# arXiv:2410.01647v2[cs.CV]13Mar2026

### 3DGS-DET: Empower 3D Gaussian Splatting with Boundary Guidance and Box-Focused Sampling for Indoor 3D Object Detection

Yang Cao∗, Yuanliang Ju∗, Dan Xu† Hong Kong University of Science and Technology

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

3DGS w/o Boundary Guidance 3DGS w/ Boundary Guidance

Figure 1. Illustration of the proposed Boundary Guidance. By incorporating Boundary Guidance in the training of 3D Gaussian Splatting (3DGS), we significantly improve the spatial distribution of Gaussian blobs relating objects and the background. To better show this improved spatial distribution, we visualize only the positions of the Gaussian blobs, omitting other attributes for clarity.

#### Abstract

Neural Radiance Fields (NeRF) have been adapted for indoor 3D Object Detection (3DOD), offering a promising approach to indoor 3DOD via view-synthesis representation. But its implicit nature limits representational capacity. Recently, 3D Gaussian Splatting (3DGS) has emerged as an explicit 3D representation that addresses the limitation. This work introduces 3DGS into indoor 3DOD for the first time, identifying two main challenges: (i) Ambiguous spatial distribution of Gaussian blobs – 3DGS primarily relies on 2D pixel-level supervision, resulting in unclear 3D spatial distribution of Gaussian blobs and poor differentiation between objects and background, which hinders indoor 3DOD; (ii) Excessive background blobs – 2D images typically include numerous background pixels, leading to densely reconstructed 3DGS with many noisy Gaussian

Under review.

blobs representing the background, negatively affecting detection. To tackle (i), we leverage the fact that 3DGS reconstruction is derived from 2D images, and propose an elegant solution by incorporating 2D Boundary Guidance to significantly enhance the spatial distribution of Gaussian blobs, resulting in clearer differentiation between objects and their background (please see Fig. 1). To address (ii), we propose a Box-Focused Sampling strategy using 2D boxes to generate object probability distribution in 3D space, allowing effective probabilistic sampling in 3D to retain more object blobs and reduce noisy background blobs. Benefiting from these innovations, 3DGS-DET significantly outperforms the state-of-the-art NeRF-based method, NeRFDet++, achieving improvements of +6.0 on mAP@0.25 and +7.8 on mAP@0.5 for the ScanNet, and the +14.9 on mAP@0.25 for the ARKITScenes. The code and models will be made publicly available upon acceptance at: https://github.com/yangcaoai/3DGS-DET.

#### 1. Introduction

Indoor 3D Object Detection (3DOD) [16, 30, 32, 52] is a fundamental task in computer vision, providing foundations for wide realistic application scenarios such as robotics and augmented reality, as accurate localization and classification of objects in 3D space are critical for these applications. Most existing indoor 3DOD methods [2–4, 37, 38] explored using non-view-synthesis representations, including point clouds and multi-view images, to perform 3D object detection. However, these approaches mainly focus on the perception perspective and lack the capability for novel view synthesis.

Neural Radiance Fields (NeRF) [28] provide an effective manner for novel view synthesis and have been adapted for indoor 3D Object Detection (3DOD) through viewsynthesis representations [16, 52]. However, as a viewsynthesis representation for indoor 3DOD, NeRF has an inherent key limitation: Its implicit nature restricts its representational capacity for indoor 3DOD. Recently, 3D Gaussian Splatting (3DGS) [18] has emerged as an explicit 3D representation that effectively addresses the limitation. Inspired by these strengths, our work is the first to introduce 3DGS into indoor 3DOD. In this exploration, we identify two primary challenges: (i) Ambiguous spatial distribution of Gaussian blobs – 3DGS primarily relies on 2D pixel-level supervision, resulting in unclear 3D spatial distribution of Gaussian blobs and insufficient differentiation between objects and background, which hinders effective indoor 3DOD; (ii) Excessive background blobs – 2D images typically contain numerous background pixels, leading to densely populated 3DGS with many noisy Gaussian blobs representing the background, hindering the detection of foreground 3D objects.

To address the above-discussed challenges, we further empower 3DGS with two novel strategies for 3D object detection (i) 2D Boundary Guidance Strategy: Given the fact that 3DGS reconstruction is optimized from 2D images, we introduce a novel strategy by incorporating 2D Boundary Guidance to achieve a more suitable 3D spatial distribution of Gaussian blobs for detection. Specifically, we first perform object boundary detection on posed images, then overlay the boundaries onto the images, and finally train the 3DGS model. This proposed strategy can facilitate the learning of a spatial Gaussian blob distribution that is more differentiable for the foreground objects and the background (see Fig. 1). (ii) Box-Focused Sampling Strategy: This strategy further leverages 2D boxes to establish 3D object probability spaces, enabling an object probabilistic sampling of Gaussian blobs to effectively preserve object blobs and prune background blobs. Specifically, we project

- the 2D boxes that cover objects in images into 3D spaces to form frustums. The 3D Gaussian blobs within the frustum have a higher probability of being object blobs com-

pared to those outside. Based on this strategy, we construct 3D object probability spaces and sample Gaussian blobs accordingly, finally preserving more object blobs and reducing noisy background blobs. In summary, the contributions of this work are fourfold:

- • To the best of our knowledge, we are the first to integrate 3DGS into indoor 3D Object Detection (3DOD), representing a novel contribution. We propose a novel method 3DGS-DET, which empowers 3DGS with Boundary Guidance and Box-Focused Sampling for indoor 3DOD.
- • We design Boundary Guidance to optimize 3DGS with the guidance of object boundaries, which achieves a significantly better spatial distribution of Gaussian blobs and clearer differentiation between objects and the background, thereby effectively enhancing indoor 3DOD.
- • We propose Box-Focused Sampling, which establishes 3D object probability spaces, enabling a higher sampling probability to be assigned to object-related 3D Gaussian blobs. This probabilistic sampling strategy preserves more foreground object blobs and suppresses noisy background blobs, further improving detection performance.
- • Boundary Guidance and Box-Focused Sampling improve detection by 5.6 points on mAP@0.25 and 3.7 points on mAP@0.5 as demonstrated in our ablation study. Furthermore, our final method, 3DGS-DET, significantly outperforms the state-of-the-art NeRF-based method, NeRFDet++ [16], on both ScanNet (+6.0 on mAP@0.25,

+7.8 on mAP@0.5) and ARKITScenes (+14.9 on mAP@0.25). Moreover, our method also clearly outperforms methods that use multi-view images as input, demonstrating the superiority of our 3DGS for indoor 3DOD.

- 2. Related Works
- 3D Gaussian Splatting (3DGS) is an effective explicit representation that models 3D scenes or objects using Gaussian blobs – small, continuous Gaussian functions distributed across 3D space. Recent works [20, 24, 42, 62] have shown that 3DGS is highly suitable for dynamic scene modeling. Additionally, some studies [12, 21, 23, 49, 50, 60] also demonstrate its efficiency in processing large-scale 3D scene data. Works [14, 34, 43, 63, 65] leverage advanced 2D foundational models, such as SAM [19] and CLIP [35], along with feature extraction methods like DINO [59], to boost perception effectiveness. Unlike previous methods that often overlook specific challenges of indoor 3D Object Detection (3DOD), our approach uniquely introduces Boundary Guidance and Box-Focused Sampling, marking the first exploration of 3DGS as a representation for the indoor 3D object detection task.

Non-View-Synthesis Representation-Based 3D Object Detection. Traditional 3D detection tasks primarily utilize the following representations: (i) Point cloud-based meth-

Sec3.3-3DGS with Boundary Guidance

###### Rendered Images

###### Posed Images

###### With Boundary

###### Noisy Blobs

[Figure 15]

[Figure 16]

|I1|
|---|

[Figure 17]

|Ibd1|
|---|

[Figure 18]

|RIbd1|
|---|

[Figure 19]

[Figure 20]

[Figure 21]

Reconstruction

[Figure 22]

[Figure 23]

BoundaryDetector

[Figure 24]

[Figure 25]

[Figure 26]

…

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

… …

[Figure 31]

[Figure 32]

[Figure 33]

…

[Figure 34]

[Figure 35]

|I(1+t)|
|---|

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

|Ibd(1+t)|
|---|

|RIbd(1+t)|
|---|

…

Rendering

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

|𝓛𝒓𝒆𝒄𝒐𝒏|
|---|

|Backbone Neck Head<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>|Block|
|---|
<br><br>Block Block Block<br><br>N<br><br>N<br><br>|N|
|---|
<br><br>|N|
|---|
<br><br>|Head|
|---|
<br><br>|Head|
|---|
<br><br>|Head|
|---|
<br><br>|Head|
|---|
<br><br>|Class.|
|---|
|Center.|
|Regress.|
<br><br>…|
|---|

|Sampling Probability<br><br>3DGS Spacial Distribution<br><br>0<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>Remove<br><br>Retain|
|---|

[Figure 55]

Box Detector

[Figure 56]

[Figure 57]

|Ibx1|
|---|

[Figure 58]

[Figure 59]

Sampling

[Figure 60]

…

|Ibx(1+t)|
|---|

[Figure 61]

| | |
|---|---|

Sec3.4-Box-Focused Sampling

2D 3D Frustum

With BBox

3DGS Object Detection

- Figure 2. Method Overview (best view in color). The blue block in the top row illustrates our Boundary Guidance (Sec. 3.3). The orange block in the bottom row shows our Box-Focused Sampling (Sec. 3.4). First, we train the 3DGS with the Boundary Guidance as shown in the blue block, which improves the 3D spatial distribution of Gaussian blobs, and thus produces clearer differentiation between objects and the background, as highlighted by the colorful dashed ellipses in the figure. Next, we perform box-focused sampling to selectively preserve object-associated blobs while effectively suppressing background noise. Finally, the sampled Gaussian blobs are fed into the detector for accurate indoor 3D object detection.

ods [2, 3, 32, 33, 37, 46, 55, 56] directly process 3D points captured by sensors like LiDAR or depth cameras. Methods such as VoteNet [32] and CAGroup3D [46] efficiently handle point clouds, capturing geometries while facing challenges in computational efficiency due to their irregular structure. Some works [6, 10, 25, 26, 29, 53, 57, 64] divide 3D space into uniform volumetric units, enabling 3D convolutional neural networks to process the data. (ii) Multi-view image-based methods [5, 11, 41, 45, 47, 48, 51] leverage 2D images from multiple views to reconstruct 3D structures.

sampling, and ordinal-residual depth supervision. Notably, NeRF-RPN focuses on class-agnostic box detection [13, 15, 17, 54], while NeRF-Det targets class-specific object detection. Our work follows the class-specific setting of NeRF-Det. However, NeRF faces a significant challenge: its implicit nature limits its representational capacity for 3D object detection. 3DGS [18] has emerged as an explicit 3D representation, effectively addressing the limitation. Motivated by that, our work introduces 3DGS into indoor 3DOD for the first time, and presents novel designs to adapt 3DGS for detection, making significant differences from NeRFbased methods [15, 52].

View-Synthesis Representation-Based Indoor 3D Object Detection. NeRF [28] have become popular for novelview-synthesis and have been adapted for indoor 3D Object Detection (3DOD) [16, 52]. These adaptations present promising solutions for detecting 3D objects using viewsynthesis representations. For instance, NeRF-RPN [15] employs voxel representations, integrating multi-scale 3D neural volumetric features to perform category-agnostic box localization [13, 17, 54], rather than category-specific object detection (our task setting). NeRF-Det [52] incorporates multi-view geometric constraints from the NeRF component into 3D detection. NeRF-Det++ [16] improves NeRF-Det for indoor multi-view 3D detection by adding

#### 3. Methodology

The pipeline of our 3DGS-DET is illustrated in Fig. 2. Initially, we train the 3D Gaussian Splatting (3DGS) on the input scenes using the proposed Boundary Guidance, which significantly enhances the spatial distribution of Gaussian blobs, resulting in clearer differentiation between objects and the background. Subsequently, we apply the proposed Box-Focused Sampling, which effectively preserves objectrelated blobs while suppressing noisy background blobs. The sampled 3DGS is then fed into the detection framework

- 2D semantic supervision, perspective-aware non-uniform

##### 3.2. Proposed Basic Pipeline of 3DGS for 3DOD

[Figure 62]

In this section, we build our basic pipeline by directly utilizing the original 3DGS for indoor 3D Object Detection (3DOD) without any further improvement. First, we train the 3DGS representation of the input scene using posed images, denoted as G = {(µi,Si,Ri,ci,αi)}Ni=1. Given that the number of Gaussian blobs N is too large for them to be input into the detector, we perform random sampling to select a subset of Gaussian blobs, denoted as Gˆ = {(µi,Si,Ri,ci,αi)}Mi=1, where M < N. We then concatenate the attributes of the Gaussian blobs along the channel dimension as follows:

[Figure 63]

[Figure 64]

[Figure 65]

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

[Figure 77]

[Figure 78]

Gˆinput = Concat(µi,Si,Ri,ci,αi) ∀i ∈ {1,...,M}.

(1) This concatenated representation Gˆinput is then fed into the subsequent detection tool. Note that since 3DGS is an explicit 3D representation, Gˆinput can be utilized with any point-cloud-based detector by retraining the detector model on 3DGS representation. In our study, the research focus is on enhancing 3DGS for 3DOD in general, rather than designing a specific detector. Therefore, we utilize the existing work [37] as the detection tool. The final detection predictions are obtained as follows:

- Figure 3. Rendered images from different views by 3DGS trained with Boundary Guidance. The category-specific boundaries are clearly rendered and exhibit multi-view consistency, showing that

- the 3D representation has successfully embedded the priors provided by Boundary Guidance.

for training. In this section, we detail our method step by step. First, we introduce the preliminary concept of 3DGS

- in Sec. 3.1. As the first to introduce 3DGS in 3D object detection, we establish the basic pipeline in Sec. 3.2, utilizing 3DGS for input and output detection predictions. We then present Boundary Guidance in Sec. 3.3. Finally, we describe the Box-Focused Sampling Strategy in Sec. 3.4.

P = F(Gˆinput) = (p,z,b), (2)

where F denotes the detector tool and P represents the predictions, including classification probabilities p, centerness z, and bounding box regression parameters b. The training loss [37] is defined as:

##### 3.1. Preliminary: 3D Gaussian Splatting

1 Npos x, ˆ y,ˆ zˆ

1{p(ˆx,y,ˆ zˆ)̸=0}Lreg(bˆ,b)

Ldet =

In our proposed method, 3DGS-DET, the input scene is represented using 3DGS [18], formulated as G = {(µi,Si,Ri,ci,αi)}Ni=1, where N denotes the number of Gaussian blobs. Each blob is characterized by its 3D coordinate µi, scaling matrix Si, rotation matrix Ri, color features ci, and opacity αi. These attributes define the Gaussian through a covariance matrix Σ = RSSTRT, centered at µ: G(x) = exp(−21(x−µ)TΣ−1(x−µ)).

+ 1{p(ˆx,y,ˆ zˆ)̸=0}Lcntr(zˆ,z) + Lcls(pˆ,p) , (3) where the number of matched positions Npos is given by

x, ˆ y,ˆ zˆ 1{p(ˆx,y,ˆ zˆ)̸=0}. Ground truth labels are indicated with a hat symbol. The regression loss Lreg is based on Intersection over Union (IoU), the centerness loss Lcntr uses binary cross-entropy, and the classification loss Lcls employs focal loss. Further details on the detection tool can be found in [37]. Building upon this basic pipeline, we develop our method, 3DGS-DET, by introducing two novel designs (Sec. 3.3 and Sec. 3.4) to improve the 3DGS representation, as illustrated in Fig. 2.

During rendering, opacity modulates the Gaussian. By projecting the covariance onto a 2D plane [66], we derive the projected Gaussian, and utilize volume rendering [27] to compute the image pixel colors: C =

K k=1 αkck kj=1−1(1−αj), where K is the number of sam-

pling points along the ray. αi is determined by evaluating a

##### 3.3. Boundary Guidance

- 2D Gaussian with covariance Σ, multiplied by the learned opacity [58]. The initial 3D coordinates of each Gaussian are based on Structure from Motion (SfM) points [39]. Gaussian attributes are refined to minimize the image recon-

Given the fact that 3DGS reconstruction is derived from 2D images, we design the novel Boundary Guidance strategy by incorporating 2D Boundary Guidance to achieve a more suitable 3D spatial distribution of Gaussian blobs for detection. In this section, we present our Boundary Guidance strategy. As illustrated in the blue block in the top row

struction loss: Lrender = (1 − λ)L1(I,Iˆ) + λLD-SSIM(I,Iˆ), where Iˆrepresents the ground truth images. Additional details can be found in [18].

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

(a) Random Sampling

(b) Box-Focused Sampling

- Figure 4. Qualitative Analysis of Box-Focused Sampling for

- 3DGS Blobs. Box-Focused Sampling significantly retains more object blobs and reduces noisy background blobs. Note that we visualize only the positions of the Gaussian blobs to highlight their spatial distribution, while omitting other attributes for clarity.

of Fig. 2, to provide the guidance priors for 3DGS reconstruction, we first generate category-specific boundaries for posed images:

Bbd = Hbd(I) = {bcbd} c ∈ C, (4)

where Hbd is the boundary generator, and bcbd represents the binary boundary map for category c. If bcbd(x,y) = 1, the pixel at (x,y) belongs to the boundary for objects of category c. The set C includes all categories. In practice, the operations of Hbd are as follows: we use Grounded SAM [36] to generate category-specific masks. Then, the SuzukiAbe algorithm [44] is employed to extract the boundaries of these masks, along with category information. The category-specific boundaries are then overlaid on the posed images in different colors:

bcbd(x,y)

Ibd(x,y) =I(x,y) · 1 −

c∈C

bcbd(x,y) · color(c), (5)

+

c∈C

where Ibd(x,y) is the pixel at position (x,y) of the final image with overlaid boundaries. I(x,y) is from the origi-

nal image, bcbd(x,y) is the boundary map for category c, and color(c) is the color associated with category c. These Ibd images are used as ground truth to train the 3DGS representation Gbd by the following loss:

Lrender = (1 − λ)L1(I,Ibd) + λLD-SSIM(I,Ibd). (6)

To effectively reduce Lrender during training, it is crucial to ensure the rendering quality of boundaries and the multiview stability of boundaries. In this way, the Boundary Guidance lead 3DGS to incorporate boundary prior information into the 3D space. As shown in Fig. 1, 3DGS trained with Boundary Guidance demonstrates improved spatial distribution of Gaussian blobs compared to those trained

without it. Besides, Fig. 3 shows the rendered images from different views by 3DGS trained with Boundary Guidance. The category-specific boundaries are clearly rendered with preserved multi-view consistency, evidencing that the 3D representation has successfully embedded the priors from Boundary Guidance.

##### 3.4. Box-Focused Sampling

Considering that 2D images often include numerous background pixels, leading to densely reconstructed 3DGS with many noisy Gaussian blobs representing the background, negatively affecting detection. To reduce the excessive background blobs, in this section, we propose the BoxFocused Sampling strategy in detail. As depicted in the orange block in the bottom row of Fig. 2, to provide priors for the following sampling, we utilize a 2D object detector to identify object bounding boxes:

Bbb = Hbb(I) = {(bbb,pC)}, (7)

where Hbb is the box detector, and we select Grounding DINO [22] as the detector in our experiments. Here, bbb denotes the predicted bounding box positions, and pC is the predicted probability vector for the box belonging to each category in C. We define pmax = maxc∈C pc as the highest category probability for a given bounding box, which helps to establish object probability spaces in later step. Then, we project the 2D boxes into 3D space:

 

  | (xi,yi) ∈ bbb,z ∈ {zmin,zmax}},

xi yi z

Fft = {K−1

(8) where Fft is the projected 3D frustum from bbb, and K−1 is the inverse camera matrix used to map 2D bounding box corners (xi,yi) and depth values zmin and zmax into 3D space. Next, we establish object probability spaces by Fft and pmax. For each bounding box, the maximum probability pmax models the likelihood of Gaussian blobs within the corresponding frustum belonging to object blobs:

pobj(gi | gi ∈ Fft) = pmax, (9)

where pobj(gi | gi ∈ Fft) denotes the probability of Gaussian blob gi within frustum Fft belonging to object blobs. To integrate priors from different view frustums, we select the maximum probability as the aggregated probability:

pobj(gi | gi ∈ Fftv), (10)

pagr(gi) = max

v∈V

where pagr(gi) is the aggregated probability for Gaussian blob gi, and V represents the set of all views. Gaussian blobs not belonging to any frustum are assigned a small probability pbg, set to 0.01 in practice. In this way, we obtain the object probability spaces Pobj, where each Gaussian blob has an associated probability of belonging to an ob-

Methods cab bed chair sofa tabl door wind bkshf pic cntr desk curt fridg showr toil sink bath ofurn mAP@0.25

VoteNet 36.3 87.9 88.7 89.6 58.8 47.3 38.1 44.6 7.8 56.1 71.7 47.2 45.4 57.1 94.9 54.7 92.1 37.2 58.7 FCAF3D 57.2 87.0 95.0 92.3 70.3 61.1 60.2 64.5 29.9 64.3 71.5 60.1 52.4 83.9 99.9 84.7 86.6 65.4 71.5 CAGroup3D 60.4 93.0 95.3 92.3 69.9 67.9 63.6 67.3 40.7 77.0 83.9 69.4 65.7 73.0 100.0 79.7 87.0 66.1 75.12

ImGeoNet 40.6 84.1 74.8 75.6 59.9 40.4 24.7 60.1 4.2 41.2 70.9 33.7 54.4 47.5 95.2 57.5 81.5 36.1 54.6 CN-RMA 42.3 80.0 79.4 83.1 55.2 44.0 30.6 53.6 8.8 65.0 70.0 44.9 44.0 55.2 95.4 68.1 86.1 49.7 58.6 ImVoxelNet 30.9 84.0 77.5 73.3 56.7 35.1 18.6 47.5 0.0 44.4 65.5 19.6 58.2 32.8 92.3 40.1 77.6 28.0 49.0

NeRF-Det 37.6 84.9 76.2 76.7 57.5 36.4 17.8 47.0 2.5 49.2 52.0 29.2 68.2 49.3 97.1 57.6 83.6 35.9 53.3 NeRF-Det++ 36.1 82.9 74.9 79.1 57.0 37.3 24.9 54.6 2.4 51.7 72.2 25.5 58.7 51.5 92.7 50.8 82.2 35.1 53.9 3DGS-DET 44.1 82.7 81.7 79.6 56.0 35.4 27.6 45.2 17.3 61.9 72.8 40.7 56.6 71.9 98.5 72.2 88.3 46.7 59.9 (+6.0)

- Table 1. Comparison of mAP@0.25 on the ScanNet dataset. The first two blocks include methods that utilize non-view-synthesis representations: the first block [32, 37, 46] features approaches based on point clouds, while the second block [38, 41, 45] focuses on multi-view image methods. The third block [16, 52] encompasses methods that employ view-synthesis representations, including NeRF-based approaches and our 3DGS-based method. Among non-view-synthesis representations, our approach clearly outperforms methods that utilize multi-view images (the second block), highlighting the superiority of our 3DGS reconstruction for detection with multi-view image inputs. In terms of view-synthesis representations, our 3DGS-DET surpasses the NeRF-based method NeRF-Det++ [16] by 6.0 points.

Methods cab fridg shlf stove bed sink wshr tolt bthtb oven dshw frplce stool chr tble TV sofa mAP@0.25

ImVoxelNet 32.2 34.3 4.2 0.0 64.7 20.5 15.8 68.9 80.4 9.9 4.1 10.2 0.4 5.2 11.6 3.1 35.6 23.6 NeRF-Det 36.1 40.7 4.9 0.0 69.3 24.4 17.3 75.1 84.6 14.0 7.4 10.9 0.2 4.0 14.2 5.3 44.0 26.7 NeRF-Det++ - - - - - - - - - - - - - - - - - 43.3 3DGS-DET (Ours) 45.2 84.4 33.3 41.4 87.3 75.5 67.6 87.2 90.8 74.3 6.0 56.4 26.3 70.3 60.6 0.7 81.8 58.2 (+14.9)

- Table 2. Comparison of the ‘whole-scene’ performance [38, 52] on the ARKITScenes validation set. Our 3DGS-DET significantly outperforms NeRF-Det++ [16] by 14.9 points. Note that we follow the setup described in the NeRF-Det supplementary materials: ‘In our experiments, we utilize the subset of the dataset with low-resolution images’, considering it is the closest work to ours. Other methods that do not use the same setting are not listed in this table. NeRF-Det++ [16] does not report per-category performance in ARKITScenes. We report its overall mAP@0.25 following the original paper [16].

ject. We then perform probabilistic sampling based on Pobj to achieve Box-Focused Sampling, resulting in the sampled

Gaussian set Gˆbsbd as:

Gˆbsbd = {g | g ∼ Pobj(g)}. (11) In this way, it allows object blobs to be better preserved due to their higher probabilities, while most background points, having lower probabilities, are effectively reduced. Then, based on Gˆbsbd, we proceed with the training of the detector, as formulated by Equ. 1-Equ. 3 as described in Sec. 3.2. As shown in Fig. 4, 3DGS sampled via Box-Focused Sampling retains more object blobs and reduces background noise.

#### 4. Experiments 4.1. Experimental setup

Dataset and Metrics: To thoroughly evaluate the performance of our proposed method in indoor 3D detection tasks, we selected two representative datasets: ScanNet [9] and ARKitScene [1]. ScanNet is a large-scale indoor scene dataset containing over 1,500 real-world 3D scanned scenes, encompassing various complex indoor environments such as residential spaces, offices, and classrooms. The ARKitScene dataset is constructed from RGBD image sequences, offering detailed geometric information and precise object annotations. For each scene, a maximum of 600 posed images are extracted. The category settings follow the standard 18 categories for ScanNet and 17 categories for ARKitScene. We use mAP@0.25 and

mAP@0.5 as the primary evaluation metrics.

##### 4.2. Main Results

Quantitative Results. For the ScanNet dataset, we present the mAP@0.25 and mAP@0.5 performances of various methods in Tab. 1 of the main paper and Tab. 3 of the supplementary, respectively. In both Tab. 1 and Tab. 3 of the supplementary, the methods listed in the first two blocks are non-view-synthesis representation-based 3D detection methods: the first block [32, 37, 46] lists approaches based on point clouds, while the second block focuses on multiview image methods [38, 41, 45]. The third block consists of view-synthesis representation-based 3DOD methods, including NeRF-Det [52], NeRF-Det++ [16] and our proposed 3DGS-DET. NeRF-Det and NeRF-Det++ are the closest work to ours, leveraging Neural Radiance Fields (NeRF). ‘3DGS-DET’ is our full method, utilizing both Boundary Guidance and Box-Focused Sampling as described in Sec. 3.4. As illustrated in Tab. 1 and Tab. 3 of the supplementary, our method significantly outperforms NeRF-Det++ by +6.0 on mAP@0.25 and +7.8 on mAP@0.5, showcasing the superiority of our approach. Besides, our method also clearly surpasses approaches utilizing multi-view images as input, demonstrating the superiority of our 3DGS reconstruction for detection.

Regarding the ARKitScene dataset, considering NeRFDet is the closest work to ours, we follow the same setup described in the NeRF-Det [52] supplementary materials: ‘In our experiments, we utilize the subset of the dataset

|[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>|
|---|

|[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>|
|---|

|[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>|
|---|

|[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]|
|---|

|[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]|
|---|

|[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]|
|---|

(a) Center Point (b) Mask (c) Boundary

- Figure 5. Analysis of guidance from different priors: (a) Center Point Guidance, (b) Mask Guidance, and (c) Boundary Guidance. In (a) and (b), the spatial distribution of Gaussian blobs for objects like the chair, trash bin and sink is incomplete and ambiguous. Gaussian blobs trained with Boundary Guidance exhibit a clearer spatial distribution. For detailed analysis, please refer to Sec. 4.3.

Methods mAP@0.25 mAP@0.5

BP 54.3 34.1 BP+BG 56.7 36.9 BP+BG+BS 59.9 37.8

Table 3. Effect of our designs.

Different Priors mAP@0.25 mAP@0.5

2D Center Point 54.4 33.9 2D Mask 54.9 34.2 2D Boundary (ours) 56.7 36.9

Table 4. Guidance from different priors.

Sampling Methods mAP@0.25 mAP@0.5

Random Sampling 56.7 36.9 Farthest Point Sampling 57.4 37.6 Box-focused Sampling (ours) 59.9 37.8

Table 5. Different sampling methods.

with low-resolution images.’ Similarly, we adopt the same subset of the ARKitScenes dataset. Other methods that report performance on ARKitScene use the full dataset, so our

- 3DGS-DET is only compared with ImVoxelNet, NeRF-Det and NeRF-Det++ under the same conditions. The results in Tab. 2 demonstrate that 3DGS-DET performs better across most categories, achieving an mAP@0.25 of 58.2, which significantly outperforms NeRF-Det++ by +14.9, highlighting the superiority of our method. Qualitative results. We provide a qualitative comparison with NeRF-Det in Fig. 1 and Fig. 2 of the supplementary. As shown, our method detects more objects in the scene with greater positional accuracy compared to NeRFDet [52], demonstrating the superiority of our method.

##### 4.3. Analysis on the Proposed Designs

In this section, we demonstrate the effectiveness of our contributions by first presenting the performance of our proposed basic 3DGS detection pipeline and then incrementally incorporating our additional designs to analyze the resulting performance improvements. We further delve into the method and provide additional analyses to for a more comprehensive understanding of our method.

Our Proposed Basic 3DGS Detection Pipeline. As shown in Tab. 3, ‘BP’ represents our basic pipeline, i.e., our proposed detection pipeline utilizing 3DGS, as described

- in Sec. 3.2. Benefiting from the advantages of 3DGS as an

explicit scene representation, our basic pipeline surpasses NeRF-Det++ (54.3 vs. 53.9), underscoring the significance of introducing 3DGS into indoor 3DOD for the first time.

Effectiveness of Boundary Guidance. In Tab. 3, ‘BP+BG’ incorporates the proposed Boundary Guidance as detailed in Sec. 3.3. Introducing Boundary Guidance into the basic pipeline results in a significant improvement of 2.4 points (56.7 vs. 54.3), demonstrating the effectiveness of the proposed Boundary Guidance. To further explore the impact of Boundary Guidance on 3DGS representations, we present a visual comparison of the spatial distribution of trained Gaussian blobs in Fig. 3 in the supplementary. As we can see, Gaussian blobs trained with Boundary Guidance demonstrate clearer spatial distribution and more distinct differentiation between objects and the background. We also present rendered images from different views by 3DGS trained with Boundary Guidance in Fig. 4 and Fig. 5 in the supplementary. As can be observed, the category-specific boundaries are clearly rendered and show multi-view stability, indicating that the 3D representation has effectively embedded the priors from Boundary Guidance. All these results clearly verify the effectiveness of the proposed Boundary Guidance for 3DGS-Det.

Effectiveness of Box-Focused Sampling. Furthermore, we introduce Box-Focused Sampling detailed in Sec. 3.4, represented by ‘BP+BG+BS’ in Tab. 3. This addition leads to a further performance boost of 3.2 points (59.9 vs. 56.7),

###### Sampling Methods mAP@0.25 mAP@0.5

Box-Focused Sampling 59.9 37.8 Box-Focused Sampling-GT-Box 76.5 68.3

- Table 6. Upper bound of Box-Focused Sampling with GT boxes. Sampling Methods Average Noisy Points Farthest Point Sampling (FPS) 80025 Box-Focused Sampling (Ours) 59476 (-25.7%)

- Table 7. Average noisy points: FPS vs. Box-Focused Sampling.

proving the effectiveness of Box-Focused Sampling. The visual comparison of sampled Gaussian blobs is shown in Fig. 6 in the supplementary. We can observe that the proposed Box-Focused Sampling significantly retains more object blobs and suppresses noisy background blobs.

Guidance from Different Priors. In this section, we analyze the impact of guidance from various priors. As described in Sec. 3.3, we utilize the object’s boundary as the guidance prior. Here, we perform an ablation study considering the object’s center point and mask as alternative priors. To obtain the center point, we detect the object’s bounding box using GroundingDINO [22] and compute its center coordinates. The mask is generated with GroundedSAM [36]. Note that all priors are category-specific, with each class associated with a fixed color. These priors are overlaid on the posed images, as shown in Fig. 5, and then used to train the 3DGS for detection. Tab. 4 presents the detection performance for 3DGS trained with the different priors. As reported in Tab. 4, the 3DGS-DET method using boundary guidance achieved 56.7% in mAP@0.25 and 36.9% in mAP@0.5, demonstrating significant superiority over the alternative priors.

Let’s explore the visualizations for further insights. In (a) and (c) of Fig. 5, we observe that the spatial distribution of Gaussian blobs with Point Guidance is less distinct compared to Boundary Guidance. This is because the center point provides only positional guidance, lacking richer information like shape or size, making it less effective compared to the boundary prior. For the mask prior, as shown in (b) and (c) of Fig. 5, the Gaussian blobs’ spatial distribution with Mask Guidance is more ambiguous than with the Boundary Guidance. Although the mask highlights shape and size information, it hides the object’s surface, reducing texture and geometric information, thus being less effective than the boundary prior. Overall, Boundary Guidance offers positional cues and richer information such as shape and size while preserving texture and geometric details on the object’s surface, leading to the best performance.

Different Sampling Methods. In this section, we compare two additional sampling methods with our Box-Focused Sampling: 1) Random Sampling and 2) Farthest Point Sampling [31]. The latter iteratively selects points farthest from those already chosen, ensuring even distribution

Class-agnostic Setting mAP@0.25 mAP@0.5

NeRF-RPN [15] 55.5 18.4 NeRF-MAE [17] 57.1 17.0 NeRF-FCM [13] 58.8 23.4 Gaussian-Det [54] 71.7 24.5 3DGS-DET (ours) 75.6 (+3.9) 52.3 (+27.8)

Table 8. Performance on the class-agnostic setting [13, 15, 17, 54], which targets class-agnostic box detection, i.e., localizing objects without predicting their semantic categories.

for better scene coverage, focusing on global distribution rather than specific geometric features of objects. The results in Tab. 5 demonstrate that our Box-Focused Sampling achieves the highest performance, with mAP@0.25 and mAP@0.5 reaching 59.9% and 37.8%, respectively. This is because 3DGS often contain excessive background blobs. Our Box-Focused Sampling is specifically designed to preserve more object-related blobs while suppressing noisy background blobs. In contrast, other sampling methods primarily focus on global scenes without differentiation between objects and background blobs.

The Upper Bound of Box-Focused Sampling. In this section, we conduct an ablation study on the upper bound of Box-Focused Sampling by utilizing ground-truth (GT) 3D bounding boxes to sample 3DGS blobs. As shown in Tab. 6, despite the significant improvements achieved with BoxFocused Sampling (3.2 points in Tab. 3), using GT 3D boxes brings further enhancements, highlighting its potential as a promising research direction.

Noise Reduction via Box-Focused Sampling. We compare the average number of noisy points in test scenes on ScanNet when using Farthest Point Sampling (FPS) and our Box-Focused Sampling. As shown in Tab. 7, our BoxFocused Sampling significantly reduces noisy background blobs by 25.7% compared with the FPS sampling, demonstrating its effectiveness in noise reduction.

##### 4.4. Performance on NeRF-RPN Setting

In this section, we adapt our 3DGS-DET to the NeRFRPN setting [15, 17], which targets class-agnostic box detection. To achieve this, we labeled all the ground-truth boxes with a single ‘object’ category and trained 3DGSDET accordingly. Additionally, NeRF-RPN uses a different train/validation split compared to the official ScanNet dataset, with its validation set overlapping the official ScanNet training set. To address this, we excluded the overlapping parts between the NeRF-RPN validation set and the ScanNet official training set from our training data. We then used the remaining scenes for training, and evaluated on the same validation set provided by NeRF-RPN. As shown in Tab. 8, 3DGS-DET achieved an mAP@0.25 of 75.6% and an mAP@0.5 of 52.3%, significantly outperforming Gaussian-Det [54]’s 71.7% and 24.5%. This

demonstrates the significant superiority of our method in the class-agnostic setting.

##### 4.5. More Ablation Studies

To comprehensively analyze our method, in Sec. 2 of the supplementary material, we further study ‘The Effect of Boundary Guidance on the Spatial Distribution of 3DGS Blobs’, ‘Comparison with SfM-Based 3D Detection Baseline’, ‘Application to Feed-Forward 3DGS’, ‘Efficiency Analysis’ , ‘Failure Cases’ and provide extensive qualitative analysis.

#### 5. Conclusion

In this work, we introduce 3DGS into indoor 3D Object Detection (3DOD) for the first time and propose 3DGS-DET, a novel approach that leverages Boundary Guidance and BoxFocused Sampling to enhance 3DGS for indoor 3DOD. Our method addresses 3DGS challenges by improving spatial distribution and reducing background noise. With 2D Boundary Guidance, we achieve clearer object-background differentiation, while Box-Focused Sampling retains more object points and suppresses noise. 3DGS-DET significantly outperforms state-of-the-art methods.

#### References

- [1] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. Arkitscenes

- a diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. In NeurIPS, 2021. 6

- [2] Yang Cao, Yihan Zeng, Hang Xu, and Dan Xu. Coda: Collaborative novel box discovery and cross-modal alignment for open-vocabulary 3d object detection. In NeurIPS, 2023. 2, 3
- [3] Yang Cao, Yihan Zeng, Hang Xu, and Dan Xu. Collaborative novel object discovery and box-guided cross-modal alignment for open-vocabulary 3d object detection. TPAMI,

2025. 3

- [4] Yang Cao, Feize Wu, Zhenyu Dave Chen, Yingji Zhong, Lanqing Hong, and Dan Xu. Vggt-det: Mining vggt internal priors for sensor-geometry-free multi-view indoor 3d object detection. In CVPR, 2026. 2
- [5] Dian Chen, Jie Li, Vitor Guizilini, Rares Andrei Ambrus, and Adrien Gaidon. Viewpoint equivariance for multi-view 3d object detection. In CVPR, 2023. 3
- [6] Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, and Jiaya Jia. Voxelnext: Fully sparse voxelnet for 3d object detection and tracking. In CVPR, 2023. 3
- [7] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In ECCV, 2024. 12
- [8] MMDetection3D Contributors. MMDetection3D: OpenMMLab next-generation platform for general 3D object

- detection. https://github.com/open-mmlab/ mmdetection3d, 2020. 12
- [9] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3D reconstructions of indoor scenes. In CVPR, 2017. 6
- [10] Jiajun Deng, Shaoshuai Shi, Peiwei Li, Wengang Zhou, Yanyong Zhang, and Houqiang Li. Voxel r-cnn: Towards high performance voxel-based 3d object detection. In AAAI,

2021. 3

- [11] Chengjian Feng, Zequn Jie, Yujie Zhong, Xiangxiang Chu, and Lin Ma. Aedet: Azimuth-invariant multi-view 3d object detection. In CVPR, 2023. 3
- [12] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Zhilin Pei, Hengjie Li, Xingcheng Zhang, and Bo Dai. Flashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering. arXiv preprint arXiv:2408.07967, 2024. 2
- [13] Hana Lebeta Goshu, Jun Xiao, Kin-Chung Chan, Cong Zhang, Mulugeta Tegegn Gemeda, and Kin-Man Lam. Nerffcm: Feature calibration mechanisms for nerf-based 3d object detection. In APSIPA ASC, pages 1–6, 2024. 3, 8
- [14] Qiao Gu, Zhaoyang Lv, Duncan Frost, Simon Green, Julian Straub, and Chris Sweeney. Egolifter: Open-world 3d segmentation for egocentric perception. In ECCV, 2025. 2
- [15] Benran Hu, Junkai Huang, Yichen Liu, Yu-Wing Tai, and Chi-Keung Tang. Nerf-rpn: A general framework for object detection in nerfs. In CVPR, 2023. 3, 8
- [16] Chenxi Huang, Yuenan Hou, Weicai Ye, Di Huang, Xiaoshui Huang, Binbin Lin, and Deng Cai. Nerf-det++: Incorporating semantic cues and perspective-aware depth supervision for indoor multi-view 3d detection. TIP, 34:2575–2587,

2025. 2, 3, 6, 12, 13, 14

- [17] Muhammad Zubair Irshad, Sergey Zakharov, Vitor Guizilini, Adrien Gaidon, Zsolt Kira, and Rares Ambrus. Nerf-mae: Masked autoencoders for self-supervised 3d representation learning for neural radiance fields. In ECCV, 2024. 3, 8
- [18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4), 2023. 2, 3, 4, 12
- [19] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 2
- [20] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian splatting for static and dynamic radiance fields. arXiv preprint arXiv:2408.03822, 2024. 2
- [21] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, et al. Vastgaussian: Vast 3d gaussians for large scene reconstruction. In CVPR, 2024. 2
- [22] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 5, 8, 12

- [23] Wenkai Liu, Tao Guan, Bin Zhu, Lili Ju, Zikai Song, Dan Li, Yuesong Wang, and Wei Yang. Efficientgs: Streamlining gaussian splatting for large-scale high-resolution scene representation. arXiv preprint arXiv:2404.12777, 2024. 2
- [24] Yang Liu, He Guan, Chuanchen Luo, Lue Fan, Junran Peng, and Zhaoxiang Zhang. Citygaussian: Real-time high-quality large-scale scene rendering with gaussians. arXiv preprint arXiv:2404.01133, 2024. 2
- [25] Anas Mahmoud, Jordan SK Hu, and Steven L Waslander. Dense voxel fusion for 3d object detection. In WACV, 2023. 3
- [26] Jiageng Mao, Yujing Xue, Minzhe Niu, Haoyue Bai, Jiashi Feng, Xiaodan Liang, Hang Xu, and Chunjing Xu. Voxel transformer for 3d object detection. In ICCV, 2021. 3
- [27] Nelson Max. Optical models for direct volume rendering. TVCG, 1(2):99–108, 1995. 4
- [28] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2, 3
- [29] Jongyoun Noh, Sanghoon Lee, and Bumsub Ham. Hvpr: Hybrid voxel-point representation for single-stage 3d object detection. In CVPR, 2021. 3
- [30] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3D classification and segmentation. In CVPR, 2017. 2
- [31] Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In NeurIPS, 2017. 8
- [32] Charles R. Qi, Or Litany, Kaiming He, and Leonidas J. Guibas. Deep hough voting for 3D object detection in point clouds. In ICCV, 2019. 2, 3, 6, 13, 14
- [33] Charles R Qi, Yin Zhou, Mahyar Najibi, Pei Sun, Khoa Vo, Boyang Deng, and Dragomir Anguelov. Offboard 3d object detection from point cloud sequences. In CVPR, 2021. 3
- [34] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In CVPR, 2024. 2
- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2
- [36] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 5, 8, 12
- [37] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. Fcaf3d: Fully convolutional anchor-free 3d object detection. In ECCV, 2022. 2, 3, 4, 6, 12, 13, 14
- [38] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. Imvoxelnet: Image to voxels projection for monocular and multi-view general-purpose 3d object detection. In WACV,

2022. 2, 6, 13, 14

- [39] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, 2016. 4, 12, 13
- [40] Johannes L Sch¨onberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In ECCV, 2016. 12, 13
- [41] Guanlin Shen, Jingwei Huang, Zhihua Hu, and Bin Wang. Cn-rma: Combined network with ray marching aggregation for 3d indoor object detection from multi-view images. In CVPR, 2024. 3, 6, 13, 14
- [42] Licheng Shen, Ho Ngai Chow, Lingyun Wang, Tong Zhang, Mengqiu Wang, and Yuxing Han. Gaussian time machine: A real-time rendering methodology for time-variant appearances. arXiv preprint arXiv:2405.13694, 2024. 2
- [43] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and ShaoHua Guan. Language embedded 3d gaussians for openvocabulary scene understanding. In CVPR, 2024. 2
- [44] Satoshi Suzuki et al. Topological structural analysis of digitized binary images by border following. Computer vision, graphics, and image processing, 30(1):32–46, 1985. 5, 12
- [45] Tao Tu, Shun-Po Chuang, Yu-Lun Liu, Cheng Sun, Ke Zhang, Donna Roy, Cheng-Hao Kuo, and Min Sun. Imgeonet: Image-induced geometry-aware voxel representation for multi-view 3d object detection. In ICCV, 2023. 3, 6, 13, 14
- [46] Haiyang Wang, Lihe Ding, Shaocong Dong, Shaoshuai Shi, Aoxue Li, Jianan Li, Zhenguo Li, and Liwei Wang. Cagroup3d: Class-aware grouping for 3d object detection on point clouds. In NeurIPS, 2022. 3, 6, 13, 14
- [47] Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, and Xiangyu Zhang. Exploring object-centric temporal modeling for efficient multi-view 3d object detection. In ICCV, 2023. 3
- [48] Yue Wang, Vitor Campagnolo Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, and Justin Solomon. Detr3d: 3d object detection from multi-view images via 3d-to-2d queries. In CoRL, 2022. 3
- [49] Zipeng Wang and Dan Xu. Pygs: Large-scale scene representation with pyramidal 3d gaussian splatting. arXiv preprint arXiv:2405.16829, 2024. 2
- [50] Butian Xiong, Xiaoyu Ye, Tze Ho Elden Tse, Kai Han, Shuguang Cui, and Zhen Li. Sa-gs: Semantic-aware gaussian splatting for large scene reconstruction with geometry constrain. arXiv preprint arXiv:2405.16923, 2024. 2
- [51] Kaixin Xiong, Shi Gong, Xiaoqing Ye, Xiao Tan, Ji Wan, Errui Ding, Jingdong Wang, and Xiang Bai. Cape: Camera view position embedding for multi-view 3d object detection. In CVPR, 2023. 3
- [52] Chenfeng Xu, Bichen Wu, Ji Hou, Sam Tsai, Ruilong Li, Jialiang Wang, Wei Zhan, Zijian He, Peter Vajda, Kurt Keutzer, et al. Nerf-det: Learning geometry-aware volumetric representation for multi-view 3d object detection. In ICCV, 2023. 2, 3, 6, 7, 12, 13, 14, 15, 16
- [53] Chi Yan and Dan Xu. Progressive gaussian transformer with anisotropy-aware sampling for open vocabulary occupancy prediction. ICLR, 2026. 3
- [54] Hongru Yan, Yu Zheng, and Yueqi Duan. Gaussian-det: Learning closed-surface gaussians for 3d object detection. In ICLR, 2025. 3, 8

- [55] Bin Yang, Wenjie Luo, and Raquel Urtasun. Pixor: Realtime 3d object detection from point clouds. In CVPR, 2018. 3
- [56] Timing Yang, Yuanliang Ju, and Li Yi. Imov3d: Learning open-vocabulary point clouds 3d object detection from only 2d images. In NeurIPS, 2024. 3
- [57] Maosheng Ye, Shuangjie Xu, and Tongyi Cao. Hvnet: Hybrid voxel network for lidar based 3d object detection. In CVPR, 2020. 3
- [58] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Oztireli,¨ and Olga Sorkine-Hornung. Differentiable surface splatting for point-based geometry processing. ACM TOG, 38(6):1–14,

2019. 4

- [59] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022. 2
- [60] Hanyue Zhang, Zhiliu Yang, Xinhe Zuo, Yuxin Tong, Ying Long, and Chen Liu. Garfield++: Reinforced gaussian radiance fields for large-scale 3d scene reconstruction. arXiv preprint arXiv:2409.12774, 2024. 2
- [61] Xiyu Zhang, Jiaqi Yang, Shikun Zhang, and Yanning Zhang. 3d registration with maximal cliques. In CVPR, 2023. 12
- [62] Yingji Zhong, Zhihao Li, Dave Zhenyu Chen, Lanqing Hong, and Dan Xu. Taming video diffusion prior with scenegrounding guidance for 3d gaussian splatting from sparse inputs. In CVPR, 2025. 2
- [63] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In CVPR, 2024. 2
- [64] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In CVPR, 2018. 3
- [65] Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di, and Mingyang Li. Fmgs: Foundation model embedded 3d gaussian splatting for holistic 3d scene understanding. In IJCV, pages 1–17. Springer, 2024. 2
- [66] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Ewa volume splatting. In VIS, 2001. 4

## Supplementary Material

#### 1. Implementation Details

To train 3DGS, we follow [18] to initialize the 3D coordinates of Gaussian blobs using Structure-from-Motion (SfM) points. The training hyperparameters are the same as those in [18]. We employ pretrained GroundedSAM [36] and the Suzuki-Abe algorithm [44] as the boundary detector in Boundary Guidance. The pretrained GroundingDINO [22] is used as the box detector in the Box-Focused Sampling strategy. Note that GroundingDINO is integrated as part of GroundedSAM. Therefore, in practice, only GroundedSAM needs to be executed to obtain both predictions. For the detection tool, we utilize FCAF3D [37] implemented in MMDetection3D [8]. The training hyperparameters are the same as those in FCAF3D. In our ablation study, to ensure a fair comparison, all model versions are trained with the same hyperparameters, such as the same number of epochs, specifically 12 epochs. All the ablation experiments (Sec. 4.3 in the main paper) are conducted on ScanNet.

#### 2. Additional Ablation Studies

##### 2.1.TheEffectofBoundaryGuidanceontheSpatial Distribution of 3DGS Blobs

Quantitative Study. To investigate the effect of Boundary Guidance on the spatial distribution of Gaussian blobs, we conducted an ablation study utilizing the first 100 scenes of ScanNet. Specifically, we evaluated the spatial distribution of Gaussian blobs by calculating the Root Mean Square Error (RMSE) metric [61] between the Gaussian blob positions and the ground-truth point clouds. Without Boundary Guidance, the average RMSE was 56.08%, whereas incorporating Boundary Guidance reduced the RMSE to 53.99%, achieving a significant improvement of 2.09 points. This result demonstrates that Boundary Guidance effectively improves the spatial distribution of Gaussian blobs.

Qualitative Study. In addition to the quantitative evaluation, we provide a qualitative analysis of the spatial distribution of Gaussian blobs in Fig. 3. Gaussian blobs trained with Boundary Guidance demonstrate clearer spatial distributions and more distinct differentiation between objects and the background. These qualitative results visually confirm the quantitative findings above, illustrating the effectiveness of Boundary Guidance in improving the spatial distribution of Gaussian blobs.

##### 2.2. Rendered Images from 3DGS Trained with Boundary Guidance

We present rendered images from different views generated by 3DGS trained with Boundary Guidance in Fig. 4 and Fig. 5. As shown, the category-specific boundaries

Method mAP@0.25 mAP@0.5

NeRF-Det [52] 53.3 29.7 NeRF-Det++ [16] 53.9 30.0 3DGS-DET (FF) 58.5 (+4.6) 31.8 (+1.8)

Table 1. When applied to the feed-forward 3DGS, the performance of our method (‘3DGS-DET (FF)’ still remains significantly ahead.

are clearly rendered and exhibit multi-view consistency, demonstrating that the 3D representation effectively incorporates the priors from Boundary Guidance.

##### 2.3. Qualitative Analysis of Box-Focused Sampling for 3DGS Blobs

Fig. 6 visually demonstrates the effect of Box-Focused Sampling. As shown, Box-Focused Sampling enhances the preservation of object blobs while effectively reducing noisy background blobs.

##### 2.4. Comparison with SfM-based 3D Detection Baseline

We implement the baseline combining SfM [39, 40] points with the FCAF3D [37]. As shown in Tab. 2, our method significantly outperforms the SfM-Detector by 11.1 points regarding mAP@0.25 and 8.1 points regarding mAP@0.5. The performance gap primarily stems from SfM reconstructions’ sparse point distributions. In contrast, our 3DGS with Boundary Guidance and Box-Focused Sampling promote superior spatial distribution of dense Gaussian blobs while suppressing background noise, ultimately yielding significant performance gains.

##### 2.5. Application to Feed-forward 3DGS

Vanilla 3DGS [18] optimizes each training scene individually, whereas the detector is trained on the entire training dataset. This fundamental difference in the optimization process prevents end-to-end training. However, to achieve end-to-end training, our method is flexible to be applied not only to Vanilla 3DGS but also to more recent feed-forward 3DGS methods [7], which are trained on entire datasets. We specifically applied our designs to the feed-forward 3DGS in an end-to-end integration manner and still achieved much better results than NeRF-Det++ [16], as shown in Tab. 1.

##### 2.6. Efficiency analysis

When applied to the feed-forward 3DGS, our method (‘3DGS-DET (FF)’ in Tab. 1) completes the entire training phase in 11 hours and the testing phase in 75 seconds on ScanNet, delivering significant performance improvements over NeRF-Det++ with an acceptable computational cost. All timing statistics were measured on two A800 GPUs.

###### Pobj mAP@0.25 mAP@0.5 SfM-Detector 48.8 29.7 Ours 59.9 (+11.1) 37.8 (+8.1)

- Table 2. Comparison with SfM-based 3D Detection Baseline [37, 39, 40].

#### 3. Additional Main Results

##### 3.1. Additional Quantitative Results

In Tab. 3, we compare the mAP@0.5 performance of various methods on the ScanNet dataset. The methods are grouped into three blocks based on their underlying representations. The first two blocks include methods that utilize non-view-synthesis representations: the first block consists of methods based on point clouds [32, 37, 46], while the second block focuses on multi-view image methods [38, 41, 45]. The third block includes methods that employ view-synthesis representations, including NeRF-based methods [16, 52] and our 3DGS-based method. Among non-view-synthesis representations, our method achieves clearly superior performance compared to methods utilizing multi-view images (the second block), demonstrating the effectiveness of our 3DGS reconstruction for detection tasks with multi-view image inputs. As for the view-synthesis representations, our 3DGS-DET outperforms the NeRFbased method NeRF-Det++ [16] by a significant margin of 7.8 points, further highlighting the advantages of our 3DGS-DET.

##### 3.2. Additional Qualitative results

We provide more qualitative results in Fig. 1 and Fig. 2. As illustrated, our method detects more objects in the scene with higher positional accuracy than NeRF-Det [52], demonstrating the advantages of our method. Note that NeRF-Det++ [16] had not released pretrained models, code, or qualitative results on the official GitHub page by the time of our submission. Therefore, we could not provide a qualitative comparison with NeRF-Det++ here.

#### 4. Failure Cases

We provide failure cases in Fig. 7. In scenes with severe occlusions, our method exhibits a small number of missed detections. These missed detections are caused by occlusions, which introduce inherent ambiguity in multi-view detection when key visual evidence is obscured.

#### 5. Limitation and Future Work

While 3DGS-DET achieves significant improvements over strong alternative methods, several limitations remain for further exploration. As the first work to introduce 3DGS into indoor 3D Object Detection (3DOD), this work mainly

focuses on the primary stage of this pipeline: empowering 3DGS for 3DOD. Extensive experiments demonstrate that our designs can lead to significant improvements. Beyond empowering the 3DGS representation, a subsequent detector specifically designed for 3DGS could hold promise in the future.

Methods cab bed chair sofa tabl door wind bkshf pic cntr VoteNet [32] 8.1 76.1 67.2 68.8 42.4 15.3 6.4 28.0 1.3 9.5 FCAF3D [37] 35.8 81.5 89.8 85.0 62.0 44.1 30.7 58.4 17.9 31.3 CAGroup3D [46] 41.4 82.8 90.8 85.6 64.9 54.3 37.3 64.1 31.4 41.1 ImGeoNet [45] 15.8 74.8 46.5 45.7 39.9 8.0 2.9 32.9 0.3 7.9 CN-RMA [41] 21.3 69.2 52.4 63.5 42.9 11.1 6.5 40.0 1.2 24.9 ImVoxelNet [38] 8.9 67.1 35.0 33.1 30.5 4.9 1.3 7.0 0.1 0.9 NeRF-Det [52] 12.0 68.4 47.8 58.3 42.8 7.1 3.0 31.3 1.6 11.6 NeRF-Det++ [16] - - - - - - - - 3DGS-DET (Our basic pipeline) 18.5 73.5 44.6 61.9 42.2 9.3 5.6 28.7 2.3 2.0 3DGS-DET (Our basic pipeline+BG) 16.1 77.0 51.6 62.4 44.7 11.7 11.3 24.4 1.7 19.0 3DGS-DET (Our basic pipeline+BG+BS) 19.2 73.8 52.7 65.2 46.2 9.6 8.2 31.8 4.2 20.9 Methods desk curt fridg showr toil sink bath ofurn mAP@0.5

VoteNet [32] 37.5 11.6 27.8 10.0 86.5 16.8 78.9 11.7 33.5 FCAF3D [37] 53.4 44.2 46.8 64.2 91.6 52.6 84.5 57.1 57.3 CAGroup3D [46] 63.6 44.4 57.0 49.3 98.2 55.4 82.4 58.8 61.3

ImGeoNet [45] 43.9 4.3 24.0 2.0 68.8 24.5 61.7 17.4 28.9 CN-RMA [41] 51.4 19.6 33.0 6.6 73.3 36.1 76.4 31.5 36.8 ImVoxelNet [38] 35.5 0.6 22.1 4.5 67.7 18.9 60.2 10.1 22.7

NeRF-Det [52] 46.0 5.8 26.0 1.6 69.0 25.5 55.8 21.1 29.7 NeRF-Det++ [16] - - - - - - - - 30.0 3DGS-DET (Our basic pipeline) 53.5 18.1 30.7 3.4 77.0 29.0 68.3 24.2 34.1 3DGS-DET (Our basic pipeline+BG) 47.4 27.2 30.4 8.3 87.0 36.3 78.3 28.8 36.9 3DGS-DET (Our basic pipeline+BG+BS) 52.4 22.2 36.9 15.7 82.6 35.1 74.0 28.9 37.8 (+7.8)

- Table 3. Comparison of mAP@0.5 on ScanNet. The first two blocks include methods that utilize non-view-synthesis representations: the first block consists of methods based on point clouds [32, 37, 46], while the second block focuses on multi-view image methods [38, 41, 45]. The third block includes methods that employ view-synthesis representations, including NeRF-based approaches and our 3DGS-based method. Among non-view-synthesis representations, our approach clearly outperforms methods that utilize multi-view images (the second block), demonstrating the superiority of our 3DGS reconstruction for detection with multi-view image inputs. In terms of view-synthesis representations, our 3DGS-DET significantly surpasses the NeRF-based method NeRF-Det++ [16] by 7.8 points. NeRF-Det++ [16] does not report per-category performance of mAP@0.5. We report its overall mAP@0.5 following the original paper [16].

NeRF-Det 3DGS-DET(Ours)

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

- Figure 1. More qualitative comparisons. Our method detects more objects with better positional precision, highlighting the advantages of our approach over NeRF-Det [52]. In this figure, the scene is represented using mesh to clearly display the boxes. Note that black and white boxes indicate predictions with incorrect categories, while boxes of other colors represent predictions with the correct categories.

NeRF-Det 3DGS-DET(Ours)

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

- Figure 2. More qualitative comparisons. Our method detects more objects with better positional precision, highlighting the advantages of our approach over NeRF-Det [52]. In this figure, the scene is represented using mesh to clearly display the boxes. Note that black and white boxes indicate predictions with incorrect categories, while boxes of other colors represent predictions with the correct categories.

3DGS w/o Boundary Guidance 3DGS w/ Boundary Guidance

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

- Figure 3. Analysis on the effect of Boundary Guidance. Gaussian blobs trained with Boundary Guidance exhibit clearer spatial distribution and more distinct differentiation between objects and background. Note that we visualize only the positions of the Gaussian blobs to highlight their spatial distribution, while omitting other attributes for clarity.

[Figure 125]

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

###### Figure 4. Rendered images from different views by 3DGS trained with Boundary Guidance. The category-specific boundaries are clearly rendered and exhibit multi-view consistency, demonstrating that the 3D representation has successfully embedded the priors provided by Boundary Guidance.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

###### Figure 5. Rendered images from different views by 3DGS trained with Boundary Guidance. The category-specific boundaries are clearly rendered and exhibit multi-view consistency, demonstrating that the 3D representation has successfully embedded the priors provided by Boundary Guidance.

Box-Focused Sampling

Random Sampling

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

- Figure 6. Qualitative Analysis of Box-Focused Sampling for 3DGS Blobs. Box-Focused Sampling significantly retains more object blobs and reduces noisy background blobs. Note that we visualize only the positions of the Gaussian blobs to highlight their spatial distribution, while omitting other attributes for clarity.

[Figure 180]

|[Figure 181]<br><br>|Failure Case|
|---|
|
|---|

- Figure 7. Failure cases. In scenes with severe occlusions, our method exhibits a small number of missed detections. These failure cases are caused by occlusions, which introduce inherent ambiguity in multi-view detection when key visual evidence is obscured.

