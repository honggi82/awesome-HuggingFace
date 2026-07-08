## Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency

Tianqi Liu1,2,∗ Zihao Huang1,2,∗ Zhaoxi Chen2 Guangcong Wang3 Shoukang Hu2 Liao Shen1,2 Huiqiang Sun1,2 Zhiguo Cao1 Wei Li2,† Ziwei Liu2,† 1Huazhong University of Science and Technology, 2S-Lab, Nanyang Technological University, 3Great Bay University https://free4d.github.io/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

∆𝑡𝑡

∆𝑡𝑡

[Figure 7]

[Figure 8]

# arXiv:2503.20785v1[cs.CV]26Mar2025

Free4D

or

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

“A rabbit playing colorful balloon.”

|[Figure 15]<br><br>|
|---|

|[Figure 16]<br><br>|
|---|

|[Figure 17]<br><br>|
|---|

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

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

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

[Figure 52]

[Figure 53]

Figure 1. Free4D can generate diverse 4D scenes from single-image or textual input. By enforcing spatial-temporal consistency in a tuning-free way, Free4D enables high-quality scene generation with explicit 4D controls.

#### Abstract

inconsistencies while fully leveraging the generated information. The resulting 4D representation enables real-time, controllable rendering, marking a significant advancement in single-image-based 4D scene generation.

We present Free4D, a novel tuning-free framework for 4D scene generation from a single image. Existing methods either focus on object-level generation, making scene-level generation infeasible, or rely on large-scale multi-view video datasets for expensive training, with limited generalization ability due to the scarcity of 4D scene data. In contrast, our key insight is to distill pre-trained foundation models for consistent 4D scene representation, which offers promising advantages such as efficiency and generalizability. 1) To achieve this, we first animate the input image using image-to-video diffusion models followed by 4D geometric structure initialization. 2) To turn this coarse structure into spatial-temporal consistent multi-view videos, we design an adaptive guidance mechanism with a point-guided denoising strategy for spatial consistency and a novel latent replacement strategy for temporal coherence. 3) To lift these generated observations into consistent 4D representation, we propose a modulation-based refinement to mitigate

#### 1. Introduction

Creating a dynamic 3D environment that closely mirrors the real world is crucial for achieving realistic and immersive digital experiences, a key objective in fields such as film production, video games, and augmented reality. However, captured images provide only limited snapshots of a scene. Generating dynamic 3D scenes from such limited observations, particularly from a single image, remains a significant challenge and an open research problem.

Existing 4D generation methods [4, 49, 50, 81] primarily focus on objects, often neglecting background generation and its dynamics. Recently, several studies [58, 71, 75, 80] have explored 4D scene generation for real-world applications. Compared to single-object generation, scene-level 4D generation presents greater challenges, requiring handling complex geometries, spatial relationships, and dynamic interactions. One line of research relies on fine-tuned video

∗Equal contribution. †Corresponding authors.

diffusion models to get temporally varying multi-view data for 4D representation fitting. By decoupling spatial and temporal dimensions [58] or enforcing consistency alternately [71], these methods aim to generate coherent multiview videos, which can be optimized to construct a 4D representation of the scene. Some works [80] further introduce data curation pipelines to generate 4D scene data for training models. However, the performance of these methods is highly dependent on the quality and scale of generated 4D scene data. Moreover, they require substantial data and computational resources to fine-tune large video diffusion models. Another line of research leverages priors from existing generative models to optimize a 4D representation, avoiding the high costs associated with fine-tuning diffusion models. A pioneering work [75] achieves text-to-4D scene generation by distilling priors from a closed-source video diffusion model [38] using score distillation sampling (SDS) [44], producing impressive results. However, it inherits common limitations of SDS, including lengthy optimization, oversaturated colors [68, 69], and limited diversity [68, 69] in the generated outputs.

To overcome these limitations, we propose Free4D, a 4D scene generation method from a single image with explicit spatial-temporal consistency in a data-efficient and tuning-free manner. To obtain a 4D representation from a single image, a straightforward solution would be generating a multi-view video from the given image and then optimize a 4D representation based on it. However, this approach presents two key challenges: 1) How to generate a multi-view video from a single image while ensuring high spatial-temporal consistency. 2) How to effectively optimize a coherent 4D representation despite inevitable minor inconsistencies in the generated multi-view video.

To address the first challenge, we adopt the dynamic reconstruction method [77], enhanced by progressive background point cloud aggregation strategy. This approach enables accurate initialization of a coherent 4D geometric structure, thus ensuring geometric alignment for subsequent generation. Subsequently, guided by the established 4D structure, we employ a point-conditioned diffusion model [76] to generate a multi-view video. To enhance spatial consistency, we introduce two strategies: an adaptive classifier-free guidance (CFG) approach, designed to maintain consistent appearance across different viewpoints, and point cloud guided denoising, aimed at reducing unintended motions of dynamic subjects in the synthesized views. Although these strategies notably improve spatial consistency, significant temporal inconsistencies remain, primarily due to the generative model’s tendency to produce inconsistent content in occluded or missing regions over time. To mitigate this, we propose reference latent replacement, a technique that substantially enhances temporal coherence, ensuring smoother and more consistent video

content over time.

With these advancements, the generated multi-view video achieves near-complete spatial-temporal consistency. However, subtle inconsistencies persist, posing challenges in constructing a fully coherent 4D representation. To overcome this second challenge, we introduce an effective optimization strategy designed to seamlessly integrate multiview videos into a unified 4D representation. Our approach begins by constructing a coarse 4D representation, utilizing only those images that share the same timestamp or viewpoint as the input image. To further refine this representation, we incorporate a modulation-based refinement, leveraging additional generated images while effectively suppressing inconsistencies. The resulting 4D representation enables real-time, controllable spatial-temporal rendering, ensuring both fidelity and coherence across views and time.

Our main contributions can be summarized as follows:

- • We present Free4D, the first tuning-free pipeline for 4D scene generation from a single image, delivering photorealistic appearances and realistic motions.
- • We employ a dynamic point-conditioned multi-view video generation approach, integrating carefully designed techniques to enhance spatial-temporal consistency.
- • We introduce a coarse-to-fine training strategy combined with a modulation-based refinement to effectively integrate the generated information while reducing inconsistencies, yielding a consistent 4D representation.

#### 2. Related Work

Video Generation. Pioneering studies achieve dynamic video generation with VAEs [3, 14, 27, 30, 37, 48] or GANs [13, 61, 62, 65], while facing limitations in temporal stability and output resolution. Subsequent methods revolutionize this field by extending image diffusion models with 3D UNet architecture [22] or auto-regressive manners [18]. Following advancements introduced controllable frameworks that enabled synthesis guided by texts [21, 23, 24, 53], images [2, 6, 32, 41] or viewpoints [5, 29] through conditional denoising techniques. Recent advancements focus on enhancing the realism and detail of generated videos [7, 9, 16, 17, 19, 36, 66], capturing intricate details and natural movements, making generated videos more lifelike and engaging. Despite these successes, video generation inherently lacks explicit 3D scene structures or support for viewpoint manipulation—critical gaps for 3D / 4D spatial-temporal modeling tasks.

3D / 4D Generation. Early explorations in 3D generation focused on static objects through point clouds [1, 11, 15] or implicit neural representations [12, 43]. Extending to 3D scene generation, SceneDreamer [10] leveraged neural radiance fields (NeRF) [39] for unbounded outdoor generation, while InfiniCity [33] proposed scalable pipelines for photorealistic urban scenes. However, these methods

primarily focus on static scene generation and lack support for dynamic spatial-temporal modeling. Subsequent methods [42, 47, 49, 50, 54, 57, 74, 79, 81] further introduce temporal deformations, enabling controllable 4D generation of single objects. Recent approaches [58, 80] attempted to unify dynamic objects and environmental interactions through space-time neural representations. However, these methods heavily depend on large-scale, highquality 4D training data, which are labor-intensive to acquire and often restrict real-world applicability. The method most closely related to ours is 4Real [75], which only supports text conditions and relatively low resolution.

4D Reconstruction. Neural Radiance Fields (NeRF) [39, 45] represents 3D scenes via implicit neural representations, while 3D Gaussian Splatting [25, 28] later introduces explicit, real-time 3D primitives. Extending these to 4D, recent advances propose dynamic 3D Gaussians, where Gaussian attributes evolve via deformation fields [70, 72]. Further innovations [35, 56] optimize spatiotemporal Gaussian kernels directly from RGB inputs. While these methods achieve photo-realistic 4D reconstruction, they remain tightly coupled with high-fidelity 4D training data, which limits scalability for real-world dynamic scenes.

#### 3. Preliminaries

Latent Diffusion Model (LDM) [51] is a computationally efficient variant of diffusion models that both the forward and the reverse process are performed in the latent space. Given an image x0, it is first encoded into a latent representation z0 = E(x0) using a VAE encoder E. The forward process progressively adds noise ϵ, formulated as:

zi = 1 − βizi−1 + βiϵ, (1)

where βi ∈ (0,1) represents the noise schedule at time step i. The cumulative noise follows the closed-form expression:

zi = √α¯iz0 + √1 − α¯iϵ, (2)

where α¯i = i1(1−βi). The reverse process removes noise from latent. We adopt DDIM [55], given by:

zi−1 = √α¯i−1z0←i + 1 − α¯i−1ϵθ(zi,i), (3)

where ϵθ(zi,i) denotes the predicted noises. Combining Eq. (2) and Eq. (3), the denoising process are rewritten as:

zi−1 = aizi + biz0←i, (4)

√α¯iai, and z0←i = (zi −

1−α¯i ,bi = √α¯i−1 −

where ai = 1−α¯

i−1

√1 − α¯iϵθ(zi,i))/√α¯i. Eq. (4) indicates that the denoising direction is determined by z0←i. Finally, the generated image is obtained via the VAE decoder: xˆ = D(z0).

#### 4. Free4D

Given a single scene image I, we aim to derive feasible 4D Gaussian representations with spatial-temporal consistency in a tuning-free approach, which enables the synthesis of high-fidelity novel views in free viewpoints at minimal cost. To achieve this, we start by converting the input image into a video V using an off-the-shelf image-to-video generator and initializing the associated 4D geometric structures P. Then, we produce spatial-temporal consistent multi-view videos S using a point-conditioned diffusion model with guidance from the obtained 4D geometric structures. The final 4D representation R is optimized via a proposed training strategy to further improve spatial-temporal consistency.

##### 4.1. 4D Geometric Structure Initialization

With an image-to-video generative model [59], we first animate the input scene image I into a reference single-view video V = {I(t,1)}Tt=1. Here, an image at time t and viewpoint k in the multi-view video is denoted as I(t,k). To initialize 4D geometric structures from the reference singleview video V, we employ point clouds P = {Pt}Tt=1 as explicit representations. This is crucial for enhancing geometric consistency and camera control capabilities for 4D scene generation. Specifically, we apply a dynamic scene reconstruction method MonST3R [77], which takes the reference video V as inputs and produces world-coordinate pointmaps {pt}Tt=1. Simultaneously, per-frame static masks {mst}Tt=1 for distinguishing invariant regions within each image are estimated. To effectively integrate geometry information within the reference video, we convert initial pointmaps into well-organized point clouds. Considering that directly using naive pointmaps might neglect some cross-frame geometry information caused by occlusions, we decompose pointmaps into static and dynamic components using static masks {mst}Tt=1. We aim to keep static components consistent across frames while dynamic components vary over time in per-frame point clouds. One might concatenate all static regions in all frames straightforwardly, but this leads to redundant points and inefficiency in the subsequent rendering process. Therefore, we propose a progressive strategy that can effectively aggregate static components. We initialize point clouds from static regions in the first frame, P1s = p1 ⊙ ms1, given that the first frame (i.e., the input scene image I) contains the highest confidence and quality. We progressively update P1s by propagating static regions from subsequent frames while avoiding redundancy:

Pts = Pts−1 ∪ (pt ⊙ mˆ st), (5)

where mˆ st = mst ∩ (1 − ti=1−1 msi) and ⊙ denotes elementwisely indexing none-zero values. This ensures a compact

yet complete representation of the static point cloud while maintaining alignment and consistency across frames. The

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

|[Figure 60]<br><br>|[Figure 61]<br><br>|[Figure 62]<br><br>|
|---|---|---|
|[Figure 63]<br><br>2|3<br><br>[Figure 64]<br><br>|[Figure 65]<br><br>|
|[Figure 66]<br><br>2|3<br><br>[Figure 67]|[Figure 68]|

|[Figure 69]<br><br>2|[Figure 70]<br><br>3|[Figure 71]<br><br>|Coarse<br><br>Loss|
|---|---|---|---|
|[Figure 72]<br><br>|[Figure 73]<br><br>|[Figure 74]<br><br>| |
|[Figure 75]<br><br>2|3<br><br>[Figure 76]<br><br>|[Figure 77]| |

1 2 3

1 2 3

[Figure 78]

𝐼𝐼′(𝑡𝑡,𝑘𝑘)

VDM

|[Figure 79]<br><br>| |
|---|---|
|[Figure 80]<br><br>[Figure 81]<br><br>| |
| | |

1

[Figure 82]

or

1

ℛ′

“Fox playing 4D-GS video games.”

3 2

𝓔𝓔

𝐼𝐼(𝑡𝑡,1)

1

[Figure 83]

1

1

[Figure 84]

[Figure 85]

[Figure 86]

MonST3R

[Figure 87]

[Figure 88]

### 𝓔𝓔

𝓔𝓔

𝐼𝐼(𝑡𝑡,𝑘𝑘)

1 2 3 1 2 3 1 2 3

[Figure 89]

[Figure 90]

[Figure 91]

|Cond|
|---|

[Figure 92]

𝓔𝓔

[Figure 93]

[Figure 94]

CFG

[Figure 95]

[Figure 96]

1 2 3 1 2 1 2 3

[Figure 97]

[Figure 98]

[Figure 99]

𝓔𝓔

3

[Figure 100]

[Figure 101]

[Figure 102]

× 𝑁𝑁

[Figure 103]

ε1 ε2

× 𝑁𝑁

1 1 1 1 1

𝑧𝑧0

[Figure 104]

[Figure 105]

t Ps

[Figure 106]

[Figure 107]

|1|𝓔𝓔<br><br>[Figure 108]<br><br>[Figure 109]|
|---|---|
| | |

𝑧𝑧0←𝑖𝑖

[Figure 110]

Static Dynamic

[Figure 111]

- 1 1 1

|[Figure 112]|
|---|

1

|[Figure 113]<br><br>|[Figure 114]<br><br>|[Figure 115]<br><br>|
|---|---|---|
|[Figure 116]<br><br>2|3<br><br>[Figure 117]<br><br>|[Figure 118]<br><br>|
|[Figure 119]<br><br>2|3<br><br>[Figure 120]|[Figure 121]<br><br>|

- 1 2 3

[Figure 122]

[Figure 123]

𝑝𝑝𝑡𝑡 ⨀ 𝑚𝑚𝑡𝑡𝑠𝑠 𝑝𝑝𝑡𝑡 ⨀ 𝑚𝑚𝑡𝑡𝑑𝑑

𝑧𝑧̃0←𝑖𝑖

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

𝓓𝓓

[Figure 128]

𝓓𝓓

[Figure 129]

[Figure 130]

1

[Figure 131]

𝐼𝐼′(𝑡𝑡,𝑘𝑘0)

[Figure 132]

Loss

1

𝐼𝐼(𝑡𝑡,𝑘𝑘)

3 2

[Figure 133]

3 2

[Figure 134]

[Figure 135]

ℛ

Fine 4D-GS

[Figure 136]

𝐼𝐼′(𝑡𝑡,2) 𝐼𝐼′(𝑡𝑡,3)

1

[Figure 137]

1

[Figure 138]

4D Geometric Structure Initialization Spatial-Temporal View Generation Consistent 4D-GS Optimization

Figure 2. Overview of Free4D. Given an input image or text prompt, we first generate a dynamic video V = {I(t, 1)}Tt=1 using an off-the-shelf video generation model [59]. Then, we employ MonST3R [77] with a progressive static point cloud aggregation strategy for dynamic reconstruction, obtaining a 4D geometric structure. Next, guided by this structure, we render a coarse multi-view video S′ = {{I′(t, k)}Tt=1}Kk=1 along a predefined camera trajectory and refine it into S = {{I(t, k)}Tt=1}Kk=1 using ViewCrafter [76]. To ensure spatial-temporal consistency, we introduce Adaptive Classifer-Free Guidance (CFG) and Point Cloud Guided Denoising for spatial coherence, along with Reference Latent Replacement for temporal coherence. Finally, we propose an efficient training strategy with a Modulation-Based Refinement to lift the generated multi-view video S into a consistent 4D representation R.

tent. 2) In terms of temporal consistency, it generates noticeable discrepancies between consecutive frames, leading to temporal flickering and a lack of smooth transitions. We tackle these issues in the following ways.

dynamic components in each frame are kept in their corresponding pointmaps. Thus, the point cloud {Pt}Tt=1 is given by Pt = PTs ∪ (pt ⊙ mdt), where mdt = 1 − mst.

##### 4.2. Spatial-Temporal View Generation

Geometry-informed Adaptive Denoising. In ViewCrafter, using the naive classifier-free guidance (CFG) [20] tends to accumulate numerical errors and cause over-saturation problems [40]. Disabling CFG by setting the guidance scale to 1 would generate low-quality results or even failure to complete in some scenarios (Sec. 5.3). We propose an Adaptive CFG strategy by deactivating CFG in regions where the point cloud rendering is visible (i.e., M(t,k) = 1) and compute the predicted noise as follows:

Due to the scarcity of 4D data, there is no available offthe-shelf multi-view video generator. Therefore, we propose a tuning-free approach to generate multi-view videos S that maintain robust spatial-temporal consistency, with the guidance of the obtained 4D geometry structures, i.e., point clouds P = {Pt}Tt=1. We consider rendering the point clouds from different camera poses to create a sequence of coarse multi-view videos S′ = {{I′(t,k)}Tt=1}Kk=1 together with visibility masks M = {{M(t,k)}Tt=1}Kk=1, with navigation of a user-defined camera trajectory (K camera poses). Despite capturing geometric structures and view relationships effectively, the coarse multi-view videos still face challenges such as occlusions, missing regions, and diminished visual fidelity.

###### ϵ1 = ϵθ(zi,c), (6)

where zi represents the latent of a specific image I(t,k) at denoising timestep i, ϵθ denotes the denoising network, and c is the condition, specifically the image I(t,1) and a default prompt text used by [76]. On the contrary, for occluded or missing regions (i.e., M(t,k) = 0), we enable CFG and compute the noise as:

To mitigate these issues, we employ ViewCrafter [76], a point-conditioned diffusion model, to refine the coarse multi-view videos. However, simply using ViewCrafter cannot guarantee strong spatial-temporal consistency in refined multi-view videos. There are two main problems: 1) From a spatial consistency perspective, it fails to maintain uniform color tones across frames and introduces unexpected motion artifacts in scenes with highly dynamic con-

ϵ2 = ϵθ(zi) + s · (ϵθ(zi,c) − ϵθ(zi)), (7)

where s is the guidance scale. Thus, the final estimated noise is obtained by a noise fusion at each denoising step :

ϵ = M(t,k) · ϵ1 + (1 − M(t,k)) · ϵ2. (8)

Moreover, we propose Point Cloud Guided Denoising by leveraging the coarse multi-view video S′ to guide the early denoising process. Specifically, we encode the specific image I′(t,k) in S′ into latent representations:

z0′ = E(I′(t,k)), (9) At an early denoising timestep i, we first apply the forward diffusion process to z0′ to obtain noisy latent zi′ following Eq. (2). We then fuse zi′ with the model-predicted latent zi based on the point cloud rendered mask m = M(t,k), which is given by:

zˆi = m · zi′ + (1 − m) · zi. (10)

By employing this adaptive approach that leverages information from point cloud renders S′ to guide the denoising process, we effectively mitigate color inconsistencies, reduce unexpected dynamic motion, and enhance spatial consistency across views.

Consistent Temporal Latent Replacement. We further refine the point cloud renders S′ at different timestamps to enhance temporal consistency in multi-view generation by proposing a Reference Latent Replacement strategy. Specifically, for a specific timestamp tj > 1, we use the multi-view images {I(1,k)}Kk=1 generated from the first frame as a reference. For simplicity, we illustrate the following process using a specific kj ∈ [1,K]. For generating image I(tj,kj), we use the first-timestamp I(1,kj) as a reference. In regions where both I(tj,kj) and I(1,kj) require completion, i.e., M(tj,kj) = 0 and M(1,kj) = 0, we replace the latent in these areas with those from the reference. Specifically, we first encode I(1,kj) into a latent as:

z0ref = E(I(1,kj)). (11) At a denoising timestep i, we first apply the forward diffusion process to z0ref to obtain noisy latent ziref following Eq. (2) and predict the latent zi for I(tj,kj). We then fuse ziref with zi based on the co-visible mask:

mˆ = (1 − M(tj,kj)) · (1 − M(1,kj)). (12) The replaced latent zˆi is given by:

zˆi = mˆ · ziref + (1 − mˆ ) · zi. (13)

This approach preserves consistency in the generated content over time, effectively reducing discrepancies and producing multi-view videos that achieve nearly-consistency in both temporal and spatial dimensions.

##### 4.3. Consistent 4D-GS Optimization

Given generated spatial-temporal consistent multi-view videos, we optimize the corresponding 4D-GS representations. Due to the high-dynamic property of generated multiview videos, directly using a standard training pipeline with

the multi-view video as supervision would cause misalignment and inconsistency in the final 4D-GS. Therefore, we propose an effective training strategy to integrate the information from the generated multi-view videos for consistent 4D-GS optimization. Our key insight is that the consistency between the reference video {I(t,1)}Tt=1 at k = 1 and the generated multi-view images {I(1,k)}Kk=1 at the first timestamp t = 1 is relatively high, as both are constrained by the input image. Thus, we first utilize these views to train a coarse 4D-GS R′. Then, we incorporate the missing information from the rest of the multi-views to obtain a refined 4D-GS R. However, it is difficult to extract useful information while preventing the propagation of inconsistencies into the 4D-GS representation. Instead of using generated images for pixel-level supervision directly, we integrate generated information into the 4D representation at a higher level. Specifically, we first render the coarse 4D-GS at specific tj and kj to obtain a rendered image Ir. We then apply the forward process of the diffusion model by adding noise to obtain the noisy renderings zTr¯, where T¯ is a predefined timestep. During the denoising stage, inspired by [63], we introduce a Modulation-Based Refinement for effective enhancement. We use the generated image I(tj,kj) as a modulation signal at each timestep, guiding the denoising process toward the desired generated context. In Eq. (4), since z0r←i (at the denoising timestep i ) estimates the noisefree latent of the rendered image and dictates the denoising direction, we propose integrating the information from the generated image into this process to adjust the denoising direction. The generated image is first encoded into a latent as z0 = E(I(tj,kj)), and the adjusted process is given by:

z˜0←i = wiγiz0 + (1 − wi)z0←i, (14) where γi = std(z

0←i)

std(z0) serves as a scaling factor to mitigate over-exposure [34, 63], while wi is a predefined weight that regulates the influence of the generated image on the denoising process. We replace the original z0←i with adjusted z˜0←i for the subsequent denoising process. This adjustment is applied at each denoising step to obtain the enhanced renderings, denoted as I˜r, which are used to improve the coarse 4D-GS and enhance both rendering quality and consistency. Loss Function. For the first timestamp (t = 1) or first viewpoint (k = 1), we use L1 loss:

Ll1 = ∥I(t,k) − Ir(t,k)∥1 (15)

where I(t,k) and Ir(t,k) represent the generated image and rendered image by 4D-GS, respectively. For other images (t > 1,k > 1), we use LPIPS loss [78], as:

Llpips = LPIPS(I˜r(t,k),Ir(t,k)) (16)

where I˜r(t,k) is the refined generated image. For the coarse stage, the total loss is L = Ll1 while for the fine stage, the total loss is L = Ll1 + λLlpips.

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

Free4D GenXD Free4D DimensionX

|[Figure 155]|[Figure 156]|
|---|---|
|[Figure 157]|[Figure 158]|

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

Free4D DimensionX

Free4D Animate124

Figure 3. Qualitative comparisons of image-to-4D. We present the results using the same single-image prompts.

Method Text Align Consistency Dynamic Aesthetic 4Real [75] 26.1% 95.7% 32.3% 50.9%

Method Consistency Dynamic Aesthetic Animate124 [79] 90.7% 45.4% 42.3%

Ours 26.1% 96.0% 47.4% 64.7% 4Dfy [4] 25.7% 91.6% 53.3% 54.5%

Ours 96.9% 40.1% 60.5% DimensionX [58] 97.2% 21.9% 56.0%

Ours 26.0% 96.9% 54.1% 61.9% D-in-4D [81] 25.0% 91.0% 53.5% 55.1%

- Ours 95.5% 22.1% 57.3%

GenXD [80] 89.8% 98.3% 38.0%

- Ours 96.8% 100.0% 57.9%

Ours 25.9% 95.2% 53.2% 65.3%

- Table 1. Text-to-4D comparisons on VBench [26]. We report the text alignment, consistency, dynamics, and aesthetics of the generated 4D videos. D-in-4D denotes Dream-in-4D [81].

Table 2. Image-to-4D comparisons on VBench [26]. We report the text alignment, consistency, dynamics, and aesthetics of the generated 4D videos.

[Figure 171]

project pages of the comparison methods. To evaluate the quality of multi-view videos rendered from 4D representations, we report common VBench [26] metrics: Consistency (average for subject/background), Dynamic Degree, Aesthetic Score, and Text Alignment (only for text-to-4D). Since there is no well-established benchmark in 4D generation field, we conduct a user study with 78 evaluators to enhance reliability. Details on these metrics and the user study are provided in Appendix B.

#### 5. Experiments

Baselines. Our baselines fall into two categories: text-to4D and image-to-4D methods. For text-to-4D, we compare our approach with 4Real [75], a state-of-the-art text-to4D scene generation method. We also include two widely used object-centric 4D generation methods: 4Dfy [4] and Dream-in-4D [81]. For image-to-4D, we compare our approach with two state-of-the-art generative models: DimensionX [58] and GenXD [80], both trained on large-scale datasets while our Free4D operates without tuning. We also include Animate124 [79], a state-of-the-art object-centric tuning-free method based on SDS [44]. Specifically, we use the same text or single-image prompts for generation. Since the official implementations of 4Real [75], DimensionX [58], and GenXD [80] have not been released, we report the results available from their respective project pages. GenXD only releases generated videos instead of videos rendered from the 4D representation. Therefore, we use a monocular reconstruction algorithm [73] to reconstruct the 4D representation and render it for comparison.

Implementation Details. We use [70] as our 4D representation. In the coarse stage, we first train for 9k iterations, followed by 1k iterations in the fine stage. We conduct all experiments on a single NVIDIA A100 (40GB) GPU. More details on the network, hyperparameter settings, and runtime are provided in Appendix A.

##### 5.1. Text-to-4D Comparisons

The qualitative comparisons are presented in Fig. 4, while the quantitative results on VBench [26] and the user study are shown in Table 1 and Fig. 6, respectively. On VBench (Table 1), our method is comparable to or outperforms 4Real [75] across all three dimensions, with notable improvements in Aesthetics and Dynamics. Compared to

Datasets and Metrics. The data used for evaluation, including single images and texts, are sourced from the

“A building on fire.”

“A cat singing.”

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

view

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

time

“A fox playing videogame.”

“A rabbit eating carrot.”

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

“A lemur holding and drinking boba.”

“A monster reading a book.”

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

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

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Free4D 4Real 4Dfy Free4D 4Real Dream-in-4D

Figure 4. Qualitative comparisons of text-to-4D. We show the results based on the same text prompts.

4D Structure Initialization (spatial-temporal consistency)

Adaptive CFG (spatial consistency)

𝐼𝐼(1,1) 𝐼𝐼(𝑡𝑡0, 1) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0)

𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0 + ∆𝑡𝑡, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0 + ∆𝑡𝑡, 𝑘𝑘0)

| |
|---|

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

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

input w/o MonST3R w/ MonST3R

input w/ CFG w/o CFG w/ Ada. CFG

Coarse-to-fine (consistency & appearance)

Guided Denoising (spatial consistency)

Latent Replacement (temporal consistency)

𝐼𝐼(𝑡𝑡0,1) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0)

𝐼𝐼(1,1) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0)

𝐼𝐼(1,1) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0)

| |
|---|

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

| |
|---|

| |
|---|

| |
|---|

input

coarse fine

input

w/o w/

input w/o w/

Modulation-based Refinement (consistency)

Ours vs. SDS (consistency & appearance)

Static Point Cloud Aggregation (storage)

𝐼𝐼(1,1) 𝐼𝐼′(1,𝑘𝑘0) 𝐼𝐼′(1,𝑘𝑘0)

𝐼𝐼(1,1) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0)

𝐼𝐼(1,1) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0) 𝐼𝐼(𝑡𝑡0, 𝑘𝑘0)

| |
|---|

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

| |
|---|

| |
|---|

input SDS Ours

input concat progressive

input

w/o w/

###### Figure 5. Qualitative Comparison of Ablation Studies.

###### Consistency

###### Dynamic

###### Aesthetic

4Real

14 86 95 91 95

30 70 74 91 89

18 82 96 92 85

4Dfy Dream-in-4D

5

26

4

9 5

43 11

8

DimensionX

15

GenXD Animate124

100 93

100 66

100 93

7

34

7

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

Win Rate% (vs. Ours)

Win Rate% (vs. Ours)

Win Rate% (vs. Ours)

###### Figure 6. Comparison of different methods based on the user study.

Method Consistency Dynamic Aesthetic wo / w MonST3R 14% / 86% 30% / 70% 9% / 91% wo / w Ada. CFG 14% / 86% 36% / 64% 25% / 75% wo / w PGD∗ 14% / 86% 11% / 89% 13% / 87% wo / w RLR† 24% / 76% 31% / 69% 17% / 83% wo / w Fine Stage 4% / 96% 21% / 79% 6% / 94% wo / w MBR‡ 5% / 95% 14% / 86% 6% / 94% SDS vs. Ours 8% / 92% 10% / 90% 9% / 91%

Table 3. User study results on ablations. PGD∗, RLR†, and MBR‡ refer to point cloud guided denoising, reference latent replacement, and modulation-based refinement, respectively.

object-level methods, our approach excels in Consistency and Aesthetics while performing comparably in Dynamics, slightly lower than Dream-in-4D [81]. This is mainly because object-level generation can more easily handle large viewpoint shifts, as it focuses only on simple single objects without complex textures or backgrounds. In contrast, our Free4D can generate more dynamic scenes with complex textures and backgrounds, as illustrated in Fig. 4. Furthermore, the user study (Fig. 6) provides strong evidence of Free4D’s superiority. Evaluators consistently found our generated results to be more advantageous across all four dimensions compared to other methods. This demonstrates that Free4D not only produces more aesthetically pleasing videos but also generates results with greater diversity, coherence, and realism. Overall, these findings highlight the effectiveness and robustness of our proposed Free4D in scene generation.

##### 5.2. Image-to-4D Comparisons

- Table 2 and Fig. 6 present the quantitative comparisons on VBench [26] and user studies, while qualitative results are shown in Fig. 3. Compared to GenXD [80], Free4D achieves more realistic and consistent free-viewpoint video reconstruction from a single image. This is more evident in the quantitative VBench [26] and is further corroborated by the qualitative results in Fig. 3. Compared to the object-centric Animate124 [79], our proposed scene-level Free4D not only incorporates the environment but also exhibits fewer artifacts and better temporal consistency. This highlights Free4D ’s ability to generate high-quality, coherent scenes with complex textures and backgrounds, a common challenge for object-centric methods. Furthermore, Free4D achieves results comparable to DimensionX [58] on the VBench [26] benchmark while outperforming it on user preferences. Evaluators in the user study consistently favored our Free4D for its diversity, consistency, and realism. This further underscores the effectiveness of our method in generating high-quality, free-viewpoint videos from single images without the need for tuning. More results can be found on our project page.

##### 5.3. Ablations and Analysis

We analyze our pipeline by systematically removing individual components and evaluating their impact. Fig. 5 presents the quantitative results, while Table 3 includes the corresponding user studies.

MonST3R provides effective 4D structure initialization. The integration of MonST3R [77] for 4D structure construction is crucial for preserving geometric and spatial consistency, outperforming [64] used by [76].

Adaptive CFG enhances view consistency. Standard CFG would introduce noticeable color shifts between views, sometimes leading to oversaturation, while disabling it weakens completion in missing regions. Our proposed Adaptive CFG achieves a well-balanced trade-off, enhancing consistency across views.

Point Cloud Guided Denoising mitigates unexpected motion. This technique stabilizes dynamic subjects, such as fluids, ensuring consistency across different views. Without this module, undesired fluid dynamics may occur.

Reference Latent Replacement is crucial for temporal consistency. Without it, generated results in occluded and missing regions exhibit significant variations across different time steps for the same viewpoint, leading to temporal inconsistencies and blurring.

Refinement with multi-view video significantly improves consistency and appearance. Without refinement, the coarse-stage results exhibit noticeable artifacts and blurring, while refinement greatly enhances overall quality.

Modulation-based Refinement aggregates generated information while preserving consistency. Direct supervision with generated multi-view videos can introduce temporal inconsistencies. Additionally, pixel-level SDS leads to unstable training and oversaturation in missing regions.

Progressive static point aggregation preserves background integrity while minimizing storage. Directly concatenating background point clouds from all frames leads to excessive data size and introduces ghosting artifacts.

#### 6. Conclusion

We introduce Free4D, the first tuning-free approach for generating consistent 4D scenes from a single image. Our approach begins with a 4D geometric structure construction module to initialize multi-view videos, followed by a pointbased generative model. To ensure spatial-temporal consistency, we incorporate adaptive classifier-free guidance and a point cloud guided denoising strategy for spatial coherence, along with reference latent replacement for temporal consistency. Finally, we apply an effective training strategy with a modulation-based refinement to lift the generated multiview video into a consistent 4D representation.

#### References

- [1] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas J. Guibas. Learning representations and generative models for 3d point clouds. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Workshop Track Proceedings. OpenReview.net, 2018. 2
- [2] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H. Campbell, and Sergey Levine. Stochastic variational video prediction. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. 2
- [3] Mohammad Babaeizadeh, Mohammad Taghi Saffar, Suraj Nair, Sergey Levine, Chelsea Finn, and Dumitru Erhan. Fitvid: Overfitting in pixel-level video prediction. arXiv preprint arXiv:2106.13195, 2021. 2
- [4] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas J. Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B. Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 7996–8006. IEEE, 2024. 1, 6, 13, 14
- [5] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. CoRR, abs/2412.07760, 2024. 2
- [6] Andreas Blattmann, Timo Milbich, Michael Dorkenwald, and Bjorn Ommer. Understanding object dynamics for interactive image-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5171–5181, 2021. 2
- [7] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 22563–22575. IEEE, 2023. 2
- [8] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 9630–9640. IEEE, 2021. 13
- [9] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 7310–7320. IEEE, 2024. 2
- [10] Zhaoxi Chen, Guangcong Wang, and Ziwei Liu. Scenedreamer: Unbounded 3d scene generation from 2d image

- collections. IEEE Trans. Pattern Anal. Mach. Intell., 45(12): 15562–15576, 2023. 2
- [11] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, Liang Pan, Dahua Lin, and Ziwei Liu. 3dtopia-xl: Scaling high-quality 3d asset generation via primitive diffusion. CoRR, abs/2409.12957, 2024. 2
- [12] Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-sdf: Conditional generative modeling of signed distance functions. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 2262–2272. IEEE, 2023. 2
- [13] Aidan Clark, Jeff Donahue, and Karen Simonyan. Adversarial video generation on complex datasets. arXiv preprint arXiv:1907.06571, 2019. 2
- [14] Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsm¨assan, Stockholm, Sweden, July 10-15, 2018, pages 1182–1191. PMLR, 2018. 2
- [15] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. GET3D: A generative model of high quality 3d textured shapes learned from images. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. 2
- [16] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 22873–

22884. IEEE, 2023. 2

- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 2
- [18] William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. Advances in Neural Information Processing Systems, 35:27953–27965, 2022. 2
- [19] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for highfidelity video generation with arbitrary lengths. CoRR, abs/2211.13221, 2022. 2
- [20] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 4
- [21] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey A. Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. CoRR, abs/2210.02303, 2022. 2

- [22] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2
- [23] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 2
- [24] Yaosi Hu, Chong Luo, and Zhenzhong Chen. Make it move: Controllable image-to-video generation with text descriptions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 18198–18207. IEEE, 2022. 2
- [25] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH 2024, Denver, CO, USA, 27 July 20241 August 2024, page 32. ACM, 2024. 3
- [26] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 21807–21818. IEEE, 2024. 6, 8, 13
- [27] Nal Kalchbrenner, A¨aron van den Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Video pixel networks. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, pages 1771–1779. PMLR, 2017. 2
- [28] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139:1– 139:14, 2023. 3, 13
- [29] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas J. Guibas, and Gordon Wetzstein. Collaborative video diffusion: Consistent multi-video generation with camera control. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. 2
- [30] Manoj Kumar, Mohammad Babaeizadeh, Dumitru Erhan, Chelsea Finn, Sergey Levine, Laurent Dinh, and Durk Kingma. Videoflow: A conditional flow-based model for stochastic video generation. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. 2
- [31] LAION-AI. aesthetic-predictor, 2022. 14
- [32] Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, and Ming-Hsuan Yang. Flow-grounded spatial-temporal video prediction from still images. In Computer Vision ECCV 2018 - 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part IX, pages 609–625. Springer, 2018. 2

- [33] Chieh Hubert Lin, Hsin-Ying Lee, Willi Menapace, Menglei Chai, Aliaksandr Siarohin, Ming-Hsuan Yang, and Sergey Tulyakov. Infinicity: Infinite-scale city synthesis. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 22751–

22761. IEEE, 2023. 2

- [34] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In WACV, pages 5404–5411, 2024. 5
- [35] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In International Conference on 3D Vision, 3DV 2024, Davos, Switzerland, March 18-21, 2024, pages 800–809. IEEE, 2024. 3
- [36] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. CoRR, abs/2303.08320, 2023. 2
- [37] Micha¨el Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016. 2
- [38] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7038– 7048, 2024. 2
- [39] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM, 65(1):99–106, 2022. 2, 3
- [40] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023. 4
- [41] Junting Pan, Chengyu Wang, Xu Jia, Jing Shao, Lu Sheng, Junjie Yan, and Xiaogang Wang. Video generation from single semantic label map. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3733–3742, 2019. 2
- [42] Zijie Pan, Zeyu Yang, Xiatian Zhu, and Li Zhang. Efficient4d: Fast dynamic 3d object generation from a singleview video. arXiv preprint arXiv:2401.08742, 2024. 3
- [43] Jeong Joon Park, Peter R. Florence, Julian Straub, Richard A. Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 165–174. Computer Vision Foundation / IEEE, 2019. 2
- [44] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 6

- [45] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 10318–10327. Computer Vision Foundation / IEEE, 2021. 3
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8748–

8763. PMLR, 2021. 13

- [47] Ohad Rahamim, Ori Malca, Dvir Samuel, and Gal Chechik. Bringing objects to life: 4d generation from 3d objects. CoRR, abs/2412.20422, 2024. 3
- [48] Marc’Aurelio Ranzato, Arthur Szlam, Joan Bruna, Micha¨el Mathieu, Ronan Collobert, and Sumit Chopra. Video (language) modeling: a baseline for generative models of natural videos. CoRR, abs/1412.6604, 2014. 2
- [49] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142,

2023. 1, 3

- [50] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems, 37:56828–56858, 2025. 1, 3
- [51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [52] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016. 14
- [53] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5,

2023. OpenReview.net, 2023. 2

- [54] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, and Yaniv Taigman. Text-to-4d dynamic scene generation. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, pages 31915–31929. PMLR, 2023. 3
- [55] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3
- [56] Colton Stearns, Adam W. Harley, Mikaela Angelina Uy, Florian Dubost, Federico Tombari, Gordon Wetzstein, and

- Leonidas J. Guibas. Dynamic gaussian marbles for novel view synthesis of casual monocular videos. In SIGGRAPH Asia 2024 Conference Papers, SA 2024, Tokyo, Japan, December 3-6, 2024, pages 30:1–30:11. ACM, 2024. 3
- [57] Qi Sun, Zhiyang Guo, Ziyu Wan, Jing Nathan Yan, Shengming Yin, Wengang Zhou, Jing Liao, and Houqiang Li. EG4D: explicit generation of 4d object without score distillation. CoRR, abs/2405.18132, 2024. 3
- [58] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. CoRR, abs/2411.04928, 2024. 1, 2, 3, 6, 8, 13, 14
- [59] KLING AI Team. Kling image-to-video model, 2024. 3, 4
- [60] Zachary Teed and Jia Deng. RAFT: recurrent all-pairs field transforms for optical flow (extended abstract). In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, pages 4839–4843. ijcai.org,

2021. 14

- [61] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1526–1535,

2018. 2

- [62] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pages 613–621, 2016. 2
- [63] Haiping Wang, Yuan Liu, Ziwei Liu, Wenping Wang, Zhen Dong, and Bisheng Yang. Vistadream: Sampling multiview consistent images for single-view scene reconstruction. arXiv preprint arXiv:2410.16892, 2024. 5
- [64] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 8, 14
- [65] Yaohui Wang, Piotr Bilinski, Franc¸ois Br´emond, and Antitza Dantcheva. Imaginator: Conditional spatio-temporal GAN for video generation. In IEEE Winter Conference on Applications of Computer Vision, WACV 2020, Snowmass Village, CO, USA, March 1-5, 2020, pages 1149–1158. IEEE, 2020. 2
- [66] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, pages 1–20, 2024. 2
- [67] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 14
- [68] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and

- diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36:8406–8441, 2023. 2
- [69] Min Wei, Jingkai Zhou, Junyao Sun, and Xuesong Zhang. Adversarial score distillation: when score distillation meets gan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8131–8141,

2024. 2

- [70] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310–20320, 2024. 3, 6, 13
- [71] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. arXiv preprint arXiv:2411.18613, 2024. 1, 2
- [72] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 20331–20341. IEEE, 2024. 3
- [73] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20331–20341, 2024. 6
- [74] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. CoRR, abs/2312.17225, 2023. 3
- [75] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, L´aszl´o A. Jeni, Sergey Tulyakov, and Hsin-Ying Lee. 4real: Towards photorealistic 4d scene generation via video diffusion models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10

- 15, 2024, 2024. 1, 2, 3, 6, 13, 14

- [76] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint

- arXiv:2409.02048, 2024. 2, 4, 8, 13, 14

[77] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint

- arXiv:2410.03825, 2024. 2, 3, 4, 8, 14

- [78] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5
- [79] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one im-

- age to 4d dynamic scene. CoRR, abs/2311.14603, 2023. 3, 6, 8, 13
- [80] Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan, Linjie Li, Zhengyuan Yang, Jianfeng Wang, Gim Hee Lee, and Lijuan Wang. Genxd: Generating any 3d and 4d scenes. CoRR, abs/2411.02319, 2024. 1, 2, 3, 6, 8, 13, 14
- [81] Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Otmar Hilliges, and Shalini De Mello. A unified approach for textand image-guided 4d scene generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7300–7309, 2024. 1, 3, 6, 8, 13

## Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency Supplementary Material

#### A. More Implementation Details

4D-GS Network. 4D Gaussian Splatting (4D-GS) [70] lies in extending static 3D Gaussian primitives [28] to dynamically model temporal-spatial scenes. In 3D-GS [28], a scene is represented by a set of anisotropic Gaussians G = {gi}Ni=1, where each Gaussian gi is parameterized by its position µi ∈ R3, rotation (quaternion qi ∈ R4), scale si ∈ R3, and opacity αi ∈ [0,1]. The covariance matrix Σi is derived from qi and si, enabling differentiable rendering via splatting.

To model 4D dynamics, each Gaussian is further augmented with time-varying parameters. For temporal coherence, we parameterize the trajectory of gi over time t through a deformation function ∆ : R4 → R9:

[∆µi(t),∆qi(t),∆si(t)] = ∆(µi,qi,si,t), (17)

where ∆ can be implemented via MLPs or explicit keyframe interpolation. The interpolated Gaussian gi(t) at time t is then rendered following the 3D-GS rendering pipeline, but with all parameters conditioned on t. Optimization typically requires multi-view RGB videos with camera poses. While achieving real-time dynamic rendering (30+ FPS), 4D-GS depends heavily on consistent multiview video supervision.

Training Setup. We adopt the 4D representation proposed in [70]. Our hyperparameter settings mainly follow those in [70]. The learning rate is initialized at 1.6 × 10−3 and gradually decays to 1.6 × 10−4 by the end of training. The Gaussian deformation decoder, implemented as a tiny MLP, starts with a learning rate of 1.6 × 10−4, which is reduced to 1.6 × 10−5 over time. The training batch size is set to 1. During the coarse stage, we train for 9k iterations, followed by an additional 1k iterations in the fine stage. The λ used in the fine-stage loss is 0.1. In modulation-based refinement, T¯ is set to 5 to improve efficiency, and wi linearly decreases from 0.5 to 0. Viewcrafter [76] uses its default denoising steps, which is 50. The guidance scale s used in CFG is the default value 7.5. For multi-view image generation at the first timestamp t = 1, we use adaptive CFG. For t > 1, CFG is disabled because the reference information from the multi-view generation at t = 1 has already been introduced into the missing regions. All experiments are conducted on a single NVIDIA A100 (40GB) GPU.

#### B. Details of User Study

User Study I: Comparison with Other Methods. We conducted the first user study to compare our method with other

existing methods. Since the source codes of these methods were not publicly available, we compared our method with the videos provided on their respective project pages. A total of 32 pairs of videos were used in this study. Each pair was generated from the same input images or text prompts to ensure a fair comparison. The methods included in this study were 4Real [75], 4Dfy [4], Dream-in-4D [81], DimensionX [58], GenXD [80], and Animate124 [79]. The user study was conducted online, and a screenshot of the interface is shown in Fig. A. Participants were asked to evaluate the generated videos based on four criteria: Consistency, Dynamic, Aesthetic, and Overall. For each pair of videos, they were required to select which method performed better for each criterion. They could skip to the next example without selecting if they found it difficult to judge. The user study was conducted anonymously, and no personally identifiable data were collected.

User Study II: Ablation Study. The second user study evaluated the impact of our method’s different components through an ablation experiment. The components included in this study were Monst3R, Adaptive CFG, Point Cloud Guided Denoising, Reference Latent Replacement, Reference Latent Replacement, Coarse-to-fine optimization, and Modulation-based Refinement. For each component, we randomly sampled 10 different scenes and generated video pairs using the full version of our method and a variant with the specific component removed or modified. Participants were asked to evaluate the generated video pairs based on the same four criteria as in User Study I: consistency, aesthetics, motion dynamic, and overall quality. The ablation study was also conducted anonymously, without collecting any personally identifiable data.

#### C. Details of VBench Metrics

To evaluate the quality of multi-view videos rendered from 4D representations, we report common VBench [26] metrics: Consistency (average for subject/background), Dynamic Degree, Aesthetic Score, and Text Alignment (only for text-to-4D).

Subject / Background Consistency. To evaluate the consistency of both subjects (e.g., a person, car, or cat) and background scenes in the video, VBench uses DINO [8] and CLIP [46] feature similarities across frames. DINO captures subject consistency by comparing frame embeddings, while CLIP assesses background stability. Together, they provide a comprehensive measure of consistency.

Dynamic Degree. Since a static video can score well in the aforementioned consistency metrics, it is important to eval-

|[Figure 271]<br><br>Input Prompt|
|---|

|Compare Method 1 & Method 2<br><br>Method 1 better<br><br>Method 2 better<br><br>Hard to judge Consistency<br><br>Dynamic Aesthetic Overall<br><br>Next Submit<br><br>1 / 10<br><br>(You can submit the results after completing the 5 groups)|
|---|

Method Time Resolution Frames Views 4Dfy [4] 10h+ 256×256 - 4Real [75] 1.5h 256×144 8 16

Ours 1h 1024×576 16 25

Table A. Comparison of runtime with other methods. Frames and Views represent the number of video frames and the number of viewpoints, respectively. The running time of Structure from Motion (SfM), such as colmap [52], is not included due to significant variations across different scenes.

|[Figure 272]<br><br>Method 1|
|---|

|[Figure 273]<br><br>Method 2|
|---|

[Figure 274]

[Figure 275]

| |
|---|

| |
|---|

[Figure 276]

Figure A. The web interface of our user studies. The input prompt can be either a single image or a short text.

[Figure 277]

[Figure 278]

uate the degree of dynamics (i.e., whether it contains large motions). To this end, the Dynamic Degree metric uses RAFT [60] to estimate the degree of dynamics in synthesized videos. Specifically, this metric takes the average of the largest 5% optical flows (considering the movement of small objects in the video). This approach ensures that minor movements (e.g., small objects or slight camera shakes) do not disproportionately influence the overall dynamic assessment.

| |
|---|

| |
|---|

Input

Multi-view images generated by ViewCrafter

Images rendered by Free4D

Figure B. Failure Case. ViewCrafter [76] struggles with blurred or defocused regions, leading to distortions that propagate into the 4DGS-rendered results.

Aesthetic Score. We evaluate the artistic and beauty value perceived by humans towards each video frame using the LAION Aesthetic Predictor [31]. This predictor is a linear model built on top of CLIP embeddings, trained to assess the aesthetic quality of images on a scale from 1 to 10. It reflects various aesthetic aspects, including the layout, richness and harmony of colors, photo-realism, naturalness, and overall artistic quality of the video frames. The Aesthetic Score metric obtains a normalized aesthetic score by applying this predictor to each frame.

supports higher resolutions while efficiently handling more frames and viewpoints, achieving the fastest optimization. Our total runtime is composed of three main steps: running MonST3R (1 min), generating multi-view videos with ViewCrafter (25 min), and optimizing 4D-GS (35 min).

#### E. Limitations and Future Work

Limitations. Since our method primarily relies on the prior from ViewCrafter [76] to generate consistent multi-view videos, it also inherits some of its limitations. Firstly, it struggles to synthesize novel views with large view ranges from limited 3D clues, such as generating a front view from only a back view. Additionally, since ViewCrafter depends on accurate point cloud geometry, it has difficulty handling severely blurred or defocused regions, which hinder depth estimation, as shown in Fig. B.

Text Alignment. This metric uses overall video-text consistency computed by ViCLIP [67] on general text prompts as an aiding metric to reflect text semantics consistency. ViCLIP is a video-text contrastive learning model that leverages a large-scale video-text dataset to learn robust and transferable representations.

#### D. Runtime Analysis

Future Work. We recognize that the accuracy of MonST3R [77]’s estimation of dynamic videos is crucial. We observed that Dust3R [64] demonstrates better robustness than MonST3R in some static scenes. Therefore, a potential approach is to use Dust3R to estimate the geometry of the first frame, and employ optical flow to link different views during the subsequent 4DGS optimization.

The runtime comparison is shown in Table A. We compare our approach with object-level methods [4] and the textto-4D scene generation method [75]. Since [58] and [80] have not reported runtime details (including feed-forward inference time and 4D representation optimization time) or released their code, they are excluded from the comparison. Notably, compared to previous methods, our approach

