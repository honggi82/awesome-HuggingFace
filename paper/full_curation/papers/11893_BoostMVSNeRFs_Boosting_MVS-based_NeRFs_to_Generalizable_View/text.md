## BoostMVSNeRFs: Boosting MVS-based NeRFs to Generalizable View Synthesis in Large-scale Scenes

Chih-Hai Su∗

Chih-Yao Hu∗

Shr-Ruei Tsai∗

oceani.c@nycu.edu.tw National Yang Ming Chiao Tung University Taiwan

jj2095813@gmail.com National Taiwan University Taiwan

0410330@gmail.com National Yang Ming Chiao Tung University Taiwan

Jie-Ying Lee∗

Chin-Yang Lin

Yu-Lun Liu

jayinnn.cs10@nycu.edu.tw National Yang Ming Chiao Tung University Taiwan

linjohn0903@gmail.com National Yang Ming Chiao Tung University Taiwan

yulunliu@cs.nycu.edu.tw National Yang Ming Chiao Tung University Taiwan

# arXiv:2407.15848v1[cs.CV]22Jul2024

https://su-terry.github.io/BoostMVSNeRFs

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

(a) MVSNeRF (b) ENeRF (c) ENeRF + ft

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Novel views

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Input views

(d) MVSNeRF + ours (e) ENeRF + ours (f) ENeRF + ours + ft

Figure 1: Our BoostMVSNeRFs enhances the novel view synthesis quality of MVS-based NeRFs in large-scale scenes. MVS-based NeRF methods often suffer from (a) limited viewport coverage from novel views or (b) artifacts due to limited input views for constructing cost volumes. (c) These drawbacks cannot be resolved even by per-scene fine-tuning. Our approach selects those cost volumes that contribute the most to the novel view and combines multiple selected cost volumes with volume rendering. (d, e) Our method does not require any training and is compatible with existing MVS-based NeRFs in a feed-forward fashion to improve the rendering quality. (f) The scene can be further fine-tuned as our method supports end-to-end fine-tuning.

### ABSTRACT

scenes. We first identify limitations in MVS-based NeRF methods, such as restricted viewport coverage and artifacts due to limited input views. Then, we address these limitations by proposing a new method that selects and combines multiple cost volumes during volume rendering. Our method does not require training and can adapt to any MVS-based NeRF methods in a feedforward fashion to improve rendering quality. Furthermore, our approach is also end-to-end trainable, allowing fine-tuning on specific scenes. We demonstrate the effectiveness of our method through experiments on large-scale datasets, showing significant rendering quality improvements in large-scale scenes and unbounded outdoor scenarios. We release the source code of BoostMVSNeRFs at https://su-terry.github.io/BoostMVSNeRFs.

While Neural Radiance Fields (NeRFs) have demonstrated exceptional quality, their protracted training duration remains a limitation. Generalizable and MVS-based NeRFs, although capable of mitigating training time, often incur tradeoffs in quality. This paper presents a novel approach called BoostMVSNeRFs to enhance the rendering quality of MVS-based NeRFs in large-scale

∗Authors contributed equally to the paper.

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s).

SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0525-0/24/07 https://doi.org/10.1145/3641519.3657416

### CCS CONCEPTS

• Computing methodologies → Rendering; Volumetric models.

### KEYWORDS

Novel View Synthesis, Neural Radiance Fields, 3D Synthesis, Neural Rendering

ACM Reference Format:

Chih-Hai Su, Chih-Yao Hu, Shr-Ruei Tsai, Jie-Ying Lee, Chin-Yang Lin, and Yu-Lun Liu. 2024. BoostMVSNeRFs: Boosting MVS-based NeRFs to Generalizable View Synthesis in Large-scale Scenes. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers ’24 (SIGGRAPH Conference Papers ’24), July 27–August 01, 2024, Denver, CO, USA. ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/ 3641519.3657416

### 1 INTRODUCTION

In computer vision, 3D reconstruction and novel view synthesis are crucial, with widespread applications from photogrammetry to AR/VR. Traditional methods relied on photo-geometry for 3D scene reconstruction using meshes. Recently, the task of novel view synthesis has advanced drastically since the emergence of the Neural Radiance Field (NeRF) and its variants [Barron et al. 2021, 2022, 2023; Cheng et al. 2024; Meuleman et al. 2023; Mildenhall et al.

- 2020; Tancik et al. 2022]. NeRF encodes 3D information into a Multilayer Perceptron (MLP) network to represent a scene. Despite such methods providing photorealistic rendering quality, these models have a huge downside as they require per-scene training with a long training time.

Recent advances in Generalizable NeRFs [Cao et al. 2022; Chen et al. 2021; Wang et al. 2021b; Xu et al. 2022; Yu et al. 2022, 2021b] improve scene adaptation by extracting input image features via

###### 2D CNNs and utilizing large datasets for training, allowing for rapid scene adaptation and enhanced rendering through fine-tuning. MVS-based methods such as MVSNeRF [Chen et al. 2021] and ENeRF [Lin et al. 2022] synthesize high-quality novel views by constructing cost volumes from a few input images, leveraging 3D CNNs and volume rendering in a feed-forward fashion. However, they are constrained by using a fixed number of input views and often struggle to reconstruct large-scale and unbounded scenes, resulting in padding artifacts at image boundaries (Fig. 1(a)) and wrongly reconstructed geometry in disocclusion regions (Fig. 1(b)). Furthermore, these issues could hardly be resolved by per-scene fine-tuning (Fig. 1(c)).

To address the problems, we propose BoostMVSNeRFs, a pipeline that is compatible with any MVS-based NeRFs to improve their rendering quality in large-scale and unbounded scenes. We first present 3D visibility scores for each sampled 3D point to indicate the proportion of contributions from individual input views. We then volume render the 3D visibility scores into 2D visibility masks to determine the contribution of each cost volume to the target novel view. Next, we combine multiple cost volumes during volume rendering to effectively expand the coverage of the novel view viewport and reduce artifacts by constructing more consistent geometry and thus alleviate the aforementioned MVS-based NeRFs’ issues. Additionally, to optimize the novel view visibility coverage, we further propose a greedy algorithm to approximate the optimal support cost volume set selection for the multiple-cost volume combined rendering. Our proposed pipeline is compatible with any MVS-based NeRFs to improve their rendering quality (Fig. 1(d, e))

and is end-to-end trainable. Therefore, our method also inherits this property from MVS-based NeRFs and can be fine-tuned to a specific scene to further improve the rendering quality (Fig. 1(f)).

We conduct experiments on two large-scale datasets, Free [Wang et al. 2023b] and ScanNet [Dai et al. 2017] datasets, which contain unbounded scenes with free camera trajectories and large-scale indoor scenes with complex structures, respectively. Experiments demonstrate that our proposed method performs favorably against other per-scene training or generalizable NeRFs in different dataset scenarios. Most importantly, our method is able to improve any MVS-based NeRF rendering quality through our extensive experiments, especially in free camera trajectories and unbounded outdoor scenes, which are the most common use cases in real-world applications.

### 2 RELATED WORK

Novel view synthesis. Novel view synthesis is a core challenge in computer vision, addressed through various techniques like imagebased rendering [Chaurasia et al. 2013; Flynn et al. 2016; Kalantari et al. 2016; Penner and Zhang 2017; Riegler and Koltun 2020] or multiplane image (MPI) [Flynn et al. 2019; Li et al. 2020; Mildenhall et al. 2019; Srinivasan et al. 2019; Tucker and Snavely 2020; Zhou et al. 2018], and explicit 3D representations, including meshs [Debevec et al. 2023; Thies et al. 2019; Waechter et al. 2014; Wood et al. 2023], voxels [Lombardi et al. 2019, 2021; Sitzmann et al. 2019], point clouds [Aliev et al. 2020; Xu et al. 2022], depth maps [Dhamo et al. 2019; Gortler et al. 1998; Hedman et al. 2021; Shih et al. 2020; Tulsiani et al. 2018]. Recently, neural representations [Jiang et al. 2020; Liu et al. 2019; Lombardi et al. 2019; Shih et al. 2020; Sitzmann et al. 2019; Wizadwongsa et al. 2021; Zhou et al. 2018], particularly Neural Radiance Fields (NeRF) [Barron et al. 2021, 2022, 2023; Meuleman et al. 2023; Mildenhall et al. 2020; Park et al. 2021; Tancik et al. 2022; Zhang et al. 2020], have achieved photorealistic rendering by representing scenes with continuous fields. Despite the advancements in areas like relighting [Boss et al. 2021; Munkberg et al. 2022; Yu et al. 2021a,b; Zhang et al. 2021a,b], dynamic scenes [Li et al. 2022; Liu et al. 2023; Park et al. 2021; Pumarola et al. 2021; Xian et al.

- 2021], and multi-view reconstruction [Oechsle et al. 2021; Wang

- et al. 2021a; Yariv et al. 2021, 2020], these methods although speed up training using hash grid [Müller et al. 2022] or voxel [Chen et al. 2022; Sun et al. 2022] as representations, still require intensive perscene optimization, thus limiting their generalizability. In contrast, our generalizable approach balances rendering quality and speed through feed-forward inference efficiently.

Multi-view stereo and generalizable radiance fields. Neural Radiance Fields (NeRF) offer photorealistic rendering but are limited by costly per-scene optimization. Recently, generalizable NeRFs [Cao

- et al. 2022; Chen et al. 2021; Wang et al. 2021b; Xu et al. 2022; Yu et al.

- 2022, 2021b] provide efficient approaches to synthesize novel views without per-scene optimization. Techniques like PixelNeRF [Yu et al. 2021b] and IBRNet [Wang et al. 2021b] merge features from adjacent views for volume rendering, while PointNeRF [Xu et al. 2022] constructs point-based fields for this purpose. Multi-view stereo (MVS) methods estimate depth using cost volumes [Oechsle et al. 2021], with MVSNet [Yao et al. 2018] utilizing 3D CNNs for feature extraction and cost volume construction, enabling end-to-end

training and further novel view synthesis. Despite amazing results from learning-based MVS, these methods are memory-intensive, prompting innovations like plane sweep [Yao et al. 2019] and coarseto-fine strategies [Chen et al. 2019; Gu et al. 2020; Yu and Gao 2020] for efficiency. Other works, such as MVSNeRF [Chen et al. 2021], ENeRF [Lin et al. 2022] and Im4D [Lin et al. 2023b], further bridge MVS methods with NeRF, introducing volumetric representations and depth-guided sampling for speed and dynamic reconstruction. Although these works advance the performance of generalizable NeRF, their rendering qualities are hindered by the limited visibility coverage of a single cost volume, leading to poor synthesis quality and visible padding artifacts near the image boundaries on large-scale or unbounded scenes. Additional research endeavors have been suggested to address these challenges. For instance, GeoNeRF [Johari et al. 2022]) introduces a novel approach to handle occlusions, while Neural Rays [Liu et al. 2022] presents an occlusionaware representation aimed at mitigating this problem. Although these methods tackle occlusions issues, the view coverage problem originated from MVS-based methods still exists. Our method overcomes this issue by selecting and combining multiple cost volumes to improve coverage and rendering confidence, enhancing the performance and robustness of MVS-based NeRF methods without any cost compared with previous methods.

Few-shot NeRFs. Prior work utilized mainly two different approaches to reconstruct scenes with sparse input views [Kim et al.

- 2022]: introducing regularization priors and training generalized model. Regularization-based methods [Deng et al. 2022; Jain et al.

2021; Niemeyer et al. 2022; Roessle et al. 2022; Seo et al. 2023; Somraj and Soundararajan 2023; Uy et al. 2023; Wang et al. 2023a; Wu et al. 2023; Wynn and Turmukhambetov 2023; Yang et al. 2023; Zhu et al.

- 2023] such as Vip-NeRF [Somraj and Soundararajan 2023] attempt to tackle this problem by obtaining visibility prior to regularize the scenes’ relative depth. Training generalized models [Chen et al. 2021, 2023; Chibane et al. 2021; Johari et al. 2022; Lin et al. 2023a; Shi et al. 2022; Trevithick and Yang 2021; Wang et al. 2021b; Yu

- et al. 2021b] on large datasets such as MVSNeRF [Chen et al. 2021] constructs cost volume to gain cross-view insight to tackle this goal. Different from this line of work, we present a novel visibility mask in a 3D fashion and serve as a visibility score to blend features while performing volume rendering.

Radiance fields fusion. Recently, several works propose to tackle scene fusion and intend to achieve large-scale reconstruction. NeRFusion [Zhang et al. 2022] performs sequential data fusion on voxels with GRU on the image level. SurfelNeRF [Gao et al. 2023] fuses scenes after converting them to surfels [Pfister et al. 2000] representation. Our approach seamlessly integrates cost volume without requiring training, thereby harnessing the capabilities of all MVS-based pre-trained models. Instead of concentrating solely on large-scale fusion, our method functions as a readily applicable tool to enhance various cost volume-based MVS applications.

### 3 METHOD

Given multi-view images in an unbounded scene, the same as other MVS-based NeRF methods (Sec. 3.1), our task is to synthesize novel view images without per-scene training. In order to tackle limited

viewport coverage from a single cost volume created by a fixed number of few (e.g., 3) input images, we propose BoostMVSNeRFs, an algorithm to consider multiple cost volumes while rendering. We first introduce a 3D visibility score for each sampled 3D point, which is used to render volume into 2D visibility masks (Sec. 3.2). Given a rendered 2D visibility mask for each cost volume, we combine multiple cost volumes in a support set to render novel views (Sec. 3.3). Finally, we present a greedy algorithm to iteratively select cost volumes and update the support set to maximize the viewport coverage and confidence of novel views (Sec. 3.4). Our pipeline is end-to-end trainable and thus can be fine-tuned on a new scene (Sec. 3.5). Our method is model-agnostic and applicable to any MVS-based NeRFs to boost the rendering quality.

### 3.1 MVS-based NeRFs Preliminaries

Given multi-view images with camera parameters, MVS-based NeRFs [Chen et al. 2021; Gu et al. 2020; Lin et al. 2022] use a shared 2D CNN to extract features for input images. Then, following MVSNet [Yao et al. 2018], we construct a feature volume by warping the input features into the target view. The warped features would be used to construct the encoding volume by computing the variance of multi-view features. Next, we apply a 3D CNN to regularize the encoding volume to build the cost volume CV to smooth the noise in the feature volume. Given a novel viewpoint, we query the color 𝑐 and density 𝜎 using an MLP with sampled 3D point coordinates 𝑥, viewing directions 𝑣, trilinear interpolated cost volume values at location 𝑝, and projected colors from input views Cin as input:

(𝑐,𝜎) = MLP𝜃 (𝑝,𝑣, CV(𝑝), Cin), (1)

where 𝜃 denotes the parameter of the MLP. Finally, we can volume render along rays to get the pixel colors in novel views.

The volume rendering equation in NeRF or MVSNeRF is evaluated by differentiable ray marching for novel view synthesis. A pixel color is computed by accumulating sample point values through ray marching. Here we consider a given ray r from the camera center 𝑜 through a given pixel on the image plane as r = 𝑜 +𝑢𝑗𝑑, where 𝑑 is the normalized viewing direction, and 𝑢𝑗 is the quadrature point constrained within the bounds of the near plane 𝑢𝑛 and the far plane 𝑢𝑓 . The final color is given by:

∑︁𝐽

𝑇 (𝑗)𝛼(𝜎𝑗𝛿𝑗)𝑐𝑗, (2)

𝐶(r) =

𝑗=1

where 𝑇 (𝑗) = exp(− 𝑠 𝑗=−11 𝜎𝑠𝛿𝑠) is the accumulated transmittance, 𝛼(𝑥) = 1 − exp(−𝑥) is the opacity of the point, and 𝛿𝑗 = 𝑢𝑗+1 −𝑢𝑗 is the distance between two quadrature points.

The existing MVS-based NeRFs only utilize a single cost volume from a few viewpoints (e.g. 3 input views). As a result, these methods often fall into limited viewport coverage, wrong geometry, and rendering artifacts (Fig. 1(a, b)). To overcome these problems, a naive solution would be training another MVS-based NeRF with more input views to construct the cost volume. Nevertheless, this solution requires training a new model with larger memory consumption, but even so, the input views could still be insufficient in inference time. Therefore, we proposed a novel method considering multiple cost volumes while rendering novel views.

| | | | | | |
|---|---|---|---|---|---|
| |M|LP| | | |
| | | | | | |
| | | | | | |

𝟏 + 𝟏 3

Novel view’s frustum

𝑚 =

𝜎,𝑐

|[Figure 24]|
|---|

[Figure 25]

𝟏 + 𝟏 + 𝟏 3 𝑚 =

Share weights

𝑚 =

Volume rendering

| | | | | | |
|---|---|---|---|---|---|
| |M|LP| | | |
| | | | | | |
| | | | | | |

𝟏 + 𝟏 3

2D visibility mask 𝐌 (Eq. 4)

𝜎,𝑐

Input view 1

3D visibility scores {𝑚 }

[Figure 26]

3D visibility scores (Eq. 3)

CV 1

⋮

⋮

CV 2

[Figure 27]

[Figure 28]

|Combined rendering (Eq. 7)| |
|---|---|
| | |

Novel view

Input view 3

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

Input view 2

Novel view

Figure 2: 3D visibility scores and 2D visibility masks. For a novel view, depth distribution is estimated from three input views, from which 3D points are sampled and projected onto each view to determine visibility. These projections yield 3D visibility scores 𝑚𝑗, normalized across the views, and are subsequently volume rendered into a 2D visibility mask M2D. This mask highlights the contribution of each input view to the cost volume and guides the rendering process, aiding in the selection of input views that optimize rendering quality and field of view coverage.

Rendered w/ CV 1

Rendered w/ CV 2

Figure 3: Combined rendering from multiple cost volumes. Using a single cost volume, as in traditional MVS-based NeRFs, often introduces padding artifacts or incorrect geometry, as indicated by the red dashed circles. Our method warps selected cost volumes to the novel view’s frustum and applies 3D visibility scores 𝑚𝑗 as weights to blend multiple cost volumes during volume rendering. Combined rendering provides broader viewport coverage and combines information from multiple cost volumes, leading to improved image synthesis and alleviating artifacts.

### 3.2 3D Visibility Scores and 2D Visibility Masks

By taking 𝐼 reference views into account in constructing a single cost volume, the maximum number of cost volumes we can refer to

is 𝐶𝐼𝑁 = 𝑁𝐼 = 𝑁 (𝑁𝐼−(1𝐼)···(−1)···𝑁1−𝐼+1) for each target view, where 𝑁 is the number of reference views. However, utilizing all cost volumes results in high memory consumption and also leads to inefficient rendering. To tackle this challenge, we propose a method to select those cost volumes with the largest contribution to viewport coverage and potential enhancement of rendering quality for novel views. To evaluate the contribution of each cost volume, we present multi-view 3D visibility scores as a metric.

3D visibility scores, we can combine the results from different cost volumes when volume rendering.

After obtaining 3D visibility scores for each cost volume, we propose the 2D visibility mask. The 2D visibility is constructed by volume rendering the 3D metrics scores to novel view, as shown in Fig. 2. Similar to Eq. 2, given ray r from the camera center 𝑜 with direction 𝑑, the value of 2D visibility mask is given by:

∑︁𝐽

For each sample point in a cost volume, we calculate its corresponding 3D visibility scores (the gray-shaded part in Fig. 2). These scores quantify the level of observation from various cost volumes, serving as a measurement of visibility. To calculate the 3D visibility scores of a single cost volume in a rendered view, we sample rays from the rendered view and aggregate the visibility weight from the reference views. Let 𝐼 represent the total number of reference views. We use 𝟙𝑖(𝑝) to indicate whether a sample point 𝑝 is in the viewport of reference view 𝑖 (bottom part in Fig. 2). The 3D visibility scores 𝑚𝑗 are calculated using the formula:

M2D(r) =

𝑇′(𝑗)𝛼 𝑚𝑗𝛿𝑗 𝑚𝑗, (4)

𝑗=1

where 𝑇′(𝑗) = exp(− 𝑠 𝑗=−11𝑚𝑠𝛿𝑠) is the transmitte considering 3D visibility scores. The 2D visibility mask will be used in cost volume

selection; we will thoroughly discuss it in Sec. 3.4.

### 3.3 Rendering by Combining Multiple Cost Volumes

Our proposed rendering differs from the traditional one (Eq. 2) by considering 3D visibility scores and combining multiple cost volumes. Below, we explain the modifications we make. First, let us only consider a single cost volume for simplicity. The pixel color output by considering only a single cost volume is given by:

𝐼 𝑖=1 𝟙𝑖(𝑝)

, (3)

𝑚𝑗 =

𝐼

where the subscript 𝑗 denotes the sampled 3D point index along the ray, and the output 3D visibility scores range from 0 to 1. Each point on the mask indicates its 3D visibility score, with larger values reflecting higher confidence in the information at a specific sample point. The visibility score can be utilized as the weight for the feature of a point on a specific cost volume. Therefore, with the

∑︁𝐽

𝑇single(𝑗)𝛼 𝜎𝑗𝛿𝑗 𝑚𝑗𝑐𝑗, (5)

𝐶single(r) =

𝑗=1

∑︁𝑗−1

𝑇single(𝑗) = exp −

(𝜎𝑠𝛿𝑠 − ln𝑚𝑠) . (6)

𝑠=1

Please refer to the supplementary material for the derivation of the transmittance considering only a single cost volume𝑇single(𝑗).

To further consider multiple cost volumes and also utilize their corresponding 3D visibility scores, we modify Eq. 9 to combine the result across multiple cost volumes. The final proposed volume rendering is given by:

∑︁𝐾

##### ∑︁𝐽

𝑇combined(𝑗)𝛼 𝜎𝑘𝑗 𝛿𝑗 𝑀𝑘𝑗 𝑐𝑘𝑗 , (7)

𝐶(r) =

𝑗=1

𝑘=1

∑︁𝑗−1

##### ∑︁𝐾

exp −

𝜎𝑠𝑘𝛿𝑠 − ln𝑀𝑠𝑘 , (8)

𝑇combined(𝑗) =

𝑠=1

𝑘=1

𝑘 𝑗

where 𝐾 is the number of selected cost volumes, and 𝑀𝑘𝑗 = 𝑚

𝐾 𝑘=1 𝑚𝑘𝑗

is the normalized 3D visibility score so that the summation of 3D visibility scores over selected cost volumes equals 1.

The illustration and effect of combining multiple cost volumes in rendering is shown in Fig. 3. Existing MVS-based NeRFs use a single cost volume to render novel views that contain padding artifacts and wrong geometry. Combining multiple cost volumes in rendering alleviates these artifacts and broadens the viewport coverage of novel views, thus improving the rendering quality.

### 3.4 Support Cost Volume Set Selection

As mentioned in Sec. 3.3, we only select 𝐾 cost volumes for combined rendering to optimize rendering efficiency. Ideally, combining selected 𝐾 cost volumes should provide maximum coverage for the rendered view. This problem can be formulated as maximum coverage problem, which is NP-hard. Thus, to complete view selection in polynomial time, we propose a greedy algorithm to construct a support set S of 𝐾 cost volumes in Algorithm 1. Nemhauser et al. [Nemhauser et al. 1978] also proved that the greedy algorithm is the optimal algorithm in polynomial time.

We show an example of the proposed selection algorithm in Fig. 4. At the beginning of the algorithm, our method selects the cost volume with the largest coverage score of the corresponding 2D visibility mask. The rendered image contains padding artifacts near the image boundaries as the viewport of this single cost volume is limited. Later on, our selection algorithm gradually selects the cost volumes that could maximize the visibility coverage and, therefore, enlarge the valid region of the rendered view. As a result, the rendering quality of novel views progressively grows as more cost volumes are selected and combined in the volume rendering.

### 3.5 End-to-end Fine-tuning

Our method is compatible with any MVS-based NeRFs to boost the rendering quality. Moreover, our approach is not optimized for a specific scene and could be generalized to new scenes, allowing it to enhance any end-to-end fine-tunable model. Fine-tuning refines geometry and color consistency within cost volumes and eliminates padding artifacts through combined rendering from multiple cost

Algorithm 1 Support cost volume set selection algorithm

Input: {CV𝑛}𝑛𝑁=1: 𝑁 candidate cost volumes Input: {M𝑛2D}𝑛𝑁=1: 2D visibility masks Output: S: a support set of 𝐾 cost volumes

- 1: S ← ∅ ⊲ Initialize the support CV set as an empty set
- 2: P0 ← 2D Mask filled with ones ⊲ Initialize the view coverage
- 3: while |S| < 𝐾 do
- 4: best_idx ← 0
- 5: max_ratio ← 0
- 6: 𝑖 ← 1 ⊲ Initialize selection iteration
- 7: while 𝑖 ≤ 𝑁 do
- 8: if CV𝑖 ∉ S then ⊲ Consider remaining views only
- 9: ratio ← (P𝑖−1 · M𝑖2D)
- 10: if ratio > max_ratio then
- 11: max_ratio ← ratio
- 12: best_idx ← 𝑖
- 13: end if
- 14: end if
- 15: 𝑖 ← 𝑖 + 1
- 16: end while
- 17: P𝑖 ← P𝑖−1 · (1 − M2Dbest_idx) ⊲ Update the view coverage
- 18: S ← S ∪ {CVbest_idx} ⊲ Add the best CV to the set
- 19: end while

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

visibilitymask Updated

RenderedRGB The best2D

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

viewcoverage

|𝐏𝟏|
|---|

|𝐏𝟐|
|---|

|𝐏𝟑|
|---|

|𝐏𝟒|
|---|

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

PSNR (dB) 18.58 21.89 24.43 25.06

Selection iteration

1 2 3 4

Figure 4: Support cost volume set selection. Initially, our greedy algorithm selects a single cost volume, providing maximum coverage yet insufficient to prevent padding artifacts (orange boxes). Subsequent iterations incorporate additional cost volumes, progressively expanding view coverage, and improving image quality, as indicated by the increasing PSNR values.

volumes. Thus, our method could augment the capabilities of advanced MVS-based NeRFs beyond ENeRF and MVS-NeRF. Please refer to the supplementary material for the fine-tuning details.

### 4 EXPERIMENTS 4.1 Experimental Settings

Datasets. We evaluate two datasets: (1) the Free dataset collected by F2-NeRF [Wang et al. 2023b] and (2) the ScanNet [Dai et al. 2017] dataset. The Free dataset consists of seven challenging scenes featuring narrow, long camera trajectories and focused foreground objects. Our evaluations on the Free dataset follow the train/test split in F2-NeRF [Wang et al. 2023b] by using one-eighth of the images

#### Table 1: Quantitative comparisons with state-of-the-art methods on the Free [Wang et al. 2023b] dataset.

Method Setting PSNR ↑ SSIM ↑ LPIPS ↓ FPS ↑ MVSNeRF [Chen et al. 2021]

20.06 0.721 0.469 1.79 MVSNeRF + Ours 20.52 0.722 0.470 1.26 ENeRF [Lin et al. 2022] 23.24 0.844 0.225 9.90 ENeRF+Ours 24.21 0.862 0.218 5.51

No per-scene optimization

25.55 0.776 0.278 3.75 Zip-NeRF [Barron et al. 2023] 25.90 0.772 0.241 0.66 MVSNeRFft [Chen et al. 2021] 20.49 0.698 0.425 1.79 MVSNeRF + Oursft 21.59 0.759 0.265 1.26 ENeRFft [Lin et al. 2022] 25.19 0.880 0.180 9.90 ENeRF+Oursft 26.14 0.894 0.171 5.51

F2-NeRF [Wang et al. 2023b]

Per-scene optimization

#### Table 2: Quantitative comparisons with state-of-the-art methods on the ScanNet [Dai et al. 2017] dataset.

Method Setting PSNR ↑ SSIM ↑ LPIPS ↓ FPS ↑ SurfelNeRF [Gao et al. 2023]

19.28 0.623 0.528 1.25 MVSNeRF [Chen et al. 2021] 23.40 0.862 0.367 1.99 MVSNeRF + Ours 23.66 0.872 0.365 1.41 ENeRF [Lin et al. 2022] 31.73 0.955 0.206 11.03 ENeRF + Ours 31.01 0.957 0.219 6.14

No per-scene optimization

F2-NeRF [Wang et al. 2023b]

28.11 0.894 0.230 4.18

SurfelNeRFft [Gao et al. 2023] 20.04 0.653 0.504 1.25 Zip-NeRF [Barron et al. 2023] 32.24 0.917 0.214 0.74

Per-scene optimization

MVSNeRFft [Chen et al. 2021] 24.69 0.872 0.316 1.99 MVSNeRF + Oursft 24.63 0.880 0.320 1.41 ENeRFft [Lin et al. 2022] 32.70 0.960 0.174 11.03 ENeRF + Oursft 32.87 0.955 0.173 6.14

for testing and the rest for training. As for the ScanNet dataset, we strictly follow the train/test splits as defined in NeRFusion [Zhang

- et al. 2022], NerfingMVS [Wei et al. 2021], and SurfelNeRF [Gao et al.
- 2023], with eight large-scale indoor scenes. We assess the rendering quality with PSNR, SSIM [Wang et al. 2004], and LPIPS [Zhang et al. 2018] metrics.

Baselines. We compare BoostMVSNeRFs with various state-ofthe-art NeRFs, including fast per-scene optimization NeRFs such as F2-NeRF [Wang et al. 2023b] and Zip-NeRF [Barron et al. 2023] and generalizable NeRFs such as MVSNeRF [Chen et al. 2021], ENeRF [Lin et al. 2022] and SurfelNeRF [Gao et al. 2023].

In particular, F2-NeRF excels in outdoor scenes with free camera trajectories. Our method employs cost volume representations similar to MVSNeRF and ENeRF but enlarges valid visible regions by fusing multiple cost volumes. Although SurfelNeRF also proposes fusing multiple surfels as a type of 3D representation, the fusion method and its scene representation differ from BoostMVSNeRFs. To ensure fairness, we used the same experimental settings as in previous studies and used official codes where possible. All the training, fine-tuning, and evaluations are done on a single RTX 4090 GPU.

Our method is compatible with MVS-based techniques, allowing us to employ pre-trained models such as MVSNeRF and ENeRF in our experiments. Unless otherwise specified, we use ENeRF as our backbone MVS-based NeRF method in all the experiments. We optimize the parameters, 𝑁 = 6, 𝐼 = 3, and 𝐾 = 4, for efficient rendering and high quality. Our method achieves similar runtime performance in rendering and fine-tuning as other generalizable NeRF methods but renders significantly improved quality. Additional implementation details of the proposed BoostMVSNeRFs, such as the sensitivity analysis of the number of selected cost volumes 𝐾 and the numerical representation (continuous or binary) of 2D visibility masks, are provided in the supplementary material.

### 4.2 Comparison with State-of-the-art Methods

Free dataset. On the Free dataset, BoostMVSNeRFs emerges as the best among no per-scene and per-scene optimization NeRF methods as shown in Table 1 and Fig. 5. Compared to F2-NeRF and SurfelNeRF, which produced blurred images, BoostMVSNeRFs leverages multiple cost volume fusion and view selection based on

visibility maps for superior rendering quality. Our method demonstrates compatibility with various camera trajectories and achieves results comparable to those of existing methods.

Our method outperforms generalizable NeRF techniques like MVSNeRF and ENeRF on the Free dataset (Table 1), enhancing rendering quality through our view selection and multiple cost volume combined rendering approach. Integrated with MVS-based NeRFs, our method achieves a PSNR improvement of 0.5-1.0 dB without requiring additional training. End-to-end fine-tuning on test scenes further enhances rendering quality, particularly in regions where a single cost volume falls short. This highlights the benefit of multiplecost volume fusion. For detailed visual comparisons, please refer to Fig. 6, 9, 10, and the supplementary material.

ScanNet dataset. We conducted a comprehensive comparison of BoostMVSNeRFs with other state-of-the-art methods in no perscene and per-scene optimization settings on the ScanNet dataset in Table 2. BoostMVSNeRFs demonstrates superior performance with a PSNR of 31.73 dB in no per-scene optimization, outperforming SurfelNeRF due to its cost volume fusion and efficient view selection strategy. In per-scene optimization, BoostMVSNeRFs excels again with a PSNR of 32.87 dB, indicating its effectiveness in cost volume fusion and per-scene adaptation. We also compare our method with two generalizable NeRF methods, MVSNeRF and ENeRF, on the ScanNet dataset in Table 2 and Fig. 12. Our method achieves better rendering quality over existing MVS-based NeRF methods in SSIM without per-scene optimization and in PSNR and LPIPS with per-scene fine-tuning.

Furthermore, our approach showed impressive results on largescale scenes, outperforming SurfelNeRF in both direct inference and per-scene fine-tuning. Unlike SurfelNeRF, which suffered from artifacts due to its surfel-based rendering approach, our model’s multiple cost volume fusion and efficient view information selection and aggregation led to high-quality and consistent renderings, as shown in Fig. 5. This indicates our cost volume fusion’s effectiveness in reconstructing large-scale scenes efficiently and accurately.

### 4.3 Ablation Study

Cost volume selection scheme. In Sec. 3.4, we propose a greedy method to select the cost volumes that will approximately maximize the view coverage. To validate the effectiveness of our method, We conducted experiments comparing two other cost volume selection

No per-scene optimization Per-scene optimization

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Ground truth MVSNeRF Ours F2-NeRF Zip-NeRF MVSNeRF + ft Ours + ft [Chen et al. 2021] [Wang et al. 2023b] [Barron et al. 2023] [Chen et al. 2021]

#### Figure 5: Qualitative comparisons of rendering quality on the Free [Wang et al. 2023b] dataset.

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

- Figure 6: Qualitative rendering quality improvements of integrating our method into MVS-based NeRF methods on the Free dataset.

- Table 3: Ablation of the cost volume selection. We compare three different strategies for cost volume selection on all scenes of the Free [Wang et al. 2023b] dataset: (a) ENeRF’s method, which is based on pose distance, (b) direct selection of cost volumes with maximum visibility, and (c) Our proposed greedy method, which maximizes the visibility coverage.

Method PSNR ↑ SSIM ↑ LPIPS ↓

- (a) ENeRF [Lin et al. 2022] 24.09 0.861 0.220
- (b) Maximize 2D visibility M𝑖2D 24.19 0.861 0.218
- (c) Maximize view coverage P𝑖 24.21 0.862 0.218

Single cost volume with more input views vs. Combining multiple cost volumes. In our method, we select multiple cost volumes and combine them in volume rendering, while ENeRF only forms one cost volume. To examine our method’s effectiveness, we train ENeRF (originally three input views) with more input views (6 in this ablation, in order to evenly compare with our proposed method). The results are shown in Table 4 and Fig. 7. We can see an increase in the number of input views which requires timeconsuming training to construct a single cost volume. However, the rendering quality improvements are subtle both with or without per-scene fine-tuning. In contrast, our cost volume selection method and combined rendering scheme improve the rendering quality by a large margin and could be further optimized with per-scene fine-tuning.

methods. These two methods are: (a) selecting 𝐾 cost volumes that are closest to the render view pose, which is adopted by ENeRF [Lin et al. 2022] and (b) selecting corresponding cost volumes directly with the highest contribution of 2D visibility mask. In particular,

- method (b) is a degenerate version of method our proposed selection
- method (c), which is based on view coverage. Table 3 shows that our greedy cost volume selection method performs better than the other two methods.

Robustness with sparse input views. Our proposed combined rendering from multiple cost volumes addresses the challenges of reconstructing large-scale and unbounded scenes due to broader viewport coverage. Therefore, our method could be more robust to sparse input views as more and farther cost volumes are considered during rendering. We conduct an experiment comparing performance across various degrees of sparse views to demonstrate the robustness of our method with sparse input views. Specifically,

#### Table 4: Different ways of combining more input views. We compare training an MVS-based NeRF with a larger number of input views (6 input views here) and our proposed cost volume selection and combined rendering on all scenes of the Free [Wang et al. 2023b] dataset.

Method Setting PSNR ↑ SSIM ↑ LPIPS ↓ ENeRF3-view [Lin et al. 2022]

23.24 0.844 0.225 ENeRF6-view [Lin et al. 2022] 23.53 0.770 0.231 ENeRF3-view + Ours 24.21 0.862 0.218

No per-scene optimization

ENeRF3-viewft [Lin et al. 2022]

25.19 0.880 0.180

Per-scene optimization

ENeRF6-viewft [Lin et al. 2022] 25.61 0.840 0.172 ENeRF3-view + Oursft 26.14 0.894 0.171

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

ENeRF3-view ENeRF6-view ENeRF3-view + Ours

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Ground truth ENeRF3-viewft ENeRF6-viewft ENeRF3-view + Oursft

- Figure 7: Visual effects of different ways of combining more input views. Artifacts in disocclusion regions cannot be resolved by including more input views for a single cost volume. Our method could alleviate these artifacts by combining more cost volumes in rendering.

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

- Figure 8: Robustness with sparse input views. With more sparse input views, the performance drop of our method is less severe than ENeRF, demonstrating the robustness of our method against sparse input views by combining multiple cost volumes in rendering.

we uniformly sub-sample the training views and evaluate the rendering quality. The results show a more significant decline in both PSNR and SSIM for ENeRF compared to ours while input views become sparse, as indicated by the curve in figure 8.

### 5 CONCLUSION

In summary, our BoostMVSNeRFs enhances MVS-based NeRFs, tackling large-scale and unbounded scene rendering challenges. Utilizing 3D visibility scores for multi-cost volume integration, BoostMVSNeRFs synthesizes significantly better novel views, enhancing viewport coverage and minimizing typical single-cost volume artifacts. Compatible with current MVS-based NeRFs, BoostMVSNeRFs supports end-to-end training for scene-specific enhancement. Experimental results validate the efficacy of our method in boosting advanced MVS-based NeRFs, contributing to more scalable and high-quality view synthesis. Future work will focus on reducing MVS dependency and optimizing memory usage, furthering the field of neural rendering for virtual and augmented reality applications.

### ACKNOWLEDGMENTS

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-E-A49-023-. The authors are grateful to Google and NVIDIA for generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan. The authors thank the anonymous reviewers for their valuable feedback.

### REFERENCES

Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lempitsky. 2020. Neural point-based graphics. In ECCV.

Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo MartinBrualla, and Pratul P Srinivasan. 2021. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV.

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman.

- 2022. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR.

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman.

- 2023. Zip-NeRF: Anti-aliased grid-based neural radiance fields. In ICCV.

Mark Boss, Raphael Braun, Varun Jampani, Jonathan T Barron, Ce Liu, and Hendrik Lensch. 2021. Nerd: Neural reflectance decomposition from image collections. In ICCV.

Ang Cao, Chris Rockwell, and Justin Johnson. 2022. Fwd: Real-time novel view synthesis with forward warping and depth. In CVPR. Gaurav Chaurasia, Sylvain Duchene, Olga Sorkine-Hornung, and George Drettakis.

2013. Depth synthesis and local warps for plausible image-based navigation. ACM TOG (2013).

Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. 2022. Tensorf: Tensorial radiance fields. In ECCV.

Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. 2021. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In ICCV.

Rui Chen, Songfang Han, Jing Xu, and Hao Su. 2019. Point-based multi-view stereo network. In ICCV.

Yuedong Chen, Haofei Xu, Qianyi Wu, Chuanxia Zheng, Tat-Jen Cham, and Jianfei Cai. 2023. Explicit Correspondence Matching for Generalizable Neural Radiance Fields. arXiv preprint arXiv:2304.12294 (2023).

Bo-Yu Cheng, Wei-Chen Chiu, and Yu-Lun Liu. 2024. Improving Robustness for Joint Optimization of Camera Poses and Decomposed Low-Rank Tensorial Radiance Fields. In AAAI.

Julian Chibane, Aayush Bansal, Verica Lazova, and Gerard Pons-Moll. 2021. Stereo radiance fields (srf): Learning view synthesis for sparse views of novel scenes. In CVPR.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. 2017. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR.

Paul E Debevec, Camillo J Taylor, and Jitendra Malik. 2023. Modeling and rendering architecture from photographs: A hybrid geometry-and image-based approach. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2.

Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. 2022. Depth-supervised nerf: Fewer views and faster training for free. In CVPR.

Helisa Dhamo, Keisuke Tateno, Iro Laina, Nassir Navab, and Federico Tombari. 2019. Peeking behind objects: Layered depth prediction from a single image. Pattern Recognition Letters (2019).

John Flynn, Michael Broxton, Paul Debevec, Matthew DuVall, Graham Fyffe, Ryan Overbeck, Noah Snavely, and Richard Tucker. 2019. Deepview: View synthesis with learned gradient descent. In CVPR.

John Flynn, Ivan Neulander, James Philbin, and Noah Snavely. 2016. Deepstereo: Learning to predict new views from the world’s imagery. In CVPR. Yiming Gao, Yan-Pei Cao, and Ying Shan. 2023. SurfelNeRF: Neural Surfel Radiance Fields for Online Photorealistic Reconstruction of Indoor Scenes. In CVPR. Jonathan Shade Steven Gortler, Li-wei He, Richard Szeliski, et al. 1998. Layered depth images. In SIGGRAPH.

Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong Tan, and Ping Tan. 2020. Cascade cost volume for high-resolution multi-view stereo and stereo matching. In CVPR.

Peter Hedman, Pratul P Srinivasan, Ben Mildenhall, Jonathan T Barron, and Paul Debevec. 2021. Baking neural radiance fields for real-time view synthesis. In ICCV. Ajay Jain, Matthew Tancik, and Pieter Abbeel. 2021. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In ICCV. Yue Jiang, Dantong Ji, Zhizhong Han, and Matthias Zwicker. 2020. Sdfdiff: Differentiable rendering of signed distance fields for 3d shape optimization. In CVPR. Mohammad Mahdi Johari, Yann Lepoittevin, and François Fleuret. 2022. Geonerf: Generalizing nerf with geometry priors. In CVPR. Nima Khademi Kalantari, Ting-Chun Wang, and Ravi Ramamoorthi. 2016. Learningbased view synthesis for light field cameras. ACM TOG (2016). Mijeong Kim, Seonguk Seo, and Bohyung Han. 2022. Infonerf: Ray entropy minimization for few-shot neural volume rendering. In CVPR.

Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. 2022. Neural 3d video synthesis from multi-view video. In CVPR. Zhengqi Li, Wenqi Xian, Abe Davis, and Noah Snavely. 2020. Crowdsampling the

plenoptic function. In ECCV.

Haotong Lin, Sida Peng, Zhen Xu, Tao Xie, Xingyi He, Hujun Bao, and Xiaowei Zhou. 2023b. Im4d: High-fidelity and real-time novel view synthesis for dynamic scenes. arXiv preprint arXiv:2310.08585 (2023).

Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xiaowei Zhou. 2022. Efficient neural radiance fields for interactive free-viewpoint video. In SIGGRAPH Asia.

Kai-En Lin, Yen-Chen Lin, Wei-Sheng Lai, Tsung-Yi Lin, Yi-Chang Shih, and Ravi Ramamoorthi. 2023a. Vision transformer for nerf-based view synthesis from a single input image. In WACV.

Lingjie Liu, Weipeng Xu, Michael Zollhoefer, Hyeongwoo Kim, Florian Bernard, Marc Habermann, Wenping Wang, and Christian Theobalt. 2019. Neural rendering and reenactment of human actor videos. ACM TOG (2019).

Yuan Liu, Sida Peng, Lingjie Liu, Qianqian Wang, Peng Wang, Christian Theobalt, Xiaowei Zhou, and Wenping Wang. 2022. Neural rays for occlusion-aware imagebased rendering. In CVPR.

Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. 2023. Robust dynamic radiance fields. In CVPR.

Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, and Yaser Sheikh. 2019. Neural volumes: Learning dynamic renderable volumes from images. ACM TOG (2019).

Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. 2021. Mixture of volumetric primitives for efficient neural rendering. ACM TOG (2021).

Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H Kim, and Johannes Kopf. 2023. Progressively optimized local radiance fields for robust view synthesis. In CVPR.

Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. 2019. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM TOG (2019).

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG (2022).

Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao, Wenzheng Chen, Alex Evans, Thomas Müller, and Sanja Fidler. 2022. Extracting triangular 3d models, materials, and lighting from images. In CVPR.

George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. 1978. An analysis of approximations for maximizing submodular set functions—I. Mathematical programming (1978).

Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. 2022. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In CVPR.

Michael Oechsle, Songyou Peng, and Andreas Geiger. 2021. Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction. In ICCV.

Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. 2021. Nerfies: Deformable neural radiance fields. In ICCV.

Eric Penner and Li Zhang. 2017. Soft 3d reconstruction for view synthesis. ACM TOG

(2017).

Hanspeter Pfister, Matthias Zwicker, Jeroen Van Baar, and Markus Gross. 2000. Surfels: Surface elements as rendering primitives. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques.

Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2021.

D-nerf: Neural radiance fields for dynamic scenes. In CVPR. Gernot Riegler and Vladlen Koltun. 2020. Free view synthesis. In ECCV. Barbara Roessle, Jonathan T Barron, Ben Mildenhall, Pratul P Srinivasan, and Matthias

Nießner. 2022. Dense depth priors for neural radiance fields from sparse input views. In CVPR.

Seunghyeon Seo, Donghoon Han, Yeonjin Chang, and Nojun Kwak. 2023. MixNeRF: Modeling a Ray with Mixture Density for Novel View Synthesis from Sparse Inputs. In CVPR.

Yue Shi, Dingyi Rong, Bingbing Ni, Chang Chen, and Wenjun Zhang. 2022. Garf: Geometry-aware generalized neural radiance field. arXiv preprint arXiv:2212.02280

(2022). Meng-Li Shih, Shih-Yang Su, Johannes Kopf, and Jia-Bin Huang. 2020. 3d photography using context-aware layered depth inpainting. In CVPR.

Vincent Sitzmann, Justus Thies, Felix Heide, Matthias Nießner, Gordon Wetzstein, and Michael Zollhofer. 2019. Deepvoxels: Learning persistent 3d feature embeddings. In CVPR.

Nagabhushan Somraj and Rajiv Soundararajan. 2023. ViP-NeRF: Visibility Prior for Sparse Input Neural Radiance Fields. (2023).

Pratul P Srinivasan, Richard Tucker, Jonathan T Barron, Ravi Ramamoorthi, Ren Ng, and Noah Snavely. 2019. Pushing the boundaries of view extrapolation with multiplane images. In CVPR.

Cheng Sun, Min Sun, and Hwann-Tzong Chen. 2022. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In CVPR.

Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. 2022. Blocknerf: Scalable large scene neural view synthesis. In CVPR.

Justus Thies, Michael Zollhöfer, and Matthias Nießner. 2019. Deferred neural rendering: Image synthesis using neural textures. ACM TOG (2019). Alex Trevithick and Bo Yang. 2021. Grf: Learning a general radiance field for 3d representation and rendering. In ICCV. Richard Tucker and Noah Snavely. 2020. Single-view view synthesis with multiplane images. In CVPR. Shubham Tulsiani, Richard Tucker, and Noah Snavely. 2018. Layer-structured 3d scene inference via view synthesis. In ECCV.

Mikaela Angelina Uy, Ricardo Martin-Brualla, Leonidas Guibas, and Ke Li. 2023. SCADE: NeRFs from Space Carving with Ambiguity-Aware Depth Estimates. In CVPR.

Michael Waechter, Nils Moehrle, and Michael Goesele. 2014. Let there be color! Large-scale texturing of 3D reconstructions. In ECCV. Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. 2023a. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In ICCV.

Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021a. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In NeurIPS.

Peng Wang, Yuan Liu, Zhaoxi Chen, Lingjie Liu, Ziwei Liu, Taku Komura, Christian Theobalt, and Wenping Wang. 2023b. F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories. In CVPR.

Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. 2021b. Ibrnet: Learning multi-view image-based rendering. In CVPR.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE TIP (2004).

Yi Wei, Shaohui Liu, Yongming Rao, Wang Zhao, Jiwen Lu, and Jie Zhou. 2021. Nerfingmvs: Guided optimization of neural radiance fields for indoor multi-view stereo. In ICCV.

Suttisak Wizadwongsa, Pakkapon Phongthawee, Jiraphon Yenphraphai, and Supasorn Suwajanakorn. 2021. Nex: Real-time view synthesis with neural basis expansion. In CVPR.

Daniel N Wood, Daniel I Azuma, Ken Aldinger, Brian Curless, Tom Duchamp, David H Salesin, and Werner Stuetzle. 2023. Surface light fields for 3D photography. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2.

Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. 2023. ReconFusion: 3D Reconstruction with Diffusion Priors. arXiv preprint arXiv:2312.02981 (2023).

Jamie Wynn and Daniyar Turmukhambetov. 2023. Diffusionerf: Regularizing neural radiance fields with denoising diffusion models. In CVPR. Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. 2021. Space-time neural irradiance fields for free-viewpoint video. In CVPR. Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. 2022. Point-nerf: Point-based neural radiance fields. In CVPR.

Jiawei Yang, Marco Pavone, and Yue Wang. 2023. FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization. In CVPR. Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. 2018. Mvsnet: Depth inference for unstructured multi-view stereo. In ECCV. Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang, and Long Quan. 2019. Recurrent mvsnet for high-resolution multi-view stereo depth inference. In CVPR. Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume rendering of neural implicit surfaces. In NeurIPS.

Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and Yaron Lipman. 2020. Multiview neural surface reconstruction by disentangling geometry and appearance. In NeurIPS.

Alex Yu, Sara Fridovich-Keil, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. 2022. Plenoxels: Radiance fields without neural networks. In CVPR.

Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. 2021a. Plenoctrees for real-time rendering of neural radiance fields. In ICCV. Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. 2021b. pixelnerf: Neural radiance fields from one or few images. In CVPR. Zehao Yu and Shenghua Gao. 2020. Fast-mvsnet: Sparse-to-dense multi-view stereo with learned propagation and gauss-newton refinement. In CVPR.

Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, and Noah Snavely. 2021a. Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting. In CVPR.

Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. 2020. Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492 (2020). Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR. Xiaoshuai Zhang, Sai Bi, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. 2022. Nerfusion: Fusing radiance fields for large-scale scene reconstruction. In CVPR.

Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul Debevec, William T Freeman, and Jonathan T Barron. 2021b. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. ACM TOG (2021).

Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. 2018. Stereo magnification: Learning view synthesis using multiplane images. In SIGGRAPH.

Bingfan Zhu, Yanchao Yang, Xulong Wang, Youyi Zheng, and Leonidas Guibas. 2023. Vdn-nerf: Resolving shape-radiance ambiguity via view-dependence normalization. In CVPR.

No per-scene optimization Per-scene optimization

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

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Ground truth MVSNeRF Ours F2-NeRF Zip-NeRF MVSNeRF + ft Ours + ft [Chen et al. 2021] [Wang et al. 2023b] [Barron et al. 2023] [Chen et al. 2021]

#### Figure 9: Qualitative comparisons of rendering quality on the Free [Wang et al. 2023b] dataset.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

#### Figure 10: Qualitative rendering quality improvements of integrating our method into MVS-based NeRF methods on the Free dataset.

No per-scene optimization Per-scene optimization

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

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Ground truth MVSNeRF SurfelNeRF Ours F2-NeRF Zip-NeRF MVSNeRF + ft SurfelNeRF + ft Ours + ft [Chen et al. 2021] [Gao et al. 2023] [Wang et al. 2023b] [Barron et al. 2023] [Chen et al. 2021] [Gao et al. 2023]

#### Figure 11: Qualitative comparisons of rendering quality on the ScanNet [Dai et al. 2017] dataset.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

#### Figure 12: Qualitative rendering quality improvements of integrating our method into MVS-based NeRF methods on the ScanNet dataset.

### A APPENDIX OVERVIEW

This supplementary material presents additional results to complement the main paper. First, we provide the detailed derivation of the combined volume rendering equation with multiple cost volumes in Sec. B. Then, we describe all the implementation details of BoostMVSNeRFs and baseline methods in Sec. C. Next, we show additional ablation studies, including the sensitivity analysis on the number of selected cost volumes and the effect of combining multiple cost volumes in 2D or in the 3D space in Sec. D. Finally, we provide complete quantitative evaluations and additional qualitative comparisons in Sec. E and Sec. F, respectively. In addition to this document, we provide video results of our method and stateof-the-art methods and show the rendering quality comparison.

### B DERIVATION OF THE COMBINED VOLUME RENDERING WITH MULTIPLE COST VOLUMES

Our proposed rendering differs from the traditional one by considering 3D visibility scores and combining multiple cost volumes. Below, we explain the modifications we make. First, let us only consider a single cost volume for simplicity. The pixel color output is given by:

∑︁𝐽

𝑇single(𝑗)𝛼 𝜎𝑗𝛿𝑗 𝑚𝑗𝑐𝑗, (9)

𝐶single(r) =

𝑗=1

𝑗−1

exp (−𝜎𝑠𝛿𝑠)𝑚𝑠 (10)

𝑇single(𝑗) =

𝑠=1

𝑗−1

exp (−𝜎𝑠𝛿𝑠) exp (ln𝑚𝑠) (11)

=

𝑠=1

𝑗−1

exp (− (𝜎𝑠𝛿𝑠 − ln𝑚𝑠)) (12)

=

𝑠=1

∑︁𝑗−1

= exp −

(𝜎𝑠𝛿𝑠 − ln𝑚𝑠) . (13)

𝑠=1

To further consider multiple cost volumes and also utilize their corresponding 3D visibility scores, we modify Eq. 9 to combine the result across multiple cost volumes. The final proposed volume rendering is given by:

∑︁𝐾

##### ∑︁𝐽

𝑇combined(𝑗)𝛼 𝜎𝑘𝑗 𝛿𝑗 𝑀𝑘𝑗 𝑐𝑘𝑗 , (14)

𝐶(r) =

𝑗=1

𝑘=1

##### ∑︁𝐾

∑︁𝑗−1

exp −

𝜎𝑠𝑘𝛿𝑠 − ln𝑀𝑠𝑘 , (15)

𝑇combined(𝑗) =

𝑠=1

𝑘=1

𝑚𝑘𝑗

, (16)

𝑀𝑘𝑗 =

𝐾 𝑘=1𝑚𝑘𝑗

where 𝐾 is the number of selected cost volumes, and 𝑀𝑘𝑗 is the normalized 3D visibility score so that the summation of 3D visibility scores over selected cost volumes equals 1.

Table 5: Sensitivity analysis on the number of selected cost volumes for combination. We compare the rendering quality of different numbers of selected cost volumes for combined rendering on all scenes of the Free [Wang et al. 2023b] dataset. The rendering quality improves with more cost volumes selected in combined rendering.

Method Setting PSNR ↑ SSIM ↑ LPIPS ↓

- 𝐾 = 1

No per-scene optimization

24.05 0.865 0.205

- 𝐾 = 2 24.55 0.872 0.202
- 𝐾 = 3 24.68 0.875 0.202
- 𝐾 = 4 24.75 0.875 0.202
- 𝐾 = 5 24.79 0.875 0.203

### C IMPLEMENTATION DETAILS

Our method is compatible with MVS-based NeRF techniques, allowing us to employ pre-trained models such as MVSNeRF and ENeRF in our experiments. For per-scene optimization, we fine-tune our methods with 11,000 iterations, following the settings of ENeRF, with an initial learning rate of 5𝑒−4 and an exponential scheduler. This fine-tuning process typically takes between 1 to 2 hours on an RTX 4090 GPU, depending on the image resolution.

For a fair comparison across methods, we evaluate them using a 731 x 468 image resolution in the Free dataset and a 640 × 480 image resolution in the ScanNet dataset. During fine-tuning, we use a resolution of 736 × 480 to match our model architecture and subsequently downsample the images to 731 × 468 for fair comparison.

In practical terms, both ENeRF and our models are configured with 2 samples per ray. When integrating MVSNeRF, we use 32 samples per ray to evaluate its generalizable rendering model; however, during fine-tuning, this is reduced to 8 samples per ray to address convergence issues. We employ 4 cost volumes (K=4) for combined rendering with multiple cost volumes. . For each rendering view in both datasets, we consider its nearest 6 training views (𝑁 = 6)

and build 𝐶3𝑁 cost volumes and select 𝐾 best cost volumes for our method.

D ADDITIONAL ABLATION STUDIES

- D.1 Sensitivity analysis on the number of selected cost volumes

In our approach, we introduce a greedy method for cost volume selection for fusion, enabling the integration of multiple cost volumes. To optimize the most effective cost volume fusion strategy, we conduct experiments by considering 𝐾 cost volumes during volume rendering. We explore scenarios where 𝐾 equals 1, 2, 3, 4, and 5, and present the corresponding results in Table 5.

- D.2 Continuous-valued 3D visibility scores.

Our method represents 3D visibility scores with continuous values. We conduct experiments comparing our method with binary-valued

- 3D visibility scores, i.e., each score can only be either 0 or 1. The

binary-valued 3D visibility scores 𝑚′𝑗 are calculated by 𝑚′𝑗 = 1 − 𝑉 𝑖=1(1−⊮𝑖(𝑝)), where𝑚′𝑗 is the binary-valued 3D visibility scores.

- Table 6: Effect of continuous or binary 3D visibility scores. We compare representing the 3D visibility score using binary values with our continuous values on all scenes of the Free [Wang et al. 2023b] dataset. Representing 3D visibility scores with continuous values performs better than binary values.

Method Setting PSNR ↑ SSIM ↑ LPIPS ↓ Discrete mask No per-scene

optimization

24.05 0.855 0.227 Continuous mask (Ours) 24.21 0.862 0.218

- Table 7: Ablation on different combined methods with multiple cost volumes. We compare the rendering quality between different combined methods on all scenes of the ScanNet dataset. Combined rendering in 3D performs slightly better than in the 2D space.

Method PSNR ↑ SSIM ↑ LPIPS ↓

3D visibility scores as multipliers on densities 17.60 0.767 0.549 Rendered images and 2D visibility masks 24.20 0.868 0.360 Sampled 3D points and 3D visibility scores (Ours) 24.22 0.868 0.361

### D.3 Different combined methods with multiple cost volumes

In our method, we use calculated 3D visibility scores as weights to combine multiple cost volumes during rendering. A straightforward variant is to multiply the 3D visibility scores onto queried density

values in volume rendering. Yet another way is to blend rendering results from different cost volumes in the 2D image domain instead of 3D sampled points. Table 7 demonstrates that our combination method using 3D sampled points and 3D visibility scores achieves the best rendering quality.

- E COMPLETE QUANTITATIVE EVALUATIONS

- E.1 Free dataset

We showall7 scenesofthequantitative comparisons on the Free [Wang et al. 2023b] dataset in Table 8.

- E.2 ScanNet dataset

We show all 8 scenes of the quantitative comparisons on the ScanNet [Dai et al. 2017] dataset following the train/test split defined in NeRFusion, NerfingMVS, and SurfelNeRF in Table 9.

- F ADDITIONAL QUALITATIVE COMPARISONS

- F.1 Free dataset

We show additional qualitative comparisons on the Free [Wang et al. 2023b] dataset in Fig. 13 and Fig. 14.

- F.2 ScanNet dataset

We show additional qualitative comparisons on the ScanNet [Dai et al. 2017] dataset, following the train/test split defined in NeRFusion, NerfingMVS, and SurfelNeRF, as depicted in Fig. 15 and Fig. 16.

#### Table 8: Complete quantitative comparisons with state-of-the-art methods on the Free [Wang et al. 2023b] dataset.

Method Setting Hydrant Lab Pillar Road Sky Stair Grass

PSNR ↑ MVSNeRF [Chen et al. 2021]

19.63 20.11 19.58 20.84 18.58 21.98 19.69 MVSNeRF + Ours 20.21 20.00 20.02 21.77 19.50 21.64 20.47 ENeRF [Lin et al. 2022] 21.85 23.48 24.3 24.73 22.27 25.52 20.53 ENeRF + Ours 22.73 23.94 26.06 25.88 23.44 25.78 21.64

No per-scene optimization

F2-NeRF [Wang et al. 2023b]

23.75 24.34 28.05 26.03 25.10 28.14 23.44 Zip-NeRF [Barron et al. 2023] 25.43 27.94 25.30 28.83 27.12 28.21 18.46 MVSNeRFft [Chen et al. 2021] 19.33 18.90 21.22 21.88 19.42 21.62 21.08 MVSNeRF + Oursft 20.81 19.05 21.75 24.31 20.04 22.18 23.02 ENeRFft [Lin et al. 2022] 23.35 24.87 27.48 26.43 23.65 27.43 23.15 ENeRF + Oursft 24.28 25.83 28.50 27.64 24.31 28.28 24.12

Per-scene optimization

SSIM ↑ MVSNeRF [Chen et al. 2021]

0.689 0.757 0.698 0.755 0.744 0.770 0.631 MVSNeRF + Ours 0.691 0.745 0.706 0.762 0.748 0.763 0.636 ENeRF [Lin et al. 2022] 0.812 0.881 0.854 0.868 0.873 0.891 0.729 ENeRF + Ours 0.837 0.889 0.888 0.889 0.881 0.897 0.755

No per-scene optimization

F2-NeRF [Wang et al. 2023b]

0.743 0.825 0.794 0.802 0.856 0.835 0.581 Zip-NeRF [Barron et al. 2023] 0.818 0.902 0.748 0.880 0.889 0.855 0.313 MVSNeRFft [Chen et al. 2021] 0.645 0.697 0.690 0.746 0.718 0.747 0.646 MVSNeRF + Oursft 0.717 0.693 0.747 0.839 0.715 0.808 0.793 ENeRFft [Lin et al. 2022] 0.847 0.908 0.905 0.898 0.892 0.917 0.791 ENeRF + Oursft 0.872 0.919 0.916 0.917 0.891 0.927 0.813

Per-scene optimization

LPIPS ↓ MVSNeRF [Chen et al. 2021]

0.458 0.389 0.532 0.474 0.438 0.462 0.528 MVSNeRF + Ours 0.458 0.397 0.543 0.465 0.431 0.471 0.528 ENeRF [Lin et al. 2022] 0.232 0.195 0.216 0.218 0.218 0.178 0.317 ENeRF + Ours 0.227 0.194 0.195 0.194 0.223 0.186 0.307

No per-scene optimization

F2-NeRF [Wang et al. 2023b]

0.283 0.262 0.233 0.270 0.237 0.215 0.448 Zip-NeRF [Barron et al. 2023] 0.185 0.163 0.235 0.156 0.166 0.167 0.613 MVSNeRFft [Chen et al. 2021] 0.434 0.383 0.474 0.421 0.391 0.418 0.451 MVSNeRF + Oursft 0.277 0.259 0.344 0.210 0.263 0.268 0.233 ENeRFft [Lin et al. 2022] 0.190 0.158 0.160 0.175 0.179 0.142 0.258 ENeRF + Oursft 0.177 0.148 0.148 0.146 0.205 0.134 0.241

Per-scene optimization

#### Table 9: Complete quantitative comparisons with state-of-the-art methods on the ScanNet [Dai et al. 2017] dataset, following the train/test split on NeRFusion, NerfingMVS, and SurfelNeRF.

Method Setting 0000 0079 0158 0316 0521 0553 0616 0653

PSNR ↑ MVSNeRF [Chen et al. 2021]

23.56 28.98 25.96 19.48 20.69 27.99 16.19 24.38 MVSNeRF + Ours 24.02 29.10 25.24 18.73 20.59 28.82 18.02 24.75 SurfelNeRF 16.57 20.33 20.43 20.42 20.89 19.88 18.58 17.12 ENeRF [Lin et al. 2022] 29.33 33.49 34.03 32.72 32.12 34.61 24.41 33.11 ENeRF + Ours 28.89 33.27 32.31 31.14 30.59 32.96 25.65 33.27

No per-scene optimization

28.03 30.28 31.75 27.21 24.84 31.55 23.24 28.02 Zip-NeRF [Barron et al. 2023] 31.56 32.53 34.86 34.42 31.85 34.57 24.80 33.34 MVSNeRFft [Chen et al. 2021] 25.24 28.02 25.72 24.64 22.35 30.12 15.76 25.69 MVSNeRF + Oursft 23.47 28.42 24.62 24.71 22.49 28.91 16.86 26.39 SurfelNeRFft 19.85 20.84 21.21 20.82 20.61 21.95 17.47 17.55 ENeRFft [Lin et al. 2022] 31.52 33.39 35.82 33.33 32.32 35.67 25.36 34.19 ENeRF + Oursft 31.86 33.26 36.09 33.43 32.29 35.18 26.25 34.57

F2-NeRF [Wang et al. 2023b]

Per-scene optimization

SSIM ↑ MVSNeRF [Chen et al. 2021]

0.831 0.909 0.905 0.873 0.833 0.936 0.713 0.899 MVSNeRF + Ours 0.838 0.911 0.904 0.880 0.847 0.940 0.757 0.899 SurfelNeRF 0.444 0.644 0.712 0.699 0.688 0.660 0.508 0.631 ENeRF [Lin et al. 2022] 0.938 0.952 0.974 0.977 0.958 0.978 0.890 0.974 ENeRF + Ours 0.943 0.953 0.973 0.975 0.955 0.976 0.902 0.975

No per-scene optimization

0.865 0.871 0.935 0.939 0.894 0.942 0.788 0.919 Zip-NeRF [Barron et al. 2023] 0.908 0.893 0.953 0.961 0.920 0.956 0.794 0.951 MVSNeRFft [Chen et al. 2021] 0.856 0.894 0.901 0.917 0.857 0.948 0.700 0.899 MVSNeRF + Oursft 0.841 0.902 0.893 0.917 0.859 0.941 0.727 0.907 SurfelNeRFft 0.547 0.664 0.728 0.714 0.659 0.750 0.482 0.677 ENeRFft [Lin et al. 2022] 0.949 0.955 0.979 0.982 0.958 0.981 0.896 0.976 ENeRF + Oursft 0.943 0.953 0.973 0.975 0.955 0.976 0.902 0.975

F2-NeRF [Wang et al. 2023b]

Per-scene optimization

LPIPS ↓ MVSNeRF [Chen et al. 2021]

0.362 0.301 0.305 0.370 0.493 0.225 0.577 0.306 MVSNeRF + Ours 0.366 0.333 0.327 0.362 0.465 0.240 0.551 0.274 SurfelNeRF 0.575 0.541 0.499 0.477 0.508 0.514 0.573 0.534 ENeRF [Lin et al. 2022] 0.195 0.211 0.193 0.219 0.219 0.171 0.281 0.161 ENeRF + Ours 0.194 0.216 0.208 0.244 0.254 0.188 0.282 0.169

No per-scene optimization

0.230 0.255 0.195 0.214 0.257 0.166 0.350 0.177 Zip-NeRF [Barron et al. 2023] 0.190 0.243 0.181 0.191 0.271 0.156 0.325 0.158 MVSNeRFft [Chen et al. 2021] 0.288 0.274 0.287 0.276 0.426 0.183 0.568 0.229 MVSNeRF + Oursft 0.284 0.270 0.285 0.280 0.426 0.189 0.544 0.223 SurfelNeRFft 0.532 0.502 0.424 0.487 0.532 0.458 0.595 0.498 ENeRFft [Lin et al. 2022] 0.162 0.189 0.144 0.154 0.209 0.128 0.271 0.133 ENeRF + Oursft 0.160 0.191 0.142 0.154 0.212 0.130 0.266 0.132

F2-NeRF [Wang et al. 2023b]

Per-scene optimization

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

Ground truth MVSNeRF Ours F2-NeRF Zip-NeRF MVSNeRF + ft Ours + ft [Chen et al. 2021] [Wang et al. 2023b] [Barron et al. 2023] [Chen et al. 2021]

#### Figure 13: Additional qualitative comparisons of rendering quality on the Free [Wang et al. 2023b] dataset.

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

#### Figure 14: Additional qualitative rendering quality improvements of integrating our method into MVS-based NeRF methods on the Free dataset.

[Figure 271]

[Figure 272]

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

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

Ground truth MVSNeRF SurfelNeRF Ours F2-NeRF Zip-NeRF MVSNeRF + ft SurfelNeRF + ft Ours + ft [Chen et al. 2021] [Gao et al. 2023] [Wang et al. 2023b] [Barron et al. 2023] [Chen et al. 2021] [Gao et al. 2023]

#### Figure 15: Additional qualitative comparisons of rendering quality on the ScanNet [Dai et al. 2017] dataset.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Ground truth MVSNeRF [Chen et al. 2021] MVSNeRF + ft ENeRF [Lin et al. 2022] ENeRF + ft

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

MVSNeRF + ours MVSNeRF + ours + ft ENeRF + ours ENeRF + ours + ft

#### Figure 16: Additional qualitative rendering quality improvements of integrating our method into MVS-based NeRF methods on the ScanNet dataset.

