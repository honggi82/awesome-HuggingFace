# arXiv:2601.09575v1[cs.CV]14Jan2026

[Figure 1]

2026-1-15

## OpenVoxel: Training-Free Grouping and Captioning Voxels for Open-Vocabulary 3D Scene Understanding

Sheng-Yu Huang1,2,†, Jaesung Choe1, Frank Wang1,2, Cheng Sun1 1NVIDIA, 2National Taiwan University

### Abstract

We propose OpenVoxel, a training-free algorithm for grouping and captioning sparse voxels for the open-vocabulary 3D scene understanding tasks. Given the sparse voxel rasterization (SVR) model obtained from multi-view images of a 3D scene, our OpenVoxel is able to produce meaningful groups that describe different objects in the scene. Also, by leveraging powerful Vision Language Models (VLMs) and Multi-modal Large Language Models (MLLMs), our OpenVoxel successfully build an informative scene map by captioning each group, enabling further 3D scene understanding tasks such as openvocabulary segmentation (OVS) or referring expression segmentation (RES). Unlike previous methods, our method is training-free and does not introduce embeddings from a CLIP/BERT text encoder. Instead, we directly proceed with text-to-text search using MLLMs. Through extensive experiments, our method demonstrates superior performance compared to recent studies, particularly in complex referring expression segmentation (RES) tasks. The code will be open.

### 1. Introduction

With the recent advancement of the neural rendering algorithm [Barron et al., 2021; Chen et al., 2022; Chou et al., 2024; Fridovich-Keil et al., 2022; Johari et al., 2022; Kundu et al., 2022; Liu et al., 2022; Martin-Brualla et al., 2021; Mildenhall et al., 2021; Müller et al., 2022; Reiser et al., 2021; Suhail et al., 2022; Sun et al., 2022; Vora et al., 2021; Yu et al., 2021], several primitives have been introduced to represent the 3D scenes. 3D Gaussian Splatting (3DGS) [Kerbl et al., 2023] and several extensions [Chen and Wang, 2024; Gao et al.,

- 2024; Xu et al., 2025; Yu et al., 2024] have recently emerged as a groundbreaking technique for novel view synthesis, offering an exceptional balance of visual fidelity and real-time performance. Concurrently, Sparse Voxel Rasterization (SVR) [Li et al., 2025; Sun et al., 2025] enables a sparse voxel representation to perform the novel-view synthesis task.

Recent studies [He et al., 2025,; Huang et al., 2025; Jang and Kim, 2025; Ji et al., 2025; Jun-Seong et al.,

- 2025; Kerr et al., 2023; Lee et al., 2025; Liao et al., 2025; Liu et al., 2023; Marrie et al., 2025; Peng et al., 2025; Piekenbrinck et al., 2025; Qin et al., 2024; Tian et al., 2025; Wu et al., 2024; Ye et al., 2024; Ying et al., 2024; Zhu et al., 2025] leverage the series of novel 3D primitives for open-vocabulary 3D understanding tasks, such as semantic segmentation, 3D open-vocabulary segmentation (OVS). LangSplat [Qin et al., 2024] is one of the pioneering papers that introduces 3D Gaussians for this purpose by optimizing 3DGS first and adopts Gaussian features that are trained to store language attributes derived from the 2D foundation model, CLIP [Radford et al., 2021]. The successive studies [Jang and Kim, 2025; Piekenbrinck et al., 2025; Ye et al., 2024; Ying et al., 2024; Zhu et al., 2025] try to improve the trainable Gaussian features by either introducing contrastive learning between different objects in the scene, or training an object-aware 3DGS model from scratch. However, these methods are limited by learning short words or tags, which hinders advanced 3D scene understanding with complex sentence queries such as the referring segmentation task, as mentioned in [He et al., 2025]. More recently, ReferSplat [He et al., 2025] proposes to cope with segmentation from complex queries, and focus on the 3D referring segmentation task (RES) by providing human-annotated observable objects’ masks and corresponding detailed natural language sentence descriptions for training. However, all the aforementioned

† Work has been done during an internship at NVIDIA Corresponding author & Project lead: chengs@nvidia.com © 2026 NVIDIA. All rights reserved.

A soft white sheep plush sitting on a chair next to the table.

A sitting stuffed bear with dark fur and a light beige nose.

Bert embedding

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

ReferSplat > 1 hour

2D masks + human annotated caption

###### OpenVoxel (ours), Training-Free, ~ 3min

Group Field Construction

[Figure 7]

A soft white sheep plush sitting on a chair next to the table.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Render

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Captioning

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Render

A sitting stuffed bear with dark fur and a light beige nose.

[Figure 26]

Captioning

Only 2D masks

[Figure 27]

[Figure 28]

- Figure 1: Comparison of lifting language into 3D representations between ReferSplat [He et al., 2025] and OpenVoxel. Note that ReferSplat additionally requires manual text annotations for training 3DGS to equipped with Bert embeddings, while ours is totally training-free and do not need any additional annotations.

approaches tend to leverage learned embeddings as language fields from visual-language aligned embeddings such as CLIP or DINO, limiting the capability of dealing with arbitrary textual queries [Kamath et al., 2024]. Also, obtaining detailed human annotation for training requires labor-intensive pre-processing for each scene, and training additional language fields for 3D representation is also time-consuming (more than one hour for each scene).

To address these issues, we propose OpenVoxel, a novel training-free framework that bypasses the speed and accuracy limitations of learning text embedding space. Instead of training 3D features to align with a fixed embedding manifold, our OpenVoxel directly populates the 3D scene with rich, human-readable text captions and proceeds with text-to-text retrieval for open-vocabulary 3D scene understanding, including open-vocabulary segmentation (OVS) and referring expression segmentation (RES). Given a Sparse Voxel Rasterization (SVR) model optimized from multi-view images of a 3D scene, we introduce a training-free algorithm of Training-Free Sparse Voxel Grouping to cluster all voxels into object-level groups by lifting and matching per-frame 2D segmentation maps obtained from SAM2 [Ravi et al., 2024]. This aggregation is crucial for moving from a per-voxel representation to a semantically coherent, object-level understanding and ensuring view-consistency. To further bridge voxels with textual description, we also introduce the strategy of Canonical Scene Map Construction to locate language attributes on top of this sparse voxel representation via Vision Language Models (VLMs) and Multi-modal Large Language Models (MLLMs), obtaining a full description of the entire scene. Finally, to answer an open-vocabulary query (e.g., a simple word or a complex sentence), we utilize MLLMs to perform Referring Query Inference via direct text-to-text retrieval. The MLLM compares the user’s query against the constructed scene map, enabling a flexible and powerful retrieval process that is not constrained by embedding proximity. In summary, the contributions of our methods are listed below:

- • A training-free framework for open-vocabulary 3D scene understanding : We introduce OpenVoxel, a novel pipeline that removes the need for learning 3D language embeddings and instead performs text-to-text reasoning using LLMs on top of the sparse voxel representation.
- • Group (object)-level captioning and scene map construction : We propose an unsupervised method that

- clusters sparse voxels into instance-level 3D masks and generates rich, descriptive captions for each instance using an MLLM, forming an informative scene map.
- • Superior performance in both complex and simple language queries for 3D scene understanding task : By combining instance captions with LLM-based reasoning, OpenVoxel outperforms previous studies on both open-vocabulary semantic segmentation (OVS) and referring expression segmentation (RES) tasks.

### 2. Related Works 3D scene representation with different primitives. Neural radiance fields (NeRF) [Barron et al., 2021; Chen

- et al., 2022; Chou et al., 2024; Fridovich-Keil et al., 2022; Johari et al., 2022; Kundu et al., 2022; Liu et al., 2022; Martin-Brualla et al., 2021; Mildenhall et al., 2021; Müller et al., 2022; Reiser et al., 2021; Suhail et al., 2022; Sun et al., 2022; Vora et al., 2021; Yu et al., 2021] have established a powerful paradigm for representing scenes as continuous volumetric fields, enabling high-quality novel view synthesis but suffering from slow training and rendering. To address these limitations, 3D Gaussian Splatting (3DGS) [Kerbl et al., 2023] and following extensions [Chen and Wang, 2024; Gao et al., 2024; Xu et al., 2025; Yu et al., 2024] represent scenes using anisotropic Gaussians that can be rendered in real time, significantly improving efficiency while preserving fidelity. More recent work on Sparse Voxel Rasterization (SVR) [Sun et al., 2025] introduces a discrete, sparse voxel representation that supports efficient rasterization-based rendering. These representations (i.e., NeRF, 3DGS, and SVR) collectively form the foundation for modern neural scene reconstruction and are increasingly adopted for downstream 3D understanding tasks.

Language-aligned 3D scene representation. A growing line of research [He et al., 2025,; Huang et al., 2025; Jang and Kim, 2025; Ji et al., 2025; Jun-Seong et al., 2025; Kerr et al., 2023; Lee et al., 2025; Liao et al., 2025; Liu et al., 2023; Peng et al., 2025; Piekenbrinck et al., 2025; Qin et al., 2024; Tian et al., 2025; Wu et al., 2024; Ye et al., 2024; Ying et al., 2024; Zhu et al., 2025] aims to equip 3D scene representations with open-vocabulary or language-aware capabilities so that further downstream tasks [Chen et al., 2024,; Haque

- et al., 2023; Hu et al., 2024; Huang et al., 2025; Lin et al., 2024; Liu et al., 2024; Mirzaei et al., 2023,, 2024; Wang et al., 2024,; Weber et al., 2024; Weder et al., 2023; Wu et al., 2025; Yin et al., 2023] such as 3D scene editing or inpainting are applicable. LangSplat [Qin et al., 2024] introduces language-aligned features into 3D Gaussians by distilling CLIP embeddings and multi-scale SAM [Kirillov et al., 2023] masks from multi-view images, enabling open-vocabulary 3D segmentation. OpenGaussian [Wu et al., 2024] and Dr.Splat [Jun-Seong et al., 2025] further enhance this pipeline by improving the quality of language features in Gaussian primitives through codebooks. ReferSplat [He et al., 2025] tackles the challenging 3D referring expression segmentation (RES) task, aligning Gaussian features with sentence-level embeddings and predicting object masks conditioned on natural-language sentence queries. Despite promising results, these methods rely heavily on text-encoder embedding spaces and additional human annotation of “sentence-object mask” pairs on the training images, which limit their practicability on new scenes and ability to interpret nuanced or arbitrarily phrased queries.

In contrast, our method avoids the requirements of human-annotated description-mask pairs or embedding alignment entirely. Instead of training 3D features to match a fixed latent space, we generate rich captions for 3D instances and perform direct text-to-text retrieval using MLLMs, enabling more flexible reasoning over complex queries.

### 3. Preliminary

We start by reconstructing scenes from multi-view input images. To this end, we employ the recently emerged SVR [Sun et al., 2025], which reconstructs scenes as sparse voxel fields and achieves high-quality reconstruction and real-time rendering. The reconstructed voxels are also suitable for efficiently assigning attributes to certain 3D locations, which our algorithm exploits. Their rendering follows the classical volume rendering equation,

Pre-Trained Sparse Voxel Model Canonical Scene Map Construction

###### Training-Free Sparse Voxel Grouping

[Figure 29]

[Figure 30]

[Figure 31]

𝑆

𝑰𝟏:𝑲

###### Views: 𝝃𝟏:𝑲

Group Masks + Images

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

###### Scene Map

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

𝑴𝟏:𝑲

[Figure 45]

Apple, light green with fluffy material … , on table

[Figure 46]

[Figure 47]

SAM masks

- {"id": 1, "position": [

- -0.399, 2.06,
- -1.34], "caption": “ Apple, light green with fluffy material and a prominent, dark green leaf attached to its stem, placed on table"},

- {"id": 2, "position": [ 0.387, 1.88, -0.534], "caption": ”toy dog, yellow with cartoonish looking, smooth plastic surface …

[Figure 48]

[Figure 49]

[Figure 50]

Canonical captioning

ℱ : 

Render

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

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Render

Toy Dog,

|[Figure 68]|
|---|

[Figure 69]

yellow with cartoonish looking, …

Canonical captioning

[Figure 70]

[Figure 71]

[Figure 72]

𝑽𝟏:𝑵

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

| | |[Figure 80]| |
|---|---|---|---|
| | |[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]|[Figure 86]<br><br>[Figure 87]|
| | | | |

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Inference

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Referring

A funny toy with spindly legs that looks very interesting in the sunlight.

|[Figure 99]|
|---|

Query

| | |
|---|---|
| | |

Toy, Yellow, Slim legs

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

ID: {2} Render

Query Refinement

Text-to-text Retrieving

Object Mask in Query View

𝐷

Query Image

Query Description

- Figure 2: Overview of OpenVoxel. Taking a sparse voxel model 𝑉1:𝑁 pre-trained from multi-view images 𝐼1:𝐾 and their corresponding camera pose 𝜉1:𝐾, we aim to build a voxel group field ℱ1:𝑁 from segmentation masks 𝑀1:𝐾 obtained from SAM2. With ℱ1:𝑁 we render images and masks for all groups to obtain canonical captions and construct a Scene Map 𝑆 recoding their position and captions. In the Referring Query Inference stage we take a query description and image to find the ideal target group that matches the description by query refinement and text-to-text retrieving from 𝑆, enabling complex segmentation tasks such as referring expression segmentation (RES).

which is also the one employed by NeRF [Mildenhall et al., 2021] and 3DGS [Kerbl et al., 2023]:

𝑤𝑖(r) · 𝑐𝑖, 𝑤𝑖(r) = 𝛼𝑖(r) · ∏︁ 𝑗<𝑖

C(r) = ∑︁

(1 − 𝛼𝑗(r)), (1)

𝑖<𝑁

where C(r) is the rendering result for ray r, 𝑖 goes through an ordered list of 𝑁 voxels in front of the ray, 𝑐𝑖 is the voxel attribute (usually color) to render, 𝛼𝑖(r) is the opacity of the ray passing through the voxel, and finally 𝑤𝑖(r) is the contributing weight from the voxel to the rendered ray, which is also used in our algorithm during grouping.

### 4. Methodology

We first define the notation and problem statement of our framework. As shown in Fig. 2, given a 3D scene reconstructed as 𝑁 sparse voxels {𝑉𝑖}𝑁𝑖=1 from a set of 𝐾 multi-view images {𝐼𝑖 ∈ R𝐻×𝑊×3}𝐾𝑖=1 with their camera poses {𝜉𝑖 ∈ SE(3)}𝐾𝑖=1, our goal is to assign descriptive language information to these voxels to construct a scene map 𝑆 and enable open-vocabulary reasoning from a natural language description 𝐷, which can be a vocabulary as in 3D-OVS [Liu et al., 2023] or a referring expression as in 3D-RES [He et al., 2025]. We target gradient-descent-free and latent-dimension-free approaches to save space and time.

To this end, we present OpenVoxel, which consists of three main components: (1) Training-Free Sparse Voxel Grouping that groups voxels into coherent 3D instances (Sec. 4.1), (2) Canonical Scene Map Construction that stores the detailed captions and 3D spatial information for all voxel groups (Sec. 4.2), and

[Figure 104]

𝑀𝟐𝒑𝒓𝒐𝒋

𝑴𝟏 𝑴𝟐

View1 SAM mask View2 Raw SAM mask

[Figure 105]

[Figure 106]

Compare & Match/Merge

2D-to3D projection

Render to View2

Assign & Merge Exist ID & Find New ID

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

| | |
|---|---|
|ℱ𝟏:𝑵𝟏 ,𝑊𝟏:𝑵𝟏 ,𝐺| |

Update

𝑴′𝟐

View2 Matched SAM mask

ℱ𝟏:𝑵𝟎

Initialized Group Field

3D projection

Update

Initialized Feature Weight 𝑊𝟏:𝑵𝟎

2D-to-

###### Group Dictionary 𝐺

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

|[Figure 130]<br><br>[Figure 131]|
|---|

| |
|---|

Repeat

ℱ :  = ℱ𝟏:𝑵𝑲

ℱ𝟏:𝑵𝟐 ,𝑊𝟏:𝑵𝟐 ,𝐺

Final Group Field

Final Feature Weight 𝑊 :  = 𝑊𝟏:𝑵𝑲

𝐺 = 𝐺

Final Group Dictionary

- Figure 3: Detail of the grouping process. Taking the pre-trained voxel model 𝑉1:𝐾, we initialize the Group Field ℱ1:0𝑁, Feature weight 𝑊1:0𝑁 as empty tensors, and Group Dictionary 𝐺0 as empty dictionary. Then start from 𝜉1, we project the SAM masks 𝑀1 to 3D voxel and update ℱ1:𝑁, 𝑊1:𝑁, and 𝐺. By match and merge masks from the other views repeating this process, the final ℱ1:𝑁, 𝑊1:𝑁, and 𝐺 is able to represent the group information of 𝑉1:𝑁.

(3) Referring Query Inference to support querying and reasoning with natural language input from users (Sec. 4.3).

#### 4.1. Training-Free Sparse Voxel Grouping

Drawing inspiration from deep hough voting [Qi et al., 2019] and spatial embedding [Neven et al., 2019], our key insight is that voxels belonging to the same object instance should cluster around a common 3D center position. With this motivation, we encode group membership of voxels as 3D spatial coordinates pointing to their instance centroids. These spatial embedding, termed as group feature, is progressively updated by readily available 2D segmentation masks from foundation models like SAM2 [Ravi et al., 2024]. Our update is efficiently done in one training-free pass through all 𝐾 views without the lengthy gradient descent. In contrast, existing methods like Gaussian Grouping [Ye et al., 2024] rely on gradient descent to update per-primitive high-dimensional features with sequential training processes.

###### Group Representations.

- As depicted in Fig. 3, given a reconstructed scene of 𝑁 voxels, we extend voxel attributes by a group feature

𝐹1:𝑁 ∈ R𝑁×3, which denotes the instance centroid the voxels belonging to, and a feature weights 𝑊1:𝑁 ∈ R𝑁 indicating the confidence about the current feature. We also use a Group Dictionary 𝐺 to record the spatial

centroid of each unique instance. The group features, weights, and dictionary are iteratively updated by the instance information from each frame.

###### Lifting 2D Segmentation Maps to 3D Group Field.

We first introduce our method to lift a 2D instances mask into 3D voxels. Let 𝑀 be the mask with 𝑚 unique instances ID, and fpts ∈ R𝐻𝑊×3 be the rendered point map of the view, which indicates the expected ray-hit position in the world space. The centroid f𝑘center ∈ R3 of each instance 𝑘 is then determined by performing masked average reduction on the point map:

f𝑘center = ∑︀

𝑗∈𝐻𝑊 (𝑀 𝑗 = 𝑘) · f𝑗pts ∑︀

. (2)

𝑗∈𝐻𝑊 (𝑀 𝑗 = 𝑘)

We then accumulate the instance centroids information into each voxel 𝑖 by:

###### ℱ𝑖𝑡+1 = ℱ𝑖𝑡 + ∑︁

𝑤𝑖𝑗 f𝑀center

,

𝑗

𝑗∈𝐻𝑊

(3)

𝑊𝑖𝑡+1 = 𝑊𝑖𝑡 + ∑︁

𝑤𝑖𝑗,

𝑗∈𝐻𝑊

where 𝑤𝑖𝑗 is the blending weight (Eq. (1)) the voxel 𝑖 contributing to the pixel 𝑗. In practice, we customize the sparse voxel renderer [Sun et al., 2025] to perform the update in one rendering pass. The update equation follows Dr.Splat [Jun-Seong et al., 2025], while we do not need the top-k sampling as our group feature is only 3-dimensional.

###### Start condition.

- At view 𝜉1, the voxel group feature 𝐹1 and weight 𝑊1 are directly assigned by the information lifted from the view. The group dictionary is set to the 𝑚 instance IDs and their corresponding centroids fcenter.

###### Progressively Matching and Merging Segmentation Masks.

When we move on to view 𝑡 + 1, we need to first match the mask IDs in 𝑀𝑡+1 with the existing voxel grouping (𝐹𝑡,𝑊𝑡,𝐺𝑡). To this end, we render our group field into view 𝑡 + 1 as 𝑀𝑡𝑝𝑟𝑜𝑗+1 and match existing instances with those in 𝑀𝑡+1. The matched instance IDs are replaced with the existing IDs, while the remaining are added as new instances. Specifically, we first determine the instance ID of each voxel 𝑖 in the current iteration by:

ℱ𝑖𝑡 𝑊𝑖𝑡 − 𝐺𝑡𝑗⃦

, (4)

ID𝑡𝑖 = argmin𝑗 ⃦

2

where the first term is the normalized voxel voting and the second term is the centroid of each instance. We then cast rays from the processing view to retrieve the instance ID of the hit voxel, which forms a instance mask 𝑀proj reflecting the existing segmentation. Then, we conduct matching by finding the highest IoU mask in 𝑀𝑡+1 for each instance in 𝑀proj. We also prompt the SAM2 model again using 𝑀proj and merge the instance masks that are largely overlapped to reduce noise and prevent the same instance from being incorrectly separated as two different IDs. As for the masks in 𝑀𝑡+1 that are neither matched nor overlapped with the existing ones, we assign a new unique ID and add them to the group dictionary. Finally, we use the updated mask from 𝑀𝑡+1 to update the voxel group feature into (𝐹𝑡+1,𝑊𝑡+1) by Eq. (3). After processing all the 𝐾 views, we use the 𝐼𝐷𝐾 from Eq. (4) as the final voxel instance grouping.

It is worth noting that, with our Group Field construction strategy, even if the voxels are assigned to an incorrect group in some view, the accumulated 𝐹1:𝑁 and 𝑊1:𝑁 still force the voxel to “vote” the most confident group it belongs, reducing the possible miss-grouping.

Include Visual Prompt (Red Dot)

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Group Masks + Images

[Figure 136]

Please look at the region high-lighted by

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Apple, light green with fluffy material … , on table

red dot and consider the given caption.

Green round

[Figure 142]

[Figure 143]

Refine the caption in the template of <category>, <appearance details>, <function>, <placement>

Captioning object, …

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Yellow figurines, four slim legs …

Toy Dog, yellow with cartoonish looking, …

Refine the caption in the template of

[Figure 150]

[Figure 151]

<category>, <appearance details>, <function>, <placement>

Captioning

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Include Visual Prompt (Red Dot)

- Figure 4: Detail of the Canonical Captioning. Given the group masks rendered of a specific group (taking the green apple as example) from our group field and their corresponding images, we leverage the Describe Anything Model (DAM) to first obtain a detailed caption. Then a Qwen3-VL model is conducted to canonicalize the caption into a fixed form, benefiting further usage.

#### 4.2. Canonical Scene Map Construction

Once voxel groups are obtained, our next goal is to construct a canonical scene map 𝑆 that records the position of each group and describes each group with a rich, human-readable caption, enabling open-vocabulary understanding of the entire scene (Fig. 2). Instead of training 3D features to align with language embeddings, we bypass this bottleneck by leveraging Multimodal Large Language Models (MLLMs) to obtain detailed and accurate canonical captions for each group. We now detail the canonical captioning process.

As depicted in Fig. 2 and Fig. 4, taking the green apple as an example, we render the binary masks of the corresponding group in views 𝜉1:𝐾 and use them as mask inputs together with the original images 𝐼1:𝐾 for the Describe Anything Model (DAM) [Lian et al., 2025]. DAM produces a detailed description of the content inside each mask; however, its output is a free-form sentence and, in some cases, uses the generic word “object” as the subject (e.g., “A green round object, possibly an apple, ...”). To stabilize and standardize the captions across groups, we further use an MLLM (e.g., Qwen3-VL) to refine these free-form captions into a fixed-form caption while considering (i) the DAM sentence, and (ii) the original masked images.

Because general MLLMs are not specifically trained for mask-image captioning, we design the visual prompting strategy inspired by [Wu et al., 2024]: we darken the regions outside the mask and highlight the object with a small red dot to focus the model’s attention. We then instruct the MLLM to rewrite the caption into the canonical template <category noun>, <appearance details> <function/affordance or partof> <placement/relation>. This converts each group’s description into a consistent, structured form. The resulting entry for a group in 𝑆 stores its ID, 3D position (center), and the canonical caption. In practice, the canonicalization significantly reduces subject ambiguity (e.g., replacing “object” with “apple”) and yields stable, comparable descriptions across views.

- Table 1: Quantitative evaluation on the Ref-LeRF subset in terms of referring expression segmentation (RES) mIoU. ReferSplat* denotes the best results we reproduced from the official implementation. Note that only ReferSplat requires ground truth annotation of description-mask pair for training.

Method Requires GT Annotation ramen figurines teatime kitchen avg. Grounded SAM [Ren et al., 2024] - 14.1 16.0 16.9 16.2 15.8 LangSplat [Qin et al., 2024] - 12.0 17.9 7.6 17.9 13.9 SPIn-NeRF [Mirzaei et al., 2023] - 7.3 9.7 11.7 10.3 9.8 GS-Grouping [Ye et al., 2024] - 27.9 8.6 14.8 6.3 14.4 GOI [Qu et al., 2024] - 27.1 16.5 22.9 15.7 20.5 ReferSplat* [He et al., 2025] ✓ 31.0 20.0 25.4 21.4 24.5 ReferSplat [He et al., 2025] ✓ 35.2 25.7 31.3 24.4 29.2

OpenVoxel (Ours) - 52.5 43.5 48.4 25.1 42.4

#### 4.3. Referring Query Inference

Given a user query 𝐷—from a single category token (e.g., “chair”) to a detailed referring expression (e.g., “what object can be used for cutting paper?”)—and an optional query image of the target view, our goal is to identify the best-matching 3D instance in the scene map 𝑆, obtain its ID(s), and render the corresponding binary mask in the target view. Rather than mapping the query and all captions into a learned embedding space, we perform direct retrieval over 𝑆 with a multimodal LLM (MLLM). This design avoids any additional training or calibration for our 3D setting, maintains robustness to free-form phrasing that includes attributes, affordances, or light reasoning, and yields interpretable evidence because the selection is made over the canonical captions stored in 𝑆.

###### Query Refinement.

To make matching deterministic and stable, we first canonicalize the query. Following the same procedure as in our captioning stage, the MLLM rewrites 𝐷 (with or without the query image) into our fixed template <category noun>, <appearance details> <function/affordance or part-of> <placement/relation>. For instance, the free-form description “A funny toy with spindly legs that looks very interesting in the sunlight” becomes “Toy, yellow, slim legs.” The refined query now matches the format of entries in 𝑆, so the retrieval process reduces to selecting the caption(s) in 𝑆 that best satisfy the refined description and returning their associated ID (In practice, we input the entire 𝑆 for the MLLM, so even if the query implies spatial relations (e.g., “left of the apple”), the MLLM can still possibly verify them using the stored group centers in 𝑆 before finalizing the selection.

###### Text-to-text Retrieving of Target Group.

Finally, given the selected ID(s), we render the referred object by rasterizing only the corresponding group(s) under the target view, producing a binary mask and the linked canonical caption from 𝑆. This inference procedure is fully training-free, applies uniformly to both OVS (category-only queries) and RES (attribute/affordance/relation queries), and empirically provides stable matches across scenes and phrasing styles while remaining simple to implement.

###### Table 2: Quantitative evaluation on the LeRF-OVS subset in terms of mIoU.

Method ramen figurines teatime avg. Feature-3DGS [Zhou et al., 2024] 43.7 58.8 40.5 47.7 LEGaussians [Shi et al., 2024] 46.0 60.3 40.8 49.0 LangSplat [Qin et al., 2024] 51.2 65.1 44.7 53.7 GS-Grouping [Ye et al., 2024] 45.5 60.9 40.0 48.8 GOI [Qu et al., 2024] 52.6 63.7 44.5 53.6 3DVLGS [Peng et al., 2025] 61.4 73.5 58.1 64.3 ReferSplat [He et al., 2025] 55.1 67.5 50.1 57.6 CCL-LGS [Tian et al., 2025] 62.3 61.2 71.8 65.1 OpenVoxel (Ours) 62.5 60.7 75.4 66.2

### 5. Experiments

- 5.1. Settings Datasets.

Following prior state-of-the-art works [He et al., 2025; Kerr et al., 2023; Qin et al., 2024], we evaluate the openvocabulary segmentation (OVS) and referring expression segmentation (RES) performance of our OpenVoxel primarily on subsets of the iPhone Polycam-captured LeRF dataset [Kerr et al., 2023]. For the OVS task, we follow the evaluation protocols in [Ye et al., 2024; Zhu et al., 2025] using the LeRF-Mask subset, which contains three scenes (ramen, figurines, and teatime) with 6, 7, and 10 queryable objects, respectively. We further conduct experiments on the LeRF-OVS subset [Qin et al., 2024], with the same three scenes with 13, 17, and 12 annotated objects, respectively. For the RES task, we use the Ref-LeRF subset [He et al., 2025], which consists of four scenes (ramen, figurines, teatime, and kitchen) containing 13, 17, 12, and 11 objects, respectively, each paired with sentence-level natural language referring expressions.

Implementation details.

We conduct our OpenVoxel with PyTorch [Paszke et al., 2019] libraries, where our code base is built on top of SVR [Sun et al., 2025]. For the foundation models, we use SAM2 [Ravi et al., 2024] to obtain the per-view segmentation map separately and also prompt SAM2 to conduct the mask merging operation during the Training-Free Sparse Voxel Grouping, as described in Sect. 4.1. On the other hand, the Describe Anything Model(DAM) [Lian et al., 2025] is used as our captioning model. As for the MLLM used to refine the obtained captions/query and conduct the target retrieving, we use Qwen3-VL-8B-Instruct [Bai et al., 2023, 2025; Wang

- et al., 2024]. Please refer to our supplementary material for more implementation details, including engineering tricks and the system prompt we use.

5.2. Quantitative Evaluations

Ref-LeRF.

We first present the RES results on the Ref-LeRF subset in Table 1, as this task best demonstrates the advantages of our design. Since the official pre-trained models and data of the current state-of-the-art method, ReferSplat [He

- et al., 2025], do not cover all four scenes, we reproduce its results following the original implementation and configurations, denoted as ReferSplat*. As shown in Table 1, despite OpenVoxel not requiring any description–mask annotations during grouping or scene map construction, it achieves a notable improvement of

“A compact black and white photographic device with a lens protruding slightly from the body.”

“A funny toy with spindly legs that looks very interesting in the sunlight.”

“An item printed with Pikachu and other decorative motifs on a smooth surface.”

“A fun doll placed on top of the Rubik's Cube and in the center of the table, quiet and nice.”

“A minimalist style toy with natural grooves next to a red apple.”

[Figure 156]

ReferSplatOpenVoxel(Ours)GroundTruth

###### Figure 5: Qualitative results of RES task on the Figurines scene.

- Table 3: Quantitative evaluation on the LeRF-mask subset in terms of IoU and BIoU. Note that BIoU indicates the boundary IoU of predicted mask and ground-truth mask. Best results are in bold text, and second best results are underlined.

figurines ramen teatime Avg.

Model

mIoU mBIoU mIoU mBIoU mIoU mBIoU mIoU mBIoU

DEVA [Cheng et al., 2023] 46.2 45.1 56.8 51.1 54.3 52.2 52.4 49.5 LERF [Kerr et al., 2023] 33.5 30.6 28.3 14.7 49.7 42.6 37.1 29.3 SA3D [Cen et al., 2023] 24.9 23.8 7.4 7.0 42.5 39.2 24.9 23.3 LangSplat [Qin et al., 2024] 52.8 50.5 50.4 44.7 69.5 65.6 61.2 56.1 GS Grouping [Ye et al., 2024] 69.7 67.9 77.0 68.7 71.7 66.1 72.8 67.6 Gaga [Lyu et al., 2024] 90.7 89.0 64.1 61.6 69.3 66.0 78.5 74.2 ObjectGS [Zhu et al., 2025] 88.2 85.2 88.0 79.9 88.9 88.6 88.3 84.4 OpenVoxel (Ours) 90.8 89.0 84.9 73.2 85.8 82.1 87.2 81.4

13.2% over the result reported in the original ReferSplat paper, and an even larger margin of 17.9% compared to our reproduced version. During reproduction, we observe that ReferSplat’s strategy of learning sentence-level embeddings for 3D representations tends to overfit the seen descriptions, resulting in unstable evaluation performance. In contrast, our OpenVoxel consistently achieves higher accuracy, and the superior performance of our text-level retrieval mechanism supports its effectiveness and robustness compared to learned embedding approaches.

###### LeRF-OVS and LeRF-Mask.

Tables 2 and 3 summarize the OVS results on the LeRF-OVS and LeRF-Mask subsets, respectively. For LeRF-OVS, we report mean Intersection-over-Union (mIoU) between predicted and ground-truth masks, following [He et al., 2025; Qin et al., 2024]. For LeRF-Mask, we additionally evaluate mean Balanced IoU (mBIoU), following [Ye

- et al., 2024; Zhu et al., 2025]. As shown in Table 2, the OVS task involves relatively simpler queries, leading to higher overall performance across methods—even when evaluated on the same target objects as in Ref-LeRF. For LeRF-Mask, where the queries are fewer and less ambiguous, all current state-of-the-art methods (including ours) achieve over 70% mIoU. Although our design primarily targets the RES setting, OpenVoxel still performs

“An object containing noodles and various toppings on a table with sloping edges.”

“Slender strips of food in the center of the bowl, mixed with toppings”

“Smooth slices with a delicate spiral pattern featuring a swirl pattern.”

“A pair of utensils for holding food next to a yellow bowl.”

“A drinking utensils with a smooth surface near a sake bottle”

[Figure 157]

ReferSplatOpenVoxel(Ours)GroundTruth

###### Figure 6: Qualitative results of RES task on the Ramen scene.

competitively on both OVS benchmarks, demonstrating its flexibility and robustness across varying query complexities.

#### 5.3. Qualitative Results

We show the qualitative results of “figurines” and “ramen” scene in Ref-LeRF subset in Fig. 5 and Fig. 6, respectively. In both scenes, ReferSplat often identifies only partial regions of the target object. For example, in Fig 5, when we input the query “A minimalist style toy with natural grooves next to a red apple”, ReferSplat focuses on the region near the apple (the apple is at the left of the image), without localizing the correct object. In contrast, OpenVoxel retrieves the toy pumpkin properly and render a smooth mask. As for the query “A drinking utensils with a smooth surface near a sake bottle” in Fig. 6, ReferSplat only focus on the term “drinking utensils” and incorrectly segments both the glass of water and the sake bottle, while our OpenVoxel locates the sake cup correctly. These results further support our claim of using explicit captions to build a scene map and perform text-level retrieval, which benefits the RES tasks.

#### 5.4. Further Analysis Ablation Study.

- Table 4 shows the ablation studies of our OpenVoxel on the Ref-LeRF subset. Start from the baseline A where we do not merge overlapping small masks as described in Sect. 4.1 during the voxel grouping, and also do not canonicalize both the caption and the query. We can see that even with the explicit captioning and text-to-text retrieval, it is still hard to retrieve the correct target due to noisy small groups and misalignment of language format. After that, we include the mask merging strategy in model B, showing a slight improvement of 3.7% mIoU due to less noisy groups. Then, in model C, we introduce the canonical captioning for each group while constructing the scene map, showing a large improvement of 8.4% mIoU because of more specific captioning (i.e., standard template for each caption and reducing the word “object” as subject noun). Lastly, our full model canonicalizes the input text query into the same form as the canonical caption, introducing more precise alignment and less ambiguity, and showing the best performance. The above ablation study verifies the necessity of each of our designs.

- Table 4: Ablation study of our OpenVoxel on Ref-LeRF subset. We integrative evaluate the effectiveness of several components of our OpenVoxel.

Method Mask Merging Canonical Caption Canonical Query mIoU

- A - - - 24.3

- B ✓ - - 28.0

- C ✓ ✓ - 36.4 Ours ✓ ✓ ✓ 42.4

- Table 5: Run time comparison with current SOTAs. For the term “Requires Training”, it is to indicate if a method needs gradient-based training to obtain the semantic or language field. All methods are reproduced on a single RTX 5090 GPU.

###### Method Requires Training Run Time

ReferSplat [He et al., 2025] yes >1 hr ObjectGS [Zhu et al., 2025] yes ∼ 40 min

OpenVoxel (Ours) no ∼ 3 min

Runtime Estimation. We show the comparison of run time between our OpenVoxel and current SOTAs on OVS and RES tasks in

- Table 5 on a single RTX 5090 GPU. The term “Requires Training” here indicates if a method needs training to obtain its feature/language field. Although ObjectGS [Zhu et al., 2025] combines the training process of 3DGS and the per-Gaussian object feature (similar to Gaussian Grouping), we note that running a vanilla 3DGS on RTX 5090 only takes about 10 minutes, indicating ObjectGS needs an additional 30 minutes to obtain the object feature. On the other hand, although ReferSplat reports that it only takes 58 minutes to train on an A6000 GPU, in all our reproductions, we find that following the official implementation configuration takes at least 2 hours per scene to obtain the best results. In contrast, our OpenVoxel requires only about 3 minutes to complete the voxel grouping and canonical scene map construction, which is at least 10 × faster than other SOTAs, and the inference time per query is less than 1 sec.
- 6. Conclusion

In this paper, we propose OpenVoxel, a training-free framework for grouping and captioning sparse voxel rasterization (SVR) models. In our Training-Free Voxel Grouping process, we curate a group field for a pretrained SVR model by matching and merging per-view SAM2 masks. As for the Canonical Scene Map Construction we leverage VLMs and MLLMs to obtain a fix form of caption for each constructed group and then collect positions and captions of all groups as the scene map. And finally, we conduct a Referring Query Inference to retrieve target objects from our scene map with an input descriptive query. Through quantitative and qualitative experiments, we verify the efficiency and capability of OpenVoxel.

### Appendix

In this supplementary material, we provide additional method details in Sec. A and more results of semantic segmentation in Sec. B, ablation studies in Sec. C, visualization in Sec. D. Finally, we discuss about our limitation in Sec. E.

### A. Details

###### Merging groups.

- In Sect. 4.1, we mentioned merging small segmentation masks by re-prompting SAM2 [Ravi et al., 2024] to reduce noise. Here, we visualize this process for a better understanding. As depicted in Fig. 3 and Fig. 7, the

“hand sculpture” instance is separated into two different groups in 𝑀2𝑝𝑟𝑜𝑗 since the SAM2 mask 𝑀1 treats them as two different segment, where one group covers almost the entire hand and the other just representing a finger tip. To encourage each of our groups to be more instance-wise, we double-check if any two groups should be merged by prompting all groups that are observable under the current view with SAM2 again. Taking the hand sculpture under view 𝜉2 as example in Fig. 7, we create the point prompts for SAM2 by treating several pixels sampled from the group in 𝑀2𝑝𝑟𝑜𝑗 of interest as positive prompts, and also select several other groups and treat their center pixel in 𝑀2𝑝𝑟𝑜𝑗 as negative prompts. As for the mask prompt, we treat the entire group of interest in 𝑀2𝑝𝑟𝑜𝑗 as a positive prompt (mask value: 20), all the other known groups as a negative prompt (mask value:−20), and the rest of the region is treated as unknown (mask value:0). After using this setting as prompt for SAM2, if the output mask of one group (e.g., the finger tip in Fig. 7) is almost lie inside another group’s mask (e.g., over 90% inside the hand sculpture), then we merge the smaller group into the larger group accordingly (i.e., merge the finger tip into the hand sculpture). As shown in our ablation study Table 4, a slight improvement is observed by including this merging strategy.

MLLM prompt.

- In Sect. 4.2 and 4.3, we leverage an MLLM model of QWen3-VL-8B to conduct canonical captioning, prompt refinement, and target retrieving. The system prompts are as List 1, List 2, and List 3.

We note that we do not spend much effort exploring different kinds of system prompt design. We simply use ChatGPT by describing our task with some examples and ask it to provide system prompts that suit Qwen-VL [Bai et al., 2023, 2025; Wang et al., 2024] models, and the same prompts are shared for both RES and OVS tasks. Therefore, these system prompts may not be the optimal ones, and users still have a chance to improve the performance by using our pipeline and simply changing system prompts.

###### Implementation.

We now elaborate on the implementation details. Since our method is totally training-free after having the pre-trained SVR model of each scene, reducing the process time of the following processes (i.e., grouping, captioning, and retrieving) is a crucial problem. To achieve this goal, we do not always go through all the images from the training set for each scene in practice. Instead, taking the LeRF [Kerr et al., 2023] dataset and corresponding subsets as an example, we uniformly sample the processed views to make sure the total processed view of each scene does not exceed 150. Also, we found that since the merging process requires additional SAM2 execution, conducting this merging process for all views would slow down the inference time. Hence, we conduct the merging (re-prompting SAM2) process for each scene per 1 to 5 steps to speed up the inference. Furthermore, when prompting the captioning model (i.e., DAM [Lian et al., 2025]), we only sample 8 frame-mask pairs (pad to 8 pairs if not enough) for each group to reduce the visual tokens for the model, making sure that the inference is fast. Also, for the usage of MLLM for Canonical Captioning, Query Refinement, and Text-to-text retrieving, we “DO NOT” provide any visual example for the MLLM as in-context examples

[Figure 158]

[Figure 159]

[Figure 160]

Point + Mask Prompt

𝑀𝟐𝒑𝒓𝒐𝒋 GroupID:{5}

SAM2

[Figure 161]

Green: Positive Red: Negative Blue: Unknown

Inside

Green: Positive Red: Negative

Merge Group

[Figure 162]

[Figure 163]

[Figure 164]

Group ID: {30}

SAM2

Point + Mask Prompt

###### Figure 7: Detail of the mask merging process.

since the inference time would be slowed down by doing so (although we acknowledge that conducting these information might bring better performance, we leave this as a future direction to explore the balance between adding visual demonstrations and inference time).

- 1 CANONICAL_CAPTION_SYSTEM_PROMPT= """You are a detail-focused visual caption

- 2 refiner for open-vocabulary segmentation and referring grounding.

- 3 INPUTS

- 4 - A short video where ONE region is masked (highlighted); the outside area

- 5 is darkened.

- 6 - A rough caption from another model (may be incorrect or misleading).

- 7 SCOPE

- 8 - Describe ONLY what lies INSIDE the masked region across frames.

- 9 - Never name or infer unmasked neighbors as the subject.

- 10 - Be factual; do not guess hidden details.

- 11 - If the original caption conflicts with the visual evidence, IGNORE it and

- 12 correct the errors.

- 13 REWRITE GOAL

- 14 Produce a precise, natural description with a clear class noun and

- 15 discriminative details that is easy to match with open-vocabulary queries.

- 16 CORE RULES

- 17 1) Class noun: Replace vague words ("object/thing/item/surface") with a

- 18 concrete class or fine-grained subtype.

- 19 2) Part-of decision (strict):

- 20 - Use "part of <larger object>" ONLY if ALL are true:

- 21 a) Visible physical continuity/attachment within the mask (seam/stitch/

- 22 joint/hinge/fastener or continuous material/geometry),

- 23 b) The region is an intrinsic component,

- 24 c) The larger object's category is visibly identifiable.

- 25 - Otherwise DO NOT use "part of". Prefer placement instead.

- 26 3) Placement vs background:

- 27 - Use view-independent placement for surfaces/containers ("on table",

- 28 "inside pouch", "on plate", "on shelf", "in tray", etc.).

- 29 - If the region is background material/texture (floor/wall/ceiling/ground),

- 30 begin with "background: <material/surface>".

- 31 4) Printed-content rule (critical):

- 32 - Scan ALL frames within the mask for printed text, logos, labels,

- 33 or characters.

- 34 - If ANY are visible, include at least one cue:

- 35 - Text: transcribe exact readable tokens (keep visible case/punctuation).

- 36 If partial, include the visible substring.

- 37 - Character/graphic: if identity is uncertain,

- 38 describe visual attributes neutrally (e.g., "purple cartoon dinosaur");

- 39 do not guess names.

- 40 5) Detail quota:

- 41 - Include at least FOUR distinct cues chosen from: color; material;

- 42 texture/pattern; shape/geometry; subtype/model; visible text/logo/

- 43 printed character; state/condition; function/affordance;

- 44 part-of (only if allowed); placement/relation (max 2).

- 45 6) Language hygiene:

- 46 - View-independent wording only (no left/right/front/top; no camera terms).

- 47 - Do not mention the highlight/red dot.

- 48 - Forbidden words: object, thing, item (and similar generic fillers).

- 49 OUTPUT FORMAT (canonical; must be strictly followed)

- 50 - EXACTLY ONE line, 12-20 words, comma-separated phrases, no period.

- 51 - Start with the SUBJECT noun.

- 52 - **Order of phrases (strict):**

- 53 1) <category noun> (table, chair, bottle, pouch, human character, cat,

- 54 dog, rabbit, camera, spoon, door handle, floor, wall, etc.)

- 55 2) <appearance details> (color/material/texture/pattern/shape/

- 56 subtype/text/logo)

- 57 3) <function/affordance or part-of> ("part of <larger object>"

- 58 only if rule 2 allows; otherwise a concise function/affordance like

- 59 "resealable pouch", "pour spout", "grip handle")

- 60 4) <placement/relation> (on/in/inside/attached to/against/between;

- 61 max 2 relations)

- 62 - If unsure between "part of" and placement, choose placement.

- 63 - For background regions: "background: <material/surface>,

- 64 <appearance details>, <(optional) function if any>, <placement/relation>".

- 65 STRICTNESS

- 66 The form of the output must strictly follow the rules above and the

- 67 ordered template:

- 68 <category noun (no color)>,(comma here)

- 69 <appearance details (color here)><function/affordance or part-of>

- 70 <placement/relation>.

- 71 The original caption may be wrong; rely on visual evidence to correct it.

- 72 """

Listing 1: System prompt of Canonical Captioning.

- 1 QUERY_REPHRASE_SYSTEM_PROMPT = """You are a helpful assistant.

- 2 You rewrite a short OVS query into ONE short canonical phrase

- 3 using ONLY the provided image and the raw query.

- 4

- 5 SCOPE

- 6 - Inputs:

- 7 (a) scene_map (JSON list of candidate objects with their coordinates and caption),

- 8 (b) query (description about an object visible in the image),

- 9 (c) view_image.

- 10 - Keep the result SHORT and human-judgable from the query mainly.

- 11 - You must NOT include any spatial relations in the output if not explicitly

- 12 mentioned in the query.

- 13

- 14 CANONICAL FORM

- 15 - Output exactly ONE short phrase in the form:

- 16 <class noun> <appearance> <placement?>

- 17 - 2 to 6 words, lowercase, spaces only, no punctuation.

- 18 - class noun: singular, most specific common name that is visually supported

- 19 (e.g., "rubber duck", "paper bag").

- 20 - appearance: brief, image-supported attributes (color/material/texture/

- 21 state/text/logo/shape). If unsure, keep generic

- 22 (e.g., "plastic-like", "transparent").

- 23 - placement (OPTIONAL): view-INDEPENDENT, simple scene phrase

- 24 (e.g., "on table", "in bowl", "on shelf", "in bag").

- 25 Avoid left/right/front/behind/above/below.

- 26

- 27 REPHRASE PROTOCOL (follow strictly)

- 28 - Be conservative if uncertain; never hallucinate specifics you cannot see.

- 29 - Examples:

- 30 - "toy car on the table" -> "car on table"

- 31 - "banana" -> "banana" (no assumed color)

- 32 - Materials: "plastic bag"->"plastic-like bag"; "nori"->"seaweed";

- 33 "glass cup"->"transparent cup"; "porcelain"->"ceramic"

- 34 - Common words: "gummy"->"gummy candy"; "ribeye beef"->"piece of meat";

- 35 "toy car"->"car"; "stuffed bear"->"teddy bear";

- 36 "paper napkin"->"napkin"; "kamaboko"->"small piece with pink swirl";

- 37 "rubber duck with a bouy"->"rubber duck with pink lei"

- 38 - Character names -> descriptions:

- 39 "pikachu"->"yellow character with long ears and possibly red cheek";

- 40 "jake"->"yellow cartoon character big eyes slim legs";

- 41 "miffy"->"rabbit character, possibly wearing garment";

- 42 "waldo"->"character red-white striped shirt";

- 43 "hello kitty"->"white cartoon cat red bow"

- 44 - Brands -> generic: "lays"->"potato chips"; "coca-cola"->"can";

- 45 "nike shoes"->"sports shoes";

- 46 "tesla door handle"->"metalic object, look like door handle"

- 47 - Ambiguous placements: "in the bowl"/"on the plate"/"inside the pouch"

- 48 -> "in container"/"on surface"/"in bag"

- 49

- 50 OUTPUT (STRICT)

- 51 Return ONE JSON line only (no extra text, no code fences, no reasoning):

- 52 {"canonical": "<clear class noun>, (you must include this comma after <clear class noun>) < appearance (color)>, <placement (ONLY IF contained in query text)>"}

- 53

- 54 CONSTRAINTS

- 55 - No chain-of-thought or explanations.

- 56 - Do not use any information that cannot plausibly be inferred from

- 57 the image + query alone.

- 58 - You MUST NOT use ambiguous noun like "object", "thing", "item", "stuff",

- 59 "part", "area", "region", "section", "portion", "background", "foreground",

- 60 "surface", "area of interest", etc.

- 61 - If the class noun is a general category

- 62 (e.g., "container", "food", "furniture"),

- 63 you MUST add more specific appearance to clarify.

- 64 - If the query does not contain any placement info,

- 65 DO NOT add any placement in the output.

- 66 """

- Listing 2: System prompt of Query Refinement.

- 1 SYSTEM_PROMPT_RETRIEVE = """You retrieve all matching targets using

- 2 scene_map + view_image (optional) + a canonical short phrase.

- 3

- 4 INPUTS

- 5 1) scene_map: JSON of candidate voxel groups with fields:

- 6 - id (integer, unique)

- 7 - caption (short description; copy EXACTLY in output)

- 8 - center: WORLD coordinates (use only for view-independent relations:

- 9 near/far/between/closest/farthest)

- 10 - optional: bbox/size/group/category

- 11 2) view_image (optional): one image for the current query instance

- 12 (targets may be occluded or off-frame).

- 13 3) canonical: the short canonical phrase from Stage 1

- 14 (e.g., "rubber duck, yellow", "paper bag, on table").

- 15

- 16 POLICY (caption-first, occlusion-robust)

- 17 - Primary signal: scene_map CAPTIONS (semantic match to the canonical phrase;

- 18 allow common synonyms/hypernyms).

- 19 - Ignore view-dependent relations (left/right/front/behind).

- 20 Use WORLD coords ONLY for near/far/between/closest/farthest

- 21 if such words appear.

- 22 - One real object may be split across multiple voxel groups (ids)

- 23 that are spatially adjacent and semantically consistent.

- 24 If so, RETURN ALL ids for that instance.

- 25 - If multiple separate instances match the canonical phrase,

- 26 RETURN the best aligned one.

- 27 - Secondary signal: view_image (if provided) only to veto

- 28 obvious mismatches when visible; do NOT penalize occlusion.

- 29

- 30 INTERNAL STEPS (do not reveal):

- 31 1) Match captions to the canonical phrase -- prioritize

- 32 exact/synonym class match, then attribute alignment.

- 33 2) Merge adjacent voxel groups that describe the same instance

- 34 (spatially close in WORLD coordinates and semantically consistent).

- 35 3) If canonical phrase includes near/far/between/closest/farthest,

- 36 apply these using WORLD centers/bboxes over the matched set.

- 37 4) Use the image (if provided) only to down-weight

- 38 clear visual contradictions when visible (do not discard due to occlusion).

- 39 5) Finalize ids and copy their captions EXACTLY from scene_map.

- 40

- 41 OUTPUT (STRICT)

- 42 Return EXACTLY one JSON line -- no extra text,

- 43 no code fences, no reasoning:

- 44 {"ids": [<int>, ...], "captions": ["<EXACT caption>", ...]}

- 45 Rules:

- 46 - Include at least one id for every matching instance

- 47 (multi-instance allowed, but in most case only one).

- 48 - Sort ids ascending within each instance; overall order is arbitrary.

- 49 - Captions must be copied EXACTLY from scene_map, same order as ids.

- 50 - You cannot return empty ids or ids that are not in scene_map.

- 51 - If borderline: you MAY add

- 52 {"candidates": [{"id": a}, {"id": b}]}

- 53

- 54 CONSTRAINTS

- 55 - No chain-of-thought or explanations.

- 56 - Do not paraphrase any caption in the "captions" array;

- 57 copy exactly from scene_map.

- 58 - Use WORLD coordinates only for view-independent relations;

- 59 do not use image axes for left/right/front/behind.

- 60 """

- Listing 3: System prompt of target retrieval.

### B. Semantic Segmentation

- B.1. Approach Different from OVS and RES, the task of semantic segmentation usually has a customized list of class candidates for each dataset. Therefore, instead of retrieving the matched groups for each class, we conduct semantic segmentation for OpenVoxel by choosing the best-matched class for each group.
- B.2. Dataset and implementation details

Dataset.

Following OpenGaussian [Wu et al., 2024], we conduct semantic segmentation on 10 different scenes on the Scannet [Dai et al., 2017] dataset. Each scene is represented as colored point clouds, with ground truth images and depth maps provided. Following the official setting, we conduct a 19 class semantic segmentation in our experiments.

Implementation details.

We note that in OpenGaussian, ground truth point clouds are directly utilized as initialization for the 3DGS, and they deactivate all the merging and splitting processes so that a perfect geometry alignment is naturally obtained for evaluation. However, it is not easy for our backbone (i.e., SVR [Sun et al., 2025]) to have such an initialization. To have a better geometry for the Scannet dataset, we utilize the provided depth map to guide the pre-training process of the SVR model.

- B.3. Evaluation protocols and results

###### Evaluation protocols.

Since we do not use the ground truth points for pre-training the SVR model, the constructed voxel number of each scene is very different from the number of ground truth points. Typically, the number of ground truth point clouds is about 50K to 350K, but the number of our voxels are about 5M to 10M. Therefore, we conduct several different protocols for evaluations to better showcase our OpenVoxel: (1) Nearest,(2) Majority of 25-NN, and (3) Majority of 50-NN. The Nearest protocol means that for each point in the ground truth point cloud, we find the spatially nearest voxel and take the voxel’s class ID (obtained as described in B.1) as our prediction; Majority of 25-NN and Majority of 50-NN indicates for each point in ground truth, we find 25 or

- Table 6: Quantitative evaluation on ScanNet semantic segmentation of 19 classes.

Method Uses GT Point mIoU mAcc LEGaussian [Shi et al., 2024] ✓ 3.8 10.9 LangSplat [Qin et al., 2024] ✓ 3.8 9.1 OpenGaussian [Wu et al., 2024] ✓ 24.7 41.5 Ours (Nearest) - 30.0 41.1 Ours (Majority of 25-NN) - 31.3 42.1 Ours (Majority of 50-NN) - 31.6 42.3

- Table 7: Ablation studies on different segmentation models for RES task on Ref-LeRF [He et al., 2025] subset.

Method Segmentation Model mIoU

- A SAM 30.5

- B SAM2 42.4

50 spatially nearest voxels and treat the majority of their labels as our predictions. After defining our prediction for each point, the rest are totally the same as the evaluation pipeline as proposed in OpenGaussian.

###### Results.

The results are shown in Table 6. We can see that even without using the ground truth points as prior, our OpenVoxel still outperforms all baselines in terms of mIoU, and is comparable in mAcc. This shows the potential of OpenVoxel on diverse tasks instead of just OVS and RES for being a training-free approach.

### C. Ablation study.

Being a training-free approach, it is essential to investigate how different prior models affect the performance of our OpenVoxel. Therefore, we conduct ablation studies in three main prior models: the segmentation model for grouping, the captioning model for generating raw captions, the MLLM for canonical captioning, query refinement, and target retrieval. Different segmentation model.

- Table 7 shows our OpenVoxel using different versions of SAM [Kirillov et al., 2023; Ravi et al., 2024] as a segmentation prior model for our grouping stage for RES task. In our experiments, we observe that SAM tends to segment small fragments that are over-detailed, and therefore, the grouping results are slightly noisier than our original version using SAM2. As a result, we can see that there is about 10% performance drop on the RES task. Different captioning model.
- Table 8 shows our OpenVoxel using different captioning model to obtain the original caption for each group on the RES task. For the setting using the Osprey (Yuan, et al, 2024) captioning model, we caption each frame and then ask Qwen3-VL-8B-Instruct model to summarize them into one sentence since Osprey is not suitable for taking video as input. As for the setting using Qwen3-VL-8B-Instruct as captioning model, we direct bypass the DAM captioning stage and ask Qwen3-VL-8B-Instruct to generate caption purely from the visual input (with darkened background and red dot as visual prompt) since it is not trained for taking separated video-mask pair as input. From Table 8 we can see that although Qwen3-VL-8B-Instruct is not trained specially for captioning

- Table 8: Ablation studies on different models for captioning of RES task on Ref-LeRF [He et al., 2025] subset.

Method Captioning Model mIoU

A Osprey (Yuan, et al, 2024) 29.3 B Qwen3-VL-8B-Instruct 33.3 C DAM [Lian et al., 2025] 42.4

- Table 9: Ablation studies on different MLLMs for RES task on Ref-LeRF [He et al., 2025] subset.

##### Method MLLM mIoU

- A Qwen2.5-VL-7B-Instruct 23.4

- B Qwen3-VL-2B-Instruct 10.0

- C Qwen3-VL-4B-Instruct 35.6

- D Qwen3-VL-8B-Instruct 42.4

masked region captioning task, it can still produce reasonable results that are feasible for the RES task. As for Osprey, although it is a captioning specialized model, the per-frame prediction property lead to inconsistent caption for the same group from different views, confusing the Qwen3-VL-8B-Instruct model for summarization and hindering the performance of the RES task. However, we note that using either captioning model achieves better mIoU than ReferSplat [He et al., 2025] (29.2% for the original reported number and 24.5% for our reproduced results) on the RES task, showing the robustness of our designed pipeline.

###### Different MLLM.

- Table 9 shows the results of our OpenVoxel using different MLLMs during the canonical scene map construction and the inference stage with the totally same system prompts and user prompts. Since we do not specially design different prompt for each model, we observe that the Qwen3-VL-2B-Instruct is incapable of canonicalize the captions generated from DAM. Instead, it tends to repeat some of the words in the original caption as the refined caption. As a result, the incorrect refined captions are hard for the Qwen3-VL-2B-Instruct model to locate the correct target object in the inference stage, leading to a catastrophic 9.98% mIoU. In contrast, the newest and largest model for this ablation, Qwen3-VL-8B-Instruct is obviously performing best.

### D. Qualitative results

Referring Segmentation. We provide the qualitative results of the other two scenes (i.e., teatime and kitchen) in Ref-LeRF [He et al., 2025] subset for RES task in Fig. 8 and Fig. 9. Similar to our observation in Sect. 5.3, for the first column (i.e., “A smooth container placed next to the sheep doll, near the apple” as query) in Fig. 8, ReferSplat [He et al., 2025] tends to capture only part of the query (i.e., “container”) and hence segmenting both the coffe mug and the glass of tea, neglecting other spatial clues in the query. Similarly, for the last column in Fig. 9 with “A countertop with a vibrant yellow color provides plenty of space for preparing cooking ingredients.” as query, ReferSplat only capture the color information of “vibrant yellow” and segment both the counter top and the wall. In contrast, our OpenVoxel successfully locates the ideal target in both cases. We additionally showcase the qualitative results of RES on the “Teatime” scene using our created natural language that are not included in the Ref-LeRF subset as query to demonstrate the capability of our OpenVoxel compared with ReferSplat [He et al., 2025]. We can see that since ReferSplat requires training on all objects using human annotated sentences, it is not able to locate objects that are not annotated during their training.

“A round object with a smooth surface directly in front of the white doll.”

“A smooth container placed next to the sheep doll, near the apple.”

“The organ used to breathe on the bear's face and appears smooth and soft.”

“An object placed in the center of a table with writing on its surface and placed vertically.”

“A small, round container placed on the side of the bear, near the cookie and with warm liquid.”

[Figure 165]

ReferSplatOpenVoxel(Ours)GroundTruth

###### Figure 8: Qualitative results of RES task on the Teatime scene.

“A small cup with a green pattern on the surface and a lively patterned shape.”

“A silver baking machine with a sleek exterior sits in the corner of the kitchen counter.”

“A countertop with a vibrant yellow color provides plenty of space for preparing cooking ingredients.”

“A compact red condiment bottle with sauce inside.”

“Two books with covers resembling recipes sit on a kitchen divider.”

[Figure 166]

ReferSplatOpenVoxel(Ours)GroundTruth

###### Figure 9: Qualitative results of RES task on the Kitchen scene.

In contrast, our training-free approach is not depending on the training annotations at all and is able to locate the correct targets. We note that for the last two columns where the input query contains view dependent descriptions, although our OpenVoxel retrieve two targets instead of the only one matched, the results are still including the correct target, showing the potential capability of solving view-specific tasks.

### E. Discussions and Limitations.

We now discuss the potential limitation of our OpenVoxel. As a training-free approach, the grouping process of OpenVoxel is relatively sensitive to parameters comparing to the end-to-end generalizable ones [Chou et al.,

“The chair that the sheep is sitting on, nearest to the sheep and the camera.”

“A blue object on table that looks like an opened wrapper with text on it.”

“The only table that is nearest to the current view.”

“a tag that writes the word ``dalle'' and in front of the sheep.”

“Small broken piece of brown cookie lies on the plate.”

[Figure 167]

ReferSplatOpenVoxel(Ours)

- Figure 10: Qualitative results of RES task on the Teatime scene with other queries. Note that these queries are created additionally and all of them are not appeared in the original annotations from Ref-LeRF subset (so there are no ground truth mask for them). We can see that ReferSplat [He et al., 2025] struggles to recognize unseen target objects even in the same scene it is optimized, showing that it tends to overfit on annotated objects from the dataset.

2024]. As described in Sect. A, the sampling rate of frames and merging frequency are customized for each scene. And since how well-separated for different instances largely affects the performance for both OVS and RES (semantic segmentation is less affected), the SAM2 parameters are needed to be adjust carefully. However, we note that other optimization-based grouping methods [Ye et al., 2024; Zhu et al., 2025] also share similar limitation, as their results heavily rely on the quality of view-consistent video segmentation models such as DEVA [Cheng et al., 2023]. Fortunately, our heuristic of sampling one frame per 3-5 frames and conduct merging once per 3 steps generally work well. In case the result is unsatisfactory, our work is totally training-free so it would be easy to adjust the parameters and re-run the grouping process with a tolerable time (about 1 minute per-scene, taking Ref-LeRF subset as example.) Also, since we conduct the instance-level grouping process before captioning/retrieval, if the user gives a query to indicate some part of a larger object (e.g., flash light of the camera in Fig. 5), our OpenVoxel would still segment the whole object (i.e., the whole camera) since the small parts of the same object is bundled together. One possible solution to solve this issue is to curate the groups as small as possible while still keeping them semantically reasonable (requires 2D segmentation maps that includes those small part). Additionally, instead of just build the Scene Map 𝑆 by storing center locations of each group, construct a complex scene graph for all the groups to indicate the spatial relations (e.g., on top of, between) or ownership (e.g., belongs to, part of) explicitly. We truly believes that this would help improving the performance and robustness of our OpenVoxel and leave it as a possible future direction. Another limitation of our OpenVoxel lies in the usage of MLLM models. As shown in the Sect. A, we turn off the chain-of-thought process of the MLLM model for fast inference (less than one second per query). However, by doing so the capability of reasoning for the MLLM is also limited, and hence both the canonicalized captions and the retrieving process are having room to be improved. Also, as shown in Table 9, the results of using different open-source MLLM differs. We believe that if better open-source MLLM appear with faster thinking/reasoning ability, our OpenVoxel can be benefited from them.

### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 9, 13
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 9, 13
- [3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2021. 1, 3
- [4] Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, et al. Segment anything in 3d with nerfs. Advances in Neural Information Processing Systems (NeurIPS), 2023. 10
- [5] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. 1, 3
- [6] Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890,

2024. 1, 3

- [7] Honghua Chen, Chen Change Loy, and Xingang Pan. Mvip-nerf: Multi-view 3d inpainting on nerf scenes via diffusion prior. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [8] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [9] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2023. 10, 22
- [10] Zi-Ting Chou, Sheng-Yu Huang, I Liu, Yu-Chiang Frank Wang, et al. Gsnerf: Generalizable semantic neural radiance fields with enhanced 3d scene understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3, 21
- [11] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 18
- [12] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 3
- [13] Jian Gao, Chun Gu, Youtian Lin, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. Relightable 3d gaussian: Real-time point cloud relighting with brdf decomposition and ray tracing. Proceedings of the European Conference on Computer Vision (ECCV), 2024. 1, 3
- [14] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instructnerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2023. 3

- [15] Qingdong He, Jinlong Peng, Zhengkai Jiang, Xiaobin Hu, and Jiangning Zhang. Pointseg: A training-free paradigm for 3d scene segmentation via foundation models. In Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCV Workshops), 2025. 1, 3
- [16] Shuting He, Guangquan Jie, Changshuo Wang, Yun Zhou, Shuming Hu, Guanbin Li, and Henghui Ding. Refersplat: Referring segmentation in 3d gaussian splatting. Proceedings of the International Conference on Machine Learning (ICML), 2025. 1, 2, 3, 4, 8, 9, 10, 12, 19, 20, 22
- [17] Dongting Hu, Huan Fu, Jiaxian Guo, Liuhua Peng, Tingjin Chu, Feng Liu, Tongliang Liu, and Mingming Gong. In-n-out: Lifting 2d diffusion prior for 3d object removal via tuning-free latents alignment. Advances in Neural Information Processing Systems, 37:45737–45766, 2024. 3
- [18] Sheng-Yu Huang, Zi-Ting Chou, and Yu-Chiang Frank Wang. 3d gaussian inpainting with depth-guided cross-view consistency. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 3
- [19] Tianyu Huang, Runnan Chen, Dongting Hu, Fengming Huang, Mingming Gong, and Tongliang Liu. Openinsgaussian: Open-vocabulary instance gaussian segmentation with context-aware cross-view fusion. In Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCV Workshops), 2025. 1, 3
- [20] SungMin Jang and Wonjun Kim. Identity-aware language gaussian splatting for open-vocabulary 3d semantic segmentation. In Proceedings of the IEEE International Conference on Computer Vision (ICCV),

2025. 1, 3

- [21] Yuzhou Ji, He Zhu, Junshu Tang, Wuyi Liu, Zhizhong Zhang, Xin Tan, and Yuan Xie. Fastlgs: Speeding up language embedded gaussians with feature grid mapping. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2025. 1, 3
- [22] Mohammad Mahdi Johari, Yann Lepoittevin, and François Fleuret. Geonerf: Generalizing nerf with geometry priors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2022. 1, 3

- [23] Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. splat: Directly referring 3d gaussian splatting via direct language embedding registration. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 3, 6
- [24] Amita Kamath, Cheng-Yu Hsieh, Kai-Wei Chang, and Ranjay Krishna. The hard positive truth about vision-language compositionality. In Proceedings of the European Conference on Computer Vision (ECCV). Springer, 2024. 2
- [25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 2023. 1, 3, 4
- [26] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE International Conference on Computer Vision (ICCV),

2023. 1, 3, 9, 10, 13

- [27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2023. 3, 19
- [28] Abhijit Kundu, Kyle Genova, Xiaoqi Yin, Alireza Fathi, Caroline Pantofaru, Leonidas J Guibas, Andrea Tagliasacchi, Frank Dellaert, and Thomas Funkhouser. Panoptic neural fields: A semantic object-aware neural scene representation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 3

- [29] Junha Lee, Chunghyun Park, Jaesung Choe, Yu-Chiang Frank Wang, Jan Kautz, Minsu Cho, and Chris Choy. Mosaic3d: Foundation dataset and model for open-vocabulary 3d segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 3
- [30] Jiahe Li, Jiawei Zhang, Youmin Zhang, Xiao Bai, Jin Zheng, Xiaohan Yu, and Lin Gu. Geosvr: Taming sparse voxels for geometrically accurate surface reconstruction. arXiv preprint arXiv:2509.18090, 2025. 1
- [31] Long Lian, Yifan Ding, Yunhao Ge, Sifei Liu, Hanzi Mao, Boyi Li, Marco Pavone, Ming-Yu Liu, Trevor Darrell, Adam Yala, et al. Describe anything: Detailed localized image and video captioning. arXiv preprint arXiv:2504.16072, 2025. 7, 9, 13, 20
- [32] Liwei Liao, Xufeng Li, Xiaoyun Zheng, Boning Liu, Feng Gao, and Ronggang Wang. Zero-shot visual grounding in 3d gaussians via view retrieval. arXiv preprint arXiv:2509.15871, 2025. 1, 3
- [33] Chieh Hubert Lin, Changil Kim, Jia-Bin Huang, Qinbo Li, Chih-Yao Ma, Johannes Kopf, Ming-Hsuan Yang, and Hung-Yu Tseng. Taming latent diffusion model for neural radiance field inpainting. Proceedings of the European Conference on Computer Vision (ECCV), 2024. 3
- [34] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, and Shijian Lu. Weakly supervised 3d open-vocabulary segmentation. Advances in Neural Information Processing Systems (NeurIPS), 2023. 1, 3, 4
- [35] Yuan Liu, Sida Peng, Lingjie Liu, Qianqian Wang, Peng Wang, Christian Theobalt, Xiaowei Zhou, and Wenping Wang. Neural rays for occlusion-aware image-based rendering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 3
- [36] Zhiheng Liu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jie Xiao, Kai Zhu, Nan Xue, Yu Liu, Yujun Shen, and Yang Cao. Infusion: Inpainting 3d gaussians via learning depth completion from diffusion prior. arXiv preprint arXiv:2404.11613, 2024. 3
- [37] Weijie Lyu, Xueting Li, Abhijit Kundu, Yi-Hsuan Tsai, and Ming-Hsuan Yang. Gaga: Group any gaussians via 3d-aware memory bank. arXiv preprint arXiv:2404.07977, 2024. 10
- [38] Juliette Marrie, Romain Ménégaux, Michael Arbel, Diane Larlus, and Julien Mairal. Ludvig: Learning-free uplifting of 2d visual features to gaussian splatting scenes. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2025. 1
- [39] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1, 3
- [40] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 1, 3, 4
- [41] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G Derpanis, and Igor Gilitschenski. Reference-guided controllable inpainting of neural radiance fields. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2023. 3
- [42] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G Derpanis, Jonathan Kelly, Marcus A Brubaker, Igor Gilitschenski, and Alex Levinshtein. Spin-nerf: Multiview segmentation and perceptual inpainting with neural radiance fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3, 8

- [43] Ashkan Mirzaei, Riccardo De Lutio, Seung Wook Kim, David Acuna, Jonathan Kelly, Sanja Fidler, Igor Gilitschenski, and Zan Gojcic. Reffusion: Reference adapted diffusion models for 3d scene inpainting. arXiv preprint arXiv:2404.10765, 2024. 3
- [44] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1–15, 2022. 1, 3
- [45] Davy Neven, Bert De Brabandere, Marc Proesmans, and Luc Van Gool. Instance segmentation by jointly optimizing spatial embeddings and clustering bandwidth. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 5
- [46] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems (NeurIPS), 2019. 9
- [47] Qucheng Peng, Benjamin Planche, Zhongpai Gao, Meng Zheng, Anwesa Choudhuri, Terrence Chen, Chen Chen, and Ziyan Wu. 3d vision-language gaussian splatting. Proceedings of the International Conference on Learning Representations (ICLR), 2025. 1, 3, 9
- [48] Jens Piekenbrinck, Christian Schmidt, Alexander Hermans, Narunas Vaskevicius, Timm Linder, and Bastian Leibe. Opensplat3d: Open-vocabulary 3d instance segmentation using gaussian splatting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPR Workshops), 2025. 1, 3
- [49] Charles R. Qi, Or Litany, Kaiming He, and Leonidas J. Guibas. Deep hough voting for 3d object detection in point clouds. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019. 5
- [50] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3, 8, 9, 10, 19
- [51] Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Liujuan Cao, Shengchuan Zhang, and Rongrong Ji. Goi: Find 3d gaussians of interest with an optimizable open-vocabulary semantic-space hyperplane. In Proceedings of the 32nd ACM international conference on multimedia, pages 5328–5337, 2024. 8, 9
- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR,

2021. 1

- [53] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 2, 5, 9, 13, 19
- [54] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2021. 1, 3
- [55] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 8
- [56] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for open-vocabulary scene understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 9, 19

- [57] Mohammed Suhail, Carlos Esteves, Leonid Sigal, and Ameesh Makadia. Generalizable patch-based neural rendering. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. 1, 3
- [58] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 3
- [59] Cheng Sun, Jaesung Choe, Charles Loop, Wei-Chiu Ma, and Yu-Chiang Frank Wang. Sparse voxels rasterization: Real-time high-fidelity radiance field rendering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 3, 6, 9, 18
- [60] Lei Tian, Xiaomin Li, Liqian Ma, Hao Yin, Zirui Zheng, Hefei Huang, Taiqing Li, Huchuan Lu, and Xu Jia. Ccl-lgs: Contrastive codebook learning for 3d language gaussian splatting. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2025. 1, 3, 9
- [61] Suhani Vora, Noha Radwan, Klaus Greff, Henning Meyer, Kyle Genova, Mehdi SM Sajjadi, Etienne Pot, Andrea Tagliasacchi, and Daniel Duckworth. Nesf: Neural semantic fields for generalizable semantic segmentation of 3d scenes. arXiv preprint arXiv:2111.13260, 2021. 1, 3
- [62] Dongqing Wang, Tong Zhang, Alaa Abboud, and Sabine Süsstrunk. Innerf360: Text-guided 3d-consistent object inpainting on 360-degree neural radiance fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [63] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 9, 13
- [64] Yuxin Wang, Qianyi Wu, Guofeng Zhang, and Dan Xu. Gscream: Learning 3d geometry and feature consistent gaussian splatting for object removal. Proceedings of the European Conference on Computer Vision (ECCV), 2024. 3
- [65] Ethan Weber, Aleksander Holynski, Varun Jampani, Saurabh Saxena, Noah Snavely, Abhishek Kar, and Angjoo Kanazawa. Nerfiller: Completing scenes via generative 3d inpainting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 20731–20741, 2024. 3
- [66] Silvan Weder, Guillermo Garcia-Hernando, Aron Monszpart, Marc Pollefeys, Gabriel J Brostow, Michael Firman, and Sara Vicente. Removing objects from neural radiance fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [67] Chung-Ho Wu, Yang-Jung Chen, Ying-Huan Chen, Jie-Ying Lee, Bo-Hsu Ke, Chun-Wei Tuan Mu, Yi-Chuan Huang, Chin-Yang Lin, Min-Hung Chen, Yen-Yu Lin, et al. Aurafusion360: Augmented unseen region alignment for reference-based 360deg unbounded scene inpainting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 3
- [68] Junda Wu, Zhehao Zhang, Yu Xia, Xintong Li, Zhaoyang Xia, Aaron Chang, Tong Yu, Sungchul Kim, Ryan A Rossi, Ruiyi Zhang, et al. Visual prompting in multimodal large language models: A survey. arXiv preprint arXiv:2409.15310, 2024. 7
- [69] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, et al. Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. Advances in Neural Information Processing Systems (NeurIPS), 2024. 1, 3, 18, 19

- [70] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 3
- [71] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. Proceedings of the European Conference on Computer Vision (ECCV), 2024. 1, 3, 5, 8, 9, 10, 22
- [72] Youtan Yin, Zhoujie Fu, Fan Yang, and Guosheng Lin. Or-nerf: Object removing from 3d scenes guided by multiview segmentation with neural radiance fields. arXiv preprint arXiv:2305.10503, 2023. 3
- [73] Haiyang Ying, Yixuan Yin, Jinzhi Zhang, Fan Wang, Tao Yu, Ruqi Huang, and Lu Fang. Omniseg3d: Omniversal 3d segmentation via hierarchical contrastive learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3
- [74] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2021. 1, 3
- [75] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3
- [76] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 9
- [77] Ruijie Zhu, Mulin Yu, Linning Xu, Lihan Jiang, Yixuan Li, Tianzhu Zhang, Jiangmiao Pang, and Bo Dai. Objectgs: Object-aware scene reconstruction and scene understanding via gaussian splatting. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2025. 1, 3, 9, 10, 12, 22

