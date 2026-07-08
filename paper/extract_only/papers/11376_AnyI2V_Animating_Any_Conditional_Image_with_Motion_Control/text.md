# AnyI2V: Animating Any Conditional Image with Motion Control

Ziye Li1 Hao Luo2,3 Xincheng Shuai1 Henghui Ding1

1Fudan University 2DAMO Academy, Alibaba group 3Hupan Lab https://henghuiding.com/AnyI2V/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

AnyModalityControl

arXiv:2507.02857v1[cs.CV]3Jul2025

Depth Skeleton

“An eagle flies through the city” “Iron Man on the beach, raising hands”

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- (a)

MixedModality

- (b)

VisualEditing

- (c)

Mesh Point Cloud

“A rose in the grassland” “A golden horse in the museum”

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Sketch+Seg “A fish swims through the coral” Depth+Hed “A helicopter flies through the glacier”

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Real LoRA+ “A lion closes its mouth” Real “A dog with sunglasses on the beach”

- Figure 1. The first frame conditional control of our Training-Free architecture AnyI2V. (a) AnyI2V supports diverse types of conditional inputs, including those that are difficult to obtain construct pairs for training, such as mesh and point cloud data. The trajectories serve as input for motion control in subsequent frames. (b) AnyI2V can accept inputs with mixed conditional types, further increasing the flexibility of the input. (c) By using LoRA [13] or different text prompts, AnyI2V can achieve the editing effect of the original image.

## Abstract

In contrast, I2V methods are limited by their dependence on real images, which restricts the editability of the synthesized content. Although some methods incorporate ControlNet to introduce image-based conditioning, they often lack explicit motion control and require computationally expensive training. To address these limitations, we propose AnyI2V, a training-free framework that animates any conditional images with user-defined motion trajectories. AnyI2V supports a broader range of modalities as the conditional image, including data types such as meshes and point clouds that are not supported by ControlNet, enabling more flexible and versatile video generation. Additionally, it supports

Recent advancements in video generation, particularly in diffusion models, have driven notable progress in text-tovideo (T2V) and image-to-video (I2V) synthesis. However, challenges remain in effectively integrating dynamic motion signals and flexible spatial constraints. Existing T2V methods typically rely on text prompts, which inherently lack precise control over the spatial layout of generated content.

Henghui Ding (henghui.ding@gmail.com) is the corresponding author with the Institute of Big Data, College of Computer Science and Artificial Intelligence, Fudan University, Shanghai, China.

mixed conditional inputs and enables style transfer and editing via LoRA and text prompts. Extensive experiments demonstrate that the proposed AnyI2V achieves superior performance and provides a new perspective in spatial- and motion-controlled video generation.

## 1. Introduction

Diffusion-based video generation has achieved significant advancements recently, with existing methods broadly categorized into text-to-video (T2V) [7, 18, 37] and image-tovideo (I2V) [1, 4, 44, 51]. Although considerable efforts have been made toward incorporating additional motion signals, e.g., object or camera motion [46], the integration of motion information with general structural information remains insufficiently explored.

Some T2V-based motion control methods enable objectspecific motion control using bounding boxes or point trajectories [23, 46, 48, 52]. While these methods can generate videos following given trajectories, they lack explicit spatial layout control of the generated content, limiting their capability to perform fine-grained adjustments, such as precisely articulating human limbs. In contrast, I2V-based methods [33, 42, 53] address this limitation by conditioning on the first-frame image, thus explicitly specifying the initial content and achieving finer-grained control. However, the dependence of these methods on real RGB images as first-frame inputs restricts their content editability. To alleviate the challenge, several video diffusion methods [15, 55] incorporate ControlNet [54], injecting frame-wise or sparse conditions [8] to achieve fine-grained layout control. Nonetheless, these approaches do not support user-defined motion control. Moreover, both ControlNet and motion control modules require computationally expensive training, and retraining is necessary whenever the base model changes, further limiting their flexibility.

To address these limitations, we propose AnyI2V, a training-free method that accepts any modality image, e.g., mesh, point cloud, edge, depth, skeleton, etc., to serve as a reference frame for initial content layout control while enabling user-defined trajectories for motion control in subsequent frames, as shown in Fig. 1. AnyI2V is a unified framework that processes the input within one model without additional modules. Beyond single-modality guidance, AnyI2V also supports mixed-modality inputs, enhancing flexibility and control. For example, depth maps effectively represent background structures, while sketches precisely define foreground details, leveraging the complementary strengths of different modalities. Additionally, by employing LoRA [13] or different text prompts, AnyI2V enables editing the visual content of the input image.

Specifically, the proposed AnyI2V consists of three key methods: structure-preserved feature injection, acrossframes alignment, and semantic mask generation. First,

to achieve first-frame guidance in a training-free manner, we identify the essential features that should be injected as guidance. Then, to mitigate bias in specific features, we propose a simple yet effective operation that suppresses unwanted appearance information while preserving the structural integrity of the input image. Based on these essential features, we further analyze their temporal characteristics by visualizing the principal components through PCA dimensionality reduction [47]. Our analysis reveals distinct attributes across these features, highlighting that the query in spatial self-attention maintains strong temporal consistency and an entity-aware semantic representation across frames. Motivated by this key insight, we propose aligning the query from spatial self-attention across different frames to ensure temporally coherent video synthesis. Furthermore, to precisely control irregularly shaped objects during this alignment, it is common practice to apply a mask that excludes unrelated content from optimization. However, since the target object may undergo deformation across frames, the use of such a static mask inevitably restricts the flexibility of object motion. To overcome this limitation, we leverage semantic information embedded within the features and cluster it into a dynamic mask, which can adaptively accommodate object deformation while maintaining precise spatial control.

In summary, our main contributions are as follows:

- • We introduce AnyI2V, which integrates the first-frame spatial condition with user-defined trajectory to control the content layout and motion, respectively. Moreover, our training-free framework eliminates the training burden and simplifies adaptation across different backbones.
- • AnyI2V offers exceptional flexibility, allowing to accept a wide range of conditional images as input for the first frame. Furthermore, AnyI2V supports mixedcondition inputs and by incorporating LoRA or different text prompts, enables effective visual editing, producing highly diverse and visually appealing results.
- • By rethinking feature injection and employing zero-shot trajectory control with semantic masks, AnyI2V demonstrates superior performance across diverse scenarios, as validated through extensive experiments, highlighting the effectiveness of our approach.

## 2. Related Work

Diffusion-Based Video Generation. Generation quality in the field of video generation has achieved great progress in recent years, especially in the domain of diffusion models [1, 4, 7, 12, 35, 37, 44, 51]. The earliest video diffusion models focus on text-to-video (T2V) generation, and many of these methods adopt a pre-trained text-to-image (T2I) diffusion model [32] to add motion to the images [7, 18, 37] where the majority of methods train extra zero-initialized temporal modules while keeping the rest of the modules to

|[Figure 33]|
|---|

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

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Input First Frame* AnyI2V (ours) DragAnything DragNUWA MOFA

- Figure 2. Comparison between AnyI2V and previous methods, DragAnything [49], DragNUWA [53], and MOFA [28]. ‘First Frame*’ indicates that the condition images for previous methods are generated using AnyI2V to ensure a more consistent and fair comparison.

be frozen. The later works [1, 4, 44, 51] explore to inject an image as the first frame to provide the spatial layout of the early stage of the generated videos which works as image-to-video (I2V) generation. These methods are mainly built upon the pre-trained T2V model which finetune the base model to achieve the ability to accept an image

- as the condition. Although they achieve high-fidelity video generation, the motion in the generated videos is sometimes limited in range or fails to match user expectations.

Spatial Condition in Diffusion Models. To control the spatial layout of generation results, several methods [11, 26, 40, 54] have been proposed. Among these, ControlNet [54] stands out as a groundbreaking approach that accepts input images from a variety of modalities to generate precisely controlled outputs. Building upon this foundation, following methods [8, 9, 15, 22, 26, 46, 55] have adapted ControlNet for video generation tasks. However, these methods typically require training the ControlNet with highly aligned datasets. Obtaining aligned inputs and corresponding real images is challenging for some modalities, e.g., meshes or point clouds. Additionally, each modalityspecific input demands an independent ControlNet, and the originally pre-trained weights may not be compatible with a new base model. To address these limitations, the recently proposed FreeControl [26] introduces a trainingfree approach to solve these issues within the text-to-image (T2I) domain. Furthermore, several works [14, 19, 20, 30, 39] have explored how to process mixed-modality inputs in the domain of T2I generation, thereby enhancing the diversity and richness of generated content.

Motion Control in Diffusion Models. Controlling motion in generated videos has been an active area of research [3, 9, 29, 31, 33, 34, 36, 42, 43, 46, 52, 53, 56, 57]. Methods

such as DragGAN [29] and DragDiffusion [34] introduced pioneering approaches to control the destination of a target point in the original image by optimizing the input latent. These methods are implemented in T2I base model and lack smooth temporal transitions. Other methods, such as MotionCtrl [46], extend motion control to T2V generation, allowing for independent control of camera motion and object motion. Additionally, drag-based methods in I2V generation, such as DragNUWA [53] and DragAnything [49], use an initial image to control the expected motion of objects. Despite their effectiveness, these drag-based T2V and I2V methods often require extensive training data, which can be difficult to obtain, and are implemented in a black-box manner, lacking interpretability. To address these limitations, some training-free methods [27, 50] have emerged. However, these approaches only accept input from real image modality and lack an editable space to control the appearance of the video using text prompts. In contrast, this paper proposes a novel method that accepts input from various image modalities and is implemented in a training-free manner, enabling users to edit videos flexibly without the need for fine-tuning the base model.

## 3. Method

### 3.1. Preliminary

Architecture. The proposed AnyI2V is adapted from a video diffusion model based on a 3D U-Net [7], with spatial and temporal components for frame content and coherence. To leverage pretrained image diffusion models, most T2V methods extend 2D models into 3D by adding temporal modules [2, 4, 7, 10, 41], and are trained on both images and videos to maintain single-frame generation ability.

Condition Modification

[Figure 73]

Align PCA features with Semantic Mask

[Figure 74]

[Figure 75]

|[Figure 76]|
|---|

|[Figure 77]<br><br>|[Figure 78]<br><br>[Figure 79]|
|---|
|
|---|

|[Figure 80]<br><br>|[Figure 81]<br><br>[Figure 82]|
|---|
|
|---|

|[Figure 83]<br><br>|[Figure 84]|
|---|
|
|---|

TemporalModuleRemoval

- U-Net 2D

[Figure 85]

- U-Net 3D

[Figure 86]

[Figure 87]

Decode

Encode

… …

Optimize

Iterative

[Figure 88]

Extract Extract

Iterative Inverse

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Optimized

|=𝑡𝑡𝛼|
|---|

=𝑡𝑡𝛼

U-Net U-Net

… …

Aligned

U-Net U-Net

Initialized

Random

|𝑧0|
|---|

𝑧𝑡’

(𝑎) (𝑏)

SpatialBlock

𝑡′ ≤ 𝑡𝛾: Optimize Latent 𝑡′ > 𝑡𝛾: Sampling Only

(𝑎)

Q K V

Q K V

Debias Residual Hidden

ResHid

Key & Value Consistency

First-Frame Injection

(𝑏)

ResBlock Self-Attn Cross-Attn

- Figure 3. Overview of Our Pipeline: Our pipeline begins by performing DDIM inversion on the conditional image. To do this, we remove the temporal module (i.e., temporal self-attention) from the 3D UNet and then extract features from its spatial blocks at timestep tα. Next, we optimize the latent representation by substituting the features from the first frame back into the U-Net. This optimization is constrained to a specific region by an auto-generated semantic mask (detailed in Fig. 6) and is only performed for timesteps t′ ≤ tγ.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Input Res Hidden Query Key

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Value Attention Map Attention Result Debiased Res Hidden

- Figure 4. The study examines the influence of injecting different features. Each feature is injected independently. The input text prompt is “A sculpture of a horse in the musuem.”

denoising stages, while finer details are established in later stages. This observation suggests there is a specific step at which the diffusion model’s features strike an optimal balance between structural information and texture. Accordingly, we extract the feature from a particular DDIM inversion [38] step tα for injection.

Next, we examine how various features at time step tα contribute to the generation result. Instead of using the noise from the end of the DDIM inversion, which retains extensive appearance information, we start with pure random noise and inject only one type of feature at a time. Fig. 4 intuitively demonstrates the primary influence of features from the ResBlock and the spatial selfattention layer. Notably, each of the residual hidden, query, and self-attention map independently yields satisfactory structure control, with the residual hidden being especially effective. However, the residual hidden also encapsulates most of the appearance information from the source image leading to unsatisfactory visual fidelity. Hence, to ensure the generated image adheres more closely to the textual guidance, we debias the appearance in the residual hidden, resulting in a more contextually accurate outcome.

Latent Optimization. Latent optimization involves iteratively updating the latent variable z by computing gradients of the guidance loss function Lguide with respect to z. Through this iterative process, the latent variable z is progressively refined to better align with the conditioning signals, ensuring that generated samples achieve higher fidelity and accurately reflect the desired conditions.

Adaptive Instance Normalization (AdaIN) [16] is a widely used technique for maintaining source structure and transferring target style. However, original AdaIN concentrates on operating global features, which leads to inferior result in local quality. To address this issue, we propose to patchify the injected residual hidden feature hi and the source residual hidden hs from the backbone into non-overlapping patches h′i and h′s. The patchified features are then manipulated using AdaIN to obtain the target hidden representation ht. Formally, the operation is

### 3.2. Rethinking Feature Injection

Given an image, it has been demonstrated in PnP [40] that diffusion models present the capability to capture structural information. However, PnP encounters difficulties when processing images from different modalities. Building on this, we conduct experiments to evaluate structural and appearance control by replacing the feature.

As indicated in previous studies [34, 40], diffusion models typically determine the overall layout in the early

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

[Figure 114]

Output

Video

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

[Figure 126]

Attention Map

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

Residual

Hidden

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

Attention Query

- Figure 5. This figure compares different PCA-reduced features in terms of their temporal consistency and entity representation. The results reveal several key observations: (1) Features of attention map exhibit low temporal consistency. (2) Features of residual hidden state fail to treat the object as a coherent entity. (3) The attention query provides both high temporal consistency and strong entity representation.

### 3.3. Zero-Shot Trajectory Control

expressed as:

The previous section discussed features without considering the temporal axis. In this section, we select the features with well-structured control ability as discussed in the previous section and use PCA dimensionality reduction to further analyze their characteristics in the temporal dimension. These features are visualized using the first three principal components of the PCA-transformed features. As shown in Fig. 5, we compare the dimensionally reduced features of the self-attention map, residual hidden state, and attention query, focusing on the moving object to assess its temporal consistency and entity representation.

h′ = Patchify(h,p), ht = AdaIN(h′i,h′s), (1)

where h ∈ RB×C×H×W is the input feature, h′ ∈ RB×

H×W

p2 ×C×p×p is the patchified feature and p denotes the patch size. The AdaIN is expressed as:

h′i − µ(h′i) σ(h′

AdaIN(h′i,h′s) = σ(h′s)

+ µ(h′s), (2)

i)

where σ and µ denote functions that compute the standard deviation and mean, respectively, across the spatial dimensions. After the AdaIN operation, the target feature ht is reshaped back to its original dimensions to match the input feature map h. As shown in Fig. 4, the debiased residual hidden exhibits well-preserved structure and natural appearance. In this way, the feature injection operation can effectively process images from different modalities while preventing appearance leakage, preserving both structure and appearance fidelity.

Our findings reveal that moving object in attention map exhibits lower temporal consistency, whereas residual hidden state and attention query demonstrate a strong correlation along the temporal axis of the moving object. Additionally, residual hidden features capture finer-grained details, which do not treat the object as a coherent entity, whereas query features encode the higher-level semantics, treating the object as a whole. This observation leads to a key insight: Aligning temporally-consistent and entity-aware features across frames enables coherent object motion. Based on this, we introduce zero-shot trajectory control by aligning the subsequent frames with the injected first frame.

To extend feature injection for controlling the first frame in the video diffusion model, we first perform DDIM inversion [38] on a single frame conditional image to extract its features. Based on our observation in Fig. 4, we then replace these features by injecting both the debiased residual hidden states and the query. The rationale for using the query instead of self-attention map is discussed in Sec. 3.3. Next, to ensure content consistency across frames in spatial self-attention, we enforce temporal coherence by setting the keys and values of subsequent frames to match those of the first frame, i.e., K2:f = K1 and V2:f = V1. These strategies not only reduce the computational cost of acquiring target features but also maintains structural control and a natural appearance for the first frame.

Cross-Frame Alignment. Inspired by previous works [29, 34], latents can be optimized to enable dragging effects on a single image. We apply this technique in trajectory control for cross-frame alignment. Specifically, based on our analysis, we choose query in spatial self-attention as the aligning target. Previous works conduct latent optimization in a point-dragging manner, restricting the optimization of latent to a small region. However, for more flexible object control, such as moving a specific part of an object or shifting the entire object, we introduce bounding box B, whose size and position in each frame can be defined

[Figure 151]

by the user. Furthermore, since we observed that lowerranked components exhibit reduced temporal consistency and struggle to define a clear layout, as shown in the supplementary material, we further propose aligning the higher-ranked principal components of the query features extracted via PCA. Finally, the following optimization objective is used to optimize the latents:

[Figure 152]

Inverse

[Figure 153]

[Figure 154]

Latent

Generated Similarity Map

Optimize

| | |
|---|---|
| |[Figure 155]<br><br>[Figure 156]<br><br>…<br><br>Align|

f

n

||[Figure 157]|
|---|
<br><br>|[Figure 158]|
|---|
<br><br>|[Figure 159]|
|---|
<br><br>|[Figure 160]|
|---|
| |
|---|---|
| | |

Align

zt∗ = arg min

L Fj[Bji], SG F1[B1i] , (3)

Aggregating & Clustering

|[Figure 161]|
|---|

zt

i=1

j=2

where L denotes the loss function, defined in Eq. (8), i indexes the bounding box groups, j represents the frame index and the operator SG denotes the stop-gradient operation. Fj[Bji] represents the extracted feature and cropped by the bounding box Bji ∈ RH defined as:

Figure 6. The process of aligning the object across frames by optimizing the latent noise while incorporating semantic masks.

SIMi,j of group i on the frame j as follows: SIMi,j(h,w) = max

- i
- j×Wji. The extracted feature is

SIMi,jk (h,w). (6)

k=1,...,K

Fj = PCA(Queryj,M), (4) where M is the number of principal components of channel dimension. Notably, the feature F1 corresponds to the injected first-frame feature and is independent of zt.

To obtain a binary mask, we apply K-Means clustering [24] to the aggregated similarity map:

- i
- j×Wji, (7)

Mji = KMeans(SIMi,j,2) ∈ RH

where 2 denotes binary clustering. The foreground is determined by selecting the cluster with the higher-value centered pixel. Based on the derived mask, we define the loss function as:

### 3.4. Semantic Mask Generation.

The bounding box mentioned above offers flexibility in defining the target region for dragging, yet it does not always enable precise object manipulation. Many objects have irregular shapes, causing unintended regions to be optimized and compromising overall accuracy. Meanwhile, a static mask can further limit the optimized area but also constrains natural deformations, reducing flexibility during dynamic transformations. To overcome these issues, we introduce an adaptive semantic mask generation method that automatically produces a mask based on the semantic information encoded in the features. By doing so, it provides more accurate, context-sensitive, and adaptive control over the target object, preserving structural integrity while allowing for natural movement.

Lij = ||M1i ⊙ Mji ⊙ (Fj[Bji] − SG(F1[B1i]))||22, (8)

where ⊙ denotes pixel-wise multiplication, and M1i ⊙ Mji indicates the overlapping region of the same instance within the bounding box group Bi between frame 1 and frame j.

## 4. Experiments

Implementation Details. We implement our method based on AnimateDiff [7] using a single Nvidia A800 GPU. Our overall pipeline can be seen in Fig. 3. The DDIM inversion consists of 1000 steps, with features extracted at tα = 201. The decoder has three cascaded spatial blocks, indexed as 0, 1 and 2. We inject residual hidden and query indexed as 0 and 1 from up blocks.1 and up blocks.2, optimizing the latent noise by aligning query 1 from up blocks.1 and query 0 from up blocks.2. The patch window is set to be p = 4 for debiasing the residual hidden. PCA reduction dimension is set to M = 64. We use 25 DDIM sampling steps, optimizing latents every 5 steps for t′ ≥ 20 with a 0.01 learning rate. Inversion stage and generation stage take approximately 8s and 35s in half-precision mode, respectively.

Given the injected first-frame feature F1, we aim to generate more precise masks in the all features F1:f corresponding to the semantic content in the F1. First, the salient points Pik are selected inside the bounding boxe of feature F1 to indicate interesting parts, where i denotes the corresponding group index of bounding box and k represents the point index. For all frames, we compute the similarity between Pik and the features within bounding boxes Bji using cosine similarity, formulated as follows:

Fj[Bji] · F1[Pik] ∥Fj[Bji]∥∥F1[Pik]∥

- i
- j×Wji, (5)

∈ RH

SIMi,jk =

### 4.1. Qualitative Evaluations

where F1[Pik] ∈ RM represents the vector at the coordinate Pik in F1. Then we can derive the aggregated similarity map

Fig. 2 presents a comparison between AnyI2V and previous state-of-the-art methods [21, 23, 28, 31, 45, 49, 53]. From

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

In-the-wildModality

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

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

MixedModality

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

- Figure 7. This picture demonstrates AnyI2V’s ability to control diverse conditions. AnyI2V can not only handle modalities that ControlNet does not support but also effectively control mixed modalities, which previously required additional training by other methods.

Table 1. Comparison with previous state-of-the-art methods.

Table 2. Ablation study of the proposed AnyI2V

|Methods|Training|FID ↓ FVD ↓ ObjMC ↓<br><br>|
|---|---|---|
|ImageConductor [21] DragAnything [49] DragNUWA [53] MOFA-Video [28]|✓ ✓ ✓ ✓<br><br>|132.23 646.50 21.14 95.83 556.09 13.60 105.02 575.78 15.02 95.63 599.48 17.72<br><br>|
|Baseline [7] FreeTraj [31] TrailBlazer [23] ObjCtrl-2.5D [45] AnyI2V (ours)|– Free Free Free Free<br><br>|141.95 970.26 38.26 128.78 672.87 24.00 112.37 620.80 23.71 111.82 605.96 23.12 104.53 569.89 16.39<br><br>|

the result images and the visualized trajectories, AnyI2V demonstrates comparable results. Fig. 7 further showcases the ability of AnyI2V to handle in-the-wild and mixedmodality images. Unlike compared methods that are limited to processing only realistic RGB images, our approach significantly enhances editability and flexibility.

### 4.2. Quantitative Evaluations

Following previous work [49], we collect data from the web and the VIPSeg dataset [25], annotating video trajectories using Co-Tracker [17] to ensure high-quality motion tracking [6]. The evaluation metrics include Fr´echet Inception Distance (FID), Fr´echet Video Distance (FVD), and ObjMC, where ObjMC quantifies the error between the ground truth trajectory and the generated result, providing a finegrained assessment of motion accuracy.

For a fair comparison, we randomly convert the first input frame into various structural representations, including canny, HED, depth, normal, and segmentation maps. AnyI2V directly utilizes these representations, while other methods first process the input frames with ControlNet. As shown in Tab. 1, AnyI2V significantly outperforms the

|Options<br><br>|FID ↓ FVD ↓ ObjMC ↓<br><br>|
|---|---|
|w/o K&V consistency w/o PCA Reduction w/ Static Mask w/o Semantic Mask opt. Residual Hidden|108.18 587.69 16.81 105.95 585.04 17.14 105.44 598.15 16.92 105.78 579.88 17.62 129.40 647.52 36.23<br><br>|
|Full (AnyI2V)|104.53 569.89 16.39<br><br>|

baseline model and achieves competitive results against state-of-the-art methods. Here, ‘baseline’ refers to the backbone model with SparseCtrl [8], which accept the reference image and conduct the experiment in a zero-shot manner (indicated by the ‘-’ training attribute).

### 4.3. Ablation Study

Tab. 2, Fig. 8, Fig. 9, and Fig. 10 show the ablation study of different design choices. Tab. 2 evaluates the impact of removing our proposed components and optimizing the residual hidden states instead of the query in self-attention. The results demonstrate that our proposed configurations enhance the temporal consistency of the generated videos, as reflected in the FVD metric, and improve the precision of target object control according to the ObjMC metric.

The line chart in Fig. 8 illustrates how PCA reduction dimensions affect FID, FVD, and ObjMC. Both overly small and large dimensions degrade performance: small dimensions leaves insufficient information for effective alignment despite strong temporal coherence, while large ones weaken consistency in lower-ranked components, hindering alignment. Based on this, we select M = 64 as the optimal PCA dimension for latent alignment.

Fig. 9 examines the impact of injecting features at different time steps. When the time step is too small, the

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

- Figure 8. The line chart in the figure demonstrates the ablation study on the impact of different PCA reduction dimensions on FID, FVD, and ObjMC, while the histogram presents the ablation study on optimization targets for ObjMC.

[Figure 202]

Input Timestep=1 Timestep=201 Timestep=501 Timestep=801

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

- Figure 9. The ablation study examines feature injection across time steps. Visual results are from the first output frames.

[Figure 207]

Input w/o Residual Hidden w/o Debias w/o Patch Full

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

- Figure 10. The ablation study examines operations on the residual hidden state. Visual results are from the first output frames.

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

VideoCrafterLavie

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

Figure 11. The generalization test results of AnyI2V on other backbones, including Lavie [44] and VideoCrafter2 [5].

model tends to overfit low-level textures, leading to unnatural artifacts. When the step is too large, noisy features hinder the model to capture accurate layout information, disrupting the overall structure. An appropriately chosen time step balances low-level feature extraction with layout preservation, resulting in higher-fidelity visuals.

### 4.4. Generalization

Since AnyI2V is a training-free method, we further implement it on different T2V backbones, including Lavie [44] and VideoCrafter2 [5]. The results shown in Fig. 11 highlight its adaptability to various architectures, demonstrating its robustness and strong generalization ability.

Fig. 10 investigates the impact of residual hidden operations. The results indicate that removing the residual hidden leads to poorer control over object details (e.g., giraffe legs). Without debiasing, the model tends to overfit the original input conditions. Moreover, without slicing patches results in incomplete debiasing of appearance features. In contrast, our approach effectively maintains layout control while preventing overfitting to the input image’s appearance.

## 5. Conclusion

We proposed AnyI2V, a training-free image-to-video (I2V) generation approach adapted from the text-to-video (T2V) backbone that integrates flexible spatial conditions from any modality and motion control using user-defined trajectories. Unlike previous methods, AnyI2V eliminates the need for extensive training and also simplifies transferring between different backbones, providing convenience for application. Limitations and Future Work. Despite its advantages, AnyI2V has limitations. It struggles with precise control of very large motion ranges and ambiguous occlusions, which can lead to unclear spatial relationships. Additionally, as feature injection occurs only at earlier denoising steps, the first frame lacks the precise control offered by methods like ControlNet. Future work can focus on improving motion consistency, handling complex occlusions, and integrating lightweight fine-tuning for better adaptability.

The histogram in Fig. 8 presents the results of the ablation study on different optimization targets. We index the j-th query in decoder up blocks.i as Query i.j. The results show that optimizing a single query, specifically Query 1.1 and Query 2.0, yields the leading performance. when optimizing queries from different resolutions, such as Query 1.1 & 2.0 (our setting), a significant improvement is observed. This enhancement occurs because optimizing queries across multiple resolutions enables the model to capture both semantic and structural information

- at various levels. For example, conditions like canny and HED often contain hollow regions that may disappear at smaller resolutions. Higher resolutions, however, provide finer texture details. Consequently, Multi-resolution optimization enhances accuracy and visual fidelity.

Acknowledgement. This project was supported by the National Natural Science Foundation of China (NSFC) under Grant No. 62472104. This work was supported by Damo Academy through Damo Academy Innovative Research Program.

## References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 3
- [3] Yujun Cai, Yiwei Wang, Yiheng Zhu, Tat-Jen Cham, Jianfei Cai, Junsong Yuan, Jun Liu, Chuanxia Zheng, Sijie Yan, Henghui Ding, et al. A unified 3d human motion synthesis model via conditional variational auto-encoder. In ICCV,

2021. 3

- [4] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2, 3
- [5] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, 2024. 8
- [6] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. MeViS: A large-scale benchmark for video segmentation with motion expressions. In ICCV, 2023. 7
- [7] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 3, 6, 7
- [8] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In ECCV, 2025. 2, 3, 7
- [9] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3
- [10] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 3

- [11] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [12] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 35, 2022. 2
- [13] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 1, 2
- [14] Minghui Hu, Jianbin Zheng, Daqing Liu, Chuanxia Zheng, Chaoyue Wang, Dacheng Tao, and Tat-Jen Cham. Cocktail:

- Mixing multi-modality control for text-conditional image generation. In NeurIPS, 2023. 3
- [15] Zhihao Hu and Dong Xu. Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073,

2023. 2, 3

- [16] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV,

2017. 4

- [17] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In ECCV, 2024. 7
- [18] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In ICCV, 2023. 2
- [19] Sungnyun Kim, Junsoo Lee, Kibeom Hong, Daesik Kim, and Namhyuk Ahn. Diffblender: Scalable and composable multimodal text-to-image diffusion models. arXiv preprint arXiv:2305.15194, 2023. 3
- [20] Jonghyun Lee, Hansam Cho, Youngjoon Yoo, Seoung Bum Kim, and Yonghyun Jeong. Compose and conquer: Diffusion-based 3d depth aware composable image synthesis. arXiv preprint arXiv:2401.09048, 2024. 3
- [21] Yaowei Li, Xintao Wang, Zhaoyang Zhang, Zhouxia Wang, Ziyang Yuan, Liangbin Xie, Yuexian Zou, and Ying Shan. Image conductor: Precision control for interactive video synthesis. arXiv preprint arXiv:2406.15339, 2024. 6, 7
- [22] Zhenyi Liao and Zhijie Deng. Lovecon: Text-driven training-free long video editing with controlnet. arXiv preprint arXiv:2310.09711, 2023. 3
- [23] Wan-Duo Kurt Ma, John P Lewis, and W Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation. In SIGGRAPH Asia, 2024. 2, 6, 7
- [24] James MacQueen et al. Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability, 1967. 6
- [25] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-scale video panoptic segmentation in the wild: A benchmark. In CVPR, 2022. 7
- [26] Sicheng Mo, Fangzhou Mu, Kuan Heng Lin, Yanli Liu, Bochen Guan, Yin Li, and Bolei Zhou. Freecontrol: Training-free spatial control of any text-to-image diffusion model with any condition. In CVPR, 2024. 3
- [27] Koichi Namekata, Sherwin Bahmani, Ziyi Wu, Yash Kant, Igor Gilitschenski, and David B Lindell. Sg-i2v: Self-guided trajectory control in image-to-video generation. arXiv preprint arXiv:2411.04989, 2024. 3
- [28] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. Mofa-video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model. In ECCV, 2024. 3, 6, 7

- [29] Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH, 2023. 3, 5
- [30] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv preprint arXiv:2305.11147, 2023. 3
- [31] Haonan Qiu, Zhaoxi Chen, Zhouxia Wang, Yingqing He, Menghan Xia, and Ziwei Liu. Freetraj: Tuning-free trajectory control in video diffusion models. arXiv preprint arXiv:2406.16863, 2024. 3, 6, 7
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2
- [33] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH, 2024. 2, 3
- [34] Yujun Shi, Chuhui Xue, Jun Hao Liew, Jiachun Pan, Hanshu Yan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. In CVPR, 2024. 3, 4, 5
- [35] Xincheng Shuai, Henghui Ding, Xingjun Ma, Rongcheng Tu, Yu-Gang Jiang, and Dacheng Tao. A survey of multimodal-guided image editing with text-to-image diffusion models. arXiv preprint arXiv:2406.14555, 2024. 2
- [36] Xincheng Shuai, Henghui Ding, Zhenyuan Qin, Hao Luo, Xingjun Ma, and Dacheng Tao. Free-form motion control: A synthetic video generation dataset with controllable camera and object motions. In ICCV, 2025. 3
- [37] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2

- [38] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 4, 5
- [39] Yanan Sun, Yanchen Liu, Yinhao Tang, Wenjie Pei, and Kai Chen. Anycontrol: create your artwork with versatile control on text-to-image generation. In ECCV, 2025. 3
- [40] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, 2023. 3, 4
- [41] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 3
- [42] Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guoqiang Wei, Liping Yuan, and Hang Li. Boximator: Generating rich and controllable motions for video synthesis. arXiv preprint arXiv:2402.01566, 2024. 2, 3
- [43] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao,

- and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. NeurIPS, 36, 2024. 3
- [44] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. IJCV, 2024. 2, 3, 8
- [45] Zhouxia Wang, Yushi Lan, Shangchen Zhou, and Chen Change Loy. Objctrl-2.5 d: Training-free object control with camera poses. arXiv preprint arXiv:2412.07721, 2024. 6, 7
- [46] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, 2024. 2, 3
- [47] Svante Wold, Kim Esbensen, and Paul Geladi. Principal component analysis. Chemometrics and intelligent laboratory systems, 2(1-3), 1987. 2
- [48] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. arXiv preprint arXiv:2406.17758, 2024. 2
- [49] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. In ECCV, 2024. 3, 6, 7
- [50] Zeqi Xiao, Yifan Zhou, Shuai Yang, and Xingang Pan. Video diffusion models are training-free motion interpreter and controller. arXiv preprint arXiv:2405.14864, 2024. 3
- [51] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In ECCV,

2025. 2, 3

- [52] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH, 2024. 2, 3
- [53] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 2, 3, 6, 7
- [54] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 3
- [55] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 2, 3
- [56] Zhenghao Zhang, Junchao Liao, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video

- generation. arXiv preprint arXiv:2407.21705, 2024. 3

[57] Haitao Zhou, Chuang Wang, Rui Nie, Jinxiao Lin, Dongdong Yu, Qian Yu, and Changhu Wang. Trackgo: A flexible and efficient method for controllable video

- generation. arXiv preprint arXiv:2408.11475, 2024. 3

