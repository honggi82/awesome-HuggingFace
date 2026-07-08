## Improving 2D Feature Representations by 3D-Aware Fine-Tuning

# arXiv:2407.20229v1[cs.CV]29Jul2024

Yuanwen Yue1 Anurag Das2 Francis Engelmann1,3 Siyu Tang1 Jan Eric Lenssen2

1 ETH Zurich 3 Google 2 Max Planck Institute for Informatics, Saarland Informatics Campus

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

(a) (b)

| |
|---|

| |
|---|

Lifting features to 3D

3D-aware fine-tuning

Downstream tasks

|𝑽𝑺|
|---|

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

(c)

###### (d)

| |
|---|

| |
|---|

Fig. 1: We propose 3D-aware fine-tuning to improve 2D foundation features. Our method starts with lifting 2D image features (e.g. DINOv2 [44]) (b) to a 3D representation. Then we finetune the 2D foundation model using the 3D-aware features (c). We demonstrate that incorporating the fine-tuned features (d) results in improved performance on downstream tasks such as semantic segmentation and depth estimation on a variety of datasets with simple linear probing (right). Feature maps are visualized using principal component analysis (PCA).

578511c8a9_DSC09683

Abstract. Current visual foundation models are trained purely on unstructured 2D data, limiting their understanding of 3D structure of objects and scenes. In this work, we show that fine-tuning on 3D-aware data improves the quality of emerging semantic features. We design a method to lift semantic 2D features into an efficient 3D Gaussian representation, which allows us to re-render them for arbitrary views. Using the rendered 3D-aware features, we design a fine-tuning strategy to transfer such 3D awareness into a 2D foundation model. We demonstrate that models fine-tuned in that way produce features that readily improve downstream task performance in semantic segmentation and depth estimation through simple linear probing. Notably, though fined-tuned on a single indoor dataset, the improvement is transferable to a variety of indoor datasets and out-of-domain datasets. We hope our study encourages the community to consider injecting 3D awareness when training 2D foundation models. Project page: https://ywyue.github.io/FiT3D.

Keywords: Representation learning · Foundation models · Gaussian splatting · Scene understanding

### 1 Introduction

Ever since the emergence of deep neural networks, vision systems are largely trained on 2D datasets. With the scalability of recent architectures, like vision transformers (ViT) [13], several large vision models [7,26,35,44,50] have been trained from a rich set of 2D images by either supervised or self-supervised learning. Visual foundation models have shown impressive utility as general feature extractors that can be applied to improve results on downstream tasks, such as segmentation [38, 57], depth estimation [32, 52, 64], or correspondence estimation [1,66]. They are trained on a large amount of readily available 2D images and, thus, learn statistics about object and scene structure in 2D-pixel space.

Images, as a simple projection of our 3D world, are easy to obtain and provide an efficient way to depict the visual world while at the same time discarding explicit 3D geometry information. It is expected that vision systems purely trained on 2D images cannot fully understand the underlying 3D structure of our world [15]. There are several promising properties of our 3D world, for example, multi-view consistency, and multi-view fusion for solving single-view ambiguities. A crucial limitation of the training setups of these models is that they don’t fully reason about the 3D structure of seen objects. Training images are presented to the network in an unstructured way, without any multi-view or video correspondences that would allow matching observations of the same object from multiple views. As a consequence, these models have limited 3D understanding of objects observed from, e.g., different views are not producing view-consistent features.

In contrast, when we humans observe images, we effortlessly achieve a holistic understanding by not only perceiving the 2D visual content but also exploiting the inferred underlying 3D structure, which we have learned through lifelong observation of stereo and temporal information. In this work, we investigate if large scale 2D vision models can also profit from equipping them with such 3D-aware understanding abilities induced by showing the right type of data.

To this end, we design a novel two-stage approach to improve the 3D-aware understanding ability of 2D foundation models. In the first stage, we aim to obtain 3D-aware features as training data. Motivated by recent advancements in neural scene representation, we design an approach to lift multi-view 2D foundation features into an efficient 3D Gaussian representation [33]. The lifting process exploits multi-view consistency and allows 2D features from different views to complement each other. Moreover, the fused features (Fig. 1 (c)) exhibit high resolution with fine details thanks to the learned 3D structure, emerging from multi-view RGB guidance. Once trained, the 3D Gaussians can render features for arbitrary views. In the following, we refer to features obtained in this way as 3D-aware. In the second stage, we utilize the rendered 3D-aware features to finetune the 2D foundation models (Fig. 2). To this end, we design an efficient fine-tuning strategy to transfer such 3D awareness into 2D foundation models. After fine-tuning, we evaluate the feature quality on downstream tasks that might profit from a better 3D understanding, namely semantic segmentation and depth estimation. Extensive experiments demonstrate that incorporating the

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

MAE

DINOv2-reg

DINOv2

CLIP

DeiT-III

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Input

CLIP (fine-tuned)

DeiT-III (fine-tuned)

DINOv2 (fine-tuned)

DINOv2-reg (fine-tuned)

MAE (fine-tuned)

- Fig. 2: Our 3D-aware fine-tuning is universal and applicable to a variety of 2D vision models, e.g. DINOv2 [44], DINOv2-reg [10], CLIP [48], MAE [26], and DeiT-III [58] (c.f. Sec. 4.5).

3D-aware features improves downstream tasks with simple linear probing and exhibits generalization ability on out-of-domain datasets.

### 2 Related Work

We give an overview about recent self-supervised 2D representation learning techniques in Sec. 2.1, and how emerging features have been distilled into 3D representations in Sec. 2.2. Then, we discuss previous work that utilizes 3D information to improve 2D representation methods in Sec. 2.3.

#### 2.1 2D Representation Learning

Representation learning [4] has achieved remarkable progress in the image domain. It aims to learn generalizable visual features from a rich set of data. Selfsupervised representation learning has gained particular interest since it does not require labeled data. Early works employ pretext tasks for pre-training, which aim to exploit inherent data attributes to automatically generate surrogate labels [12,14,21,43,45,60]. Later, contrastive learning [25] has been popularly used for representation learning by leveraging discriminative signals between images or groups of images [6,7,23,27,44]. More recently, motivated by BERT [11], a new paradigm of masked image modeling [3,8,26] has been proposed for scalable visual learning. Nevertheless, all those methods are only trained on 2D image data, without accessing the underlying 3D structure. Our work aims to supplement the features purely learned from 2D observations with 3D awareness.

#### 2.2 Distilled Feature Fusion Fields

Neural radiance fields (NeRF) [42] emerge as a promising scene representation for high-quality 3D reconstruction and novel view synthesis. Recently, some works [16, 22, 34, 36, 59] explore distilling pre-trained image features (e.g. DINO [7], CLIP [48], LSeg [37], or OpenSeg [20]) into NeRF via neural rendering. Without requiring any labels, such distilled feature fusion fields enable several zeroshot 3D scene understanding tasks, e.g. segmentation, scene editing, and openvocabulary queries. We share similar inspiration from these works by distilling

- 2D features into a 3D representation. However, instead of focusing on perception tasks with feature fields, we are interested in leveraging the rendered 3D-aware features to in turn improve the 2D feature extractor. We demonstrate that the transferred 3D awareness can readily improve the 2D features on both semantic and geometric tasks. Moreover, we extend the recent Gaussian-based representation [33] by designing a method to distill 2D features into 3D Gaussians while keeping high efficiency and memory under bound. There are several concurrent works introducing 3D Gaussians with semantic features [47, 54, 68]. However, none of these works distill features back into 2D models. Our work shows, for the first time, that semantic features fused into 3D representations can effectively improve 2D foundation models via fine-tuning.

#### 2.3 Injecting 3D Priors to 2D

Existing works mainly focus on fusing multi-view 2D features into the 3D representation [24, 30, 31, 41, 46, 53, 56, 61]. Little attention has been paid to the other direction of incorporating 3D awareness into 2D representation learning. Pri3D [29] uses geometric constraints (multi-view consistency and 2D-3D correspondence) from RGB-D reconstructions to learn 3D priors for image-based representations with contrastive learning. Recently, inspired by the masked autoencoder (MAE) [26], several works adopt the masked image modeling strategy to learn 3D priors [2,28,62]. However, all these methods require pre-training the

- 2D feature extractor, typically a Vision Transformer (ViT) backbone [13], using their hand-crafted pretext tasks. The pre-trained models are then employed to downstream tasks via fine-tuning. By contrast, we aim to transfer the 3D awareness embedded in multi-view fused features to the 2D feature extractor through fine-tuning with little computational resources. Our 3D-aware features readily improve downstream task performance with simple linear probing. In addition, we find our 3D-aware features exhibit cleaner and more detailed feature maps compared with the original 2D features (see Sec. D in appendix), while several concurrent works specifically denoise or sharpen 2D feature maps [18,63].
- 3 Method

In this section, we introduce our method for fine-tuning 2D foundation models with 3D-aware features. We present a two-stage pipeline (c.f. Fig 3). In the first stage, we lift per-view 2D features into a multi-view consistent and 3D-aware representation. The representation and setup are described in Sec. 3.1. In the second stage, we use the obtained 3D-aware feature representations as training dataset to finetune the 2D feature extractor, which is detailed in Sec. 3.2. Last, we describe the linear probing methodology for feature evaluation in Sec. 3.3.

#### 3.1 Lifting Features to 3D

Lifting semantic 2D features into 3D has been a trend recently and several different options exist (c.f. Sec. 2). For our purposes of using larger amounts of

###### Stage I: Lifting Features to 3D (Per-scene) Stage II: 3D-Aware Fine-Tuning (Multi-scene)

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Feature Gaussian Splatting

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

… …

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Feature Gaussian Splatting

Feature Gaussian Splatting

[Figure 38]

… …

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

…

…

…

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Feature Gaussian Splatting

[Figure 54]

2D

2D

Multi-view images

… …

[Figure 55]

- Fig. 3: Overall pipeline. We present a two-stage pipeline. In the first stage, we lift 2D foundation features (e.g. DINOv2 [44]) into 3D-aware features by training 3D Gaussian

representation Gi. In the second stage, we use the rendered features to finetune the 2D foundation model ε2D. With → we denote gradient flow.

scenes as training data for 2D models, the most important aspect is efficiency. The representation needs to (1) be able to efficiently fit a large number of scenes into 3D representations and (2) have a fast rendering mechanism for efficient integration into a fine-tuning loop of a 2D foundation model. Thus, we utilize the recent advances in 3D Gaussian splatting [33], which enable fast optimization and real-time rendering. Fig. 4 illustrates how we extend Gaussian splatting to lift 2D foundation features and we detail the method below.

- 3D feature Gaussians. Adapting the formulation of 3D Gaussian splatting [33], we define a set of 3D Gaussians as

G = {(µ,s,R,α,SH,f)j)}1≤j≤M, (1)

where µ ∈ R3 is the 3D mean of the Gaussian, S = diag(s) ∈ R3×3 is the Gaussian scale, R ∈ R3×3 its orientation, α ∈ R is a per-Gaussian opacity, and SH a vector of spherical harmonic coefficients, encoding view-dependent color. The Gaussian covariance matrix is obtained by combining scale and orientation as Σ = RSS⊤R⊤. In addition to the original parameters, we introduce a perGaussian feature vector f ∈ RD to store distilled 2D features in 3D space. Those feature vectors are rasterized into a 2D feature image with our designed feature rasterizer. Inspired by the differentiable color rasterizer of Gaussian splatting, we rasterize the features using point-based α-blending as follows:

- i−1
- j=1

Flow =

(1 − αi) (2)

fiαi

i∈N

where N is a set of ordered Gaussians overlapping the pixel, fi is the feature of each Gaussian and αi is given by evaluating a 2D Gaussian with covariance Σ multiplied with a learned per-point opacity.

Up-projecting features. A strong limitation of 3D Gaussians as representation is their memory consumption. Since there can be millions of Gaussians per scene, it is impossible to store, e.g., the 384-dimensional DINO features directly on

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

Color Rasterizer

[Figure 71]

|𝜀2D|
|---|

[Figure 72]

[Figure 73]

Rendered Images

GT Images

[Figure 74]

[Figure 75]

[Figure 76]

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

CNN

Feature

Rasterizer

[Figure 98]

3D Feature Gaussians

[Figure 99]

[Figure 100]

[Figure 101]

- Fig. 4: Lifting 2D features into 3D Gaussian representation. We equip each Gaussian with a low-dimensional feature vector f. We render colors using the same color rasterizer as Gaussian splatting [33]. We design a feature rasterizer to render a lowdimensional feature image Flow, which is subsequently projected to a high-dimensional feature image Fhigh using a simple CNN. We use 2D foundation features F from model ε2D to supervise the feature learning.

each of the 3D Gaussians. Therefore, to stay memory efficient and keep the fast rendering process, we opt for storing lower dimensional features f ∈ RD with D << 384 and train a scene-specific pixel-space CNN decoder d : Flow  → Fhigh to up-project feature images into high-dimensional feature space after rendering. We analyze the trade-off introduced by this approach in Sec. 4.6.

Optimization. For a given scene, the full 3D Gaussian representation, including our distilled features, is obtained using optimization. Let {Ii}1≤i≤N be a set of multi-view images of a scene with corresponding camera parameters, {Fi}1≤i≤N a corresponding set of feature maps from a 2D feature extractor (e.g. DINOv2 [44]), and rrgb, rfeat rasterization functions that render a set of Gaussians into an RGB or feature image, respectively, using the camera pose Pi of image i. Then, we optimize the Gaussian parameters, to optimally represent images Ii and feature images Fi:

N

Gˆ = arg min

{(µ,s,R,α,SH,f)i}

i=1

Lc(rrgb(G,Pi),Ii) + Lf(d(rfeat(G,Pi),Fi), (3)

where Lc is a pixel-wise l1 loss combined with a D-SSIM term on RGB images, and Lf is a pixel-wise l1 loss on feature images. Notably, we only optimize f with gradients coming from Lf (feature images) and the rest of the parameters only on Lc (RGB loss). This has proven to be essential to obtain a consistent

- 3D feature representation, as a loss from feature space does not lead to correct Gaussian mean, covariance and opacity. We speculate that the reason for this is the missing 3D consistency of the 2D feature extractor. Only through forcing them into a 3D consistent representation, we make them consistent in return.

Algorithm 1 3D-aware fine-tuning algorithm

Input: Pre-trained Feature Gaussian representations {G1, ..., GK}, pre-trained 2D fea-

ture extractor ε2θD, a set of images {Ii}Ni=1 and associated camera poses {Pi}Ni=1. Output: Fine-tuned 2D feature extractor εθ2ˆD.

- 1: Load G ∼ {G1, ..., GK}
- 2: while fine-tuning do
- 3: Sample an image Ii and camera pose Pi, i ∼ U{1, N}
- 4: Retrieve associated feature Gaussian G and CNN decoder d
- 5: Render Fhigh ← d(rfeat(G, Pi))
- 6: Step θ by minimizing L(ε2θD(Ii), Fhigh)
- 7: end while return ε2θˆD

#### 3.2 3D-Aware Fine-Tuning

The procedure described in the last section is used to fit 3D feature Gaussian representations of K scenes. The algorithm of 3D-aware fine-tuning is outlined in Algorithm 1. The fine-tuning process requires training pairs of original 2D feature maps and 3D-aware feature maps. Since it is memory-intensive to save the feature maps, we generate the training pairs on the fly. Considering it is time-consuming to load each pre-trained Gaussian when rendering features, we pre-load all the Gaussians into CPU memory. In each step of the training loop, we randomly sample a view from all the training images, then retrieve its associated feature Gaussian and scene-specific CNN decoder and finally render features Fhigh as the ground truth features for fine-tuning. The fine-tuning loss is a l1 loss between Fhigh (resized) and the output features of the fine-tuned 2D feature extractor.

The above design makes the fine-tuning process efficient and keeps memory consumption under control. Notably, we only need to fine-tune the 2D feature extractor with a small number of epochs (e.g. 1 epoch for DINOv2 [44]) with a small learning rate without additionally introducing any network component. The fine-tuning process is fast and computation-friendly. An analysis of finetuning time is in Sec. 4.6.

#### 3.3 Linear Probing for Downstream Tasks

After fine-tuning on 3D-aware features, we evaluate the emerging features on a set of standard benchmark downstream tasks. To this end, we train a linear head on top of the features to solve tasks of semantic segmentation and depth estimation on several datasets.

Semantic segmentation. A linear layer is trained to predict class logits from patch tokens. The linear layer produces a low-resolution logit map, which is then upsampled to full resolution to obtain a segmentation map.

Depth estimation. We concatenate the [CLS] token of the ViT to each patch token. We divide the depth prediction range into 256 uniformly distributed

bins [5] and use a linear normalization. Then a simple linear layer is trained using a classification loss.

Feature assembly. We concatenate original 2D features with our fine-tuned features. We observe this is key to preserving the generalization ability of the original 2D feature extractor while incorporating the 3D awareness in our finetuned features. Different strategies for feature assembly are evaluated in Sec. 4.6.

### 4 Experiments

#### 4.1 Datasets

Training. We train the feature Gaussians on ScanNet++ [65], which is a largescale dataset of 3D indoor scenes containing sub-millimeter resolution laser scans, registered DSLR images, and commodity RGB-D streams from iPhone. We train on the official training split of 230 scenes, which contain 140451 views.

Evaluation. To examine the effectiveness of the fine-tuned features, we conduct extensive experiments on downstream 2D tasks including semantic segmentation and depth estimation. There is no direct competitor in our study and we instead focus on whether our 3D-aware fine-tuning can bring performance gains compared with the standard 2D feature extractor. We conduct most of the experiments with DINOv2 [44] while also demonstrating the universality of our approach with other vision models in Sec. 4.5. We first evaluate on ScanNet++ [65] validation set, which contains 50 scenes with 30638 images. Then we move on to other indoor datasets ScanNet [9] and NYUd [55]. which have a similar data distribution with ScanNet++ but were captured with different sensors. To investigate the generalization ability of the fine-tuned features, we also perform out-of-domain evaluation on generally distributed datasets including ADE20k [67], Pascal VOC [17] and the outdoor dataset KITTI [19].

#### 4.2 Implementation Details

Feature Gaussians. We wrote custom CUDA kernels for feature rasterization. Each Gaussian is initialized with a random feature vector with a dimension of 64. We implement the up-projecting CNN with a single convolutional layer with a kernel size of 3×3. We train the feature Gaussians of each scene for novel view synthesis and feature rendering jointly for 30000 iterations.

Fine-tuning. We finetune DINOv2 small with a feature dimension of 384 with a batch size of 2 with a learning rate of 1e-5 for 1 epoch. We use horizontal flip as data augmentation. We use the AdamW [40] optimizer with a weight decay factor 1e-4. The fine-tuning on a single Nvidia Tesla A100 takes 8.5 hours.

Linear probing. We follow the linear probing protocol with DINOv2 [44] to ensure a fair comparison. For semantic segmentation, we train the linear layer for 40K iterations with 8 GPUs. For depth estimation, we train the linear layer for 38400 iterations with 8 GPUs. In addition, we use the same data augmentation and learning rate schedule with DINOv2.

#### 4.3 Within-domain Evaluation

Quantitative comparison. We demonstrate the effectiveness of incorporating our 3D-aware features on downstream semantic segmentation (see Tab. 1) and depth estimation task (see Tab. 2) for indoor scenes. For semantic segmentation task, our 3D aware features consistently improve DINOv2 features, achieving a significant performance gain of 2.6%, 2.0% mIoU, and 1.2% on ScanNet++ [65], NYUv2 [55] and ScanNet [9] datasets, respectively. Our 3D-aware DINOv2 features also improve performance on the depth estimation task. In particular, our enhanced features consistently reduce the RMSE across datasets by achieving 0.34 vs. 0.37 (DINOv2) for ScanNet++ [65], 0.42 vs. 0.44 (DINOv2) for NYUv2 [55] and 0.29 vs. 0.31 (DINOv2) for ScanNet [9] datasets.

Qualitative comparison. We qualitatively show the benefits of 3D-aware features in Fig. 5 and Fig. 6. We observe the improvements are mainly reflected in two aspects: (1) cleaner segmentation/depth estimation in homogeneous or textureless regions, e.g. on walls and boards, and (2) better prediction with finegrained details, e.g. on legs of chairs or tables. For (1), during the lifting of

- 2D features to 3D, features from multiple views are aggregated into a holistic representation, thus information from one view implicitly complements other views. We hypothesize that such multi-view awareness is transferred to DINOv2 through fine-tuning. By contrast, standard DINOv2 struggles to infer accurate segmentation or depth from a single image when with ambiguity, thus leading to noisy prediction. For (2), in our feature lifting process, we train the geometry properties (e.g. position and opacity) of Gaussians with RGB color as supervision. The RGB guidance helps feature Gaussians learn detailed 3D structure and render high-resolution feature maps (c.f. Fig. 1 (c)). During the fine-tuning process, the model learns to estimate fine-grained features of objects (c.f. Fig. 1 (d) vs. (b)), which is helpful for capturing detailed structure in downstream tasks.

- Table 1: Semantic segmentation scores on indoor datasets. 3D-aware finetuning consistently leads to improved performance on semantic segmentation in comparison to standard DINOv2 across different indoor datasets.

ScanNet++ [65] NYUv2 [55] ScanNet [9] Method mAcc (↑) mIoU (↑) aAcc (↑) mAcc (↑) mIoU (↑) aAcc (↑) mAcc (↑) mIoU (↑) aAcc (↑)

DINOv2 [44] 40.84 30.19 80.25 76.88 65.55 82.43 55.86 43.6 73.54 + Ours 43.4 32.76 83.54 80.52 67.5 83.37 58.32 44.84 74.37

- Table 2: Depth estimation scores on indoor datasets. 3D-aware fine-tuning consistently leads to improved performance on depth estimation in comparison to standard DINOv2 across different indoor datasets.

ScanNet++ [65] NYUv2 [55] ScanNet [9]

Method RMSE (↓) Rel (↓) RMSE (↓) Rel (↓) RMSE (↓) Rel (↓) DINOv2 [44] 0.3742 0.2836 0.4423 0.1392 0.3089 0.1557 + Ours 0.3361 0.2401 0.4198 0.1300 0.2921 0.1459

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

ScanNet++ScanNetNYUv2

[Figure 106]

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

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Input DINOv2 Ours Ground Truth

- Fig. 5: Semantic segmentation on indoor datasets with linear probing. Incorporating our 3D-aware fine-tuned features helps obtain cleaner and more compact segmentation results, especially for detailed structures and in homogeneous regions.

#### 4.4 Out-of-domain Evaluation

We train feature Gaussians and fine-tune DINOv2 on ScanNet++, a dataset that contains only indoor scenes with the usual content, e.g. tables, chairs and other indoor furniture. We want to analyze how the gains obtained in this setting generalize to other domains, e.g. outdoor scenes. For semantic segmentation, we conduct linear probing on ADE20k [67] and Pascal VOC [17]. For depth estimation, we conduct linear probing on KITTI [19].

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

| |
|---|

| |
|---|

ScanNet++ScanNetNYUv2

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

| |
|---|

| |
|---|

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

| |
|---|

| |
|---|

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

| |
|---|

| |
|---|

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

| |
|---|

| |
|---|

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

| |
|---|

| |
|---|

Input DINOv2 Ours Ground Truth

- Fig. 6: Depth estimation on indoor datasets with linear probing. Incorporating our 3D-aware fine-tuned features helps obtain cleaner depth in textureless regions and more detailed depth on fine-grained structures, e.g. legs of tables or chairs.

Quantitative comparison. We observe that the improvement brought by 3Daware features is generalizable to out-of-domain challenging datasets and also outdoor driving scenes, although to a smaller degree. As shown in Tab. 3, for semantic segmentation task, incorporating our 3D-aware features brings a gain of 1.6% mIoU on the ADE20k [67] and a gain of 1.2% mIoU on Pascal VOC [17] over standard DINOv2 features. Furthermore, we also compare our performance on urban scene dataset KITTI [19] for depth estimation and observe our 3Daware features help to reduce RMSE from 3.03 (DINOv2) to 2.91.

- Table 3: Quantitative performance on out-of-domain datasets. 3D-aware finetuning noticeably improves semantic segmentation on ADE20k and Pascal VOC and depth estimation on KITTI, demonstrating the transferability of the fine-tuned features, even under a significant domain gap.

ADE20k [67] Pascal VOC [17] KITTI [19] Method mAcc (↑) mIoU (↑) aAcc (↑) mAcc (↑) mIoU (↑) aAcc (↑) RMSE (↓) Rel (↓)

DINOv2 [44] 56.74 44.28 79.73 90.61 81.14 95.72 3.03 0.10 + Ours 58.71 45.93 81.05 91.04 82.35 96.14 2.91 0.09

- Table 4: Generalization on other 2D vision models. Our 3D-aware fine-tuning applies to other 2D vision models and readily improves their performance.

DINOv2-reg CLIP MAE DeiT-III

mIoU (↑) RMSE (↓) mIoU (↑) RMSE (↓) mIoU (↑) RMSE (↓) mIoU (↑) RMSE (↓) Original 30.92 0.4190 25.61 0.4324 17.19 0.4855 18.62 0.4350

###### + Ours 33.39 0.3824 28.82 0.3960 20.27 0.4795 22.98 0.3820

Qualitative comparison. We show qualitative comparison on out-of-domain datasets in Fig. 7. We observe similar improvements as in the within-domain datasets. Even though the 3D-aware fine-tuning is only conducted on indoor dataset ScanNet++, the fine-tuned features exhibit transferability to improve segmentation results for the detailed structures of common objects like bicycle and animal, and help achieve more compact segmentation of objects like tree, building and pillar. On depth estimation, the incorporated 3D-aware features are helpful in capturing the detailed structure of trees.

#### 4.5 Generalization to Other Vision Models

We conduct experiments on more vision models (DINOv2-reg [10], CLIP [48], MAE [26], DeiT-III [58]) to prove the universality of our method. We show the linear probing results of semantic segmentation and depth estimation on ScanNet++ validation set in Tab. 4. Our method consistently improves all the models. We also visualize the features in Fig. 2. Note that there is little visual difference between the original MAE features and the fined-tuned features but our method still improves them.

#### 4.6 Ablation Studies and Analysis

We conduct ablation studies on semantic segmentation on NYUv2 dataset with DINOv2.

Feature dimension of each Gaussian. We attach a low-dimensional feature vector with each Gaussian and then up-project it to the same space with DINOv2. Tab. 5 indicates that increasing the feature dimension from 32 to 64 will improve the performance of fine-tuned DINOv2 with an acceptable higher memory and longer training time. Increasing the feature dimension further to 128 is

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

DINOv2 Ours

ADE20kPascalVOCKITTI

DINOv2 Ours

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

DINOv2 Ours

DINOv2 Ours

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

DINOv2 Ours

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

DINOv2 Ours DINOv2 Ours DINOv2 Ours

[Figure 166]

[Figure 167]

| |
|---|

| |
|---|

DINOv2 Ours

- Fig. 7: Results on out-of-domain datasets with linear probing. Our 3D-aware fine-tuned features help achieve better segmentation and capture detailed structure.

not feasible in our hardware due to the large memory consumption. We chose a feature dimension of 64 as a good compromise between model performance, memory consumption, as well as training time.

Feature assembly strategy. We study different strategies to assemble the finetuned features with the original DINOv2 features in Tab. 7. We explore simple channel-wise adding and concatenation. Alternatively, we first concatenate the fine-tuned features with the original DINOv2 features then use a liner layer to fuse them to the same feature space of DINOv2. We observe simple concatenation works well in incorporating learned 3D-aware features while preserving original generalization ability.

Fine-tuning epochs. We finetune DINOv2 using the features rendered by the pre-trained feature Gaussians. Tab. 8 suggests that a single epoch is sufficient to transfer the 3D awareness to DINOv2 and more epochs may harm the model’s generalization ability.

Fine-tuning vs. adapter. Besides directly fine-tuning DINOv2, we explore an alternative strategy where we keep DINOv2 frozen and introduce an adapter on top of that. The adapter is a single Swin Transformer block [39]. We observe the

- Table 5: Ablation study on feature dimension of 3D Gaussian. Increasing feature dimensions improves performance at the cost of larger memory consumption and longer training time.

Table 6: Ablation study on fine-tuning vs. adapter. An adapter is a tiny network plugged into the frozen DINOv2 features.

Feature dimension Average memory (MB) Average Training time (h) mAcc (↑) mIoU (↑) aAcc (↑)

32 370 1.3 78.77 67.15 83.44 64 495 1.6 80.52 67.5 83.37

128 750 2.5 - - -

Strategy mAcc (↑) mIoU (↑) aAcc (↑) Fine-tuning 91.04 82.35 96.14

Adapter 90.97 82.02 95.96

Table 7: Ablation study on feature assembly. We study different strategies to assemble fine-tuned features with the original DINOv2.

Table 8: Ablation study on fine-tuning epochs. We find fine-tuning with a single epoch with 8.5 hours is sufficient to achieve good performance.

Strategy mAcc (↑) mIoU (↑) aAcc (↑)

Adding 77.97 66.0 82.85 Linear fusion 78.22 66.39 82.89

Concatenation 80.52 67.5 83.37

Epochs Fine-tun. time (h) mAcc (↑) mIoU (↑) aAcc (↑)

- 1 8.5 80.52 67.5 83.37
- 2 17 78.72 67.25 83.54
- 3 25.5 79.5 67.18 83.24

adapter can achieve comparable performance with fine-tuning (Tab. 6), however, with longer training time. We stick with the fine-tuning strategy for simplicity without introducing any additional network component.

Limitations and discussion. Our work makes an initial step to transfer multiview consistent and 3D-aware features encoded by a 3D Gaussian representation to 2D foundation model via fine-tuning. We demonstrate the 3D-aware features are helpful for downstream tasks. However, we still need the original features to keep the generalization ability. We attribute this to the limited diversity of our

- 3D training data (only a single indoor dataset) and hypothesize that this can be remedied by fine-tuning on larger-scale data.

### 5 Conclusion

In this work, we present a method to inject 3D awareness into 2D foundation models. We first lift 2D foundation features into a 3D Gaussian representation and then use the rendered multi-view consistent and 3D-aware features to in turn fine-tune the 2D foundation model. Our experiments show that incorporating the fine-tuned features readily leads to improved performance on both semantic and geometric tasks through simple linear probing. Although we only conduct the 3D-aware fine-tuning on a single dataset ScanNet++, we demonstrate the learned 3D awareness is transferable across a variety of datasets in different domains. We hope our work inspires future research to consider equipping 2D foundation models with 3D-aware understanding.

Acknowledgements Francis Engelmann is partially supported by an ETH AI Center postdoctoral research fellowship and an ETH Zurich Career Seed Award. This project was also partially supported by Saarbrücken Research Center for Visual Computing, Interaction and AI.

### References

- 1. Amir, S., Gandelsman, Y., Bagon, S., Dekel, T.: Deep ViT Features as Dense Visual Descriptors. In: European Conference on Computer Vision (ECCV) Workshops

(2022) 2

- 2. Bachmann, R., Mizrahi, D., Atanov, A., Zamir, A.: MultiMAE: Multi-modal Multitask Masked Autoencoders. In: European Conference on Computer Vision (ECCV)

(2022) 4

- 3. Bao, H., Dong, L., Piao, S., Wei, F.: BEiT: BERT Pre-training of Image Transformers. In: International Conference on Learning Representations (ICLR) (2022) 3
- 4. Bengio, Y., Courville, A., Vincent, P.: Representation Learning: A Review and New Perspectives. Transactions on Pattern Analysis and Machine Intelligence (PAMI)

(2013) 3

- 5. Bhat, S.F., Alhashim, I., Wonka, P.: Adabins: Depth Estimation Using Adaptive Bins. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2021) 8
- 6. Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A.: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. In: International Conference on Neural Information Processing Systems (NeurIPS) (2020) 3
- 7. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging Properties in Self-Supervised Vision Transformers. In: International Conference on Computer Vision (ICCV) (2021) 2, 3
- 8. Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., Sutskever, I.: Generative Pretraining From Pixels. In: International Conference on Machine Learning (ICML) (2020) 3
- 9. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2017) 8, 9, 23
- 10. Darcet, T., Oquab, M., Mairal, J., Bojanowski, P.: Vision Transformers Need Registers. In: International Conference on Learning Representations (ICLR) (2024) 3, 12
- 11. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL (2018) 3
- 12. Doersch, C., Gupta, A., Efros, A.A.: Unsupervised Visual Representation Learning by Context Prediction. In: International Conference on Computer Vision (ICCV)

(2015) 3

- 13. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale. In: International Conference on Learning Representations (ICLR) (2020) 2, 4
- 14. Dosovitskiy, A., Springenberg, J.T., Riedmiller, M., Brox, T.: Discriminative Unsupervised Feature Learning With Convolutional Neural Networks. In: International Conference on Neural Information Processing Systems (NeurIPS) (2014) 3
- 15. El Banani, M., Raj, A., Maninis, K.K., Kar, A., Li, Y., Rubinstein, M., Sun, D., Guibas, L., Johnson, J., Jampani, V.: Probing the 3D Awareness of Visual Foundation Models. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 2
- 16. Engelmann, F., Manhardt, F., Niemeyer, M., Tateno, K., Tombari, F.: OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered

Novel Views. In: International Conference on Learning Representations (ICLR)

(2024) 3

- 17. Everingham, M., Eslami, S.A., Van Gool, L., Williams, C.K., Winn, J., Zisserman, A.: The Pascal Visual Object Classes Challenge: A Retrospective. International Journal of Computer Vision (2015) 8, 10, 11, 12
- 18. Fu, S., Hamilton, M., Brandt, L., Feldman, A., Zhang, Z., Freeman, W.T.: FeatUp: A Model-Agnostic Framework for Features at Any Resolution. In: International Conference on Learning Representations (ICLR) (2024) 4
- 19. Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision Meets Robotics: The Kitti Dataset. The International Journal of Robotics Research (2013) 8, 10, 11, 12, 20
- 20. Ghiasi, G., Gu, X., Cui, Y., Lin, T.Y.: Scaling Open-Vocabulary Image Segmentation With Image-Level Labels. In: European Conference on Computer Vision (ECCV) (2022) 3
- 21. Gidaris, S., Singh, P., Komodakis, N.: Unsupervised Representation Learning by Predicting Image Rotations. In: International Conference on Learning Representations (ICLR) (2018) 3
- 22. Goel, R., Sirikonda, D., Saini, S., Narayanan, P.: Interactive Segmentation of Radiance Fields. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2023) 3
- 23. Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al.: Bootstrap Your Own Latent-a New Approach to Self-Supervised Learning. In: International Conference on Neural Information Processing Systems (NeurIPS) (2020) 3
- 24. Ha, H., Song, S.: Semantic Abstraction: Open-world 3D Scene Understanding From 2D Vision-Language Models. In: Conference on Robot Learning (CoRL) (2022) 4
- 25. Hadsell, R., Chopra, S., LeCun, Y.: Dimensionality Reduction by Learning an Invariant Mapping. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2006) 3
- 26. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked Autoencoders Are Scalable Vision Learners. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2022) 2, 3, 4, 12
- 27. He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.: Momentum Contrast for Unsupervised Visual Representation Learning. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2020) 3
- 28. Hou, J., Dai, X., He, Z., Dai, A., Nießner, M.: Mask3D: Pre-training 2D Vision Transformers by Learning Masked 3D Priors. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2023) 4
- 29. Hou, J., Xie, S., Graham, B., Dai, A., Nießner, M.: Pri3D: Can 3D Priors Help 2D Representation Learning? In: International Conference on Computer Vision (ICCV) (2021) 4
- 30. Huang, R., Peng, S., Takmaz, A., Tombari, F., Pollefeys, M., Song, S., Huang, G., Engelmann, F.: Segment3D: Learning Fine-Grained Class-Agnostic 3D Segmentation without Manual Labels. European Conference on Computer Vision (ECCV)

(2024) 4

- 31. Jatavallabhula, K.M., Kuwajerwala, A., Gu, Q., Omama, M., Chen, T., Li, S., Iyer, G., Saryazdi, S., Keetha, N., Tewari, A., et al.: ConceptFusion: Open-set Multimodal 3D Mapping. Robotics: Science and Systems (RSS) (2023) 4
- 32. Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R.C., Schindler, K.: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. In: International Conference on Computer Vision and Pattern Recognition (CVPR)

(2024) 2

- 33. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics (2023) 2, 4, 5, 6
- 34. Kerr, J., Kim, C.M., Goldberg, K., Kanazawa, A., Tancik, M.: LERF: Language Embedded Radiance Fields. In: International Conference on Computer Vision (ICCV) (2023) 3
- 35. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment Anything. In: International Conference on Computer Vision (ICCV) (2023) 2
- 36. Kobayashi, S., Matsumoto, E., Sitzmann, V.: Decomposing Nerf for Editing via Feature Field Distillation. In: International Conference on Neural Information Processing Systems (NeurIPS) (2022) 3
- 37. Li, B., Weinberger, K.Q., Belongie, S., Koltun, V., Ranftl, R.: Language-driven Semantic Segmentation. In: International Conference on Learning Representations (ICLR) (2022) 3
- 38. Li, F., Zhang, H., Xu, H., Liu, S., Zhang, L., Ni, L.M., Shum, H.Y.: Mask DINO: Towards a Unified Transformer-Based Framework for Object Detection and Segmentation. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2023) 2
- 39. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. In: International Conference on Computer Vision (ICCV) (2021) 13
- 40. Loshchilov, I., Hutter, F.: Decoupled Weight Decay Regularization. In: International Conference on Learning Representations (ICLR) (2019) 8
- 41. Mazur, K., Sucar, E., Davison, A.J.: Feature-realistic Neural Fusion for Real-time, Open Set Scene Understanding. In: International Conference on Robotics and Automation (ICRA) (2023) 4
- 42. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In: European Conference on Computer Vision (ECCV) (2020) 3
- 43. Noroozi, M., Favaro, P.: Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. In: European Conference on Computer Vision (ECCV) (2016) 3
- 44. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: DINOv2: Learning Robust Visual Features Without Supervision. Transactions on Machine Learning Research (2023) 1, 2, 3, 5, 6, 7, 8, 9, 12, 19, 20
- 45. Pathak, D., Girshick, R., Dollár, P., Darrell, T., Hariharan, B.: Learning Features by Watching Objects Move. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2017) 3
- 46. Peng, S., Genova, K., Jiang, C., Tagliasacchi, A., Pollefeys, M., Funkhouser, T.: OpenScene: 3D Scene Understanding with Open Vocabularies. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2023) 4
- 47. Qin, M., Li, W., Zhou, J., Wang, H., Pfister, H.: LangSplat: 3D Language Gaussian Splatting. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 4
- 48. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning Transferable Visual Models From Natural Language Supervision. In: International Conference on Machine Learning (ICML) (2021) 3, 12

- 49. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision Transformers for Dense Prediction. In: International Conference on Computer Vision (ICCV) (2021) 20
- 50. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-Resolution Image Synthesis With Latent Diffusion Models. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2022) 2
- 51. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recognition challenge. International journal of computer vision (2015) 20
- 52. Saxena, S., Herrmann, C., Hur, J., Kar, A., Norouzi, M., Sun, D., Fleet, D.J.: The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation. In: International Conference on Neural Information Processing Systems (NeurIPS) (2024) 2
- 53. Shen, W., Yang, G., Yu, A., Wong, J., Kaelbling, L.P., Isola, P.: Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation. In: Conference on Robot Learning (CoRL) (2023) 4
- 54. Shi, J.C., Wang, M., Duan, H.B., Guan, S.H.: Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 4
- 55. Silberman, N., Hoiem, D., Kohli, P., Fergus, R.: Indoor Segmentation and Support Inference From RGBD Images. In: European Conference on Computer Vision (ECCV) (2012) 8, 9, 19, 20, 23
- 56. Takmaz, A., Fedele, E., Sumner, R.W., Pollefeys, M., Tombari, F., Engelmann, F.: OpenMask3D: Open-Vocabulary 3D Instance Segmentation. In: International Conference on Neural Information Processing Systems (NeurIPS) (2023) 4
- 57. Tan, H., Wu, S., Pi, J.: Semantic Diffusion Network for Semantic Segmentation. In: International Conference on Neural Information Processing Systems (NeurIPS)

(2022) 2

- 58. Touvron, H., Cord, M., Jégou, H.: Deit III: Revenge of the ViT. In: European Conference on Computer Vision (ECCV) (2022) 3, 12
- 59. Tschernezki, V., Laina, I., Larlus, D., Vedaldi, A.: Neural Feature Fusion Fields: 3D Distillation of Self-Supervised 2D Image Representations. In: International Conference on 3D Vision (3DV) (2022) 3
- 60. Wang, X., Gupta, A.: Unsupervised Learning of Visual Representations Using Videos. In: International Conference on Computer Vision (ICCV) (2015) 3
- 61. Weder, S., Blum, H., Engelmann, F., Pollefeys, M.: LabelMaker: Automatic Semantic Label Generation from RGB-D Trajectories. In: International Conference on 3D Vision (3DV) (2024) 4
- 62. Weinzaepfel, P., Leroy, V., Lucas, T., Brégier, R., Cabon, Y., Arora, V., Antsfeld, L., Chidlovskii, B., Csurka, G., Revaud, J.: CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion. In: International Conference on Neural Information Processing Systems (NeurIPS) (2022) 4
- 63. Yang, J., Luo, K.Z., Li, J., Weinberger, K.Q., Tian, Y., Wang, Y.: Denoising Vision Transformers. In: European Conference on Computer Vision (ECCV) (2024) 4
- 64. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., Zhao, H.: Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 2
- 65. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes. In: International Conference on Computer Vision (ICCV) (2023) 8, 9, 19, 20, 22

- 66. Zhang, J., Herrmann, C., Hur, J., Polania Cabrera, L., Jampani, V., Sun, D., Yang, M.H.: A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence. In: International Conference on Neural Information Processing Systems (NeurIPS) (2023) 2
- 67. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene Parsing Through ade20K Dataset. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2017) 8, 10, 11, 12, 19
- 68. Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S., Wang, Z., Kadambi, A.: Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields. In: International Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 4

In the appendix, we provide (1) experiments with other DINOv2 ViT variants (Appendix A) (2) experiments on more tasks and heads (Appendix B) (3) experiments on impact of feature dimensions for linear probing (Appendix C) (4) more visualization and K-Means clustering of features (Appendix D).

### A Experiments With More DINOv2 ViT Variants

To demonstrate that the effectiveness of our 3D-aware fine-tuning is agnostic to DINOv2 architecture variants, we conduct additional experiments using the ViT-Base architecture with a feature dimension of 768. We show the results of semantic segmentation and depth estimation across multiple in-domain and out-of-domain datasets in Tab. 9 and Tab. 10, respectively. We observe a similar trend of improvement with the ViT-B architecture. For example, on the finetuning dataset ScanNet++, incorporating our fine-tuned features brings an improvement of 3.47% mIoU on semantic segmentation and reduces 0.03 RMSE on depth estimation. On other indoor datasets NYUv2 and out-of-domain dataset ADE20k, our 3D-aware fine-tuning consistently improves the original DINOv2. This experiment indicates that our 3D-aware fine-tuning is applicable to different ViT architectures and readily benefits downstream tasks.

- Table 9: Results of ViT variants on semantic segmentation. Our 3D-aware fine-tuning yields consistent improvements on semantic segmentation for both ViT-S and ViT-B architectures.

ScanNet++ [65] NYUv2 [55] ADE20k [67] Method Arch. mAcc (↑) mIoU (↑) aAcc (↑) mAcc (↑) mIoU (↑) aAcc (↑) mAcc (↑) mIoU (↑) aAcc (↑)

DINOv2 [44] ViT-S 40.84 30.19 80.25 76.88 65.55 82.43 56.74 44.28 79.73 + Ours ViT-S 43.4 32.76 83.54 80.52 67.5 83.37 58.71 45.93 81.05

DINOv2 [44] ViT-B 42.99 32.72 82.05 80.56 68.45 84.03 59.11 47.16 80.79 + Ours ViT-B 46.35 36.19 85.5 80.58 70.56 85.72 62.18 49.5 82.52

##### Table 10: Results of ViT variants on depth estimation. Our 3D-aware finetuning yields consistent improvements on depth segmentation for both ViT-S and ViT-

B architectures.

ScanNet++ [65] NYUv2 [55] KITTI [19] Method Arch. RMSE (↓) Rel (↓) RMSE (↓) Rel (↓) RMSE (↓) Rel (↓) DINOv2 [44] ViT-S 0.3742 0.2836 0.4423 0.1392 3.0322 0.0965 + Ours ViT-S 0.3361 0.2401 0.4198 0.1300 2.9125 0.0891 DINOv2 [44] ViT-B 0.3439 0.2576 0.3986 0.1218 2.9071 0.095

+ Ours ViT-B 0.3174 0.2324 0.3802 0.1171 2.7923 0.0897

##### Table 11: Results on image classification. Our features do not improve image classification results.

Table 12: Results with DPT head on depth estimation. Beyond linear probing, we evaluate with DPT head for depth estimation and observe consistent improvement.

Method Acc. (↑) DINOv2 [44] 80.02

+ Ours 80.00

Method RMSE (↓) Rel (↓) DINOv2 [44] 0.3027 0.2149

+ Ours 0.2830 0.1936

### B Experiments on More Tasks and Heads

Image classification. We additionally evaluate our approach with DINOv2 small on image classification. We train a linear probing on ImageNet-1K [51] for 12500 iterations on a single GPU. As shown in tab. 11, our features do not improve image classification results. This is expected as classification mainly relies on CLS token of ViT while our method aims to improve image patch features.

DPT head. Beyond linear probing, we evaluate DINOv2 small with the DPT head [49] for depth estimation on ScanNet++. In comparison with the linear probing results (Tab. 2 in the main paper), the DPT head improves both results and our features are still helpful in this setup (see Tab. 12). This demonstrates that improvement brought the 3D-aware features is not limited to linear probing but also applicable to more complex heads.

### C Experiments on Feature Dimensions

We concatenate original 2D features with our fine-tuned features, which will introduce increased feature dimension. In this experiment, we compare with DINOv2 small with duplicate features for linear probing of semantic segmentation and depth estimation on ScanNet++. As shown in Tab. 13, simply duplication ○2 only leads to little improvement compared with incorporating our fine-tuned features ○.3 This verifies that it is not the number of feature dimensions that leads to improvement.

Table 13: Results of duplicating DINOv2 features for linear probing. We verify that it is not the number of feature dimensions that leads to improvement by showing that simple duplication of original features does not help.

Fdim mIoU (↑) RMSE (↓)

- ○1 DINOv2 384 30.19 0.3742
- ○2 DINOv2 × 2 768 30.31 0.3676
- ○3 DINOv2 + Ours 768 32.76 0.3361

### D Visual Analysis of Features

We train feature Gaussians and conduct 3D-aware fine-tuning on ScanNet++. In Fig 8, we visualize the features rendered by pre-trained feature Gaussians (4th column), features of DINOv2 (2nd column) and our fine-tuned features (3rd column). The colors of features in all visualizations are produced using principle component analysis (PCA). The standard DINOv2 features suffer from noise and rough object boundaries. After lifting those features to 3D by training feature Gaussians, we observe the rendered features enjoy cleaner and sharper object boundaries. We then fine-tune DINOv2 using those rendered features, which results in compact and clean feature representations.

Although the fine-tuning is only conducted on ScanNet++, we observe the resulting fine-tuned DINOv2 can generalize to other indoor datasets (e.g. NYUv2 and ScanNet) and produces cleaner feature maps and more pronounced structure details (Fig. 9). Similar patterns can also be found in out-of-domain datasets (e.g. Pascal VOC, ADE20k and KITTI), as shown in Fig. 10. Visualizations of these feature representations indicate that 3D-aware fine-tuning is helpful and transferable. We observe the improvements are mainly reflected in two aspects: (1) cleaner and more compact feature maps. (2) clearer object boundaries and structured details emerge.

Feature clustering. We also use a simple K-Means clustering to directly examine the semantic concepts encoded in the feature representations. We show the K-means clustering results in Fig. 11. The improvements in our features are directly reflected in those simple clustering results. As shown in Fig. 11, the K-Means results of DINOv2 (3rd column) are strongly affected by artifacts and noise. By contrast, our clustering results (5th column) are much cleaner and more compact. In addition, we observe the PCA features and K-Means clustering of our 3D-aware fine-tuned features exhibit higher temporal consistency than the standard DINOv2 features. Please check our demos on our project page to see the full visualizations of video sequences.

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

Input DINOv2 DINOv2 (fine-tuned) Feature Gaussians

- Fig. 8: Feature visualization on ScanNet++ [65]. We visualize the features rendered by pre-trained feature Gaussians (4th column), features of DINOv2 (2nd column) and our fine-tuned features (3rd column).

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

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

Input DINOv2 DINOv2 (fine-tuned) Input DINOv2 DINOv2 (fine-tuned)

##### Fig. 9: Feature visualization on indoor datasets NYUv2 [55] and ScanNet [9]. Our 3D-aware fine-tuning helps obtain more compact features and capture detailed structures.

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

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

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Input DINOv2 DINOv2 (fine-tuned) Input DINOv2 DINOv2 (fine-tuned)

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

Input DINOv2 DINOv2 (fine-tuned)

##### Fig. 10: Feature visualization on out-of-domain datasets. Our 3D-aware finetuning is generalizable to out-of-domain datasets and helps obtain more compact features and capture detailed structures.

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

[Figure 343]

[Figure 344]

Input DINOv2 DINOv2 (fine-tuned)

##### Fig. 11: K-Means clustering of features. We show the PCA features and K-Means clustering results of DINOv2 (2, 3th columns) and our 3D-aware fine-tuning features (4, 5th columns). Our K-Means clustering results are more compact and detailed than DINOv2.

