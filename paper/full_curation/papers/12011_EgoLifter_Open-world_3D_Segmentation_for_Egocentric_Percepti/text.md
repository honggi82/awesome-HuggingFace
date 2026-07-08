# arXiv:2403.18118v2[cs.CV]22Jul2024

## EgoLifter: Open-world 3D Segmentation for Egocentric Perception

Qiao Gu1,2⋆ , Zhaoyang Lv2 , Duncan Frost2 , Simon Green2 , Julian Straub2 , and Chris Sweeney2

1 University of Toronto, Toronto, ON M5S 1A1, Canada q.gu@mail.utoronto.ca 2 Meta Reality Labs, Redmond, WA 98052, USA {zhaoyang, frost, simongreen, jstraub, sweeneychris}@meta.com

Abstract. In this paper we present EgoLifter, a novel system that can automatically segment scenes captured from egocentric sensors into a complete decomposition of individual 3D objects. The system is specifically designed for egocentric data where scenes contain hundreds of objects captured from natural (non-scanning) motion. EgoLifter adopts 3D Gaussians as the underlying representation of 3D scenes and objects and uses segmentation masks from the Segment Anything Model (SAM) as weak supervision to learn flexible and promptable definitions of object instances free of any specific object taxonomy. To handle the challenge of dynamic objects in ego-centric videos, we design a transient prediction module that learns to filter out dynamic objects in the 3D reconstruction. The result is a fully automatic pipeline that is able to reconstruct 3D object instances as collections of 3D Gaussians that collectively compose the entire scene. We created a new benchmark on the Aria Digital Twin dataset that quantitatively demonstrates its state-of-the-art performance in open-world 3D segmentation from natural egocentric input. We run EgoLifter on various egocentric activity datasets which shows the promise of the method for 3D egocentric perception at scale. Please visit project page at https://egolifter.github.io/.

Keywords: Egocentric Perception · Open-world Segmentation · 3D Reconstruction

### 1 Introduction

The rise of personal wearable devices has led to the increased importance of egocentric machine perception algorithms capable of understanding the physical 3D world around the user. Egocentric videos directly reflect the way humans see the world and contain important information about the physical surroundings and how the human user interacts with them. The specific characteristics of egocentric motion, however, present challenges for 3D computer vision and machine perception algorithms. Unlike datasets captured with deliberate "scanning" motions, egocentric videos are not guaranteed to provide complete coverage of the

⋆ Work done during internship at Reality Labs, Meta.

Input Egocentric Video

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

EgoLifter

Novel-view 2D Segmentation

[Figure 5]

[Figure 6]

Open-world 3D Detection/Segmentation

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

3D Gaussians with Color Features (Clustered)

3D Object Extraction & Scene Editing

- Fig. 1: EgoLifter solves 3D reconstruction and open-world segmentation simultaneously from egocentric videos. EgoLifter augments 3D Gaussian Splatting [18] with instance features and lifts open-world 2D segmentation by contrastive learning, where 3D Gaussians belonging to the same objects are learned to have similar features. In this way, EgoLifter solves the multi-view mask association problem and establishes a consistent 3D representation that can be decomposed into object instances. EgoLifter enables multiple downstream applications including detection, segmentation, 3D object extraction and scene editing. See project webpage for animated visualizations.

scene. This makes reconstruction challenging due to limited or missing multiview observations. The specific content found in egocentric videos also presents challenges to conventional reconstruction and perception algorithms. An average adult interacts with hundreds of different objects many thousands of times per day [5]. Egocentric videos capturing this frequent human-object interaction thus contain a huge amount of dynamic motion with challenging occlusions. A system capable of providing useful scene understanding from egocentric data must therefore be able to recognize hundreds of different objects while being robust to sparse and rapid dynamics.

To tackle the above challenges, we propose EgoLifter, a novel egocentric 3D perception algorithm that simultaneously solves reconstruction and open-world

3D instance segmentation from egocentric videos. We represent the geometry of the scene using 3D Gaussians [18] that are trained to minimize photometric reconstruction of the input images. To learn a flexible decomposition of objects that make up the scene we leverage SAM [22] for its strong understanding of objects in 2D and lift these object priors into 3D using contrastive learning. Specifically, 3D Gaussians are augmented with additional N-channel feature embeddings that are rasterized into feature images. These features are then learned to encode the object segmentation information by contrastive lifting [2]. This technique allows us to learn a flexible embedding with useful object priors that can be used for several downstream tasks.

To handle the difficulties brought by the dynamic objects in egocentric videos, we design EgoLifter to focus on reconstructing the static part of the 3D scene. EgoLifter learns a transient prediction network to filter out the dynamic objects from the reconstruction process. This network does not need extra supervision and is optimized together with 3D Gaussian Splatting using solely the photometric reconstruction losses. We show that the transient prediction module not only helps with photorealistic 3D reconstruction but also results in cleaner lifted features and better segmentation performance.

EgoLifter is able to reconstruct a 3D scene while decomposing it into 3D object instances without the need for any human annotation. The method is evaluated on several egocentric video datasets. The experiments demonstrate strong 3D reconstruction and open-world segmentation results. We also showcase several qualitative applications including 3D object extraction and scene editing. The contributions of this paper can be summarized as follows:

- – We demonstrate EgoLifter, the first system that can enable open-world 3D understanding from natural dynamic egocentric videos.
- – By lifting output from recent image foundation models to 3D Gaussian Splatting, EgoLifter achieve strong open-world 3D instance segmentation performance without the need for expensive data annotation or extra training.
- – We propose a transient prediction network, which filters out transient objects from the 3D reconstruction results. By doing so, we achieve improved performance on both reconstruction and segmentation of static objects.
- – We set up the first benchmark of dynamic egocentric video data and quantitatively demonstrate the leading performance of EgoLifter. On several largescale egocentric video datasets, EgoLifter showcases the ability to decompose a 3D scene into a set of 3D object instances, which opens up promising directions for egocentric video understanding in AR/VR applications.

### 2 Related Work

- 2.1 3D Gaussian Models

##### 3D Gaussian Splatting (3DGS) [18] has emerged as a powerful algorithm for novel view synthesis by 3D volumetric neural rendering. It has shown promising performance in many applications, like 3D content generation [4, 51, 62],

SLAM [17,30,57] and autonomous driving [58,65]. Recent work extend 3DGS to dynamic scene reconstruction [7, 27, 56, 59, 60]. The pioneering work from Luiten et al. [27] first learns a static 3DGS using the multi-view observations at the initial timestep and then updates it by the observations at the following timesteps. Later work [56,60] reconstructs dynamic scenes by deforming a canonical 3DGS using a time-conditioned deformation network. Another line of work [7,59] extends 3D Gaussians to 4D, with an additional variance dimension in time. While they show promising results in dynamic 3D reconstruction, they typically require training videos from multiple static cameras. However, in egocentric perception, there are only one or few cameras with a narrow baseline. As we show in the experiments, dynamic 3DGS struggles to track dynamic objects and results in floaters that harm instance segmentation feature learning.

#### 2.2 Open-world 3D Segmentation

Recent research on open-world 3D segmentation [9, 13, 15, 16, 19, 23, 24, 31, 38, 44,45,52,54] has focused on lifting outputs from 2D open-world models - large, powerful models that are trained on Internet-scale datasets and can generalize to a wide range of concepts [22, 35, 40, 42]. These approaches transfer the ability of powerful 2D models to 3D, require no training on 3D models, and alleviate the need for large-scale 3D datasets that are expensive to collect. Early work [16,19,38] lifts dense 2D feature maps to 3D representations by multi-view feature fusion, where each position in 3D is associated with a feature vector. This allows queries in fine granularity over 3D space, but it also incurs high memory usage. Other work [12,26,49] builds object-decomposed 3D maps using

- 2D open-world detection or segmentation models [22,25], where each 3D object is reconstructed separately and has a single feature vector. This approach provides structured 3D scene understanding in the form of object maps or scene graphs but the scene decomposition is predefined and the granularity does not vary according to the query at inference time. Recently, another work [2] lifts

- 2D instance segmentation to 3D by contrastive learning. It augments NeRF [33] with an extra feature map output and optimizes it such that pixels belonging to the same 2D segmentation mask are pulled closer and otherwise pushed apart. In this way, multi-view association of 2D segmentation is solved in an implicit manner and the resulting feature map allows instance segmentation by either user queries or clustering algorithms. Concurrent Work. We briefly review several recent and unpublished preprints that further explore topics in this direction using techniques similar to ours. Concurrently, OmniSeg3D [63] and GARField [20] follow the idea of [2], and focus on learning 3D hierarchical segmentation. They both take advantage of the multi-scale outputs from SAM [22] and incorporate the scales into the lifted features. GaussianGrouping [61] also approaches the open-world 3D segmentation problem but they rely on a 2D video object tracker for multi-view association instead of directly using 2D segmentation via contrastive learning. Similar to our improvement on 3DGS, FMGS [66], LangSplat [39] and Feature3DGS [64] also augment 3DGS with feature rendering. They learn to embed

the dense features from foundation models [36,40] into 3DGS such that the 3D scenes can be segmented by language queries. While the concurrent work collectively also achieves 3D reconstruction with the open-world segmentation ability, EgoLifter is the first to explicitly handle the dynamic objects that are commonly present in the real-world and especially in egocentric videos. We demonstrate this is a challenge in real-world scenarios and show improvements on it brought by EgoLifter.

#### 2.3 3D Reconstruction from Egocentric Videos

NeuralDiff [55] first approached the problem of training an egocentric radiance field reconstruction by decomposing NeRF into three branches, which capture ego actor, dynamic objects, and static background respectively as inductive biases. EPIC-Fields [53] propose an augmented benchmark using 3D reconstruction by augmenting the EPIC-Kitchen [6] dataset using neural reconstruction. They also provide comprehensive reconstruction evaluations of several baseline methods [10,29,55]. Recently, two datasets for egocentric perception, Aria Digital Twin (ADT) dataset [37] and Aria Everyday Activities (AEA) Dataset [28], have been released. Collected by Project Aria devices [8], both datasets feature egocentric video sequences with human actions and contain multimodal data streams and high-quality 3D information. ADT also provides extensive ground truth annotations using a motion capture system. Preliminary studies on egocentric 3D reconstruction have been conducted on these new datasets [28,48] and demonstrate the challenges posed by dynamic motion. In contrast, this paper tackles the challenges in egocentric 3D reconstruction and proposes to filter out transient objects in the videos. Compared to all existing work, we are the first work that holistically tackles the challenges in reconstruction and open-world scene understanding, and set up the quantitative benchmark to systematically evaluate performance in egocentric videos.

### 3 Method

#### 3.1 3D Gaussian Splatting with Feature Rendering

- 3D Gaussian Splatting (3DGS) [18] has shown state-of-the-art results in 3D reconstruction and novel view synthesis. However, the original design only reconstructs the color radiance in RGB space and is not able to capture the rich semantic information in a 3D scene. In EgoLifter, we augment 3DGS to also render a feature map of arbitrary dimension in a differentiable manner, which enables us to encode high-dimensional features in the learned 3D scenes and lift segmentation from 2D to 3D. These additional feature channels are used to learn object instance semantics in addition to photometric reconstruction.

Formally, 3DGS represents a 3D scene by a set of N colored 3D Gaussians S = {Θi|i = 1,··· ,N}, with location and shape represented by a center position pi ∈ R3, an anisotropic 3D covariance si ∈ R3 and a rotation quaternion qi ∈ R4.

Egocentric Video

Transient Mask

Rendered Image Rendered Feature map (PCA)

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

|[Figure 23]<br><br>[Figure 24]|
|---|

|[Figure 25]<br><br>[Figure 26]|
|---|

Transient Prediction Network

| |
|---|

| |
|---|

EgoLifter (Ours) EgoLifter-Static (Without Transient Prediction)

[Figure 27]

Image Segmentation Model (SAM)

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

|[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]|
|---|

[Figure 53]

[Figure 54]

|[Figure 55]<br><br>[Figure 56]|
|---|

|[Figure 57]<br><br>[Figure 58]|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

| |
|---|

| |
|---|

3D Gaussians with Features

Segmentation Masks

Rendered Image Rendered Feature map (PCA)

- Fig. 2: Naive 3D reconstruction from egocentric videos creates a lot of "floaters" in the reconstruction and leads to blurry rendered images and erroneous instance features (bottom right). EgoLifter tackles this problem using a transient prediction network, which predicts a probability mask of transient objects in the image and guides the reconstruction process. In this way, EgoLifter gets a much cleaner reconstruction of the static background in both RGB and feature space (top right), which in turn leads to better object decomposition of 3D scenes.

The radiance of each 3D Gaussian is described by an opacity parameter αi ∈ R and a color vector ci, parameterized by spherical harmonics (SH) coefficients. In EgoLifter, we additionally associate each 3D Gaussian with an extra feature vector f ∈ Rd, and thus the optimizable parameter set for i-th Gaussian is Θi = {pi,si,qi,αi,ci,fi}.

To train 3DGS for 3D reconstruction, a set of M observations {Ij,θj|j = 1,··· ,M} is used, where Ij is an RGB image and θj is the corresponding camera parameters. During the differentiable rendering process, all 3D Gaussians are splatted onto the 2D image plane according to θj and α-blended to get a rendered image ˆIj. Then the photometric loss is computed between the rendered image ˆIj and the corresponding ground truth RGB image Ij as

LRGB(Ij,ˆIj) = LMSE(Ij,ˆIj) =

∥Ij[u] − f(ˆIj[u])∥22, (1)

u∈Ω

where LMSE is the mean-squared-error (MSE) loss, Ω is set of all coordinates on the image and Ij[u] denotes the pixel value of Ij at coordinate u. f(·) is an image formation model that applies special properties of the camera (e.g. vignetting, radius of valid pixels) on the rendered image. By optimizing LRGB, the location, shape, and color parameters of 3D Gaussians are updated to reconstruct the geometry and appearance of the 3D scene. A density control mechanism is also used to split or prune 3D Gaussians during the training process [18].

In EgoLifter, we also implement the differentiable feature rendering pipeline similar to that of the RGB images, which renders to a 2D feature map Fˆ ∈

RH×W×d according to the camera information. During the training process, the feature vectors are supervised by segmentation output obtained from 2D images and jointly optimized with the location and color parameters of each Gaussian. We also include gradients of feature learning for the density control process in learning 3DGS. More details may be found in the supplementary material.

#### 3.2 Learning Instance Features by Contrastive Loss

Egocentric videos capture a huge number of different objects in everyday activities, and some of them may not exist in any 3D datasets for training. Therefore, egocentric 3D perception requires an ability to generalize to unseen categories (open-world) which we propose to achieve by lifting the output from 2D instance segmentation models. The key insight is that 2D instance masks from images of different views can be associated to form a consistent 3D object instance and that this can be done together with the 3D reconstruction process. Recent work has approached this problem using linear assignment [46], video object tracking [61], and incremental matching [12,26,49].

To achieve open-world 3D segmentation, we use f as instance features to capture the lifted segmentation and their similarity to indicate whether a set of Gaussians should be considered as the same object instance. Inspired by Contrastive Lift [2], we adopt supervised contrastive learning, which pulls the rendered features belonging to the same mask closer and pushes those of different masks further apart. Formally, given a training image Ij, we use a 2D segmentation model to extract a set of instance masks Mj = {Mkj|k = 1,··· ,mi} from Ij. The feature map Fˆj at the corresponding camera pose θj is then rendered, and the contrastive loss is computed over a set of pixel coordinates U, for which we use a uniformly sampled set of pixels U ⊂ Ω due to GPU memory constraint. The contrastive loss is formulated as

1 |U| u∈U

Lcontr(Fˆj,Mj) = −

′∈U+ exp(sim(Fˆj[u],Fˆj[u′];γ) u′∈U exp(sim(Fˆj[u],Fˆj[u′];γ)

log u

, (2)

where U+ is the set of pixels that belong to the same instance mask as u and Fˆj[u] denotes the feature vector of the Fˆj at coordinate u. We use a Gaussian RBF kernel as the similarity function, i.e. sim(f1,f2;γ) = exp(−γ∥f1 − f2∥22).

In the contrastive loss, pixels on the same instance mask are considered as positive pairs and will have similar features during training. Note that since the

- 2D segmentation model does not output consistent object instance IDs across different views, the contrastive loss is computed individually on each image. This weak supervision allows the model to maintain a flexible definition of object instances without hard assignments and is key to learning multi-view consistent instance features for 3D Gaussians that enables flexible open-world 3D segmentation.

#### 3.3 Transient Prediction for Egocentric 3D Reconstruction

Egocentric videos contain a lot of dynamic objects that cause many inconsistencies among 3D views. As we show in Fig. 2, the original 3DGS algorithm on the egocentric videos results in many floaters and harms the results of both reconstruction and feature learning. In EgoLifter, we propose to filter out transient phenomena in the egocentric 3D reconstruction, by predicting a transient probability mask from the input image, which is used to guide the 3DGS reconstruction process.

Specifically, we employ a transient prediction network G(Ij), which takes in the training image Ij and outputs a probability mask Pj ∈ RH×W whose value indicates the probability of each pixel being on a transient object. Then Pj is used to weigh the reconstruction loss during training, such that when a pixel is considered transient, it is filtered out in reconstruction. Therefore the reconstruction loss from Eq. (1) is adapted to

LRGB-w(Ij,ˆIj,Pj) =

(1 − Pj[u])∥Ij[u] − ˆIj[u]∥22, (3)

u∈Ω

where the pixels with lower transient probability will contribute more to the reconstruction loss. As most of the objects in egocentric videos remain static, we also apply an L-1 regularization loss on the predicted Pj as Lreg(Pj) =

p∈Pj |p|. This regularization also helps avoid the trivial solution where Pj equals one and all pixels are considered transient. The transient mask Pj is also used to guide contrastive learning for lifting instance segmentation, where the pixel set U is only sampled on pixels with the probability of being transient less than a threshold δ. As shown in Fig. 2 and Fig. 3, this transient filtering also helps learn cleaner instance features and thus better segmentation results.

In summary, the overall training loss on image Ij is a weighted sum as

L = λ1LRGB-w(Ij,ˆIj,Pj) + λ2Lcontr(Fˆj,Mj) + λ3Lreg(Pj), (4) with λ1, λ2 and λ3 as hyperparameters.

#### 3.4 Open-world Segmentation

After training, instance features f capture the similarities among 3D Gaussians, and can be used for open-world segmentation in two ways, query-based and clustering-based. In query-based open-world segmentation, one or few clicks on the object of interest are provided and a query feature vector is computed as the averaged features rendered at these pixels. Then a set of 2D pixels or a set of 3D Gaussians can be obtained by thresholding their Euclidean distances from the query feature, from which a 2D segmentation mask or a 3D bounding box can be estimated. In clustering-based segmentation, an HDBSCAN clustering algorithm [32] is performed to assign 3D Guassians into different groups, which gives a full decomposition of the 3D scene into a set of individual objects. In our experiments, query-based segmentation is used for quantitative evaluation, and clustering-based mainly for qualitative results.

### 4 Experiments

Implementation. We use a U-Net [43] with the pretrained MobileNet-v3 [14] backbone as the transient prediction network G. The input to G is first resized to 224 × 224 and then we resize its output back to the original resolution using bilinear interpolation. We use feature dimension d = 16, threshold δ = 0.5, temperature γ = 0.01, and loss weights λ1 = 1, λ2 = 0.1 and λ3 = 0.01. The 3DGS is trained using the Adam optimizer [21] with the same setting and the same density control schedule as in [18]. The transient prediction network is optimized by another Adam optimizer with an initial learning rate of 1 × 10−5. EgoLifter is agnostic to the specific 2D instance segmentation method, and we use the Segment Anything Model (SAM) [22] for its remarkable instance segmentation performance.

Datasets. We evaluate EgoLifter on the following egocentric datasets:

- – Aria Digital Twin (ADT) [37] provides 3D ground truth for objects paired with egocentric videos, which we used to evaluate EgoLifter quantitatively. ADT dataset contains 200 egocentric video sequences of daily activities, captured using Aria glasses. ADT also uses a high-quality simulator and motion capture devices for extensive ground truth annotations, including 3D object bounding boxes and 2D segmentation masks for all frames. ADT does not contain an off-the-shelf setting for scene reconstruction or open-world 3D segmentation. We create the evaluation benchmark using the GT 2D masks and 3D bounding boxes by reprocessing the 3D annotations. Note that only the RGB images are used during training, and for contrastive learning, we used the masks obtained by SAM [22].
- – Aria Everyday Activities (AEA) dataset [28] provides 143 egocentric videos of various daily activities performed by multiple wearers in five different indoor locations. Different from ADT, AEA contains more natural video activity recordings but does not offer 3D annotations. For each location, multiple sequences of different activities are captured at different times but aligned in the same 3D coordinate space. Different frames or recordings may observe the same local space at different time with various dynamic actions, which represent significant challenges in reconstruction. We group all daily videos in each location and run EgoLifter for each spatial environment. The longest aggregated video in one location (Location 2) contains 2.3 hours of video recording and a total of 170K RGB frames. The dataset demonstrates our method can not only tackle diverse dynamic activities, but also produce scene understanding at large scale in space and time.
- – Ego-Exo4D [11] dataset is a large and diverse dataset containing over one thousand hours of videos captured simultaneously by egocentric and exocentric cameras. Ego-Exo4D videos capture humans performing a wide range of activities. We qualitatively evaluate EgoLifter on the egocentric videos of Ego-Exo4D.

We use the same process for all Project Aria videos. Since Aria glasses use fisheye cameras, we undistort the captured images first before training. We use

Table 1: Quantitative evaluation of 2D instance segmentation (measured in mIoU) and novel view synthesis (measured in PNSR) on the ADT dataset. The evaluations are conducted on the frames in the novel subset of each scene.

Evaluation mIoU (In-view) mIoU (Cross-view) PSNR Object set Static Dynamic All Static Dynamic All Static Dynamic All

SAM [22] 54.51 32.77 50.69 - - - - - Gaussian Grouping [61] 35.68 30.76 34.81 23.79 11.33 21.58 21.29 14.99 19.97 EgoLifter-Static 55.67 39.61 52.86 51.29 18.67 45.49 21.37 15.32 20.16 EgoLifter-Deform 54.23 38.62 51.49 51.10 18.02 45.22 21.16 15.39 19.93 EgoLifter (Ours) 58.15 37.74 54.57 55.27 19.14 48.84 22.14 14.37 20.28

the image formation function f(·) in Eq. (1) to capture the vignetting effect and the radius of valid pixels, according to the specifications of the camera on Aria glasses. We use high-frequency 6DoF trajectories to acquire RGB camera poses and the semi-dense point clouds provided by each dataset through the Project Aria Machine Perception Services (MPS).

Baselines. We compare EgoLifter to the following baselines.

- – SAM [22] masks serve as input to EgoLifter. The comparison on segmentation between EgoLifter and SAM shows the benefits of multi-view fusion of 2D masks. As we will discuss in Sec. 4.1, SAM only allows prompts from the same image (in-view query), while EgoLifter enables segmentation prompts from different views (cross-view query) and 3D segmentation.
- – Gaussian Grouping [61] also lifts the 2D segmentation masks into 3D Gaussians. Instead of the contrastive loss, Gaussian Grouping uses a video object tracker to associate the masks from different views and employs a linear layer for identity classification. Gaussian Grouping does not handle the dynamic objects in 3D scenes.

Ablations. We further provide two variants of EgoLifter in particular to study the impact of reconstruction backbone.

- – EgoLifter-Static disabled the transient prediction network. A vanilla static 3DGS [18] is learned to reconstruct the scene. We use the same method to lift and segment 3D features.
- – EgoLifter-Deform uses a dynamic variant of 3DGS [60] instead of the transient prediction network to handle the dynamics in the scene. Similar to [60], EgoLifter-Deform learns a canonical 3DGS and a network to predict the location and shape of each canonical 3D Gaussian at different timestamps.

We also compare EgoLifter with NeRF-based methods [50,63] on the ADT dataset and evaluate EgoLifter on non-egocentric datasets [1,47]. Please see Appendix A3 for the results.

#### 4.1 Benchmark Setup on ADT

We use the ADT dataset [37] for the quantitative evaluation. We use 16 video sequences from ADT, and the frames of each sequence are split into seen and

novel subsets. The seen subset are used for training and validation, while the novel subset contains a chunk of consecutive frames separate from the seen subset and is only used for testing. The evaluation on the novel subset reflects the performance on novel views. The objects in each video sequence are also tagged dynamic and static according to whether they move. Each model is trained and evaluated on one video sequence separately. We evaluate the performance of the query-based open-world 2D instance segmentation and 3D instance detection tasks, as described in Sec. 3.4. For the Gaussian Grouping baseline [61], we use their learned identity encoding for extracting query features and computing similarity maps. Please refer to Appendix for more details of the evaluation settings, the exact sequence IDs and splits we used.

- Open-world 2D instance segmentation. We adopt two settings in terms of query sampling for 2D evaluation, namely in-view and cross-view. In both settings, a similarity map is computed between the query feature and the rendered feature image. The instance segmentation mask is then obtained by cutting off the similarity map using a threshold that maximizes the IoU with respect to the GT mask, which resembles the process of a human user adjusting the threshold to get the desired object. In the in-view setting, the query feature is sampled from one pixel on the rendered feature map in the same camera view. For a fair comparison, SAM [22] in this setup takes in the rendered images from the trained 3DGS and the same query pixel as the segmentation prompt. For each prompt, SAM predicts multiple instance masks at different scales, from which we also use the GT mask to pick one that maximizes the IoU for evaluation. The cross-view setting follows the prompt propagation evaluation used in the literature [3,34,41,63]. We randomly sample 5 pixels from the training images (in the seen subset) on the object, and their average feature is used as the query for segmentation on the novel subset. To summarize, the in-view setting evaluates how well features group into objects after being rendered into a feature map, and the cross-view setting evaluates how well the learned feature represents each object instance in 3D and how they generalize to novel views.
- Open-world 3D instance detection. For 3D evaluation, we use the same query feature obtained in the above cross-view setting. The similarity map is computed between the query feature and the learned 3D Gaussians, from which a subset of 3D Gaussians are obtained by thresholding, and a 3D bounding box is estimated based on their coordinates. The 3D detection performance is evaluated by the IoU between the estimated and the GT 3D bounding boxes. We also select the threshold that maximizes the IoU with the GT bounding box. We only evaluate the 3D static objects in each scene.

Novel view synthesis. We evaluate the synthesized frames in the novel subset using PSNR metric. We use "All" to indicate full-frame view synthesis. We also separately evaluate the pixels on dynamic and static regions using the provided the 2D ground truth dynamic motion mask.

#### 4.2 Quantitative Results on ADT

The quantitative results are reported in Tab. 1 and 2. As shown in Tab. 1, EgoLifter consistently outperforms all other baselines and variants in the reconstruction and segmentation of static object instances in novel views. Since transient objects are deliberately filtered out during training, EgoLifter has slightly worse performance on dynamic objects, However, the improvements on static objects outweigh the drops on transient ones in egocentric videos and EgoLifter still achieves the best overall results in all settings. Similarly, this trend also holds in 3D, and EgoLifter has the best 3D detection performance as shown in Tab. 2.

Table 2: 3D instance detection performance for the static objects in the ADT dataset.

Method mIoU Gaussian Grouping [61] 7.48

EgoLifter-Static 21.10 EgoLifter-Deform 20.58 EgoLifter (Ours) 23.11

#### 4.3 Qualitative Results on Diverse Egocentric Datasets

In Fig. 3, we visualize the qualitative results on several egocentric datasets [11, 28,37]. Please refer to project webpage for the videos rendered by EgoLifter with comparison to baselines. As shown in Fig. 3, without transient prediction, EgoLifter-Static creates 3D Gaussians to overfit the dynamic observations in some training views. However, since dynamic objects are not geometrically consistent and may be at a different location in other views, these 3D Gaussians become floaters that explain dynamics in a ghostly way, harming both the rendering and segmentation quality. In contrast, EgoLifter correctly identifies the dynamic objects in each image using transient prediction and filters them out in the reconstruction. The resulting cleaner reconstruction leads to better results in novel view synthesis and segmentation, as we have already seen quantitatively in Sec. 4.2. We also compare the qualitative results with Gaussian Grouping [61] in Fig. 4, from which we can see that Gaussian Grouping not only struggles with floaters associated with transient objects but also has a less clean feature map even on the static region. We hypothesize this is because our contrastive loss helps learn more cohesive identity features than the classification loss used in [61]. This also explains why EgoLifter-Static significantly outperforms Gaussian Grouping in segmentation metrics as shown in Tab. 1 and 2.

#### 4.4 3D Object Extraction and Scene Editing

Based on the features learned by EgoLifter, we can decompose a 3D scene into individual 3D objects, by querying or clustering over the feature space. Each extracted 3D object is represented by a set of 3D Gaussians which can be photorealistically rendered. In Fig. 5, we show the visualization of 3D objects extracted from a scene in the ADT dataset. This can further enable scene editing applications by adding, removing, or transforming these objects over the 3D space. In Fig. 1, we demonstrate one example of background recovery by removing all segmented 3D objects from the table.

EgoLifter-Static EgoLifter (Ours) GT Render Feature Trans. map Render Feature

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

- Fig. 3: RGB images and feature maps (colored by PCA) rendered by the EgoLifter Static baseline and EgoLifter. The predicted transient maps (Trans. map) from EgoLifter are also visualized, with red color indicating a high probability of being transient. Note that the baseline puts ghostly floaters on the region of transient objects, but EgoLifter filters them out and gives a cleaner reconstruction of both RGB images and feature maps. Rows 1-3 are from ADT, rows 4-5 from AEA, and rows 6-7 from Ego-Exo4D.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

GT Render from [61] Feature from [61] Our Render Our Feature

- Fig. 4: Rendered images and feature maps (visualised in PCA colors) by Gaussian Grouping [61] and EgoLifter (Ours).

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

[Figure 126]

Rendered Image and Features from EgoLifter Extracted 3D Objects

[Figure 127]

- Fig. 5: Individual 3D object can be extracted by querying or clustering over the 3D features from EgoLifter. Note object reconstructions are not perfect since each object might be partial observable in the egocentric videos rather than scanned intentionally.

### 5 Conclusion and Limitation

We present EgoLifter, a novel algorithm that simultaneously solves the 3D reconstruction and open-world segmentation problem for in-the-wild egocentric perception. By lifting the 2D segmentation into 3D Gaussian Splatting, EgoLifter achieves strong open-world 2D/3D segmentation performance with no 3D data annotation. To handle the rapid and sparse dynamics in egocentric videos, we employ a transient prediction network to filter out transient objects and get more accurate 3D reconstruction. EgoLifter is evaluated on several challenging egocentric datasets and outperforms other existing baselines. The representations obtained by EgoLifter can also be used for several downstream tasks like

- 3D object asset extraction and scene editing, showing great potential for personal wearable devices and AR/VR applications. Limitations: We observe the transient prediction module may mix the regions that are hard to reconstruct with transient objects. As shown in rows (4) and (5) of Fig. 3, the transient prediction module predicts a high probability for pixels on the windows, which have over-exposed pixels that are hard to be reconstructed from LDR images. In this case, EgoLifter learns to filter them out to improve reconstruction on that region. Besides, the performance of EgoLifter may also be dependent on the underlying 2D segmentation model. EgoLifter is not able to segment an object if the 2D model consistently fails on it. Potential Negative Impact: 3D object digitization for egocentric videos in the wild may pose a risk to privacy considerations. Ownership of digital object rights of physical objects is also a challenging and complex topic that will have to be addressed as AR/VR becomes more ubiquitous.

### Acknowledgments

The authors thank the Project Aria team for providing open-source support and Nickolas Charron for helping with the ADT dataset evaluation. The authors also thank Fangzhou Hong, Kevin QH Lin, Hyo Jin Kim, Lingni Ma, Lambert Mathias, Pierre Moulon, Richard Newcombe, Tianwei Shen, Nan Yang, and Wang Zhao for discussions and useful feedback over the course of this project.

### Appendix

- A1 Video Qualitative Results Please refer to videos on the project page 3, which contains:

- – Videos qualitative results of the multiple applications of EgoLifter (corresponding to Fig. 1).
- – Video qualitative results on the ADT dataset, comparing EgoLifter and its variants (corresponding to Fig. 3).
- – Video qualitative results on the ADT dataset, comparing with Gaussian Grouping [61] (corresponding to Fig. 4).
- – Video qualitative results on the AEA and Ego-Exo4D datasets. (corresponding to Fig. 3).
- – Demonstration video of the interactive visualization and segmentation system.

- A2 Experiment Details

- A2.1 Image Formation Model for Project Aria

Aria Glasses [8] use a fisheye camera, and thus recorded images have a fisheye distortion and vignette effect, but 3DGS uses a linear camera model and does not have a vignette effect. Therefore we account for these effects in training

- 3D Gaussian models using the image formation model f(·) in Eq. 1, such that not the raw rendered image but a processed one is used for loss computation. Specifically, we apply an image processing pipeline as shown in Fig. A.1. In the pipeline, the raw recorded images are first rectified to a pinhole camera model using projectaria_tools4, and then multiplied with a valid-pixel mask that removes the pixels that are too far from the image center. The rendered image from 3DGS is multiplied with a vignette mask and also the valid-pixel masks. Then the photometric losses are computed between the processed rendered image and the processed GT image during training. This pipeline models the camera model used in Aria glasses and leads to better 3D reconstruction. Empirically we found that without this pipeline, 3DGS will create a lot of floaters to account for the vignette effect in the reconstruction and significantly harm the results.

- 3 https://egolifter.github.io/
- 4 Link

Vignetting Mask

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Photometric

Undistortion

Loss

Rendered Image

Processed Rendered Image GT Image (distorted)

undistorted GT Image

[Figure 133]

Valid-pixel Mask

- Fig. A.1: Image processing pipeline during training. The symbol indicates elementwise multiplication.

#### A2.2 Additional Training Details

Due to the GPU memory constraint, we sampled at most |U| = 4096 pixels within the valid-pixel mask for computing the contrastive loss in Eq. 2. Note that for EgoLifter where the transient prediction is used, the samples are additionally constrained to be pixels with transient probability less than δ = 0.5.

For the segmentation masks generated by SAM, some masks may have overlapped with each other. In our experiments, we discarded the information about overlapping and simply overlaid all masks on the image space to get a one-hot segmentation for each pixel. While making use of these overlapping results leads to interesting applications like hierarchical 3D segmentation as shown in [20,63], this is beyond the scope of EgoLifter and we left this for future exploration. The images used for training are of resolution of 1408 × 1408 and segmentation masks from SAM are in the resolution of 512 × 512. Therefore, during training, two forward passes are performed. In the first pass, only the RGB image is rendered at the resolution of 1408×1408 and in the second, only the feature map is rendered at 512 × 512. The losses are computed separately from each pass and summed up for gradient computation. Note that the view-space gradients from both passes are also summed for deciding whether to split 3D Gaussians.

For optimization on the 3D Gaussian models, we adopt the same setting as used in the original implementation [18], in terms of parameters used in the optimizer and scheduler and density control process. The learning rate for the additional per-Gaussian feature vector fi is 0.0025, the same as that for updating color ci. All models are trained for 30,000 iterations on each scene in the ADT dataset, and for 100,000 iterations on scenes in the AEA and Ego-Exo4D

Table A.1: 2D instance segmentation results (measured in mIoU) and novel view synthesis results (measured in PNSR) on seen subsets in the ADT dataset.

Evaluation mIoU (In-view) mIoU (Cross-view) PSNR Object set Static Dynamic All Static Dynamic All Static Dynamic All

SAM [22] 62.74 52.48 61.00 - - - - - Gaussian Grouping [61] 40.86 42.24 41.09 32.26 26.23 31.24 27.97 19.13 25.53 EgoLifter-Static 64.34 57.71 63.21 62.20 35.39 57.64 27.65 19.60 25.64 EgoLifter-Deform 63.33 57.11 62.27 62.24 34.91 57.59 28.60 19.89 26.24 EgoLifter (Ours) 65.08 52.12 62.88 63.65 33.70 58.56 26.86 16.02 23.34

datasets, as these two datasets contain more frames in each scene. In the latter case, the learning rate scheduler and density control schedule are also proportionally extended.

#### A2.3 Timing

Using one NVIDIA A100 (40GB), training EgoLifter on one ADT sequence takes around 130 minutes (training vanilla 3DGS takes around 100 minutes). For a trained EgoLifter model, rendering both the RGB image and the instance feature map of 1408 × 1408 resolution runs at around 103 fps. If only RGB images are rendered, the speed goes to 158 fps. Note that we use a different implementation than the original 3DGS, where we made several changes like not caching images on GPU to enable training on large datasets, e.g. AEA and Ego-Exo4D.

#### A2.4 ADT Dataset Benchmark

Sequence selection Based on the 218 sequences in the full ADT datasets [37], we filter out the sequences that have too narrow baselines for 3D reconstruction (sequences with name starting with Lite_release_recognition) or do not have segmentation annotation on human bodies. From the rest of the sequences, we select 16 sequences for evaluation, where 6 of them contain recordings of Aria glasses from two human users in the scene (sequences with multiskeleton in the name), and the rest 10 only have recordings from one user, although there may be multiple two persons in the scene (sequences with multiuser in the name). The names of the selected sequences are listed as follows:

- Apartment_release_multiskeleton_party_seq121
- Apartment_release_multiskeleton_party_seq122
- Apartment_release_multiskeleton_party_seq123

- Apartment_release_multiskeleton_party_seq125
- Apartment_release_multiskeleton_party_seq126
- Apartment_release_multiskeleton_party_seq127 Apartment_release_multiuser_cook_seq114 Apartment_release_multiuser_meal_seq140 Apartment_release_multiuser_cook_seq143

Apartment_release_multiuser_party_seq140 Apartment_release_multiuser_clean_seq116 Apartment_release_multiuser_meal_seq132 Apartment_release_work_skeleton_seq131 Apartment_release_work_skeleton_seq140 Apartment_release_meal_skeleton_seq136 Apartment_release_decoration_skeleton_seq137

#### A2.5 Comparison with NeRF

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

- Fig. A.2: Qualitative results on ADT datasets. From left to right: GT image; Render by Nerfacto; Render by EgoLifter.

Subset Splitting For sequences that only have a recording from one pair of Aria glasses, the first 4/5 of the video is considered as seen views and the rest are considered as novel ones. For sequences that have videos from two pairs, the video from one pair is considered as seen views and the other is considered as novel views. During training, every 1 out of 5 consecutive frames in the seen views

Method Nerfacto EgoLifter PSNR (all) 17.22 20.28

Table A.2: Comparison to Nerfacto on the ADT dataset.

Method INGP-Big M-NeRF360 3DGS EgoLifter PSNR 25.59 27.69 27.21 27.26

Table A.3: Quantitative comparison on the MipNeRF 360 dataset.

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

- Fig. A.3: Qualitative results on MipNeRF 360. From left to right: GT image; EgoLifter RGB render; EgoLifter feature map (PCA).

are used for validation the remaining 4 are used for training. The entire novel subset is hidden from training and solely used for evaluation. For evaluation on 2D instance segmentation, we uniformly sampled at most 200 frames from each subset for fast inference. The objects in each video sequence are also split into dynamic and static subsets, according to whether their GT object positions have changed by over 2cm over the duration of each recording. Humans are always considered dynamic objects.

### A3 Additional Results

#### A3.1 Results on ADT Seen Subset

For completeness, we also report the 2D instance segmentation and photometric results on the seen subset of ADT in Tab. A.1. Note that the frames used for evaluation in the seen subset are closer to those for training, and therefore these results mostly reflect how well the models overfit the training viewpoints in each scene, rather than generalize to novel views. As we can see from Tab. A.1, EgoLifter outperforms the baselines in segmenting static objects using both in-view and cross-view queries. When both static and dynamic objects are considered (the “All” column), EgoLifter still achieves the best results in cross-view, which is a harder setting for open-world segmentation. EgoLifter also has the second place in the in-view setting.

Method Office0 Office1 Office2 Office3 Office4 Room0 Room1 Room2 Mean

|MVSeg [34] SA3D [3] OmniSeg3D [63]|31.4 40.4 30.4 30.5 25.4 31.1 40.7 29.2 84.4 77.0 88.9 84.4 82.6 77.6 79.8 89.2 83.9 85.3 89.0 87.2 78.3 83.0 79.4 88.9<br><br>|32.4 83.0 84.4<br><br>|
|---|---|---|
|EgoLifter (Ours)|82.9 78.4 85.1 84.1 80.0 77.0 85.4 84.3<br><br>|82.1|

Table A.4: Instance Segmentation results (mean IoU) on Replica dataset.

[Figure 146]

[Figure 147]

[Figure 148]

- Fig. A.4: Qualitative result on Replica datasets. From left to right: GT image; OmniSeg3D feature map; EgoLifter feature map

#### A3.2 NeRF on ADT Dataset

In Tab. A.2 and Fig. A.2, we compare EgoLifter with the (default) nerfacto model in Nerfstudio [50] on the ADT dataset. As we can see from Fig. A.2, although Nerfacto uses per-image appearance embeddings to filter out transient phenomena in reconstruction, it still fails on challenging egocentric datasets like ADT and results in many floaters in the rendering. Quantitatively, EgoLifter also outperforms as shown in Fig. A.2.

#### A3.3 Non-egocentric public benchmarks

Non-egocentric benchmarks (Replica, ScanNet, MipNeRF 360) use careful handheld scanning motions and lack dynamic phenomena. Therefore, they do not reflect the full capability of EgoLifter. We evaluate EgoLifter on the MipNeRF 360 dataset [1] in Tab. A.3 and Fig. A.3, where we use EgoLifter-static variant as the scenes are all static. Due to the lack of GT segmentation masks, we provide qualitative results on learned instance features in Fig. A.3. As shown in Tab. A.3 and Fig. A.3, EgoLifter has a similar PSNR as the original 3DGS and learns clean instance features that distinguish different instances.

We also test EgoLifter on Replica [47] and compare to OmniSeg3D [63], a recent feature lifting method based on NeRF representation and contrastive learning [2]. We evaluate the instance segmentation task using the multi-view mask propagation protocol [3,34,63], where the GT mask from one view is used for computing reference instance features and masks on other view are computed based on the feature distance from the reference ones. We follow the evaluation protocol in [63] and use Eq. (11) in [63] for computing the similarity scores. Similar to the experiments on MipNeRF 360, we used EgoLifter-static as there is no dynamic content in Replica scenes.

We report the quantitative results (in mIoU) in Tab. A.4 and a qualitative example in Fig. A.4. From Tab. A.4, we can see that EgoLifter has similar performance with the state-of-the-art NeRF-based segmentation methods [3,63] on the non-egocentric Replica dataset. From the qualitative example in Fig. A.4, we can see that EgoLifter also results in clean and sharp feature boundaries on Replica as contemporary work OmniSeg3D [63], which distinguish different object instances and even the parts within each object.

### A4 Additional Discussion on Limitations

Due to form factor and power constraints, egocentric videos are often captured with more challenges. Due to rapid head motion and lighting condition changes in the egocentric videos, the images contain significant motion blur that causes challenges in recovering sharp reconstructions from them. This explains in part the blurry results shown in some of the reconstruction results by EgoLifter. We leave how to improve the reconstruction quality from egocentric videos for future work.

### References

- 1. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mipnerf 360: Unbounded anti-aliased neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5470– 5479 (2022)
- 2. Bhalgat, Y., Laina, I., Henriques, J.F., Zisserman, A., Vedaldi, A.: Contrastive lift: 3d object instance segmentation by slow-fast contrastive fusion. In: Advances in Neural Information Processing Systems (2023)
- 3. Cen, J., Zhou, Z., Fang, J., Shen, W., Xie, L., Jiang, D., Zhang, X., Tian, Q., et al.: Segment anything in 3d with nerfs. Advances in Neural Information Processing Systems 36, 25971–25990 (2023)
- 4. Chen, Z., Wang, F., Wang, Y., Liu, H.: Text-to-3d using gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21401–21412 (2024)
- 5. Crabtree, A., Tolmie, P.: A day in the life of things in the home. In: Proceedings of the 19th ACM Conference on Computer-Supported Cooperative Work & Social Computing. pp. 1738–1750 (2016)
- 6. Damen, D., Doughty, H., Farinella, G.M., Fidler, S., Furnari, A., Kazakos, E., Moltisanti, D., Munro, J., Perrett, T., Price, W., et al.: Scaling egocentric vision: The epic-kitchens dataset. In: Proceedings of the European conference on computer vision (ECCV). pp. 720–736 (2018)
- 7. Duan, Y., Wei, F., Dai, Q., He, Y., Chen, W., Chen, B.: 4d-rotor gaussian splatting: Towards efficient novel view synthesis for dynamic scenes. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–11 (2024)
- 8. Engel, J., Somasundaram, K., Goesele, M., Sun, A., Gamino, A., Turner, A., Talattof, A., Yuan, A., Souti, B., Meredith, B., Peng, C., Sweeney, C., Wilson, C., Barnes, D., DeTone, D., Caruso, D., Valleroy, D., Ginjupalli, D., Frost, D., Miller, E., Mueggler, E., Oleinik, E., Zhang, F., Somasundaram, G., Solaira, G., Lanaras,

H., Howard-Jenkins, H., Tang, H., Kim, H.J., Rivera, J., Luo, J., Dong, J., Straub, J., Bailey, K., Eckenhoff, K., Ma, L., Pesqueira, L., Schwesinger, M., Monge, M., Yang, N., Charron, N., Raina, N., Parkhi, O., Borschowa, P., Moulon, P., Gupta, P., Mur-Artal, R., Pennington, R., Kulkarni, S., Miglani, S., Gondi, S., Solanki,

- S., Diener, S., Cheng, S., Green, S., Saarinen, S., Patra, S., Mourikis, T., Whelan,
- T., Singh, T., Balntas, V., Baiyya, V., Dreewes, W., Pan, X., Lou, Y., Zhao, Y., Mansour, Y., Zou, Y., Lv, Z., Wang, Z., Yan, M., Ren, C., Nardi, R.D., Newcombe, R.: Project aria: A new tool for egocentric multi-modal ai research. arXiv preprint arXiv:2308.13561 (2023)

- 9. Engelmann, F., Manhardt, F., Niemeyer, M., Tateno, K., Pollefeys, M., Tombari, F.: OpenNerf: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views. In: International Conference on Learning Representations (2024)
- 10. Gao, H., Li, R., Tulsiani, S., Russell, B., Kanazawa, A.: Monocular dynamic view synthesis: A reality check. Advances in Neural Information Processing Systems 35, 33768–33780 (2022)
- 11. Grauman, K., Westbury, A., Torresani, L., Kitani, K., Malik, J., Afouras, T., Ashutosh, K., Baiyya, V., Bansal, S., Boote, B., et al.: Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19383–19400 (2024)
- 12. Gu, Q., Kuwajerwala, A., Morin, S., Jatavallabhula, K.M., Sen, B., Agarwal, A., Rivera, C., Paul, W., Ellis, K., Chellappa, R., et al.: Conceptgraphs: Openvocabulary 3d scene graphs for perception and planning. In: IEEE International Conference on Robotics and Automation (2023)
- 13. Gu, Q., Okorn, B., Held, D.: Ossid: online self-supervised instance detection by (and for) pose estimation. IEEE Robotics and Automation Letters 7(2), 3022– 3029 (2022)
- 14. Howard, A., Sandler, M., Chu, G., Chen, L.C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., et al.: Searching for mobilenetv3. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1314–1324 (2019)
- 15. Huang, C., Mees, O., Zeng, A., Burgard, W.: Audio visual language maps for robot navigation. In: Proceedings of the International Symposium on Experimental Robotics (ISER). Chiang Mai, Thailand (2023)
- 16. Jatavallabhula, K.M., Kuwajerwala, A., Gu, Q., Omama, M., Iyer, G., Saryazdi, S., Chen, T., Maalouf, A., Li, S., Keetha, N.V., Tewari, A., Tenenbaum, J.B., de Melo, C.M., Krishna, K.M., Paull, L., Shkurti, F., Torralba, A.: ConceptFusion: Open-set multimodal 3d mapping. In: Robotics: Science and Systems (2023)
- 17. Keetha, N., Karhade, J., Jatavallabhula, K.M., Yang, G., Scherer, S., Ramanan, D., Luiten, J.: Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21357–21366 (2024)
- 18. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (ToG) 42(4), 1–14 (2023)
- 19. Kerr, J., Kim, C.M., Goldberg, K., Kanazawa, A., Tancik, M.: LERF: Language embedded radiance fields. In: International Conference on Computer Vision (ICCV) (2023)
- 20. Kim, C.M., Wu, M., Kerr, J., Goldberg, K., Tancik, M., Kanazawa, A.: Garfield: Group anything with radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21530–21539 (2024)

- 21. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: International Conference on Learning Representations (2015)
- 22. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4015–4026

(2023)

- 23. Kobayashi, S., Matsumoto, E., Sitzmann, V.: Decomposing nerf for editing via feature field distillation. Advances in Neural Information Processing Systems 35, 23311–23330 (2022)
- 24. Liu, K., Zhan, F., Zhang, J., Xu, M., Yu, Y., Saddik, A.E., Theobalt, C., Xing, E., Lu, S.: 3d open-vocabulary segmentation with foundation models. arXiv preprint arXiv:2305.14093 (2023)
- 25. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023)
- 26. Lu, S., Chang, H., Jing, E.P., Boularias, A., Bekris, K.: Ovir-3d: Open-vocabulary 3d instance retrieval without training on 3d data. In: Conference on Robot Learning. pp. 1610–1620. PMLR (2023)
- 27. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In: 2024 International Conference on 3D Vision (3DV). pp. 800–809. IEEE (2024)
- 28. Lv, Z., Charron, N., Moulon, P., Gamino, A., Peng, C., Sweeney, C., Miller, E., Tang, H., Meissner, J., Dong, J., Somasundaram, K., Pesqueira, L., Schwesinger, M., Parkhi, O.M., Gu, Q., Nardi, R.D., Cheng, S., Saarinen, S., Baiyya, V., Zou, Y., Newcombe, R.A., Engel, J.J., Pan, X., Ren, C.: Aria everyday activities dataset. arXiv preprint arXiv:2402.13349 (2024)
- 29. Martin-Brualla, R., Radwan, N., Sajjadi, M.S., Barron, J.T., Dosovitskiy, A., Duckworth, D.: Nerf in the wild: Neural radiance fields for unconstrained photo collections. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7210–7219 (2021)
- 30. Matsuki, H., Murai, R., Kelly, P.H., Davison, A.J.: Gaussian splatting slam. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18039–18048 (2024)
- 31. Mazur, K., Sucar, E., Davison, A.J.: Feature-realistic neural fusion for real-time, open set scene understanding. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). pp. 8201–8207. IEEE (2023)
- 32. McInnes, L., Healy, J., Astels, S.: hdbscan: Hierarchical density based clustering. The Journal of Open Source Software 2(11), 205 (2017)
- 33. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021)
- 34. Mirzaei, A., Aumentado-Armstrong, T., Derpanis, K.G., Kelly, J., Brubaker, M.A., Gilitschenski, I., Levinshtein, A.: Spin-nerf: Multiview segmentation and perceptual inpainting with neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20669–20679 (2023)
- 35. OpenAI: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 36. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

- 37. Pan, X., Charron, N., Yang, Y., Peters, S., Whelan, T., Kong, C., Parkhi, O., Newcombe, R., Ren, Y.C.: Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20133–20143 (2023)
- 38. Peng, S., Genova, K., Jiang, C., Tagliasacchi, A., Pollefeys, M., Funkhouser, T., et al.: Openscene: 3d scene understanding with open vocabularies. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 815–824 (2023)
- 39. Qin, M., Li, W., Zhou, J., Wang, H., Pfister, H.: Langsplat: 3d language gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20051–20060 (2024)
- 40. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)
- 41. Ren, Z., Agarwala, A., Russell, B., Schwing, A.G., Wang, O.: Neural volumetric object selection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6133–6142 (2022)
- 42. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 43. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. pp. 234–241. Springer (2015)
- 44. Shafiullah, N.M.M., Paxton, C., Pinto, L., Chintala, S., Szlam, A.: Clip-fields: Weakly supervised semantic fields for robotic memory. In: Bekris, K.E., Hauser, K., Herbert, S.L., Yu, J. (eds.) Robotics: Science and Systems (2023)
- 45. Shen, W., Yang, G., Yu, A., Wong, J., Kaelbling, L.P., Isola, P.: Distilled feature fields enable few-shot language-guided manipulation. In: 7th Annual Conference on Robot Learning (2023)
- 46. Siddiqui, Y., Porzi, L., Bulò, S.R., Müller, N., Nießner, M., Dai, A., Kontschieder, P.: Panoptic lifting for 3d scene understanding with neural fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9043–9052 (2023)
- 47. Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J., MurArtal, R., Ren, C., Verma, S., Clarkson, A., Yan, M., Budge, B., Yan, Y., Pan, X., Yon, J., Zou, Y., Leon, K., Carter, N., Briales, J., Gillingham, T., Mueggler, E., Pesqueira, L., Savva, M., Batra, D., Strasdat, H.M., Nardi, R.D., Goesele, M., Lovegrove, S., Newcombe, R.: The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797 (2019)
- 48. Sun, J., Qiu, J., Zheng, C., Tucker, J., Yu, J., Schwager, M.: Aria-nerf: Multimodal egocentric view synthesis. arXiv preprint arXiv:2311.06455 (2023)
- 49. Takmaz, A., Fedele, E., Sumner, R.W., Pollefeys, M., Tombari, F., Engelmann, F.: OpenMask3D: Open-Vocabulary 3D Instance Segmentation. In: Advances in Neural Information Processing Systems (NeurIPS) (2023)
- 50. Tancik, M., Weber, E., Ng, E., Li, R., Yi, B., Wang, T., Kristoffersen, A., Austin, J., Salahi, K., Ahuja, A., et al.: Nerfstudio: A modular framework for neural radiance field development. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–12

(2023)

- 51. Tang, J., Ren, J., Zhou, H., Liu, Z., Zeng, G.: Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023)
- 52. Tsagkas, N., Mac Aodha, O., Lu, C.X.: Vl-fields: Towards language-grounded neural implicit spatial representations. arXiv preprint arXiv:2305.12427 (2023)
- 53. Tscherlnezki, V., Darkhalil, A., Zhu, Z., Fouhey, D., Laina, I., Larlus, D., Damen, D., Vedaldi, A.: Epic fields: Marrying 3d geometry and video understanding. Advances in Neural Information Processing Systems 36 (2024)
- 54. Tschernezki, V., Laina, I., Larlus, D., Vedaldi, A.: Neural feature fusion fields: 3d distillation of self-supervised 2d image representations. In: International Conference on 3D Vision (3DV). IEEE (2022)
- 55. Tschernezki, V., Larlus, D., Vedaldi, A.: Neuraldiff: Segmenting 3d objects that move in egocentric videos. In: 2021 International Conference on 3D Vision (3DV). pp. 910–919. IEEE (2021)
- 56. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20310–20320 (2024)
- 57. Yan, C., Qu, D., Xu, D., Zhao, B., Wang, Z., Wang, D., Li, X.: Gs-slam: Dense visual slam with 3d gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19595–19604 (2024)
- 58. Yan, Y., Lin, H., Zhou, C., Wang, W., Sun, H., Zhan, K., Lang, X., Zhou, X., Peng, S.: Street gaussians for modeling dynamic urban scenes. arXiv preprint arXiv:2401.01339 (2024)
- 59. Yang, Z., Yang, H., Pan, Z., Zhu, X., Zhang, L.: Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642 (2023)
- 60. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20331– 20341 (2024)
- 61. Ye, M., Danelljan, M., Yu, F., Ke, L.: Gaussian grouping: Segment and edit anything in 3d scenes. arXiv preprint arXiv:2312.00732 (2023)
- 62. Yi, T., Fang, J., Wang, J., Wu, G., Xie, L., Zhang, X., Liu, W., Tian, Q., Wang, X.: Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)
- 63. Ying, H., Yin, Y., Zhang, J., Wang, F., Yu, T., Huang, R., Fang, L.: Omniseg3d: Omniversal 3d segmentation via hierarchical contrastive learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20612–20622 (2024)
- 64. Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S., Wang, Z., Kadambi, A.: Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21676–21685 (2024)
- 65. Zhou, X., Lin, Z., Shan, X., Wang, Y., Sun, D., Yang, M.H.: Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21634–21643 (2024)
- 66. Zuo, X., Samangouei, P., Zhou, Y., Di, Y., Li, M.: Fmgs: Foundation model embedded 3d gaussian splatting for holistic 3d scene understanding. arXiv preprint arXiv:2401.01970 (2024)

