## UniGeo: Taming Video Diffusion for Unified Consistent Geometry Estimation

Yang-Tian Sun1 Xin Yu1 Zehuan Huang2 Yi-Hua Huang1

Yuan-Chen Guo3 Ziyi Yang1 Yan-Pei Cao3 Xiaojuan Qi1† 1The University of Hong Kong 2Beihang University 3VAST

arXiv:2505.24521v1[cs.CV]30May2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

(a) (b)

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

# . . .

[Figure 14]

[Figure 15]

[Figure 16]

| |
|---|

[Figure 17]

[Figure 18]

[Figure 19]

| |
|---|

| |
|---|

| |
|---|

- View #1
- View #2

| |
|---|

| |
|---|

[Figure 20]

| |
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

[Figure 27]

[Figure 28]

| |
|---|

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

| |
|---|

- View #1
- View #2 Frame #1 Frame #2 Frame #3

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

[Figure 33]

Figure 1. UniGeo utilizes video diffusion models to jointly estimate geometric properties—such as surface normals and coordinatesfrom either multi-view images (a) or video sequences (b). Rather than predicting within the local camera coordinate system of each frame, UniGeo infers geometric attributes in a unified global reference frame. Such design facilitates consistent estimation across frames for patches corresponding to the same 3D region by effectively leveraging inter-frame patch correspondences embedded in the video prior. Moreover, the estimated properties can be seamlessly integrated into downstream tasks such as 3D reconstruction. Project page: https://sunyangtian.github.io/UniGeo-web/

### Abstract

Recently, methods leveraging diffusion model priors to assist monocular geometric estimation (e.g., depth and normal) have gained significant attention due to their strong generalization ability. However, most existing works focus on estimating geometric properties within the camera coordinate system of individual video frames, neglecting the inherent ability of diffusion models to determine inter-frame correspondence. In this work, we demonstrate that, through appropriate design and fine-tuning, the intrinsic consistency of video generation models can be effectively harnessed for consistent geometric estimation. Specifically, we 1) select geometric attributes in the global coordinate system that share the same correspondence with video frames as the prediction targets, 2) introduce a novel and efficient conditioning method by reusing positional encodings,

and 3) enhance performance through joint training on multiple geometric attributes that share the same correspondence. Our results achieve superior performance in predicting global geometric attributes in videos and can be directly applied to reconstruction tasks. Even when trained solely on static video data, our approach exhibits the potential to generalize to dynamic video scenes.

### 1. Introduction

Estimating 3D geometric information, such as depth and surface normals, from RGB input frames is a fundamental task in computer vision with applications spanning VR/AR, robotics, and autonomous driving. Recently, single-image 3D geometry estimation has attracted significant attention [16, 26, 37, 66, 67, 71]. Approaches like Marigold [26]

and Geowizard [16] have demonstrated that diffusion-based image generators, when fine-tuned, can achieve remarkable performance in depth and normal prediction tasks. These findings suggest that priors learned by image generation models from large-scale datasets can enhance the accuracy and generalizability of geometric estimations.

However, directly applying image-based geometric estimation methods to videos in a frame-wise manner often leads to noticeable inconsistencies. To mitigate this issue, recent works [24, 48] have explored leveraging consistency priors from video diffusion models for depth estimation, treating video frames as conditioning inputs while predicting depth across frames in camera coordinates as the output. Despite these efforts, the consistency required for geometric properties such as depth and surface normals differs fundamentally from that of RGB video frames. For instance, video priors typically enforce appearance to be similar for the same object across frames, whereas its depth and normal vary according to camera motion (see Fig. 2 (c)). This discrepancy can lead to inaccurate geometric predictions. Furthermore, RGB conditioning is introduced into video diffusion models via channel concatenation, altering the input format compared to the pretrained model. This necessitates architectural modifications and is hard to fully exploit the potential of video diffusion priors.

In this paper, we introduce UniGeo, a unified framework that reformulates video-based geometry estimation tasks– including global position and surface normals– as a video generation problem. Our key insight is grounded in the discovery that pre-trained video generation models inherently possess the capability to extract inter-frame consistency, as can be visualized through attention weights across tokens, illustrated in Fig. 2(a). Such correspondence motivates us to repurpose a pretrained video diffusion model for consistent video geometry estimation.

First, to better exploit consistency priors, we propose representing geometric properties within a shared global coordinate system. This approach naturally aligns geometric correspondences across frames, mirroring the consistency in RGB videos (see Fig. 2(c)). In contrast, prior methods [10, 24, 48] estimate geometry in camera-centric coordinates, which inherently introduces inconsistencies.

Second, instead of stacking RGB inputs in the channel dimension as conditions– an approach that misaligns with the pretrained video diffusion model– we propose treating RGB frames as additional inputs within a unified video sequence. Specifically, we organize them alongside the noised geometry sequence, enabling direct adaptation of video diffusion models without architectural modifications (refer to Fig. 3). Then, motivated by the observation that attention weights between tokens naturally capture inter-frame correspondences, with these weights strongly influenced by token positional embeddings (see Fig. 2), we propose a novel

[Figure 34]

Figure 2. Our key insights lie in (a) pre-trained video diffusion models capture accurate inter-frame correspondence (the same patches in different frames highlight in the attention maps), (b) the correspondence can be specified by applying identical positional encodings onto different frames, and (c) geometric properties within a shared global coordinate system naturally exhibit alignment across frames.

shared positional encoding strategy that reuses positional embeddings from images and applying them to geometric properties. This achieves precise conditioning from images to geometric properties and effectively harnesses the pretrained model’s inter-token correspondence learning for improved geometry estimation.

Finally, to effectively utilize available training datasets for learning generalized models, we explore training a single network to predict multiple geometric attributes simultaneously. Our novel formulation enables these tasks to share the same learned correspondences, allowing them to mutually reinforce each other. Surprisingly, experimental results demonstrate that this multi-task approach not only offers the added advantage of inferring multiple attributes within a unified model but also outperforms individually trained networks for specific tasks.

To the best of our knowledge, our work is the first capable of simultaneously predicting multiple geometric attributes (e.g., radius, normals) from video data, ensuring global consistency suitable for direct reconstruction tasks (see Fig. 1). Compared to image-based methods, our approach achieves superior performance without additional camera information (Table 1), and delivers reconstruction quality comparable to models trained on large-scale datasets (Table 2). Notably, despite being trained exclusively on static data, our model benefits from video diffusion priors, enabling robust generalization to certain dynamic scenes (Fig. 7). In summary, our contributions are:

- • We propose a unified formulation of video-based geometry estimation as a video generation task, enabling the direct use of pretrained video diffusion models to achieve consistent predictions across frames.
- • We introduce global coordinate representation and a novel RGB conditioning method with a shared positional encoding strategy, which allow pretrained video diffusion models to transfer learned consistency priors without requiring architectural modifications.

- • We explore a multi-task learning approach that harnesses shared knowledge across tasks, enabling a unified model to simultaneously predict multiple geometric attributes from videos.
- • We demonstrate that our approach improves consistency and accuracy across geometric tasks and achieves competitive performance compared to state-of-the-art methods on geometry prediction and reconstruction.

### 2. Related Work

#### 2.1. Monocular Geometry Estimation

Early works [14, 49] used CNNs to estimate depth from annotated datasets [18, 51]. Depth and normal were soon jointly predicted due to their interdependence [13, 30, 39, 40]. Early works’ reliance on small datasets hindered generalization to new domains. Larger datasets proved critical for improving generalization [11, 12, 63]. MiDas [41] introduced an affine-invariant loss for depth estimation, enabling effective training across diverse datasets. MoGe [60] further improved the loss design for coordinate prediction. Enhanced strategies like probabilistic modeling [5] and iterative refinement [4] also improved normal estimation.

With larger annotated datasets, powerful architectures became crucial. Vision Transformers (ViTs) were applied to depth [37, 42] and normal estimation [4]. Depth Anything (DA) [66, 67] showed that ViTs trained with synthetic data can preserve depth details. Diffusion models [53] emerged as scalable architectures for image generation [21, 45, 54] and proved effective for geometry estimation [65]. Marigold [27] fine-tuned SD’s U-Net [45] for high-quality depth estimation. DepthFM [19] improved efficiency by reducing sampling steps via flow matching. GeoWizard [16] and DMP [29] utilized diffusion priors for depth and normal estimation, while StableNormal [69] refined normals through iterative diffusion. GenPercept [65] analyzed pre-trained diffusion models, offering insights for advancing monocular diffusion-based perception.

#### 2.2. Video / Multi-view Geometry Estimation

DUSt3R [62] introduces a dual ViT architecture to predict dense geometry from image pairs. Subsequent works extend DUSt3R to handle multi-view images or videos using techniques such as spatial memory [58], multi-view fusion [56], zero-convolution [34], and recurrent neural networks [59]. To enhance DUSt3R for dynamic scenes, MonST3R [72] separates the supervision of dynamic foreground and static background. Stereo4D [25] leverages stereo videos to annotate 3D tracked points and trains a time-dependent DUSt3R. Several works extend DA to video geometry. Prompt Depth Anything [33] uses LiDAR-based low-resolution depth maps as prompts for accurate video depth estimation. Video Depth Anything [9] adds temporal

layers to DA for relative video depth prediction.

Diffusion models also demonstrate exceptional performance in video generation. Commercial products such as SORA [8], Pika [3], Keling [2], and Hailuo [1] have revolutionized media creation. Open-source projects like Stable Video Diffusion (SVD) [7] and CogVideo [22] leverage Unet [47] with temporal attention to extend capabilities from image generation to video. With the DiT [35] architecture showcasing superior scalability, implementations like HunyuanVideo [28] and CogVideoX [68] have achieved remarkable results. Building on the success of video diffusion models, video geometry estimation methods have emerged by leveraging rich learned priors. DepthCrafter [24] predicts video depth using pre-trained image-to-video priors, while ChronoDepth [48] refines depth estimation with a fine-tuned SVD. Finetuning a video diffusion model to predict geometry attributes with diverse supervision modalities could scale to larger datasets, which are underexplored.

### 3. Preliminaries

Diffusion Models. Diffusion models [20] can model a specific data distribution through an iterative denoising process. Specifically, Gaussian noise at different levels t ∈ {1,...,T} is progressively added to the data point x0 in the forward process, generating a noisy sample sequence {xt}Tt=1, formulated as

xt = √α¯tx0 + √1 − α¯tϵ, (1)

where ϵ ∈ N(0,I), α¯t := ts=1(1 − βs), and β1,...,βT is the variance schedule of a process with T steps. The de-

noising model ϵθ(·) parameterized with parameters θ aims to gradually reverse this process by modeling the probability pθ(xt−1|xt).

During the training phase, the model takes noisy data sample xt and timestep t as input, and predicts the noise ϵˆ = ϵθ(xt,t). Paramters θ is updated by minimizing the following objective function

0,ϵ∼N(0,I),t∼U(T)||ϵ − ϵˆ||22. (2)

L = Ex

At the inference phase, x0 is obtained by iteratively denoising a sampled Gaussian noise.

To reduce the computational cost of high-resolution inputs, Latent Diffusion Model [44] (LDM) is often adopted by using a pre-trained VAE to encode the data into a latent space for probability modeling.

Video Diffusion Models. Given an RGB video of shape H × W × F × 3, it is first compressed into the latent space using a pre-trained VAE encoder, obtaining a latent representation of shape h × w × f × c. Typically, diffusion models are implemented using a U-Net architecture [46]. Recently, Diffusion Transformers (DiT) [36] have demonstrated significant potential due to their superior generation

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

- Figure 3. Method overview. Our method targets at predicting geometric properties that are defined in the global coordinate system, where “radius” represents the distance from a 3D points to the origin (left). We efficiently adapt a pre-trained video diffusion model that inherently encodes inter-frame correspondence into a consistent geometry estimation model (right), where we process both rgb sequence and geometry sequence through our proposed SPE-Transformer.

high correlation with color measurements—for example, the same point in space consistently appears with similar colors across different video frames. This aligns well with the consistency priors inherent in video generation models.

quality and greater flexibility. The DiT architecture applies a patchify operation to video latent representations, converting them into tokens, which are then concatenated into a long sequence for denoising. Our approach is based on the DiT architecture video diffusion [68].

Therefore, in our design, we directly predict geometric properties such as position and normal in a global coordinate system across all frames, which is proven to achieve more accurate results compared to predicting in their own coordinate systems in Sec 5.5. Additionally, our approach offers the advantage that the predicted global results can be directly used for reconstruction without requiring camera information as input.

### 4. Method

Our primary goal is to efficiently adapt a pre-trained video diffusion model into a video-based consistent geometry estimation model. The key insight is that pre-trained video generation models inherently encode inter-frame correspondence, which can be leveraged for consistent video geometric property estimation through appropriate design. In Section 4.1, we propose predicting geometric properties consistent with video frames in a global coordinate system. In Section 4.2, we propose a unified joint-training and inference framework for multiple geometry tasks to improve generalization. In Section 4.3, we introduce a conditioning method with a novel shared positional encoding strategy to enhance RGB-geometry alignment and inter-frame consistency. In Section 4.4, we accelerate training and inference by employing single-step deterministic training.

#### 4.2. Multi-Attributes Joint Training

Our method focuses on leveraging the inter-frame consistency of RGB frames to learn globally consistent geometry. For different geometric attributes, such as position and normal, since they share the exact same correspondence, they can be effectively integrated and trained together. Therefore, we employ a unified training framework by utilizing multiple RGB-geometry data pairs, as shown in Fig. 3 (left). Formally, given a collection of datasets D = {Dk}Kk=1, where each dataset Dk = {(Ij,Gj)}M

j=1 contains two components: an RGB video sequence Ij ∈ RH×W×F×3, and a corresponding geometry sequence Gj ∈ RH×W×F×d, where H and W represent spatial dimensions, F denotes the number of frames, and d indicates the channel dimension of the geometry attribute (e.g., d = 3 for normals). We train the model on these multiple datasets with an attribute identifier k (indicating, for instance, normals or positions). The attribute identifier k is integrated directly into the Transformer layers to explicitly guide the model toward the desired geometric property. At inference time, the attribute label k can be specified to control the desired geo-

k

#### 4.1. Geometry in Global Coordinate System

Video generation models can produce consistent and temporally coherent video frames. Although some existing works have explored geometric estimation (e.g., depth estimation [23, 48]) based on video diffusion, these methods focus on per-frame estimation in their respective camera coordinate systems. As a result, they fail to transfer interframe consistency to geometric properties.

Unlike existing per-frame prediction methods, we find that when geometric properties are defined in a global coordinate system, their geometric measurements exhibit a

metric estimation type, enabling a single model to handle depth estimation, normal prediction, and other geometric tasks within a unified framework.

Leveraging Multi-view Data. Finetuning Video Diffusion requires paired video data with corresponding geometric property labels. However, such data is extremely scarce in practice. Note that the correspondence between video frames and their geometric properties also applies to multi-view images, we propose a mixed training strategy to fully utilize existing high-quality multi-view datasets, such as Hypersim [43]. Specifically, we first group the data based on the bidirectional overlap between multi-view frames. For multi-view images within the same group, their latents are obtained by individually encoding each image using the video VAE, following a method similar to [6, 52]. Please refer to the supplementary material for more details regarding to the dataset grouping.

#### 4.3. Shared Positional Encoding (SPE)

Conditioning by Extending Tokens. A common approach for conditional generation in video models is channel-wise concatenation [16, 26]. However, this requires modifying the original network architecture and introducing new parameters, which harms pre-trained prior and shows limited performance (see Tab. 3). In contrast, we propose directly treating the conditional RGB sequence as extended frames, thus eliminating the need to modify the network architecture. Formally, as shown in Fig. 3 (right), given the pre-trained VAE encoder Enc, we extract the RGB tokens zrgb = Enc(Ij) ∈ Rh×w×f×c and geometry tokens zgeo = Enc(Gj) ∈ Rh×w×f×c. After adding noise to the target geometry tokens, i.e., ztgeo = √α¯tz0geo + √1 − α¯tϵ (see Sec. 3), we concatenate their VAE tokens along the token dimension, jointly treating them as an extended token sequence:

ztinput = [zrgb;ztgeo], (3)

where [; ] denotes concatenation along the token dimension. Subsequently, in the forward process of the DiT network, self-attention is applied across the entire sequence, enabling full feature exchange. To obtain the predicted denoised geometry result, we only retain the final half of the tokens from the output sequence and decode them to the pixel space using the VAE decoder.

Shared Positional Encoding (SPE). To fully utilize the learned inter-frame consistency in video diffusion, we further propose a Shared Positional Encoding (SPE) strategy without changing the network architecture. Specifically, as illustrated in Fig. 2, we observe that attention weights among tokens strongly correlate with their positional embeddings: by repeating one frame’s positional embeddings to another, different tokens corresponding to the same positional embedding consistently exhibit significantly higher mutual attention weights. Motivated by this insight, we

propose to explicitly reuse the positional embeddings from RGB tokens for geometry tokens.

Formally, let prgbi and pgeoi denote the positional embeddings associated with the i-th RGB and geometry tokens, respectively. During training and inference, we discard the original geometry positional embeddings pgeoi and replace them with the RGB positional embeddings prgbi :

pgeoi ← prgbi , ∀i. (4)

SPE effectively enforces spatial alignment and transfers inter-frame consistency to the geometry estimation, leading to improved coherence between RGB conditions and predicted geometry maps. Compared to channel concatenation, this method does not require modifying the input features of the denoising network, providing a more flexible finetuning mechanism. Ablation experiments 5.5 demonstrate that this approach propagated the inherent inter-frame consistency more effectively, ensuring consistent and coherent geometric predictions across video frames.

#### 4.4. One-step Deterministic Training

Building on recent research on fine-tuning image diffusion models for geometric estimation [17, 64], we find that video diffusion models can also be fine-tuned as one-step deterministic models for geometric estimation. Following [17], we no longer randomly sample timestep t during training but instead fix t = T. Additionally, we replace the Gaussian noise with its expectation, i.e., zero, and input it into the model along with the RGB latent representation. The video diffusion model is fine-tuned to match the latents of GT geometry attributes with an MSE loss. While significantly reducing computational cost, we find that the singlestep model produces more accurate geometric predictions, as reported in Sec 5.5.

### 5. Experiment 5.1. Setup

Dataset. Following previous work [26, 69], we trained our model exclusively on high-quality synthetic data. The datasets used for training include: Hypersim [43], an indoor multi-view dataset, using its position and normal labels, providing 40,000 samples after data grouping; InteriorNet [31], an indoor video dataset, using its position and normal supervision, providing 30,000 samples after filtering; MatrixCity [32]: an outdoor video dataset, using its normal data, providing 80,000 samples after filtering.

We choose the ScanNet++ [70] and 7scenes [50] dataset to evaluate the effectiveness of our method. Both are realworld scene datasets that has not been used during the training process. For Scannet++, the annotated geometric properties of each frame are re-rendered from a mesh scanned by a high-power LiDAR sensor (FARO Focus Premium laser

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

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

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

| |
|---|

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

| |
|---|

[Figure 138]

[Figure 139]

|[Figure 140]|[Figure 141]|[Figure 142]|[Figure 143]|[Figure 144]|
|---|---|---|---|---|

Input E2E FT StableNormal VDA Ours GT

- Figure 4. Qualitative comparisons on normal estimation. Our method produces more consistent normals, remains robust to highlight reflections, and generates results closest to the GT. It effectively removes noise from the GT, delivering smooth predictions.

scanner) and IMU camera poses, which can be used for depth and normal evaluation. 7scenes is used for the reconstruction point cloud evaluation. We preprocess these two datasets into video clips for evaluation.

Implementation Details. Our model is fine-tuned based on CogVideoX [22] 5B, which employs RoPE [55] for positional encoding. During training, we use the AdamW optimizer with a learning rate of 1e-4 and momentum parameters set to β = (0.9,0.999). The resolution of training video frames is (512, 384), and the batch size is set to 1. The entire training process runs for approximately 3 days on 8 A800 GPUs.

- 5.2. Consistent Video Geometry Estimation

only in the local camera coordinate system. To enable a fair comparison with our approach, we first transform their results into the global coordinate system using the ground truth camera parameters before conducting the same evaluation.

Normal Estimation. Following the metrics used in [69], we compute the angular error between the predicted normals and the ground truth normals. We report both the mean and median angular errors, where lower values indicate higher accuracy. Additionally, we measure the percentage of pixels with an angular error less than 11.25°, where a higher value indicates better accuracy.

Radius Estimation. Note that depth is typically defined as the z-values of the 3D coordinate in the camera coordinate system. To convert it into consistent geometric properties, we use the distance from the 3D point to the origin of the global coordinate system (“radius” in Fig 3) as a substitute, which is aligned with GT by a least-square fitting. following [10, 26]. We report the mean absolute relative error (AbsRel), defined as the average relative difference between the GT radius and the aligned counterpart at each pixel; the root mean square error (RMSE); and the percentage of pixels where the ratio of the aligned predicted radius to the GT is less than 1.25 (δ1 accuracy).

Our method can estimate position and normal from a video clip. We conduct evaluations separately for these two aspects with current state-of-the-art methods. For imagebased estiamtion methods, we choose Marigold [26] and its normal version1, E2E FT [15] and GeoWizard [16]. Additionally, we compare with the video-based geometric estimation method Video Depth Anything [10] (VDA). Since there are currently no existing works for estimating normals directly from videos, we recompute normals based on VDA’s depth results and camera intrinsic. Note that all the aforementioned methods estimate geometric properties

As shown in Table 1, our method achieves more consistent results in the global coordinate system compared to existing approaches, demonstrating its superior consistency

1https://huggingface.co/prs-eth/marigold-normals-lcm-v0-1

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

| |
|---|

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

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

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

| |
|---|

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | |[Figure 194]| |

Input E2E FT VDA Marigold Ours GT

- Figure 5. Qualitative comparisons on radius estimation. Our method achieves both accuracy and consistency, producing results closest to the ground truth. The entire video is normalized for consistent visualization.

###### Normal Radius

###### Reconstruction Depth Acc↓ Comp↓ NC↑ AbsRel↓ δ1↑

Methods

Methods

Mean↓ Median↓ 11.25↑ AbsRel↓ δ1↑

Marigold [26] 20.93 11.36 53.31 11.2 90.1 Geowizard [15] 21.33 12.61 49.23 11.5 89.6 E2E FT [17] 18.32 8.22 65.02 9.8 89.4 Stable Normal [69] 23.51 13.16 61.68 / / Video DA [10] 28.54 18.70 49.78 13.5 86.2 Ours 18.15 7.91 63.38 10.2 90.5

Dust3R [61] 0.216 0.073 0.573 25.8 61.3 Spann3R [57] 0.115 0.038 0.605 21.70 61.9 Ours 0.184 0.058 0.602 20.69 62.2

Table 2. Evaluation on 7Scenes. Our method achieves competitive results despite using significantly less training data.

Table 1. Evaluation on Scannet++ dataset. Our method achieves state-of-the-art results in both normal and radius estimation.

sequence and directly predicts the global coordinates for all frames at once. For a fair comparison, we remove Dust3R’s final alignment step in our evaluation. Please refer to the supplementary video materials for more comparisons.

in geometric estimation. Notably, unlike other methods, our approach does not require camera parameters as input, further proving the advantage of leveraging video diffusion model priors for this task.

We reported the accuracy, completion and normal consistency, by directly comparing the predicted coordinate map with back-projected pixel depth. As shown in Table 2, our method achieves comparable performance to Spann3R and significantly outperforms Dust3R. Notably, our method is trained on far less data than the compared approaches, further demonstrating its potential.We also demonstrate in Fig 6 and the supplementary video materials that our method exhibits better inter-frame consistency compared to the baseline approaches.

#### 5.3. Video Reconstruction

Since our method directly predicts the geometric properties of each frame’s pixels in a unified coordinate system, it can be directly applied to reconstruction. We also compared our method with unposed image-based reconstruction methods, such as Dust3R [61] and Spann3R [57].

Dust3R estimates the coordinates of each image pair in their respective local frames, then aligns them through an optimization-based global alignment. Spann3R, on the other hand, maintains an external spatial memory and predicts subsequent frame coordinates based on existing status. In contrast, our method treats all frames as a single token

#### 5.4. Dynamic Video Estimation

Thanks to the rich external priors from video diffusion, our method can also be applied to consistent normal and depth

[Figure 195]

[Figure 196]

[Figure 197]

| |
|---|

[Figure 198]

| |
|---|

[Figure 199]

[Figure 200]

Variants Chan Cat Seq Cat w/o Global Coord w/o Multi-Attr Ours

Metric

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Mean↓ 19.13 19.75 19.91 20.29 18.15 Median↓ 8.35 8.66 9.21 9.85 7.91 11.25↑ 61.17 61.32 59.85 59.92 63.38

| |
|---|

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Table 3. Ablation study. We ablate the key components in our design to show their effectiveness.

| |
|---|

| |
|---|

| |
|---|

[Figure 213]

[Figure 214]

[Figure 215]

ods : channel-wise concatenation (Chan Cat), sequential concatenation (Seq Cat), as well as our approach of reusing positional embeddings. We also compared the results of predicting a single attribute (w/o Multi-Attr) versus jointly predicting multiple attributes. Our findings show that optimizing multiple geometric properties together significantly improves performance. Additionally, we compared with the results of predicting geometric properties in the seperate camera coordinate system of each frame (w/o Global Coord), This demonstrates that optimizing in a unified coordinate system better leverages inter-frame consistency, leading to improved performance. Please refer to the supplementary materials for visual comparison.

| |
|---|

| |
|---|

| |
|---|

[Figure 216]

[Figure 217]

[Figure 218]

Dust3R Ours GT

- Figure 6. Comparison to Dust3R. Our method demonstrates better inter-frame consistency in multi-frame sequence reconstruction. Please refer to the videos in Supp for clearer comparisons.

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

[Figure 233]

[Figure 234]

Input E2E FT Ours

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

| |
|---|

[Figure 239]

|[Figure 240]|
|---|

| |
|---|

| |
|---|

[Figure 241]

|[Figure 242]|
|---|

[Figure 243]

[Figure 244]

| |
|---|

[Figure 245]

[Figure 246]

- Figure 7. Comparison to E2E. Our method predicts normals of better consistency with more details preserved .

### 6. Limitation

Our method processes entire video clips but, due to storage constraints, can handle only limited-length segments per iteration. A significant challenge arises when stitching results from multiple short clips, as this can lead to accumulated errors. Integrating long-term memory into the current framework remains an open research problem. Additionally, the high computational cost restricts fine-tuning to lower resolutions (512×384), occasionally causing blurry artifacts. Future work should explore efficient model distillation techniques to better capture inter-frame consistency, enhancing geometric predictions and overall output quality.

### 7. Conclusion

In this paper, we present UniGeo, a unified framework to adapt pre-trained video generation models for consistent geometry estimation by leveraging their inherent inter-frame consistency. Specifically, we advocate optimizing geometric attributes in a global coordinate system rather than local camera coordinates, thus fully exploiting inter-frame priors encoded in pre-trained models. We further introduce a shared positional encoding method to precisely condition geometric attributes from RGB frames without modifying the network architecture. Additionally, our framework naturally integrates multiple geometric attributes in joint training, capitalizing on their shared correspondences to enhance overall performance. Extensive experiments demonstrate that our method effectively predicts consistent geometric attributes, with the resulting global geometry directly applicable to reconstruction tasks.

estimation in dynamic scenes. We present some estimation results on DAVIS videos [38], along with comparisons to single-frame estimation methods. For additional comparisons, please refer to the supplementary materials.

#### 5.5. Ablation Study

We evaluated the impact of different conditioning methods on the results in Table 3, including different condition meth-

### References

- [1] Hailuo. https://hailuoai.video, 2023. 3
- [2] Keling. https://kling.kuaishou.com, 2023. 3
- [3] Pika. www.imagine.art, 2023. 3
- [4] Gwangbin Bae and Andrew J Davison. Rethinking inductive biases for surface normal estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9535–9545, 2024. 3
- [5] Gwangbin Bae, Ignas Budvytis, and Roberto Cipolla. Estimating and exploiting the aleatoric uncertainty in surface normal estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13137–13146,

2021. 3

- [6] A. Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, and Dominik Lorenz. Stable video diffusion: Scaling latent video diffusion models to large datasets. ArXiv, abs/2311.15127, 2023. 5
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [8] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1:8, 2024. 3
- [9] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. arXiv preprint arXiv:2501.12375, 2025. 3
- [10] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. arXiv:2501.12375, 2025. 2, 6, 7
- [11] Weifeng Chen, Zhao Fu, Dawei Yang, and Jia Deng. Singleimage depth perception in the wild. Advances in neural information processing systems, 29, 2016. 3
- [12] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multitask mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10786–10796, 2021. 3
- [13] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In Proceedings of the IEEE international conference on computer vision, pages 2650–2658,

2015. 3

- [14] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. Advances in neural information processing systems, 27, 2014. 3
- [15] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. ArXiv, abs/2403.12013, 2024. 6, 7

- [16] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In European Conference on Computer Vision, pages 241–258. Springer, 2024. 1, 2, 3, 5, 6
- [17] Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan de Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. ArXiv, abs/2409.11355, 2024. 5, 7, 1
- [18] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal of robotics research, 32(11):1231–1237,

2013. 3

- [19] Ming Gui, Johannes Schusterbauer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Bj¨orn Ommer. Depthfm: Fast monocular depth estimation with flow matching. arXiv preprint arXiv:2403.13788, 2024. 3
- [20] Jonathan Ho, Ajay Jain, and P. Abbeel. Denoising diffusion probabilistic models. ArXiv, abs/2006.11239, 2020. 3
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [22] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3, 6
- [23] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. ArXiv, abs/2409.02095, 2024. 4
- [24] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. arXiv preprint arXiv:2409.02095, 2024. 2, 3
- [25] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4d: Learning how things move in 3d from internet stereo videos. arXiv preprint arXiv:2412.09621, 2024. 3
- [26] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9492–9502, 2023. 1, 5, 6, 7
- [27] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9492– 9502, 2024. 3
- [28] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3

- [29] Hsin-Ying Lee, Hung-Yu Tseng, and Ming-Hsuan Yang. Exploiting diffusion prior for generalizable dense prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7861–7871, 2024. 3
- [30] Bo Li, Chunhua Shen, Yuchao Dai, Anton Van Den Hengel, and Mingyi He. Depth and surface normal estimation from monocular images using regression on deep features and hierarchical crfs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1119–1127,

2015. 3

- [31] Wenbin Li, Sajad Saeedi, John McCormac, Ronald Clark, Dimos Tzoumanikas, Qing Ye, Yuzhong Huang, Rui Tang, and Stefan Leutenegger. Interiornet: Mega-scale multisensor photo-realistic indoor scenes dataset. In British Machine Vision Conference (BMVC), 2018. 5
- [32] Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3205–3215, 2023. 5
- [33] Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, and Bingyi Kang. Prompting depth anything for 4k resolution accurate metric depth estimation. arXiv preprint arXiv:2412.14015, 2024. 3
- [34] Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, and Yuan Liu. Align3r: Aligned monocular depth estimation for dynamic videos. arXiv preprint arXiv:2412.03079, 2024. 3
- [35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 3

- [36] William S. Peebles and Saining Xie. Scalable diffusion models with transformers. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182, 2022. 3
- [37] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10106–10116, 2024. 1, 3
- [38] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 8
- [39] Xiaojuan Qi, Renjie Liao, Zhengzhe Liu, Raquel Urtasun, and Jiaya Jia. Geonet: Geometric neural network for joint depth and surface normal estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 283–291, 2018. 3
- [40] Xiaojuan Qi, Zhengzhe Liu, Renjie Liao, Philip HS Torr, Raquel Urtasun, and Jiaya Jia. Geonet++: Iterative geometric neural network with edge-aware refinement for joint depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(2):969–984,

2020. 3

- [41] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 3
- [42] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021. 3
- [43] Mike Roberts and Nathan Paczan. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10892–10902, 2020. 5
- [44] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2021. 3
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [46] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. ArXiv, abs/1505.04597, 2015. 3
- [47] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 3
- [48] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors. ArXiv, abs/2406.01493, 2024. 2, 3, 4
- [49] Evan Shelhamer, Jonathan T Barron, and Trevor Darrell. Scene intrinsics and depth from a single image. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 37–44, 2015. 3
- [50] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew William Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. 2013 IEEE Conference on Computer Vision and Pattern Recognition, pages 2930–2937, 2013. 5
- [51] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pages 746–760. Springer, 2012. 3
- [52] Uriel Singer, Adam Polyak, Thomas Hayes, Xiaoyue Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. ArXiv, abs/2209.14792, 2022. 5
- [53] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using

- nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015. 3
- [54] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [55] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. ArXiv, abs/2104.09864, 2021. 6
- [56] Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan. Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. arXiv preprint arXiv:2412.06974, 2024. 3
- [57] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. ArXiv, abs/2408.16061, 2024. 7
- [58] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061, 2024. 3
- [59] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387, 2025. 3
- [60] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. arXiv preprint arXiv:2410.19115, 2024. 3
- [61] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and J´erˆome Revaud. Dust3r: Geometric 3d vision made easy. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20697–

- 20709, 2023. 7

[62] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–

- 20709, 2024. 3

- [63] Ke Xian, Chunhua Shen, Zhiguo Cao, Hao Lu, Yang Xiao, Ruibo Li, and Zhenbo Luo. Monocular relative depth perception with web stereo data supervision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 311–320, 2018. 3
- [64] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. What matters when repurposing diffusion models for general dense perception tasks? arXiv preprint arXiv:2403.06090,

2024. 5

- [65] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. What matters when repurposing diffusion models for general dense perception tasks? arXiv preprint arXiv:2403.06090,

2024. 3, 1

- [66] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024. 1, 3

- [67] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2025. 1, 3
- [68] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. ArXiv, abs/2408.06072, 2024. 3, 4
- [69] Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. ACM Transactions on Graphics,

2024. 3, 5, 6, 7

- [70] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023. 5
- [71] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9043–9053, 2023. 1
- [72] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825, 2024. 3

## UniGeo: Taming Video Diffusion for Unified Consistent Geometry Estimation Supplementary Material

### S1. Outline

In this supplementary file, we provide detailed description of multi-view data grouping, ablation comparison, and further results that could not be included in the main paper due to space constraints.

### S2. Multi-view Data Processing

Our primary dataset is Hypersim, where each scene consists of images captured from several discrete camera viewpoints. Therefore, we first preprocess the dataset by grouping the images.

[Figure 247]

Figure S1. Given a query image, we use its depth and camera parameters to project it into other viewpoints. We then compute the overlap pixel ratio with the corresponding image to determine their relevance.

As illustrated by Fig S1, for each image in each scene, we we compute its relationship with any other images in this scene based on camera viewpoints and depth, denoted as Rij.

Specifically, for each scene, we compute the overlap between image pairs based on camera viewpoints and depth. For the i − th image, we compute its bidirectional overlap with all other images using (Rij + Rji)/2. We then select the top (NumView - 1) images with the highest overlap to form a data group with the i-th image for training.

### S3. Coordinate Frame Definition

UniGeo predicts geometric properties directly within a global coordinate space, thereby ensuring consistency for the same 3D point across different video frames. The global frame of reference is defined by the coordinate system of the first image in the video sequence.

To prepare the training data, given a sequence of L frames with associated depth maps Di, surface normals Ni, camera extrinsic parameters Ei, and intrinsic parameters Ki (i = 1,2,··· ,L), we transform these local geometric properties into the global coordinate space as follows:

Coordci = Ki−1([U;V ;1] · Di) (5) Coordgi = E1Ei−1h(Coordci) (6)

Normalgi = r(E1Ei−1)Ni (7)

Here, U and V denote the pixel coordinate grids along the x and y axes, respectively; h represents the transformation to homogeneous coordinates; and r extracts the rotational component from the transformation matrix.

### S4. Ablation Study Visualization

In this section, we present visualizations of ablation experiments to demonstrate the effectiveness of our design choices.

Effectiveness of concatenation method. We present a comparison of convergence speed across different concatenation methods. The evaluation is conducted on a small dataset consisting of 10 sequences, showing the results after 100 steps of training, as shown in Fig S3. The results demonstrate that our proposed method, which reuses positional embeddings, achieves faster convergence, indicating that it effectively leverages the video diffusion prior.

Effectiveness of Multi-attributes Joint Training. We find that training multiple geometric properties simultaneously yields better results than training a single property in isolation. In Fig S2, we compare the normal estimation results when training only normal versus jointly training normal and position. The results show that joint optimization helps the model develop a better understanding of spatial geometry, leading to more reasonable predictions.

[Figure 248]

Figure S2. After the same number of fine-tuning steps, our method aligns RGB with geometric properties more quickly and produces more consistent predictions. This demonstrates that our proposed approach preserves the original prior as much as possible without disruption.

Effectiveness of one-step deterministic training. In the realm of image-based geometry estimation tasks, works such as [17, 65] have achieved comparable or even supe-

[Figure 249]

- Figure S3. After the same number of fine-tuning steps, our method aligns RGB with geometric properties more quickly and produces more consistent predictions. This demonstrates that our proposed approach preserves the original prior as much as possible without disruption.

rior performance by replacing the multi-step denoising generation of diffusion with a single-step process, while significantly reducing computational overhead. We have attempted to apply a similar approach to video diffusion models.

By fixing the timestep t at T during the training process and initializing the noise to the mean of a Gaussian distribution, i.e., zero, we have trained a single-step deterministic diffusion model for consistent geometric estimation. In Fig. S4, we present a comparison of the results between single-step and multi-step approaches under the condition of the same number of training steps.

[Figure 250]

- Figure S4. The figure illustrates a comparison between the results of multi-step and single-step approaches. The yellow boxes indicate areas where the multi-step method achieves sharper results, while the red boxes highlight regions where the multi-step approach exhibits prediction errors.

### S5. More Results

In this section, we present additional experimental results, including comparisons with current video depth estimation methods on additional datasets and more visualization.

Comparison with video depth estimation methods. We further evaluate our approach on three static datasets (“ScanNet”, “Neural RGBD”, and “Replica”) and one dynamic dataset (“Bonn”), comparing it with SOTA video depth estimation methods. Here we report “Depth” and “Radius” metrics in “local” and “global” coordinate systems, respectively. Despite being trained exclusively on limited static data, our method still outperforms competitors and generalizes effectively to dynamic scenes, demonstrating the successful incorporation of the video diffusion model prior.

ScanNet NRGBD Replica Bonn AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ DepthCrafter

Method

local 7.28 95.8 7.33 95.6 6.21 97.9 7.47 95.9 global 7.35 95.3 7.52 94.3 6.41 97.3 7.82 95.2

local 7.35 94.9 6.02 97.2 8.31 95.4 8.03 94.6 global 7.76 94.2 6.91 96.4 8.65 94.9 8.24 94.1

ChronoDepth

local 15.5 79.1 15.6 79.5 15.8 78.6 15.2 78.3 global 16.8 76.3 16.4 77.0 16.7 77.6 16.6 76.9

DepthAnyVideo

local 6.95 95.9 7.14 95.6 6.35 97.6 8.09 95.1 global 6.69 96.3 6.58 96.4 5.53 98.3 8.03 95.6

UniGeo

Table S1. Comparison with other video depth methods on additional datasets.

More visualization results. Here we visualize more comparison results. Please refer to the project page for video results.

Based on the results presented in Fig. S4 and the numerical outcomes of the ablation study in the main text, it is evident that, overall, the one-step approach still holds a significant advantage in such deterministic prediction tasks.

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

|[Figure 275]|
|---|

|[Figure 276]|
|---|

| |
|---|

| |
|---|

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

|[Figure 283]|
|---|

|[Figure 284]|
|---|

| |
|---|

| |
|---|

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

|[Figure 291]|
|---|

|[Figure 292]|
|---|

| |
|---|

| |
|---|

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

|[Figure 299]|
|---|

|[Figure 300]|
|---|

| |
|---|

| |
|---|

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

| |
|---|

| |
|---|

| |
|---|

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

| |
|---|

| |
|---|

| | |
|---|---|
|[Figure 316]| |

|[Figure 317]|
|---|

|[Figure 318]|
|---|

Input E2E FT StableNormal VDA Ours GT

- Figure S5. We show more visual comparisons of predicted normal on scannetpp dataset. The inconsistency is marked in red rectangles. It can be seen that ours achieve the most consistent visual effect while E2E and StableNormal provide inconsistent results, and VDA provides erroneous and inconsistent normals.

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

Input E2E FT VDA Marigold Ours GT

- Figure S6. We show more visual comparisons of predicted radius on scannetpp dataset. Compared with other depth estimation methods, our approach produces more consistent and accurate geometry estimation.

