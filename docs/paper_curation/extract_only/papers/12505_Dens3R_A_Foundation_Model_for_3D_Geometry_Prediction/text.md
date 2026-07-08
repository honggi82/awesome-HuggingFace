# arXiv:2507.16290v1[cs.CV]22Jul2025

## Dens3R: A Foundation Model for 3D Geometry Prediction

XIANZE FANG∗, Alibaba Group, China JINGNAN GAO∗, Shanghai Jiao Tong University, China ZHE WANG, Alibaba Group, China ZHUO CHEN, Shanghai Jiao Tong University, China XINGYU REN, Shanghai Jiao Tong University, China JIANGJING LV†, Alibaba Group, China QIAOMU REN, Alibaba Group, China ZHONGLEI YANG, Alibaba Group, China XIAOKANG YANG, Shanghai Jiao Tong University, China YICHAO YAN‡, Shanghai Jiao Tong University, China CHENGFEI LV, Alibaba Group, China

[Figure 1]

Fig. 1. Dens3R is a feed-forward visual foundation model that takes unposed images as input and outputs high-quality 3D pointmap with unified geometric dense prediction. Dens3R also accepts generalized inputs, supporting both multi-view and multi-resolution inputs. As a versatile backbone, Dens3R achieves robust dense prediction under several scenarios and can be easily extended to downstream applications.

∗Equal Contribution †Project Leader ‡Corresponding Author

Authors’ Contact Information: Xianze Fang, Alibaba Group, China, fangxianze.fxz@ taobao.com; Jingnan Gao, Shanghai Jiao Tong University, China, gjn0310@sjtu.edu.cn; Zhe Wang, Alibaba Group, China, xingmi.wz@taobao.com; Zhuo Chen, Shanghai Jiao Tong University, China, ningci5252@sjtu.edu.cn; Xingyu Ren, Shanghai Jiao Tong University, China, rxy_sjtu@sjtu.edu.cn; Jiangjing Lv, Alibaba Group, China, jiangjing.ljj@taobao.com; Qiaomu Ren, Alibaba Group, China, renqiaomu.rqm@taobao. com; Zhonglei Yang, Alibaba Group, China, zhonglei.yzl@taobao.com; Xiaokang Yang, Shanghai Jiao Tong University, China, xkyang@sjtu.edu.cn; Yichao Yan, Shanghai Jiao Tong University, China, yanyichao@sjtu.edu.cn; Chengfei Lv, Alibaba Group, China, chengfei.lcf@taobao.com.

Recent advances in dense 3D reconstruction have led to significant progress, yet achieving accurate unified geometric prediction remains a major challenge. Most existing methods are limited to predicting a single geometry quantity from input images. However, geometric quantities such as depth, surface normals, and point maps are inherently correlated, and estimating them in isolation often fails to ensure consistency, thereby limiting both accuracy and practical applicability. This motivates us to explore a unified framework that explicitly models the structural coupling among different geometric properties to enable joint regression. In this paper, we present Dens3R, a 3D foundation model designed for joint geometric dense prediction and adaptable to a wide range of downstream tasks. Dens3R adopts a two-stage training framework to progressively build a pointmap representation that is both generalizable and intrinsically invariant. Specifically, we design a lightweight shared encoder-decoder backbone and introduce

position-interpolated rotary positional encoding to maintain expressive power while enhancing robustness to high-resolution inputs. By integrating image-pair matching features with intrinsic invariance modeling, Dens3R accurately regresses multiple geometric quantities such as surface normals and depth, achieving consistent geometry perception from single-view to multi-view inputs. Additionally, we propose a post-processing pipeline that supports geometrically consistent multi-view inference. Extensive experiments demonstrate the superior performance of Dens3R across various dense 3D prediction tasks and highlight its potential for broader applications. CCS Concepts: • Computing methodologies → Computer Vision. Additional Key Words and Phrases: Visual Foundation Model, 3D Geometry Prediction

- 1 Introduction

Recovering 3D geometric structures from static images is a longstanding and fundamental problem in computer vision. Classical approaches, such as Structure-from-Motion (SfM) and Multi-View Stereo (MVS), demonstrate strong performance in controlled settings and have been widely adopted in a broad range of 3D reconstruction applications. However, in unconstrained scenarios—where camera intrinsics, extrinsics, or viewpoint information are unavailable—achieving accurate and dense geometric prediction remains highly challenging. These conditions demand more generalizable and robust solutions capable of handling diverse and unstructured visual inputs.

Existing methods for dense geometric prediction primarily fall into two categories. The first category mostly adopts generative models, utilizing strong image priors from pre-trained diffusion models or large-scale training datasets for dense prediction. For example, GenPercept [Xu et al. 2025] is used for depth prediction, and StableNormal [Ye et al. 2024] for normal estimation. This raises a key issue: while image generation tasks typically benefit from their inherent ambiguity and multi-modal output characteristics, geometric prediction is fundamentally different. Geometric prediction is essentially a deterministic task that needs to closely reflect the structural information of the underlying scene. Moreover, the pixel continuity and spatial smoothness required by geometric representations are difficult to naturally obtain through standard diffusion sampling mechanisms without structural constraints. Therefore, the direct application of diffusion models in geometric regression tasks faces significant challenges, especially in such tasks where a strict one-to-one correspondence between input and output needs to be maintained. Based on this, we adopt a regression-oriented framework to construct geometric mapping models in a more efficient and interpretable way. Furthermore, the aforementioned methods mainly handle only one geometric quantity prediction and cannot generalize to output multiple geometric quantities in a single forward pass. The second category includes DUSt3R [Wang et al. 2024] and its follow-up works [Leroy et al. 2024; Wang et al. 2025a,b]. These methods use regression models that can regress 3D point map representations with geometric properties, applied to dense prediction, including image pair matching and depth estimation. However, these methods typically focus on a single prediction task, and other geometric quantities suffer severe performance degradation due to representation influences.

This raises a natural question: can we build a unified model that simultaneously regresses multiple geometric quantities with high quality? We observe that existing methods like DUSt3R, when handling dense geometric regression tasks, overlook a crucial geometric information—surface normals. Traditionally, normals have been used to add high-frequency details to rough geometric structures to enhance rendering quality. However, our research finds that introducing normal information during geometric prediction can significantly improve the accuracy of point maps, resulting in more detailed and structurally consistent 3D representations. This is mainly because: 1) From the perspective of normal prediction, the inherent image pair matching capability in dense vision backbone networks helps alleviate monocular ambiguity and improve the stability and accuracy of normal prediction; 2) From the feature modeling perspective, normals possess good intrinsic invariance, which simplifies the mapping learning process and aids in model convergence and generalization. This modeling approach enables the model to simultaneously predict multiple geometric quantities (such as depth, normals, and point maps) from a single view, effectively reducing dependence on multi-view supervision and simplifying the training process. However, training such a multi-task, multi-output 3D foundation model still faces significant challenges. Geometric quantities are tightly coupled, and how to coordinate these relationships to achieve optimal overall performance requires carefully designed training strategies and architectural support.

In this paper, we present Dens3R, a foundation model for highquality geometric prediction. To this end, we design a two-stage training framework that gradually builds a versatile pointmap representation, which generalizes well to various downstream tasks. Specifically, we first construct a dense vision backbone network with multi-task prediction capabilities. This network adopts a shared encoder-decoder architecture, which significantly reduces model parameters while maintaining expressive power. To accommodate high-resolution inputs, we introduce position-interpolated rotary positional encoding, which effectively mitigates prediction degradation caused by increased input resolution. For the training strategy, we propose a novel two-staged approach. In the first stage, the model leverages image pair matching features to learn scale-invariant point maps, capturing consistent spatial geometric structures across viewpoints. Subsequently, to fully exploit the one-to-one mapping property in normal estimation, we extend the pointmap representation into an intrinsically invariant form. This allows the model to independently attend to each viewpoint, thereby improving the accuracy of normal prediction. The learned geometric structures also assist in estimating other geometric quantities, such as depth, thereby simplifying their training processes. Finally, we design a simple and efficient post-processing pipeline that supports multi-view inputs during inference, which enhances the geometric consistency of the model in real-world applications. In summary, we make the following contributions:

• We introduce Dens3R, a dense 3D visual foundation model that demonstrates high-quality performance in various 3D tasks including pointmap reconstruction, depth estimation, normal prediction and image matching under several benchmark evaluations.

- • We design a novel training strategy with the intrinsicinvariant pointmap and shared Encoder-Decoder visual backbone to incorporate surface normals in unconstrained image-based dense 3D reconstruction, simplifying the training complexity of other 3D quantities and achieving better results without requiring excessive computation resources.
- • We employ a position-interpolated rotary positional encoding to preserve prediction accuracy at higher resolutions and support multi-resolution inputs.
- • Extensive experiments on various benchmarks showcase our high-quality predictions of 3D geometric quantities, which further enable a wide range of applications.

- 2 Related Works

- 2.1 Monocular depth and normal prediction

Monocular depth prediction has been extensively investigated and demonstrates strong capability in providing geometric priors for a multitude of downstream tasks like image understanding and 3D reconstruction. The earliest pioneering researchers [Bhat et al. 2021, 2023; Eigen et al. 2014; Hu et al. 2024; Piccinelli et al. 2024; Yin et al. 2023] addressed this issue by estimating depth with a metric scale. These methods usually rely heavily on data from specific sensors, which restricts the applicability and deteriorates the performance when confronted with complex scenes. Subsequently, deep learning approaches involve predicting relative depth either through direct regression [Chen et al. 2016, 2020; Godard et al. 2019; Li and Snavely 2018a; Ranftl et al. 2022; Yang et al. 2024a,b] or via generative modeling based on diffusion priors [Fu et al. 2024; Gui et al. 2024; Ke

- et al. 2024; Wan et al. 2023]. While monocular depth estimation has made significant strides, accurate 3D shape reconstruction from depth maps remains fundamentally dependent on precise camera intrinsic parameters. Meanwhile, normal maps serve as a supervision for neural scene representation, bridging 2D and 3D worlds. The accurate estimation of the normal map can open up broader applications like material decomposition and relighting. On one hand, regression-based methods [Bansal et al. 2016; Eftekhar et al. 2021; Ranftl et al. 2021; Wang et al. 2015] utilize large-scale training datasets for robust estimation. DSINE [Bae and Davison 2024] proposes to leverage the per-pixel ray direction and try to model the inductive biases for surface normal estimation correctly. On the other hand, diffusion-based methods [Fu et al. 2024; Long et al. 2024; Ye et al. 2024] adapt the pretrained diffusion model as a geometric cues predictor. Geowizard includes a geometry switcher to disentangle mixed-sourced data into distinct sub-distributions for normal prediction. StableNormal repurposes the diffusion model for deterministic estimation tasks and can estimate sharp normals steadily. Nevertheless, these normal estimation methods often suffer from monocular ambiguity, leading to inaccurate and inconsistent results for complex scenes. In contrast, our method allows the communication between 3D geometric representation and normal prediction without known camera poses. This not only resolves the ambiguity but also achieves accurate 3D reconstruction with accurate normals.

- 2.2 Image Pair Matching in 3D

Dense matching [Edstedt et al. 2023, 2024; Efe et al. 2021; Melekhov et al. 2019; Sarlin et al. 2020; Sun et al. 2021; Truong et al. 2021, 2020, 2023; Zhu and Liu 2023] has been proved to be effective in many scenarios and results in top performance in many benchmarks. However, these approaches cast matching as a 2D problem, which restricts the application for visual localization. Thus anchoring image correspondence in 3D space is essential when these 2D-based methods fall short. Early methods [Bhalgat et al. 2023; He et al. 2020; Toft et al. 2020; Wang et al. 2020a; Yao et al. 2019; Yifan et al. 2022; Zhou et al. 2021] leverage epipolar constraints in order to improve accuracy or robustness. Recently, researchers [Wang et al. 2023b; Zhang et al. 2024] leverage diffusion models for pose estimation and demonstrate promising results by incorporating 3D geometric constraints into estimation formulation. MASt3R [Leroy et al. 2024] retrieves correspondences via 3D reconstruction from uncalibrated images by explicitly training local features for pairwise matching. However, MASt3R only grounds image-pair matching and overlooks other geometric predictions like depth and normal, while our method achieves unified geometric predictions and better matching.

- 2.3 Dense Unconstrained Geometric Representations

Neural scene reconstructions [Barron et al. 2021, 2023; Kerbl et al. 2023; Lu et al. 2024; Martin-Brualla et al. 2021; Mildenhall et al. 2020; Wang et al. 2021a, 2023a; Yariv et al. 2021; Yu et al. 2024] usually require the camera intrinsic parameters and poses for optimization. The reconstruction quality of these methods is highly dependent on the accuracy of the camera intrinsics and poses. Later methods [Hong et al. 2024; Smart et al. 2024; Ye et al. 2025] propose to optimize the scene without known camera poses, but these methods usually take longer time and sacrifice reconstruction quality. To bypass estimation of camera parameters and poses, DUSt3R [Wang et al. 2024] proposes to directly map two input images in a single forward pass, leading to a more straightforward geometry representation. Subsequently, Spann3R [Wang and Agapito 2025] and Fast3R [Yang et al. 2025] augment DUSt3R to process an ordered set of images. MoGe [Wang et al. 2025b] further proposes affine-invariant pointmaps for monocular geometry estimation. VGGT [Wang et al. 2025a] utilizes 3D pointmaps and multiple prediction heads to predict geometric quantities from multi-view images input. However, the aforementioned methods overlook the normal attribute and fall short in prediction for complex scenarios. In contrast, our model takes advantage of pointmap representation and employs several prediction heads including the normal head to achieve unified geometric predictions.

3 Method

This work aims to utilize a single model to predict various geometric data from unconstrained images, including 3D pointmaps, depth maps, normal maps, and image-pair matching. To this end, we built a backbone network based on dense visual transformers and designed input configurations that adapt to multi-resolution and multi-view requirements (Sec. 3.1). Since achieving accurate results through direct training with a single model is challenging, we adopted a twostage training approach. In the first stage, we train the backbone

[Figure 2]

Fig. 2. Overview of Dens3R. We propose Dens3R, a dense visual transformer backbone featuring a shared encoder-decoder architecture and multiple task-specific heads for geometric prediction. To train this foundation model, we adopt a two-stage strategy. In Stage 1, we learn a scale-invariant pointmap by enforcing cross-view mapping consistency across multiple viewpoints. In Stage 2, we incorporate surface normals and leverage one-to-one correspondence constraints to transform the representation into an intrinsic-invariant pointmap. Built upon this unified backbone, additional geometric prediction heads and downstream task branches can be seamlessly integrated to support a wide range of applications.

and heads to obtain scale-invariant pointmaps. In the second stage, we fine-tune the backbone on this foundation to obtain intrinsicinvariant pointmaps (Sec. 3.2). Finally, we further fine-tune the prediction heads for each downstream task to adapt to different application scenarios. Meanwhile, extending the model inputs to multi-view images in the inference stage significantly improves the overall inference quality. (Sec. 3.3).

- 3.1 Model Formulation Shared Backbone. Motivated by recent advances in 3D vision [Jin

- et al. 2025; Leroy et al. 2024; Wang et al. 2025a, 2024], we aim to build a foundation model capable of predicting diverse geometric quantities across different scenes and tasks. To this end, we adopt a dense visual transformer as the backbone, learning from rich 3D

annotated data. Given an image pair of image sequence (𝐼𝑖)𝑖2=1 ∈ R3×𝐻×𝑊 , Dens3R’s dense visual transformer is a function 𝑓 that maps the input to a corresponding set of 3D quantities per frame:

(𝐶𝑖,𝑃𝑖,𝐷𝑖, 𝑁𝑖,𝑀𝑖)𝑖2=1 = 𝑓 ((𝐼𝑖)𝑖2=1), (1)

where 𝐶𝑖 ∈ R9 is the camera parameters including both intrinsics and extrinsics, 𝐷𝑖 ∈ R𝐻×𝑊 is the depth map, 𝑁𝑖 ∈ R3×𝐻×𝑊 is the normal map, and 𝑀𝑖 ∈ R𝐶×𝐻×𝑊 is the image-pair-matching with 𝐶-dimensional features.

The overall architecture is illustrated in the upper part of Fig. 2. Similar to prior DUSt3R-based approaches [Leroy et al. 2024; Wang et al. 2025a,b, 2024], we first employ a shared-weight encoder to process input image sequences and extract image features 𝐹𝑒𝑎𝑖, which are then fed into the decoder. Unlike previous works, our approach introduces a novel weight-sharing mechanism within the decoders, allowing the backbone to better capture spatial relationships across viewpoints and to model the holistic 3D scene structure. Given the need to predict a wider range of geometric outputs, this

design also significantly reduces memory and computational overhead, keeping the training and inference efficient. Moreover, the shared-weight strategy facilitates high-resolution input processing while effectively preventing memory overflow.

Multi-resolution Input. Existing methods represented by DUSt3R perform excellently at fixed resolutions (such as 512), but their prediction accuracy significantly decreases when processing higherresolution inputs. The main challenge for this issue lies in the rotary positional encoding (RoPE) used in their ViT structure, which becomes unstable when inferring images beyond the training resolution range. Inspired by context window extension techniques in LLMs [Chen et al. 2023], we incorporate the position-interpolated RoPE into the ViT as a simple yet effective improvement. We adapt the idea from context window to image resolution in the image domain, addressing the instability at higher resolutions. Considering the smooth properties of trigonometric functions in RoPE, interpolation is more stable than direct extrapolation when handling high resolutions. Specifically, let the original RoPE be 𝑅, the input sequence length be 𝐿, and for any RoPE embedding vector 𝑥, we obtain a new encoding representation 𝑅′ through interpolation. That is:

𝑚𝐿

𝐿′ ), (2) where 𝑚 is the position index and 𝐿′ is the longer sequence. This position-interpolation encoding strategy significantly enhances the model’s robustness under high-resolution inputs, effectively avoiding the performance degradation caused by RoPE extrapolation.

𝑅′(𝑥,𝑚) = 𝑅(𝑥,

3.2 Foundation Model Training

The main challenge in training 3D geometric foundation models lies in the coupling among multiple prediction outputs, where mutual interference often leads to performance degradation. Existing methods typically focus on only one or two geometric tasks, resulting in poor generalization to others. To this end, we propose to build upon

a unified geometric representation since all geometric representations are inherently interconvertible. We adopt a two-stage training paradigm, progressively learns a strong geometric prior, which can be efficiently transferred to a variety of 3D geometry prediction tasks via lightweight fine-tuning.

Scale-Invariant Pointmap Training. In the first stage, we train the ViT backbone (Encoder-Decoder), pointmap head, and matching head to obtain a scale-invariant pointmap 𝑃𝑖. Following MASt3R’s [Leroy et al. 2024], we adopted (1) local 3D regression loss Lpts_loc , (2) Global 3D Regression Loss Lpts_glb , (3) Pointmap Normal Loss Lpts_n, (4) Pixel Matching Loss Lmatch. The details are as follows:

- (1) Local 3D Regression Loss Lpts_loc . For a predicted camera, we use the local 3D regression loss to quantify the pointmap in its own coordinate frame. We apply a mask derived from the ground-truth data to the pointmap and only evaluate the valid points when calculating the loss. We also employ a normalization factor to handle the scale ambiguity between ground-truth and the predicted pointmaps.

We set the factor 𝑧𝑣 as the average distance of all valid points in 𝑣𝑡ℎ camera coordinate frame to the origin:

𝑧𝑣 = 𝑃𝑚𝑎𝑠𝑘𝑒𝑑1,𝑣 + 𝑃𝑚𝑎𝑠𝑘𝑒𝑑2,𝑣 ,𝑣 ∈ {1, 2}, 𝑧¯𝑣 = 𝑃 ¯𝑚𝑎𝑠𝑘𝑒𝑑1,𝑣 + 𝑃 ¯𝑚𝑎𝑠𝑘𝑒𝑑2,𝑣 ,𝑣 ∈ {1, 2},

(3)

where 𝑧¯𝑣 is the corresponding factor of the ground-truth. Then the local 3D regression loss can be formulated as:

Lpts_loc =

1 𝑧𝑣

𝑃𝑚𝑎𝑠𝑘𝑒𝑑𝑣,𝑣 −

1 𝑧¯𝑣

𝑃¯𝑚𝑎𝑠𝑘𝑒𝑑𝑣,𝑣 ,𝑣 ∈ {1, 2}, (4)

where 𝑃𝑛,𝑚 denotes the pointmap from camera 𝑛 expressed in the coordinate frame of camera 𝑚.

- (2) Global 3D Regression Loss Lpts_glb . The global 3D regression loss is applied to quantify the pointmap expressed in another camera’s coordinate frame. This loss function simultaneously optimizes for two objectives. It not only constrains the network to fit the pointmap shape of the image, but also aligns the pointmap to another paired image. The global regression loss is formulated as:

Lpts_glb =

1 𝑧𝑡

𝑃𝑚𝑎𝑠𝑘𝑒𝑑𝑣,𝑡 −

1 𝑧¯𝑡

𝑃¯𝑚𝑎𝑠𝑘𝑒𝑑𝑣,𝑡 ,𝑣,𝑡 ∈ {1, 2},𝑣 ≠ 𝑡, (5)

where 𝑧𝑡 and 𝑧¯𝑡 is the normalization factor of the pointmap and the ground-truth.

- (3) Pointmap Normal Loss Lpts_n. To train an intrinsic-invariant pointmap from the scale-invariant pointmap, we use a pointmap normal loss to encourage the pointmap learn smooth surface and sharp edge, making the pointmap perceives the normal information and the intrinsic-invariant property. Suppose 𝑁𝑣,𝑣 is the groundtruth view-space normal expressed in its own camera coordinate frame and 𝑁𝑣,𝑡 is the ground-truth normal expressed in another camera coordinate frame, the pointmap normal loss is the absolute error loss between the transformed normal and the ground-truth normal:

Lpts_n = L1(𝑁𝑣,𝑣, 𝑁ˆ𝑣,𝑣) + L1(𝑁𝑣,𝑡, 𝑁ˆ𝑣,𝑡),𝑣,𝑡 ∈ {1, 2},𝑣 ≠ 𝑡, (6)

where the 𝑁ˆ𝑣,𝑣 is the normal transformed from the local pointmap and 𝑁ˆ𝑣,𝑡 is the normal transformed from the global pointmap.

[Figure 3]

Fig. 3. Normal comparison. We demonstrate that the normal derived directly from the scale-invariant pointmap and MoGe both are not accurate enough.

(4) Pixel Matching Loss Lmatch. We utilize the pixel matching loss proposed in MASt3R [Leroy et al. 2024] to learn accurate imagematching. This loss is based on the infoNCE [Oord et al. 2018] loss and ensures that each pixel’s descriptor in the first image match at most one pixel’s descriptor in another image. Suppose Mˆ = (𝑖, 𝑗) is the set of ground-truth correspondences where the 𝑖𝑡ℎ pixel in the first image matches the 𝑗𝑡ℎ pixel in another, the loss can then be formulated as:

Lmatch = − ∑︁

𝑠𝜏 (𝑖, 𝑗) 𝑘∈P1 𝑠𝜏 (𝑘, 𝑗)

𝑠𝜏 (𝑖, 𝑗) 𝑘∈P2 𝑠𝜏 (𝑖,𝑘)

log

+ log

,

(𝑖,𝑗)∈Mˆ

(7)

𝑠𝜏 (𝑖, 𝑗) = exp −𝜏𝐷𝑖1⊤𝐷2𝑗 ,

where 𝜏 is a hyper-parameter, and 𝐷𝑖 and 𝐷𝑗 are the corresponding descriptors in each image.

With the above losses, we summarize the training objective as:

L𝑠𝑡𝑎𝑔𝑒1 = Lpts_loc + 𝜂1Lpts_glb + 𝜂2Lpts_n + 𝜂3Lmatch, (8)

where the loss weights 𝜂1, 𝜂2, and 𝜂3 are set as 1.0, 0.1 and 0.075, respectively. After training, we obtained a scale-invariant pointmap capable of capturing rich spatial information. However, as shown in Fig. 3, the accuracy of normals obtained directly from the pointmap at this stage is still not ideal.

Intrinsic-Invariant Pointmap Training. Although the pointbased representation learned in the first stage achieves good performance, it remains limited in its ability to generalize to other tasks—particularly surface normal estimation. Existing methods often struggle with monocular ambiguity in normal prediction, leading to inaccurate and inconsistent results.

To this end, we expanded the pointmap representation in the second stage, proposing an intrinsic-invariant pointmap. This representation allows the model to form consistent geometric understanding of the same structure under different viewpoints, effectively improving the stability and generalization capability of normal estimation. Specifically, we introduce high-quality normal supervision based on the first stage’s point map, and jointly finetune the encoder-decoder module, point map prediction head, and newly added normal prediction head to achieve end-to-end optimization. In terms of supervision mechanism, we adjusted the initial "one-to-many" mapping (one image corresponding to multiple view supervisions) to a "one-to-one" mapping, enabling the model to independently optimize normal prediction under a single viewpoint. This strategy not only significantly reduces the ambiguity brought by multi-view supervision but also simplifies the training process and improves training efficiency and stability.

We observe that the commonly-used confidence loss in previous works [Leroy et al. 2024; Wang et al. 2025a, 2024] tends to cause models to ignore complex scenarios such as reflective surfaces

and low textured areas. However, naively removing the loss without additional constraints leads to degraded performance, since previous models rely heavily on confidence weighting for point-view regression. In contrast, by utilizing the deterministic nature of normals, we obviate the need to rely on additional views, which further enables stable and accurate prediction.

For detailed implementation, we explicitly connect normal to the pointmap representation, that is

𝑃𝑖𝑛 = 𝑃𝑖 ⊕ 𝑛, (9) where ⊕ represents feature concatenation operation. The normal prediction head is connected after the initial point map training is completed, allowing the model to consistently output coherent normal mappings from the same input image, thereby internalizing this intrinsic invariance in the point map and maintaining geometric consistency across different views.

In the second stage, we add a normal loss Ln for finetuning.

(5) Predicted Normal Loss Ln. Apart from the intrinsic-invariant pointmap, we also design a normal head to predict the view-space normal of each frame in input image pairs. We also use the L1 loss to supervise the normal prediction:

Ln = L1(𝑁𝑣,𝑣, 𝑁¯𝑣,𝑣),𝑣 ∈ {1, 2}, (10)

where 𝑁 is the ground-truth normal and 𝑁¯ is the direct prediction of the normal prediction head. The complete training objective for training stage 2 is as follows:

L𝑠𝑡𝑎𝑔𝑒2 = Lpts_loc + 𝜆1Lpts_glb + 𝜆2Lpts_n + 𝜆3Ln, (11)

where the loss weights 𝜆1, 𝜆2, and 𝜆3 are set as 1.0, 0.1 and 1.0, respectively.

To further improve the performance of Dens3R on high-resolution inputs, we introduce a coarse-to-fine training strategy. Specifically, we first fine-tune the model on 512 resolution images to establish a stable geometric prior, and then fine-tune it on 1024 resolution images to further improve the prediction accuracy. In addition, combining high-resolution data also significantly improves the fidelity of point-based representations, ultimately enhancing the overall quality of dense 3D prediction.

3.3 Model Inference

Heads Training. After training, we fine-tune it for downstream tasks by adding task-specific prediction heads on top of the frozen backbone network. Training only these new heads enables extension to various tasks such as depth estimation, normal estimation, matching estimation, and even segmentation and object detection. Multi-view Inputs. To enable Dens3R to efficiently process multiview inputs during inference, we design a simple yet effective postprocessing step. This step ensures both computational efficiency in multi-view data processing and the consistency and accuracy of results. Specifically, based on Dens3R’s high-precision image pair matching predictions, we establish geometric mappings between different viewpoints by constructing and optimizing a dense correspondence network across views. This approach effectively guides the model to understand geometric consistency between multiple viewpoints and accurately captures spatial relationships between views. Additionally, while maintaining high computational

efficiency, it significantly improves the performance and stability of multi-view processing.

4 Experiments

- 4.1 Normal and Matching Prediction

We evaluateourDens3Ron severalsurface normal prediction datasets that include both indoor and outdoor scenes. We compare our method with regression-based methods such as DSINE [Bae and Davison 2024] and diffusion-based methods like StableNormal [Ye

- et al. 2024], GeoWizard [Fu et al. 2024] and Lotus [He et al. 2025]. Quantitative results are shown in Tab. 1, where Dens3R outperforms other methods across multiple benchmarks. Qualitative comparisons are provided in Fig. 4, also demonstrating that Dens3R generates more accurate and detailed normal maps. On the DIODE dataset, our method produces more accurate normals for reflective surfaces (e.g., car window) and finer details in backgrounds and tree structures. On in-the-wild scenes, Dens3R handles both object-centric and unbounded scenarios, producing more stable and intricate surface normals. Thanks to the intrinsic-invariant pointmap and the proposed training strategy, our method effectively reduces the ambiguity from monocular estimation, enabling more accurate and detailed predictions across various settings.

For the image-matching task, we evaluate our method on the ZEB benchmark as shown in Tab. 2. We compare our method with previous dense image-matching methods and MASt3R [Leroy et al. 2024]. It can be seen that our method yields higher accuracy and surpasses previous methods across nearly all datasets, demonstrating our superior performance across various evaluation protocols.

4.2 Pointmap and Depth Prediction

For monocular depth prediction, we evaluate our model on several datasets containing both indoor and outdoor scenes. We compare our method with MoGe [Wang et al. 2025b], VGGT [Wang

- et al. 2025a], MASt3R [Leroy et al. 2024] and DUSt3R [Wang et al. 2024]. The quantitative results are presented in Tab. 3. It can be seen that our method achieves accurate results in depth estimation. The qualitative comparison of the depth prediction is shown in Fig. 5 with the corresponding pointmaps. Our method achieves high-quality pointmap prediction and depth estimation with the intrinsic-invariant pointmap and the novel training strategy. As for pointmap prediction, MoGe and VGGT often fail to recover depth for reflective surfaces and tend to produce flattened pointmaps in background regions. In contrast, our method accomplishes to predict accurate depth with high-quality pointmaps. Moreover, our method yields more stable and high-quality predictions than MASt3R. It is also noteworthy that our method generates more accurate depth maps than DUSt3R, which can be reflected from the depth predictions for the Chandeliers.

- 4.3 Ablation Study

We present high-quality geometric predictions for high-resolution inputs and various scenarios in Fig. 6 and Fig. 7. More evaluations and comparisons are also provided in the Supplementary Materials.

[Figure 4]

- Fig. 4. Qualitative comparison of normal prediction. Dens3R generates more accurate and detailed normal maps than previous methods. Our method is capable of predicting accurate normals for reflective surfaces and in backgrounds. For both object-centric and unbounded scenes, our method also accomplishes to predict stable and intricate surface normals.

NYUv2 ScanNet IBims-1 Sintel DIODE-outdoor

Method Mean Med 𝛿11.25◦ Mean Med 𝛿11.25◦ Mean Med 𝛿11.25◦ Mean Med 𝛿11.25◦ Mean Med 𝛿11.25◦ DSINE 18.6 9.9 56.1 18.6 9.9 56.1 18.8 8.3 64.1 34.9 28.1 21.5 22.0 14.5 39.6 Lotus* 17.5 8.6 58.7 18.1 8.8 58.2 19.2 5.6 66.2 35.7 28.0 20.5 24.7 15.9 32.9

GeoWizard 20.4 11.9 47.0 21.4 13.9 37.1 19.7 9.7 58.4 41.6 34.3 11.8 27.0 19.8 24.0 StableNormal 19.7 10.5 53.0 18.1 10.1 56.0 17.2 8.1 66.7 35.0 27.0 19.5 26.9 16.1 36.1

Ours 16.1 7.4 62.5 16.9 7.1 64.0 16.0 4.3 72.2 30.7 21.4 28.9 20.8 12.8 43.0

- Table 1. Quantitative comparison of normal prediction. We report the mean and median angular errors with each cell colored to indicate the best and the second . Dens3R achieves accurate normal prediction for both indoor and outdoor scenes. *Note that we utilize Lotus-G for a fair comparison.

Method

Mean Real AUC@5° ↑ Simulate AUC@5° ↑ AUC@5° ↑ GL3 BLE ETI ETO KIT WEA SEA NIG MUL SCE ICL GTA

SIFT 31.8 43.5 33.6 49.9 48.7 35.2 21.4 44.1 14.7 33.4 7.6 14.8 43.9 SuperGlue 34.3 43.2 34.2 58.7 61.0 29.0 28.3 48.4 18.8 34.8 2.8 15.4 36.5 LoFTR 39.1 50.6 43.9 62.6 61.6 35.9 26.8 47.5 17.6 41.4 10.2 25.6 45.0 DKM 51.2 63.3 53.0 73.9 76.7 43.4 34.6 52.5 24.5 56.6 32.2 42.5 61.6 ROMA 53.2 61.8 53.8 76.7 82.7 43.2 36.7 53.2 26.6 60.7 33.8 45.4 64.3 MASt3R 59.9 57.8 52.3 66.2 78.1 46.2 52.8 70.5 43.7 70.1 53.9 60.1 67.7 Ours 64.5 61.3 59.2 74.7 81.1 55.6 57.4 71.7 50.4 71.3 53.7 66.3 71.7

- Table 2. Benchmark on image matching on ZEB dataset. We report the AUC values with each cell colored to indicate the best and the second . Our method outperforms previous methods in almost all the metrics.

|Method<br><br>|NYUv2 REL↓ RMSE↓ 𝛿1 ↑ 𝛿2 ↑ 𝛿3 ↑|DIODE-indoor REL↓ RMSE↓ 𝛿1 ↑ 𝛿2 ↑ 𝛿3 ↑<br><br>|DIODE-outdoor REL↓ RMSE↓ 𝛿1 ↑ 𝛿2 ↑ 𝛿3 ↑|
|---|---|---|---|
|GenPercept Lotus* DepthAnythingV2 DUSt3R VGGT MoGe Ours|0.052 0.214 96.7 99.3 99.8<br><br>0.053 0.262 96.5 99.1 99.7<br><br><br>0.049 0.204 97.3 99.3 99.8 0.046 0.197 97.1 99.3 99.8 0.038 0.194 98.0 99.4 99.8 0.035 0.167 97.9 99.4 99.9 0.042 0.189 97.5 99.3 99.8<br><br>|0.107 0.924 89.1 96.0 98.1 0.111 1.123 88.7 96.0 98.4 0.091 0.878 92.5 97.3 98.6 0.083 0.375 92.0 97.7 99.0 0.064 0.404 93.1 98.0 99.2 0.080 0.879 92.6 97.3 98.7 0.072 0.372 93.7 97.5 98.8<br><br>|0.727 5.571 67.3 84.2 90.6 0.488 9.960 47.1 63.3 71.8 0.705 5.525 67.8 83.4 89.7 0.451 5.217 67.7 84.3 90.7 0.400 4.861 70.6 84.9 90.6 0.578 5.177 72.8 86.7 91.9 0.387 4.740 72.2 87.0 92.3<br><br>|

- Table 3. Quantitative comparison on monocular depth prediction. We report the relative point error (REL), root mean square error (RMSE) and the percentage of inliers 𝛿1,𝛿2,𝛿3 with each cell colored to indicate the best and the second . *Note that we utilize Lotu-G disparity model for comparison.

We then conducted comprehensive ablation studies for our key components: the position-interpolated rotary positional encoding, the intrinsic-invariant training and the coarse-to-fine training strategy.

Method NYUv2 ScanNet IBims

Mean ↓ 𝛿11.25◦ ↑ Mean ↓ 𝛿11.25◦ ↑ Mean ↓ 𝛿11.25◦ ↑

w/o IIT 17.8 50.6 18.6 49.4 20.2 56.8 w/o C2F 17.6 50.5 17.8 58.8 18.6 63.9

Ours 16.1 62.5 16.9 64.0 16.0 72.2

Table 4. Normal quantitative metrics for ablation. We demonstrate that both the intrinsic-invariant training and coarse-to-fine strategy contributes to accurate normal predictions.

Position-Interpolated Rotary Positional Encoding. Dens3R can supportmulti-resolutionimageinputs. With the position-interpolated rotary positional encoding and the coarse-to-fine training strategy, our method can prevent performance degradation when handling high-resolution inputs. As shown in Fig. 8a, we can generate accurate and well-structured pointmaps with the position-interpolated

RoPE, preventing the model from producing overlapping or inconsistent pointmaps at higher resolutions.

Intrinsic-Invariant Training. Our approach first learns a scaleinvariant pointmap, which is then transformed into an intrinsicinvariant pointmap via subsequent intrinsic-invariant training. We find that jointly training the pointmap and normal at the initial scale-invariant stage leads to instability and poor convergence. This is because pointmaps and normal maps lie in different data domains, and coupling their supervision potentially increases training complexity. While GeoWizard [Fu et al. 2024] addresses this domain gap with a task switcher, we adopt a two-stage training scheme to learn an intrinsic-invariant pointmap, ensuring stable learning. As shown in Tab. 4 and in Fig. 8b, the performance of the model will degrade without the intrinsic-invariant training.

Coarse-to-Fine Training. Our model is trained on diverse training dataset of varying quality. To better utilize the full training set, we implement a coarse-to-fine training strategy that gradually increases resolution and data fidelity. In the coarse stage, we set the max resolution of the training images as 512 and enable all the training data. In the fine stage, we increase the image resolution to 1024 pixels and restrict training to the high-resolution data only. As demonstrated in Tab. 4 and in Fig. 8b, this strategy improves prediction accuracy, particularly for high-resolution outputs.

- 4.4 Downstream Applications

Segmentation Head Training. Dens3R serves as a visual foundation model that can be finetuned for several downstream tasks. We demonstrate this by training a new prediction head for segmentation task while keeping our backbone frozen. As shown in Fig. 8c, the segmentation head can generate accurate results, with much more effortless training than a large segmentation model.

Surface Reconstruction. Dens3R can improve surface reconstruction quality by its sharp and accurate normals. We demonstrate this by utilizingourpredictednormals as the supervision for NeuS [Wang et al. 2021a] training. The results are showcased in Fig. 8d. It can be seen that the final reconstruction results are improved due to the strong normal prior provided by our Dens3R.

5 Conclusion

We propose Dens3R, a 3D foundation model for dense geometric prediction that jointly regresses multiple geometric quantities, including depth, surface normals, and pointmaps, from unconstrained image inputs. Unlike previous approaches that estimate geometry in isolation, Dens3R explicitly models the structural coupling among these properties to ensure consistency and improves overall accuracy. We utilize a two-stage training framework with coarse-to-fine strategy and build an accurate intrinsic-invariant pointmap representation. In addition, we design a lightweight encoder-decoder architecture and position-interpolated rotary positional encoding to enable scalable and high-fidelity inference for high-resolution inputs. Moreover, Dens3R incorporates a geometrically consistent post-processing pipeline for multi-view inputs. Extensive experiments demonstrate our superior performance across various 3D prediction benchmarks and highlight the potential as a versatile backbone for broader downstream applications.

References

Eduardo Arnold, Jamie Wynn, Sara Vicente, Guillermo Garcia-Hernando, Áron Monszpart, Victor Adrian Prisacariu, Daniyar Turmukhambetov, and Eric Brachmann. 2022. Map-free Visual Relocalization: Metric Pose Relative to a Single Image. In ECCV, Vol. 13661. 690–708.

Gwangbin Bae and Andrew J. Davison. 2024. Rethinking Inductive Biases for Surface Normal Estimation. In CVPR. 9535–9545. Aayush Bansal, Bryan C. Russell, and Abhinav Gupta. 2016. Marr Revisited: 2D-3D Alignment via Surface Normal Prediction. In CVPR. 5965–5974.

Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo MartinBrualla, and Pratul P. Srinivasan. 2021. Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields. In ICCV. 5835–5844.

Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.

2023. Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields. In ICCV. 19697– 19705.

Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. 2021. ARKitScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile RGB-D Data. In NeurIPS Datasets and Benchmarks.

Yash Bhalgat, João F. Henriques, and Andrew Zisserman. 2023. A Light Touch Approach to Teaching Transformers Multi-view Geometry. In CVPR. 4958–4969. Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. 2021. AdaBins: Depth Estimation Using Adaptive Bins. In CVPR. 4009–4018.

Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. 2023. ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth. arxiv preprint arXiv:2302.12288 (2023).

Ming-Fang Chang, John W Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, and James Hays. 2019. Argoverse: 3D Tracking and Forecasting with Rich Maps. In CVPR. 8748–8757.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending Context Window of Large Language Models via Positional Interpolation. arxiv preprint arXiv:2306.15595 (2023).

Weifeng Chen, Zhao Fu, Dawei Yang, and Jia Deng. 2016. Single-Image Depth Perception in the Wild. In NeurIPS. 730–738.

Weifeng Chen, Shengyi Qian, David Fan, Noriyuki Kojima, Max Hamilton, and Jia Deng. 2020. OASIS: A Large-Scale Dataset for Single Image 3D in the Wild. In CVPR. 676–685.

Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn. 2019. A large RGB-D dataset for semi-supervised monocular depth estimation. arXiv preprint arXiv:1904.10230 (2019).

Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. 2022. ABO: Dataset and Benchmarks for Real-World 3D Object Understanding. In CVPR. 21094–21104.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. 2023. Objaverse-XL: A Universe of 10M+ 3D Objects. In NeurIPS.

Johan Edstedt, Ioannis Athanasiadis, Mårten Wadenbäck, and Michael Felsberg. 2023. DKM: Dense Kernelized Feature Matching for Geometry Estimation. In CVPR. 17765– 17775.

Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, and Michael Felsberg.

2024. RoMa: Robust Dense Feature Matching. In CVPR. 19790–19800. Ufuk Efe, Kutalmis Gokalp Ince, and A. Aydin Alatan. 2021. DFM: A Performance Baseline for Deep Feature Matching. In CVPRW. 4284–4293.

Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. 2021. Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans. In ICCV. 10766–10776.

David Eigen, Christian Puhrsch, and Rob Fergus. 2014. Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. In NeurIPS. 2366–2374.

Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. 2024. GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image. In ECCV, Vol. 15080. 241–258.

Adrien Gaidon, Qiao Wang, Yohann Cabon, and Eleonora Vig. 2016. Virtual worlds as proxy for multi-object tracking analysis. In CVPR. 4340–4349.

Clément Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J. Brostow. 2019. Digging Into Self-Supervised Monocular Depth Estimation. In ICCV. 3827–3837.

Ming Gui, Johannes S. Fischer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Björn Ommer. 2024. DepthFM: Fast Monocular Depth Estimation with Flow Matching. arxiv preprint arXiv:2403.13788 (2024).

Jose L. Gómez, Manuel Silva, Antonio Seoane, Agnés Borràs, Mario Noriega, German Ros, Jose A. Iglesias-Guitian, and Antonio M. López. 2025. All for one, and one for all: UrbanSyn Dataset, the third Musketeer of synthetic driving scenes. Neurocomputing

637 (2025), 130038.

Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Liu, Bingbing Liu, and Ying-Cong Chen. 2025. Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction. In ICLR.

Yihui He, Rui Yan, Katerina Fragkiadaki, and Shoou-I Yu. 2020. Epipolar Transformers. In CVPR. 7776–7785.

Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jiaolong Yang, Seungryong Kim, and Chong Luo. 2024. Unifying Correspondence, Pose and NeRF for Pose-Free Novel View Synthesis from Stereo Pairs. In CVPR.

Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. 2024. Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation. IEEE Trans. Pattern Anal. Mach. Intell. 46, 12 (2024), 10579–10596.

Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. 2025. LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias. In ICLR.

Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. 2024. Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. In CVPR. 9492–9502.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4 (2023), 139:1–139:14.

Vincent Leroy, Yohann Cabon, and Jérôme Revaud. 2024. Grounding Image Matching in 3D with MASt3R. In ECCV, Vol. 15130. 71–91.

Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R. Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, C. Karen Liu, Jiajun Wu, and Li Fei-Fei. 2022. BEHAVIOR-1K: A Benchmark for Embodied AI with 1, 000 Everyday Activities and Realistic Simulation.. In CORL, Vol. 205. 80–93.

Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. 2023. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In ICCV. 3205–3215.

- Zhengqi Li and Noah Snavely. 2018a. MegaDepth: Learning Single-View Depth Prediction From Internet Photos. In CVPR. 2041–2050.
- Zhengqi Li and Noah Snavely. 2018b. MegaDepth: Learning Single-View Depth Prediction from Internet Photos. In CVPR. 2041–2050.

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. 2024. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In CVPR. 22160–22169.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. 2024. Wonder3D: Single Image to 3D Using Cross-Domain Diffusion. In CVPR. 9970–9980.

Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. 2024. Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering. CVPR

(2024).

Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duckworth. 2021. NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections. In CVPR. 7210–7219.

Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andrés Bruhn.

2023. Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo. In CVPR. 4981–4991.

Iaroslav Melekhov, Aleksei Tiulpin, Torsten Sattler, Marc Pollefeys, Esa Rahtu, and Juho Kannala. 2019. DGC-Net: Dense Geometric Correspondence Network. In WACV. 1034–1042.

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.

Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 2019. 3D Ken Burns Effect from a Single Image. ACM Trans. Graph. 38, 6 (2019), 184:1–184:15. Aaronvanden Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation Learning with Contrastive Predictive Coding. arXiv preprint arXiv:1807.03748 (2018).

Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segù, Siyuan Li, Luc Van Gool, and Fisher Yu. 2024. UniDepth: Universal Monocular Metric Depth Estimation. In CVPR. 10106–10116.

Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, Zeyu Ma, and Jia Deng. 2024. Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation. In CVPR. 21783–21794.

René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. 2021. Vision Transformers for Dense Prediction. In ICCV. 12159–12168.

René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. 2022. Towards Robust Monocular Depth Estimation: Mixing Datasets for ZeroShot Cross-Dataset Transfer. IEEE Trans. Pattern Anal. Mach. Intell. 44, 3 (2022),

1623–1637.

Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. 2021. Common Objects in 3D: Large-Scale Learning and Evaluation of Real-Life 3D Category Reconstruction. In ICCV. 10881–10891.

Stephan R Richter, Vibhav Vineet, Stefan Roth, and Vladlen Koltun. 2016. Playing for data: Ground truth from computer games. In ECCV, Vol. 9906. 102–118.

Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. 2021. Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding. In ICCV. 10892–10902.

German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, and Antonio M. Lopez.

2016. The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes. In CVPR. 3234–3243.

Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich.

2020. SuperGlue: Learning Feature Matching With Graph Neural Networks. In CVPR. 4937–4946.

Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. 2019. Habitat: A Platform for Embodied AI Research. In ICCV. 9338–9346.

Philipp Schröppel, Jan Bechtold, Artemij Amiranashvili, and Thomas Brox. 2022. A Benchmark and a Baseline for Robust Multi-view Depth Estimation. In 3DV. 637– 645.

Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. 2024. Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs. arxiv preprint arXiv:2408.13912 (2024).

Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. 2021. LoFTR: Detector-Free Local Feature Matching With Transformers. In CVPR. 8922–8931. Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. 2020. Scalability in Perception for Autonomous Driving: Waymo Open Dataset. In CVPR. 2443–2451.

Carl Toft, Daniyar Turmukhambetov, Torsten Sattler, Fredrik Kahl, and Gabriel J. Brostow. 2020. Single-Image Depth Prediction Makes Feature Matching Easier. In ECCV, Vol. 12361. 473–492.

Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. 2021. SMD-Nets: Stereo Mixture Density Networks. In CVPR. 8942–8952. Prune Truong, Martin Danelljan, Luc Van Gool, and Radu Timofte. 2021. Learning Accurate Dense Correspondences and When To Trust Them. In CVPR. 5714–5724.

Prune Truong, Martin Danelljan, and Radu Timofte. 2020. GLU-Net: Global-Local Universal Network for Dense Flow and Correspondences. In CVPR. 6257–6267. Prune Truong, Martin Danelljan, Radu Timofte, and Luc Van Gool. 2023. PDC-Net+: Enhanced Probabilistic Dense Correspondence Network. IEEE Trans. Pattern Anal. Mach. Intell. 45, 8 (2023), 10247–10266.

Qiang Wan, Zilong Huang, Bingyi Kang, Jiashi Feng, and Li Zhang. 2023. Harnessing Diffusion Models for Visual Perception with Meta Prompts. arxiv preprint arXiv:2312.14733 (2023).

Hengyi Wang and Lourdes Agapito. 2025. 3D Reconstruction with Spatial Memory. In 3DV.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. 2025a. VGGT: Visual Geometry Grounded Transformer. In CVPR.

Jianyuan Wang, Christian Rupprecht, and David Novotný. 2023b. PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment. In ICCV. 9739–9749.

Kaixuan Wang and Shaojie Shen. 2020. Flow-motion and depth network for monocular stereo and beyond. IEEE Robotics and Automation Letters 5, 2 (2020), 3307–3314. Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021a. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. In NeurIPS. 27171–27183.

Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. 2021b. Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In ICME. 1–6.

Qianqian Wang, Xiaowei Zhou, Bharath Hariharan, and Noah Snavely. 2020a. Learning Feature Descriptors Using Camera Pose Supervision. In ECCV, Vol. 12346. 757–774.

Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. 2025b. MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision. In CVPR.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jérôme Revaud.

2024. DUSt3R: Geometric 3D Vision Made Easy. In CVPR. 20697–20709.

Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. 2020b. TartanAir: A Dataset to Push the Limits of Visual SLAM. (2020), 4909–4916.

Xiaolong Wang, David F. Fouhey, and Abhinav Gupta. 2015. Designing deep networks for surface normal estimation. In CVPR. 539–547.

Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and Lingjie Liu. 2023a. NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction. In ICCV.

Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. 2024. RGBD Objects in the Wild: Scaling Real-World 3D Object Learning from RGB-D Videos. In CVPR. 22378–22389.

Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. 2025. What Matters When Repurposing Diffusion Models for General Dense Perception Tasks?. In ICLR.

Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. 2025. Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass. In CVPR.

Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024a. Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. In CVPR. 10371–10381.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024b. Depth Anything V2. In NeurIPS. Yuan Yao, Yasamin Jafarian, and Hyun Soo Park. 2019. MONET: Multiview SemiSupervised Keypoint Detection via Epipolar Divergence. In ICCV. 753–762.

Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. 2020. BlendedMVS: A Large-Scale Dataset for Generalized Multi-View Stereo Networks. In CVPR. 1787–1796.

Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume Rendering of Neural Implicit Surfaces. In NeurIPS. 4805–4815.

Botao Ye, Sifei Liu, Haofei Xu, Li Xueting, Marc Pollefeys, Ming-Hsuan Yang, and Peng Songyou. 2025. No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images. In ICLR.

Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. 2024. StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal. ACM Trans. Graph. 43, 6 (2024), 250:1– 250:18.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. 2023. ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes. In ICCV. 12–22. Wang Yifan, Carl Doersch, Relja Arandjelovic, João Carreira, and Andrew Zisserman.

2022. Input-level Inductive Biases for 3D Reconstruction. In CVPR. 6166–6176. Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. 2023. Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image. In ICCV. 9009–9019.

Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024. MipSplatting: Alias-free 3D Gaussian Splatting. CVPR (2024).

Amir R Zamir, Alexander Sax, , William B Shen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. 2018. Taskonomy: Disentangling Task Transfer Learning. In CVPR. 3712–3722.

Jason Y. Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. 2024. Cameras as Rays: Pose Estimation via Ray Diffusion. In ICLR.

Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. 2020. Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling. In ECCV, Vol. 12354. 519–535.

Qunjie Zhou, Torsten Sattler, and Laura Leal-Taixé. 2021. Patch2Pix: Epipolar-Guided Pixel-Level Correspondences. In CVPR. 4669–4678. Shengjie Zhu and Xiaoming Liu. 2023. PMatch: Paired Masked Image Modeling for Dense Geometric Matching. In CVPR. 21909–21918.

[Figure 5]

- Fig. 5. Qualitative comparison of depth maps and pointmaps. We compare our method with previous DUSt3R-based methods and demonstrate high-quality depth prediction results. Dens3R also reconstructs more stable and accurate pointmap than previous methods.

[Figure 6]

Fig. 6. High-quality geometric predictions for high-resolution (2K) inputs. Please zoom in to better observe the fine-grained details.

[Figure 7]

Fig. 7. High-quality unified geometric predictions for various scenarios. We demonstrate accurate normal and depth predictions with high-quality 3D pointmaps for challenging object-centric, indoor and outdoor scenes.

[Figure 8]

(a) High-resolution inference comparison. Our method supports highresolution input and generates accurate and well-structured pointmaps.

[Figure 9]

(c) Segmentation results. Our model can be easily extended to segmentation tasks by training a new prediction head with the backbone frozen.

[Figure 10]

(b) Normal comparison for ablation. The intrinsic-invariant training enables accurate normal prediction and the coarse-to-fine training enhances details.

[Figure 11]

(d) Normal supervision results. We demonstrate the effectiveness of using our normal as the supervision of surface reconstruction.

Fig. 8. Ablation and downstream applications.

- A Implementation Details

Datasets. To train the visual foundation model, we collect and reorganize a large-scale training dataset containing various data types. The dataset includes indoor scenes, outdoor scenes, and object-level data. It is noteworthy that the quality of training data has a substantial impact on model performance. We then make the most of high-quality synthetic data in the training process for more accurate and robust predictions. We divide all the data into three types based on their quality. Data of type A is collected from synthetic rendering process with the highest quality. Data of type B also originates from synthetic rendering, but they possess certain quality issues like insufficient resolution or absence of background or imprecise original 3D models, etc. Data of type C is obtained from the real world using cameras and depth sensors. We also carefully allocate the proportions of each dataset to attain the optimal model training performance. We summarize and present this dataset information in Tab. 5.

Training Details. During our coarse-to-fine training, we first utilize all the images with 512 resolution and train our model for about two weeks in the coarse-stage training. Then we only utilize the images from type A dataset and a minor portion of type B dataset and set the image resolution to 1024 for the fine-stage training. We utilize 32 Nvidia A800 GPUs for both the coarse and fine stage training. As for model inference, our model only requires a single Nvidia RTX3090 GPU for 1024-resolution image inputs.

- B Additional Visualization

We provide additional unified geometric prediction results of monocular inputs in Fig. 9 and 2-view inputs in Fig. 10. It can be seen that our model achieves robust and high-quality unified geometric predictions across several scenarios.

- C Normal and Depth Comparison

Dens3R predicts robust and accurate normal and depth for various scenarios. As shown in Fig. 11, we demonstrate that the intrinsicinvariant training assists the pointmap to capture the geometric information from normal. Then the normal prediction head further predicts sharper edges and more accurate results.

We provide the normal prediction comparison of the Kitti dataset in Fig. 12. It can be seen that our method generates the most accurate and sharp normals. We also provide more comparison of normal map prediction in Fig 15 using in-the-wild images and in Fig. 16 using DL3DV dataset. It can be seen that our method predicts sharper and more accurate normal across various scenarios. We also compare our method with the normal map derived from the predicted depth map of MoGe [Wang et al. 2025b], the results are shown in

- Fig. 13. It can be seen that Dens3R can handle normal predictions for reflective surfaces and accomplishes to generate richer details. We also provide the full quantitative comparison in Tab. 6 and full ablation comparison in Tab. 7, which are partly shown in Tab. 1 and Tab. 4 in the main paper correspondingly.

We provide the depth prediction comparison in Fig. 17, it can be seen that our method generates the most accurate depth maps even for reflective surfaces. Since VGGT [Wang et al. 2025a] also predicts

multiple quantities including depth and matching, we further compare our predicted depth map with VGGT. We demonstrate more accurate depth predictions of NYUv2 dataset in Fig. 18. We also showcase the accurate prediction of both indoor scenes of NYUv2 dataset and outdoor scenes of Kitti dataset. It can be seen in Fig. 19 and Fig. 20 that our model also achieves accurate human depth estimation that can be further utilized for detection and autonomous driving.

- D Camera Pose Estimation Comparison

Dens3R can also perform accurate camera pose estimation through a single feed-forward pass. We conduct extended experiments to demonstrateitsaccuracy. Weutilize the map-free benchmark [Arnold et al. 2022] following the MASt3R protocol [Leroy et al. 2024], which is a challenging dataset aiming at localizing the camera in metric space given a single reference image without any map. We present the camera pose estimation (Map-free relocalization) comparison in Tab. 8. It can be seen that Dens3R outperforms previous methods in nearly all the metrics, demonstrating highly accurate camera pose estimation results.

- E Image Matching Visualization Results

For image-matching, apart from the ZEB dataset, we also provide the quantitative comparison of the Scannet-1500 dataset in Tab. 9 and the MegaDepth-1500 dataset in Tab. 10. The comparisons on the ScanNet-1500 and the MegaDepth-1500 benchmarks further demonstrate our superior performance over pervious DUSt3R-based method MASt3R [Leroy et al. 2024] and VGGT [Wang et al. 2025a].

We also demonstrate our dense and accurate matching results of several challenging cases in Fig. 21. We visualize the matching results from MegaDepth-1500 dataset, ScanNet-1500 dataset and Aachen dataset. It can be seen that our method can handle imagematching for 1) inputs with different lighting conditions, 2) inputs taken from different views with large-angle difference, 3) inputs with small overlapping regions. The accuracy of our method is shown on the upper left of each matching image-pair.

- F High-Resolution Inference Comparison

We showcasemorecomparisonofhigh-resolution inputs with DUSt3R [Wang et al. 2024] and VGGT [Wang et al. 2025a] in Fig. 22. It can be seen that our method can handle higher-resolution inputs without causing degenerated predictions like previous methods with our proposed position-interpolated rotary positional encoding.

- G Shared Encoder-Decoder Backbone Ablation

Dens3R employs a dense visual transformer backbone designed to capture spatial relationships across viewpoints and capture the global 3D geometric information of scenes. Different from previous methods, both the encoder and decoder components in our architecture share weights. The comparison of the network parameters and the memory cost is shown in Tab. 11. Since our model deals with more 3D quantities than previous methods, the framework initially requires a higher memory cost. Employing the shared encoderdecoder structure also resolves this issue, reducing the memory cost and network parameters without losing the prediction quality.

[Figure 12]

### Fig. 9. Additional visual results for unified geometric predictions of monocular inputs by our methods

[Figure 13]

### Fig. 10. Additional visual results for unified geometric predictions of 2-view images inputs by our method.

Applied Losses Image

Dataset Type

Ratio

Lpts_loc, Lpts_glb Lpts_n Lmatch Ln Pairs Hypersim[Roberts et al. 2021] A ✓ ✓ ✓ ✓ 1.8M 6.77% UnrealStereo4K[Tosi et al. 2021] A ✓ ✓ ✓ ✓ 0.9M 6.77% MatrixCity[Li et al. 2023] A ✓ ✓ ✓ ✓ 0.7M 6.77% Infinigen[Raistrick et al. 2024] A ✓ ✓ ✓ ✓ 2.8M 6.77% Behavior [Li et al. 2022] A ✓ ✓ ✓ ✓ 6.8M 6.77% Structure3D[Zheng et al. 2020] A ✓ ✓ ✓ ✓ 0.2M 4.06% GTASFM[Wang and Shen 2020] A ✓ ✓ ✓ ✓ 0.2M 13.53% GTAV[Richter et al. 2016] A ✓ ✓ ✓ ✓ 0.6M 13.53% VirtualKitti[Gaidon et al. 2016] A ✓ ✓ ✓ ✓ 4.0M 13.53% IRS[Wang et al. 2021b] A ✓ ✓ ✓ ✓ 74K 0.41% UrbanSyn[Gómez et al. 2025] A ✓ ✓ ✓ ✓ 7.0K 0.41% Spring[Mehl et al. 2023] A ✓ ✓ ✓ ✓ 10K 0.41%

ScanNet++[Yeshwanth et al. 2023] B ✓ ✓ ✓ ✓ 3.5M 1.35% ABO[Collins et al. 2022] B ✓ ✓ ✓ ✓ 2.0M 1.35% GObjaverseXL[Deitke et al. 2023] B ✓ ✓ ✓ ✓ 6.8M 1.35% StaticThings3D[Schröppel et al. 2022] B ✓ ✓ ✓ ✓ 0.3M 1.35% BlendedMVS[Yao et al. 2020] B ✓ ✓ ✓ ✓ 1.1M 1.35% Habitat[Savva et al. 2019] B ✓ ✓ ✓ ✓ 1.3M 0.68% Taskonomy[Zamir et al. 2018] B ✓ ✓ ✓ ✓ 1.8M 0.68% ARKitScenes[Baruch et al. 2021] B ✓ ✓ ✓ ✓ 2.2M 0.68% Tartanair[Wang et al. 2020b] B ✓ ✓ ✓ ✓ 4.5M 0.68% Synthia[Ros et al. 2016] B ✓ ✓ ✓ ✓ 2.6M 0.68% KenBurns[Niklaus et al. 2019] B ✓ ✓ ✓ ✓ 0.3M 0.68%

MegaDepth[Li and Snavely 2018b] C ✓ ✓ ✓ 1.8M 1.35% Waymo[Sun et al. 2020] C ✓ ✓ ✓ 1.1M 1.35% Co3dv2[Reizenstein et al. 2021] C ✓ ✓ ✓ 1.2M 1.35% WildRGBD[Xia et al. 2024] C ✓ ✓ ✓ 1.1M 1.35% NianticMapFree[Arnold et al. 2022] C ✓ ✓ ✓ 3.7M 1.35% DL3DV[Ling et al. 2024] C ✓ ✓ ✓ 1.2M 1.35% DIMLIndoor[Cho et al. 2019] C ✓ ✓ ✓ 0.9M 0.68% ArgoverseStereo[Chang et al. 2019] C ✓ ✓ ✓ 4.0K 0.68%

- Table 5. Training dataset information. We reorganize a large-scale training dataset and divide the data into three types based on their quality. We also showcase the training objectives we apply during training, the number of image pairs and the corresponding dataset ratio.

[Figure 14]

Fig. 11. Normal prediction comparison of different training stages.

H Multi-view Image Inputs

Apart from multi-resolution image inputs, our model also design a simple yet effective post-processing pipeline and supports multiview inputs. The results are demonstrate in Fig. 23. It can be seen that our method enables high-quality 3D reconstruction even without known camera poses. In addition, the predictions of Dens3R can be used to initialize robust and accurate 3D reconstructions by

integrating the predicted attributes with modern structure-frommotion (SfM) pipelines such as Glomap.

I Limitation

Although Dens3R outperforms previous methods in geometric predictions, predicting accurate results for inputs with thin structures remains a significant challenge. Restricted by the network’s limited capacity and the presence of noisy training data, our method may predict inaccurate results for these inputs. As shown in Fig. 14, the prediction quality for thin structures still require further improvement.

Method Mean ↓ Med ↓ 𝛿11.25° ↑ 𝛿22.5° ↑ 𝛿30° ↑ NYUv2 (indoor)

DSINE 18.6 9.9 56.1 76.9 82.6 Lotus 17.5 8.6 58.7 76.4 82.0

- GeoWizard 20.4 11.9 47.0 73.8 80.8

StableNormal 19.7 10.5 53.0 75.9 81.7 Ours 16.1 7.4 62.5 78.8 84.0

ScanNet (indoor) DSINE 18.6 9.9 56.1 76.9 82 Lotus 18.1 8.8 58.2 75.3 80.8

- GeoWizard 21.4 13.9 37.1 71.7 79.7

StableNormal 18.1 10.1 56.0 78.8 84.1 Ours 16.9 7.1 64.0 78.1 82.7

IBims-1 (indoor) DSINE 18.8 8.3 64.1 78.6 82.2

Lotus 19.2 5.6 66.2 74.9 78.1 GeoWizard 19.7 9.7 58.4 77.6 81.6

StableNormal 17.2 8.1 66.7 81.1 84.6 Ours 16.0 4.3 72.2 80.1 83.0

Sintel (outdoor) DSINE 34.9 28.1 21.5 41.5 52.7

Lotus 35.7 28.0 20.5 41.8 52.8 GeoWizard 41.6 34.3 11.8 31.8 43.9

StableNormal 35.0 27.0 19.5 42.4 54.6 Ours 30.7 21.4 28.9 51.9 62.2

DIODE-outdoor (outdoor) DSINE 22.0 14.5 39.6 67.5 75.4

Lotus 24.7 15.9 32.9 63.9 71.9 GeoWizard 27.0 19.8 24.0 56.6 68.9

StableNormal 26.9 16.1 36.1 60.6 67.5 Ours 20.8 12.8 43.0 70.7 77.0

- Table 6. Full quantitative comparison of normal prediction. We report the mean and median angular errors with each cell colored to indicate the best and the second .

|Dataset|Metrics|w/o IIT w/o C2F Ours<br><br>|
|---|---|---|
|NYUv2<br><br>|Mean ↓ 𝛿11.25° ↑<br><br>|17.8 17.6 16.1 50.6 50.5 62.5|
|ScanNet|Mean ↓ 𝛿11.25° ↑<br><br>|18.6 17.8 16.9 49.4 58.8 64.0<br><br>|
|IBims<br><br>|Mean ↓ 𝛿11.25° ↑|20.2 18.6 16.0 56.8 63.9 72.2<br><br>|
|Sintel<br><br>|Mean ↓ 𝛿11.25° ↑<br><br>|35.9 35.8 30.7 18.9 22.3 28.9|
|DIODE-outdoor|Mean ↓ 𝛿11.25° ↑<br><br>|23.5 21.6 20.8 33.7 40.2 43.0<br><br>|

- Table 7. Full normal quantitative metrics for ablation. We demonstrate that both the intrinsic-invariant training and coarse-to-fine strategy contributes to accurate normal predictions.

[Figure 15]

- Fig. 12. Normal comparison of Kitti dataset. We present more normal comparison of outdoor scenes, our method produces more accurate and sharper normals than previous methods.

[Figure 16]

- Fig. 13. Normal comparison with MoGe. We provide more normal comparison with the normal maps derived from the depth map of MoGe [Wang et al. 2025b]. Our method yields sharper and more accurate predictions.

Method Reproj. Error ↓ Precision ↑ AUC ↑ Median Error (m) ↓ Median Error (°) ↓ Pose Precision ↑ Pose AUC ↑

DUSt3R 125.8 px 45.2% 0.704 1.10 m 9.4° 17.0% 0.344 MASt3R 57.2 px 75.9% 0.934 0.46 m 3.0° 51.7% 0.746

VGGT 48.8 px 78.9% 0.789 0.36 m 3.6° 57.7% 0.577 Ours 30.4 px 82.1% 0.944 0.24 m 3.4° 65.5% 0.852

- Table 8. Camera pose estimation results of the Map-free dataset. We report the metrics with each cell colored to indicate the best and the second .

Method AUC@5° ↑ AUC@10° ↑ AUC@20° ↑

ROMA 31.8 53.4 70.9 VGGT 33.9 55.2 73.4

MASt3R 62.4 77.4 86.9 Ours 65.6 80.3 89.2

- Table 9. Two-view matching comparison on ScanNet-1500 Dataset. We report the AUC values with each cell colored to indicate the best and the second . Our method achieves state-of-the-art for two-view matching, surpassing all the previous methods.

Method AUC@5° ↑ AUC@10° ↑ AUC@20° ↑

SP+SG 42.2 61.2 76.0 SP+LG 49.9 67.0 80.1 LoFTR 52.8 69.2 81.2

MASt3R 73.3 84.1 90.9 Ours 73.9 84.4 91.2

- Table 10. Two-view matching comparison on MegaDepth-1500 Dataset. We report the AUC values with each cell colored to indicate the best and the second . Our method also achieves state-of-the-art for the two-view matching using the MegaDepth-1500 Dataset.

Setting Compute Cost Memory Cost Network Params w/o Shared 1.362 TFlops 4.6 GB 737.591 M

w/ Shared 1.362 TFlops 4.1 GB 624.152 M

- Table 11. Ablation on shared encoder-decoder structure. We conduct experiments for both of the model on image pairs with 512 resolution. With the shared encoder-decoder structure, our model yields lower memory cost and less network parameters.

[Figure 17]

- Fig. 14. Limitations. Despite that our method outperforms previous methods in geometric predictions, the prediction quality for thin structures still require further improvement.

[Figure 18]

### Fig. 15. More qualitative comparison of normal map. We provide more normal comparison of both object-centric and human scenes. Dens3R is able to produce more accurate and sharper results

[Figure 19]

### Fig. 16. More qualitative comparison of normal map. We provide more normal comparison of both indoor and outdoor scenes. Dens3R is able to produce sharper and more accurate results and surpasses previous methods.

[Figure 20]

### Fig. 17. Additional depth comparison. We provide more depth comparison with previous methods and our method can predict more accurate and detailed results.

[Figure 21]

### Fig. 18. Additional depth comparison with VGGT. We compare our depth prediction results with VGGT and Dens3R demonstrates more robust and accurate predictions.

[Figure 22]

Fig. 19. Additional depth comparison of indoor scenes with VGGT. Dens3R demonstrates more accurate results for human depth estimation.

[Figure 23]

- Fig. 20. Additional depth comparison of outdoor scenes with VGGT. We compare our depth prediction results of autonomous driving dataset. Our methods achieves much more accurate predictions.

[Figure 24]

### Fig. 21. Image-matching visualization. We provide the visualization of our dense and accurate image-matching results. The accuracy is also presented on the upper left of each matching image-pair.

[Figure 25]

### Fig. 22. Additional high-resolution inference comparison. We provide more high-resolution inference results to demonstrate the effectiveness of the proposed position-interpolated rotary positional encoding. We present the pointmap of the main frame and our method accomplishes to prevent the degeneration problem that occured in previous methods.

[Figure 26]

### Fig. 23. Multi-view reconstruction results. We demonstrate high-quality 3D reconstruction for various scenarios without known camera poses. The predicted camera poses are also shown on the right of the reconstruction results.

