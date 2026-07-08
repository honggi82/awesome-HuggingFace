## arXiv:2601.05573v1[cs.CV]9Jan2026

### Orient Anything V2: Unifying Orientation and Rotation Understanding

Zehan Wang1,2∗, Ziang Zhang1∗, Jiayang Xu1, Jialei Wang1, Tianyu Pang3†, Chao Du3, Hengshuang Zhao4, Zhou Zhao1,2‡ 1Zhejiang University; 2Shanghai AI Lab; 3Sea AI Lab; 4The University of Hong Kong https://orient-anythingv2.github.io/

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

Relative Rotation

Absoluate Orientation

|Fine Labels|
|---|

- ▪ Full Coverage
- ▪ Perfect Balance
- ▪ Diverse Pose
- ▪ Realistic Texture
- ▪ Strong Robustness
- ▪ Rotational Symmetry

Pseudo ensemble

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

Joint Encoder

3D Assets

share

Encoder Encoder

[Figure 9]

|[Figure 10]<br><br>[Figure 11]<br><br>| |
|---|---|

|[Figure 12]|[Figure 13]|
|---|---|

|[Figure 14]|
|---|

Generative model

Class Names

Data: Scalable Data Engine Model: Orient Anything V2 New Features: Symmetry & Rotation

Figure 1: Overview of Orient Anything V2. We upgrade the foundation orientation estimation model from both Data and Model perspectives. It uniﬁes the understanding of object orientation and rotation, achieving better estimation accuracy and gaining the New Features to handle rotational symmetry and relative rotation. Zoom in for the best view.

Abstract

This work presents Orient Anything V2, an enhanced foundation model for uniﬁed understanding of object 3D orientation and rotation from single or paired images. Building upon Orient Anything V1, which deﬁnes orientation via a single unique front face, V2 extends this capability to handle objects with diverse rotational symmetries and directly estimate relative rotations. These improvements are enabled by four key innovations: 1) Scalable 3D assets synthesized by generative models, ensuring broad category coverage and balanced data distribution; 2) An efﬁcient, model-in-the-loop annotation system that robustly identiﬁes 0 to N valid front faces for each object; 3) A symmetry-aware, periodic distribution ﬁtting objective that captures all plausible front-facing orientations, effectively modeling object rotational symmetry; 4) A multi-frame architecture that directly predicts relative object rotations. Extensive experiments show that Orient Anything V2 achieves state-of-the-art zero-shot performance on orientation estimation, 6DoF pose estimation, and object symmetry recognition across 11 widely used benchmarks. The model demonstrates strong generalization, signiﬁcantly broadening the applicability of orientation estimation in diverse downstream tasks.

∗Equal Contribution. †Project Leader. ‡Corresponding Author.

Preprint. Under review.

###### 1 Introduction

Estimating object orientation from images is a fundamental task of computer vision. 3D object orientation information plays crucial roles in robot manipulation [49, 37, 22], autonomous driving [25], AR/VR [11, 30, 4, 57], and spatial-aware image understanding [21, 28, 60] and generation [56, 50, 35].

Orient Anything V1 [48] is a foundation model for estimating the object orientation aligned with an object’s unique front face. While it exhibits strong robustness and accuracy in absolute orientation estimation, it lacks an understanding of rotation (despite its intrinsic link to orientation). This deﬁciency results in difﬁculties handling numerous rotationally symmetric objects (simply classifying them as having no front face) and understanding object rotation relative to a speciﬁed reference frame. These limitations around rotation understanding restrict its utility in many downstream tasks.

In this work, we aim to develop an enhanced orientation estimation model, Orient Anything V2, with stronger generalization and deeper understanding of both object orientation and rotation. Our contributions include a scalable data engine and a more elegant model framework.

From the data perspective, Orient Anything V1 uses advanced VLM [12, 31] to annotate real 3D assets from Objaverse [8, 7]. Building on this data-driven motivation, we leverage advanced 3D generation models [51, 61] to further speed up data scaling-up and improve the data coverage and balance. Additionally, we assemble pseudo labels predicted by the V1 model across multi-view renderings and reﬁne them through model-in-the-loop calibration. The proposed data engine enables highly cost-effective and ﬂexible data scaling up, delivers robust annotation performance, and shows a strong understanding of rotationally symmetric objects. Our ﬁnal dataset includes 600K assets, 12× larger than the existing orientation dataset, with signiﬁcantly higher annotation quality, accurately identifying 0 to N valid front faces.

From the model perspective, we ﬁrst propose symmetry-aware orientation distribution, explicitly teaching the model to capture and predict rotational symmetry. Moreover, our model supports multiframe input to directly predict relative rotations between frames. This design effectively bridges the knowledge transfer between absolute orientation and relative rotation, showing strong potential in reference-known scenarios.

Our experiments demonstrate the enhanced and novel capabilities of our model. It achieves superior performance on zero-shot orientation estimation and sets new records on zero-shot rotation estimation (i.e., 6DoF pose estimation [49, 26]), while also accurately handling and predicting different rotational symmetries.

To summarize, we propose Orient Anything V2, which improves Orient Anything V1 as follows:

- • We propose a data engine that cost-efﬁciently scales up 3D asset collection and robustly annotates the 0 to N valid front faces to capture different object rotational symmetries.
- • We introduce symmetry-aware distribution ﬁtting as a learning objective, allowing the model to directly predict all plausible object orientations.
- • We extend the model architecture to support multi-frame input, enabling it to directly estimate relative object rotations over the reference frame.
- • Our model demonstrates strong zero-shot generalization across absolute orientation estimation, relative rotation estimation, and object symmetry recognition.

###### 2 Related Work

- 2.1 Object Rotational Symmetry

Rotational symmetry [36, 34] indicates that an object may retain its original shape after being rotated by certain angles. This property is commonly found across various objects. Understanding an object’s rotational symmetry is critical for 3D object recognition and generation [24, 59], pose estimation [17, 6], and robotic manipulation [39]. While some existing works [40] attempt to detect

###### 3D rotational symmetry from single-view 2D images, they are constrained by limited training data and lack zero-shot generalization to open-world scenarios.

[Figure 15]

[Figure 16]

|低质量纹理|
|---|

|低质量纹理<br><br>[Figure 17]|
|---|

|T-pose|
|---|

|T-pose<br><br>[Figure 18]|
|---|

|T-pose<br><br>[Figure 19]|
|---|

|[Figure 20]|
|---|

(a) Low-quality Texture (b) Limited Pose&Shape

- Figure 2: Real assets from Objavese suffer from (a) low-quality texture and (b) limited realism.

Our focus is on object orientation relative to a semantic "front" face. The number of possible valid front-facing orientations an object possesses is determined by its rotational symmetry around its vertical axis. For example, 180-degree symmetry means there are two distinct valid front faces. Objects with continuous rotational symmetry (symmetric at any angle), like balls, are considered to have no meaningful direction. In this work, we broaden the applicability of orientation estimation models by enabling the prediction of an object’s azimuthal symmetry from a single 2D image. Our model demonstrates impressive zero-shot rotational symmetry recognition performance.

- 2.2 Relative Rotation Estimation

Predicting an object’s rotation in the query frame relative to the reference frame is a fundamental capability in 6DoF pose estimation [26, 14, 55] and is crucial for robotics applications. Early methods [23, 18, 9] focused on speciﬁc instances or object categories. More recent approaches like OnePose [44] and OnePose++ [15] estimate object rotation by solving 2D-3D correspondences across views. POPE [10] follows a similar idea and achieves zero-shot rotation estimation with a single reference frame with the help of SAM [19] and DINOv2 [33]. However, the reliance on pixel matching makes these methods prone to failure under large viewpoint changes.

In contrast, we propose a purely implicit learning approach. Leveraging the inherent coupling between rotation and orientation, we extend the orientation estimation model to support multi-frame inputs, enabling direct zero-shot relative rotation prediction between arbitrary views.

- 2.3 Single-view Orientation Estimation

Estimating an object’s 3D front-facing orientation (interpreted as its rotation relative to the canonical front view) from a single view, requires the model to have an inherent understanding of different objects’ standard poses and front-facing appearances. Earlier works [53, 46, 42] are mainly limited to a small number of categories or speciﬁc domains. More recently, ImageNet3D [29] introduce a large-scale dataset with manually annotated 3D orientations. Orient Anything [48] achieves robust orientation estimation for any objects in any scenes by leveraging an advanced automated annotation pipeline, improved learning objectives, and real-world knowledge from the pre-trained vision model.

In this work, we further address several limitations of Orient Anything and upgrade the orientation estimation model from both data-driven (novel and scalable data engine) and model-driven (direct symmetry and rotation prediction) perspectives, resulting in Orient Anything V2.

###### 3 Revisiting Orient Anything V1

Orient Anything V1 pioneers zero-shot object orientation estimation from single images. It introduces a VLM-based pipeline to annotate front faces of Objaverse 3D assets [8, 7], learns orientation estimation from their renderings via distribution ﬁtting. It also provides a conﬁdence score to indicate whether an object has a unique front face. To further advance the orientation prediction foundation model, we ﬁrst dig into the potential limitations in its training data and framework.

Disadvantages of Real 3D Assets 1) Imbalanced Category Distribution: Stemming from the human biases in asset creation, real 3D datasets, such as Objaverse [8, 7], suffer from signiﬁcant class imbalance. Common categories like buildings and characters make up a large proportion, while

|Class Tag|LLM|
|---|---|
| |Rewrite|

|Caption|Image|
|---|---|
| |Generation|

|Image|3D Mesh|
|---|---|
| |Generation|

|3D Mesh|
|---|

[Figure 21]

[Figure 22]

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

| |
|---|

| |
|---|

|[Figure 26]|
|---|

|Cage|
|---|

|Computer|
|---|

|Heater|
|---|

|"matte plastic, ethereal cage"|
|---|

|"muted tones, dreamlike computer"|
|---|

|"rusted, cinematic heater"|
|---|

[Figure 27]

[Figure 28]

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

| |
|---|

| |
|---|

|[Figure 32]|
|---|

|Farmer|
|---|

|Fence|
|---|

|Fighter|
|---|

|"cartoon, dramatic farmer"|
|---|

|"granite, symmetrical, fence"|
|---|

|"polished wood, pastel fighter"|
|---|

.. ..

# .. ..

# .. ..

.. ..

[Figure 33]

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

| |
|---|

##### . . .

. . .

. . .

|Fox|
|---|

|Worker|
|---|

|"gold made, ambient glow fox"|
|---|

|"dusty, realistic, sidelight,<br><br>worker"|
|---|

. . .

- Figure 3: Overview of 3D Asset Synthesis Pipeline. We begin with class tags and use a series of advanced generative models to progressively generate high-quality 3D assets.

others, like uncommon animals, are severely underrepresented. 2) Inconsistent Data Quality: Current large-scale 3D datasets often lack high-quality assets with complete geometry and rich surface details. (Fig. 2 a) Moreover, many human-created meshes exhibit ﬁxed poses, leading to a substantial domain gap from real-world object variations (Fig. 2 b).

Limitations of Object Rotation Understanding 1) Ignored Rotational Symmetries: Orient Anything V1 deﬁnes orientation based on the single, unique front face, overlooking the different rotational symmetries (i.e., multiple valid "front" faces). For the many symmetric objects in real world, the model cannot effectively distinguish or identify their potential orientations. 2) Unsupported Relative Rotations: The relative rotation between two views and the front-facing orientation (essentially the rotation relative to the front view) are inherently coupled. However, estimating relative rotation through independent absolute orientation predictions suffers from signiﬁcant error accumulation, causing Orient Anything V1 to often fail in relative rotation estimation.

- 4 Scalable Data Engine

- 4.1 3D Asset Synthesis

Motivated by the recent remarkable progress in generative models and the successful application of synthetic data in downstream tasks [45, 56], we explore whether synthetic 3D assets can serve as scalable, high-quality data sources for orientation learning. To fully harness modern generative models, we construct our asset synthesis pipeline as a structured process: Class Tag → Caption → Image → 3D Mesh, as detailed below:

Step 1: Class Tag → Caption. To ensure broad category coverage and diversity, we follow SynCLR’s approach [45], starting from ImageNet-21K [38] category tags, and use Qwen-2.5 [54] to generate rich captions that describe detailed object attributes and diverse poses. Step 2: Caption → Image. We use the state-of-the-art text-to-image model, FLUX.1-Dev [20], to generate images following the captions. Besides, we enhance captions with positional descriptors to promote explicit 3D structure and upright pose. Step 3: Image → 3D mesh. We employ the leading open-source image-to-3D model, Hunyuan-3D-2.0 [61], to produce high-quality 3D meshes from the synthesized images.

Finally, we generate 600k 3D assets in total, with approximately 30 items for each class tag in ImageNet-21K. These assets feature complete geometry, detailed textures, and balanced category coverage. In terms of scale, the new synthetic dataset is 12× larger than the ﬁltered real dataset used in Orient Anything V1.

- 4.2 Robust Annotation

Orient Anything V1 employs VLM to annotate the unique canonical front view of 3D assets. However, this approach is limited by the VLM’s underdeveloped spatial perception ability and struggles to handle diverse rotational symmetries. To address these challenges, we introduce a more effective and robust system for annotating 3D asset orientations.

|Multi-view Rendering|V1 model|
|---|---|
| |Annotating|

|Pseudo Label|Voting &|
|---|---|
| |Standardize|

|Automatic Label|Manual|
|---|---|
| |Calibration|

|Final Label|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|[Figure 41]|
|---|---|
|[Figure 42]|[Figure 43]|

|[Figure 44]|[Figure 45]|
|---|---|
|[Figure 46]|[Figure 47]|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|[Figure 58]|
|---|---|
|[Figure 59]|[Figure 60]|

|[Figure 61]|[Figure 62]|
|---|---|
|[Figure 63]|[Figure 64]|

- Figure 4: Overview of Robust Annotation Pipeline. "Pseudo Label" visualizes the azimuth direction of pseudo labels and objects in the horizontal plane. By ﬁtting the pseudo labels to standard periodic distribution, we can robustly derive the orientation and symmetry label. Human calibration is only required for categories with symmetry inconsistencies.

Intra-asset Ensemble Annotation We ﬁrst train an improved orientation estimation model as our automatic annotator, based on the Orient Anything V1 paradigm and incorporating the additional real-world orientation dataset, ImageNet3D. Next, for each 3D asset, we employ this model to produce pseudo-labels for various renderings. Finally, we project these pseudo-labels, obtained from different viewpoints, back into a canonical 3D world coordinate system.

- As shown in Fig. 4, the overall distribution of pseudo-labels on the horizon plane clearly indicates the object’s possible orientations. To capture the main direction and rotational symmetries, we ﬁrst arrange the discrete predicted azimuth angles over [0°, 360°) into a probability distribution

Ppseudo ∈ R360. This distribution is then ﬁtted to a periodic Gaussian distribution using the least squares method:

(¯φ,α,¯ σ¯) = arg min

φ,α,σ

2

 Ppesudo(i) −

 

exp cos(ασ(2i−φ)) 2πI0 σ 12

359

i=0

(1)

where σ¯ is the ﬁtted variance. The phase φ¯ ∈ [0◦,360◦) represents the main azimuth direction. The periodicity α¯ ∈ {1,2,...,N} signiﬁes 360/α¯-degree rotational symmetry, possessing α¯ valid front faces, while α¯ = 0 indicates no dominant orientation.

Ensembling multiple pseudo labels in the 3D world effectively suppresses outlier errors from singleview predictions, resulting in signiﬁcantly more reliable annotations.

Inter-assets Consistency Calibration Building on the rotational symmetry and orientation annotations for individual assets, we further perform human-in-the-loop consistency calibration across assets. Speciﬁcally, since our 3D assets are generated based on object category tags, they are naturally grouped by category. We assume that objects of the same category should share the same type of rotational symmetry. Based on this assumption, we analyze the annotated rotational symmetries within each category. If all assets within the same category demonstrate the same symmetries, we directly consider the annotations to be correct. If inconsistencies are found, we manually review all assets in that category to re-annotate or ﬁlter out incorrect annotations.

As each asset is annotated independently, the cross-asset consistency check and manual calibration offer an orthogonal perspective that efﬁciently and effectively enhances annotation reliability. Statistically, across 21k source category tags, we observe only minor inconsistencies in around 15% of categories, each involving a small number of assets. The ﬁnding further validates the accuracy and robustness of our ensemble annotation strategy.

[Figure 65]

|[Figure 66]|
|---|

[Figure 67]

[Figure 68]

[Figure 69]

|[Figure 70]|Orientation Token<br><br>[Figure 71]<br><br>[Figure 72]<br><br>Encoder|
|---|---|
| | |

[Figure 73]

#### Joint Encoder

|[Figure 74]|
|---|

[Figure 75]

[Figure 76]

|[Figure 77]|[Figure 78]<br><br>[Figure 79]<br><br>Encoder<br><br>Rotation Token|
|---|---|
| | |

[Figure 80]

- Figure 5: Framework of Orient Anything V2. One or two input frames are tokenized by DINOv2 and then jointly encoded using transformer blocks. We ﬁnally employ MLP heads to predict the orientation or rotation distributions from the encoded learnable tokens of each frame.

- 5 Framework

- 5.1 Symmetry-aware Distribution

Orient Anything V1 proposes an orientation distribution ﬁtting task that guides the model to learn circular Gaussian distributions over azimuth, polar, and in-plane rotation angles, that preserve the similarity between neighboring angles. Each angle is modeled with a unimodal target distribution centered on a unique front-facing orientation. For symmetric objects with multiple or no semantic front faces, the model additionally predicts a low orientation conﬁdence to ﬁlter them out.

To recognize different types of rotational symmetry and enable general orientation prediction for objects with multiple front faces, we further introduce the symmetry-aware periodic distribution as the training target. As discussed in Sec. 4.2, our ensemble annotation and consistency calibration approach enables accurate and robust labeling of 0 to N valid front-facing directions over the horizontal plane. To incorporate these annotations into prediction, we directly model 0 to N valid front faces within the azimuth angle distribution. This design naturally replaces V1’s extra orientation conﬁdence design. Instead, different kinds of rotational symmetries are captured directly from the predicted probability distribution. This more elegant framework enables the model to inherently share knowledge across all object categories.

For training, the target Pazi ∈ R360 for the azimuth angle, originally represented as a circular Gaussian distribution, is adapted to be periodic:

Pazi (i|φ,¯ α,σ¯ ) =

exp cos(¯ασ(2i−φ¯)) 2πI0 σ 12

(2)

where φ¯ and α¯ are the phase (azimuth angle) and periodicity (rotation symmetry) ﬁtted from Sec. 4.2, σ is the variance hyper-parameter, and i = 0◦,...,359◦ is the angle index. Target probability distributions for the polar angle Ppol ∈ R180 and in-plane rotation angle Prot ∈ R360 are constructed using a similar method, but without the periodicity parameter.

During inference, the predicted angle distributions are ﬁtted to a standard distribution model using the least squares method, similar to Eq. 1. The resulting parameters (azimuth periodicity αˆ, azimuth angle φˆ, polar angle σˆ and rotation angle δˆ), directly indicate the object’s αˆ valid front faces (i.e., symmetric with 360/α degree rotation) and their corresponding front-facing directions in 3D space.

- 5.2 Relative Rotation Estimation

To establish a connection between absolute orientation and relative rotation, enabling knowledge sharing and transferring, we modify the network architecture to support dynamic inputs from one or multiple images.

- As shown in Fig. 5, we mainly follow VGGT [47], ﬁrst using a visual encoder, DINOv2 [33], to encode each input image into K tokens, augmented with learnable tokens. The combined set of

SUN-RGBD ARKitScenes Pascal3D+ Objectron ImageNet3D† Ori_COCO Med↓ Acc30°↑ Med↓ Acc30°↑ Med↓ Acc30°↑ Med↓ Acc30°↑ Med↓ Acc30°↑ Acc↑

Model

OriAny.V1 33.94 48.5 77.58 35.8 22.90 55.0 30.67 49.6 13.34 71.3 72.4 OriAny.V2 26.00 55.4 36.48 43.2 15.02 72.7 22.62 56.4 15.26 65.2 86.4

- Table 1: Zero-shot Absolute Orientation Estimation. †: ImageNet3D is used for training Orient Anything V2. To ensure a fair comparison, the compared V1 model is ﬁne-tuned on ImageNet3D. Best results are highlighted in bold.

tokens from all frames is then passed into a uniﬁed transformer block. The ﬁnal learnable token corresponding to each frame is used for prediction. Speciﬁcally, the learnable token for the ﬁrst frame is initialized differently and is used to predict the absolute orientation using the symmetry-aware distribution described in Sec. 5.1. Tokens from subsequent frames predict the object rotation relative to the ﬁrst frame through a similar probability ﬁtting task, but without considering symmetry.

- 5.3 Training Setting

Our model is initialized from VGGT, a large feed-forward transformer with 1.2 billion parameters pre-trained on 3D geometry tasks. We repurpose its original "camera" token, designed to predict camera extrinsics, to predict object orientation and rotation. This leverages the inherent correlation between camera pose and object rotation. We train the model to ﬁt target orientation (or rotation) distributions using Binary Cross-Entropy (BCE) loss for 20k iterations. A cosine learning rate scheduler is used with an initial rate of 1e-3. Input frames are resized to 518, and random patch masking is used for data augmentation to simulate real-world occlusion. The effective batch size is set to 48, where 1-2 frames are randomly sampled for each training sample. The training dataset comprises the ImageNet3D training set and newly collected 600k synthetic assets. Furthermore, we observe that most objects exhibit only four types of rotational symmetry: {0,1,2,4}. Therefore, we restrict our training to consider only these four cases. Any ﬁtted periodicity α¯ > 4 is mapped to 0.

- 6 Experiment

- 6.1 Zero-shot Orientation Estimation

Benchmark & Baselines Predicting the 3D orientation of objects from a single image is our core focus. We mainly compare with Orient Anything V1 [48] on ImageNet3D [29] test set and unseen test datasets, SUN-RGBD [41], ARKitScenes [3], Pascal3D+ [52], Objectron [1] and the Ori_COCO [48]. Since current testing datasets often provide only one ground truth orientation, even for symmetric objects, when Orient Anything V2 predicts multiple orientations, we simply select the one closest to facing the camera as the prediction. The main evaluation metrics are the median 3D angle error (Med↓) and accuracy within 30 degrees (Acc30°↑). For Ori_COCO, where 20 samples are collected for each class and annotated within 8 horizontal orientations, recognition accuracy (Acc↑) is used.

Main Results In Tab. 1, we present the comparative results on single view-based orientation estimation. Overall, Orient Anything V2 signiﬁcantly improves upon V1, beneﬁting from diverse synthetic data and robust ensemble annotation. On the representative Ori_COCO benchmark, our method achieves 86.4% accuracy and performs well on categories where V1 struggled, such as bicycles. Achieving state-of-the-art results on numerous real-world image datasets highlights our method’s generalization ability.

- 6.2 Zero-shot Rotation Estimation

Benchmark & Baselines We benchmark zero-shot 6DoF object pose estimation performance under a single reference view. Evaluation is conducted on four widely used datasets: LINEMOD [16], YCB-Video [5], OnePose++ [15], and OnePose [44]. Objects are prepared using the cropping and matching, following [10]. Comparisons are made against three state-of-the-art zero-shot 6DoF object pose estimation methods: Gen6D [27], LoFTR [43], and POPE [10]. Standard metrics for relative object pose estimation are used: median error (Med) and accuracy within 15◦ and 30◦ (Acc15 and Acc30), computed for each sample pair.

LINEMOD YCB-Video OnePose++ OnePose Med↓ Acc30↑ Acc15↑ Med↓ Acc30↑ Acc15↑ Med↓ Acc30↑ Acc15 Med↓ Acc30↑ Acc15↑ POPE’s Sampling (Average rotation angle: 14.85°)

Model

Gen6D 44.86 36.4 9.6 54.48 23.2 7.7 35.43 41.1 15.8 17.78 89.3 38.9 LoFTR 33.04 56.2 32.4 19.54 68.6 47.8 9.01 89.1 70.3 4.35 96.3 91.8

POPE 15.73 77.0 48.3 13.94 80.1 54.4 6.27 89.6 72.8 2.16 96.2 91.1 OriAny.V2 7.82 98.07 89.7 6.07 91.6 86.4 6.18 99.7 96.6 6.76 99.7 95.7

Random Sample (Average rotation angle: 78.22°)

POPE 98.03 10.3 4.3 41.88 40.9 27.2 88.21 25.6 19.8 45.73 45.1 37.3 OriAny.V2 28.83 51.6 28.3 15.78 61.2 48.7 12.83 85.5 58.8 11.72 86.7 63.4

- Table 2: Zero-shot Relative Rotation Estimation (i.e., pose estimation with one reference view). We evaluate two strategies for sampling query-reference view pairs: (1) query-reference pairs provided by POPE [10], and (2) randomly sampled pairs. The average rotation angles between views for each sampling strategy are 14.85° and 78.22°, respectively.

Main Results Tab. 2 includes zero-shot two-view relative rotation estimation results compared with state-of-the-art pose estimation methods. With small relative rotations (using POPE’s sampling), our model achieves the overall best performance across the four datasets. More importantly, our method’s advantage is signiﬁcantly larger when the relative rotation between the query and reference frame is larger (using random sampling). The signiﬁcant performance drop of the previous method stems from the reliance on explicit feature matching, which becomes unreliable with large rotations due to less view overlap and scarcer reliable matching points. In contrast, our approach understands images from different viewpoints by considering overall meaning rather than just detailed matching. This makes it more robust to challenging large rotations.

- 6.3 Zero-shot Symmetry Recognition

Benchmark & Baselines We assess our method’s zero-shot performance in predicting object rotational symmetry in the horizontal plane. This evaluation uses the recent, large-scale 3D object datasets with rotational symmetry annotations: Omni6DPose [58], which contain 149 distinct object classes. To ensure their orientation deﬁnition aligns with our front-facing direction, we manually select a subset of 3-5 assets per category and render 2 views per 3D asset for testing. This resulted in 838 testing sample. During inference, models receive a single rendering and predict the four kinds of rotational symmetry predictions. As there are currently no dedicated zero-shot models for predicting object rotational symmetry from a single view, we employ advanced VLMs (Qwen2.5VL-72B [2], GPT-4o [31], GPT-o3 [32], and Gemini-2.5-pro [13]) as baselines. We evaluate their ability to predict horizontal plane rotational symmetry using a multiple-choice format, with recognition accuracy as the metric.

Main Results We present a comparison of our method against various advanced general VLMs for identifying object horizontal rotational symmetry in Tab. 3. Our results indicate that recognizing object rotational symmetry is a challenging problem even for the strongest VLMs, thereby limiting their ability to fully understand the 3D spatial state from 2D images. In contrast, beneﬁting from high-quality annotations and a uniﬁed learning objective, our model achieves 65% accuracy in distinguishing object rotational symmetry. Combining this strong symmetry recognition ability alongside the robust and accurate absolute orientation estimation performance demonstrated in Sec. 6.1, our model can accurately infer multiple potential orientations from a single image in real applications.

Omni6DPose Acc↑ Random 25.0

Qwen2.5VL-72B 55.8

Gemini-2.5-pro 44.4 GPT-4o 62.5 GPT-o3 53.7

OriAny. V2 65.2

Table 3: Zero-shot horizontal rotational symmetry recognition.

0 Front Face 1 Front Faces 2 Front Faces 4 Front Faces

|[Figure 81]<br><br>[Figure 82]|
|---|

|[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]|
|---|

|[Figure 86]<br><br>[Figure 87]|
|---|

|[Figure 88]<br><br>[Figure 89]|
|---|

|[Figure 90]<br><br>[Figure 91]|
|---|

|[Figure 92]<br><br>[Figure 93]|
|---|

|[Figure 94]<br><br>[Figure 95]|
|---|

|[Figure 96]<br><br>[Figure 97]|
|---|

[Figure 98]

[Figure 99]

[Figure 100]

|[Figure 101]<br><br>[Figure 102]|
|---|

|[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]|
|---|

|[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]|
|---|

|[Figure 109]<br><br>[Figure 110]|
|---|

|[Figure 111]<br><br>[Figure 112]|
|---|

|[Figure 113]<br><br>[Figure 114]|
|---|

|[Figure 115]<br><br>[Figure 116]|
|---|

|[Figure 117]<br><br>[Figure 118]|
|---|

[Figure 119]

[Figure 120]

Figure 6: Visualization of synthetic 3D assets and robust annotation.

Orientation Estimation Rotation Estimation

Assets Type

Assets Number

Initialized Weights

Objectron Ori_COCO LINEMOD YCB-Video Med↓ Acc15↑ Acc↑ Med↓ Acc15↑ Med↓ Acc15↑

Row

- 1 Real 40K VGGT 25.05 54.3 74.8 10.70 69.8 15.49 72.5
- 2 Synthetic 40K VGGT 24.44 54.6 74.6 10.16 74.1 7.28 76.2

- 3 Synthetic 200K VGGT 23.82 55.0 75.2 10.22 74.5 7.49 78.6
- 4 Synthetic 400K VGGT 25.09 53.9 75.4 9.78 76.3 6.48 80.5
- 5 Synthetic 600K VGGT 22.62 54.8 86.4 7.82 89.7 6.07 86.4

- 6 Synthetic 600K DINOv2 26.70 52.6 79.0 15.11 49.1 13.78 52.6
- 7 Synthetic 600K None 62.08 13.5 25.6 16.54 45.3 13.93 52.2 Table 4: Ablation study. For the rotation estimation, we employ POPE’s sampling pairs.

- 6.4 Ablation Study

Quality of Synthetic 3D Assets Fig. 6 visualizes our synthetic dataset and the labelled orientation, qualitatively demonstrating the high quality of both the synthetic data and its annotations. Quantitatively, Rows 1 and 2 of Tab. 4 show the comparison of training with an equal amount of annotated real or synthetic 3D assets. We observe that both data sources yield comparable results for absolute orientation estimation. However, for rotation estimation (on LINEMOD and YCB-Video), training with synthetic assets provides a signiﬁcant advantage. This may be because synthetic assets possess richer, more realistic textures, which are more crucial for understanding rotation.

Effect of Scaling Data In Rows 2, 3, 4, and 5 of Tab. 4, we explore the impact of data scale on the performance of ﬁnal orientation and rotation estimation. Overall, with the same training step, encountering more diverse data and 3D assets during training leads to better overall performance. Speciﬁcally, we ﬁnd that rotation estimation is more sensitive to data scale than orientation estimation. This may be because orientation relies on overall semantics and structure, while rotation estimation requires understanding diverse textures and ﬁne-grain details to capture cross-view relationships.

Effect of Geometry Pre-training Tab. 4 (Rows 5-7) presents our experiments of different model initialization strategies. Training without any pre-trained initialization yields the worst results. Initializing the separated visual encoder with DINOv2 introduces valuable high-quality semantic and object structure information, leading to substantial performance gains. We observe further improvements in rotation estimation by using VGGT, pre-trained speciﬁcally on 3D geometric tasks, which boosts the model’s comprehension of object geometry.

- 7 Conclusion

We present Orient Anything V2, an advanced model for uniﬁed object orientation and rotation understanding. Through introducing the scalable data engine, a symmetry-aware distribution learning target, and a multi-frame framework, our model enables: 1) Stronger single-view absolute orientation estimation. 2) Advanced two-frame object relative pose rotation estimation. 3) Powerful object horizontal rotational symmetry recognition. In practice, the model can simultaneously and accurately predict multiple valid front faces of objects, making it well-suited for diverse objects and real-world application scenarios.

Limitation While our models exhibit strong generalization to diverse in-the-wild objects in real images, we ﬁnd that the inherent ambiguity of monocular images leads to less accurate predictions in views with very low information or severe occlusion. Furthermore, the current framework supports a maximum of two input frames. Extending the model to handle more frames will be an important direction for supporting video understanding applications.

Acknowledgements

This work was supported in part by National Key R&D Program of China (No. 2022ZD0162000) and National Natural Science Foundation of China (No. 62222211, U24A20326, 624B2128, 62422606 and 62201484)

References

- [1] Adel Ahmadyan, Liangkai Zhang, Artsiom Ablavatski, Jianing Wei, and Matthias Grundmann. Objectron: A large scale dataset of object-centric videos in the wild with pose annotations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7822–7831, 2021.
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [3] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021.
- [4] Lonni Besançon, Anders Ynnerman, Daniel F Keefe, Lingyun Yu, and Tobias Isenberg. The state of the art of spatial interfaces for 3d visualization. In Computer Graphics Forum, volume 40, pages 293–326. Wiley Online Library, 2021.
- [5] Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M Dollar. The ycb object and model set: Towards common benchmarks for manipulation research. In 2015 international conference on advanced robotics (ICAR), pages 510–517. IEEE, 2015.
- [6] Enric Corona, Kaustav Kundu, and Sanja Fidler. Pose estimation for objects with rotational symmetry. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7215–7222. IEEE, 2018.
- [7] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023.
- [8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023.
- [9] Ufuk Efe, Kutalmis Gokalp Ince, and Aydin Alatan. Dfm: A performance baseline for deep feature matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4284–4293, 2021.
- [10] Zhiwen Fan, Panwang Pan, Peihao Wang, Yifan Jiang, Dejia Xu, and Zhangyang Wang. Pope: 6-dof promptable pose estimation of any object in any scene with one reference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7771–7781, 2024.
- [11] Aaron L Gardony, Shaina B Martis, Holly A Taylor, and Tad T Brunyé. Interaction strategies for effective augmented reality geo-visualization: Insights from spatial cognition. Human–Computer Interaction, 36(2):107–149, 2021.
- [12] Google. Gemini-2.0-ﬂash, 2025. https://aistudio.google.com/prompts/new_chat?model= gemini-2.0-flash-exp.
- [13] Google. Gemini-2.5-pro, 2025. https://deepmind.google/technologies/gemini/pro/.
- [14] Qi Guan, Zihao Sheng, and Shibei Xue. Hrpose: Real-time high-resolution 6d pose estimation network using knowledge distillation. Chinese Journal of Electronics, 32(1):189–198, 2023.

- [15] Xingyi He, Jiaming Sun, Yuang Wang, Di Huang, Hujun Bao, and Xiaowei Zhou. Onepose++: Keypointfree one-shot object pose estimation without cad models. Advances in Neural Information Processing Systems, 35:35103–35115, 2022.
- [16] Stefan Hinterstoisser, Vincent Lepetit, Slobodan Ilic, Stefan Holzer, Gary Bradski, Kurt Konolige, and Nassir Navab. Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes. In Asian conference on computer vision, pages 548–562. Springer, 2012.
- [17] Tomas Hodan, Daniel Barath, and Jiri Matas. Epos: Estimating 6d pose of objects with symmetries. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11703–11712, 2020.
- [18] Wei Jiang, Eduard Trulls, Jan Hosang, Andrea Tagliasacchi, and Kwang Moo Yi. Cotr: Correspondence transformer for matching across images. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6207–6217, 2021.
- [19] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.
- [20] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [21] Phillip Y Lee, Jihyeon Je, Chanho Park, Mikaela Angelina Uy, Leonidas Guibas, and Minhyuk Sung. Perspective-aware reasoning in vision-language models via mental imagery simulation. arXiv preprint arXiv:2504.17207, 2025.
- [22] Dingzhe Li, Yixiang Jin, Yuhao Sun, Hongze Yu, Jun Shi, Xiaoshuai Hao, Peng Hao, Huaping Liu, Fuchun Sun, Jianwei Zhang, et al. What foundation models can bring for robot learning in manipulation: A survey. arXiv preprint arXiv:2404.18201, 2024.
- [23] Jiankun Li, Peisen Wang, Pengfei Xiong, Tao Cai, Ziwei Yan, Lei Yang, Jiangyu Liu, Haoqiang Fan, and Shuaicheng Liu. Practical stereo matching via cascaded recurrent network with adaptive correlation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16263–16272, 2022.
- [24] Xiang Li, Zixuan Huang, Anh Thai, and James M Rehg. Symmetry strikes back: From single-image symmetry detection to 3d generation. arXiv preprint arXiv:2411.17763, 2024.
- [25] Chuang Lin, Bingbing Zhuang, Shanlin Sun, Ziyu Jiang, Jianfei Cai, and Manmohan Chandraker. Drive-1to-3: Enriching diffusion priors for novel view synthesis of real vehicles. arXiv preprint arXiv:2412.14494, 2024.
- [26] Jian Liu, Wei Sun, Hui Yang, Zhiwen Zeng, Chongpei Liu, Jin Zheng, Xingyu Liu, Hossein Rahmani, Nicu Sebe, and Ajmal Mian. Deep learning-based object pose estimation: A comprehensive survey. arXiv preprint arXiv:2405.07801, 2024.
- [27] Yuan Liu, Yilin Wen, Sida Peng, Cheng Lin, Xiaoxiao Long, Taku Komura, and Wenping Wang. Gen6d: Generalizable model-free 6-dof object pose estimation from rgb images. In European Conference on Computer Vision, pages 298–315. Springer, 2022.
- [28] Wufei Ma, Haoyu Chen, Guofeng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. arXiv preprint arXiv:2412.07825, 2024.
- [29] Wufei Ma, Guofeng Zhang, Qihao Liu, Guanning Zeng, Adam Kortylewski, Yaoyao Liu, and Alan Yuille. Imagenet3d: Towards general-purpose object-level 3d understanding. Advances in Neural Information Processing Systems, 37:96127–96149, 2024.
- [30] Pedro Monteiro, Guilherme Gonçalves, Hugo Coelho, Miguel Melo, and Maximino Bessa. Hands-free interaction in immersive virtual reality: A systematic review. IEEE Transactions on Visualization and Computer Graphics, 27(5):2702–2713, 2021.
- [31] OpenAI. Gpt-4o, 2025. https://openai.com/index/introducing-4o-image-generation/.
- [32] OpenAI. Gpt-o3, 2025. https://openai.com/index/introducing-o3-and-o4-mini/.
- [33] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

- [34] Jonathan Palacios and Eugene Zhang. Rotational symmetry ﬁeld design on surfaces. ACM Transactions on Graphics (TOG), 26(3):55–es, 2007.
- [35] Karran Pandey, Paul Guerrero, Matheus Gadelha, Yannick Hold-Geoffroy, Karan Singh, and Niloy J Mitra. Diffusion handles enabling 3d edits for diffusion models by lifting activations to 3d. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7695–7704, 2024.
- [36] V Shiv Naga Prasad and Larry S Davis. Detecting rotational symmetries. In Tenth IEEE International Conference on Computer Vision (ICCV’05) Volume 1, volume 2, pages 954–961. IEEE, 2005.
- [37] Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, et al. Sofar: Language-grounded orientation bridges spatial reasoning and object manipulation. arXiv preprint arXiv:2502.13143, 2025.
- [38] Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for the masses. arXiv preprint arXiv:2104.10972, 2021.
- [39] Yifei Shi, Zixin Tang, Xiangting Cai, Hongjia Zhang, Dewen Hu, and Xin Xu. Symmetrygrasp: Symmetryaware antipodal grasp detection from single-view rgb-d images. IEEE Robotics and Automation Letters, 7(4):12235–12242, 2022.
- [40] Yifei Shi, Xin Xu, Junhua Xi, Xiaochang Hu, Dewen Hu, and Kai Xu. Learning to detect 3d symmetry from single-view rgb-d images with weak supervision. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4882–4896, 2022.
- [41] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 567–576, 2015.
- [42] Hao Su, Charles R Qi, Yangyan Li, and Leonidas J Guibas. Render for cnn: Viewpoint estimation in images using cnns trained with rendered 3d model views. In Proceedings of the IEEE international conference on computer vision, pages 2686–2694, 2015.
- [43] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8922–8931, 2021.
- [44] Jiaming Sun, Zihao Wang, Siyu Zhang, Xingyi He, Hongcheng Zhao, Guofeng Zhang, and Xiaowei Zhou. Onepose: One-shot object pose estimation without cad models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6825–6834, 2022.
- [45] Yonglong Tian, Lijie Fan, Kaifeng Chen, Dina Katabi, Dilip Krishnan, and Phillip Isola. Learning vision from models rivals learning vision from data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15887–15898, 2024.
- [46] He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, and Leonidas J Guibas. Normalized object coordinate space for category-level 6d object pose and size estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2642–2651, 2019.
- [47] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651, 2025.
- [48] Zehan Wang, Ziang Zhang, Tianyu Pang, Chao Du, Hengshuang Zhao, and Zhou Zhao. Orient anything: Learning robust object orientation estimation from rendering 3d models. ICML, 2025.
- [49] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchﬁeld. Foundationpose: Uniﬁed 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17868–17879, 2024.
- [50] Ziyi Wu, Yulia Rubanova, Rishabh Kabra, Drew Hudson, Igor Gilitschenski, Yusuf Aytar, Sjoerd van Steenkiste, Kelsey Allen, and Thomas Kipf. Neural assets: 3d-aware multi-object scene synthesis with image diffusion models. Advances in Neural Information Processing Systems, 37:76289–76318, 2024.
- [51] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024.
- [52] Yu Xiang, Roozbeh Mottaghi, and Silvio Savarese. Beyond pascal: A benchmark for 3d object detection in the wild. In IEEE winter conference on applications of computer vision, pages 75–82. IEEE, 2014.

- [53] Yang Xiao, Vincent Lepetit, and Renaud Marlet. Few-shot object detection and viewpoint estimation for objects in the wild. IEEE transactions on pattern analysis and machine intelligence, 45(3):3090–3106, 2022.
- [54] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [55] Honghong Yang, Hongxi Liu, Yumei Zhang, and Xiaojun Wu. Fmr-gnet: Forward mix-hop spatial-temporal residual graph network for 3d pose estimation. Chinese Journal of Electronics, 33(6):1346–1359, 2024.
- [56] Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han. Hi3dgen: High-ﬁdelity 3d geometry generation from images via normal bridging. arXiv preprint arXiv:2503.22236, 3, 2025.
- [57] Difeng Yu, Xueshi Lu, Rongkai Shi, Hai-Ning Liang, Tilman Dingler, Eduardo Velloso, and Jorge Goncalves. Gaze-supported 3d object manipulation in virtual reality. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pages 1–13, 2021.
- [58] Jiyao Zhang, Weiyao Huang, Bo Peng, Mingdong Wu, Fei Hu, Zijian Chen, Bo Zhao, and Hao Dong. Omni6dpose: A benchmark and model for universal 6d object pose estimation and tracking. In European Conference on Computer Vision, pages 199–216. Springer, 2024.
- [59] Zhaoxuan Zhang, Bo Dong, Tong Li, Felix Heide, Pieter Peers, Baocai Yin, and Xin Yang. Single depth-image 3d reﬂection symmetry and shape prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8896–8906, 2023.
- [60] Zheyuan Zhang, Fengyuan Hu, Jayjun Lee, Freda Shi, Parisa Kordjamshidi, Joyce Chai, and Ziqiao Ma. Do vision-language models represent space and how? evaluating spatial frame of reference under ambiguities. arXiv preprint arXiv:2410.17385, 2024.
- [61] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.

A More Visualizations of Images in The Wild

In 7 8, 9, 10, 11, 12, 13, we present more visualizations of images from various domains containing different objects. In these images, our model shows strong abilities in single-view absolute orientation estimation, powerful object horizontal rotational symmetry recognition and two-frame object relative pose rotation estimation, further highlighting the impressive zero-shot capability of Orient Anything V2.

[Figure 121]

Figure 7: Relative pose rotation estimation for images in the wild.

[Figure 122]

- Figure 8: Orientation estimation and Rotational symmetry recognition results on objects has no front direction.

[Figure 123]

- Figure 9: Orientation estimation and Rotational symmetry recognition results on objects has one

- front direction. Part 1.

[Figure 124]

- Figure 10: Orientation estimation and Rotational symmetry recognition results on objects has one
- front direction. Part 2.

[Figure 125]

###### Figure 11: Orientation estimation and Rotational symmetry recognition results on objects has two front direction. Part 1.

[Figure 126]

###### Figure 12: Orientation estimation and Rotational symmetry recognition results on objects has two front direction. Part 2.

[Figure 127]

###### Figure 13: Orientation estimation and Rotational symmetry recognition results on objects has four front direction.

- 1 A More Visualizations of Images in The Wild
- 2 In 1 2, 3, 4, 5, 6, 7, we present more visualizations of images from various domains containing
- 3 different objects. In these images, our model shows strong abilities in single-view absolute orientation
- 4 estimation, powerful object horizontal rotational symmetry recognition and two-frame object relative
- 5 pose rotation estimation, further highlighting the impressive zero-shot capability of Orient Anything
- 6 V2.

[Figure 128]

Figure 1: Relative pose rotation estimation for images in the wild.

[Figure 129]

###### Figure 2: Orientation estimation and Rotational symmetry recognition results on objects has no front direction.

[Figure 130]

###### Figure 3: Orientation estimation and Rotational symmetry recognition results on objects has one

[Figure 131]

###### Figure 4: Orientation estimation and Rotational symmetry recognition results on objects has one front direction. Part 2.

[Figure 132]

###### Figure 5: Orientation estimation and Rotational symmetry recognition results on objects has two

[Figure 133]

###### Figure 6: Orientation estimation and Rotational symmetry recognition results on objects has two front direction. Part 2.

[Figure 134]

###### Figure 7: Orientation estimation and Rotational symmetry recognition results on objects has four front direction.

