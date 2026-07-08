## VGGHeads: 3D Multi Head Alignment with a Large-Scale Synthetic Dataset

Orest Kupyn1,2 Eugene Khvedchenia3 Christian Rupprecht1 1 University of Oxford 2 Piñata Farms, Los Angeles, USA 3 Ukrainian Catholic University

# arXiv:2407.18245v3[cs.CV]17Jun2026

[Figure 1]

Figure 1. VGGHeads Top: Our fully synthetic large-scale dataset for 3D Multi Head Alignment with a wide variety of scenes, numbers of people, and rich annotations for every human head. Bottom: The model trained on this fully synthetic data generalizes to in-the-wild scenes.

### Abstract

Human head detection, keypoint estimation, and 3D head model fitting are essential tasks with many downstream applications. However, recent progress has been limited, as real-world detection datasets face significant privacy and ethical barriers, while existing crop-based datasets are unable to capture the multi-person scenarios required for real-world applications. Here, we introduce VGGHeads—a large-scale synthetic dataset of over 1 million multi-person images generated with controllable diffusion models, providing comprehensive 3D annotations for every head while significantly reducing privacy concerns. Using this dataset, we train a novel unified architecture that performs simultaneous head detection and coarse 3D mesh estimation from unconstrained images in a single forward pass. Extensive evaluations demonstrate that models trained exclusively on synthetic data achieve state-of-the-art performance on realworld benchmarks, proving that synthetic datasets can match the effectiveness of traditional data collection approaches Furthermore, the versatility of our dataset provides a robust foundation for downstream detailed reconstruction methods while enabling diverse head modeling tasks.

### 1. Introduction

The demand for high-quality datasets has surged across computer vision, particularly for tasks involving human head representations. Accurate head modeling is crucial for applications from facial recognition [11] and animation [67] to augmented reality and medical imaging. Traditional datasets often fall short, focusing narrowly on specific aspects, such as facial landmarks [79], or offering limited resolution and annotation types.

Head processing traditionally follows a multi-stage pipeline: face detection, landmark detection, and 3D alignment on cropped regions. This sequential approach has fundamental limitations. Face detection operates within limited pose ranges and fails to capture the necessary variability for accurate 3D modeling. Importantly, a significant gap exists between coarse detection and fine-grained reconstruction, requiring practitioners to bridge representation levels through separate, specialized models.

Severe data limitations compound methodological challenges. Established datasets, such as VGG-Face [47], FFHQ [28], and CelebA [41], assume tightly cropped, pre-aligned single faces under controlled conditions. While enabling advances in face recognition and generation, they cannot

address multi-person, unconstrained scenarios, prevalent in real-world applications. Privacy and ethical challenges further hold back progress. Unlike some other computer vision domains, facial data requires explicit consent, creating fundamental barriers. Major benchmarks have been withdrawn: MS-Celeb-1M [20], VGG-Face [47], and DukeMTMC [52] due to consent violations, while ImageNet [10] required face blurring in 2021 [68]. These limitations create a dual bottleneck: existing methods cannot effectively bridge the detection-reconstruction gap, while privacy constraints prevent the collection of diverse, well-annotated datasets needed for unified models. Synthetic data generation offers a unique solution, enabling diverse multi-person scenarios with perfect annotations while eliminating privacy concerns.

We introduce VGGHeads, bridging coarse head detection and parametric 3D modeling through end-to-end learning. Our unified model processes unconstrained images and outputs complete 3D head representations for all visible people in a single forward pass, eliminating the need for separate detection, cropping, and reconstruction stages while jointly optimizing from bounding boxes to parametric 3D mesh parameters. To enable this approach, we construct a large-scale synthetic dataset using latent diffusion models conditioned on 2D body skeletons from real image collections. Our dataset comprises over 1 million multi-person scenes with comprehensive 3D annotations for every head. Conditioning on 2D body skeletons effectively models diverse in-the-wild scenes while eliminating privacy concerns.

Our contributions are as follows:

Unified Multi-Scale Head Representation: We present the first end-to-end real-time model that bridges the gap between coarse head detection and fine-grained 3D reconstruction, jointly optimizing bounding boxes, 3D vertices, pose, and landmarks in a single forward pass through knowledge distillation from sequential pipelines, demonstrating superior performance over the original teacher while eliminating separate cropping and alignment stages.

Large-Scale Multi-Person Synthetic Dataset: We introduce a comprehensive dataset of over 1 million images with 2.2 million annotated heads, with each head annotated with a detailed 3D head mesh. Unlike existing cropped and pre-aligned datasets, our data contains multiple faces per frame. Compared to detection benchmarks, it is over 20 times larger than WIDERFace [69] (32,303 images), while providing substantially more information per frame. The detailed representation enables various downstream tasks from face detection to 3D alignment, while generation with diffusion models [50] eliminates privacy concerns.

Synthetic-to-Real Generalization: We demonstrate that models trained exclusively on synthetic data achieve state-ofthe-art performance on real-world benchmarks, establishing that synthetic datasets can exceed the traditional real-data approaches. This proves the viability of privacy-preserving

synthetic data pipelines for advancing computer vision research.

### 2. Related Work

Head Detection: The simplest head representation is a face bounding box. Early detectors used hand-crafted features [8, 59], but deep learning significantly reshaped the field [51]. Current methodologies are categorized into singlestage [27, 38] and two-stage methods [51]. Single-stage approaches like S3FD [74] and PyramidBox [57] efficiently detect smaller facial structures. Two-stage methods based on Faster R-CNN [51] and R-FCN [7] integrate multi-scale techniques [61, 71]. However, these approaches are limited to facial areas and struggle with side poses and out-ofdistribution scenarios. HollywoodHeads [60] extends to full head detection with movie scenes, while SCUT-Head [49] introduces 4,405 classroom images with cascade detection. Both datasets are limited to specific scenes, not reflecting real-world distributions. RetinaFace [12] improved accuracy through multi-task learning of bounding boxes and facial landmarks, taking a first step toward general head representation. img2pose [3] directly regresses 6DoF face pose, recovering bounding boxes and 3D head pose. However, these methods only predict pose or limited facial landmarks, insufficient for comprehensive head modeling.

3D Morphable Model: Early 3DMMs provided generalpurpose head representations [4]—statistical models of 3D facial shapes and textures from scanned faces enabling realistic 3D model creation with few parameters. The Basel Face Model [48] uses matrix decomposition with parameters from 200 scans. Recent models like FLAME [34] cover full 3D head meshes trained on larger datasets. Several methods address regressing 3DMM parameters from head crops. RingNet [55] estimates 3D face shape without direct 3D supervision. 3DDFA [18] uses Cascaded CNNs for dense 3DMM prediction, while 3DDFA-v2 [19] improves through meta-joint optimization. DAD-3DHeads [44] introduces the first in-the-wild 3D head dataset with direct supervision. DECA [17], EMOCA [9], and MICA [80] address finer geometry modeling but require robust detection and alignment for high-quality crop extraction. VGGHeads bridges this gap, providing robust multi-person head detection and coarse 3D estimates that serve as reliable initialization for detailed reconstruction approaches. RetinaFace [12] first recovers 3D landmarks end-to-end but is limited to facial areas and uses only 5 ground-truth 2D landmarks for pseudo 3D information, lacking consistency on challenging samples. PIXIE [16] and PyMAF-X [72] attempt joint head and body reconstruction but model limited parameters, lack robustness on out-of-distribution poses, and rely on separate coarse predictions.

Lack of large-scale datasets for direct head mesh regression and task complexity due to high-dimensional represen-

[Figure 2]

- Figure 2. Data Generation. The predicted 2D body pose [2] and scene description [33] condition the image generation process. Binary detection model predicts head bounding boxes and 3DMM regressor [44] generates final annotation for each head crop.

tations limits progress. VGGHeads addresses these issues with a large-scale synthetic dataset containing dense 3D annotations and, to our knowledge, the first model to directly recover multi-person head meshes from images.

Synthetic Data Generation: Early synthetic data generation used 3D rendering engines for 2D vision problems. These approaches were constrained by 3D model domains and required modifications for each dataset and subtask. Virtual KITTI [5] is limited to street driving scenes. Recent methods using generative adversarial networks (GANs) [66] offer greater flexibility and generalization. However, these primarily sample from original dataset distributions, limiting their ability to incorporate new information. Diffusion-based methods like Diffumask [65] and DatasetDM [64] generate synthetic images and annotations for semantic segmentation and depth prediction. Instance Augmentation [30] generates separate objects in images, providing frameworks to augment and anonymize datasets. However, these methods require fine-tuning for specific datasets and have limited application to tasks with low ground truth data.

Synthetic Face Data: Synthetic face generation typically uses rendering engines. Fake It Till You Make It [62] releases the first large-scale synthetic dataset combining procedurally generated 3D face models with hand-crafted assets to render training images with 3D head meshes and 2D landmarks. However, the dataset contains only tight face crops and requires manual mesh texture generation, limiting scalability. Alternative approaches manipulate 2D images instead of 3D graphics pipelines. Some methods [79] fit 3DMMs to face images and warp them for head pose augmentation, while others [46] composite hand images onto faces to improve detection. These approaches make only minor adjustments to existing images, limiting utility. In contrast, VGGHeads directly samples from large diffusion model distribution, easily scaling to arbitrary numbers of training samples and scene types while capturing diverse multi-person scenarios impossible with crop-based approaches.

### 3. Dataset

Our goal is to create a large-scale dataset of image-label pairs where each image contains multiple people and every visible head is annotated with precise bounding boxes and complete 3D morphable model parameters. Unlike existing crop-based datasets assuming pre-aligned single faces, our dataset captures unconstrained multi-person scenarios essential for real-world applications. We generate fully synthetic images using pretrained latent diffusion models [50], then employ multi-stage annotation and filtering to ensure high-quality 3D annotations.

#### 3.1. Generation Pipeline

The dataset generation process consists of the following stages: (1) images are generated with a latent diffusion model conditioned on a large real-world dataset, (2) a small subset of data is manually labeled with head bounding boxes to train a binary head detector on synthetic data, (3) for each detected head in generated images we predict the 3D head model parameters and (4) multi-level quality filtering ensures dataset reliability. The full pipeline is illustrated in Fig. 2.

Image Generation: Latent diffusion models have achieved remarkable progress in image generation [24, 50]. However, generating diverse multi-person scenes with coherent spatial compositions remains challenging [65]. To create a dataset that generalizes to in the wild settings, we require scenes with varied backgrounds, different numbers of people, and complex inter-person interactions. We address this through controllable generation using 2D human pose skeletons as an intermediate representation. Body poses encode positions, scales, and spatial relationships of multiple people, providing the structural control necessary for reliable multiperson scene generation. We condition image generation on poses extracted from LAION [56] using pose estimation [2], ensuring our synthetic scenes reflect real-world spatial distributions. To eliminate privacy concerns, we use BLIP-VQA [33] to generate subject-neutral scene descriptions rather than using original captions that may contain identifying information. This approach also enables bias reduction by

- Table 1. Dataset Comparison. Overview of prominent face/head datasets showing scale, multi-person capability, and annotation types. VGGHeads combines large scale with comprehensive multi-person 3D annotations, enabling unified multi-task head modeling.

##### Dataset Images Heads Multi-Person Head Box Face Box 3D Landmarks Status/Issues

DAD-3DHeads [44] 44,898 44,898 ✗ ✓ ✓ ✓(5,023) Cropped & Aligned CelebA [41] 202,599 202,599 ✗ ✗ ✓ ✗ Cropped & Aligned VGG-Face [47] 2,600,000 2,600,000 ✗ ✗ ✓ ✗ Withdrawn MS-Celeb-1M [20] 10,000,000 10,000,000 ✗ ✗ ✓ ✗ Withdrawn

FDDB [26] 2,845 5,171 ✓ ✓ ✗ ✗ Limited scenes WIDER FACE [69] 32,203 393,703 ✓ ✗ ✓ ✗ Face Labels Only HollywoodHeads [60] 224,740 369,846 ✓ ✓ ✗ ✗ Limited scenes

VGGHeads (Ours) 1,022,944 2,219,146 ✓ ✓ ✓ ✓(5,023) Synthetic/Private

modifying prompts to ensure diverse demographic sampling. The T2I-Adapter [45] injects skeleton maps into SDXL [50] decoder blocks, generating high-resolution synthetic images up to 1280×1280 pixels with precise control over human placement and pose diversity.

Annotation Process: Existing face detectors [12] are optimized for frontal regions and fail on extreme poses and multi-person scenarios. We manually annotate 10,000 uniformly selected images with full head bounding boxes—the smallest axis-parallel rectangles encompassing all visible head pixels, including hair and occlusions. This serves dual purposes: training a robust binary head detector and establishing annotation standards. We remove harmful or privacy-sensitive content during manual review. RT-DETR [43] trained on this subset achieves 0.925 mAP on 2000 validation images, demonstrating feasibility of training robust detectors on synthetic data. Using our detector, we locate and crop all heads for 3D annotation. We employ FLAME [34], which parameterizes head geometry using 413 disentangled parameters covering shape, expression, and pose. The state-of-the-art FLAME regressor [44] generates comprehensive annotations including mesh vertices, pose rotations, and expression coefficients.

Multi-Stage Quality Filtering: Despite advances in diffusion models, generation failures can occur, particularly in complex multi-person scenes. We implement a comprehensive four-stage filtering pipeline specifically designed to identify and remove problematic samples while maintaining high recall of valid data.

Basic Validity: Remove images with zero detected heads, indicating complete generation failure.

Consistency Verification: Apply head detection to both original and horizontally flipped images. Remove samples where head counts differ, indicating unstable detections or generation artifacts.

Cross-Method Validation: Verify that our head detector’s outputs are consistent with face detection using RetinaFace [12]. Remove images where detected faces have no spa-

tial overlap with detected heads, catching cases where head detection fails while face detection succeeds.

Scale Consistency: Split images vertically and verify that the sum of heads detected in both halves equals the total in the original image. This removes images with many tiny, often deformed heads due to resolution limitations.

Our pipeline achieves 97.3% recall, treating minor artifacts as beneficial data augmentation given our coarse task nature. This ensures dataset scale while guaranteeing taskrelevant quality.

#### 3.2. Safety & Privacy

Traditional face datasets face significant privacy challenges, with major benchmarks withdrawn due to consent violations. Our synthetic approach eliminates these concerns while enabling unlimited scalability.

Content Safety: We implement multi-layered harmful content prevention: filter source data, employ CLIP-based NSFW filtering [1] with high sensitivity, incorporate negative embeddings for problematic keywords, and apply opensource NSFW classification [31] achieving 99.49% recall and 83.37% precision on 2,101 manually verified images.

Content Privacy: Controlled experiments fine-tuning Stable Diffusion on RaFD [32] with 61 individuals show that multi-person training does not enable identity memorization and re-identification models [11] fail to match generated faces to training individuals. However, single-person finetuning enables memorization, suggesting identity information is stored in text encoders linked to names. We mitigate residual risks using GliNER [70] to detect and remove personal names, BLIP [33] for privacy-neutral descriptions, and filtering against CelebA [41] celebrity databases. This multicomponent approach eliminates privacy concerns causing major dataset withdrawals.

#### 3.3. Dataset Statistics

From 1.7 million LAION-FACE [75] images, we filter 20.6% for privacy/NSFW content, retaining only pose skeletons

[Figure 3]

- Figure 3. Model Architecture. VGGHeads extends YOLO-NAS [2] architecture to predict the 3D Morphable Model parameters along with the head bounding boxes from the multi-scale feature maps (D1-D3).

and neutral descriptions while discarding visual data. After generating 1.3 million synthetic images and applying quality filtering (removing 19.9%), our final dataset contains 1,022,944 images with 2,219,146 annotated heads, over 20× larger than WIDERFace [69] while providing complete 3D annotations for every head. The generation process required 4,000 GPU hours but demonstrates the scalability: our pipeline can generate unlimited additional data as computational resources permit, offering an alternative to privacyconstrained real datasets.

- 4. Method

metric structure: shape, expression, jaw pose, global head pose, translation, and scale. Each module operates on shared multi-scale features, enabling efficient computation while preserving semantic meaning of parameter groups.

#### 4.1. Unified Multi-Scale Architecture

VGGHeads bridges the representation gap between coarse detection and parametric 3D modeling. The YOLO-NAS backbone extracts hierarchical features at multiple scales, enabling head detection across wide size ranges. Rather than treating detection and 3D estimation as separate problems, we jointly optimize both through shared feature representations. Multi-scale design is crucial for multi-person scenarios where head sizes vary dramatically. Traditional crop-based methods struggle with scale variation due to normalized input assumptions. Our approach naturally handles scale diversity by leveraging multi-scale pyramids inherent in modern detection architectures. Each detection head predicts traditional outputs (classification scores, bounding boxes) and 3D regression outputs (FLAME parameters). Joint prediction ensures 3D estimates are geometrically consistent with detected bounding boxes, eliminating misalignment issues in sequential approaches.

We present a unified architecture combining head detection and coarse 3D mesh estimation in a single forward pass, eliminating sequential processing pipelines. We extend object detection architectures to predict comprehensive 3D morphable model parameters alongside standard bounding box outputs. Building upon YOLO-NAS [2], we augment the real-time detector to regress FLAME [34] parameters directly from multi-scale feature maps. This design preserves single-stage detector efficiency while enabling comprehensive 3D modeling suitable for real-time applications. VGGHeads distills a traditional sequential pipeline (head detection → cropping → 3D regression) into a unified model that outperforms the original teacher through joint optimization and error accumulation elimination. For each detected head, our model predicts a complete vector of 3DMM parameters disentangled into shape, expression, and pose components. This parametric representation enables recovery of head and face bounding boxes, 3D mesh vertices, pose rotations, and facial landmarks through differentiable rendering. Compared to existing methods with limited representations, our approach provides a unified foundation for diverse downstream tasks. The architecture Fig. 3) employs six specialized prediction heads decoding FLAME’s para-

#### 4.2. Objective Functions

We design a multi-component loss providing supervision for all aspects of our unified representation. The loss combines standard detection objectives with novel 3D geometry losses, enabling end-to-end optimization. Our loss consists of five components: two detection losses (Classification LC and Bounding Box Regression Lbbox) and three 3D losses (3D Vertices L3D, Rotation LR, and Reprojection Lproj). This combination ensures accurate detection and precise 3D estimation while maintaining computational efficiency.

Reprojection Loss: We measure the discrepancy between reprojected 3D vertices and ground truth 2D keypoint

[Figure 4]

Figure 4. Versatility. Our single model predicts many types of head annotations and works across all datasets.

coordinates, providing direct supervision for the camera projection model:

1 N

Lreproj(vp,vgt) =

N

i=1

vpi − vgti

where N is the number of keypoints. This loss ensures that our 3D predictions project correctly onto the image plane, maintaining geometric consistency.

3D Vertices Loss: Following DAD-3D [44], we compute L2 loss over normalized, canonical 3D head vertices. We set global rotation to zero to evaluate shape and expression predictions independently of pose:

1 N

L3D(vp,vgt) =

N

i=1

vpi|R=0 − vgti 2

We subsample vertices by removing ears, eyeballs, and neck regions, then normalize both predicted and ground truth meshes to unit cubes. This canonical comparison enables robust shape supervision across different head sizes and orientations.

Rotation Loss: Rather than using standard Euler angle representations that suffer from discontinuities, we predict 6D rotation representations and apply geodesic distance loss on the resulting rotation matrices. This approach respects the SO(3) manifold geometry:

LR(Rp,Rgt) = cos−1

tr(RpRgtT ) − 1 2

The geodesic distance measures the shortest path between rotations, providing stable gradients.

Detection Losses: We employ focal loss Lc [37] for classification, addressing class imbalance in multi-person scenarios, and Complete IoU loss Lreg [76] for bounding box regression, incorporating center distance and aspect ratio for improved localization.

The final loss combines all components with learned weights: L = α3DL3D +αRLR +αreprojLreproj +αcLc +αregLreg

Our ablation studies (Table 7) demonstrate that each loss component contributes meaningfully to final performance, with the 3D losses enabling substantially better geometric understanding compared to detection-only baselines.

#### 4.3. Implementation Details

We implement our model in PyTorch, initializing the backbone with COCO [35] pre-trained weights to leverage general object detection knowledge. The differentiable FLAME layer remains fixed during training, with shape and expression parameters set to 300 and 100 dimensions respectively, providing sufficient representational capacity while maintaining computational efficiency. Training requires 4 RTX A6000 GPUs with batch size 80, converging after 7 days. We resize all images to 640×640 pixels while preserving aspect ratios through padding, ensuring consistent head scales across the dataset. To bridge the synthetic-to-real domain gap, we apply extensive data augmentation including blur, noise, compression, and color manipulation, simulating realworld image degradations. This implementation achieves real-time performance suitable for interactive applications while maintaining the accuracy necessary for downstream 3D modeling tasks, demonstrating the effectiveness of our unified architecture design.

### 5. Experimental Evaluation

We extensively evaluate both our synthetic dataset and unified architecture across multiple head-related tasks. Our evaluation demonstrates that: (1) models trained exclusively on synthetic data can achieve state-of-the-art performance on real benchmarks, (2) our dataset improves existing methods beyond our specific architecture, and (3) our unified representation enables superior performance across diverse downstream tasks.

#### 5.1. Dataset Evaluation

We first validate our VGGHeads dataset by comparing our method’s performance when trained on different datasets and by demonstrating cross-method improvements.

Cross-Dataset Training: Table 3 shows our method trained on various traditional datasets compared to VGGHeads training. Training on individual traditional datasets

- Table 2. 3D Head Pose Estimation. VGGHeads achieves state-of-the-art among end-to-end methods on AFLW [78] and BIWI [15], and is competitive with non-end-to-end approaches.

AFLW BIWI

Model End to End 3DMM MAE ↓ Pitch ↓ Roll ↓ Yaw ↓ MAE ↓ Pitch ↓ Roll ↓ Yaw ↓ 6DRepNet [22] ✗ ✗ 3.61 4.58 2.98 3.27 3.78 5.32 2.78 3.23 3DDFA-V2 [19] ✗ ✓ 7.56 8.48 9.89 4.30 8.81 12.08 7.54 6.80 DAD-3DNet [44] ✗ ✓ 3.66 4.76 3.15 3.08 3.98 5.24 2.92 3.79 RetinaFace [12] ✓ ✗ 6.22 9.64 3.92 5.10 4.49 6.42 2.97 4.07 Img2Pose [3] ✓ ✗ 3.91 5.03 3.28 3.43 3.79 3.55 3.24 4.57 VGGHeads ✓ ✓ 3.76 4.91 3.37 3.00 3.79 5.24 2.65 3.47

- Table 3. Dataset Evaluation. The model trained on VGGHeads shows better performance on 3D Head Alignment and Pose Estimation compared to training on traditional face and head datasets (FDDB [26], SCUT [49], HH [60] and WIDER [69].

Dataset

DAD-3D AFLW BIWI FDDB NME↓ Z5 Acc.↑ CD↓ PoseErr↓ MAE↓ MAE↓ AP50↑

FDDB 6.22 0.84 7.21 0.51 8.42 5.23 97.1 SCUT 5.57 0.84 6.23 0.43 8.91 5.44 84.1 HH 3.91 0.91 5.02 0.29 8.12 4.92 89.2 WIDER 5.03 0.87 5.83 0.38 9.83 5.17 96.1 VGGHeads 2.92 0.93 4.00 0.18 3.67 3.58 96.1

- Table 4. Cross-Method Dataset Validation. Training img2pose [3] on VGGHeads dataset improves head pose estimation and detection performance compared to WIDER.

Method Dataset

AFLW BIWI FDDB MAE↓ Yaw↓ MAE↓ Yaw↓ AP50↑

img2pose WIDER 3.91 3.43 3.79 4.57 96.1 img2pose VGGHeads 3.82 3.25 3.74 4.26 96.3

(FDDB [26], SCUT-Head [49], HollywoodHeads [60], WIDER [69]) consistently underperforms SH3D across all 3D (DAD-3D [44]) and pose estimation benchmarks (AFLW [78], BIWI [15]). This demonstrates that our synthetic dataset provides superior training data on a large scale for 3D head modeling tasks, even when compared to real datasets.

Cross-Method Validation: Table 4 addresses the critical question of whether our dataset benefits existing methods beyond our architecture. We retrained img2pose on VGGHeads versus WIDER, showing consistent improvements across AFLW (3.91→3.82 MAE), BIWI (3.79→3.74 MAE), and FDDB (96.1→96.3 AP50). This proves our dataset’s value is not limited to our specific method design.

- 5.2. Head Pose Estimation

3D comprises 2,000 subjects with 68 3D landmarks and pose annotations, while BIWI contains laboratory-recorded RGBD data with head rotations up to ±75° yaw, ±60° pitch, and ±50° roll.

Results: Table 2 shows VGGHeads outperforms end-toend methods [3, 12] and achieves comparable performance to crop-based 3DMM estimators that assume perfect head localization. This is significant because crop-based methods operate on pre-aligned, tightly cropped heads, while our method must first detect heads. Achieving superior performance on this more challenging problem validates both our synthetic data and unified architecture design.

#### 5.3. 3D Head Alignment

We utilize the DAD-3DHeads Benchmark [44] to evaluate 3D dense head alignment and robustness to extreme poses. The benchmark consists of 2,746 images with FLAME topology meshes, measuring pose fitting and shape matching under challenging conditions including extreme poses, illumination, and occlusions. For multi-head images, we extract the head with highest ground truth bounding box overlap.

Results: Table 5 shows VGGHeads significantly outperforms RingNet [55] and 3DDFA-v2 [19], despite these methods being trained on real images and optimized for tight crops. We achieve comparable results to DAD-3D [44] even though it was trained on the same distribution as this benchmark. Our FLAME-based representation provides geometric consistency with the skull center consistently projecting to image center regardless of orientation (Fig. 5), essential for downstream 3D modeling tasks.

#### 5.4. Face Detection

The last step is to validate the model’s ability to detect heads and faces under different conditions. While our model predicts head bounding boxes, we can derive face boxes from FLAME vertex projections, enabling evaluation on established face detection benchmarks [26, 69].

For face box computation, we calculate minimum bounding rectangles around facial vertex subsets and filter detections where |yaw| > π2 (profile views where faces are not

We evaluate 3D head pose estimation accuracy on AFLW2000-3D [78] and BIWI [15] datasets. AFLW2000-

[Figure 5]

- Figure 5. Head Alignment. VGGHeads introduce more consistent alignment across various poses by ensuring the center of the head in 3D is reprojected to the center of the aligned image. VGGHeads, RetinaFace [12].

- Table 5. DAD3D-Heads. VGGHeads achieves superior performance to RingNet and 3DDFA-V2, while being only slightly inferior to the methods trained on the benchmark. E2E: end to end.

Model E2E NME↓ Z5 Acc.↑ Chamf. Dist.↓ Pose Err.↓

3DDFA-V2 [19] ✗ 3.580 - 6.170 0.527 RingNet [55] ✗ 8.757 0.880 5.166 0.438 DAD-3DNet [44] ✗ 2.302 0.954 3.178 0.138 VGGHeads ✓ 2.917 0.933 4.002 0.179

- Table 6. Face Detection. The model trained on VGGHeads data is not optimized for tiny faces detection but with additional finetuning it shows comparable results to state-of-the-art detectors while recovering more complete representation.

Model Dataset

FDDB WIDER Val (AP50) AP50↑ Easy↑ Med.↑ Hard↑

RetinaFace [12] WIDER 96.2 94.6 93.0 80.4 img2pose [3] WIDER 96.1 86.5 82.9 61.3

VGGHeads VGGHeads 96.1 56.3 51.0 29.2 VGGHeads VGGHeads + WIDER 96.6 92.6 88.9 70.3

- Table 7. Ablation. Removing any loss reduces the performance on all three datasets.

VGGHeads Val AFLW FDDB NME↓ FR↓ MAE↓ MAE↓ AP50↑

Model

VGGHeads 1.975 0.122 2.470 5.03 88.5 w/o LR 2.052 0.113 2.863 6.27 87.9 w/o L3D 2.050 0.128 2.600 5.13 87.9

visible). Importantly, unlike methods explicitly optimized for face detection, we do not target extremely small faces (few pixels wide) since 3DMM parameter estimation becomes highly ambiguous at such scales. There is insufficient visual information for reliable 3D reconstruction.

Results: Tab. 6 shows that VGGHeads achieves comparable performance on FDDB [26] to specialized face detectors. This is remarkable considering VGGHeads recovers a more complete head representation and is not optimized for face detection. The performance validates that our unified ap-

proach does not sacrifice detection quality while gaining substantial 3D modeling capabilities.

#### 5.5. Ablation Study

We analyze the contribution of our novel loss components by systematically removing them during training Tab. 7. We ablate the 3D vertices loss (L3D) and rotation loss (LR) as these represent the 3D supervision contributions, while the standard detection losses (Lreproj, Lc, Lreg) are essential for basic functionality and cannot be removed. Each ablated component proves essential: the 3D vertices loss enables better shape reconstruction through direct geometric supervision, while the geodesic rotation loss significantly reduces pose estimation errors via manifold-aware optimization. Ablation experiments use our manually annotated subset for computational efficiency.

### 6. Conclusion

We have presented VGGHeads, a unified approach that bridges head detection and coarse 3D mesh estimation in a single forward pass, thereby eliminating fragmented (pre)processing pipelines. Training this model was only possible using our new synthetic dataset, since it is the first to contain multi-task labels. Our large-scale synthetic dataset, comprising over 1 million multi-person images, captures unconstrained scenarios while addressing privacy concerns that have plagued major real-world datasets. Importantly, we demonstrate that models trained exclusively on synthetic data achieve state-of-the-art performance on real-world benchmarks, often outperforming methods trained on real data despite solving the more challenging problem of joint detection and parametric 3D modeling.Our coarse 3D estimates provide robust initialization for detailed reconstruction methods, bridging the gap between detection and high-fidelity modeling. The dataset, code, and models will be made available, supporting further research in unified head modeling and the broader potential of synthetic data generation.

Acknowledgements. We would like to thank Tetiana Martyniuk and Iro Laina for paper proofreading and valuable feedback. O.K. is supported by a Google unrestricted gift. We also thank the Armed Forces of Ukraine for providing security to complete this work.

### References

- [1] Machine vision & learning group lmu. safety checker model card. https://huggingface.co/CompVis/ stable-diffusion-safety-checker, 2022. Accessed: 2023-11-16. 4
- [2] Shay Aharon, Louis-Dupont, Ofri Masad, Kate Yurkova, Lotem Fridman, Lkdci, Eugene Khvedchenya, Ran Rubin, Natan Bagrov, Borys Tymchenko, Tomer Keren, Alexander Zhilko, and Eran-Deci. Super-gradients, 2021. 3, 5, 12
- [3] Vitor Albiero, Xingyu Chen, Xi Yin, Guan Pang, and Tal Hassner. img2pose: Face alignment and detection via 6dof, face pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7617–7627, 2021. 2, 7, 8, 14
- [4] Volker Blanz and Thomas Vetter. A morphable model for the synthesis of 3d faces. In Conference on Computer graphics and interactive techniques, pages 187–194, 1999. 2
- [5] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 3
- [6] Xiangxiang Chu, Liang Li, and Bo Zhang. Make repvgg greater again: A quantization-aware approach. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 11624–11632, 2024. 12
- [7] Jifeng Dai, Yi Li, Kaiming He, and Jian Sun. R-fcn: Object detection via region-based fully convolutional networks. Advances in neural information processing systems, 29, 2016. 2
- [8] Navneet Dalal and Bill Triggs. Histograms of oriented gradients for human detection. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), pages 886–893. Ieee, 2005. 2
- [9] Radek Danˇeˇcek, Michael J Black, and Timo Bolkart. Emoca: Emotion driven monocular face capture and animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20311–20322, 2022. 2, 13
- [10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 2
- [11] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019. 1, 4
- [12] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multi-level face localisation in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5203–5212, 2020. 2, 4, 7, 8, 14
- [13] Bardia Doosti, Shujon Naha, Majid Mirbagheri, and David J. Crandall. Hope-net: A graph-based model for hand-object pose estimation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 14
- [14] Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, and Qi Tian. Centernet: Keypoint triplets for

- object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6569–6578, 2019. 12
- [15] Gabriele Fanelli, Thibaut Weise, Juergen Gall, and Luc Van Gool. Real time head pose estimation from consumer depth cameras. In Joint pattern recognition symposium, pages 101–110. Springer, 2011. 7, 13, 14
- [16] Yao Feng, Vasileios Choutas, Timo Bolkart, Dimitrios Tzionas, and Michael J Black. Collaborative regression of expressive bodies using moderation. In 2021 International Conference on 3D Vision (3DV), pages 792–804. IEEE, 2021. 2, 13, 14
- [17] Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart. Learning an animatable detailed 3d face model from in-thewild images. ACM Transactions on Graphics (ToG), 40(4): 1–13, 2021. 2, 13
- [18] Jianzhu Guo, Xiangyu Zhu, and Zhen Lei. 3ddfa. https: //github.com/cleardusk/3DDFA, 2018. 2, 14
- [19] Jianzhu Guo, Xiangyu Zhu, Yang Yang, Fan Yang, Zhen Lei, and Stan Z Li. Towards fast, accurate and stable 3d dense face alignment. In European Conference on Computer Vision (ECCV), 2020. 2, 7, 8, 13, 14
- [20] Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, and Jianfeng Gao. Ms-celeb-1m: A dataset and benchmark for large-scale face recognition. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, pages 87–102. Springer, 2016. 2, 4
- [21] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 37(9):1904–1916, 2015. 12
- [22] Thorsten Hempel, Ahmed A Abdelrahman, and Ayoub AlHamadi. Toward robust and unconstrained full range of rotation head pose estimation. IEEE Transactions on Image Processing, 33:2377–2387, 2024. 7, 14
- [23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 12
- [24] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. The Journal of Machine Learning Research, 23(1):2249–2281, 2022. 3
- [25] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. pmlr, 2015. 12
- [26] Vidit Jain and Erik Learned-Miller. Fddb: A benchmark for face detection in unconstrained settings. Technical report, UMass Amherst technical report, 2010. 4, 7, 8, 14
- [27] Peiyuan Jiang, Daji Ergu, Fangyao Liu, Ying Cai, and Bo Ma. A review of yolo algorithm developments. Procedia computer science, 199:1066–1073, 2022. 2
- [28] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In

- Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 1
- [29] Davis E. King. Dlib-ml: A machine learning toolkit. Journal of Machine Learning Research, 10:1755–1758, 2009. 14
- [30] Orest Kupyn and Christian Rupprecht. Dataset enhancement with instance-level augmentations. arXiv preprint arXiv:2406.08249, 2024. 3
- [31] Gant Laborde. Deep nn for nsfw detection. 4
- [32] Oliver Langner, Ron Dotsch, Gijsbert Bijlstra, Daniel HJ Wigboldus, Skyler T Hawk, and AD Van Knippenberg. Presentation and validation of the radboud faces database. Cognition and emotion, 24(8):1377–1388, 2010. 4, 12
- [33] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR,

2022. 3, 4

- [34] Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 36(6), 2017. 2, 4, 5
- [35] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 6
- [36] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017. 12
- [37] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 6
- [38] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 2
- [39] Feng Liu, Luan Tran, and Xiaoming Liu. 3d face modeling from diverse raw scan data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9408– 9418, 2019. 13, 14
- [40] Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path aggregation network for instance segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8759–8768, 2018. 12
- [41] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Large-scale celebfaces attributes (celeba) dataset. Retrieved August, 15(2018):11, 2018. 1, 4, 12, 13
- [42] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 12
- [43] Wenyu Lv, Shangliang Xu, Yian Zhao, Guanzhong Wang, Jinman Wei, Cheng Cui, Yuning Du, Qingqing Dang, and Yi

- Liu. Detrs beat yolos on real-time object detection. arXiv preprint arXiv:2304.08069, 2023. 4
- [44] Tetiana Martyniuk, Orest Kupyn, Yana Kurlyak, Igor Krashenyi, Jiˇri Matas, and Viktoriia Sharmanska. Dad3dheads: A large-scale dense, accurate and diverse dataset for 3d head alignment from a single image. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 20942–20952, 2022. 2, 3, 4, 6, 7, 8, 13, 14
- [45] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 4, 12
- [46] Behnaz Nojavanasghari, Charles E Hughes, Tadas Baltrušaitis, and Louis-Philippe Morency. Hand2face: Automatic synthesis and recognition of hand over face occlusions. In 2017 Seventh International Conference on Affective Computing and Intelligent Interaction (ACII), pages 209–215. IEEE,

2017. 3

- [47] Omkar Parkhi, Andrea Vedaldi, and Andrew Zisserman. Deep face recognition. In BMVC 2015-Proceedings of the British Machine Vision Conference 2015. British Machine Vision Association, 2015. 1, 2, 4
- [48] Pascal Paysan, Reinhard Knothe, Brian Amberg, Sami Romdhani, and Thomas Vetter. A 3d face model for pose and illumination invariant face recognition. In IEEE International Conference on Advanced Video and Signal Based Surveillance, pages 296–301, 2009. 2
- [49] Dezhi Peng, Zikai Sun, Zirong Chen, Zirui Cai, Lele Xie, and Lianwen Jin. Detecting heads using feature refine net and cascaded multi-scale architecture. arXiv preprint arXiv:1803.09256, 2018. 2, 7
- [50] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3, 4, 12, 13
- [51] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015. 2
- [52] Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision, pages 17–35. Springer, 2016. 2
- [53] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 12
- [54] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 12
- [55] Soubhik Sanyal, Timo Bolkart, Haiwen Feng, and Michael J Black. Learning to regress 3d face shape and expression from an image without 3d supervision. In IEEE Conference

- on Computer Vision and Pattern Recognition (CVPR), pages 7763–7772, 2019. 2, 7, 8, 13, 14
- [56] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 3
- [57] Xu Tang, Daniel K Du, Zeqiang He, and Jingtuo Liu. Pyramidbox: A context-assisted single shot face detector. In Proceedings of the European conference on computer vision (ECCV), pages 797–813, 2018. 2
- [58] Roberto Valle, José M Buenaposada, and Luis Baumela. Multi-task head pose estimation in-the-wild. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(8): 2874–2881, 2020. 14
- [59] Paul Viola and Michael J Jones. Robust real-time face detection. International journal of computer vision, 57:137–154,

2004. 2

- [60] Tuan-Hung Vu, Anton Osokin, and Ivan Laptev. Contextaware cnns for person head detection. In Proceedings of the IEEE International Conference on Computer Vision, pages 2893–2901, 2015. 2, 4, 7
- [61] Yitong Wang, Xing Ji, Zheng Zhou, Hao Wang, and Zhifeng Li. Detecting faces using region-based fully convolutional networks. arXiv preprint arXiv:1709.05256, 2017. 2
- [62] Erroll Wood, Tadas Baltrušaitis, Charlie Hewitt, Sebastian Dziadzio, Thomas J Cashman, and Jamie Shotton. Fake it till you make it: face analysis in the wild using synthetic data alone. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3681–3691, 2021. 3
- [63] Cho-Ying Wu, Qiangeng Xu, and Ulrich Neumann. Synergy between 3dmm and 3d landmarks for accurate 3d facial geometry. arXiv preprint arXiv:2110.09772, 2021. 14
- [64] Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, and Chunhua Shen. Datasetdm: Synthesizing data with perception annotations using diffusion models. arXiv preprint arXiv:2308.06160,

2023. 3

- [65] Weijia Wu, Yuzhong Zhao, Mike Zheng Shou, Hong Zhou, and Chunhua Shen. Diffumask: Synthesizing images with pixel-level annotations for semantic segmentation using diffusion models. arXiv preprint arXiv:2303.11681, 2023. 3
- [66] Zhenyu Wu, Lin Wang, Wei Wang, Tengfei Shi, Chenglizhao Chen, Aimin Hao, and Shuo Li. Synthetic data supervised salient object detection. In Proceedings of the 30th ACM International Conference on Multimedia, pages 5557–5565,

2022. 3

- [67] Yuelang Xu, Benwang Chen, Zhe Li, Hongwen Zhang, Lizhen Wang, Zerong Zheng, and Yebin Liu. Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians. arXiv preprint arXiv:2312.03029, 2023. 1
- [68] Kaiyu Yang, Jacqueline H Yau, Li Fei-Fei, Jia Deng, and Olga Russakovsky. A study of face obfuscation in imagenet. In International Conference on Machine Learning, pages 25313–25330. PMLR, 2022. 2
- [69] Shuo Yang, Ping Luo, Chen-Change Loy, and Xiaoou Tang. Wider face: A face detection benchmark. In Proceedings of

- the IEEE conference on computer vision and pattern recognition, pages 5525–5533, 2016. 2, 4, 5, 7, 14, 15
- [70] Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois. Gliner: Generalist model for named entity recognition using bidirectional transformer, 2023. 4
- [71] Changzheng Zhang, Xiang Xu, and Dandan Tu. Face detection using improved faster rcnn. arXiv preprint arXiv:1802.02142, 2018. 2
- [72] Hongwen Zhang, Yating Tian, Yuxiang Zhang, Mengcheng Li, Liang An, Zhenan Sun, and Yebin Liu. Pymaf-x: Towards well-aligned full-body model regression from monocular images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):12287–12303, 2023. 2, 13
- [73] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 12, 13
- [74] Shifeng Zhang, Xiangyu Zhu, Zhen Lei, Hailin Shi, Xiaobo Wang, and Stan Z Li. S3fd: Single shot scale-invariant face detector. In Proceedings of the IEEE international conference on computer vision, pages 192–201, 2017. 2
- [75] Yinglin Zheng, Hao Yang, Ting Zhang, Jianmin Bao, Dongdong Chen, Yangyu Huang, Lu Yuan, Dong Chen, Ming Zeng, and Fang Wen. General facial representation learning in a visual-linguistic manner. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18697–18709, 2022. 4
- [76] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-iou loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12993– 13000, 2020. 6
- [77] Yijun Zhou and James Gregson. Whenet: Real-time finegrained estimation for wide range head pose. arXiv preprint arXiv:2005.10353, 2020. 14
- [78] Xiangyu Zhu, Zhen Lei, Xiaoming Liu, Hailin Shi, and Stan Z Li. Face alignment across large poses: A 3d solution. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 146–155, 2016. 7, 14
- [79] Xiangyu Zhu, Zhen Lei, Xiaoming Liu, Hailin Shi, and Stan Z Li. Face alignment across large poses: A 3d solution. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 146–155, 2016. 1, 3, 14
- [80] Wojciech Zielonka, Timo Bolkart, and Justus Thies. Towards metrical reconstruction of human faces. In European conference on computer vision, pages 250–269. Springer, 2022. 2, 13

### A. Data Quality

Table 8. Generators. We experiment with different image generators [50, 53] and control mechanisms [45, 73]. Our method is not specific to any combination and improves with better generators.

Model Condition FID↓ IS↑

SD 1.5 ControlNet 7.86 13.59 SDXL ControlNet 4.18 15.09 SDXL T2I 3.22 14.37

We evaluate the impact of different diffusion models and conditioning mechanisms by generating 20,000 images per configuration. Using FID [23] and Inception Score [54], we measure scene diversity and realism, evaluating full images rather than face crops focusing on scene composition and multi-person layout quality, crucial for robust real-world performance.

Tab. 8 demonstrates that advanced diffusion models [50] significantly outperform earlier versions [53] in generating high-quality multi-person scenes. While ControlNet [73] and T2I-Adapter [45] achieve comparable metrics, ControlNet frequently produces anatomical deformations, leading us to choose T2I-Adapter for more reliable scene generation.

### B. Re-Identification on Synthetic Data

Figure 6 provides visual examples of generated samples that were matched with public figures from Celeb-A [41] dataset. Even though the data generation include the prompt anonymization, some parts of the prompts, such as the movie name still might encode identity of famous people. Identifying and removing such samples improve the privacy aspect. Fig. 7 shows an example of an image generated by diffusion model that was finetuned on RaFD [32] dataset, and 2 nearest neighbors from the dataset it was trained on. Even though the model learns features from the original images, in most cases it does combine features from multiple people and fails to recover finer details that encode the identity.

### C. Model Details

The architecture is inspired by object detection methods like CenterNet [14] or YOLO-NAS-Pose [2], enabling efficient end-to-end learning. The model is based on the YOLO-NAS [2] architecture for object detection. YOLO-NAS use a neural architecture search engine to enhance the YOLO family of models by optimzing the sizes and structures of stages, block types, the number of blocks, and the number of channels in each stage. We employ the YOLO-NAS-L backbone, though the model is agnostic to the choice of encoder. The “neck” is used to fuse the features generated by the backbone. The visual features from the encoder maps neck are fused

Table 9. Model Architecture Comparison. Analysis of different VGGHeads variants in terms of parameters, computational cost, and speed.

Model Total Parameters FLOPs (B) FPS VGGHeads-L 50,442,706 83.51 60.13 VGGHeads-M 32,378,236 52.01 69.38 VGGHeads-S 17,004,954 22.92 72.34

by the Spatial Pyramid Pooling [21] module at different scales and processed by Feature Pyramid Network [36] to generate features at different semantic levels. Similarly to other YOLO models, we adopt an anchor-based multi-scale detection scheme. Path Aggregation Network (PAN) [40] transfers positioning features bottom-up. We combine them with the features from FPN to obtain a better feature fusion effect and then directly use the multi-scale fusion feature maps in the PAN for detection. Thus, the detection heads predict bounding boxes and 3DMM parameters on different scales, ensuring high accuracy on different object sizes. This approach allows us to optimize FLAME parameters only on positive anchors, improving training efficiency. It also offers flexibility for applications that may not require full 3D meshes, allowing the extraction of bounding boxes in real time without computing full 3D mesh. The detection head in the YOLO-NAS model predicts the offset of the bounding box position and the scaling of the height and width, as well as the confidence of the prediction. We extend the detection head to also predict the 3DMM parameters by introducing six separate 3D parameter prediction modules, each consisting of two RepVGG blocks [6] and a final 1 × 1 convolution that predicts the final set of parameters. Each RepVGG block consists of three branches: a 3 × 3 Convolution followed by BatchNorm [25], a branch of a 1x1 Convolution with bias and a residual branch. Predicting different 3DDM parameters components separately achieves an extra level of disentanglement.

The weights of the loss components are set to α1 = 50,α2 = 1,α3 = 1,α4 = 0.5,α5 = 2.5 in the final version.

The framework is agnostic to the choice of backbone and can be adapted to use larger transformer models [42]. Yet, we observe the string performance on a smaller fully convolutional models and stick to this design to allow for real-time multi head mesh recovery. Furthermore the versatility of backbones allow to train even smaller models suitable for wide range of tasks and applications.

### D. Head Pose Estimation

The full results on AFLW and BIWI datasets are presented in Tab. 10, Tab. 11. VGGHeads outperforms most of method optimized solely for 3D Head Pose Estimation even though it does not operate on tight head crops.

[Figure 6]

- Figure 6. Re-ID on Celeb-A. We automatically detect and removed samples where face in the generated image is matched with one of the faces from Celeb-A dataset [41].

[Figure 7]

[Figure 8]

[Figure 9]

- Figure 7. Preserving Identity. With subject neutral prompts diffusion Model blends feature of different people from the set it was trained on.

### E. 3D Face Reconstruction

VGGHeads is more coarse than 3D reconstruction methods so detailed 3D face reconstruction is not the goal of our method and a standalone task in itself. Nonetheless, we evaluate the model on the Feng et al. benchmark [39]. The model achieves comparable results to other coarse 3D face reconstruction methods [19, 44, 55] despite not being optimized for shape and expression disentanglement. This performance is surprising since our method predicts the 3D face from a full image, not a tight crop as is typical for this task. While our method aims at solving a different task, it achieves good performance. Moreover, the VGGHeads dataset and model can be complementary to detailed face reconstruction methods such as DECA [17], MICA [80] or EMOCA [9], which often need to be initialized with a crop or initial coarse face shape.

### F. Full Body Mesh Recovery

Our approach proves the viability of using synthetic data from diffusion models for body modeling, paving the way for future fully synthetic all-in-one methods. VGGHeads is the first step towards this goal. While full-body reconstruction methods [16, 72], achieved significant progress and attention, they often still rely on upstream face predictors and lack robustness in edge cases. To validate it we evaluate PIXIE’s [16] performance on the BIWI [15] dataset for 3D Head Pose.

### G. Controllable Generation

Conditioning the image generation on full head mesh helps to preserve the head shape and expression which is crucial for many AR applications. We trained the ControlNet [73] for SDXL [50] model that is conditioned on meshes recovered by our model. The meshes are rendered by mapping to RGB space with Projected Normalized Coordinate Code (PNCC)

- Table 10. AFLW

Model End to End 3DMM MAE ↓ Pitch MAE ↓ Roll MAE ↓ Yaw MAE ↓ Dlib [29] ✗ ✗ 13.29 12.60 9.00 18.27 HopeNet [13] ✗ ✗ 6.16 6.56 5.44 6.47 6DRepNet [22] ✗ ✗ 3.61 4.58 2.98 3.27 RingNet [55] ✗ ✓ 8.27 4.39 13.51 6.92

- 3DDFA-V2 [19] ✗ ✓ 7.56 8.48 9.89 4.30 3DDFA [18] ✗ ✓ 7.39 8.53 7.39 5.40 DAD-3DHeads [44] ✗ ✓ 3.66 4.76 3.15 3.08 SynergyNet [63] ✗ ✓ 3.35 4.09 2.55 3.42

RetinaFace [12] ✓ ✗ 6.22 9.64 3.92 5.10 Img2Pose [3] ✓ ✗ 3.91 5.03 3.28 3.43 VGGHeads ✓ ✓ 3.76 4.91 3.37 3.00

Table 11. BIWI

Model End to End 3DMM MAE ↓ Pitch MAE ↓ Roll MAE ↓ Yaw MAE ↓ Dlib (68 points) [29] ✗ ✗ 12.25 13.80 6.19 16.76 HopeNet [13] ✗ ✗ 4.90 6.61 3.27 4.81 WHENet [77] ✗ ✗ 3.81 4.39 3.06 3.99 6DRepNet [22] ✗ ✗ 3.78 5.32 2.78 3.23 MNN [58] ✗ ✗ 3.66 4.61 2.39 3.98 3DDFA [18] ✗ ✓ 19.07 12.25 8.78 36.18

- 3DDFA-V2 [19] ✗ ✓ 8.81 12.08 7.54 6.80 RingNet [55] ✗ ✓ 7.34 5.37 7.82 8.82 DAD-3DNet [44] ✗ ✓ 3.98 5.24 2.92 3.79

RetinaFace [12] ✓ ✗ 4.49 6.42 2.97 4.07 Img2Pose [3] ✓ ✗ 3.79 3.55 3.24 4.57 VGGHeads ✓ ✓ 3.79 5.24 2.65 3.47

### H. Qualitative Results

Median(mm)↓ Mean(mm)↓ Std(mm)↓

Model 3DRMSE↓

HQ LQ HQ LQ HQ LQ

Additional data samples from VGGHeads dataset are presented in Fig. 9.

3DDFA-V2[19] 2.998 1.500 1.779 1.942 2.350 1.704 2.149 RingNet[55] 2.809 1.698 1.634 2.161 2.113 1.832 1.831 DAD-3DNet [44] 2.749 1.558 1.624 1.940 2.082 1.581 1.795 VGGHeads 2.996 1.622 1.801 2.079 2.353 1.801 2.054

We also include more visual results on AFLW [78] Fig. 12, BIWI [15] Fig. 13, DAD-3D [44] Fig. 14, WIDER [69] Fig. 8 and FDDB [26] benchmarks Fig. 11.

###### Table 12. Feng et al.[39]

Table 13. Comparison with Full Body Reconstruction [16]. VGGHeads achieves superior performance to full body recovery method [16] on BIWI 3D Head Pose Estimation. This shows that all in one methods still fail on challenging head understanding benchmarks.

Model MAE↓ Pitch MAE↓ Roll MAE↓ Yaw MAE↓ PIXIE [16] 10.97 16.80 6.19 9.93 VGGHeads 3.79 5.24 2.65 3.47

[79], with the 3D coordinate of each vertex of the normalized head mesh encoded as RGB (NCCx = R, NCCy = G, NCCz = B). The heads on the generated images preserve the pose, expression and shape of the original photo Fig. 10.

### I. Limitations and Broader Impact

The dataset annotations are based on DAD-3D [44] so we don’t aim to model neck, ears and eyeball vertices that are a part of the FLAME topology. The generation pipeline still can produce deformed small faces due to the limited resolution so we don’t label and predict the 3D model parameters of the tiny faces. Also, the more advanced filtering methods and nsfw detection methods are a suitable venue for future explorations as on the large scale it is not feasible to guarantee the absolute correctness of the generated samples, even by adding the human evaluation into the process. By leveraging synthetic data generated through diffusion models, we reduce the privacy, ethics, and safety issues in human subject research, as no real personal data is used so that privacy and ethical standards are upheld. Furthermore, the synthetic dataset’s high resolution and detailed annotations

[Figure 10]

- Figure 8. Qualitative Evaluation. VGGHeads is able to accurately recover 3D head models on various complex scenes from WIDER Face [69] dataset.

provide a robust and versatile resource for developing and testing new models. This approach not only enhances the generalizability and accuracy of models trained on this data but also promotes ethical research practices by eliminating the need for real human subjects. The ability to generate large-scale synthetic datasets paves the way for safer and more inclusive research, free from the constraints and risks associated with real-world data collection. Thus, our work promotes ethical AI practices and sets a standard for future research in this area.

### J. Fail Cases

The dataset fail cases are presented in Fig. 15. Typical failure cases are head detector failure (both FP and FN), misaligned mesh on out-of-distribution shapes and poses, severe occlusions, and deformed tiny faces in crowded scenes.

[Figure 11]

###### Figure 9. Dataset Examples. The synthetic data generation pipeline generates complex realistic real world scenes with multiple objects, covering wide range of poses and backgrounds while reducing age, gender and ethnical biases present in small real world datasets.

[Figure 12]

###### Figure 10. ControlNet with VGGHeads. The 3D condition provides a strong degree of control for the generative model, preserving shape, pose and expression of the input image.

[Figure 13]

###### Figure 11. Qualitative Evaluation on FDDB.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

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

###### Figure 12. Qualitative Evaluation on AFLW.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

###### Figure 13. Qualitative Evaluation on BIWI.

[Figure 46]

###### Figure 14. Qualitative Evaluation on DAD-3D

[Figure 47]

###### Figure 15. VGGHeads Dataset Fail Cases

