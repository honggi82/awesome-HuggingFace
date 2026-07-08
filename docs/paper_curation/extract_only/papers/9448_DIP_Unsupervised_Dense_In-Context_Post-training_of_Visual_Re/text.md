# arXiv:2506.18463v3[cs.CV]9Sep2025

## DIP: Unsupervised Dense In-Context Post-training of Visual Representations

Sophia Sirko-Galouchenko1,2 Spyros Gidaris1 Antonin Vobecky1,3,4 Andrei Bursuc1 Nicolas Thome2,5

1Valeo.ai 2Sorbonne Universit´e, CNRS, ISIR, F-75005 Paris, France 3FEE CTU 4CIIRC CTU Prague 5 Institut universitaire de France (IUF)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

(a) Query (b) Ground Truth (c) DIP output (d) DIP 1st NN (e) Dv2R output (f) Dv2R 1st NN

##### cat dog horse sheep cow aeroplane car bus bottle diningtable motorbike train

Figure 1. Dense retrieval-based semantic segmentation in low-shot regimes (40 examples) using patch feature similarities. For each query image (a), we show the ground truth (b), our DIP model’s prediction (c), and its retrieved neighbor (d) for the patch marked with a red cross in (a). In (e) and (f), we display DINOv2R’s (Dv2R) output and its nearest neighbor, respectively. Our DIP representations retrieve more coherent neighbors, yielding more accurate segmentations than DINOv2R.

### Abstract

We introduce DIP, a novel unsupervised post-training method designed to enhance dense representations in largescale pretrained vision encoders for in-context scene understanding. Unlike prior approaches using complex selfdistillation architectures, our method trains the vision encoder using pseudo-tasks that simulate downstream incontext scenarios, inspired by meta-learning principles. To enable post-training on unlabeled data, we propose an automatic mechanism for generating in-context tasks that combines a pretrained diffusion model and the vision encoder. DIP is simple, unsupervised, and computationally efficient, requiring under 9 hours on a single A100 GPU. By learning dense representations through pseudo incontext tasks, it achieves strong performance across a variety of downstream real-world in-context scene understanding tasks. It outperforms both the initial vision encoder and prior methods, offering a practical and effective solution for improving dense representations. See code here

### 1. Introduction

Our goal is to learn dense image representations from unlabeled data for effective in-context scene understanding. In-context learning, which enables models to adapt to tasks without updating parameters, has seen remarkable success in large language models (LLMs) [7]. By providing a few examples within the input prompt, LLMs generalize effectively to new tasks. Inspired by this, recent efforts aim to bring similar in-context learning capabilities to vision-only models [3, 5]. A notable approach by Balazevic et al. [3] reformulates dense prediction tasks as nearest-neighbor retrieval problems using patch feature similarities, demonstrating strong performance, especially with limited data. However, while recent self-supervised Vision Transformers (ViTs) [3, 56, 75] show promise for in-context scene understanding, they still fall short of matching LLMs’ incontext learning success. To bridge this gap, it is crucial to learn dense features that establish strong semantic correspondences between patches in test and training images.

A key insight from prior work [58, 64, 84] is that leverag-

ing pretrained Vision Foundation Models (VFMs), such as DINOv2R [16, 56], and specializing them through unsupervised post-training is more computationally efficient and effective than training from scratch. This aligns with trends in LLMs, where large-scale pretraining is followed by cheaper post-training stages to specialize in specific skills of interest. We refer to the pretrained VFM undergoing posttraining as the base model.

Most methods, whether trained from scratch or posttrained, rely on self-distillation, where the teacher is an exponential moving average of student weights [26]. These frameworks use architectures and objectives designed to promote dense nearest-neighbor retrieval skills. For instance, state-of-the-art methods enforce object consistency across images and views [46] or between patch rankings from two random views of the same image [58]. However, self-distillation often suffers from poor stability [83] and hyperparameter sensitivity [2]. Moreover, these approaches introduce additional complexity, such as differentiable sorting [58], RoI alignment units [29, 58, 84], Sinkhorn-Knopp optimization [64, 84], or learnable patch pooling [3, 46]. While improving performance, these components make the methods harder to interpret and tune, and less transparent in terms of their underlying mechanisms.

In this work, we adopt a post-training approach due to its computational efficiency and effectiveness. Within this context, we address two key questions: (1) Are there simpler and more effective alternatives to existing self-distillation frameworks with complex architectures and objectives?, and (2) Beyond the base model, can we leverage other vision foundation models to achieve our goal?

We propose a simpler post-training approach with an unsupervised learning objective explicitly designed for incontext scene understanding. Inspired by meta-learning, which learns general knowledge from diverse tasks to address unseen but related tasks, we automatically construct multiple in-context tasks from unlabeled data and train the model on them. This enables the model to acquire transferable dense representations for efficiently solving new incontext tasks during downstream stages. Unlike prior methods, our approach eliminates complex training objectives and architectures, and avoids self-distillation frameworks.

To achieve unsupervised post-training, we generate incontext pseudo-tasks that mimic the tasks encountered during downstream use. Each pseudo-task consists of support examples with segmentation masks and a query image segmented using the support examples. The support set includes a positive example (sharing objects with the query) and “distractor” examples (likely unrelated). Our pseudotasks resemble meta-learning episodes [76] with the difference that we do not use any manual annotations.

Instead, we leverage pretrained VFMs to construct these pseudo-tasks. Specifically, apart from the base model (DI-

NOv2R) that we post-train, we use an auxiliary pretrained generative model, Stable Diffusion (SD) [27]. SD provides high-fidelity, class-agnostic image segments in an unsupervised manner, while the base model retrieves candidate positives for each query and assigns pseudo-labels to the segments. Although SD is pre-trained on image-caption pairs, we consider our post-training unsupervised as we do not use captions to extract image segments from SD, and neither SD nor the base model rely on dense segmentation labels.

The main contributions of this paper are as follows:

- • We introduce DIP, a novel unsupervised post-training method that uses retrieval-based in-context learning to improve dense image representations. Unlike prior work, DIP is based on meta-training principles and eliminates the need for complex self-distillation architectures. Furthermore, DIP is computationally efficient, requiring less than 9 hours on a single A100 GPU for post-training.
- • To enable post-training on unlabeled data, we propose an automatic mechanism for generating in-context tasks. In addition to the base model, this approach leverages the Stable Diffusion model (through the unsupervised, training-free DiffCut [15] technique) to produce highquality pseudo-segmentations from unlabeled images.
- • Experiments show that the learned dense representations generalize effectively to a wide variety of downstream tasks. These include six semantic segmentation datasets and one monocular depth prediction dataset, even in lowshot scenarios. Furthermore, we extend our approach to other VFMs, such as CLIP [61], and show that it consistently enhances their dense representations.
- • Compared to DINOv2R, DIP produces more semantically coherent neighbors, leading to more accurate retrieval-based scene understanding, particularly in lowshot settings (see Fig. 1). Additionally, DIP outperforms recent state-of-the-art post-training methods in most retrieval-based tasks while being simpler.

### 2. Related Work

Self-supervised learning. Self-supervised methods learn rich representations from vast amounts of raw data via pretext tasks providing supervision signals from the data, later enabling fine-tuning on downstream tasks with minimal annotations. The main self-supervised approaches for images include: contrastive objectives that distinguish similar and dissimilar views [11, 30, 54], clustering-based objectives [1, 8, 23], self-distillation objectives that match representations across augmentations [6, 9, 12, 24, 26], and reconstruction objectives that predict pixel colors [31, 80] or masked patch features from a teacher network [2, 25, 39, 55, 75, 83]. Recent approaches use pretrained VFMs for distillation in cooperation with self-supervised objectives [4, 17, 36, 59]. These methods focus on learning

global image representations for classification tasks. However, without full-finetuning or task decoders, they are less suitable for dense scene-understanding tasks like segmentation. We build DIP on contrastive objectives [54] equipped with a memory bank [30].

Dense self-supervision. Learning localized image representations that can be rapidly adapted to scene understanding requires specialized self-supervised objectives that mimic downstream tasks [3, 10, 33, 34, 46, 68, 71, 78, 79]. One approach contrasts features from dissimilar pixels or pseudo-segments within images, improving object detection [33, 34, 53, 78, 79]. Others employ clustering strategies to produce object-aware supervision signals to improve semantic segmentation [10, 71, 84], while some extend patchlevel contrastive learning across images [3] with nearestneighbor consistency [28, 46, 58]. Although we share similar objectives with [3, 58] for in-context scene understanding, we introduce a novel self-supervised objective inspired by few-shot learning that leverages automatically computed object segment pseudo-labels.

Recent works build upon pretrained VFMs [28, 41, 58, 69, 84] casting their self-supervised training as a posttraining stage for preparing the VFM for dense downstream tasks. Similarly, DIP aims to rapidly endow pretrained VFMs with in-context dense reasoning skills.

Unsupervised semantic segmentation. These works aim to produce object segments via self-supervised training. Earlier works employed clustering objectives, with segmentations refined via consistency across augmented views [13, 37, 38, 40, 77]. Other works explore pretrained features from self-supervised VFMs [28, 45, 47, 66, 77] or diffusion UNet encoders [15, 73]. The recent DiffCut [15] leverages recursive Normalized Cuts in the final attention layer of a pretrained diffusion UNet encoder to produce object segments without supervision. We use DiffCut to extract pseudo-segmentation maps for our in-context post-training, as these segments - while sometimes noisy - serve as effective proxies for actual object segments in our framework.

Few-shot learning. Few-shot learning develops models capable of rapid adaptation from limited labeled examples. These models are typically trained on few-shot tasks (or episodes) sampled from the training data, that simulate test-time conditions. Existing approaches include memory-based methods that match queries to labeled examples [50, 65, 76], metric-learning techniques that learn distance functions from few samples [70, 72], optimizationbased methods that learn efficient adaptation through few gradient steps [20, 62], and generative approaches that predict classifier weights [21, 22, 60]. Inspired by Vinyals et al. [76], we devise a self-supervised pretext objective consisting of multiple in-context scene understanding tasks, each with a set of support samples stored in a memory and a

query image to be segmented based on the support samples. In contrast to few-shot learning, we use no human annotation during post-training, generating labels automatically.

### 3. Method

Our goal is to fine-tune a pretrained vision encoder, typically a ViT [18], to produce dense features suitable for incontext dense prediction tasks framed as nearest-neighbor retrieval. Given a set of support images (i.e., training images) with semantic segmentation annotations and a query image (i.e., test image), we aim to ensure that patch-wise features extracted from the query image are highly similar (e.g., high cosine similarity) to support patches of the same object category, while being dissimilar to features of different objects.

To achieve this, we propose an unsupervised posttraining approach (Fig. 2) that fine-tunes the encoder on dense nearest-neighbor retrieval tasks automatically generated from unlabeled data. Training on pseudo-tasks that simulate real scenarios, the encoder learns robust features transferable to real in-context tasks.

In the following sections, we first detail our dense incontext post-training approach (Sec. 3.1) and then describe the automatic process for constructing dense in-context tasks (Sec. 3.2), including the pseudo-labeling of classagnostic segments and the selection of support examples.

#### 3.1. Dense In-Context Training

During training, we sample an in-context task T from dataset D. Each task T consists of K support examples S = {(Xs

)}Ki=1 and one query example (Xq,Yq), where Xq and Xs

,Ys

i

i

are their one-hot semantic segmentation pseudo-labels. Among the K support examples, one is a “positive” example sharing pseudo-classes with the query, while the remaining K − 1 are “distractor” examples unlikely to share any pseudoclass. Since our method is unsupervised, these tasks and pseudo-labels are automatically constructed (see Sec. 3.2).

are images, and Yq and Ys

i

i

Feature extraction. Let f(·) be a pretrained encoder that our method fine-tunes. Given an input image X (either query or support), f(·) produces patch-wise features f(X) ∈ RL×D, where L is the number of patches and D is the feature dimension. We further apply a patch-wise multilayer perceptron (MLP) network h(·) to project the features into a D′-dimensional space: F = h(f(X)) ∈ RL×D

′

.

The MLP includes an ℓ2-normalization layer at the end. Let θ denote the parameters of (h ◦ f)(·) model.

Label pre-processing. To align label resolution with patch features, we patchify the one-hot pseudo-labels Ys

. We divide Ys

i

into patches matching the encoder’s patch size and average the one-hot labels within each patch, yielding Ys′

i

∈ RL×C, where C is the number of pseudo-classes.

i

[Figure 13]

- Figure 2. Our unsupervised Dense In-context Post-training (DIP) method. During post-training, the model is given a pseudo incontext task, created automatically without human input. Each task includes a query image and a pseudo-labeled support set with a positive example (sharing objects with the query, as shown in the zebra example) and “distractor” examples that contain different object categories. The model predicts the pseudo-labeled semantic segmentation of the query image using the support set as reference. To do this, it (1) extracts patch-wise features from both the query and support images using the vision encoder f(·) and projection head h(·); (2) computes segmentation predictions for the query image through cross-attention (query patch features as queries, support patch features as keys, and pseudo-labeled support patches as values). A pixel-wise cross-entropy loss is applied to the predictions. By training on these pseudo tasks, DIP enables the encoder to learn transferable dense representations, which are later used to efficiently solve new real in-context tasks.

In-context dense predictions. Using patch-wise features and patchified labels from the support set S, we define a soft nearest-neighbor classification function cS(·) to predict dense labels Yˆq = cS(Xq) for the query image Xq. First, we compute attention scores for each query patch over all K × L support patches:

Fq · FST τ

, (1)

A = softmax

′

concatenates patch-wise features Fs

where FS ∈ R(L·K)×D

from all support images, and τ is the softmax temperature. The softmax normalization is over the support patches. The attention weights A ∈ [0,1]L×(L·K) are used to compute a weighted average of the patchified support labels:

i

Yˆq′ = A · YS′, (2)

where YS′ ∈ R(L·K)×C concatenates patchified labels Ys′

i

from all support images, and Yˆq′ ∈ RL×C represents the predicted patchified labels for query. Essentially, this de-

fines a cross-attention layer [74], with Fq as queries, FS as keys, and YS′ as values.

Finally, we use nearest-neighbor interpolation to upsample Yˆq′ to the original image size, yielding the final label prediction Yˆq = cS(Xq).

Training objective. For a single task T ={S,(Xq,Yq)}, we minimize the pixel-wise cross-entropy loss LCE(Yq,cS(Xq),θ) between the predicted labels cS(Xq) and the pseudo-labels Yq. The model, comprising the

pretrained encoder and randomly initialized MLP (h◦f)(·) with parameters θ, is trained by optimizing the expected loss over a collection of tasks sampled from D:

q,Yq)}∼D [LCE(Yq,cS(Xq),θ)]. (3)

E{S,(X

min

θ

Downstream stage. For downstream in-context scene understanding tasks, we remove the head h(·) and use the fine-tuned encoder f(·). Following [3, 57, 58], we construct a larger memory bank (support set) than used during post-training, with 10,240,000 patch-wise features randomly sampled from the available training images for each dataset. For a query image, we extract its patch-wise features using f(·), retrieve the top-30 nearest neighbors from the memory bank via cosine similarity, and apply crossattention (as in Eqs. (1) and (2)) with temperature τ = 0.07.

#### 3.2.AutomaticDenseIn-ContextTaskConstruction

To apply our dense in-context post-training approach to unlabeled data, we automatically generate in-context pseudotasks with spatially coherent segmentation labels. This involves two steps: (1) generating pseudo-segmentation maps for dataset images and (2) selecting query and support examples to form each in-context task. These steps create meaningful and challenging tasks, enabling the model to learn transferable in-context visual learning skills. Examples of constructed pseudo-tasks are shown in Fig. 3.

Generating pseudo semantic segmentation labels. We generate pseudo-segmentation maps by first dividing each image into non-overlapping class-agnostic segments and then assigning a pseudo-label to each segment.

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

(a) Query (b)Pseudo labels (c) Positive (d) Pseudo labels

- Figure 3. Examples of automatically constructed in-context scene understanding tasks. Each row shows a query image and its corresponding positive support example. (a) and (b) display the query image and its pseudo segmentation labels, while (c) and (d) show the positive support image and its pseudo segmentation labels. Despite being generated in a fully unsupervised manner, the segmentation masks for salient objects are highly accurate, closely matching the actual objects. In addition, the query and positive image pairs share a common object with the same pseudo-label. During post-training with our DIP method, the model predicts the query image’s segmentation using the positive example as a reference, along with distractor support examples (randomly sampled from other images in the mini-batch, not shown here for brevity).

For the first step, we use DiffCut [15], a training-free zero-shot image segmentation method. DiffCut leverages features from a pretrained diffusion model, SSD-1B [27], along with a recursive graph partitioning algorithm to produce fine-grained segmentation maps. These maps, called DiffCut masks, are generated and stored for the entire pretraining dataset. Since DiffCut is unsupervised, our pretraining approach remains fully annotation-free.

For the second step, we assign pseudo-labels to each segment using the self-supervised DINOv2R feature encoder. We compute the mean DINOv2R feature for each DiffCut mask by pooling dense features within the mask region. To group visually similar segments, we apply K-means clustering on these pooled features1. The resulting clusters serve as pseudo-classes for annotating the DiffCut masks. To assign a pseudo-class to each DiffCut mask, we first assign each pixel-wise DINOv2R feature the cluster ID of its closest K-means centroid. Then, for each DiffCut mask, we perform majority voting on all pixels within the mask and assign the most frequent cluster ID. This ensures robust and consistent pseudo-labels, even with noisy features.

Selecting examples for in-context tasks. To generate incontext tasks, we pair each query image with one “positive”

1For COCO [48], clustering is performed on the full dataset; for ImageNet [44], clustering is performed on a random subset of 200,000 training images for efficiency.

DINOv2R NeCo DIP (ours)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

80

40

mIoU(%)

35

70

30

60

25

1 128

1 64

1 8

1 1

1 128

1 64

1 8

1 1

Fraction of Data

Fraction of Data (b) ADE20K

(a) PascalVOC

Figure 4. In-context scene understanding in low-shot regimes. mIoU results with ViT-B/14 versus training data size.

support image and K-1 “distractor” support images.

To find positive images sharing similar visual content, we use DINOv2R global image representations. For each image, we retrieve five nearest neighbors using DINOv2R global features. This yields five positive pairs per image. We then retain pairs where the query and positive-support images share a common pseudo-class occupying more than 5% of each image’s area. This ensures semantically meaningful positive pairs. The final refined list contains all possible query-positive support pairs for training.

For the K-1 “distractor” examples, we randomly sample images from the current mini-batch during training. Including distractors encourages the model to distinguish relevant from irrelevant content, improving feature discriminability.

### 4. Experiments

#### 4.1. Experimental Setup

Compared methods. We compare our method with state-of-the-art unsupervised learning approaches, including DINO [9], Leopart [84], TimeT [64], iBOT [83], CrOC [71], CrIBO [46], DINOv2R [16, 56], and NeCo [58]. NeCo, a recent method (ICLR’25), is the most related to ours, as it also post-trains DINOv2R for dense representation learning in in-context scene understanding tasks. However, our post-training methodology differs significantly: we explicitly train on in-context pseudo-tasks relevant to our target applications, while NeCo adopts a self-distillation-based approach.

Training setup. We apply our unsupervised post-training to DINOv2 [56] with registers [16] (DINOv2R), using ViTS/14 and ViT-B/14 models. In-context pseudo-tasks are generated using DiffCut and the base model (e.g., DINOv2R ViT-B/14 is used to generate pseudo-tasks for posttraining DINOv2R ViT-B/14; see Sec. 3.2). Except otherwise stated, post-training is on COCO for 5 epochs with 1,000 pseudo-classes. We also explore post-training CLIP

PascalVOC ADE20K

Method Backbone

1 1

1 128 DINO† [9] ViT-S/16 48.7 41.3 30.5 26.4 17.9 15.0 11.0 9.5 SelfPatch† [81] ViT-S/16 50.8 43.2 32.6 28.4 17.7 14.7 10.9 10.0 CroC† [71] ViT-S/16 60.5 53.8 41.8 34.0 17.3 15.2 10.8 8.7 TimeT† [64] ViT-S/16 62.3 55.2 43.8 38.1 23.2 18.9 14.1 12.1 Leopart† [84] ViT-S/16 64.5 58.4 49.7 44.6 23.9 19.6 14.8 12.9 CriBO† [46] ViT-S/16 72.4 66.9 59.9 53.9 26.6 22.7 17.3 14.6 DINOv2R‡ [16] ViT-S/14 79.4 75.2 67.7 60.7 39.3 33.2 24.9 22.6 NeCo‡ [58] ViT-S/14 81.0 77.3 71.5 65.8 38.9 32.5 24.1 21.9 DIP (ours)‡ ViT-S/14 81.0 77.7 71.4 65.9 39.7 33.7 25.6 23.2

1 8

1 64

1 128

1 1

1 8

1 64

DINO† [9] ViT-B/16 57.3 49.8 37.7 33.1 21.5 18.2 13.5 11.5 Leopart† [84] ViT-B/16 69.5 63.1 54.7 50.1 26.7 21.8 16.8 14.6 Hummingbird† [3] ViT-B/16 71.8 64.3 57.2 50.5 29.6 22.3 15.1 11.7 CriBO† [46] ViT-B/16 74.2 69.2 61.8 55.9 28.4 24.4 18.4 15.9 DINOv2R‡ [16] ViT-B/14 79.0 75.3 67.6 60.3 40.8 35.3 27.3 24.9 NeCo‡ [58] ViT-B/14 82.4 78.7 71.7 65.0 41.2 35.2 27.2 25.1 DIP (ours)‡ ViT-B/14 82.1 79.6 75.1 70.1 42.6 36.8 29.4 27.0

- Table 1. In-context scene understanding with few training examples. Dense nearest neighbor retrieval performance on ADE20K and

PascalVOC datasets, evaluated using the mIoU metric across varying proportions of training data. The fractions xy below the dataset names indicate the proportion of training data used. Results marked with † are from NeCo [58], while those marked with ‡ are our own.

and MAE models with ViT-B/16 (using pseudo-labels from DiffCut and DINOv2R ViT-B/14), as well as post-training on ImageNet. Our post-training is efficient, requiring 17h52min on V100 (ViT-B) and 8h37min on A100 (ViTS). Pseudo-label generation for COCO, which is done only once and offline, requires 17h on a V100 GPU. More implementation details in Sec. A.

Evaluation setup. To assess our representations’ generality, we evaluate them on diverse tasks and datasets. These include semantic segmentation on PascalVOC [19], PascalContext [51], ADE20K [82], Cityscapes [14], COCO [48] (mIoU↑), and monocular depth prediction on NYUv2 [52] (using RMSE↓). COCO is in-domain, as its training data is used for post-training, while the others are outof-domain. These datasets test generalization to domain shifts between post-training and downstream tasks. We also evaluate robustness to intra-task domain shifts using the CS→ACDC [63] setting, where Cityscapes (CS) provides the support set and ACDC provides the query images for retrieval-based segmentation. While both datasets contain autonomous driving images, ACDC introduces challenging conditions like snow, night, fog, and rain, unlike Cityscapes’ clear-weather daylight images.

All results reported in this paper are produced using our implementation unless marked with †. For DINOv2R and NeCo, we evaluate the publicly available pretrained model checkpoints to ensure a fair comparison.

#### 4.2. Retrieval-based Scene Understanding

Comparison with state-of-the-art in low-shot regimes. In Tab. 1, we compare our DIP approach with prior state-

of-the-art methods for dense image representation learning on retrieval-based semantic segmentation using the Pascal VOC and ADE20K datasets. We evaluate both full-data (11) and low-shot regimes (18, 641 , and 1281 ), where only a fraction of the training examples is used in the support set.

Our DIP approach consistently outperforms DINOv2R, with the performance gap widening as the number of training examples decreases, particularly for the PascalVOC dataset (see Fig. 4). Notably, DIP achieves superior results compared to prior work in most settings for both ViT-S and ViT-B, while remaining competitive in others.

Comparison with DINOv2R and NeCo. We compare our DIP method with DINOv2R, the base model for our post-training, and NeCo, a recent method that also posttrains DINOv2R. In Tab. 2 we evaluate these methods on in-context semantic segmentation tasks across six datasets.

Our approach shows consistent improvements over DINOv2R on all datasets, both in-domain (COCO) and out-of-domain (ADE20K, PascalVOC, Pascal-Context, and Cityscapes), highlighting the generalization capability of our representations. Unlike NeCo, our DIP consistently improves over DINOv2R, achieving higher mIoU and average performance improvements (see “Avg. Delta” in Tab. 2).

Intra-task domain shift. In Tab. 2, the CS→ACDC setting evaluates the robustness of the learned representations to domain shifts between the support set and query images during the downstream stage, as explained in Sec. 4.1. Although our post-training DIP method is not specifically designed to improve this type of robustness, it still achieves a 1.6 mIoU improvement over DINOv2R for ViT-B.

Method Backbone ADE20K PascalVOC Pascal-Context Cityscapes CS→ACDC COCO Avg. Delta DINOv2R [16] ViT-S/14 39.3 79.4 48.0 55.6 47.4 72.6 NeCo [58] ViT-S/14 38.9 (−0.4) 81.0 (+1.6) 49.4 (+1.4) 53.9 (−1.7) 47.2 (−0.2) 74.3 (+1.7) +0.40 DIP (ours) ViT-S/14 39.7 (+0.4) 81.0 (+1.6) 49.5 (+1.5) 55.8 (+0.2) 47.4 (+0.0) 74.0 (+1.4) +0.85

DINOv2R [16] ViT-B/14 40.8 79.0 49.0 58.4 50.5 72.9 NeCo [58] ViT-B/14 41.2 (+0.4) 82.4 (+2.4) 51.2 (+2.2) 57.9 (−0.5) 51.5 (+1.0) 75.4 (+2.5) +1.50 DIP (ours) ViT-B/14 42.6 (+1.8) 82.1 (+2.1) 51.5 (+2.5) 59.5 (+1.1) 52.1 (+1.6) 76.0 (+3.1) +2.00

- Table 2. In-context scene understanding benchmark. Dense nearest neighbor retrieval performance for semantic segmentation on six scene-centric datasets: ADE20K, PascalVOC, Pascal-Context, Cityscapes, CS→ACDC, and COCO. The first five are out-of-domain, while COCO is in-domain (used for post-training). Performance is measured using the mIoU metric. For NeCo and our DIP, which post-train DINOv2R, improvements over DINOv2R are shown in parentheses. The “Avg. Delta” column reports the average improvement.

Method DINOv2R NeCo DIP (ours) RMSE↓ .771 .769 .756

- Table 3. In-context monocular depth prediction on NYUv2 dataset [67]. RMSE scores (lower is better) scaled by 10 for readability reasons. All methods use ViT-S/14.

Method Backbone PascalVOC ADE20K DINOv2R [16] ViT-B/14 79.0 40.8

+DIP ViT-B/14 82.1 (+2.1) 42.6 (+1.8) CLIP [61] ViT-B/16 71.8 29.0

+ DIP ViT-B/16 73.8 (+2.0) 30.1 (+1.1) MAE [31] ViT-B/16 13.9 5.1

+ DIP ViT-B/16 47.3 (+33.4) 11.8 (+6.7)

- Table 4. Post-training with other base models. Dense nearest neighbor retrieval performance (mIoU) for semantic segmentation on ADE20K and PascalVOC, using DINOv2R, CLIP, and MAE as base models, before and after post-training.

Method Backbone PascalVOC ADE20K

SD [27] SSD-1B 59.4 24.4 SAM [43] ViT-B/16 32.8 12.9 DINOv2R [16] ViT-B/14 79.0 40.8 RADIOv2.5 [32] ViT-B/16 81.3 42.1 DIP (ours) ViT-B/14 82.1 42.6

Table 5. Comparison with Superised Baselines. Dense nearest neighbor retrieval performance (mIoU) for semantic segmentation on PascalVOC and ADE20K.

requires stronger supervision (manual segmentation masks) than SD, while RADIOv2.5 distills multiple model types (SAM, DINOv2, and CLIP-like features). Thus, both baselines leverage more supervision than our method. Additionally, RADIOv2.5 requires significantly more training compute. Despite these advantages, Tab. 5 shows DIP surpasses RADIOv2.5 in in-context semantic segmentation, confirming our method’s effectiveness. SAM underperforms in this setting, consistent with prior findings [42]. Importantly, DIP outperforms the SD features it uses for pseudosegmentation.

Monocular depth prediction. In Tab. 3, we report incontext depth results on NYUv2 (RMSE). Both DIP and NeCo improve over DINOv2R, with DIP achieving the best RMSE. This is notable, as our in-context pseudo-tasks are designed for semantic segmentation, yet the representations generalize well to depth prediction.

Additionally, we report linear segmentation results with consistent gains over DINOv2R and competitive results compared to NeCo in Tab. 8 in supplementary.

Post-training other base models. In Tab. 4, we evaluate in-context segmentation on ADE20K and PascalVOC using CLIP [61] and MAE [31] as base models, before and after DIP post-training. Our method consistently improves performance across all tested models, showing its versatility. MAE improves by +33.4 mIoU on PascalVOC. These gains transform MAE from weak to improved at in-context segmentation, highlighting our approach’s effectiveness in enhancing dense features.

#### 4.3. Ablations

In Tab. 6, we evaluate key design choices of DIP approach. In-context vs Direct dense prediction (Tab. 6a). We compare our in-context dense prediction objective with direct dense prediction, where the model predicts pseudo semantic segmentation maps using a classification head. Our in-context approach significantly outperforms the direct method on PascalVOC and reduces RMSE on NYUv2 depth prediction, while both perform similarly on ADE20K. Impact of “distractor” examples (Tab. 6b). Removing the “distractor” support examples significantly reduces performance, as the absence of “distractors” simplifies the post-training task, limiting the model’s ability to learn discriminative dense representations.

Comparison with supervised baselines To automatically generate pseudo-segmentations, our post-training approach leverages Stable Diffusion (SD) features [27], trained on weakly annotated (internet-scraped) image-caption pairs. In Tab. 5, we compare DIP to SD and supervised encoder features (SAM [43] and RADIOv2.5 [32]). Crucially, SAM

Objective PascalVOC ADE20K NYUv2↓

In-context 81.0 39.7 .756 Direct. 79.9 39.9 .776

(a) In-context pretraining is more effective.

Distractor PascalVOC ADE20K

✓ 81.0 39.7 ✗ 78.5 38.4

(b) Distractor examples are important.

Positive example PascalVOC ADE20K

Nearest neighbor 81.0 39.7 Two random crops 79.5 39.4

###### (c) Constructing positive examples.

#Pseudo-classes PascalVOC ADE20K

Segm. Labels PascalVOC ADE20K

300 80.8 39.5 500 80.9 39.7

Pseudo-labels w/ DiffCut 81.0 39.7 Pseudo-labels w/o DiffCut 59.1 25.5

Dataset PascalVOC ADE20K

1000 81.0 39.7 2000 80.8 39.8

COCO 81.0 39.7 ImageNet 80.8 39.6

Ground truth labels 81.9 39.8 (d) Pseudo-labels generation.

(e) Number of pseudo-classes.

(f) Post-training dataset.

Table 6. Ablation Study of DIP. Default settings in light blue . “Two random crops” in (c) refers to the case where the query and positive image are constructed as two random crops of the same original image.

Construction of positive examples (Tab. 6c). We compare two strategies for creating positive examples: (1) retrieving nearest neighbors using DINOv2R image-wise features and (2) using two random crops from the same image. The nearest neighbor strategy outperforms random crops, validating our design choice and showing that our method avoids reliance on complex augmentations common in selfdistillation and contrastive approaches.

Impact of DiffCut (Tab. 6d). We assess the role of DiffCut [15], which generates class-agnostic segments for pseudo segmentation maps before K-means clustering. Compared to direct K-means on DINOv2R features, DiffCut significantly improves performance, demonstrating its effectiveness as a training-free, unsupervised method leveraging Stable Diffusion. Furthermore, our pseudo labels achieve results nearly matching ground-truth semantic segmentation, underscoring their quality.

Impact of number of pseudo classes (Tab. 6e). We study the effect of pseudo-class count (K-means clusters) for pseudo-label generation. Our approach is robust to this hyperparameter, with stable performance across values.

Post-training data: scene-centric vs object-centric (Tab. 6f). We compare DIP post-trained on scene-centric COCO vs. object-centric ImageNet. Results are comparable, COCO slightly outperforms ImageNet. This indicates that our method does not depend on human-curated, objectcentric data and performs better on scene-centric data, easier to collect at scale.

#### 4.4. Qualitative results

- In Fig. 5, we present correlation maps between a query image patch and a reference image, comparing dense representations from DINOv2R and our DIP. While DINOv2R produces localized, part-based correlations, DIP captures semantic-level correspondences, more accurately delineating entire objects of the same semantic type as the query.

This shows that DIP provides better semantic correspondences. As visualized in Fig. 1, this enables DIP to retrieve more semantically coherent nearest neighbors, particularly

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

(a) Query (b) Reference (c) DINOv2R (d) DIP Figure 5. Correlation maps between a query image patch (highlighted with a red cross) and a reference image, comparing DINOv2R and DIP. DIP generates coherent, object-level correlations, while DINOv2R produces localized, part-level responses.

in low-shot regimes (as in Fig. 1), resulting in more accurate segmentation outputs compared to DINOv2R.

### 5. Conclusion

We introduced DIP, a novel unsupervised post-training method that enhances dense image representations for incontext scene understanding. Leveraging meta-learning principles and automatically generating in-context tasks with pseudo-segmentations using Stable Diffusion, our approach avoids complex self-distillation architectures. DIP is computationally efficient (<9h on an A100) and generalizes well across downstream dense retrieval tasks, including semantic segmentation and depth prediction. It outperforms both the initial pretrained vision encoder and prior state-ofthe-art post-training methods, providing a simpler and more effective solution for improving dense representations in vision models.

Acknowledgements This work was performed using HPC resources from GENCI–IDRIS (Grants 2025-AD011015037R1, 2024-AD011012884R3, and 2025-AD011016523). We thank PEPR Sharp (ANR-23-PEIA-0008, ANR, FRANCE 2030).

### References

- [1] Yuki Markus Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In ICLR, 2020. 2
- [2] Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Mike Rabbat, and Nicolas Ballas. Masked siamese networks for label-efficient learning. In ECCV, 2022. 2
- [3] Ivana Balaˇzevi´c, David Steiner, Nikhil Parthasarathy, Relja Arandjelovi´c, and Olivier J. H´enaff. Towards in-context scene understanding. In NeurIPS, 2023. 1, 2, 3, 4, 6, 13
- [4] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In ICLR, 2022. 2
- [5] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei A. Efros. Visual prompting via image inpainting. In NeurIPS, 2022. 1
- [6] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for selfsupervised learning. In ICLR, 2022. 2
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 1
- [8] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020. 2
- [9] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 2, 5, 6
- [10] Mathilde Caron, Neil Houlsby, and Cordelia Schmid. Location-aware self-supervised transformers for semantic segmentation. In WACV, 2024. 3
- [11] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 2
- [12] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2021. 2
- [13] Jang Hyun Cho, Utkarsh Mall, Kavita Bala, and Bharath Hariharan. Picie: Unsupervised semantic segmentation using invariance and equivariance in clustering. In CVPR,

2021. 3

- [14] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In CVPR,

2016. 6

- [15] Paul Couairon, Mustafa Shukor, Jean-Emmanuel Haugeard, Matthieu Cord, and Nicolas Thome. Diffcut: Catalyzing

- zero-shot semantic segmentation with diffusion features and recursive normalized cut. In NeurIPS, 2024. 2, 3, 5, 8
- [16] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In ICLR,

2024. 2, 5, 6, 7

- [17] Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu, and Baining Guo. Peco: Perceptual codebook for bert pre-training of vision transformers. In AAAI, 2023. 2
- [18] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3
- [19] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge. IJCV, 2010. 6
- [20] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Modelagnostic meta-learning for fast adaptation of deep networks. In ICML, 2017. 3
- [21] Spyros Gidaris and Nikos Komodakis. Dynamic few-shot visual learning without forgetting. In CVPR, 2018. 3
- [22] Spyros Gidaris and Nikos Komodakis. Generating classification weights with gnn denoising autoencoders for few-shot learning. In CVPR, 2019. 3
- [23] Spyros Gidaris, Andrei Bursuc, Nikos Komodakis, Patrick P´erez, and Matthieu Cord. Learning representations by predicting bags of visual words. In CVPR, 2020. 2
- [24] Spyros Gidaris, Andrei Bursuc, Gilles Puy, Nikos Komodakis, Matthieu Cord, and Patrick P´erez. Obow: Online bag-of-visual-words generation for self-supervised learning. In CVPR, 2021. 2
- [25] Spyros Gidaris, Andrei Bursuc, Oriane Simeoni, Anton´ın Vobeck`y, Nikos Komodakis, Matthieu Cord, and Patrick P´erez. Moca: Self-supervised representation learning by predicting masked online codebook assignments. TMLR, 2024. 2
- [26] Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. In NeurIPS, 2020. 2
- [27] Yatharth Gupta, Vishnu V Jaddipal, Harish Prabhala, Sayak Paul, and Patrick Von Platen. Progressive knowledge distillation of stable diffusion xl using layer level loss. arXiv preprint arXiv:2401.02677, 2024. 2, 5, 7
- [28] Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T. Freeman. Unsupervised semantic segmentation by distilling feature correspondences. In ICLR,

2022. 3

- [29] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask R-CNN. In ICCV, 2017. 2
- [30] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 2, 3
- [31] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2, 7

- [32] Greg Heinrich, Mike Ranzinger, Yin Hongxu, Yao Lu, Jan Kautz, Andrew Tao, Bryan Catanzaro, and Pavlo Molchanov. RADIOv2. 5: Improved baselines for agglomerative vision foundation models. In CVPR, 2025. 7
- [33] Olivier J H´enaff, Skanda Koppula, Jean-Baptiste Alayrac, Aaron Van den Oord, Oriol Vinyals, and Joao Carreira. Efficient visual pretraining with contrastive detection. In ICCV,

2021. 3

- [34] Olivier J H´enaff, Skanda Koppula, Evan Shelhamer, Daniel Zoran, Andrew Jaegle, Andrew Zisserman, Jo˜ao Carreira, and Relja Arandjelovi´c. Object discovery and representation networks. In ECCV, 2022. 3
- [35] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 12
- [36] Zejiang Hou, Fei Sun, Yen-Kuang Chen, Yuan Xie, and SunYuan Kung. Milan: Masked image pretraining on language assisted representation. arXiv preprint arXiv:2208.06049,

2022. 2

- [37] Jyh-Jing Hwang, Stella X Yu, Jianbo Shi, Maxwell D Collins, Tien-Ju Yang, Xiao Zhang, and Liang-Chieh Chen. Segsort: Segmentation by discriminative sorting of segments. In ICCV, 2019. 3
- [38] Xu Ji, Joao F Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In ICCV, 2019. 3
- [39] Ioannis Kakogeorgiou, Spyros Gidaris, Bill Psomas, Yannis Avrithis, Andrei Bursuc, Konstantinos Karantzalos, and Nikos Komodakis. What to hide from your students: Attention-guided masked image modeling. In ECCV, 2022. 2
- [40] Asako Kanezaki. Unsupervised image segmentation by backpropagation. In ICASSP, 2018. 3
- [41] Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. Dino-foresight: Looking into the future with dino. arXiv preprint arXiv:2412.11673, 2024. 3
- [42] Tommie Kerssies, Daan de Geus, and Gijs Dubbelman. How to benchmark vision foundation models for semantic segmentation? In CVPRw, 2024. 7
- [43] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 7
- [44] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012. 5
- [45] Mengcheng Lan, Xinjiang Wang, Yiping Ke, Jiaxing Xu, Litong Feng, and Wayne Zhang. Smooseg: smoothness prior for unsupervised semantic segmentation. NeurIPS, 2024. 3
- [46] Tim Lebailly, Thomas Stegm¨uller, Behzad Bozorgtabar, Jean-Philippe Thiran, and Tinne Tuytelaars. Cribo: Selfsupervised learning via cross-image object-level bootstrapping. In ICLR, 2024. 2, 3, 5, 6
- [47] Kehan Li, Zhennan Wang, Zesen Cheng, Runyi Yu, Yian Zhao, Guoli Song, Chang Liu, Li Yuan, and Jie Chen. Acseg: Adaptive conceptualization for unsupervised semantic segmentation. In CVPR, 2023. 3

- [48] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Doll´ar. Microsoft coco: Common objects in context. In ECCV, 2014. 5, 6
- [49] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 12
- [50] Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. A simple neural attentive meta-learner. In ICLR,

2018. 3

- [51] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In CVPR, 2014. 6
- [52] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 6
- [53] Pedro O O Pinheiro, Amjad Almahairi, Ryan Benmalek, Florian Golemo, and Aaron C Courville. Unsupervised learning of dense visual representations. In NeurIPS, 2020. 3
- [54] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 2, 3
- [55] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. In TMLR, 2023. 2
- [56] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. TMLR, 2024. 1, 2, 5
- [57] Valentinos Pariza, Mohammadreza Salehi, and Yuki Asano. Hummingbird evaluation for vision encoders, 2024. 4, 13
- [58] Valentinos Pariza, Mohammadreza Salehi, Gertjan J Burghouts, Francesco Locatello, and Yuki M Asano. Near, far: Patch-ordering enhances vision foundation models’ scene understanding. In ICLR, 2025. 1, 2, 3, 4, 5, 6, 7
- [59] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366, 2022. 2
- [60] Siyuan Qiao, Chenxi Liu, Wei Shen, and Alan L Yuille. Fewshot image recognition by predicting parameters from activations. In CVPR, 2018. 3
- [61] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 7
- [62] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In ICLR, 2017. 3
- [63] Christos Sakaridis, Haoran Wang, Ke Li, Ren´e Zurbr¨ugg, Arpit Jadon, Wim Abbeloos, Daniel Olmeda Reino, Luc Van Gool, and Dengxin Dai. ACDC: The adverse conditions dataset with correspondences for robust semantic driving scene perception. In ICCV, 2021. 6

- [64] Mohammadreza Salehi, Efstratios Gavves, Cees GM Snoek, and Yuki M Asano. Time does tell: Self-supervised timetuning of dense image representations. In ICCV, 2023. 1, 2, 5, 6
- [65] Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In ICML, 2016. 3
- [66] Hyun Seok Seong, WonJun Moon, SuBeen Lee, and Jae-Pil Heo. Leveraging hidden positives for unsupervised semantic segmentation. In CVPR, 2023. 3
- [67] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV. Springer, 2012. 7
- [68] Walter Simoncini, Andrei Bursuc, Spyridon Gidaris, and Yuki Asano. No train, all gain: Self-supervised gradients improve deep frozen representations. In NeurIPS, 2024. 3
- [69] Sophia Sirko-Galouchenko, Alexandre Boulch, Spyros Gidaris, Andrei Bursuc, Antonin Vobecky, Patrick P´erez, and Renaud Marlet. Occfeat: Self-supervised occupancy feature prediction for pretraining bev segmentation networks. In CVPRw, 2024. 3
- [70] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In NeurIPS, 2017. 3
- [71] Thomas Stegm¨uller, Tim Lebailly, Behzad Bozorgtabar, Tinne Tuytelaars, and Jean-Philippe Thiran. Croc: Crossview online clustering for dense visual representation learning. In CVPR, 2023. 3, 5, 6
- [72] Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to compare: Relation network for few-shot learning. In CVPR, 2018. 3
- [73] Junjiao Tian, Lavisha Aggarwal, Andrea Colaco, Zsolt Kira, and Mar Gonzalez-Franco. Diffuse attend and segment: Unsupervised zero-shot segmentation using stable diffusion. In CVPR, 2024. 3
- [74] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 4
- [75] Shashanka Venkataramanan, Valentinos Pariza, Mohammadreza Salehi, Lukas Knobel, Spyros Gidaris, Elias Ramzi, Andrei Bursuc, and Yuki M Asano. Franca: Nested matryoshka clustering for scalable visual representation learning. arXiv preprint arXiv:2507.14137, 2025. 1, 2
- [76] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. In NeurIPS, 2016. 2, 3
- [77] Antonin Vobecky, David Hurych, Oriane Sim´eoni, Spyros Gidaris, Andrei Bursuc, Patrick P´erez, and Josef Sivic. Unsupervised semantic segmentation of urban scenes via crossmodal distillation. IJCV, 2025. 3
- [78] Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, and Lei Li. Dense contrastive learning for self-supervised visual pre-training. In CVPR, 2021. 3
- [79] Zhenda Xie, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. Propagate yourself: Exploring pixel-level consistency for unsupervised visual representation learning. In CVPR, 2021. 3

- [80] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In CVPR, 2022. 2
- [81] Sukmin Yun, Hankook Lee, Jaehyung Kim, and Jinwoo Shin. Patch-level representation learning for self-supervised vision transformers. In CVPR, 2022. 6
- [82] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017. 6
- [83] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. In ICLR, 2022. 2, 5
- [84] Adrian Ziegler and Yuki M Asano. Self-supervised learning of object parts for semantic segmentation. In CVPR, 2022. 1, 2, 3, 5, 6

## DIP: Unsupervised Dense In-Context Post-training of Visual Representations Supplementary Material

### A. Implementation details

During post-training, we fine-tune only the last three transformer blocks of the pretrained ViT encoder f(·) while keeping the remaining layers frozen. Our MLP projector h(·) consists of two linear layers with a non-linear activation function GELU [35]. Hidden feature dimension is set to 7D and fixed output dimension 6144. ℓ2-normalization is applied at the output of the MLP. We set the temperature parameter τ of the softmax operator to 0.07 (see Tab. 7). Our ablation study demonstrates that the method is robust to the choice of temperature, with values ranging from 0.03 to 0.09 yielding similar performance on the PascalVOC dataset. During post-training our support set consists of 1 positive and 7 “distractor” examples (8 total examples). During training, we use the AdamW [49] optimizer with a learning rate of 2.25×10−7 and a weight decay of 0.05. We train for 5 epochs. We employ cosine learning rate schedule.

Temperature 0.09 0.07 0.03 PascalVOC 80.8 81.0 81.0

Table 7. Ablation of temperature τ during post-training.

Generation time of pseudo semantic segmentation labels. COCO pseudo-label generation required 17 hours on 1 V100 GPU, with DiffCut inference as the bottleneck due to its current lack of batch processing optimization and suboptimal GPU utilization. However, this one-time offline process can be used to post-train multiple encoders.

### B. Additional quantitative results

##### In-context scene understanding: impact of memory size.

- In Fig. 6, we analyze the effect of memory size on incontext semantic segmentation using the ADE20K dataset (full) for DINOv2R, NeCo, and our DIP. Results show that DIP outperforms both DINOv2R and NeCo across all memory sizes.

Linear segmentation. Tab. 8 presents linear segmentation results on COCO and ADE20K benchmarks. For fair comparison, we re-evaluated both DINOv2R and NeCo using our implementation, ensuring consistent evaluation protocols across all methods. Our approach consistently improves over the strong DINOv2R baseline and shows improvements over NeCo. Notably, with the ViT-B/14 back-

Method Backbone COCO ADE20K DINOv2R ViT-S/14 82.1 33.5 NeCo ViT-S/14 81.1 (−1.0) 33.1 (−0.4) DIP (ours) ViT-S/14 82.6 (+0.5) 33.7 (+0.2)

DINOv2R ViT-B/14 85.5 38.6 NeCo ViT-B/14 85.2 (−0.3) 39.5 (+0.9)

DIP (ours) ViT-B/14 86.7 (+2.0) 39.5 (+0.9)

Table 8. Linear segmentation results on COCO and ADE20K datasets. All methods (DINOv2R, NeCo, and DIP) are evaluated using our implementation. DIP consistently improves over our base model DINOv2R and outperforms NeCo across both datasets.

DINOv2R NeCo DIP (ours)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

42

42

40

40

mIoU(%)

38

38

36

36

34

34

105 106 107

105 106 107

Memory Size

Memory Size (a) ViT-B/14

(a) ViT-S/14

Figure 6. In-context scene understanding: impact of memory size. Semantic segmentation performance on ADE20K (full dataset) using dense nearest neighbor retrieval, evaluated across varying memory sizes.

bone on COCO, our method achieves 86.7 mIoU, surpassing DINOv2R by 2.0 points.

PascalVOC 1 18 641 1281

Method Backbone

DINOv2R ViT-B 79.0 75.3 67.6 60.3 DIP (ours) ViT-B 82.1 79.6 75.1 70.1

DINOv2R ViT-L 76.9 72.8 61.4 54.4 DIP (ours) ViT-L 81.1 78.7 70.0 64.6

Table 9. Larger backbones evaluation We show performance of DIP and DINOv2R on a larger backbone ViT-L. Dense nearest neighbor retrieval performance (mIoU) for semantic segmentation on PascalVOC across varying proportions of training data.

Larger backbones. While ViT-L (DINOv2) underperforms ViT-B in in-context segmentation [3, 57], our method still improves results with ViT-L, as shown in Tab. 9. This demonstrates DIP’s scalability across backbone sizes.

PascalVOC ADE20K

1 18 641 1281 1 18 641 1281 Two Crops 79.5 75.1 67.7 61.2 39.4 33.2 24.7 22.4 NN 81.0 77.7 71.4 65.9 39.7 33.7 25.6 23.2

Table 10. Additional ablation on the construction of positive examples. Dense nearest neighbor retrieval performance (mIoU) for semantic segmentation on PascalVOC and ADE20K across varying proportions of training data.

Nearest Neighbors (NN) vs. Two Crops We compare two strategies for creating positive examples: (1) retrieving nearest neighbors using DINOv2R image-wise features (NN) and (2) using two random crops from the same image. NN consistently outperforms Two Crops, with the performance gap widening when fewer training examples are available (see Tab. 10). This scalability advantage justifies our design choice.

### C. Additional qualitative results

We present additional examples of automatically constructed in-context tasks in Fig. 7, showing the quality of our pseudo-labeling approach. We display query images paired with their corresponding positive support examples, along with both pseudo-labels and ground truth labels, included only for comparison.

[Figure 38]

[Figure 39]

(a) Query image (b) Pseudo labels (c) GT labels (d) Positive image (e) Pseudo labels (f) GT labels

Figure 7. Examples of automatically constructed in-context scene understanding tasks. Each row shows a query image and its corresponding positive support example. (a) and (b) display the query image and its pseudo segmentation labels, while (d) and (e) show the positive support image and its pseudo segmentation labels. (c) and (f) present the ground truth segmentation labels for the query and positive images, respectively, included only for comparison with the pseudo labels.

