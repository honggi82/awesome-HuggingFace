### Visual In-Context Prompting

Feng Li♠, Qing Jiang♯, Hao Zhang♠, Tianhe Ren†, Shilong Liu¶, Xueyan Zou§, Huaizhe Xu♠, Hongyang Li♯, Chunyuan Li‡, Jianwei Yang‡1, Lei Zhang†2, Jianfeng Gao‡2

♠ HKUST ‡ Microsoft Research, Redmond † IDEA ♯ SCUT ¶ Tsinghua § UW-Madison

1. Project Lead 2. Equal Advisory Contribution

[Figure 1]

# arXiv:2311.13601v1[cs.CV]22Nov2023

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

DINOv

DINOv

…

N in-context examples

|Visual Prompting Generic Segmentation|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

DINOv

DINOv

|Visual Prompting Referring Segmentation|
|---|

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

DINOv

|Zero-shot Video Object and Part Segmentation|
|---|

Figure 1. Our model DINOv supports generic and referring segmentation to associate multiple or single objects with the user input visual prompts. A user can input one or more in-context visual prompts (scribbles, masks, boxes, etc.) to improve the segmentation performance.

#### Abstract

prompts like strokes, boxes, and points. We further enhance it to take an arbitrary number of reference image segments as the context. Our extensive explorations show that the proposed visual in-context prompting elicits extraordinary referring and generic segmentation capabilities to refer and detect, yielding competitive performance to close-set in-domain datasets and showing promising results on many open-set segmentation datasets. By joint training on COCO and SA-1B, DINOv achieves 57.7 PQ on COCO and 23.2 PQ on ADE20K. Code will be available at https://github.com/UX-Decoder/DINOv

In-context prompting in large language models (LLMs) has become a prevalent approach to improve zero-shot capabilities, but this idea is less explored in the vision domain. Existing visual prompting methods focus on referring segmentation to segment the most relevant object, falling short of addressing many generic vision tasks like open-set segmentation and detection. In this paper, we introduce a universal visual in-context prompting framework for both tasks, as shown in Fig. 1. In particular, we build on top of an encoder-decoder architecture, and develop a versatile prompt encoder to support a variety of

#### 1. Introduction

|Prompt|
|---|

|Tasks|
|---|

Visual In-Context Prompt

The recent progress in large language models (LLMs) like GPT [1,27] has shown promising results towards artificial general intelligence (AGI) by training unified models on large amounts of text data. These giant LLMs manifest themselves with intriguing emerging capabilities such as incontext learning. Nevertheless, similar paradigms have not yet succeeded in solving all vision tasks due to the diversity of scenarios in computer vision [15]. Some works [23,50] have combined LLMs and vision models to tackle complex image understanding tasks with text outputs such as visual question answering, but challenges remain in fine-grained tasks that require pixel-level outputs, like instance masks, rather than just text.

[Figure 16]

[Figure 17]

###### SegGPT

…

N in-context examples

Visual prompt

Ours

[Figure 18]

SAM

Image prompt

SEEM

[Figure 19]

OWL-VIT/ MQ-Det

Text Prompt

Grounding DINO/ GLIP

OVSeg

<cat>

Referring Generic

[Figure 20]

| |[Figure 21]|
|---|---|
| | |
|or| |

| |[Figure 22]<br><br>[Figure 23]|
|---|---|

[Figure 24]

To this end, the community has observed a growing interest in the development of language-enhanced vision foundation models. These models demonstrate profound competencies in open-world visual understanding tasks using text prompts, encompassing areas like open-set detection [19, 24, 45] and segmentation [38, 42, 45, 51]. Visual prompt, a different prompting mechanism has been explored in some recent segmentation models [13, 17, 52]. In these works, different visual prompting formats (e.g., points, boxes and strokes, etc) have been explored to facilitate the segmentation of visual contents specified by users.

Input Image Segmentation Output

Figure 2. Comparison with related works. Generic: segment all objects of the same semantic concept that match the user prompt. Refer: segment a particular object with the user input visual prompts. Image prompt: crop the image regions as prompts. (single) Visual prompt: one image-prompt example to segment. Incontext prompt: one or multiple image-prompt examples. We can do single-image and cross-image visual prompting tasks and support referring and generic segmentation.

In-context learning, an appealing capability in LLMs, has been less explored. It specifies the new task instruction using examples, and allows models to adapt to new tasks or domains without explicit retraining. One pioneering work for visual in-context learning is SegGPT [34], which demonstrates the ability to output an image mask based on visual examples. However, these works focus on associating a user visual prompt with one most relevant object and have the limited ability to identify multiple objects of the same semantic concept. More importantly, prompting in the image pixels with colorful masks inherently fails to generalize to novel concepts. As such, it is not competent to address many generic vision tasks like open-set object detection and segmentation, which often require the segmentation of multiple objects of a given concept. On the other hand, textual prompting in vision models exhibit notable flexibility in managing both referring and generic tasks in detection or segmentation [24, 51]. Nevertheless, they are arguably not favorable for in-context settings in that they cannot take segmentation masks as the inputs. In this paper, we strive to develop a model that supports visual in-context prompting for all types of image segmentation tasks. A comparison between our work and previous work is shown in Fig. 2. Besides supporting both single-image and cross-image visual prompting, our model distinguishes itself by effectively handling both referring and generic segmentation problems.

support versatile visual prompting capabilities, based on the unified detection and segmentation model MaskDINO [18]. DINOv follows the general encoder-decoder design with an extra prompt encoder to formulate and sample visual prompts. The decoder takes in segmentation queries and reference prompt queries to generate segmentation masks and target visual prompts, and we associate the output segmentation masks with the target prompt queries for the final output. We can define the visual in-context samples with a set of reference image (Q) - visual prompt (A) pairs. The visual prompt can be in various types, including mask, scribble, box, etc. With the in-context examples, our model takes in a target image and outputs the masks. The creation of target visual prompts involves an initial step where a prompt encoder extracts reference visual prompts from a Q-A pair. This is followed by a decoder to get the target visual prompt by attending reference visual prompts to the target image. During training, to construct positive and negative samples for generic segmentation, we sample reference visual prompts in a batch across different images. To address task and data discrepancies, we formulate generic latent queries and point queries for generic and referring segmentation, respectively. By joint training on COCO [21] and SA-1B [13] for generic and referring segmentation, our model attains competitive performance on in-domain segmentation tasks compared with text-prompted models and shows promising generalization capability on a wide range

To achieve this goal, we build a model called DINOv to

Loss Pivot Matching & Loss

Loss Pivot

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

|0|
|---|

|0|
|---|

|1|
|---|

|0|
|---|

- 0

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

- 1

|0|
|---|

|0|
|---|

|0|
|---|

|1|
|---|

| | |
|---|---|
| | |

[Figure 29]

[Figure 30]

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

DINOv

Prompt Encoder

[Figure 31]

|1|
|---|

DINOv

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

Share DINOv

0

|0|
|---|

|1|
|---|

|0|
|---|

|0|
|---|

|1|
|---|

DINOv

|0|
|---|

|1|
|---|

|0|
|---|

|0|
|---|

|1|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

DINOv

|0|
|---|

|0|
|---|

|0|
|---|

|1|
|---|

Visual Prompts

Visual Prompt

Segmentation Query

Sample & Aggregate

Vision Tokens

[Figure 32]

Prompt Encoder

DINOv

| | |
|---|---|
| | |

Prompt Encoder

[Figure 33]

|[Figure 34]<br><br>[Figure 35]|
|---|

Vision Encoder

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

###### ...

|[Figure 42]|
|---|

###### ...

Input Images Input Image Visual Prompts

Input Images

Visual Prompts

(a) General Framework (b) Visual Generic Segmentation (c) Visual Referring Segmentation

Figure 3. DINOv is a universal segmentation framework that can do generic segmentation and referring image segmentation. The vision encoder is used to extract image features.(b) An illustration of losses for visual generic segmentation. In the example, there are 6 visual prompts sampled from 6 masks from 3 categories. The visual prompts from the instances of the same class are averaged as the class embedding. Each colume of the matching matrix is a 3-dimension one-hot vector which is a one-hot class label of the instance; (c) An illustration of losses for visual referring segmentation. Each visual prompt is classified to one of the 6 instances.

The interested objects can be a particular object for referring segmentation or all objects of the same semantic concept for generic segmentation. Note that the reference image can be identical to the target image, in which the task reduces to single-image visual prompting segmentation.

of open-set segmentation benchmarks using purely visual prompts.

To summarize, our contributions are threefold:

- • We are the first to extend visual in-context prompting to support generic vision tasks like open-set generic segmentation and detection, and achieve comparable performance with text prompt-based open-set models.
- • We build DINOv, a unified framework for referring segmentation and generic segmentation based on visual in-context prompting. This unification simplifies model design and allows our model to consume both semantically-labelled and unlabelled data for better performance.
- • We conduct extensive experiments and visualizations to show that our model can handle generic, referring, and video object segmentation tasks. Our early attempts exhibit promising results on open-set segmentation and detection with visual prompting.

To address these tasks, DINOv utilizes a comprehensive query-based encoder-decoder architecture. This architecture comprises a vision encoder, denoted as Enc, responsible for extracting image features, a prompt encoder referred to as PromptEncoder, designed to extract visual prompt features by combining image features and userprovided visual prompts, and a general decoder represented as Decoder, which generates masks and visual concepts based on the segmentation query and visual prompt features. Upon receiving the input image and user-provided visual prompts, our initial step involves extracting image features denoted as Z using the vision encoder. Subsequently, we feed both the image features and visual prompts into the prompt encoder to extract the reference visual prompt F and subsequently sample the query visual prompt features Qp.Formally, we have:

#### 2. Method 2.1. Unified Formulation for Segmentation Tasks

Z = Enc(I),Z = Enc(It) F = PromptEncoder(P,Z)

(1)

In this paper, we concentrate on visual prompting tasks involving images, encompassing both generic segmentation and referring segmentation tasks. Given N reference images I = {I1,...,IN} ∈ RN×H×W×3 with the corresponding visual prompts P = {p1,...,pN}, DINOv aims to segment objects of interest on a new target image It. The visual prompts include masks, boxes, scribbles, points, etc.

Qp = PromptSample(F)

In addition to the visual prompt features Qp, DINOv incorporates segmentation queries Qs for proposal extraction. A shared decoder is employed to decode outputs for both Qs and Qp while performing cross-attention with respect to the target image feature Z.

Train Inference

DINOv

DINOv

[Figure 43]

[Figure 44]

...

...

[Figure 45]

Learnable Content Queries

Learnable Content Queries

[Figure 46]

...

Duplicate ... Duplicate

Learnable Position Queries (Boxes)

Sampled Position Queries (Points)

(a) Generic queries for generic segmentation

(b) Interactive point queries for referring segmentation

Figure 4. DINOv query formulation of generic and referring segmentation tasks.

[Figure 47]

object features. g is the linear projection for generic segmentation task. Each of Ns objects is classified into one of Np classes. For visual referring segmentation, the objective differs. Here, each visual prompt is employed to identify the most closely matched instance within the target image. This task can be framed as a classification problem, where each visual prompt is assigned to a specific instance within the target image. It’s important to note that during our training phase, the target image and the reference image are identical. The matching score matrix for referring segmentation is structured as follows:

Figure 5. Prompt encoder to encode visual prompt from reference images. We use three masked cross-attention from the vision encoder small feature map to large feature map.

###### Cr = h(Op)h(OsT),Cr ∈ RN

###### s×Nq (4)

h is the linear projection for referring segmentation task. Fig. 6(b) and (c) provide an illustrative representation of the two tasks. In our implementation, the generic segmentation task involves finding the most suitable visual prompt for each mask proposal, effectively pivoting the loss from a query to all prompts. Conversely, the referring segmentation task focuses on matching a given visual prompt to a specific mask proposal, with the loss pivot transitioning from a prompt to all proposals. As indicated in Equations 3 and 4, the PromptClassifier for both generic and referring segmentation tasks share a similar formulation. Consequently, they can share the entire framework, except for the two distinct linear layers denoted as g and h.

Os = Decoder(Qs;Z) Op = Decoder(Qp;Z)

(2)

⟨M,B⟩ = MaskHead(Os) Cg,Cr = PromptClassifier(Os,Op)

Here, Os represents the decoded segmentation query features, Op corresponds to the decoded target visual prompt features, while M and B denote the predicted masks and boxes, respectively. Furthermore, we have Cg and Cr as the predicted matching scores for generic segmentation and referring segmentation tasks. These scores are derived through the use of a PromptClassifier, which computes the similarity between Os and Op.

###### 2.2. Visual Prompt Formulation

PromptClassifier. We clarify the definition of the prompt classifier, denoted as PromptClassifier(·,·), for both generic segmentation and referring segmentation tasks here. In the case of generic segmentation tasks like instance and panoptic segmentation, the typical objective is to classify object features Os into respective categories. When employing visual prompting for generic segmentation tasks, the distinction lies in the utilization of visual prompt features Op as class embeddings. This is illustrated in the following equation:

The heart part of our DINOv is the proposed visual prompting mechanism. As shown in Eq. 1 and Eq. 2, we employ two modules to get the final visual prompt:

- • A PromptEncoder to encode the reference visual prompt F from the reference image features (followed by a sampling process to get query visual prompt Qp).
- • A Decoder (shared with the segmentation decoder)

to decode outputs for the target visual prompt Op by interacting with the target image features.

###### Cg = g(Os)g(OpT),Cg ∈ RN

p×Ns (3) where Np and Ns are the number of visual prompts and

This design allows our model to first encode the reference visual prompt and then adapt the prompt to the target image

CVPR

#15147

CVPR 2024 Submission #15147. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.

CVPR

#15147

- 432
- 433
- 434
- 435
- 436
- 437
- 438
- 439
- 440
- 441
- 442
- 443
- 444
- 445
- 446
- 447
- 448
- 449
- 450
- 451
- 452
- 453
- 454
- 455
- 456
- 457
- 458
- 459
- 460
- 461
- 462
- 463
- 464
- 465
- 466
- 467
- 468
- 469
- 470
- 471
- 472
- 473
- 474
- 475
- 476
- 477
- 478
- 479
- 480
- 481
- 482
- 483
- 484
- 485

Algorithm 1: Pseudo code of Prompt Sampling for Generic Segmentation Task.

# Inputs: A list of encoded reference visual prompts F with length M, M is the number of visual in-context prompting examples. The ground-truth semantic category (i.e., dogs) of each reference visual prompt forms another list C with length M. During training, F is acquired from a batch of training images (i.e., 64 images). During inference, the batch is the whole training image set.

# Variables: Defined maximum in-context length N for each semantic category. # Output: Visual prompt features Qp

- 1 def Prompt Sample(F):

- 2 C=Unique(C);# C is a list that contains all the semantic categories in this training batch.
- 3 Fc=Dict();# Fc is visual prompt dict, where key is the semantic category and value are the reference prompt features.
- 4 Fc[c]=[] for c in C; # Initialize visual prompt dict by semantic category.
- 5 Fc[c].append(f) for c, f in zip(C, F); # Gather visual prompts of the same semantic category.
- 6 n = Randint(1, N; # Randomly select the number of in-context examples.
- 7 Sc = RandomSelect(F[c], n) for c in C; # For each semantic category, randomly select n prompts to represent a semantic category.
- 8 Qp = [Aggregate(Sc[c]) for c in C]; # Perform prompt aggregation to get one reference prompt token from multiple in-context prompt features for each semantic category.

Algorithm 1: Pseudo code of Prompt Sampling for Generic Segmentation Task.

###### 2.3. Prompt Sampling

as its representative visual prompt feature. This ensures that our inference stage has the same number of categories as in traditional open-set evaluation to prevent information leakage.

# Inputs: A list of encoded reference visual prompt F with length M, M is the total number of possible prompting examples. The ground-truth semantic category (i.e., dogs) of each reference visual prompt forms another list C with length M. During training, F is acquired from a batch of training images (i.e., 64 images). During inference, the batch is the whole training image set.

We propose two prompt sampling strategies for referring segmentation and generic segmentation, respectively.

Referring segmentation. For referring segmentation, we perform “self-referring” during training, where the reference image is the same as the target image. We sample a prompt from an instance and train the model to refer to the same instance. Therefore, we can utilize large-scale segmentation data such as SA-1B to train our model. Although trained on the same instances, our model can do cross-image referring during inference. As illustrated in Fig. 3(c), we can change target images into different images to perform cross-image referring.

# Variables: Defined maximum in-context length N for each semantic category. # Output: Query visual prompt Qp

- 1 def Prompt Sample(F):

- 2 C=Unique(C);# C is a list that contains all the semantic categories in this training batch.
- 3 Fc=Dict();# Fc is visual prompt dict, where key is the semantic category and value are the reference prompt features.
- 4 Fc[c]=[] for c in C; # Initialize visual prompt dict by semantic category.
- 5 Fc[c].append(f) for c, f in zip(C, F); # Gather visual prompts of the same semantic category.
- 6 n = Randint(1, N; # Randomly select the number of in-context examples.
- 7 Sc = RandomSelect(F[c], n) for c in C; # For each semantic category, randomly select n prompts to represent a semantic category.
- 8 Qp = [Aggregate(Sc[c]) for c in C]; # Perform prompt aggregation to get one reference prompt token from multiple in-context prompt features for each semantic category.

###### 2.4. Decoder Query Formulation

In DINOv, we designed two types of segmentation queries to address two different tasks as depicted in Fig. 4. For generic segmentation, the query is a number of learnable ones similar to MaskDINO [16]. For the visual referring task, we adopt the interactive point query following Semantic-SAM [17], so that we can exploit the rich granularities in SA-1B [13]. Similar to Semantic-SAM, the visual prompts (points or boxes) are both converted into anchor box format, and then the position of each visual prompt will be encoded into position queries. Each position query is duplicated and subsequently combined with content queries of different granularities as the final segmentation queries. For the training on SA-1B, in order to avoid excessive computational overhead on the model, we selectively sample a subset of points contained within this visual concept as positive point queries. Concurrently, we randomly sample a subset of points from the remaining areas to serve as negative points. During the inference stage, we sample the initial point position queries on 20×20 uniformly distributed grid as the initial point position for a single frame. [Hao: The query type names should be unified with Fig.4]

Generic segmentation. The sampling strategies are slightly different during training and inference:

in a flexible way. As we attempt to express visual concepts through visual prompts, a straightforward way is to employ a pre-trained vision encoder (e.g., CLIP [29]) to process the reference images guided by user prompts [26]. However, it may encounter several challenges: (i) the vision encoder takes cropped images as inputs, which causes substantial domain shift, especially for small objects [48]; (ii) The visual features extracted from CLIP tend to be more semantic and may not meet the demands in VOS tasks. As we will show in our ablation study, employing a CLIP vision encoder to extract visual prompts has a clear inferior generalization ability.

to process the features at the corresponding positions to get the visual prompt features.

the same instances, our model demonstrates the capability to perform cross-image referring during inference. As illustrated in Fig. 6(c), we can change the target images to various different images, enabling the model to effectively engage in cross-image referring tasks.

in a flexible way. As we attempt to express visual concepts through visual prompts, a straightforward way is to employ a pre-trained vision encoder (e.g., CLIP [29]) to process the reference images guided by user prompts [26]. However, it may encounter several challenges: (i) the vision encoder takes cropped images as inputs, which causes substantial domain shift, especially for small objects [48]; (ii) The visual features extracted from CLIP tend to be more semantic and may not meet the demands in VOS tasks. As we will show in our ablation study, employing a CLIP vision encoder to extract visual prompts has a clear inferior generalization ability.

- • Training. During training, it is critical to construct positive and negative visual prompt samples. Therefore, we construct visual prompts through a large image training batch. As shown in Algorithm 1, we initially group together reference visual prompts of the same semantic category across all images within a batch. For each semantic category, we randomly select [1,N] in-context examples and perform an aggregation to create a single reference visual prompt fea-

ture for each semantic category Qp, which will be fed to the decoder to interact with the target image to get the final target visual prompt Op. Hence, the number of semantic categories is the same as the target visual prompts. Note that a batch of images may not contain all the semantic categories in the dataset, therefore the number of semantic categories varies during training.

- • Inference. During the inference stage, taking the COCO Dataset as an example, we pre-extract corresponding visual prompt features based on mask prompts for all semantic categories within the training set. Subsequently, for evaluation, we randomly select N (default to 16) features for each semantic category

###### 2.3. Prompt Sampling

We introduce two prompt sampling strategies tailored for referring segmentation and generic segmentation, respectively.

Generic segmentation. The sampling strategies are slightly different during training and inference:

Referring segmentation. In the case of referring segmentation, we employ a “self-referring” approach during training, wherein the reference image is identical to the target image. Here, we sample a prompt from an instance and train the model to refer to the same instance. This approach allows us to leverage extensive segmentation data, such as SA-1B, for training our model effectively. Despite being trained on the same instances, our model demonstrates the capability to perform cross-image referring during inference. As illustrated in Fig. 6(c), we can change the target images to various different images, enabling the model to effectively engage in cross-image referring tasks.

- • Training. In the training process, it is crucial to create both positive and negative visual prompt samples. To achieve this, we generate visual prompts by utilizing a large image training batch. As depicted in Algorithm 1, our approach begins by grouping together reference visual prompt F of the same semantic category across all images within a training batch. For each semantic category, we then randomly select a variable number of in-context examples, ranging from 1 to N, and perform an aggregation process to generate refer-

ence visual prompt tokens Qp, where each reference visual prompt token corresponds to a specific semantic category. Qp is subsequently fed into the decoder, where it interacts with the target image to produce the final target visual prompt Op. Consequently, we attain the same number of target visual prompts to semantic categories. It is noteworthy that a given batch of images may not encompass all semantic categories present in the dataset, resulting in a variable number of semantic categories during the training process.

- • Inference. During the inference stage, using the COCO dataset as an example, we pre-extract the respective visual prompt features based on mask prompts for all semantic categories established during the training phase. For evaluation purposes, we adopt a random selection approach, where we choose N (16 by default) features for each semantic category. These selected features act as representative visual prompt features for each category. This practice ensures that

To address these issues, we reuse the vision encoder in our model and develop a simple yet effective prompt encoder. It extracts visual features corresponding to the locations indicated by various forms of visual prompts. To capture visual details of different granularities, we have incorporated multiple layers (default to 3) of the Mask Cross Attention Layer, as shown in Fig. 5. Each layer takes the image features extracted at different levels (output multi-scale features from the vision encoder, ranging from lower to higher resolutions) as inputs, utilizes the regions defined by the visual inputs as masks, and employs learnable queries

##### 3. Experiments 3.1. Setup

To address these issues, we reuse the vision encoder in our model and develop a simple yet effective prompt encoder. It extracts visual features corresponding to the locations indicated by various forms of visual prompts. To capture visual details of different granularities, we have incorporated multiple layers (default to 3) of the Mask Cross Attention Layer, as shown in Fig. 5. Each layer takes the image features extracted at different levels (output multi-scale features from the vision encoder, ranging from lower to higher resolutions) as inputs, utilizes the regions defined by the visual inputs as masks, and employs learnable queries to process the features at the corresponding positions to get the visual prompt features.

Dataset and Settings. In our experiments, we jointly train on two types of data: segmentation data with semantic labels and segmentation data with only pixel annotations (SA-1B [13]. For semantically-labeled data, we use COCO2017 [21] panoptic segmentation dataset with around 110K images. For SA-1B, we employ a 20% portion subset with around 2M images. We evaluate our model

Generic segmentation. The sampling strategies are slightly different during training and inference:

• Training. In the training process, it is crucial to create both pos-

5

###### 2.3. Prompt Sampling

We introduce two prompt sampling strategies tailored for referring segmentation and generic segmentation, respectively.

Referring segmentation. In the case of referring segmentation, we employ a “self-referring” approach during training, wherein the reference image is identical to the target image. Here, we sample a prompt from an instance and train the model to refer to the same instance. This approach allows us to leverage extensive segmentation data, such as SA-1B, for training our model effectively. Despite being trained on

- 486
- 487
- 488
- 489
- 490
- 491
- 492
- 493
- 494
- 495
- 496
- 497
- 498
- 499
- 500
- 501
- 502
- 503
- 504
- 505
- 506
- 507
- 508
- 509
- 510
- 511
- 512
- 513
- 514
- 515
- 516
- 517
- 518
- 519
- 520
- 521
- 522
- 523
- 524
- 525
- 526
- 527
- 528
- 529
- 530
- 531
- 532
- 533
- 534
- 535
- 536
- 537
- 538
- 539

our inference stage maintains the same number of categories as in traditional open-set evaluation, effectively preventing any potential information leakage.

###### 2.4. Decoder Query Formulation

In DINOv, we designed two types of segmentation queries to address two different tasks as depicted in Fig. 4. For generic segmentation, the query is a number of learnable ones similar to MaskDINO [16]. For the visual referring task, we adopt the interactive point query following Semantic-SAM [17], so that we can exploit the rich granularities in SA-1B [13]. Similar to Semantic-SAM, the visual prompts (points or boxes) are both converted into anchor box format, and then the position of each visual prompt will be encoded into position queries. Each position query is duplicated and subsequently combined with content queries of different granularities as the final segmentation queries. For the training on SA-1B, in order to avoid excessive computational overhead on the model, we selectively sample a subset of points contained within this visual concept as positive point queries. Concurrently, we randomly sample a subset of points from the remaining areas to serve as negative points. During the inference stage, we sample the initial point position queries on 20×20 uniformly distributed grid as the initial point position for a single frame.

#### 3. Experiments 3.1. Setup

Dataset and Settings. In our experiments, we jointly train on two types of data: segmentation data with semantic labels and segmentation data with only pixel annotations (SA-1B [13]. For semantically-labeled data, we use COCO2017 [21] panoptic segmentation dataset with around 110K images. For SA-1B, we employ a 20% portion subset with around 2M images. We evaluate our model on a wide range of tasks and datasets with only visual prompts, including: 1. Open-set panoptic segmentation on COCO2017 [21] and ADE20K [49]; 2. Segmentation in the wild (SegInW) [51] which includes 25 instance segmentation datasets; 3. Object detection in the wild (ODinW) [19] that encompasses over 35 datasets; 4. Zero-shot Video object segmentation (VOS) on DAVIS2017 [28], DAVIS2016Interactive [28], and Youtube-VOS 2018 [39].

Implementation Details. We provide implementation details in the Appendix.

Evaluation Metrics. For all segmentation and detection tasks, we use standard evaluation metrics: PQ (Panoptic Quality) for panoptic segmentation, AP (Average Precision) for instance segmentation (mask AP) and detection (box AP), and mIoU (mean Intersection over Union) for semantic segmentation. For VOS tasks, we follow previous semisupervised models to use region similarity J and contour

accuracy F. We also adopt the averaged score J&F as the metric for DAVIS2017 and averaged overall score G for Youtube-VOS 2018. Note that Youtube-VOS 2018 also reports J and F for seen and unseen splits.

###### 3.2. Generic Segmentation and Detection

We evaluate our visual prompt based generic segmentation performance in Table 1.

In-domain Segmentation on COCO. Compared to other models trained for visual prompts, we achieve significantly better results. For example, we surpass SegGPT [34] and Painter [33] by 14.3 PQ and 25.5 PQ. In addition, With just a few visual in-context prompts for each category, our model achieves comparable results with previous close-set or open-set models on COCO. For example, the panoptic segmentation performance gap between DINOv and our baseline Mask DINO is only 0.6 PQ (57.7 PQ vs 58.3 PQ). Out-domain open-set segmentation on ADE20K. After training with visual prompt on COCO and SAM, we do zero-shot evaluation on ADE20K to validate its open-set segmentation capability when seeing novel visual concepts. To our best knowledge, it is the first time to use visual prompt for open-set segmentation. Compared with previous text-prompted open-set models, we achieve comparable or better performance with only COCO semantic data and no semantic knowledge from large pre-trained models. Especially, compared with our baseline OpenSeed, we achieve better performance with much fewer data. Note that FC-CLIP [42] employs a frozen CLIP to do text-based open-set segmentation. As the text and visual features are aligned in CLIP, we also attempt to prompt a pre-trained FC-CLIP with visual features from CLIP to test its open-set ability with visual prompts. However, its visual prompting performance largely lags behind its text-prompted results. Therefore, it is non-trivial to transfer a multi-modal text-based open-set model to do visual-prompted recognition well. The results indicate that visual prompts can generalize well to new concepts.

Segmentation and detection in the wild. We also validate the generalization capability of visual prompting on some diversified and domain-specific datasets including SegInW and ODinW, which in total encompass more than 60 datasets. These datasets contain many real-scenario or rare categories. As these datasets all focus on instance-level segmentation, we report the average and median AP (APAverage and AP-Median) over all datasets. We first evaluate the Segmentation in the Wild (SegInW) benchmark, which consists of 25 datasets. With visual prompting, DINOv achieves a significant performance improvement over our baseline OpenSeed. For example, Our best AP-Average exceeds OpenSeed by 4.5 AP. We further evaluate Object Detection in the Wild (ODinW), which is composed of 35 datasets with bounding box annotations. As shown in Ta-

- Table 1. One suit of weights for generic visual in-context segmentation on multiple datasets. Our model is trained on COCO and SA-1B data. Note: “−” denotes the model does not have number reported or does not have the ability for the specific task. ⋆ means it is the test set results. † FC-CLIP adopts a frozen CLIP for open-set (text), we prompt the FC-CLIP with CLIP visual features to simulate visual promoting. # FC-CLIP and ODISE rely on frozen CLIP and Stable Diffusion knowledge. Mask DINO [18] is our baseline for comparison.

Method Semantic Data Type

COCO (in-domain) ADE (out-domain) SegInW (out-domain)

PQ mask AP box AP mIoU PQ mask AP box AP mIoU AP-Average AP-Median Mask2Former-T [2] COCO

Closed-set

53.2 43.3 46.1 63.2 − − − − −

Mask2Former-B [2] COCO 56.4 46.3 49.5 67.1 − − − − − − Mask2Former-L [2] COCO 57.8 48.6 52.1 67.4 − − − − − − OneFormer-L [9] COCO 57.9 48.9 − 67.4 − − − − − − MaskDINO-L [16] COCO 58.3 50.6 56.2 67.5 − − − − − − Pano/SegFormer-B [36] COCO 55.4 − − − − − − − − − kMaX-DeepLab-L [43] COCO 48.7 58.1 − − − − − − − −

GLIPv2-H [46] COCO+O365+GOLDG+...

Text Open-set

− 48.9⋆ − − − − − − − − MaskCLIP (L) [5] YFCC100M − − − − − 15.1 6.0 − 23.7 − #ODISE-H [37] COCO (Stable diffusion)) 45.6 38.4 − 52.4 23.4 13.9 − 28.7 − − #FC-CLIP-L [42] COCO (CLIP) 54.4 44.6 − 63.7 26.8 16.8 − 34.1 − − OpenSeed-T [45] COCO+O365 55.4 47.6 52.0 64.0 19.8 14.1 17.0 22.9 33.9 21.5 X-Decoder-T [51] COCO+CC3M+.. 51.4 40.5 43.6 62.8 18.8 9.8 − 25.0 22.7 15.2 X-Decoder-L [51] COCO+CC3M+.. 56.9 46.7 − 67.5 21.8 13.1 − 29.6 36.1 38.7 OpenSeed-L [45] COCO+O365 59.5 53.2 58.2 68.6 19.7 15.0 17.7 23.4 36.1 38.7

FC-CLIP†-L [42] COCO − − − − 2.3 4.1 − 7.8 − − SegGPT-L [34] COCO+ADE+VOC+..

Visual Prompt

43.4 − − − − − − − − − Painter-L [33] COCO+ADE+NYUv2 34.4 − − − − − − − − −

DINOv-T (Ours) COCO 49.0 41.5 45.2 57.0 19.4 11.4 12.8 21.9 39.5 41.6 DINOv-L (Ours) COCO 57.7 50.4 54.2 66.7 23.2 15.1 14.3 25.3 40.6 44.6

- Table 2. One suit of weights on ODinW benchmark. Average and median AP across 35 datasets are reported for simplicity.

|Model<br><br>|Pretrain Data<br><br>|Average|Median|
|---|---|---|---|
|MDETR [12] GLIP-T [19] OpenSeed (T) (Ours) OpenSeed (L) (Ours)<br><br>DINOv (T) (Ours) DINOv (L) (Ours)|GOLDG, REFC Object365 Object365, COCO Object365, COCO COCO, SAM COCO, SAM<br><br>|10.7 11.4 14.2 15.2 14.9 15.7<br><br>|3.0 1.6 3.1 5.0 5.4 4.8|

ble 2, though we only employ much fewer semantically labeled data, we achieve better performance compared with previous models under similar settings.

- 3.3. Video Object Segmentation

DAVIS2016-Interactive, and Youtube-VOS 2018. The results of DAVIS2017 and Youtube-VOS 2018 indicate our model achieves better performance than SEEM and PerSAM. In addition, DINOv can also do interactive VOS, and our performance on DAVIS16-Interactive achieves significantly better performance compared with models not using video data for training.

###### 3.4. Ablation

Effectiveness of Query Formulation. In Table 4, we ablate the effectiveness of using different query formulations for different tasks. The results indicate our double query formulation outperforms using only one type of query.

Effectiveness of Visual Prompt Formulation. In Table 5, we attempt to use a pre-trained CLIP vision encoder to encode the features of the visual prompt by cropping the prompted region into images for CLIP to process. As CLIP features contain rich semantics with few appearance features, which could not apply to referring segmentation tasks. Therefore, we ablate on generic segmentation tasks and find that the final model could not generalize well on open-set datasets like ADE. This result verifies our hypothesis that CLIP vision features could not generalize well on in-context visual prompting.

Video object segmentation (VOS) aims to segment an interested object in a video by giving text or visual clues. Our model focuses on the semi-supervised setting, which segments a particular object throughout a video by giving visual clues in the first frame. In DINOv, the visual prompt originates from one single image (generic/referring segmentation) or other images in one batch (generic segmentation). Therefore, our model has learned to prompt with visual features from other images. Therefore, DINOv is able to do video object segmentation (VOS) by replacing current frame visual prompt features with previous frames. For more accurate tracking, we also store the visual features of the predicted mask in previous frames. These features, denoted as memory visual prompts, will be averaged together with the first frame’s given prompt to construct the visual prompt of the current frame. Details of the memory visual prompt and ablations are in the Appendix. By default, the memory length is set to 8. In Table 3, we conduct (interactive) video object segmentation evaluation on DAVIS17,

Effectiveness of Unifying Tasks and Data. We unify visual generic segmentation and visual referring segmentation to use both semantically labeled data (COCO) and data with only segmentation annotations (SA-1B). In Table 6, the results indicate that employing both datasets improves each individual task.

Training batch size for generic segmentation. In Table 7, the results show that increasing training batch size consistently improves the generic segmentation performance. The

- Table 3. Zero-shot video object segmentation. Without training with video or pairwise image data, our approach is able to do video object segmentation in a zero-shot manner. (#Concurrent work.)

Method Segmentation Data Type Refer-Type

ZeroShot

DAVIS17 DAVIS16-Interactive YouTube-VOS 2018

JF J F JF J F G Js Fs Ju Fu With Video Data

|AGSS [20] VOS+DAVIS AGAME [11] (Synth)VOS+DAVIS SWEM [22] Image+VOS+DAVIS XMem [3] Image+VOS+DAVIS SiamMask [32] COCO+VOS MiVOS [4] BL30K+VOS+DAVIS ReferFormer-B [35] RefCOCO(+/g)+VOS+DAVIS|Video<br><br>|Mask Mask Mask Mask Box Mask Text| |67.4 64.9 69.9 − − − 71.3 71.3 65.5 75.2 73.1 70.0 67.2 72.7 − − − 66.0 66.9 * 61.2 * 84.3 81.2 87.4 − − − 82.8 82.4 86.9 77.1 85.0 − − − − − − 86.1 85.1 89.8 80.3 89.2 * 54.3 58.5 69.8 71.7 67.8 * 60.2 58.2 45.1 47.7<br><br>84.5 81.7 87.4 91.0 89.6 92.4 82.6 81.1 85.6 77.7 86.2 61.1 58.1 64.1 − − − * * * * *|
|---|---|---|---|---|
|UNINEXT-T [41] Image+Video UNINEXT-L [41] Image+Video UNINEXT-L [41] Image+Video|Generalist<br><br>|Mask Mask Text| |74.5 71.3 77.6 − − − 77.0 76.8 81.0 70.8 79.4 77.2 73.2 81.2 − − − 78.1 79.1 83.5 71.0 78.9 66.7 62.3 71.1 − − − * * * * *|

Without Video Data Painter-L [33] COCO+ADE+NYUv2

Generalist

Mask ✓ 34.6 28.5 40.8 − − − 24.1 27.6 35.8 14.3 18.7 SegGPT-L [34] COCO+ADE+VOC+... Mask ✓ 75.6 72.5 78.6 − − − 74.7 75.1 80.2 67.4 75.9 PerSAM-L [47] SAM+DAVIS Mask ✗ 60.3 56.6 63.9 − − − * * * * * SEEM-T [52] ✓ 60.4 57.6 63.3 62.7 58.9 66.4 51.4 55.6 44.1 59.2 46.9 SEEM-L [52]

COCO+LVIS Mask

✓ 58.9 55.0 62.8 62.2 58.3 66.0 50.0 57.2 38.2 61.3 43.3 DINOv-T (Ours)

COCO+SAM

Mask ✓ 73.3 71.0 75.7 77.0 72.9 81.2 60.9 65.3 70.0 52.3 57.9 DINOv-L (Ours) ✓ 72.3 69.8 74.8 75.4 71.3 79.4 59.6 61.7 65.7 52.3 58.8

- Table 4. Ablation of using difference queries to do both incontext reference and generic segmentation. By default, we use both generic query and interactive query. We remove one type of query at a time to ablate their effectiveness.

Method

COCO DAVIS17

PQ mask AP box AP mIoU JF J F

DINOv-SwinT 49.6 42.7 47.0 58.0 73.3 71.0 75.7 only point query 45.2 31.0(11.7) 34.7(-12.3) 52.7 71.4 68.8 74.0 only generic query 46.2 38.3(-4.4) 41.5(-6.0) 53.3 68.9 66.5 71.3

- Table 5. Ablation of using different ways to encode the visual prompt on our Swin-T model. Under the same setting, we change our prompt encoding method and use a pre-trained CLIP to crop and encode the prompted objects in the image.

Prompt Encoding

COCO (in-domain) ADE (out-domain) PQ mask AP box AP mIoU PQ mask AP box AP mIoU

Ours 49.6 42.7 47.0 58.0 19.4 11.4 12.8 21.9 CLIP 48.5 40.7 43.5 54.9 12.6 1.4 1.3 13.3

- Table 6. Ablation of the effectiveness of unifying tasks and data.

Table 7. Ablation of image batchsize sampling in training.

#Batchsize for Prompt Sampling

COCO PQ mask AP box AP mIoU

Method

DINOv-SwinT 1 28.9 23.2 25.3 33.7 DINOv-SwinT 4 45.1 37.0 40.4 50.6 DINOv-SwinT 8 47.3 39.2 43.1 53.1 DINOv-SwinT 32 47.8 40.3 44.1 56.2 DINOv-SwinT 64 49.0 45.2 41.5 57.0

#### 4. Related Works

###### 4.1. Visual Perception Through Text Prompt

Innovations in open-vocabulary object detection [7, 12, 19,24,26,44,45,48] and open-vocabulary segmentation [6, 8, 14, 31, 37, 45], have shown great potential in generic visual perception, by leveraging large pre-trained visionlanguage models like CLIP [30] and ALIGN [10]. These approaches demonstrate significant strides in zero-shot and few-shot performance, adapting to a variety of visual contexts through text prompts. However, the reliance on text alone introduces limitations due to linguistic ambiguity and the potential mismatch between textual descriptions and complex visual scenes [40]. This highlights the ongoing need to refine the integration of visual inputs for more accurate and comprehensive image perception.

COCO DAVIS17

Method Data

PQ mask AP box AP mIoU JF J F

DINOv-SwinT COCO, SAM 49.6 42.7 47.0 58.0 73.3 71.0 75.7 DINOv-SwinT COCO 48.9 41.7 45.9 57.1 63.3 60.8 65.7 DINOv-SwinT SAM N/A − − − 68.4 66.0 70.8

reason for this phenomenon is that a larger batch size helps to sample more positive and negative visual in-context examples across different images, which better matches the inference setting with random visual examples.

###### 4.2. Visual Perception Through Image Example

Building upon the foundations set by text-based visual perception methodologies, the field has seen a notable shift towards incorporating image examples to enhance accuracy and context sensitivity. OV-DETR [44] extends its open-vocabulary object detection capability beyond text,

Inference In-Context Examples. In Fig. 7, we ablate the impact of using different in-context lengths. Increasing the in-context example exhibits diminishing returns, especially when the number of examples is more than eight.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

## DINOv

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Figure 6. DINOv can do open-set segmentation by giving visual prompts.

encoder like CLIP [30], visual prompt-based methods typically use visual instructions (e.g. box, point, mask, scribble, and refereed regions of another image) to guide a model for a specific visual task. SAM [13], for instance, introduces a promptable model for interactive image segmentation, fostering research in computer vision foundation models. It is followed by some works that adapt SAM for visual prompting through personalized examples [47]. SEEM [52] stands out as an interactive and versatile model for segmenting objects, accommodating various types of prompts, and is semantic-aware compared to SAM. Semantic-SAM [17] excels in semantic awareness and recognizing granularity, and is capable of various segmentation tasks including panoptic and part segmentation. Painter [33] and SegGPT [34] take a generalist approach, coping with various segmentation tasks by formulating segmentation as an in-context coloring problem. Our work resembles them with the same goal while presenting a new visual prompting mechanism to support all types of segmentation tasks.

60

50

40

COCO PQ

Metric

ADE20K PQ

ODinW AP-Average SegInW AP-Average

30

20

10

1 2 4 8 16 32

Inference In-context Examples

Figure 7. DINOv query formulation of different tasks.

by utilizing both the image encoder and text encoder from CLIP [30], allowing for object detection guided by visual examples. Similarly, OWL-ViT [26] leverages large-scale image text examples in its contrastive pre-training phase, and propose to adopt image example for one-shot imageconditioned object detection. MQ-Det [40] utilizes image examples to enhance text descriptions for better openvocabulary object detection performance. These methods typically adopt the image encoder in CLIP to extract visual features from given image examples for a more accurate perception of objects and scenes, and demonstrate that visual examples can bridge the gap between textual ambiguity and the complex nature of visual perception.

#### 5. Conclusion

We present DINOv, a unified framework for in-context visual prompting to accommodate both referring segmentation and generic segmentation tasks. To effectively formulate in-context visual prompts, we designed a simple prompt encoder to encoder reference visual prompts from the reference image and adopted a shared decoder to decode the final target visual prompts from the target image. We also formulate generic latent queries and point queries to align different tasks and data. The experimental results indicate that DINOv demonstrates impressive referring and generic segmentation capabilities to refer and detect with in-context visual prompting. Notably, DINOv delivers competitive per-

###### 4.3. Visual Perception Through Visual Prompt

Different from image example-based methods that take an image as input, which are then processed by multi-modal

formance compared to close-set segmentation on in-domain datasets and show promising results on many open-set segmentation benchmarks. We hope our early exploration of visual in-context prompting could inspire the community.

Limitations. We employ limited semantically labeled data (COCO), which can be scaled up for better performance and extended to text prompts for multi-modal understanding.

#### References

- [1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. 2
- [2] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1290–1299, 2022. 7
- [3] Ho Kei Cheng and Alexander G. Schwing. XMem: Longterm video object segmentation with an atkinson-shiffrin memory model. In ECCV, 2022. 8
- [4] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Modular interactive video object segmentation: Interaction-to-mask, propagation and difference-aware fusion. In CVPR, 2021. 8
- [5] Zheng Ding, Jieke Wang, and Zhuowen Tu. Openvocabulary panoptic segmentation with maskclip. arXiv preprint arXiv:2208.08984, 2022. 7
- [6] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In European Conference on Computer Vision, pages 540–557. Springer, 2022. 8
- [7] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921,

2021. 8

- [8] Dat Huynh, Jason Kuen, Zhe Lin, Jiuxiang Gu, and Ehsan Elhamifar. Open-vocabulary instance segmentation via robust cross-modal pseudo-labeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7020–7031, 2022. 8
- [9] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. arXiv preprint arXiv:2211.06220, 2022. 7
- [10] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,

2021. 8

- [11] Joakim Johnander, Martin Danelljan, Emil Brissman, Fahad Shahbaz Khan, and Michael Felsberg. A generative appearance model for end-to-end video object segmentation,

2018. 8

- [12] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetrmodulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1780–1790, 2021. 7, 8
- [13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything, 2023. 2, 6, 9
- [14] Shiyi Lan, Zhiding Yu, Christopher Choy, Subhashree Radhakrishnan, Guilin Liu, Yuke Zhu, Larry S Davis, and Anima Anandkumar. Discobox: Weakly supervised instance segmentation and semantic correspondence from box supervision. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3406–3416, 2021. 8
- [15] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao. Multimodal foundation models: From specialists to general-purpose assistants. arXiv preprint arXiv:2309.10020, 1, 2023. 2
- [16] Feng Li, Hao Zhang, Shilong Liu, Lei Zhang, Lionel M Ni, Heung-Yeung Shum, et al. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. arXiv preprint arXiv:2206.02777, 2022. 6, 7, 12
- [17] Feng Li, Hao Zhang, Peize Sun, Xueyan Zou, Shilong Liu, Jianwei Yang, Chunyuan Li, Lei Zhang, and Jianfeng Gao. Semantic-sam: Segment and recognize anything at any granularity. arXiv preprint arXiv:2307.04767, 2023. 2, 6, 9, 12
- [18] Feng Li, Hao Zhang, Huaizhe Xu, Shilong Liu, Lei Zhang, Lionel M Ni, and Heung-Yeung Shum. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3041–3050, 2023. 2, 7
- [19] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975, 2022. 2, 6, 7, 8
- [20] Huaijia Lin, Xiaojuan Qi, and Jiaya Jia. Agss-vos: Attention guided single-shot video object segmentation. In ICCV,

2019. 8

- [21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 2, 6
- [22] Zhihui Lin, Tianyu Yang, Maomao Li, Ziyu Wang, Chun Yuan, Wenhao Jiang, and Wei Liu. Swem: Towards realtime video object segmentation with sequential weighted expectation-maximization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1362–1372, 2022. 8

- [23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2
- [24] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 2, 8
- [25] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021. 12
- [26] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In European Conference on Computer Vision, pages 728–755. Springer, 2022. 5, 8, 9
- [27] OpenAI. Gpt-4 technical report, 2023. 2
- [28] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 6
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 5
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 8, 9
- [31] Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu. Denseclip: Language-guided dense prediction with contextaware prompting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18082–18091, 2022. 8
- [32] Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, and Philip HS Torr. Fast online object tracking and segmentation: A unifying approach. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2019. 8
- [33] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning, 2023. 6, 7, 8, 9
- [34] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Segmenting everything in context. arXiv preprint arXiv:2304.03284, 2023. 2, 6, 7, 8, 9
- [35] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as queries for referring video object segmentation. arXiv preprint arXiv:2201.00487, 2022. 8
- [36] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and ef-

- ficient design for semantic segmentation with transformers. arXiv preprint arXiv:2105.15203, 2021. 7
- [37] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2955–2966, 2023. 7, 8
- [38] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models, 2023. 2
- [39] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327, 2018. 6
- [40] Yifan Xu, Mengdan Zhang, Chaoyou Fu, Peixian Chen, Xiaoshan Yang, Ke Li, and Changsheng Xu. Multimodal queried object detection in the wild. arXiv preprint arXiv:2305.18980, 2023. 8, 9
- [41] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. arXiv preprint arXiv:2303.06674, 2023. 8
- [42] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional clip. arXiv preprint arXiv:2308.02487, 2023. 2, 6, 7
- [43] Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. k-means mask transformer. In European Conference on Computer Vision, pages 288–307. Springer, 2022. 7
- [44] Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, and Chen Change Loy. Open-vocabulary detr with conditional matching. In European Conference on Computer Vision, pages 106–122. Springer, 2022. 8
- [45] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianwei Yang, and Lei Zhang. A simple framework for open-vocabulary segmentation and detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1020–1031, 2023. 2, 7, 8
- [46] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. arXiv preprint arXiv:2206.05836, 2022. 7
- [47] Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Hao Dong, Peng Gao, and Hongsheng Li. Personalize segment anything model with one shot, 2023. 8, 9
- [48] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Regionbased language-image pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16793–16803, 2022. 5, 8
- [49] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through

- ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017. 6
- [50] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023. 2
- [51] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al. Generalized decoding for pixel, image, and language. arXiv preprint arXiv:2212.11270, 2022. 2, 6, 7
- [52] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. arXiv preprint arXiv:2304.06718, 2023. 2, 8, 9

#### A. Implementation Details

Our model framework is mainly based on Mask DINO [17], which is a unified framework for detection and segmentation. DINOv is a general encoder-decoder architecture composed of a vision encoder. We use SwinT/L [25] as the vision encoder. As our decoder supports both generic query and point query, we adopt 300 latent generic queries following Mask DINO [16] and six level queries for each input point following Semantic-SAM [17]. Especially, when using point query, we sample 10 foreground and 40 background points during training and employ grid sample for 20×20 points during inference. For inference on general segmentation and detection tasks, we use 16 in-context examples for each category by default. For VOS inference, we average eight previous frame predictions as the reference to segment the current frame.

Table 8. Ablation of Inference Memory Length on DAVIS2017 with a SwinL backbone.

|Method Memory Length|DAVIS2017<br><br>J& F J F<br><br>|
|---|---|
|DINOv-SwinT 1<br><br>DINOv-SwinT 2 DINOv-SwinT 4 DINOv-SwinT 8 DINOv-SwinT 16<br><br><br>|62.1 58.7 65.4 69.6 66.7 72.6<br><br>71.5 68.7 74.3<br>72.3 69.8 74.8 68.0 65.4 70.7<br>|

frame. We employ a priority queue to manage the memory. For simplicity, the priority score of each prompt is positively correlated to the frame number, which indicates that we only store the memory prompts that are near the current frame in time sequence. By default, the memory length is set to 8. In Tab. 8, we show the influence of using different number of memory length.

#### B. Video Object Segmentation Inference

Video object segmentation (VOS) aims to segment an interested object in a video by giving text or visual clues. Our model focuses on the semi-supervised setting, which segments a particular object throughout a video by giving visual clues in the first frame. When doing VOS, an intuitive way is to first extract reference visual prompt features from the first frame image and the corresponding visual prompts with our prompt encoder. When processing each frame in a video, we are able to utilize reference visual prompt features in the first frame as in the current frame.

In DINOv, as we train with visual in-context prompting with multiple examples for generic segmentation, we can also apply this strategy to VOS for better performance. More concretely, we also compute and store the reference visual features of the predicted mask in previous frames. These features, denoted as memory reference visual prompts, will be averaged together with the first frame’s given prompt to construct the visual prompt of the current

