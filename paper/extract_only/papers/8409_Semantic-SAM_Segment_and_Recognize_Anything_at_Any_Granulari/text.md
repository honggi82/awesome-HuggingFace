# arXiv:2307.04767v1[cs.CV]10Jul2023

## Semantic-SAM: Segment and Recognize Anything at Any Granularity

#### Feng Li♠∗, Hao Zhang♠∗, Peize Sun♯, Xueyan Zou§, Shilong Liu¶, Chunyuan Li‡ Jianwei Yang‡1, Lei Zhang†2, Jianfeng Gao‡2

♠ HKUST ‡ Microsoft Research, Redmond † IDEA ♯ HKU § UW-Madison ¶ Tsinghua

∗ Equal Contribution 1. Project Lead 2. Equal Advisory Contribution

### Abstract

In this paper, we introduce Semantic-SAM, a universal image segmentation model to enable segment and recognize anything at any desired granularity. Our model offers two key advantages: semantic-awareness and granularity-abundance. To achieve semantic-awareness, we consolidate multiple datasets across granularities and train on decoupled objects and parts classification. This allows our model to facilitate knowledge transfer among rich semantic information. For the multigranularity capability, we propose a multi-choice learning scheme, enabling each click point to generate masks at multiple levels that correspond to multiple groundtruth masks. Notably, this work represents the first attempt to jointly train a model on SA-1B, generic, and part segmentation datasets. Experimental results and visualizations demonstrate that our model successfully achieves semantic-awareness and granularity-abundance. Furthermore, combining SA-1B training with other segmentation tasks, such as panoptic and part segmentation, leads to performance improvements. We will provide code and a demo for further exploration and evaluation at https://github.com/UX-Decoder/Semantic-SAM.

### 1 Introduction

The universal and interactive AI systems that follow human intents have shown their potential in natural language processing [46, 47] and controllable image generation [52, 66]. However, such a universal system for pixel-level image understanding remains less explored. We argue that a universal segmentation model should possess the following important properties: universal representation, semantic-awareness, and granularity-abundance. Regardless of the specific image domain or prompt context, the model is capable of acquiring a versatile representation, predicting segmentation masks in multi-granularity, and understanding the semantic meaning behind each segmented region.

Previous works [31, 70, 58] attempted to investigate these properties, but only achieved part of the goals. The main obstacles impeding the progress of such a universal image segmentation model can be attributed to limitations in both model architecture flexibility and training data availability.

• Model Architecture. The existing image segmentation model architectures are dominated by the single-input-single-output pipeline that discards any ambiguity. While this pipeline is prevalent in both anchor-based CNN architectures [24] and query-based Transformer architectures [4, 11], and has demonstrated remarkable performance in semantic, instance, and panoptic segmentation tasks [39, 68, 30], it inherently restricts the model to predict multi-granularity segmentation masks in an end-to-end manner. Although clustering postprocessing techniques [13] can produce multiple masks for a single object query, they are neither efficient nor effective solutions for a granularity-aware segmentation model.

Preprint. Under review.

[Figure 1]

Figure 1: Our model is capable of dealing with various segmentation tasks including open-set and interactive segmentation. (a) Our model can do instance, semantic, panoptic segmentation, and part segmentation. (b) Our model is able to output multi-level semantics with different granularities. The red point on the left-most image is the click.(c) We connect our model with an inpainting model to perform multi-level inpainting. The prompts are "Spider-Man" and "BMW car", respectively. Note that only one click is needed to produce the results in (b) and (c), respectively.

• Training Data. Scaling up segmentation datasets that possess both semantic-awareness and granularity-awareness is a costly endeavor. Existing generic object and segmentation datasets such as MSCOCO [39] and Objects365 [53] offer large amounts of data and rich semantic information, but only at the object level. On the other hand, part segmentation datasets such as Pascal Part [9], PartImageNet [23], and PACO [49] provide more fine-grained semantic annotations, but their data volumes are limited. Recently, SAM [31] has successfully scale up the multi-granularity mask data to millions of images, but it does not include semantic annotations. In order to achieve the dual objectives of semantic-awareness and granularity-abundance, there is a pressing need to unify segmentation training on various data formats to facilitate knowledge transfer. However, the inherent differences in semantics and granularity across different datasets pose a significant challenge to joint training efforts.

In this paper, we introduce Semantic-SAM, a universal image segmentation model designed to enable segmenting and recognizing objects at any desired granularity. Given one click point from a user, our model addresses the spatial ambiguity by predicting masks in multiple granularities, accompanied by semantic labels at both the object and part levels. As shown in Figure 1, our model generates multi-level segmentation masks ranging from the person head to the whole truck.

The multi-granularity capability is achieved through a multi-choice learning design [37, 22] incorporated into the decoder architecture. Each click is represented with multiple queries, each containing a different level of embedding. These queries are trained to learn from all available ground-truth masks representing different granularities. To establish a correspondence between multiple masks and ground-truths, we employ a many-to-many matching scheme to ensure that a single click point could generate high-quality masks in multiple granularities.

To accomplish semantic-awareness with a generalized capability, we introduce a decoupled classification approach for objects and parts, leveraging a shared text encoder to encode both objects and parts independently. This allows us to perform object and part segmentation separately, while adapting the loss function based on the data type. For instance, generic segmentation data lacks part classification loss, whereas SAM data does not include classification loss.

To enrich semantics and granularity within our model, we consolidate seven datasets on three types of granularities, including generic segmentation of MSCOCO [39], Objects365 [53], ADE20k [68], part segmentation of PASCAL Part [9], PACO [49], PartImagenet [23], and SA-1B [31]. Their

data formats are reorganized to match our training objectives accordingly. After joint training, our model obtains a strong performance across a variety of datasets. Notably, we find that learning from interactive segmentation could improve generic and part segmentation. For example, by jointly training SA-1B promptable segmentation and COCO panoptic segmentation, we achieve a gain of 2.3 box AP and a gain of 1.2 mask AP. In addition, through comprehensive experiments, we demonstrate that our granularity completeness is better than SAM with more than 3.4 1-IoU.

### 2 Data Unification: Semantics and Granularity

In order for multi-level semantics, we include seven datasets that contain different granularitylevel masks. The datasets are SA-1B, COCO panoptic, ADE20k panoptic, PASCAL part, PACO, PartImageNet, and Objects365. Within them, COCO and ADE20k panoptic datasets contain objectlevel masks and class labels. PASCAL part, PACO, and PartImageNet contain part-level masks and class labels. SA-1B contains up to 6-level masks without labels, while Objects365 contains abundant class labels for object-level instances. The details of these datasets are shown in Table 1. We further visualize the data distribution of different data type in Fig 2.

|Type Data #Images|Semantic Concept<br><br>Part Object|Granularity Level<br><br>Part Whole<br><br>|
|---|---|---|
|Class-agnostic SA-1B 11B<br><br>|✗ ✗|✓ ✓<br><br>|
|Object-level<br><br>Objects365 1.7M<br><br>COCO 110K ADE20K 20K<br><br>|✗ 365<br><br>✗ 133<br><br>✗ 150<br><br><br>|✗ ✓<br>✗ ✓<br>✗ ✓<br>|
|Part-level<br><br>PACO-LVIS 45K<br><br>PartImageNet 16K Pascal Part 5K<br><br>|201 75 13 11 30 20<br><br>|✓ ✓ ✓ ✓ ✓ ✓|

[Figure 2]

Table 1: The data statistics in Semantic-SAM. Figure 2: Semantics-Granularity 2D chart.

### 3 Semantic-SAM

#### 3.1 Model

Our Semantic-SAM follows [33] to exploit a query-based mask decoder to produce semantic-aware and multi-granularity masks. In addition to the generic queries, it supports two types of prompts including point and box, similar to SAM [31]. The overall pipeline is shown in Fig. 3.

We represent both click and box prompts into anchor boxes as a unified format. In particular, we convert user click point (x,y) into an anchor box (x,y,w,h) with small width w and height h, so that the anchor box can closely approximate the point. To capture different granularities of masks, each click is first encoded to position prompt and combined with K different content prompts, where each content prompt is represented as a trainable embedding vector for a given granularity level. Here we empirically choose K = 6, considering there are at most 6 levels of masks per user click for the majority of images in SA-1B [31]. More specifically, a click/box b = (x,y,w,h) is encoded into K content embeddings and one position embedding, respectively. We represent its content embeddings as a set of query vectors Q = (q1,··· ,qK). For the i-th query,

qi = qleveli + qtypei , (1) where

- • qlevel is the embedding for granularity level i,
- • qtype distinguishes the query type, chosen from either the click or the box embeddings.

The position embedding of c is implemented via sine encoding. Assuming that the output image feature from vision encoder is F, the mask decoder of the proposed Semantic-SAM represents the click on the input image as:

O = DeformDec(Q,b,F) with O = (o1,··· ,oK), (2) where DeformDec(·,·,·) is a deformable decoder that takes query feature, reference box, and image features as input to output queried features. oi is the model output for the ith input query qi.

Truck Window/Truck

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Training GT Types:

…

|Hungarian Matching|
|---|

Generic segmentation (i.e, COCO) Part segmentation(i.e, Pascal Part) Class-agnostic multi-granularity (SAM)

Multiple-choice Learning

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

…

|Text Encoder| |
|---|---|
| | |

|…|
|---|

Vision tokens

|Vision Encoder| |
|---|---|
| | |

Semantic SAM

[Figure 12]

Interactive whole-part

Generic

referring

|… …<br><br>|
|---|

|Duplicate|
|---|

Content Prompts

Position Prompts

Point

Box Generic

Figure 3: Semantic-SAM is a universal segmentation framework that can take multiple types of segmentation data including generic, part, and class-agnostic segmentation data. The Vision Encoder is used to extract image features. The mask decoder can do both generic segmentation and promptable segmentation with various types of prompts. For point and box, we input them via anchor boxes to the mask decoder. Since there is an ambiguity of granularity for a point input, we duplicate each point 6 times and give them different levels of embeddings. The output masks of point prompts match with multiple GT masks of different granularities.

Each oi = (ci,mi) consists of the predicted semantic category ci and mask mi, which are used to construct the concept recognition loss and mask prediction loss, respectively.

#### 3.2 Training

[Figure 13]

Recognize Anything. As we train with various types of data with different semantic annotations, in which some contain object-level annotations (COCO), some contain both object and part-level annotations (Pascal Part), and SA-1B has no semantic annotations but contains masks of all semantic levels. Note that a large number of part concepts are shared across different objects, for example,

Figure 4: Decoupled object and part classification.

head for all animals. We aim to transfer the part concept knowledge across objects trained with only object-level annotations in our joint training. To address this discrepancy between semantic annotations and better transfer semantics of different granularity, we propose to decouple object and part recognition. As shown in Fig 4, we utilize a shared text encoder to encode objects and parts, which are used to perform object and part segmentation separately.

Importantly, while all types of segmentation data share a unified format, the loss varies for different data types. We summarize the loss items to construct the training objective in Semantic-SAM in Table 2. It is the part-level data that bridges the gap to recognize semantic concepts between part and object levels, and it is the use of SAM data in Hungarian matching that bridges the gap to segment masks at any granularity.

|Data<br><br>|Recognize Anything Part Object|Segment at Any Granularity Box Mask #GT in Matching<br><br>|
|---|---|---|
|SAM data Object-level data Part-level data<br><br>|✗ ✗<br>✗ ✓<br><br><br>✓ ✓|✓ ✓ Many<br><br>✓ ✓ One ✓ ✓ One<br><br>|

- Table 2: The loss items to construct the training objective in Semantic-SAM. The four loss items are part classification, object classification, box loss and mask loss, respectively. The last column indicates the number of ground-truth mask in the matching.

Segment at any granularity. To endow the model with a multi-granularity segmentation ability, we propose a many-to-many matching method during training. We found that SAM fails in providing

[Figure 14]

- Figure 5: Inteactive learning strategy comparison between a) One-to-one: traditional interactive segmentation models that focus on object-level, i.e, SEEM, b) Many-to-one: multi-choice learning for single granularity, i.e, SAM, and c) Many-to-many: ours. We enforce the model to predict all the possible granularities of a single click for more controllable segmentation. d) As a result, our output granularity are richer to generate diverse output masks.

good multi-level segmentation results with a single click because SAM uses many-to-one matching during training. In other words, the three SAM-predicted masks for each click only match with one GT mask. This causes that points located in masks of small levels cannot predict large masks with high quality according to our observation. In contrast, to enable multi-level mask prediction with a single click, we fully leverage the structures in both data and algorithm. First, we re-organize the data by clustering multiple GT masks of different levels sharing the same click. To allow multiple predictions of the same click to match with the GT masks, we employ the Hungarian algorithm to enable the many-to-many matching. The similarity matrix and scores vary based on the availability of different segmentation data components.

For box input and generic segmentation, we follow existing methods. Specifically, to generate a mask from an input box, we follow a similar idea as in denoising training (DN) [33]. We add noises to ground-truth boxes to simulate inaccurate box inputs from users, and these noised boxes serve as spatial prompts for the decoder. The model is trained to reconstruct the original boxes and masks given noised boxes. For the content part of box prompts, we adopt a learnable token as a general prompt. Note that this is the only difference from DN, as DN uses ground-truth label embedding as the content prompts. For generic segmentation, we follow the same pipeline as in Mask DINO [33].

Discussion. As shown in Fig. 5, compared with previous interactive segmentation models, SemanticSAM differs from previous segmentation models in two aspects. Firstly, we train the model to output all the possible segmentation masks with one click. Secondly, our output granularities are richer to generate diverse output masks.

### 4 Experiments

#### 4.1 Experimental Setup

Implementation Details. In our experiments, we jointly train on three types of data, as shown in Table 1. We implement our model based on Mask DINO [33] . Mask DINO is a unified detection and segmentation framework which simultaneously predicts box and mask. We follow [33] to use 300 latent queries and nine decoder layers for all segmentation tasks. For the visual backbone, we adopt pre-trained Swin-T/L [41] by default. For the language backbone, we adopt the pre-trained base model in UniCL [62]. As SA-1B [31] dominates the data, during training, we first train on only SA-1B data. Then, we add object and part-level data to jointly train the three types of data. During training, the image resolution is 1024 × 1024 for all data. We use AdamW [43] as the optimizer. We use large-scale jittering for object and part-level data and did not use data augmentations for SA-1B data, as SA-1B images are abundant. We set the learning rate to 0.0001, which is decayed at 0.9 and 0.95 fractions of the total number of steps by 10.

Evaluation. We mainly evaluate two datasets, including COCO Val2017 and a subset of SA-1B [31] with 1000 images. For evaluation metrics, we evaluate PQ and AP for generic and part segmentation datasets. For single-granularity interactive segmentation, we report Point (Max) and Point (Oracle). Max denotes we select the output mask with the maximum confidence score. Oracle denotes we select the output mask with the max IoU by calculating the IoU between the prediction and target mask. For multi-granularity interactive segmentation, we report 1-IoU@All Granularity that matches

|Method|Type<br><br>|Training Data|PQ<br><br>|mIoU<br><br>|AP box mask|APs box mask<br><br>|APm box mask|APl box mask<br><br>|
|---|---|---|---|---|---|---|---|---|
|Mask2Former (T) [11] X-Decoder (T) [69]<br><br>|Close-set Open-set<br><br>|COCO COCO+VL|53.2 52.6<br><br>|63.2 62.4|46.1 43.3 43.6 41.3<br><br>|− − − −|− − − −<br><br>|− − − −|
|OpenSeed (T) [65] Semantic-SAM (T) (ours) Semantic-SAM (T) (ours)|Open-set Open-set Open-set<br><br>|COCO+O365 COCO COCO+SAM<br><br>|55.4 54.6 55.2<br><br>|63.8 63.2 63.4<br><br>|51.2 47.1 50.1 46.1<br>52.3(+2.2) 47.4(+1.3)<br>|34.5 27.4 34.4 27.1 36.1(+1.7) 28.3(+1.2)<br><br>|54.3 50.4 53.2 49.4<br><br>55.6(+2.4) 50.7(+1.3)<br><br><br>|66.2 66.8<br><br>66.1 66.1<br>67.3 66.2<br>|

##### Table 3: Results for Semantic-SAM and other panoptic segmentation models on COCO val2017. Our model is jointly trained on COCO [8] and [31] (1/10 data) and directly evaluates COCO.

|Method<br><br>|Type|Training Data<br><br>|AP box mask|APs box mask<br><br>|APm box mask|APl box mask<br><br>|
|---|---|---|---|---|---|---|
|VLPart [55] Semantic-SAM (ours) Semantic-SAM (ours)<br><br>|Open-set Open-set Open-set<br><br>|Pascal Part Pascal Part Pascal Part+SAM<br><br>|− 27.4<br><br>27.0 30.5<br>28.0 31.4<br>|− −<br><br>16.6 19.1<br><br>17.3 19.9<br><br><br>|− − 38.1 41.6 40.0 42.5|− − 43.8 49.1 45.7 49.7<br><br>|

##### Table 4: Results for Semantic-SAM and other part segmentation models on Pascal Part. Our model is jointly trained on Pascal Part [15] and SA-1B [31] (1/10 data) and directly evaluates Pascal Part.

all the possible ground-truth masks for a single click to the multi-granularity predictions and then calculate the average IoU of all granularities.

- 4.2 Semantic Segmentation of Anything

Generic Segmentation As shown in Table 3, to validate the compatibility of multi-granularity interactive segmentation and generic segmentation, we jointly train with SA-1B [31] (1/10 data) and COCO panoptic segmentation. The result indicates that interactive segmentation with SAM can significantly help the instance-level detection and segmentation with a performance improvement of +2.2 AP on the box and +1.3 AP on the mask. Notably, OpenSeed [65] and Semantic-SAM are both based on Mask DINO [33]. Our joint training with SA-1B even outperforms OpenSeed which is trained with Object365 [53]. In addition, adding SA-1B mainly improves small object detection (APs and APm), as there are a large number of small objects in SA-1B.

Part Segmentation We also validate the compatibility of joint training SA-1B (1/10 data) and part segmentation. As shown in Table 4, adding SA-1B brings a decent performance improvement on Pascal Part [15].

Single-granularity Interactive Segmentation In Table 5, we evaluate the 1-click mIoU (denoted as 1-IoU) for SAM and our model on COCO Val2017. Our model outperforms SAM under the same settings.

Multi-granularity Interactive Segmentation In Table 6, we compare SAM [31] and our model on the output granularities for a single click. We adopt a Hungarian Matching to match all the possible

|Method<br><br>|COCO Point (Max) Point (Oracle)<br><br>1-IoU 1-IoU|
|---|---|
|SAM (B) SAM (L) Semantic-SAM (T) Semantic-SAM (L)<br><br>|52.1 68.2 55.7 70.5 54.5 73.8 57.0 74.2|

##### Table 5: Comparison with previous models on point interactions. Both SAM [31] and our model are trained with only SA-1B and directly evaluate on COCO Val2017 for fair comparison. Max denotes we select the output with the max confidence score prediction. Oracle denotes we select the output with the max IoU by calculating the IoU between the prediction and target mask.

|Method Granularity<br><br>|1-IoU@All Granularity|
|---|---|
|SAM (B)† 3 SAM (L)† 3 SAM (H)† 3 SAM (B)†∗ 6 SAM (L)†∗ 6 SAM (H)†∗ 6 Semantic-SAM(T) 6 Semantic-SAM(L) 6<br><br>|75.6 82.5 83.5 79.3 85.6 86.5 88.1 89.0|

- Table 6: Granularity comparison between SAM and our model on a subset of SA-1B with 1000 images. We did not train on this subset of images but SAM did. For each click, we evaluate all the possible ground-truth masks to calculate the 1-IoU@All Granularity. SAM [31] and Semantic-SAM adopts three and six prompts for a single click of a mask, respectively. † denotes that SAM has been trained on this validation subset while we did not. ∗ denotes that we click two points for a single mask to produce six output masks.

|Method Match|1-IoU@All Granularity|
|---|---|
|Semantic-SAM(T) Many-to-one Semantic-SAM(T) Many-to-many|73.2 88.1<br><br>|

Table 7: Different match strategy comparison on output granularity.

target masks with the predicted masks for the click and calculate the average IoU score. As SAM has only three prompts, we also sample two clicks from a single mask to produce six output masks for a fair comparison. Notably, SAM has been trained on this validation set while we did not.

#### 4.3 Abaltions

Match Strategy As shown in Table 7, we compare different match strategies in our model. When using many-to-many matching to match all the possible ground-truth masks for each click, the 1-IoU@All Granularity performance is significantly improved. This validates our matching strategy is effective to learn complete granularities.

Box Interactive Evaluation We also evaluate the 1-IoU given boxes in Table 8. We achieve better performance compared with object-level interactive segmentation model SEEM [70] and multi-granularity model SAM [31].

Increasing SA-1B Training data In Table 9, we show the performance improvement on COCO Val 2017 when training with more SA-1B data. The performance is saturated after using more than 15% of the total data. It indicates that we do not need to train with the whole SA-1B data to get a good zero-shot performance.

#### 4.4 Visualization

We compare our model with SAM to show that our model can output more levels of high-quality masks, as shown in Fig. 6.

|Method<br><br>|Box 1-IoU|
|---|---|
|SAM [31](B) SEEM [70](T) Semantic-SAM(T)<br><br>|50.7 73.7 76.1|

- Table 8: Box 1-IoU evaluation on COCO Val2017. Both SEEM [70] and our model are trained on COCO and we additionally train on SA-1B [31].

|Method|Data Portion of SA-1B<br><br>|COCO Point (Max) Point (Oracle)<br><br>1-IoU 1-IoU<br><br>|
|---|---|---|
|SAM (L) Semantic-SAM (L) Semantic-SAM (L) Semantic-SAM (L) Semantic-SAM (L) Semantic-SAM (L)|100% 3% 15% 30% 50% 100%<br><br>|55.7 70.5<br><br>55.2 73.5<br>56.7 73.6 55.7 73.7 55.3 73.9<br>57.0 74.2<br>|

- Table 9: Comparison of using different portions of SA-1B [31] data. Our model is only trained with SA-1B and directly evaluated on COCO Val2017.

Multi-Level Masks Our model outputs more meaningful granularities of masks. SAM outputs three masks at most and different levels of outputs are sometimes duplications, While, the output masks of our model are more diverse.

Mask Qualities It is also proved that our model output masks with higher quality. SAM sometimes outputs masks with artifacts such as holes or islands especially for large masks when the click is within a small-scale mask, while our model output high-quality masks for all levels.

Compare with SA-1B Ground-truth Granularity We output more meaningful granularity on SAM data compared with the original annotation.

Query semantics We also find that each point content prompt embeddings learns to correspond to a fixed granularity. As shown in Fig. 7, when we visualize masks in a specific order of the corresponding content embeddings, the masks follow the order from small to large in each row consistently. This proves that each content embedding represents a semantic granularity level in our model.

### 5 Related works

#### 5.1 Generic Segmentation

Segmenting visual concepts is well-documented within the expansive field of computer vision [17, 16, 71, 45]. Broad segmentation methodologies comprise several subdivisions, such as instance segmentation, semantic segmentation, and panoptic segmentation [24, 6, 30], each catering to a unique semantic degree. For example, semantic segmentation’s goal is to detect and assign a label to each pixel in an image according to its corresponding semantic class [7, 11, 42]. Conversely, instance segmentation seeks to cluster pixels associated with the same semantic class into distinct object instances [24, 3, 33]. Panoptic segmentation is the hybrid of these two tasks. Recently, Transformerbased methods [56, 4] have contributed to significant progress in segmentation tasks [38, 11, 33, 26, 64]. Generic object detection and segmentation have led to the development of abundant datasets, such as MSCOCO [39], LVIS [21], Objects365 [53], PASCAL [15],CityScapes [12],ADE20k [68], etc.

#### 5.2 Part Segmentation

Beyond generic segmentation, part segmentation aims to more fine-grained visual understanding. Most early works were bottom-up methods by grouping super-pixels into parts and then objects [1, 20, 2]. Later, based on high-performance object detection networks [51, 24], top-down methods were developed by firstly detecting an object and then parsing it to part segmentation [34, 63, 27]. To segment the scene in multi-granularity, part-aware panoptic segmentation [13] is introduced. PPS [13] establishes the baseline through assembling panoptic and part segmentation models. JPPF [25] simplifies the model by a shared image encoder for both panoptic segmentation and part segmentation. By representing thing, stuffs, and parts as object queries, Panoptic-PartFormer [35] proposes a unified architecture based on Transformer. While part segmentation data is much expensive than object detection and segmentation data, a number of public datasets are available. Datasets for specific

|[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]|
|---|

- (a) Ours

- (b) SAM

|[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]|
|---|

|[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>≈|
|---|

- (a) Ours
- (b) SAM

|[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>≈|
|---|

|[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]|
|---|

- (a) Ours
- (b) SAM

|[Figure 34]<br><br>[Figure 35]|
|---|

- (c) GT

|[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]|
|---|

- (c) GT

|[Figure 39]<br><br>[Figure 40]|
|---|

- (c) GT

|[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]|
|---|

- Figure 6: (a)(b) are the output masks of our model and SAM, respectively. The red points on the left-most image of each row are the use clicks. (c) shows the GT masks that contain the user clicks. The outputs of our model have been processed to remove duplicates.

domains include cars [54], birds [57], and fashion [29]. General objects include Pascal-Part [9], PartImageNet [23], ADE20K [67], Cityscapes-Panoptic-Parts [44], and PACO [49]. More recently, SAM [31] provides a large-scale multi-granularity class-agnostic segmentation dataset. Our work is jointly trained on these datasets and contributes to a multi-granularity segmentation model.

#### 5.3 Open-Vocabulary Segmentation

While generic segmentation and part segmentation have made remarkable progress, they can only segment the image in a close-set vocabulary. To expand the vocabulary size, recent works leverage the visual-semantic knowledge from large-scale foundation models like CLIP [48], ALIGN [28] and Diffusion models [60] to various segmentation tasks. LSeg [32], OpenSeg [18], GroupViT [59]

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

Prompt 1 Prompt 2 Prompt 3 Prompt 4 Prompt 5 Prompt 6

- Figure 7: We visualize the prediction of each content prompt embedding of points with a fixed order for our model. We find all the output masks are from small to large. This indicates each prompt embedding represents a semantic level.

achieves open-vocabulary semantic segmentation ability on ADE20K and PASCAL. DenseCLIP [50] and MaskCLIP [14] achieves open-vocabulary instance and panoptic segmentation on COCO dataset. More recently, X-Decoder [69] proposes a unified approach to tackle various segmentation and visionlanguage tasks for open-vocabulary segmentation, OpenSeeD [65] proposes to use a large amount of detection data and a joint training method to improve segmentation. To segment open-vocabulary masks in part-level, VLPart [55] leverages three part segmentation datasets and learns from the dense correspondence [5] between base objects and novel objects. Our work unifies these tasks into one architecture and builds up open-vocabulary segmentation in multi-granularity.

#### 5.4 Interactive Segmentation

Interactive segmentation refers to the process of separating objects by actively integrating user inputs. This enduring challenge has seen notable advancements [36, 19, 61, 40, 10, 31]. Previous works only focus on a small set of data or semantic-agnostic instance masks. Recently, SAM [31] enlarges the training data from 0.12M COCO images to 10M SAM fine-grained images. And SEEM [70] enriches the modality to language and function to both generic and grounded segmentation with an impressive compositionality.

### 6 Conclusion

In this paper, we have presented Semantic-SAM, which can segment and recognize anything at any desired granularity. Apart from performing generic open-vocabulary segmentation, Semantic-SAM demonstrates the advantages of semantic awareness and granularity abundance. To achieve such advantages, we have proposed improvements on data, model, and training where we utilized datasets from multiple granularity and semantic levels, multi-choice learning for training, and a universal framework for modeling. Comprehensive experiments and visualizations have verified the semantic awareness and granularity abundance of our model. Further, Semantic-SAM is the first successful attempt to jointly train on SA-1B and other classic segmentation datasets. Experimental results also show that training with SA-1B improves other tasks such as panoptic and part segmentation.

### References

- [1] Pablo Arbelaez, Michael Maire, Charless Fowlkes, and Jitendra Malik. Contour detection and hierarchical image segmentation. IEEE transactions on pattern analysis and machine intelligence, 33(5):898–916, 2010.
- [2] Pablo Arbeláez, Jordi Pont-Tuset, Jonathan T Barron, Ferran Marques, and Jitendra Malik. Multiscale combinatorial grouping. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 328–335, 2014.
- [3] Daniel Bolya, Chong Zhou, Fanyi Xiao, and Yong Jae Lee. Yolact: Real-time instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9157–9166, 2019.

- [4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European Conference on Computer Vision, pages 213–229. Springer, 2020.
- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [6] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence, 40(4):834–848, 2017.
- [7] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.
- [8] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.
- [9] Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect what you can: Detecting and representing objects using holistic models and body parts, 2014.
- [10] Xi Chen, Zhiyan Zhao, Yilei Zhang, Manni Duan, Donglian Qi, and Hengshuang Zhao. Focalclick: towards practical interactive image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1300–1309, 2022.
- [11] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1290–1299, 2022.
- [12] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
- [13] Daan de Geus, Panagiotis Meletis, Chenyang Lu, Xiaoxiao Wen, and Gijs Dubbelman. Partaware panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5485–5494, 2021.
- [14] Zheng Ding, Jieke Wang, and Zhuowen Tu. Open-vocabulary panoptic segmentation with maskclip. arXiv preprint arXiv:2208.08984, 2022.
- [15] Mark Everingham and John Winn. The pascal visual object classes challenge 2012 (voc2012) development kit. Pattern Analysis, Statistical Modelling and Computational Learning, Tech. Rep, 8(5), 2011.
- [16] Pedro F Felzenszwalb, Ross B Girshick, David McAllester, and Deva Ramanan. Object detection with discriminatively trained part-based models. IEEE transactions on pattern analysis and machine intelligence, 32(9):1627–1645, 2009.
- [17] King-Sun Fu and JK Mui. A survey on image segmentation. Pattern recognition, 13(1):3–16, 1981.
- [18] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Open-vocabulary image segmentation. arXiv preprint arXiv:2112.12143, 2021.
- [19] Leo Grady. Random walks for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 28(11):1768–1783, 2006.
- [20] Matthias Grundmann, Vivek Kwatra, Mei Han, and Irfan Essa. Efficient hierarchical graphbased video segmentation. In 2010 ieee computer society conference on computer vision and pattern recognition, pages 2141–2148. IEEE, 2010.
- [21] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5356–5364, 2019.
- [22] Abner Guzman-Rivera, Dhruv Batra, and Pushmeet Kohli. Multiple choice learning: Learning to produce multiple structured outputs. Advances in neural information processing systems, 25, 2012.
- [23] Ju He, Shuo Yang, Shaokang Yang, Adam Kortylewski, Xiaoding Yuan, Jie-Neng Chen, Shuai Liu, Cheng Yang, and Alan Yuille. Partimagenet: A large, high-quality dataset of parts. arXiv preprint arXiv:2112.00933, 2021.

- [24] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.
- [25] Sravan Kumar Jagadeesh, René Schuster, and Didier Stricker. Multi-task fusion for efficient panoptic-part segmentation. arXiv preprint arXiv:2212.07671, 2022.
- [26] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. arXiv preprint arXiv:2211.06220, 2022.
- [27] Ruyi Ji, Dawei Du, Libo Zhang, Longyin Wen, Yanjun Wu, Chen Zhao, Feiyue Huang, and Siwei Lyu. Learning semantic neural tree for human parsing. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16, pages 205–221. Springer, 2020.
- [28] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021.
- [29] Menglin Jia, Mengyun Shi, Mikhail Sirotenko, Yin Cui, Claire Cardie, Bharath Hariharan, Hartwig Adam, and Serge Belongie. Fashionpedia: Ontology, segmentation, and an attribute localization dataset. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 316–332. Springer, 2020.
- [30] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. Panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9404–9413, 2019.
- [31] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything, 2023.
- [32] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. Languagedriven semantic segmentation. arXiv preprint arXiv:2201.03546, 2022.
- [33] Feng Li, Hao Zhang, Shilong Liu, Lei Zhang, Lionel M Ni, Heung-Yeung Shum, et al. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. arXiv preprint arXiv:2206.02777, 2022.
- [34] Qizhu Li, Anurag Arnab, and Philip HS Torr. Holistic, instance-level human parsing. arXiv preprint arXiv:1709.03612, 2017.
- [35] Xiangtai Li, Shilin Xu, Yibo Yang, Guangliang Cheng, Yunhai Tong, and Dacheng Tao. Panopticpartformer: Learning a unified model for panoptic part segmentation. In European Conference on Computer Vision, pages 729–747. Springer, 2022.
- [36] Yin Li, Jian Sun, Chi-Keung Tang, and Heung-Yeung Shum. Lazy snapping. ACM Transactions on Graphics (ToG), 23(3):303–308, 2004.
- [37] Zhuwen Li, Qifeng Chen, and Vladlen Koltun. Interactive image segmentation with latent diversity. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 577–585, 2018.
- [38] Zhiqi Li, Wenhai Wang, Enze Xie, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, Ping Luo, and Tong Lu. Panoptic segformer: Delving deeper into panoptic segmentation with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1280–1289, 2022.
- [39] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.
- [40] Qin Liu, Zhenlin Xu, Gedas Bertasius, and Marc Niethammer. Simpleclick: Interactive image segmentation with simple vision transformers. arXiv preprint arXiv:2210.11006, 2022.
- [41] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021.
- [42] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3431–3440, 2015.
- [43] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [44] Panagiotis Meletis, Xiaoxiao Wen, Chenyang Lu, Daan de Geus, and Gijs Dubbelman. Cityscapes-panoptic-parts and pascal-panoptic-parts datasets for scene understanding. arXiv preprint arXiv:2004.07944, 2020.

- [45] Shervin Minaee, Yuri Y Boykov, Fatih Porikli, Antonio J Plaza, Nasser Kehtarnavaz, and Demetri Terzopoulos. Image segmentation using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 2021.
- [46] OpenAI. Chatgpt. https://openai.com/blog/chatgpt, 2022.
- [47] OpenAI. Gpt-4 technical report, 2023.
- [48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.
- [49] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, Amir Mousavi, Yiwen Song, Abhimanyu Dubey, and Dhruv Mahajan. PACO: Parts and attributes of common objects. In arXiv preprint arXiv:2301.01795, 2023.
- [50] Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu. Denseclip: Language-guided dense prediction with context-aware prompting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18082–18091, 2022.
- [51] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28:91–99.
- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [53] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019.
- [54] Xibin Song, Peng Wang, Dingfu Zhou, Rui Zhu, Chenye Guan, Yuchao Dai, Hao Su, Hongdong Li, and Ruigang Yang. Apollocar3d: A large 3d car instance understanding benchmark for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5452–5462, 2019.
- [55] Peize Sun, Shoufa Chen, Chenchen Zhu, Fanyi Xiao, Ping Luo, Saining Xie, and Zhicheng Yan. Going denser with open-vocabulary part segmentation, 2023.
- [56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [57] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. technical report, 2011.
- [58] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Segmenting everything in context. arXiv preprint arXiv:2304.03284, 2023.
- [59] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18134–18144, 2022.
- [60] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. arXiv preprint arXiv:2303.04803, 2023.
- [61] Ning Xu, Brian Price, Scott Cohen, Jimei Yang, and Thomas S Huang. Deep interactive object selection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 373–381, 2016.
- [62] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified contrastive learning in image-text-label space. In CVPR, 2022.
- [63] Lu Yang, Qing Song, Zhihui Wang, and Ming Jiang. Parsing r-cnn for instance-level human analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 364–373, 2019.
- [64] Hao Zhang, Feng Li, Huaizhe Xu, Shijia Huang, Shilong Liu, Lionel M Ni, and Lei Zhang. Mpformer: Mask-piloted transformer for image segmentation. arXiv preprint arXiv:2303.07336, 2023.

- [65] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianfeng Gao, Jianwei Yang, and Lei Zhang. A simple framework for open-vocabulary segmentation and detection. arXiv preprint arXiv:2303.08131, 2023.
- [66] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023.
- [67] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017.
- [68] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset, 2018.
- [69] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al. Generalized decoding for pixel, image, and language. arXiv preprint arXiv:2212.11270, 2022.
- [70] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. arXiv preprint arXiv:2304.06718, 2023.
- [71] Zhengxia Zou, Zhenwei Shi, Yuhong Guo, and Jieping Ye. Object detection in 20 years: A survey. arXiv preprint arXiv:1905.05055, 2019.

