## FrozenSeg: Harmonizing Frozen Foundation Models for Open-Vocabulary Segmentation

Xi Chen 1 Haosen Yang 2 Sheng Jin 3 Xiatian Zhu 2 Hongxun Yao 1 1Harbin Institute of Technology 2 University of Surrey 3 Nanyang Technological University

# arXiv:2409.03525v1[cs.CV]5Sep2024

### Abstract

Open-vocabulary segmentation poses significant challenges, as it requires segmenting and recognizing objects across an open set of categories in unconstrained environments. Building on the success of powerful vision-language (ViL) foundation models, such as CLIP, recent efforts sought to harness their zero-short capabilities to recognize unseen categories. Despite notable performance improvements, these models still encounter the critical issue of generating precise mask proposals for unseen categories and scenarios, resulting in inferior segmentation performance eventually. To address this challenge, we introduce a novel approach, FrozenSeg, designed to integrate spatial knowledge from a localization foundation model (e.g. SAM) and semantic knowledge extracted from a ViL model (e.g. CLIP), in a synergistic framework. Taking the ViL model’s visual encoder as the feature backbone, we inject the space-aware feature into the learnable queries and CLIP features within the transformer decoder. In addition, we devise a mask proposal ensemble strategy for further improving the recall rate and mask quality. To fully exploit pre-trained knowledge while minimizing training overhead, we freeze both foundation models, focusing optimization efforts solely on a lightweight transformer decoder for mask proposal generation – the performance bottleneck. Extensive experiments demonstrate that FrozenSeg advances state-of-the-art results across various segmentation benchmarks, trained exclusively on COCO panoptic data, and tested in a zero-shot manner. Code is available at https://github.com/ chenxi52/FrozenSeg.

### 1. Introduction

Image segmentation is a fundamental task in computer vision, enabling a wide range of applications such as object recognition [6, 32], scene understanding [34, 41], and image manipulation [49]. However, traditional techniques are often tailored to specific datasets and segmentation tasks, resulting in a significant gap compared to human visual intelligence, which can perceive diverse visual concepts in the

[Figure 1]

Figure 1. Comparision of mask recall of unseen classes and final results performance between FC-CLIP [43] and our approach. Evaluating the performance on the Cityscapes and PC459 datasets with IoU thresholds of 0.5 and 0.75, our FrozenSeg approach significantly increases the mask average recall (AR) of unseen classes and delivers improved final results in Panoptic Quality (PQ) and Mean Intersection-over-Union (mIoU).

open world. To bridge this disparity, the concept of openvocabulary segmentation has emerged. In this task, the segmenter is trained to recognize and segment instances and scene elements from any category, mirroring the broad capabilities of human perception.

Parallel to these efforts, significant advancements have been made in the field of purpose-generic image-level large-dataset pretrained Vision Language (ViL) representation learning, exemplified by foundational models such as CLIP [29] and ALIGN [14]. These models are pivotal in understanding open scenes, as they leverage rich, descriptive language cues to enhance models’ ability to generalize across a wide array of unseen categories. However, the absence of sufficient pixel-level annotations often leads to challenges in dense-level image-text alignment. Recent studies have utilized these pre-trained ViL models for region classification [18, 23, 38], necessitating further training of a segmentation model [6] for precise pixel-level alignment, often resulting in inefficiencies and reduced effectiveness. Alternatively, mask proposals generated with the CLIP visual encoder [43] are still suboptimal due to their limited fine-grained pixel-level understanding, which becomes a performance bottleneck as the mask proposal generation may overfit to the training classes, undermining

the model’s generalizability to unseen classes. As shown in Fig. 1, existing methods such as FC-CLIP [43] struggle to generalize to unseen categories under different IoU thresholds, significantly limiting their practical utility.

In this paper, to overcome the above limitation, we introduce FrozenSeg, a system that harnesses the capabilities of localization foundation model SAM to synergistically and efficiently enhance the coarse semantic features extracted from CLIP by incorporating generalized fine space-aware features. FrozenSeg has three key modules: (1) Query Injector, which aggregate local space-aware features from SAM to serve as the spatial query for the corresponding mask region, enhancing the learnability of queries in a transformer decoder. (2) Feature Injector, designed to enrich each pixel’s CLIP feature by incorporating comprehensive global spatial information from SAM. (3) OpenSeg Ensemble Module, designed to further boost the quality of mask predictions based on the spatial information injection of SAM during training by ensembling with zero-shot mask proposals from SAM. Building upon these modules, as shown in Fig. 1, the recall metrics of unseen categories on the challenging CityScapes dataset [7] showed significant improvement, consequently boosting PQ from 44.3 to 45.8. This upward trend is further supported by the results in PC-459 [26], with mIoU increase from 17.3 to 19.7, validating the observed enhancement.

Our contributions can be summarized as follows: (1) Addressing an acknowledged limitation in mask proposal quality, we introduce FrozenSeg, a framework that incorporates foundational models to tackle the open-vocabulary segmentation task effectively. (2) We propose three critical components: the Query Injector, the Feature Injector, and the OpenSeg Ensemble Module. These components are designed to enhance the integration of SAM features into the transformer decoder, facilitating generalized mask generation. (3) Extensive experiments on various segmentation tasks demonstrate the superiority of our FrozenSeg in generating mask proposals and achieving enhanced final performance, surpassing previous approaches.

### 2. Related Works

#### 2.1. Open-vocabulary Segmentation

Open-vocabulary segmentation aims to segment objects even without seeing those classes during training. Previous approaches [20, 23, 38] typically employ a twostage process, where an additional segmentation model generates class-agnostic mask proposals, which are then interacted with CLIP features. In the context of openvocabulary panoptic segmentation, which necessitates instance segmentation and interaction with multiple mask proposals [9, 18], methods such as OPSNet [4] combine

query embeddings with the last-layer CLIP embeddings and applies an IoU branch to filter out less informative proposals. MaskCLIP [9] integrates learnable mask tokens with CLIP embeddings and class-agnostic masks. Despite these advancements, challenges remain in effectively aligning segmenters with CLIP.

Alternatively, one-stage open-vocabulary segmentation faces challenges in extending vision-language models without dedicated segmentation models and addressing overfitting in an end-to-end format. CLIP’s pre-training on imagetext pairs necessitates reconciling the region-level biases of the vision-language model. Research such as FC-CLIP and F-VLM [16] indicates that convolutional CLIP models generally exhibit superior generalization capabilities compared to ViT-based [10] counterparts, primarily due to their capability to handle larger input resolutions effectively. This finding highlights a promising direction for adapting CLIP for improved performance in segmentation tasks. Despite these advancements, a fundamental issue persists: accurately generating mask proposals for unseen categories and scenarios. This challenge is compounded by the methods’ dependence on a static Vision and ViL model, which is not equipped to discern intricate pixel-level details, thereby limiting its effectiveness in mask proposal generation.

#### 2.2. Large-scale Foundation Models

Recent advances in large-scale foundation models, pretrained on extensive datasets, have showcased exceptional zero-shot capabilities. Multi-modal foundation models, such as CLIP and ALIGN, exhibit strong generalization across various downstream tasks. Although these models are trained on image-level data with inherent noise, they can be effectively fine-tuned for various applications. Common strategies include prompt learning [35, 48] and the use of adapters [46], with CLIP often remaining frozen to preserve its broad generalization.

In the realm of the segmentation foundation models, significant progress is exemplified by the SAM model [1], which leverages the extensive SA-1B dataset to achieve notable zero-shot generalization. SAM can adapt to new datasets without additional training by using input prompts. Subsequent models, such as HQ-SAM [15] and GenSAM [13] have built upon this foundation by optimizing output tokens and integrating textual semantic reasoning, respectively. Despite these advancements, these methods often rely on manually crafted prompts, which constrains their wider applicability and scalability.

Recent research [30, 40, 44] has explored the use of bounding boxes generated through open-vocabulary detection methods as prompts, combining SAM and CLIP to exploit their complementary strengths in open-vocabulary segmentation. These approaches aim to combine SAM’s zero-shot generalization capabilities with CLIP’s robust

feature representations. Despite these efforts, significant challenges remain in achieving fully automatic openvocabulary segmentation and transitioning from instance segmentation to semantic and panoptic segmentation.

### 3. Method

Our objective is to achieve efficient open-vocabulary segmentation using frozen foundation models. In this section, we start by defining the problem. Subsequently, we present our method, FrozenSeg, which integrates frozen foundation models for open-vocabulary segmentation through two key components: the Query Injector and the Feature Injector, as illustrated in Fig. 2. Finally, we detail our inference strategy, the OpenSeg Ensemble Module.

#### 3.1. Problem Definition

Open-vocabulary segmentation involves training with ground-truth masks corresponding to a predefined set of class labels, Ctrain. During testing, the model encounters a different set of class labels, Ctest, which includes novel classes not seen during training. This process requires segmenting images in an open-world context, where the model must categorize pixels into semantic classes for semantic segmentation, identify individual instances for instance segmentation, or combine both for panoptic segmentation. The notation C represents either Ctrain or Ctest, depending on whether the phase is training or testing.

#### 3.2. Our Approach FrozenSeg

Overall Architecture Following the approach of [36, 43], we adopt Mask2Former [6] as our framework. A set of N learnable queries that represent all things and stuff in the input image is processed through the transformer decoder to get mask predictions m. To adapt the framework for open vocabulary segmentation, we replace the original classification layer with the text embeddings derived from the CLIP text encoder, resulting in class prediction pd, where d denotes the mask detector. Post-training, the embeddings for each mask and its corresponding category text are projected into a shared embedding space, facilitating effective categorization within the open-vocabulary framework. In line with [16, 43], we adopt the convolution-based CLIP visual encoder as our image feature extractor, leveraging its pretrained, frozen weights to obtain high-resolution semantic information.

To address the limitations of CLIP’s coarse features, we introduce two key modules: the Query Injector and the Feature Injector. These modules integrate the spatial features from SAM into the mask proposal generation process, as depicted in Fig. 2. Unlike [5], which incorporates multi-level spatial information into the vision transformer, our injectors focus on infusing spatial information

directly into mask queries. Additionally, we propose the OpenSeg Ensemble Module to further enhance segmentation performance during inference. We detail our approach below:

Query Injector To improve local spatial understanding, we introduce the Query Injector, which enhances the learnable query with space-aware features derived from SAM. The transformer decoder uses masked multi-head attention to bolster cross-attention between the image’s foreground region and the learnable queries. This mechanism facilitates the integration of both content and spatial information within the query, a concept supported by prior studies such as [2, 22, 25]. However, capturing detailed spatial information remains challenging when the backbone is frozen.

To address this challenge, we devise the Query Injector, which leverages newly generated masks at each decoder layer to pool and transform SAM visual features into a spatial query. The process for generating the spatial query is defined as follows:

xl = f(pool(Ml,Fsam)) (1)

Here, l represents the layer index in the transformer decoder, f denotes a linear projection function, and pool refers to the mask pooling operation. Fsam represents the SAM-derived image features. This spatial query is specifically designed to concentrate on a region encompassing the mask region. Subsequently, the spatial query is integrated with the learnable query via element-wise addition.

Feature Injector To refine the CLIP features for mask generation on a global scale, we introduce the Feature Injector, which uses the multi-head cross-attention mechanism (MHCA) as detailed in [31]. This mechanism is renowned for its effectiveness in amalgamating diverse information. In our approach, we extend MHCA to enhance the coarse semantic features from CLIP. Specifically, the Feature Injector integrates semantic content from CLIP with pixellevel spatial awareness from SAM, providing a more nuanced understanding at the pixel scale. The mathematical formulation of this feature integration is as follows:

F = SoftMax

fq(Fclip) · fk(Fsam) √

D · fv(Fsam) (2)

Here, fq, fk, and fv are linear projection functions in MHCA. Fclip and Fsam represent the features extracted from CLIP and SAM, respectively, while D denotes the dimensionality of the projected features.

Inference Strategy Previous works such as [16, 28, 43] have validated the efficacy of mask pooling on CLIP features within class ensemble methodologies to enhance open-vocabulary segmentation capabilities. Building on these techniques, our approach introduces a novel mask en-

[Figure 2]

- Figure 2. Overview of our FrozenSeg approach: (Top) We introduce three key components: the Query Injector, Feature Injector and OpenSeg Ensemble Module to enhance open-vocabulary dense-level understanding. Given N queries, spatial information from SAM is injected into these queries within intermediate layers of the transformer encoder, leading to N class and N corresponding mask predictions. The OpenSeg Ensemble Module then integrates these predictions with zero-shot SAM masks to generate the final results. (Bottom) Detailed design of the two injectors.

[Figure 3]

- Figure 3. Overview of OpenSeg Ensemble Module. SAM masks are generated through uniform sampling of point prompts. The module employs a novel mask ensemble strategy, injecting

Msam = {mˆ i}Ni=1, are generated by uniformly sampling points prompts across the image. These masks are used to

pool CLIP features and derive classification scores Psam = {pˆi}Ni=1 by aligning with CLIP text features. A threshold ξ = 0.5 is applied to filter these masks based on the maximum probability, resulting in the selected probability-mask pairs {( ˆmi,pˆi) | argmaxc pˆi > ξ}N

′

i=1.

In the context of semantic segmentation, the SAM mask predictions, denoted as rˆ, are computed similarly as follows: rˆ = N

′

i=1 pˆi(c)·mˆ i[x,y]. The final mask prediction, r

′

, is obtained by integrating the predictions r and rˆthrough a mask ensemble approach:

- SAM mask predictions into unseen mask predictions to enhance the generalization of mask proposals.

semble strategy. As illustrated in Fig. 3, our OpenSeg Ensemble Module initiates with the class ensemble process:

(pi,d(j))(1−α) · (pi,cl(j))α, if j ∈ Ctrain (pi,d(j))(1−β) · (pi,cl(j))β, else

pi(j) =

(3)

Here, pi(j) denotes the combined probability for class j in proposal i, integrating inputs from both the detector (pi,d) and CLIP (pi,cl). The mask predictions r for N queries are then generated by aggregating the products of these probability-mask pairs: Ni=1 pi(c) · mi[x,y] = r ∈ RC×HW.

Drawing inspiration from the class ensemble, we utilize zero-shot mask predictions from SAM to perform a mask ensemble on r. The SAM masks, denoted as

r[x,y](j),if j ∈ Ctrain (1 − ϵ) ∗ r[x,y](j) + ϵ ∗ rˆ[x,y](j),else

r′[x,y](j) =

(4)

Subsequently, the final semantic segmentation results are determined by assigning each pixel [x,y] a class based on argmaxc∈{1,...,|C|} r′[x,y].

In the context of panoptic segmentation, the efficacy of the results significantly depends on the performance of individual queries. This dependency reduces the effectiveness of integrating class-agnostic mask predictions. Therefore, the final results are determined by assigning each pixel to one of the N predicted probability pairs. This assignment is performed through the following expression: argmaxi:c

i̸=∅ pi(ci) · mi[x,y]. In this expression, ci represents the most likely class label, which is determined by ci = argmaxc∈{1,...,|C|,∅} pi(c). Here, ∅ denotes the class

of ’no object’.

### 4. Experiments

#### 4.1. Datasets and Evaluation Protocal

For training, we use the COCO panoptic [21] dataset, which includes 133 classes. Our evaluation covers openvocabulary panoptic, semantic, and instance segmentation tasks in a zero-shot setting spanning several test datasets. In semantic segmentation, we access performance on ADE20K dataset [47], which includes both a subset with 150 classes (A-150) and a full version with 847 classes (A-847). Additionally, we evaluate on PASCAL VOC [11](PAS-21), which has 20 object classes and one background class, and PASCAL-Context [26], an extension of PASCAL VOC with 459 classes (PC459). For panoptic segmentation, the datasets used are ADE20K [47], Cityscapes [7], Mapillary Vistas [27], and BDD100K [42], alongside the closed-set COCO validation dataset. For instance segmentation, we choose to evaluate LVIS v1.0 [12], which features 337 rare categories. The evaluation metrics include mean intersection-over-union (mIoU) and frequency-weighted IoU (FWIoU) that offer a comprehensive evaluation of overall performance for semantic segmentation, panoptic quality (PQ), average precision (AP), and mIoU for panoptic segmentation, as well as AP for instance segmentation.

#### 4.2. Baselines

We compare with multiple state-of-art approaches as follows: OPSNet [4], MaskCLIP [9], MasQCLIP [39], ODISE [36], CLIPSelf [33], FC-CLIP [43], Ovseg [20],

- SAN [37] , RegionSpot [40] and Open-Vocabulary SAM [44].

#### 4.3. Implement Details

We use 250 queries for both training and testing, with CLIP serving as the backbone for open-vocabulary textimage alignment. Specifically, we employ the RN50x64 and ConvNext-Large [24] versions of CLIP. Additionally, we validate our approach using the ViT-Base [10] model from SAM, with the selection rationale detailed in the Supplementary. To obtain multi-level semantic features, we apply feature pyramid networks (FPN) after CLIP. SAM processes input images I ∈ RH,W, where H = W = 1024. As demonstrated by PlainViT [19], the deepest feature of ViT contains sufficient information for multi-scale object recognition, and given that SAM is frozen, we do not use FPN for SAM. Instead, we utilize a single convolution layer to project the features to the necessary resolution and then feed them into a single-scale deformable attention transformer [51] as the pixel decoder in the SAM branch. Our transformer decoder comprises L = 9 layers. Feature

[Figure 4]

Figure 4. Qualitative illustration of panoptic segmentation results on Cityscapes. White boxes highlight areas with notable differences between methods. Compared to FC-CLIP, FrozenSeg shows improved performance in predicting small objects (row 1), more accurate entity segmentation (row 2), and better generalization to the unseen class ’rider’ (row 3).

maps with resolutions of 1/8, 1/16, and 1/32 are processed by successive decoder layers in a round-robin fashion. During training, we follow the strategy and losses outlined in FC-CLIP, selecting the model from the final iteration for our primary results. Training is conducted on 4 Tesla A100 GPUs with a batch size of 16.

#### 4.4. Inference Details

During inference, we adhere to the FC-CLIP by resizing images such that the shortest side is 800 pixels for general datasets and 1024 for the Cityscapes and Mapillary Vistas datasets. We employ a 32x32 grid of points research to generate masks from the SAM ViT-Huge model. The default parameters are set as follows: α = 0.4 and β = 0.8 in Eq.(3), and a mask ensemble parameter ϵ = 0.2 in Eq. (4).

#### 4.5. Evaluation on Open-vocabulary Segmentation 4.5.1 Open-vocabulary Panoptic Segmentation

Tab. 1 presents a comparison of FrozenSeg with leading methods in zero-shot open-vocabulary panoptic segmentation. Our approach, FrozenSeg with RN50x64, notably surpasses other works and the baseline FC-CLIP, achieving improvements of +1.8 PQ, +0.3 AP, and +2.0 mIoU on ADE20K; +2.6 PQ, +1.6 AP, and +0.9 mIoU on Cityscapes; and +4.8 mIoU on BDD100K. Additionally, the FrozenSeg configuration with ConvNeXt-L delivers enhanced performance on open-set datasets without compromising results on the closed-set COCO validation dataset. Significant improvements include +0.8 PQ and +1.6 mIoU on ADE20K,

- Table 1. Performance of open-vocabulary panoptic segmentation. We present results obtained using both CLIP RN50x64 and ConvNext-L. Bold represents best, underline indicates second best. * denotes re-implemented final results.

ADE20K Cityscapes Mapillary Vistas BDD100K COCO (seen) Method ViL Model PQ AP mIoU PQ AP mIoU PQ mIoU PQ mIoU PQ AP mIoU OPSNet RN50 19.0 - 25.4 41.5 - - - - - - 57.9 - 64.8

MaskCLIP ViT-L/14 15.1 6.0 23.7 - - - - - - - 30.9 - MaskQCLIP ViT-L/14 23.3 - 30.4 - - - - - - - - - ODISE ViT-L/14 23.4 13.9 28.7 - - - - - - - 45.6 38.4 52.4 ODISE+CLIPSelf ViT-L/14 23.7 13.6 30.1 - - - - - - - 45.7 38.5 52.3

FC-CLIP* RN50x64 21.3 13.2 28.7 42.6 27.3 55.1 18.2 27.4 13.8 41.4 55.3 46.5 64.8 FrozenSeg RN50x64 23.1 13.5 30.7 45.2 28.9 56.0 18.1 27.7 12.9 46.2 55.7 47.4 65.4

FC-CLIP* ConvNeXt-L 25.1 16.4 32.8 44.3 27.9 56.0 18.1 27.9 17.9 49.4 56.4 47.4 65.3 FrozenSeg ConvNeXt-L 25.9 16.4 34.4 45.8 28.4 56.8 18.5 27.3 19.3 52.3 56.2 47.3 65.5

- Table 2. Performance of cross-dataset open-vocabulary semantic segmentation. ’IN’ refers to the ImageNet(50K) [8] dataset, and ’Panop.+Cap.’ signifies the combined use of COCO panoptic [21] and COCO caption [3] datasets. Bold represents best, underline indicates second best. * denotes the results from re-implemented final evaluations.

PC-459 PAS-21 A-847 A-150 Method ViL Model Training Dataset mIoU FWIoU mIoU FWIoU mIoU FWIoU mIoU FWIoU OPSNet RN50 COCO Panop.+IN - - - - - - 25.4 Ovseg ViT-L/14 COCO Stuff 12.4 - - - 9.0 - 29.6 SAN ViT-L/14 COCO Stuff 17.1 - - - 13.7 - 33.3 MaskCLIP ViT-L/14 COCO Panoptic 10.0 - - - 8.2 - 23.7 MasQCLIP ViT-L/14 COCO Panoptic 18.2 - - - 10.7 - 30.4 ODISE ViT-L/14 COCO Panop.+Cap. 13.8 - 82.7 - 11.0 - 29.9 ODISE+CLIPSelf ViT-L/14 +COCO Stuff - - - - - - 30.1 -

FC-CLIP* RN50x64 COCO Panoptic 15.6 50.0 81.1 91.4 10.8 44.6 28.7 52.5 FrozenSeg RN50x64 COCO Panoptic 18.7 60.1 82.3 92.1 11.8 52.8 30.7 56.6

FC-CLIP* ConvNeXt-L COCO Panoptic 17.3 56.7 83.0 92.4 14.0 48.1 32.8 56.1 FrozenSeg ConvNeXt-L COCO Panoptic 19.7 60.2 82.5 92.1 14.8 51.4 34.4 59.9

[Figure 5]

- Figure 5. Qualitative comparison of semantic segmentation results. White boxes indicate areas of discrepancy. Our FrozenSeg (col. 4) has contextually appropriate results compared to FC-CLIP (col. 2) and ground truth annotations (col. 5).

- Table 3. Performance (APr) of open-vocabulary instance segmentation on rare categories in LVIS v1.0 dataset. * denotes the results from re-implemented final results.

Method Proposals LVISr

RegionSpot

GLIP-T(B) [17] 12.7 SAM 14.3 GLIP-T [17] 20.0

Open-Vocabulary SAM Detic [50] 24.0 FC-CLIP* - 25.0

FrozenSeg - 25.6

+1.5 PQ, +0.5 AP, and +0.8 mIoU on Cityscapes, +0.4 PQ on Mapillary Vistas, and +1.4 PQ and +2.9 mIoU on BDD10K. Qualitative results of panoptic segmentation on Cityscapes, depicted in Fig. 4, show improvements in segmentation, particularly for small objects, entity recognition, and novel class recognition. Additional details and additional results are available in the Supplementary.

- 4.5.2 Open-vocabulary Semantic Segmentation

- Tab. 2 presents a comparative analysis of FrozenSeg in cross-dataset open-vocabulary semantic segmentation. Using the RN50x64 backbone, FrozenSeg significantly outperforms the baselines. Compared to FC-CLIP, FrozenSeg achieves gains of +3.1 mIoU and +10.1 FWIoU on PC459, +1.0 mIoU and +8.2 FWIoU on A-847, and +2.0 mIoU and +4.1 FWIoU on A-150. These improvements are also reflected in the ConvNeXt-L configuration. Overall, FrozenSeg sets a new benchmark in performance across the datasets PC-459, A-847, and A-150. It is important to note that PAS-21’s categories fully overlap with those of the training dataset, which suggests that FC-CLIP may overfit to base classes and thus limit its generalization. For qualitative insights, refer to Fig. 5, where FrozenSeg delivers segmentations that are contextually more accurate compared to both the baseline and ground truth annotations, demonstrating its effective handling of complex scenes. Additional results are available in the Supplementary.

4.5.3 Open-vocabulary Instance Segmentation

- Tab. 3 presents results for rare categories in the LVIS dataset. We compare FrozenSeg with approaches that integrate SAM with CLIP for open-vocabulary segmentation tasks, specifically RegionSpot and Open-Vocabulary SAM. Both of these methods rely on proposals as location prompts and are trained on datasets beyond the COCO panoptic to align the models. Our method achieves the highest performance, with an improvement of +0.6 AP over FC-CLIP.

#### 4.6. Ablation Studies

We perform a series of ablation studies on our method. All findings are presented using the ConvNext-L version of CLIP and the ViT-B version of SAM.

##### 4.6.1 The Effectiveness of Each Component

We perform ablation studies to assess the effectiveness of each component of our method. Tab. 4 presents the results of these ablations on three challenging out-of-vocabulary datasets, with rows 2-6 highlighting the contribution of each component to overall performance. Specifically, row 1 illustrates the scenario where only proposals from SAM are utilized. In this setup, SAM masks are used to pool CLIP features, providing basic semantic understanding without explicit semantic guidance. This configuration achieves approximately 6.5 mIoU on PC-459 and A-847, and 25.4 mIoU on ADE20K, demonstrating the fundamental generalization capability of SAM masks. Therefore, we integrate OpenSeg Ensemble to address the limitation of unseen mask proposals. This enhancement is evident in the comparison between ablation cases (2) and (3), and also between cases (5) and (6). Fig. 5 provides a clearer visual comparison, showing improved segmentation accuracy for objects such as ‘people’ and ‘fences’ in the PC-459(2) example, particularly in columns 3 and 4.

##### 4.6.2 Where to Inject

Tab. 5 presents an ablation examining the impact of layer insertion for the Query Injector within the transformer decoder, which consists of a total of 9 layers. Since SAM Vision Transformers provide the final layer features as the most relevant feature maps, we explore the optimal layer for query injection based on its interaction with corresponding CLIP feature maps. The results indicate that injecting SAM query features at layers l = 3,6,9 yields the most significant improvement, with layer l +1 leveraging the newly introduced queries for further refinement. For the Feature Injector, due to the exponential increase in computational complexity associated with cross-attention computations as feature size expands, we restrict the application of the Feature Injector to 1/32-sized features, specifically at layers l = 1,4,7.

##### 4.6.3 Speed and Model Size

As shown in Tab. 6, incorporating SAM along with two custom injectors results in a slight reduction in inference speed, manifesting as a 0.56 and 0.09 decrement in frames per second (FPS) during single-image processing. Despite this, the adjustment leads to a notable improvement of 1.8 PQ on the Cityscapes datasets, with minimal impact on COCO. This

Table 4. Ablations of the proposed modules: results following complete training iterations with mIoU metrics. ’Inj.’: Injector. PC-459 A-847 ADE20K # Query Inj. Feature Inj. OpenSeg Ensemble mIoU mIoU PQ AP mIoU

- (1) only SAM ✗ ✗ ✗ 6.6 6.5 - - 25.4

- (2) Baseline ✗ ✗ ✗ 17.3 14.0 25.1 16.4 32.8

- (3) - ✗ ✗ ✓ 17.6 14.5 - - 33.5

- (4) - ✓ ✗ ✗ 18.1 14.2 25.7 16.5 33.6

- (5) - ✓ ✓ ✗ 18.5 14.4 25.9 16.5 33.8

- (6) FrozenSeg ✓ ✓ ✓ 19.7 14.8 - - 34.4

Table 5. Impact of selected insertion layer in transformer decoder on Query Injector performance: results after 55K iterations. The ’Size’ column is the relative interacted image feature size of multi-level feature maps.

COCO Cityscapes Size Layers PQ AP mIoU PQ AP mIoU

1/32 1, 4, 7 52.6 42.7 62.6 40.4 20.8 53.0 1/16 2, 5, 8 52.5 42.5 62.4 40.1 20.7 53.3 1/8 3, 6, 9 52.7 42.8 62.7 40.0 21.6 54.0

Table 6. Comparative analysis of FPS performance and Trainable vs. Frozen parameter counts using a single A100. All results are obtained from the average time on the validation set, including post-processing.

COCO Cityscapes Parmas[M] Method PQ FPS PQ FPS Frozen Trainable

FC-CLIP 56.4 2.71 44.3 0.87 200.0 21.0 FrozenSeg 56.2 2.15 45.8 0.78 293.5 26.5

reflects a well-balanced trade-off between enhanced performance and computational efficiency. Compared to FCCLIP, our model requires a modest increase of only 5.5M training parameters and 93.5M frozen parameters, demonstrating its effectiveness.

### 5. Conclusion

In this study, we introduced FrozenSeg, a method designed to enhance mask proposal quality in open-vocabulary segmentation by leveraging SAM’s dense-prediction capabilities. FrozenSeg employs the Query Injector and Feature Injector modules to integrate SAM visual features with learned queries and CLIP visual features, thereby refining mask proposals through multiple transformer decoder layers. Additionally, we introduce the OpenSeg Ensemble Module for inference, which aggregates zero-shot SAM masks to improve out-vocabulary predictions further. Our experiments demonstrated that FrozenSeg significantly enhances the mask proposal quality in open-vocabulary scenarios, highlighting its versatility.

### 6. Appendix

Our supplementary material begins with the in-depth analysis of the FC-CLIP baseline’s performance. Next, we explore more numerical results across various datasets, including evaluating mask recalls and an ablation study focusing on the co-training size of SAM. Finally, we present further qualitative visualization findings, featuring mask attention maps and segmentation results for two challenging datasets, A-847 [47] and PC-459 [26].

#### 6.1. Further discussion of FC-CLIP baseline

FC-CLIP adopts a checkpoint selection strategy based on the PQ accuracy within the ADE20K benchmark [47], a dataset known for its complexity with 150 diverse classes. Upon executing the FC-CLIP code and analyzing the finalround results, marked by * in Tab. 7, we observed tendencies towards overfitting and a subsequent decline in generalizability, This was accompanied by reduced effectiveness across various other open-vocabulary evaluation datasets, although there was improved performance on the COCO validation dataset. Despite FC-CLIP’s strategies to mitigate overfitting, the method’s effectiveness in open-vocabulary scenarios, especially in the context of ADE20K’s datasets, remains questionable. This raises concerns about the transparency of its model selection methodology. In contrast, our proposed framework, FrozenSeg, which leverages the last iteration’s checkpoints for inference, performs comparably on both the ADE20K and A-847 datasets. It demonstrates consistent and robust performance across all tested scenarios, thus eliminating the need for selective model evaluation.

#### 6.2. More numerical results 6.2.1 Comparative recall across datasets

In Tab. 8, we present the recall rates for our method and FC-CLIP across four datasets. This comparison is between the predicted mask proposals and class-agnostic semantic ground truth. We detail recall rates at IoU thresholds of 0.5, 0.75, and 0.9. The results demonstrate that FrozenSeg generally outperforms in generalizing to mask proposals for

- Table 7. Comparative analysis of FC-CLIP models’ performance on ADE20K, Cityscapes [7], COCO, PC-459, and A-847. Results in the first row are sourced from the paper [43] and are based on checkpoints selected from the ADE20K validation set. The * denotes outcomes from the final iteration. FC-CLIP* is re-implemented using the official code.

ADE20K Cityscapes COCO(seen) PC-459 A-847 Method PQ AP mIoU PQ AP mIoU PQ AP mIoU mIoU FWIoU mIoU FwIoU

FC-CLIP [43] 26.8 16.8 34.1 44.0 26.8 56.2 54.4 44.6 63.7 18.2 58.2 14.8 51.3 FC-CLIP* 25.1 16.4 32.8 44.3 27.9 56.0 56.4 47.4 65.3 17.3 56.3 14.0 48.1

Ours(*) 25.9 16.4 34.4 45.8 28.4 56.8 56.2 47.3 65.5 19.7 60.2 14.8 51.4

- Table 8. Evaluating open-vocabulary recall with semantic segmentation annotations on Cityscapes, PC-459, A-847, and PAS-21 [11]: insights into seen (S) and unseen (U) Classes.

Cityscapes(S/U) PC-459(S/U) A-847(S/U) PAS-21(S) Method 0.5 0.75 0.9 0.5 0.75 0.9 0.5 0.75 0.9 0.5 0.75 0.9

FC-CLIP 68.5/21.4 45.6/6.6 10.3/0.0 91.2/77.1 73.1/55.0 43.3/27.7 84.2/65.8 59.6/40.3 27.8/13.8 90.6 75.6 49.0 Ours 68.2/24.2 46.0/7.5 13.3/0.1 92.6/81.0 76.0/59.9 47.0/31.4 78.9/82.9 52.3/59.4 23.6/24.6 92.0 76.4 46.1

- Table 9. Open-vocabulary performance with SAMs on Cityscapes, Mapillary Vistas [27], PC-459, A-847, and PC-59 [26]. Bold highlights optimal results.

Cityscapes Mapillary Vistas PC-459 A-847 PC-59 SAM PQ AP mIoU PQ mIoU mIoU FWIoU mIoU FWIoU mIoU FWIoU

ViT-T [45] 43.0 25.9 55.0 18.6 28.0 17.6 57.0 14.0 48.1 56.8 66.1 ViT-B [1] 45.8 28.4 56.8 18.5 27.3 18.5 59.0 14.4 51.5 58.1 68.7 ViT-L [1] 44.2 27.1 56.2 18.5 27.5 17.4 57.0 14.2 48.6 56.7 66.2 ViT-H [1] 44.2 28.0 56.2 18.4 27.4 17.3 56.7 13.9 48.1 56.6 66.2

unseen classes.

##### 6.2.2 Comparison with different SAMs

In Tab. 9, we provide detailed results of FrozenSeg (w/o. mask ensemble), using ConvNeXt-L CLIP [29] alongside different size of co-trained SAM [1]: ViT-T (Tiny) [45], ViT-B (Base), ViT-L (Large) and ViT-H (Huge). Across the board, the ViT-B configuration stands out, delivering better performance in our evaluations. We also provide visualizations of the k-means clustering results for feature embeddings from the SAM image encoders. The visualization demonstrates that ViT-B balances segmentation accuracy and connectivity, offering precise segmentation with good instance connectivity. ViT-T provides coarse boundaries, while ViT-L and ViT-H, though more precise, have reduced instance connectivity and may be less effective for panoptic segmentation with CLIP. Thus, ViT-B’s balanced performance makes it a robust choice.

#### 6.3. More qualitative visualizations

##### 6.3.1 Attention maps

To illustrate the refinement of query features facilitated by injectors, we identify the query with the highest confidence

and present its corresponding attention map from the final cross-attention layer within the transformer decoder. We map the attention map back to the original image for visualization purposes. The results are depicted in Fig. 6. It is evident that our queries exhibit heightened attention towards both the object boundaries and intra-content regions, indicative of the effectiveness of our approach in mask proposal generation.

##### 6.3.2 More results

We have expanded our visual comparisons in PC-459 dataset shown in Fig. 8, and the A-847 dataset, depicted in Fig. 9. In Fig. 8, it can be seen that our method generates more precise masks which are highlighted by red boxes. Notably, we draw attention to the areas enclosed by white boxes, which exhibit coarse or imprecise annotations. For instance, the ’door’ is overlooked in the first column, and the ’chair’ annotations fail to precisely demarcate the chair legs. Meanwhile, in the second column, although the ground truth predominantly annotates the background as ’grass’, a closer inspection reveals a composite of ’soil’ and ’grass’, with ’sidewalks’ situated in the lower left quadrant. Fig. 9 exemplifies the efficacy of our proposed method

[Figure 6]

- Figure 6. Comparative visualization of query attention maps on the PAS-21 dataset. The enclosed white box delineates the queried object intended for visualization. Our learned queries distinctly emphasize the object’s boundaries and intra-content, showcasing the accuracy of attention allocation.

[Figure 7]

- Figure 7. K-means clustering visualization of feature embeddings from various SAM image encoders on the PC-459 dataset. Note that the cluster colors are randomly assigned.

in producing high-quality masks, extending across a diverse array of novel classes. These include but are not limited to, ’toys’, ’painted pictures’, ’baptismal fonts’, ’altars’, ’decorative elements’, ’columns’, ’pipes’, and ’fluorescent lighting’, among others.

[Figure 8]

###### Figure 8. Qualitative visualizations in PC-459 dataset. Enhanced sections are delineated by red boxes, while white boxes underscore regions with imprecise ground truth annotations.

[Figure 9]

###### Figure 9. Qualitative visualizations in A-847 dataset.

### References

- [1] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Alex Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 2, 9
- [2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European Conference on Computer Vision, 2020. 3
- [3] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C. Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server. arXiv:1504.00325, 2015. 6
- [4] Xi Chen, Shuang Li, Ser-Nam Lim, Antonio Torralba, and Hengshuang Zhao. Open-vocabulary panoptic segmentation with embedding modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision,

2023. 2, 5

- [5] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. VISION TRANSFORMER ADAPTER FOR DENSE PREDICTIONS. In International Conference on Learning Representations, 2023. 3
- [6] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Advances in Neural Information Processing Systems, 2021. 1, 3
- [7] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016. 2, 5, 9
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2009. 6
- [9] Zheng Ding, Jieke Wang, and Zhuowen Tu. Openvocabulary universal image segmentation with MaskCLIP. In Proceedings of the 40th International Conference on Machine Learning, 2023. 2, 5
- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 2, 5
- [11] Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (VOC) challenge. International Journal of Computer Vision, 2010. 5, 9
- [12] Agrim Gupta, Piotr Doll´ar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. 5
- [13] Jian Hu, Jiayi Lin, Weitong Cai, and Shaogang Gong. Relax image-specific prompt requirement in SAM: A single generic prompt for segmenting camouflaged objects. In Pro-

ceedings of the AAAI Conference on Artificial Intelligence,

2024. 2

- [14] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021. 1
- [15] Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, and Fisher Yu. Segment anything in high quality. In Advances in Neural Information Processing Systems, 2023. 2
- [16] Weicheng Kuo, Yin Cui, Xiuye Gu, AJ Piergiovanni, and Anelia Angelova. F-VLM: Open-vocabulary object detection upon frozen vision and language models. In International Conference on Learning Representations, 2023. 2, 3
- [17] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 7
- [18] Shuai Li, Minghan Li, Pengfei Wang, and Lei Zhang. OpenSD: Unified open-vocabulary segmentation and detection. arXiv:2312.06703, 2023. 1, 2
- [19] Yanghao Li. Exploring plain vision transformer backbones for object detection. In European Conference on Computer Vision, 2022. 5
- [20] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted CLIP. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 2, 5
- [21] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Doll´ar. Microsoft COCO: Common objects in context. In European Conference on Computer Vision, 2014. 5, 6
- [22] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. DAB-DETR: Dynamic anchor boxes are better queries for DETR. In International Conference on Learning Representations, 2022. 3
- [23] Yong Liu, Sule Bai, Guanbin Li, Yitong Wang, and Yansong Tang. Open-vocabulary segmentation with semantic-assisted calibration. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2024. 1, 2
- [24] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2022. 5
- [25] Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, and Jingdong Wang. Conditional DETR for fast training convergence. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021. 3
- [26] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the

- IEEE conference on computer vision and pattern recognition, 2014. 2, 5, 8, 9
- [27] Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulo, and Peter Kontschieder. The mapillary vistas dataset for semantic understanding of street scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision,

2017. 5, 9

- [28] Jie Qin, Jie Wu, Pengxiang Yan, Ming Li, Ren Yuxi, Xuefeng Xiao, Yitong Wang, Rui Wang, Shilei Wen, Xin Pan, and Xingang Wang. FreeSeg: Unified, universal and openvocabulary image segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2024. 3
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In In International conference on machine learning, 2021. 1, 9
- [30] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv:2401.14159, 2024. 2
- [31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. 3
- [32] Haochen Wang, Xiaolong Jiang, Haibing Ren, Yao Hu, and Song Bai. Swiftnet: Real-time video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, June 2021. 1
- [33] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Xiangtai Li, Wentao Liu, and Chen Change Loy. CLIPSelf: Vision transformer distills itself for open-vocabulary dense prediction. In International Conference on Learning Representations, 2024. 5
- [34] Yu-Huan Wu, Yun Liu, Xin Zhan, and Ming-Ming Cheng. P2t: Pyramid pooling transformer for scene understanding. TPAMI, 2023. 1
- [35] Yinghui Xing, Qirui Wu, De Cheng, Shizhou Zhang, Guoqiang Liang, Peng Wang, and Yanning Zhang. Dual modality prompt tuning for vision-language pre-trained model. IEEE Transactions on Multimedia, 2023. 2
- [36] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2023. 3, 5
- [37] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2023. 5
- [38] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for openvocabulary semantic segmentation with pre-trained visionlanguage model. In European Conference on Computer Vision, 2022. 1, 2

- [39] Xin Xu, Tianyi Xiong, Zheng Ding, and Zhuowen Tu. MasQCLIP for open-vocabulary universal image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 5
- [40] Haosen Yang, Chuofan Ma, Bin Wen, Yi Jiang, Zehuan Yuan, and Xiatian Zhu. Optimization efficient open-world visual region recognition. arxiv:2311.01373, 2024. 2, 5
- [41] Changqian Yu, Jingbo Wang, Changxin Gao, Gang Yu, Chunhua Shen, and Nong Sang. Context prior for scene segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2020. 1
- [42] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. BDD100k: A diverse driving dataset for heterogeneous multitask learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2020. 5
- [43] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional CLIP. In Advances in Neural Information Processing Systems, 2023. 1, 2, 3, 5, 9
- [44] Haobo Yuan, Xiangtai Li, Chong Zhou, Yining Li, Kai Chen, and Chen Change Loy. Open-vocabulary SAM: Segment and recognize twenty-thousand classes interactively. In European Conference on Computer Vision, 2024. 2, 5
- [45] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong. Faster segment anything: Towards lightweight sam for mobile applications. arXiv preprint arXiv:2306.14289, 2023. 9
- [46] Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free CLIP-adapter for better visionlanguage modeling. In European Conference on Computer Vision, 2022. 2
- [47] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ADE20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2017. 5, 8
- [48] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 2022. 2
- [49] Peng Zhou, Bor-Chun Chen, Xintong Han, Mahyar Najibi, Abhinav Shrivastava, Ser-Nam Lim, and Larry Davis. Generate, segment, and refine: Towards generic manipulation segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020. 1
- [50] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Kr¨ahenb¨uhl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In European Conference on Computer Vision, 2022. 7
- [51] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. DEFORMABLE DETR : DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION. In International Conference on Learning Representations, 2021. 5

