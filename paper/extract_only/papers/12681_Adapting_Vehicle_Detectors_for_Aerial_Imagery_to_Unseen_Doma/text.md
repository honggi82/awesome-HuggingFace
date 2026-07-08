# arXiv:2507.20976v1[cs.CV]28Jul2025

## Adapting Vehicle Detectors for Aerial Imagery to Unseen Domains with Weak Supervision

Xiao Fang1, Minhyek Jeon1, Zheyang Qin1, Stanislav Panev1, Celso de Melo2, Shuowen Hu2, Shayok Chakraborty1,3, Fernando De la Torre1 1Carnegie Mellon University, 2DEVCOM Army Research Laboratory, 3Florida State University, {xfang2, minhyekj, zheyangq, spanev}@andrew.cmu.edu, {celso.m.demelo.civ, shuowen.hu.civ}@army.mil, shayok@cs.fsu.edu, ftorre@cs.cmu.edu

#### Abstract

Detecting vehicles in aerial imagery is a critical task with applications in traffic monitoring, urban planning, and defense intelligence. Deep learning methods have provided state-of-the-art (SOTA) results for this application. However, a significant challenge arises when models trained on data from one geographic region fail to generalize effectively to other areas. Variability in factors such as environmental conditions, urban layouts, road networks, vehicle types, and image acquisition parameters (e.g., resolution, lighting, and angle) leads to domain shifts that degrade model performance. This paper proposes a novel method that uses generative AI to synthesize high-quality aerial images and their labels, improving detector training through data augmentation. Our key contribution is the development of a multi-stage, multi-modal knowledge transfer framework utilizing fine-tuned latent diffusion models (LDMs) to mitigate the distribution gap between the source and target environments. Extensive experiments across diverse aerial imagery domains show consistent performance improvements in AP50 over supervised learning on source domain data, weakly supervised adaptation methods, unsupervised domain adaptation methods, and open-set object detectors by 4-23%, 6-10%, 7-40%, and more than 50%, respectively. Furthermore, we introduce two newly annotated aerial datasets from New Zealand and Utah to support further research in this field. Project page is available at: https://humansensinglab.github.io/ AGenDA

#### 1. Introduction

Recent advancements in Generative AI have led to the development of diffusion-based models [50] that can synthesize images with unprecedented realism, making them

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]|
|---|

| |
|---|
|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>|

[Figure 7]

| |[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>|
|---|---|
| |[Figure 12]<br><br>[Figure 13]|

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

| | | |
|---|---|---|
| |[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>| |

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Figure 1. Overview. We propose a pipeline that generates highquality aerial images along with their labels. Our method outperforms baseline detectors trained on source images and open-set detectors directly inferred on target images.

nearly indistinguishable from real-world data [12, 36, 52]. Additionally, integrating natural human language has become crucial to modern vision systems. This integration can take the form of conditioning the generation process (for example, in text-to-image generative models [41, 46] utilizing cross-attention mechanism) or serving as an additional modality that shares a common latent space with

vision (as seen in models like CLIP [42]). One key reason for the success of these new models is that they have been trained on extensive datasets, such as LAION-5B [49] or WIT [42], consisting of hundreds of millions or billions of text-image pairs. As a result, the scientific community often refers to them as Foundational Models.

Upon their inception, researchers have explored how to use foundational models to generate training data for downstream tasks like classification, object detection, and semantic segmentation. Methods such as ALIA [8], DatasetDM [60], Dataset Diffusion [35], DiffusionEngine [70] are particularly useful in scenarios with limited or no access to natural training data.

Despite the promising results these methods have achieved augmenting general-purpose datasets like VOC [9] and COCO [28], little to no effort has been made to generate diverse, high-resolution annotated synthetic datasets on aerial views. These datasets are important for training detectors on small objects like vehicles. The primary reason for this gap is that off-the-shelf generative models, struggle to generate these views due to insufficient aerial imagery representation in their training datasets. Hence, fine-tuning is unavoidable in this case. Recent efforts have primarily produced generative models constrained by geospatial resolution, such as DiffusionSat [17] and SatDiffMoE [51], and therefore cannot be utilized for vehicle detection.

The lack of sufficient large-scale aerial view training data impacts the performance of not only generative foundational models but also other types of Vision Large Language Models [67] (VLLMs), such as BLIP2 [22], InternVL3 [75], and Gemini [55]. Our analysis reveals that these models perform poorly in zero-shot settings for tasks such as car presence classification and center localization in aerial imagery. (see Appendix Sec. C). This limitation also extends to open-set object detectors [23, 34, 72], which, as shown in Table 1, exhibit unsatisfactory results on aerial imagery.

On the other hand, detection annotations should also be produced for each generated image object. However, textto-image models do not naturally provide these. There are two potential options to address this challenge: 1) the annotations (e.g., semantic segmentation maps, bounding box layouts) to be created before the image generation and used as a spatial conditioning signal [26, 69], or 2) products from the generative process, such as cross- and self-attention maps extracted from the denoising U-Net [12] to be employed to image produce annotations [35, 54, 60, 70]. In the first case, studies reported that the generative models often do not fully obey the imposed conditioning, especially for small objects such as vehicles in aerial view images [62], which may result in generated annotations that do not match the corresponding synthetic images. Therefore, we studied the prospects of the second approach.

To this end, we introduce a new method for annotated

aerial view vehicle detection via synthetic image dataset generation, which employs stacked (multi-channel) crossattention maps, learnable text prompt tokens, and multistage cross-environment (source to target) knowledge transfer. We consider the scenario where a fully annotated dataset with bounding box annotations from the source environment and a target environment dataset with weak binary annotations (whether or not a vehicle is present in a given image) are available, which are much easier to obtain than full bounding box annotations. Our extensive experiments show considerable improvements over baseline (model trained only on the source environment data) performance demonstrated by four popular object detectors.

Our contributions can be summarized as follows:

- • We introduce a novel approach for generating aerial view annotated synthetic image datasets based on multichannel cross-attention maps and multi-stage crossenvironment knowledge transfer.
- • We conduct extensive experiments involving four popular state-of-the-art object detectors to assess the effectiveness of our approach. We compared the performance of our method with other published methods for open-set detection, unsupervised object detector adaptation, and weakly supervised model adaptation. The results corroborate the promise and potential of our framework.
- • We introduce two new real-world aerial view datasets captured in Selwyn (New Zealand), 2,078,077 images, and Utah (USA), 2,684,658 images, including car (small vehicle) detection task annotations.

#### 2. Related Work

##### 2.1. Diffusion Models for Perception Tasks

Diffusion models [6, 12] have undergone significant developments and have emerged as prominent generative models in modern research. These models work by progressively corrupting data with noise, and then learning to reverse this process to reconstruct the original data. Once trained, new data can be generated by applying the learned denoising process to randomly sampled noise. Latent diffusion models (LDM) [41, 46] perform diffusion process in the latent image space, which reduces the computation cost towards high-resolution image synthesis. Customized image generation can be achieved by adding various types of control, such as texts [47], edges [69], segmentation masks [38], geometric layouts [3], and images [43].

Recent research demonstrates that diffusion models can also be utilized in various perception tasks. Diffusion Classifier [21] reveals that pre-trained diffusion models can be directly employed for zero-shot classification tasks. DoGE [59] conditions Stable Diffusion [46] on CLIP image embedding difference between two domains, improving classification and semantic segmentation accu-

racy. DGInStyle [14] combines semantic masks with style prompts to generate training data for semantic segmentation. DatasetDM [60] uses a few labeled real images to train a mask decoder, leading to a robust synthetic data generator. Thanks to the cross-attention mechanism in Stable Diffusion, several works [35, 58, 61] obtain high-quality segmentation labels through cross-attention maps between images and target concepts. AttnDreambooth [37] further incorporates learnable tokens before target concepts to enhance the accuracy of cross-attention maps. Compared to previous works, our method differs in several aspects. First, we explore Stable Diffusion for cross-domain object detection where diffusion models are rarely trained, while previous works mostly focus on classification and segmentation. Second, we synthesize accurate labels by introducing learnable tokens to encode foreground and background concepts for more accurate cross-attention maps and labeling based on style-less cross-attention maps to remove domain gaps.

##### 2.2. Cross-domain Object Detection

Cross-domain object detection addresses the challenge of detecting objects when a domain shift exists between the source and target environments. It can be categorized into subfields based on the availability of data and labels during the training and testing phases.

Open-set detection is helpful in detecting novel categories in the target domain since it identifies both known and unknown objects during inference. One approach is aligning textual and visual features by jointly training object detection [23]. Another approach includes integrating transformer-based architectures or YOLO framework with open-set object detection, improving robustness in detecting objects without category constraints [4, 31]. Vision transformer-based models [33, 34] further enhance openvocabulary object detection by utilizing large-scale textimage pre-training, enabling models to detect beyond predefined categories. More recent research [7, 72] refines open-set detection through improved representation learning and domain generalization techniques.

Unsupervised domain adaptation for object detection aims to improve detection performance in an unlabeled target domain by leveraging various techniques. A common approach is style transfer [39, 59, 65], which aligns the visual characteristics of source and target domains to mitigate domain shifts. Recent research also uses knowledge distillation [16, 74], where knowledge from a well-trained teacher model is transferred to a student model, enabling adaptation to the target domain. Other methods include adversarial feature learning [27, 71] which minimizes domain discrepancies through adversarial training, or graph reasoning [24] that captures structural relationships between objects, enhancing robustness in cross-domain object detection.

Weakly supervised domain adaptation enhances object

detection in target domains where only limited supervision is available and can be categorized into networkbased methods and generative models. Network-based methods improve feature alignment and domain generalization through approaches such as hierarchical feature learning [66], transformer-based adaptations [19], and pseudolabeling [13]. These methods refine object detection models by utilizing structured architectural modifications that help bridge domain gaps. Generative models [14, 18] enhance adaptation by learning domain-invariant representations through style transfer and task-adaptive pretraining. By synthesizing target-like visual features, these approaches help models generalize more effectively to new environments with limited supervision, mitigating domain discrepancies while preserving task-relevant information.

#### 3. Method

Given the fully annotated (where all vehicles in each image are annotated with bounding boxes) source domain

data represented as DS = xSi ,yiS Ni=1S , and imagelevel annotated (whether or not a vehicle is present in a

given image) target domain data DT = xTi ,yiT Ni=1T , the goal is to synthesize a fully annotated dataset DG =

xGi ,yiG Ni=1G in the target domain, where x and y denote the images and their corresponding labels, respectively. NS, NT and NG denote the total number of images in each dataset. In Sec. 3.1, we present our approach for generating synthetic images with Stable Diffusion [46] by fine-tuning the model to adapt to both source and target domains. In Sec. 3.2, we leverage multi-channel cross-attention maps for object localization in synthetic images. In Sec. 3.3, we propose an automatic labeling strategy to generate pseudo labels for synthetic target domain data. A summary of the full pipeline is provided in the Appendix Sec. A.

##### 3.1. Text-to-image generation

Stable Diffusion consists of an image encoder E, a conditional U-Net ϵθ, a text encoder τθ, and a latent decoder D. We fine-tune the U-Net ϵθ on both DS and DT . In the forward process, an image x is encoded into a latent representation z0 = E(x). A noisy latent zt at any time t is then sampled using the following sampling function [12]:

zt = √α¯tz0 + √1 − α¯tε, ε ∼ N(0,I), (1)

where α¯t = ts=1 αs, and t is uniformly sampled from {1,...,T}. To learn the reverse process, the latent zt is passed to the U-Net ϵθ, along with the timestep t and the prompt embedding τθ(c). To encode domain-specific information, we design distinct prompts for source and target domain images, denoted as cS and cT . cS and cT follow the format of “an aerial image with [category] in [S]” and “an aerial image with [category] in [T]”, where [category]

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

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|

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

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| |
|---|

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

- Figure 2. Overview of our pipeline. It consists of two stages. First, we finetune Stable Diffusion and synthesize both source and target domain images. Second, we automatically label synthetic target domain images via cross-attention maps.

represents the object type, and [S] and [T] are unique identifiers to distinguish between the source and target domain prompts, as inspired by [47]. The U-Net ϵθ is trained to predict the noise added to the latent z0:

LLDM = EE(x),c,ϵ,t ∥ϵ − ϵθ(zt,t,τθ(c))∥22 (2)

where c ∈ {cS,cT } depends on the domain of the input image x. During image generation, a pure noise latent zT is iteratively denoised through the U-Net ϵθ for T steps, followed by decoding via the latent decoder D to generate the final image. By conditioning the U-Net ϵθ on distinct prompts cS and cT , we can independently synthesize images corresponding to the source and target domains.

##### 3.2. Concept Localization

It has been observed in [54] that in a well-trained textto-image diffusion model, cross-attention maps between the feature maps and the conditioning text assign higher weights to regions that align with the text concept. This indicates that cross-attention maps effectively help locate the concept. Therefore, we propose leveraging cross-attention maps corresponding to “[category]” in a fine-tuned Stable Diffusion to facilitate downstream visual tasks. Specifically, cross-attention maps can be obtained in each of the four layers in the U-Net ϵθ, which corresponds to four different resolutions. We compute the attention map A by

averaging all the cross-attention maps across these resolutions. Finally, we normalize A to (0,1) to emphasize regions with higher attention weights.

While cross-attention maps for the object category effectively highlight relevant regions, their robustness can be further improved by cross-verifying with additional maps. Inspired by [10, 37], we introduce two learnable tokens in the prompt to generate complementary cross-attention maps. These tokens are designed to capture both the object and the background concept, which includes all regions outside the objects. The insight is that combining a learnable context token with the object category enhances the localization of the target objects, while a background token helps identify non-target regions. By effectively localizing the background, we can further refine object delineation. To implement this approach, we design a two-stage pipeline. Throughout the two stages, we set the prompt as “an aerial view image with [V1] [category] in [V2] [S]” for source domain data and “an aerial view image with [V1] [category] in [V3] [T]” for target domain data. [V1] represents the learnable token for the object concept, while [V2] and [V3] correspond to the learnable token for the source domain and target domain background concept, respectively.

In the first stage, we fine-tune both the U-Net ϵθ and learnable tokens [V1], [V2], and [V3] to capture the new concepts. To facilitate this process, we introduce a novel crossattention map regularization loss that encourages similarity

between the attention maps of [V1] and [category], while penalizing similarity between the attention maps of [V2], [V3] and [category]. We denote the normalized cross-attention map of [V1], [V2], [V3], and [category] as AV

, AV

, AV

1

2

3

and Ac, respectively. To enforce similarity between AV

1

and Ac, we further normalize them into discrete probability distributions where pixel values sum to one while preserving their relative differences. We denote them as AˆV

1

and Aˆc. We then employ the total variation distance metric to minimize the difference between AˆV

and Aˆc, which is equivalent to half of the L1 distance between them [20]:

1

- 1

- 2 x,y |AˆV

(x,y) − Aˆc(x,y)| (3)

Lobj =

1

Similarly, to penalize similarity between Abg and Ac where Abg ∈ {AV

3}, which is equivalent to enforce similarity between Abg and A∗c where A∗c = 1 − Ac, we first normalize both Abg and A∗c into discrete probability distributions and then compute the distribution difference using the total variation difference metric as follows:

,AV

2

- 1

- 2 x,y |Aˆbg(x,y) − Aˆ∗c(x,y)| (4)

Lbg =

The total loss function can be formulated as L = LLDM + Lobj + Lbg.

In the second stage, we fix the learned tokens [V1], [V2], and [V3], and further fine-tune the U-Net ϵθ. This is because Stable Diffusion learns to fit the data distribution with varying prompts in the previous stage, which might not align well with the data distribution conditioned on the final learned prompt. We continue to apply the loss function L to guide the learning of the attention maps and avoid any embedding misalignment.

##### 3.3. Automatic Labeling via Cross-Attention Maps

After fine-tuning Stable Diffusion as described in Sec. 3.1 and Sec. 3.2, we generate synthetic source domain data

- DGS = xGSi , AGSi

NGS i=1

and target domain data

- DGT = xGTi , AGTi

NGT i=1

. For each synthetic data sam-

ple Di ∈ DGS ∪ DGT , we extract the cross-attention maps Ai,c, Ai,V

and Ai,bg during denoising steps. The enhanced cross-attention map Ai is then obtained by stacking these components. Since cross-attention maps are grayscale images that highlight object regions, they contain less style information than RGB images. Therefore, we propose using these maps to generate bounding box annotations for target domain data. First, we train a detector FS(·;θ) on the fully annotated real source domain data DS. This detec-

1

tor is then used to predict reliable pseudo labels yiGS Ni=1GS on the synthetic source domain images xGSi Ni=1GS. Next,

we train another detector FA(·;θ) on the combined dataset DGS∗ = AGSi ,yiGS

NGS i=1

, which comprises synthetic source domain enhanced cross-attention maps and their corresponding pseudo labels. Finally, we use the welltrained detector FA(·;θ) to test the target domain cross-

NGT i=1

attention maps AGTi

and predict a set of pseudo la-

bels yiGT Ni=1GT for synthetic target domain data DGT . This results in a fully annotated synthetic target domain dataset

xGTi ,yiGT Ni=1GT , which can be directly used for downstream object detection tasks.

Determining the confidence score threshold for bounding box labels is challenging because it can vary across datasets. To address this, we propose a classifier refinement method that selects more reliable labels for synthetic target domain data, inspired by [53]. Predicted bounding boxes with high confidence scores represent foreground objects, while those with low scores represent background regions. We define bounding boxes above a high threshold λhigh as positive samples and those below a low threshold λlow as negative samples. We train a classifier using these samples on cropped image patches, refining predictions for samples with intermediate confidence scores, resulting in more reliable labels for synthetic target domain data.

#### 4. Datasets

We use three real-world aerial view vehicle detection datasets for our experiments—the publicly available DOTA [64], and two additional datasets we created, called LINZ and UGRC (Figure 3). All three have ground sampling distance (GSD) of 12.5 cm per px and have been sampled to 112 px × 112 px image size. Utilizing this image size is essential for our method because diffusion models are known for struggling with generating small objects caused by the cross-attention mechanism’s limited resolution. Thus, we increase the relative object size within the images. We use only one object class (small vehicle) and location labels (object centers).

DOTA: We used the DOTA-v2 dataset’s training and validation sets, which contain 1,830 and 593 images, respectively, with 169,268 and 56,062 unique small car instances. The image sizes range from 421 px × 346 px to 29,200 px × 27,620 px. We only used images with GSD of 15 cm per px or less and scale them to match the target 12.5 cm per px. The images with larger GSD or whose metadata do not provide such information are rejected. Finally, we produce our experiments’ training and validation sets by placing randomly rotated square sampling windows with size 112 px × 112 px uniformly distributed across each original image area. This results in 455,099 training images (28,453 contain objects) and 142,262 validation images (9,460 contain objects), with 62,178 and 20,194 total object instances,

[Figure 81]

[Figure 82]

- Figure 3. Image samples from our datasets. (left) LINZ sample, (right) UGRC, (green markers) small vehicle location annotations. For more examples, check the Supplementary Material.

respectively. The original bounding box labels were converted to object locations to match the annotations of the other two datasets described below.

LINZ: We created this dataset by manually annotating vehicle locations in aerial images captured in Selwyn, New Zealand in December 2012. They were obtained from the Land Information New Zealand (LINZ) online platform [29]. The original tiles’ size is 240 m × 360 m (1920 px × 2880 px) with GSD of 12.5 cm per px. A uniformly distributed sampling window produces the final samples with size 112 px × 112 px. We finally have

- 1,451,144 training images (19,564 of which include objects), 188,744 validation images (2,108 of which include objects), and 438,189 test images (2,629 of which include objects). The number of object instances in the training set is 29,495, in the validation set, 2,574, and in the test set, 3,640. The Appendix Sec. B provides more details. UGRC: We used the same approach as with LINZ to construct this dataset. We obtained 40 high-resolution (12.5 cm per px) aerial image tiles captured in Utah, USA and annotated the locations of small vehicle instances. The images were downloaded from Utah Geospatial Resource Center (UGRC) [57] online platform. Their native size is
- 2,000 m × 2,000 m (16,000 px × 16,000 px). After performing dataset sampling using a 112 px square window, we constructed splits with the following quantities: 2,142,849 training images (15,631 of which include objects), 271,252 validation images (3,912 of which include objects), and 270,557 test images (1,510 of which include objects). The number of object instances in the training set is 26,001, in the validation set 9,878, and in the test set 1,900. More information can be found in the Appendix Sec. B.

#### 5. Experiments

##### 5.1. Implementation Details

Working with location-based annotations: As discussed in Sec. 4, LINZ and UGRC datasets contain small vehicle location labels. DOTA’s bounding box annotations were

converted to location labels for consistency. To leverage the abundance of bounding box-based open-source object detection frameworks, we reformulate the location detection problem as a bounding box detection task. Specifically, we define a decision circle with a 12 px radius centered on the vehicle’s centroid. A predicted center is considered a true positive if it falls within this circle. Then we introduce a 42.36 px square pseudo-bounding box centered at the vehicle’s centroid. This specific size is chosen to facilitate the evaluation using the AP50 object detection metric. Mathematically, this approach is functionally equivalent to using the aforementioned decision circle, with minimal error.

Automatic Labeling: To establish detection baselines, we employ the MMDetection framework [2], and evaluate a diverse set of models, including Faster-RCNN [45], YOLOv5 [15], YOLOv8 [44] and ViTDet [25], which represent one-stage, two-stage and transformer-based object detectors. The automatic labeling process follows the methodology outlined in section 3.3, ensuring consistency by utilizing the same detector throughout the entire process. For label refinement, we fine-tune a pre-trained ResNet [11] on the ImageNet [48] dataset for 80 epochs with a batch size of 256. The predefined confidence thresholds λhigh and λlow are determined based on the confidence score that yields the optimal F1-score for synthetic source domain crossattention maps during detector training. Specifically, we set λhigh = 0.7 and λlow = 0.3 for both YOLOv5 and YOLOv8, while for ViTDet and Faster-RCNN, we assign values of λhigh = 0.95 and λlow = 0.5.

Image Generation: We employ Stable diffusion V1.4 [46], pre-trained on the images of size 512 × 512, a batch size of 64, and a learning rate of 10−6 on two RTX A6000 GPUs for approximately 15 epochs. Following section 3.2, we adopt a two-stage fine-tuning strategy to capture both object and background concepts. For the source domain, the guidance prompt is “An aerial view image with [V1] cars in [V2][id]” for images with small vehicles since most small vehicles fall under the broader classification of cars (see Appendix Sec. B for more details), and “An aerial view image in [V2][id]” for images without small vehicles, where [id] is replaced with “New Zealand” or “DOTA”. For the target domain, [V2] is substituted with token [V3] to learn unique background concept in target domain, and [id] is replaced with “Utah”. In the first stage, we fine-tune the learnable tokens in the prompt along with the U-Net. In the second stage, we freeze the learned tokens and fine-tune U-Net to better align the model with the data distribution. Each stage is trained for approximately two epochs using a batch size of 8 and a learning rate of 5 × 10−7 on two RTX A6000 GPUs. We synthesize 10k images containing cars for both the source and target domain, and 10k images without cars for the target domain. Additional implementation details are provided in the Appendix Sec. D.

###### Method Backbone LINZ→UGRC DOTA→UGRC

AP50(%) AP50(%) Supervised Object detection (Source-only)

Faster-RCNN [45] Faster R-CNN [45] 53.1 52.5 YOLOv5 [15] YOLOv5 [15] 63.2 57.6 YOLOv8 [15] YOLOv8 [44] 62.9 71.4 ViTDet [25] ViTDet [25] 55.7 50.4

Open-set object detection GLIP-T [23] Swin-T [32] 8.7 8.7 OmDet-Turbo [72] Swin-T [32] 14.4 14.4 OWLV2 [34] CLIP ViT [42] 17.9 17.9

Unsupervised domain adaptation object detection

SIGMA [24] Faster R-CNN [45] 36.2 34.8 TIA [71] Faster R-CNN [45] 49.2 42.4 Adapt. Teacher [27] Faster R-CNN [45] 29.0 37.0 CycleGAN-Turbo [39] YOLOv5 [15] 56.0 60.8 SSDA-YOLO [74] YOLOv5 [15] 52.3 49.6

Cross domain weakly supervised object detection

OCUD [73] Faster R-CNN [45] 63.1 65.3 H2FA R-CNN [66] Faster R-CNN [45] 61.8 68.3

Ours Faster R-CNN [45] 69.3 75.5 Ours YOLOv5 [15] 68.8 68.5 Ours YOLOv8 [44] 75.4 75.7 Ours ViTDet [25] 72.0 67.1

Table 1. Cross-domain object detection results. LINZ to UGRC and DOTA to UGRC. We report the AP50 result.

##### 5.2. Comparison with State-of-the-art Methods

Table 1 summarizes the cross-domain detection results for adaptation from the LINZ and DOTA datasets to the UGRC dataset. First, our method consistently outperforms detectors trained solely on source domain data, demonstrating its robustness and generalizability. Second, open-set detectors perform poorly on aerial images, highlighting the lack of training of foundational models in this domain. Third, compared to domain adaptation methods [24, 27, 66, 71, 73] based on Faster-RCNN [45], our pipeline achieves the highest AP50 of 75.4% for LINZ → UGRC, and 75.7% for DOTA → UGRC, surpassing the best unsupervised method by 20.1% and 33.1%, and the best weakly supervised method by 6.2% and 7.2%, respectively. Similarly, using YOLOv5 [15] as the backbone, our pipeline outperforms existing methods [39, 74] by 12.8% and 7.7%. These results strongly corroborate the promise and potential of our method against competing baselines, for the challenging task of vehicle detection from aerial imagery.

##### 5.3. Ablation Studies 5.3.1. Effectiveness of learnable tokens

We evaluate the effectiveness of stacking three crossattention maps of the word “car”, learnable tokens of car concept and background for small vehicle detection using YOLOv5 and YOLOv8, as summarized in Table 2.

LINZ→UGRC DOTA→UGRC AP50(%) AP50(%)

Method Backbone

Base YOLOv5 62.1 68.0 Base YOLOv8 69.8 74.5 Stack YOLOv5 68.8 68.5 Stack YOLOv8 75.4 75.7

Table 2. Comparison between labeling using single crossattention map and multi-channel cross-attention maps. “Base” denotes using cross-attention maps of word “car” only. “Stack” denotes stacked cross-attention maps extracted from the word “car” and two learnable tokens for the car concept and background.

Compared to using only the cross-attention map of the word “car”, incorporating multiple cross-attention maps improves detection accuracy by 5.6% when adapting from LINZ to UGRC, and by 1.2% from DOTA to UGRC. Additionally, we provide visualizations of cross-attention maps corresponding to different tokens and analyze their impact on labeling synthetic images, as illustrated in Figure 4. The visualizations reveal that different cross-attention maps tend to focus on distinct regions of the cars. By cross-verifying using multiple maps, the labeling process becomes more robust, leading to improved label quality.

###### 5.3.2. Effectiveness of label refinement

To evaluate the effectiveness of fine-tuning a classifier to refine labels, we compare our method against a fixed filtering confidence threshold set at 0.3, 0.4, 0.5, 0.6, and 0.7 using YOLOv8. As shown in Figure 5, the detection performance varies across different datasets without a consistent pattern about the confidence threshold. When compared to the results obtained using different thresholds, our method demonstrates competitive performance, suggesting that it produces a reliable set of labels. Additionally, compared to our method, when setting a lower threshold the detector labels more false positive samples, while setting a higher threshold the model labels fewer cars, as shown in Figure 6. This indicates our method can be helpful to select a more reliable set of labels.

#### 6. Conclusions

This paper introduces a novel approach that leverages diffusion models to automatically generate synthetic aerial view images alongside object location annotations based on multi-channel cross-attention maps. Extensive experiments conducted with four object detectors, as well as comparisons with open-set detectors, unsupervised domain adaptation methods, and weakly supervised models, demonstrate the effectiveness of our method in object detection for aerial imagery. Additionally, we introduce two new aerial view datasets from New Zealand and Utah, including annotations for small vehicle detection tasks. For future work, we plan

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

(a) (b) (c) (d) (e) (f) (g)

- Figure 4. Cross-attention maps for different tokens. We analyze the effectiveness of utilizing multi-channel cross-attention maps by assessing the label quality synthetic UGRC images. (a) Synthetic UGRC images. (b) Labels generated using only the cross-attention map of the word “car”. (c) Labels generated by using multi-channel heatmaps. (d) Multi-channel cross-attention maps. (e) Cross-attention maps of the word “car”. (f) Cross-attention maps of token [V1], which is designed to capture the concept of cars. (g) Cross-attention maps of token [V3] for background concept, which are further inverted for better comparison. In (b) and (c), bounding boxes with dotted lines denote the predicted pseudo-bounding box labels while dots denote predicted vehicle centers. In (e), (f), and (g), grayscale cross-attention maps are displayed as color heatmaps to highlight the intensity difference.

|74.1<br><br>74.7 74.6<br><br>75.2<br><br>74.5<br><br>75.3<br><br>74.1<br><br>74.5<br><br>73.1<br><br>74.6<br><br>75.4 75.7<br><br>70<br><br>72<br><br>74<br><br>76<br><br>LINZ → UGRC DOTA → UGRC<br><br>0.3<br><br>0.4 0.5<br><br>0.6<br><br>0.7<br><br>Ours|
|---|

- Figure 5. Quantitative comparison with varying thresholds. We report the AP50 result.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

(a) (b) (c) (d) (e)

- Figure 6. Comparison of our method with different thresholds for label quality. Blue bounding boxes represent pseudo labels generated using fixed thresholds, while red bounding boxes correspond to labels obtained through refinement. The blue and red dots indicate the predicted car centers. (a) to (e) show the results for thresholds of 0.3, 0.4, 0.5, 0.6, and 0.7, respectively.

to integrate VLLMs trained for aerial view images to automatically identify the presence of vehicles in aerial imagery. This would allow us to generate weak labels within the pipeline itself, enabling a fully unsupervised workflow.

Despite its strengths, our method has certain limitations that should be considered. First, we sample large aerial view images into smaller patches of resolution 112 px × 112 px to mitigate challenges in detecting small objects. This limitation arises because Stable Diffusion progressively downsamples the cross-attention map to a size of 8 × 8. Consequently, target objects may occupy less than a 1 × 1 grid, making it difficult for the model to generate precise attention maps. Second, our approach may encounter difficulties in handling overlapping objects. When multiple objects are close together, their cross-attention maps overlap, making it challenging for detectors to separate individual objects when using styleless cross-attention maps for synthetic data annotations.

#### 7. Acknowledgement

We thank Jessica Hodgins, Justin Macey, Melanie Danver, Katie Tender, and Eric Yu for their valuable help in creating the datasets. This work has been funded by the DEVCOM Army Research Laboratory.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. 2
- [2] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, and Dahua Lin. MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019. 6
- [3] Kai Chen, Enze Xie, Zhe Chen, Yibo Wang, Lanqing HONG, Zhenguo Li, and Dit-Yan Yeung. GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation. In The Twelfth International Conference on Learning Representations, 2024. 2
- [4] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. YOLO-World: Real-Time Open-Vocabulary Object Detection. In Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [5] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2
- [6] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion Models Beat GANs on Image Synthesis. In Advances in Neural Information Processing Systems, 2021. 2
- [7] Penghui Du, Yu Wang, Yifan Sun, Luting Wang, Yue Liao, Gang Zhang, Errui Ding, Yan Wang, Jingdong Wang, and Si Liu. LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction. In Proceedings of the European conference on computer vision (ECCV), 2024. 3
- [8] Lisa Dunlap, Alyssa Umino, Han Zhang, Jiezhi Yang, Joseph E Gonzalez, and Trevor Darrell. Diversify your vision datasets with automatic diffusion-based augmentation. In Advances in Neural Information Processing Systems, pages 79024–79034. Curran Associates, Inc., 2023. 2
- [9] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results. http://www.pascalnetwork.org/challenges/VOC/voc2012/workshop/index.html,

2012. 2

- [10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In The Eleventh International Conference on Learning Representations, 2023. 4
- [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 6
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In Advances in Neural Infor-

- mation Processing Systems, pages 6840–6851. Curran Associates, Inc., 2020. 1, 2, 3
- [13] Naoto Inoue, Ryosuke Furuta, Toshihiko Yamasaki, and Kiyoharu Aizawa. Cross-domain weakly-supervised object detection through progressive domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5001–5009, 2018. 3
- [14] Yuru Jia, Lukas Hoyer, Shengyu Huang, Tianfu Wang, Luc Van Gool, Konrad Schindler, and Anton Obukhov. DGInStyle: Domain-Generalizable Semantic Segmentation with Image Diffusion Models and Stylized Semantic Control. In Computer Vision – ECCV 2024, pages 91–109, Cham, 2025. Springer Nature Switzerland. 3
- [15] Glenn Jocher. Yolov5 by ultralytics, 2020. 6, 7, 2
- [16] Justin Kay, Timm Haucke, Suzanne Stathatos, Siqi Deng, Erik Young, Pietro Perona, Sara Beery, and Grant Van Horn. Align and distill: Unifying and improving domain adaptive object detection. arXiv preprint arXiv:2403.12029, 2024. 3
- [17] Samar Khanna, Patrick Liu, Linqi Zhou, Chenlin Meng, Robin Rombach, Marshall Burke, David B. Lobell, and Stefano Ermon. DiffusionSat: A Generative Foundation Model for Satellite Imagery. In The Twelfth International Conference on Learning Representations, 2024. 2
- [18] Neehar Kondapaneni, Markus Marks, Manuel Knott, Rogerio Guimaraes, and Pietro Perona. Text-Image Alignment for Diffusion-Based Perception. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13883–13893, 2024. 3
- [19] Tyler LaBonte, Yale Song, Xin Wang, Vibhav Vineet, and Neel Joshi. Scaling novel object detection with weakly supervised detection transformers. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 85–96, 2023. 3
- [20] David A Levin and Yuval Peres. Markov chains and mixing times. American Mathematical Soc., 2017. 5
- [21] Alexander C Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly a zero-shot classifier. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2206–2217,

2023. 2

- [22] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

2023. 2, 3

- [23] Liunian Harold Li*, Pengchuan Zhang*, Haotian Zhang*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In CVPR, 2022. 2, 3, 7
- [24] Wuyang Li, Xinyu Liu, and Yixuan Yuan. Sigma: Semanticcomplete graph matching for domain adaptive object detection. In CVPR, 2022. 3, 7
- [25] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In Computer Vision – ECCV 2022, pages 280–296, Cham, 2022. Springer Nature Switzerland. 6, 7, 2
- [26] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee.

- Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22511–22521, 2023. 2, 3
- [27] Yu-Jhe Li, Xiaoliang Dai, Chih-Yao Ma, Yen-Cheng Liu, Kan Chen, Bichen Wu, Zijian He, Kris Kitani, and Peter Vajda. Cross-Domain Adaptive Teacher for Object Detection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3, 7
- [28] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014. 2
- [29] LINZ. Selwyn 0.125m Urban Aerial Photos (2012-2013). https://data.linz.govt.nz/layer/51926selwyn-0125m-urban-aerial-photos-20122013/, 2014. 6
- [30] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge, 2024. 2
- [31] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying DINO with Grounded Pre-training for Open-Set Object Detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024. 3, 2
- [32] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 7
- [33] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. Simple open-vocabulary object detection. In Computer Vision – ECCV 2022, pages 728–755, Cham, 2022. Springer Nature Switzerland. 3, 2
- [34] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling Open-Vocabulary Object Detection. In Advances in Neural Information Processing Systems, pages 72983–

73007. Curran Associates, Inc., 2023. 2, 3, 7

- [35] Quang Ho Nguyen, Truong Tuan Vu, Anh Tuan Tran, and Khoi Nguyen. Dataset Diffusion: Diffusion-based Synthetic Data Generation for Pixel-Level Semantic Segmentation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2, 3
- [36] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 1

- [37] Lianyu Pang, Jian Yin, Baoquan Zhao, Feize Wu, Fu Lee Wang, Qing Li, and Xudong Mao. AttnDreamBooth: Towards Text-Aligned Personalized Text-to-Image Generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3, 4

- [38] Dong Huk Park, Grace Luo, Clayton Toste, Samaneh Azadi, Xihui Liu, Maka Karalashvili, Anna Rohrbach, and Trevor Darrell. Shape-Guided Diffusion With Inside-Outside Attention. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 4198–4207,

2024. 2

- [39] Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036, 2024. 3, 7
- [40] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Qixiang Ye, and Furu Wei. Grounding multimodal large language models to the world. In The Twelfth International Conference on Learning Representations, 2024. 3
- [41] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 1, 2
- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2, 7
- [43] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2, 6

- [44] Dillon Reis, Jordan Kupec, Jacqueline Hong, and Ahmad Daoudi. Real-time flying object detection with yolov8. arXiv preprint arXiv:2305.09972, 2023. 6, 7, 2
- [45] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2015. 6, 7, 2
- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-Resolution Image Synthesis With Latent Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 1, 2, 3, 6, 7
- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22500–22510, 2023. 2, 4
- [48] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015. 6
- [49] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo

- Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. 2
- [50] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015. 1
- [51] Bowen Song, Zhaoxu Luo, and Liyue Shen. Satdiffmoe: A mixture of estimation method for satellite image superresolution with latent diffusion models. In ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling, 2024. 2
- [52] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 1
- [53] Peng Tang, Xinggang Wang, Xiang Bai, and Wenyu Liu. Multiple Instance Detection Network With Online Instance Classifier Refinement. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2017. 5

- [54] Raphael Tang, Linqing Liu, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Pontus Stenetorp, Jimmy Lin, and Ferhan Ture. What the DAAM: Interpreting Stable Diffusion Using Cross Attention. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5644–5659, Toronto, Canada, 2023. Association for Computational Linguistics. 2, 4
- [55] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2
- [56] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2
- [57] UGRC. Utah High Resolution Orthophotography (HRO) 2012 Images. https://gis.utah.gov/ products/sgid/aerial- photography/highresolution-orthophotography/, 2012. 6
- [58] Jinglong Wang, Xiawei Li, Jing Zhang, Qingyuan Xu, Qin Zhou, Qian Yu, Lu Sheng, and Dong Xu. Diffusion model is secretly a training-free open vocabulary semantic segmenter. CoRR, abs/2309.02773, 2023. 3
- [59] Yinong Oliver Wang, Younjoon Chung, Chen Henry Wu, and Fernando De la Torre. Domain Gap Embeddings for Generative Dataset Augmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28684–28694, 2024. 2, 3
- [60] Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, and Chunhua

- Shen. DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2, 3
- [61] Weijia Wu, Yuzhong Zhao, Mike Zheng Shou, Hong Zhou, and Chunhua Shen. DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 1206– 1217, 2023. 3
- [62] Yiming Wu, Qihe Pan, Zhen Zhao, Zicheng Wang, Sifan Long, and Ronghua Liang. SOEDiff: Efficient Distillation for Small Object Editing. ACM Trans. Multimedia Comput. Commun. Appl., 2025. 2
- [63] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-ofexperts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024. 2
- [64] Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, and Liangpei Zhang. DOTA: A Large-Scale Dataset for Object Detection in Aerial Images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2018. 5

- [65] Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, and Joyce Chai. CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation. In Thirtyseventh Conference on Neural Information Processing Systems, 2023. 3
- [66] Yunqiu Xu, Yifan Sun, Zongxin Yang, Jiaxu Miao, and Yi Yang. H2FA R-CNN: Holistic and Hierarchical Feature Alignment for Cross-Domain Weakly Supervised Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14329–14339, 2022. 3, 7
- [67] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 11(12), 2024. 2
- [68] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid Loss for Language Image Pretraining. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 2
- [69] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3836–3847, 2023. 2
- [70] Manlin Zhang, Jie Wu, Yuxi Ren, Ming Li, Jie Qin, Xuefeng Xiao, Wei Liu, Rui Wang, Min Zheng, and Andy J Ma. Diffusionengine: Diffusion model is scalable data engine for object detection. arXiv preprint arXiv:2309.03893, 2023. 2
- [71] Liang Zhao and Limin Wang. Task-Specific Inconsistency Alignment for Domain Adaptive Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14217–14226, 2022. 3, 7
- [72] Tiancheng Zhao, Peng Liu, Xuan He, Lu Zhang, and Kyusong Lee. Real-time Transformer-based Open-Vocabulary

- Detection with Efficient Fusion Head. arXiv preprint arXiv:2403.06892, 2024. 2, 3, 7
- [73] Yuanyi Zhong, Jianfeng Wang, Jian Peng, and Lei Zhang. Boosting weakly supervised object detection with progressive knowledge transfer. In Computer Vision – ECCV 2020, pages 615–631, Cham, 2020. Springer International Publishing. 7
- [74] Huayi Zhou, Fei Jiang, and Hongtao Lu. SSDA-YOLO: Semi-supervised domain adaptive YOLO for cross-domain object detection. Computer Vision and Image Understanding, 229:103649, 2023. 3, 7
- [75] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2

## Adapting Vehicle Detectors for Aerial Imagery to Unseen Domains with Weak Supervision

### Supplementary Material

Algorithm 1 Summary of the pipeline Input: Source domain data DS, target domain data DT

- 1: Fine-tune Stable Diffusion [46] on both DS and DT using domain-specific prompts
- 2: Generate synthetic images xGS and extract stacked

- cross-attention maps AGS for the source domain

3: Generate synthetic images xGT and extract stacked

- cross-attention maps AGT for the target domain

- 4: Train a detector FS(·;θ) on DS
- 5: Run FS(·;θ) on DS to obtain pseudo labels yGS
- 6: Train a detector FA(·;θ) on AGS,yGS
- 7: Run FA(·;θ) on AGT to obtain pseudo labels yGT
- 8: Train the final detector FT (·;θ) on xGT ,yGT
- 9: Test FT (·;θ) on real target domain images from DT

#### A. Methods

In this section, we present a summary of steps regarding our proposed pipeline, as shown in Algorithm 1.

#### B. Datasets

In this section, we provide additional details and examples of the LINZ and UGRC datasets. Figure 8 illustrates the geographic regions from which our data samples were obtained. The LINZ online platform captured aerial imagery from nine areas in Selwyn. For dataset construction, we designated one of these nine areas as the testing region, from which test set images were sampled, while the remaining eight areas were used for training and validation samples. Similarly, the UGRC online platform collected aerial imagery from seven regions in Utah. One of these seven areas was designated as the testing region, with the remaining six serving as sources for training and validation data. This spatial partitioning strategy ensures that our datasets do not suffer from data leakage, as the training and testing areas are spatially independent. Within each area, we randomly sample square images of size 112 px × 112 px. Due to the sampling strategy, a single vehicle can appear in multiple images. Figure 7 presents the subcategories within the small vehicle class. Except for the Pickup truck category, all other small vehicle subcategories fall under the broader classification of cars. Accordingly, we utilize the word “car” in prompts to guide the Stable Diffusion [46] during image generation, leveraging its pre-trained perceptual understanding related to cars. Figure 9 provides visual ex-

Microcar

Station wagon

[Figure 114]

[Figure 115]

Minivan

Hatchback

[Figure 116]

[Figure 117]

Convertible

SUV

[Figure 118]

[Figure 119]

Pickup truck

Coupé

[Figure 120]

[Figure 121]

Sedan

[Figure 122]

Figure 7. Vehicles belonging to the object class small vehicle.

amples of both LINZ and UGRC images, highlighting distinct visual characteristics: UGRC includes a notable proportion of off-road vehicles, reflecting its sandy and rocky terrain, while LINZ images primarily feature urban vehicles within structured road networks. Lastly, Figure 10 compares the performance of four object detectors under two settings: cross-dataset evaluation (trained on LINZ and tested on UGRC) and within-dataset evaluation (trained and tested on UGRC). The results show that within-dataset performance surpasses cross-dataset performance by at least 25.7% higher AP50 across all detectors, underscoring a significant domain gap between the two datasets.

#### C. Limitation of Foundation Models

In this section, we provide more examples of limitations of various types of foundation models, including open-set detectors, diffusion models, and vision large language models, when detecting small vehicles in aerial view images.

| |
|---|

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

| |
|---|

(a) Selwyn (New Zealand)

(b) Utah (USA)

Figure 8. Geographic regions where we construct LINZ and UGRC datasets. Red bounding boxes denote the testing area.

any cars in aerial images, highlighting its limitations in this specific context. These findings underscore the challenges faced by open-set object detectors in accurately identifying vehicles in aerial imagery.

Method Vision Backbone LINZ→UGRC

Precision(%) Recall(%) Vision Large Language Model

- Gemini 1.5 Flash [56] - 2.9 44.5
- Gemini 2.0 Flash-Lite [5] - 6.6 26.3 InternVL3-8B [75] InternViT [75] 4.7 22.0 Qwen2.5-VL-7B [1] ViT [1] 0.4 4.8 DeepSeek-VL2-Tiny [63] SigLIP-SO400M [68] 9.2 26.8 LLaVA-NeXT [30] CLIP ViT [42] 5.5 4.7

##### C.2. Diffusion Models

As shown in Figure 15 (a), when employing the pre-trained Stable Diffusion [46] with the prompt “an aerial image with cars in Utah”, the model fails to understand the geographical reference to “Utah” and does not generate images reflective of the state’s landscape. Additionally, Stable Diffusion struggles with generating small objects such as cars in aerial images, resulting in low-quality vehicle depictions. ControlNet [69] may also fail to effectively guide the generation direction, leading to outputs that do not align with the provided edge and segmentation map conditions. As shown in Figure 15 (b), DoGE [59] frequently fails to generate cars and surroundings in the precise locations corresponding to the input LINZ image under the guidance of canny edges and semantic segmentation maps. Moreover, without finetuning on UGRC images, DoGE fails to encode the domain difference by modeling the average CLIP [42] image embedding difference and is unable to produce images

Ours Faster R-CNN [45] 63.8 68.2 Ours YOLOv5 [15] 67.2 67.3 Ours YOLOv8 [44] 70.0 76.3 Ours ViTDet [25] 72.0 67.1

Table 3. Comparison between our methods and VLLMs on UGRC dataset. We report the precision and recall metrics.

##### C.1. Open-set Detectors

As shown in Figure 13, Grounding-DINO [31] often detects cars in background images. OmDet-Turbo [72] and OWLv2 [34] often produce false positives by misclassifying objects such as rectangular tanks and boxes, which share visual similarities with cars. OWL-ViT [33] fails to detect

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

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

- Figure 9. Examples of images from our real-world datasets. (first row) LINZ images containing small vehicles; (second row) LINZ images without vehicles; (third row) UGRC images containing small vehicles; (fourth row) UGRC images without vehicles;

| |
|---|

LINZ

| |
|---|

UGRC

|63.2<br><br>88.9<br><br>50<br><br>60<br><br>70<br><br>80<br><br>90<br><br>100<br><br>YOLOv5|
|---|

|62.9<br><br>91.2<br><br>50<br><br>60<br><br>70<br><br>80<br><br>90<br><br>100<br><br>YOLOv8|
|---|

|53.1<br><br>85.7<br><br>50<br><br>60<br><br>70<br><br>80<br><br>90<br><br>100<br><br>Faster R-CNN|
|---|

|55.7<br><br>91<br><br>50<br><br>60<br><br>70<br><br>80<br><br>90<br><br>100<br><br>ViTDet|
|---|

- Figure 10. Comparison between cross-dataset generalization and within-dataset performance. The purple bars represent the model trained on the LINZ dataset and evaluated on the UGRC dataset, while the pink bars correspond to both training and testing conducted on the UGRC dataset. We report the AP50 result.

that accurately reflect Utah’s landscape. Figure 15(c) illustrates that even after fine-tuning GLIGEN [26] on UGRC data, the model frequently fails to comply with the specified bounding box layout conditions. It often generates extraneous cars outside the designated bounding boxes or omits cars within the expected bounding areas. These limitations highlight the challenges associated with ensuring diffusion models faithfully adhere to spatial and semantic constraints in conditioned image generation.

##### C.3. Vision Large Language Models

We evaluate the capabilities of Vision Large Language Models (VLLMs) for car presence classification and center localization (VLLMs), as shown in Figure 11 and Table 3. For classification, BLIP2 [22] generates captions based on the images while Kosmos2 [40] is prompted to complete the sentence “an aerial image of {}”. We consider a model to have predicted the presence of a car in an image if the generated caption includes the word “car”. As shown in Figure 11 (a), only 13.73% of the images predicted to contain cars by BLIP2 are true positives in the LINZ dataset. A similar trend is observed in Figure 11 (b), (c) and (d),

###### Absolute quantities

###### Normalized w.r.t. predicted labels

###### Normalized w.r.t. true labels

|[Figure 161]<br><br>421,827 13,733<br><br>443 2,186|
|---|

|[Figure 162]<br><br>96.85% 3.15%<br><br>16.85% 83.15%|
|---|

|[Figure 163]<br><br>99.90% 86.27%<br><br>0.10% 13.73%|
|---|

non-carimages

(435,560imgs) carimages

Truelabel

(2,629imgs)

non-car images (422,270 imgs)

car images (15,919 imgs)

non-car images (422,270 imgs)

car images (15,919 imgs)

non-car images (422,270 imgs)

car images (15,919 imgs)

Predicted label

Predicted label

Predicted label

###### (a) LINZ - BLIP2 Image Captioning

###### Absolute quantities

###### Normalized w.r.t. predicted labels

###### Normalized w.r.t. true labels

|[Figure 164]<br><br>432,802 2,758<br><br>896 1,733|
|---|

|[Figure 165]<br><br>99.37% 0.63%<br><br>34.08% 65.92%|
|---|

|[Figure 166]<br><br>99.79% 61.41%<br><br>0.21% 38.59%|
|---|

non-carimages

(435,560imgs) carimages

Truelabel

(2,629imgs)

non-car images (433,698 imgs)

car images (4,491 imgs)

non-car images (433,698 imgs)

car images (4,491 imgs)

non-car images (433,698 imgs)

car images (4,491 imgs)

Predicted label

Predicted label

Predicted label

###### (b) LINZ - Kosmos2 Image Captioning

###### Absolute quantities

###### Normalized w.r.t. predicted labels

###### Normalized w.r.t. true labels

|[Figure 167]<br><br>258,826 10,221<br><br>719 791|
|---|

|[Figure 168]<br><br>96.20% 3.80%<br><br>47.62% 52.38%|
|---|

|[Figure 169]<br><br>99.72% 92.82%<br><br>0.28% 7.18%|
|---|

non-carimages

(269,047imgs) carimages

Truelabel

(1,510imgs)

non-car images (259,545 imgs)

car images (11,012 imgs)

non-car images (259,545 imgs)

car images (11,012 imgs)

non-car images (259,545 imgs)

car images (11,012 imgs)

Predicted label

Predicted label

Predicted label

###### (c) UGRC - BLIP2 Image Captioning

###### Absolute quantities

###### Normalized w.r.t. predicted labels

###### Normalized w.r.t. true labels

|[Figure 170]<br><br>265,506 3,541<br><br>800 710|
|---|

|[Figure 171]<br><br>98.68% 1.32%<br><br>52.98% 47.02%|
|---|

|[Figure 172]<br><br>99.70% 83.30%<br><br>0.30% 16.70%|
|---|

non-carimages

(269,047imgs) carimages

Truelabel

(1,510imgs)

non-car images (266,306 imgs)

car images (4,251 imgs)

non-car images (266,306 imgs)

car images (4,251 imgs)

non-car images (266,306 imgs)

car images (4,251 imgs)

Predicted label

Predicted label

Predicted label

(d) UGRC - Kosmos2 Image Captioning

- Figure 11. VLMs captioning capabilities analysis tested on LINZ and UGRC datasets.

where the true positive rates for predicted car images are 38.59%, 7.18%, and 16.70%, respectively. Furthermore, among all images that contain cars, only 52.38% are correctly identified by BLIP2 in the UGRC dataset, as shown in Figure 11 (c). A similar trend is observed in Figure 11 (d), where 47.02% of all images that actually contain cars are correctly predicted by Kosmos2. Figure 12 supports that VLLMs struggle to accurately detect cars in aerial imagery by presenting the F1 score, Precision, and Recall metrics of their classification performance.

###### Accuracy

###### F1 Score

100% 96.76%95.96% 99.17%98.40%

100%

Dataset

80%

LINZ UGRC

| |
|---|

80%

| |
|---|

60%

60%

48.68%

40%

40%

24.65%

23.57%

20%

20%

12.63%

0%

0%

BLIP2 Kosmos2

BLIP2 Kosmos2

###### Recall

###### Precision

100%

100%

83.15%

80%

80%

71.80%

65.92% 52.38%

60%

60%

47.02%

38.59%

40%

40%

16.70%

20%

20%

13.73%

0%

0%

BLIP2 Kosmos2

BLIP2 Kosmos2

Figure 12. Two popular VLLMs (BLIP2 and Kosmos2) tested as zero-shot image car presence classifier. The severe imbalance of the positive and negative classes causes the high levels of accuracy, which is deceptive. F1 score, Precision and Recall metrics clearly show that the classification quality is less than ideal.

|Backbone<br><br>|Task<br><br>|Stage 1 bs lr<br><br>|Stage 2 bs lr<br><br>|Stage 3 bs lr|
|---|---|---|---|---|
|Faster-RCNN Faster-RCNN YOLOv5 YOLOv5 YOLOv8 YOLOv8 ViTDet ViTDet<br><br>|LINZ → UGRC DOTA → UGRC LINZ → UGRC DOTA → UGRC LINZ → UGRC DOTA → UGRC LINZ → UGRC DOTA → UGRC|1024 0.2 1024 0.2<br><br>1600 0.001 2400 0.001 4096 0.001 4096 0.001<br><br>192 0.001<br><br>192 0.0001<br><br>|192 0.02 192 0.001 192 0.0001 192 0.0001 192 0.0001 192 0.0001<br><br>96 0.0001 96 0.001|512 0.2 1024 0.2<br><br>2400 0.001 2400 0.001 1024 0.001 2048 0.001<br><br>192 0.001<br><br>192 0.0001|

Table 4. Training parameters of each stage, where “bs” denotes the batch size and “lr” denotes the base learning rate, which will be scaled during training based on batch size following the MMDetection framework.

For localization, we assess the detection performance of VLLMs as shown in Table 3. Since VLLMs do not provide confidence scores for the predicted bounding boxes, we define detection accuracy as the proportion of predicted bounding boxes that achieve an Intersection over Union (IoU) greater than 0.5 with at least one ground truth bounding box. Based on this definition, we compute precision and recall, which are reported in Table 3. To establish pseudo labels on the UGRC test set for our method, we set the detection threshold according to the highest F1 score achieved by each detector on UGRC test set. Under this setting, VLLMs exhibit significantly lower performance compared to our method, often producing a large number of false positives in aerial imagery. These findings highlight the current limitations of pre-trained VLLMs in accurately detecting and localizing vehicles in aerial imagery.

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

- (a)
- (b)
- (c)
- (d)

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

- Figure 13. Failure cases of Open-set detectors. (a) Detection results of Grounding-DINO. (b) Detection results of Omdet-Turbo. (c) Detection results of OWLV2. (d) Detection results of Owlvit. The blue bounding boxes with dotted lines denote the predicted pseudo bounding box labels while the dots denote the predicted car centers.

[Figure 205]

∆𝑦 ∆𝑥

(a) (b) (c)

[Figure 206]

- Figure 14. Illustration of how we obtain the 42.36 px bounding box size. (a) The black bounding boxes denote the ground truth pseudo bounding box labels while the red bounding boxes denote the predicted pseudo bounding box labels. The dots denote the corresponding centers. ∆x and ∆y denote the x-and y-coordinate difference between the ground truth center and the predicted center. (b) Isocontour of

Intersection of Union (IoU). The yellow arc is 14 of the decision circle with a 12 px radius while the black curve represents the isocontour where IoU = 0.5. (c) An example of a decision circle with a radius of 12 px centered at the car’s center with the corresponding 42.36 px

pseudo-bounding box.

#### D. More Implementation Details

##### D.1. Decision circle and Pseudo-bounding box label

In this section, we provide a detailed explanation for defining the pseudo-bounding box size to be 42.36 px. As illustrated in Figure 14 (b), any point p within the region en-

closed by the isocontour of IoU = α represents a predicted pseudo-bounding box centered at p with an IoU ≥ α relative to the ground truth pseudo-bounding box. Ideally, all true positive predicted centers should be contained within the decision circle. However, no isocontour perfectly fits the arc of the decision circle. To minimize this discrep-

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

(a) Stable Diffusion (b) ControlNet + Domain Gap Embedding (c) GLIGEN

- Figure 15. Failure cases of diffusion models. (a) Images generated by pre-trained Stable Diffusion V1.4. (b) Images generated by pretrained Stable UnCLIP [43] with canny edge maps, semantic segmentation maps, and average CLIP image embedding difference between the LINZ and the UGRC dataset as conditions. From left to right: Real images from LINZ dataset, edge maps, semantic segmentation maps, and synthetic images. (c) Images generated by GLIGEN. Left: Real images from the LINZ dataset. Right: Synthetic UGRC images. The bounding boxes that have the same color in left and right images correspond to the same location.

ancy, we determine the square pseudo-bounding box size a such that the area enclosed by the isocontour at IoU = 0.5 matches 14 of the decision circle. Let ∆x and ∆y denote the x- and y-coordinate differences, respectively, between the centers of the ground truth and predicted pseudo-bounding boxes, as shown in Figure 14 (a). Without loss of generality, we assume the predicted pseudo-bounding box center lies in the first quadrant. The IoU can then represented as IoU = 2a2(−a−(a∆−x∆)(xa)(−a∆−y∆) y). By setting IoU = 0.5, we solve for ∆y in terms of ∆x, treating a as a constant, which can be represented as ∆y = a3((aa−−3∆∆xx)). We then integrate ∆y with respect to ∆x to compute the area under the isocontour of IoU = 0.5, which is a function of a. Finally we equate this integral to 14 of the area of the decision circle and solve for a.

##### D.2. Multi-Stage Training

In this section, we provide more details regarding the training process of the detectors. As outlined in Sec. 3.3, the labeling of synthetic target domain (UGRC) images is conducted in three stages. In the first stage, we train a detector FS on fully annotated real source domain data (LINZ or DOTA) and subsequently generate pseudo labels for the synthetic source domain images. In the second stage, we train another detector FA on the multi-channel

Backbone Ac + Afg Ac + Abg Ac + Afg + Abg

YOLOv5 [15] 63.7 65.5 68.8 YOLOv8 [44] 69.1 73.1 75.4

Table 5. Comparison of different cross-attention map configurations for adaptation from LINZ to UGRC.

cross-attention maps of synthetic source domain images and use it to predict pseudo labels for the multi-channel cross-attention maps of synthetic target domain images. Finally, in the third stage, we train a detector FT on the synthetic target domain images. For Faster-RCNN [45], we use ResNet50 [11] as backbone. For YOLOv5 [15], we utilize the YOLOv5-M variant, while for YOLOv8 [44], we employ the YOLOv8-M variant. For ViTDet [25], we disable the mask head. In all training stages, we scale the image resolution to 128 px × 128 px, as YOLOv5 requires input dimensions to be multiples of 32. The specific training parameters for each stage are provided in Table 4. Except for these adjustments, we adhere to the MMDetection [2] framework for implementation.

#### E. More Ablation Studies

In this section, we present two additional ablation studies to further validate the effectiveness of our proposed pipeline. First, we investigate the impact of stacking different combinations of cross-attention maps. Specifically, we compare our approach with alternative configurations that stack two channels of the object category cross-attention map Ac and one channel from either the learned foreground cross-attention map Afg or background cross-attention map Abg, ensuring compatibility with object detectors that accept only three-channel inputs. As shown in Table 5, integrating both background and foreground information, as in our method, yields the best performance in AP50.

Second, we analyze the effectiveness of our two-stage design, which first fine-tunes Stable Diffusion [46] on both source and target domain datasets, and then introduces learnable tokens to extract cross-attention maps that capture both foreground and background information. We compare this design with a one-stage baseline that jointly finetunes Stable Diffusion and learns tokens simultaneously. The two-stage setup is motivated by the limited localization ability of unseen prompts in pre-trained Stable Diffusion models when applied to aerial view images. For instance, prompts such as “an aerial view image with cars in Utah” often fail to localize vehicles, leading to inaccurate attention and suboptimal token learning. By first fine-tuning the model to better align the concept of “cars” with actual vehicle locations, we enable subsequent token learning to more precisely focus on relevant regions. Experimental results show that the one-stage pipeline yields an 8.5% lower AP50 on YOLOv5 [15] when adapting from LINZ to UGRC.

