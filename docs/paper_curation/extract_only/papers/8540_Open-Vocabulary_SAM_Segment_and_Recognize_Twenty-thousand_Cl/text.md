# arXiv:2401.02955v2[cs.CV]14Sep2024

## Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively

Haobo Yuan1 , Xiangtai Li1 , Chong Zhou1 , Yining Li2 , Kai Chen2 , and Chen Change Loy1

1 S-Lab, Nanyang Technological University 2 Shanghai AI Laboratory Project Page: https://www.mmlab-ntu.com/project/ovsam E-mail: yuanhaobo@whu.edu.cn, xiangtai94@gmail.com

Abstract. The CLIP and Segment Anything Model (SAM) are remarkable vision foundation models (VFMs). SAM excels in segmentation tasks across diverse domains, whereas CLIP is renowned for its zero-shot recognition capabilities. This paper presents an in-depth exploration of integrating these two models into a unified framework. Specifically, we introduce the Open-Vocabulary SAM, a SAM-inspired model designed for simultaneous interactive segmentation and recognition, leveraging two unique knowledge transfer modules: SAM2CLIP and CLIP2SAM. The former adapts SAM’s knowledge into the CLIP via distillation and learnable transformer adapters, while the latter transfers CLIP knowledge into SAM, enhancing its recognition capabilities. Extensive experiments on various datasets and detectors show the effectiveness of Open-Vocabulary SAM in both segmentation and recognition tasks, significantly outperforming the naïve baselines of simply combining SAM and CLIP. Furthermore, aided with image classification data training, our method can segment and recognize approximately 22,000 classes.

Keywords: Scene Understanding · Promptable Segmentation

### 1 Introduction

The Segment Anything Model (SAM) [30] and CLIP [53] have made significant strides in various vision tasks, showcasing remarkable generalization capabilities in segmentation and recognition, respectively. SAM, in particular, has been trained with a massive dataset of mask labels, making it highly adaptable to a wide range of downstream tasks through interactive prompts. On the other hand, CLIP’s training with billions of text-image pairs has given it an unprecedented ability in zero-shot visual recognition. This has led to numerous studies [18,64,72,78] exploring the extension of CLIP to open vocabulary tasks, such as detection and segmentation.

While SAM and CLIP offer considerable advantages, they also have inherent limitations in their original designs. SAM, for instance, lacks the capability to recognize the segments it identifies. Efforts to overcome this by integrating a

Point Prompt

SAM OV-SAM Box Prompt SAM OV-SAM

Image-Crop Baseline Feature-Crop Baseline

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Open-Vocabulary SAM

Accuracy

OV-SAM (Ours) 1180, 84.3

| |
|---|

Pen

Cat

3x Efficiency +29.2 Accuracy

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

| |
|---|

SAM + CLIP 3545, 55.1

GFLOPs

Vermillion Flycatcher

Bus

Accuracy

|[Figure 13]|
|---|

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

84.3

55.1

46.2

Beagling

Cyanobacteria

SAM + CLIP OV-SAM

- Fig. 1: Open-Vocabulary SAM not only can segment anything with prompts just like SAM but also has the capability of recognition in the real world, like CLIP. With drastically lower computational cost, Open-Vocabulary SAM has a higher recognition performance than directly combining SAM and CLIP with image or feature cropping (measured on the COCO open vocabulary benchmark).

classification head have been made [32,84], but these solutions are constrained to specific datasets or closed-set settings. On the other hand, CLIP, which is trained using image-level contrastive losses, faces challenges in adapting its representations for dense prediction tasks. To address this, several studies [18,57,66,67,76] have investigated ways to align CLIP’s representation for dense predictions. However, these approaches tend to be dataset-specific and not universally applicable. For example, some research has focused on open vocabulary segmentation on the ADE-20k [85] dataset, using the COCO [43] dataset for pre-training. Merging SAM and CLIP in a naïve manner, as illustrated in Fig. 2 (a) and (b), proves to be inefficient. This approach incurs substantial computational expenses and yields subpar results, including recognition of small-scale objects, as evidenced by our experimental results.

In this study, we address these challenges with a unified encoder-decoder framework that integrates a CLIP encoder and a SAM decoder, as depicted in

- Fig. 2 (c). To bridge these two distinct components effectively, we introduce two novel modules, SAM2CLIP and CLIP2SAM, facilitating dual knowledge transfer. First, we distill knowledge from the SAM encoder to the CLIP encoder using SAM2CLIP. This distillation process is uniquely executed not directly on the CLIP encoder, which is kept frozen to maintain its existing knowledge, but rather on a lightweight transformer-like adapter using a pixel-wise distillation loss. The adapter takes multi-scale features as input, with the goal of aligning CLIP features with SAM representation. On the decoding side, the CLIP2SAM module transfers knowledge from the frozen CLIP encoder to the SAM decoder. In particular, we design a feature pyramid adapter with a RoIAlign operator to be jointly trained with the SAM decoder. Both modules are lightweight and naturally combine the strengths of CLIP and SAM.

##### Following the spirit of SAM, we enhance our model’s recognition capabilities by harnessing the power of established semantic datasets, including COCO [43], LVIS [19], and ImageNet-22k [11]. This strategy elevates our model to the ver-

prompt

prompt

prompt

SAM Decoder

SAM Decoder

SAM Decoder

mask

mask

mask

SAM Encoder

image

SAM Encoder

SAM Encoder

image

image

[Figure 19]

[Figure 20]

SAM2CLIP

crop

Adapters

CLIP

Adapters

CLIP

CLIP

label

Visual Encoder

Visual Encoder

label

Visual Encoder

label crop

Adapters

text

text

[Figure 21]

[Figure 22]

CLIP2SAM

text

(a) Image Cropping Baseline

(b) Feature Cropping Baseline

(c) Open-Vocabulary SAM

- Fig. 2: Comparison of two simple SAM-CLIP combination baselines (a) and (b), and our proposed single encoder architecture (c). The adapters for (a) and (b) are optional and can be replaced with various designs (please refer to Sec. 4.1 for details). Note that, in our method, the SAM encoder will be discarded during inference.

satility of SAM, endowing it with enhanced capability to segment and recognize any objects, as shown in Fig. 1. As our approach is an adaptation of SAM, it is flexible enough to be integrated with various detectors, making it suitable for both closed-set and open-set environments.

We conduct extensive experiments across a range of datasets and scenarios, encompassing closed-set and open-vocabulary interactive segmentation. Notably, when compared to basic combined baselines, our approach demonstrates superior performance, achieving over 2% improvement in IoU and 3% in mAP with various detectors on the COCO dataset. In particular, in the case of recognition on LVIS, our approach achieves over 20% improvements over previous adapters. Furthermore, by expanding our approach with a more diverse array of datasets, we have developed a versatile, interactive tool suitable for practical applications. For detailed results, we direct the reader to Sec. 4 and the appendix.

### 2 Related Work

Vision Language Models (VLMs). Vision-language pre-training has given rise to models with aligned image and text representations [25, 26, 28, 41, 53]. Recent studies on contrastive vision-language pre-training [26, 53, 58, 80] have significantly improved the generalization ability of recognition models. Meanwhile, several works [28, 33–35] aim to design better optimization goals for downstream multi-modal tasks, including caption and visual question answering. Among these works, CLIP models [53] that are pre-trained on billion-scale image-text pairs have shown impressive zero-shot classification performance on a wide range of datasets. Our goal is to enable SAM to perform recognition tasks with the help of pre-trained VLMs.

Open Vocabulary Dense Prediction. This direction aims to recognize region visual concepts of arbitrary categories described by texts, which includes object detection [18,63,64,69,79], semantic segmentation [37,38,40,71,87,88], and panoptic segmentation [70,75,76]. This necessitates the alignment between region and text representations with the help of VLMs [26,53,58]. For open-vocabulary

detection, a series of works [18,57,66,78] distill knowledge from the CLIP models to recognize novel objects. In contrast to distillation-based methods, several works [31,68] directly build object detectors upon frozen CLIP CNNs. For open-vocabulary segmentation, the typical works [12, 70, 72, 76] first generate class-agnostic mask proposals and then classify the proposals with CLIP. Recently, several works [70,76] build the mask generator upon the frozen diffusion model [54] and CLIP model [53]. Meanwhile, several studies [27,47,51,52] focus on class-agnostic segmentation and detection to enrich generalization ability in various domains. However, most approaches are trained and tested on specific datasets. Our approach is based on SAM, which provides a general, interactive tool to support different open vocabulary detectors.

Prompting in Computer Vision. Prompting, originating from in-context learning in natural language processing (NLP) as seen in works like Brown et al. [4] and Rubin et al. [55], leverages a large language model to infer unseen tasks through context-specific input-output pairs. Recent studies [1,3,16,45,61,62,89] have explored in-context learning for visual tasks. Common techniques involve mask image modeling [2, 21, 77] for cross-task visual prompting, as employed by approaches like Painter [61] and Bar et al. [3]. SAM [30] demonstrates incontext learning through interactive segmentation, using diverse visual prompts like points, boxes, and masks, although it is limited to class-agnostic mask prediction. Meanwhile, other studies [8,17,23,39,42] have concentrated on efficient parameter tuning of visual foundation models, typically focusing on a single model. Our work uniquely bridges two models, CLIP and SAM, exploring their combined potential for enhanced general segmentation and recognition capabilities. In particular, we adopt visual prompts (box, point) as the model’s inputs. Segmentation Anything Model. SAM [30] presents a new data engine and portable model for segmentation. Subsequent research has employed SAM as an interactive segmentation tool for various vision tasks, including grounding [44], tracking [10], distillation [81, 86], medical analysis [5, 65], and generation [82]. While some studies use SAM and CLIP for segmentation [6,20,36,49,59], none have yet integrated VLMs and SAM into a unified model capable of both segmentation and recognition of novel classes. Our work makes the first attempt to merge the capabilities of VLMs with SAM for enhanced task versatility.

### 3 Methodology

We first review the SAM, CLIP, and combined baselines in Sec. 3.1. Then, we detail our Open Vocabulary SAM in Sec. 3.2. Last, we present our model’s training details and application in Sec. 3.3.

#### 3.1 Preliminaries and Baselines

SAM. SAM is a prompt-driven segmentor. It contains an image encoder, a prompt encoder, and a light-weight mask decoder. Here, we use box prompts as an example. We denote an input image as X ∈ RH×W×3 and input visual

SAM2CLIP

training only

Trainable

[Figure 23]

[Figure 24]

[Figure 25]

SAM Encoder

[Figure 26]

Distillation Loss

SAM2CLIP

CLIP Encoder SAM Encoder

Neck

[Figure 27]

CLIP Encoder

CLIP2SAM

Language Embeddings

| | | |
|---|---|---|
| | | |

[Figure 28]

CLIP2SAM

[Figure 29]

Pen

[Figure 30]

Prompt Encoder

SAM

Decoder Cat

RoIAlign

Classification Loss

CLIP Encoder

FPN

- Fig. 3: Illustration of Open-Vocabulary SAM. For training, the SAM encoder is as a teacher network, while SAM2CLIP plays the role of a student network and aligns the knowledge of SAM into CLIP. The CLIP2SAM transfers the CLIP knowledge to the SAM decoder and performs joint segmentation and classification for close-set and open vocabulary settings.

prompts as P ∈ RN×4, where H ×W are the spatial size, N is the number of box prompts. The image encoder is a modified vision transformer (ViT). It encodes an image into dense feature FSAM ∈ R16H ×W16×d. The prompt encoder encodes P into sparse prompts Qsp. Meanwhile, mask tokens Qmask and an IoU token QIoU are initialized for the mask decoder.

The mask decoder takes the image feature F, sparse prompts Qsp, mask tokens Qmask, and the IoU token QIoU as input. All the inputs will be concatenated and encoded with a lightweight two-way transformer. Consequently, each mask token is transformed into a dynamic linear classifier, capable of calculating the foreground mask probability for every sparse prompt. Simultaneously, the IoU token is tasked with predicting the confidence score for each mask. Considering the multi-granular nature of SAM’s data annotations, encompassing both instance and part level, Qmask naturally encodes multi-granularity. Our study concentrates exclusively on the object level, which aligns more closely with prevalent real-world applications and datasets such as COCO [43] and LVIS [19]. CLIP. Given an input image X and a corresponding caption C, the CLIP framework processes these modalities to produce respective embeddings: the image embedding EI, derived from its image encoder, and the text embedding t, obtained from its text encoder. In the context of open-vocabulary object detection and segmentation, CLIP’s capability to generalize beyond fixed class labels is leveraged to replace traditional classifiers. For instance, in open-vocabulary detection scenarios, the text embedding tc for the c-th object category is generated by inputting the category name into the CLIP text encoder. This process can employ a single template prompt, such as "a photo of {category}," or multiple prompt templates. Subsequently, for a given region embedding r, that is produced by the RoI-Align [22], the classification score for the c-th category is computed as: pc = exp(τ·<r,t

c>) C

i=0 exp(τ·<r,ti>), where < ·,· > denotes the cosine similarity, and τ is a learnable or fixed temperature to re-scale the value.

Combined Baselines. We introduce two different baselines for combining CLIP and SAM, as depicted in Fig. 2 (a) and (b). The first approach, termed the

‘cropped image baseline’, employs the SAM mask decoder’s output to segment and resize the original input image. This processed image then serves as the input for the CLIP image encoder, and, in conjunction with the CLIP text embedding, the mask is classified using the similarities between visual and text embeddings. The second approach referred to as the ‘cropped CLIP image feature baseline’, employs the same initial CLIP feature extraction step. However, in this method, masks predicted by the SAM decoder are used to crop the CLIP image features. Subsequent pooling of these masked features yields the final label, akin to baseline (a).

While both baselines enable zero-shot inference of images, they exhibit a noticeable knowledge gap on specific datasets. To address this, we draw inspiration from recent advancements in visual prompting or adapters [8,89]. Specifically, we propose incorporating additional learnable tokens as an adapter to fine-tune the model for enhanced performance on downstream datasets. These zero-shot inference capabilities and the fine-tuned models constitute our primary comparison baselines under various experimental conditions, detailed in Sec. 4.1.

#### 3.2 Open Vocabulary SAM

While both baseline models can be enhanced through visual prompting or adapters, as we will discuss in Sec. 4, they face several challenges in real-world applications. First, the requirement for two independent backbones in the combined model increases computational costs (Prob.1). Second, SAM and CLIP are trained with distinct objectives – SAM through supervised learning and CLIP via contrastive learning – and there is limited research on knowledge transfer between such diverse architectures (Prob.2). Third, despite adapter integration, significant performance gaps remain in recognizing small objects (Prob.3). Fourth, there is a lack of exploration into integrating open-vocabulary capabilities for SAM and CLIP, particularly in the context of feature fusion and data scaling (Prob.4). Our work aims to solve these problems in a unified yet effective framework.

Unified Architecture. We design a unified architecture for both segmentation and recognition to address Prob.1. Specifically, we adopt the frozen CLIP visual encoder as our feature extractor. Then, both SAM’s mask decoder and prompt encoder are appended behind the CLIP encoder. The meta-architecture of openvocabulary SAM is shown in Fig. 2 (c), with the more detailed version shown in

- Fig. 3. This unified architecture is made possible via the SAM2CLIP, which transfers knowledge of SAM to CLIP with distillation, and CLIP2SAM, which employs CLIP knowledge and combines the SAM mask decoder for recognition. We have chosen convolution-based visual backbones for the frozen CLIP backbone, aligning with previous studies that have highlighted their superiority in capturing spatial structures [31,73]. The efficacy of different CLIP backbones is further explored in Sec. 4.2. SAM2CLIP. To resolve Prob.2, we design the SAM2CLIP module that bridges the gap in feature representations learned by SAM and CLIP, using adaptation and distillation methods. Through comprehensive experiments, we discovered that employing distillation loss Ldistill along with transformer-based

adapters [14], yields effective results. Specifically, the distillation process involves a simple pixel-wise approach, where SAM-Huge serves as the teacher, and the frozen CLIP equipped with an adapter assumes the student’s role. We then implement a per-pixel mean squared error (MSE) loss to align the SAM feature Fsam with the CLIP feature EI, as detailed below:

Ldistill = MSE(Fsam,EI). (1)

We design a multi-scale adapter Asam2clip to align the features from CLIP and SAM. In particular, we take pyramid CLIP features EIi,i = 1,2,3 as the inputs. Such pyramid features contain both high-resolution and semantic information, which is proven crucial for semantic segmentation [29]. The MSE loss is revised as follows:

Ldistill = MSE(Fsam,Asam2clip(Fusion(EIi))), (2)

where Asam2clip comprises several transformer layers, and Fusion is achieved by bilinear upsampling and addition.

With SAM2CLIP, we can even achieve comparable segmentation results with the SAM-Huge with much lower computational costs. As detailed in Sec. 4.2, we observe that employing convolution-based methods specifically designed for backbone adaptation [8, 15] results in sub-optimal outcomes. The reason for this might be in the inherent architecture of the SAM encoder, which is purely based on ViT. A symmetrical structure is crucial for effective knowledge transfer. With the implementation of SAM2CLIP, we can achieve segmentation results comparable to those of SAM-Huge across various detectors, while significantly reducing computational costs.

CLIP2SAM. This module aims to leverage CLIP’s knowledge to enhance the recognition capabilities of the SAM decoder. A straightforward approach involves appending a label token Qlabel to the existing mask token Qmask and IoU token QIoU. Using Qlabel, we introduce a specialized adapter to facilitate the transfer of knowledge from the frozen CLIP to the SAM decoder. Subsequently, the enhanced Qlabel, combined with the output of the prompt encoder and adapted CLIP features, is fed into a two-way transformer. Following the cross-attention process, the improved Qlabel undergoes further refinement through a multilayer perceptron (MLP), ensuring better alignment with CLIP’s text embedding. The final labels are derived by calculating the distance between the refined label token and the CLIP text embedding.

This design, however, falls short of recognizing small objects (Prob.3) since the adaptation only involves the single-scale feature, which is mainly focused on segmentation. We present a simple yet effective solution to handle this issue, introducing a lightweight feature pyramid network (FPN) for CLIP2SAM adaption. As shown in Fig. 3, the pyramid network extracts multi-scale CLIP features as the inputs. Then, we apply the RoI-Align [22] operation to extract region features. Like the R-CNN framework [22], we apply one convolution layer and a MLP to learn the feature embedding without introducing cross-attention

in the mask decoder. In particular, for point prompts, we first obtain the corresponding masks via the SAM decoder and obtain the box via the corresponding masks. For box prompts, we can directly send it to the FPN for region feature extraction. Given that our method incorporates only a few convolution layers, it does not significantly increase computational costs compared to the original SAM.

Open Vocabulary. To tackle Prob.4, the open-vocabulary challenge, we leverage the knowledge embedded in the frozen CLIP backbone, which aids in recognizing novel and unseen objects during inference. In line with previous studies [31,67], we fuse the learned class scores with those from the frozen CLIP via a geometric mean to leverage information from both the CLIP and CLIP2SAM. Additionally, we investigate various strategies to expand the vocabulary size, such as joint training with multiple datasets, as detailed in Sec. 4.2. Our experimental results show that the model scales effectively with large datasets.

While our approach can address the open vocabulary challenge, it is important to distinguish the setting of Open-Vocabulary SAM from that of previous open vocabulary segmentation methods [6,13,24,74]. Unlike previous techniques, which typically depend on a separate segmentor to produce mask proposals, our method only uses visual prompts, such as boxes and points, to generate masks and labels. Furthermore, our method specifically targets foreground objects, aligning with the ImageNet [11] datasets and the purpose of box prompts. To show the effectiveness of our method, we compare the performance of our method with previous open vocabulary segmentation methods in Sec. 4.1.

#### 3.3 Training and Application

Training and Loss Function. We first use the SA-1B (1%) dataset [30] for training the SAM2CLIP module to transfer SAM’s knowledge into openvocabulary SAM, with the loss Ldistill (Equ. (2)). Then, we joint train the CLIP2SAM and mask decoder using segmentation mask and label annotations from COCO or LVIS. The final loss function is given as L = λclsLt_cls + λceLt_ce + λdiceLt_dice. Here, Lt_ce is the Cross-Entropy (CE) loss for mask classification, and Lt_ce and Lt_dice are mask Cross Entropy (CE) loss and Dice loss [48] for segmentation, respectively. In addition, we adopt joint training with the ImageNet dataset for classifying over 22,000 categories.

Inference and Demo Tools. Our model performs inference like SAM, with points and boxes as visual prompts. Specifically, we test boxes and points as visual prompts for the encoder in Sec. 4. In the appendix, we show a demo of our model, which can segment and recognize with prompts.

### 4 Experiments

Datasets and Metrics. We mainly use COCO [43] and LVIS [19] datasets for the experiments. Moreover, we also use part of SA-1B [30] data (1%) for

- Table 1: Comparison of combined baselines and Open-Vocabulary SAM using visual prompts. “*” indicates using mask center point as prompts, while others indicate using

ground truth boxes prompts. IoUb and IoUn refer to the average IoU for each mask of base classes and novel classes, respectively.

COCO LVIS

Method

FLOPs #Param

IoUb IoUn Acc IoUb IoUn Acc

|Image-Crop baseline Feature-Crop baseline<br><br>|78.1 81.4 46.2 78.1 81.4 55.1<br><br>|78.3 81.6 9.6 78.3 81.6 26.5|3,748G 808M 3,545G 808M<br><br>|
|---|---|---|---|
|Image-Crop baseline + CoOp [89] Feature-Crop baseline + CoOp [89]|79.6 82.1 62.0 79.6 82.1 70.9<br><br>|80.1 82.0 32.1 80.1 82.0 48.2<br><br>|3,748G 808M 3,545G 808M<br><br>|
|Open-Vocabulary SAM<br><br>|81.5 84.0 84.3|80.4 83.1 66.6<br><br>|1,180G 304M|
|Image-Crop baseline* Feature-Crop baseline*|60.7 66.7 24.5 60.7 66.7 32.1<br><br>|53.0 62.3 6.2 53.0 62.3 11.0|3,748G 808M 3,545G 808M<br><br>|
|Image-Crop baseline + CoOp [89]* Feature-Crop baseline + CoOp [89]*|64.7 66.7 28.2 64.7 66.7 35.1<br><br>|58.9 64.2 8.3<br><br>58.9 64.2 13.2<br><br>|3,748G 808M 3,545G 808M|
|Open-Vocabulary SAM*<br><br>|68.4 65.2 76.7|63.6 67.9 60.4<br><br>|1,180G 304M|

SAM2CLIP knowledge transfer. For COCO, we report the results of both closeset and open-vocabulary settings for the instance segmentation task. In particular, following Zareian et al. [79], we split 48 base classes with annotations and 17 target classes without annotations. We use the base class annotations for training. For LVIS datasets, we adopt the open-vocabulary setting and report the results of APrare for novel classes. For evaluation metrics, we report the accuracy of recognition for reference to evaluate the recognition capability. Meanwhile, each prompt’s intersection-over-union (IoU) with its ground truth mask is also adopted to evaluate the segmentation ability of our method. As mentioned in Sec. 3.2, different from previous open vocabulary segmentation tasks, boxes or points serve as visual prompts in our method.

Baselines. As shown in Fig. 2 (a) and (b), based on different adapter designs, we append these adapters to the different locations of the combined models. For example, when using CoOp [89], we append the learnable tokens by combining them with CLIP features. For several convolution-based adapters [8], we add the extra convolution layers along with SAM or CLIP backbone for fair comparison. By default, we adopt SAM-huge and CLIP R50x16.

Implementation Details. We implement our models in PyTorch [50] with MMDetection [7]. We use 8 A100 GPUs for distributed training. Each minibatch has two images per GPU. The optimizer is AdamW [46] with a weight decay of 0.0001. We adopt full image size for a random crop in the pre-training and training process following Cheng et al. [9]. All the class names are transferred into CLIP text embedding, following previous works [18]. We train each model for 12 epochs for fair comparison. Due to the limitation of computation costs, we do not adopt joint SAM data and COCO data training. We first perform training the SAM2CLIP on SA-1B (1%), and then we finetune the model on COCO or LVIS data. Please refer to the supplementary material for more details.

###### Table 2: Comparison of combined baselines and Open-Vocabulary SAM on prompts generated by the open vocabulary detector. For the LVIS dataset, only ‘normal’ and ‘frequent’ classes are in the training set. The labels are generated by each baseline or our method. We adopt Detic [90] as the OV-Detector to provide box prompts.

COCO LVIS

Method

FLOPs #Params

APbase APnovel AP APrare APnorm APfreq AP

Image-Crop baseline + CoOp [89] 26.2 31.2 27.3 19.8 18.3 16.3 17.2 3,748G 808M Feature-Crop baseline + CoOp [89] 28.0 33.8 29.5 24.2 21.4 18.6 20.8 3,545G 808M

Open-Vocabulary SAM 31.1 36.0 32.4 24.0 21.3 22.9 22.4 1,180G 304M

###### Table 3: Comparison of Open-Vocabulary SAM with previous methods on open vocabulary instance segmentation. The results presented in the table all use the MaskRCNN [22] with ResNet-50 backbone. Different from previous works, Open-Vocabulary SAM uses the bounding box as the prompt to generate the mask. We report mask AP50 following base-novel setting as [24].

Constrained Generalized

Method Venue

Base Novel Base Novel All

|XPM [24] MaskCLIP [13] MasQCLIP [74]<br><br>|CVPR’22 ICML’23 ICCV’23|42.4 24.0 42.8 23.2 40.9 30.1<br><br>|41.5 21.6 36.3<br>42.6 21.7 37.2 40.7 28.4 37.5<br>|
|---|---|---|---|
|Open-Vocabulary SAM<br><br>|(Ours)|41.7 37.5<br><br>|39.3 39.8 39.4|

###### Table 4: Comparison of mask quality with various detectors on COCO dataset. We report the mask mean AP for comparison. The masks are generated by each method, while the labels are from the corresponding detectors.

Method Detectors mAP AP50 AP75 APS APM APL #Params FLOPs

|SAM-Huge Faster-RCNN (R50) SAM-Huge (finetuned) Faster-RCNN (R50) Open-Vocabulary SAM Faster-RCNN (R50)<br><br>|35.6 54.9 38.4 35.8 55.0 38.4 35.8 55.6 38.3<br><br>|17.2 39.1 51.4 16.5 38.6 53.0 16.0 38.9 53.1|641M 3,001G 641M 3,001G 304M 1,180G<br><br>|
|---|---|---|---|
|SAM-Huge Detic (swin-base) SAM-Huge (finetuned) Detic (swin-base) Open-Vocabulary SAM Detic (swin-base)|36.4 57.1 39.4 36.8 57.4 39.8 36.7 57.2 39.7<br><br>|21.4 40.8 54.6 20.8 40.6 55.1 20.7 40.8 54.9|641M 3,001G 641M 3,001G 304M 1,180G<br><br>|
|SAM-Huge ViTDet (Huge) SAM-Huge (finetuned) ViTDet (Huge) Open-Vocabulary SAM ViTDet (Huge)<br><br>|46.3 72.0 49.8 46.5 72.3 50.3 48.8 73.8 52.9|25.2 45.5 59.6 25.2 45.8 60.1 24.8 46.3 64.2<br><br>|641M 3,001G 641M 3,001G 304M 1,180G|

Benchmark Setup. Open Vocabulary SAM is different from both SAM [30] and open vocabulary segmentation methods [13,24,74]. Compared with OpenVocabulary SAM, SAM [30] cannot recognize segmented objects while open vocabulary segmentation methods [13, 24, 74] usually rely on a separate segmentor (e.g., Mask-RCNN [22]) instead of prompts like points or boxes. For a complete evaluation of Open-Vocabulary SAM, we set the benchmark as three parts, including 1).comparison with combined baselines to verify the effectiveness; 2).comparison with open-vocabulary segmentation methods to evaluate the recognition; 3).comparision with SAM models to evaluate segmentation.

###### Table 5: Scaling up with large-scale datasets.

Datasets Accuracy #vocaulary #images LVIS 83.1 1,203 99K V3Det 78.7 13,204 183K I-21k 44.5 19,167 13M V3Det + LVIS 82.7 13,844 282K V3Det + LVIS + I-21k 83.3 25,898 13M V3Det + LVIS + I-21k + Object365 83.0 25,970 15M

- Table 6: The effectiveness of each component. We use Detic [90] as the detector. The labels are generated by the corresponding model. S2C and C2S denote SAM2CLIP and CLIP2SAM respectively. The baseline refers to the image-crop variant.

Setting AP APbase APnovel FLOPs (G) #Params (M)

|Baseline + CoOp [89]|29.5 28.0 33.8 3,545 808|
|---|---|
|Our + S2C Our + S2C + C2S|28.7 27.3 33.3 1,127 291 34.4 33.1 38.0 1,180 304<br><br>|

#### 4.1 Main Results

Comparison with Combined Baselines Using Ground Truth. To avoid the influence of other modules, we first demonstrate the recognition ability of our model in Tab. 1. Compared to the simple combined approaches, adding adapters with joint co-training leads to better results. However, the recognition ability is still limited on both COCO and LVIS. Our Open-Vocabulary SAM achieves the best results on both boxes and points as visual prompts. We observe more significant gains on LVIS datasets. We argue that LVIS contains more small objects, which is more challenging than COCO. Our method can solve Prob.2 and lead to over 20% accuracy improvement. Although the segmentation quality is pretty good (about 80 IoU on COCO and LVIS with box prompt), our method still achieves 2% IoU improvements. This indicates the effectiveness of our joint co-training on mask prediction and classification. Compared with boxes as prompts, using points as prompts is more challenging since the location clues of points are much weaker than boxes. However, our approach is still better than combined baselines or them with adapters.

Comparison with Combined Baselines on OV-Detector. In Tab. 2, we adopt a more challenging setting by using the box prediction from the existing open-vocabulary detector to simulate the interactive segmentation process with deviation. We choose the representative Detic [90] as the open-vocabulary detector. Again, our method also achieves the best performance on both COCO and LVIS datasets. In particular, on COCO, compared with previous works [89], our method achieves 3.0 mask mAP improvements with much lower costs.

Comparison with Open-vocabulary Segmentation. In Tab. 3, we compare our method with previous open-vocabulary instance segmentation methods. Open-Vocabulary SAM uses the boxes generated by Mask- RCNN [22] and

###### Table 7: Ablation studies on COCO open-vocabulary dataset. We use boxes of the ground truth as a prompt to generate masks and labels. (left): Ablation on SAM2CLIP design. (right): Ablation on CLIP2SAM design.

Setting IoU Acc APbase APnovel

Setting IoU FLOPs (G) #Params (M)

SAM 78.7 3,001 641 Conv-L + MultiScale Neck 78.3 1,313 321 Conv-L + SingleScale Neck 73.6 1,280 307 R50x16 + MultiScale Neck 78.1 1,180 304

|SAM2CLIP (baseline)|78.1 54.2 27.6 33.2<br><br>|
|---|---|
|+ Cls Token<br><br>+ Cls Token & CLIP MLP fusion<br><br>+ light FPN (Ours)|81.3 79.3 29.8 34.5<br><br>80.9 78.9 29.0 33.9<br><br>81.5 84.3 31.1 36.0<br>|

R50 + MultiScale Neck 77.3 728 165

###### Table 8: (left): Ablation study on different CLIP backbone. We test results on the COCO open-vocabulary dataset. We use boxes of the ground truth as a prompt to generate masks and labels. (right): Comparison with other SAM models.

Backbone IoU Acc #FLOPs(G) #Params (M)

Method 1-IoU (COCO) cls. open.

RN50 77.3 50.8 728 165 RN50x16 78.1 55.1 1,180 304 RN50x64 78.1 54.1 2,098 568

SAM [30] (H) 78.2 - SEEM [91] (T) 73.7 ✓ Semantic-SAM [32] (T) 76.1 ✓ OV-SAM (ours) 81.7 ✓ ✓

ConvNeXt-L 78.3 59.1 1,313 321 ViT-L-14 38.6 14.3 2,294 441

previous works use the masks of Mask-RCNN. The results show that our OpenVocabulary SAM performs strongly, especially in the novel classes.

Comparison with SAM on Various Detectors. In Tab. 4, we test the mask prediction quality of our model and original SAM on different detectors. Our method performs better than the original SAM and performs comparably with fine-tuned SAM. It is worth noting that our Open-Vocabulary SAM has much lower computational costs and parameters than SAM.

Comparison on Interactive Segmentation. In Tab. 8 (right), we compare interactive segmentation performance. Notably, our approach excels beyond interactive segmentation and can recognize classes in an open-vocabulary setting. Visualization Comparison. In Fig. 4, we compare our approach with the feature-crop baseline. Our model shows better performance in classifying small and rare object classification and handling occlusion scenarios.

Model as a Zero Shot Annotation Tool. In addition to COCO and LVIS standard datasets training, following the spirit of SAM, we also scale up our model by training it with more data (Tab. 5). In particular, we adopt more detection data (V3Det [60], Object365 [56]) and classification data (ImageNet22k [11]). Owing to significant costs, we have not conducted comparisons with other baselines for this setting. Rather, we have adapted our method into an interactive annotation tool capable of segmenting and recognizing over 22,000 classes.

#### 4.2 Ablation Studies and Analysis

Effectiveness of SAM2CLIP and CLIP2SAM. We first verify the effectiveness of our proposed two modules in Tab. 6. We adopt image-crop variant of the baseline for comparison. In particular, by sharing a single backbone, we observe

###### Input SAM + CLIP OV SAM

[Figure 31]

[Figure 32]

[Figure 33]

Ski Boot

Ski Pole

| |
|---|

[Figure 34]

[Figure 35]

[Figure 36]

Turtleneck

Scarf

[Figure 37]

[Figure 38]

[Figure 39]

| |
|---|

Television Set

Cat

- Fig. 4: Visualization Comparison. We compare the mask and classification results of the image-crop baseline (SAM + CLIP) and Open-Vocabulary SAM (OV SAM). The predicted labels are presented on the mask.

a significant drop in the number of parameters and FLOPs, with a little drop in segmentation performance. The slight drop is caused by the domain gap between SAM data and COCO data during the training of the SAM2CLIP module. However, after adding our CLIP2SAM module and joint co-training with mask classification and prediction, a significant improvement in both segmentation and classification is observed, with just a negligible increase in compute cost.

Detailed Design on SAM2CLIP. In Tab. 7, we explore the detailed design of SAM2CLIP in the first stage of open-vocabulary SAM training. The results show that distillation benefits most when multi-scale features are adopted, suggesting that both high-resolution features and high-level semantics are important to align CLIP’s feature with the SAM’s feature.

Detailed Design on CLIP2SAM. In Tab. 7, we present extensive design for the CLIP2SAM module. We compare two designs: a simple classification token with cross attention (Cls Token) and a combination of this token with mask pooled CLIP feature (CLS Token & CLIP MLP fusion). These designs work better than the combined baseline shown in the first row. Nonetheless, due to resolution constraints, these variants cannot handle small objects well, as shown in Fig. 4. In contrast, our design improves the performance considerably.

Ablation on Different CLIP Backbones. In Tab. 8, we explore the effect of frozen CLIP visual backbone. We do not add the CLIP2SAM. Motivated by recent works [31,67,73,76], CNN-based CLIPs encapsulate more structural information, which is good for our goal since we have location-sensitive visual prompts. Thus, adopt CNN-based CLIPs, aligned with previous works. As shown

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Faucet Ski Bird Blanket Taxi

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Sheep Vase Sheep Toilet Doughnut

[Figure 50]

[Figure 51]

[Figure 52]

- Fig. 5: Qualitative results of Open-Vocabulary SAM. In the visualization, boxes refer to box prompts (up), and the red stars refer to the point prompts (middle). The masks can be merged into a segmentation map (bottom). We show the mask and labels generated by the proposed Open-Vocabulary SAM. Our method can segment and recognize open vocabulary objects in diverse scenes identified by prompts.

in the table, we find ConvNext large achieves the best performance, but we choose RN50x16 since it has comparable performance and better efficiency.

### 5 Conclusion

We present Open Vocabulary SAM, a SAM-inspired method for interactive segmentation and recognition. Unlike previous open-vocabulary detection and segmentation methods, our method explores interactive open-vocabulary segmentation for the first time. Given the user’s inputs, such as boxes or points, the proposed approach can interactively segment and label each visual prompt. Compared with the combined baselines and various visual adapters, our proposed CLIP2SAM and SAM2CLIP are both efficient and effective in various settings. Our open vocabulary segmentation is compatible with different detectors, including open-vocabulary detectors and close-set detectors. With more data, our model plays a similar role as SAM, offering an effective annotation tool for both segmentation and instance labeling. In particular, our method can perform large vocabulary segmentation and recognition over 22K classes. We hope our OpenVocabulary SAM can provide a solid baseline for combining the strengths of different forms of vision foundation models and inspire further research.

## Appendix

Overview. In the supplementary, we provide more details and results to support our main paper. The contents are presented as follows: we first present more details and discussions of our method in Sec. S1. Then, we report more results in Sec. S2. Finally, we present a demo and discuss future work in Sec. S3. Please refer to our GitHub repo at https://github.com/HarborYuan/ovsam for more details about the demo and implementation.

### S1 Method Discussion

#### S1.1 Comparison with Recent Joint SAM and CLIP Models

Several recent works [6,20,59,83] also explore joint segmentation and recognition as one system. Recognize Anything [83] adopts a paradigm for image tagging. The focus of that work is to build large-scale tagging datasets via automatic text semantic parsing. The SAM model is only an external module. Therefore, it cannot perform well with box or mask-level recognition [83]. SAM-CLIP [59] integrates multi-task learning, continual learning techniques, and teacher-student distillation, where its goal is to build a single model capable of segmentation and classification. However, its classification capability is limited to the image level since it does not provide optimization for small objects and falls short of the original CLIP. Conversely, our Open-Vocabulary SAM fuses knowledge from SAM and CLIP via SAM2CLIP and CLIP2SAM, which keeps the original CLIP and can support interactive segmentation, instance-level classification, and scale-up beyond the frozen CLIP. SSA [6] also builds a system that integrates the knowledge from CLIP and SAM. It involves a semantic voting branch for refining the semantic segmentation results generated by an external semantic segmentation model and refined by SAM. On the contrary, Open-Vocabulary SAM does not require the external segmentor and takes visual prompts rather than coarse semantic segmentation results as input. Moreover, Open-Vocabulary SAM focuses on instance segmentation rather than semantic segmentation. Sambor [20] builds on frozen SAM and CLIP, but adds a SideFormer to fuse the knowledge. The SideFormer serves as an adapter for generating object proposals. Compared to Open-Vocabulary SAM, however, the design keeps two foundation models, which introduces extra computational costs similar to combined baselines. In addition, Sambor does not perform interactive segmentation like Open-Vocabulary SAM.

Table S1: Comparison with Recent Joint SAM and CLIP models [6,20,59,83].

|Property<br><br>|SSA [6] SAM-CLIP [59] RecognizeAnything [83] Sambor [20] Ours|
|---|---|
|Single Backbone Object-Level Classification<br><br>Interactive Segmentation<br><br>|✗ ✓ ✗ ✗ ✓<br>✗ ✗ ✗ ✓ ✓<br>✗ ✓ ✗ ✗ ✓<br>|

SAM2CLIP

training only

Trainable

[Figure 53]

[Figure 54]

[Figure 55]

SAM Encoder

[Figure 56]

Distillation Loss

SAM2CLIP

CLIP Encoder SAM Encoder

Neck

[Figure 57]

CLIP Encoder

CLIP2SAM

Language Embeddings

| | | |
|---|---|---|
| | | |

[Figure 58]

CLIP2SAM

[Figure 59]

Pen

[Figure 60]

Prompt Encoder

SAM

Decoder Cat

RoIAlign

Classification Loss

CLIP Encoder

FPN

[Figure 61]

[Figure 62]

Inference Mode

[Figure 63]

[Figure 64]

: Parameters Trainable : Parameters Frozen

CLIP Encoder

- - SAM encoder is dropped during the inference.
- - Use open-vocabulary classifier.
- - CLIP encoder is only inference once.
- - Interactive segmentation and recognition

| | | |
|---|---|---|
| | | |

SAM2CLIP CLIP2SAM

Pen

Prompt Encoder

SAM

Decoder Cat

- Fig. S1: Illustration of Open-Vocabulary SAM. For training, the SAM encoder is adopted as a teacher network, while SAM2CLIP plays the role of a student network and aligns the knowledge of SAM into CLIP. The CLIP2SAM transfers the CLIP knowledge to the SAM decoder and performs joint segmentation and classification for close-set and open vocabulary settings. For inference, we drop the SAM encoder and only take the CLIP encoder. Each visual prompt is only encoded and decoded by the prompt encoder and lightweight SAM decoder, which is the same as SAM.

#### S1.2 Comparison with Open-Vocabulary Methods

Although our method can recognize objects in an open-vocabulary setting, our work is in a different setting compared to previous open-vocabulary methods. Specifically, we aim to build a SAM-like model with interactive prompts, such as points and boxes. On the contrary, previous open-vocabulary detection methods [18, 78, 79, 90] need to generate region proposals or mask proposals to recall all possible objects in the scene. Thus, most approaches aim to improve the recall of novel objects in region proposal networks or mask proposal networks [9,79]. Meanwhile, previous open-vocabulary semantic segmentation methods [63,64,67,72] focus on cross-dataset evaluation mode. Our Open-Vocabulary SAM is different from all these works. Our model, following the same spirit of SAM, mainly takes visual proposals for segmentation and recognition. Thus, not only can users deploy our model to take the visual prompts from humans but also deploy our model on various OV-detectors [67,90] to achieve instance segmentation or class-agnostic detectors to achieve joint segmentation and recognition. Moreover, once scaled up with massive datasets, our method is more practical than previous open-vocabulary works mainly designed for several specific datasets, such as ADE-20k, Pascal-VOC, or LVIS. We provide an interactive demo that can take visual prompts from humans to demonstrate real-world performance.

Table S2: Comparison of mask quality with various detectors on LVIS dataset. We report the mask mean AP for comparison. The masks are generated by each method, while the labels are from the corresponding detectors.

Method Detectors mAP AP50 AP75 APS APM APL APr APn APf #Params FLOPs

|SAM-Huge Detic (swin-base) Open-Vocabulary SAM Detic (swin-base)<br><br>|43.5 59.8 46.5 42.6 59.5 45.1|31.4 55.3 60.6 29.7 53.6 61.6<br><br>|42.6 43.3 44.2 42.0 42.5 42.9|641M 3,001G 304M 1,180G<br><br>|
|---|---|---|---|---|
|SAM-Huge ViTDet (Huge) TAP-L [49] ViTDet (Huge)<br><br>Open-Vocabulary SAM ViTDet (Huge)|44.5 61.5 47.4<br><br>42.6 – –<br><br>44 61.7 47.1<br><br>|34.2 56.1 60.4 29.8 55.5 64.8<br><br>33 54.9 61.6|34.6 45.4 47.8<br><br>33.3 43.6 45.5<br><br>34.5 44.9 47.3<br><br><br>|641M 3,001G 312M 1502G 304M 1,180G|

#### S1.3 Training and Inference Details

- In Fig. S1, we present a more detailed visualization of both training and inference of our Open-Vocabulary SAM. As shown in the figure, only three components are learned during the training, including SAM2CLIP (129M), CLIP2SAM (3.8M), and SAM decoder (4.0M). The remaining parameters are frozen. The total parameters are 304M. After SAM2CLIP distillation, the heavy SAM encoder (637M) is dropped when fine-tuning our model on the detection and classification datasets, which speeds up the CLIP2SAM process. The CLIP2SAM module is co-trained using a diverse mixture of datasets, encompassing detection, segmentation, and classification tasks. Specifically, for segmentation, the training integrates both mask and classification labels, while for detection and classification tasks, the focus is solely on training the classification head. The classification dataset, notable for its extensive range of class categories, coupled with the inclusion of several recent works that feature an extensive vocabulary, significantly enhances the model’s capabilities. After this extensive co-training process, our model exhibits the remarkable ability to recognize and segment over twenty thousand class categories, all within a single, unified framework.

S2 More Experimental Results

- S2.1 More Comparison with SAM

In the main paper, we already compare our method with SAM on the COCO dataset (Tab.4), which shows that Open-Vocabulary SAM has good performance compared to the original SAM. In Tab. S2, we further compare our method with SAM and a recent SAM-like method TAP [49]. The results of TAP-L [49] are from the original paper. Regarding the mask quality, we notice that OpenVocabulary SAM has comparable or slightly worse results compared to the original SAM. However, we want to note that Open-Vocabulary SAM has a much lower computational cost. When comparing with recently proposed TAP [49], Open-Vocabulary SAM has better performance and lower computational cost.

- S2.2 More Visualization Comparison

- In Fig. S2, we compare our method with the feature-crop baseline with visualization results. We noticed that although the recognition capability of the

Baseline Open-Vocabulary SAM Baseline Open-Vocabulary SAM

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

- Fig. S2: Visualization comparison of Open-Vocabulary SAM and the feature-crop baseline. We print the class name output by the model on the corresponding masks.

feature-crop baseline is significantly poor compared to Open-Vocabulary SAM, especially on the small objects, the errors are relatively traceable. We noticed that the feature-crop baseline can extract useful information from features, but it may be confused with adjacent objects. We speculate that this may be because it only performs image-level training and lacks object-level recognition capabilities. In contrast, Open-Vocabulary SAM has great advantages at object-level recognition, especially in the recognition of small objects.

#### S2.3 Segment and Recognize Everything

- In Fig. S3, we demonstrate demonstrate the capabilities of Open-Vocabulary SAM on segment and recognize everything. With the box prompts as input, our method can segment every identifiable object within a given image, showcasing its versatility and effectiveness in handling diverse objects. Based on the results presented in Fig. S3, Open-Vocabulary SAM not only excels in segmenting a wide range of objects but also in recognizing them, which shows its potential as a powerful tool for comprehensive image analysis.

### S3 Failure Cases, Demo and Future Work

#### S3.1 Failure Cases

In Figure S4, we present a series of instances where our Open-Vocabulary SAM does not perform well. First, Open-Vocabulary SAM struggles to differentiate

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

Fig. S3: Visualization of everything mode.

between classes that are subtly distinct from one another, such as mistaking "Truck" for "Minivan". Moreover, some challenging cases, such as extreme occlusion, also confuse Open-Vocabulary SAM. For example, Open-Vocabulary SAM incorrectly classified “Vase” as “Bowl” in the bottom right example. Besides these challenging scenarios, another notable and interesting issue is the model’s confusion with overlapping objects. It misclassified the “Toy” to the “Toothbrush” (the third example in the figure), where the two objects have significant overlapping. These examples highlight areas where further research could be beneficial in enhancing the performance of Open-Vocabulary SAM.

#### S3.2 Short Introduction To Demo

We present an online demo in addition to our paper. It shows the functionality of our Open-Vocabulary SAM (please also refer to Fig. S5 for illustration), which can segment and recognize various classes on many scenes. Please refer to our GitHub repo at https://github.com/HarborYuan/ovsam for more details.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Pred: Table GT: Coffee Table

Pred: Armchair GT: Chair

Pred: Toothbrush GT: Toy

Pred: Apple GT: Orange

[Figure 90]

[Figure 91]

Pred: Bowl GT: Vase

Pred: Truck GT: Minivan

Fig. S4: Failure cases of Open-Vocabulary SAM.

#### S3.3 Future Work

While users can efficiently interact with specific objects using point-and-click or box-dragging techniques, future work will explore using coarse masks or language descriptions as interactive prompts. We aim to continue investigating these promising new directions. Also, our work mainly focuses on foreground categories (e.g., person), and future work may consider introducing background categories (e.g., sky) for comprehensive segmentation and recognition.

Acknowledgements. This study is supported under the RIE2020 Industry Alignment Fund-Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contributions from the industry partner(s). The project is also supported by Singapore MOE AcRF Tier 1 (RG16/21) and the National Key R&D Program of China (No. 2022ZD0161600).

### References

- 1. Balažević, I., Steiner, D., Parthasarathy, N., Arandjelović, R., Hénaff, O.J.: Towards in-context scene understanding. In: NeurIPS (2023)
- 2. Bao, H., Dong, L., Piao, S., Wei, F.: BEiT: BERT pre-training of image transformers. In: ICLR (2022)
- 3. Bar, A., Gandelsman, Y., Darrell, T., Globerson, A., Efros, A.: Visual prompting via image inpainting. In: NeurIPS (2022)

[Figure 92]

Fig. S5: With a simple click, our Open-Vocabulary SAM can generate the mask and label of the object. Please refer to tools_demo.mov for the interactive demo.

- 4. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. In: NeurIPS (2020)
- 5. Chen, C., Miao, J., Wu, D., Yan, Z., Kim, S., Hu, J., Zhong, A., Liu, Z., Sun, L., Li, X., Liu, T., Heng, P.A., Li, Q.: MA-SAM: Modality-agnostic SAM adaptation for 3d medical image segmentation. arXiv preprint arXiv:2309.08842 (2023)
- 6. Chen, J., Yang, Z., Zhang, L.: Semantic segment anything (2023)
- 7. Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., et al.: MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155 (2019)
- 8. Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., Qiao, Y.: Vision transformer adapter for dense predictions. In: ICLR (2023)
- 9. Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. In: CVPR (2022)
- 10. Cheng, Y., Li, L., Xu, Y., Li, X., Yang, Z., Wang, W., Yang, Y.: Segment and track anything. arXiv preprint arXiv:2305.06558 (2023)
- 11. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A large-scale hierarchical image database. In: CVPR (2009)
- 12. Ding, J., Xue, N., Xia, G.S., Dai, D.: Decoupling zero-shot semantic segmentation. In: CVPR (2022)
- 13. Ding, Z., Wang, J., Tu, Z.: Open-vocabulary universal image segmentation with MaskCLIP. In: ICML (2023)
- 14. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021)
- 15. Fang, Y., Yang, S., Wang, S., Ge, Y., Shan, Y., Wang, X.: Unleashing vanilla vision transformer with masked image modeling for object detection. In: ICCV (2023)

- 16. Fang, Z., Li, X., Li, X., Buhmann, J.M., Loy, C.C., Liu, M.: Explore in-context learning for 3d point cloud understanding. In: NeurIPS (2023)
- 17. Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: CLIPAdapter: Better vision-language models with feature adapters. IJCV (2024)
- 18. Gu, X., Lin, T.Y., Kuo, W., Cui, Y.: Open-vocabulary object detection via vision and language knowledge distillation. In: ICLR (2021)
- 19. Gupta, A., Dollar, P., Girshick, R.: LVIS: A dataset for large vocabulary instance segmentation. In: CVPR (2019)
- 20. Han, X., Wei, L., Yu, X., Dou, Z., He, X., Wang, K., Han, Z., Tian, Q.: Boosting segment anything model towards open-vocabulary learning. arXiv preprint arXiv:2312.03628 (2023)
- 21. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: CVPR (2022)
- 22. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask R-CNN. In: ICCV (2017)
- 23. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: ICLR (2022)
- 24. Huynh, D., Kuen, J., Lin, Z., Gu, J., Elhamifar, E.: Open-vocabulary instance segmentation via robust cross-modal pseudo-labeling. In: CVPR (2022)
- 25. Jayaraman, D., Grauman, K.: Zero-shot recognition with unreliable attributes. In: NeurIPS (2014)
- 26. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: ICML (2021)
- 27. Kim, D., Lin, T.Y., Angelova, A., Kweon, I.S., Kuo, W.: Learning open-world object proposals without learning to classify. RA-L (2022)
- 28. Kim, W., Son, B., Kim, I.: ViLT: Vision-and-language transformer without convolution or region supervision. In: ICML (2021)
- 29. Kirillov, A., He, K., Girshick, R., Rother, C., Dollár, P.: Panoptic segmentation. In: CVPR (2019)
- 30. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: ICCV (2023)
- 31. Kuo, W., Cui, Y., Gu, X., Piergiovanni, A.J., Angelova, A.: F-VLM: Openvocabulary object detection upon frozen vision and language models. In: ICLR

(2023)

- 32. Li, F., Zhang, H., Sun, P., Zou, X., Liu, S., Yang, J., Li, C., Zhang, L., Gao, J.: Semantic-SAM: Segment and recognize anything at any granularity. arXiv preprint arXiv:2307.04767 (2023)
- 33. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: ICML (2023)
- 34. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: ICML (2022)
- 35. Li, J., Selvaraju, R.R., Gotmare, A., Joty, S.R., Xiong, C., Hoi, S.C.: Align before fuse: Vision and language representation learning with momentum distillation. In: NeurIPS (2021)
- 36. Li, S., Cao, J., Ye, P., Ding, Y., Tu, C., Chen, T.: ClipSAM: CLIP and SAM collaboration for zero-shot anomaly segmentation. arXiv preprint arXiv:2401.12665

(2024)

- 37. Li, X., Ding, H., Zhang, W., Yuan, H., Cheng, G., Jiangmiao, P., Chen, K., Liu, Z., Loy, C.C.: Transformer-based visual segmentation: A survey. arXiv pre-print

(2023)

- 38. Li, X., You, A., Zhu, Z., Zhao, H., Yang, M., Yang, K., Tong, Y.: Semantic flow for fast and accurate scene parsing. In: ECCV (2020)
- 39. Li, X., Yuan, H., Li, W., Ding, H., Wu, S., Zhang, W., Li, Y., Chen, K., Loy, C.C.: OMG-Seg: Is one model good enough for all segmentation? In: CVPR (2024)
- 40. Li, X., Zhang, J., Yang, Y., Cheng, G., Yang, K., Tong, Y., Tao, D.: Sfnet: Faster and accurate domain agnostic semantic segmentation via semantic flow. IJCV

(2023)

- 41. Li, Y., Fan, H., Hu, R., Feichtenhofer, C., He, K.: Scaling language-image pretraining via masking. In: CVPR (2023)
- 42. Lian, D., Zhou, D., Feng, J., Wang, X.: Scaling & shifting your features: A new baseline for efficient model tuning. In: NeurIPS (2022)
- 43. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: ECCV (2014)
- 44. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023)
- 45. Liu, Y., Qi, L., Tsai, Y.J., Li, X., Chan, K.C.K., Yang, M.H.: Effective adapter for face recognition in the wild. arXiv preprint (2023)
- 46. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)
- 47. Maaz, M., Rasheed, H., Khan, S., Khan, F.S., Anwer, R.M., Yang, M.H.: Classagnostic object detection with multi-modal transformer. In: ECCV (2022)
- 48. Milletari, F., Navab, N., Ahmadi, S.: V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In: 3DV (2016)
- 49. Pan, T., Tang, L., Wang, X., Shan, S.: Tokenize anything via prompting. arXiv preprint arXiv:2312.09128 (2023)
- 50. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. In: NeurIPS (2019)
- 51. Qi, L., Kuen, J., Shen, T., Gu, J., Guo, W., Jia, J., Lin, Z., Yang, M.H.: Highquality entity segmentation. In: ICCV (2023)
- 52. Qi, L., Kuen, J., Wang, Y., Gu, J., Zhao, H., Torr, P., Lin, Z., Jia, J.: Open world entity segmentation. IEEE TPAMI (2022)
- 53. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)
- 54. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022)
- 55. Rubin, O., Herzig, J., Berant, J.: Learning to retrieve prompts for in-context learning. arXiv:2112.08633 (2021)
- 56. Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., Sun, J.: Objects365: A large-scale, high-quality dataset for object detection. In: ICCV (2019)
- 57. Shi, C., Yang, S.: Edadet: Open-vocabulary object detection using early dense alignment. In: ICCV (2023)
- 58. Sun, Q., Fang, Y., Wu, L., Wang, X., Cao, Y.: EVA-CLIP: Improved training techniques for CLIP at scale. arXiv preprint arXiv:2303.15389 (2023)
- 59. Wang, H., Vasu, P.K.A., Faghri, F., Vemulapalli, R., Farajtabar, M., Mehta, S., Rastegari, M., Tuzel, O., Pouransari, H.: SAM-CLIP: Merging vision foundation models towards semantic and spatial understanding. arXiv preprint arXiv:2310.15308 (2023)
- 60. Wang, J., Zhang, P., Chu, T., Cao, Y., Zhou, Y., Wu, T., Wang, B., He, C., Lin, D.: V3det: Vast vocabulary visual detection dataset. In: ICCV (2023)

- 61. Wang, X., Wang, W., Cao, Y., Shen, C., Huang, T.: Images speak in images: A generalist painter for in-context visual learning. In: CVPR (2023)
- 62. Wang, X., Zhang, X., Cao, Y., Wang, W., Shen, C., Huang, T.: Seggpt: Segmenting everything in context. In: ICCV (2023)
- 63. Wu, J., Li, X., Ding, H., Li, X., Cheng, G., Tong, Y., Loy, C.C.: Betrayed by captions: Joint caption grounding and generation for open vocabulary instance segmentation. In: ICCV (2023)
- 64. Wu, J., Li, X., Xu, S., Yuan, H., Ding, H., Yang, Y., Li, X., Zhang, J., Tong, Y., Jiang, X., Ghanem, B., Tao, D.: Towards open vocabulary learning: A survey. IEEE TPAMI (2024)
- 65. Wu, J., Fu, R., Fang, H., Liu, Y., Wang, Z., Xu, Y., Jin, Y., Arbel, T.: Medical SAM Adapter: Adapting segment anything model for medical image segmentation. arXiv preprint arXiv:2304.12620 (2023)
- 66. Wu, S., Zhang, W., Jin, S., Liu, W., Loy, C.C.: Aligning bag of regions for openvocabulary object detection. In: CVPR (2023)
- 67. Wu, S., Zhang, W., Xu, L., Jin, S., Li, X., Liu, W., Loy, C.C.: CLIPSelf: Vision transformer distills itself for open-vocabulary dense prediction. In: ICLR (2024)
- 68. Wu, X., Zhu, F., Zhao, R., Li, H.: CORA: Adapting CLIP for open-vocabulary detection with region prompting and anchor pre-matching. In: CVPR (2023)
- 69. Xie, J., Li, W., Li, X., Liu, Z., Ong, Y.S., Loy, C.C.: Mosaicfusion: Diffusion models as data augmenters for large vocabulary instance segmentation. arXiv preprint arXiv:2309.13042 (2023)
- 70. Xu, J., Liu, S., Vahdat, A., Byeon, W., Wang, X., De Mello, S.: Open-vocabulary panoptic segmentation with text-to-image diffusion models. In: CVPR (2023)
- 71. Xu, M., Zhang, Z., Wei, F., Hu, H., Bai, X.: Side adapter network for openvocabulary semantic segmentation. In: CVPR (2023)
- 72. Xu, M., Zhang, Z., Wei, F., Lin, Y., Cao, Y., Hu, H., Bai, X.: A simple baseline for open-vocabulary semantic segmentation with pre-trained vision-language model. In: ECCV (2022)
- 73. Xu, S., Li, X., Wu, S., Zhang, W., Li, Y., Cheng, G., Tong, Y., Chen, K., Loy, C.C.: Dst-det: Simple dynamic self-training for open-vocabulary object detection. arXiv pre-print (2023)
- 74. Xu, X., Xiong, T., Ding, Z., Tu, Z.: Masqclip for open-vocabulary universal image segmentation. In: ICCV (2023)
- 75. Yang, J., Peng, W., Li, X., Guo, Z., Chen, L., Li, B., Ma, Z., Zhou, K., Zhang, W., Loy, C.C., et al.: Panoptic video scene graph generation. In: CVPR (2023)
- 76. Yu, Q., He, J., Deng, X., Shen, X., Chen, L.C.: Convolutions die hard: Openvocabulary segmentation with single frozen convolutional CLIP. In: NeurIPS

(2023)

- 77. Yu, X., Tang, L., Rao, Y., Huang, T., Zhou, J., Lu, J.: Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In: CVPR (2022)
- 78. Zang, Y., Li, W., Zhou, K., Huang, C., Loy, C.C.: Open-vocabulary DETR with conditional matching. In: ECCV (2022)
- 79. Zareian, A., Rosa, K.D., Hu, D.H., Chang, S.F.: Open-vocabulary object detection using captions. In: CVPR (2021)
- 80. Zhai, X., Wang, X., Mustafa, B., Steiner, A., Keysers, D., Kolesnikov, A., Beyer, L.: LiT: Zero-shot transfer with locked-image text tuning. In: CVPR (2022)
- 81. Zhang, C., Han, D., Qiao, Y., Kim, J.U., Bae, S.H., Lee, S., Hong, C.S.: Faster segment anything: Towards lightweight SAM for mobile applications. arXiv preprint arXiv:2306.14289 (2023)

- 82. Zhang, R., Jiang, Z., Guo, Z., Yan, S., Pan, J., Dong, H., Gao, P., Li, H.: Personalize segment anything model with one shot. arXiv preprint arXiv:2305.03048 (2023)
- 83. Zhang, Y., Huang, X., Ma, J., Li, Z., Luo, Z., Xie, Y., Qin, Y., Luo, T., Li, Y., Liu, S., et al.: Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514 (2023)
- 84. Zhao, X., Ding, W., An, Y., Du, Y., Yu, T., Li, M., Tang, M., Wang, J.: Fast segment anything. arXiv preprint arXiv:2306.12156 (2023)
- 85. Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., Torralba, A.: Semantic understanding of scenes through the ade20k dataset. IJCV (2019)
- 86. Zhou, C., Li, X., Loy, C.C., Dai, B.: EdgeSAM: Prompt-in-the-loop distillation for on-device deployment of SAM. arXiv preprint arXiv:2312.06660 (2023)
- 87. Zhou, C., Loy, C.C., Dai, B.: Extract free dense labels from CLIP. In: ECCV (2022)
- 88. Zhou, H., Shen, T., Yang, X., Huang, H., Li, X., Qi, L., Yang, M.H.: Rethinking evaluation metrics of open-vocabulary segmentaion. arXiv preprint arXiv:2311.03352 (2023)
- 89. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. IJCV (2022)
- 90. Zhou, X., Girdhar, R., Joulin, A., Krähenbühl, P., Misra, I.: Detecting twentythousand classes using image-level supervision. In: ECCV (2022)
- 91. Zou, X., Yang, J., Zhang, H., Li, F., Li, L., Gao, J., Lee, Y.J.: Segment everything everywhere all at once. In: NeurIPS (2023)

