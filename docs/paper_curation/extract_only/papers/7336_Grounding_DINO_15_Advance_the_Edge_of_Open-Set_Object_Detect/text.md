# arXiv:2405.10300v2[cs.CV]1Jun2024

## Grounding DINO 1.5: Advance the “Edge” of Open-Set Object Detection

Tianhe Ren∗, Qing Jiang∗, Shilong Liu∗, Zhaoyang Zeng∗, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, Yuda Xiong, Hao Zhang, Feng Li, Peijun Tang, Kent Yu, Lei Zhang†

International Digital Economy Academy (IDEA), IDEA Research https://deepdataspace.com/home

###### Abstract

This paper introduces Grounding DINO 1.5, a suite of advanced open-set object detection models developed by IDEA Research, which aims to advance the “Edge”1 of open-set object detection. The suite encompasses two models: Grounding DINO 1.5 Pro, a high-performance model designed for stronger generalization capability across a wide range of scenarios, and Grounding DINO 1.5 Edge, an efficient model optimized for faster speed demanded in many applications requiring edge deployment. The Grounding DINO 1.5 Pro model advances its predecessor by scaling up the model architecture, integrating an enhanced vision backbone, and expanding the training dataset to over 20 million images with grounding annotations, thereby achieving a richer semantic understanding. The Grounding DINO 1.5 Edge model, while designed for efficiency with reduced feature scales, maintains robust detection capabilities by being trained on the same comprehensive dataset. Empirical results demonstrate the effectiveness of Grounding DINO 1.5, with the Grounding DINO 1.5 Pro model attaining a 54.3 AP on the COCO detection benchmark and a 55.7 AP on the LVIS-minival zero-shot transfer benchmark, setting new records for open-set object detection. Furthermore, the Grounding DINO 1.5 Edge model, when optimized with TensorRT, achieves a speed of 75.2 FPS while attaining a zero-shot performance of 36.2 AP on the LVIS-minival benchmark, making it more suitable for edge computing scenarios. Model examples and demos with API will be released at https://github.com/IDEA-Research/Grounding-DINO-1.5-API.

###### 1 Introduction

In this paper, we introduce Grounding DINO 1.5, a series of powerful and practical open-set object detection models developed by IDEA Research. Object detection is a fundamental task in computer vision, with recent efforts focusing on developing generic detectors capable of performing detection across a wide variety of real-world applications. A key strategy for improving model generalization across diverse object categories is the integration of language modality, which has received increasingly more attention and has undergone extensive development in recent research.

Grounding DINO [18] represents a significant advancement in this area. Building on the Transformerbased DINO [33] architecture, it incorporates linguistic information to enable open-set object detection in various scenarios. Following GLIP [16], Grounding DINO redefines object detection as

∗Equal contributions. List order is random. †Project lead and corresponding author. 1We use “edge” for its dual meaning both as in pushing the boundaries and as in running on edge devices.

Technical report.

a phrase grounding task, facilitating large-scale pre-training using both detection and grounding datasets. This approach, coupled with self-training on pseudo-labeled grounding data derived from an almost unlimited pool of image-text pairs, enhances the model’s applicability to open-world settings due to its robust architecture and semantic-rich dataset.

Building upon the success of Grounding DINO, Grounding DINO 1.5 extends the model’s capabilities in two key areas: stronger detection performance and faster inference speed, designated as the Grounding DINO 1.5 Pro and Grounding DINO 1.5 Edge models, respectively.

The Grounding DINO 1.5 Pro model significantly expands both the model’s capacity and the dataset size, with the goal of creating a more potent and versatile open-set object detection model. Specifically, we have upscaled the model by incorporating the pre-trained ViT-L [6] architecture and developed a data engine capable of producing over 20 million images with grounding annotations from diverse sources, thereby enriching the model’s semantic comprehension.

By contrast, the Grounding DINO 1.5 Edge model is tailored for edge devices, focusing on computational efficiency without compromising detection quality. We develop an efficient feature enhancer that leverages only high-level image features, removing the need of multi-scale features. This streamlined approach maintains the model’s strong context-aware detection capabilities, after training on the same 20 million images as used for the Pro model.

Extensive results from our experiments validate the superiority of Grounding DINO 1.5. Specifically, Grounding DINO 1.5 Pro achieves a 54.3 AP on the COCO detection zero-shot transfer benchmark and simultaneously achieves a 55.7 AP and a 47.6 AP on the LVIS-minival and LVIS-val zeroshot transfer benchmarks, respectively, setting new records on these benchmarks. Moreover, under TensorRT optimization, the Grounding DINO 1.5 Edge model reaches a speed of 75.2 FPS and achieves a zero-shot performance of 36.2 AP on the LVIS-minival benchmark, making it more suitable for edge computing scenarios.

###### 2 Model Training

###### 2.1 Model Architecture

We present the overall framework of Grounding DINO 1.5 series in Figure 1. This framework retains the dual-encoder-single-decoder structure of Grounding DINO and extends it into Grounding DINO

- 1.5 for both the Pro and Edge models, which are introduced in Sections 2.1.1 and 2.1.2, respectively.

|Image<br><br>Backbone|
|---|

Image

Backbone

|Text Backbone|
|---|

Text Backbone

|FeatureEnhancer|
|---|

FeatureEnhancer

[Figure 1]

A cat sitting on a car .

|Decoder|
|---|

Decoder

|Language-guided QuerySelection|
|---|

Language-guidedLanguage-guided QuerySelection

Queries

Text Feature

Image Feature

Image Feature

Detected

Objects

|Box Loss|
|---|

Box Loss

|Contrastive Loss|
|---|

Contrastive Loss

|(a) Model Framework|
|---|

|(b) Contrastive Loss|
|---|

[Figure 2]

dog . blue sky .

[Figure 3]

a cat sitting on a car

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]<br><br>[Figure 11]|
|---|

[Figure 12]

| |
|---|

[Figure 13]

| |
|---|

[Figure 14]

| |
|---|

[Figure 15]

dog . blue sky .

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

|[Figure 23]<br><br>[Figure 24]<br><br>|
|---|

[Figure 25]

[Figure 26]

[Figure 27]

Figure 1: Grounding DINO 1.5 series overall framework.

- 2.1.1 Grounding DINO 1.5 Pro

Grounding DINO 1.5 Pro preserves the core architecture of Grounding DINO while incorporating a larger Vision Transformer backbone. We adopt the pre-trained ViT-L [6] model as our primary vision backbone for its superior performance on downstream tasks and its pure Transformer design, which lays a solid foundation for optimizing the training and inference processes.

Following the methodologies of Grounding DINO [18] and GLIP [16], Grounding DINO 1.5 Pro employs a deep early fusion strategy during feature extraction. This involves cross-attention mecha-

nisms between language and image features before the decoding phase, facilitating a more integrated information fusion.

We also compared the strategies of early fusion and later fusion. We observe that models trained with early fusion architecture design tend to yield a higher detection recall and better bounding box precision accuracy. However, this approach can also lead to increased model hallucinations, such as predicting objects not present in the input images. In contrast to early fusion, the models with a late fusion design, which integrate language and image modalities only in the loss calculation phase, generally demonstrate robustness against hallucinations but may lead to lower detection recall. This is primarily due to the increased challenge of vision and language alignment as late fusion keeps features from different modalities separately until the loss phase.

Consequently, to simultaneously enhance the model’s prediction capability and maintain its robustness during inference, we have retained the early fusion design while introducing a more comprehensive training sampling strategy, which increases the proportion of negative samples during training. Such improvements facilitate achieving a balance between the advantages and drawbacks of the early fusion architecture.

###### 2.1.2 Grounding DINO 1.5 Edge

Deploying Grounding DINO on edge devices is highly desired by many applications, including autonomous driving, medical image processing, computational photography, etc. However, there is a large gap between the computational cost required by an open-set detection model and the limited resources available on edge devices. Grounding DINO uses multi-scale image features and a heavy computational feature enhancer for faster training and better performance, which is impractical for real-time scenarios in real-world applications.

To overcome this obstacle, we propose a novel efficient feature enhancer, as shown in Fig.2. Recognizing that lower-level image features lack semantic information and introduce excessive computational costs, as demonstrated in Lite-DETR [15], we limit cross-modality fusion to high-level image features (P5 level) only. This approach greatly reduces the number of tokens that need to be processed, significantly cutting the computational demands of the feature enhancer. To facilitate easier deployment on edge-side GPUs, we replace deformable self-attention with vanilla self-attention, and introduce a cross-scale feature fusion module [37] to integrate low-level image features (from P3 and P4 levels). Such a design effectively balances feature enhancement and computational efficiency.

In our edge-optimized model, Grounding DINO 1.5 Edge, we replace the original feature enhancer with our newly proposed efficient one and employ EfficientViT-L1 [1] as the image backbone for rapid multi-scale feature extraction. We deploy the model on the NVIDIA Orin NX platform, achieving an inference speed of over 10 FPS at an input size of 640 × 640. The visualization of the model predictions on NVIDIA Orin NX is shown in Figure 16, which demonstrates the effectiveness of our modifications in real-world edge computing environments.

Updated Image Features

Updated Image Features

Updated Text Features

Updated Text Features

Cross-Scale Feature Fusion

FFN

FFN

Image Features: {P3, P4}

Text-to-image Cross-Attention

Text-to-image Cross-Attention

Q K,V

Q K,V

Image-to-text Cross-Attention

Image-to-text Cross-Attention

K,V Q

K,V Q

Deformable Self-Attention

Self-Attention

Self-Attention

Q,K,V Q,K,V

× 𝐿

Image Features: {P5}

Text Features

Text Features Multi-scale Image Features

(a) Origin Feature Enhancer (b) Efficient Feature Enhancer

Figure 2: Comparison of the Origin Feature Enhancer and the New Efficient Feature Enhancer.

###### 2.2 Training Dataset

To train a robust open-set detector, it is crucial to construct a high-quality grounding dataset that is sufficiently rich in categories and encompasses a wide range of detection scenarios. Grounding DINO 1.5 is pre-trained on over 20M grounding images, termed Grounding-20M, which are collected from publicly available sources. We have carefully developed a series of annotation pipelines and post-processing rules to guarantee the high quality of the grounding annotations.

###### 3 Model Evaluation

We compare Grounding DINO 1.5 with other related works on both the zero-shot and fine-tuning settings. The best and the second best results are indicated in bold and with underline.

###### 3.1 Zero-Shot Transfer of Grounding DINO 1.5 Pro

Following the implementation of Grounding DINO [18], we evaluate our model’s zero-shot transfer performance on the COCO [17] dataset, which comprises 80 common categories and the LVIS [7] datasets, which includes 1203 more diverse and long-tailed categories. We report the performance of fixed AP [4] on both the LVIS-val and LVIS-minival splits. As shown in Table 1, our model shows a significant performance improvement compared to the previous Grounding DINO models. For instance, on the COCO zero-shot transfer benchmark, Grounding DINO 1.5 Pro achieves a 54.3 AP, improving upon Grounding DINO Swin-L by 1.8 AP. On the LVIS-minival and LVIS-val zero-shot transfer benchmarks, Grounding DINO 1.5 Pro achieves a 55.7 AP and a 47.6 AP, outperforming the previous best model, DetCLIPv3, by 6.9 AP and 6.2 AP, respectively. Furthermore, compared with the Grounding DINO Swin-T model, our model demonstrates a remarkable improvement of 28.3 AP (55.7 AP vs. 27.4 AP) on the LVIS-minival zero-shot transfer benchmark.

- Table 1: Performance of Grounding DINO 1.5 Pro on the COCO, LVIS, ODinW35 [16] and ODinW13 [16] benchmarks compared to previous methods. Gray numbers indicate that the training dataset includes images or annotations from COCO or LVIS datasets.

Method Backbone Pre-training data COCO LVISminival LVISval ODinW35 ODinW13

APall APall APr APc APf APall APr APc APf APavg APavg Supervised Models (Pre-training data includes COCO, LVIS, etc.)

GLIPv2 [35] Swin-H [32] FourODs,COCO,GoldG,CC15M,SBU 60.6 50.1 - - - - - - - - 55.5 Grounding DINO [18] Swin-L [19] O365,OID,GoldG,Cap4M,COCO,RefC 60.7 33.9 22.2 30.7 38.8 - - - - - APE (B) [24] ViT-L COCO,LVIS,O365,OID,VG 57.7 62.5 - - - 57.0 - - - 29.4 59.8 APE (D) [24] ViT-L [6] COCO,LVIS,O365,OID,VG,RefC,SA-1B,GoldG,PhraseCut 58.3 64.7 - - - 59.6 - - - 28.8 57.9 GLEE-Pro [27] ViT-L [6] GLEE-merged-10M (COCO,LVIS,etc) 62.0 - - - - 55.7 49.2 - - - 53.4

Zero-shot Transfer Models

OWL-ViT [22] ViT-L [5] O365,OID,VG,LiT 42.2 - - - - 34.6 31.2 - - - MDETR [11] ResNet101 [8] COCO,GoldG - 22.5 7.4 22.7 25.0 - - - - - GLIP [16] Swin-L FourODs,GoldG,Cap24M 49.8 37.3 28.2 34.3 41.5 26.9 17.1 23.3 35.4 - 52.1 Grounding DINO [18] Swin-T O365,GoldG,Cap4M 48.4 27.4 18.1 23.3 32.7 - - - - 22.3 49.8 Grounding DINO [18] Swin-L O365,OID,GoldG 52.5 - - - - - - - - 26.1 56.9 OpenSeeD [34] Swin-L COCO,O365 - 23.0 - - - - - - - 15.2 UniDetector [26] ResNet50 [8] COCO,O365,OID - - - - - 19.8 18.0 19.2 21.2 - 47.3 OmDet-Turbo-B [36] ConvNeXt-B [20] O365,GoldG,PhraseCut,Hake,HOI-A 53.4 34.7 - - - - - - - 30.1 54.7 OWL-ST [21] CLIP L/14 [23] WebLI2B - 40.9 41.5 - - 35.2 36.2 - - 24.4 53.0 MQ-GLIP [28] Swin-L O365 - 43.4 34.5 41.2 46.9 34.7 26.9 32.0 41.3 23.9 54.1 DetCLIP [30] Swin-L O365,GoldG,YFCC1M - 38.6 36.0 38.3 39.3 28.4 25.0 27.0 31.6 - DetCLIPv2 [29] Swin-L O365,GoldG,CC15M - 44.7 43.1 46.3 43.7 36.6 33.3 36.2 38.5 - DetCLIPv3 [31] Swin-L O365,V3Det,GoldG,GranuCap50M - 48.8 49.9 49.7 47.8 41.4 41.4 40.5 42.3 - YOLO-World [3] YOLOv8-L [10] O365,GoldG,CC3M 45.1 35.4 27.6 34.1 38.0 - - - - - DINOv [14] Swin-L COCO,SA-1B - - - - - - - - - 15.7 T-Rex2 (visual) [9] Swin-L O365,OID,HierText,CrowdHuman,SA-1B 46.5 47.6 45.4 46.0 49.5 45.3 43.8 42.0 49.5 27.8 T-Rex2 (text) [9] Swin-L O365,OID,GoldG,CC3M,SBU,LAION 52.2 54.9 49.2 54.8 56.1 45.8 42.7 43.2 50.2 22.0 -

Grounding DINO 1.5 Pro (zero-shot) ViT-L [6] Grounding-20M 54.3 55.7 56.1 57.5 54.1 47.6 44.6 47.9 48.7 30.2 58.7

We further evaluate the generalization capability of our model in multiple real-world scenarios using the ODinW (Object Detection in the Wild) [16] benchmark, which encompasses 35 datasets covering a wide range of application domains. We observe that within the ODinW benchmark, several datasets exhibit significant quality issues in terms of the annotated category names. To mitigate such problems, during testing, we performed prompt engineering to refine category names on datasets where their performance was particularly poor to better align their category names with actual testing scenarios. Grounding DINO 1.5 Pro achieves an average of 58.7 AP over 13 datasets on the ODinW13 benchmark and set a new record on the ODinW35 benchmark with an average of 30.2

###### Table 2: Detailed zero-shot performance comparison between Grounding DINO 1.5 Pro and related works on ODinW13 benchmark.

Method Backbone PascalVOC AerialDrone Aquarium Rabbits EgoHands Mushrooms Packages Raccoon Shellfish Vehicles Pistols Pothole Thermal APavg

GLIP Swin-L 61.7 7.1 26.9 75.0 45.5 49.0 62.8 63.3 68.9 57.3 68.6 25.7 66.0 52.1 GLIPv2 Swin-H 66.3 10.9 30.4 74.6 55.1 52.1 71.3 63.8 66.2 57.2 66.4 33.8 73.3 55.5 Grounding DINO Swin-L 66.0 12.6 28.1 72.8 52.1 73.0 63.9 67.9 64.8 62.7 71.7 31.4 78.4 56.9 OmDet-Turbo-B ConvNeXt-B 63.7 16.2 28.5 70.6 55.7 71.5 65.6 63.6 39.7 61.9 65.5 30.2 78.4 54.7 OWL-ST CLIP L/14 53.9 19.9 32.3 84.9 47.1 76.6 70.9 63.8 35.0 58.5 62.6 27.5 55.6 53.0 MQ-GLIP-L Swin-L 64.7 17.4 30.3 71.8 57.2 63.9 53.0 58.1 63.0 63.2 74.4 27.0 58.7 54.1 APE (B) ViT-L - - - - - - - - - - - - - 59.8 APE (D) ViT-L - - - - - - - - - - - - - 57.9 GLEE-Pro-Scale ViT-L 69.1 13.7 34.7 75.6 38.9 57.8 50.6 65.6 62.7 67.3 69.0 30.7 59.1 53.4 T-Rex2 (visual prompt) Swin-L 65.8 16.0 27.0 70.0 61.9 83.7 58.9 67.1 53.0 66.4 69.0 24.1 61.4 55.7

Grounding DINO 1.5 Pro ViT-L 67.1 19.0 38.5 65.7 61.8 82.1 58.1 72.5 62.0 64.3 71.9 29.0 71.4 58.7

AP over 35 datasets, improving upon Grounding DINO by 4.2 AP. The comprehensive per-dataset performance of Grounding DINO 1.5 Pro on ODinW13 are presented in Table 2. Furthermore, the detailed per-dataset performance of Grounding DINO 1.5 Pro on ODinW35 is available in Appendix Section 7.1.

3.2 Fine-tuning Results on Downstream Datasets

We further investigate the transferability of Grounding DINO 1.5 Pro by fine-tuning it on various downstream datasets. As shown in Table 3, on the LVIS dataset, the fine-tuned Grounding DINO 1.5 Pro model achieves a 68.1 AP and a 63.5 AP on the LVIS-minival and LVIS-val splits, respectively, which represent an enhancement of 12.4 AP and 15.9 AP over the Grounding DINO 1.5 Pro zero-shot setting.

###### Table 3: Fine-tuning performance of Grounding DINO 1.5 Pro on the LVIS-minival, LVIS-val, ODinW35 and ODinW13 benchmarks. The fixed AP [4] on LVIS-minival and val splits are reported. †indicates results of fine-tuning with LVIS base categories only.

Method Backbone LVISminival LVISval ODinW35 ODinW13 APall APr APc APf APall APr APc APf APavg APavg

GLIP Swin-L - - - - - - - - - 68.9 GLEE-Pro ViT-L - - - - - - - - - 69.0 GLIPv2 Swin-H 59.8 - - - - - - - - 70.4 OWL-ST+FT † CLIP L/14 54.4 46.1 - - 49.4 44.6 - - - DetCLIPv2 Swin-L 60.1 58.3 61.7 59.1 53.1 49.0 53.2 54.9 - 70.4 DetCLIPv3 Swin-L 60.5 60.7 - - - - - - - 72.1 DetCLIPv3 † Swin-L 60.8 56.7 63.2 59.4 54.1 45.8 55.4 56.4 - -

Grounding DINO 1.5 Pro (zero-shot) ViT-L 55.7 56.1 57.5 54.1 47.6 44.6 47.9 48.7 30.2 58.7 Grounding DINO 1.5 Pro ViT-L 68.1 (+12.4) 68.7 69.5 66.6 63.5 (+15.9) 64.0 63.8 63.0 70.6 (+40.4) 72.4 (+13.7)

After fine-tuning on the ODinW35 dataset, the Grounding DINO 1.5 Pro model sets new records by achieving an average of 70.6 AP across 35 datasets on the ODinW35 benchmark and 72.4 AP over 13 datasets on the ODinW13 benchmark, respectively. This represents a significant improvement over the zero-shot setting, with respective improvements of 40.5 AP and 13.7 AP. The detailed fine-tuning performance of Grounding DINO 1.5 Pro on the ODinW13 benchmark is reported in Table 4.

###### Table 4: Detailed fine-tuning performance of Grounding DINO 1.5 Pro on the ODinW13 benchmark.

Method Backbone PascalVOC AerialDrone Aquarium Rabbits EgoHands Mushrooms Packages Raccoon Shellfish Vehicles Pistols Pothole Thermal APall

GLIP Swin-L 69.6 32.6 56.6 76.4 79.4 88.1 67.1 69.4 65.8 71.6 75.7 60.3 83.1 68.9 GLEE-Pro ViT-L 72.6 36.5 58.1 80.5 74.1 92.0 67.0 76.5 66.4 70.5 66.4 55.7 80.6 69.0 GLIPv2 Swin-H 74.4 36.3 58.7 77.1 79.3 88.1 74.3 73.1 70.0 72.2 72.5 58.3 81.4 70.4 DetCLIPv2 Swin-L 74.4 44.1 54.7 80.9 79.9 90.0 74.1 69.4 61.2 68.1 80.3 57.1 81.1 70.4 Grounding DINO Swin-T 73.6 36.6 57.7 78.7 79.2 92.8 74.7 74.7 61.2 69.6 75.9 60.4 85.9 70.9 MQ-GLIP-L Swin-L - - - - - - - - - - - - - 71.3 DetCLIPv3 Swin-L 76.4 51.2 57.5 79.9 80.2 90.4 75.1 70.9 63.6 69.8 82.7 56.2 83.8 72.1

Grounding DINO 1.5 Pro ViT-L 77.6 37.0 60.2 75.1 78.6 89.2 72.1 81.8 70.8 74.6 77.6 62.3 84.0 72.4

###### 3.3 Main Results of Grounding DINO 1.5 Edge

After pre-training on Grounding-20M, we directly evaluate Grounding DINO 1.5 Edge on the COCO dataset and LVIS dataset in a zero-shot manner. Following previous works[18, 3, 35], we evaluate on both LVIS-val and LVIS-minival splits and report the fixed AP [4] for comparison. The main results are shown in Table 5. Compared with current real-time open-set detectors in an end-to-end test setting, which do not use language cache, Grounding DINO 1.5 Edge achieves a zero-shot AP of 45.0 on COCO. Regarding the zero-shot performance on LVIS-minival, Grounding DINO 1.5 Edge achieves remarkable performance, an AP score of 36.2, which surpasses all other state-of-the-art algorithms (OmDet-Turbo-T 30.3 AP, YOLO-Worldv2-L 32.9 AP, YOLO-Worldv2-M 30.0 AP, YOLO-Worldv2-S 22.7 AP). Notably, deploying Grounding DINO 1.5 Edge model optimized with TensorRT on NVIDIA Orin NX achieves an inference speed of over 10 FPS at an input size of 640 × 640.

Table 5: Zero-shot Results of Grounding DINO 1.5 Edge on COCO and LVIS. Speed measurement is performed on an A100 GPU, expressed in frames per second (FPS). The format used is PyTorch speed / TensorRT FP32 speed. And FPS on NVIDIA Orin NX is also reported. †indicates results of YOLO-World are reproduced by the latest official codes. ‡ indicates it uses language cache, which does not calculate the latency of the text encoder.

Method Backbone Pre-training data test size COCO LVISminival LVISval FPS(A100/TensorRT) FPS(Orin NX)

APall APr APc APf APall APr APc APf End-to-End Open-Set Object Detection

GLIP-T Swin-T O365,GoldG,Cap4M 800 × 1333 46.3 26.0 20.8 21.4 31.0 - - - - - Grounding DINO-T Swin-T O365,GoldG,Cap4M 800 × 1333 48.4 27.4 18.1 23.3 32.7 - - - - 9.4 / 42.6 1.1

Real-time End-to-End Open-Set Object Detection

YOLO-Worldv2-S† YOLOv8-S O365,GoldG 640 × 640 - 22.7 16.3 20.8 25.5 17.3 11.3 14.9 22.7 47.4 / - YOLO-Worldv2-M† YOLOv8-M O365,GoldG 640 × 640 - 30.0 25.0 27.2 33.4 23.5 17.1 20.0 30.1 42.7 / - YOLO-Worldv2-L† YOLOv8-L O365,GoldG 640 × 640 - 33.0 22.6 32.0 35.8 26.0 18.6 23.0 32.6 37.4 / - YOLO-Worldv2-L† YOLOv8-L O365,GoldG,CC3M-Lite 640 × 640 - 32.9 25.3 31.1 35.8 26.1 20.6 22.6 32.3 37.4 / - OmDet-Turbo-T‡ Swin-T O365,GoldG 640 × 640 42.5 30.3 - - - - - - - 21.5 / 140.0 -

Grounding DINO 1.5 Edge EfficientViT-L1 Grounding-20M 640 × 640 42.9 33.5 28.0 34.3 33.9 27.3 26.3 25.7 29.6 21.7 / 111.6 10.7 Grounding DINO 1.5 Edge EfficientViT-L1 Grounding-20M 800 × 1333 45.0 36.2 33.2 36.6 36.3 29.3 28.1 27.6 31.6 18.5 / 75.2 5.5

4 Case Analysis and Qualitative Visualization

In this section, we visualize the detection results of Grounding DINO 1.5 models in real-world scenarios. The images and text prompts are primarily sourced from the COCO [17], LVIS [7], V3Det [25], OpenImages [13], CC3M [2] and SA-1B [12]. We are deeply grateful for their contributions, which have significantly benefited the community.

###### 4.1 Common Object Detection

The visualizations presented in Figures 3 and 4 demonstrate the robust performance of Grounding DINO 1.5 Pro for common object detection scenarios. Our model’s proficiency is evident not only in its handling of typical cases but also in its ability to accurately detect objects under challenging conditions.

The model adeptly identifies objects in monochromatic images, where color cues are minimal, as illustrated by the first example in Figure 3. This showcases the model’s reliance on shape and texture to distinguish objects. The detection of blurry objects, as seen in the second example of Figure 3, is a testament to the model’s robustness against common image degradations, maintaining high accuracy even when visual clarity is compromised.

The ability to detect small and partially occluded objects is crucial for many applications. Grounding DINO 1.5 Pro’s success in these scenarios, as shown in the last image in Figure 3, indicates its fine-grained understanding and the nuanced integration of multi-modal information in autonomous driving scenes. The visualizations also highlight the model’s versatility in handling objects of varying sizes and shapes, from petite to sprawling, each accurately localized and identified.

###### common image

[Figure 28]

[Figure 29]

sparrow

microphone. guitar. man

[Figure 30]

[Figure 31]

[Figure 32]

button . bow-tie plate. pastry.

book . cupcake . sweatshirt . headband

doughnut. muffin

[Figure 33]

[Figure 34]

[Figure 35]

doorknob. plate. lettuce. poster. jean. ring. earring

peach

ferris wheel. clock. ferry

[Figure 36]

[Figure 37]

[Figure 38]

short pants. choker. frisbee. dog

egg. sandwich. magazine

wheel. car

Figure 3: Model predictions on common objects with Grounding DINO 1.5 Pro (part 1).

###### 4.2 Long-tailed Object Detection

This subsection delves into the capability of Grounding DINO 1.5 Pro in detecting long-tailed objects, which are less frequently encountered categories that pose unique challenges for object detection

[Figure 39]

[Figure 40]

[Figure 41]

bolt . antenna . dog . dog collar . steering wheel . windshield wiper

swimsuit . necklace . clock . frisbee

polo shirt . belt . sunglasses

[Figure 42]

[Figure 43]

[Figure 44]

printer . speaker . computer keyboard . mouse . laptop computer . water bottle . drawer . desk

cellular telephone . ring . sweater . earring . handbag

cow

[Figure 45]

[Figure 46]

[Figure 47]

mirror . police cruiser . garbage truck

mouse . mug . plastic bag . computer keyboard

knife . sandwich . tomato

[Figure 48]

[Figure 49]

[Figure 50]

grape . lemon . cat . trousers orange . apple

earring . boot . skirt . necklace . tights

Figure 4: Model predictions on common objects with Grounding DINO 1.5 Pro (part 2).

models. The examples provided in Figure 5 highlight the model’s nuanced capability of understanding

and detecting such uncommon objects. The model demonstrates its ability to recognize a diverse set of objects, including those that are not commonly found in everyday settings.

For instance, the second image within the figure illustrates the model’s capacity to identify a fungus, which is a category that requires specialized knowledge to discern. The third image showcases the model’s fine-grained detection capabilities, accurately pinpointing a popsicle amidst a variety of potential distractors. The model’s contextual understanding is evident in its ability to detect a taco in the last image of the third line, an object that may be challenging to recognize without the appropriate contextual cues.

###### 4.3 Short Caption Grounding

Grounding models can correlate objects within images to their corresponding mentions in accompanying captions. This capability is particularly significant for enhancing the contextual understanding of visual content across various domains. In Figure 6, we present Grounding DINO 1.5 Pro’s proficiency in short caption grounding, highlighting its versatility and accuracy.

The model exhibits a robust performance in grounding objects across different visual domains. It adeptly handles real-world images while also demonstrating a keen ability to interpret objects within cartoons, sketches, and animations. By aligning textual descriptions with visual features, the model can accurately identify and localize objects, even when they appear in stylized or abstract forms.

###### 4.4 Long Caption Grounding

Our model, Grounding DINO 1.5 Pro, extends its capability beyond the realm of standard imagecaption pairs to adeptly handle long captions, as depicted in Figure 7, Figure 8 and Figure 9. Long captions offer a richer tapestry of details that can more comprehensively describe the visual content of an image. The ability to map each noun phrase in a long caption to corresponding objects within an image is a significant step toward deeper image understanding.

An intriguing observation is the model’s ability to generalize from pre-training on captions with shorter context windows to effectively processing longer contexts. This adaptability suggests that larger models may inherently possess flexibility that can be leveraged for various lengths of textual input.

Grounding DINO 1.5 Pro exhibits an impressive capacity to correlate textual phrases with visual elements. This capability is not only showcased in the model’s handling of long captions but also in its nuanced analysis of images at multiple granularities. Moreover, we notice that the model can detect some terms that did not show in the training data, like fiat logo in the third image. It presents the strong generalization ability of the model.

###### 4.5 Dense Object Detection

Grounding DINO 1.5 Pro showcases an exceptional capability to discern objects within dense scenarios, where multiple objects are closely positioned or overlapping, making detection a challenging task. This ability is vividly demonstrated through the visualizations presented in Figure 10 and Figure 11.

The model’s performance is noteworthy across a wide spectrum of object nomenclature. It adeptly identifies objects labeled with common names such as coin, tree, flower, and land. Moreover, the model also excels at recognizing objects denoted by specialized terminology, including kohlrabi, atlantic puffin, and oxalis purpurea.

###### 4.6 Video Object Detection

We present video detection results of Grounding DINO 1.5 Pro in Figure 12. We notice the model can produce consistent object bounding boxes in most cases. The videos are processed offline.

###### 4.7 Side-by-side Comparison

We present the side-by-side comparison between the results of Grounding DINO 1.5 Pro and Grounding DINO in Figure 13 and Figure 14. The Grounding DINO 1.5 Pro model demonstrates superior

#### long tail image

[Figure 51]

[Figure 52]

[Figure 53]

clothing.person.sportsequipment fungus popsicle

[Figure 54]

[Figure 55]

[Figure 56]

bagel. boiled egg remote control sunglasses. necklace. ring.

necktie. camera

[Figure 57]

[Figure 58]

[Figure 59]

mirror. grill. truck

taco

dinghy

[Figure 60]

[Figure 61]

[Figure 62]

telescopic sight . gun

propeller vent . hinge . tassel . rearview mirror

Figure 5: Model predictions on long-tailed categories with Grounding DINO 1.5 Pro.

performance over the Grounding DINO model in terms of dense scene detection, long-tailed object detection, and the accuracy of semantic understanding.

Moreover, we compare the object hallucinations of Grounding DINO 1.5 Pro and Grounding DINO, as shown in Figure 15. The results show that Grounding DINO 1.5 Pro has better accuracy and fewer object hallucinations. The last line in Figure 15 demonstrates the better context understanding ability of our model.

[Figure 63]

[Figure 64]

[Figure 65]

a neon sign of goodbye is placed in the center of the wall.

a yellow boat sitting in the snow near some trees.

the little devil is flying on

the crescent in the night.

[Figure 66]

[Figure 67]

[Figure 68]

a black and yellow snake in its hand with a white ring.

an owl is sitting on the branch with gifts.

two hands hold the globe.

[Figure 69]

[Figure 70]

[Figure 71]

man brushing his teeth with a toothbrush in hand.

hand drawn birds in a circle with inscription.

a family playing football on the beach.

[Figure 72]

[Figure 73]

[Figure 74]

a boy in a helmet with a

green houseplant in glass containers.

cactus with red flowers, as seen in the garden.

smart phone rides a hoverb

self balancing scooter.

Figure 6: Phrase grounding on short captions with Grounding DINO 1.5 Pro.

###### 4.8 Advanced Object Detection on Edge Devices

We present the practical, real-time application potential of Grounding DINO 1.5 Edge through a series of demonstrations in Figure 16. The model’s adept performance in office environments is particularly highlighted, offering significant utility for the field of robotics research.

[Figure 75]

The image shows a large mosque with a white facade, highlighted by a prominent golden dome, additional smaller domes, and towering minarets with arched windows. The mosque, surrounded by a bustling urban scene, features cars, electrical lines, and streetlights, indicative of its location in a lively city. Coniferous trees and shadows enhance the well-maintained landscape under a clear blue sky.

[Figure 76]

The image shows an individual lying prone on grassy ground, aiming a bolt-action rifle with

their right eye close to the sight. This person is wearing a military-style uniform with a steel helmet with netting, suggesting a military or historical reenactment context. On the

individual's back, you can see a backpack and what appears to be a canteen, both typical of military field equipment.

[Figure 77]

The image shows a car on display at a motor show, prominently featuring the Fiat logo in red along the side. The car is painted gray with a glossy finish, highlighted by orange and black racing stripes and red accents on the front splitter. Its sporty design includes

a large rear spoiler and white multi-spoke wheels, suggesting additional aerodynamic features. In the background. Several people navigate the space, including an individual in a blue jacket with a logo, likely an event staff member.

[Figure 78]

The image shows a military tank rolling along a city street, with crew members visible on top. Painted in camouflage and equipped with a long barrel, the tank navigates through an intersection, suggesting a setting likely in Europe or Russia. The surroundings, marked by street signs, a no parking sign, and a directional sign pointing right, indicate the tank's participation in a parade or military demonstration. Spectators, including adults and children, stand behind barriers on the sidewalk, observing the tank against a backdrop of buildings and leafless trees, under an overcast sky.

[Figure 79]

The image captures a young person extending their arm to showcase a Rubik's Cube towards the camera. The cube is the focal point, while the individual's face is intentionally blurred, perhaps for privacy or artistic effect. The person is dressed in semi-formal attire, consisting of a patterned vest over a dress shirt and a bow tie, accessorized with a smartwatch on their wrist.

Figure 7: Phrase grounding on long captions with Grounding DINO 1.5 Pro (part 1).

The image captures a sunny day at an airfield, prominently featuring a large orange aircraft towing vehicle, built robustly with massive tires,

[Figure 80]

designed specifically as an aircraft tug or tow tractor for ground movements of aircraft. Attached to this towing vehicle through a tow bar is a small camouflaged aircraft with tandem wings, one set above and behind the cockpit and

another set towards the front. In the background, several flagpoles with a variety of international flags are visible. The clear blue sky with minimal

cloud streaks sets a backdrop that indicates favorable weather conditions.

[Figure 81]

The photo captures two individuals in a wooded setting during a season. On the right, a young child stands out in bright colors, dressed in a purple patterned coat,

pink pants, and an animal-themed hat. To the child's left, an adult is dressed in a black coat, blue jeans, leaning against a tree

trunk. Both faces are blurred for privacy, and the natural backdrop suggests they are enjoying an outdoor excursion.

[Figure 82]

The image shows an individual with a blurred face for privacy, dressed in floral jacket and floral pants with a white-pink background and green leaves, and a beige top underneath. The jacket features gold buttons and they are wearing a dark necklace with gem-like adornments. The person stands casually with hands in pockets in front of a shop window displaying mannequins. One mannequin wears a white shirt with “CHOOSE JUICY” in gold letters.

[Figure 83]

The image shows a CPR training session outdoors, where an individual with a buzz cut, wearing a dark jacket with "Paul's Team" text, is keenly practicing chest compressions on a CPR manikin placed on a

purple mat. The participant is intensely focused on mastering the technique. Nearby, two individuals in high-visibility vests labeled "POLICE" are observed.

The setup on a carpet indicates the location is adapted for comfort, likely to ease the strain of kneeling during the exercise.

Figure 8: Phrase grounding on long captions with Grounding DINO 1.5 Pro (part 2).

The image captures a moment in a road race, likely a marathon, focusing on a runner's outstretched arm reaching for a plastic water bottle. In the background, three runners, identified by their bib numbers, are in motion. The runner closest to the foreground is dressed in a blue tank top and black shorts, and is possibly wearing a black headband or cap. Another runner, wearing an orange shirt, is also visible. The scene is set against an overcast gray sky and features metal structures indicative of an urban setting. A hedge and a metal fence line the race path, separating the runners from the adjacent grassy area.

[Figure 84]

The image depicts a living space features a large Lshaped grey couch with cushions, alongside a wooden chair with fabric seating. Decorative elements include a large abstract art piece with gold tones and a small framed picture on the walls, with wooden beams overhead adding architectural interest. A decorative chest resembling a trunk. The kitchen, viewed through a wooden framed opening, includes a kitchen table, a kitchen counter and kitchen towels. Hanging potted plants by the window and cooking pots

[Figure 85]

contribute to the room’s functional yet inviting vibe. A white door in the kitchen hints at additional spaces, while two stockings nearby suggest the photo was taken during the holiday season.

The photograph captures a lively dance scene at a social event, possibly a wedding reception, indicated by a lady wearing a wrist corsage, often a mark of special recognition. In the foreground, a woman in a blue dress and corsage executes a dance move with her right hand extended, palm out, and her left hand on the shoulder of a man beside her dressed in a light blue polo shirt and dark jeans. Behind this pair, other participants are blurred, contributing to the dynamic atmosphere of the scene. A man in a white shirt and brown trousers and a woman in a black outfit are also part of the group. Curtains and interior decorations suggest an indoor venue.

[Figure 86]

[Figure 87]

The image shows a postage stamp from the former Soviet Union, distinguished by its Cyrillic text. The brown-toned stamp captures an intense moment from a volleyball match, depicting three female volleyball players engaged in a game. The action centers on one player on the left, who is leaping to hit the ball over the volleyball net, while her two opponents on the right prepare defensivelyone ready to block and the other set to dig or pass.

Figure 9: Phrase grounding on long captions with Grounding DINO 1.5 Pro (part 3).

##### dense object

[Figure 88]

[Figure 89]

[Figure 90]

coin

tree bicycle wheel.person. bicycle. bicycle helmet

[Figure 91]

[Figure 92]

banana . umbrella

land vehicle

[Figure 93]

[Figure 94]

[Figure 95]

broccoli zebrasoma flavescens flower

[Figure 96]

[Figure 97]

[Figure 98]

dish . brussels red paper lantern. model. flower. pot. house.

kue lapis. coffee.

decoration . woman. doorframe.

cup. epergne. dish

- Figure 10: Model predictions on dense object scenarios with Grounding DINO 1.5 Pro (part 1).

### dense object

[Figure 99]

[Figure 100]

[Figure 101]

worker bee dish . brush . bottle . taralli

kohlrabi

[Figure 102]

[Figure 103]

atlantic puffin wine rack . bottle . basket

[Figure 104]

[Figure 105]

[Figure 106]

diamond

flower

saltwater fish

[Figure 107]

[Figure 108]

[Figure 109]

pot . oxalis purpurea

wild strawberry home plate . black morel

- Figure 11: Model predictions on dense object scenarios with Grounding DINO 1.5 Pro (part 2).

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

head. arm. leg

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

race car

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

fighter jet. smoke

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

dinosaur

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

person. boat. boat engine. horse

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

butterfly

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

car. cotorcycle

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

bird. helmet. smile

Figure 12: Model predictions on video object detection with Grounding DINO 1.5 Pro.

Input Image Grounding DINO 1.5 Pro Grounding DINO

[Figure 142]

[Figure 143]

[Figure 144]

dolmas. dinner table. fork. citron

[Figure 145]

[Figure 146]

[Figure 147]

dish. brush. electric rage. bottle. taralli

[Figure 148]

[Figure 149]

[Figure 150]

bowl. vessel. instrumentality. clothing. wind window. range hood. artifact. electric range. kitchen knife. women. board. socket.

[Figure 151]

[Figure 152]

[Figure 153]

building. window. tree.

- Figure 13: Side-by-side comparison between Grounding DINO 1.5 Pro and Grounding DINO [18]

- (part 1).

Input Image Grounding DINO 1.5 Pro Grounding DINO

[Figure 154]

[Figure 155]

[Figure 156]

gulf fritillary. gaillardia pinnatifida

[Figure 157]

[Figure 158]

[Figure 159]

blue cheese. post it note. furniture

[Figure 160]

[Figure 161]

[Figure 162]

corydalis flavula. green violetear

[Figure 163]

[Figure 164]

[Figure 165]

baby. hair clip. chair. footwear. clothing. adult. short sleeve t-shirt. suit. shirt. cravat

- Figure 14: Side-by-side comparison between Grounding DINO 1.5 Pro and Grounding DINO [18]

- (part 2).

Input Image Grounding DINO 1.5 Pro Grounding DINO

[Figure 166]

[Figure 167]

[Figure 168]

popcorn. luggage and bags. panda. cattle. luggage and bags. sofa bed. mirror. office building. mechanical fan

[Figure 169]

[Figure 170]

[Figure 171]

sink. bicycle. orange. bowl. firsbee. bottle. sheep. vase. toothbrush. toilet. cell phone

[Figure 172]

[Figure 173]

[Figure 174]

an old woman with her family in a hospital

- Figure 15: Side-by-side comparison between Grounding DINO 1.5 Pro and Grounding DINO [18] regarding object hallucinations.

[Figure 175]

[Figure 176]

[Figure 177]

toy eye mouth

[Figure 178]

[Figure 179]

[Figure 180]

bottle bottle cap tissue

[Figure 181]

[Figure 182]

[Figure 183]

plant card watch

[Figure 184]

[Figure 185]

[Figure 186]

red rope hand book

[Figure 187]

[Figure 188]

[Figure 189]

wire glasses phone

- Figure 16: The visualization of Grounding DINO 1.5 Edge on NVIDIA Orin NX. The top left of the screen displays the FPS and prompts, while the top right shows a camera view of the recorded scene.

###### 5 Conclusion

This paper has presented Grounding DINO 1.5, a series of models to advance the field of open-set object detection. The flagship model, Grounding DINO 1.5 Pro, has established new records on the COCO and LVIS zero-shot benchmarks, signifying a major stride in detection accuracy and reliability. Moreover, the Grounding DINO 1.5 Edge model enables real-time object detection across various applications, further expanding the practical utility of the Grounding DINO 1.5 series.

###### 6 Contributions and Acknowledgments

We would like to express our gratitude to everyone involved in the Grounding DINO 1.5 project. The contributions are as follows (in no particular order):

- • Grounding DINO 1.5 Pro Model Design, Training Infra Development, Data Collection, Model Training and Model Evaluation: Tianhe Ren, Qing Jiang, Shilong Liu, and Zhaoyang Zeng.
- • Grounding DINO 1.5 Edge Model Design, Training, Evaluation, and Runtime Optimization: Wenlong Liu, Han Gao, Qing Jiang, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, and Yihao Chen.
- • Grounding-20M Data Collection and Annotation Pipeline Construction: Yihao Chen, Hao Zhang, Yuda Xiong, Tianhe Ren, Zhaoyang Zeng, Qing Jiang, Shilong Liu, and Peijun Tang.
- • Provide Great Insight and Technical Support: Hao Zhang and Feng Li.
- • Grounding DINO 1.5 Edge Model Lead: Kent Yu.
- • Overall Project Lead of Grounding DINO 1.5: Lei Zhang.

We would also like to thank everyone involved in the Grounding DINO 1.5 demo support, including application lead Wei Liu, product manager Qin Liu and Xiaohui Wang, front-end developers Yuanhao Zhu, Ce Feng, and Jiongrong Fan, back-end developers Weiqiang Hu and Zhiqiang Li, UX designer Xinyi Ruan, tester Yinuo Chen, and Zijun Deng for helping with demo videos.

- 7 Appendix 7.1 Detailed results on the ODinW benchmark We report the detailed results of Grounding DINO 1.5 Pro on the ODinW35 benchmarks in Table 6

Table 6: Detailed Zero-shot Results of Grounding DINO 1.5 Pro on the ODinW35 benchmark.

Datasets ODinW13 ODinW35 Grounding DINO 1.5 Pro (zero-shot) Grounding DINO 1.5 Pro (fine-tuning)

AerialMaritimeDrone_large ✓ ✓ 19.0 37.0 AerialMaritimeDrone_tiled ✓ 18.2 45.0 AmericanSignLanguageLetters ✓ 13.7 83.3 Aquarium ✓ ✓ 38.5 60.2 BCCD ✓ 22.8 64.0 ChessPieces ✓ 6.8 80.5 CottontailRabbits ✓ ✓ 65.7 75.1 DroneControl ✓ 8.3 79.8 EgoHands_generic ✓ ✓ 61.8 78.6 EgoHands_specific ✓ 16.7 78.7 HardHatWorkers ✓ 20.5 46.4 MaskWearing ✓ 16.7 63.2 MountainDewCommercial ✓ 24.9 33.1 NorthAmericaMushrooms ✓ ✓ 82.1 89.2 OxfordPets_by_breed ✓ 0.9 89.7 OxfordPets_by_species ✓ 61.6 91.5 PKLot ✓ 4.4 96.5 Packages ✓ ✓ 58.1 72.1 PascalVOC ✓ ✓ 67.1 77.6 Raccoon ✓ ✓ 72.5 81.8 ShellfishOpenImages ✓ ✓ 62.0 70.8 ThermalCheetah ✓ 20.7 58.3 UnoCards ✓ 1.7 89.3 VehiclesOpenImages ✓ ✓ 64.3 74.6 WildfireSmoke ✓ 28.9 57.5 boggleBoards ✓ 0.8 77.0 brackishUnderwater ✓ 10.1 76.8 dice ✓ 0.6 79.5 openPoetryVision ✓ 0.9 81.2 pistols ✓ ✓ 71.9 77.6 plantdoc ✓ 3.3 62.6 pothole ✓ ✓ 29.0 62.4 selfdrivingCar ✓ 7.4 53.1 thermalDogsAndPeople ✓ ✓ 71.4 84.0 websiteScreenshots ✓ 2.3 41.6

ODinW13 Average AP 58.7 72.4 ODinW35 Average AP 30.2 70.6

###### References

- [1] Han Cai, Junyan Li, Muyan Hu, Chuang Gan, and Song Han. Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 17302–17313, October 2023. 3
- [2] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts. CVPR, 2021. 6
- [3] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. YOLOWorld: Real-Time Open-Vocabulary Object Detection. CVPR, 2024. 4, 6
- [4] Achal Dave, Piotr Dollár, Deva Ramanan, Alexander Kirillov, and Ross Girshick. Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details. arXiv preprint arXiv:2102.01066, 2022. 4, 5, 6
- [5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2020. 4
- [6] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. EVA-02: A Visual Representation for Neon Genesis. arXiv preprint arXiv:2303.11331, 2023. 2, 4
- [7] Agrim Gupta, Piotr Dollár, and Ross Girshick. LVIS: A Dataset for Large Vocabulary Instance Segmentation, 2019. 4, 6
- [8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. CVPR, 2015. 4
- [9] Qing Jiang, Feng Li, Zhaoyang Zeng, Tianhe Ren, Shilong Liu, and Lei Zhang. T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy. arXiv preprint arXiv:2403.14610,

2024. 4

- [10] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLOv8, January 2023. 4
- [11] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. MDETR - Modulated Detection for End-to-End Multi-Modal Understanding. ICCV,

2021. 4

- [12] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment Anything. ICCV, 2023. 6
- [13] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale. IJCV, 2020. 6
- [14] Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Chunyuan Li, Jianwei Yang, et al. Visual In-Context Prompting. CVPR, 2024. 4
- [15] Feng Li, Ailing Zeng, Shilong Liu, Hao Zhang, Hongyang Li, Lei Zhang, and Lionel M. Ni. Lite DETR : An Interleaved Multi-Scale Encoder for Efficient DETR. CVPR, 2023. 3
- [16] Liunian Harold Li*, Pengchuan Zhang*, Haotian Zhang*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded Language-Image Pre-training. CVPR, 2022. 1, 2, 4
- [17] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft COCO: Common Objects in Context. ECCV, 2014. 4, 6
- [18] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv preprint arXiv:2303.05499, 2023. 1, 2, 4, 6, 18, 19, 20
- [19] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, 2021. 4

- [20] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. CVPR, 2022. 4
- [21] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling Open-Vocabulary Object Detection. NeurIPS, 2023. 4
- [22] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. Simple Open-Vocabulary Object Detection with Vision Transformers. ECCV, 2022. 4
- [23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. ICML,

2021. 4

- [24] Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang, Ke Li, Xing Sun, Yunsheng Wu, Shaohui Lin, and Rongrong Ji. Aligning and Prompting Everything All at Once for Universal Visual Perception. CVPR, 2024. 4
- [25] Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. V3Det: Vast Vocabulary Visual Detection Dataset. ICCV, 2023. 6
- [26] Zhenyu Wang, Yali Li, Xi Chen, Ser-Nam Lim, Antonio Torralba, Hengshuang Zhao, and Shengjin Wang. Detecting Everything in the Open World: Towards Universal Object Detection. CVPR, 2023. 4
- [27] Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, and Song Bai. General Object Foundation Model for Images and Videos at Scale. CVPR, 2024. 4
- [28] Yifan Xu, Mengdan Zhang, Chaoyou Fu, Peixian Chen, Xiaoshan Yang, Ke Li, and Changsheng Xu. Multi-modal Queried Object Detection in the Wild. NeurIPS, 2023. 4
- [29] Lewei Yao, Jianhua Han, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, and Hang Xu. DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment. CVPR, 2023. 4
- [30] Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing Xu, and Hang Xu. DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pretraining for Open-world Detection. NeurIPS, 2022. 4
- [31] Lewei Yao, Renjie Pi, Jianhua Han, Xiaodan Liang, Hang Xu, Wei Zhang, Zhenguo Li, and Dan Xu. DetCLIPv3: Towards Versatile Generative Open-vocabulary Object Detection. CVPR,

2024. 4

- [32] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, Ce Liu, Mengchen Liu, Zicheng Liu, Yumao Lu, Yu Shi, Lijuan Wang, Jianfeng Wang, Bin Xiao, Zhen Xiao, Jianwei Yang, Michael Zeng, Luowei Zhou, and Pengchuan Zhang. Florence: A New Foundation Model for Computer Vision. arXiv preprint arXiv:2111.11432, 2021. 4
- [33] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and HeungYeung Shum. DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. ICLR, 2023. 1
- [34] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianwei Yang, and Lei Zhang. A Simple Framework for Open-Vocabulary Segmentation and Detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1020–1031, 2023. 4
- [35] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. GLIPv2: Unifying Localization and Vision-Language Understanding. NeurIPS, 2022. 4, 6
- [36] Tiancheng Zhao, Peng Liu, Xuan He, Lu Zhang, and Kyusong Lee. Real-time Transformerbased Open-Vocabulary Detection with Efficient Fusion Head. arXiv preprint arXiv:2403.06892,

2024. 4

- [37] Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen. DETRs Beat YOLOs on Real-time Object Detection. CVPR, 2023. 3

