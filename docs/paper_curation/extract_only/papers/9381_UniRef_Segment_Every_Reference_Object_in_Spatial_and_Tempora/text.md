## UniRef++: Segment Every Reference Object in Spatial and Temporal Spaces

Jiannan Wu1, Yi Jiang2, Bin Yan3, Huchuan Lu3, Zehuan Yuan2, Ping Luo1,4

1The University of Hong Kong 2ByteDance 3Dalian University of Technology 4Shanghai AI Laboratory

# arXiv:2312.15715v1[cs.CV]25Dec2023

#### Abstract

|[Figure 1]<br><br>Image+M Image+M+L Image+L<br><br>VOS/FSS RVOS RIS<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]|
|---|

Output Mask

The reference-based object segmentation tasks, namely referring image segmentation (RIS), few-shot image segmentation (FSS), referring video object segmentation (RVOS), and video object segmentation (VOS), aim to segment a specific object by utilizing either language or annotated masks as references. Despite significant progress in each respective field, current methods are task-specifically designed and developed in different directions, which hinders the activation of multi-task capabilities for these tasks. In this work, we end the current fragmented situation and propose UniRef++ to unify the four reference-based object segmentation tasks with a single architecture. At the heart of our approach is the proposed UniFusion module which performs multiway-fusion for handling different tasks with respect to their specified references. And a unified Transformer architecture is then adopted for achieving instancelevel segmentation. With the unified designs, UniRef++ can be jointly trained on a broad range of benchmarks and can flexibly complete multiple tasks at run-time by specifying the corresponding references. We evaluate our unified models on various benchmarks. Extensive experimental results indicate that our proposed UniRef++ achieves state-of-the-art performance on RIS and RVOS, and performs competitively on FSS and VOS with a parametershared network. Moreover, we showcase that the proposed UniFusion module could be easily incorporated into the current advanced foundation model SAM and obtain satisfactory results with parameter-efficient finetuning. Codes and models are available at https://github.com/ FoundationVision/UniRef.

###### UniRef++

###### Multiway-Fusion

Mask Reference

Language Reference

Images

Figure 1: A single, jointly trained UniRef++ can perform three different reference-based tasks by specifying the corresponding references. ‘L’ and ‘M’ represent language and mask references, respectively.

mentation (VOS) [75], which are the fundamental tasks for vision understanding. Over time, many advanced methods have ballooned in their respective fields and rapidly improves the state-of-the-art performance.

Despite witnessing the significant progress, these tasks are separately tackled with specialized designed models. In that regard, the individual methods need extra training time and produce different sets of model weights on each task. This would cause expensive computational cost and yield redundant parameters. More importantly, the independent designs prevent the synergy and facilitation of different tasks. We argue that the current fragmented situation is unnecessary as the four tasks have essentially the same definition in a high-level aspect: they all use the references (language or annotated mask) as guidance to perform the perpixel segmentation of the target object. This motivates us to build a unified model within the same parameters, which can perform different tasks at run-time by specifying the corresponding references.

#### 1. Introduction

The reference-guided object segmentation aims at segmenting the specified object with the given references (e.g., language or annotated mask). The four representative tasks include referring image segmentation (RIS) [122], few-shot segmentation(FSS) [78], referring video object segmentation (RVOS) [42] and semi-supervised video object seg-

Towards the unification of reference-based object segmentation tasks, it poses great challenges in connecting the isolated landscapes as a whole: (1) The mainstream methods in different fields vary greatly. RIS meth-

ods [119, 36, 114] mostly focus on the deep cross-modal fusion of vision and language information. FSS community is advocating the correlation-based methods [32, 108] for dense semantic correspondence. VOS has been long dominated by the space-time memory network [72, 16, 116, 15] for pixel-level matching. While the recenet RVOS methods heavily rely on the query-based methods [6, 102]. (2) The image-level methods cannot be simply extended to the video domain. The image tasks only require to segment the referred target in a single image. For the video tasks, however, the objects may encounter occlusion, fast motion or disapperance-reappearance in many complex scenes, which requires the networks to leverage the spatio-temporal information to track the objects throughout the whole video. Hence, simply adopting the image-level methods for each frame independently cannot ensure the temporal consistency for target object in videos. (3) The video tasks (VOS and RVOS) are solved in two different paradigms currently. The previous state-of-the-art RVOS methods [6, 102] take the whole video as input and generate the prediction results for all frames in one single step, which termed as offline methods. In contrast, VOS methods [72, 16] operate in an online fashion where they readout the historical features to propagate the target masks frame by frame.

In this work, we conquer the challenges above and propose a unified model, UniRef++, for the reference-based object segmentation tasks. The key idea behind our approach is to formulate all four tasks as instance-level segmentation problems, and the information of references can be injected into the network through an attention-based fusion process regardless of their modalities. As illustrated in Figure 1, for different tasks, UniRef++ receives the current frame and utilizes the corresponding references to perform the fusion process, termed as multiway-fusion. Specifically, the annotated mask for reference image is leveraged as reference for FSS and VOS. The reference comes from language description for RIS. And we emphasize that, for RVOS, both the language and mask references are used. This design not only tackles RVOS in an online fashion, but also can utilize the historical information for mask propagation to ensure the temporal consistency for target object, establishing a new paradigm for RVOS.

Practically, we introduce a UniFusion module to fuse the visual features and the specified references. Afterwards, the visual features of current frame are fed into a unified Transformer architecture, where queries are employed for instance-level segmentation of the target object. Thanks to the unified architecture, our model can be jointly trained on the broad range of benchmarks to learn the general knowledge, and can flexibly perform multi-tasks at run-time by specifying the corresponding references.

To summarize, our contributions are as follows:

- • We propose UniRef++, a unified model to perform four reference-based object segmentation tasks (RIS, FSS, RVOS, VOS) with the same model weights.
- • We introduce a UniFusion module to inject the reference information into the network regardless of their modalities. And we establish a new online paradigm for RVOS by leveraging both language and mask as references.
- • Extensive experiments demonstrate that our models achieve state-of-the-art performance for RIS and RVOS, and perform competitively for FSS, VOS.

#### 2. Related Work

##### 2.1. Unified Model

Towards achieving general artificial intelligence, the vision community has clearly witnessed the trend of building unified models recently. One line of works [1, 123, 120, 96, 9, 10, 132, 45, 94, 65] is to design the general interface for vision or vision-language (VL) tasks. For example, UnifiedIO [65] unifies broad range of image-level tasks (e.g., image classification [19], image caption [11] and VQA [2]) in a sequence-to-sequence generation paradigm. Another line of works [13, 40, 47, 126, 111, 133, 125, 59, 38, 5, 3, 112, 101] is to build the unified architecture for the closely related tasks. GLIP [47] formulates both the object detection and phrase grounding tasks as the word-region alignment problem. OneFormer [38] rules the universal image segmentation tasks with a single Transformer network. Unicorn [111] proposes the designs of target priors to tackle four tracking tasks. However, these methods with unified architecture either focus on the image domain or only consider the visualonly tasks. We aim to bridge the gap by building the unified model for the reference-based object segmentation tasks.

##### 2.2. Task-specific Object Segmentation

Referring Image Segmentation. The objective of RIS [33] is to generate a pixel-level mask for the target object described by a language description in an image. Prior research has primarily focused on the multi-modal feature interaction techniques, either employing attention mechanism in CNNs [121, 8, 119, 36, 35, 39, 114] or using multimodal Transformers [22, 43, 99]. As RIS is closely related to referring expression comprehension (REC), which aims to predict the bounding box of the referred object, some works [66, 49, 130] also explore the unified frameworks that can accomplish these two tasks simultaneously.

Few-shot Segmentation. FSS task aims to provide a mask prediction for a query image with a handful of support samples. The early methods mainly utilize the prototype-based networks [93, 113, 61], where the network computes the class prototype by a masked average pooling from support

Current Frame

Output

[Figure 10]

[Figure 11]

Visual Encoder

Pixel Decoder

𝛾, 𝛽,𝛼

|Scale, Shift, Gate|
|---|

| | |
|---|---|
|Multi-Head Cross-Attention| |

|[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>Mask Reference<br><br>Language Reference The largest sheep.| |
|---|---|
| | |

Q

|Reference Encoding|K, V|
|---|---|
| | |

Linear

Q K,V

Transformer Encoder

Transformer Decoder

| | |
|---|---|
|poo|ling|
| | |

UniFusion

Linear Linear

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Visual Features

Reference Features

(a) Main pipeline of our framework (b) UniFusion module

- Figure 2: Illustration of (a) the overall framework of UniRef++. For sake of clarity, we omit the visualization of prediction heads which are on top of Transformer decoder. The core network (in blue) is shared for all tasks. (b) The details of UniFusion module. The reference features come from the language or mask references.

ory networks [107, 15, 41] or proposing reliable memory readout strategies [83, 16, 116, 118]. These previous works view the VOS task as the pixel-level binary classification problem, lacking the understanding of object. Different from theirs, we tackle the VOS as the instance segmentation problem.

set and then refine the query image features with the prototype information. As these methods have significant information loss due to the pooling operation, the correlationbased methods [32, 108, 34, 92] have been proposed to model the densely pixel-level relationships between query and support images.

Referring Video Object Segmentation. RVOS can be considered as an extension of RIS in the video domain. Some previous methods process the video frames independently [119, 128, 24, 46] or simply adopt 3D CNNs [91, 69, 90] to extract the spatio-temporal features for a video clip. Recently, state-of-the-art methods [6, 102] are based on query-based Transformers and process the videos in an offline fashion. They receive the whole video as input and employ the queries to segment and track the target object simultaneously. However, such methods are not suitable for long videos or ongoing videos. In contrast to these works, our UniRef++ belongs to the online method and can utilize the historical information for mask propagation, which can ensure the temporal consistency of target object and improve segmentation accuracy.

#### 3. Method

##### 3.1. Overview

We present UniRef++, a simple and unified architecture that can segment arbitrary objects with the given references in images and videos. Conceptually, it allows us to train a single network on all related benchmarks and simultaneously solves the aforementioned tasks (RIS, FSS, RVOS, VOS).

The overall architecture of UniRef++ is illustrated in Fig. 2a. Our framework consists of a visual encoder, two reference encoders (for text and mask, respectively), a proposed UniFusion module and a transformer-based object detector. Given an image I ∈ RH×W×3 and the corresponding references, we first use the visual encoder EncV

Video Object Segmentation. Given a video with the target mask annotations in the first frame, the VOS algorithms need to propagate the masks to the entire video. The previous approaches could be broadly categorized into two groups: (i) Template-based methods. These works [88, 95, 89, 81, 12] regard the annotated frame as template and investigate how to fuse the template information into the current frame. (ii) Memory-based methods. The pioneering work STM [72] leverages a memory network to embed past-frame predictions and learns the space-time pixel-level correspondence on the memory to propagate the mask information. This type of works has achieved significant improvement and dominated the VOS community. The subsequent works mainly focus on improving memorized embeddings [115, 117, 48, 53, 110, 68], designing novel mem-

to extract the multi-scale features F = {Fℓ}4ℓ=1 of current image, where ℓ denotes the level index of the hier-

archical visual features, with the spatial strides from 4 to 32. Then the reference encoders are applied to encode the reference information, followed by the UniFusion module to inject the information into the visual features. Finally, the network can produce a binary mask for the target object m ∈ RH×W via a unified transformer-based architecture.

In the following subsections, we are going to details of UniRef++ by introducing the reference encoding (Sec. 3.2), the multi-scale UniFusion module (Sec. 3.3), a unified encoder-decoder architecture (Sec. 3.4), the training and inference process of UniRef++ (Sec. 3.5).

##### 3.2. Reference Encoding

In this part, we introduce how to encode the reference information for the four reference-based tasks. Before that, we would like to clarify that the only task-specific design of UniRef++ is to use different reference encoders (text encoder EncT and mask encoder EncM) for processing different modalities.

Few-shot Segmentation and Video Object Segmentation. For FSS and VOS tasks, the mask annotation for the reference image is provided as the reference. And the network needs to propagate the mask throughout the video. Inspired by the spirit of STCN [16] that computing the similarity of two frames for once, we use the same visual encoder EncV to extract the hierarchical visual features FfV = FV,ℓf of reference frame Iref. Then, a lightweight mask encoder (e.g., ResNet18 [30]) receives the reference frame Iref, object mask annotation mo and the encoded frame features FfV to generate the multi-scale mask features FmV = FV,ℓm for the target object in reference frame. Here, ℓ = 2,3,4 for FV,ℓf and FV,ℓm .

FfV = EncV (Iref) (1)

FmV = EncM(Iref,mo,FfV ) (2) Referring Image Segmentation. The reference for RIS task is the language description T. To encode such linguistic information, we apply an off-the-shelf text encoder (e.g., BERT [20] or RoBERTa [60]) to extract the language features FT ∈ RL×C, where L is the sentence length and C is the channel dimension.

FT = EncT(T) (3)

Referring Video Object Segmentation. RVOS requires the model to not only understand the language description, but also track the referred object in the whole video. To this end, we encode both linguistic and visual information for this task. Similarly, we use Eq. 3 to extract language feature and apply Eq. 1 and Eq. 2 for mask features encoding. It should be noted that the mask annotation is available during training. And we use the predicted mask in the previous frame as the visual reference during inference.

##### 3.3. Multi-scale UniFusion Module

After the reference encoding, a natural question is raised: How to inject the reference information into the network? In this subsection, we introduce our proposed multi-scale UniFusion module for the reference information injection.

The details of UniFusion module is illlustrated in Fig. 2b. We fuse the visual features F and the reference features in a hierarchical manner. For simplicity, we take the ℓ-th (ℓ = 2,3,4) visual level for illustration. The UniFusion module receives three inputs: the ℓ-th level visual feature Fℓ

of current image and the corresponding key and value embeddings (Frk and Frv) from reference features. For mask reference, Frk = FfV , Frv = FmV . For language reference, Frk = Frv = FT. These inputs are first linearly projected and further reformulated as three vectors, namely Qℓ, Kℓ and Vℓ. We first perform the mutli-head cross-attention operation between these vectors. Then, reference features Frk are pooled and regressed to obtain the scale, shift and gate parameters γ,β,α, which are applied after the attention block. Finally, the output features are injected into the original visual features via residual connection. The process of UniFusion is represented as:

Oℓ = Attention(Qℓ,Kℓ,Vℓ) (4) γ,β,α = Linear(Pooling(Frk)) (5) Fℓ′ = Fℓ + α(Oℓ(1 + γ) + β) (6)

where Oℓ is the intermediate results after the attention operation. Fℓ′ is the final output of UniFusion. Notably, the UniFusion module shares the same parameters in all visual scales. We emphasize the distinguished characteristics of UniFusion are two-folds: (1) We implement the crossattention operation using FlashAttention [18, 17]. This leads to the high efficiency and low memory cost when computing on the dense feature maps. (2) Inspired by the adaLN-zero block from [74], the linear layer for regressing the scale, shift and gate parameters are zero-initialized. This helps network gradually learn the knowledge from references and make UniFusion easily plug in the pretrained object segmentation models.

Thanks to the unifying fusion format, the reference information in different tasks can be injected into the visual features using the same UniFusion module. For the FSS and VOS tasks, the reference features are from the annotated masks. And the reference is language description for RIS. We emphasize here, for RVOS task, both the language features and reference frame visual features are fused with the visual feature of current frame. In this fashion, the network can not only find the referred object by language, but also propagate the target mask across frames for tracking. This also unifies the paradigms of VOS and RVOS to the online pattern.

##### 3.4. Unified Architecture

The fused multi-scale visual features F′ = {Fℓ′}4ℓ=2 have discriminative representations in highlighting the specific target by reference. We next adopt a unified transformbased architecture to predict the target mask.

Transformer. We use the two-stage version DeformableDETR [131] as our object detector. It receives the fused hierarchical visual features F′ as input and perform multiscale deformable self-attention in the encoder. In the decoder, N object queries are iteratively refined over stacked

Table 1: Comparison with the state-of-the-art methods on three referring image segmentation (RIS) benchmarks. RN101 denotes ResNet-101 [30], WRN101 refers to Wide ResNet-101 [124], and DN53 denotes Darknet-53 [79].

|Method<br><br>| |Visual Backbone|Text<br><br>Encoder<br><br>|RefCOCO| | |RefCOCO+| | |RefCOCOg| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | |val|test A<br><br>|test B<br><br>|val|test A<br><br>|test B|val-u<br><br>|test-u|
|oIoU|CMSA [119]<br><br>STEP [8]<br><br>BRINet [35]<br><br>CMPC [36] LSCM [37] CMPC+ [58]<br><br>MCN [66]<br><br>EFN [27]<br><br>LTS [39]<br><br>ReSTR [43] LAVT [114]<br><br>|RN101 RN101 RN101 RN101 RN101 RN101 DN53<br><br>WRN101<br><br>DN53 ViT-B Swin-B<br><br>|LSTM Bi-LSTM LSTM LSTM LSTM LSTM Bi-GRU Bi-GRU Bi-GRU Transformer BERT-base|58.32 60.04 60.98 61.36 61.47 62.47 62.44 62.76 65.43 67.22 72.73<br><br>|60.61 63.46 62.99 64.53 64.99 65.08 64.20 65.69 67.76 69.30 75.82<br><br>|55.09 57.97 59.21 59.64 59.55 60.82 59.71 59.67 63.08 64.45 68.79|43.76 48.19 48.17 49.56 49.34 50.25 50.62 51.50 54.21 55.78 62.14<br><br>|47.60 52.33 52.32 53.44 53.12 54.04 54.99 55.24 58.32 60.44 68.38<br><br>|37.89 40.41 42.11 43.23 43.50 43.47 44.69 43.01 48.02 48.27 55.10|49.22 54.40 61.24<br><br>|49.40 54.25 62.09|
| |UniRef-R50 UniRef-L UniRef++-R50 UniRef++-L<br><br>|RN50 Swin-L RN50 Swin-L<br><br>|RoBERTa-base RoBERTa-base BERT-base BERT-base<br><br>|75.04 79.79 75.63 79.13<br><br>|77.28 81.81 78.75 82.21<br><br>|72.43 77.02 72.91 77.45<br><br>|63.25 69.26 63.29 68.43<br><br>|68.12 74.11 68.68 73.98<br><br>|55.56 63.14 56.26 61.45<br><br>|66.96 73.04 68.38 71.37<br><br>|68.77 73.36 69.71 72.84<br><br>|
|mIoU|VLT [22]<br><br>CRIS [99]<br><br>SeqTR [130]<br><br>RefTr [49]<br><br>LAVT [114]<br><br>PolyFormer-L [57]<br><br>|DN53 RN101 DN53 RN101 Swin-B Swin-L<br><br>|Bi-GRU GPT-2 Bi-GRU BERT-base BERT-base BERT-base|65.65 70.47 71.70 74.34 74.46 76.94<br><br>|68.29 73.18 73.31 76.77 76.89 78.49|62.73 66.10 69.82 70.87 70.94 74.83<br><br>|55.50 62.27 63.04 66.75 65.81 72.15<br><br>|59.20 68.06 66.73 70.58 70.97 75.71<br><br>|49.36 53.68 58.97 59.40 59.23 66.73|52.99 59.87 64.69 66.63 63.34 71.15<br><br>|56.65 60.36 65.74 67.39 63.62 71.17<br><br>|
| |UniRef-R50 UniRef-L UniRef++-R50 UniRef++-L<br><br>|RN50 Swin-L RN50 Swin-L<br><br>|RoBERTa-base RoBERTa-base BERT-base BERT-base<br><br>|78.14 81.90 78.97 81.84<br><br>|80.09 83.03 81.29 83.48<br><br>|75.94 79.61 76.51 80.44<br><br>|69.09 73.81 69.53 74.02<br><br>|73.64 78.30 74.93 78.04<br><br>|62.62 68.33 63.35 68.31<br><br>|71.76 76.65 73.37 76.00<br><br>|73.10 77.09 74.16 77.20<br><br>|

oIoU

decoder layers and converted into the query representations Qobj ∈ RN×C finally. Three predictions heads (class head, box head and mask head) are further built on top of the decoder to predict the object scores S ∈ RN×1, boxes B ∈ RN×4 and mask dynamic convolution [86, 14, 13] kernel parameters G = {gi}Ni=1, respectively.

Mask Decoder. We take the output features (from strides 8 to 32) of Transformer encoder and hierarchically fuse them in a FPN-like [54, 104] manner. The feature map with 4× strides of backbone, namely F1, is also added in this process. This is helpful for preserving the reference-agnostic and fine-grained information of images. Consequently, we obtain the high-resolution mask features Fseg ∈ RH4 ×W4 ×C. Finally, the masks of target object are generated by performing dynamic convolution between Fseg and G:

mi = Upsample DynamicConv(Fseg,gi) , i = 1,...,N

(7)

During inference, we choose the mask with the highest score as the final result m for the target object. Notably, we empirically find that using more object queries leads to higher performance, despite that one object query is sufficient for the reference-based tasks.

##### 3.5. Training and Inference

We train UniRef++ on all related benchmarks of reference-based object segmentation tasks (RIS, FSS, RVOS, VOS). The model with the same weights can perform different tasks at run-time by specifying the references.

Training. The network predicts N predictions of object scores, box coordinates and segmentation masks, where the object score indicates whether the object is visible in current frame. During training, we apply the set prediction loss [7, 131, 85] on these predictions. There is only one ground-truth for the reference-based object segmentation tasks. We assign multiple predictions to the ground-truth by selecting the top-k predictions with the least cost according to an optimal transport method [28, 29, 105]. The matching cost is formulated as:

C = λcls · Ccls + λL1 · CL1 + λgiou · Cgiou (8)

where Ccls is the focal loss [55]. The box losses include the widely-used ℓ1 loss and generalized IoU (GIoU) loss [80]. The top-k predictions with the least cost are assigned as

positive samples and others as negatives. UniRef++ is optimized by minimizing the following loss function:

L = λcls · Lcls + λL1 · LL1 + λgiou · Lgiou + λmask · Lmask + λdice · Ldice

(9)

where the class loss and boxes losses are the same as those in Eq. 8. The mask-related losses contain the mask binary cross-entropy loss and DICE loss [70].

Inference. For RIS and FSS, we directly output the predicted mask of the query that has the highest score. For RVOS and VOS, our method infers the video in a frame-byframe online fashion without the complex post-processing. Specifically, for the current frame, the network uses the corresponding references to produce the mask of target object. The mask would be output if its object score is higher than a pre-determined threshold σ. Otherwise the output mask values are all set to zeros. To handle the videos that contain multiple objects, we adopt the soft-aggregation method commonly used in prior works [72, 16].

#### 4. Experiments

In this section, we conduct comprehensive experiments on all reference-based tasks (RIS, FSS, RVOS and VOS) to evaluate the effectiveness of our proposed UniRef++. The experimental settings will be first introduced in Sec. 4.1. We then compare UniRef++ with state-of-the-art methods on the prevalent benchmarks in Sec. 4.2. The ablation studies are presented in Sec. 4.3.

##### 4.1. Experimental Setup

Datasets. We evaluate our UniRef++ on four tasks to verify its effectiveness. The specific datasets leveraged in this work for evaluation are presented in the following. (i) RIS: RefCOCO [122] consists of 142,209 language descriptions for 50,000 objects in 19,994 images. RefCOCO+ [122] has 141,564 expressions for 49,856 objects in 19,992 images. RefCOCOg [67] includes 85,474 referring expressions for 54,822 objects in 26,711 images. And we use the UMD split for RefCOCOg [67]. (ii) FSS: FSS-1000 [50] is a large-scale dataset for FSS task. It contains 10,000 images from 1,000 classes. (iii) RVOS: Ref-Youtube-VOS [82] is a large-scale referring video object segmentation dataset which contains 3,978 videos with around 15k langauge descriptions. Ref-DAVIS17 [42] provides the referring expressions for each object in DAVIS17 [75]. It contains 90 videos in total. (iv) VOS: Youtube-VOS1 [109] is the popular benchmark for video object segmentation. There are 474 and 507 videos in the validation set for 2018 and 2019 version, respectively. LVOS [31] is a long-term video object segmentation benchmark consisting of 220 videos. The

1Youtube-VOS and Ref-Youtube-VOS are evaluated using the official server https://youtube-vos.org/.

Table 2: Comparison with the state-of-the-art methods on FSS-1000 validation set.

|Method Venue<br><br>|mIoU 1-shot 5-shot|
|---|---|
|Specialist Models DAN [92] ECCV’20 HSNet [71] ICCV’21 SSP [26] ECCV’22 VAT [32] ECCV’22 DACM [108] ECCV’22|85.2 88.1<br><br>86.5 88.5<br><br>87.3 88.6<br><br><br>90.3 90.8 90.8 91.7<br><br>|
|Generalist Models Painter [97] CVPR’23 SegGPT [98] ICCV’23 UniRef++-R50 this work UniRef++-L this work<br><br>|61.7 62.3 85.6 89.3 79.1 85.5 85.4 89.9<br><br>|

videos in LVOS have an average duration of 1.59 minutes, and the videos in Youtube-VOS last 6 seconds on average. MOSE [21] is a newly proposed dataset for evaluating VOS algorithms in complex scenes, such as occlusion and disappearance. It have 2,149 videos clips and 5,200 objects from 36 categories, with a total of 431,725 annotated masks.

Implementation Details. We experiment with two prevalent backbones as our visual encoder: ResNet50 [30] and Swin Transformer-Large [62]. The text encoder is selected as BERT-base [20] and we set the max length of sentences as 77. The Transformer architecture has 6 encoders and 6 decoders with the channel dimension of 256. The number of object queries is set as 300 by default. The loss coefficients in Eq. 9 are set as λcls = 2.0, λcls = 2.0, λL1 = 5.0, λmask = 2.0 and λdice = 5.0, respectively.

The entire training process includes three sequential stages, in which pretrained weights from the previous stage are loaded and used for further training. (1) Objects365 [84] pretraining. In this stage, we do not incorporate the UniFusion module but aim to learn a strong object detector for a massive of objects. Due to absence of mask annotation in the dataset, we also apply the BoxInst [87] loss for mask supervision. (2) Image-level training. We first combine the training set of RefCOCO/+/g to train the full network, and then train the network for RIS and FSS tasks. (3) Video-level training. At this stage, we randomly sample two frames from a video, where the first frame is considered as the reference frame. To avoid the knowledge forgetting for the RIS task, we also generate pseudo videos for RefCOCO/+/g. The network is jointly trained on all the related benchmarks, including RefCOCO/+/g [122, 67], Ref-YoutubeVOS [82], Ref-DAVIS17 [42], COCO [56], Youtube-VOS19 [109], OVIS [76] and LVOS [31].

In this work, we use Pytorch toolkit [73] to conduct all experiments on NVIDIA A100 GPUs. Unless otherwise

Table 3: Comparison with the state-of-the-art methods for referring video object segmentation (RVOS). † and ‡ denote the model uses the tiny and base version of Video Swin Transformer [63] as visual encoders, respectively.

Visual

Method

Encoder J &F J J Ref-Youtube-VOS

|CMSA [119] URVOS [82] YOFO [46] ReferFormer [102] UniRef-R50 UniRef++-R50<br><br>|ResNet-50<br><br>|36.4 34.8 38.1<br><br>47.2 45.3 49.2<br>48.6 47.5 49.7 58.7 57.4 60.1<br><br><br>60.6 59.0 62.3<br><br>61.5 59.7 63.3<br><br><br>|
|---|---|---|
|PMINet + CFBI [25] CITD [52]<br><br>|Ensemble<br><br>|54.2 53.0 55.5 61.4 60.0 62.7<br><br>|
|MTTR† [6] VLT‡ [23] ReferFormer‡ [102]<br><br>|Video-Swin<br><br>|55.3 54.0 56.6<br><br>63.8 61.9 65.6<br>64.9 62.8 67.0<br>|
|ReferFormer [102] UniRef-L UniRef++-L<br><br>|Swin-L<br><br>|64.2 62.3 66.2 67.4 65.5 69.2 66.9 64.8 69.0<br><br>|

Ref-DAVIS17

|CMSA [119] URVOS [82] YOFO [46] ReferFormer [102] UniRef-R50 UniRef++-R50<br><br>|ResNet-50<br><br>|40.2 36.9 43.5 51.5 47.3 56.0 54.4 50.1 58.7 58.5 55.8 61.3 63.5 60.0 67.0 62.5 58.7 66.3<br><br>|
|---|---|---|
|VLT‡ [23] ReferFormer‡ [102]<br><br>|Video-Swin<br><br>|61.6 58.9 64.3 61.1 58.1 64.1|
|PolyFormer-L [57] ReferFormer [102] UniRef-L<br><br>UniRef++-L|Swin-L<br><br>|61.5 57.2 65.8 60.5 57.6 63.4<br><br>66.3 62.9 69.7<br><br>67.2 63.4 70.9<br><br><br>|

stated, we use 4 × 8 A100 GPUs for the objects365 pretraining and 2 × 8 GPUs for the following image-level and video-level training. We adopt AdamW [64] as the optimizer and set the batch size as 2 for each GPU. We refer the readers to Appendix for more implementation details.

##### 4.2. Quantitative Results

We employed ResNet50 [30] and Swin TransformerLarge [62] as visual backbones in our experiments, denoted as UniRef++-R50 and UniRef++-L, respectively. For each version, all results are computed with one suit of weights.

Referring Image Segmentation. We compare UniRef++ with state-of-the-art methods in Table 1. Following the previous works, we use both overall intersection-over-union (oIoU) and mean intersection-over-union (mIoU) as the evaluation metrics. It can be seen that UniRef++ with

Table 4: Comparison with the state-of-the-art methods on LVOS [31] and MOSE [21] validation set. These methods are all not trained on the MOSE dataset.

LVOS val MOSE val Method J &F J F J &F J F AFB-URR [53] 34.8 31.3 38.2 - - CFBI [115] 50.0 45.0 55.1 - - RDE [48] 53.7 48.3 59.2 46.8 42.4 51.3 STCN [16] 45.8 41.1 50.5 52.5 48.5 56.6 AOT [116] 59.4 53.6 65.2 58.4 54.3 62.6 XMem [15] 50.0 45.5 54.4 56.3 52.1 60.6 DeAOT [118] - - - 59.0 54.6 63.4

UniRef-R50 55.7 51.5 60.0 - - UniRef-L 60.9 57.2 64.6 - - UniRef++-R50 60.1 55.8 64.3 54.7 51.3 58.2 UniRef++-L 67.2 62.9 71.5 59.0 55.7 62.3

ResNet-50 backbone surpasses the previous methods on nearly all splits and it also has significant improvement over UniRef [103]. When equipped with Swin-Large backbone, UniRef++ sets new SoTA performance on several splits. For example, our UniRef++-L has 4.99 mIoU performance gain over the SoTA method PolyFormer-L [57] on the testA split of RefCOCO. We hypothesize the reason for the similar performance between UniRef and UniRef++ is that UniRef experienced the Visual Genome pretraining, which makes it more friendly for the grounding tasks.

Few-shot Segmentation. Following [50, 32], we divide the 1,000 classes of FSS-1000 [50] dataset into 520, 240 and 240 classes, which are used for training, validation and testing, respectively. We evaluate our models on the validation set and report the results in Table 2. It is observed that our models significantly benefit from the few-shot samples and achieve comparable results with the state-of-the-art specialist models.

Referring Video Object Segmentation. For the RVOS task, we use the region jaccard J , boundary accuracy F and the average score J &F as the evaluation metrics. The comparison of UniRef++ and state-of-the-art methods are presented in Table 3. We empirically find that solely using the language as reference would lead to better performance on Ref-DAVIS17, possibly due to salient objects and simple scenes within the dataset. According to Table 3, it indicates that UniRef++ has significant improvement over all the previous methods on the two datasets. UniRef++ with ResNet50 visual encoder achieves the state-of-the-art performance and has notable 2.8 J &F gain over ReferFormer [102] on Ref-YoutubeVOS. On Ref-DAVIS17, UniRef++ sets new state-of-the-art performance of 67.2 J &F, surpassing the previous best method VLT [22] by 5.4 J &F.

Table 5: Comparison with the state-of-the-art methods on three video object segmentation (VOS) benchmarks.

|Youtube-VOS 2018 val|Youtube-VOS 2019 val<br><br>|DAVIS17 val|
|---|---|---|
|G Js Fs Ju Fu|G Js Fs Ju Fu<br><br>|J &F J F|

Method

###### Memory-based Methods

STM [72] 79.4 79.7 84.2 72.8 80.9 - - - - - 81.8 79.2 84.3 AFB-URR [53] 79.6 78.8 83.1 74.1 82.6 - - - - - 76.9 74.4 79.3 CFBI [115] 81.4 81.1 85.8 75.3 83.4 81.0 80.6 85.1 75.2 83.0 81.9 79.1 84.6 RDE [48] - - - - - 81.9 81.1 85.5 76.2 84.8 84.2 80.8 87.5 STCN [16] 83.0 81.9 86.5 77.9 85.7 82.7 81.1 85.4 78.2 85.9 85.4 82.2 88.6 AOT-B [116] 83.5 82.6 87.5 77.7 86.0 83.3 82.4 87.1 77.8 86.0 82.5 79.7 85.2 AOT-L [116] 83.8 82.9 87.9 77.7 86.5 83.7 82.8 87.5 78.0 86.7 83.8 81.1 86.4 XMem [15] 85.7 84.6 89.3 80.2 88.7 85.5 84.3 88.6 80.3 88.6 86.2 82.9 89.5 DeAOT [118] 86.0 84.9 89.9 80.4 88.7 85.9 84.6 89.4 80.8 88.9 85.2 82.2 88.2

Non-memory Methods FRTM [81] 72.1 72.3 76.2 65.9 74.1 - - - - - 76.7 73.9 79.6 LWL [4] 81.5 80.4 84.9 76.4 84.4 81.0 79.6 83.8 76.4 84.2 70.6 67.9 73.3 UniRef-R50 81.4 81.6 85.9 75.6 82.4 81.2 80.8 84.9 76.2 83.0 - - UniRef-L 82.6 83.2 87.5 76.2 83.7 82.7 82.9 86.9 76.8 84.1 - - UniRef++-R50 81.9 82.3 86.8 75.9 82.6 81.9 81.9 86.2 76.5 83.2 81.5 78.1 87.6 UniRef++-L 83.2 83.8 88.5 76.8 83.8 83.0 83.1 87.7 77.3 84.1 83.9 80.8 89.8

Table 6: Ablation experiments of UniRef. We evaluate our model on the RefCOCO, FSS-1000, Ref-Youtube-VOS and Youtube-VOS2018 validation set, respectively. Our default settings are marked in gray .

(a) Task-specific training.

(b) Parameter-sharing for UniFusion.

(c) Query number.

RIS FSS RVOS VOS oIoU mIoU J &F G

Training

Signle-task 72.1 81.8 58.3 81.9 Multi-task 74.7 80.3 60.1 81.9

RIS FSS RVOS VOS oIoU mIoU J &F G

Tasks Levels

✓ 73.6 64.1 60.7 81.9 ✓ 74.9 75.2 61.5 80.9 ✓ ✓ 74.7 80.3 60.1 81.9

RIS FSS RVOS VOS oIoU mIoU J &F G

Query

1 72.1 75.5 58.4 79.0 100 75.1 77.7 60.6 81.9 300 74.7 80.3 60.1 81.9

Video Object Segmentation. To evaluate the performance on Youtube-VOS [109], region jaccard J and countour accuracy F are computed for ”seen” and ”unseen” classes separately, denoted by subscripts s and u. G is the average J &F for both seen and unseen classes. For DAVIS17 [75], LVOS [31] and MOSE [21] datasets, J &F, J and F are adopted as the evaluation metrics. We provide a comprehensive comparison of different methods on the three classic datasets in Table 5. Our proposed UniRef++-L outperforms non-memory-based methods, achieving the best results with 83.2/83.0 G on the Youtube-VOS 2018/2019. Our models also display competitive results on DAVIS17 even if the dataset is not included during training. Unlike memorybased methods [72, 16], our approach does not rely on predicted masks from past frames, making it more memoryefficient and suitable for long videos. We further demonstrate the advantages of our model for handling long videos in complex scenes in Table 4. In this situation, UniRef++ have obvious performance improvement compared with the classic memory-based methods.

##### 4.3. Ablation Study

The ablation experiments are evaluated on RefCOCO [122], FSS-1000 [50], Ref-Youtube-VOS [82] and Youtube-VOS2018 [109] validation set to study UniRef++ in detail. Unless otherwise stated, we use the ResNet-50 as visual backbone and only conduct the image-level training and video-level training for quick validation.

Task-specific Training. In Table 6a, we compare results of single training and multi-task joint training. Single-task models are trained on the corresponding datasets, while multi-task models are trained jointly on all datasets. The ablation results indicate that multi-task learning offers significant benefits for both RIS and RVOS. Specifically, RefYoutube-VOS achieves 60.1 J &F, which is 1.8 points higher than the task-specific model. This can attribute to the fact that the jointly trained model is better at learning mask propagation through VOS training. For FSS, the performance of multi-task model is slightly lower than the singletask model. This is due to that we didn’t use the FSS dataset

a cow is standing a fair distance behind the brown cow which is moving forward. L

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

80%

20% 40%

0%

60%

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

L + M

80% a fire truck pulling out with another behind it.

0% 20%

40%

60%

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

L

80%

0%

20%

60%

40%

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

L + M

60%

0%

40%

20%

80%

- Figure 3: Comparison of the use of mask references for RVOS. The red masks highlight the predicted objects. The percentages indicate the relative temporal position of each frame in the video. ‘L’ and ‘M’ represent language and mask references, respectively.

during video-level training. In summary, multi-task joint training improves the performance of task-specific models and saves a significant number of parameters.

Parameter-sharing for UniFusion Module. Our UniRef++ leverages a parameter-sharing UniFusion module to fuse information from both mask and language references for multi-scale visual features. In Table 6b, we present ablation experiments on two weight-sharing variants for UniFusion module. As shown in the table, when we use different UniFusion modules for different tasks, the performance of FSS drops significantly because its data scale is relatively small. When applying different UniFusion for different visual levels, G metric on Youtube-VOS2018 decreases from 81.9 to 80.9. This suggests that a single parameter-sharing UniFusion module is more effective in learning frame similarities for multi-scale visual features.

Query Number. Reference-based tasks have one specific target for each reference, making it possible to complete the tasks with just one query. In Table 6c, we present an ablation study on the number of queries to investigate its impact on performance. As observed, increasing the number of queries from 1 to 100 leads to higher performance. This is reasonable as the model can have more candidates to find the target object, which is particularly helpful in complex scenes where many similar objects co-exist. And we also observe that using 100 and 300 queries would obtain the similar results.

Table 7: Ablation on the references used for RVOS. In the table, ‘Lang’ means language. Results are evaluated on Ref-Youtube-VOS validation set.

###### ResNet-50 Swin-L J &F J F J &F J F

Reference

Lang 60.4 58.9 61.9 65.5 63.6 67.4 Lang + Mask 61.5 59.7 63.3 66.9 64.8 69.0

Does Mask Reference Help RVOS? For the RVOS task, UniRef++ processes the videos in an online-fashion. Specifically, we not only use the language reference as guidance, but also leverage the predicted masks in the previous frames for mask propagation. To study the effectiveness of the mask reference, we use our final version models and provide the ablation results in Table 7. As illustrated in the table, when additionally using the mask references, the model gets 1.1 and 1.4 J &F improvement for UniRef++R50 and UniRef++-L, respectively. This evidently proves that the mask propagation helps the model to achieve temporal consistency for the target object.

##### 4.4. Qualitative Results

In order to demonstrate the effectiveness of employing mask references in RVOS, we present the qualitative results in Figure 3. As depicted in the first example, utilizing language references alone struggles in identifying the referred

Table 8: Comparison with other SAM [44]-variant methods. Ref-YT, Ref-D mean Ref-Youtube-VOS and Ref-DAVIS17 datasets. YT-18, YT-19 and D-17 are the abbreations for Youtube-VOS-18, Youtube-VOS-19 and DAVIS17, respectively.

|Task Dataset Metric|RIS RefCOCO val oIoU mIoU<br><br>|FSS FSS-1000 1-shot 5-shot<br><br>|RVOS Ref-YT Ref-D<br><br>J &F J &F|VOS YT-18 YT-19 D-17<br><br>G G J &F|
|---|---|---|---|---|
|ReferSAM [106] PerSAM [127] PerSAM-F [127] RefSAM [51] SAM-PT [77]<br><br>|64.6 71.1<br><br>- -<br><br>- -<br><br>- -<br><br>- -<br><br><br>|- 81.6 86.3 -<br><br>- -<br><br>- -<br><br><br>|- -<br><br>- -<br><br>- -<br><br><br>55.1 66.1 - -<br><br>|- - -<br>- - -<br>- - -<br>- - -<br><br><br>67.5 - 76.6|
|SAM [44] + UniRef<br><br>|65.9 70.4<br><br>|77.8 84.2|52.4 61.2<br><br>|73.4 72.8 78.4|

and video-level training as described in Sec. 4.1. The experiment is highly efficient, being able to be completed within 20 hours using 8 A100s.

|[Figure 35]<br><br>❄ frozen trainable<br><br>[Figure 36]|
|---|

mask

We compare the results with other SAM-variant methods in Table 8. It should be noted that other methods are finetuned on the corresponding datasets while our model has one suit of weights. For FSS, our model lags behind the PerSAM [127] under 1-shot setting since FSS-1000 dataset is not included during video-level training. But the performance gap could be eliminated with few samples (5-shot). From the table, we show that combining SAM [44] with UniFusion could achieve satisfactory across the referencebased tasks.

[Figure 37]

lightweight mask decoder

| | |
|---|---|
|UniFusion<br><br>[Figure 38]| |

|image encoder<br><br>[Figure 39]<br><br>❄|
|---|

reference encoder

[Figure 40]

#### 6. Conclusion

We present UniRef++, a unified model for four reference-based object segmentation tasks (RIS, FSS, RVOS and VOS). By introducing a UniFusion module to incorporate different types of references, our model can flexibly perform multi-tasks at run-time by specifying the corresponding references and achieves superior performance with a single network. We also show that UniFusion could be play as the plug-in component for the foundation models (e.g., SAM [44]) for efficient finetuning.

language / mask image

- Figure 4: Our proposed UniFusion module plays as the plug-in component for SAM [44]. Analogously, UniFusion could be also easily plugged in other object segmentation foundation models.

object in a complex scene with multiple similar objects. By integrating mask references, the network can leverage mask propagation to accurately track the target object. This figure illustrates the efficacy of incorporating mask references in RVOS for improving temporal consistency of target object. We also show more visualization examples for VOS and RVOS in the Appendix.

#### Acknowledgements

This paper is partially supported by the National Key R&D Program of China No.2022ZD0161000 and the General Research Fund of Hong Kong No.17200622. The paper is supported in part by the National Natural Science Foundation of China under grant No.62293540, 62293542, U1903215 and the Fundamental Research Funds for the Central Universities No.DUT22ZD210.

#### 5. Inserting UniFusion into SAM

At the core of UniRef++ is the UniFusion module for injecting the reference information into the network. To explore its applicability as a plug-in component, we insert the UniFusion module into the advanced foundation model SAM [44]. We keep the heavy image encoder frozen and train the lightweight reference encoders, UniFusion and mask decoder. The network goes through the image-level

#### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022. 2
- [2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015. 2
- [3] Ali Athar, Alexander Hermans, Jonathon Luiten, Deva Ramanan, and Bastian Leibe. Tarvis: A unified approach for target-based video segmentation. arXiv preprint arXiv:2301.02657, 2023. 2
- [4] Goutam Bhat, Felix J¨aremo Lawin, Martin Danelljan, Andreas Robinson, Michael Felsberg, Luc Van Gool, and Radu Timofte. Learning what to learn for video object segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 777–794. Springer, 2020. 8
- [5] Deblina Bhattacharjee, Tong Zhang, Sabine S¨usstrunk, and Mathieu Salzmann. Mult: an end-to-end multitask learning transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12031– 12041, 2022. 2
- [6] Adam Botach, Evgenii Zheltonozhskii, and Chaim Baskin. End-to-end referring video object segmentation with multimodal transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4985–4995, 2022. 2, 3, 7
- [7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 213–229. Springer, 2020. 5
- [8] Ding-Jie Chen, Songhao Jia, Yi-Chen Lo, Hwann-Tzong Chen, and Tyng-Luh Liu. See-through-text grouping for referring image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7454–7463, 2019. 2, 5
- [9] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. arXiv preprint arXiv:2109.10852,

2021. 2

- [10] Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J Fleet, and Geoffrey Hinton. A unified sequence interface for vision tasks. arXiv preprint arXiv:2206.07669, 2022. 2
- [11] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 2
- [12] Xi Chen, Zuoxin Li, Ye Yuan, Gang Yu, Jianxin Shen, and Donglian Qi. State-aware tracker for real-time video object segmentation. In Proceedings of the IEEE/CVF Conference

- on Computer Vision and Pattern Recognition, pages 9384– 9393, 2020. 3
- [13] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1290–1299, 2022. 2, 5
- [14] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. Advances in Neural Information Processing Systems, 34:17864–17875, 2021. 5
- [15] Ho Kei Cheng and Alexander G Schwing. Xmem: Longterm video object segmentation with an atkinson-shiffrin memory model. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVIII, pages 640–658. Springer, 2022. 2, 3, 7, 8
- [16] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Rethinking space-time networks with improved memory coverage for efficient video object segmentation. Advances in Neural Information Processing Systems, 34:11781–11794,

2021. 2, 3, 4, 6, 7, 8

- [17] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. 2023. 4, 17
- [18] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022. 4, 17
- [19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 2
- [20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 4, 6, 17
- [21] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. Mose: A new dataset for video object segmentation in complex scenes. arXiv preprint arXiv:2302.01872, 2023. 6, 7, 8
- [22] Henghui Ding, Chang Liu, Suchen Wang, and Xudong Jiang. Vision-language transformer and query generation for referring segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16321–16330, 2021. 2, 5, 7
- [23] Henghui Ding, Chang Liu, Suchen Wang, and Xudong Jiang. Vlt: Vision-language transformer and query generation for referring segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. 7
- [24] Zihan Ding, Tianrui Hui, Junshi Huang, Xiaoming Wei, Jizhong Han, and Si Liu. Language-bridged spatialtemporal interaction for referring video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4964–4973,

2022. 3

- [25] Zihan Ding, Tianrui Hui, Shaofei Huang, Si Liu, Xuan Luo, Junshi Huang, and Xiaoming Wei. Progressive multimodal

- interaction network for referring video object segmentation. The 3rd Large-scale Video Object Segmentation Challenge, page 7, 2021. 7
- [26] Qi Fan, Wenjie Pei, Yu-Wing Tai, and Chi-Keung Tang. Self-support few-shot semantic segmentation. In European Conference on Computer Vision, pages 701–719. Springer,

2022. 6

- [27] Guang Feng, Zhiwei Hu, Lihe Zhang, and Huchuan Lu. Encoder fusion network with co-attention embedding for referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15506–15515, 2021. 5
- [28] Zheng Ge, Songtao Liu, Zeming Li, Osamu Yoshie, and Jian Sun. Ota: Optimal transport assignment for object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 303–312,

2021. 5

- [29] Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021. 5
- [30] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 4, 5, 6, 7, 17
- [31] Lingyi Hong, Wenchao Chen, Zhongying Liu, Wei Zhang, Pinxue Guo, Zhaoyu Chen, and Wenqiang Zhang. Lvos: A benchmark for long-term video object segmentation. arXiv preprint arXiv:2211.10181, 2022. 6, 7, 8, 17, 18
- [32] Sunghwan Hong, Seokju Cho, Jisu Nam, Stephen Lin, and Seungryong Kim. Cost aggregation with 4d convolutional swin transformer for few-shot segmentation. In European Conference on Computer Vision, pages 108–126. Springer,

2022. 2, 3, 6, 7

- [33] Ronghang Hu, Marcus Rohrbach, and Trevor Darrell. Segmentation from natural language expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pages 108–124. Springer, 2016. 2
- [34] Tao Hu, Pengwan Yang, Chiliang Zhang, Gang Yu, Yadong Mu, and Cees GM Snoek. Attention-based multi-context guiding for few-shot semantic segmentation. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8441–8448, 2019. 3
- [35] Zhiwei Hu, Guang Feng, Jiayu Sun, Lihe Zhang, and Huchuan Lu. Bi-directional relationship inferring network for referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4424–4433, 2020. 2, 5
- [36] Shaofei Huang, Tianrui Hui, Si Liu, Guanbin Li, Yunchao Wei, Jizhong Han, Luoqi Liu, and Bo Li. Referring image segmentation via cross-modal progressive comprehension. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10488–10497, 2020. 2, 5
- [37] Tianrui Hui, Si Liu, Shaofei Huang, Guanbin Li, Sansi Yu, Faxi Zhang, and Jizhong Han. Linguistic structure guided context modeling for referring image segmentation. In

- Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part X 16, pages 59–75. Springer, 2020. 5
- [38] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. arXiv preprint arXiv:2211.06220, 2022. 2
- [39] Ya Jing, Tao Kong, Wei Wang, Liang Wang, Lei Li, and Tieniu Tan. Locate then segment: A strong pipeline for referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9858–9867, 2021. 2, 5
- [40] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetrmodulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1780–1790, 2021. 2
- [41] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. arXiv e-prints, pages arXiv– 2310, 2023. 3
- [42] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part IV 14, pages 123–141. Springer, 2019. 1, 6
- [43] Namyup Kim, Dongwon Kim, Cuiling Lan, Wenjun Zeng, and Suha Kwak. Restr: Convolution-free referring image segmentation using transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18145–18154, 2022. 2, 5
- [44] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 10
- [45] Alexander Kolesnikov, Andr´e Susano Pinto, Lucas Beyer, Xiaohua Zhai, Jeremiah Harmsen, and Neil Houlsby. Uvim: A unified modeling approach for vision with learned guiding codes. arXiv preprint arXiv:2205.10337, 2022. 2
- [46] Dezhuang Li, Ruoqi Li, Lijun Wang, Yifan Wang, Jinqing Qi, Lu Zhang, Ting Liu, Qingquan Xu, and Huchuan Lu. You only infer once: Cross-modal meta-transfer for referring video object segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 1297–1305, 2022. 3, 7
- [47] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975, 2022. 2
- [48] Mingxing Li, Li Hu, Zhiwei Xiong, Bang Zhang, Pan Pan, and Dong Liu. Recurrent dynamic embedding for video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1332–1341, 2022. 3, 7, 8

- [49] Muchen Li and Leonid Sigal. Referring transformer: A one-step approach to multi-task visual grounding. Advances in neural information processing systems, 34:19652–19664, 2021. 2, 5
- [50] Xiang Li, Tianhan Wei, Yau Pun Chen, Yu-Wing Tai, and Chi-Keung Tang. Fss-1000: A 1000-class dataset for fewshot segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2869–2878, 2020. 6, 7, 8, 18
- [51] Yonglin Li, Jing Zhang, Xiao Teng, and Long Lan. Refsam: Efficiently adapting segmenting anything model for referring video object segmentation. arXiv preprint arXiv:2307.00997, 2023. 10
- [52] Chen Liang, Yu Wu, Tianfei Zhou, Wenguan Wang, Zongxin Yang, Yunchao Wei, and Yi Yang. Rethinking cross-modal interaction from a top-down perspective for referring video object segmentation. arXiv preprint arXiv:2106.01061, 2021. 7
- [53] Yongqing Liang, Xin Li, Navid Jafari, and Jim Chen. Video object segmentation with adaptive feature bank and uncertain-region refinement. Advances in Neural Information Processing Systems, 33:3430–3441, 2020. 3, 7, 8
- [54] Tsung-Yi Lin, Piotr Doll´ar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017. 5
- [55] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Doll´ar. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 5
- [56] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 6, 17, 18
- [57] Jiang Liu, Hui Ding, Zhaowei Cai, Yuting Zhang, Ravi Kumar Satzoda, Vijay Mahadevan, and R Manmatha. Polyformer: Referring image segmentation as sequential polygon generation. arXiv e-prints, pages arXiv–2302, 2023. 5, 7
- [58] Si Liu, Tianrui Hui, Shaofei Huang, Yunchao Wei, Bo Li, and Guanbin Li. Cross-modal progressive comprehension for referring segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):4761–4775,

2021. 5

- [59] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 2
- [60] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019. 4

- [61] Yongfei Liu, Xiangyi Zhang, Songyang Zhang, and Xuming He. Part-aware prototype network for few-shot semantic segmentation, 2020. 2
- [62] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 6, 7
- [63] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202–3211, 2022. 7
- [64] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [65] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. arXiv preprint arXiv:2206.08916, 2022. 2
- [66] Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng, and Rongrong Ji. Multitask collaborative network for joint referring expression comprehension and segmentation. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 10034–10043, 2020. 2, 5
- [67] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016. 6, 17, 18
- [68] Yunyao Mao, Ning Wang, Wengang Zhou, and Houqiang Li. Joint inductive and transductive learning for video object segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9670–9679,

2021. 3

- [69] Bruce McIntosh, Kevin Duarte, Yogesh S Rawat, and Mubarak Shah. Visual-textual capsule routing for textbased video segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9942–9951, 2020. 3
- [70] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016. 6
- [71] Juhong Min, Dahyun Kang, and Minsu Cho. Hypercorrelation squeeze for few-shot segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6941–6952, 2021. 6
- [72] Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. Video object segmentation using space-time memory networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9226–9235, 2019. 2, 3, 6, 8
- [73] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library.

Advances in neural information processing systems, 32,

2019. 6

- [74] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195– 4205, 2023. 4
- [75] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 1, 6, 8
- [76] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded video instance segmentation: A benchmark. International Journal of Computer Vision, 130(8):2022–2039, 2022. 6, 17, 18
- [77] Frano Rajiˇc, Lei Ke, Yu-Wing Tai, Chi-Keung Tang, Martin Danelljan, and Fisher Yu. Segment anything meets point tracking. arXiv preprint arXiv:2307.01197, 2023. 10
- [78] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In International conference on learning representations, 2016. 1
- [79] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018. 5
- [80] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 658–666,

2019. 5

- [81] Andreas Robinson, Felix Jaremo Lawin, Martin Danelljan, Fahad Shahbaz Khan, and Michael Felsberg. Learning fast and robust target models for video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7406–7415, 2020. 3, 8
- [82] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16, pages 208–223. Springer,

2020. 6, 7, 8, 18

- [83] Hongje Seong, Seoung Wug Oh, Joon-Young Lee, Seongwon Lee, Suhyeon Lee, and Euntai Kim. Hierarchical memory matching network for video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12889–12898, 2021. 3
- [84] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019. 6, 18
- [85] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al. Sparse r-cnn: End-to-end object detection with learnable proposals. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14454–14463, 2021. 5

- [86] Zhi Tian, Chunhua Shen, and Hao Chen. Conditional convolutions for instance segmentation. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 282–298. Springer, 2020. 5
- [87] Zhi Tian, Chunhua Shen, Xinlong Wang, and Hao Chen. Boxinst: High-performance instance segmentation with box annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5443–5452, 2021. 6
- [88] Paul Voigtlaender, Yuning Chai, Florian Schroff, Hartwig Adam, Bastian Leibe, and Liang-Chieh Chen. Feelvos: Fast end-to-end embedding learning for video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9481– 9490, 2019. 3
- [89] Paul Voigtlaender, Jonathon Luiten, Philip HS Torr, and Bastian Leibe. Siam r-cnn: Visual tracking by re-detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6578–6588, 2020. 3
- [90] Hao Wang, Cheng Deng, Fan Ma, and Yi Yang. Context modulated dynamic networks for actor and action video segmentation with language queries. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 12152–12159, 2020. 3
- [91] Hao Wang, Cheng Deng, Junchi Yan, and Dacheng Tao. Asymmetric cross-guided attention network for actor and action video segmentation from natural language query. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3939–3948, 2019. 3
- [92] Haochen Wang, Xudong Zhang, Yutao Hu, Yandan Yang, Xianbin Cao, and Xiantong Zhen. Few-shot semantic segmentation with democratic attention networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16, pages 730–746. Springer, 2020. 3, 6
- [93] Kaixin Wang, Jun Hao Liew, Yingtian Zou, Daquan Zhou, and Jiashi Feng. Panet: Few-shot image semantic segmentation with prototype alignment. In proceedings of the IEEE/CVF international conference on computer vision, pages 9197–9206, 2019. 2
- [94] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR, 2022. 2
- [95] Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, and Philip HS Torr. Fast online object tracking and segmentation: A unifying approach. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, pages 1328–1338, 2019. 3
- [96] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and visionlanguage tasks. arXiv preprint arXiv:2208.10442, 2022. 2

- [97] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023. 6
- [98] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Segmenting everything in context. arXiv preprint arXiv:2304.03284,

2023. 6

- [99] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. Cris: Clipdriven referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11686–11695, 2022. 2, 5
- [100] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV), pages 3–19, 2018. 17
- [101] Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, and Song Bai. General object foundation model for images and videos at scale. arXiv preprint arXiv:2312.09158,

2023. 2

- [102] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as queries for referring video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4974– 4984, 2022. 2, 3, 7
- [103] Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, and Ping Luo. Segment every reference object in spatial and temporal spaces. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2538–2550,

2023. 7

- [104] Junfeng Wu, Yi Jiang, Wenqing Zhang, Xiang Bai, and Song Bai. Seqformer: a frustratingly simple model for video instance segmentation. arXiv preprint arXiv:2112.08275, 2021. 5
- [105] Junfeng Wu, Qihao Liu, Yi Jiang, Song Bai, Alan Yuille, and Xiang Bai. In defense of online models for video instance segmentation. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVIII, pages 588–605. Springer,

2022. 5

- [106] Yuhang Xiao Xiao. Refersam. https://github.com/ mydcxiao/ReferSAM, 2023. 10
- [107] Haozhe Xie, Hongxun Yao, Shangchen Zhou, Shengping Zhang, and Wenxiu Sun. Efficient regional memory network for video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1286–1295, 2021. 3
- [108] Zhitong Xiong, Haopeng Li, and Xiao Xiang Zhu. Doubly deformable aggregation of covariance matrices for fewshot segmentation. In European Conference on Computer Vision, pages 133–150. Springer, 2022. 2, 3, 6
- [109] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327, 2018. 6, 8, 17, 18

- [110] Xiaohao Xu, Jinglu Wang, Xiao Li, and Yan Lu. Reliable propagation-correction modulation for video object segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 2946–2954, 2022. 3
- [111] Bin Yan, Yi Jiang, Peize Sun, Dong Wang, Zehuan Yuan, Ping Luo, and Huchuan Lu. Towards grand unification of object tracking. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXI, pages 733–751. Springer, 2022. 2
- [112] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15325–15336, 2023. 2
- [113] Boyu Yang, Chang Liu, Bohao Li, Jianbin Jiao, and Qixiang Ye. Prototype mixture models for few-shot semantic segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16, pages 763–778. Springer, 2020. 2
- [114] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18155–18165, 2022. 2, 5
- [115] Zongxin Yang, Yunchao Wei, and Yi Yang. Collaborative video object segmentation by foreground-background integration. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V, pages 332–348. Springer, 2020. 3, 7, 8
- [116] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. Advances in Neural Information Processing Systems, 34:2491–2502, 2021. 2, 3, 7, 8
- [117] Zongxin Yang, Yunchao Wei, and Yi Yang. Collaborative video object segmentation by multi-scale foregroundbackground integration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):4701–4712,

2021. 3

- [118] Zongxin Yang and Yi Yang. Decoupling features in hierarchical propagation for video object segmentation. arXiv preprint arXiv:2210.09782, 2022. 3, 7, 8
- [119] Linwei Ye, Mrigank Rochan, Zhi Liu, and Yang Wang. Cross-modal self-attention network for referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10502– 10511, 2019. 2, 3, 5, 7
- [120] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 2
- [121] Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet: Modular attention network for referring expression comprehension. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1307–1315, 2018. 2

- [122] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 1, 6, 8, 17, 18
- [123] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021. 2
- [124] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016. 5
- [125] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianwei Yang, and Lei Zhang. A simple framework for open-vocabulary segmentation and detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1020–1031, 2023. 2
- [126] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. In Advances in Neural Information Processing Systems, 2022. 2
- [127] Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Hao Dong, Peng Gao, and Hongsheng Li. Personalize segment anything model with one shot. arXiv preprint arXiv:2305.03048, 2023. 10
- [128] Wangbo Zhao, Kai Wang, Xiangxiang Chu, Fuzhao Xue, Xinchao Wang, and Yang You. Modeling motion with multi-modal features for text-based video segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11737–11746, 2022. 3
- [129] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Kr¨ahenb¨uhl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In Computer Vision– ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IX, pages 350–

368. Springer, 2022. 17

- [130] Chaoyang Zhu, Yiyi Zhou, Yunhang Shen, Gen Luo, Xingjia Pan, Mingbao Lin, Chao Chen, Liujuan Cao, Xiaoshuai Sun, and Rongrong Ji. Seqtr: A simple yet universal network for visual grounding. In Computer Vision– ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXV, pages 598–

615. Springer, 2022. 2, 5

- [131] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020. 4, 5
- [132] Xizhou Zhu, Jinguo Zhu, Hao Li, Xiaoshi Wu, Hongsheng Li, Xiaohua Wang, and Jifeng Dai. Uni-perceiver: Pretraining unified architecture for generic perception for zeroshot and few-shot tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16804–16815, 2022. 2

[133] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al. Generalized decoding for pixel, image, and language. arXiv preprint arXiv:2212.11270, 2022. 2

#### Appendix A. Architecture

and RefCOCO/+/g [122, 67], we apply two different augmentations on the same image to generate the pseudo videos for training. And for OVIS [76], we convert the dataset into a class-agnostic format to make it suitable for VOS training.

Reference Encoding Figure 5 illustrates the process of reference encoding. (i) For mask references, we employ the same visual encoder EncV for both the current and reference frames to generate multi-scale features (i.e., C3, C4, C5). We denote the encoded features of the reference frame as FfV , where the ℓ-th feature (ℓ = 2,3,4) has a size of Hℓ × Wℓ × C, with a spatial stride of 2ℓ+1 relative to the original size. Next, we use a lightweight mask encoder (ResNet-18 in all our experiments) that takes the reference frame and annotated mask as inputs. We concatenate the last three layer features with the corresponding level features in FfV and further process them with two ResBlocks [30] and a CBAM block [100] to obtain the final outputs, denoted as FmV . Finally, we flatten each level feature in FfV and FmV into 1-dimensional vectors. (ii) For language references, we directly use off-the-shelf text encoder BERT [20] to extract the 1-d linguistic features.

Inference Details. For both RVOS and VOS tasks, all the videos are rescaled to 480p for inference. And the score thresholds are set as 0.4 for VOS datasets and 0.3 for RVOS datasets, respectively. For these two tasks, both the masks in the first frame and previous frame are adopted as references.

#### Appendix C. More Results

Reference Frames for Mask Propagation. In this study, we investigate the effect of reference frames for mask propagation, which is presented in Table 9. Specifically, we analyze the impact of discarding the first frame and the previous frame on performance for Youtube-VOS2018 and RefYoutube-VOS. These two datasets are evaluated for VOS and RVOS tasks, respectively. On Youtube-VOS2018, the first frame provides a reliable annotated mask, while the previous frame has the highest similarity with the current frame. Therefore, discarding either of these frames would result in a significant drop in performance. on Ref-YoutubeVOS, there is no ground-truth mask in the first frame. Thus the performance drop is less noticeable. Nevertheless, our findings support the conclusion that utilizing both the first frame and the previous frame as references yields the best results for mask propagation.

[Figure 41]

Visual Encoder

[Figure 42]

𝑯𝒍𝑾𝒍 × 𝑪

[Figure 43]

Mask Encoder

[Figure 44]

𝑯𝒍𝑾𝒍 × 𝑪

- (a) Mask Reference
- (b) Language Reference

giraﬀe kicked by an animal in the yard of a zoo.

|Text Encoder| |
|---|---|
| | |

Table 9: Ablation on the reference frames used for mask propagation during inference. We use the final model with ResNet-50 visual backbone in this ablation. Our default settings are marked in gray .

[Figure 45]

| | | | | |
|---|---|---|---|---|

𝑳 × 𝑪

Figure 5: The process of reference encoding for (a) mask references and (b) language references.

Youtube-VOS2018 Ref-Youtube-VOS G Js Fs Ju Fu J &F J F

First Previous

#### Appendix B. Implementation Details

✓ 76.6 78.7 82.8 69.6 75.2 60.8 59.1 62.6 ✓ 80.0 80.7 85.1 73.4 80.7 61.0 59.4 62.7 ✓ ✓ 81.9 82.3 86.8 75.9 82.6 61.5 59.7 63.3

Training Details. Our training process consists of three sequential stages: Objects365 pretraining, image-level training and video-level training. We train models on NVIDIA A100 GPUs and it takes 5-7 days (depends on the visual backbone) to complete the whole training. The text encoder is unfrozen during the first two stages and then frozen for the final stage. The detailed configurations are summarized in Table 10. We follow the implementation of Detic [129] for the multi-dataset training. The learning rate is reduced by the factor of 10 when the iteration reaches the specified step in the table. Data augmentation includes random horizontal flip and scale jitter for resizing the input images. In the table, short side means the range of values for the shortest side and long side represents the maximum value for the longest side. During video-level training, for COCO [56]

Efficiency Comparison. We compare the efficiency of using FlashAttention [18, 17] on two VOS datasets, namely Youtube-VOS18 [109] and LVOS [31], as displayed in Table 11. The results clearly show that FlashAttention could improve the FPS during inference, especially when employing the multi-head attention. Also, it can greatly reduce the GPU memory cost during training, consuming only 12.7G with 8 heads.

Our method has a constant memory cost, while the classic memory-based methods have linear memory complexity with respect to the video duration. This suggests that our method is more efficient for the long-term videos. To better

Table 10: The detailed configurations for the three training stages.

Sampling Weight

Batch Size

Short Side

Long Side

GPU Number

Learning Rate

Weight Decay

Max Iteration

Stage Task Dataset

Step

- I Det Objects365 [84] 1 2 480 ∼ 800 1333 32 0.0002 0.05 340,000 310,000

- II 16 0.0002 0.05 90,000 75,000

|Det|COCO [56] 1 2 480 ∼ 800 1333|
|---|---|
|RIS<br><br>|RefCOCO/+/g [122, 67] 1 2 480 ∼ 800 1333|
|FSS<br><br>|FSS-1000 [50] 0.05 2 480 ∼ 800 1333|

- III

|VOS<br><br>|COCO [56] 0.40 2 320 ∼ 640 768<br><br>Youtube-VOS2019 [109] 0.30 2 320 ∼ 640 768 LVOS [31] 0.20 2 320 ∼ 640 768 OVIS [76] 0.10 2 320 ∼ 640 768|
|---|---|
|RVOS<br><br>|RefCOCO/g/+ [122, 67] 0.50 2 480 ∼ 800 1333 Ref-Youtube-VOS [82] 0.50 2 320 ∼ 640 768|

16 0.0001 0.05 90,000 75,000

Table 11: Efficiency comparison of FlashAttention. ‘YT-VOS18’ represents Youtube-VOS2018 dataset. FPS is measured during inference using A100 GPU. Memory is the GPU memory cost during training.

100

30

20

10

Mean Mean Flash Num

Dataset

FPS Memory Frames Objects Attention Heads

FPS

UniRef

✓ 1 12.4 11.6G ✓ 8 11.1 12.7G

STCN

- 0.1
- 1

YT-VOS18 27 1.9

- ✗ 1 11.9 13.4G
- ✗ 8 8.3 29.9G

✓ 1 24.4 11.6G ✓ 8 20.5 12.7G

1,000 2,000 3,000 4,000 5,000 6,000 7,000

Number of processed frames

LVOS 574 1.3

Figure 6: FPS scaling of our method and the representative memory-based method STCN.

- ✗ 1 23.4 13.4G
- ✗ 8 15.9 29.9G

highlight the advantages of our method, we further compare our method with the representative memory-based method STCN in terms of the single-object FPS-scaling in Figure 6.

#### Appendix D. Visualization Results

We provide the visualization results of UniRef-L for RVOS tasks in Figure 7 and Figure 8. It can be seen that our model can segment the referred objects correctly and accurately in various challenging scenes, e.g., partial display, similar objects and fast moving, as illustrated in Figure 7.

Visualization results for the VOS tasks are presented in

- Figure 9 and Figure 10. Notably, our model reveals strong ability in handling long-term videos that typically last for over a minute, such as those in LVOS [31]. As shown in
- Figure 10, our model can accurately segment the target objects throughout the whole video, despite the objects have significant pose variation. We further provide a video demo in the supplementary material.

a dog is waiting to catch the ball shown to him. a hand is showing a ball to a dog. a lawn tennis ball in the hand of a person.

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

a whale swimming from the bottom to the top of the water. a whale on the top right swimming underwater.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

a skateboard being rolled through a road filled with cars and people. a boy in black shorts and white tee shirt roller skating.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Figure 7: Visualization results on Ref-Youtube-VOS validation set.

a go-cart type car. a person driving the go cart. person at the back of the go-cart without a helmet.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

a man wearing a green helmet. a motor-bike.

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

a blonde haired girl dancing in a blue dress.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

- Figure 8: Visualization results on Ref-DAVIS17 validation set.

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

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

- Figure 9: Visualization results on Youtube-VOS2018 validation set.

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

Figure 10: Visualization results on LVOS validation set.

