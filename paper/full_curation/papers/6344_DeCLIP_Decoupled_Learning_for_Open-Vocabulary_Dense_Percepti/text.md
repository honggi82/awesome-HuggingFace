# arXiv:2505.04410v1[cs.CV]7May2025

## DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception

Junjie Wang1 Bin Chen2,3,* Yulin Li1 Bin Kang3 Yichi Chen3 Zhuotao Tian1,* 1School of Computer Science and Technology, HIT, Shenzhen 2International Research Institute for Artificial Intelligence, HIT, Shenzhen 3University of Chinese Academy of Sciences

### Abstract

Dense visual prediction tasks have been constrained by their reliance on predefined categories, limiting their applicability in real-world scenarios where visual concepts are unbounded. While Vision-Language Models (VLMs) like CLIP have shown promise in open-vocabulary tasks, their direct application to dense prediction often leads to suboptimal performance due to limitations in local feature representation. In this work, we present our observation that CLIP’s image tokens struggle to effectively aggregate information from spatially or semantically related regions, resulting in features that lack local discriminability and spatial consistency. To address this issue, we propose DeCLIP, a novel framework that enhances CLIP by decoupling the self-attention module to obtain “content” and “context” features respectively. The “content” features are aligned with image crop representations to improve local discriminability, while “context” features learn to retain the spatial correlations under the guidance of vision foundation models, such as DINO. Extensive experiments demonstrate that DeCLIP significantly outperforms existing methods across multiple open-vocabulary dense prediction tasks, including object detection and semantic segmentation. Code is available at https://github.com/xiaomoguhz/DeCLIP.

### 1. Introduction

In the era of deep learning, dense prediction tasks like object detection [44, 55] and image segmentation [12, 57] have rapidly advanced and are widely used. However, traditional methods [7, 40, 91] recognize only a fixed set of predefined categories. This restriction hinders the practical application of these methods in real-world settings, where the range of visual concepts is virtually boundless. Consequently, increasing attention has been drawn to open-vocabulary methods [14, 67, 70, 82], which aim to detect and segment objects from any category using textual descriptions.

*Corresponding authors

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Figure 1. DeCLIP outperforms previous state-of-the-art models on a broad range of open-vocabulary dense prediction benchmarks.

Building on the success of Vision-Language Models (VLMs) [13, 41, 52, 61] pre-trained on image-text pairs, such as CLIP [52], researchers have started leveraging these models for open-vocabulary dense prediction tasks. Among these [8, 65, 67–69, 84], transfer-learning approaches [11, 30, 37, 65, 68, 78] have shown outstanding performance. These methods utilize the image encoder of VLM as a feature extractor and exclusively train lightweight task-specific components. Whereas using VLMs as feature extractors offers significant advantages due to their comprehensive pre-training, directly applying these image-level models to dense prediction often leads to domain shift issues [68, 70].

What hinders CLIP in dense perception? To assess VLM’s constraints in dense perception, we analyze CLIP’s attention maps across various layers (Figure 3(a)). Our experiments reveal that CLIP’s [CLS] token may interfere

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]|
|---|

[Figure 72]

###### 74.3

+35.3

CLIP DeCLIP

Image

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

###### 39.0

[Figure 78]

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

25.3

+15.6

CLIP

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

9.7

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

Ours

Region Classification (Top1 mAcc)

Semantic Segmentation (mIoU)

(a) (b)

- Figure 2. Quantitative and qualitative comparisons between our method and CLIP. (a) Performance comparisons of open-vocabulary dense predictions on COCO [43]. (b) Attention map comparisons, with the anchor image token marked in red.

with the correlations among other image tokens, leading to suboptimal performance in dense prediction tasks.

Specifically, we have observed that in deeper layers (behind the 9th layer), the [CLS] token shifts focus away from primary objects within the image and attends highly to certain background tokens, as highlighted by the bright spots in the first row of Figure 3(a). Moreover, image tokens (rows 2 and 3, Figure 3(a)) exhibit similar behavior with the [CLS] token, showing high attention to certain background tokens regardless of their positions.

This observation sheds light on why CLIP struggles in dense prediction tasks: its image tokens fail to aggregate information from spatially or semantically related regions, resulting in dense features that lack local discriminability and spatial consistency. As shown in Figure 2(a), directly using CLIP features on the COCO dataset yields relatively inferior performance in open-vocabulary region classification and semantic segmentation. To tackle this, an intuitive approach is to enhance CLIP’s local representations through fine-tuning. However, balancing the optimizations of both local feature spatial correlations and vision-language semantic alignment within a unified architecture becomes a new challenge. Therefore, is it feasible to disentangle CLIP’s features and apply separate guiding constraints to obtain diverse features within a unified architecture?

Our solution. To address these challenges, we propose DeCLIP, a general unsupervised fine-tuning method aimed at enhancing both the discriminability and spatial consistency of CLIP’s local features. The core idea is to decouple the self-attention module of CLIP and learn from different teacher models separately.

Specifically, DeCLIP decouples the features in the selfattention module into “content” and “context” components. The “content” features, responsible for local discriminability, are fine-tuned by aligning pooled region features with their corresponding image crop [CLS] representations. Meanwhile, the “context” features, responsible for spatial

consistency, are learned from the feature correlations generated by Vision Foundation Models (VFMs). This decoupled distillation design effectively mitigates optimization conflicts, improving the generalization ability when applying CLIP to downstream open-vocabulary dense prediction tasks. As shown in Figure 2, DeCLIP significantly outperforms CLIP in local discriminability and spatial consistency. To summarize, our contributions are as follows:

- • We analyze CLIP and find that its limitation in openvocabulary dense prediction arises from image tokens failing to aggregate information from spatially or semantically related regions.
- • To address this issue, we propose DeCLIP, a simple yet effective unsupervised fine-tuning framework, to enhance the discriminability and spatial consistency of CLIP’s local features via a decoupled feature enhancement strategy.
- • Extensive experiments demonstrate that DeCLIP can be decently applied to mainstream open-vocabulary dense prediction tasks, including object detection and semantic segmentation. As illustrated in Figure 1, DeCLIP outperforms state-of-the-art methods across a broad range of benchmarks, achieving superior performance metrics in all evaluated task domains.

### 2. Background and Motivation

In the following, we provide a concise overview of foundational concepts pertinent to this study in Section 2.1, and highlight important findings in Section 2.2, which offer valuable insights for motivating the proposed approach.

#### 2.1. Preliminaries

Contrastive Language-Image Pre-training (CLIP) [52] is built upon two encoders, one for images and one for text. The visual encoder of CLIP can be a CNN series [27, 45] or ViT [19], and the text encoder is a Transformer [62]. This paper focuses on the CLIP model with the ViT architecture,

Layer6 Layer9 Layer12

Layer11

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

CLIP CLS Token

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

CLIP Image

- Token1

VFM CLS Token

VFM Image Token1

Layer6 Layer9 Layer11 Layer12

CLIP Image

- Token2

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

VFM Image Token2 (a)

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

(b)

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

- Figure 3. Visualization of attention maps across different encoding layers of CLIP and VFM. The attention weights are calculated at a low resolution, then averaged across different heads, and finally upsampled to the original image resolution for visualization. The anchor image token is marked in red. We observe the occurrence of the “proxy” token phenomenon in CLIP, but not in VFM. Furthermore, when the position of the anchor image token is shifted, VFM shows a better correlation for image tokens with the same semantics.

(a) Using Pseudo Regions (b) Using Self-Distillation (c) Decouple Distillation (Ours)

CLIP Image Encoder

Content Context

Decouple

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

CLIP Image Encoder

[Figure 124]

Content Distillation Context Distillation

[Figure 125]

| | |
|---|---|
| | |

Vision Foundation Model

CLIP Image Encoder

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Align

[Figure 132]

CLIP Image Encoder

CLIP Text Encoder

CLIP Image Encoder

[Figure 133]

Align

Bus traveling along a country road

Horse on a summer pasture

[Figure 134]

[Figure 135]

- Figure 4. Pre-fine-tuning methods for adapting CLIP to dense prediction tasks. Existing work considers establishing region-text alignment through cost-effective methods via: (a) using images as pseudo regions or (b) using self-distillation on image patches. The former regards the entire image as a region, which results in a loss of details. The latter uses self-distillation on the image patches thereby gaining more fine-grained information, but still fails to apply to pixel-level image segmentation. (c) Unlike prior approaches, we use VFM to guide the spatial consistency of CLIP’s features, and decouple CLIP’s features for distillation separately to avoid optimization conflicts.

which adopts the [CLS] token to represent the overall features of an image. CLIP learns vision-language alignment by maximizing the cosine similarity between the [CLS] token and text features of matched image-text pairs, and minimizing the similarity for unmatched pairs.

Dense feature extraction with CLIP. ViT-based CLIP consists of a series of stacked attention blocks. For example, the ViT-B version of CLIP includes 12 attention block layers. Let X = {x0,x1,··· ,xh×w} denotes the input to the last attention block, where xi ∈ R1×D. The computation within this attention block can be expressed as:

##### Q = Projq(X), K = Projk(X), V = Projv(X), (1)

- Y = X + Proj(Attnqk · V), (2)
- Z = Y + FFN(Y), (3)

where Q, K, and V represent the query, key, and value embeddings, respectively; Proj denotes projection lay-

##### √

d represents the selfattention process, with d denoting the dimension of each attention head. FFN denotes a feed-forward network. For simplicity, normalization operations are omitted.

ers; Attnqk = SoftMax QK⊤/

After passing through the final attention block, Z[0] represents the global [CLS] token. The remaining image patch embeddings Z[1 : h × w] can be reshaped to obtain dense feature representations Xdense ∈ RC×H×W1.

Adapting CLIP to dense prediction tasks. Several studies have attempted to alleviate the domain shift issue in applying CLIP to dense prediction tasks via fine-tuning strategies. These approaches fall into two main categories:

• Joint fine-tuning. These methods fine-tune CLIP while training task-specific components [14, 30, 31, 39, 42, 72, 77]. For instance, CAT-Seg [14] proposes an attention fine-tuning strategy based on ViT CLIP, which general-

1The final V-L projection layer is omitted here for brevity.

izes well to unseen categories. MAFT [30] leverages attention bias to fine-tune CLIP for mask classification.

• Pre-fine-tuning. These methods directly fine-tune CLIP using cost-efficient techniques [49, 68–70, 85], which are more closely aligned with the approach proposed in this paper. As illustrated in Figure 4(a), CLIM [69] employs a mosaic augmentation technique to stitch multiple images into a single image, enabling each sub-image to serve as a pseudo-region for region-text contrastive learning. CLIPSelf [68] enhances CLIP’s region classification accuracy by maximizing cosine similarity between its region representations and the corresponding image crop representations, as illustrated in Figure 4(b).

#### 2.2. Key Observations

Despite the promising results of the two categories of finetuned methods in Section 2.1, they continue to exhibit certain limitations. Joint fine-tuning methods are typically specific to tasks or models and heavily rely on labor-intensive annotations of dense prediction tasks. On the other hand, pre-fine-tuning methods demonstrate broader applicability. However, their region-level fine-tuning technique remains limited in image segmentation tasks that require pixel-level details. To tackle this issue, we investigate the feasibility of incorporating pixel-level details into CLIP’s pre-finetuning, enabling it to better align with open-vocabulary dense prediction tasks. In the following, we start by analyzing CLIP’s attention maps across various layers.

The “proxy” token phenomenon. As shown in Figure 3(a), we found that in CLIP’s shallow layer, the attention weights of CLIP’s [CLS] token are widely distributed across the image (i.e., layer 6). However, in the deeper layers, the [CLS] token shifts its focus away from primary objects in the image and attends to specific tokens, as highlighted by the bright spots within the image background. Additionally, we found that image tokens (rows 2 and 3) exhibit similar behavior to the [CLS] token, showing high attention to certain tokens in the background, regardless of their position.

These background tokens may serve as “proxies” for the [CLS] token. This suggests that these tokens aggregate essential information from other image tokens, enabling the [CLS] token to form an approximate “global view” by summarizing content from them, thereby facilitating image classification. However, these “proxy” tokens negatively affect the feature correlations between image tokens. As illustrated in Figure 3(a), when we shift the position of the anchor image token (from the bird to the branch), we observe that the new image token still pays high attention to the “proxy” tokens. This results in a lack of correlation between image patches that share the same semantics, which is detrimental to dense prediction tasks.

VFMs exhibit better dense correlations. Considering the inherent constraints that impede CLIP’s efficacy in dense

Table 1. Performance of different distillation schemes.

Region Classification (mAcc) Semantic Segmentation (mIoU) COCO (Thing) COCO (Stuff) Context59 CityScape

Distillation Type

Self Distillation [68] 69.5 44.6 29.4 25.6 Self+VFM Distillation [36] 65.6 (-3.9) 41.3 (-3.3) 32.4 (+3.0) 28.7 (+3.1) Self+VFM+Decouple 75.0 (+5.5) 51.8 (+7.2) 35.3 (+5.9) 32.3 (+6.7)

perception tasks, we instead observe that VFMs such as the DINO series [5, 51], trained in a self-supervised manner, and the SAM series [36, 54], trained on large-scale segmentation data, are capable of extracting features with strong spatial consistency, as shown in Figure 3(b).

In particular, the attention map of VFMs does not exhibit the “proxy” token phenomenon observed in CLIP. Furthermore, when we change the position of the anchor image token, the VFM shows a better correlation for image tokens with the same semantics. Therefore, we consider whether VFMs can be incorporated into the pre-fine-tuning process to further improve the feature correlations of CLIP. However, this straightforward approach fails to achieve satisfactory results. Conducting VFM distillation2 and selfdistillation3 simultaneously results in reduced region classification performance, as shown in Table 1 (row 2). We hypothesize that this observation stems from the fact that spatial feature correlation and vision-language alignment have different optimization focuses, and optimizing them simultaneously within a single model results in trade-offs.

### 3. Method

Through the above analysis, we found that CLIP underperforms in dense prediction tasks since its image tokens fail to effectively aggregate information from semantically related regions. Observations of VFMs’ attention maps inspired us to incorporate them into CLIP’s pre-fine-tuning process. Considering the optimization conflict between feature correlations and visual-language alignment, we applied a decoupled feature enhancement strategy to CLIP.

In this section, we introduce DeCLIP, an unsupervised fine-tuning framework for adapting CLIP to dense prediction tasks. We first explain how to decouple CLIP’s selfattention mechanism into “content” and “context” components in Sec.3.1, then describe how these components learn from different “teacher” models in Sec.3.2 by distillation.

#### 3.1. Decoupled Attention

The unsuccessful attempts to simultaneously perform selfdistillation and VFM distillation on Xdense (Table 1, row 2) prompted us to explore the feasibility of a decoupled distillation. In the following, we propose decoupling CLIP’s

- 2“VFM distillation” indicates aligning the feature self-correlations be-

tween CLIP’s Xdense and that of the VFM.

- 3“Self-distillation” refers to aligning region features from Xdense with

their corresponding [CLS] representation.

|[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]|
|---|

|[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]|
|---|

[Figure 196]

VFM Single Token Correlations

CLIP Single Token Correlations

Vision Foundation Model

Parameter Tuned

[Figure 197]

VFM

IE CLIPEncoderImage

Parameter Frozen

[Figure 198]

Reshape

Reshape

[Figure 199]

| | | | | | | |
|---|---|---|---|---|---|---|

[Figure 200]

Region Feature Pooling

VFM Feature Correlations

CLIP Context Correlations Reshape

CLS Tokens

[Figure 201]

Encoder Layer L

[Figure 202]

[Figure 203]

V-L proj CLSCLSCLSCLSImage Tokens

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Encoder Layer 2

IE Teacher VFM Models

[Figure 208]

Encoder Layer 1

Softmax MatMul MatMul

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

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Wquery Wvalue

Wquery

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

- Figure 5. Illustration of the DeCLIP framework. We decouple CLIP’s final attention module into context and content features for distillation, avoiding optimization conflicts between feature correlations and visual-language alignment. CLIP itself serves as the teacher for content features to improve region classification accuracy. A VFM serves as the teacher for context features to enhance spatial consistency.

self-attention module to obtain “content” and “context” features, and separately optimize the local discriminability and spatial consistency abilities, as illustrated in Figure 4(c).

Rethinking the self-attention. As described in Sec.2.1, in CLIP’s last attention block, the V features are weighted and summed under the guidance of the attention map (Attnqk) derived from Q and K, which define spatial or semantic relationships among image tokens. Studies [38, 59, 63, 71] have shown that CLIP’s dense features Xdense can be directly used for semantic segmentation by per-pixel classification, indicating that each pixel of Xdense contains independent semantic information. Inspired by this, we regard Q and K as anchors for improving spatial consistency, and Xdense as an anchor for enhancing local discriminability.

Additionally, recent training-free OVS studies [38, 63] have further promoted us to decouple CLIP’s self-attention followed by distillation. They modify CLIP’s attention block from Attnqk to Attnqq and remove the residual connections, simplifying the optimization of local feature consistency by focusing on Q alone. Based on our rethinking of CLIP’s self-attention and inspired by these methods, we propose decoupling CLIP’s last attention block to obtain “content” and “context” features for distillation as follows:

Xcontext = Projq(X), V = Projv(X), (4) Xcontent = Proj(Attncontext · V), (5)

√

Attncontext = SoftMax XcontextX⊤context/

d . (6)

Specifically, V is aggregated based on the attention map (Attncontext) generated from Xcontext. Xcontext determines

which image tokens are semantically or spatially related. Xcontent carries the semantic information of each image token in the visual-language space. By decoupling the features in this manner, we can apply different guidance constraints to Xcontext and Xcontent to obtain diverse feature representations in a unified architecture without interference.

As observed in Sec. 2.2, VFM exhibits a strong correlation for image tokens with the same semantics, thus we leverage it as guidance for Xcontext to improve CLIP’s local feature spatial consistency. Meanwhile, we employ the selfdistillation technique as guidance for Xcontent to enhance the visual-language alignment of CLIP’s region feature.

As demonstrated in Table 1 (row 3), this decoupled optimization significantly improves the local discriminability and spatial consistency of CLIP’s features, leading to simultaneous enhancements in both region classification accuracy and semantic segmentation performance.

#### 3.2. DeCLIP

The previous section presents a method for obtaining the decoupled “context” and “content” features from CLIP. In this section, we elaborate on how the decoupled features Xcontent and Xcontext learn from their respective teacher models to enhance CLIP’s performance on open-vocabulary dense prediction tasks.

Content feature distillation. As shown in Figure 5, the first teacher model in DeCLIP is itself, which is known as self-distillation [9, 49, 50, 68]. we employ an image patching method to align the region representations of the student model’s feature map with the corresponding image crop representations (i.e., [CLS] token) of the teacher model.

Specifically, the input image I is first divided into k sub-

regions. Subsequently, these sub-regions are cropped from the original image, resulting in a set of sub-images S = {I′1,I′2,...,I′k}. The student model takes the image I as input and outputs the content feature Xcontent ∈ RC×H×W and the context feature Xcontext ∈ RD×H×W, as mentioned in Eq.(6). Here, D represents the dimension of the CLIP visual encoder, and C represents the shared dimension of the vision-language modality. Then, the student model uses RoI Align [28] to pool region features from Xcontent based on the cropping coordinates of S, resulting in a region feature set Fs = {f1s,f2s,...,fks}, where fis ∈ R1×C.

Meanwhile, the teacher model takes the sub-image set S as input and outputs a series of [CLS] tokens corresponding to the cropped sub-images, resulting in [CLS] token set Ft = {f1t,f2t,...,fkt}, where fit ∈ R1×C. Finally, we use a cosine similarity loss to align the [CLS] tokens from Ft with the region features from Fs as follows:

k

fit · fis ∥fit∥ · ∥fis∥

1 k

. (7)

Lcontent =

1 −

i=1

The intuition behind this distillation branch is that, for objects within an image, classifying them using image crops (i.e., [CLS] token) achieves higher accuracy than using region features [68]. This is because CLIP is pre-trained on image-text pairs using contrastive learning, as mentioned in Sec.2.1. Therefore, the distillation learning of Xcontent enhances the discriminability of CLIP’s region features, i.e., Fs = {f1s,f2s,...,fks}, by mimicking the [CLS] tokens obtained from the image crops, i.e., Ft = {f1t,f2t,...,fkt}. However, as previously discussed in Sec.2.2, the regionlevel fine-tuning remains limited in image segmentation that requires pixel-wise scene understanding.

Context feature distillation. As discussed in Sec.2.2, VFMs do not exhibit CLIP’s “proxy” token issue and better correlate semantically related image tokens, which may be conducive to the fine-grained local perception. Therefore, we distilled these correlations into CLIP’s Xcontext features.

As illustrated in Figure 5, the same image I is input into the VFM to obtain its dense feature representations XVFMdense ∈ RD×HW. To ensure consistency in the number of image tokens after patch embedding, different input resolutions are typically used for the VFM and the student CLIP. To transfer VFM’s correlations between image tokens to CLIP, an intermediary is required to represent the correlation volume between two image tokens. Cosine similarity is used in our method, specifically as follows:

xi · xj ∥xi∥ · ∥xj∥

. (8)

rij =

Here, xi ∈ R1×D and xj ∈ R1×D represent the i-th and j-th image patch tokens. rij denotes the correlation volume between patch tokens xi and xj. We use the L2 loss to

align the discrepancy in the correlation volume between the image tokens of XVFMdense and Xcontext, specifically as follows:

H

W

1 HW

Lcontext =

i=1

j=1

rijVFM − rijCLIP 2 , (9)

where rijVFM and rijCLIP denote the correlation volume between xi and xj for VFM and CLIP, respectively. Finally, the entire distillation learning process of DeCLIP can be expressed as follows:

Ltotal = Lcontent + λLcontext, (10) where λ4 represents the loss scaling hyperparameter.

### 4. Experiments

#### 4.1. Datasets and Evaluation

We conducted extensive evaluations across multiple openvocabulary dense prediction benchmarks, encompassing object detection, semantic segmentation, and segmentation based on VLM features. Due to space limitations, detailed descriptions of the datasets, evaluation metrics, and implementation specifics are provided in the Appendix.

#### 4.2. Benchmark Results

Open-Vocabulary Detection. Table 2 presents DeCLIP’s performance on OV-COCO and OV-LVIS benchmarks. On OV-COCO, DeCLIP improves the F-ViT [68] baseline by

###### 3.5 and 1.9 mAP, and the OV-DQUO [65] baseline by 6.9 and 2.7 mAP on novel classes. On OV-LVIS, it achieves gains of 1.5 and 2.3 mAP with F-ViT, as well as 1.3 and 2.2 mAP with OV-DQUO on rare classes. Cross-dataset evaluations of F-ViT+DeCLIP trained on OV-LVIS (Table 3) further confirm DeCLIP’s superiority over existing methods.

Open-Vocabulary Semantic Segmentation. Table 4 displays the performance of the CAT-Seg [14] model using DeCLIP as the backbone across various open-vocabulary semantic segmentation benchmarks. The results show that DeCLIP significantly enhances segmentation performance on all datasets. Notably, even with the ViT-B/16 version of DeCLIP, CAT-Seg nearly surpasses all existing SOTA methods that utilize substantially larger encoders like ConvNeXt-L. When employing the ViT-L/14 version of DeCLIP, the model achieves new SOTA results in openvocabulary semantic segmentation tasks.

Open-Vocabulary Semantic Segmentation Based on VLM Features. Following existing methods [38, 59, 63], in this experiment, we assign each pixel in the feature map the category with which it has the highest cosine similarity. The low-resolution prediction result is up-sampled to the

4The sensitivity analysis is in the appendix.

76

77.5

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | |DeC CLIP|LIP self| |
| | | | | | | | | |
| | | | | | |Reg|ionCLIP| |
| | | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | |DeC|LIP| |
| | | | | |CLI<br><br>Reg|Pself ionCLIP| |
| | | | | | | | |

MaskPooling(Thing)mAcc

50

MaskPooling(Stuff)mAcc

75.0

RoIAlign(Thing)mAcc

74

72.5

45

72

DeCLIP

70.0

CLIPself

40

67.5

RegionCLIP

70

65.0

35

68

62.5

30

66

60.0

336 448 560 672 784 896 1024

336 448 560 672 784 896 1024

336 448 560 672 784 896 1024

Resolution

Resolution

Resolution

- Figure 6. Comparisons between DeCLIP and existing methods in terms of open-vocabulary region classification ability at different resolutions on the COCO panoptic dataset.

- Table 2. Comparison with state-of-the-art open-vocabulary object detection methods. Caption supervision indicates that the method learns from extra image-text pairs, while CLIP supervision refers to transferring knowledge from CLIP. †: DETR-based detectors [4].

(a) OV-COCO benchmark

Method Supervision Backbone APNovel50

ViLD [24] CLIP RN50 27.6 Detic [89] Caption RN50 27.8 OV-DETR† [81] CLIP RN50 29.4 BARON-KD [67] CLIP RN50 34.0 SAS-Det [84] CLIP RN50 37.4 OV-DQUO† [65] CLIP RN50 39.2 RegionCLIP [85] Captions RN50x4 39.3 CORA† [70] CLIP RN50x4 41.7 OV-DQUO† [65] CLIP RN50x4 45.6

RO-ViT [34] CLIP ViT-L/16 33.0 CFM-ViT [33] CLIP ViT-L/16 34.1 CLIPSelf [68] CLIP ViT-B/16 37.6 CLIPSelf [68] CLIP ViT-L/14 44.3

F-ViT [68]+DeCLIP CLIP ViT-B/16 41.1 (+3.5) F-ViT [68]+DeCLIP CLIP ViT-L/14 46.2 (+1.9) OV-DQUO+DeCLIP† CLIP ViT-B/16 46.1 (+6.9) OV-DQUO+DeCLIP† CLIP ViT-L/14 48.3 (+2.7)

(b) OV-LVIS benchmark

Method Supervision Backbone mAPr

ViLD [24] CLIP RN50 16.3 OV-DETR† [81] CLIP RN50 17.4 BARON-KD [67] CLIP RN50 22.6 RegionCLIP [85] Caption RN50x4 22.0 OV-SAM [79] CLIP RN50x16 24.0 CORA+† [70] Caption RN50x4 28.1 F-VLM [37] CLIP RN50x64 32.8

CLIPSelf [68] CLIP ViT-B/16 25.3 OV-DQUO† [65] CLIP ViT-B/16 29.7 Detic [89] Caption Swin-B 33.8 RO-ViT [34] CLIP ViT-H/16 34.1 CLIPSelf [68] CLIP ViT-L/14 34.9 OV-DQUO† [65] CLIP ViT-L/14 39.3

F-ViT [68]+DeCLIP CLIP ViT-B/16 26.8 (+1.5) F-ViT [68]+DeCLIP CLIP ViT-L/14 37.2 (+2.3) OV-DQUO+DeCLIP† CLIP ViT-B/16 31.0 (+1.3) OV-DQUO+DeCLIP† CLIP ViT-L/14 41.5 (+2.2)

- Table 3. Transfer evaluation of the LVIS-trained detector on COCO and Objects365 datasets.

region classification performance of DeCLIP, RegionCLIP [85], and CLIPSelf [68] at various resolutions on the COCO-Panoptic validation set. Using RoI Align [28] and Mask Pooling, we extract local features from the feature maps based on annotated bounding boxes and masks, assigning categories based on maximum cosine similarity. As illustrated in Figure 6, the Top-1 mean accuracy (mAcc) results demonstrate that DeCLIP consistently surpasses existing methods in region recognition across all resolutions.

COCO Objects365 [58]

Method

AP AP50 AP75 AP AP50 AP75 Supervised Baseline [24] 46.5 67.6 50.9 25.6 38.6 28.0 ViLD [24] 36.6 55.6 39.6 11.8 18.0 12.6 DetPro [20] 34.9 53.8 37.4 12.1 18.8 12.9 BARON [67] 36.2 55.7 39.1 13.6 21.0 14.5 F-VLM [37] 37.9 61.6 41.2 16.2 27.4 17.5 CoDet [47] 39.1 57.0 42.3 14.2 20.5 15.3 RO-ViT [35] - - - 17.7 27.4 19.1 CLIPSelf [68] 40.5 63.8 44.3 19.5 31.3 20.7 DeCLIP 41.0 64.6 44.8 20.0 32.2 21.2

#### 4.3. Ablation Study

original resolution to obtain the final segmentation map. As shown in Table 5, DeCLIP outperforms all existing methods in terms of average mIoU across eight benchmarks, highlighting the effectiveness of our approach in improving the discriminability and spatial consistency of VLM features.

Open-Vocabulary Region Classification. We assess the

The impact of VFMs. We analyzed the impact of various VFM configurations on DeCLIP performance. As shown in Table 6, DeCLIP distilled from DINO [5] performs moderately in segmentation but trails SAM [36, 54] and DINOv2 [51] in region classification. DeCLIP distilled from SAM excels in region classification but shows lower segmentation performance compared to DINO. DINOv2 achieves balance in both region classification and segmentation.

Table 4. Results on open-vocabulary semantic segmentation. † indicates results re-experimented by CAT-Seg [14].

Method Backbone Training Set ADE847 Context459 ADE150 Context59 VOC20 VOC21 ZegFormer† [17] ViT-B/16 COCO-Stuff 5.6 10.4 18.0 45.5 89.5 65.5 ZSseg [75] ViT-B/16 COCO-Stuff 7.0 - 20.5 47.7 88.4 OVSeg [42] ViT-L/14 COCO-Stuff 9.0 12.4 29.6 55.7 94.5 SAN [76] ViT-L/14 COCO-Stuff 13.7 17.1 33.3 60.2 95.5 ODISE [74] ViT-L/14 COCO-Panoptic 11.1 14.5 29.9 57.3 - 84.6 MAFT [30] ConvNeXt-L COCO-Stuff 13.1 17.0 34.4 57.5 93.0 FC-CLIP [78] ConvNeXt-L COCO-Panoptic 14.8 18.2 34.1 58.4 95.4 81.8 FrozenSeg [11] ConvNeXt-L COCO-Panoptic 14.8 19.7 34.4 - - 82.5 CAT-Seg [14] ViT-B/16 COCO-Stuff 12.0 19.0 31.8 57.5 94.6 77.3 CAT-Seg [14] ViT-L/14 COCO-Stuff 16.0 23.8 37.9 63.3 97.0 82.5

CAT-Seg+DeCLIP ViT-B/16 COCO-Stuff 15.3 (+3.3) 21.4 (+2.4) 36.3 (+4.5) 60.6 (+3.1) 96.6 (+2.0) 81.3 (+4.0) CAT-Seg+DeCLIP ViT-L/14 COCO-Stuff 17.6 (+1.6) 25.9 (+2.1) 40.7 (+2.8) 63.9 (+0.6) 97.7 (+0.7) 83.9 (+1.4)

Table 5. Results on open-vocabulary semantic segmentation based on VLM features.

With a background category Without background category

Method

Avg

VOC21 Context60 COCO-Obj VOC20 CityScape Context59 ADE COCO-Stf CLIP [52] 18.8 9.9 8.1 49.4 6.5 11.1 3.1 5.7 14.1 MaskCLIP [87] 43.4 23.2 20.6 74.9 24.9 26.4 11.9 16.7 30.3 GroupViT [73] 52.3 18.7 27.5 79.7 18.5 23.4 10.4 15.3 30.7 ReCo [60] 25.1 19.9 15.7 57.7 21.6 22.3 11.2 14.8 23.5 TCL [6] 51.2 24.3 30.4 77.5 23.5 30.3 14.9 19.6 33.9 OVSeg [42] 53.8 20.4 25.1 - - - 5.6 - SCLIP [63] 59.1 30.4 30.5 80.4 32.2 34.2 16.1 22.4 38.2 ClearCLIP [38] 51.8 32.6 33.0 80.9 30.0 35.9 16.7 23.9 38.1 CLIP-DINOiser [71] 62.1 32.4 34.8 80.9 31.7 35.9 20.0 24.6 40.3

###### DeCLIP (ours) 59.7 35.3 36.4 85.0 32.8 39.2 21.9 25.3 41.9

|[Figure 264]|
|---|
|[Figure 265]|

|[Figure 266]|
|---|
|[Figure 267]|

|[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>[Figure 311]<br><br>[Figure 312]<br><br>[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]|
|---|
|[Figure 316]|

|[Figure 317]|
|---|
|[Figure 318]|

[Figure 319]

ImageSAMDeCLIPDINO/8DINOv2

|[Figure 320]|
|---|

|[Figure 321]|
|---|

|[Figure 322]|
|---|

|[Figure 323]|
|---|

|[Figure 324]|
|---|

|[Figure 325]|
|---|

|[Figure 326]|
|---|

|[Figure 327]|
|---|

|[Figure 328]|
|---|

|[Figure 329]|
|---|

|[Figure 330]|
|---|

|[Figure 331]|
|---|

- Figure 7. Qualitative comparisons of attention maps between VFMs and DeCLIP. The anchor image token is marked in red.

Qualitative results. Figure 7 presents the visual comparison of attention maps between DINO, SAM, DINOv2, and DeCLIP. Experimental results show that DeCLIP effectively focuses on regions spatially or semantically asso-

Table 6. Ablation studies on the impact of different VFMs on open-vocabulary region classification and segmentation.

Region Classification (mAcc) Semantic Segmentation (mIoU) COCO (Thing) COCO (Stuff) Context59 COCO-Stf ADE

VFMs Arch

DINO [5] ViT-B/8 68.4 49.4 37.3 23.2 19.5 DINO [5] ViT-B/16 67.6 47.4 38.1 23.7 20.4 SAM [36] ViT-B/16 75.0 51.8 35.3 22.0 18.5 SAM [36] ViT-L/16 76.8 52.6 37.7 23.0 20.0 DINOv2 [51] ViT-B/14 77.2 52.5 39.2 25.3 21.9 DINOv2 [51] ViT-L/14 77.6 53.1 38.0 24.1 21.3

ciated with the anchor image token. Moreover, this experiment reveals why DeCLIP distilled from DINOv2 works best: SAM lacks semantic association ability, while DINO focus indiscriminately on all primary objects in the image.

### 5. Conclusion

This paper analyzes the limitations of CLIP in dense prediction tasks from the perspective of its attention map. We observed that CLIP’s [CLS] token negatively affects the attention map of image tokens. To address this issue, we proposed DeCLIP, a decoupled feature enhancement strategy. Extensive experiment results on open-vocabulary dense prediction benchmarks demonstrate that DeCLIP outperforms state-of-the-art methods, achieving excellent performance across all evaluated task domains.

### References

- [1] Benedikt Alkin, Lukas Miklautz, Sepp Hochreiter, and Johannes Brandstetter. Mim-refiner: A contrastive learning boost from intermediate pre-trained representations. arXiv preprint arXiv:2402.10093, 2024. 21

- [2] Xiang An, Kaicheng Yang, Xiangzi Dai, Ziyong Feng, and Jiankang Deng. Multi-label cluster discrimination for visual representation learning. In European Conference on Computer Vision, pages 428–444. Springer, 2025. 21

- [3] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1209–1218, 2018. 17, 20

- [4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. Endto-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer,

2020. 7, 20

- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 4, 7, 8, 21

- [6] Junbum Cha, Jonghwan Mun, and Byungseok Roh. Learning to generate text-grounded mask for open-world semantic segmentation from only image-text pairs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11165–11174, 2023. 8

- [7] Fangyi Chen, Han Zhang, Kai Hu, Yu-Kai Huang, Chenchen Zhu, and Marios Savvides. Enhanced training of querybased object detection via selective query recollection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23756– 23765, 2023. 1

- [8] Fangyi Chen, Han Zhang, Zhantao Yang, Hao Chen, Kai Hu, and Marios Savvides. Rtgen: Generating region-text pairs for open-vocabulary object detection. arXiv preprint arXiv:2405.19854, 2024. 1, 17

- [9] Jun Chen, Deyao Zhu, Guocheng Qian, Bernard Ghanem, Zhicheng Yan, Chenchen Zhu, Fanyi Xiao, Sean Chang Culatana, and Mohamed Elhoseiny. Exploring open-vocabulary semantic segmentation from clip vision encoder distillation only. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 699–710, 2023. 5

- [10] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9640–9649, 2021. 21

- [11] Xi Chen, Haosen Yang, Sheng Jin, Xiatian Zhu, and Hongxun Yao. Frozenseg: Harmonizing frozen foundation models for open-vocabulary segmentation. arXiv preprint arXiv:2409.03525, 2024. 1, 8, 21

- [12] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In

- Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 1
- [13] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818–2829, 2023. 1

- [14] Seokju Cho, Heeseong Shin, Sunghwan Hong, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. Catseg: Cost aggregation for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4113– 4123, 2024. 1, 3, 6, 8, 15, 20, 21

- [15] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016. 17, 20

- [16] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023. 13, 21

- [17] Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. Decoupling zero-shot semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11583–11592, 2022. 8, 20, 21

- [18] Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. Decoupling zero-shot semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11583–11592, 2022. 20

- [19] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2, 13

- [20] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary object detection with vision-language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14084–14093, 2022. 7, 17

- [21] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88:303–338, 2010. 20

- [22] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. Image and Vision Computing, 149:105171,

2024. 15

- [23] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In European Conference on Computer Vision, pages 540–557. Springer, 2022. 20

- [24] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921,

2021. 7, 17, 21

- [25] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In

- Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5356–5364, 2019. 17, 18
- [26] Kunyang Han, Yong Liu, Jun Hao Liew, Henghui Ding, Jiajun Liu, Yitong Wang, Yansong Tang, Yujiu Yang, Jiashi Feng, Yao Zhao, et al. Global knowledge calibration for fast open-vocabulary segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 797–807, 2023. 20

- [27] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 2, 13

- [28] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017. 6, 7

- [29] Joonhyun Jeong, Geondo Park, Jayeon Yoo, Hyungsik Jung, and Heesu Kim. Proxydet: Synthesizing proxy novel classes via classwise mixup for open-vocabulary object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2462–2470, 2024. 17

- [30] Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, and Humphrey Shi. Learning mask-aware clip representations for zero-shot segmentation. Advances in Neural Information Processing Systems, 36:35631–35653, 2023. 1, 3, 4, 8, 21

- [31] Siyu Jiao, Hongguang Zhu, Jiannan Huang, Yao Zhao, Yunchao Wei, and Humphrey Shi. Collaborative vision-text representation optimizing for open-vocabulary segmentation. In European Conference on Computer Vision, pages 399–416. Springer, 2025. 3, 21

- [32] Laurynas Karazija, Iro Laina, Andrea Vedaldi, and Christian Rupprecht. Diffusion models for zero-shot open-vocabulary segmentation. arXiv preprint arXiv:2306.09316, 2023. 20, 21

- [33] Dahun Kim, Anelia Angelova, and Weicheng Kuo. Contrastive feature masking open-vocabulary vision transformer. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 15556–15566, 2023. 7, 17, 20

- [34] Dahun Kim, Anelia Angelova, and Weicheng Kuo. Regionaware pretraining for open-vocabulary object detection with vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11144–11154, 2023. 7, 17

- [35] Dahun Kim, Anelia Angelova, and Weicheng Kuo. Regionaware pretraining for open-vocabulary object detection with vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11144–11154, 2023. 7, 17, 20

- [36] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 4, 7, 8, 21

- [37] Weicheng Kuo, Yin Cui, Xiuye Gu, AJ Piergiovanni, and Anelia Angelova. F-vlm: Open-vocabulary object detec-

- tion upon frozen vision and language models. arXiv preprint arXiv:2209.15639, 2022. 1, 7, 17, 20
- [38] Mengcheng Lan, Chaofeng Chen, Yiping Ke, Xinjiang Wang, Litong Feng, and Wayne Zhang. Clearclip: Decomposing clip representations for dense vision-language inference. arXiv preprint arXiv:2407.12442, 2024. 5, 6, 8, 15, 17, 18

- [39] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Ren´e Ranftl. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546, 2022. 3, 20, 21

- [40] Feng Li, Hao Zhang, Huaizhe Xu, Shilong Liu, Lei Zhang, Lionel M Ni, and Heung-Yeung Shum. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3041–3050, 2023. 1

- [41] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23390– 23400, 2023. 1

- [42] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7061–7070, 2023. 3, 8, 20, 21

- [43] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 2, 17, 18, 20

- [44] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. DAB-DETR: Dynamic anchor boxes are better queries for DETR. In International Conference on Learning Representations, 2022. 1

- [45] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 13

- [46] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 20

- [47] Chuofan Ma, Yi Jiang, Xin Wen, Zehuan Yuan, and Xiaojuan Qi. Codet: Co-occurrence guided region-word alignment for open-vocabulary object detection. Advances in Neural Information Processing Systems, 36, 2024. 7, 17

- [48] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 891–898, 2014. 17, 20

- [49] Jishnu Mukhoti, Tsung-Yu Lin, Omid Poursaeed, Rui Wang, Ashish Shah, Philip HS Torr, and Ser-Nam Lim.

- Open vocabulary semantic segmentation with patch aligned contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19413–19423, 2023. 4, 5, 21
- [50] Muhammad Ferjad Naeem, Yongqin Xian, Xiaohua Zhai, Lukas Hoyer, Luc Van Gool, and Federico Tombari. Silc: Improving vision language pretraining with self-distillation. In European Conference on Computer Vision, pages 38–55. Springer, 2025. 5

- [51] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4, 7, 8, 20, 21

- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 1, 2, 8, 15, 16, 20, 21

- [53] Mike Ranzinger, Greg Heinrich, Jan Kautz, and Pavlo Molchanov. Am-radio: Agglomerative vision foundation model reduce all domains into one. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12490–12500, 2024. 21

- [54] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 4, 7, 21

- [55] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015. 1, 20

- [56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 17, 19

- [57] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 1

- [58] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019. 7, 17, 18, 19

- [59] Tong Shao, Zhuotao Tian, Hang Zhao, and Jingyong Su. Explore the potential of clip for training-free open vocabulary semantic segmentation. In European Conference on Computer Vision, pages 139–156. Springer, 2025. 5, 6, 13

- [60] Gyungin Shin, Weidi Xie, and Samuel Albanie. Reco: Retrieve and co-segment for zero-shot transfer. Advances in

Neural Information Processing Systems, 35:33754–33767,

2022. 8

- [61] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 1, 15, 20, 21

- [62] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 2

- [63] Feng Wang, Jieru Mei, and Alan Yuille. Sclip: Rethinking self-attention for dense vision-language inference. arXiv preprint arXiv:2312.01597, 2023. 5, 6, 8, 17, 18

- [64] Haoxiang Wang, Pavan Kumar Anasosalu Vasu, Fartash Faghri, Raviteja Vemulapalli, Mehrdad Farajtabar, Sachin Mehta, Mohammad Rastegari, Oncel Tuzel, and Hadi Pouransari. Sam-clip: Merging vision foundation models towards semantic and spatial understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3635–3647, 2024. 21

- [65] Junjie Wang, Bin Chen, Bin Kang, Yulin Li, YiChi Chen, Weizhi Xian, and Huifeng Chang. Ov-dquo: Openvocabulary detr with denoising text query training and open-world unknown objects supervision. arXiv preprint arXiv:2405.17913, 2024. 1, 6, 7, 15, 17, 20, 21

- [66] Luting Wang, Yi Liu, Penghui Du, Zihan Ding, Yue Liao, Qiaosong Qi, Biaolong Chen, and Si Liu. Object-aware distillation pyramid for open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11186–11196, 2023. 20, 21

- [67] Size Wu, Wenwei Zhang, Sheng Jin, Wentao Liu, and Chen Change Loy. Aligning bag of regions for openvocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15254–15264, 2023. 1, 7, 17, 20, 21

- [68] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Xiangtai Li, Wentao Liu, and Chen Change Loy. CLIPSelf: Vision transformer distills itself for open-vocabulary dense prediction. In The Twelfth International Conference on Learning Representations, 2024. 1, 4, 5, 6, 7, 15, 16, 17, 18, 20, 21

- [69] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Wentao Liu, and Chen Change Loy. Clim: Contrastive languageimage mosaic for region representation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6117– 6125, 2024. 1, 4, 17, 21

- [70] Xiaoshi Wu, Feng Zhu, Rui Zhao, and Hongsheng Li. Cora: Adapting clip for open-vocabulary detection with region prompting and anchor pre-matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7031–7040, 2023. 1, 4, 7, 17, 18, 20, 21

- [71] Monika Wysocza´nska, Oriane Sim´eoni, Micha¨el Ramamonjisoa, Andrei Bursuc, Tomasz Trzci´nski, and Patrick P´erez. Clip-dinoiser: Teaching clip a few dino tricks. arXiv preprint arXiv:2312.12359, 2023. 5, 8

- [72] Bin Xie, Jiale Cao, Jin Xie, Fahad Shahbaz Khan, and Yanwei Pang. Sed: A simple encoder-decoder for openvocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3426–3436, 2024. 3, 21

- [73] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18134–18144, 2022. 8, 20

- [74] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2955–2966, 2023. 8

- [75] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for openvocabulary semantic segmentation with pre-trained visionlanguage model. In European Conference on Computer Vision, pages 736–753. Springer, 2022. 8

- [76] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2945– 2954, 2023. 8

- [77] Xin Xu, Tianyi Xiong, Zheng Ding, and Zhuowen Tu. Masqclip for open-vocabulary universal image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 887–898, 2023. 3, 21

- [78] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional clip. Advances in Neural Information Processing Systems, 36, 2024. 1, 8, 21

- [79] Haobo Yuan, Xiangtai Li, Chong Zhou, Yining Li, Kai Chen, and Chen Change Loy. Open-vocabulary sam: Segment and recognize twenty-thousand classes interactively. In ECCV,

2024. 7, 21

- [80] Nir Zabari and Yedid Hoshen. Open-vocabulary semantic segmentation using test-time distillation. In European Conference on Computer Vision, pages 56–72. Springer,

2022. 20

- [81] Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, and Chen Change Loy. Open-vocabulary detr with conditional matching. In European Conference on Computer Vision, pages 106–122. Springer, 2022. 7, 17, 20

- [82] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and ShihFu Chang. Open-vocabulary object detection using captions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14393–14402, 2021. 1, 18

- [83] Heng Zhang, Qiuyu Zhao, Linyu Zheng, Hao Zeng, Zhiwei Ge, Tianhao Li, and Sulong Xu. Exploring regionword alignment in built-in detector for open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16975– 16984, 2024. 17

- [84] Shiyu Zhao, Samuel Schulter, Long Zhao, Zhixing Zhang, Vijay Kumar B G, Yumin Suh, Manmohan Chandraker, and Dimitris N. Metaxas. Taming self-training for openvocabulary object detection. In Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13938–13947, 2024. 1, 7, 17,

- 20, 21

- [85] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Regionbased language-image pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16793–16803, 2022. 4, 7, 16, 17, 20,

21

- [86] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127:302–321, 2019. 17, 20

- [87] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In European Conference on Computer Vision, pages 696–712. Springer, 2022. 8, 17, 18

- [88] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832,

- 2021. 21

[89] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Kr¨ahenb¨uhl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In European Conference on Computer Vision, pages 350–368. Springer,

- 2022. 7, 17, 20

- [90] Chaoyang Zhu and Long Chen. A survey on openvocabulary detection and segmentation: Past, present, and future. arXiv preprint arXiv:2307.09220, 2023. 20

- [91] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020. 1

## DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception Supplementary Material

### Overview

This material provides supplementary details to the main paper, including the following sections:

- • (6) Details of Proxy Token Phenomenon
- • (7) Additional Experiments

- – (7.1) Ablation Studies
- – (7.2) sanity Checks
- – (7.3) Further Details on Benchmark Results

- • (8) Additional Qualitative Analysis

- – (8.1) Analyses of Feature Correlations
- – (8.2) Comparison of Semantic Segmentation Results
- – (8.3) Comparison of Attention Maps

- • (9) Details of Experimental Settings

- – (9.1) Datasets and Evaluation Protocols
- – (9.2) Implementation Details

- • (10) Related Work

- – (10.1) Open-Vocabulary Dense Prediction
- – (10.2) Transferring VLMs to Dense Prediction Tasks
- – (10.3) Vision Foundation Models

### 6. Details of Proxy Token Phenomenon

This section primarily supplements the details of the proxy token phenomenon observed in CLIP, offering deeper insights into the rationale behind our proposed DeCLIP.

Observation. As stated in the main paper, ViT-based [19] CLIP utilizes the [CLS] token to represent the overall features of an image and performs image-text contrastive learning accordingly. Therefore, it is commonly believed that the [CLS] token comprehensively attends to all image tokens during the forward pass to obtain a “global view”, thereby enhancing the image classification process.

Unexpectedly, the [CLS] token ceased to focus on the primary object in the image starting from the 7th layer and instead redirected its attention to several image tokens in the background as shown in the first row of Figure 8. These specific image tokens continued to receive significant attention from the [CLS] token in the following encoding layers.

A similar pattern was observed in the attention maps of CLIP’s image tokens. As shown in the second row of Figure 8, we first randomly selected an image token located on the primary object in the image as the anchor image token, and then visualized its attention maps across different encoder layers. The experimental results show that the at-

tention of the anchor image token in layers 1-6 is primarily distributed over the object it belongs to. However, after the 7th layer, which is when the [CLS] token shifted its attention to several specific image tokens in the background, the anchor image token also began to focus on these specific image tokens.

Moreover, as illustrated in the third row of Figure 8, when the position of the anchor image token is shifted, the new anchor image token continues to exhibit high attention towards these specific tokens. This demonstrates that this phenomenon is not limited to a particular image token but is instead widespread across the image tokens in CLIP.

Analysis. One possible explanation for this phenomenon could be the redundancy present in image data. Images inherently carry a higher information load than text, encompassing substantial background details that are unrelated to image classification tasks. These specific background tokens may serve as “proxies” for the [CLS] token. This suggests that these tokens aggregate essential information from other image tokens, enabling the [CLS] token to form an approximate “global view” by summarizing content from them, thereby facilitating image classification. This perspective is also supported by recent studies [16, 59].

In over a decade of CNN [27, 45] development, no studies have reported similar phenomena. Therefore, we speculate that the second reason for this phenomenon may stem from the ViT architecture [19]. The classic ResNet [27] architecture consists of four stages, in which the feature resolution is halved and the number of channels is doubled at each stage. This is a process of learning sparse features, where redundant image details are progressively discarded, and feature semantics are continually enhanced. However, CLIP with a ViT architecture lacks this process. After patch embedding, the size and the number of channels in the feature map remain unchanged. As a result, the model spontaneously generates “proxy” tokens to mimic the process of learning sparse features, akin to CNN.

Effects. As discussed above, the proxy token phenomenon allows ViT CLIP to learn sparse features, which facilitate the extraction of key information from images, enhance image-text contrastive learning and reduce the optimization burden.

However, this phenomenon causes the image tokens in CLIP to indiscriminately focus on the proxy tokens in the background, rather than on the regions that are spatially or semantically related to them. Consequently, this leads to CLIP’s dense features to lack local discriminability and spatial consistency, affecting its performance in open-

CLIP CLS Token

Layer1 Layer3 Layer6 Layer7 Layer8 Layer9 Layer10 Layer11 Layer12

CLIP Image

- Token1

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

|[Figure 335]|
|---|

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

|[Figure 344]|
|---|

|[Figure 345]|
|---|

|[Figure 346]|
|---|

|[Figure 347]|
|---|

|[Figure 348]|
|---|

|[Figure 349]|
|---|

CLIP Image

- Token2

|[Figure 350]|
|---|

|[Figure 351]|
|---|

|[Figure 352]|
|---|

|[Figure 353]|
|---|

|[Figure 354]|
|---|

|[Figure 355]|
|---|

|[Figure 356]|
|---|

|[Figure 357]|
|---|

|[Figure 358]|
|---|

- Figure 8. Visualization of the “proxy” token phenomenon in the attention maps of the CLIP visual encoder. Specifically, the input image resolution is 224*224. We extract the attention weights from each attention block of CLIP and average them across the multihead dimension (after Softmax), yielding attention maps M ∈ R197×197. M[0, 1:] ∈ R1×196 represents the attention map from the [CLS] token to other image tokens (first row). M[1:197, 1:197] ∈ R196×196 represents the attention map between each image token and all image tokens. We randomly select specific image tokens’ attention map (the second and third rows, indicated by the red dots) for visualization, each with dimensions of 1*196. We reshape them to 1*14*14 and apply bilinear upsampling to 1*224*224 for better visualization.

Table 7. Ablation study on types of XContext.

Region Classification (mAcc) Semantic Segmentation (mIoU) COCO (Thing) COCO (Stuff) PASCAL Context59 ADE

XContext

Q 77.2 52.5 38.7 21.8 K 76.5 51.0 39.4 21.6

###### Q + K 77.3 53.8 39.2 21.9

vocabulary dense prediction tasks.

### 7. Additional Experiments

#### 7.1. Ablation Studies

In this section, we conduct a thorough ablation study on DeCLIP, encompassing the examination of various Xcontext implementations, the variation in the number of fine-tuning layers, the impact of the hyperparameter λ in the loss function, and the influence of the distillation baseline.

Except for the region classification experiment in Table 7, which was conducted at a resolution of 1024×1024, the region classification performance in all other experiments was assessed at a resolution of 560×560. Additionally, the semantic segmentation performance of all ablation experiments was assessed at a resolution of 336×336.

Types of Context. Since there are various implementations of Xcontext, including Q, K, and Q + K, we performed an ablation study on their performance in dense prediction tasks, including region classification (mAcc) and semantic segmentation (mIoU), as shown in Table 7. Specifically, implementing Xcontext based on K means that the last attention block of CLIP leverages K to compute the attention weight. Additionally, implementing Xcontext based on Q + K in-

Table 8. Ablation study on number of fine-tuning layers.

Region Classification (mAcc) Semantic Segmentation (mIoU) COCO (Thing) COCO (Stuff) PASCAL Context59 ADE

Fine-tuning Layers

3 62.7 47.0 38.0 21.8 6 67.1 47.8 39.0 22.3 9 70.7 50.5 39.0 22.1

12 72.2 51.3 38.7 21.8

volves first computing the attention weights of Q and K separately, and then summing them. The experimental results indicate that the performance differences among the three implementations are minimal, while the Q and K exhibits slightly better performance in dense prediction tasks. Number of fine-tuning layers. We performed an ablation study to examine the relationship between the number of fine-tuning attention blocks and dense prediction performance. The experiment was conducted on the ViT-B version of CLIP, which comprises a total of 12 attention blocks. we experiment with updating the last 3, 6, 9, and 12 attention blocks. As shown in Table 8, we observed that as the number of fine-tuning layers increased, the performance of region classification continuously improved, reaching its peak at 12 layers. However, the performance of semantic segmentation peaked at 6 layers, and as the number of layers increased further, the performance slightly declined. In practice, to balance the performance of both tasks, we chose to fine-tune all attention blocks in the implementation of DeCLIP.

Sensitivity Analysis of λ. In DeCLIP, we employ a hyperparameter λ to balance the weight between Lcontent and Lcontext. We performed an ablation study to examine the

Table 9. Ablation Study on EVA-CLIP for open-vocabulary semantic segmentation

Method Backbone Training Set ADE847 Context459 ADE150 Context59 VOC20 VOC21

CAT-Seg+CLIP [52] ViT-B/16 COCO-Stuff 12.0 19.0 31.8 57.5 94.6 77.3 CAT-Seg+CLIP [52] ViT-L/14 COCO-Stuff 16.0 23.8 37.9 63.3 97.0 82.5

CAT-Seg+EVA-CLIP [61] ViT-B/16 COCO-Stuff 11.9 17.6 30.4 52.3 94.2 74.2 CAT-Seg+EVA-CLIP [61] ViT-L/14 COCO-Stuff 14.2 21.3 34.8 56.2 95.8 80.1

CAT-Seg+DeCLIP ViT-B/16 COCO-Stuff 15.3 21.4 36.3 60.6 96.6 81.3 CAT-Seg+DeCLIP ViT-L/14 COCO-Stuff 17.6 25.9 40.7 63.9 97.7 83.9

Table 10. Ablation Study on EVA-CLIP for open-vocabulary semantic segmentation based on VLM features.

With a background category Without background category

Method

Avg VOC21 Context60 COCO-Obj VOC20 CityScape Context59 ADE COCO-Stf

CLIP [52] 18.8 9.9 8.1 49.4 6.5 11.1 3.1 5.7 14.1 EVA-CLIP [61] 23.4 12.8 15.3 55.9 12.8 13.9 7.7 9.7 18.9

ClearCLIP [38] 51.8 32.6 33.0 80.9 30.0 35.9 16.7 23.9 38.1 EVA-ClearCLIP 47.0 29.7 30.2 78.3 26.3 29.4 16.7 20.4 34.7

###### DeCLIP 59.7 35.3 36.4 85.0 32.8 39.2 21.9 25.3 41.9

Table 11. Sentitivity Analysis of hyperparameter λ.

Region Classification (mAcc) Semantic Segmentation (mIoU) COCO (Thing) COCO (Stuff) PASCAL Context59 ADE

λ

- 0.1 72.4 50.6 37.9 21.3

- 0.2 72.4 51.0 38.4 21.7 0.25 72.2 51.3 38.7 21.8

- 0.3 71.9 51.4 38.7 21.7

relationship between the hyperparameter λ and dense prediction performance. The experimental results demonstrate that our method exhibits strong robustness, and the dense prediction performance of DeCLIP does not fluctuate drastically with changes in λ. Furthermore, the results indicate that λ = 0.25 strikes a good balance between region classification capability and image segmentation performance.

Distillation Baseline. In our experiments, we used EVACLIP [61] as the baseline for DeCLIP, as we found that it demonstrated improved performance after distillation, as shown in Table 12. This can be attributed to two main factors: (1) EVA-CLIP uses the EVA02 [22] model for initializing the visual encoder. EVA02 was trained using Masked Image Modeling (MIM), thereby enhancing its compatibility with Vision Foundation Models (VFMs). (2) EVA-CLIP’s [CLS] token exhibits superior zero-shot classification capability compared to OpenAI’s model [68]. In Sec. 7.2, we conducted comprehensive sanity checks to verify whether the performance improvement of DeCLIP in dense prediction tasks is due to the use of EVA-CLIP.

#### 7.2. sanity Checks

To eliminate potential biases that EVA-CLIP [61] might introduce, we conducted additional sanity check experiments.

Table 12. Comparison of different distillation baselines.

Region Classification (mAcc) Semantic Segmentation (mIoU) COCO (Thing) COCO (Stuff) PASCAL Context59 ADE

Source

OpenAI 65.0 38.8 36.2 18.6 EVA-CLIP 72.2 51.3 38.7 21.8

Specifically, we first apply vanilla EVA-CLIP as the backbone network in the CAT-Seg [14] model and compare its performance with DeCLIP in the Open-Vocabulary Semantic segmentation (OVSS) task, as shown in Table 9. Furthermore, we re-implemented ClearCLIP [38] based on EVA-CLIP and named it EVA-ClearCLIP. Then, we compared the performance between EVA-CLIP, EVAClearCLIP, and DeCLIP in the OVSS based on VLM features task, as shown in Table 10. We did not conduct further open-vocabulary detection experiments because the baseline detectors, OV-DQUO [65] and F-ViT [68], have already used EVA-CLIP as the backbone network in their respective studies.

OVSS. As shown in Table 9, experimental results demonstrate that directly applying EVA-CLIP to CAT-Seg performs worse than OpenAI’s model. In contrast, DeCLIP significantly improves CAT-Seg’s performance across all semantic segmentation benchmarks.

OVSS based on VLM feautures. As shown in Table 10, experimental results indicate that EVA-CLIP performs slightly better than CLIP in this task, while EVAClearCLIP underperforms in comparison to ClearCLIP. However, both EVA-CLIP and EVA-ClearCLIP fall significantly short of DeCLIP’s average performance of 41.9 across the eight benchmarks.

Layer1 Layer3 Layer6 Layer7 Layer8 Layer9 Layer10 Layer11 Layer12

[Figure 359]

CLIPRegionCLIPCLIPSelfDeCLIP

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

- Figure 9. Qualitative comparison of feature correlations between DeCLIP and existing pre-fine-tuning approaches [68, 85]. Specifically, the input image resolution is 336*336. We extract the output features from each attention block of CLIP, where each feature F ∈ R441×D. Then, we compute the feature correlations FC ∈ R441×441 between the image tokens within F using cosine similarity. We randomly select a specific image token’s feature correlation (indicated by the red dots) and upsample it to a resolution of 336*336 for visualization.

Based on the results of the aforementioned experiments, we conclude that the performance improvement of DeCLIP is not attributable to the introduction of EVA-CLIP, but is instead due to the superiority of the decoupled feature enhancement strategy.

#### 7.3. Further Details on Benchmark Results

We present detailed results for the OV-COCO, OV-LVIS, and cross-dataset benchmarks to provide a comprehensive comparison of the open-vocabulary object detection task, as shown in Tables 13 and 14.

### 8. Additional Qualitative Analysis

This section further presents a qualitative experimental analysis of our proposed DeCLIP method in comparison to existing methods, including feature correlation analysis, semantic segmentation results, and attention map comparisons, thereby providing a more comprehensive demonstration of the superiority of DeCLIP’s decoupled feature enhancement strategy.

#### 8.1. Analyses of Feature Correlations

We have analyzed CLIP and found that its limitation in open-vocabulary dense prediction arises from image tokens failing to aggregate information from spatially or semantically related regions. Figure 9 presents a comparison of feature correlations among CLIP [52], DeCLIP, and exist-

ing pre-finetuning methods [68, 85] at each vision encoder layer.

This experiment provide insight into how the output features of each layer in CLIP’s visual encoder changed after fine-tuning. In this experiment, we randomly select an image token from the primary object within the image (i.e., the bird) as the anchor and visualize the cosine similarity between the anchor and the other image tokens. The experimental results indicate that the impact of various finetuning methods on the correlation of CLIP’s output features becomes noticeable starting from the 6th encoder layer.

CLIP vs. existing pre-fine-tuning methods. Rows 1, 2, and 3 of Figure 9 exhibit the changes in feature correlations of CLIP after region-level fine-tuning [68, 85]. The experimental results indicate that region-level fine-tuning enhances the feature correlations of the anchor image token to start converging towards the object it belongs to (rows 2 and 3), rather than being randomly scattered across the image (row 1).

This change is highly effective for open-vocabulary object detection tasks. As relevant features become more focused, region features exhibit enhanced discriminative power in the visual-language space when extracting the object’s region features from the image for recognition. However, these methods remain constrained in image segmentation tasks that demand pixel-level precision. As shown in the feature correlation results in rows 2 and 3 of Figure 9, most of the pixels surrounding the bird will be misclassified

- Table 13. Detailed comparison on OV-COCO and OV-LVIS benchmarks. Caption supervision indicates that the method learns from extra image-text pairs, while CLIP supervision refers to transferring knowledge from CLIP. †: Detection Transformer based detectors.

(a) OV-COCO benchmark [43]

Method Supervision Backbone APNovel50 APBase50 AP50 ViLD [24] CLIP RN50 27.6 59.5 51.2 Detic [89] Caption RN50 27.8 51.1 45.0 OV-DETR† [81] CLIP RN50 29.4 61.0 52.7 ProxyDet [29] Caption RN50 30.4 52.6 46.8 RegionCLIP [85] Caption RN50 31.4 57.1 50.4 RTGen [8] Caption RN50 33.6 51.7 46.9 BARON-KD [67] CLIP RN50 34.0 60.4 53.5 CLIM [69] CLIP RN50 36.9 - SAS-Det [84] CLIP RN50 37.4 58.5 53.0 RegionCLIP [85] Captions RN50x4 39.3 61.6 55.7 CORA† [70] CLIP RN50x4 41.7 44.5 43.8 OV-DQUO† [65] CLIP RN50x4 45.6 - -

RO-ViT [34] CLIP ViT-L/16 33.0 - 47.7 CFM-ViT [33] CLIP ViT-L/16 34.1 - 46.0 F-ViT [68] CLIP ViT-B/16 37.6 54.9 50.4 BIND [83] CLIP ViT-L/16 41.5 58.3 54.8 F-ViT [68] CLIP ViT-L/14 44.3 64.1 59.0

F-ViT+DeCLIP CLIP ViT-B/16 41.1 57.8 53.5 F-ViT+DeCLIP CLIP ViT-L/14 46.2 65.2 60.3 OV-DQUO+DeCLIP† CLIP ViT-B/16 46.1 56.3 53.6 OV-DQUO+DeCLIP† CLIP ViT-L/14 48.3 60.0 56.9

(b) OV-LVIS benchmark [25]

Method Supervision Backbone mAPr mAPc mAPf mAP ViLD [24] CLIP RN50 16.6 24.6 30.3 25.5 OV-DETR† [81] CLIP RN50 17.4 25.0 32.5 26.6 BARON-KD [67] CLIP RN50 22.6 27.6 29.8 27.6 RegionCLIP [85] Caption RN50x4 22.0 32.1 36.9 32.3 CORA+† [70] Caption RN50x4 28.1 - - SAS-Det [84] CLIP RN50x4 29.1 32.4 36.8 33.5 CLIM [69] CLIP RN50x64 32.3 - - F-VLM [37] CLIP RN50x64 32.8 - - 34.9 F-ViT [68] CLIP ViT-B/16 25.3 21.8 29.1 25.2 RTGen [8] Caption Swin-B 30.2 39.9 41.3 38.8 BIND [83] CLIP ViT-L/16 32.5 33.4 35.3 33.2 Detic [89] Caption Swin-B 33.8 - - 47.0 CFM-ViT [33] CLIP ViT-L/14 33.9 - - 36.6 RO-ViT [34] CLIP ViT-H/16 34.1 - - 35.1 F-ViT [68] CLIP ViT-L/14 34.9 34.6 35.6 35.1 ProxyDet [29] Caption Swin-B 36.7 - - 41.5 CoDet [47] Caption ViT-L/14 37.0 46.3 46.3 44.7 OV-DQUO† [65] CLIP ViT-L/14 39.3 - - -

F-ViT+DeCLIP CLIP ViT-B/16 26.8 22.4 29.8 26.0 F-ViT+DeCLIP CLIP ViT-L/14 37.2 35.2 36.5 36.0 OV-DQUO+DeCLIP† CLIP ViT-B/16 31.0 - - 27.7 OV-DQUO+DeCLIP† CLIP ViT-L/14 41.5 - - 34.6

- Table 14. Detailed comparison of transferring LVIS-trained detectors to the COCO and Objects365 datasets.

#### 8.2. Comparison of Semantic Segmentation Results

Figure 10 shows a qualitative comparison of MaskCLIP [87], SCLIP [63], ClearCLIP [38], and our proposed DeCLIP across the Context59 [48], COCOStuff [3], Cityscapes [15], and ADE20K [86] datasets. We observe that, compared to other methods, DeCLIP consistently produces higher-quality and more precise segmentation maps.

COCO [43] Objects365 [58]

Method

AP AP50 AP75 AP AP50 AP75 APs APm APl Supervised Baseline [24] 46.5 67.6 50.9 25.6 38.6 28.0 - - ViLD [24] 36.6 55.6 39.6 11.8 18.0 12.6 - - DetPro [20] 34.9 53.8 37.4 12.1 18.8 12.9 4.5 11.5 18.6 BARON [67] 36.2 55.7 39.1 13.6 21.0 14.5 5.0 13.1 20.7 F-VLM [37] 37.9 59.6 41.2 16.2 25.3 17.5 - - CoDet [47] 39.1 57.0 42.3 14.2 20.5 15.3 - - RO-ViT [35] - - - 17.7 27.4 19.1 - - CLIPSelf [68] 40.5 63.8 44.3 19.5 31.3 20.7 9.7 23.2 35.5 DeCLIP 41.0 64.6 44.8 20.0 32.2 21.2 10.0 24.4 36.7

Specifically, benefiting from content feature distillation, which improves the discriminability of local features, DeCLIP successfully recognizes trees, people, and curbs in the images, as shown in columns 1, 5, and 6 of Figure 10, whereas other models fail. Furthermore, our observation indicates that the distillation of context features improves the spatial consistency of DeCLIP’s local features, leading to smoother and less noisy segmentation results compared to other models, as demonstrated in columns 2, 3, 4, and 7 of

as “bird” rather than to be “background”.

CLIP vs. DeCLIP. Rows 1 and 4 of Figure 9 exhibit the changes in feature correlations of CLIP after decoupled feature enhancement strategy. The experimental results indicate that DeCLIP enhances the feature correlations of the anchor image token to closely align with the object it represents, in clear contrast with other existing pre-fine-tuning approaches (row 2 and 3). This experiment reveals why DeCLIP is better suited for image segmentation tasks than existing methods. Additionally, the experiment demonstrates DeCLIP’s also superiority over current pre-finetuning approaches in region classification tasks. As shown in the feature correlation map of DeCLIP’s 12th layer, the image regions corresponding to the same object as the anchor image token display a strong red color, indicating a very high feature correlation strength in these regions, thereby enhancing the discriminative power of region features within the visual-language space.

- Figure 10. This demonstrates the superiority of our decoupled feature enhancement strategy.

8.3. Comparison of Attention Maps

- Figure 11 offers a detailed comparison of attention maps between CLIP and our proposed DeCLIP approach. As DeCLIP involves unsupervised fine-tuning, we conducted tests using diverse cross-domain image styles to thoroughly assess its generalization capability. Specifically, we utilized generative models [56] to generate test images in various styles such as ink painting, watercolor, sketch, animation, and oil painting, which are depicted on the left side of Figure 11. These cross-domain test images were not part of the fine-tuning dataset for DeCLIP (i.e., COCO2017 [43]).

Context59

COCO Stuff

CityScape

ADE

|[Figure 375]|
|---|

|[Figure 376]|
|---|

|[Figure 377]|
|---|

|[Figure 378]|
|---|

|[Figure 379]|
|---|

|[Figure 380]|
|---|

|[Figure 381]|
|---|

Image

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

|[Figure 385]|
|---|

|[Figure 386]|
|---|

|[Figure 387]|
|---|

|[Figure 388]|
|---|

GTSCLIPClearCLIPDeCLIP

|[Figure 389]|
|---|

|[Figure 390]|
|---|

|[Figure 391]|
|---|

|[Figure 392]|
|---|

|[Figure 393]|
|---|

|[Figure 394]|
|---|

|[Figure 395]|
|---|

MaskCLIP

|[Figure 396]|
|---|

|[Figure 397]|
|---|

|[Figure 398]|
|---|

|[Figure 399]|
|---|

|[Figure 400]|
|---|

|[Figure 401]|
|---|

|[Figure 402]|
|---|

|[Figure 403]|
|---|

|[Figure 404]|
|---|

|[Figure 405]|
|---|

|[Figure 406]|
|---|

|[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

|[Figure 410]|
|---|

|[Figure 411]|
|---|

|[Figure 412]|
|---|

|[Figure 413]|
|---|

|[Figure 414]|
|---|

|[Figure 415]|
|---|

|[Figure 416]|
|---|

- Figure 10. Qualitative comparison of the open-vocabulary semantic segmentation results between DeCLIP and existing approaches [38, 63, 87].

In addition, we performed a detailed comparison of attention maps between CLIP and DeCLIP on in-domain images. Specifically, we selected a subset of images from the Object365 [58] validation set for testing, with the results shown on the right-hand side of Figure 11. During the testing phase, we first resized the images to 336×336 pixels and then fed them into the model to extract features. Subsequently, we randomly selected an anchor image token and visualized its attention map in the 12th attention block, as indicated by the red dots on the test images in Figure 11. For details on the calculation process of the attention map, please refer to Figure 8.

As depicted in Figure 11 , due to the proxy token phenomenon, the heatmap generated by the anchor image token in vanilla CLIP frequently lacks semantic consistency with its corresponding object. In contrast, despite being fine-tuned only on the natural scene dataset COCO, DeCLIP demonstrates significant semantic relevance for both in-domain and cross-domain test images. Moreover, benefiting from context feature distillation, DeCLIP’s semantic correlations demonstrate remarkably fine granularity, effectively outlining the boundaries of each object semantically associated with the anchor image token.

### 9. Details of Experimental Settings

In this section, we present further details and configurations utilized in our experiments.

#### 9.1. Datasets and Evaluation Protocols

Open-Vocabulary Detection. Following established settings [68, 70, 82], we evaluated our model on the OV-COCO [43], OV-LVIS [25], COCO, and Object365 [58] datasets. The OV-COCO dataset includes 48 base categories and 17 novel categories. The training set contains only base categories, totaling 107,761 images, while the validation set comprises 4,836 images featuring both base and novel categories. We report the mean Average Precision (mAP) at an Intersection over Union (IoU) threshold of 0.5 for novel categories. The OV-LVIS dataset consists of 1,203 categories. Its training set includes only 461 common and 405 frequent categories, totaling 100,170 images. The validation set contains 19,809 images with common, frequent, and rare categories. We report the mAP for rare categories at IoU thresholds ranging from 0.5 to 0.95. Additionally, we provide cross-dataset evaluation results on the COCO and Object365 validation sets for models trained on OV-LVIS to assess generalization across domains.

Image CLIP Ours Image CLIP Ours

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

- Figure 11. Comprehensive comparison of attention maps between CLIP and DeCLIP. The left side presents images of various styles generated by generative models [56]. The images presented on the right-hand side comes from a subset of images in the Object365 [58] validation set. Anchor image token marked in red.

Open-Vocabulary Semantic Segmentation. In line with prior studies [14], we trained our model on the COCOStuff dataset [3], which comprises 118,000 images with dense annotations across 171 categories. We then evaluated the model on the ADE20K [86], PASCAL VOC [21], and PASCAL-Context [48] datasets. ADE20K [86] includes 20,000 training images and 2,000 validation images, with two category sets: A-150 (150 common categories) and A847 (847 categories) [18]. PASCAL-Context consists of 5,000 training and validation images, with category sets PC59 (59 categories) and PC-459 (459 categories). The PASCAL VOC dataset includes 1,500 images for training and validation, featuring category sets PAS-20 (20 categories) and PAS-21 (20 object categories plus one background class). We used mean Intersection over Union (mIoU) as the evaluation metric in all experiments.

Open-Vocabulary Semantic Segmentation Based on VLM Features. To further evaluate DeCLIP, we assessed it on six commonly used semantic segmentation benchmarks: PASCAL VOC 2012 [21], PASCAL Context [48], Cityscapes [15], ADE20K [86], COCO-Stuff [43], and COCO-Object [3]. For datasets including a background category, we refer to them as VOC21 and Context60; those without a background category are termed VOC20 and Context59. Consistent with previous experiments, we used mIoU as the evaluation metric across these benchmarks.

#### 9.2. Implementation Details

DeCLIP. DeCLIP was trained on training set images from the COCO2017 [43] dataset using 8 GPUs, each with a batch size of 2, for 6 epochs (about 44 min/epoch on 8×4090 GPUs). The AdamW [46] optimizer with a learning rate of 1e−5 and a weight decay of 0.1 was employed during the training process.

During the content feature distillation process, the image is divided into k blocks, where k = m × n, and m and n are randomly sampled from the range [1, 6]. After cropping k image blocks from the original image, the patches are resized to a resolution of 224×224 and subsequently fed into the teacher model to generate the corresponding [CLS] tokens for content feature distillation. Unless stated otherwise, our experiments were conducted using EVA-CLIP [61].

In the process of context feature distillation, given the distinct image preprocessing methods with varying means and standard deviations used by CLIP and VFM during pretraining, we incorporated the corresponding parameters during the distillation process. Additionally, to address the potential variation in patch sizes between CLIP and VFM (e.g., CLIP uses a 16-patch size while DINOV2 uses a 14patch size), we adjusted the image resolutions to maintain consistency in the number of image tokens. For example, we set the resolution of CLIP to 1024 and that of DINOV2

to 896, ensuring both models possess 4096 image tokens. The weight λ for context feature distillation is established at 0.25. Unless specified otherwise, our default VFM is DINOv2 [51].

Open-vocabulary detection. In the open-vocabulary detection experiment, DeCLIP was evaluated in two model baselines: F-ViT [68] and OV-DQUO [65]. These baselines are constructed based on transfer learning principles, utilizing the image encoder of CLIP for feature extraction while maintaining the backbone network frozen during training and only training the task-specific components. The two baseline models utilize distinct detector architectures: F-ViT employs the traditional Faster R-CNN [55] architecture, whereas OV-DQUO utilizes the modern Detection Transformer [4] architecture. This enables a thorough assessment of the efficacy of our proposed approach.

We maintained the default training strategies and hyperparameter configurations from the original studies for both baseline models to uphold experiment fairness. The only modification was to the temperature parameter when integrating DeCLIP for object detection. For F-ViT, the temperature was set to 45 for the OV-COCO benchmark and 90 for the OV-LVIS benchmark. In OV-DQUO, the temperature was set to 50 for both the OV-COCO and OV-LVIS benchmarks.

Open-Vocabulary Semantic Segmentation. In the openvocabulary semantic segmentation experiments, we applied DeCLIP to the CAT-Seg [14] baseline. For all experiments, we adhered to the default training and inference settings of vanilla CAT-Seg, replacing only the image encoder with DeCLIP.

Open-Vocabulary Semantic Segmentation Based on VLM Features. During inference, we resized the shorter side of images to 448 pixels and employed a sliding window strategy with a window size of 336×336 and a stride of 112×112. For all datasets, we generate textual descriptions by utilizing the standard ImageNet prompts [52] in conjunction with their respective class names. No post-processing steps were applied.

### 10. Related Work

#### 10.1. Open-Vocabulary Dense Prediction

Open-vocabulary dense prediction aims to detect and segment visual concepts from novel categories using textual descriptions, extending beyond the base categories on which the model was trained. According to recent surveys [90], methods in this field can be broadly classified into four categories: knowledge distillation-based [26, 66, 67, 81], pseudo-labeling [65, 80, 84, 85, 89], region-aware training [23, 33, 35, 70, 73], and transfer learning-based approaches [17, 32, 37, 39, 42, 65, 68].

Knowledge distillation-based methods, such as ViLD [24], BARON [67], and OADP [66], propose various distillation frameworks to transfer the generalized classification knowledge of VLMs [52, 61] into dense prediction models. Pseudo-labeling methods like RegionCLIP [85] and SAS-Det [84] enhance region-text alignment by generating pseudo-labels for image-text pairs using VLMs or selftraining techniques. Region-Aware Training methods, exemplified by CORA [70], improve the object classification accuracy of CLIP by learning region prompts.

Transfer Learning-Based methods [14, 17, 30–32, 42, 65, 68, 77, 78] utilize the image encoder of VLM as a feature extractor and exclusively train lightweight task-specific components. These methods have become mainstream in open-vocabulary dense prediction due to their broad applicability. While leveraging VLMs as feature extractors offers significant advantages due to their comprehensive pre-training, directly applying these image-level models to dense prediction tasks often results in domain shift issues [68, 70], thereby limiting their performance. In this paper, we integrate DeCLIP into transfer learning-based object detection baselines F-ViT and OV-DQUO, as well as the image segmentation baseline CATSeg, to enhance their performance in open-vocabulary dense prediction tasks.

#### 10.2. Transferring VLMs to Dense Prediction Tasks

As VLMs [52, 61] were initially trained on image-text pairs, the direct application of these image-level models to dense prediction tasks, which require region-level or pixel-level semantic understanding, results in significant performance degradation. Several studies have attempted to address this limitation through fine-tuning strategies. These approaches can be broadly categorized into joint fine-tuning and prefine-tuning approaches.

Joint fine-tuning methods fine-tune CLIP while training task-specific components [14, 30, 31, 39, 42, 72, 77]. For instance, CAT-Seg [14] proposes an attention fine-tuning strategy based on ViT CLIP, which generalizes well to unseen categories. MAFT [30] leverages attention bias to finetune CLIP for mask classification.

Pre-fine-tuning methods directly fine-tune CLIP using cost-efficient techniques [49, 68–70, 85]. For instance, CLIM [69] employs a mosaic augmentation technique to stitch multiple images into a single image, enabling each sub-image to serve as a pseudo-region for region-text contrastive learning. CLIPSelf [68] enhances CLIP’s region classification accuracy by maximizing cosine similarity between its region representations and the corresponding image crop representations.

Despite the promising results of the two categories of fine-tuned methods, they continue to exhibit certain limitations. In contrast to these studies, we conduct an analysis of CLIP and identify that its limitation in open-vocabulary

dense prediction stems from the inability of image tokens to effectively aggregate information from spatially or semantically related regions. To address this, we propose integrating VFMs into the pre-fine-tuning process and decoupling features for distillation, thereby improving the discriminability and spatial consistency of CLIP’s local features.

#### 10.3. Vision Foundation Models

Vision foundation models, including the Self-Supervised Representation Learning (SSL) series [1, 2, 5, 10, 51, 88] and the SAM series [36, 54], which are trained on largescale segmentation data, demonstrate the ability to extract features that exhibit strong spatial consistency.

SSL is a key area in computer vision that focuses on learning meaningful visual features without manual annotations [1, 2, 5, 10, 51, 88]. Vision models trained through SSL can extract image features with excellent spatial understanding. For example, the DINO series [5, 51] can identify similar semantic regions across different images and segment main objects without explicit supervision. Another prominent vision foundation model is SAM [36, 54], which demonstrates similarly outstanding spatial understanding. Trained on the extensive SA-1B segmentation dataset, SAM can accurately capture and segment objects regions in images based on prompts.

Recently, some studies have explored the combination of CLIP with VFM, such as SAM-CLIP [64], OV-SAM [79], and FrozenSeg [11], with the goal of integrating SAM’s powerful image segmentation capabilities and CLIP’s zeroshot semantic perception capabilities. AM-RADIO [53] trains a unified vision model through multi-teacher distillation from multiple foundational vision models such as CLIP, DINOv2, and SAM. However, SAM-CLIP, OVSAM, and FrozenSeg focus on integrating CLIP into SAM rather than enhancing CLIP itself as DeCLIP does. AMRADIO does not support OVSS, as confirmed by its authors in Github issues (No. 81, 55, and 42). Another study that solves similar problems to DeCLIP is ViT-Register [16]. However, unlike DeCLIP, ViT-Register [16] does not solve the dense perception deficiency arising from CLIP’s imagetext alignment.

