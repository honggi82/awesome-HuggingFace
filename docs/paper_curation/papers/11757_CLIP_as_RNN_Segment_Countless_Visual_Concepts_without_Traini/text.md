## CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor

# arXiv:2312.07661v3[cs.CV]7May2024

Shuyang Sun1,2* Runjia Li1* Philip Torr1 Xiuye Gu2† Siyang Li2† 1University of Oxford 2Google Research

{kevinsun, runjia, phst}@robots.ox.ac.uk {siyang, xiuyegu}google.com

https://torrvision.com/clip_as_rnn/

[Figure 1]

Figure 1. We propose CLIP as RNN (CaR) to segment concepts in a vast vocabulary, including fictional characters, landmarks, brands, everyday objects, and referring expressions. This figure shows our qualitative results. More visualizations are included in the supplementary material. Best viewed in color and with zoom-in.

### Abstract

Existing open-vocabulary image segmentation methods require a fine-tuning step on mask labels and/or image-text datasets. Mask labels are labor-intensive, which limits the number of categories in segmentation datasets. Consequently, the vocabulary capacity of pre-trained VLMs is severely reduced after fine-tuning. However, without finetuning, VLMs trained under weak image-text supervision tend to make suboptimal mask predictions. To alleviate these issues, we introduce a novel recurrent framework that progressively filters out irrelevant texts and enhances mask quality without training efforts. The recurrent unit is a two-stage segmenter built upon a frozen VLM. Thus, our model retains the VLM’s broad vocabulary space and equips it with segmentation ability. Experiments show that our method outperforms not only the training-free counterparts, but also those fine-tuned with millions of data samples, and sets the new state-of-the-art records for both zeroshot semantic and referring segmentation. Concretely, we improve the current record by 28.8, 16.0, and 6.9 mIoU on Pascal VOC, COCO Object, and Pascal Context.

*The first two authors contribute equally to this work. The majority of this work was done during Shuyang’s internship at Google Research.

†Equal advising.

### 1. Introduction

Natural language serves as a bridge to connect visual elements with human-communicable ideas by transforming colors, shapes, and objects etc. into descriptive language. On the other hand, human can use natural language to easily instruct computers and robotics to perform their desired tasks. Built upon the revolutionary vision-language model trained on Internet-scale image-text pairs, e.g., CLIP [49], a variaty of studies [10, 35, 39, 42, 50, 55, 68, 76, 84] have explored using pre-trained VLMs for open-vocabulary image segmentation — to segment any concepts in the image described by arbitrary text queries.

Among these advances, several works [35, 40, 76] have integrated pre-trained VLMs with segmenters fine-tuned on bounding boxes and masks. While these methods exhibit superior performances on segmentation benchmarks with common categories, their ability to handle a broader vocabulary is hampered by the small category lists in the segmentation datasets used for fine-tuning. As depicted in Figure 2, even though all three methods incorporate CLIP [49], those relying on fine-tuning with mask annotations [35, 40] fail to recognize the concepts like Pepsi and Coca Cola.

Since box and mask annotations are expensive, another line of works [10, 39, 42, 50, 51, 68] seek to fine-tune the VLM and/or auxiliary segmentation modules with image-

OVSeg [35] Grounded SAM [40] CaR (Ours)

[Figure 2]

[Figure 3]

[Figure 4]

- Figure 2. Our method CaR can fully inherit the vast vocabulary space of CLIP, by directly using features from a pre-trained VLM, i.e., CLIP, without any fine-tuning. Although the scene in the image is simple, state-of-the-art methods fine-tuned on segmentation datasets [35, 40] fail to segment and recognize Pepsi and Coca Cola correctly.

level annotations only, e.g., paired image-text data obtained from the Internet. This would lead to a complicated finetuning pipeline. Besides, these segmentation models often have suboptimal mask qualities, as image-level labels cannot directly supervise pixel grouping.

In this paper, we eliminate the fine-tuning on mask annotations or additional image-text pairs to fully preserve the extensive vocabulary space of the pre-trained VLM. However, the pre-training objectives of VLMs are not specifically designed for dense predictions. As a result, existing approaches [14, 37, 84] that do not fine-tune the VLMs struggle to generate accurate visual masks corresponding to the text queries, particularly when some of the text queries refer to non-existing objects in the image. To address this issue, we repeatedly assess the degree of alignment between mask proposals and text queries, and progressively remove text queries with low confidence. As the text queries become cleaner, better mask proposals are consequently obtained. To facilitate this iterative refinement, we propose a novel recurrent architecture with a two-stage segmenter as the recurrent unit, maintaining the same set of weights across all time steps. The two-stage segmenter consists of a mask proposal generator and a mask classifier to assess the mask proposals. Both are built upon a pre-trained CLIP model without modifications. Given an input image and multiple text queries, our model recurrently aligns the visual and textual spaces and generates refined masks as the final output, continuing until a stable state is achieved. Owing to its recurrent nature, we name our entire framework as CLIP as RNN (CaR).

Experimental results demonstrate our approach is remarkably effective. In comparison with methods that do not use additional training data, i.e., zero-shot open-vocabulary semantic segmentation, our approach outperforms the prior art by 28.8, 16.0, and 6.9 mIoU on Pascal VOC [19], COCO Object [36], and Pascal Context [45], respectively. Impressively, even when pitted against models fine-tuned on extensive additional data, our strategy surpasses the best record by 12.6, 4.6, and 0.1 on the three aforementioned datasets, respectively. To assess our model’s capacity to handle more complex text queries, we evaluate on the re-

ferring image segmentation benchmarks, Ref-COCO, RefCOCO+ and RefCOCOg. CaR outperforms the zero-shot counterparts by a large margin. Moreover, we extend our method to the video domain, and establish a zero-shot baseline for the video referring segmentation on Ref-DAVIS 2017 [29]. As showcased in Figure 1, our proposed approach CaR exhibits remarkable success across a broad vocabulary spectrum, effectively processing diverse queries from celebrities and landmarks to referring expressions and general objects. Our contributions can be summarized as follows: 1. By constructing a recurrent architecture, our method CaR performs visual segmentation with arbitrary text queries in a vast vocabulary space without the need of fine-tuning. 2. When compared with previous methods on zero-shot openvocabulary semantic segmentation and referring segmentation, our method CaR outperforms the prior state of the art by a large margin.

### 2. Related Work

Open-vocabulary segmentation with mask annotations. The success of VLMs [25, 34, 49, 59, 73, 78, 79] has motivated researchers to push the boundaries of traditional image segmentation tasks, moving them beyond fixed label sets and into an open vocabulary by fine-tuning or training VLMs on segmentation datasets [20, 22, 26, 32, 35, 40, 44, 70, 76, 80, 81, 85]. However, as collecting mask annotations for a vast range of fine-grained labels is prohibitively expensive, existing segmentation datasets, e.g. [4, 19, 36, 45, 83] have limited vocabularies. Methods fine-tuned on these mask annotations reduce the openvocabulary capacity inherited from the pre-trained VLMs. In this work, we attempt to preserve the completeness of the vocabulary space in pre-trained VLMs.

Open-vocabulary segmentation without mask supervision. Several works [6, 10, 11, 23, 42, 46, 50, 51, 55, 68, 69, 84] avoid the aforementioned vocabulary reduction issue by not fine-tuning on any mask annotations. Instead, researchers allow semantic grouping to emerge automatically without any mask supervision. GroupViT [68] learns to progressively group semantic regions with weak supervision, using only image-text datasets. Furthermore, it is possible to use a pre-trained VLM for open-vocabulary segmentation without any additional training [27, 55, 84]. For example, MaskCLIP [84] enables CLIP to perform openvocabulary segmentation by only modifying its image encoder. However, these methods often suffer from inferior segmentation performance due to the lack of mask supervision, and the modification of the pre-trained VLMs. CaR is closely related to these approaches, we are both in a zeroshot manner without training. CaR stands out by proposing a recurrent framework on a VLM with fixed weights and no alternation on its architecture. Note that our zeroshot is different from the zero-shot semantic segmentation [2, 3, 17, 24, 33, 66, 84] that mirrors the seen/unseen

(a) Compressed Pipeline

(b) Unfolded Pipeline

Final Mask Prediction

|Mask 𝑦|
|---|

|Mask 𝑦|
|---|

|Mask 𝑦|
|---|

Stop & Post-Process

Manchester United, Barcelona, Arsenal, Manchester City,

Manchester United, Barcelona, Arsenal, Manchester City

Y

𝝈 𝝈 ManchesterUnited,

N

… ℎ == ℎ ?

…

ℎ

Segmenter

Segmenter

Segmenter

Manchester City

| | |
|---|---|
|Image 𝑥| |

| | |
|---|---|
|Image 𝑥| |

| | |
|---|---|
|Image 𝑥| |

ℎ ℎ ℎ

Text query ℎ

(c) Details of each stage

Manchester United, Barcelona, Arsenal, Manchester City,

…

Segmenter

[Figure 5]

MaskClassifier (CLIP)𝑔(⋅,⋅)

Manchester United, Barcelona, Arsenal, Manchester City

[Figure 6]

…

Proposal Generator (CLIP+CAM) 𝑓(⋅.⋅)

[Figure 7]

[Figure 8]

𝝈

…

[Figure 9]

[Figure 10]

Image 𝑥

Text query ℎ Final Mask Prediction

Mask Proposals 𝑦

Image+Visual Prompts 𝑥

- Figure 3. The overall framework of our method CaR. (a), (b): given an image, the user provides a set of text queries that they are

interested to segment. This initial set, denoted by h0, may refer to non-existing concepts in the image, e.g., Barcelona and Arsenal. In the t-th time step, the frozen segmenter evaluates the degree of alignment between each mask and text query from the previous time step, ht−1, and then low-confidence queries are eliminated by the function σ. (c) depicts the detailed architecture of our two-stage segmenter. It consists a mask proposal generator f(·, ·), and a mask classifier g(·, ·) that assesses the alignment of each mask-text pairs.

sequential data, such as text, speech, and time series. A basic RNN, commonly known as a vanilla RNN, uses the same set of weights to process data at all time steps. At each time step t, the process can be expressed as follows:

class separation from zero-shot classification in earlier ages. Segmentation with VLM-generated pseudo-labels. As an alternative direction, recent works have exploited pretrained VLMs to generate pseudo-masks in a fixed label space, requiring only image-level labels or captions for training [1, 37, 41, 52, 55, 67, 71, 84]. Once pseudo mask labels are obtained, a segmenter with a fixed vocabulary (e.g., DeepLab [12, 13]) can be trained in a fully supervised manner. Among these, CLIP-ES [37] is particularly relevant as it directly uses CLIP for pseudo-mask generation given the class names in ground-truth. However, CLIPES [37] requires pseudo-label training while we do not.

ht = σ(Whhht−1 + Wxhxt + bh), (1) yt = Whyht + by. (2)

xt represents the input, and ht represents the hidden state which serves as the “memory” to store information from previous steps. yt denotes the output. Whh, Wxh, and Why are weight matrices, bh and by refer to bias terms, and σ denotes a thresholding function to introduce non-linearity.

Progressive refinement for image segmentation. Progressive refinement in image segmentation has seen significant advancements through various approaches. Recent works [8, 15, 16, 60, 62, 75] such as Cascade RCNN [7], DETR [8] and CRF-RNN [82] combine a detector (R-CNN [21]), a transformer [61] or a segmenter (denseCRF [31]) repeatedly for iterative refinement. We kindly note that all these works are designed for supervised image instance or semantic segmentation in a closed-set vocabulary. Our method does not require any training effort, yet our way of progressive refinement is fundamentally different from these methods.

An RNN’s core lies in its hidden state, ht, which captures information from past time steps. This empowers RNNs to exploit temporal dynamics within sequences. In our approach CaR, we design a similar process - iteratively aligning the textual and visual domains by assessing the accuracy of each text query through a segmenter, using the same set of weights as well. The text queries at each step act like the RNN’s hidden state, representing the entities identified in the image at each specific time step.

#### 3.2. Overview

As depicted in Figure 3(a) and (b), our training-free framework operates in a recurrent manner, with a fixedweight segmenter shared across all time steps. In the t-th time step, the segmenter receives an image xt ∈ R3×H×W and a set of text queries ht−1 from the preceding step as the input. It then produces two outputs: a set of masks

### 3. CLIP as Recurrent Neural Networks

#### 3.1. A Recap on Recurrent Neural Networks

We begin with a concise overview of recurrent neural networks (RNN). RNNs are specifically designed to process

Algorithm 1 Pseudo-code of CLIPasRNN in PyTorch style.

# img: the input image with shape (3, H, W) # h_0: a list of the initial N_0 text queries. # clip: the CLIP model encoding the image and texts. # cam: the gradient-based CAM model for mask proposal

generation. # eta: a threshold to binarize the masks for visual

prompting. # theta: a threshold defined in Eq. 6. h_{t-1} = h_0 while len(h_{t-1}) > 0:

# logits: [1, len(h_{t-1})] logits = clip(img, h_{t-1}) scores = softmax(logits, dim=-1)

# proposals: [len(h_{t-1}), H, W] proposals = cam(clip, img, scores) # prompted_img: [len(h_{t-1}), H, W] prompted_imgs = apply_visual_prompts(img, proposals

, eta) # mask_logits: [len(h_{t-1}), len(h_{t-1})]

mask_logits = clip(prompted_imgs, h_{t-1}) mask_scores = softmax(mask_logits, dim=-1) # diag_scores: [len(h_{t-1})]

diag_scores = diagonal(mask_scores) h_t = [] for score, label in zip(diag_scores, h_{t-1}):

if score > theta: h_t.append(label) if len(h_t) == len(h_{t-1}): break h_{t-1} = h_t final_masks = post_process(proposals)

t−1×H×W corresponding to Nt−1 input text queries, and the updated text queries ht for the subsequent step. For image segmentation, all different time steps share the same xt.

yt ∈ [0,1]N

To delve deeper into the design of our framework, we formulate its operations through Eq. (3) to Eq. (5).

yt = f(xt,ht−1;Wf). (3)

Here the function f(·,·) represents the mask proposal generator and Wf denotes its pre-trained weights. The mask proposal generator processes the input image xt and the text queries at previous step ht−1 to generate candidate mask proposals yt. Given the mask proposal generator is not pretrained for dense prediction, the mask proposals yt from f(·,·) are inaccurate. To assess these mask proposals, we draw visual prompts e.g., red circles or background blur, to the input xt, based on mask proposals to highlight the masked area on the image. The visual prompting function v(·,·) is defined as:

x′t = v(xt,yt). (4)

Here x′t represent Nt−1 images with the visual prompts. The prompted images x′t are then passed to the mask classifier g(·,·) with the pre-trained weights Wg, along with the text queries ht−1, to compute a similarity matrix Pt. The entire process of the mask classifier can be defined as:

Pt = g(x′t,ht−1;Wg). (5)

Finally, after going through a thresholding function σ(·), text queries with similarity scores lower than the threshold

Background Blur

Background Gray

Background Mask

Red Circle Red Contour

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Figure 4. Examples of visual prompts given a mask on the man wearing the jersey of Manchester United.

θ will be removed so that the text queries ht = σ(Pt) for the next step t are obtained. ht is a potentially reduced set of ht−1. Details of the thresholding function will be given in Section 3.3. This recurrent process continues until the text queries remain unchanged between consecutive steps, i.e., ht == ht−1. We use T to denote this terminal time step. Finally, we apply post-processing described in Section 3.4 to the mask proposals yT generated in the final time step.

The pseudo-code in PyTorch-style is given in Algorithm 1. Note that users provide the initial text queries h0, which are unrestricted and can include general object classes (“cat”), proper nouns (“Space Needle”), referring phrases (“the man in red jacket”), etc.

#### 3.3. The Two-stage Segmenter

In this section, we explain the two components of our segmenter, i.e. a mask proposal generator and a mask classifier, which serve as the recurrent unit. As illustrated in Figure 3(c), the mask proposal generator first predicts a mask for each text query and then the mask classifier filters out irrelevant text queries based on the degree of alignment with their associated masks. We use the frozen pre-trained CLIP model weights for both the proposal generator and classifier to fully preserve the knowledge encapsulated in CLIP.

Mask proposal generator. To predict the mask proposal yt, a gradient-based Class-Activation Map (gradCAM) [37, 54] is applied to the pre-trained CLIP. More specifically, the image xt and text queries ht−1 are first fed into CLIP to get a score between the image and each text. We then backpropagate the gradients of the score of each text query (i.e., class) from the feature maps of the CLIP image encoder to obtain a heatmap. Unless otherwise specified, we use the Class Activation Map (CAM) and class affinity (CAA) module from CLIP-ES [37] as our mask proposal generator, with no further training of the CLIP model required. Apart from the text queries at the current step, we explicitly add a set of background queries describing categories that do not exist in the user text queries and calculate their gradients. This helps to suppress the activation from irrelevant texts (e.g., Barcelona and Arsenal in Figure 3) in the subsequent mask classification process. More details of how CLIP works with gradCAM are provided in the appendix.

Mask classifier. The masks from the proposal generator may be noisy because the input texts are from an unrestricted vocabulary and may refer to non-existing objects in the input image. To remove this type of proposals, we apply

another CLIP model to compute a similarity score between each query and its associated mask proposal. A straightforward approach is blacking out all pixels outside the mask region, as shown in the rightmost image in Figure 4, and then computing the visual embedding for the foreground only. However, recent works [41, 56] have found several more effective visual prompts which can highlight the foreground as well as preserve the context in the background. Inspired by this, we apply a variety of visual prompts, e.g., red circles, bounding boxes, background blur and gray background to guide the CLIP model to focus on the foreground region. A threshold η is set to first binarize the mask proposals yt before applying these visual prompts to the images. Please refer to the supplementary material for more implementation details. After applying visual prompts, we obtain Nt−1 different prompted images, corresponding to Nt−1 text queries (ht−1). We feed these images and text queries into the CLIP classifier g(·,·) followed by a softmax operation along the text query dimension to get the similarity matrix Pt ∈ RN

t−1×Nt−1 given the image and text embeddings. We only keep the diagonal elements of Pt as the matching score between the i-th mask and the i-th query. If the score is lower than a threshold θ, the query and its mask are filtered out. Mathematically, the thresholding function σ(·) is defined as follows:

hit−1, if Ptii ≥ θ NULL, if Ptii < θ

hit = σ(Ptii) =

(6)

where Ptii is the i-th element of the diagonal of the normalized similarity matrix, and θ is a manually set threshold. NULL represents that the i-th text query is filtered out and will not be input to next steps.

#### 3.4. Post-Processing

Once the recurrent process stops, we start to post-process yT, the masks from the final step T. We employ dense conditional random field (CRF) [31] to refine mask boundaries. When constructing the CRF, the unary potentials are calculated based on the mask proposals of the last step. All hyper-parameters are set to the defaults in [31]. Finally, an argmax operation is applied to the mask output of denseCRF along the dimension of text queries. Thus, for each spatial location of the mask we only keep the class (text query) with the highest response.

Additionally, we propose to ensemble the CRF-refined masks with SAM [30], as an optional post-processing module. This begins with generating a set of mask proposals from SAM using the automask mode, without entering any prompts into SAM. To match these SAM proposals with the masks processed by denseCRF, we introduce a novel metric: the Intersection over the Minimum-mask (IoM). If the IoM between a mask from SAM and a CRFrefined mask surpasses a threshold ϕiom, we consider them matched. Then all SAM proposals matched to the same

CRF-refined mask are combined into one single mask. Finally, we compute the IoU between the combined mask and the original CRF-refined mask. If the IoU is greater than a threshold ϕiou, we adopt the combined mask to replace the original mask, otherwise, we keep the CRF-refined mask. The detailed post-processing steps are explained in the supplementary material.

### 4. Experiments

#### 4.1. Zero-shot Semantic Segmentation

Datasets. Since our method does not require training, below we introduce only the datasets utilized for evaluation purposes. We conduct assessments for semantic segmentation using the validation (val) splits of Pascal VOC, Pascal Context, and COCO Object. Specifically, Pascal VOC [18] encompasses 21 categories: 20 object classes (VOC-20) alongside one background class (VOC-21). For Pascal Context [45], our evaluation employs the prevalent version comprising 59 classes including both “things” and “stuff” categories (PC-59), and one background (“other”) class for the concepts outside of the 59 classes (PC-60). Following [68], we construct the COCO Object dataset as a derivative of COCO Stuff [5]. We kindly emphasize that the COCO Object dataset is not COCO Stuff since it merges all “stuff” classes into one background class and thus has 81 classes (80 “things” + 1 background) in total. We use the standard mean Intersection-over-Union (mIoU) metric to evaluate our method’s segmentation performance. Besides, we report the performance of CaR on ADE-150 (A150), ADE-847 (A-847) and Pascal Context 459 (PC-459) that consist of 150, 847 and 459 classes respectively.

Implementation details. Our proposed method CaR utilizes the foundational pre-trained CLIP models as the backbone. More precisely, we harness the CLIP model with ViT-B/16 to serve as the underlying framework for the mask proposal generator f(·,·). Concurrently, for the mask classifier g(·,·), we adopt a larger ViT-L/14 version for higher precision based on our ablation study. Unless otherwise specified, the reported quantitative results are postprocessed solely with a denseCRF, with no SAM masks involved. When setting the threshold hyper-parameters, we assign η = 0.4, θ = 0.6, and λ = 0.4 for Pascal VOC, and η = 0.5, θ = 0.3, λ = 0.5 for COCO and η = 0.6, θ = 0.2, λ = 0.4 for Pascal context. The specific background queries used for the mask generator f(·,·) are ablated in Section 4.2 and detailed in the supplementary material. For Pascal Context, we use separate groups of background queries for “thing” and “stuff”. For “thing” categories, all “stuff” categories are added as background queries and vice versa for “stuff” categories. As an optional strategy, we utilize a matching algorithm and perform an ensemble with SAM masks. We set both thresholds, ϕiom and ϕiou, to 0.7 for all three datasets. We enable half-precision floating point for CLIP. Since CaR is just a framework de-

|Models<br><br>Is VLM pre-trained?<br><br>w/ aux trainable module?<br><br>Additional Training Data|#Images<br><br>Additional Supervision|VOC20‡<br><br>VOC21<br><br>COCO Object<br><br>PC59‡<br><br>PC60<br><br>A150<br><br>A847<br><br>PC459<br><br>|
|---|---|---|
|zero-shot methods fine-tuned with additional data ViL-Seg [39] ✓ ✓ CC12M GroupViT [68] × ✓ CC12M+YFCC GroupViT [68] × ✓ CC12M+RedCaps SegCLIP [42] × ✓ CC3M+COCO SegCLIP [42] ✓ ✓ CC3M+COCO ZeroSeg [11] ✓ ✓ IN-1K ViewCo [51] ✓ ✓ CC12M+YFCC MixReorg [6] ✓ ✓ CC12M CLIPpy [50] ✓ × HQITP-134M OVSegmenter [69] ✓ ✓ CC4M TCL [10] ✓ ✓ CC15M TCL+PAMR [10] ✓ ✓ CC15M|12M text+self<br><br>26M text 24M text<br><br>3.4M text+self 3.4M text+self 1.3M self<br><br>26M text+self<br><br>12M text<br><br>134M text+self<br><br>4M text<br><br>15M text+self 15M text+self<br><br>|- 34.4 16.4 - 16.3 - - 74.1 52.3 24.3 23.4 22.4 10.6 4.3 4.9 79.7 50.8 27.5 - 23.7 - - -<br><br>- 33.3 15.2 - 19.1 - - -<br>- 52.6 26.5 - 24.7 8.7 - -<br>- 40.8 20.2 - 20.4 - - -<br>- 52.4 23.5 - 23.0 - - -<br>- 47.9 - - 23.9 - - -<br>- 52.2 32.0 - - 13.5 - -<br><br>- 53.8 25.1 - 20.4 - - -<br><br><br>77.5 51.2 30.4 24.3 30.3 14.9 - 83.2 55.0 31.6 33.9 30.4 17.1 - -<br><br>|
|zero-shot methods with SAM SAMCLIP [63] (w/ [30]) ✓ ✓ CC15M+YFCC+IN21k CaR+SAM (Ours, w/ [28]) ✓ - -|41M text+self<br><br>- -|- 60.6 - - 29.2 17.1 - -<br><br>- 70.2 37.6 - 31.1 - - -<br><br><br>|
|zero-shot methods without fine-tuning on CLIP ReCo† [55] ✓ × MaskCLIP† [84] ✓ × CaR (Ours, w/o SAM) ✓ × ∆ w/ the state-of-the-art w/o additional data ∆ w/ the state-of-the-art w/ additional data<br><br>|- -<br>- -<br><br>- -<br><br><br>|57.7 25.1 15.7 22.3 19.9 11.2 - 74.9 38.8 20.6 26.4 23.6 9.8 - 91.4 67.6 36.6 39.5 30.5 17.7 8.1 13.9<br><br>+16.5 +28.8 +16.0 +13.1 +6.9 +6.5 - -<br><br>+8.2 +12.6 +4.6 +5.6 +0.1 +0.6 +3.8 +9.0<br><br>|

- Table 1. Comparison to state-of-the-art zero-shot semantic segmentation approaches. Results annotated with a † are reported by Cha et al. [10]. A ✓ is placed if either the visual or text encoder of the VLM is pre-trained. The table shows that our method outperforms not only counterparts without fine-tuning by a large margin, but also those fine-tuned on millions of data samples. For fair comparison, we compare with methods using CLIP [49] as the backbone. ‡VOC-20 and PC-59 represent Pascal VOC and Pascal Context without background. VOC-21 and PC-60 represent Pascal VOC and Pascal Context with an additional background class. A-150, A-847 and PC459 represent ADE-150, ADE-847, and Pascal Context 459 datasets, respectively.

|Dataset<br><br>|w/ recurrence? CAM|mIoU|
|---|---|---|
|Pascal VOC<br><br>|CLIP-ES [37] ✓ CLIP-ES [37] ✓ gradCAM [54]<br><br>|15.2 67.6 41.1|

- Table 2. Effect of applying our recurrent architecure and different CAM methods. The recurrence plays a vital role in improving the performance.

signed for inference, all experiments in this paper are conducted on just one NVIDIA V100 GPU.

Efficiency. CaR without SAM operates at 950 ms with CPU-based CRF (200ms) for 500×500 images. GPU CRF is ∼ 5× faster. CaR is memory-efficient, consuming only

- 3.6GB GPU memory on Pascal VOC. CaR significantly outperforms methods without additional training. We also compare CaR with training-free methods like MaskCLIP [84] and ReCo [55]. Across the benchmarks, our model consistently demonstrates an impressive performance uplift. Under a similar setting with no additional training data, CaR surpasses previous state-ofthe-art method by 28.8, 16.0 and 6.9 mIoU on Pascal VOC, COCO Object and Pascal Context, respectively. Training-free CaR even outperforms several methods with additional fine-tuning. As shown in Table 1, we compare our method with previous state-of-the-art meth-

ods including ViL-Seg [39], GroupViT [68], SegCLIP [42], ZeroSeg [11], ViewCo [51], CLIPpy [50], and TCL [10], which are augmented with additional data. The prior best results of different datasets are achieved by different methods. Specifically, TCL [10], employing a fully pretrained CLIP model and fine-tuned on 15M additional data, achieves the highest mIoU (55.0 and 30.4) on Pascal VOC and Pascal Context. CLIPpy [50] sets the previous highest record on COCO Object but also requires extensive data for fine-tuning. Concretely, it first utilizes a ViT-based image encoder pre-trained with DINO [9] and a pre-trained T5 text encoder [48], and then fine-tunes both encoders with 134M additional data. Our method, incurring no cost for fine-tuning, still outperforms these methods by 12.6, 4.5, and 0.1 mIoU on the Pascal VOC, COCO Object, and Pascal Context datasets, respectively. Since “stuff” appears less frequently in the pre-training image-text data for CLIP, CaR also exhibits less sensitivity to “stuff” on Pascal Context.

CaR+SAM further boosts the performance. When integrated with SAM [28, 30], we compare CaR with a concurrent method SAMCLIP [63] and outperform it by 9.6 and 1.9 on Pascal VOC and Pascal Context. We use the recent variant HQ-SAM [28] with no prompt given (automask mode), and then match the generated masks with metrics designed in Section 3.4. In other words, SAM is only used as a post-processor to refine the boundary of results from

|Mask Proposal Generator f(·, ·)<br><br>Mask Classifier g(·, ·)<br><br>|Pascal VOC<br><br>COCO Object|
|---|---|
|ViT-B/16<br><br>ViT-B/16 ViT-L/14<br><br>|54.1 15.9 67.6 36.6|
|ViT-L/14<br><br>ViT-B/16 ViT-L/14<br><br>|50.6 14.1 57.6 32.5|

- Table 3. Effect of CLIP backbones. We compare various CLIP backbones on Pascal VOC and COCO Object. Results show that we can improve the performance by scaling up the mask classifier.

|Dataset<br><br>|Visual Prompts circle contour blur gray mask<br><br>|mIoU|
|---|---|---|
|Pascal VOC|✓<br><br>✓<br><br>✓<br><br>✓<br><br>✓<br><br>✓ ✓<br><br>✓ ✓<br><br>✓ ✓<br><br>✓ ✓ ✓ ✓ ✓<br><br>|66.9 66.0 66.4 66.1 61.8 67.6 67.1 66.5 66.3 66.8|

- Table 4. Effect of different visual prompts. When multiple visual prompts are checked, we will apply all checked visual prompts simultaneously on one image. The experiments are conducted on Pascal VOC. Results for COCO and Pascal Context are shown in supplementary materials.

CaR. By applying SAM into our framework, our results can be further boosted by 2.6, 1.1 and 0.6 mIoU on Pascal VOC, COCO Object and Pascal Context, respectively.

#### 4.2. Ablation Studies.

Effect of Recurrence. As illustrated in Table 2, the incorporation of the recurrent architecture is crucial to our method. Without recurrence (a.k.a T = 1), our method functions similarly to CLIP-ES [37] with an additional CLIP classifier, and achieves only 15.2% in mIoU. The recurrent framework can lead to a 52.4% improvement, reaching an mIoU of 67.6%. The significant improvement validates the effectiveness of the recurrent design of our framework. For VOC and COCO, most images require two steps, and a small portion of images goes beyond two.

Effect of different CAM methods. Table 2 exhibits that our framework is compatible with different CAM methods and could be potentially integrated with other CAM-related designs. When integrated with CLIP-ES [37], our method is 26.5 mIoU higher than that with gradCAM [54]. We kindly note that we do not carefully search the hyper-parameters on gradCAM so the performance could be further improved.

Effect of different CLIP Backbones. We experiment with different settings of CLIP backbones used in the mask proposal generator f and mask classifier g, on Pascal VOC and

|Pascal VOC η θ λ mIoU|COCO Object η θ λ mIoU<br><br>|
|---|---|
|0.3 0.6 0.4 67.0<br>0.4 0.6 0.4 67.6<br>0.5 0.6 0.4 67.0<br><br><br>0.4 0.5 0.4 67.4 0.4 0.7 0.4 67.5<br>0.4 0.6 0.3 67.3 0.4 0.6 0.5 67.0<br>|0.5 0.3 0.6 35.4<br><br>0.5 0.3 0.4 36.1<br><br>0.4 0.3 0.5 35.8<br><br>0.5 0.3 0.5 36.6<br><br>0.6 0.3 0.5 35.9<br><br><br>0.5 0.4 0.5 36.3<br><br>0.5 0.5 0.5 36.0<br>|

Table 5. Effect of different hyper-parameters: the threshold to binarize mask proposals (η), the threshold to remove text queries (θ), and parameter of CLIP-ES’s[37] (λ). Experiments are conducted on Pascal VOC and COCO Object.

COCO Object datasets. Results are displayed in Table 3. As for the mask proposal generator, ViT-B/16 outperforms the ViT-L/14 by over 10 mIoU on both Pascal VOC and COCO Object. There is significant mIoU gains when employing the larger ViT-L/14 for the mask classifier over ViTB/16. Similar observations have been found by Shtedritski et al. [56] that a larger backbone can better understand the visual prompts, which indicates that the performance of our method can be potentially improved by employing large backbones for the mask classifier.

Effect of different visual prompts. There are various forms of visual prompts, including circle, contour, background blur (blur), background gray (gray), and background mask (mask), etc. We study the effects of different visual prompts on the Pascal VOC dataset and Table 4 summarizes the results when applying one or a combination of two of the aforementioned visual prompt types. The highest mIoU score is achieved with the combination of circle and blur, yielding a mIoU of 67.6. Notably, using mask alone results in the lowest mIoU of 61.8, which is a common practice for most previous open-vocabulary segmentation approaches, e.g., [35, 76]. We also evaluate the effect of different visual prompts on COCO Object and Pascal Context, and show the results in the supplementary material.

Effect of hyper-parameters. We perform an ablation study on the performance impact of various hyper-parameter configurations on Pascal VOC, and present the results in Table 5. Hyper-parameters include the mask binarization threshold, η, defined in Section 3.3, the threshold θ employed in the thresholding function defined in Eq. (6), and the parameter λ defined in CLIP-ES [37]. The peak performance is recorded at an mIoU of 67.6 for η = 0.4, θ = 0.6,

- and λ = 0.4 on Pascal VOC and 36.6 for η = 0.5, θ = 0.3,
- and λ = 0.5 on COCO Object. Different parameter combinations result in mIoU scores that range from 67.0 to 67.6 on Pascal VOC and from 35.4 to 36.6 on COCO Object. Effect of background queries. In Table 6, we explore how different background queries (classes not existing in the input queries) can affect CaR’s performance. We find that the segmentation quality improves as we include more diverse

|Dataset<br><br>|Background queries Terrestrial<br><br>Aquatic Atmospheric<br><br>Man-Made|mIoU|
|---|---|---|
|Pascal VOC|× × × ✓ × × × ✓ × × × ✓ ✓ ✓ × × ✓ ✓ ✓ × ✓ ✓ ✓ ✓<br><br>|64.3 65.6 64.9 66.4 65.8 66.4 65.8 67.6<br><br>|

- Table 6. Effect of background queries on Pascal VOC. We divide background queries into: Terrestrial, Aquatic, Atmospheric, and Man-Made. We use “None” as the background query for the result in the first row. Specific background queries of each category are shown in the supplementary material.

|Models RefCOCO val testA testB<br><br>|RefCOCO+ val testA testB<br><br>|RefCOCOg val test(U) val(G)|GRES|
|---|---|---|---|
|weakly-supervised TSEG [58] 25.95 - -<br><br>|22.62 - -<br><br>|23.41 - -|-<br><br>|
|zero-shot<br><br>GL CLIP [77] 26.20 24.94 26.56 CaR(Ours) 33.57 35.36 30.51<br><br>|27.80 25.64 27.84 34.22 36.03 31.02<br><br>|33.52 33.67 33.61 36.67 36.57 36.63|16.8<br><br>|

- Table 7. Comparison to state-of-the-art methods on referring image segmentation in mIoU. CaR is better than all comparison methods in all benchmarks.

background queries: The combination of all three types of background queries delivers the highest mIoU of 67.6. For more details about the background queries of each class, please refer to the supplementary material.

#### 4.3. Referring Segmentation

We evaluate CaR on the referring segmentation task for both images and videos. Again, our method is an inference-only pipeline built upon pre-trained CLIP models, and does not need training/fine-tuning on any types of annotations. For referring segmentation we only use denseCRF [31] for postprocessing, and SAM is not involved for all experiments in this section for fair comparison. Please refer to the supplementary material for the implementation details.

Datasets. Following [72, 77], we evaluate on RefCOCO [74], RefCOCO+ [74], RefCOCOg [43, 47] and GRES [38] for the referring image segmentation task. Images used in all three datasets are sourced from the MS COCO [36] dataset and the masks are paired with descriptive language annotations. In RefCOCO+, descriptions about location are prohibited, making the task more challenging. There are two separate splits of the RefCOCOg dataset, one by UMD (U) [47] and another by Google (G) [65]. Following previous work, we use the standard mIoU metric. Apart from referring image segmentation, we also set up a new baseline for zero-shot referring video

segmentation on Ref-DAVIS 2017 [29]. Following [29], we adopt region similarity J , contour accuracy F, and the averaged score J &F as the metrics for evaluation.

Experimental results. Table 7 compares the performance of CaR with other methods on the referring image segmentation tasks across RefCOCO, RefCOCO+, and RefCOCOg. Comparing with other zero-shot methods, our method CaR outperforms Global-Local CLIP (GL CLIP) on all splits of these benchmarks. The performance gap is most pronounced on RefCOCO’s testA split, where CaR outperforms 10.42 mIoU, and similarly on RefCOCO+’s testA split, with a lead of 10.72 mIoU. We also note that GL CLIP [77] uses a pre-trained segmenter Free-SOLO [64] for mask extraction, while CaR is built without any pre-trained segmenter. As the first zero-shot method on GRES [38], CaR achieves 16.8 mIoU with no training effort incurred.

J &F J F 30.34 28.15 32.53

Table 8. Results on Ref-DAVIS 2017.

For referring video segmentation, we demonstrate in Table 8 that our method achieves 30.34, 28.15 and 32.53 for J &F, J and F on Ref-DAVIS 2017 [29]. Considering that our method CaR requires neither fine-tuning nor annotations and operates in a zero-shot manner, this performance establishes a strong baseline.

### 5. Conclusion

We introduce CLIP as RNN (CaR), which preserves the entire large vocabulary space of pre-trained VLMs, by eliminating the fine-tuning process. By constructing a recurrent pipeline with a shared segmenter in the loop, CaR can perform zero-shot semantic and referring segmentation without any additional training efforts. Experiments show that CaR outperforms previous state-of-the-art counterparts by a large margin on eight different benchmarks, i.e. Pascal VOC, COCO Object, Pascal Context, ADE-150, ADE-847 and Pascal Context 459 on zero-shot semantic segmentation. We also demonstrate that CaR can handle referring expressions and segment fine-grained concepts like anime characters and landmarks, and also achieves state-of-the-art performance on RefCOCO, RefCOCO+, RefCOCOg and GRES for zero-shot referring segmentation. We hope our work sheds light on future research in open-vocabulary segmentation aiming to further expand the vocabulary space.

Acknowledgement. The majority of this work was done during Shuyang’s internship at Google Research. We would like to thank Jindong Gu and Jianhao Yuan at Oxford University, Anurag Arnab, Xingyi Zhou, Huizhong Chen and Neil Alldrin at Google Research for their insightful discussion, Zhongli Ding for image donation. Shuyang Sun and Philip Torr are supported by UKRI grants: Turing AI Fellowship EP/W002981/1. We would also like to thank the Royal Academy of Engineering and FiveAI.

### References

- [1] Nikita Araslanov and Stefan Roth. Single-stage semantic segmentation from image labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4253–4262, 2020. 3
- [2] Donghyeon Baek, Youngmin Oh, and Bumsub Ham. Exploiting a joint embedding space for generalized zero-shot semantic segmentation. In ICCV, 2021. 2
- [3] Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, and Patrick P´erez. Zero-shot semantic segmentation. Advances in Neural Information Processing Systems, 32, 2019. 2
- [4] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2018. 2
- [5] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1209–1218, 2018. 5, 15
- [6] Kaixin Cai, Pengzhen Ren, Yi Zhu, Hang Xu, Jianzhuang Liu, Changlin Li, Guangrun Wang, and Xiaodan Liang. Mixreorg: Cross-modal mixed patch reorganization is a good mask learner for open-world semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1196–1205, 2023. 2, 6
- [7] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In CVPR, 2018. 3
- [8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 3
- [9] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 6
- [10] Junbum Cha, Jonghwan Mun, and Byungseok Roh. Learning to generate text-grounded mask for open-world semantic segmentation from only image-text pairs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11165–11174, 2023. 1, 2, 6, 13
- [11] Jun Chen, Deyao Zhu, Guocheng Qian, Bernard Ghanem, Zhicheng Yan, Chenchen Zhu, Fanyi Xiao, Mohamed Elhoseiny, and Sean Chang Culatana. Exploring open-vocabulary semantic segmentation without human labels. arXiv preprint arXiv:2306.00450, 2023. 2, 6
- [12] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs. In ICLR, 2015. 3
- [13] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV, 2018. 3
- [14] Peijie Chen, Qi Li, Saad Biaz, Trung Bui, and Anh Nguyen. gscorecam: What objects is clip looking at? In Proceedings of the Asian Conference on Computer Vision, pages 1959– 1975, 2022. 2

- [15] Bowen Cheng, Alexander G Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In NeurIPS, 2021. 3
- [16] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. CVPR, 2022. 3
- [17] Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. Decoupling zero-shot semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11583–11592, 2022. 2
- [18] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. IJCV, 88:303–338, 2010. 5
- [19] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88:303–338, 2010. 2, 13, 15
- [20] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In European Conference on Computer Vision, pages 540–557. Springer, 2022. 2
- [21] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 580–587, 2014. 3
- [22] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921,

2021. 2

- [23] Wenbin He, Suphanut Jamonnak, Liang Gou, and Liu Ren. Clip-s4: Language-guided self-supervised semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11207–11216,

2023. 2

- [24] Ping Hu, Stan Sclaroff, and Kate Saenko. Uncertainty-aware learning for zero-shot semantic segmentation. Advances in Neural Information Processing Systems, 33:21713–21724,

- 2020. 2

[25] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,

- 2021. 2

- [26] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetrmodulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1780–1790, 2021. 2
- [27] Laurynas Karazija, Iro Laina, Andrea Vedaldi, and Christian Rupprecht. Diffusion models for zero-shot open-vocabulary segmentation. arXiv preprint arXiv:2306.09316, 2023. 2
- [28] Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, and Fisher Yu. Segment anything in high quality. arXiv preprint arXiv:2306.01567, 2023. 6, 14
- [29] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In

- Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part IV 14, pages 123–141. Springer, 2019. 2, 8
- [30] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 5, 6
- [31] Philipp Kr¨ahenb¨uhl and Vladlen Koltun. Efficient inference in fully connected crfs with gaussian edge potentials. Advances in neural information processing systems, 24, 2011. 3, 5, 8
- [32] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In International Conference on Learning Representations, 2022. 2
- [33] Peike Li, Yunchao Wei, and Yi Yang. Consistent structural relation learning for zero-shot segmentation. NeurIPS, 2020. 2
- [34] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23390– 23400, 2023. 2
- [35] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7061–7070, 2023. 1, 2, 7, 13, 15, 18
- [36] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 2, 8, 13
- [37] Yuqi Lin, Minghao Chen, Wenxiao Wang, Boxi Wu, Ke Li, Binbin Lin, Haifeng Liu, and Xiaofei He. Clip is also an efficient segmenter: A text-driven approach for weakly supervised semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15305–15314, 2023. 2, 3, 4, 6, 7, 13, 14
- [38] Chang Liu, Henghui Ding, and Xudong Jiang. GRES: Generalized referring expression segmentation. In CVPR, 2023. 8
- [39] Quande Liu, Youpeng Wen, Jianhua Han, Chunjing Xu, Hang Xu, and Xiaodan Liang. Open-world semantic segmentation via contrasting and clustering vision-language embedding. In European Conference on Computer Vision, pages 275–292. Springer, 2022. 1, 6
- [40] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 1, 2, 15, 18
- [41] Timo L¨uddecke and Alexander Ecker. Image segmentation using text and image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7086–7096, 2022. 3, 5

- [42] Huaishao Luo, Junwei Bao, Youzheng Wu, Xiaodong He, and Tianrui Li. Segclip: Patch aggregation with learnable centers for open-vocabulary semantic segmentation. In International Conference on Machine Learning, pages 23033–

23044. PMLR, 2023. 1, 2, 6

- [43] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016. 8, 13
- [44] M Minderer, A Gritsenko, A Stone, M Neumann, D Weissenborn, A Dosovitskiy, A Mahendran, A Arnab, M Dehghani, Z Shen, et al. Simple open-vocabulary object detection with vision transformers. arxiv 2022. arXiv preprint arXiv:2205.06230, 2022. 2
- [45] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014. 2, 5, 13, 14
- [46] Jishnu Mukhoti, Tsung-Yu Lin, Omid Poursaeed, Rui Wang, Ashish Shah, Philip HS Torr, and Ser-Nam Lim. Open vocabulary semantic segmentation with patch aligned contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19413–19423, 2023. 2, 13
- [47] Varun K Nagaraja, Vlad I Morariu, and Larry S Davis. Modeling context between objects for referring expression understanding. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 792–807. Springer,

2016. 8, 13

- [48] Jianmo Ni, Gustavo Hern´andez Abrego,´ Noah Constant, Ji Ma, Keith B Hall, Daniel Cer, and Yinfei Yang. Sentencet5: Scalable sentence encoders from pre-trained text-to-text models. arXiv preprint arXiv:2108.08877, 2021. 6
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2, 6, 13
- [50] Kanchana Ranasinghe, Brandon McKinzie, Sachin Ravi, Yinfei Yang, Alexander Toshev, and Jonathon Shlens. Perceptual grouping in vision-language models. arXiv preprint arXiv:2210.09996, 2022. 1, 2, 6
- [51] Pengzhen Ren, Changlin Li, Hang Xu, Yi Zhu, Guangrun Wang, Jianzhuang Liu, Xiaojun Chang, and Xiaodan Liang. Viewco: Discovering text-supervised segmentation masks via multi-view semantic consistency. arXiv preprint arXiv:2302.10307, 2023. 1, 2, 6
- [52] Lixiang Ru, Yibing Zhan, Baosheng Yu, and Bo Du. Learning affinity from attention: End-to-end weakly-supervised semantic segmentation with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16846–16855, 2022. 3
- [53] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,

- Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. IJCV, 115:211–252, 2015. 13
- [54] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pages 618–626,

2017. 4, 6, 7, 13

- [55] Gyungin Shin, Weidi Xie, and Samuel Albanie. Reco: Retrieve and co-segment for zero-shot transfer. Advances in Neural Information Processing Systems, 35:33754–33767,

2022. 1, 2, 3, 6, 13

- [56] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does clip know about a red circle? visual prompt engineering for vlms. arXiv preprint arXiv:2304.06712, 2023. 5, 7
- [57] Richard Sinkhorn. A relationship between arbitrary positive matrices and doubly stochastic matrices. The annals of mathematical statistics, 35(2):876–879, 1964. 14
- [58] Robin Strudel, Ivan Laptev, and Cordelia Schmid. Weaklysupervised segmentation of referring expressions. arXiv preprint arXiv:2205.04725, 2022. 8
- [59] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 2
- [60] Shuyang Sun, Weijun Wang, Qihang Yu, Andrew Howard, Philip Torr, and Liang-Chieh Chen. Remax: Relaxing for better training on efficient panoptic segmentation. arXiv preprint arXiv:2306.17319, 2023. 3
- [61] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [62] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Max-deeplab: End-to-end panoptic segmentation with mask transformers. In CVPR, 2021. 3
- [63] Haoxiang Wang, Pavan Kumar Anasosalu Vasu, Fartash Faghri, Raviteja Vemulapalli, Mehrdad Farajtabar, Sachin Mehta, Mohammad Rastegari, Oncel Tuzel, and Hadi Pouransari. Sam-clip: Merging vision foundation models towards semantic and spatial understanding. arXiv preprint arXiv:2310.15308, 2023. 6
- [64] Xinlong Wang, Zhiding Yu, Shalini De Mello, Jan Kautz, Anima Anandkumar, Chunhua Shen, and Jose M Alvarez. Freesolo: Learning to segment objects without annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14176–14186, 2022. 8
- [65] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv:1609.08144, 2016. 8
- [66] Yongqin Xian, Subhabrata Choudhury, Yang He, Bernt Schiele, and Zeynep Akata. Semantic projection network for zero-and few-label semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8256–8265, 2019. 2

- [67] Jinheng Xie, Xianxu Hou, Kai Ye, and Linlin Shen. Clims: Cross language image matching for weakly supervised semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4483–4492, 2022. 3
- [68] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In CVPR, 2022. 1, 2, 5, 6, 13
- [69] Jilan Xu, Junlin Hou, Yuejie Zhang, Rui Feng, Yi Wang, Yu Qiao, and Weidi Xie. Learning open-vocabulary semantic segmentation models from natural language supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2935–2944, 2023. 2, 6
- [70] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2955–2966, 2023. 2
- [71] Lian Xu, Wanli Ouyang, Mohammed Bennamoun, Farid Boussaid, and Dan Xu. Multi-class token transformer for weakly supervised semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4310–4319, 2022. 3
- [72] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18155–18165, 2022. 8
- [73] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 2
- [74] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 8, 13
- [75] Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. k-means Mask Transformer. In ECCV, 2022. 3
- [76] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional clip. arXiv preprint arXiv:2308.02487, 2023. 1, 2, 7
- [77] Seonghoon Yu, Paul Hongsuck Seo, and Jeany Son. Zeroshot referring image segmentation with global-local context features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19456– 19465, 2023. 8
- [78] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18123–18133, 2022. 2
- [79] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. arXiv preprint arXiv:2303.15343, 2023. 2

- [80] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Li, Xiyang Dai, Lijuan Wang, Lu Yuan, JenqNeng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. Advances in Neural Information Processing Systems, 35:36067–36080, 2022. 2
- [81] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianwei Yang, and Lei Zhang. A simple framework for open-vocabulary segmentation and detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1020–1031, 2023. 2
- [82] Shuai Zheng, Sadeep Jayasumana, Bernardino RomeraParedes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, and Philip HS Torr. Conditional random fields as recurrent neural networks. In ICCV, 2015. 3
- [83] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 2019. 2
- [84] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In European Conference on Computer Vision, pages 696–712. Springer, 2022. 1, 2, 3, 6, 13
- [85] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Kr¨ahenb¨uhl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In European Conference on Computer Vision, pages 350–368. Springer, 2022. 2

Appendix

- A. More Experimental Results

- A.1. Quantitative Analysis on Vocabulary Space.

We demonstrate that our method CaR has a larger vocabulary space compared to the methods fine-tuned with mask annotations. Here we compare our method with OVSeg [35], which is fine-tuned on ImageNet [53] and COCO [36] with a pre-trained CLIP backbone for the task of referring image segmentation. We believe that referring expressions (e.g., “the person in the red shirt” or “the cat in the mirror”) refers to a specific segment using a broad vocabulary. We conduct a comparative analysis between a robust open-vocabulary segmentation benchmark, OVSeg [35], and CaR, utilizing standard referring image segmentation benchmarks [43, 47, 74]. We note that RefCOCO and COCO share the same set of images so OVSeg fine-tuning on COCO may not be counted as zero-shot on RefCOCO. The results, as detailed in Table I, demonstrate that CaR significantly surpasses OVSeg in performance. This disparity in performance suggests that CaR encompasses a more expansive vocabulary space than OVSeg.

- A.2. Evaluation without Background

Following [37], our methodology benefits from using background queries in CLIP [49] classification to suppress false positives (predictions not belonging to the input text queries), enhancing segmentation results. Nevertheless, for more comprehensive comparison, we also assess our approach using an alternate evaluation setting, previously established, which omits the background class. Consequently, less emphasis is placed on object boundaries in this setting. We test our method on two datasets: Pascal VOC [19] without background (termed VOC-20) and Pascal Context [45] without background (termed Context-59). This setting tests the ability of various methods to discriminate between different classes. Our method CaR significantly outperforms previous methods on VOC-20 and Context-59, where all methods use the same setting that ignores the background class. We reached out to the PACL authors to confirm that they did not evaluate background.

- B. Implementation Details of CAM

In this paper, we integrate two kinds of gradient-based CAM, i.e., Grad-CAM [54] and CLIP-ES [37], respectively.

Integration with Grad-CAM. When integrating GradCAM [54] into our framework, we first extract the image and text feature vectors vx = fI(x),vh = fT(h) from the image and text encoder fI(·),fT(·) given an image x and text queries h. We compute a similarity score between the image and text features using the dot product s = softmax(vx·vh⊺), where softmax is applied along the di-

|Models|RefCOCO<br><br>val testA testB<br><br>|RefCOCO+ val testA testB|RefCOCOg<br><br>val test(U) val(G)|
|---|---|---|---|
|OVSeg [35] CaR(Ours)|22.58 19.38 25.63 33.57 35.36 30.51<br><br>|19.13 15.74 25.30 34.22 36.03 31.02|27.87 29.09 28.31 36.67 36.57 36.63<br><br>|

- Table I. Comparison to mask-supervised open-vocabulary methods on referring image segmentation in mIoU. CaR is better than the comparison method, OVSeg, in all splits of the three benchmarks.

|Model|Is VLM pre-trained?<br><br>#Additional Images<br><br>|w/o Background VOC-20 Context-59|
|---|---|---|
|GroupViT† [68] PACL [46] TCL [10]<br><br>|× 26M ✓ 40M ✓ 15M<br><br>|79.7 23.4 72.3 50.1 77.5 30.3|
|MaskCLIP† [84]<br><br>ReCo† [55] CaR (Ours)|✓ ✓ ✓ -<br><br>|74.9 26.4 57.5 22.3 91.4 39.5<br><br>|

- Table J. Comparison with methods under the setting where background is ignored. We compare CaR with prior work on VOC-20, Context-59 in a setting that considers only the foreground pixels (decided by ground truth). Our method shows comparable performance to prior works despite only relying on pretrained feature extractors. †: numbers are from [10].

mension of vh. This score s quantifies the alignment (a.k.a similarity) between the image x and the text h as perceived by the CLIP model. Here h contains multiple queries. To integrate Grad-CAM into our framework, we first compute the gradients of the similarity score with respect to the feature maps of the image encoder by:

∂s ∂Ak

g =

,

where Ak represents the feature maps and g denotes the gradients. Then we compute the neuron importance weights by average-pooling the gradients:

1 Z i j

αk =

gijk .

Here, αk is the neuron importance weights and Z is the number of pixels in each feature map. We then calculate a weighted combination of the feature maps Ak and the neuron importance αk:

L = ReLU

k

αkAk ,

an activation function ReLU is applied to filter out all negative activations. Specifically, we use the feature map after the first normalization layer of the last residual block to compute the gradients for CAM.

Integration with CLIP-ES. In summary, the CLIPES [37] we adopted is composed of a Grad-CAM and a class-aware attention-based affinity (CAA) module. The CAA module is introduced to enhance the vanilla multihead self-attention (MHSA) for the Vision Transformer in CLIP. Given an image, class-wise CAM maps Mc ∈ Rh×w for each target class c and the attention weight Wattn ∈ Rhw×hw are obtained from MHSA. For the attention weight, which is made asymmetric due to the use of different projection layers by the query and key, Sinkhorn normalization [57] is applied (alternately applying rownormalization and column-normalization) to convert it into a doubly stochastic matrix D, and the symmetric affinity matrix A can be derived as follows:

D + DT 2

, where D = Sinkhorn(Wattn). (7)

A =

For the CAM map Mc ∈ Rh×w, a mask map for each target class c can be obtained by thresholding the CAM with λ. Then a set of bounding boxes can be generated based on the thresholded masks. These boxes are used to mask the affinity weight matrix A, and then each pixel can be refined based on the masked affinity weight and its semantically similar pixels. This refinement process can be formalized as follows:

Mcaff = Bc ⊙ At · vec(Mc), (8)

where Bc ∈ R1×hw represents the box mask obtained from the CAM of class c, ⊙ denotes the Hadamard product, t indicates the number of refining iterations, and vec(·) denotes the vectorization of a matrix. It should be noted that the attention map and CAM are extracted in the same forward pass. Therefore, CAA refinement is performed in real time and does not need an additional stage. Our implementation uses the attention maps from the last 8 layers of Vision Transformer for CAA.

### C. Implementation Details of Visual Prompts

The Python code of visual prompts is shown in Algorithm B, which is at the end of the supplementary material.

### D. Breakdown of Background Tokens

We break down the background tokens into 3 sub-categories for ablation study (experiment results are shown in the main manuscript in Table 6):

• Terrestrial: [‘ground’, ‘land’, ‘grass’, ‘tree’, ‘mountain’, ‘rock’, ‘valley’, ‘earth’,‘terrain’, ‘forest’, ‘bush’, ‘hill’, ‘field’, ‘pasture’, ‘meadow’, ‘plateau’, ‘cliff’, ‘canyon’, ‘ridge’, ‘peak’, ‘plain’, ‘prairie’, ‘tundra’, ‘savanna’, ‘steppe’, ‘crag’, ‘knoll’, ‘dune’, ‘glen’, ‘dale’, ‘copse’, ‘thicket’]

- • Aquatic-Atmospheric: [‘sea’, ‘ocean’, ‘lake’, ‘water’, ‘river’, ‘sky’, ‘cloud’, ‘pond’, ‘stream’, ‘lagoon’, ‘bay’, ‘gulf’, ‘fjord’, ‘estuary’, ‘creek’, ‘brook’, ‘reservoir’, ‘pool’, ‘spring’, ‘marsh’, ‘swamp’, ‘wetland’, ‘glacier’, ‘iceberg’, ‘atmosphere’, ‘stratosphere’, ‘mist’, ‘fog’, ‘rain’, ‘drizzle’, ‘hail’, ‘sleet’, ‘snow’, ‘thunderstorm’, ‘breeze’, ‘wind’, ‘gust’, ‘hurricane’, ‘tornado’, ‘monsoon’, ‘cumulus’, ‘cirrus’, ‘stratus’, ‘nimbus’]
- • Man-Made: [ ‘building’, ‘house’, ‘wall’, ‘road’, ‘street’, ‘railway’, ‘railroad’, ‘bridge’, ‘edifice’, ‘structure’, ‘apartment’, ‘condominium’, ‘skyscraper’, ‘highway’, ‘boulevard’, ‘lane’, ‘alley’, ‘byway’, ‘avenue’, ‘expressway’, ‘freeway’, ‘path’, ‘overpass’, ‘underpass’, ‘viaduct’, ‘tunnel’, ‘footbridge’, ‘crosswalk’, ‘culvert’, ‘dam’, ‘archway’, ‘causeway’, ‘plaza’, ‘square’, ‘station’, ‘terminal’ ]

### E. Implementation Details of Mutual Background for Pascal Context

Our approach involves creating a list of background queries to minimize false positive predictions in mask proposals. However, in the Pascal Context dataset [45], many “stuff” categories (e.g. sky, ground, sea) serve as background queries for “object” categories (e.g. bird, car, boat). Directly removing these ’stuff’ categories from the background query list and generating object and stuff masks using CAM leads to noisy results due to the lack of false positive background suppression. To address this issue, we adopt a mutual background strategy. In this method, object and stuff masks are produced separately, using object categories as the background queries for stuff masks and vice versa. This technique not only maintains the benefit of reducing false positives but also significantly enhances performance in the Pascal Context dataset.

### F. Implementation Details of Referring Image Segmentation.

We use ViT-B/16 as the backbone of the visual encoder for both the mask proposal generator and mask classifier, and use circle and background blur as the visual prompts for the inputs of mask classifier. The η, θ, λ were set to (0.5, 0.3, 0.5), (0.2, 0.1, 0.5), (0.5, 0.1, 0.6) for refCOCO, refCOCO+ and refCOCOg, respectively. All splits of these three datasets share the same set of hyperparameters. We note that we do not apply SAM for referring image segmentation.

### G. More Visualization Results

#### G.1. Visualization results on different postprocessors

Figures E and F present a comparative visualization of the post-processing techniques Conditional Random Field (CRF) and Segment Anything Model (SAM) [28], applied

to randomly chosen samples from the VOC [19] and COCO Object datasets [5]. Initial observations reveal that the application of CRF in CaR facilitates the generation of highquality masks, albeit with notable limitations in delineating boundaries between distinct semantic masks. The integration of SAM enhances the precision of these masks, yielding clearer and more well-defined boundaries. Nevertheless, the implementation of SAM is not without drawbacks; it occasionally leads to false negative predictions, stemming from mismatches between CaR raw masks and SAM candidate masks (the matching algorithm is introduced in the main manuscript), or false positive predictions due to the overly coarse nature of SAM masks. Meanwhile, we find SAM is not very sensitive to stuff classes, so combining SAM on Pascal Context will not lead to much increase in mIoU.

#### G.2. Visualization comparison for different openvocabulary segmentation methods.

Figure G presents a qualitative comparison of openvocabulary segmentation results for a variety of nonstandard subjects, including unique characters, brands, and landmarks. These subjects are notably distinct from common objects. The Grounded SAM [40] method demonstrates proficiency in segmenting prominent objects with precision, yet it often misclassify these segments. The OVSeg [35] approach also generates low-quality segmentation masks and inaccurate class predictions. In contrast, our methodology CaR excels by creating high-quality masks with accurate semantic class predictions, showcasing its superior capability in the realm of open-vocabulary segmentation.

### H. Limitation

The primary limitation of our method is that its performance is bounded by the pre-trained VLM. For example, since the CLIP model utilizes horizontal flipping augmentation during training, it becomes challenging for our model to successfully distinguish between the concepts “left” and “right”. However, we believe that this issue can be easily resolved through adjustments, such as incorporating better data augmentation techniques during the pre-training phase.

### I. Future Potentials and Broader Impact

CaR is simple, straightforward yet highly efficient. To enhance its performance further, we provide two ways to explore. First, incorporating additional trainable modules such as Feature-Pyramid Networks can significantly improve its capability in handling small objects. Second, since our method is fundamentally compatible with various Vision-Language Models (VLMs), it presents an intriguing opportunity to investigate integration with other VLMs. Moreover, CaR can serve the purpose of generating pseudolabels for other open-vocabulary segmenters.

Algorithm B Pseudo-code of CLIP as RNN in PyTorch style.

import cv2 import numpy as np import torch from scipy.ndimage import binary_fill_holes def apply_visual_prompts(

image_array, mask, visual_prompt_type=(’circle’), visualize=False, color=(255, 0, 0), thickness=1, blur_strength=(15, 15)):

prompted_image = image_array inv_mask = (1 - mask)[:, :, None] if ’blur’ in visual_prompt_type:

# blur the part out side the mask # Blur the entire image blurred = cv2.GaussianBlur(prompted_image,

blur_strength, 0) # Get the sharp region using the mask sharp_region = cv2.bitwise_and(

prompted_image, prompted_image, mask=np.clip(mask, 0, 255)))

# Get the blurred region using the inverted mask blurred_region = (blurred * inv_mask) # Combine the sharp and blurred regions prompted_image = cv2.add(sharp_region,

blurred_region) if ’gray’ in visual_prompt_type: gray = cv2.cvtColor(prompted_image, cv2.

COLOR_BGR2GRAY) # make gray part 3 channel gray = np.stack([gray, gray, gray], axis=-1) # Get the sharp region using the mask color_region = cv2.bitwise_and(

prompted_image, prompted_image, mask=np.clip(mask, 0, 255))

# Get the blurred region using the inverted mask inv_mask = 1 - mask gray_region = (gray * inv_mask) # Combine the sharp and blurred regions prompted_image = cv2.add(color_region,

gray_region) if ’black’ in visual_prompt_type: prompted_image = cv2.bitwise_and(

prompted_image, prompted_image, mask=np.clip(mask, 0, 255))

if ’circle’ in visual_prompt_type: mask_center, mask_height, mask_width = mask2chw( mask) center_coordinates = (mask_center[1], mask_center[0]) axes_length = (mask_width // 2, mask_height // 2)

prompted_image = cv2.ellipse(prompted_image, center_coordinates, axes_length, 0, 0, 360,

color, thickness) if ’rectangle’ in visual_prompt_type:

mask_center, mask_height, mask_width = mask2chw( mask) center_coordinates = (mask_center[1], mask_center[0]) start_point = (mask_center[1] - mask_width //

2, mask_center[0] - mask_height // 2) end_point = (mask_center[1] + mask_width //

2, mask_center[0] + mask_height // 2)

prompted_image = cv2.rectangle(prompted_image, start_point, end_point, color, thickness)

if ’contour’ in visual_prompt_type: # Find the contours of the mask # fill holes for the mask mask = binary_fill_holes(mask) contours, hierarchy = cv2.findContours(mask, cv2

.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Draw the contours on the image prompted_image = cv2.drawContours(

prompted_image, contours, -1, color, thickness) return prompted_image

Image CaR CaR+SAM GT

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

- Figure E. Comparison of different post-processors on randomly selected images from PASCAL VOC.

Image CaR CaR+SAM GT

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

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

- Figure F. Comparison of different post-processors on randomly selected images from COCO Object.

Image OVSeg [35] Grounded SAM [40] Ours

[Figure 56]

Figure G. Visualization comparison of different open-vocabulary segmentation methods.

