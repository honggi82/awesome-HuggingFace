# arXiv:2503.19903v2[cs.CV]3Aug2025

[Figure 1]

2025-8-5

## Scaling Vision Pre-Training to 4K Resolution

###### Baifeng Shi1,2* Boyi Li1,2 Han Cai2 Yao Lu2 Sifei Liu2 Marco Pavone2 Jan Kautz2 Song Han2 Trevor Darrell1 Pavlo Molchanov2 Hongxu Yin2

1UC Berkeley 2NVIDIA

### Abstract

High-resolution perception of visual details is crucial for daily tasks. Current vision pre-training, however, is still limited to low resolutions (e.g., 378×378 pixels) due to the quadratic cost of processing larger images. We introduce PS3 that scales CLIP-style vision pre-training to 4K resolution with a near-constant cost. Instead of contrastive learning on global image representation, PS3 is pre-trained by selectively processing local regions and contrasting them with local detailed captions, enabling high-resolution representation learning with greatly reduced computational overhead. The pre-trained PS3 is able to both encode the global image at low resolution and selectively process local high-resolution regions based on their saliency or relevance to a text prompt. When applying PS3 to multi-modal LLMs (MLLMs), the resulting model, named VILA-HD, significantly improves high-resolution visual perception compared to baselines without high-resolution vision pre-training such as AnyRes and S2 while using up to 4.3× fewer tokens. PS3 also unlocks appealing scaling properties of VILA-HD, including scaling up resolution for free and scaling up test-time compute for better performance. Compared to state of the arts, PS3 and VILA-HD outperform previous vision encoders (e.g., SigLIP2 and Perception Encoder) and MLLMs (e.g., NVILA and Qwen2.5-VL) respectively across multiple benchmarks and achieve better efficiency than latest token pruning approaches. Finally, we find current benchmarks do not require 4K-resolution perception, which motivates us to propose 4KPro, a new benchmark of image QA at 4K resolution, on which VILA-HD outperforms all previous MLLMs, including a 16.1% improvement over GPT-4o and a 7.5% improvement and 1.67× speedup over Qwen2.5-VL.

Project page: https://nvlabs.github.io/PS3

### 1. Introduction

Vision models with large-scale pre-training [78, 80, 44, 30] have been the workhorses for both fundamental vision tasks [53, 116] and numerous downstream applications [81, 82, 49]. Notably, CLIPstyle vision pre-training (i.e., vision-language contrastive learning) such as CLIP [80] and SigLIP [122] have driven significant advancements in multi-modal large language models (MLLMs) by providing general-purpose language-aligned visual understanding in real-world tasks [69, 61, 56].

However, modern vision models including CLIP and SigLIP have one defect: they are pre-trained with low resolution only. Visual perception at high resolution (e.g., 4K resolution) is essential in many real-world scenarios such as spotting the stop sign while driving (Figure 1(Left)). On the other hand, SigLIP, for example, is only pre-trained with a maximum resolution of 378×378 [122], making it incapable of perceiving visual details and thus unsuitable for assisting humans in everyday tasks. Existing methods propose to run pre-trained vision models at higher resolution in a training-free manner for downstream tasks [87, 62, 22]. However, this prevents the model from leveraging large-scale pre-training data to learn high-quality high-resolution perception, resulting in suboptimal performance [87].

What blocks the current vision pre-training from scaling to higher resolution? The computational cost.

*Work done during an internship at NVIDIA.

Pre-Training Resolution

###### PS3 Pre-Training

PS3 SigLIP

[Figure 2]

SigLIP

SigLIP Pre-Training

PS3

[Figure 3]

378x378

ViT

ViT

[Figure 4]

Pre-Training Cost (4K Res)

[Figure 5]

A street view in the morning.

The red sign says stop.

[Figure 6]

[Figure 7]

SigLIP

PS3

1134x1134

[Figure 8]

Accuracy on 4KPro

378 x 378

[Figure 9]

Stop.

[Figure 10]

[Figure 11]

3780x3780

Qwen2.5-VL

Patch Selection

VILA-HD

[Figure 12]

VILA-HD

Inference Latency

The red sign says stop. / What does the red sign say?

What does the red sign say?

[Figure 13]

[Figure 14]

select

[Figure 15]

Qwen2.5-VL

Resolution: 3780 x 3780

VILA-HD

- Figure 1: Left: Regular vision models such as SigLIP processes images at a low resolution (e.g., 378 × 378 pixels), which is not enough for many daily tasks such as spotting the stop sign while driving. In contrast, PS3 is able to both encode low-res features and efficiently process high-res information of 4K-resolution images via top-down patch selection, i.e., selectively processing relevant patches based on any text prompt. Top Right: SigLIP is pre-trained by contrasting global vision features and global captions, which is costly for high-resolution images. PS3 is pre-trained with additional contrast between local high-res features with local captions, enabling pre-training at 4K resolution with 79× less cost than SigLIP. Bottom Right: VILA-HD with PS3 as the vision encoder is able to select high-res regions to process based on the user prompt. VILA-HD outperforms state-of-the-art MLLMs such as Qwen2.5-VL [4] by 7.5% on the proposed 4KPro benchmark while achieving 1.67× speedup.

The compute spent by the vision model grows quadratically for CNNs and quartically for ViTs with increasing image resolution, making it even infeasible to pre-train over 1K resolution [78, 122].

In this work, we introduce Pre-training with Scale-Selective Scaling, or PS3, that scales CLIP-style pre-training to 4K resolution with a near-constant cost. The key insight is that, instead of contrasting between global images and captions for the whole high-res image, it suffices to contrast between local regions and local captions to learn detailed feature extraction in high-resolution images. For example, in Figure 1(Left, Top Right), to learn to recognize the text on the stop sign, the model only needs to extract the high-resolution feature around the local region of the text and align it with the detailed description about the region. This is analogous to top-down selection mechanism in human vision [128, 11], i.e., one usually focuses on a small portion of the scene that is relevant to the high-level task (e.g., spotting the stop sign). In this way, the model enjoys greatly reduced computational cost by being scale-selective, i.e., selectively processing a small region at fine-grained scale. By disentangling the region size from the image resolution, we are able to scale PS3 pre-training to 4K resolution with a near-constant cost, reducing the pre-training compute by 79× compared to global contrastive learning of SigLIP (Figure 1(Top Right)).

The success of PS3 pre-training hinges on addressing three challenges: data, model, and algorithm. First, since the low-resolution image-text pairs used for CLIP pre-training is not suitable for PS3 pre-training, we collect 75M images with up to 4K resolution and build an automatic pipeline to curate 282M pairs of detailed captions and bounding boxes of salient local regions in the images. Second, we design a vision model that can not only extract low-resolution global features, but also select local patches based on image saliency or text queries and process high-resolution details of the patches. Third, we design an algorithm that pre-trains high-resolution perception through contrastive loss between local regions and local captions and pre-trains patch selection with supervision from the curated bounding boxes.

We show PS3 enables high-quality and efficient high-resolution perception in multi-modal LLMs (MLLMs).

Specifically, we train a modern MLLM [69] using pre-trained PS3 as the vision encoder. The resulting MLLM, named VILA-HD, is capable of capturing the global image at low resolution and extracting high-resolution details in the local regions selected based on the user prompt. Evaluated on seven benchmarks that require high-res perception, VILA-HD significantly improves the performance over baselines that use either the original low-res SigLIP or approaches such as S2 [87] and AnyRes [22, 62] that scale up the resolution of SigLIP without high-resolution vision pre-training, while using 4.3× fewer tokens compared to the AnyRes baseline. Additionally, PS3 unlocks several intriguing scaling properties of VILA-HD, for example, scaling up the resolution without extra cost by selecting a constant number of high-res patches and trading more compute for higher performance at test time by selecting larger high-res regions. Comparing to the state of the arts, PS3 outperforms previous vision encoders such as SigLIP2 [99] and Perception Encoder [9] on 23 benchmarks under a controlled setting, and with a more advanced MLLM training recipe, VILA-HD is able to surpass state-of-the-art MLLMs such as NVILA [69] and Qwen2.5-VL [4] on various benchmarks and achieve superior efficiency and performance over latest token pruning approaches [8, 17, 117].

Despite the superior performance of PS3, we find most existing benchmarks do not actually require 4K resolution. Therefore, we introduce 4KPro, a benchmark that evaluates visual perception at 4K resolution in four professional use cases including autonomous vehicle, household, gaming, and UI understanding. For each category, 4KPro contains image QA pairs where each question can only be answered under 4K resolution. On 4KPro, PS3 shows a significant improvement of 15% over S2 and AnyRes baselines and achieves state-of-the-art results compared to both proprietary and open-sourced MLLMs including GPT-4o [47] and Qwen2.5-VL [4] in terms of both performance and efficiency (Figure 1(Bottom Right)).

### 2. PS3: Vision Pre-Training at 4K Resolution

Based on the paradigm of contrastive language-image pre-training (CLIP) [80] which optimizes a contrastive loss between global images and global captions, we propose PS3 which instead optimizes the contrast loss between local regions and detailed captions about the regions (Figure 1(Upper Right)). In this way, the model efficiently learns language-aligned detailed representation by being scale-selective, i.e., only processing the selected regions at a fine-grained scale. This detaches the computational cost from the global image size, allowing us to scale up to ultra-high image resolution during pre-training by controlling the size of the local regions.

The scale-selective pre-training requires a redesign of data, model, and algorithm. We first collect 75M high-resolution images with 282M pairs of bounding boxes and captions of salient local regions (Section 2.1). We then design the architecture of PS3 that can both encode low-resolution global images and select local high-resolution patches to process based on image saliency or their relevance to a text prompt (Section 2.2). We finally pre-train PS3 jointly with localized contrastive loss for high-res perception and box supervision for patch selection (Section 2.3).

#### 2.1. Pre-Training Data of PS3

To learn fine-grained perception in high-res images through contrastive loss between local regions and captions, we need to collect high-res images together with bounding boxes and captions of local regions in each image. We need to make sure the local regions contain rich details in order for the model to learn fine-grained representation. In this work, we collect 75M high-res images with 282M pairs of bounding boxes and captions for both natural images and document images. An example of the pre-training data is shown in Figure 3 and more examples of natural and document images can be found in Figure 15-16 in the Appendix. The curation of each component of the data is detailed below.

[Figure 16]

[Figure 17]

Segment Everything

Salient Region Detection Caption

[Figure 18]

[Figure 19]

###### MLLM

(e.g., Qwen2-VL-7B)

- Figure 2: Curation of bounding boxes and captions of salient regions in the pre-training data. For each high-resolution image, we segment all the masks, detect salient regions with small or dense masks, and use an MLLM to generate captions about the local regions.

|[Figure 20]<br><br>|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

...

|There is a banner with text and images related to Thai cooking, with the words "Thai Cooking With Your Heart"...|
|---|

|There is a metal vent or grill on the left side, and a table with a black tablecloth in the middle. On the table, there are two silver pots...|
|---|

...

Figure 3: Pre-training data example. Each instance contains an image with resolution up to 4K, bounding boxes of the salient regions in the image, and captions about details in the regions such as text or small objects.

High-resolution images. We collect 75M images with 1K - 4K resolution. These include 38M natural images from DataComp [38] and SA-1B [53] and 37M document images from IDL [7] and PDFA [75].

Local captions and bounding boxes of salient regions for natural images. We propose a pipeline of first detecting salient regions and then generating local captions (Figure 2). For saliency detection, inspired by recent work on segmenting everything in an image [53, 127, 104], we first use EfficientViT-SAM [127] to generate segmentation masks of the whole image, and then select local regions containing small or dense masks as salient regions. The intuition is that a local region should contain small or cluttered objects in order to have rich details. The saliency detection algorithm is explained in more details in the Appendix A. For local captions, we use an off-the-shelf MLLM (e.g., Qwen2-VL [102]) as an captioner. Specifically, we zoom in and crop the local region, send it along with the global image to the MLLM, and let it describe the local region based on the global context. This results in 3 - 4 pairs of local captions and bounding boxes per image and 134M pairs in total, with an average box size around 400×400.

Local captions and bounding boxes for documents. Both IDL [7] and PDFA [75] provide bounding boxes and OCR results of sentences or words in each PDF. Therefore, we simply sample from these bounding boxes and use the corresponding OCR results as the local captions. We generate 148M pairs of boxes and captions with an average box size around 50×400.

Global captions. Our pre-training also uses global captions (see Section 2.3). We use the same MLLM captioner to generate global captions for natural images. For document images we do not use any global captions.

#### 2.2. Model Design of PS3

We design the model such that given a high-res image, it can 1) extract low-res global features, 2) select local regions based on saliency or their relevance to an input text prompt, and 3) extract high-res features of the selected regions. The whole model can be divided into three stages corresponding to these three capabilities respectively (Figure 4). The design of each stage is detailed below.

Stage 1: Low-res feature extraction. We use the same vision transformer (ViT) architecture as

Low-res tokens High-res tokens Prompt embedding Auxiliary high-res features Shared weights

Two men sitting on the dock.

[Figure 23]

or

N x

N x

Top-k multi-scale high-res patches

[Figure 24]

[Figure 25]

756 x 756

Selection Score Prediction Light-Weight High-Res Encoder

Att MLP

Att MLP

...

[Figure 26]

[Figure 27]

378 x 378

3780 x 3780

Low-res KV cache

1512 x 1512

###### Stage 1: Low-Res Feature Extraction Stage 2: Top-Down or Bottom-Up Patch Selection Stage 3: High-Res Multi-Scale Feature Extraction

- Figure 4: Model architecture of PS3. The model consists of 3 stages. In Stage 1, the model encodes global low-resolution features. In Stage 2, based on the low-resolution features as well as auxiliary high-resolution features extracted by a light-weight encoder, the model selects local regions that are either relevant to a text prompt (top-down selection) or salient by themselves (bottom-up selection). In Stage 3, the model processes multi-scale high-res patches from the selected regions with the same encoder from Stage 1. KV cache from the low-res tokens in Stage 1 is added to the self-attention layers to provide a global context for local high-res encoding.

SigLIP-SO400M [122] to extract low-res features. The image is resized to 378×378 which corresponds to 27×27 output tokens.

###### Stage 2: Top-down or bottom-up patch selection. In this stage, the model selects important regions either based on their relevance to a text prompt (i.e., top-down selection) or based on the saliency of the region itself (i.e., bottom-up selection) [128, 11]. See Figure 6(Left) for examples of such selection. To achieve this, the model predicts a selection score for each spatial position of the image by calculating the cosine similarity between the low-res visual features (from Stage 1) and the embedding of the prompt. The prompt embedding is either the text embedding in the case of top-down selection or a constant learnable vector in the case of bottom-up selection, following [86]. The text embedding comes from the text encoder in our contrastive pre-training.

The selection score is calculated with low-res features only, making it infeasible to locate fine-grained details. To alleviate this issue, we predict additional high-res selection score following the same process but with auxiliary high-res features extracted by a light-weight encoder. The light-weight encoder is a ConvNeXt [68] model with only 3 blocks and extracts features at 1512 resolution. The high-res and low-res selection scores are then interpolated to the same size and averaged as the final score.

###### Stage 3: High-res multi-scale feature extraction. Stage 3 consists of a few key steps. 1) Selecting top-𝑘 multi-scale high-res patches. The model first resizes the high-res image to a set of pre-defined scales and patchifies each. For example, we use three scales of 756×756, 1512×1512, and 3780×3780 for a maximum resolution of 4K. Each is then patchified to 54×54, 108×108, and 270×270 patches, respectively. The selection score from Stage 2 is also interpolated into the same each size. Then for each scale, top-𝑘 patches with the highest score are selected. Note that 𝑘 can vary for different scales. During pre-training, we set 𝑘 for each scale to be proportional to the total number of patches at that scale, e.g., 𝑘 equals to 80, 320, 2000 for scales of 756×756, 1512×1512, and 3780×3780. This ensures the exact same image region is selected for each scale. However, the user can flexibly decide which scale to select more patches by setting different 𝑘 for different scales, which we find benefits different downstream tasks (Section 7.3). 2) Scale-aware positional embedding. After embedding each selected patch with the same patch embedding module as in Stage 1, we then add positional embedding for each token by interpolating the original low-res positional embedding and selecting the

Low-res vision features High-res vision features text Global caption text Local caption global caption embedding local caption embedding [2280, ..., 2448] local bounding box

The player's jersey has number 22.

[Figure 28]

[Figure 29]

|[Figure 30]|
|---|

[Figure 31]

The player's jersey has number 22.

supervise

[2280, 1888, 2680, 2448]

|[Figure 32]|
|---|

[Figure 33]

contrast

There are several black sprinkler heads on the ground.

- (b) top-down patch selection supervision
- (c) bottom-up patch selection supervision

[2280, 1888,

[Figure 34]

[Figure 35]

2680, 2448] ... [2576, 1104,

|[Figure 36]|
|---|

supervise

Interior of a church with marble columns.

2968, 1672]

(a) Vision-language contrastive learning

- Figure 5: Pre-training algorithm of PS3. (a) During training, PS3 extracts the high-res features from the labeled local regions and contrasts them with embeddings of the local captions. To maintain the low-res feature quality, we also mix pairs of low-res features and global caption embedding in each batch. Both high-res and low-res features are extracted in the same way as Figure 4. (b) The top-down patch selection score is supervised by ground-truth score map generated from the bounding box corresponding to the local caption. (c) The supervision for bottom-up selection is similar to top-down selection, except that the ground-truth selection score is generated from all the labeled bounding boxes of the image.

positional embeddings that correspond to the selected patches. On top of that, we add a scale-specific positional embedding to tokens from each scale such that the ViT is able to discriminate tokens from the same spatial position but different scales. 3) High-res feature extraction with low-res KV cache. The selected tokens from different scales are gathered and simultaneously processed by the same ViT as in Stage 1. To make the local high-res features aware of the global visual context, we augment the keys and values in the self-attention layers with the keys and values from the corresponding layer in the low-res feature extraction stage, similar to the KV cache in modern LLMs [79]. The effect of each step above is studied in Section 7.1.

#### 2.3. Pre-Training Algorithm of PS3

PS3 is pre-trained to jointly learn 1) detailed visual representation through localized contrastive loss and 2) top-down and bottom-up patch selection from box supervision.

Learning high-res visual representation. As illustrated in Figure 5(a), given the paired data of high-res images and detailed local captions, we use PS3 to extract the high-res features of the local regions that are relevant to the local captions as described in Section 2.2, extract text embedding of the local captions using a text encoder, and optimize a contrastive loss between the high-res visual features and the text embeddings. The total number of selected high-res patches for each image is limited to 2560 during pre-training for efficiency, while one can choose to select more tokens for downstream applications (Section 3.1). We use the same sigmoid contrastive loss as in SigLIP [122]. Both the ViT backbone in PS3 and the text encoder are initialized with the pre-trained SigLIP.

There are several key designs in the contrastive pre-training. The effect of each design is studied in Section 7. 1) Using ground-truth selection score for patch selection. Normally PS3 selects patches based on the local caption. However, in pre-training, to avoid inaccuracy in the selection score predicted

|[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

There is a red symbol on the ground.

How many people are there on the boat?

What is the name of the ship?

Who is the author of the paper?

What is the website released in the paper?

The number on the player's jersey is 22.

- Figure 6: Qualitative examples of patch selection. Left: PS3 is pre-trained to perform bottom-up selection based on image saliency (denoted by ∅) or top-down selection based on local captions. The selection process is detailed in Figure 4 and Section 2.2. Middle & Right: We fine-tune PS3 with MLLM to select patches based on questions about local regions (Figure 7 and Section 3.1).

by the model which may lead to selecting irrelevant regions, we use selection score generated from the ground-truth bounding box. This is similar to Teacher Forcing [109] in training recurrent neural networks, where the ground truth of an intermediate prediction is given in order to better train the consequent prediction of the model. The ground truth selection score is generated by setting the score inside the box to 1 and others to 0. 2) Pooling only tokens in the ground-truth boxes. SigLIP uses attention pooling to compress all the output tokens into one for contrastive loss. When a box contains fewer patches than the pre-set 𝑘, the model will select patches outside the box as well. Pooling both tokens inside and outside the box results in aligning irrelevant visual features to the text embedding in contrastive loss. To avoid this, we constrain the attention pooling to only tokens inside the box. 3) Mixing global and local contrast. We empirically find that optimizing contrastive loss only between local regions and captions can degrade the quality of global low-res representations. To this end, we mix global and local contrast, i.e., each batch contains both pairs of global low-resolution features and global caption embeddings and pairs of local high-resolution features and local caption embeddings. 4) Avoiding intra-image contrast. Since we have multiple local boxes and captions for each high-res image, there is a chance that one batch contains multiple local regions from the same image. It can be problematic to contrast between different regions of the same image if those regions are visually similar [13]. We make sure each image only appears once in a batch to avoid intra-image contrast. The effect of each design is studied in Section 7.1.

Learning top-down and bottom-up patch selection. As illustrated in Figure 5(b), we use box supervision to learn top-down and bottom-up patch selection. During pre-training, the model predicts both the bottom-up selection score (i.e., without any text prompt) and the top-down selection score based on the local caption for each image. To learn top-down patch selection, we supervise the selection score with the ground-truth bounding box that corresponds to the local caption. Specifically, we treat it as a binary semantic segmentation problem, where the predicted likelihood of binary classification is the selection score at each position and the ground truth label is 1 inside the ground-truth box and 0 outside. A position-wise cross entropy loss as well as a DICE loss [90] is then optimized between the selection score map and the ground-truth map. For bottom-up selection, since we already label the bounding boxes of salient regions in each image, we directly use these boxes to generate ground truth segmentation map by setting the label of a position to 1 as long as it belongs to one of the boxes. We then supervise the bottom-up selection score using the same loss functions as above. Figure 6(Left) shows examples of the learned top-down and bottom-up patch selection.

There are two men...

###### LLM

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

x N

How many men are there sitting on the dock?

High-Res Patch Selection

##### ...

- Figure 7: Model design of VILA-HD. For any input image and text prompt, VILA-HD first extracts the low-res image features using PS3 and sends them along with the text tokens to the LLM. The last-layer embedding of the last token is used to select high-res patches in PS3, whose features are then extracted by PS3, added with additional positional embedding, and sent to the LLM. Although PS3 only processes at most 2560 high-res patches at a time, one can run the patch selection and high-res feature extraction for N times (N can be an arbitrary number) to encode up to 2560×N high-res patches.

### 3. VILA-HD: Enabling High-Resolution MLLM with PS3

We apply PS3 to MLLMs to enhance their high-resolution perception capability. Specifically, we propose VILA-HD, an MLLM with PS3 as the vision encoder that shows suprior performance and efficiency in processing images of up to 4K resolution. In the following we introduce the model design (Section 3.1) and training data (Section 3.2) of VILA-HD.

#### 3.1. Building VILA-HD with PS3

VILA-HD’s model architecture is based on NVILA [69]. NVILA is a LLaVA-style [61] MLLM with SigLIP [122] as the vision encoder, and we replace the vision encoder with PS3 to build VILA-HD. The model design of VILA-HD is illustrated in Figure 7. We first extract the global low-res features following Stage 1 of PS3 and send them along with the text tokens to LLM. We then select high-res patches in either a bottom-up or a top-down way. Bottom-up selection is exactly the same as in pre-training. For top-down selection, since we need to select regions that can help answer the user’s question, instead of using the text embedding from the SigLIP text encoder as the prompt, we use the latent embedding of the last token in the user’s text input from the last layer of LLM as the prompt embedding. This is inspired by LISA [54] which uses the same embedding for reasoning segmentation. Finally, the selected high-res patches are encoded by Stage 3 of PS3 and sent to LLM after the text tokens, from which the following text generation resumes. We also add an additional positional embedding to the high-res features before sending them to LLM such that LLM is aware of the spatial positions of the selected patches, similar to the scale-aware positional embedding Stage 3 of PS3. Note that while the number of selected high-res patches is limited to 2560 during pre-training, one can select an arbitrary number of patches when applying to MLLMs by running patch selection and high-res feature extraction for multiple times. For example, to select 3840 high-res patches, one can first select the top 2560 patches to process in Stage 3, and then select another top 1280 patches among the rest of the unselected patches and process them in Stage 3. The multiple runs of Stage 3 are independent from each other. Then all the high-res tokens are gathered together and sent to LLM.

Pre-training data

Regular image QA data

512 x 512

[Figure 55]

[Figure 56]

There are four containers with broccoli, mashed potato, and fruits...

What do you see in the lunch tray?

Two men sitting on the dock.

[2280, 1888, 2680, 2448]

LLaMA-3.1: Generate a question based on the local caption...

paste on large-size solid backgronud

[Figure 57]

[Figure 58]

There are four containers with broccoli, mashed potato, and fruits...

How many men are there sitting on the dock?

[2280, 1888, 2680, 2448]

What do you see in the lunch tray?

4096 x 4096

MLLM ﬁne-tuning data for patch selection

MLLM ﬁne-tuning data for high-res feature alignment

- Figure 8: Additional fine-tuning data for MLLMs with PS3. Left: To fine-tune top-down patch selection, we generate data with pairs of high-res image, question about a local region, and the bounding box of the region. This is generated by taking the PS3 pre-training data and retargeting the local captions into questions using LLaMA-3.1. Right: To align the PS3 high-res features to the LLM text space, it requires fine-tuning data that contains QA pairs on high-res images. We generate this by taking regular low-res image QA data and pasting the image onto a large-size background to get the new high-res image while the question and answer are inherited.

#### 3.2. Fine-Tuning Data of VILA-HD

Except for regular fine-tuning data for training MLLMs in previous work [69, 56], we collect additional data for 1) learning top-down patch selection in VILA-HD and 2) aligning the high-res features from PS3 to the text space of VILA-HD. Each type of data is introduced below and Figure 8 illustrates the curation of the data.

Fine-tuning data for patch selection. When training VILA-HD, we jointly fine-tune the top-down patch selection since it uses different prompt embeddings compared to pre-training. For this purpose, we collect data of high-res images paired with questions about local regions as well as their bounding boxes. As shown in Figure 8(a) This is achieved by directly sampling images and bounding boxes from the pre-training data (Section 2.1) and then automatically generate questions based on the local captions using LLaMA-3.1 [31]. We generate 500k data samples from the natural images and document images in the pre-training dataset respectively. To check the quality of the generated data, we randomly sample 500 examples and manually check whether the generated questions correctly corresponds to the original captions and local regions, and find 95.8% of the questions are accurate. During training, we run VILA-HD with PS3 on this data to perform top-down patch selection based on the questions, and then supervise the patch selection following the same objective as in Section 2.3. Note that this data does not have answers to the questions so we do not optimize the next-token-prediction loss on this data. See Figure 6(Middle & Right) for visualization of the learned patch selection.

Fine-tuning data for high-res vision feature alignment. Since most of the existing fine-tuning data for MLLMs only contains low-resolution images, the high-res vision features from PS3, especially the features extracted at 4K resolution, are not properly aligned to the LLM’s text space if they are only trained on low-resolution data. To solve this problem, we collect 225k VQA data at up to 4K resolution. The data curation pipeline is shown in Figure 8(b). Specifically, we take existing low-res VQA data,

paste the images onto a solid-color background image with larger size such as 4096×4096, and use the new synthesized image with the original question as the high-res VQA data. By training on this data, VILA-HD is forced to utilize the high-res features from PS3 to answer the question because the informative region in the synthesized image is small, e.g., a 256×256 region in a 4K image. During training, we mix this data with other fine-tuning data. We find this simple approach works reasonably well in our empirical studies (Section 7.4).

### 4. Scaling Properties of PS3

In this section, we evaluate how well the performance of PS3 scales with the pre-training resolution. Specifically, we pre-train PS3 at different resolutions and compare the performance of VILA-HD with the PS3 encoders as well as baseline vision encoders using a fixed MLLM training recipe. We pre-train PS3 at three resolutions of 756, 1512, and 3780, where for each resolution we extract multi-scale high-res features at scales of (756), (756, 1512), and (756, 1512, 3780), respectively. We show four types of scaling: 1) Whole-image resolution scaling. PS3 selects all the high-res patches at each resolution for VILA-HD. This is to compare the high-res feature quality of PS3 with the baselines that also process all high-res patches (see Experiment settings below). 2) Constant-cost scaling. PS3 selects a constant number of high-res patches when scaling the resolution for VILA-HD. This evaluates if performance scales ‘‘for free’’, i.e., by maintaining a constant MLLM training and inference cost (note that the pre-training cost is already near-constant). 3) Constant-resolution scaling. At the same resolution, PS3 selects increasingly more high-res patches for VILA-HD. This evaluates if we gain benefits from selecting more patches in downstream without changing the pre-training and fine-tuning resolution. 4) Test-time scaling. Similar to 3), but we increase the number of high-res patches at test time. The scaling curves are shown in Figure 9 and more detailed quantitative results are reported in Table 1.

Experiment settings. PS3 is initialized with SigLIP-SO400M [122] before pre-training. Please see detailed pre-training setting in Appendix B. VILA-HD is trained using a subset of the training data in NVILA. Please see the detailed training recipe of VILA-HD in Appendix C. For evaluation, we report average accuracy on seven resolution-sensitive benchmarks: TextVQA [89], ChartQA [72], DocVQA [73], InfoVQA [74], OCRBench [66], V*Bench [110], and RealWorldQA [112]. We compare PS3 to the original SigLIP as well as two baselines, AnyRes [22, 62] and S2 [87], that run SigLIP at larger resolution in a training-free way by splitting large images into smaller tiles. For AnyRes baselines, we set the maximum number of tiles to 4 for 756 resolution and 9 for 1512 resolution. For S2 baselines, instead of resizing the feature maps from different scales to the smallest scale as in the original algorithm, we resize them to the largest scale to get the best performance. For PS3, the number of selected patches for multiple image scales at test time might vary between benchmarks to obtain the best performance (Section 7.3). Please see Appendix C for the detailed evaluation setting.

#### 4.1. Whole-Image Resolution Scaling

We first scales up the resolution of PS3 and make it select all the high-res patches at each resolution for VILA-HD. Results are shown in Figure 9(a). We can see that the effect of scaling up resolution is significant, where PS3 at 1512 resolution improves 14.2% over SigLIP baseline. Compared to AnyRes and S2, PS3 shows consistent improvements across different resolution while using a similar number of high-res tokens. For example, at 1512 resolution, PS3 improves by 2.4% over S2 and 6.9% over AnyRes which is commonly used by modern MLLMs [56, 22]. Since all the methods are processing the whole high-res image, the advantage of PS3 mainly comes from the improved high-res feature quality which is brought by our high-res pre-training. Note that we cannot scale up to 4K resolution while selecting all the high-res patches due to the drastic computational cost, although it is achievable with

[Figure 59]

[Figure 60]

#HR token for MLLM

Original (SigLIP)

S AnyRes

0 729 1600 3840

PS3

PS3 (test-time scaling)

[Figure 61]

(a) Whole-image resolution scaling (b) Constant-cost scaling

[Figure 62]

[Figure 63]

(c) Constant-resolution scaling (d) Test-time scaling

- Figure 9: Scaling properties of PS3 on VILA-HD. (Left) Overall results. We report average performance of the MLLM on seven benchmarks under different maximum input resolution. The size of each data point indicates the number of high-res vision tokens input to the LLM. (a) When selecting all high-res patches for MLLM, the performance of PS3 scales better with the resolution than the baselines without high-resolution pre-training. (b) PS3 is able to process higher resolution and improve performance while selecting a fixed number of high-res patches for MLLM. (c) Within the same resolution, PS3 is able to trade compute for performance by selecting more high-res patches. (d) PS3 can select more high-res patches at test time even if its selects a fixed number of high-res patches during MLLM training.

Table 1: Full results of scaling properties of PS3 on VILA-HD. #HR Token is the number of high-res tokens input to the MLLM. Select (Train) is the percentage of high-res patches PS3 selects when training the MLLMs and Select (Test) is the selection ratio at test time. Note the number of HR tokens is usually 1/4 of the number of selected patches due to the 2×2 downsampling connector in NVILA [69]. †#HR tokens for AnyRes depends on the input resolution and we report the maximum number of tokens.

Vision Encoder

Max Res.

#HR Token

Select (Train)

Select (Test)

Text VQA

Chart QA

Doc VQA

Info VQA

OCR Bench

V* Bench

Real World

Avg

SigLIP [122] 378 0 - - 62.3 56.6 51.9 30.7 387 51.8 57.1 49.9 AnyRes [62] 756 784† - - 65.3 58.0 60.6 32.7 416 59.2 59.1 53.8 S2 [87] 756 729 - - 65.9 65.5 63.0 32.3 471 53.1 59.6 55.2

756 320 44% 44% 66.7 62.8 62.6 33.1 460 56.3 61.7 55.6 PS3

756 729 100% 100% 66.8 63.5 64.6 33.9 462 56.5 61.7 56.2

AnyRes [62] 1512 3136† - - 67.4 58.4 67.9 34.1 468 60.2 59.0 56.3 S2 [87] 1512 2916 - - 66.1 71.0 78.3 41.1 526 55.2 61.0 60.8

1512 729 20% 20% 67.3 64.7 66.5 34.8 505 60.7 62.6 58.2 1512 1600 20% 44% 67.7 65.9 70.7 35.7 515 62.0 62.6 59.4 1512 1600 44% 44% 68.4 68.0 74.5 37.3 509 63.1 65.0 61.0 1512 3645 44% 100% 68.4 68.0 76.5 39.4 522 66.7 62.0 61.9

PS3

1512 3645 100% 100% 69.3 71.1 79.4 41.3 534 64.0 63.8 63.2

AnyRes [62] 3780 19600† - - OOM S2 [87] 3780 18225 - - OOM PS3 3780 3840 18% 18% 69.8 70.9 79.1 40.5 543 67.8 64.7 63.9

constant-cost scaling in Section 4.2.

#### 4.2. Constant-Cost Scaling

Scaling up resolution comes at a cost of quadratically increasing number of tokens in Section 4.1. However, for PS3, higher resolution is still beneficial even when selecting a constant number of patches (Figure 9(b)). For example, using 729 (20%) high-res tokens with 1512 resolution improves the accuracy by 2% over using 729 (100%) tokens with 756 resolution (Table 1). This is because at 756 resolution, not all patches are relevant to the user’s questions, and by scaling up the resolution to 1512, even it selects the same number of patches, there are fewer irrelevant 756-resolution patches but more relevant 1512-resolution patches being selected, thanks to our top-down selection mechanism. Comparing to AnyRes, PS3 achieves 58.2% accuracy at resolution of 1512 through constant-cost scaling, improving over AnyRes by 2% accuracy while using 5× fewer tokens. Notably, while neither AnyRes or S2 can scale to 4K resolution, PS3 is able to scale up to 4K resolution with a constant training and inference cost, where PS3 achieves 63.9% accuracy, improving over S2 at 1512-resolution by 3.1% while using a similar number of tokens. Specifically, the accuracy on V*Bench improves from 64.0% to 67.8% by scaling to 4K resolution, which shows that 4K resolution is important for such detailed visual perception. We observe an even larger advantage for tasks that require 4K resolution (Section 5.1).

#### 4.3. Constant-Resolution Scaling

Pre-trained at a fixed resolution, PS3 can flexibly select different number of patches for VILA-HD to trade compute for performance. As shown by Figure 9(c), selecting more patches at 1512 resolution consistently improves performance. By increasing the number of high-res tokens from 729 (20%) to 1600 (44%), PS3 is able to outperform S2 with only half the number of high-res tokens (Table 1). Further increasing to 3645 (100%) tokens boosts the performance by another 2.2%. We observe that the improvement is especially significant for tasks that has dense visual information and requires processing most parts of the image such as document understanding in DocVQA. On the other hand, tasks such as TextVQA and RealWorldQA which only require understanding of local regions only have minor performance drops (2.0% and 1.2%, respectively) when we select only 20% of the high-res patches. In practice, PS3 allows one to select different number of high-res patches based on the computational budget and the requirements for high-res information density in different tasks.

#### 4.4. Test-Time Scaling

Constant-resolution scaling is still valid at test time, i.e., we select a fixed number of high-res patches during training but select more at test time. As shown in Figure 9(d), at 1512 resolution, we can train with 20% high-res patches and test with 44% patches, which improves the accuracy by 1.2%. Similarly, training with 40% patches but testing with 100% patches gives an improvement of 0.9%. Note that scaling at test time still performs worse than training time, which is possibly because 1) the MLLM receives lower-quality supervision if seeing fewer patches at training time, and 2) the LLM is not adapted to longer context length during training.

Key Finding 1: Scaling up the pre-training resolution of PS3 significantly improves performance of downstream MLLM over baselines without high-res pre-training. This is achievable with near-constant pre-training cost and even with no extra downstream training and inference cost by selecting a fixed number of high-res patches. PS3 can also flexibly select different number of patches to trade training or test-time compute for better performance.

Image Resolution Minimum Recognizable Resolution

2K - 4K

1K - 2K

< 1K

TextVQAChartQAOCRBenchRealWorldQADocVQAV*BenchInfoVQA 4KPro

- Figure 10: Image resolution and MRR of different benchmarks. Existing benchmarks contain high-res images but the resolution required to answer the questions (MRR) is mostly under 1K. In contrast, 4KPro contains questions only solvable at 4K resolution.

### 5. 4KPro: Benchmarking PS3 at 4K Resolution

Despite the suprior performance of PS3 and VILA-HD on existing benchmarks as shown in the previous section, we find those benchmarks do not actually require high resolution visual perception, especially 4K-resolution perception, even though some of them contain high-resolution images. To verify this, we examine the minimum recognizable resolution (MRR) of the existing benchmarks, i.e., the minimum resolution required to answer the questions. We calculate the MRR by randomly sampling examples from each benchmark, manually checking the minimum resolution (4K, 2K, or 1K) under which the visual details are clear enough to answer the question for each example, and averaging the minimum resolutions of different samples for each benchmark. As shown in Figure 10, we can see for all previous benchmarks their MRR is lower than the average image resolutions, which means these benchmarks don’t require visual perception at the same resolution as the images. Specifically, even though benchmarks like DocVQA and V*Bench contain images at 4K resolution, the MRR is mostly around 1K. InfoVQA has the highest MRR of 2K, although it is solely focused on infographic understanding.

To effectively evaluate 4K-resolution perception in real-world tasks, we introduce 4KPro, an image QA benchmark with MRR of 4K from four professional use cases including Autonomous Vehicle, Household, Gaming, and UI Understanding. Each QA pair is in the form of multi-choice problem with four options. Examples of 4KPro are shown in Figure 11, where we can see that all the questions cannot be accurately answered without 4K resolution. We detail the data curation process in Appendix F.

Experiment settings. The training setting is the same as Section 4 where we pre-train PS3 at resolutions of 756, 1512, and 3780, train VILA-HD with each PS3 encoder and evaluate on 4KPro. When comparing with state-of-the-art MLLMs, we use SigLIP2-SO400M [99] as initialization during PS3 pre-training and evaluate VILA-HD with PS3 pre-trained at 1512 and 3780 resolutions, which we denote by VILA-HD-8B-PS3-1.5K-SigLIP2 and VILA-HD-8B-PS3-4K-SigLIP2, and we use the full training recipe in NVILA-8B [69] with differences detailed in Appendix C.

#### 5.1. Main Results

Scaling properties of PS3. We evaluate the performance of VILA-HD when scaling up the resolution of PS3. Results are shown in Figure 12. We can see PS3 outperforms other baselines at resolution of 756 and 1512, although by a relatively small margin, which implies 4KPro is genarlly unsolvable with resolution lower than 4K. However, while it is infeasible to scale to 4K resolution for the baselines, PS3 enables 4K resolution perception by selecting the same number of high-res patches as 1512 resolution, using the same constant-cost scaling scheme as in Section 4. This improves the performance by

What is the next exit? What is next to the black plugin?

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

Nothing.

###### 72A.

756

756

Qwen2-VL

Qwen2-VL

Resolution

Resolution

|[Figure 68]|
|---|

|[Figure 69]|
|---|

A light switch.

###### 64C.

1512

1512

GPT-4o

GPT-4o

|[Figure 70]|
|---|

|[Figure 71]|
|---|

A light switch.

72A.

3780

3780

VILA-HD

###### VILA-HD

What is the level number on top left? What is the RAM usage?

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]<br><br>|
|---|

###### 160.

###### 62%.

756

756

Qwen2-VL

Qwen2-VL

Resolution

Resolution

|[Figure 76]|
|---|

|[Figure 77]|
|---|

52.

62%.

1512

1512

GPT-4o

GPT-4o

|[Figure 78]|
|---|

|[Figure 79]|
|---|

82%.

155.

3780

3780

VILA-HD

VILA-HD

- Figure 11: Examples from 4KPro and comparison of different models. Each example corresponds to one out of four categories (Autonomous Vehicle, Household, Gaming, and UI Understanding) and each question can only be answered without ambiguity under 4K resolution. VILA-PS3 improves the accuracy over the state-of-the-art MLLMs such as GPT-4o and Qwen2-VL.

4.8%. Taking a step further, we can double the number of high-res patches at test time to boost the performance by another 8.2%. This achieves an improvement of ∼15% over AnyRes at 1512 resolution, which is more significant than the improvements we get by scaling to 4K resolution in Section 4, indicating that 4K-resolution pre-training is most helpful for downstream tasks requiring ultra high-res perception. On the other direction, we can also shrink the number of patches by 3×, achieving 48.4% accuracy which is 3.2% higher than AnyRes at 1512 resolution while using 2.5× fewer tokens.

Comparison to state of the arts. We compare the performance of VILA-HD-8B-PS3-1.5K-SigLIP2 and VILA-HD-8B-PS3-4K-SigLIP2 with other proprietary or open-source MLLMs (Table 2), where we abbreviated the two models as VILA-HD-1.5K and VILA-HD-4K. All the proprietary MLLMs do not perform well, with the best accuracy at 59.7% for GPT-4o and Gemini-1.5-Pro. We find part of the errors for proprietary MLLMs are caused by erroneous instruction following or refusing to answer the question when unable to recognize the details. For open-source models, most achieve accuray around 50% - 60%. Qwen2-VL-7B-Instruct performs the best, reaching an accuracy of 71.0%, although at a cost of larger latency than other models due to processing the full high-resolution images in its vision encoder. VILA-HD-4K, on the other hand, achieves 75.8% accuracy at 3780 resolution when selecting 35% patches, which improves over both the proprietary models and the state-of-the-art open-source model (Qwen2-VL) while having a lower latency thanks to the top-down patch selection mechanism. VILA-HD-4K also significantly improves over VILA-HD-1.5K by 8.1%. Note that the performance under 1.5K resolution is still better than random guess (25%) even though the visual details are not clear, which is probably because the blurry details still provide some clues for an informed guess. By selecting

- Figure 12: Scaling properties of PS3 on 4KPro. PS3showsconsistentlyimprovedperformance by scaling to 4K resolution and achieves further improvements by scaling up number of highres tokens at test time, greatly outperforming the baselines.

Original (SigLIP) PS3 (test-time scaling)

AnyRes

0 #HR token 7680

S

PS3

[Figure 80]

###### Table 2: Comparing VILA-HD-8B-PS3-1.5K-SigLIP2 and VILA-HD-8B-PS3-4K-SigLIP2 to state-of-theart MLLMs on 4KPro. †We limit the maximum number of pixels to 4,000,000 to maintain a reasonable latency.

Model Select Latency Acc. GPT-4o [47] - - 59.7 Claude 3.5 Sonnet [1] - - 29.0 Gemini-1.5-Pro [94] - - 59.7 NVILA-8B [69] - 0.82s 58.1 Cambrian-1-8B [96] - 2.78s 50.0 InternVL2-8B [95] - 1.65s 58.1 IXC-2.5-7B [124] - 2.11s 32.3 LLaVA-OneVision-7B [56] - 1.75s 67.7 InternVL3-8B [129] - 2.78s 61.3 Qwen2-VL-7B-Instruct† [102] - 3.61s 71.0 Qwen2.5-VL-7B-Instruct† [4] - 2.98s 68.3 VILA-HD-1.5K 33% 0.46s 59.7 VILA-HD-1.5K 100% 1.20s 67.7 VILA-HD-4K 18% 1.22s 71.0 VILA-HD-4K 35% 1.78s 75.8

fewer patches (e.g., 18%), VILA-HD-4K still maintains superior performance of 71.0% while enjoying only 1/3 of the latency comparing to Qwen2-VL. We also show qualitative results in Figure 11. We can see that compared to state-of-the-art MLLMs, VILA-HD with PS3 is able to accurately recognize extremely fine-grained details under 4K resolution, e.g., tiny text on highway signs or monitor screens and small light switch on a distant wall.

Key Finding 2: Most of previous benchmarks do not require perception at 4K resolution despite the images at 4K resolution, and on 4KPro benchmark that does require 4K resolution perception, VILA-HD with PS3 significantly improves over previous MLLMs in both accuracy and efficiency.

### 6. Comparing PS3 and VILA-HD to State of the Arts

In this section, we compare PS3 and VILA-HD with other state-of-the-art models. Specifically, we compare the visual representation quality of PS3 with state-of-the-art vision encoders [99, 83, 9, 35, 21], compare the performance of VILA-HD with state-of-the-art MLLMs [47, 1, 94, 59, 96, 69, 123, 56, 29, 124, 95, 102], compare the efficiency of patch selection in PS3 and VILA-HD with other MLLM token pruning methods [8, 17, 117], and also verify the generalizability and effectiveness of PS3 pre-training pipeline on different state-of-the-art vision encoders [122, 83].

#### 6.1. Comparing PS3 to State-Of-The-Art Vision Encoders

Experiment settings. To compare with state-of-the-art vision encoders, we first pre-train the PS3 encoder at 1512 resolution using SigLIP2-SO400M [99] as the initialization, which we call PS3-1.5KSigLIP2. Then we evaluate PS3-1.5K-SigLIP2 as well as each baseline vision encoder by training an MLLM

- Table 3: Comparison to state-of-the-art vision encoders. We train an MLLM on top of each vision encoder with the same recipe and compare the performance on various types of benchmarks including high-resolution VQA, VQA over text, documents, and charts, and general VQA. We compare

PS3-1.5K-SigLIP2 and PS3Lang-1.5K-SigLIP2 (abbreviated as PS3 and PS3Lang here) with previous state-of-the-art vision encoders [99, 83, 9, 35, 21]. For a fair comparison, we use similar ViT sizes (300M - 400M) for every encoder and use S2 to scale all the vision encoders (except for PS3 which natively supports high resolution) to a similar high resolution round 1.5K. We compare the vision encoders with and without LLM alignment separately.

w/o LLM alignment w/ LLM alignment Benchmark SigLIP2 C-RADIOv2 PECore AIMv2 PS3 InternViT2.5 PELang PS3Lang High-Resolution VQA

4KPro 37.1 32.3 46.8 43.5 50.0 43.5 45.2 53.2 HRBench4K [103] 55.0 51.8 56.1 51.5 58.5 54.6 52.0 62.9 HRBench8K [103] 49.9 44.8 50.3 47.6 52.5 47.3 49.9 50.5 MME-RealWorld [126] 43.7 40.7 44.1 43.4 42.1 44.9 43.5 40.4 V* [110] 58.8 59.7 64.9 63.1 64.2 57.4 61.1 62.7 Avg 48.9 45.9 52.4 49.8 53.5 49.5 50.3 53.9 Text / Doc / Chart

TextVQA [89] 71.5 68.1 72.0 72.6 71.9 70.2 72.3 73.6 ChartQA [72] 71.3 72.3 73.8 72.3 74.0 75.8 76.4 78.0 CharXiVdq [107] 40.7 40.3 39.3 41.1 41.3 42.7 45.6 41.8 CharXiVrq [107] 15.4 16.2 21.3 15.6 18.2 18.2 24.6 22.9 DocVQA [73] 79.1 82.5 84.5 81.8 82.9 86.0 87.8 87.2 InfoVQA [74] 39.3 40.4 44.2 42.2 44.2 45.9 48.2 48.7 OCRBench [66] 566 542 608 590 558 647 581 659 SEEDBench-2-Plus [55] 60.0 61.3 62.7 61.8 63.6 62.8 64.1 65.1

- Avg 54.2 54.4 57.3 55.8 56.5 58.3 59.6 60.4 General VQA BLINK [37] 44.2 45.4 44.1 44.3 44.0 43.8 43.2 43.4 MME [36] 1952 1983 1897 1962 1995 2029 2000 2031 MMBenchen_dev [65] 75.2 74.1 76.4 75.9 76.0 73.2 73.9 76.5 MMBenchcn_dev [65] 74.3 73.3 75.4 74.6 75.3 70.8 72.1 74.1 MMBench-v1.1en_dev [65] 73.0 72.0 74.6 73.9 73.6 70.5 72.1 74.5 MMBench-v1.1cn_dev [65] 72.8 71.4 74.1 73.1 73.5 68.1 70.0 73.2 MMMU-Prostandard [120] 27.3 26.8 28.2 28.0 27.2 28.1 27.9 29.3 MMMU-Provision [120] 12.3 12.9 12.1 13.0 12.3 13.5 14.2 14.6 MMStar [16] 51.1 51.3 51.9 50.1 52.9 50.4 53.7 53.9 MMVet [119] 44.7 43.7 44.1 43.9 46.3 42.2 44.1 47.1 RealWorldQA [112] 60.3 62.1 61.6 60.9 64.6 59.0 61.4 61.7

- Avg 55.0 54.9 55.5 55.3 56.1 53.8 54.9 55.5

on top of it using the same training recipe as in Section 4 and testing it on various MLLM benchmarks. We do not compare the 4K-resolution PS3 here because we are not able to train MLLM with any baseline encoder at 4K resolution due to OOM issue and thus cannot build a fair comparison between them and PS3-4K-SigLIP2. We compare PS3-1.5K-SigLIP2 with state-of-the-art vision encoders including SigLIP2-SO400M [99], C-RADIOv2-L [83], PECore-L [9], and AIMv2-L [35]. Additionally, we follow the previous literature [9] and present PS3Lang-SigLIP2-1.5K which is the PS3 model that is further aligned with LLMs by co-training with VILA-HD (Section 6.2). Specifically, after training VILA-HD, we take out the vision encoder part as an independent vision encoder. We compare the LLM-aligned version of PS3 with other LLM-aligned vision encoders such as PELang-L [9] and InternViT2.5-300M [21]. For a fair comparison, we train MLLMs with each vision encoder at roughly the same resolution around 1.5K to

- 1.7K. Since every vision encoder except for PS3 is pre-trained at low resolutions, we use S2 to scale them to high resolution when training MLLMs. We evaluate on three types of benchmarks, including high-resolution VQA, VQA related to texts, documents, and charts, and general VQA.

Results. Table 7 shows the full results. For vision encoders without LLM alignment, PECore is the most competitive encoder between the baselines and PS3-SigLIP2-1.5K is able to obtain better performance for high-resolution VQA and general VQA. Especially, PS3 improves the performance by 2% - 4% on high-res benchmarks such as 4KPro and HRBench, indicating the effectiveness of high-res pre-training of PS3. Note that PS3 is worse than PECore on MME-RealWorld, which is possibly because the benchmark requires capabilities such as counting that are beyond simple high-resolution perception which is the main pre-training objective of PS3. For benchmarks with texts, documents, and charts, PS3 is competitive with the state-of-the-art encoder while achieving the best results on benchmarks such

- as ChartQA and CharXiVdq, which is partially credited to the document images in the pre-training data. However, the PS3 pre-training data is not specifically designed for text recognition in natural images, as reflected by the lower performance on OCRBench and TextVQA, leaving huge room for improvement in the future. For general VQA, PS3 also achieves the best performance on average, showing significant improvements on benchmarks such as MMVet and RealWorldQA. On the other hand,

PS3Lang-SigLIP2-1.5K achieves significantly better results compared to other encoders with or without LLM alignment, rendering it the state-of-the-art vision encoder across the benchmarks we evaluate here. Compared to PELang, PS3Lang dominates the performance on most of the benchmarks, including

- 8.0% and 10.9% improvements on highres benchmarks of 4KPro and HRBench4K. It is also worth noting that if we compare the PS3 encoders with and without LLM alignment, the improvements are mostly on the text-related benchmarks, which implies the current MLLM training is still language-centric and fails to improve the vision-centric capabilities of vision encoders.

Key Finding 3: Compared to previous vision encoders with low-resolution pre-training only, PS3 achieves state-of-the-art results on various MLLM benchmarks, especially the ones that require high-resolution. Aligning PS3 with LLMs further boosts its performance on text-related benchmarks.

#### 6.2. Comparing VILA-HD to State-Of-The-Art MLLMs

Experiment settings. To compare with state-of-the-art MLLMs, we train VILA-HD using the full recipe in NVILA-8B [69] (except that we use less data in stage 2 as described in Appendix C) and compare its performance with state-of-the-art MLLMs. We train VILA-HD with two PS3 encoders, PS3-1.5K-SigLIP2 and PS3-4K-SigLIP2, and denote them by VILA-HD-8B-PS3-1.5K-SigLIP2 and VILA-HD-8B-PS3-4KSigLIP2, respectively. We evaluate on several high-resolution VQA benchmarks (4KPro, HRBench [103], MME-RealWorld [126], and V* [110]) and other general VQA benchmarks (TextVQA [89], ChartQA [72], DocVQA [73], InfoVQA [74], MathVista [70], MMBench [65], MMMU-Pro [120], OCRBench [66], and

- Table 4: Comparing VILA-HD to state-of-the-art MLLMs on high-resolution benchmarks. Here VILA-HD-8B-4K refer to VILA-HD-8B-PS3-4K-SigLIP2. Bold and underlined results mean 1st and 2nd best performance for each task. VILA-HD-8B-4K achieves top-2 performance in 4 of the 6 benchmarks. †For most of the models, the resolution and number of tokens are dependent on the maximum number of tiles to use at test time, and we set the maximum number of tiles such that the number of tokens are similar to VILA-HD-8B-4K when it is selecting 18% patches for a fair comparison.

Res.† #Token† 4KPro HRBench4K HRBench8K MME-RW V*att V*pos

NVILA-8B [69] 1552 3072 58.1 59.0 55.5 52.0 68.7 65.8 PLM-8B [26] 1735 4096 59.7 71.3 61.1 63.1 72.2 68.4 InternVL2.5-8B [21] 1735 4096 66.1 65.5 57.4 58.9 75.8 76.5 InternVL3-8B [129] 1735 4096 56.5 67.5 62.5 60.7 72.2 75.0 Qwen2-VL-7B [102] 1735 3840 69.4 66.9 62.1 58.1 76.5 67.1 Qwen2.5-VL-7B [4] 1735 3840 66.1 65.6 64.1 57.1 78.3 71.1 VILA-HD-8B-4K 3780 4036 71.0 67.8 65.3 50.9 76.5 71.1

RealWorldQA [112]). For some benchmarks, we balance the patch selection ratio across different scales such that it reaches the optimal performance at test time. Please see Appendix C for more detailed training recipe and evaluation setting.

Results. We first show the results on high-res benchmarks in Table 4. We compare with state-of-the-art MLLMs under approximately the same number of vision tokens. We can see VILA-HD-8B-4K can process higher resolution with similar number of tokens, thanks to the top-down patch selection mechanism. As a result, VILA-HD achieves the best results on 4KPro and HRBench8K with improvements of 4.9% and

- 2.2% respectively on 4KPro and HRBench8K over Qwen2.5-VL-7B [4]. On HRBench4K and V*att, VILA-HD also achieves second-to-best results. Notably, VILA-HD achieves 7.3% improvement on average over NVILA-8B, which showcases the advantage of high-res vision pre-training considering it is the only difference between the two models. The only benchmark where VILA-HD fails to perform well is MME-RealWorld, which is probably because the benchmark requires more complicated capabilities such as counting which is beyond the scope of PS3 pre-training which is focused on basic capabilities of high-res perception and recognition.

We show the results on other general benchmarks in Table 5, VILA-HD shows competitive performance compared to state-of-the-art MLLMs such as NVILA and Qwen2-VL and achieves the best results in 6 out of 11 benchmarks. Specifically, VILA-HD-1.5K achieves the best results on benchmarks that have the MRR around 512-1K including MathVista and MMMU-Pro, and VILA-HD-4K obtains state-of-the-art performance on benchmarks that require more detailed understanding (MRR between 1K and 4K) such as V*Bench and 4KPro, surpassing NVILA despite using the same recipe and less data, showing the effect of PS3 pre-training. VILA-HD-4K also outperforms all proprietary MLLMs on these high-MRR benchmarks, for example, improving by 13.0% and 11.3% over Gemini-1.5-Pro on V*Bench and 4KPro. Note that this is achieved by selecting only 18% of the high-res patches, showing both the efficacy and efficiency of our model. On the other hand, VILA-HD has slightly worse results than NVILA and Qwen2-VL on OCR-related benchmarks including ChartQA, DocVQA, TextVQA, InfoVQA, and OCRBench. This is probably because the pre-training data of PS3 is not specifically optimized for OCR and text recognition and adding such data in pre-training can potentially solve the problem. We further show that PS3 can achieve even higher efficiency with only minor performance degradation on most benchmarks. Specifically, by selecting only 6% patches for PS3, it maintains competitive performance of VILA-HD-4K while only using 1476 tokens for a maximum resolution of 4K which is less than 1/10 of #token of Qwen2-VL. We can see the performance stays similar compared to 18% patch selection for

- Table 5: Comparing VILA-HD to state-of-the-art MLLMs on other common benchmarks. Here VILA-HD-8B-1.5K and VILA-HD-8B-4K refer to VILA-HD-8B-PS3-1.5K-SigLIP2 and VILA-HD-8B-PS34K-SigLIP2, respectively. Res. is the maximum resolution each model supports. Some models (e.g., Qwen2-VL, InternVL2) can accept input images of different aspect ratios, for which the resolution is calculated as square root of the maximum number of pixels the model can take in. Select is the high-res patch selection ratio of PS3 at test time. #Token is the total number of visual tokens fed into LLM under the maximum input resolution. VILA-HD-8B-4K achieves state-of-the-art performance on high-resolution benchmarks such as V*Bench and 4KPro.

RealWorldQA

MMMU-Pro

OCRBench

MathVista

(standard)

MMBench

(testmini)

*VBench

TextVQA

ChartQA

InfoVQA

DocVQA

(en-dev)

4KPro

(test)

(test)

(test)

(test)

(test)

(test)

(test)

(val)

Res. Select #Token

Proprietary

GPT-4o [47] - - - 85.7 92.8 - 63.8 - 54.0 736 53.7 58.6 - 59.7 Claude 3.5 Sonnet [1] - - - 90.8 95.2 49.7 67.7 - 55.0 788 23.0 59.9 - 29.0 Gemini-1.5-Pro [94] - - - 87.2 93.1 81.0 63.9 - 49.4 754 60.3 70.4 78.7 59.7

Open-source VILA-1.5-8B [59] 336 - 576 52.7 40.6 25.9 36.7 68.9 - - - 52.7 68.5 33.9 Cambrian-1-8B [96] 1024 - - 73.3 77.8 - 49.0 75.9 - 624 59.2 64.2 71.7 50.0 MM1.5-7B [123] 2016 - 1440 78.6 88.1 59.5 47.6 - - 635 - 62.5 76.5 NVILA-8B [69] 1552 - 3072 86.1 93.7 70.7 65.4 87.6 33.6 794 67.2 66.4 80.1 58.1 LLaVA-OV-7B [56] 2304 - 7252 80.0 87.5 68.8 63.2 80.8 29.5 - 69.2 66.3 - 67.7 IXC2-4KHD [29] 2479 - 7920 81.0 90.0 68.6 57.8 80.2 - 675 - - 77.2 42.8 IXC-2.5-7B [124] 2743 - 10000 82.2 90.9 70.0 59.6 82.2 - 690 45.6 67.8 78.2 32.3 InternVL2-8B [95] 2833 - 10496 83.3 91.6 74.8 58.3 81.7 32.5 794 65.8 64.4 77.4 58.1 Qwen2-VL-7B [102] 3584 - 16384 83.0 94.5 76.5 58.2 - - 866 71.0 70.1 84.3 71.0

1512 33% 1411 84.4 89.5 62.3 66.2 92.0 35.0 795 66.5 69.3 79.5 59.7

VILA-HD-8B-1.5K 1512 67% 2626 85.8 92.8 68.0 65.8 92.3 35.2 817 68.0 69.7 80.3 67.7 1512 100% 3841 85.4 93.1 70.0 66.6 92.4 35.0 819 68.5 70.8 80.4 67.7 3780 6% 1476 83.6 89.0 59.7 65.8 91.7 34.9 791 66.5 68.2 79.9 56.5

VILA-HD-8B-4K 3780 12% 2756 84.7 92.3 66.3 66.2 92.4 35.2 794 69.7 69.5 80.4 59.7 3780 18% 4036 84.7 92.7 68.2 66.6 92.6 34.5 802 73.3 69.9 80.5 71.0

benchmarks such as TextVQA and RealWorldQA, and only has minor drops for CharQA and V*Bench. Tasks like DocVQA require denser visual information and thus have larger performance drop.

#### 6.3. Comparing PS3 to State-Of-The-Art Token Pruning Methods

Experiment settings. We compare the efficiency of PS3 on high-res images with state-of-the-art token pruning methods for MLLMs including ToMe [8], FastV [17], and VisionZip [117]. We test the models in both 1512 and 3780 resolution and try different token selection ratio for each resolution. All the baselines use PS3 as the vision encoder for a fair comparison but instead of using the PS3 patch selection module, they use PS3 to process the whole image and prune the tokens in their own way. The PS3 encoder is pre-trained on top of SigLIP-SO400M as in Section 4.

Results. Full results are shown in Table 6. First of all, we can see that PS3 achieves similar efficiency on the LLM backbone when pruning the same number of tokens, while significantly reduces the ViT latency (e.g., 0.286s → 0.096s when selecting 25% patches at 1512 resolution). This is because PS3 reduces the tokens in both ViT and LLM while other methods only prune the tokens in LLM. This advantage is especially important for 4K-resolution images where the ViT latency dominates the LLM latency. For the same reason, the previous methods all run out of memory on 4K resolution images while PS3 is able to process 4K resolution efficiently. In the meantime, PS3 achieves superior results over all previous methods under the same selection ratio. This is due to several possible reasons. For example,

- Table 6: Comparing PS3 to state-of-the-art token pruning methods. PS3 has consistently lower ViT latency and achieve better performances than previous methods. PS3 is also the only method that can handle 4K resolution images.

Select (Test)

ViT Latency

LLM Latency

Text VQA

Chart QA

Doc VQA

Info VQA

OCR Bench

V* Bench

Real World

Method

Avg

###### 1512 Resolution

- Full 100% 0.286s 0.375s 78.6 84.1 92.2 68.1 787 67.9 69.8 77.1

ToMe [8] 50% 0.286s 0.260s 74.1 70.2 59.7 47.3 622 66.8 67.2 63.9 FastV [17] 50% 0.286s 0.264s 78.2 81.2 90.0 60.4 769 66.2 69.0 74.6 VisionZip [117] 50% 0.286s 0.260s 75.2 77.2 79.8 55.7 722 64.0 67.1 70.2 PS3 50% 0.167s 0.260s 77.7 83.4 89.8 60.8 774 67.9 69.1 75.2

ToMe [8] 25% 0.286s 0.180s 72.5 65.5 51.7 42.8 61.1 62.2 63.4 59.9 FastV [17] 25% 0.286s 0.185s 76.1 66.3 78.1 49.5 651 64.6 65.2 66.6 VisionZip [117] 25% 0.286s 0.180s 74.6 76.0 72.8 51.5 694 62.7 64.6 67.4 PS3 25% 0.096s 0.180s 76.8 80.4 84.4 54.6 738 65.7 67.8 71.9

3780 Resolution

- Full 100% 1.812s OOM - - - - - - - -

ToMe [8] 20% 1.812s OOM - - - - - - - FastV [17] 20% 1.812s OOM - - - - - - - VisionZip [117] 20% 1.812s OOM - - - - - - - PS3 20% 0.417s 0.383s 77.8 83.9 91.6 65.0 773 72.8 70.1 76.9

the previous methods are heuristic-based, either merging visual tokens based on their similarities or pruning tokens based on the attention score in the vision encoder or LLM backbone, while PS3 adopts a learning-based approach to select important or relevant visual tokens, which is more accurate than heuristic-based methods given enough data. On the other hand, unlike previous methods such as ToMe and VisionZip that are prompt-agnostic, PS3 is able to select patches based on the user prompt which can be more accurate.

Key Finding 4: PS3’s learning-based top-down patch selection achieves better performance and efficiency than heuristic-based token pruning methods.

#### 6.4. Generalizability of PS3 Pre-Training to State-Of-The-Art Vision Encoders

We verify the generalizability of PS3 pre-training pipeline by pre-training PS3 on top of several stateof-the-art vision encoders and compare with the same vision encoders without high-res pre-training. Specifically, we take the state-of-the-art language-aligned vision model, SigLIP-SO400M [122], and agglomerative vision model, C-RADIO-v2-L [83], as the baselines. We take their pre-trained model and continue pre-training on high-res images using PS3 pre-training pipeline to get PS3-SigLIP-SO400M and PS3-C-RADIO-v2-L. We compare them with the original models as well as their AnyRes [62] and S2 [87] variants. Table 7 shows the results. We can see that for both the base models, the S2 variant achieves the best performance compared to AnyRes variant as well as the original version, while PS3 encoders pre-trained on top of each base model are able to achieve better results, showing that PS3 is a general high-res pre-training pipeline that can be applied to any base vision encoder. We also observe that C-RADIO baseline consistently outperforms SigLIP baseline, and PS3-C-RADIO-v2-L is able to inherit that advantage and outperforms PS3-SigLIP.

- Table 7: Generalizability of PS3 pre-training to state-of-the-art vision encoders. #Param of ViT is the number of parameters of the ViT backbone. Max Res. is the maximum resolution each model processes in MLLM. Max #Tok is the maximum number of vision tokens in MLLM. All the PS3 models select all the high-res patches.

Vision Encoder

#Param of ViT

Max Res.

Max #Tok

Text VQA

Chart QA

Doc VQA

Info VQA

OCR Bench

V* Bench

Real World

Avg

SigLIP-SO400M [122] 400M 378 196 62.3 56.6 51.9 30.7 387 51.8 57.1 49.9 + AnyRes [62] 400M 1512 3332 67.4 58.4 67.9 34.1 468 60.2 59.0 56.3 + S2 [87] 400M 1512 2916 66.1 71.0 78.3 41.1 526 55.2 61.0 60.8

PS3-SigLIP-SO400M 400M 1512 3841 69.3 71.1 79.4 41.3 534 64.0 63.8 63.2 C-RADIOv2-L [83] 320M 384 144 65.0 58.8 53.1 30.9 405 51.5 57.5 51.0

+ AnyRes [62] 320M 1536 2448 68.1 62.8 70.0 35.8 497 65.9 62.8 59.3 + S2 [87] 320M 1536 2304 68.1 72.3 82.5 40.4 542 59.7 62.1 62.8

PS3-C-RADIOv2-L 320M 1536 3024 68.4 72.6 83.2 43.4 569 68.2 61.5 64.9

### 7. Ablations and Analysis

#### 7.1. Key Designs in Pre-Training Algorithm and Model Architecture

We conduct ablation studies on the pre-training algorithm designs (Section 2.3), PS3 model designs (Section 2.2), and MLLM model design (Section 3.1). We follow the same experimental setting as in Section 4 and use PS3 with max resolution of 1512 and 100% patch selection as the baseline. For the ablation of each design, we report the improvement of the baseline model on the average accuracy of the seven benchmarks compared to the model without the design. Results are shown in Table 8. Overall all the designs are helpful. Among the pre-training algorithm designs, we can see that it is crucial to select the patches in the ground-truth boxes during patch selection and pool only their corresponding tokens before calculating the contrastive loss. This is because otherwise the model will select irrelevant patches and contrast them with the local caption, leading to noisy pre-training and degrading the performance by over 8%. We also find avoiding contrast between local regions in the same image improves the performance by 5.3% which aligns with the observation in previous work [13]. The model architecture designs also help with the performance. Notably, the low-res KV cache allows the model to see the global context while extracting the local high-res features, which significantly improves the performance by 8.8%. Additionally, extracting features at multiple scales (e.g., 756 and 1512) performs better than only extracting features at the largest scale (e.g., 1512 only) as in AnyRes. Adding scale-aware positional embeddings in PS3 and additional vision positional embeddings when feeding PS3 features to LLM also improve the performance by a small margin.

Key Finding 5: High-resolution pre-training with localized contrastive learning requires several key algorithm designs, such as contrasting within the ground-truth selection boxes and avoiding contrasting between different regions in the same image, and key model designs including low-res KV cache that makes high-res features aware of the global context.

#### 7.2. Top-Down and Bottom-Up Patch Selection Matters

We compare the performance of using random, bottom-up, and top-down patch selection, as shown in Table 9. We report benchmark accuracy as well as the patch recall rate which evaluates how many patches out of the ground-truth region are selected at test time. This is evaluated on a test set split

- Table 8: Ablation of PS3 pre-training, model, and MLLM designs. ∆ is the change of the average performance on the seven benchmarks after adding the design.

Training and Model Design Choices Δ Pre-training algorithm designs (Section 2.3)

- - using ground truth selection score +5.1
- - pooling only tokens in ground-truth boxes +3.7
- - mixing global and local contrast +1.0
- - w/o intra-image contrast +5.3

PS3 model designs (Section 2.2)

- - multi-scale feature extraction +1.0
- - scale-aware pos. emb. +0.8
- - low-res KV cache +8.8

MLLM model design (Section 3.1)

- additional vision pos. emb. +0.8

Table 9: Ablation of top-down and bottom-up patch selection. Select (Train) and Select (Test) are the percentage of high-res patches PS3 selects at training and test time. Recall is the recall rate of how many patches in the ground-truth regions are selected at test time.

Patch Selection

Select (Train)

Select (Test)

Recall (Test)

Avg. Acc.

Random 44% 44% 43.7% 52.3 Bottom-up 44% 44% 87.4% 59.7 (+7.4) Top-down 44% 44% 91.2% 61.0 (+8.7)

Random 44% 100% 100% 56.5 Bottom-up 44% 100% 100% 61.1 (+4.6) Top-down 44% 100% 100% 61.9 (+5.4)

Top-down 100% 100% 100% 63.2

from the data used to train patch selection in MLLM. We can see that both bottom-up and top-down selection significantly improves the recall and the accuracy over random selection. For example, when selecting 44% patches, bottom-up selection improves the recall rate by 43.7% and the accuracy by

- 7.4% over random selection. Top-down selection further improves the recall rate by 3.8% and the accuracy by 1.3%. Interestingly, patch selection not only affects test time but training as well. When we train MLLM with 44% patches, even if we select all patches at test time, top-down selection is still better than the other two. The reason is likely that top-down selection provides more informative visual context for MLLM during training, which leads to less noisy training dynamics. Note that this model has performance comparable to training with 100% patches, which means that training with top-down patch selection improves training efficiency without hurting the performance too much.

Key Finding 6: Top-down and bottom-up patch selection provides more relevant visual information to MLLM, which does not only improve performance at test time but also improve the model optimization during training.

#### 7.3. Trade-Off Between Different Image Scales

We find the optimal balance of patch selection between image scales varies for different tasks (Figure 13). For example, when using in total 729 (20%) high-res tokens, V*Bench performance peaks when selecting no patches (0%) at 756 scale and only patches (25%) at 1512 scale, while it performs the best on DocVQA when selecting 67% 756-scale patches and only 8% 1512-scale patches. Note that selecting more patches at 756 scale covers more regions of the image because one 756-scale patch represents larger area in the original image than a 1512-scale patch, but lose more details at the 1512 scale. This indicates V*Bench needs smaller coverage of image regions but requires more high-res information at 1512 resolution to perform well. DocVQA, on the other hand, needs more coverage of the image, which is probably because one usually needs to read the whole document to answer the questions. We also observe degraded performance on all benchmarks when selecting only tokens at 756 scale, which is because not all tokens at 756 scales are relevant to the question while the relevant information from 1512 scale is completely lost in this way.

V*Bench

RealWorldQA

InfoVQA

ChartQA

35.25

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

63.0

Performance

Performance

Performance

Performance

60

35.00

- 62
- 63
- 64

58

62.5

34.75

56

34.50

62.0

54

34.25

0%/25%20%/20%33%/17%50%/13%67%/8%100%/0%

0%/25%20%/20%33%/17%50%/13%67%/8%100%/0%

0%/25%20%/20%33%/17%50%/13%67%/8%100%/0%

0%/25%20%/20%33%/17%50%/13%67%/8%100%/0%

Select @ 768/1512

Select @ 768/1512

Select @ 768/1512

Select @ 768/1512

OCRBench

TextVQA

DocVQA

Avg Acc

67.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 56
- 57
- 58

- 65
- 66
- 67

Performance

Performance

Performance

Performance

500

490

67.0

480

66.5

470

0%/25%20%/20%33%/17%50%/13%67%/8%100%/0%

0%/25%20%/20%33%/17%50%/13%67%/8%100%/0%

0%/25%20%/20%33%/17%50%/13%67%/8%100%/0%

0%/25%20%/20%33%/17%50%/13%67%/8%100%/0%

Select @ 768/1512

Select @ 768/1512

Select @ 768/1512

Select @ 768/1512

- Figure 13: Trade-off between image scales for different benchmarks. Select @ 756/1512 are the percentage of selected patches at 756 and 1512 scales at test time, respectively. PS3 can flexibly adjust token selection ratios at different image scales to achieve the best performance for different downstream tasks.

Table 10: Ablation of MLLM SFT data for high-resolution feature alignment. The high-resolution SFT data (HR Data) generally improves high-resolution perception, especially on natural images.

Real World

Vision Encoder

Max Res.

HR Data

Text VQA

Chart QA

Doc VQA

Info VQA

OCR Bench

V* Bench

Avg

PS3 1512 ✗ 68.8 71.2 79.6 39.6 535 60.9 62.9 62.4 PS3 1512 ✓ 69.3 71.1 79.4 41.3 534 64.0 63.8 63.2

(+0.5) (-0.1) (-0.2) (+1.7) (-0.1) (+3.1) (+0.9) (+0.8)

Key Finding 7: The optimal balance of patch selection between different image scales varies for different tasks. Selecting from smaller scales covers more image regions, which is better for tasks with denser visual information. Selecting from larger scales covers less region but captures more detailed information, which is favored by tasks that only require localized understanding.

#### 7.4. SFT Data for High-Resolution Feature Alignment

We study the effect of high-resolution SFT data for MLLMs using PS3 as the vision encoder. Commonlyused image QA data normally contains only low-res images or lacks questions about details in high-res images, from which it is hard for MLLM to learn to utilize the high-res vision features. We hypothesize the high-res QA data in Section 3.2, though generated in a naive way, can help align the high-res visual features to the text space of LLM, thus improving the high-res perception capability. We conduct an ablation study in Table 10. We find that the high-res SFT data in general helps with the performance on resolution-sensitive benchmarks. The improvements are especially significant on natural images, e.g., a

- 3.1% improvement on V*Bench. On the other hand, the performance on ChartQA and DocVQA does not change, which is probably because the current high-res SFT data does not contain document images.

|[Figure 81]|
|---|

[Figure 82]

[Figure 83]

[Figure 84]

|[Figure 85]|
|---|

SigLIP-S (3780 x 3780)

SigLIP-DynRes (3780 x 3780)

|[Figure 86]|
|---|

[Figure 87]

[Figure 88]

[Figure 89]

|[Figure 90]|
|---|

PS3 (756 x 756)

PS3 (1512 x 1512)

PS3 (3780 x 3780)

- Figure 14: PCA visualization of visual features. The baselines, S2 and AnyRes, have either noisy or blurry features at 4K resolution, while PS3 shows extremely fine-grained features that highlight details such as small texts on the banners.

#### 7.5. Visualization of PS3 Visual Representations

We visualize the visual features in PS3 using principal component analysis (PCA) and compare with the baselines of S2 and AnyRes that does not pre-train vision models at high resolutions. Results are shown in Figure 14. We visualize the features of SigLIP-S2 and SigLIP-AnyRes at 3780×3780 resolution and the features from three different scales (756×756, 1512×1512, and 3780×3780) for PS3. We can see the feature map of SigLIP-S2 is blurry compared to other models because S2 concatenates both low-res and high-res features and the PCA visualization is partially dominated by the low-res components. The AnyRes feature map, which is equivalent to S2 but only with the high-res features, shows noisy patterns. One possible reason is AnyRes splits the high-res image into small tiles and individually processes each, which means the features lack global context and are inconsistent when running PCA on the whole feature map. On the other hand, PS3 shows sharp and fine-grained features

- at 4K resolution, e.g., at the edges of objects. We also observe PS3 features from different scales tend to group pixels at different scales. For example, the features at 756 and 1512 scales group large-scale objects or things such as the dock and the sky, while the features at 3780 resolution usually group small-scale patterns such as the texts on the banners.

### 8. Related Work

Scaling up vision pre-training. The enormous efforts from the past few years in both supervised pre-training [42, 68, 80, 30, 122, 121, 25, 27, 23, 35] and self-supervised pre-training [19, 43, 10, 40, 2, 32, 18, 44, 78] have successfully scaled up the size of vision model to billions of parameters and pre-training data to billions of images, which is proven effective in numerous downstream tasks [116, 81, 82, 111, 48, 49]. However, all these vision models are pre-trained at a low resolution (e.g., 384×384) and unable to scale up to higher pre-training resolution due to the costly processing

of full images. Previous work has studied the importance of scaling up input resolution in vision pretraining [92, 93, 28, 5] and attempted at incorporating high-resolution pre-training data [78, 33, 68, 67]. However, they are mostly limited to tasks such as ImageNet classification [85] that are less sensitive to input resolution by nature, and none of these work has tried scaling pre-training to ultra-high resolution (e.g., over 1K resolution) which proves to be important for various real-world applications including MLLMs [87, 95, 29, 102]. RADIO [83, 45] uses a maximum pre-training resolution of 1024×1024 although it does not explore CLIP-style pre-training and is pre-trained by distilling from other pre-trained vision encoders and thus is limited by the resolution of the teacher encoders. Another thread of work tries to improve fine-grained visual perception via localized contrastive learning [13, 6, 91] but none of these work scales up the pre-training resolution with the proposed approaches. In this work we scale up vision pre-training to 4K resolution and show the significant improvement on different MLLM benchmarks.

High-resolution visual processing in MLLMs. The success of LLMs [76, 98, 77, 3] has spurred the progresses of MLLMs [63, 60, 88, 102, 22, 62, 56] with evolving recipe [69, 34, 108, 96] and functionality [41, 46, 24, 12]. Recent work shows the importance of high-resolution perception in MLLMs [62, 22, 87, 57, 114] for tasks that require fine-grained detail perception. Different approaches to high-resolution image processing for MLLMs have been proposed, including splitting the high-res image into smaller tiles and processing each tile separately with a pre-trained lowresolution vision model [87, 58, 62, 22], using multiple vision models to encode an image at different resolution [57, 88, 71, 96], and directly using one single encoder to process high-res images [102, 100]. However, most of these work does not scale up image resolution during the vision pre-training stage and directly processes high-res images with vision encoders pre-trained on low-res images, which is shown to be suboptimal [87]. Furthermore, all the previous methods exhaustively process every pixel in the high-res images without any saliency-based or prompt-aware region selection, leading to extreme inefficiency and rendering them unable to process ultra-high resolution such as 4K resolution. In contrast, PS3 scales up to 4K resolution during the pre-training stage while using a top-down patch selection mechanism that significantly improves the efficiency.

Token pruning for vision encoders and MLLMs. Redundancy in visual information often leads to inefficiency encoding in vision models and MLLMs. Previous work has explored improving the efficiency of vision encoders by pruning redundant vision tokens [84, 118, 20, 101, 8], although the models are all optimized for a specific task such as image classification and cannot adapt to general VQA tasks where token dropping should be based on the input prompt. AbSViT [86] learns to focus on different image regions based on any text prompt but it does not prune irrelevant tokens and is more computationally expensive. Other work has explored adaptively focusing on important regions for fine-grained image recognition [105, 106] although is still limited to low resolution. With the recent progress of MLLMs, visual token pruning methods for MLLMs have been proposed [17, 125, 117, 113] which removes redundant tokens based on either the self-attention in ViT or the text-image cross attention inside LLM. However, these methods still need to process the whole image in the ViT, resulting in large ViT latency especiall for high-res images and making it unable to scale up to 4K resolution. SEAL [110] processes high-res images in a search-and-focus style although the searching process is guided by LLM which adds on huge latency. PS3, on the other hand, processes 4K resolution images with significantly improved efficiency and demonstrates superior performance over previous methods.

### 9. Current Limitations and Future Directions

#### 9.1. Current Limitations

Data. Designs of the pre-training data are not extensively studied in this work due to limit of resources: 1) Quality of high-res images. It is not enough to have images that are just high-res. They should contain rich details as well so the model has the incentive to learn fine-grained representations. This is supported by a preliminary experiment where we compare the effect of SA-1B and DataComp data in pre-training. Specifically, we separately remove the SA-1B and DataComp data from pre-training and find that removing SA-1B has larger impact on the performance despite its smaller size than DataComp. We hypothesize this is because the images in SA-1B usually contains dense or small objects while DataComp images usually have large or even single object in spite of the high resolution. In this work, we do not add any filtering on the richness of visual details in the images, which is necessary for future improvement. 2) Quality of local bounding boxes of salient regions. In this work, we detect salient regions that contain dense or small SAM masks. This heuristic can be improved, for example, by running OCR model to detect regions with texts for text-heavy downstream applications or by filtering out SAM masks that only contain textures but not meaningful objects. 3) Quality of local captions. We use off-the-shelf MLLMs to generate local captions without any data cleaning or filtering. Future improvements can include borrowing the data filtering techniques from LLM synthetic data generation or using visual fact checkers [39] to ensure accuracy of the generated captions. Furthermore, the quality of high-res SFT data for MLLM also needs improving. Currently we vanillaly generate high-res QA data from low-res data, which does not reflect the real distribution of high-res images in the wild. Manual or automatic curation of QA pairs on natural high-res images is better.

Training. In the current model, patch selection is supervised by labeled bounding boxes of salient regions. This can have several drawbacks. For example, since we first generate boxes and then generate captions from the boxes, it is possible the caption also corresponds to other regions outside the box but we treat those regions as negative when training patch selection. On the other hand, the current patch selection only select important regions and process all the scales in those regions, while it is usually the case that not all the scales need to be processed. Learning to select which scale to process is hard to be supervised because of lack of labeled data. All these issues can be addressed by using, for example, reinforcement learning instead of supervised learning where the model learns to find an optimal patch selection strategy itself in order to maximize the downstream performance.

Model. There are several defects in the current model design. First, when we run Stage 2 and Stage 3 of PS3 for multiple times, each group of high-res patches only have the context of the low-res patches but not high-res patches from other groups. This might affect PS3’s ability to model long-range correlation of high-res details. On the other hand, when selecting high-res patches, the selected patches can be scattered around the whole image and sparse within a local region. The sparsity of information affects the quality of local features. This is remedied by adding a soft locality constraint on patch selection, e.g., the model is encouraged to select more centralized and clustered patches. We verify this by adding gaussian smoothing on the selection score map such that patches around any high-score patch also have high scores and each group of selected patches is more clustered. We find this improves the performance when selecting all the patches, supporting our hypothesis. Nevertheless, this degrades the performance when selecting only parts of the patches, which is possibly because the selection score of salient objects is smoothed out if they are small.

#### 9.2. Future Directions

Beyond vision-language pre-training. While PS3 follows the paradigm of CLIP-style vision-language pre-training, the same idea can be applied to any vision pre-training paradigm such as self-supervised pre-training. Taking DINOv2 [78] as an example, the original pre-training objective is to improve the representation consistency of two different views of the image, and using the similar idea of localized processing, one can scale DINOv2 pre-training to 4K resolution by applying the DINOv2 objective to local regions. Furthermore, despite the recent progress on vision expert aggregation [88, 97, 83, 45, 57, 64], it is still an open question of how to combine vision-language and self-supervised pre-training by either joint or sequential training and how it affects performances of MLLM and other downstream tasks. Ideally, pre-training on 4K resolution images with both types of objectives can help the model learn both verbalizable and nonverbalizable features, for example, action-related features for robotic tasks.

Beyond images. PS3 processes 4K-resolution images by removing the redundant patches. Videos, especially high-res, high-fps, and long-form videos, contain much more redundancy and thus can benefit more from the similar idea of patch selection. Furthermore, learning such a patch selection mechanism can force the model to better grasp the world knowledge behind the videos such as physics and dynamics. Although video pre-training still faces lots of problems such as lack of video data and high-quality captions, such bottom-up or top-down spatiotemporal patch selection is a promising way to scale up video pre-training.

### 10. Conclusion

We propose PS3 that scales up CLIP-style vision pre-training to 4K resolution with a near-constant cost. PS3 learns high-resolution perception through local-global region-caption contrast. It encodes low-resolution global image and selectively processes only informative high-resolution regions. PS3 enables VILA-HD, a high-res MLLM whose performance scales with the PS3 pre-training resolution and surpasses state-of-the-art MLLMs with better efficiency. We also introduce 4KPro, a benchmark that evaluates visual perception at 4K resolution, on which VILA-HD outperforms state-of-the-art MLLMs. Acknowledgement. We would like to thank Ekta Prashnani, Joohwan Kim, Alex Naumann, Subhashree Radhakrishnan, Vishwesh Nath, Yucheng Tang, Dong Yang, Holger Roth, and Daguang Xu for their assistance in data curation for the 4KPro benchmark. We are grateful to Greg Heinrich, Mike Ranzinger, Yi Dong, and Zhiding Yu for their feedback on the project. We thank Ligeng Zhu, Zhijian Liu, Yukang Chen, Yunhao Fang, and Xiuyu Li for helping with the codebase. We appreciate Long Lian, Junyi Zhang, and Haven Feng for discussion on the project. We thank Yin Cui and Ming-Yu Liu for their help with the benchmark data curation tools. We thank Andrew Tao, Kari Briski, Bryan Catanzaro, and Bill Dally for their feedback on the manuscript.

### References

- [1] Anthropic. Claude 3.5 sonnet, 2024. 15, 19
- [2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619--15629,

2023. 24

- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 25
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 3, 15, 18
- [5] Irwan Bello, William Fedus, Xianzhi Du, Ekin Dogus Cubuk, Aravind Srinivas, Tsung-Yi Lin, Jonathon Shlens, and Barret Zoph. Revisiting resnets: Improved training and scaling strategies. Advances in Neural Information Processing Systems, 34:22614--22627, 2021. 25
- [6] Ioana Bica, Anastasija Ilić, Matthias Bauer, Goker Erdogan, Matko Bošnjak, Christos Kaplanis, Alexey A Gritsenko, Matthias Minderer, Charles Blundell, Razvan Pascanu, et al. Improving fine-grained understanding in image-text pre-training. arXiv preprint arXiv:2401.09865, 2024. 25
- [7] Ali Furkan Biten, Ruben Tito, Lluis Gomez, Ernest Valveny, and Dimosthenis Karatzas. Ocr-idl: Ocr annotations for industry document library dataset. In European Conference on Computer Vision, pages 241--252. Springer,

2022. 4, 37

- [8] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022. 3, 15, 19, 20, 25
- [9] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, et al. Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181, 2025. 3, 15, 16, 17
- [10] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650--9660, 2021. 24
- [11] Marisa Carrasco. Visual attention: The past 25 years. Vision research, 51(13):1484--1525, 2011. 2, 5
- [12] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14455--14465, 2024. 25
- [13] Hong-You Chen, Zhengfeng Lai, Haotian Zhang, Xinze Wang, Marcin Eichner, Keen You, Meng Cao, Bowen Zhang, Yinfei Yang, and Zhe Gan. Contrastive localized language-image pre-training. arXiv preprint arXiv:2410.02746, 2024. 7, 21, 25
- [14] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021. 39
- [15] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 39
- [16] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056--27087, 2024. 16

- [17] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19--35. Springer, 2025. 3, 15, 19, 20, 25
- [18] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691--1703. PMLR, 2020. 24
- [19] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597--1607. PmLR,

2020. 24

- [20] Xuanyao Chen, Zhijian Liu, Haotian Tang, Li Yi, Hang Zhao, and Song Han. Sparsevit: Revisiting activation sparsity for efficient high-resolution vision transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2061--2070, 2023. 25
- [21] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 15, 16, 17, 18
- [22] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 1, 3, 10, 25
- [23] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185--24198, 2024. 24
- [24] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision language model. arXiv preprint arXiv:2406.01584, 2024. 25
- [25] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818--2829, 2023. 24
- [26] Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, et al. Perceptionlm: Open-access data and models for detailed visual understanding. arXiv preprint arXiv:2504.13180, 2025. 18
- [27] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, pages 7480--7512. PMLR, 2023. 24
- [28] Piotr Dollár, Mannat Singh, and Ross Girshick. Fast and accurate model scaling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 924--932, 2021. 25
- [29] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. arXiv preprint arXiv:2404.06512, 2024. 15, 19, 25
- [30] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1, 24
- [31] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 9

- [32] Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. arXiv preprint arXiv:2401.08541, 2024. 24
- [33] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19358--19369, 2023. 25
- [34] Yunhao Fang, Ligeng Zhu, Yao Lu, Yan Wang, Pavlo Molchanov, Jang Hyun Cho, Marco Pavone, Song Han, and Hongxu Yin. Vila2: Vila augmented vila. arXiv preprint arXiv:2407.17453, 2024. 25
- [35] Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, et al. Multimodal autoregressive pre-training of large vision encoders. arXiv preprint arXiv:2411.14402, 2024. 15, 16, 17, 24
- [36] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 16
- [37] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148--166. Springer, 2024. 16
- [38] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36, 2024. 4, 37
- [39] Yunhao Ge, Xiaohui Zeng, Jacob Samuel Huffman, Tsung-Yi Lin, Ming-Yu Liu, and Yin Cui. Visual fact checker: Enabling high-fidelity detailed caption generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14033--14042, 2024. 26
- [40] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33: 21271--21284, 2020. 24
- [41] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13796--13806, 2024. 25
- [42] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770--778, 2016. 24
- [43] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729--9738, 2020. 24
- [44] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000--16009, 2022. 1, 24
- [45] Greg Heinrich, Mike Ranzinger, Yao Lu, Jan Kautz, Andrew Tao, Bryan Catanzaro, Pavlo Molchanov, et al. Radio amplified: Improved baselines for agglomerative vision foundation models. arXiv preprint arXiv:2412.07679,

2024. 25, 27

- [46] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporal-localization assistant. arXiv preprint arXiv:2403.19046, 2024. 25

- [47] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 3, 15, 19, 40
- [48] Shubham Juneja, Povilas Daniušis, and Virginijus Marcinkevičius. Visual place recognition pre-training for end-to-end trained autonomous driving agent. IEEE access, 11:128421--128428, 2023. 24
- [49] Shubham Juneja, Povilas Daniušis, and Virginijus Marcinkevičius. Dino pre-training for vision-based end-to-end autonomous driving. arXiv preprint arXiv:2407.10803, 2024. 1, 24
- [50] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648--5656, 2018. 39
- [51] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision--ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11--14, 2016, Proceedings, Part IV 14, pages 235--251. Springer, 2016. 39
- [52] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision, pages 498--517. Springer, 2022. 39
- [53] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015--4026, 2023. 1, 4, 37
- [54] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579--9589, 2024. 8
- [55] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790,

2024. 16

- [56] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 9, 10, 15, 19, 25
- [57] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024. 25, 27
- [58] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26763--26773, 2024. 25
- [59] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. CVPR, 2024. 15, 19
- [60] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689--26699, 2024. 25
- [61] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 1, 8
- [62] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 1, 3, 10, 11, 20, 21, 25

- [63] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 25
- [64] Shikun Liu, Linxi Fan, Edward Johns, Zhiding Yu, Chaowei Xiao, and Anima Anandkumar. Prismer: A vision-language model with an ensemble of experts. arXiv preprint arXiv:2303.02506, 3, 2023. 27
- [65] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 16, 17
- [66] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 10, 16, 17
- [67] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012--10022, 2021. 25
- [68] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976--11986, 2022. 5, 24, 25
- [69] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. arXiv preprint arXiv:2412.04468,

2024. 1, 3, 8, 9, 11, 13, 15, 17, 18, 19, 25, 39

- [70] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 17
- [71] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixtureof-resolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024. 25
- [72] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 10, 16, 17, 39
- [73] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200--2209, 2021. 10, 16, 17, 39
- [74] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697--1706, 2022. 10, 16, 17
- [75] Pablo Montalvo and Ross Wightman. https://huggingface.co/datasets/pixparse/pdfa-eng-wds. 4, 37
- [76] OpenAI. GPT-4 technical report. Technical report, OpenAI, 2023. https://arxiv.org/abs/2303.08774. 25
- [77] OpenAI. ChatGPT: Optimizing language models for dialogue. https://openai.com/blog/chatgpt, 2023. Accessed: 2023. 25
- [78] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1, 2, 24, 25, 27
- [79] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 6

- [80] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748--8763. PMLR, 2021. 1, 3, 24
- [81] Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell. Real-world robot learning with masked visual pre-training. In Conference on Robot Learning, pages 416--426. PMLR,

2023. 1, 24

- [82] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022. 1, 24
- [83] Mike Ranzinger, Greg Heinrich, Jan Kautz, and Pavlo Molchanov. Am-radio: Agglomerative vision foundation model reduce all domains into one. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12490--12500, 2024. 15, 16, 17, 20, 21, 25, 27
- [84] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. DynamicViT: Efficient vision transformers with dynamic token sparsification. In NeurIPS, 2021. 25
- [85] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211--252, 2015. 25
- [86] Baifeng Shi, Trevor Darrell, and Xin Wang. Top-down visual attention from analysis by synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2102--2112,

2023. 5, 25

- [87] Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, and Trevor Darrell. When do we not need larger vision models? In European Conference on Computer Vision, pages 444--462. Springer, 2025. 1, 3, 10, 11, 20, 21, 25
- [88] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024. 25, 27
- [89] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317--8326, 2019. 10, 16, 17
- [90] Carole H Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M Jorge Cardoso. Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: Third International Workshop, DLMIA 2017, and 7th International Workshop, ML-CDS 2017, Held in Conjunction with MICCAI 2017, Québec City, QC, Canada, September 14, Proceedings 3, pages 240--248. Springer, 2017. 7
- [91] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Alpha-clip: A clip model focusing on wherever you want. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13019--13029, 2024. 25
- [92] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105--6114. PMLR, 2019. 25
- [93] Mingxing Tan and Quoc Le. Efficientnetv2: Smaller models and faster training. In International conference on machine learning, pages 10096--10106. PMLR, 2021. 25
- [94] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 15, 19
- [95] OpenGVLab Team. Internvl2: Better than the best—expanding performance boundaries of opensource multimodal models with the progressive scaling strategy. https://internvl.github.io/blog/ 2024-07-02-InternVL-2.0/. 15, 19, 25

- [96] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 15, 19, 25
- [97] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568--9578, 2024. 27
- [98] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 25
- [99] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 3, 13, 15, 16, 17
- [100] Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, et al. Fastvlm: Efficient vision encoding for vision language models. arXiv preprint arXiv:2412.13303, 2024. 25
- [101] Thomas Verelst and Tinne Tuytelaars. Dynamic convolutions: Exploiting spatial sparsity for faster inference. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 2320--2329,

2020. 25

- [102] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 4, 15, 18, 19, 25, 37
- [103] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7907--7915, 2025. 16, 17
- [104] Xudong Wang, Jingfeng Yang, and Trevor Darrell. Segment anything without supervision. arXiv preprint arXiv:2406.20081, 2024. 4
- [105] Yulin Wang, Kangchen Lv, Rui Huang, Shiji Song, Le Yang, and Gao Huang. Glance and focus: a dynamic approach to reducing spatial redundancy in image classification. Advances in Neural Information Processing Systems, 33:2432--2444, 2020. 25
- [106] Yulin Wang, Rui Huang, Shiji Song, Zeyi Huang, and Gao Huang. Not all images are worth 16x16 words: Dynamic transformers for efficient image recognition. Advances in neural information processing systems, 34:11960--11973, 2021. 25
- [107] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569--113697, 2024. 16
- [108] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021. 25
- [109] Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2):270--280, 1989. 7
- [110] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084--13094,

2024. 10, 16, 17, 25

- [111] Penghao Wu, Li Chen, Hongyang Li, Xiaosong Jia, Junchi Yan, and Yu Qiao. Policy pre-training for autonomous driving via self-supervised geometric modeling. arXiv preprint arXiv:2301.01006, 2023. 24
- [112] x.ai. https://x.ai/blog/grok-1.5v. 10, 16, 18
- [113] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024. 25
- [114] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. arXiv preprint arXiv:2403.11703, 2024. 25
- [115] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 39
- [116] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371--10381, 2024. 1, 24
- [117] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. arXiv preprint arXiv:2412.04467, 2024. 3, 15, 19, 20, 25
- [118] Hongxu Yin, Arash Vahdat, Jose M Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-vit: Adaptive tokens for efficient vision transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10809--10818, 2022. 25
- [119] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490,

2023. 16

- [120] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. 16, 17
- [121] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12104--12113,

- 2022. 24

[122] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975--11986,

- 2023. 1, 2, 5, 6, 8, 10, 11, 15, 20, 21, 24

- [123] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, et al. Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566, 2024. 15, 19
- [124] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024. 15, 19
- [125] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417, 2024. 25
- [126] Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024. 16, 17

- [127] Zhuoyang Zhang, Han Cai, and Song Han. Efficientvit-sam: Accelerated segment anything model without performance loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7859--7863, 2024. 4, 37
- [128] Li Zhaoping. Understanding vision: theory, models, and data. Oxford University Press (UK), 2014. 2, 5
- [129] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 15, 18
- [130] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. arXiv preprint arXiv:2304.06939, 2023. 39

### A. Details of PS3 Pre-Training Data Curation

The full pre-training data sources and statistics are listed in Table 11.

- Table 11: Data sources and statistics. We collect in total 75M images with 1K - 4K resolution and 282M pairs of bounding boxes and detailed captions about salient local regions in the images.

1K - 2K Res. 2K - 4K Res. #Img #Box

Data Source

#Img #Box

Avg. Box size

Avg. Box size

Natural images

DataComp [38] 18M 54M 424×438 9M 36M 562×578 SA-1B [53] - - - 11M 44M 302×312

Documents

IDL [7] 12M 48M 28×286 7M 28M 30×330 PDFA [75] 12M 48M 80×461 6M 24M 84×569

Agg. 42M 150M - 33M 132M -

High-resolution images. The images consists of two types, natural images and documents. For natural images, we collect 18M images with 1K - 2K resolution and 20M images with 2K - 4K resolution from DataComp [38] and SA-1B [53]. For documents, we take all 37M PDF pages from IDL [7] and PDFA [75] and convert each page into image with DPI of 150, which normally results in images with resolution above 1.5K.

Local captions and bounding boxes of salient regions for natural images. In the saliency detection pipeline described in Section 2.1, we first use EfficientViT-SAM [127] to generate all the masks in each image, similar to the ‘‘segment everything’’ mode in the original SAM [53]. We use EfficientViT-SAM with model size of XL1. The arguments used for generating masks are listed in Table 12. Notably, points_per_side and crop_n_layers largely affect how dense and detailed the generated masks are.

After generating all the masks, we locate local bounding boxes of salient regions that contain small or dense masks. The detailed process is as follows: 1) We preset a set of boxes which are square, have the same sizes, and are uniformly distributed in the image with the distance between adjacent boxes equal to the size of the box. The box size is set depending on the size of the image. For example, for SAM, we set the box size as 1/5 of the shortest side of the image. This results in a typical box size of 300 - 400 in a 2K resolution image. For each square box, we also preset two boxes at the same position with the same area as the square box but with aspect ratios of 1.5 : 1 and 1 : 1.5, respectively. 2) For each box, we calculate the saliency score of the box. The saliency score is the accumulation of the scores contributed by each mask that has overlaps with the box. The contribution to the score from a mask is calculated as max(Area(𝑚𝑎𝑠𝑘), 401·40) / Area(𝑖𝑚𝑎𝑔𝑒) · AreaArea(𝑚𝑎𝑠𝑘(𝑚𝑎𝑠𝑘∩ 𝑏𝑜𝑥) ), where the first term is larger when the area of the mask is smaller compared to the area of the whole image and the second term is larger when a larger portion of the mask resides inside the box. We then select the top-𝑘 boxes with the highest saliency scores. To encourage the selected boxes to cover more areas, we ensure no overlap between boxes.

Finally, after detecting the local salient boxes, we generate captions about the local region by sending two images, the local crop and the global image, to an MLLM and asking it to generate a caption about the local details given the global context. Here we use Qwen2-VL [102] as the MLLM since it has superior results to other open-source MLLMs and can handle multiple images. We set max_pixels = 256 · 28 · 28 for Qwen2-VL to handle both the global image and the local crop. The prompt following the two images is: The second image is a crop from the first image. Given the context of the first image, please describe the second image briefly. Make sure to cover all the objects and texts in the second image. Make sure to describe all the attributes, including color, shape, spatial relations, of each object in the second image. If there’s no text in the second image, you don’t need to mention there’s no text. Please only describe the

###### Table 12: Arguments for EfficientViT-SAM mask generation. Argument Value

points_per_side 24 points_per_batch 128 crop_n_layers 1 crop_n_points_downscale_factor 1 pred_iou_thresh 0.6 stability_score_thresh 0.85 min_mask_region_area 0

###### Table 13: Hyperparameters of PS3 pre-training.

Argument Value #epochs 75 #samples each epoch 1e6 global batch size 8192 / 4096 (for 3780 resolution) %global caption in each batch 25% learning rate 5e-6 warmup iterations 1500

- beta1 0.9
- beta2 0.95 weight decay 3e-4

objects in the foreground and don’t describe the background such as the sky or the weather. Please only describe the objects in the second image and don’t describe the objects in the first image. Please use 1-2 sentences.

Local captions and bounding boxes for documents. IDL provides bounding boxes and OCR results of each sentence, and PDFA provides the same labels for each word. For IDL, we randomly sample one sentence and use the bounding box as well as its OCR result as the caption. For PDFA, we sample 15 consecutive words and use the union of their bounding box and concatenate these words as the caption.

Global captions. We directly use Qwen2-VL to caption the whole image. The prompt is as follows: Please describe the image briefly. Make sure to cover all the objects and texts in the image. Make sure to describe all the attributes, including color, shape, spatial relations, of each object in the image. If there’s no text in the image, you don’t need to mention there’s not text. Please only describe the objects in the foreground and don’t describe the background such as the sky or the weather. Please use 2-3 sentences.

Examples of the pre-training data. See Figure 15-16.

### B. Additional Details of PS3 Pre-Training Algorithm

During pre-training, since the data comes from different data sources (SAM, DataComp, IDL, and PDFA), we sample from each data source such that the probability each data source is sampled is the same for all data sources. Table 13 shows the pre-training hyperparameters.

### C. Additional Details of Training and Evaluation for VILA-HD

- Detailed training and evaluation setting of Section 4. The overall MLLM design and training pipeline of VILA-HD in Section 4 follows NVILA [69]. Specifically, VILA-HD uses PS3 as the vision encoder, a two-layer MLP with 2×2 spatial-to-channel reshaping as the projector, and Qwen2 [115] as the LLM. While the training pipeline of NVILA consists of five stages, we only use stage 1, 3, and 4 in training VILA-HD in this section, where stage 1 only trains the projector, stage 3 trains the trains the LLM and the projector, and stage 4 trains the whole model. For the training data, except for the self-collected data for patch selection and high-res feature alignment, we use the same stage 1 data as in NVILA, use only ShareGPT4V-Pretrain [15] for stage 3, and use a small subset of NVILA SFT data including ShareGPT4V-100K [15], ShareGPT4V-SFT [15], DVQA [50], ChartQA [72], AI2D [51], DocVQA [73], GeoQA [14], and SynthDoG-en [52] for stage 4 training. For training efficiency, instead of mixing the patch selection data with other data in the same stage of training, we use a separate stage to solely train on patch selection data. When training VILA-HD-4K, we adopt a dynamic-resolution training scheme, where we select high-res patches only at scales of 756 and 1512 and none of the patches at 3840 scale for images with resolution under 2646, and select patches for all three scales when the image resolution is above 2646, such that larger image regions are covered for low-res images while more 4K-resolution patches are selected for high-res images. The number of patches selected for each scale is proportional to the image area of that scale. For evaluation of VILA-HD-4K, we only select the high-res patches at 756 and 1512 scales for all benchmarks except for V* where we select patches from all three scales.
- Detailed training and evaluation setting of Section 5 and 6. The training setting in Section 5 (the part of comparing to state-of-the-art MLLMs) and 6 is roughly the same as Section 4, except that we adopt the first four stages in NVILA training pipeline instead of just three of them. The training data for each stage is the same as NVILA except that in stage 3 we only use 10% of MMC4 [130] data. For both VILA-HD-1.5K and VILA-HD-4K, we select 10240 high-res patches for each image during training, which is 70.2% patches for VILA-HD-1.5K and 11.7% patches for VILA-HD-4K. We limit the number of images for each instance to 4 during the first three stages and to 3 during stage 4 in order to improve training efficiency. During evaluation, for VILA-HD-1.5K selecting 44% patches, the number of patches from 756 and 1512 scales are 2560 and 3840. For VILA-HD-4K selecting 18% patches, we select all the patches from 756 and 1512 scales and none from 3840 scales for all the tasks except for high-res benchmarks which we select 1536, 6144, and 7680 patches for the 756, 1512, and 3780 scales. For VILA-HD-4K selecting 12% patches, we select 2048 and 8192 patches for 756 and 1512 scales. For VILA-HD-4K selecting 6% patches, we select 2048 and 3072 patches for 756 and 1512 scales. For VILA-HD-4K selecting 35% patches on 4KPro, we select 1536, 6144, and 23040 patches for 756, 1512, and 3780 scales.

### D. Additional Comments on Training VILA-HD

Parallel training of next-token prediction. Since the high-res patch selection is dependent on the latent embedding of the previous tokens, we cannot train next-token prediction for every token in parallel. Therefore, in training, we first run LLM on the low-res vision features and the text tokens to get the last-layer embedding of the last token, use it to extract high-res vision features, and then run LLM on the full sequence again for next-token prediction. Although this requires running LLM twice, we empirically observe the additional computational cost is marginal since the first run is on a shorter sequence and it does not require gradient because it is only for obtaining the last-layer embedding which is detached before the patch selection. Note that during inference the next token prediction is

###### Table 14: Full results of scaling properties of PS3 on 4KPro.

Max Res.

#HR Token

Select (Train)

Select (Test)

Vision Encoder

Acc

SigLIP 378 0 - - 35.5 SigLIP-AnyRes 756† 784† - - 37.1 SigLIP-S2 756 784 - - 40.7 PS3 756 729 100% 100% 41.9 SigLIP-AnyRes 1512† 3136† - - 45.2 SigLIP-S2 1512 3136 - - 43.6 PS3 1512 3645 100% 100% 46.8 PS3 3780 1280 6% 6% 48.4 PS3 3780 3840 18% 18% 51.6 PS3 3780 7680 18% 35% 59.8

sequential so we only need to run LLM once for each token.

### E. Qualitative Examples of Patch Selection Fine-tuned with MLLMs

We show additional qualitative examples of bottom-up and top-down patch selection of PS3 in Figure 17 and 18.

### F. 4KPro Data Curation

For each category, we first collect videos with 4K resolution (3840×2160) from YouTube, and for each video, we manually select 5 frames containing rich details that are only recognizable under high resolution and label the bounding boxes around the local details. To obtain QA pairs about the details for each frame, we first use GPT-4o [47] to generate a candidate QA pair based on the context of the global image and the content of the local crop, and then manually review and screen each generated QA pair to ensure each question is answerable under and only under 4K resolution and each answer is correct.

### G. Additional Qualitative Results on 4KPro

We show additional qualitative comparison between PS3 and state-of-the-art MLLMs such as GPT-4o and Qwen2-VL in Figure 19.

### H. Full Results of Scaling Properties on 4KPro

Table 14 shows the full results of the experiment of scaling properties of PS3 on 4KPro (Figure 9 in Section 5.1).

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

|The image shows a close-up of a motorcycle's rear section. The motorcycle has a maroon-colored fuel tank with a black seat and a black leather<br><br>saddlebag attached to the side. The motorcycle's license plate is blurred and unreadable. The background consists of green foliage and a blurred road.|
|---|

|The image shows a close-up of a police ofﬁcer and a motorcyclist. The ofﬁcer is wearing a highvisibility vest and holding a device, possibly a breathalyzer. The motorcyclist is wearing a helmet and has a motorcycle with visible details such as the engine and exhaust. The background includes trees and foliage.|
|---|

|The second image is a close-up crop from the ﬁrst image, focusing on the motorcycle rider. The rider is wearing a black helmet and a black jacket with a white stripe. The<br><br>motorcycle is a classic design with a black frame and red and black body. The background shows a forested area with green trees and some buildings partially visible in the distance.|
|---|

|The image shows a police ofﬁcer in a high-visibility vest with the letters "ANC" on the back, standing next to a motorcycle rider. The ofﬁcer is wearing a cap and has a badge on his chest. The motorcycle rider is wearing a helmet and gloves. The background features greenery and a house in the distance.|
|---|

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

|The image shows a vibrant green and orange building with a staircase leading up to a balcony. The building has a door and several lifebuoys hanging on the wall. There are two people sitting on a bench in front of the building, and a bicycle is parked nearby. The text on the building reads "20000".|
|---|

|The image shows a motorcycle parked on a platform with an orange railing. The motorcycle is blue with a black helmet placed on its seat. The background features a green tree and some bushes. The platform appears to be part of a larger structure, possibly a dock or a pier, with a green and orange railing.|
|---|

|The second image shows a portion of the ﬁrst image, focusing on the area with the boat and the green and orange structure. The boat is a large, colorful vessel with a cabin and a deck, and it is positioned on the water. The green and orange structure appears to be a part of the boat, possibly a cabin or a storage area. The background includes some greenery and a few buildings, indicating a coastal or riverine setting.|
|---|

|The second image is a close-up crop from the ﬁrst image, focusing on a section of the boat. The boat has a black hull with an orange deck and a green structure on top. There are two Thai ﬂags on the boat, and a small stream of water is visible coming out of the boat. The water is brown and there are green plants in the foreground.|
|---|

- Figure 15: Examples of pre-training data with natural images. Here each image is labeled with bounding boxes of four salient regions (highlighted by different colors), together with the local captions of each region. The local captions, generated by Qwen2-VL, contains details in the crops although there are still occasional hallucinations.

[Figure 101]

|[Figure 102]|
|---|

|</b><br><b>Primary Date: </b>3/18/1996 12:45:18 PM<br><b>Last Modiﬁed Date: </b>2001-Nov-20|
|---|

|[Figure 103]|
|---|

|RJR0000000507104110|
|---|

|[Figure 104]|
|---|

|<br><b>Sent Date: </b>1996-Mar-18 12:45:18<br><b>Received Date: </b>1996-Mar-18|
|---|

|[Figure 105]|
|---|

|</head>|
|---|

|[Figure 106]|
|---|

[Figure 107]

|dried out in air for at least 48 hours, and rods which have been|
|---|

|[Figure 108]|
|---|

| |
|---|

|257|
|---|

|[Figure 109]|
|---|

|23.27|
|---|

|[Figure 110]|
|---|

|Standardisatior of Drying Procedure for Measurement of EFCOT Rods|
|---|

[Figure 111]

|[Figure 112]|
|---|

|surveillance or monitoring of employees without giving|
|---|

|[Figure 113]|
|---|

|evaluation or for disciplinary perposes.|
|---|

|[Figure 114]|
|---|

|a presidential veto and the lack of votes to override it.|
|---|

|[Figure 115]|
|---|

|months or even years, away. While it won't be enacted|
|---|

- Figure 16: Examples of pre-training data with document images. Here each image is labeled with four bounding boxes (highlighted by different colors), together with the OCR results as the captions of each region.

|[Figure 116]|
|---|

[Figure 117]

[Figure 118]

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

[Figure 122]

[Figure 123]

What is the number of green leaves surrounding the cluster of pink ﬂowers?

What is the name of the university mentioned on the sign?

What type of objects are the two items placed side by side on the brick pathway?

What is the object held by the person's hand?

|[Figure 124]|
|---|

|[Figure 125]|
|---|

[Figure 126]

[Figure 127]

|[Figure 128]|
|---|

|[Figure 129]|
|---|

[Figure 130]

[Figure 131]

What is the name of the city that is prominently displayed on the runner's shirt?

What is the shape of the taillights?

What is installed on the top of the car?

What is the material of the runner's shoes?

|[Figure 132]|
|---|

|[Figure 133]|
|---|

[Figure 134]

[Figure 135]

|[Figure 136]|
|---|

|[Figure 137]|
|---|

[Figure 138]

[Figure 139]

What is the material that the curved handle of the basket is made of?

Which direction is the wave moving?

What is the color of the surfboard?

What type of material is the person's headdress primarily made of?

- Figure 17: Qualitative examples of patch selection on natural images. PS3 is able to locate different parts of the image that are relevant to the question.

[Figure 140]

|[Figure 141]|
|---|

|[Figure 142]|
|---|

[Figure 143]

What month does the "Retail Promotion" activity occur in the period from January to December?

What is the target increase in competitive interaction with Camel from 1996 to 1997?

[Figure 144]

|[Figure 145]|
|---|

|[Figure 146]|
|---|

[Figure 147]

What is the extent to which the confounding of occupational grade and smoking has been examined in this example?

What is the alternative approach to examining the contribution of smoking to mortality that the authors suggest?

|[Figure 148]|
|---|

[Figure 149]

|[Figure 150]|
|---|

[Figure 151]

What is the purpose of the curved ends of the cover plate in the described device?

What is the date of the cigarette ﬁlter feed application?

- Figure 18: Qualitative examples of patch selection on document images. PS3 is able to locate different parts of the text in the document based on the question. For example, in the second example, when asked about the alternative approach of examination or the extent to which the confounding is examined, PS3 is able to locate the texts that discuss these topics.

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|What is the distance to La Cienega Blvd as indicated by the green sign?|
|---|

|Which lane is indicated by the sign showing the number 5?|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

###### 1/2 mile. 1/2 mile.

###### 3/4 mile.

Right lane.

###### Right lane.

Left lane.

Qwen2-VL

GPT-4o

PS3 Qwen2-VL

GPT-4o

PS3

|[Figure 160]<br><br>|
|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|What time is displayed on the stove's digital clock?|
|---|

|What is the text printed at the top of the paper on the table?|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

###### 8:15.

7:45. 7:45.

###### 190967.

190967. 196957.

Qwen2-VL

GPT-4o

###### PS3

Qwen2-VL

GPT-4o

PS3

|[Figure 168]|
|---|

|[Figure 169]|
|---|

[Figure 170]

[Figure 171]

|What is the rank of XEXOR XD as shown in the score display?|
|---|

|What text is displayed on the upper section of the biker patch on the man's vest?|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

VAGA BOND.

###### 2nd.

2nd. 1st.

###### NOMAD.

NOMAD.

Qwen2-VL

GPT-4o

PS3

Qwen2-VL

GPT-4o

PS3

|[Figure 176]<br><br>|
|---|

|[Figure 177]|
|---|

|[Figure 178]<br><br>|
|---|

|[Figure 179]|
|---|

|What number is displayed in the blue notiﬁcation badge on the red icon at the top right corner?|
|---|

|What is the version number of Chrome as indicated in the text?|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

88.0.4324. 150.

87.0.4324. 150.

88.0.4324. 150.

###### 9+.

5+. 9+.

Qwen2-VL

GPT-4o

PS3

Qwen2-VL

GPT-4o

###### PS3

- Figure 19: Qualitative examples on 4KPro. The four rows show examples from categories of autonomous vehicle, household, gaming, and UI understanding, respectively. For each instance, the local crop is shown under the resolutions of 756, 1512, and 3780.

