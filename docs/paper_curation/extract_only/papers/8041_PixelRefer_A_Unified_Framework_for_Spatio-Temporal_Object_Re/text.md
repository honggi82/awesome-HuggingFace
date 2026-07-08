# arXiv:2510.23603v2[cs.CV]1Nov2025

[Figure 1]

[Figure 2]

[Figure 3]

## PixelRefer: A Unified Framework for Spatio-Temporal Object Referring with Arbitrary Granularity

Yuqian Yuan1, Wenqiao Zhang†1, Xin Li2,3, Shihao Wang4, Kehan Li2,3, Wentong Li†, Jun Xiao1, Lei Zhang4, Beng Chin Ooi1

1Zhejiang University 2DAMO Academy, Alibaba Group 3Hupan Lab 4The Hong Kong Polytechnic University

†Corresponding Authors

Multimodal large language models (MLLMs) have demonstrated strong general-purpose capabilities in open-world visual comprehension. However, most existing MLLMs primarily focus on holistic, scenelevel understanding, often overlooking the need for fine-grained, object-centric reasoning. In this paper, we present PixelRefer, a unified region-level MLLM framework that enables advanced fine-grained understanding over user-specified regions across both images and videos. Motivated by the observation that LLM attention predominantly focuses on object-level tokens, we propose a Scale-Adaptive Object Tokenizer (SAOT) to generate compact and semantically rich object representations from free-form regions. Our analysis reveals that global visual tokens contribute mainly in early LLM layers, inspiring the design of PixelRefer-Lite, an efficient variant that employs an Object-Centric Infusion module to pre-fuse global context into object tokens. This yields a lightweight Object-Only Framework that substantially reduces computational cost while maintaining high semantic fidelity. To facilitate fine-grained instruction tuning, we curate PixelRefer-2.2M, a high-quality object-centric instruction dataset. Extensive experiments across a range of benchmarks validate that PixelRefer achieves leading performance with fewer training samples, while PixelRefer-Lite offers competitive accuracy with notable gains in efficiency.

Homepage https://circleradon.github.io/PixelRefer

Demo https://huggingface.co/spaces/lixin4ever/PixelRefer

Code https://github.com/alibaba-damo-academy/PixelRefer

HuggingFace https://huggingface.co/collections/Alibaba-DAMO-Academy/pixelrefer

Date: November 4, 2025

1 Introduction

Multi-modal large language models (MLLMs) [35, 36, 42, 3, 51, 37, 31, 12] have demonstrated impressive general-purpose capabilities in open-world visual comprehension, spanning both static images and dynamic videos. While most existing MLLMs are designed to perform holistic image-level and video-level interpretations and reasoning, they often overlook the need for fine-grained, object-centric understanding, also known as visual referring [23, 39, 69, 74, 67], which aims to precisely interpret and reason the semantics of specific, localized regions within visual scenes. This fine-grained understanding is critical for a wide range of applications that demand accurate object-level comprehension, nuanced event analysis, and reliable predictive reasoning in complex real-world environments, such as human-computer interaction [40], embodied AI [71, 15, 13], medical diagnostics [1, 32, 84, 26, 63] and remote sensing interpretation [82, 55].

Early efforts such as SoM [66] attempt to enhance MLLMs by overlaying visual markers directly onto the input image. However, these markers can sometimes be visually ambiguous, blending into the background or obscuring key content, which limits their effectiveness. Moreover, modifying object markers requires reprocessing the

[Figure 4]

Spatial Temporal

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Image Object Referring Video Object Referring

[Figure 13]

[Figure 14]

Object-levelScene-level

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Temporal QA

What is the ranking of <object> at the end of the video?

[Figure 20]

[Figure 21]

Tiny Object Understanding

Part-level Category

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

It is a blue circular roundabout sign with three white arrows forming a circle, indicating that vehicles must travel around the roundabout in the direction shown.

[Figure 26]

| |
|---|

[Figure 27]

The player was in first place at the end of the video.

[Figure 28]

[Figure 29]

[Figure 30]

Future Prediction

Handle of the cup

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

What is a likely future event of <object> ?

[Figure 36]

Relationship Analysis

[Figure 37]

[Figure 38]

Which is taller? <object0> or <object1> ?

A likely future event is that it will continue to fly away from the grassy area towards the water, possibly landing on the riverbank or in the water.

[Figure 39]

[Figure 40]

[Figure 41]

Retrieval

[Figure 42]

[Figure 43]

<object0> is taller. <object0> is a skyscraper, while <object1> is under construction and not ye completed.

What is the doing in the video?

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

The man was Trump, who stood in the crowd waving his fist.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

General Image Understanding General Video Understanding

[Figure 54]

###### Give me a short introduction of the painting. What is unusual in the video?

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

The unusual aspect of the video is that a large bear and two smaller ones are seated at a wooden table, calmly eating sushi in a tranquil forest setting.

The painting vividly portrays a night scene with a large tree in the foreground and a town behind it. The star-filled sky, rendered in swirling brushstrokes and bright colors reminiscent of Van Gogh’s Starry Night, creates a serene yet striking composition that harmonizes tree, town, and stars.

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

- Figure 1 PixelRefer, a unified region-level MLLM, supports a broad range of tasks at both object-level and scene-level, spanning spatial (images) and temporal (videos) domains. It enables fine-grained spatiotemporal reasoning over user-specified region with arbitrary semantic granularity, while preserving general-purpose capabilities for holistic visual understanding.

entire image, posing challenges for interaction flexibility and inference efficiency. To overcome these limitations, a growing number of region-level MLLMs [80, 70, 19, 67, 9, 7, 22, 78, 64, 73, 87, 48, 6, 59, 75, 18, 13, 33] leverage explicit visual prompts or localized queries to extract object-level visual representations, which are aligned with LLMs to enable accurate spatially localized image understanding. In contrast, the study of spatiotemporal object understanding in dynamic videos remains relatively limited. Some works [61, 68] use bounding boxes as text prompts or rely on external RoI tracking [46], but these often yield coarse and temporally inconsistent representations in complex video scenarios.

Recently, research has increasingly shifted toward a unified region-level understanding across both images and videos, aiming to support fine-grained spatiotemporal understanding. For instance, the Describe Anything Model (DAM) [28] introduces a focal prompt mechanism to encode user-specified regions and employed a localized vision backbone that integrates global image context into regional representations via gated crossattention. While DAM effectively captures finer details, its architecture is inherently limited to describing a single object at a time, requiring repeated image encoding for multiple regions and thereby incurring substantial computational overhead. The Perception Anything Model (PAM) [34] extends SAM 2 [50] by incorporating a learnable semantic perceiver as the interface between vision backbone and LLM. Despite showing promising results, PAM remains largely constrained to captioning tasks, limiting its ability to handle more complex reasoning (e.g., object-level QA, multi-object understanding). Besides, its reliance on semantics-agnostic SAM 2 features necessitates large-scale training data (e.g. 8M samples) to achieve sufficient alignment with the LLM. More critically, the task-specific architectures of both DAM and PAM undermine the inherent general-purpose capabilities of MLLMs, hindering their flexibility and scalability.

In this work, we revisit the design of a flexible and unified region-level MLLM for fine-grained spatiotemporal object understanding in both images and videos. Unlike prior methods focused primarily on single-object captioning, we advocate a framework that supports a broad range of object-centric referring tasks, while preserving the general-purpose capabilities of modern MLLMs. To this end, we argue that such a framework should be built upon a general-purpose MLLM backbone, with region-level object representations integrated in a modular and flexible manner, enabling seamless interaction with the base model without compromising its versatility. A preliminary version of this framework was introduced in our prior work [72], where we demonstrated its effectiveness for video-based scenarios. In this paper, we present PixelRefer, a unified regionlevel MLLM that enables advanced fine-grained understanding over user-specified regions. As illustrated in

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

14.9 2.3

SPHINX-V

Image Video

8.0

PAM DAM Ours

1.5 1.4

[Figure 68]

Description

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

QA

0.8

[Figure 73]

|[Figure 74]|
|---|

[Figure 75]

|[Figure 76]|
|---|

[Figure 77]

[Figure 78]

# Training samples (M)

[Figure 79]

|[Figure 80]|
|---|

[Figure 81]

|[Figure 82]|
|---|

(b)

|[Figure 83]|
|---|

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

MemoryUse(GB)

Runtime(ms/item)

[Figure 88]

[Figure 89]

[Figure 90]

Ours-7B Ours-2B Ours-Lite

Ours-Lite -2B

DAM -3B

PAM -3B

-7B

(c)

(a)

- Figure 2 Quantitative Evaluation and Efficiency Analysis. (a) Performance Comparison: PixelRefer and PixelReferLite consistently outperform state-of-the-art object-level MLLMs across diverse image (LVIS [70], PACO [70], DLC-Bench [28]) and video (VideoRefer-Bench, HC-STVG [58]) benchmarks. (b) Data Efficiency: Our method achieves leading performance with fewer training samples compared to existing methods. (c) Runtime and Memory Efficiency: PixelRefer-Lite notably reduces inference time and memory usage, clearly demonstrating its efficiency.

Fig. 1, PixelRefer supports a variety of perception and reasoning tasks across spatial and temporal dimensions, ranging from object-level to scene-level comprehension. Prior to detailing the full model design, we conduct an in-depth preliminary analysis of how our initial framework [72] interprets object-level representations, offering insights that inform the core design of our PixelRefer.

First, our empirical finding reveals that the LLM’s attention is predominantly focused on the region-level tokens corresponding to the referred objects. This highlights the critical role of object tokens quality in determining model performance on object-centric tasks. Motivated by this observation, we introduce the Scale-Adaptive Object Tokenizer (SAOT), a novel object-level tokenizer designed to generate precise and semantically rich region representations. It leverages a unified pixel-level mask representation to support arbitrary free-form regions, dynamically adapts to varying object scales, and preserving spatial context, producing compact yet informative object tokens. SAOT is architecture-agnostic and can be seamlessly incorporated into general-purpose MLLM with minimal modifications.

Then, our second empirical finding examines the interaction between global visual tokens and object tokens within the LLM. We observe that attention to global visual tokens is predominantly concentrated in the early layers, while object tokens remain active throughout the LLM. However, the global visual tokens contribute excessively to the LLM’s overall computational overhead, as also noted in prior studies [10, 27]. These insights motivate the design of PixelRefer-Lite, an efficient variant of our method based on an Object-Only Framework. Specially, we introduce a lightweight Object-Centric Infusion (OCI) module, which pre-fuses global visual context into object tokens via a hierarchical cross-attention mechanism. By retaining only the fused object tokens as input to the LLM, our approach achieves substantial reductions in computational cost while preserving high semantic fidelity.

We further curate a new open-source dataset, PixelRefer-2.2M, structured into two categories: Foundational Object Perception and Visual Instruction Tuning, to support fine-grained alignment between language and both global visual context and local object regions. Extensive experiments are conducted across a wide range of object-centric tasks with varying semantic granularity, including image-level benchmarks such as Category Recognition [70], Phrase-level and Detailed Caption [8, 24, 28], and Reasoning Questions [67], as well as video-level benchmarks including VideoRefer-BenchD [72], VideoRefer-BenchQ [72], and HC-STVG [58]. As shown in Fig. 2-(a)&(b), our approach consistently achieves state-of-the-art performance, despite being trained on fewer instruction samples than prior advanced counterparts [33, 28, 34], clearly demonstrating both its effectiveness and data efficiency. Notably, PixelRefer-Lite delivers competitive accuracy while offering substantial improvements in runtime and memory consumption (Fig. 2-(c)), highlighting its practicality for real-world applications.

[Figure 91]

[Figure 92]

[Figure 93]

Layer1 Layer7 Layer14 Layer28

|sys| | | | | |
|---|---|---|---|---|---|
|vision| | | | | |
|text| | | | | |
|object| | | | | |
|ans| | | | | |
| |sys|vision|text|object|ans|

| | | | | |sys|[Figure 94]<br><br>[Figure 95]| | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |vision| | | | | |
| | | | | |text| | | | | |
| | | | | |object| | | | | |
| | | | | |ans| | | | | |
|sys|vision|text|object|ans| |sys|vision|text|object|ans|

|sys| | | | | |
|---|---|---|---|---|---|
|vision| | | | | |
|text| | | | | |
|object| | | | | |
|ans| | | | | |
| |sys|vision|text|object|ans|

|sys|
|---|
|vision|
|text|
|object|
|ans|

textsysvisionobjectans

textsysvisionobjectans

textsysvisionansobject

textsysvisionobjectans

- Figure 3 Visualization of attention maps across different layers (Layer 1, 7, 14 and 28) of the LLM. The input sequence includes system tokens (sys), global image token (vision), text prompts (text), object-level tokens (object), and answer tokens (ans). For clarity, image tokens are average pooled by a factor of 8. The figure showcases how attention patterns evolve across layers over different tokens. 2 Related Work

- 2.1 Multimodal Large Language Models

Large language models (LLMs) have significantly advanced the field of artificial intelligence by proving their capabilities to tackle diverse tasks related to language comprehension and generation [41]. To leverage the potential of LLMs for visual understanding, recent research has focused on multimodal LLMs (MLLMs) [35, 36, 42, 3, 37, 31, 62, 4, 76, 83], which integrate vision and language into a unified representation space. Evolving from image-based MLLMs, recent efforts have explored Video Large Language Models (Video LLMs) [14, 29, 77, 38, 85], aiming to extend multimodal reasoning to dynamic spatiotemporal contexts. Most Video LLMs empoly pre-trained visual encoders to extract frame-wise or clip-level features, which are then interleaved with textual tokens and processed by LLMs to generate responses [57]. While these models have shown promising progress, they fall short in supporting fine-grained spatial and temporal reasoning, especially in object-centric tasks.

- 2.2 Region-level Multimodal Large Language Models

Unlike traditional MLLMs that emphasize holistic understanding, region-level MLLMs aim for fine-grained, object-centric reasoning. Early method like SoM [66] enhances MLLMs by overlaying visual markers onto the image to indicate object locations, but suffers from ambiguity and limited flexibility. Recent regionlevel MLLMs [80, 70, 19, 67, 9, 7, 22, 78, 64, 73, 87, 48, 6, 59, 75, 18] introduce explicit visual prompts or region-based queries to generate instance-level representations, improving localized region understanding. For videos, several works [68, 61, 46] adopt sparse temporal sampling and coarse object-level references, lacking support for multi-object interactions and temporal coherence. To address this, VideoRefer [72], introduces a simple yet effective architecture for fine-grained region-level video understanding, supported by large-scale video instruction data and comprehensive benchmarks. More recently, DAM [28] employs a focal prompt mechanism and localized vision backbone to enable image and video captioning. PAM [34] extends SAM 2 [50] by introducing a semantic perceiver that bridges the visual backbone and LLM, leveraging intermediate SAM

- 2 features for enhanced region-level understanding. Despite their promising results, these models remains largely constrained to captioning tasks and fall short in more complex reasoning scenarios. Moreover, these task-specific architectures compromise the general-purpose nature of MLLMs.

- 2.3 Benchmarks and Datasets for Region-level MLLMs

Benchmarks. Prior works [19, 22, 48, 68, 46] typically assess region-level captioning using traditional languagebased metrics [2, 5, 30, 45, 60]. These metrics often measure surface-level textual similarity, fail to reflect factual correctness or fine-grained semantic alignment. To address this, recent studies have explored more semantically grounded evaluations. Osprey [70] utilizes Sentence-BERT [52] to compute sentence-level semantic similarity, along with a semantic IoU metric for word-level alignment. Ferret-Bench [67] leverages GPT-4 [44]

to score the alignment between predictions and reference captions. DLC-Bench [28] further eliminates the reliance on reference captions by scoring model outputs against predefined sets of positive and negative attributes for each region. Nevertheless, these benchmarks largely focus on object-level captioning, leaving a notable gap in evaluating spatiotemporal understanding, particularly for complex reasoning in dynamic video scenarios.

Datasets. While several region-level instruction-tuning datasets exist across images [70, 19, 33, 13] and videos [28, 34], they predominantly support single-object captions. This limits their suitability for higherorder visual reasoning tasks like multi-object relationship, and multi-turn QA in human-centric, real-world interactions.

#### 3 How Do MLLMs Understand Object Tokens?

In this section, we conduct an in-depth investigation into how MLLMs interpret and utilize object-level tokens. Given the complexity of this topic, our preliminary analysis focuses on a Vision-Object Framework, and examines the role of object tokens within the LLM through attention patterns.

- 3.1 Vision-Object Framework

As shown in Fig. 5-(a), the Vision-Object Framework comprises four components: a vision encoder, an object tokenizer, a text tokenizer, and an instruction-following LLM.

Given a video1 V ∈ RN×H×W×C, where N, H, W, C denote the frame number, height, width and channels, respectively. The vision encoder Ev encodes the input and extracts a feature map Z, which encodes spatialtemporal scene-level information as a sequence of visual tokens TZ. To focus object-centric semantics, we define a set of user-specified region R = {R1,R2,...,Rn}, where n is the number of target objects. Notably, when n = 0, the framework naturally degenerates to a general visual understanding task. Each object is represented as a collection Rj = {mij|i ∈ T}, where mij ∈ M denotes the binary mask corresponding to a free-form region of interest, and T being a set containing one or multiple timestamps. The object tokenizer ER generates enriched object-level representations from Z, yielding object tokens TR = ER(R,Z). Finally, the visual tokens TZ, object-level tokens TR, and linguistic tokens TX are jointly fed into the LLM to generate fine-grained semantic understanding Y. Formally, this process is formulated as:

Y = Φ(TZ,TR,TX), (1) where Φ denotes the autoregressive decoding function of the LLM.

The Vision-Object Framework thus enables flexible integration of global scene context, localized object semantics, and linguistic instructions, supporting both fine-grained scene-level and object-level understanding across spatial and temporal dimensions.

- 3.2 Preliminary Analyses

To gain deeper insights into how object tokens are utilized within the model, we conduct an empirical analysis of attention distributions across LLM layers, spanning from shallow to deep layers. Representative visualizations are presented in Fig. 3. Some key patterns can be observed:

Answer tokens prioritize object tokens. Across all layers, from shallow to deep layers, answer tokens consistently exhibit stronger attention toward object tokens than toward global visual tokens. This pattern indicates that object tokens serve as the primary semantic anchor for question answering. Much like human cognition, where specific objects often provide more informative cues than holistic scenes. Consequently, region-level MLLMs rely on well-structured object representations to generate accurate answers. This finding highlights the pivotal role of high-quality object tokenization in enabling precise and context-aware object-centric understanding.

1As images are viewed as single-frame videos, we do not explicitly differentiate between images and videos throughout this work.

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

### LLM

### LLM

[Figure 102]

[Figure 103]

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

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

##### Vision Encoder

[Figure 120]

Infusion Module

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Object Tokenizer

Vision Encoder

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Object Tokenizer

[Figure 134]

[Figure 135]

(a) (b)

###### Figure 5 Frameworks of two complementary paradigms for region-level representations in our approach: (a) illustrates Vision-Object Framework, while (b) presents Object-Only Framework.

[Figure 136]

[Figure 137]

Answer-to-image token attention is sparse. In contrast to object tokens, answer tokens exhibit sparse attention to global image tokens, often manifesting as strip-like patterns in the attention maps. This suggests that LLMs selectively attend to only a small subset of image tokens deemed relevant to the current query. To assess the interpretability of this selection, we further visualize answer-to-image attention distributions across different queried regions (Fig. 4), revealing that the model adaptively highlights the corresponding semantically aligned object areas, while occasionally incorporating contextual cues from broader spatial context (e.g., background or surrounding objects). These findings indicate that global image tokens serve as auxiliary references, complementing the stronger semantic guidance provided by object tokens.

[Figure 138]

[Figure 139]

[Figure 140]

Raw image Please describe . Please describe .

[Figure 141]

Figure 4 Visualization of answer-to-image attention heatmaps for different query regions. The model adaptively focuses on relevant objects while incorporating contextual cues from the surrounding areas.

Early fusion of object and image tokens. As illustrated in Fig. 3, the early layers of the LLM exhibit broad mutual interaction between object tokens and global image tokens, with attention patterns densely spanning the entire visual token space. However, as depth increases, the attention gradually shifts, concentrating more heavily on object tokens. This shift suggests that object tokens serve as compact, information-rich summaries of the relevant visual content, thereby reducing the reliance on raw image tokens in deeper reasoning stages of the LLM. This progression from distributed to focused attention reflects a hierarchical processing strategy: the early layers facilitate comprehensive visual integration by combining both object-specific and global scene features, whereas the later layers selectively retain task-relevant object-level semantics to support accurate answer generation.

#### 4 Methodology

- 4.1 Overview

Motivated by the insights from our attention analysis, we propose two complementary paradigms for objectcentric understanding in the our approach. The first paradigm, PixleRefer, builds upon the Vision-Object Framework (Fig. 5-(a)), combining both global visual tokens with object-level tokens. This design enables the model to reason over holistic scene context while leveraging well-constructed object-level representations for more precise and comprehensive semantic understanding. The second paradigm, PixelRefer-Lite, based on the Object-Only Framework (Fig. 5-(b)), introduces a lightweight infusion module that pre-integrates global image context into object tokens prior to LLM processing. By retaining only object tokens for subsequent LLM decoding, this design substantially reduces computational cost while preserving strong discriminative

[Figure 142]

[Figure 143]

[Figure 144]

###### Scale-Adaptive Object Tokenizer

[Figure 145]

x x x

x

[Figure 146]

| |
|---|

DynamicObjectProcessing

x x

x x

MaskFeatureExtractor

MLP

###### Flatten

VisionEncoder

x x x

x

Image Feat. & Mask

X n

Small object Expand Object Token

[Figure 147]

Mask Feature

x

x

[Figure 148]

[Figure 149]

[Figure 150]

AbundantFeature Aggregation

[Figure 151]

[Figure 152]

x x

x

###### Flatten

x

Image Feat. & Mask

Crop Object Token

Large object Expand Mask Feature

Dynamic Object Processing Abundant Feature Aggregation

x x

x x x

x

Pooling

Resize Resize x

MLP

Mask Feature Cluster Aggregation

- Figure 6 Architecture of our proposed Scale-Adaptive Object Tokenizer. For an input image and a given object, we first perform Dynamic Object Processing to adaptively scale the objects. Subsequently, vision features are extracted from the cropped and expanded sections of the image. To address redundancy prevalent in large objects, we further introduce Abundant Feature Aggregation for efficient feature integration.

capability.

- 4.2 PixelRefer

As outlined in Sec. 3.1, the Vision-Object Framework consists of four components: a vision encoder, an object tokenizer, a text tokenizer, and an LLM. Building on the insights from Sec. 3.2, our finding highlights the critical role of high-quality object tokenization for enabling precise and context-aware object-centric understanding. To this end, the PixelRefer framework introduces a novel Scale-Adaptive Object Tokenizer, which employs a dynamic processing strategy to adaptively process objects of varying sizes and shapes. This design ensures robust and consistent object-level embeddings across diverse visual inputs.

- 4.2.1 Scale-Adaptive Object Tokenizer

In this section, we propose a Scale-Adaptive Object Tokenizer (SAOT) designed to generate accurate and informative object tokens across different scales. Unlike prior approaches that rely on a naive RoI Pooling-based or Mask Pooling-based strategy to encode each region [80, 70, 72, 19, 48, 18, 81], our method addresses the common issue of unreliable feature extraction from extremely small or scale-variant regions without comprising fine-grained low-level cues. As illustrated in Fig. 6, some regions are relatively small, after patchification, may occupy less than a single token, making it difficult to extract reliable features. In contrast, our SAOT dynamically adjusts region scale, preserves spatial context, and aggregates redundant features, thereby yielding object tokens that are both accurate and semantically informative.

Given an input image I ∈ R3×H

I×WI and a target region represented by a binary mask M, we perform Dynamic Object Processing to handle scale variations across objects. Specifically, we extract the bounding box BR = (xb,yb,wb,hb) corresponding to the region of interest and adaptively compute a scaling ratio s for region enlargement or reduction:



Ω · 100 |M|

, if |M| > 100 · Ω



(2)

Ω · n |M|

s =

, elseif |M| < n · Ω



1, otherwise

Token Similarity Matrix (50token)

[Figure 153]

###### Distribution of Token Pair Similarities (100token)

###### Distribution of Token Pair Similarities (50token)

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Numberoftokenpairs

[Figure 158]

[Figure 159]

Numberoftokenpairs

Similarity

Similarity

- Figure 7 Cosine similarity analysis of object tokens. Left: Pairwise similarity matrix of 50 object tokens randomly

sampled from a single object. Right: Histograms of pairwise similarities before (FMP) and after (FMP′) Abundant Feature Aggregation for objects with 50 tokens (top) and 100 tokens (bottom), respectively. Aggregated tokens show reduced intra-object similarity, indicating effective redundancy reduction and improved representational compactness.

where |M| = i,j 1(Mi,j = 1) denotes the number of foreground pixels in the mask, and Ω = patchh×patchw is the patch size of the vision encoder. Here, n denotes the number of tokens assigned to each target object. Once the scaling ratio s is determined, small objects (|M| < n · Ω) are upscaled by s to retain fine-grained details, wherea large objects (|M| > 100 · Ω) are downscaled by s to reduce redundancy and computational overhead. This scale-adaptive strategy effectively normalizes region sizes across varying object scales, ensuring that both small and large objects are encoded with high fidelity while maintaining computational efficiency.

We subsequently apply contextual padding to enlarge the cropped bounding box of the target object, yielding an expanded region IB ∈ R3×h

b×wb′ . This padded region is then fed into the shared vision encoder to obtain region-level embeddings FR, which are enriched with contextual information. To isolate object-specific features, we introduce a Mask Feature Extractor. Specially, we extract the masked spatial features FM ∈ Rn×D

′

I by applying the binary mask M to the feature map FR:

FM = FR ⊙ M. (3)

Since contextual padding disrupts the original spatial alignment of the object within the global image, we introduce relative positional encoding to alleviate localization ambiguity:

 

- p(0)i,j = (j/wb′) · wb + xb /(WI − 1),
- p(1)i,j = (i/h′b) · hb + yb /(HI − 1),

(4)



where 0 ≤ i < h′b, 0 ≤ j < wb′. These coordinates are projected through a linear layer and fused with the masked features to form position-aware object tokens:

FMP = (FM + Linear(pi,j))[M = 1]. (5)

As visualized in Fig. 7, we observe that the resulting object tokens often exhibit high intra-object similarity, particularly in large or homogeneous regions. To further mitigate redundancy, we propose an Abundant Feature Aggregation strategy. Specifically, we employ k-means clustering to merge redundant tokens: initial n centroids are randomly selected, clustering proceeds for k iterations, and the mean embedding of each cluster Ci is preserved, resulting in n representative object tokens:

1 |Ci| j∈C

, ∀i ∈ {1,...,n}. (6)

FMP′ =

###### FMP

j

i

Finally, an MLP is applied to generate the final tokens representations of each target object.

- 4.3 PixelRefer-Lite

As revealed in our preliminary analysis (Sec. 3.2), attention to global visual tokens is primarily concentrated in the early layers of LLM, while object-level tokens maintain strong activation throughout all layers. This

Algorithm 1 Object-Centric Infusion Module

Input: Object tokens TR ∈ Rn×D; Raw image I ∈ R3×HI×WI ; Object mask m ∈ RHI×WI Output: Fused object tokens TO ∈ Rn×D

- 1: Fl ← Ev Resize(LocalCrop(I, m)) ▷ Extract local features
- 2: TˆR ← LN(TR)
- 3: Tl ← TR + Local-to-Object Attn(TˆR, Fl, Fl)
- 4: ▷ Pre-Norm + Residual
- 5: Fg ← Ev Resize(I) ▷ Extract global features
- 6: Tˆl ← LN(Tl)
- 7: TO ← Tl + Global-to-Object Attn(Tˆl, Fg, Fg)
- 8: ▷ Pre-Norm + Residual

Return TO

observation suggests that semantic fusion between global scene context and object-centric representations is largely completed at shallow layers. However, global visual tokens still comprise the majority of the input sequence of LLM, especially for high-resolution and long video inputs, resulting in significant computational overhead. This inefficiency has also been highlighted in prior studies [27, 10] as a major bottleneck in MLLMs. To improve the overall efficacy of our approach, we introduce PixelRefer-Lite, an efficient variant of our method based on the Object-Only Framework.

Object-Only Framework. The architecture of the Object-Only Framework is illustrated in Fig. 5-(b). In contrast to the Vision-Object Framework, which directly concatenates both global visual and object-level tokens as input to the LLM, this framework incorporates a lightweight infusion module to streamline visual processing. Specially, the infusion module integrates global visual context TZ into object tokens TR, enabling each object to be enriched with global contextual cues. This design significantly reduces the total number of vision tokens passed to the LLM while preserving critical semantic content. Formally, the infusion module is defined as:

TO = Ψ(TR,TZ), (7)

where Ψ denotes the infusion function. Here, TO represents the enhanced object tokens that are integrated with scene-level visual context. Subsequently, the refined object tokens TO are concatenated with linguistic tokens TX and fed into the LLM for decoding, yielding precise context-aware object-level semantic understanding:

Y = Φ(TO,TX), (8)

where Φ denotes the LLM’s decoding function. By eliminating the need to retain dense global visual tokens, the Object-Only Framework offers a token-efficient yet semantically rich alternative, particularly well-suited for processing high-resolution images or long video sequences.

Object-CentricInfusionModule. Within our PixelRefer-Lite framework, we introduce an Object-Centric Infusion (OCI) module designed to hierarchically integrates contextual visual information into object tokens, thereby enhancing their semantic representations. To model broader contextual understanding based on long-range dependencies, OCI module adopts a two-step cross-attention infusion strategy that progressively incorporates local and global visual context into the object tokens. In the first step, Local-to-Object Attention, fine-grained visual embeddings extracted from locally expanded image regions are injected into the object tokens. This operation enables refined tokens to capture detailed local context from the object’s immediate surroundings, preserving fidelity to object’s original appearance while becoming more robust against occlusion or noise. In the second step, Global-to-Object Attention, object tokens are further conditioned on scene-level embeddings derived from the raw image. This global integration introduces long-range dependencies and holistic semantics, complementing the previously injected local details and enabling more scene-aware object understanding. The detailed processing flow of the proposed OCI module is presented in Algorithm 1. This hierarchical injection mechanism mirrors human perception, where object recognition is progressively refined by situating local detail within its broader scene context. For implementation, we adopt standard attention operations for cross-attention, enabling direct compatibility with recent advances in efficient attention kernels [16].

Extension to Videos. Given that a video can be regarded as a sequence of images across different timestamps, the Object-Only Framework naturally generalizes to the video domain by processing temporally ordered sequences of object tokens, each associated with its respective frame. To incorporate temporal information,

###### JOURNAL OF LATEX CLASS FILES, VOL. A, NO. B, AUGUST 2025 12

###### Image Detailed Caption (47.9%)

VideoRefer-Detailed (125k)

HC-STVG (10k)

DAM-SAM (592k)

###### Region Recognition (25.9%)

DAM-OpenImages (64k)

DAM-LVIS (90k)

VG (100k)

DAM-COCOStuff (28k)

VideoRefer-Short (100k)

DAM-Mapillary (17k)

LVIS (99k)

DAM-PACO (24k)

PACO (82k)

Osprey (29k)

PascalPart (11k)

MDVP-Data (15k)

PartImagenet (12k)

Refcoco (17k)

###### Video Detailed Caption (26.2%)

Refcoco+ (17k) Refcocog (22k) MEVIS (4k)

DAM-SAV (37k)

- Fig. 12. Data used for Foundational Object Perception training (Stage1). Left: category distribution (outer ring: all categories; inner ring: subsets). Right: detailed dataset breakdown (two columns).

[Figure 160]

Visual Instruction Tuning 0.8M

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

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

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Image Region QA (24.1%)

VCR (118k)

Osprey-QA (68k) MDVP-LVIS (7k) MDVP-PACO (5k)

Video Object QA (9.1%)

VideoRefer-QA (75k)

Region-level Caption (30.4%)

Caption Sampled from Stage1 (250k)

General QA (36.5%)

Video QA (173k)

Text QA (81k)

Image QA (46k)

- Fig. 13. Data used for Visual Instruction Tuning training (Stage2). Left: data distribution (outer ring: categories; inner ring: subsets). Right: detailed dataset breakdown (two columns).

JOURNAL OF LATEX CLASS FILES, VOL. A, NO. B, AUGUST 2025 12

JOURNAL OF LATEX CLASS FILES, VOL. A, NO. B, AUGUST 2025 12

[Figure 209]

Image Detailed Caption (47.9%)

VideoRefer-Detailed (125k)

[Figure 210]

Image Detailed Caption (47.9%)

VideoRefer-Detailed (125k)

[Figure 211]

[Figure 212]

[Figure 213]

HC-STVG (10k)

[Figure 214]

[Figure 215]

[Figure 216]

HC-STVG (10k)

[Figure 217]

[Figure 218]

DAM-SAM (592k)

[Figure 219]

###### Region Recognition (25.9%)

[Figure 220]

[Figure 221]

DAM-SAM (592k)

[Figure 222]

###### Region Recognition (25.9%)

DAM-OpenImages (64k)

[Figure 223]

[Figure 224]

DAM-OpenImages (64k)

[Figure 225]

[Figure 226]

[Figure 227]

DAM-LVIS (90k)

[Figure 228]

[Figure 229]

DAM-LVIS (90k)

VG (100k)

Object Perception 1.4M

[Figure 230]

[Figure 231]

[Figure 232]

VG (100k)

DAM-COCOStuff (28k)

[Figure 233]

DAM-COCOStuff (28k)

VideoRefer-Short (100k)

[Figure 234]

VideoRefer-Short (100k)

DAM-Mapillary (17k)

[Figure 235]

DAM-Mapillary (17k)

LVIS (99k)

[Figure 236]

[Figure 237]

LVIS (99k)

[Figure 238]

DAM-PACO (24k)

DAM-PACO (24k)

[Figure 239]

PACO (82k)

PACO (82k)

Osprey (29k)

[Figure 240]

Osprey (29k)

[Figure 241]

PascalPart (11k)

[Figure 242]

PascalPart (11k)

[Figure 243]

[Figure 244]

MDVP-Data (15k)

[Figure 245]

MDVP-Data (15k)

[Figure 246]

PartImagenet (12k)

[Figure 247]

[Figure 248]

PartImagenet (12k)

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Refcoco (17k)

Refcoco (17k)

###### Video Detailed Caption (26.2%)

###### Video Detailed Caption (26.2%)

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

Refcoco+ (17k) Refcocog (22k) MEVIS (4k)

Refcoco+ (17k) Refcocog (22k) MEVIS (4k)

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

DAM-SAV (37k)

DAM-SAV (37k)

[Figure 265]

[Figure 266]

- Fig. 12. Data used for Foundational Object Perception training (Stage1). Left: category distribution (outer ring: all categories; inner ring: subsets). Right: detailed dataset breakdown (two columns).

Image Region QA (24.1%)

VCR (118k)

Osprey-QA (68k) MDVP-LVIS (7k) MDVP-PACO (5k)

Video Object QA (9.1%)

VideoRefer-QA (75k)

Region-level Caption (30.4%)

Caption Sampled from Stage1 (250k)

General QA (36.5%)

Video QA (173k)

Text QA (81k)

Image QA (46k)

- Fig. 13. Data used for Visual Instruction Tuning training (Stage2). Left: data distribution (outer ring: categories; inner ring: subsets). Right: detailed dataset breakdown (two columns).

- Fig. 12. Data used for Foundational Object Perception training (Stage1). Left: category distribution (outer ring: all categories; inner ring: subsets). Right: detailed dataset breakdown (two columns).

Image Region QA (24.1%)

VCR (118k)

Osprey-QA (68k) MDVP-LVIS (7k) MDVP-PACO (5k)

Video Object QA (9.1%)

VideoRefer-QA (75k)

Region-level Caption (30.4%)

Caption Sampled from Stage1 (250k)

General QA (36.5%)

Video QA (173k)

Text QA (81k)

Image QA (46k)

- Fig. 13. Data used for Visual Instruction Tuning training (Stage2). Left: data distribution (outer ring: categories; inner ring: subsets). Right: detailed dataset breakdown (two columns).

- Figure 8 Overview of datasets used for model training. Left: Data distribution for Foundational Object Perception training (1.4M samples). Right: Data used for Visual Instruction Tuning (0.8M samples).

model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024. 9

we prepend timestamps embeddings to each object token, enabling the model to distinguish objects across different frames. Each object mask is independently processed through the proposed OCI module. For object mask extraction, we employ SAM 2 [50] to generate high-quality segmentation masks on sampled video frames.

[69] L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang, “An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models,” in ECCV. Springer, 2024, pp. 19–35. 8

- 5 Dataset

With the growing demand in fine-grained, pixel-level object understanding, recent studies have developed instruction-tuning datasets to advance this task, including image-level data [70, 19, 33] and video-level data [28, 34]. However, most existing datasets remain limited to single-object semantic captioning for objectlevel recognition, especially in video scenes, which falls short in supporting complex visual reasoning required for human-centric video interactions in real-world scenarios. To address this gap, we introduce VideoRefer-700K in our prior work [72], a meticulously curated large-scale region-text video instruction dataset. It features region-level descriptions and multi-turn question–answer (QA) pairs spanning basic inquiries, compositional reasoning, and future event prediction. Beyond this, we further carefully collect diverse open-source image-level and video-level datasets and systematically organize them into two categories: Foundational Object Perception and Visual Instruction Tuning, thereby providing a robust knowledge foundation for model fine-tuning.

model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024. 9

model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024. 9

[69] L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang, “An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models,” in ECCV. Springer, 2024, pp. 19–35. 8

[69] L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang, “An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models,” in ECCV. Springer, 2024, pp. 19–35. 8

- 5.1 Data Collection

We curate a collection of open-source datasets and systematically organize them into two categories: Foundational Object Perception and Visual Instruction Tuning.

- 5.1.1 Foundational Object Perception Data

While pre-trained LLMs and Vision Transformers (ViTs) encode rich general priors about the world, they lack precision at the regional level. To address this limitation, we first strengthen fine-grained regional alignment through carefully curated supervision before advancing to instruction tuning. Figure 8 illustrates the composition of data in this stage, totaling 1.4M samples across three complementary categories.

Region Recognition. A critical foundation for high-quality visual understanding lies in region recognition. To this end, we curate a multi-scale cognitive dataset encompassing objects, parts, and temporal dynamics. Object-level annotations from LVIS [20], Visual Genome [24], and RefCOCO/RefCOCO+ [69] are strategically combined with fine-grained part-level datasets (PACO [47], Pascal-Part [11], PartImageNet [21]), enabling hierarchical learning from whole objects to constituent parts. Furthermore, temporal alignment is introduced through VideoRefer-Short captions and MEVIS [17], bridging static region understanding with dynamic scene perception.

Regional Image Detailed Caption. For region-level image captioning, we integrate diverse domain data sources to enhance descriptive richness. Specifically, we aggregate DAM [28] samples constructed from SAM, OpenImages, LVIS, COCOStuff, Mapillary, and PACO, ensuring broad coverage of object categories and contexts. To further expand descriptive variety, we incorporate Osprey-caption and MDVP-Data, which contribute more fine-grained narrative supervision.

###### Regional Video Detailed Caption. Compared with image data, detailed video captions remain scarce yet essential for modeling temporal and contextual understanding. To fill this gap, we leverage three complementary resources: our self-constructed VideoRefer-Detailed captions, DAM-SAV [28] dataset, and HC-STVG [58].

- 5.1.2 Visual Instruction Tuning Data

Visual instruction tuning aims to endow LMMs with the ability to understand, follow, and respond to natural-language instructions grounded in visual inputs. Achieving this requires supervision that is not only large in scale but also rich in instruction diversity and reasoning depth. To this end, we curate four complementary types of data, as illustrated in Figure 8, totaling 0.8M samples.

Image Region QA. To strengthen localized reasoning capabilities at the region level, we incorporate OspreyQA [70], MDVP-LVIS [33], and MDVP-PACO [33], all annotated using GPT-4/GPT-4o to ensure high-quality, instruction-rich supervision. In addition, we include VCR [74] data, which extends beyond recognition toward commonsense reasoning over visual contexts, pushing the model to infer intent, causality, and social cues from images.

Video Object QA. Compared with images, object-centric QA in videos remains limited. To address this gap, we constructed 75K VideoRefer-QA samples using a multi-agent pipeline [72], enabling precise question–answer supervision over dynamic objects in temporal sequences. This enriches the model’s capacity to reason about object identity, attributes, and interactions across time, an essential skill for video understanding.

Region Caption. To consolidate the model’s ability for regional captioning, we sample 250K captions from the data in Foundational Object Perception. This prompts continuity between foundational perception pretraining and instruction tuning, aligning descriptive generation with QA-based reasoning. By jointly incorporating caption-style and QA-style supervision, we balance generative expressiveness and discriminative reasoning.

General QA. To broaden the model’s instruction-following capabilities, we introduce general visual QA beyond region-specific tasks. These are sampled from sources such as LLaVA-Video [86] and LLaVA-OV [25], providing a wide range of open-ended queries. This component equips the model with the flexibility to handle heterogeneous instruction types, ranging from factual recognition to complex visual reasoning.

#### 6 Experiment

- 6.1 Experiment Setup

Implementation Details. Our base model is built upon VideoLLaMA 3 [76], a robust unified architecture capable of understanding both images and videos. Its vision encoder processes images with dynamic resolutions using Rotary Position Embedding (RoPE) [56], enabling fine-grained perception of small regions compared to strategies based on fixed input sizes. For video inputs, the model efficiently reduces the number of vision tokens by leveraging their similarity, resulting in more precise and compact video representations. We initialize our model with the VideoLLaMA 3-Image weights. Benefiting from large volume of high-quality vision-text pre-training data, VideoLLaMA 3 exhibits strong and resilient vision understanding capabilities. Additionally, we adopt a progressive training strategy consisting of two stages: foundational object perception training (Stage 1) and visual instruction tuning (Stage 2). The global batch size is configured to 256, and each stage is trained for one epoch. A cosine learning rate scheduler is employed across all stages, with a warm-up ratio of 0.03 applied to the learning rate. In Stage 1, the learning rates are configured as follows: 1 × 10−5 for the LLM and projector, and 1 × 10−3 for the object encoder. In Stage 2, the learning rate for all parameters is uniformly set to 1 × 10−5. The number of object tokens n is set to 32 in the main experiments. For efficiency analysis, experiments are carried out using one NVIDIA A100 80GB GPU.

Efficiency Metrics. To comprehensively assess model efficiency, we adopt three key metrics: FLOPs, GPU memory usage and inference time, providing a holistic view of computational complexity. For the calculation of FLOPs, as in FastV [10], we calculate the FLOPs for the multi-head attention and the feed-forward network (FFN) modules as 4nd2 + 2n2d + 2ndm. Here, n denotes the token count, d is the hidden state size, and m represents FFN’s intermediate dimension. Considering the projection in k and v is not equal to q, and there are three linear layer in FFN of Qwen-style LLM, the FLOPs is modified as 2nd2 + 2nddkv + 2n2d + 3ndm.

###### Table 1 Performance on image-level region understanding benchmarks: including category-level (LVIS and PACO), detailed captioning (DLC-Bench and Ref-L4 [CLAIR]), phrase-level (Ref-L4 and VG) and reasoning-level (Ferret-Reasoning). The best results are bold and the second-best results are underlined.

LVIS PACO DLC-Bench Ref-L4 VG Ferret SSim SIoU SSim SIoU Pos Neg Avg CLAIR METEOR CIDER METEOR CIDER Reasoning

Method

DAM-3B [28] – – – – 52.3 82.2 67.3 – 17.2 56.4 – – – PAM-3B [34] 88.6 78.3 87.4 74.9 – – – – 17.2 59.7 20.8 142.3 – Osprey-7B [70] 65.2 38.2 73.1 52.7 – – – – – – – – 67.8 Ferret-7B [67] 63.8 36.6 58.7 26.0 14.2 46.8 30.5 45.2 10.7 39.7 – – 67.3 DAM-8B [28] 89.0 77.7 84.2 73.2 – – – 57.9 19.4 70.0 – – – SPHINX-V-13B [33] 87.1 62.9 79.9 55.0 26.3 71.6 49.0 51.2 10.7 32.4 20.6 141.8 70.4 PixelRefer-Lite-2B 89.4 82.0 89.3 81.9 41.1 80.2 60.7 63.5 13.0 92.9 18.6 155.0 74.5 PixelRefer-Lite-7B 89.6 82.5 90.3 83.1 48.2 72.4 60.3 56.2 12.6 89.4 19.0 161.4 78.1 PixelRefer-2B 89.8 82.5 90.1 82.7 46.8 85.4 66.1 60.9 14.1 102.2 19.7 161.2 78.5 PixelRefer-7B 90.5 82.7 91.7 85.3 49.6 87.0 68.3 60.8 13.8 98.2 19.7 168.2 83.1

[Figure 267]

[Figure 268]

Part-level

Prompt: What is the category of <object>? Answer: Plush toy.

The wheel features a black tire with a rugged tread pattern, mounted on a dark-colored rim. The rim has a circular design with multiple evenly spaced spokes radiating from the center hub to the outer edge.

<object>

[Figure 269]

Prompt: Please describe <object> in detail. Answer: A plush toy with a blue, textured surface resembling scales or feathers. It has large, round, white eyes with black pupils and small, triangular ears protruding from the top of its head.

Part-level

The windshield is a large, slightly curved glass panel with a black wiper blade attached to the left side. The top edge of the windshield has a slight overhang, and there are two small stickers or decals on the lower right corner.

[Figure 270]

Object-level

Prompt: What is <object> used for? Answer: It is used for decoration. It is a stuffed animal that is placed on the table

A white cab-over-engine style truck with a gray tarpaulincovered cargo area. The truck has a single rear axle and black

tires. The front of the truck features a rectangular grille, round headlights, and orange turn signals on either side of the grille. There is a small red logo on the lower right corner of the door.

to add some fun and personality to the dining area.

(a) (b)

- Figure 9 (a) Multi-perspective object understanding with PixelRefer. The model generates diverse responses according to different prompts. (b) Granular visual understanding with PixelRefer. PixelRefer yields distinct detailed descriptions at part-level and object-level based on specified region granularity. For our Vision-Object Framework, the FLOPs are calculated by:

S

Ks(2(LR + LZ)d2 + 2(LR + LZ)ddkv + 2(LR + LZ)2d + 3(LR + LZ)dm).

FLOPs =

(9)

###### For our Object-Only Framework, FLOPs are computed as:

S

Ks 2LRd2 + 2LRddkv + 2L2Rd + 3LRdm

FLOPs =

(10)

+ 2(LR + LZL)d2 + 2(LR + LZG)d2,

where LR, LZ, LZ

denote the numbers of region tokens, vision tokens, local-to-object tokens, and global-to-object tokens, respectively.

and LZ

L

G

6.2 Main Results

- 6.2.1 Image-level Benchmarks

We begin by evaluating the model on image-level region understanding benchmarks, which encompass three key aspects: category recognition, phrase-level captioning, and detailed captioning. Table 1 summarizes the comparison results. Additionally, we provide visualization examples in Fig. 9-(a) to demonstrate the model’s adaptability to instructions, which showcases varied responses based on different prompts. In Fig. 9-(b), we highlight the model’s capability to offer diverse, detailed descriptions of each region, with different granularity from object-level to part-level details.

Category Recognition. This task requires the model to output the category or part-level category corresponding to a given region. Following Osprey [70], we adopt object-level LVIS [20] and part-level PACO [47] as

evaluation benchmarks. Our approach achieves state-of-the-art (SOTA) performance on both datasets. On the PACO benchmark, a particularly challenging category recognition task involving both whole objects and object parts in complex scenes, which requires the model to distinguish whether a region corresponds to an object or a part, our PixelRefer-7B attains 91.7% semantic similarity (SSim) and 85.3% semantic IoU (SIoU), surpassing the previous best by 4.3% and 10.4%, respectively. In addition, our lightweight PixelRefer-Lite-7B also surpasses the previous SOTA by 2.9% SSim and 8.4% SIoU. Notably, the PACO dataset is dominated by small part-level regions. The substantial performance gains in these part regions highlight the effectiveness of our proposed Scale-Adaptive Object Tokenizer, particularly in handling fine-grained, small-scale visual components.

Phrase-level Caption. This task requires the model to generate a short phrase or brief description for each given region. We evaluated performance on VG [24] and Ref-L4 [8] datasets. Our PixelRefer-7B model achieves comparable performance to existing methods on VG-METEOR, and attains the best performance on VG-CIDER, with a score of 168.2%, outperforming PAM-3B [34] by 45.9%. For Ref-L4, we conduct zero-shot evaluation on the Objects365 [53] split. Following the evaluation protocol of [28], both model predictions and ground-truth captions are first summarized by GPT-4o [43], and then evaluated using short captioning metrics. In this setting, our PixelRefer-2B achieves a 32.2% improvement in CIDER over the previous best model. While its METEOR score is slightly lower, this may be attributed to formatting mismatches between the generated outputs and ground-truth annotations.

DetailedCaption. In this setting, the model is expected to generate comprehensive and fine-grained descriptions of each region, going beyond short phrases to capture nuanced attributes and contextual information. To assess this capability, we conduct evaluations on DLC-Bench [28] and Ref-L4 (CLAIR) [8] benchmarks on the Objects365 subset. Our models demonstrate strong performance. In particular, PixelRefer-7B achieves state-of-the-art results on DLC-Bench with 68.3%, while PixelRefer-Lite-2B surpasses the previous best model on Ref-L4-CLAIR by 5.6%.

Reasoning Questions. In this setting, the model is required to perform reasoning based on one or more referred regions correctly. We evaluate this ability on the Referring Reasoning task of Ferret-Bench [67], which involves commonsense reasoning in visual context. Our PixelRefer demonstrates superior performance on this task, improving the score from 70.4% to 83.1% (+12.7%). These results indicate that our approach effectively narrows the gap between visual perception and high-level reasoning, enabling more accurate interpretation of complex visual scenarios.

- 6.2.2 Video-level Benchmarks

To thoroughly evaluate video-level object understanding, we conduct experiments on both caption-level and question-answering (QA)-level subtasks, leveraging both existing established benchmarks and ours newly constructed VideoRefer-Bench designed for this study. For caption-level tasks, we employ VideoRefer-BenchD and HC-STVG [58]. For QA-based tasks that require answering dynamic and context-aware queries, we adopt challenging VideoRefer-BenchQ. This benchmark enables us to evaluate how effectively models can infer answers to complex questions, capturing spatiotemporal dependencies and contextual understanding in video-based scenarios. Qualitative results are presented in Fig. 13. Our method, PixelRefer, exhibits strong capability across diverse video referring tasks, including video object captioning, multi-object question answering, and zero-shot spatial understanding.

VideoRefer-BenchD. We benchmark our method on VideoRefer-BenchD and compare it against several advanced generalist models, including GPT-4o [43], GPT-4o-mini [43], InternVL2 [12], Qwen2-VL [65], LLaVAOV [25], LongVA [79], and LongVU [54], as well as region-level specialist models for object-level understanding, such as Elysium [61], Artemis [46], DAM [28] and PAM [34]. In the single-frame (S) mode, we use the first frame containing the target object and its aligned boundary as input for generalist models, while image-level methods process a random frame paired with the corresponding region prompt. In the multi-frame (M) mode, object masks are generated for each key frame using the off-the-shelf SAM 2 [49]. For our PixelRefer, we simply sample a random single frame in the single-frame (S) mode. Table 2 presents the comparison results. Our approach achieves leading average performance in regional-temporal video understanding. Notably, in the single-frame setting, PixelRefer achieves top scores of 4.70 in Subject Correspondence (SC), 3.59 in Appearance Description (AD), and 3.39 in Temporal Description (TD), surpassing DAM-8B [28] by an average

###### Table 2 Performance comparisons on VideoRefer-BenchD. The best results are bold and the second-best results are underlined. For general baselines, masks of the targets are overlaid on the original video. S: single-frame mask, M: multi-frame masks.

Method Mode SC AD TD HD Avg. Generalist Models

LongVU-7B [54] S 2.02 1.45 1.98 1.12 1.64 LongVA-7B [79] S 2.63 1.59 2.12 2.10 2.11 LLaVA-OV-7B [25] S 2.62 1.58 2.19 2.07 2.12 Qwen2-VL-7B [62] S 2.97 2.24 2.03 2.31 2.39 InternVL2-26B [12] S 3.55 2.99 2.57 2.25 2.84 GPT-4o [43] S 3.34 2.96 3.01 2.50 2.95 GPT-4o-mini [43] S 3.56 2.85 2.87 2.38 2.92

Region-level Models

DAM-3B [28] M 3.62 2.86 2.81 2.67 2.99 PAM-3B [34] S 3.92 2.84 2.88 2.94 3.14 Elysium-7B [61] S 2.35 0.30 0.02 3.59 1.57 Artemis-7B [46] S 3.42 1.34 1.39 2.90 2.26 VideoRefer-7B [72] S 4.44 3.27 3.10 3.04 3.46 DAM-8B [28] M 4.69 3.61 3.34 3.09 3.68 PixelRefer-Lite-2B M 4.56 3.41 3.08 3.12 3.53 PixelRefer-Lite-7B M 4.69 3.56 2.28 3.06 3.64 PixelRefer-2B S 4.59 3.40 3.25 3.09 3.58 PixelRefer-7B S 4.70 3.59 3.39 3.13 3.70

###### Table 3 Performance comparisons on VideoRefer-BenchQ. BQ: Basic Questions, SQ: Sequential Questions, RIQ: Relationship Questions,RsQ: Reasoning Questions, FP: Future Prediction.

Method BQ SQ RlQ RsQ FP Avg. Generalist Models

LongVU-7B [54] 47.2 61.3 57.5 85.3 65.8 61.0 LongVA-7B [79] 56.2 62.5 52.0 83.9 65.8 61.8 InternVL2-26B [12] 58.5 63.5 53.4 88.0 78.9 65.0 GPT-4o-mini [43] 57.6 67.1 56.5 85.9 75.4 65.8 Qwen2-VL-7B [65] 62.0 69.6 54.9 87.3 74.6 66.0 LLaVA-OV-7B [25] 58.7 62.9 64.7 87.4 76.3 67.4 GPT-4o [43] 62.3 74.5 66.0 88.0 73.7 71.3

Region-level Models

Osprey-7B [70] 45.9 47.1 30.0 48.6 23.7 39.9 Ferret-7B [67] 35.2 44.7 41.9 70.4 74.6 48.8 Elysium-7B [61] - - - - - Artemis-7B [46] - - - - - PAM-3B [34] - - - - - DAM-8B [28] - - - - - VideoRefer-7B [72] 75.4 68.6 59.3 89.4 78.1 71.9 PixelRefer-Lite-2B 70.3 58.8 56.4 80.7 73.7 65.7 PixelRefer-Lite-7B 81.2 72.7 68.5 88.4 81.6 76.9 PixelRefer-2B 82.1 73.0 64.7 90.2 81.6 76.5 PixelRefer-7B 84.5 76.9 71.5 89.5 79.7 79.4

###### Table 4 Quantitative comparisons with video object-centric methods on HC-STVG benchmark.

Method METEOR CIDER BLEU@4 ROUGE-L SPICE

DAM-3B [28] 18.2 72.7 – – – PAM-3B [34] 23.3 70.3 – – – Elysium-7B [61] – – – – – Merlin-7B [68] 11.3 10.5 3.3 26.0 20.1 Artemis-7B [46] 18.0 53.2 15.5 40.8 25.4 VideoRefer-7B [72] 18.7 68.6 – – – DAM-8B [28] 21.0 91.0 19.8 45.9 31.4 PixelRefer-Lite-2B 21.1 91.3 19.0 45.8 31.2 PixelRefer-Lite-7B 21.9 92.7 20.7 46.5 31.3 PixelRefer-2B 19.5 78.9 17.2 43.8 30.1 PixelRefer-7B 21.1 97.4 20.1 46.1 32.5

Table 5 FLOPs and memory consumption of different VideoRefer model variants under image and video settings. Experiments are conducted on DLC-Bench [28] (Image) and HC-STVG [58] (Video).

Method LR LZ LZ

L FLOPs(T)↓ Memory(GB)↓ DLC-Bench (Image)

LZ

G

###### PixelRefer-2B 32 ∼ 1408 – – 1.51 13.2 PixelRefer-2B-Lite 32 0 576 256 0.03 4.9 PixelRefer-7B 32 ∼ 1408 – – 7.08 25.1 PixelRefer-7B-Lite 32 0 576 256 0.17 15.8

HC-STVG (Video)

PixelRefer-2B 32 ∼ 7185 – – 11.15 24.6 PixelRefer-2B-Lite 32 0 576 256 0.11 5.1 PixelRefer-7B 32 ∼ 7185 – – 43.83 36.9 PixelRefer-7B-Lite 32 0 576 256 0.61 17.6

of +0.02, despite the latter leveraging multi-frame inputs with denser object masks.

VideoRefer-BenchQ. We further evaluate our method on VideoRefer-BenchQ, which assesses a model’s ability to answer multiple-choice questions involved in referred video regions. Notebaly, some specialist models like DAM [28], PAM [34], Elysium [61] and Artemis [46] lack the capability to support this task. Therefore, we compare our method against generalist models as well as image-based region-level baselines to provide a comprehensive performance analysis. As presented in Table 3, our approach achieves the best average performance, scoring 79.4%, and exceeding the closed-source GPT-4o by 8.1%. Additionally, PixelRefer-7B achieves top scores across multiple subcategories, including Basic Questions (BQ) at 84.5%, Sequential Questions (SQ) at 76.9%, Relationship Questions (RQ) at 71.5%, Reasoning Questions (RQ) at 89.5%, and Future Prediction (FP) at 79.7%. These results clearly validate the effectiveness of our method in addressing the challenges of spatiotemporal video understanding. The lightweight PixelRefer-Lite variant shows relatively lower performance, primarily due to architectural constraints that prevent it from leveraging global scene-level choice-based training data, thereby limiting its instruction-following capability.

HC-STVG [58]. This benchmark assesses a model’s ability to generate detailed object-level descriptions in videos, with a particular focus on human-centric scenarios. As reported in Table 4, our proposed PixelRefer achieves leading performance, surpassing the previous best model, DAM-8B [28]. Specifically, PixelRefer-7B

###### PACOSSim DLC-Bench VideoRefer-BenchD HC-STVGCIDER

[Figure 271]

- Figure 10 Effects of object token scaling across four typical benchmarks, including PACO, DLC-Bench, VideoReferBenchD, and HC-STVG. We evaluate the impact of varying object token numbers (1, 8, 16, 32) under two model configurations: Vision-Object and Object-Only.

###### Table8 Ablation results for the design of the Scale-Adaptive Object Tokenizer across both image and video benchmarks.

Table7 Ablation results on different object token numbers with both Vision-Object Framework and Object-Only Framework.

Method LVISAvg DLC-Bench VideoReferD HC-STVGCIDER

Token Num. PACOSSim DLC-Bench VideoReferD HC-STVGCIDER Vision-Object Framework

###### Mask Pooling 79.4 56.8 3.50 76.6 w/o expansion 81.0 65.4 3.56 78.0 w/o position emb. 86.0 64.3 3.52 77.2

###### 1 87.2 56.9 3.53 75.9 8 89.9 62.3 3.53 78.7

Ours 86.2 66.1 3.58 78.9

###### Table 9 Ablation results on the design of Object-Centric Infusion Module.

16 90.1 61.2 3.54 79.1 32 90.1 66.1 3.58 78.9

Object-Only Framework

L-Attn G-Attn LVISSSim DLC-Bench VideoReferD HC-STVGCIDER

1 83.7 48.8 2.95 56.4 8 87.6 54.0 3.45 76.6

###### ✗ ✗ 85.0 59.0 3.37 69.6

✓ ✗ 85.2 60.5 3.46 73.7

16 88.5 59.7 3.49 78.6 32 89.3 60.7 3.53 91.3

###### ✗ ✓ 88.2 60.6 3.48 77.2

✓ ✓ 89.4 60.7 3.53 91.3

attains 21.1 METEOR (+0.1), 97.4 CIDER (+6.1), 20.1 BLEU@4 (+0.3), 46.1 ROUGE-L (+0.2), and 32.5 SPICE (+1.1), compared to DAM-8B [28], demonstrating consistent improvements across all metrics. In addition, PixelRefer-Lite-7B delivers performance comparable to PixelRefer-7B, while offering greater efficiency.

- 6.3 Efficiency Analysis

Table 6 Inference time and memory usage on DLC-Bench [28] (Image) and HC-STVG [58] (Video). We report per-item inference time (s/item) and peak GPU memory (GB).

DLC-Bench HC-STVG

Model Infer time(s)↓ Memory(GB)↓ Infer time(s)↓ Memory(GB)↓

DAM-3B [28] 1.29 7.8 2.68 10.4 PAM-3B [34] 1.09 9.4 1.51 12.7 PixelRefer-2B 1.04 13.2 0.82 24.6 PixelRefer-Lite-2B 0.88 4.9 0.68 5.2 PixelRefer-7B 1.44 25.1 1.25 36.9 PixelRefer-Lite-7B 1.10 15.8 0.74 17.6

The FLOPs and memory usage of our model are reported in Table 5. In the video setting, we uniformly sample 20 frames for each video to ensure a fair comparison. As shown, the Object-Only Framework significantly reduces computational and memory demands. For instance, with video inputs, PixelRefer-2B requires 11.15T FLOPs and 24.6GB of GPU memory, whereas the object-only variant PixelRefer-2B-Lite reduces the cost to merely 0.11T FLOPs and 5.1GB of memory. Similar reductions are also observed for larger models and in the image input setting. These results highlight that the Object-Only Framework is highly efficient in minimizing computational overhead and memory consumption, providing a scalable and cost-effective solution for large-scale applications without compromising performance. As shown in Table 6, we further provide a detailed comparison of inference time and memory usage with DAM [28] and PAM [34]. Notably, PixelRefer-2B achieves significant reductions in both metrics, particularly in the video setting.

- 6.4 Ablation Study

###### Table 10 Ablation results on the impact of various training data types. We utilize SSim for the LVIS benchmark and METEOR for the HC-STVG, and the average scores for the remaining benchmarks.

Image-Region-Bench Video-Region-Bench General-Bench LVIS DLC-Bench HC-STVG VideoReferD VideoReferQ MVBench POPE

Data #Samples

###### Region Recognition 390K 89.6 61.2 11.9 2.94 72.3 60.3 87.3 + Image Detailed Cap. 860K 89.7 66.4 13.0 2.97 71.9 58.7 88.2 + Video Detailed Cap. 180K 89.7 66.0 19.1 3.69 74.8 61.9 88.0 + Region QA 560K 89.7 66.6 19.6 3.62 75.8 61.6 83.9 + General QA 300K 89.8 66.1 19.5 3.58 76.5 63.4 88.7

Scaling Object Tokens. We study how the number of object tokens in our scale-adaptive tokenizer influences both the Vision–Object and Object-Only frameworks. The results are presented in Table 7 and Fig. 10. For the Vision–Object Framework, increasing the number of object tokens from 1 to 8 yields the most substantial gains across benchmarks. Beyond this point, improvements largely plateau: PACOSSim and VideoRefer-BenchD show little change beyond 8 tokens, HC-STVG drops marginally, while DLCBench continues to benefit up to 32 tokens. In contrast, the Object-Only Framework exhibits consistent improvements as the number of tokens increases. Adding more tokens progressively narrows the performance gap relative to the Vision–Object model, and on HC-STVG, it even surpasses the latter when using 32 tokens. These findings highlight the complementary role of global vision tokens, which provide scene-level context and allow the Vision–Object model to achieve strong results with relatively few object tokens. Conversely, the Object-Only model relies more heavily on a larger token budget to capture fine-grained object details and relational information. We further explore the role of the number of object tokens by visualizing the attention patterns between different object tokens and image tokens in Fig. 11 (using 16 tokens as an example). The visualization reveals that different object tokens focus on distinct regions of the objects, thereby supplementing detailed information.

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Prompt: Please describe the <region> in detail.

[Figure 276]

Figure 11 Visualization of attention map between object tokens and image tokens with 16 tokens.

Design of Scale-Adaptive Object Tokenizer (SAOT).

We conduct an in-depth analysis of the design in the Scale-adaptive Object Tokenizer, as illustrated in Table 8. First, we compare our design with the vanilla Mask Pooling method [70, 72]. Our tokenizer achieves significant gains on both image and video benchmarks, outperforming the baseline with 6.8% on LVIS, 8.6% on DLC-Bench and 1.4% on HC-STVG. To further investigate its efficacy, we divide the regions into two groups based on pixel count: small regions (<2000 pixels) and large regions (≥2000 pixels). As depicted in Fig. 12, the performance gap is particularly pronounced for smaller regions, with improvements of 15.6% on LVIS and 9.6% on DLC-Bench. These results clearly showcase the effectiveness of our design in preserving object details, especially in scenarios involving tiny objects.

[Figure 277]

| | | | |
|---|---|---|---|
| |[Figure 278]<br><br>[Figure 279]| | |
| | | | |
| |[Figure 280]<br><br>[Figure 281]<br><br>-9.6%| | |
| | | | |
| |[Figure 282]| | |
| |[Figure 283]| | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| |[Figure 284]<br><br>-15.6%| | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

[Figure 285]

[Figure 286]

Figure 12 Performance comparisons between the proposed Scale-Adaptive Object Tokenizer (SAOT) and MaskPooling on LVIS and DLC-Bench with large and small regions. Our SAOT demonstrates consistently strong performance across both region sizes, while MaskPooling suffers significant degradation on small regions, highlighting the importance of scale-aware object representation.

We further analyze the impact of the expansion operation, which incorporates surrounding context after region cropping to enrich feature representations. As shown in Table 8, omitting this expansion results in a noticeable decline in performance, particularly on image benchmarks, with decreases of 5.2% on LVIS and 0.7% on DLC-Bench. These results underscore the key role of contextual information in enhancing region-level feature extraction.

[Figure 287]

[Figure 288]

###### Video Input

[Figure 289]

###### Video Input

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

<object1>

<object0>

<object1>

<object0>

Prompt What distinguishes <object0>'s attire from <object1>'s?

Prompt Which is higher? <object0> or <object1>? Answer <object0> is higher.

Answer

<object0> is wearing a blue shirt, while <object1> is wearing a black jacket.

[Figure 295]

Prompt What can be inferred about the role of <object1> in the scene?

###### Video Input Comparison with other models

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Answer

<object1> appears to be assisting <object0>, possibly a store employee or customer, in handling the large box. His role is likely to provide support and ensure that the box is properly placed on the cart.

<object0>

###### Video Input

Prompt How many chairs are around the <object0>?

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

<object>

There are no chairs around the <object0> in the provided frames. The frames show various parts of a kitchen and bedroom, but none of them

###### Qwen2.5-VL depict chairs around an object labeled as <object0>.

Incorrect

Prompt Please describe <object> in the video detail. Answer

[Figure 305]

[Figure 306]

A round, metallic object with a smooth, reflective surface, likely a clock, is prominently featured...The object maintains a consistent

A duck is initially seen walking on the grass, its body facing forward. As the video progresses, the duck begins to fly, transitioning from a stationary position to an airborne state. The bird's wings are spread wide, and it takes off with a graceful motion, showcasing its ability to navigate both land and air. The background remains subdued, allowing the focus to remain on the duck's movement and transformation.

Simply description

DAM-3B orientation throughout the sequence.

[Figure 307]

[Figure 308]

There are four chairs around it.

PixelRefer

Figure 13 Left: Versatile video referring with PixelRefer. PixelRefer handles diverse video referring tasks, including video object captioning, multi-object question answering, and zero-shot spatial understanding. Right: Comparing PixelRefer with Qwen2.5-VL [4], DAM [28] on video object referring task. PixelRefer exhibits the ability to accurately identify specific objects while also comprehending the overall context of the video.

Finally, we examine the design of position embedding, which incorporates relative positional features into object tokens. As shown in Table 8, this design yields improvements in both image and video benchmarks, particularly for tasks requiring detailed descriptions. These tasks necessitate not only accurate category recognition but also a coherent understanding of each object’s spatial location within the image or video sequence.

Design of Object-Centric Infusion (OCI) Module. We analyze the effects of Local-to-Object Attention (L-Attn) and Global-to-Object Attention (G-Attn) within the Object-Centric Infusion Module. Table 9 presents the results. The baseline uses only object features, without either L-Attn or G-Attn, meaning the model can only “see” the object itself, without contextual cues. Introducing local context through L-Attn yields consistent improvements across all benchmarks, confirming that nearby contextual information aids in disambiguating object understanding. Adding global context via G-Attn leads to even larger gains, highlighting the importance of scene-level cues when interpreting small or ambiguous regions. When both mechanisms are combined, performance reaches its highest level: +4.4% on LVIS, +1.7% on DLC-Bench, +0.16 on VideoRefer-D, and a striking +21.7% on HC-STVG. These results confirm that local and global contexts are complementary, where local cues refine details, while global cues provide holistic scene information, together enabling more effective object-centric representation.

Impact of Diverse Training Data. To evaluate the effectiveness of the datasets collected in our PixelRefer-2.2M, we classify the datasets we used into six types: Region Recognition, Image Detailed Caption, Video Caption, Region QA and General QA. Table 10 reports results across diverse benchmarks, spanning region- and scene-level, image- and video-level, QA- and description-level tasks. Starting with only the region recognition datasets, the model exhibits basic category cognition with 89.6% on LVIS, which is relatively easy, but struggles on tasks requiring detailed descriptions or QA. Incorporating image and video captioning data substantially enhances captioning performance while preserving region recognition ability. The inclusion of Region QA data further enhances QA performance, most notably on VideoRefer-BenchQ. Lastly, incorporating General QA data strengthens the model’s QA capabilities without impairing other tasks, thereby mitigating the risk of catastrophic forgetting.

- 7 Conclusion

We presented PixelRefer, a unified region-level MLLM framework designed to support fine-grained spatiotemporal object-centric understanding across images and videos with arbitrary granularity. By introducing the Scale-Adaptive Object Tokenizer (SAOT), PixelRefer generated compact and semantically rich object representations from free-form regions. Building upon empirical analysis of attention patterns within LLMs, we further developed PixelRefer-Lite, an efficient Object-Only Framework that employs an Object-Centric Infusion module to pre-fuse global context into object tokens, significantly improving efficiency without sacrificing accuracy. To support robust training, we curated PixelRefer-2.2M, a high-quality object-centric instruction dataset. Extensive experiments across diverse tasks, ranging from captioning and recognition to complex reasoning, demonstrated PixelRefer’s state-of-the-art performance with fewer training samples. Meanwhile, the PixelRefer-Lite variant offers comparable accuracy with notable efficiency gains, highlighting the practicality and scalability of our proposed framework.

References

- [1] Rawan AlSaad, Alaa Abd-Alrazaq, Sabri Boughorbel, Arfan Ahmed, Max-Antoine Renault, Rafat Damseh, and Javaid Sheikh. Multimodal large language models in health care: applications, challenges, and future outlook. Journal of medical Internet research, 26:e59505, 2024.
- [2] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In ECCV, pages 382–398, 2016.
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv:2308.12966, 2023.
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [5] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In ACL Workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pages 65–72, 2005.
- [6] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In CVPR, pages 12914–12923, 2024.
- [7] Chi Chen, Ruoyu Qin, Fuwen Luo, Xiaoyue Mi, Peng Li, Maosong Sun, and Yang Liu. Position-enhanced visual instruction tuning for multimodal large language models. arXiv preprint arXiv:2308.13437, 2023.
- [8] Jierun Chen, Fangyun Wei, Jinjing Zhao, Sizhe Song, Bohuai Wu, Zhuoxuan Peng, S-H Gary Chan, and Hongyang Zhang. Revisiting referring expression comprehension evaluation in the era of large multimodal models. In CVPR, pages 513–524, 2025.
- [9] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.
- [10] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In ECCV, pages 19–35. Springer, 2024.
- [11] Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect what you can: Detecting and representing objects using holistic models and body parts. In CVPR, pages 1971–1978, 2014.
- [12] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.
- [13] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In NeurIPS, 2024.
- [14] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.

- [15] Ronghao Dang, Yuqian Yuan, Yunxuan Mao, Kehan Li, Jiangpin Liu, Zhikai Wang, Xin Li, Fan Wang, and Deli Zhao. Rynnec: Bringing mllms into embodied world. arXiv preprint arXiv:2508.14160, 2025.
- [16] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.
- [17] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. Mevis: A large-scale benchmark for video segmentation with motion expressions. In ICCV, pages 2694–2703, 2023.
- [18] Hao Fei, Shengqiong Wu, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. In NeurIPS, 2024.
- [19] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In CVPR, pages 13796–13806, 2024.
- [20] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In CVPR, pages 5356–5364, 2019.
- [21] Ju He, Shuo Yang, Shaokang Yang, Adam Kortylewski, Xiaoding Yuan, Jie-Neng Chen, Shuai Liu, Cheng Yang, Qihang Yu, and Alan Yuille. Partimagenet: A large, high-quality dataset of parts. In ECCV, pages 128–145. Springer, 2022.
- [22] Xiaoke Huang, Jianfeng Wang, Yansong Tang, Zheng Zhang, Han Hu, Jiwen Lu, Lijuan Wang, and Zicheng Liu. Segment and caption anything. In CVPR, pages 13405–13417, 2024.
- [23] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, pages 787–798, 2014.
- [24] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 123(1):32–73, 2017.
- [25] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [26] Sijing Li, Tianwei Lin, Lingshuai Lin, Wenqiao Zhang, Jiang Liu, Xiaoda Yang, Juncheng Li, Yucheng He, Xiaohui Song, Jun Xiao, et al. Eyecaregpt: Boosting comprehensive ophthalmology understanding with tailored dataset, benchmark and model. arXiv preprint arXiv:2504.13650, 2025.
- [27] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. arXiv preprint arXiv:2407.02392, 2024.
- [28] Long Lian, Yifan Ding, Yunhao Ge, Sifei Liu, Hanzi Mao, Boyi Li, Marco Pavone, Ming-Yu Liu, Trevor Darrell, Adam Yala, et al. Describe anything: Detailed localized image and video captioning. In ICCV, 2025.
- [29] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2023.
- [30] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004.
- [31] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, pages 26689–26699, 2024.
- [32] Tianwei Lin, Wenqiao Zhang, Sijing Li, Yuqian Yuan, Binhe Yu, Haoyuan Li, Wanggui He, Hao Jiang, Mengze Li, Xiaohui Song, et al. Healthgpt: A medical large vision-language model for unifying comprehension and generation via heterogeneous knowledge adaptation. arXiv preprint arXiv:2502.09838, 2025.
- [33] Weifeng Lin, Xinyu Wei, Ruichuan An, Peng Gao, Bocheng Zou, Yulin Luo, Siyuan Huang, Shanghang Zhang, and Hongsheng Li. Draw-and-understand: Leveraging visual prompts to enable mllms to comprehend what you want. In ICLR, 2025.
- [34] Weifeng Lin, Xinyu Wei, Ruichuan An, Tianhe Ren, Tingwei Chen, Renrui Zhang, Ziyu Guo, Wentao Zhang, Lei Zhang, and Hongsheng Li. Perceive anything: Recognize, explain, caption, and segment anything in images and videos. In NeurIPS, 2025.
- [35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.

- [36] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pages 26296–26306, 2024.
- [37] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge. https://llava-vl.github.io/blog/2024-01-30-llava-next/, 2024.
- [38] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024.
- [39] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, pages 11–20, 2016.
- [40] Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. A survey of multimodal retrieval-augmented generation. arXiv preprint arXiv:2504.08748, 2025.
- [41] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. arXiv preprint arXiv:2402.06196, 2024.
- [42] OpenAI. Gpt-4v(ision) system card. https://cdn.openai.com/papers/GPTV_System_Card.pdf, 2023.
- [43] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024.
- [44] R OpenAI. Gpt-4 technical report. arxiv 2303.08774. View in Article, 2(5):1, 2023.
- [45] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, pages 311–318, 2002.
- [46] Jihao Qiu, Yuan Zhang, Xi Tang, Lingxi Xie, Tianren Ma, Pengyu Yan, David Doermann, Qixiang Ye, and Yunjie Tian. Artemis: Towards referential understanding in complex videos. In NeurIPS, 2024.
- [47] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, et al. Paco: Parts and attributes of common objects. In CVPR, pages 7141–7151, 2023.
- [48] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In CVPR, pages 13009–13018, 2024.
- [49] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
- [50] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In ICLR, 2025.
- [51] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [52] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.
- [53] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, pages 8430–8439, 2019.
- [54] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024.
- [55] Yan Shu, Bin Ren, Zhitong Xiong, Danda Pani Paudel, Luc Van Gool, Begum Demir, Nicu Sebe, and Paolo Rota. Earthmind: Towards multi-granular and multi-sensor earth observation with large multimodal models. arXiv preprint arXiv:2506.01667, 2025.
- [56] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [57] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. arXiv preprint arXiv:2312.17432, 2023.
- [58] Zongheng Tang, Yue Liao, Si Liu, Guanbin Li, Xiaojie Jin, Hongxu Jiang, Qian Yu, and Dong Xu. Human-centric spatio-temporal video grounding with visual transformers. TCSVT, 32(12):8238–8249, 2021.
- [59] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. Chatterbox: Multi-round multimodal referring and grounding. arXiv preprint arXiv:2401.13307, 2024.
- [60] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, pages 4566–4575, 2015.
- [61] Han Wang, Yongjie Ye, Yanjie Wang, Yuxiang Nie, and Can Huang. Elysium: Exploring object-level perception in videos via mllm. In ECCV, pages 166–185. Springer, 2024.
- [62] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [63] Yihan Xie, Sijing Li, Tianwei Lin, Zhuonan Wang, Chenglin Yang, Yu Zhong, Wenqiao Zhang, Haoyuan Li, Hao Jiang, Fengda Zhang, et al. Heartcare suite: Multi-dimensional understanding of ecg with raw multi-lead signal modeling. arXiv preprint arXiv:2506.05831, 2025.
- [64] Shiyu Xuan, Qingpei Guo, Ming Yang, and Shiliang Zhang. Pink: Unveiling the power of referential comprehension for multi-modal llms. In CVPR, pages 13838–13848, 2024.
- [65] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [66] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023.
- [67] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. In ICLR, 2024.
- [68] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. In ECCV, pages 425–443. Springer, 2025.
- [69] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In ECCV, pages 69–85. Springer, 2016.
- [70] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In CVPR, pages 28202–28211, 2024.
- [71] Yuqian Yuan, Ronghao Dang, Long Li, Wentong Li, Dian Jiao, Xin Li, Deli Zhao, Fan Wang, Wenqiao Zhang, Jun Xiao, et al. Eoc-bench: Can mllms identify, recall, and forecast objects in an egocentric world? arXiv preprint arXiv:2506.05287, 2025.
- [72] Yuqian Yuan, Hang Zhang, Wentong Li, Zesen Cheng, Boqiang Zhang, Long Li, Xin Li, Deli Zhao, Wenqiao Zhang, Yueting Zhuang, et al. Videorefer suite: Advancing spatial-temporal object understanding with video llm. In CVPR, pages 18970–18980, 2025.
- [73] Tongtian Yue, Jie Cheng, Longteng Guo, Xingyuan Dai, Zijia Zhao, Xingjian He, Gang Xiong, Yisheng Lv, and Jing Liu. Sc-tune: Unleashing self-consistent referential comprehension in large vision language models. In CVPR, pages 13073–13083, 2024.
- [74] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In CVPR, pages 6720–6731, 2019.
- [75] Yufei Zhan, Yousong Zhu, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon v2: Advancing multimodal perception with high-resolution scaling and visual-language co-referring. arXiv preprint arXiv:2403.09333, 2024.
- [76] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.

- [77] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.
- [78] Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, et al. Ferret-v2: An improved baseline for referring and grounding with large language models. arXiv preprint arXiv:2404.07973, 2024.
- [79] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- [80] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023.
- [81] Tao Zhang, Xiangtai Li, Zilong Huang, Yanwei Li, Weixian Lei, Xueqing Deng, Shihao Chen, Shunping Ji, and Jiashi Feng. Pixel-sail: Single transformer for pixel-grounded understanding. arXiv preprint arXiv:2504.10465, 2025.
- [82] Wei Zhang, Miaoxin Cai, Tong Zhang, Yin Zhuang, Jun Li, and Xuerui Mao. Earthmarker: A visual prompting multi-modal large language model for remote sensing. TGRS, 2024.
- [83] Wenqiao Zhang, Tianwei Lin, Jiang Liu, Fangxun Shu, Haoyuan Li, Lei Zhang, He Wanggui, Hao Zhou, Zheqi Lv, Hao Jiang, et al. Hyperllava: Dynamic visual and language expert tuning for multimodal large language models. arXiv preprint arXiv:2403.13447, 2024.
- [84] Wenqiao Zhang, Zheqi Lv, Hao Zhou, Jia-Wei Liu, Juncheng Li, Mengze Li, Yunfei Li, Dongping Zhang, Yueting Zhuang, and Siliang Tang. Revisiting the domain shift and sample uncertainty in multi-source active domain transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16751–16761, 2024.
- [85] Yuanhan Zhang, Bo Li, Haotian Liu, Yong Jae, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model. https://llava-vl.github.io/blog/ 2024-04-30-llava-next-video/, 2024.
- [86] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. TMLR, 2025.
- [87] Liang Zhao, En Yu, Zheng Ge, Jinrong Yang, Haoran Wei, Hongyu Zhou, Jianjian Sun, Yuang Peng, Runpei Dong, Chunrui Han, et al. Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning. arXiv preprint arXiv:2307.09474, 2023.

