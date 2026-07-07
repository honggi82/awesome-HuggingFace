# arXiv:2501.00599v3[cs.CV]25Mar2025

#### VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM

Yuqian Yuan1,2*, Hang Zhang2, Wentong Li1, Zesen Cheng2, Boqiang Zhang2, Long Li1,2*, Xin Li2, Deli Zhao2, Wenqiao Zhang1†, Yueting Zhuang1, Jianke Zhu1†, Lidong Bing3

1Zhejiang University 2DAMO Academy, Alibaba Group 3Shanda AI Research Institute

Project Page Code

##### Abstract

Video Large Language Models (Video LLMs) have recently exhibited remarkable capabilities in general video understanding. However, they mainly focus on holistic comprehension and struggle with capturing fine-grained spatial and temporal details. Besides, the lack of high-quality object-level video instruction data and a comprehensive benchmark further hinders their advancements. To tackle these challenges, we introduce the VideoRefer Suite to empower Video LLM for finer-level spatial-temporal video understanding, i.e., enabling perception and reasoning on any objects throughout the video. Specially, we thoroughly develop VideoRefer Suite across three essential aspects: dataset, model, and benchmark. Firstly, we introduce a multi-agent data engine to meticulously curate a largescale, high-quality object-level video instruction dataset, termed VideoRefer-700K. Next, we present the VideoRefer model, which equips a versatile spatial-temporal object encoder to capture precise regional and sequential representations. Finally, we meticulously create a VideoRefer-Bench to comprehensively assess the spatial-temporal understanding capability of a Video LLM, evaluating it across various aspects. Extensive experiments and analyses demonstrate that our VideoRefer model not only achieves promising performance on video referring benchmarks but also facilitates general video understanding capabilities.

##### 1. Introduction

Multi-modal Large Language Models (MLLMs) [2, 8, 20– 23, 28, 36] have demonstrated remarkable general-purpose capabilities for open-world image understanding through language-based dialogues over the past year. In constant,

*Work is done during internship at DAMO Academy, Alibaba Group †Corresponding author

extending their capabilities to the video domain presents unique challenges, as videos comprise dynamic sequences that not only showcase visual content but also convey the timing and relationships among various events and objects. Currently, existing Video Large Language Models (Video LLMs) [9, 16, 18, 27, 52, 57] primarily focus on holistic scene understanding. Unfortunately, these approaches often fall short in capturing the nuanced elements of video content. For instance, they often struggle to focus on userspecific objects, such as accurately describing a particular object. Fig. 1-(a) illustrates a typical example from general VideoLLaMA2 [9]. The ability to discern such finer details in video content is crucial for applications that require precise object description, detailed event analysis, and predictive reasoning in dynamic environments.

To achieve fine-grained object understanding, numerous efforts have been devoted to image-based MLLMs, such as GPT4RoI [55], Ferret [46, 53] and Osprey [48]. These methods typically utilize a region encoder to obtain objectlevel embeddings, adapting them to LLMs for static image region-text alignment. In contrast, research on video-based object understanding remains limited. Some works [43, 47] directly convert the bounding box coordinates of object from specific frames into textual prompts to assist the LLM in identifying referred objects within the video. However, these methods are plagued by impractical object referring and suffer from imprecise regional understanding. Alternatively, Artemis [33] employs an external RoI tracker to capture an object across the video and extract box-level features for aligning with the LLM. However, as illustrated in Fig. 1(b), it primarily focuses on single-object referencing using coarse box-level representations, which restricts its capacity to handle complex tasks, such as analyzing relationships among multiple objects and performing intricate reasoning. Therefore, developing an interactive Video LLM that facilitates a comprehensive understanding of objects within video represents a nontrivial research challenge.

[Figure 1]

[Figure 2]

[Figure 3]

###### Video Relationship Analysis

Video Object Referring

###### Video Object Retrieval

[Figure 4]

[Figure 5]

| |[Figure 6]<br><br>[Figure 7]| |
|---|---|---|
| | | |
| | | |

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

| |
|---|

… … …

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

What is the man <object> doing in the video?

What is the ranking of contestant <object> at the end of the video?

What is the relationship between <object1> and <object2> ?

[Figure 19]

[Figure 20]

[Figure 21]

The knife <object1> moved the scallions from the chopping board <object2> to the pot.

The player was in first place at the end of the video.

The man was Trump, who stood in the crowd waving his fist.

###### VideoRefer VideoRefer VideoRefer

Video Understanding Methods Video Referring Methods

General MLLMs

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

The first image shows a man, the next images are from one video. Wh at is the man doing in the video?

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

What is the ranking of the athlete on the fifth track at the end?

[Figure 30]

[Figure 31]

[Figure 32]

ROI tracking

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Complex Referential Language

[Figure 37]

The pot is full of food, and is being cooked by a person.

The athlete on the fifth track was in last place at the end of the race.

[Figure 38]

In the video, the man...

VideoLLaMA2

Artemis

(a) (b) (c)

GPT-4O

Figure 1. Comparisons with previous general and specialized MLLMs. Our VideoRefer excels in multiple fine-grained regional and temporal video understanding tasks, including basic video object referring, complex video relationship analysis, and video object retrieval.

multiple frames while producing flexible, enriched regional representations. The image-level and object-level embeddings are interleaved with language instructions to form the input sequence for the LLM, facilitating a detailed object understanding of the input video.

In this work, we revisit the design of the Video LLM for finer-level video understanding. We contend that achieving this necessitates three essential components: a largescale dataset containing high-quality object-level video instruction data, an architecture that integrates object embeddings with temporal cues, and a thorough benchmark for performance assessment. To this end, we introduce VideoRefer Suite, designed to empower Video LLMs with spatialtemporal object comprehension.

Benchmark. Furthermore, to evaluate the regional video understanding capabilities of a Video LLM comprehensively, we develop a benchmark named VideoRefer-Bench, which consists of two subbenchmarks: VideoRefer-BenchD, which focuses on description generation from four aspects, and VideoRefer-BenchQ, which emphasizes multiplechoice question answering across five aspects. VideoReferBench thoroughly assesses the model’s performance across various timestamps and objects, evaluating the abilities in comprehensive captioning and reasoning, complex multi-object relationships, and future predictions.

Dataset. Firstly, to achieve regional alignment between video content and language embeddings, we meticulously curate a large-scale region-text video instruction dataset named VideoRefer-700K. Specifically, we present a multi-agent data engine to create high-quality video-based mask-text description pairs. This data engine leverages several expert models that excel in various tasks, collaborating meticulously to produce a diverse range of object-level instruction data for each object across the video. Our curated VideoRefer-700K comprises descriptions and multi-round QA pairs covering basic questions, complex reasoning and future predictions.

As illustrated in Fig. 1, our VideoRefer unlocks a range of advanced finer-level video understanding capabilities, including basic video object referring, intricate relationship analysis among objects and object retrieval tasks, maintaining user interactivity. In particular, VideoRefer can be seamlessly integrated with the off-the-shelf SAM 2 [35] to further enhance user interactivity by enabling a comprehensive understanding of everything user click on. Extensive experiments conducted on VideoRefer-Bench and general video understanding benchmarks, yield compelling results and demonstrate the efficacy of our approach. Notably, VideoRefer not only significantly surpasses the state-of-theart methods in regional video understanding across temporal, sequential and relationship reasoning, but also advances the general video understanding abilities.

Model. Next, we introduce an effective Video LLM, named VideoRefer, that enables fine-grained perceiving, reasoning and retrieval for user-defined regions at any specified timestamps. To accommodate both single-frame and multi-frame region inputs, we propose a versatile spatialtemporal object encoder. Specially, a Spatial Token Extractor is developed to generate accurate object-level encoding at any frame, leveraging a unified pixel-level mask representation to allow arbitrary free-form input regions. We then propose an adaptive Temporal Token Merge Module, which captures temporal contextual information across

##### 2. Preliminary

[Figure 39]

[Figure 40]

[Figure 41]

Description

###### Video + caption

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Appearance description & Motion description

Subject: [xxx] Other nouns: [xxx]

###### 2.1. Background and Video-referring Task.

[Figure 47]

[Figure 48]

Annotator

Analyzer

To attain precise regional comprehension, MLLMs can be incorporated with instance-level visual representations. This integration allows models to generate semantic understandings that focus on specific regions. As for image-based MLLMs, recent researchs [4–6, 11, 13, 34, 41, 44, 46, 48, 49, 51, 53, 55, 58] has demonstrated a significant trend to enable the image referring with spatial visual prompts. In contrast, research focused on video-based regional understanding across sequential scenes is relatively limited.

[Figure 49]

Yes Description

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Masks Segmentor

Refiner

[Figure 56]

[Figure 57]

[Figure 58]

No

Reviewer

[Figure 59]

[Figure 60]

[Figure 61]

QA data

Video + masks

[Figure 62]

[Figure 63]

[Figure 64]

Appearance description & Motion description

[Figure 65]

Multi-round QA pairs

[Figure 66]

[Figure 67]

Annotator Refiner

Figure 2. A multi-agent data engine for the construction of our VideoRefer-700K.

The video referring task involves comprehending userspecific regions at designated moments or a time periods within a video [33, 43, 47]. The basic video referring task focuses on captioning, while more complex tasks involve reasoning about the relationships between objects, and inferring their future states or interactions. Video referring tasks can significantly enhance the functionality and applicability of video analysis for Video LLM across multiple domains, such as navigation, surveillance, and interactive robotics.

ing high-quality instruction-following object-level annotations; a Video LLM, VideoRefer, capable of pixel-level regional and temporal comprehension; and an evaluation benchmark, VideoRefer-Bench, developed to evaluate models across various video referring tasks.

###### 3.1. VideoRefer-700K Dataset

###### 3.1.1 Multi-agent Data Engine

We develop an automatic multi-agent data engine to create VideoRefer-700K, a large-scale and high-quality object-level video instruction-following dataset. Specially, we utilize off-the-shelf expert models that excel in tasks such as captioning, detection, segmentation and summation as collaborative agents to carefully create diverse types of object-level instruction data. As illustrated in Fig. 2, our curation pipeline involves five components: (i) Analyzer for noun extraction; (ii) Annotator for object-level caption generation; (iii) Segmentor for mask generation; (iv) Reviewer for correspondence verification; and (v) Refiner for summarization&refinement. This multi-agent data engine effectively eliminates noisy or irrelevant contexts, ensuring that the data maintains its accuracy and relevance.

###### 2.2. Task Formulation.

For basic video object referring, the model processes questions phrased as “What is <object> doing in this video?”, where the <object> is specified by the user at a specific moment t or over a duration of time. In more complex scenarios involving various object relationships, the model requires multiple user-defined regions, such as <object1>, <object2> and <objectK> along with the corresponding questions, like “How do <object1> and <object2> interact with each other?”. To address these nuanced regional and temporal tasks, we provide a unified problem formulation.

For a given video V ∈ RN×W×H×C, where N, W, H, C denote the frame number, height, width and channels, respectively. We define all the <object> as R, where R = {R1,R2,...,Rn}. Here, n represents the total number of objects specified by the user. Rj is expressed as Rj = {rij | i ∈ T}, with rij representing a region within a single frame, and T being a set containing one or multiple timestamps. For a Video LLM, the model optimization process aims to maximize the log-likelihood of generating text conditioned on V , R, and text-based prompt x across the entire training dataset to produce the desired output:

Analyzer for Noun Extraction. Considering that most available video datasets contain the short scene-level caption, we begin by analyzing the raw captions to accurately capture the nouns within the sentences, i.e., objects occurred in the video scene. To achieve this, we employ an Analyzer to extract nouns, encompassing both subjects and other relevant nouns. The Qwen2-Instruct-7B model [45] serves as our analytical tool in this process.

Annotator for Object-level Caption Generation. To obtain detailed descriptions of the extracted nouns, we employ a general video understanding model as an annotator. We prompt the model to provide comprehensive descriptions focused specifically on the objects, rather than the holistic narrative of the whole video. To enhance accuracy and detail, we query the model twice: emphasizing dynamic actions&movements, and highlighting static appearances&states, respectively. Specifically, we filter out static actions related to the subjects to maintain variability and

log P(y | V, R1, ..., Rn, x), (1)

L =

(V,R,x,y)

where y denotes the ground truth label.

##### 3. VideoRefer Suite

Our VideoRefer Suite consists of three crucial components: a comprehensive dataset, VideoRefer-700K, contain-

dynamism in the videos. The open-source InternVL2-26B model [8] serves as our annotator.

Segmentor for Mask Generation. To acquire pixellevel masks as object-level region representations for each extracted noun, we first select a random frame from the video and extract the bounding box using GroundingDINO [24] through open-set grounding, with the extracted noun serving as the input text prompt. Subsequently, HQSAM [14] is employed to generate the high-quality mask based on the corresponding box prompt. To accommodate multi-frame input, we further generate masks for each video frame using SAM 2 [35].

Reviewer for Correspondence Verification. To address potential errors and mismatches in this data construction pipeline, we introduce a Reviewer to verify the correspondence between masks and descriptions. Initially, we employ Osprey [48] to provide a region-level description for a specific frame. The Reviewer then assesses whether the descriptions from Osprey and the Annotator refer to the same object. After this filtering process, we retain only 40% of samples to ensure accuracy. Qwen2-Instruct-7B model [45] is chosen as the Reviewer for this task, due to its efficiency and suitability for handling the complexity of this process.

Refiner for Summarization&Refinement. Finally, we utilize a reliable Refiner, GPT-4o [29], to summarize and refine the temporal and appearance-related captions generated by the annotator. This process aims to further eliminate repetition and hallucinations, ensuring a coherent and accurate final object-level instruction-following dataset.

###### 3.1.2 Data Characteristics

By leveraging our multi-agent data engine, we meticulously create three primary types of object-level video instruction data: detailed captions, short captions, and multi-round question-answer (QA) pairs.

Object-level Detailed Caption. We utilize a subset of large-scale Panda-70M [7], which has a short caption for each video. We generate 125K high-quality object-level detailed captions through our full multi-agent data engine.

Object-level Short Caption. To generate short captions, primarily for aligning object-level encoder with the LLM for pre-training, we employ a portion of the pipeline, which only includes the Analyzer and Segmentor. Specifically, in the Analyzer, we extract only singular subject nouns, enabling the reusing of raw captions for short descriptions. Using this approach, we produce 500K short captions.

Object-level QA. To generate instruction data that explicitly specifies particular objects or their relationships, we collect MeViS [10], Ref-YouTube-VOS [37] and A2DSentence datasets. Both provide reliable short descriptions with mask annotations for each object region. By utilizing these short descriptions and masked videos, we first employ

Annotator to generate object-level descriptions for each region, and then employ Refiner to generate QA pairs related to the objects within the videos, using a variety of prompts. Three types of region-based QA data have been created: (i) Basic Questions: These cover object types, attributes, actions, locations, and interactions over time. (ii) Reasoning Questions: These require reasoning and background knowledge to explain events without relying on specific visual details. (iii) Future Predictions: These involve anticipating future actions or events related to a given object. We generate 75K QA pairs in total.

###### 3.2. VideoRefer Model

###### 3.2.1 Overall Architecture

In this section, we introduce the VideoRefer framework, which ensures the next token predictions of Video LLM, enabling fine-grained mask-level comprehension at any specific regions and any timestamps for a given video. Given that the current Video LLM already exhibits strong general scene-level video understanding capabilities, we develop our model upon a well-established Video LLM, VideoLLaMA2.1 [9]. Our primary innovation is to introduce a versatile and unified spatial-temporal object encoder to obtain object-level representations across video scenes.

The overall architecture of our framework is illustrated in Fig. 3. VideoRefer adopts a visual encoder and STC connector [9] to encode the global scene-level visual representations, a pretrained text tokenizer to capture the language embeddings, and an instruction-following LLM for language decoding. To achieve video referring, we present a versatile and unified spatial-temporal encoder, denoted as REnc, to derive object-level representations. For a specific object Rj ∈ R, we define Rj = {rij | i ∈ T}, where each rij represents a unified 2D binary mask M designed to accommodate free-form input regions, assigning a value of 1 inside the region and 0 outside. The set of objects R, along with the image feature map Z extracted from the shared visual encoder, is then fed into the introduced object encoder REnc, which generates enriched objectlevel tokens, expressed as TR = REnc(R,Z). Finally, the interleaved scene-level tokens TZ, object-level tokens TR and linguistic tokens Tx are sent to the LLM to obtain the fine-grained semantic understandings Y , formulated as Y = Φ(TZ,TR,Tx), where Φ denotes the LLM.

###### 3.2.2 A Versatile Spatial-Temporal Object Encoder

To support various spatial-temporal video understanding tasks, our presented object encoder not only captures masklevel spatial features within the single frame at a specific timestamp, but also aggregates temporal information across multiple frames over a duration of time. Consequently, we devise two modes for our object encoder: single-frame and

[Figure 68]

###### Object Encoder

Single frame

- object1 and object2 are walking in the same direction, with
- object2 consistently trailing behind object1 throughout the entire sequence.

|[Figure 69]|[Figure 70]| |
|---|---|---|
| | | |

Spatial Token Extractor

TTM

###### MLP

###### VideoRefer

Feature Map &Mask

Multi frames

### Large Language Model

Temporal Token Merge

- t1 t2 t3 t5 … tn

- 1. Extract region tokens by Spatial Token Extractor

t1 t3

t4

t5

t6

… tn

- 2. Calculate similarity between adjacent tokens

t4 t6

STC Connector

Object Encoder

Embedding

Visual Encoder

Visual Encoder

Shared

How is <object1>’s position relate to <object2>’s?

t2

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

…

t4 t56 … tn

t123

Single frame &Mask Multi frames&Masks

Video frames

3. Merge the top n-r similar tokens

Figure 3. Model architecture of our VideoRefer for spatial-temporal video object understanding.

Merge Module, which is designed to effectively capture essential object-level tokens across the temporal dimension. Specifically, starting with spatial object tokens O ∈ Rk×C, we first compute the cosine similarity between each pair of adjacent tokens, formulated as:

multi-frame. For the sake of brevity for better illustration, we use a single object Rj as an example. If multiple objects are specified by the user, we adopt the same manner to extract features for each object individually.

Single-Frame. For single-frame mode, the input consists of a randomly selected frame along with the corresponding regions specified by the user in that frame. Here, T contains only a randomly chosen timestamp. To generate the object-level token representations, we present the Spatial Token Extractor. In detail, the image feature is initially extracted by the shared visual encoder to generate the global image feature FI ∈ R1×H

Om · Om+1 ∥Om∥ · ∥Om+1∥

,0 ≤ m<k. (2)

Sm,m+1 =

Subsequently, we select the top k−u similarity scores from S, where u is a predefined constant. The corresponding pairs of tokens are then merged into a single union, resulting in u unions. For each union, we apply straightfoward average pooling to produce a single distinct representative token. Ultimately, u tokens, represented as O ∈ Ru×C, are generated following an MLP layer for each object, ensuring both spatial integrity and temporal coherence without disrupting spatial structure.

I×WI×DI, where HI, WI, DI denote the height, width and dimension of the image feature, respectively. Each binary mask M of an object is then resized to match the shape of the image feature. We utilize the Mask Pooling operation upon image feature to extract object-level spatial feature FO ∈ R1×D

I for each mask, which pools all features within the region M to generate an object-level representation. Finally, an MLP layer is employed to adapt and produce the object-level token O ∈ R1×C for each object region.

###### 3.3. VideoRefer-Bench

To comprehensively evaluate the models’ capability on video-based regional comprehension, we have developed a benchmark named VideoRefer-Bench. This benchmark assesses the models in two key areas: Description Generation, corresponding to VideoRefer-BenchD, and Multiple-choice Question-Answer, corresponding to VideoRefer-BenchQ. Fig. 4 and Fig. 5 provide exemplar visual illustrations and data characteristics, respectively.

Multi-Frame. In the multi-frame mode, the input consists of a list of selected frames from the video, accompanied with their respective object regions, i.e., T contains a list of timestamps from the video. The frame-level feature is extracted using the shared visual encoder to generate the image feature FI ∈ Rk×H

I×WI×DI, where k represents the number of selected frames. We then employ the Spatial Token Extractor to generate the object-level tokens for each frame. Hence, we obtain the object tokens O ∈ Rk×C. To aggregate distinct temporal object-level representations across multiple frames over a time duration while minimizing redundant tokens, we propose the Temporal Token

###### 3.3.1 VideoRefer-BenchD

We introduce a sub-benchmark, VideoRefer-BenchD specifically designed to evaluate the description generation performance of video-based referring models. The bench-

Sequential Questions

Basic Questions

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

VideoRefer-BenchD GT Description: A middle-aged man wearing a suit and a red scarf walked over to talk to someone who looked like a superhero, and then left.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

3%

people animal transportation

[Figure 98]

[Figure 99]

3%

4%

[Figure 100]

[Figure 101]

[Figure 102]

- 24%
- 25%

26%

8%

[Figure 103]

9%

VideoRefer BenchQ

item

[Figure 104]

57%

[Figure 105]

[Figure 106]

|Eval: Multidimensional Evaluation by GPT|
|---|

[Figure 107]

environment

11%

Future Predictions

16%

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

###### VideoRefer-BenchQ

forniture

[Figure 118]

Sequential Question

[Figure 119]

[Figure 120]

VideoRefer BenchD

14%

Q: How does <object1> move?

Relationship Questions

[Figure 121]

others

[Figure 122]

Complex Questions

- (A) In a straight line
- (B) In a zigzag pattern
- (C) In circles
- (D) Randomly

[Figure 123]

(a) Category list in BenchD (b) Question types in BenchQ

Eval: Accuracy Calculation

Figure 5. Data characteristics of VideoRefer-Bench.

Figure 4. Exemplar visual illustration of VideoRefer-Bench.

##### 4. Experiments

mark comprises a total of 400 curated data entries. We curated the test set based on Panda-70M [7], employing the pipeline described in Section 3.1, followed by a meticulous human check. Furthermore, we developed an evaluation pipeline utilizing the GPT-4o model. This pipeline rigorously assesses various capabilities of the model by assigning scores to the generated predictions on a scale range from 0 to 5 across the following four dimensions:

###### 4.1. Implementation Details

We adopt siglip-so400m-patch14-384 [50] as the vision encoder, Qwen-2 [45] as the LLM. The AdamW [25] is used as the optimizer and the cosine annealing scheduler [26] is used to adjust learning rate. We use a hybrid strategy including both single-frame and multi-frame modes during training. We leverage a progressive training scheme, which consists of image-text alignment pretraining (Stage 1), region-text alignment pre-training (Stage 2), high-quality knowledge learning (Stage 2.5) and visual instruction tuning (Stage 3) stages, respectively. Please refer to the Appendix for detailed introduction to each stage. At the first and second stages, we set global batch size to 256 and learning rate to 1×10−3 for one epoch. In stage 2.5 and stage 3, the learning rate is reduced to 2×10−5 with a global batch size of 128 for one epoch. Unless otherwise specified, all models adopt the 7B LLM.

- • Subject Correspondence (SC): This dimension evaluates whether the subject of the generated description accurately corresponds to that specified in the ground truth.
- • Appearance Description (AD): This criterion assesses the accuracy of appearance-related details, including color, shape, texture, and other relevant visual attributes.
- • Temporal Description (TD): This aspect analyzes whether the representation of the object’s motion is consistent with the actual movements.
- • Hallucination Detection (HD): This facet identifies discrepancies by determining if the generated description includes any facts, actions, or elements absent from reality, like imaginative interpretations or incorrect inferences.

###### 4.2. Main Results

To evaluate the efficacy of our VideoRefer model, we conduct experiments on both video referring tasks and general video understanding tasks to demonstrate its capabilities.

###### 3.3.2 VideoRefer-BenchQ

###### 4.2.1 Video Referring Tasks

VideoRefer-BenchD. We compare our approach on VideoRefer-BenchD with the previous generalist models, including GPT-4o [29], GPT-4o-mini [29], InternVL2 [8], Qwen2-VL [45], LLaVA-OV [15], LongVA [54], LongVU [38] and specialist models for object-level understanding, including image-level Osprey [48], Ferret [46], and video-level Elysium [43], Artemis [33]. Both single-frame and multi-frame modes are adopted for evaluation. In the single-frame mode, we select the first frame that contains the specific object with its aligned boundary for the generalist models. For image-level region understanding models, we utilize a random frame along with the corresponding region prompt as input. In the multi-frame mode, we uniformly sample 16 frames with mask contours for generalist models. For image-level methods, we obtain the description frame by frame and then

The other sub-benchmark VideoRefer-BenchQ is designed to evaluate the proficiency of MLLMs in interpreting video objects. We meticulously curated a dataset comprising 198 videos sourced from various datasets, including DAVIS-2017 [32] and the test set of MeViS [10]. To facilitate a robust evaluation, we annotated a set of 1,000 high-quality multiple-choice questions. These questions are crafted to assess different dimensions of understanding, including Basic Questions, Sequential Questions, Relationship Questions, Reasoning Questions, and Future Predictions. The annotations were performed by researchers with extensive research experience in visionlanguage learning. Importantly, each QA pair is required to be explicitly linked to a specific video region. This ensures that the MLLMs cannot provide answers without actually analyzing the video or the designated object.

Single-Frame Multi-Frame SC AD TD HD Avg. SC AD TD HD Avg.

Method

Generalist Models LongVU-7B [38] 2.02 1.45 1.98 1.12 1.64 2.33 1.80 2.39 1.68 2.05 LongVA-7B [54] 2.63 1.59 2.12 2.10 2.11 3.02 2.30 1.92 2.51 2.44 LLaVA-OV-7B [15] 2.62 1.58 2.19 2.07 2.12 3.09 1.94 2.50 2.41 2.48 Qwen2-VL-7B [45] 2.97 2.24 2.03 2.31 2.39 3.30 2.54 2.22 2.12 2.55 InternVL2-26B [8] 3.55 2.99 2.57 2.25 2.84 4.08 3.35 3.08 2.28 3.20 GPT-4o-mini [29] 3.56 2.85 2.87 2.38 2.92 3.89 3.18 2.62 2.50 3.05 GPT-4o [29] 3.34 2.96 3.01 2.50 2.95 4.15 3.31 3.11 2.43 3.25 Specialist Models Image-level models

Ferret-7B [46] 3.08 2.01 1.54 2.14 2.19 3.20 2.38 1.97 1.38 2.23 Osprey-7B [48] 3.19 2.16 1.54 2.45 2.34 3.30 2.66 2.10 1.58 2.41 Video-level models

Elysium-7B [43] 2.35 0.30 0.02 3.59 1.57 – – – – – Artemis-7B [33] – – – – – 3.42 1.34 1.39 2.90 2.26 VideoRefer-7B 4.41 3.27 3.03 2.97 3.42 4.44 3.27 3.10 3.04 3.46

- Table 1. Performance comparisons on VideoRefer-BenchD. The best results are bold and the second-best results are underlined. “–” means that the model does not support the certain input form. Grey entries denote cases where the original method cannot accomplish the task; for these tests, masks of the targets were overlaid on the original video (the same below).

Basic Sequential Relationship Reasoning Future Method

Questions Questions Questions Questions Predictions

Average Generalist Models

LongVU-7B [38] 47.2 61.3 57.5 85.3 65.8 61.0 LongVA-7B [54] 56.2 62.5 52.0 83.9 65.8 61.8 InternVL2-26B [8] 58.5 63.5 53.4 88.0 78.9 65.0 GPT-4o-mini [29] 57.6 67.1 56.5 85.9 75.4 65.8 Qwen2-VL-7B [45] 62.0 69.6 54.9 87.3 74.6 66.0 LLaVA-OV-7B [15] 58.7 62.9 64.7 87.4 76.3 67.4 GPT-4o [29] 62.3 74.5 66.0 88.0 73.7 71.3

Specialist Models

Osprey-7B [48] 45.9 47.1 30.0 48.6 23.7 39.9 Ferret-7B [46] 35.2 44.7 41.9 70.4 74.6 48.8 VideoRefer-7B 75.4 68.6 59.3 89.4 78.1 71.9

- Table 2. Performance comparisons on VideoRefer-BenchQ. Note: Video-level specialist models, including Elysium [43] and Artemis [33], do not have the ability to handle multi-choice questions on VideoRefer-BenchQ.

generate a summary using GPT-4o. For Elysium [43] and Artemis [33], we adhere to the official settings provided in their respective papers. For our VideoRefer, we randomly select a single frame and uniformly sample 16 frames as inputs for the single-frame and multi-frame modes, respectively. Table 1 presents the comparison results. Our approach achieves the leading average performance in regional-temporal video understanding compared to previous methods in both single-frame and multi-frame modes. Notably, VideoRefer attains top scores of 4.41,

- 3.27, and 3.03 for Subject Correspondence (SC), Appearance Description (AD), and Temporal Description (TD) in single-frame mode, and scores of 4.44 and 3.04 for SC and Hallucination Detection (HD) in multi-frame mode. Fig. 6 illustrates a typical visual comparison.

VideoRefer-BenchQ. We then compare our VideoRefer against the previous methods on VideoRefer-BenchQ. Here, we set single-frame mode following settings of VideoReferBenchD. As shown in Table 2, our VideoRefer achieves the best average performance with 71.9, which significantly outperforms the previous regional methods. Especially, our approach excels in basic questions, relationship questions, reasoning questions and future predictions with 75.4, 59.3, 89.4 and 78.1 scores with best or second-best places, respectively. These results clearly demonstrate the superiority of our method in spatial-temporal video understanding.

Previous Video Referring Metrics. Following the previous state-of-the-art video referring approach, Artmis [33], we further conduct experiments on the test set of HCSTVG [40].

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

It is a small dog, specifically a French Bulldog. The dog has a stocky build, a broad chest, and a short, muscular body. Its face is characterized by a flat, wrinkled snout, and it has large, expressive eyes. The ears are erect and bat-like, typical of the breed. The coat appears to be short and smooth, with a light tan or fawn color and darker markings around the eyes and snout. The dog is seen on a bed, interacting with the person filming, and there is a laptop nearby. The setting appears to be a bedroom with a television and various pieces of furniture in the background.

[Figure 136]

5

4

###### GPT-4O

5

2 4

[Figure 137]

[Figure 138]

2

[Figure 139]

5

4

4

[Figure 140]

2

[Figure 141]

[Figure 142]

[Figure 143]

5 5 5 5

TD HD

SC AD

A white and brown dog is sitting on Artemis the bed and looking at the camera.

Elysium french bulldog.

GPT4oArtemisElysiumVideoRefer

[Figure 144]

[Figure 145]

[Figure 146]

The French Bulldog in the video is small and has a white coat with brown patches. It is sitting on a bed next to a person who is using a laptop. The dog appears to be calm and is looking around, occasionally moving its head.

[Figure 147]

###### VideoRefer

- Figure 6. Visual comparisons between our VideoRefer with general GPT-4o and regional video-level Elysium and Artemis. Here we provide detailed illustrations on VideoRefer-BenchD.

VideoRefer-BenchD VideoRefer-BenchQ TD HD Avg. SQ RQ Avg. Single-frame 3.03 2.97 3.42 68.3 59.1 71.9 Multi-frame 3.10 3.04 3.46 70.6 60.5 72.1

Method BLEU@4 METEOR ROUGE L CIDER SPICE

Mode

Merlin [47] 3.3 11.3 26.0 10.5 20.1 Artemis [33] 15.5 18.0 40.8 53.2 25.4 VideoRefer 16.5 18.7 42.4 68.6 28.3

- Table 3. Exprimental results on video-based referring metrics on the HC-STVG [40] test set.

Method Perception-Test MVBench VideoMME

VideoLLaMA2 [9] 51.4 54.6 47.9/50.3 VideoLLaMA2.1 [9] 54.9 57.3 54.9/56.4 Artemis [33] 47.1 34.1 28.8/35.3 VideoRefer 56.3 59.6 55.9/57.6

- Table 4. Exprimental results on general video understanding tasks.

Table 5. Results using different modes during the inference. Here, SQ and RQ are Sequential Questions and Relationship Questions.

###### 4.3. Ablation Study

Single-frame vs. Multi-frame. We first validate the impacts on the single-frame and multi-frame modes, i.e. with or without Temporal Token Merge (TTM) module to encode the multi-frame sequences during the inference. As shown in Table 5, our approach utilizing multi-frame mode exhibits improvements over the single-frame mode in both VideoRefer-BenchD and VideoRefer-BenchQ across all metrics. Notably, for sequential relation-based metrics, including Temporal Description (TD), Sequential Questions (SQ), and Relationship Questions (RQ), as well as hallucination-related metrics such as Hallucination Detection (HD), multi-frame mode showcases the superiority.

Table 3 presents the comparison results. Our approach outperforms Artmis [33] by +1.0%, +0.7%, +1.6%, +15.4%, and +2.9% on BLEU4 [30], METEOR [3], ROUGE L [19], CIDEr [42] and SPICE [1] metrics. These results demonstrate the superiority of our VideoRefer.

Ablation on VideoRefer-700K Dataset. Table 6 summarizes the ablation results for various data types in the constructed VideoRefer-700K dataset. The results indicate that using a short description yields a score of 2.43 on BenchD and 68.3 on BenchQ, along with an MVBench score of 58.0. Incorporating question-answering (QA) data improves the performance to 2.45 for BenchD and 71.7 for BenchQ, while maintaining an MVBench score of 58.4. Notably, the method employing detailed descriptions achieves the best results, with scores of 3.42 on BenchD, 71.9 on BenchQ, and 59.6 on MVBench. These results demonstrate that the inclusion of more comprehensive data significantly enhances overall performance.

###### 4.2.2 General Video Understanding

To demonstrate the capabilities of our method, we conduct performance evaluation on general video understanding tasks. As shown in Table 4, VideoLLaMA2.1 [9] achieves scores of 54.9% on Perception-Test [31], 57.3% on MVBench [16], and 54.9%/56.4% on VideoMME [12]. Based on that, our VideoRefer exhibits performance gains of +1.4%, +2.3%, and +1.0%/+1.2%, respectively. In contrast, Artemis demonstrates subpar performance. These results clearly indicate that our approach not only excels in object-level analysis, but also enhances the ability of general video understanding.

###### Impacts of Different Union Numbers in TTM. The

###### Method BenchD BenchQ MVBench

- 0 w/o Regional data – – 57.9

- 1 + Short description 2.43 68.3 58.0
- 2 + QA 2.45 71.7 58.4

- 3 + Detailed description 3.42 71.9 59.6

- Table 6. Ablation results on various data types in VideoRefer700K dataset. Bench denotes VideoRefer-Bench for simplicity.

Union u

VideoRefer-BenchD VideoRefer-BenchQ

TD HD SQ RQ 32 3.17 3.01 68.7 58.1 16 3.20 2.99 69.3 58.5

8 3.18 3.02 69.6 57.8 4 3.10 3.04 70.6 60.5 1 3.08 2.98 68.9 60.9

- Table 7. Temporal and sequential performance comparisons for various union u in the TTM module under multi-frame mode.

Temporal Token Merge (TTM) Module is designed to capture essential object-level tokens across the temporal dimension in multi-frame mode. Fig. 7 visualizes the similarity scores between adjacent object token pairs. It is evident that most adjacent tokens exhibit high similarity, making it necessary to merge those tokens with significant similarity. We conducted ablation experiments using temporal and sequential metrics to investigate the effects of varying numbers of token unions u. The experimental results are detailed in Table 7. Notably, with u = 4, VideoRefer achieves the best performance in Hallucination Detection (HD) and Sequential Questons (SQ), and ranks second in Reasoning Questions (RQ). We adopt u = 4 to strike a balance between performance and token costs in our approach.

##### 5. Related Works

###### 5.1. Video Large Language Models

Large Language Models (LLMs) have revolutionized the field of artificial intelligence by proving their capability to tackle diverse tasks related to language comprehension and generation. To fully leverage the potential of LLMs for visual understanding, researchers have increasingly turned their attention to image-based Multimodal Large Language Models (MLLMs) [2, 8, 17, 20–23, 28, 56], which integrate language and visual data within a unified feature space. This integration has emerged as a significant area of research focus. In parallel, Video Large Language Models (Video LLMs) [9, 18, 27, 52, 57] have garnered increasing attention fueled by advancements in image-based MLLMs. Most Video LLMs primarily follow the trend of utilizing pre-trained vision models to extract sequence-based information from videos, which is then interleaved with textual embeddings for LLM to generate responses [39]. Despite

their promising results, current Video LLMs still face challenges in fine-grained regional and temporal understanding.

###### 5.2. Regional Understanding with MLLMs

To attain fine-grained regional object-level comprehension, MLLMs can be incorporated with instance-level visual representations. This integration allows models to generate semantic understandings that focus on specific regions. In the context of image-based MLLMs, recent researchs [4– 6, 11, 13, 34, 41, 44, 46, 48, 49, 51, 53, 55, 58] has demonstrated a significant trend to enable the image referring with spatial visual prompts. In contrast, research focused on video-based regional understanding across dynamic sequence-based scenes is relatively limited. Merlin [47] first explored video-based referring and future reasoning by employing three manually selected frames as visual input, which limits the model’s ability to comprehend longer and more intricate scenes. Elysium [43] introduces a million-scale dataset for object-level tasks in videos; however, the provided descriptions tend to be quite simplistic. Another reseach work is Artemis [33], but it primarily emphasizes basic single object descriptions, thereby constraining its capacity to analyze relationships among various objects or perform more complex tasks on specific objects within dynamic sequences. Moreover, Artemis utilizes a sparse bounding box representation, which inadequately captures the nuances of the objects. Compounding these challenges is the lack of large-scale, high-quality regionlevel video instruction data and benchmarks for thorough evaluation, which further hampers progress in this domain. To address these issues, we introduce the VideoRefer Suite to advance spatial-temporal understanding.

##### 6. Conclusion

In this work, we introduced the VideoRefer Suite to empower Video LLM for fine-grained spatial and regional video understanding. Three key components have been proposed: 1) VideoRefer-700K: A large-scale, high-quality region-level video instruction data curated by a developed multi-agent engine; 2) VideoRefer: A Video LLM equipped with a versatile spatial-temporal object encoder that includes a Spatial Token Extractor and an adaptive Temporal Token Merge Module to enabling precise sequential regional representation; and 3) VideoRefer-Bench: a comprehensive benchmark that thoroughly evaluates model performance across multiple aspects, ensuring a holistic assessment of spatial-temporal capabilities. Extensive experimental results and analyses have demonstrated the efficacy of our VideoRefer Suite, substantially advancing finer-level video understanding and analysis.

##### References

- [1] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In ECCV, pages 382–398, 2016. 8
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv:2308.12966, 2023. 1, 9
- [3] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In ACL Workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pages 65–72, 2005. 8
- [4] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In CVPR, pages 12914–12923, 2024. 3, 9
- [5] Chi Chen, Ruoyu Qin, Fuwen Luo, Xiaoyue Mi, Peng Li, Maosong Sun, and Yang Liu. Position-enhanced visual instruction tuning for multimodal large language models. arXiv preprint arXiv:2308.13437, 2023.
- [6] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 3, 9
- [7] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In CVPR, pages 13320–13331,

2024. 4, 6

- [8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 1, 4, 6, 7, 9
- [9] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 1, 4, 8, 9, 12, 13
- [10] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. Mevis: A large-scale benchmark for video segmentation with motion expressions. In ICCV, pages 2694–2703, 2023. 4, 6
- [11] Hao Fei, Shengqiong Wu, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. In NeurIPS, 2024. 3, 9
- [12] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 8
- [13] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei

- Liu. Regiongpt: Towards region understanding vision language model. In CVPR, pages 13796–13806, 2024. 3, 9
- [14] Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, and Fisher Yu. Segment anything in high quality. In NeurIPS, 2023. 4
- [15] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6, 7
- [16] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, pages 22195–22206, 2024. 1, 8
- [17] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. arXiv preprint arXiv:2407.02392, 2024. 9
- [18] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2023. 1, 9
- [19] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 8
- [20] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, pages 26689–26699, 2024. 1, 9
- [21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.
- [22] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pages 26296–26306, 2024. 13
- [23] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge. https: //llava-vl.github.io/blog/2024-01-30llava-next/, 2024. 1, 9
- [24] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024. 4
- [25] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In ICLR, 2017. 6
- [26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 6
- [27] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL,

2024. 1, 9

- [28] OpenAI. Gpt-4v(ision) system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf,

2023. 1, 9

- [29] OpenAI. Hello gpt-4o. https://openai.com/ index/hello-gpt-4o/, 2024. 4, 6, 7
- [30] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, pages 311–318, 2002. 8

- [31] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. In NeurIPS, 2024. 8
- [32] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 6
- [33] Jihao Qiu, Yuan Zhang, Xi Tang, Lingxi Xie, Tianren Ma, Pengyu Yan, David Doermann, Qixiang Ye, and Yunjie Tian. Artemis: Towards referential understanding in complex videos. In NeurIPS, 2024. 1, 3, 6, 7, 8, 9
- [34] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In CVPR, pages 13009–13018, 2024. 3, 9
- [35] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 2, 4
- [36] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1
- [37] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In ECCV, pages 208–223. Springer,

2020. 4

- [38] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 6, 7
- [39] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. arXiv preprint arXiv:2312.17432, 2023. 9
- [40] Zongheng Tang, Yue Liao, Si Liu, Guanbin Li, Xiaojie Jin, Hongxu Jiang, Qian Yu, and Dong Xu. Human-centric spatio-temporal video grounding with visual transformers. TCSVT, 32(12):8238–8249, 2021. 7, 8
- [41] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. Chatterbox: Multi-round multimodal referring and grounding. arXiv preprint arXiv:2401.13307, 2024. 3, 9
- [42] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, pages 4566–4575, 2015. 8
- [43] Han Wang, Yongjie Ye, Yanjie Wang, Yuxiang Nie, and Can Huang. Elysium: Exploring object-level perception in videos

- via mllm. In ECCV, pages 166–185. Springer, 2024. 1, 3, 6, 7, 9
- [44] Shiyu Xuan, Qingpei Guo, Ming Yang, and Shiliang Zhang. Pink: Unveiling the power of referential comprehension for multi-modal llms. In CVPR, pages 13838–13848, 2024. 3, 9
- [45] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 3, 4, 6, 7
- [46] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. In ICLR, 2024. 1, 3, 6, 7, 9
- [47] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. In ECCV, pages 425–443. Springer,

2025. 1, 3, 8, 9

- [48] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In CVPR, pages 28202–28211, 2024. 1, 3, 4, 6, 7, 9
- [49] Tongtian Yue, Jie Cheng, Longteng Guo, Xingyuan Dai, Zijia Zhao, Xingjian He, Gang Xiong, Yisheng Lv, and Jing Liu. Sc-tune: Unleashing self-consistent referential comprehension in large vision language models. In CVPR, pages 13073–13083, 2024. 3, 9
- [50] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, pages 11975–11986, 2023. 6
- [51] Yufei Zhan, Yousong Zhu, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon v2: Advancing multimodal perception with high-resolution scaling and visual-language co-referring. arXiv preprint arXiv:2403.09333, 2024. 3, 9
- [52] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1, 9
- [53] Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, et al. Ferretv2: An improved baseline for referring and grounding with large language models. arXiv preprint arXiv:2404.07973,

2024. 1, 3, 9

- [54] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 6, 7
- [55] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on regionof-interest. arXiv preprint arXiv:2307.03601, 2023. 1, 3, 9
- [56] Wenqiao Zhang, Tianwei Lin, Jiang Liu, Fangxun Shu, Haoyuan Li, Lei Zhang, He Wanggui, Hao Zhou, Zheqi Lv, Hao Jiang, et al. Hyperllava: Dynamic visual and language expert tuning for multimodal large language models. arXiv preprint arXiv:2403.13447, 2024. 9

- [57] Yuanhan Zhang, Bo Li, Haotian Liu, Yong Jae, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model. https: //llava-vl.github.io/blog/2024-04-30llava-next-video/, 2024. 1, 9
- [58] Liang Zhao, En Yu, Zheng Ge, Jinrong Yang, Haoran Wei, Hongyu Zhou, Jianjian Sun, Yuang Peng, Runpei Dong, Chunrui Han, et al. Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning. arXiv preprint arXiv:2307.09474, 2023. 3, 9

Appendix

- A. More Qualitative Results

We provide additional visualization results to emphasize performance across a variety of tasks, such as single-object referring, video relationship analysis, complex reasoning, future prediction, and video object retrieval. Besides, we present the examplar cases to demonstrate the capabilities in general video understanding and image object understanding. Fig. 12 showcases these visual examples. A randomly selected mask along with its corresponding frame is used as the region input.

- B. Additional Implemental Details B.1. Training Stages

The training pipeline of our model is structured into four distinct stages. Fig. 8 presents the data distribution for each stage.

###### Stage 1: Image-Text Alignment Pre-training. In this

initial pre-training phase, we utilize the same dataset as employed in the first stage of VideoLLaMA2.1 [9]. During this phase, the parameters of both the vision encoder and the large language model are frozen, and training is conducted solely on the STC connector [9], enabling the alignment of image and text modalities.

###### Stage 2: Region-Text Alignment Pre-training. This

stage further incorporates the Object Encoder to capture object-level features based on the weights obtained from

- Stage 1. The training focus is exclusively on the spatialtemporal Object Encoder to ensure the alignment of intricate object-level features with corresponding language embeddings. We use the generated 500K region-level short descriptions, along with video and image referring segmentation datasets as the training data. During this stage, all the data are processed in single-frame mode to focus solely on alignment.
- Stage 2.5: High-Quality Knowledge Learning. At

this intermediate stage, the weights of vision encoder remain frozen, while the STC connector, Object Encoder, and LLM undergo fine-tuning. This stage aims to infuse the model with high-quality captioning data, utilizing a diverse dataset that includes 118K image-caption pairs, 30K videocaption pairs, 79K image-level region caption data, and 125K video-level region caption data, inclusive of the detailed descriptions we curated. For object-level video data, we employ a balanced approach, using half in single-frame mode and half in multi-frame mode.

Stage 3: Visual Instruction Tuning. The training configuration for this stage closely mirrors that of Stage 2.5. The primary objective is to enhance the model’s ability to accurately interpret user instructions and tackle complex

[Figure 148]

[Figure 149]

[Figure 150]

0 3 8 12 15 19 22 26 30

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

0 32

token similarity

0.93 0.94 0.95 0.96 0.97 0.98 0.99

[Figure 160]

0 4 8 13 17 21 25 26 30

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

0 32

token similarity

0.7 0.75 0.8 0.85 0.9 0.95

- Figure 7. Visualizations of similarity among adjacent object-level token pairs across the temporal dimension. Here, we use cosine similarity as the measurement.

- Stage2: Region-Text Alignment Pre-training

|511K<br><br>[Figure 170]|180K<br><br>[Figure 171]|
|---|---|

Stage2.5: High-Quality Knowledge Learning

|[Figure 172]<br><br>30K|125K<br><br>[Figure 173]|[Figure 174]<br><br>125K|79K<br><br>[Figure 175]|
|---|---|---|---|

- Stage3: Visual Instruction Tuning

|[Figure 176]<br><br>478K|115K<br><br>[Figure 177]|523K<br><br>[Figure 178]|394K<br><br>[Figure 179]|
|---|---|---|---|

[Figure 180]

Video Image Video-referring Image-referring

[Figure 181]

[Figure 182]

[Figure 183]

|[Figure 184]<br><br>10M|[Figure 185]<br><br>600K|
|---|---|

Stage 1: Image-Text Alignment Pre-training

- Figure 8. Visual illustrations of the data distribution for each training stage.

40K

500K

VideoRefer -700K

21K

125K

14K

Figure 9. Data distributions of our VideoRefer-700K dataset, encompassing five different data types.

stage.

##### C. More Details of VideoRefer-700K Dataset and Benchmark

object-level understanding tasks. For video-level data, we utilize the same dataset segments as those used in VideoLLaMA2.1 [9]. For image-level data, we employ the datasets from LLaVA [22]. In addition, we incorporate 294K imagelevel region data and 115K previously constructed videolevel region data to further strengthen the model’s capabilities. We also employ a balanced approach using half in single-frame mode and half in multi-frame mode in this

###### C.1. Human Evaluation on Reviewer

In our muliti-agent data engine, we introduce the Reviewer to address potential errors and mismatches, thereby ensuring the quality of our VideoRefer-700K dataset. To assess the effectiveness of the Reviewer, we conducted a manual

|[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>The young man is focused on a piece of machinery, using a screwdriver with steady precision. His short, light brown hair complements the light gray, button-up shirt with rolledup sleeves and a dark gray tie he wears.<br><br>[Figure 189]<br><br>[Figure 190]<br><br>Annotator<br><br>a man working on a machine<br><br>[Figure 191]<br><br>Osprey<br><br>[Figure 192]<br><br>Reviewer Human<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>TP|[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>The person is holding a black card labeled "Surface" and placing it on a white box while wearing a black shirt. Their light skin tone and short, dark hair are visible as they repeatedly pick up the card and a white paper, placing them back on the white box in a methodical manner.<br><br>[Figure 200]<br><br>[Figure 201]<br><br>Annotator<br><br>Osprey a hand holding a paper<br><br>[Figure 202]<br><br>[Figure 203]<br><br>Reviewer Human<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]<br><br>FP|
|---|---|
|The young man is seated in the driver's seat of a car, wearing dark sunglasses and a white t-shirt. He holds a black object in his right hand, likely a smartphone or camera, while gesturing with his left hand as he speaks.<br><br>[Figure 208]<br><br>[Figure 209]<br><br>Annotator<br><br>person wearing black pants<br><br>[Figure 210]<br><br>Osprey<br><br>[Figure 211]<br><br>Reviewer Human<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>FN<br><br>|[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>The woman is wearing a blue T-shirt, seated at a table on a patio, holding a jar of pickles with one hand while using a spoon to scoop out pickles. She places the pickles onto a sandwich on the plate in front of her.<br><br>[Figure 222]<br><br>[Figure 223]<br><br>Annotator<br><br>man wearing a blue shirt<br><br>[Figure 224]<br><br>Osprey<br><br>[Figure 225]<br><br>Reviewer Human<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]<br><br>TN|

Figure 10. Visual illustrations of human check process. TP, TN, FP and FN are introduced for the assessment on Reviewer.

[Figure 230]

## 🐼 Panda-70M

Video caption A man is fishing on a boat and catching a large fish. Video clips

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

###### Step1: Noun extraction

Subject: [man], Others: [ boat, fish]

[Figure 237]

Analyzer

###### Step3: Mask Generation

###### Step2: object-level Caption generation

[Figure 238]

[Figure 239]

InternVL2

Prompt: Describe the man’s … in detail…

[Figure 240]

Choose a random frame

[Figure 241]

Detailed description on Action/Movement

Grounding-DINO HQ-SAM

Annotator

Segmentor

The man in the video is initially standing on the edge of a boat, holding a fishing rod. He is wearing a camouflage shirt and a blue baseball cap. The man is focused on the water, where a fish is jumping and splashing. He then bends down towards the water, reaching out to grab the fish. The fish is struggling, causing water to splash around. The man successfully catches the fish and lifts it out of the water. He then holds the fish up, showing it to the camera. Throughout this process, the man's movements are deliberate and focused, with significant changes in motion as he reaches for the fish and lifts it.

[Figure 242]

|[Figure 243]|
|---|

man

Detailed description on Appearance

[Figure 244]

[Figure 245]

SAM2

[Figure 246]

[Figure 247]

The man in the video is wearing a camouflage-patterned long-sleeve shirt, gray shorts, and a blue baseball cap. He has a beard and is wearing sunglasses.

- Step4: Correspondence Verification Yes/No

The man is standing on the edge of a boat, wearing a camouflage-patterned long-sleeve shirt, gray shorts, and a blue baseball cap. With a focused expression, he watches the water as a fish jumps and splashes nearby. He bends down, reaching out to grab the struggling fish, causing water to splash around him. Successfully catching it, he lifts the fish out of the water and holds it up to show the camera, his deliberate movements reflecting his concentration on the task.

- Step5: Summarization & Refinement Refiner

[Figure 248]

[Figure 249]

Image-level region short description

[Figure 250]

[Figure 251]

The man is wearing a camouflage.

Osprey

[Figure 252]

Reviewer

[Figure 253]

GPT-4o

[Figure 254]

Figure 11. A detailed illustrative example of the construction pipeline in our multi-agent data engine.

evaluation of its outputs. We define the evaluation metrics as follows:

as relevant and accurate, which are confirmed to be true upon manual inspection.

• TN (True Negatives): Items that the Reviewer discarded

• TP (True Positives): Items that the Reviewer identified

Manually True Manually False

Reviewer True 88 (TP) 12 (FP) Reviewer False 36 (FN) 64 (TN)

- Table 8. Confusion matrix of the randomly sampled 100 items in the Reviewer evaluation.

as irrelevant or inaccurate, which are indeed false according to the manual check.

- • FP (False Positives): Items that the Reviewer considered as true, but are found to be false during manual verification.
- • FN (False Negatives): Items that the Reviewer discarded as false, but are actually true upon manual review.

We randomly sampled 100 items each from both the data discarded and retained by the Reviewer. The detailed results are represented in Table 8, and the corresponding metrics are calculated as follows:

##### D. Limitations

In this work, our VideoRefer is designed on object-level spatial-temporal video understanding, without the abilities on grounding. This limitation may affect the applicability of our method in real-world scenarios, which requires identifying and associating objects within their dynamic contexts. In the future work, we will address this gap by integrating grounding abilities into our framework, extending our dataset and benchmark to improve the system’s overall utility in practical applications.

TP + TN TP + TN + FP + FN

= 0.76, (3)

Accuracy =

TP TP + FP

Precision =

= 0.88, (4)

TP TP + FN

Recall =

= 0.71, (5)

Precision × Recall Precision + Recall

F1 Score = 2 ×

= 0.79. (6)

The precision value stands at 88%, indicating that the majority of samples identified as positive by the reviewer are indeed positive, thereby ensuring the data’s quality.

###### C.2. Example Illustrations

We provide a typical example to better exhibit the construction pipeline of our multi-agent data engine, as shown in Fig. 11. Additionally, the data distribution of our VideoRefer-700K dataset is illustrated in Fig. 9. Fig. 13 further showcases the additional visual samples from the VideoRefer-700K dataset.

###### C.3. More Benchmark Visualization

We present more visualizations of our benchmark, VideoRefer-BenchD and VideoRefer-BenchQ, as shown in Fig. 14. These visualizations aim to provide a deeper understanding of benchmarks’ structure and content. VideoReferBenchD focuses on detailed description tasks, facilitating the analysis of nuanced object references and relationships within videos. Meanwhile, VideoRefer-BenchQ is designed for question-and-answer scenarios, capturing the essence of interactive video comprehension.

[Figure 255]

Single Object Referring

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Describe the <object> in the video.

Describe the <object> in the video.

[Figure 280]

[Figure 281]

The little girl with dark skin and curly hair is wearing a pink shirt, paired with light blue pants. She sits on a pink mat, her legs spread apart in a split position, while another child stands nearby.

###### The horse in the video is brown with a black mane. It is standing still and appears calm. The horse is wearing a

VideoRefer

VideoRefer bridle and is being held by a person wearing a gray shirt.

[Figure 282]

Video Relationship Analysis

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

How does the movement of <object1> relate to <object2>?

How does <object2> interact with <object1>?

[Figure 311]

[Figure 312]

<object1> is the bicycle being ridden by <object2>, who is the person wearing a black and yellow helmet. The movement of the bicycle is directly controlled by the rider, indicating that the rider's actions are responsible for the bike's motion.

<object2> is dancing with <object1>, who is wearing a pink dress. They are moving in sync, and their dance appears to be part of a larger performance or celebration.

VideoRefer VideoRefer

[Figure 313]

Complex Reasoning

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

What can be inferred about <object1>’s role in the scene? Is the behavior of <object1> challenging?

[Figure 333]

[Figure 334]

<object1> appears to be a store employee, as indicated by his uniform and the context of the scene, which takes place

Yes, the behavior of <object1> is challenging as it involves navigating a rocky terrain with a bicycle.

VideoRefer in a retail environment. VideoRefer

[Figure 335]

[Figure 336]

[Figure 337]

Future Prediction

Video Object Retrival

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

What is a likely future event of <object1>?

What is the person <object> doing in the video?

[Figure 352]

[Figure 353]

A likely future event for <object1> is that it will continue to fly away from the grassy area towards the water, possibly

This person is Kobe, wearing a red hooded jacket. He initially stood on stage holding a microphone for an interview with the host, then sat down on a chair to begin speaking.

[Figure 354]

VideoRefer

VideoRefer landing on the riverbank or in the water.

[Figure 355]

[Figure 356]

General Video Understanding Image Object Understanding

[Figure 357]

[Figure 358]

[Figure 359]

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

Describe the video in detail.

[Figure 373]

Describe the <object> in the image.

[Figure 374]

In the video, a lioness is seen lying down with her two cubs. She is grooming them by licking their fur. The cubs are sitting close to her, enjoying the attention. The lioness is focused on cleaning and caring for her young ones.

[Figure 375]

The red sports car is a Ferrari. It is parked in a lot, and its door is open.

VideoRefer

VideoRefer

- Figure 12. Visualization results of VideoRefer across various tasks, including single-object referring, video relationship analysis, complex reasoning, future prediction, video object retrieval, as well as general video understanding and image object understanding.

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

A black bird is eating something on the grass. A guy on the left is doing karate.

- (a) Samples from our VideoRefer-700K dataset (Short description)

Question: <video>\nPlease describe the object <region> in the video in brief.

- (b) Samples from our VideoRefer-700K dataset (Detailed description)

Question: <video>\nPlease describe in detail the object <region> in the video.

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

The motorcyclist is adjusting his black helmet with a visor, ensuring it fits securely as he prepares for his ride. He wears a red t-shirt emblazoned with “CANADA” and a white maple leaf, complementing his youthful appearance with short, dark hair. After making slight adjustments, he lifts the helmet off, revealing his face and smiling at the camera.

The man in a black shirt and blue jeans stands at a podium before walking towards the audience. With long black hair tied back in a ponytail, he bends down to shake hands with a woman, then stands up and continues to move through the group of people.

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

Question: What is the primary action being performed by <object0>?

Question: What is the relationship between <object0> and <object1>?

<object0> is climbing up towards the crib, using it for support as he ascend.

<object0> is the child being supported by <object1>, the adult in black, who is helping the child learn to walk in the corridor.

Question: How does the position of <object1> change over time?

Question: How does <object0> maintain balance while walking?

<object1> remains stationary throughout the sequence, consistently holding onto the wooden bars of the crib without any significant change in position or activity.

<object0> maintains balance by occasionally touching the wall for support as she walks forward.

(c) Samples from our VideoRefer-700K dataset (QA)

- Figure 13. Visual samples from our VideoRefer-700 dataset, typical including short descriptions, detailed descriptions, and QA pairs.

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

VideoRefer-BenchD

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

…

… …

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

The little girl in the video has short light brown hair and is wearing a light blue dress with a floral pattern. Behind her is a man and a woman who appear to be her mom and dad. She looks around with a curious expression.

The piglet in the video has a predominantly white coat with some gray patches. Its body is small and round, its legs are short, and it has pink ears. It walks back and forth in front of a cage with chickens in it and jumps around excitedly on a white cloth.

The ambulance in the video is white with a blue and yellow stripe. It is parked near the school building, and there are people gathered around it. The ambulance is stationary and not in motion.

[Figure 420]

[Figure 421]

person animal transportation

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

VideoRefer-BenchQ

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

…

…

…

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

Question: What is <object1> doing in the video?

Question: How many times did <object1> kick the ball?

Question: What is <object2>?

- (A) Sitting on the table and moving
- (B) Being held by a person's hand and placed on the scale
- (C) Running around the room
- (D) Sleeping on the table

- (A) One
- (B) Two
- (C) Three
- (D) Four

- (A) A piece of paper
- (B) A plate
- (C) A phone
- (D) A cup

[Figure 445]

Basic Questions Sequential Questions Sequential Questions

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

…

… …

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

Question: What might be a reason for <object2> walking by <object1>?

Question: What is the relative position of <object3> to <object1> at the beginning of the video?

Question: What will <object1> do next?

- (A) <object1> will continue going straight
- (B) <object1> will turn around and walk back
- (C) <object1> will take a seat on the bench
- (D) <object1> will stop and interact with someone

- (A) <object2> is her pet providing companionship
- (B) <object2> is a stray dog looking for food
- (C) <object2> is being walked by another person
- (D) <object2> is lost and trying to find its way home

- (A) <object3> is to the right of <object1>
- (B) <object3> is to the left of <object1>
- (C) <object3> is behind <object1>
- (D) <object3> is in front of <object1>

[Figure 466]

[Figure 467]

[Figure 468]

Relationship Questions Reasoning Questions Future Predictions

Figure 14. Visual examples of our VideoRefer-Bench, including VideoRefer-BenchD and VideoRefer-BenchQ.

