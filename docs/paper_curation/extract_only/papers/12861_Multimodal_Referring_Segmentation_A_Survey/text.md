## Multimodal Referring Segmentation: A Survey

Henghui Ding, Song Tang, Shuting He, Chang Liu, Zuxuan Wu, Yu-Gang Jiang, Fellow, IEEE

### arXiv:2508.00265v2[cs.CV]5Aug2025

Abstract—Multimodal referring segmentation aims to segment target objects in visual scenes, such as images, videos, and 3D scenes, based on referring expressions in text or audio format. This task plays a crucial role in practical applications requiring accurate object perception based on user instructions. Over the past decade, it has gained significant attention in the multimodal community, driven by advances in convolutional neural networks, transformers, and large language models, all of which have substantially improved multimodal perception capabilities. This paper provides a comprehensive survey of multimodal referring segmentation. We begin by introducing this field’s background, including problem definitions and commonly used datasets. Next, we summarize a unified meta architecture for referring segmentation and review representative methods across three primary visual scenes, including images, videos, and 3D scenes. We further discuss Generalized Referring Expression (GREx) methods to address the challenges of real-world complexity, along with related tasks and practical applications. Extensive performance comparisons on standard benchmarks are also provided. We continually track related works at https://github.com/henghuiding/Awesome-Multimodal-Referring-Segmentation.

Index Terms—Survey, Multimodal Referring Segmentation, Referring Expression Segmentation, Referring Video Object Segmentation, Referring Audio-Visual Segmentation, 3D Referring Expression Segmentation, Multimodal Learning, Vision-Language

✦

1 INTRODUCTION

|[Figure 1]|
|---|

|[Figure 2]|
|---|

###### Text:

[Figure 3]

# M

ULTIMODAL referring segmentation [1], [2], [3], [4], [5], [6], [7], [8], [9] aims to segment the target object in a

“The young man in the red hat and shirt smiles widely at the older man.”

visual scene of image [3], [4], [7], video [1], [2], [10], or 3D [5], [9], [11] according to a referring expression, such as free-form text or audio. For example, as shown in Fig. 1(b), given the text referring expression “The bird flying away”, the model is expected to segment and track the described target object in the video. This task presents a fundamental and challenging problem in multimodal understanding and supports a wide range of practical applications, such as image/video editing [12], [13], robotics [14], autonomous driving [15], etc. Because of its significant potential in practical applications, multimodal referring segmentation has received growing attention in recent years, as shown in Fig. 3.

Image

(a) RES

[Figure 4]

[Figure 5]

Text / Audio:

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

“The bird flying away.”

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

time

time

Video

(b) RVOS

Omnimodal:

###### / /

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

“The one that sounds like <sound: > , played by someone like <image: >.”

[Figure 26]

[Figure 27]

[Figure 28]

time

time

Video w/ Audio (c) OmniAVS

Text:

[Figure 29]

[Figure 30]

[Figure 31]

“A blue bed in the room, poisoned directly opposite the sofa.”

Segmentation [16], [17], [18] is one of the fundamental tasks in computer vision, forming the basis for many visual understanding tasks and applications [19]. Classic segmentation methods, e.g., semantic segmentation [16] and instance segmentation [17], typically segment the given visual scenes into a set of predefined categories. Although open-vocabulary segmentation [20] expands the category coverage, it remains reliant on explicit category names, e.g., person and car. Different from these classical segmentation tasks, referring segmentation enables more flexible and userfriendly segmentation by leveraging free-form referring expressions to identify specific target objects within a scene. A referring expression is a human-understandable linguistic construct used to describe an object in any way that uniquely and unambiguously identifies it. Such expressions are not limited to naming object categories. They may refer to the target object’s position, visual attributes, motion, or relationships with other objects. As long as the expression leads to an unambiguous identification of the target, any descriptive strategy is considered valid. This high degree of expressive freedom introduces considerable challenges for fine-

(d) 3D-RES

3D

Referring Expression Scene

Output

###### Fig. 1: Multimodal Referring Segmentation.

grained multimodal understanding and alignment. It also raises requirements for model robustness against diverse expression styles and linguistic-visual variations. Depending on the modality of the referring signal (e.g., text or audio) and the type of visual scene (e.g., image, video, auditory video, or 3D), referring segmentation can be further categorized into different tasks, as shown in Fig. 1.

Despite the inherent homogeneity shared across different referring segmentation tasks, most existing surveys [26], [27], [28], [29], [30] remain limited in scope, often concentrating on isolated modalities or specific tasks. For example, a recent survey [31] focuses exclusively on referring expression segmentation within 2D images, while neglecting extensions to video and 3D scenes. As a result, a critical gap remains in the literature due to the absence of a comprehensive survey that systematically covers the diverse task formulations, input modalities, and challenges within referring segmentation. Addressing this gap is essential for fostering a deeper understanding of the field and for advancing the

- • H. Ding, S. Tang, Z. Wu, Y.G. Jiang are with Fudan University, China. henghui.ding@gmail.com
- • S. He is with Shanghai University of Finance and Economics, China.
- • C. Liu is with ByteDance Inc.

###### Introduction Background Meta Architecture RES RVOS R-AVS 3D-RES GREx Related Task Conclusion

###### (§1) (§2) (§3) (§4) (§5) (§6) (§7) (§8) (§9) (§10)

Problem Definition

3D Referring Expression Segmentation

Two-stage Methods

Per-frame & Online Methods

Paradigm

Audio-Visual Segmentation

- (§4.1)
- (§4.2)
- (§4.3)
- (§4.4)
- (§4.5)
- (§4.6)
- (§4.7)

- (§5.1)
- (§5.2)
- (§5.3)
- (§5.4)

- (§6.1)
- (§6.2)
- (§6.3)

- (§2.1)
- (§2.2)

- (§3.1)
- (§3.2)
- (§3.3)
- (§3.4)
- (§3.5)
- (§3.6)

- (§7.1)
- (§7.2)

One-stage Methods

Offline One-stage Methods

Referring Audio-Visual Segmentation

Datasets

Feature Extraction

Referring 3D Gaussian Splatting Segmentation

Multi-Task Learning

Two-Stage Methods

Multimodal Interaction

Omnimodal Referring Audio-Visual Segmentation

Temporal Processing

Weakly-Supervised Methods

Other Task Settings

Segmentation Head

Semi-Supervised Methods

Performance Comparison

(§Appendix)

Training Objectives

Zero-shot Methods

R-AVS Benchmarking

RES Benchmarking

RVOS Benchmarking

(§A.1) (§A.2) (§A.3)

Other Task Settings

GREx Benchmarking ReasonSeg Benchmarking

3D-RES Benchmarking

(§A.4) (§A.5) (§A.6)

###### Fig. 2: Overview of this survey. Different colors represent specific sections. Best viewed in color.

[Figure 32]

[Figure 33]

GRES

###### #Papers

3D-RES

6.9%

100 80 60 40 20

6.2%

RES

2025 Annual Estimate Jan-July 2025

51.3% 21.0%

RVOS

0

14.6%

RES

RVOS R-AVS

R-AVS

20232025 2019 2021

AVS Ref-AVS OmniAVS

3D-RES GRES

Task

2015 2017

Year

- Fig. 3: Publication statistics of multimodal referring segmentation papers in top conferences/journals of computer vision, machine learning, and artificial intelligence, collected up to July 2025.

3D Data Video w/o Audio

Text

Audio

[Figure 34]

[Figure 35]

Omni AVS

[Figure 36]

[Figure 37]

Image Video w/ Audio

VLT

SegPoint

MeViS LISA

ReferSplat

TeSO

OmniAVS

Output: Masks

[Figure 38]

AVS

RES 3D-RES RVOS

3D-AVS

3D-AVS

[Figure 39]

AudioRVOS

[Figure 40]

Ref-AVS

DsHmp

Ref-AVS

MeViSv2

Referring Expression

Visual Scene

Omni modal

[21] [9] [22]

[4] [5] [1] [23]

[24] [2] [25]

[10]

- Fig. 4: Overview of Multimodal Referring Segmentation tasks. Representative works are cited in the top-left corners of each task.

development of generalizable and multimodal solutions.

To this end, we conduct a comprehensive review of over 600 papers in the field of multimodal referring segmentation. This survey seeks to unify diverse referring modalities across various visual scenes. Our goal is to offer a cohesive and structured understanding of the field to enhance accessibility and facilitate cross-task insights. In addition, we highlight practical applications of referring expression techniques, demonstrating their transformative potential in emerging domains such as embodied AI.

• Scope. This survey focuses on recent advances in referring segmentation across three major visual scenes: image-based, videobased (including salient and auditory videos), and 3D-based scenarios, along with three primary referring modalities: text, audio, and omnimodal, as shown in Fig. 4. It primarily reviews deep learning-based methods, highlighting influential works published in top-tier conferences and journals, along with recent preprints that reflect emerging trends and future directions.

• Organization. An overview of the survey is shown in Fig. 2. We begin with background on problem definitions and datasets in Sec. 2, followed by a unified meta architecture in Sec. 3 that spans various referring segmentation tasks. Based on this framework, representative methods across image, video, and 3D scenes are systematically reviewed in Sec. 4 to Sec. 7. Considering the real-world complexity, we further discuss Generalized Referring Expression (GREx) in Sec. 8. Related tasks and applications are explored in Sec. 9, followed by the conclusion and discussion in Sec. 10. Benchmark results are provided in the Appendix.

2 BACKGROUND

2.1 Problem Definition

- 2.1.1 A Unified Formulation The primary objective of this survey is to provide a systematic investigation of the family of referring segmentation tasks. To this end, we propose a unified formulation that generalizes across different task variants within this domain. Specifically, let X and Y denote the input and output spaces, respectively. The objective is to learn an optimal mapping function f defined as:

f : X  → Y, where X = V × E, (1)

where f is typically instantiated as a neural network. The input space X consists of two components: the visual input V (e.g., image, silent video, auditory video, or 3D data), and the referring signal E (e.g., text, audio, etc.) that specifies the target object(s) of interest. The output space Y consists of segmentation mask(s) of the referred entities within V. Based on this unified formulation, we construct a comprehensive taxonomy of referring segmentation tasks and formally define each task in the following sections.

- 2.1.2 Image Scene

• Referring Expression Segmentation (RES). RES, also known as Referring Image Segmentation (RIS), aims to segment the target object in a given image I ∈ RH×W×3 according to a natural language referring expression E. The output is a binary mask M ∈ {0,1}H×W that precisely delineates the object referred to by E. Compared to semantic or instance segmentation tasks that rely on predefined object categories, RES introduces new challenges in comprehending the linguistic content of E and reasoning about the complex visual relationships among objects in the scene, such as spatial configurations, attributes, and interactions.

(1). “The one in red and white (2). “People with long hair.”

(4). “Girl wearing a brown leather jacket.”

Image striped shirt.” (3). “The two on the right.”

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

|(Empty)|
|---|

###### Segmentation

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

RES GRES RES GRES RES GRES RES GRES

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

|(Empty)|
|---|

Localization

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

REC GREC REC GREC

REC GREC REC GREC

- Fig. 5: Classic Referring Expression Segmentation (RES) and Comprehension (REC) handle expressions that refer to a single target object, as shown in example (1). In contrast, Generalized Referring Expression Segmentation (GRES) [3] and Comprehension (GREC) [32], [33] support expressions referring to any number of target objects, including multi-target expressions like (2) and

(3), as well as no-target expressions such as (4), thereby enhancing their applicability in complex and diverse real-world scenarios.

- • Reasoning Segmentation. Reasoning Segmentation is a special case of RES where the referring expression requires indirect reasoning rather than explicit object descriptions, e.g., “the food with the most Vitamin C”. Recent advances in LLMs/MLLMs have made it feasible to handle such expressions by leveraging their strong reasoning and commonsense capabilities.

- 2.1.3 Video Scene

- • Referring Video Object Segmentation (RVOS). RVOS extends referring segmentation to the video domain. Given a video V =

{Vt}Tt=1 with T frames, where each frame Vt ∈ RH×W×3, and a natural language referring expression E, the goal of RVOS is

to generate a sequence of binary masks M = {Mt}Tt=1, where each Mt ∈ {0,1}H×W denotes the pixel-level segmentation of the referred object in frame Vt. Compared to RES, RVOS introduces additional challenges, such as maintaining temporal consistency across frames, dealing with occlusions and appearance variations, and tracking the referred object despite partial or full occlusion.

- • Audio-Visual Segmentation (AVS). AVS aims to segment the sound-emitting objects throughout an auditory video. Given an

auditory video {V,A}, where V = {Vt}Tt=1 denotes T visual frames and A = {At}Tt=1 is the corresponding audio stream, the goal of AVS is to predict binary masks M = {Mt}Tt=1, with each Mt ∈ 0,1H×W highlighting the regions in Vt associated with the sound source in At. AVS is a special case of referring video segmentation, where the query is implicitly defined as “segment the sound-emitting objects in the video.”

- • Referring Audio-Visual Segmentation (Ref-AVS). Ref-AVS aims to segment target objects in an auditory video {V,A} according to a text referring expression E. The desired output

is a sequence of binary masks M = {Mt}Tt=1, where each Mt ∈ {0,1}H×W is the pixel-level segmentation of the referred object in frame Vt. Ref-AVS enables handling scenarios that are difficult to address in AVS and RVOS, e.g., “segment the person singing bass in the a cappella group”. This poses the importance of leveraging multi-modal cues to guide visual segmentation.

- • Omnimodal Referring Audio-Visual Segmentation (OmniAVS). OmniAVS aims to segment specific target objects in an auditory video {V,A} according to a multimodal expression E that flexibly combines text, speech, sound, and visual cues. The

desired output is a sequence of binary masks M = {Mt}Tt=1, where each Mt ∈ {0,1}H×W denotes the pixel-level segmentation of the referred object in frame Vt. The ability to handle diverse multimodal expressions makes OmniAVS both practical for real-world applications and well-suited for advancing omnimodal models with fine-grained perceptual capabilities.

- 2.1.4 3D Scene

- • 3D Referring Expression Segmentation (3D-RES). 3D-RES aims to segment the target object within a 3D scene based on a referring expression E. Given a 3D point cloud consisting of N

points, denoted as P = {Pi}Ni=1, and a referring expression E, the objective is to produce a binary mask M ∈ {0,1}N that identifies the subset of points corresponding to the object referred to by E. Compared to 2D RES on structured image grids, 3D-RES involves segmenting target points in unordered, irregular, and sparse point clouds, requiring both effective language–vision alignment and a deep understanding of geometric structures.

- • Referring 3D Gaussian Splatting Segmentation (R3DGS). Given a scene with S multi-view RGB images I = {Ii}Si=1 and L

corresponding referring expressions E = {El}Ll=1 during training, R3DGS aims to segment the target object in a novel-view image

I ∈ RH×W×3 of the scene based on a given expression E. The output is a binary mask M ∈ {0,1}H×W that delineates the target object, potentially under occlusion. Unlike conventional 2D referring segmentation, R3DGS focuses on 3D scene reconstructed from multi-view images. Compared to existing 3D referring tasks that rely on point clouds and 3D mask supervision, R3DGS learns from 2D images without requiring explicit 3D annotations, offering a more scalable and annotation-efficient paradigm.

- 2.1.5 GREx

• Generalized Referring Expression Segmentation (GRES). As shown in Fig. 5, GRES [3] extends the scope of referring segmentation by allowing expressions to refer to any number of target objects. Given a visual input V and a referring expression E, GRES aims to predict a binary mask M covering all relevant pixels or points corresponding to the described target object(s). Unlike conventional settings that focus solely on single object, GRES supports single-target, multi-target, and no-target expressions,

###### TABLE 1: Representative Datasets for Referring Segmentation and Their Characteristics

Dataset #Imgs/Vids/3D Scenes #Expressions #Objects Characterization Image Scene

ReferItGame [8] 19,894 130,525 96,654 Focusing on real-world expressions but is limited by simpler descriptions.

RefCOCO [34] 19,994 142,209 50,000 Allowing both location- and appearance-based references in images. RefCOCO+ [34] 19,992 141,564 49,856 Focusing on appearance-based expressions without location-based descriptions. RefCOCOg [34] 25,799 95,010 49,822 Including longer and more complex expressions without restrictions on location.

PhraseCut [35] 77,262 345,486 - Focusing on fine-grained expressions covering categories, attributes, and relationships in diverse scenes.

ReasonSeg [21] 1,218 - - Focusing on implicit, reasoning-based expressions requiring world knowledge and complex understanding. Video Scene

MeViS [1] 2,006 28,570 8,171 Focusing on motion attributes and enabling multi-object expressions for enhanced RVOS challenges. MeViSv2 [2] 2,006 33,072 8,171 Based on MeViS, further adding no-target and motion reasoning expressions, adding audio expressions.

- Ref-DAVIS16 [36] 50 100 50 Containing single-object video sequences annotated with referring expressions.

- Ref-DAVIS17 [36] 90 1,544 205 Presenting more challenging multi-object scenarios involving occlusions and distractors.

A2D Sentences [37] 3,782 6,656 4,825 Enriching A2D [38] with elaborate natural language descriptions for actor and action segmentation. J-HMDB Sentences [37] 928 928 928 Supplementing J-HMDB [39] with natural language descriptions, mainly concerning human actions. Refer-Youtube-VOS [40] 3,975 27,899 7,451 Offering pixel-level RVOS across multiple object categories.

ReVOS [41] 1,042 35,074 - Focusing on implicit, complex text queries requiring world knowledge and video context for segmentation. ReasonVOS [42] 91 458 - Evaluating models’ reasoning ability using complex language queries and world knowledge.

###### Audio-Video Scene

AVSBench-S4 [43] 4,932 - - Focusing on individual sound-making objects in semi-supervised single sound source AVS tasks. AVSBench-MS3 [43] 424 - - Focusing on concurrent sound sources in fully supervised multiple sound source AVS scenarios.

AVSBench-Semantic [44] 12,356 - - Offering semantic annotations for fully supervised audio-visual semantic segmentation.

Ref-AVS [23] 4,002 20,261 6,888 Providing pixel-level annotations for objects described in corresponding multimodal-cue expressions. OmniAVS [10] 2,098 59,458 4,262 Providing 8 different omnimodal expressions flexibly consisting of text, speech, sound, and image.

- 3D Scene ScanRefer [45] 800 51,583 11,046 Pioneering the first large-scale dataset for object localization via natural language expressions.

Nr3D [46] 707 41,503 5,878 Providing human-annotated descriptions for precise 3D object localization in real-world scenes. Sr3D [46] 1,273 83,572 11,375 Offering synthetically generated expressions with simplified language patterns.

Instruct3D [9] 280 2,565 - Supporting 3D instruction segmentation from complex texts, including both multi- and zero-target scenes. Ref-LERF [5] 4 295 59 Focusing on spatial relationships for referring 3D gaussian splatting segmentation.

###### GREx

gRefCOCO [3] 19,994 278,232 60,287 Extending RefCOCO [34] by supporting multi-target and no-target expressions. Ref-ZOM [47] 55,078 90,199 74,942 Supporting GRES tasks with annotations built on COCO dataset. Multi3DRefer [48] 800 61,926 11,609 Extending ScanRefer [45] with zero/single/multiple target descriptions for supporting 3D-GREC task. Multi3DRes [11] 800 61,926 11,609 Adapting Multi3DRefer [48] with instance masks to support 3D-GRES task.

improving models’ adaptability in real-world scenarios. This generalization introduces new challenges, particularly in achieving precise alignment between different modalities when dealing with ambiguous, descriptive, or compositional expressions.

• Generalized Referring Expression Comprehension (GREC). Parallel to GRES, GREC [32] is introduced to expand the scope of the classic REC task, see Fig. 5. In contrast to classic REC that generates a single bounding box for a sentence, GREC pursues the generation of a collection of bounding boxes within the input V, denoted as B = {Bi}, wherein each bounding box Bi ∈ R4 encloses an object among the entirety of target objects indicated by the expression E. The number of bounding boxes may vary from 0 to multiple, depending on the given expression.

Beyond image scenes, GRES and GREC can be applied to video and 3D scenes, leading to task variants such as VideoGRES [1], AV-GRES [10], and 3D-GRES [11], expanding the applicability of referring segmentation to real-world scenarios.

###### 2.2 Datasets

We briefly introduce commonly used referring segmentation datasets. TABLE 1 lists more datasets and summarizes their key characteristics, and Fig. 6 presents some representative examples.

- 2.2.1 Image Scene

- • ReferItGame [8] contains 130,525 referring expressions for 96,654 objects across 19,894 images. As the first large-scale dataset for referring expression understanding, it is a valuable resource despite its relatively simple language expressions.
- • RefCOCO/+/g [34] contains 142,209 expressions for 50,000 objects (RefCOCO), 141,564 expressions for 49,856 objects (RefCOCO+), and 104,560 expressions for 54,822 objects (RefCOCOg). These datasets include over 19,000 images for RefCOCO and RefCOCO+, and 26,711 images for RefCOCOg. RefCOCO allows the use of spatial clues. RefCOCO+ limits

the use of such terms, focusing more on the visual attributes. RefCOCOg contains longer and more complex expressions.

• ReasonSeg [21] sets a benchmark for the reasoning segmentation. It includes 1,218 image-instruction-mask samples with implicit text instructions and target masks.

Beyond the above, several other datasets [35], [49], [50], [51], [52], [53], [54], [55] contribute unique challenges to RES research.

- 2.2.2 Video Scene

- • MeViS [1] contains 2,006 videos with 28,570 motion-based referring expressions, emphasizing temporal motion for segmentation and excelling in tasks requiring long-term motion comprehension. MeViS has 4.28 objects per video on average.
- • MeViSv2 [2] extends MeViS [1] with 4,502 additional motion reasoning and no-target expressions, and further provides audio expressions to support audio-based RVOS.
- • Ref-DAVIS [36] comprises 1,200 referring expressions annotated for over 400 objects across 150 videos with around 10K frames, built upon the DAVIS16 and DAVIS17 [56] datasets.
- • Ref-YouTube-VOS [40] contains 3,978 videos and 15,009 expressions with pixel-level annotations. Two types of expressions are provided: 1) full-video expressions containing both static appearance and dynamic temporal clues, and 2) first-frame expressions only using static appearance clues of the first frame.
- • A2D Sentences [37] contains 6,656 sentences describing actions performed by actors across 3,782 videos, offering actor-action segmentation for videos. It has 1.28 objects per video on average.
- • J-HMDB Sentences [37] provides 928 referring expressions for human actions in 928 videos of J-HMDB dataset, supporting action localization and segmentation.

- 2.2.3 Auditory Video Scenes

• AVSBench-object [43] provides two settings: Single-source (S4) includes 4,932 videos from 23 categories with sparse supervision (only the first frame labeled), while Multi-source (MS3) contains 424 fully annotated videos with multiple sound sources.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

RefCOCO

RefCOCO+

RefCOCOg

gRefCOCO

There is red colored truck in between the other trucks.

A man along with his two dogs.

Left guy in blue.

Man in yellow tie.

|[Figure 72]<br><br>RefAVS<br><br>[Figure 73]|
|---|

|[Figure 74]<br><br>OmniAVS<br><br>[Figure 75]|
|---|

[Figure 76]

[Figure 77]

ReasonSeg

PhraseCut

The one makes <sound: Siren> and moves to ?

Something that helps in the pollination of plants.

Man wearing glasses.

The woman singing.

[Figure 78]

|[Figure 79]<br><br>A2D-Sentences|
|---|

|[Figure 80]<br><br>JHMDB-Sentences|
|---|

|[Figure 81]<br><br>MeViS|
|---|

A person is walking backwards while being followed by a dog.

Man jumping over the platform.

The goldfish swimming in an aquarium.

|[Figure 82]<br><br>Ref-DAVIS|
|---|

|[Figure 83]<br><br>Ref-DAVIS|
|---|

|[Figure 84]<br><br>Ref-Yotube-VOS|
|---|

A person performing bike tricks.

A white cockatoo in its cage.

A flamingo drinking water.

[Figure 85]

[Figure 86]

[Figure 87]

ScanRefer

ReferIt3D

Multi3DRes

This tall trashcan is by a dark cabinet that goes from wall to wall.

Chairs around the round table.

The large bed.The bed is next to the guitar.

- Fig. 6: Examples from 17 commonly used referring segmentation datasets, including image, video, and 3D scene data.

- • Ref-AVS [23] provides 20,261 text expressions for 6,888 objects in 4002 auditory videos. The text expressions adopt multimodal cues to describe objects in audio-visual scenes.
- • OmniAVS [10] provides 59,458 omnimodal expressions for 4,262 objects in 2,098 auditory videos. The expressions integrate

- 4 different modalities: text, speech, sound, and image, forming 8 different referring expressions: 1) Text; 2) Speech; 3) Text with Sound; 4) Speech with Sound; 5) Text with Image; 6) Speech with Image; 7) Text with Sound and Image; and 8) Speech with Sound and Image. Each expression includes either text or speech, as they provide essential instructions for the task and other modalities

- 2.2.4 3D Scene

- • ScanRefer [45] comprises 51,583 natural language queries for 11,046 objects across 800 ScanNet [57] scenes, with an average of 13.81 objects and 64.48 queries per scene.
- • Instruct3D [9] consists of 280 scenes from ScanNet++ [58] and 2,565 expression–object pairs, designed for 3D instruction-based (or reasoning-based) segmentation, with a focus on interpreting implicit user intent from natural language.
- • Ref-LERF [5] consists of 772 training images and 22 testing images sourced from LERF [59], and is newly annotated with 295 language expressions for 59 objects. The expressions focus on spatial relationships, providing rich contextual cues for referring segmentation in 3D Gaussian Splatting.

- 2.2.5 GREx Datasets.

- • gRefCOCO [3] contains 278,232 referring expressions across 19,994 images, supporting both GRES [3] and GREC [32] tasks. It extends RefCOCO by adding multi-target and no-target expressions alongside single-target ones. No-target expressions are carefully curated to be contextually relevant yet intentionally misleading, without being entirely unrelated to the image. These characteristics make gRefCOCO more challenging than traditional RES datasets and more representative of real-world scenarios.
- • Multi3DRes [11] extends Multi3DRefer [48] by incorporating segmentation masks to support the 3D-GRES task [11]. It provides 61,926 expressions for 11,609 objects across 800 scenes, enabling more robust handling of real-world scenarios.

3 META ARCHITECTURE

- 3.1 Paradigm

- • Two-Stage Paradigm. Two-stage referring segmentation methods [60], [61], [62], [63], [64] follows a sequential process: the model first generates mask or tracklet proposals covering all objects in the scene or across the video, then matches these proposals to the referring expression, and finally selects the bestmatched mask as the final prediction. These region proposals are typically obtained using off-the-shelf, well-trained instance segmentation models [17]. These candidate masks are evaluated based on their feature similarity to the referring expression. These two-stage methods, e.g., TGNN [61] and MAttNet [60], have been widely adopted in early referring segmentation methods due to their modularity and interpretability. However, this paradigm is susceptible to error propagation, where inaccuracies in the proposal stage directly affect final performance. Moreover, it often incurs higher computational costs, limiting its practicality for realtime or resource-constrained applications.
- • One-Stage Paradigm. One-stage methods [4], [65], [66], [67], [68], [69] addresses the limitations of the two-stage methods by directly predicting the target object from the input visual scene and referring expression in a single forward pass. This end-to-end architecture eliminates the need for separate proposal generation and matching steps, potentially reducing error propagation and improving efficiency. One-stage methods typically employ dense prediction mechanisms where each spatial location in the feature map interacts with the referring expression. Following DETR [70], recent advances in transformer-based architectures, e.g., VLT [4], [7] and ReLA [3], have significantly enhanced the performance by enabling more effective multimodal fusion and context modeling. These models can better capture the complex relationships between visual elements and linguistic descriptions, leading to more accurate segmentation results across various visual scenes. 3.2 Feature Extraction
- • Vision Encoder. Vision encoder extracts visual features from visual inputs. For image-based tasks, early methods [6], [60] rely on convolutional neural networks (CNN) such as ResNet [71]. Recently, Vision Transformers (ViT) [72] and their variants [73] have demonstrated superior performance and have been widely adopted by many methods [3], [4]. For video-based tasks, models [37], [40] incorporate temporal modeling capabilities via 3D CNN [74], [75], or transformer with temporal attention, e.g., Video Swin Transformer [76]. For 3D tasks [9], [77], specialized encoders, e.g., Sparse 3D U-Net [78] and PointNet [79], process point clouds or voxel to capture spatial geometry and relationships.

- • Text Encoder. Text encoders transform referring expressions into text features. Early methods [6], [80], [81], [82], [83] use recurrent neural networks to model sequential dependencies in language. With the advancement of pre-trained language models, recent methods [84], [85], [86], [87] have adopted transformerbased architectures, e.g., BERT [88] and RoBERTa [89], to obtain richer and more contextualized representations. More recently, text encoders from vision-language models, e.g., CLIP [90], have gained popularity for producing text embeddings that align well with visual features in a shared semantic space, enhancing multimodal alignment in referring segmentation [91], [92].
- • Audio Encoder. Audio encoders extract acoustic features for audio-based tasks. Raw audio is typically converted into spectrograms or mel-frequency cepstral coefficients, which are then processed by neural networks. CNN-based models [93] treat spectrograms as images, while transformer-based models [94], [95] capture temporal dependencies in audio. Pretrained encoders like VGGish [93] and wav2vec [94] have shown strong performance in generating robust audio features for downstream tasks including AVS [43], Ref-AVS [96], and OmniAVS [10].

###### 3.3 Multimodal Interaction

- A. Multimodal Fusion: the process of integrating features from different modalities to create a unified representation.

- • Concatenation-based Fusion. This fusion strategy [6], [69], [80], [81], [83] combines multimodal features via simple concatenation, followed by convolutional layers or MLPs. For example, image and text features are concatenated to form a joint represen-

tation: ffused = Conv(Concat[fimage;ftext]). While efficient, such simple fusion fails to capture complex inter-modal interactions, limiting its effectiveness in fine-grained cross-modal reasoning.

- • Attention-based Fusion. To address the limitations of simple fusion, attention-based methods [4], [84], [97], [98] enable dynamic and context-aware interactions across modalities. They selectively emphasize relevant features from one modality conditioned on the other [99], facilitating fine-grained alignment. For example, in image-text tasks, attention guides the model to focus on specific word tokens when processing visual regions, and vice versa. Representative strategies include visual attention [4], [7], symmetric co-attention [92], [100], and multimodal transformers employing self- and cross-attention [66], [101]. These methods enhance models’ ability to capture complex inter-modal relationships and have shown superior performance over simple fusion.

- B. Multimodal Alignment: the process of establishing meaningful correspondences between elements of different modalities.

- • Contrastive Learning-based Alignment. These methods [91], [100], [102] leverage contrastive objectives to align multimodal representations in a shared embedding space. Methods like CLIP [90] and ALIGN [103] train on paired image–text data to maximize the similarity of matched pairs and minimize that of mismatched pairs. This results in a unified semantic space where semantically related concepts from different modalities are closely positioned, supporting cross-modal retrieval and understanding.
- • Self-supervised Alignment. These methods [66], [100], [104], [105] leverage naturally co-occurring multimodal data to learn alignment without explicit supervision. Techniques include 1) masked multimodal modeling [66] that learns to predict masked content in one modality according to another modality and 2) reconstruction-based methods [100], [105] that encode information from one modality and decode it into another.

###### 3.4 Temporal Information Processing

For video tasks, temporal information processing is essential for motion understanding [1] and consistent object segmentation across frames, especially under complex scenes [106], [107].

- • 3D Convolutional Networks. 3D CNNs [37], [108], [109] extend 2D convolutions by adding a temporal dimension to learn spatiotemporal features from videos. They process multiple frames simultaneously, capturing motion patterns and temporal context. Methods like C3D [74] and I3D [75] treat videos as 3D volumes, applying convolutions across spatial and temporal dimensions to extract features modeling object movements and transformations.
- • Temporal Attention Mechanisms. Attention-based methods [110], [111], [112], [113] capture temporal dependencies by dynamically weighting the importance of different frames. Temporal attention allows models to focus on the frames that are most informative for object tracking and segmentation, enhancing the handling of long-range dependencies and diverse motion patterns.
- • Memory Networks. Memory-based methods [40], [114], [115], [116] explicitly store and update objects’ features across frames, capturing clues about their appearance, location, and context. These memory modules enable models to retrieve relevant historical cues when processing new frames, enhancing temporal consistency. Such mechanisms are particularly beneficial in complex and long videos, where target objects may temporarily disappear, reappear, or undergo significant appearance variations [1], [106].
- • Optical Flow and Motion Estimation. A number of temporal modeling methods [36], [37], [105], [117], [118] leverage explicit motion cues by incorporating optical flow estimation. Optical flow captures pixel-level correspondences between consecutive frames, providing detailed information about object motion. This motion guidance can be effectively combined with appearance features to enhance tracking and temporal coherence of segmentation masks. 3.5 Segmentation Head Segmentation head plays a key role by converting features into final masks. Existing designs can be broadly categorized as follows.
- • CNN-based Segmentation Head. Traditional segmentation heads [6], [69], [81], [83], [110] typically use a series of convolutional layers followed by upsampling operations as in FCN [16] to restore the spatial resolution of feature maps.
- • Transformer-based Segmentation Head. Prominent architectures such as DETR [70] and Mask2Former [119] employ transformer decoders to generate object queries, which are subsequently mapped to segmentation masks. Transformer-based segmentation heads [3], [77], [120], [121] are particularly effective at capturing global context and are well-suited for complex scenes.
- • Promptable Segmentation Head. Recent advances have introduced flexible and generalizable segmentation heads [116], [122], [123], [124], [125], [126] that respond to diverse types of prompts. Segment Anything Model (SAM) [127] exemplifies this design by supporting point, box, and text-based prompts without requiring task-specific fine-tuning, enabling generalization across a wide range of segmentation tasks. Building on this foundation, SAM2 [128] extends the promptable framework to video segmentation.

###### 3.6 Training Objectives

Herein we list the most commonly used training objectives.

• Segmentation Objectives. For segmentation tasks [3], [4], [6], [7], [43], the primary training objectives typically include binary cross-entropy (BCE) loss and Dice loss. BCE measures pixel-wise

classification error, while Dice loss directly optimizes the overlap between predicted and ground truth masks. These losses are often combined to improve boundary accuracy and region completeness.

- • Grounding Objectives. To enhance grounding performance, several methods [68], [85] incorporate grounding objectives (e.g., L1, L2, IoU, and focal loss) to enforce accurate correspondence between visual regions and referring expressions. Set-based bipartite matching [32], [70] is used to address permutation invariance.
- • Multimodal Alignment Objectives. Beyond visual perception losses, many referring segmentation models [4], [22], [77], [91], [129], [130] incorporate alignment objectives to bridge visual and linguistic modalities. Contrastive losses are commonly used to pull together matched visual-language pairs and push apart mismatched ones. Effectively combining these objectives is essential for building robust models capable of accurate segmentation across diverse scenes and referring modalities.
- • Multi-Task Learning Objectives. Some methods adopt multitask learning strategies that couple referring segmentation with auxiliary tasks such as referring comprehension or generation. For example, MCN [68] jointly models RES and REC with consistency and suppression objectives to reduce task conflict. Chen et al. [131] integrate referring generation to enforce captionaware consistency. Liu et al. [99] introduce iterative languagevision interaction and reconstruction to preserve linguistic cues. These strategies show that jointly optimizing related tasks can enhance RES performance and multimodal understanding.

4 REFERRING EXPRESSION SEGMENTATION

- 4.1 Two-stage Methods

Two-stage methods [34], [60], [62], [132] for RES typically involve an initial segmentation step followed by a matching process, as shown in Fig. 7(a). These methods often leverage off-theshelf instance segmentation models to generate object proposals, and then match them to the referring expression to identify the best-matched object. For example, MAttNet [60] first segments all objects using an instance segmentation network, Mask RCNN [17], and then employs a modular network to identify the best-matched instance. CMN [62] also adopts modular networks that parse referring expressions into subject, relationship, and object components using three soft attention maps. It then aligns these textual representations with image regions. ISF [133] and WiCo [134] combine the advantages of both two-stage and onestage methods to improve the performance of RES.

However, this paradigm suffers from error accumulation and high computational cost, making it less practical in real-time settings. The majority of existing RES methods adopt a one-stage paradigm, which we categorize in the following subsections.

###### 4.2 One-stage Methods

- 4.2.1 Better Representations

• Optimizing Representation Extraction. Several works [82], [135], [136] optimize feature extraction to obtain richer and more discriminative representations. TV-Net [135] enhances vision features of the referent by retrieving relevant external images. KWA [82] enhances language feature by assigning higher weights to words critical for object identification. MCRES [136] builds virtual datasets and combines them with meta-learning optimization, enabling model to better capture the semantic and visual representations of concepts when handling expressions with novel

###### (a) Two-Stage Method.

Proposal Features

Segmentation

[Figure 88]

[Figure 89]

[Figure 90]

Image Encoder

Instance

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Text Encoder

“The cat lying on the sofa.”

Feature Matching & Ranking

[Figure 95]

###### (b) One-Stage Method.

Image Feature

[Figure 96]

Image Encoder

FeatureFusion

MaskDecoder

[Figure 97]

Text Encoder

“The cat lying on the sofa.”

[Figure 98]

Text Feature

Fig. 7: Architecture Overview of Referring Expression Segmentation. (a) Two-stage method uses an off-the-shelf instance segmentation model to generate region proposals, followed by vision-language feature matching and ranking to select the top1 mask. (b) One-stage method fuses image and text features, performing pixel-level segmentation directly on the fused features.

compositions. Some methods [137], [138], [139] focus on optimizing feature extraction to obtain language-guided vision representations or vision-guided language representations. For example, ASDA [137] adaptively selects the most relevant vision features according to sentence-level language cues. VG-LAW [138] treats the vision backbone as an expression-specific feature extractor by generating dynamic weights for various expressions.

- • Data Augmentation Methods. Data augmentation helps learn better representations by increasing the diversity and quantity of training data. However, standard techniques often distort spatial relationships or alter contextual cues described in referring expressions, making the augmented samples invalid. To address this, several works [140], [141], [142], [143] propose RES-specific data augmentation techniques. NeMo [140] constructs mosaic images by combining a target image with three CLIP-selected negative images, generating more diverse and challenging training examples. MaskRIS [141] introduces a masking-based data augmentation technique that combines both image and text masking to generate diverse image-text pairs for training.

4.2.2 Enhancing Multi-Modal Interaction

- •Multimodal Fusion. Various methods [4], [6], [144] have been proposed to fuse vision and language features. Early methods [6], [80], [81], [83] typically extract vision features using CNNs and encode referring expressions via LSTMs. These features are then fused using simple operations such as concatenation, followed by convolutional layers. Kesen et al. [145] extend this by performing feature fusion in both the upsampling and downsampling paths of U-Net. With the advent of attention mechanisms, some works [97], [98], [99], [110], [146] leverage cross-modal attention to capture fine-grained interactions between modalities. CMSA [110] constructs comprehensive multimodal features by concatenating image, spatial, and language features, then processes them by a cross-modal self-attention module to model long-range dependencies between words and spatial regions. EFN [98] leverages an asymmetric co-attention module to enhance the matching between multi-modal features and strengthen their targeting capability.

A major milestone in RES is the shift toward Transformerbased architectures, which naturally support long-range dependencies across spatial and linguistic tokens. Traditional CNNs struggle with such global context due to their localized receptive fields. VLT [4], [7] is the first to introduce a Transformer-based approach to RES, reformulating it as an attention problem where language features act as queries over vision tokens. This enables joint reasoning across both modalities. Since VLT, Transformer-based methods [84], [147], [148], [149], [150] have quickly become the common and state-of-the-art methods in RES. LAVT [84] injects language into vision features at intermediate levels of a vision Transformer to enhance multimodal fusion. To mitigate query collapse, LQMFormer [150] enforces a margin of separation between query representations. To address the high computational cost of Transformer, ReMamber [151] adopts a Mamba-based [152] architecture for more efficient vision-language fusion in RES.

• Multimodal Alignment. Multimodal alignment [91], [100], [102], [153] in RES aims to explicitly or implicitly associate information from different modalities (i.e., language and vision), ensuring consistency within a shared semantic space. Several methods [87], [100], [101], [102], [154] employ cross-modal attention to strengthen alignment across modalities. For example, ReSTR [87] employs a self-attention encoder to effectively align visual and language features in a shared semantic space.

Other methods [91], [149], [155], [156], [157] employ contrastive learning to improve multimodal alignment. By contrasting matched and mismatched modality pairs, these methods pull aligned pairs closer while pushing apart misaligned ones. CRIS [91] introduces a text-to-pixel contrastive loss to explicitly link textual and pixel-level visual features, addressing CLIP’s limitations in RES. CGFormer [149] integrates contrastive learning with a grouping strategy to associate tokens with corresponding masks. Some methods [66], [158] adopt reconstruction-based techniques to align diverse modalities. Inspired by MAE [159], BTMAE [158] introduces bidirectional reconstruction of missing features in both image and language tokens to effectively model high-dimensional relationships among multimodal tokens. Inspired by BEiT-3 [160], OneRef [66] introduces a one-tower modality-shared Transformer with a mask referring strategy that jointly models referring-aware image and language masks.

- •Parameter-Efficient Tuning Methods. While large foundation models have advanced RES, fully fine-tuning remains computationally expensive. Parameter-Efficient Tuning (PET) offers a costeffective alternative by freezing most of the pre-trained model and updating only a small subset of parameters, often achieving comparable performance. Recent works [92], [161], [162], [163] have explored applying PET to RES, mainly focusing on enhancing modal interaction while maintaining computational efficiency. For example, ETRIS [92] leverages a vision-language bridge to fuse visual inductive biases with linguistic cues, while BarLeRIa [161] employs bi-directional intertwined vision-language adapters for efficient cross-modal fusion with minimal learnable parameters.

- 4.2.3 Optimizing Mask Decoder

To improve segmentation quality, several methods [69], [83], [164], [165], [166], [167], [168] adopt multi-stage optimization strategies that progressively refine segmentation masks. Early methods [69], [83], [108], [109], [169] leverage ConvLSTM [167] to iteratively decode multimodal features, refining the prediction mask. SADLR [166] iteratively refines predictions using accumulated object context to produce accurate segmentation masks.

JMCELN [164] proposes a multi-stage cascade framework that refines segmentation by dynamically updating contextual embeddings based on intermediate mask predictions.

Segment Anything Model (SAM) [127] shows strong performance across various segmentation tasks using point, box, or mask prompts, but remains limited in text-guided segmentation. To bridge this gap, several methods [124], [153], [170], [171], [172] have been proposed to adapt SAM for RES. DIT-SAM [124] addresses SAM’s limitations by projecting text into the semantic space of SAM’s image encoder. Prompt-RIS [172] bridges CLIP and SAM via prompt learning. Grounded SAM [171] combines Grounding DINO [173] with SAM to enable detection and segmentation of arbitrary regions based on free-form text inputs. FLMM [170] uses a frozen LLM to generate segmentation priors, followed by a CNN-based and a SAM-based mask decoder to produce the final segmentation mask. Recent works [21], [174], [175] enhance SAM’s text-guided segmentation by generating prompt embeddings via LLMs (see Sec. 4.7).

4.2.4 Improving Training Objectives

Several methods [85], [176], [177], [178], [179] focus on optimizing training objectives. LTS [176] decouples RES into a “LocateThen-Segment” framework, first predicting a position prior and then refining the segmentation mask. Other works [85], [177], [180], [181] reformulate segmentation as point-based sequence prediction, representing masks as polygons or point sequences. Text4Seg [182] introduces a text-as-mask paradigm, casting RES as a text generation task without relying on additional decoders. UFO [183] treats segmentation as an embedding retrieval problem, generating masks by computing similarity between mask token embeddings and image features. Training objectives for enhancing multimodal interaction are discussed in Sec. 4.2.2.

###### 4.3 Multi-Task Learning

Multi-task learning is widely used in segmentation, detection, and generation, typically with a shared backbone and task-specific heads. Building on this paradigm, several methods [68], [184], [185], [186] jointly address RES and referring expression comprehension (REC). MCN [68] implements an explicit constraint strategy by introducing consistency loss to ensure similarity between feature activation maps in REC and RES tasks. Referring Transformer [120] and MDETR [187] use an implicit approach by sharing multi-modal representations across REC and RES heads. SeqTR [181] and Polyformer [85] further unify both tasks as sequence-to-sequence point prediction problems. Beyond REC and RES, some works [131], [188] also explore joint modeling of RES and referring expression generation (REG).

Generalist models supporting multiple vision-language tasks [189], [190], [191], [192], [193], [194], [195] have shown strong performance on RES. X-Decoder [189] introduces a unified framework for segmentation and vision-language tasks, including RES. SEEM [190] extends this versatility by supporting diverse prompts (clicks, boxes, polygons, scribbles, text, and image regions) via a prompt encoder in a joint visual-semantic space. AnyRef [196] leverages MLLMs for pixel-level grounding and region-aware expression generation across text, regions, images, and audio.

###### 4.4 Weakly-Supervised Methods

To reduce the annotation burden, weakly-supervised RES methods [197], [198], [199], [200] leverage incomplete, imprecise, or noisy annotations to minimize reliance on dense pixel-level labels.

- • Text-Only Supervision. TSEG [197] is a pioneering weakly supervised RES method that only uses image-level referring expressions as supervision. Subsequent methods [198], [199], [201], [202], [203], [204] adopt textual supervision and employ strategies such as visual entity discovery and gathering [199], textto-image response mapping [205], and enhanced Grad-CAM for saliency refinement [201]. TRIS [205] directly learns the text-toimage response map by contrasting target-related positive texts with target-unrelated negative texts. PPT [198] leverages a point generator to connect frozen CLIP and SAM models while adopting curriculum learning to facilitate the gradual learning of the point generator. PCNet [206] decomposes the description into multiple cues to guide progressive target localization. WeakMCN [184] jointly learns WeakREC and WeakRES in a collaborative manner.
- •Bounding Box Supervision. Feng et al. [207] propose a weakly supervised RES method using bounding box annotations, where pseudo labels are generated from object contours and refined through filtering. In contrast, GTMS [208] enhances pseudo label quality by incorporating both structural and semantic information.
- • Point Supervision. Beyond text and bounding box, PKS [200] employs click-based annotations (i.e., object center/corner clicks) for model training, achieving commendable performance.

###### 4.5 Semi-Supervised Methods

Semi-supervised RES methods reduce reliance on labor-intensive annotations by leveraging a small set of labeled image-text pairs alongside abundant unlabeled samples. Several semi-supervised RES methods [209], [210] adopt pseudo-labeling strategies, selecting high-confidence predictions as supervision for model refinement. RESMatch [211] introduces the first dedicated semisupervised RES framework, incorporating a quality assessment mechanism to evaluate pseudo-labels and strong–weak supervision pairs. SemiRES [209] leverages SAM’s edge-segmentation capabilities to generate high-quality pseudo-labels. PseudoRIS [210] produces multiple candidate masks from distinctive words and filters misleading captions to obtain reliable supervisory signals. Some semi-supervised RES methods [180], [212] adopt a mixed-supervision paradigm, typically using a small portion of mask annotations alongside a larger proportion of bounding box annotations as supervisory signals. Partial-RES [180] and Safari [212] adopt an auto-regressive contour-based sequence prediction strategy, requiring only a small fraction of mask annotations along with supplementary bounding box annotations.

###### 4.6 Zero-Shot Methods

Zero-shot RES methods [213], [214], [215], [216] leverage visionlanguage foundation models (e.g., CLIP [90]) to perform segmentation without task-specific training. Global-Local CLIP [213] is the first method to explore zero-shot RES using CLIP. TAS [217] incorporates additional caption embeddings, negative text embeddings, and a spatial rectifier to enhance CLIP predictions, while CaR [218] recurrently applies CLIP to iteratively refine the segmentation mask. HybridGL [219] introduces a global-local feature extraction method that combines mask-specific details with contextual information to improve mask representation.

###### 4.7 Other Task Settings

•Reasoning Segmentation. Large language models (LLMs) and multimodal LLMs (MLLMs) [220], [221], [222] have significantly advanced vision-language tasks by enabling strong commonsense

reasoning, opening new opportunities for RES. Building on this, LISA [21] pioneers the concept of reasoning segmentation, allowing models to handle complex expressions requiring external knowledge, such as “Segment the food with the highest protein content.” LISA introduces an embedding-as-mask paradigm by extending the MLLM’s vocabulary with a special [SEG] token to prompt SAM [127] for mask generation. Inspired by these developments, several methods [28], [223], [224], [225], [226], [227], [228], [229], [230] explore reasoning segmentation with LLMs/MLLMs. CoReS [231] employs chain-of-thought (CoT) [232] to tackle complex implicit text queries that require multi-step reasoning. SAM4MLLM [175] encodes object masks as discrete text prompts, while READ [233] converts text-image similarity maps into differentiable points to prompt SAM. Unlike LISA’s paradigm, LLM-Seg [234] adopts a two-stage approach, decoupling the reasoning over text from the segmentation process.

Several methods [174], [235], [236], [237] focus on addressing the challenges of multi-target or zero-target scenarios in images, known as GRES [3]. GSVA [174] addresses the GRES [3] task by introducing shared-weight [SEG] tokens for multi-target segmentation and a [REJ] token to discard empty targets. SESAME [236] handles false-premise queries by enabling models to detect object presence, provide corrective feedback, and segment only when appropriate. In parallel, interactive and conversational segmentation has gained interest [67], [238]. For example, SegLLM [67] supports multi-turn interactions with visual and textual queries, enabling the inference of object relationships such as spatial, interactive, and hierarchical dependencies. More recently, methods [239], [240], [241], [242] have explored reinforcement learning (RL) to enhance reasoning capabilities in segmentation. Seg-Zero [239] employs pure RL to develop emergent reasoning and improve outof-domain generalization. POPEN [240] incorporates preferencebased optimization to align large vision-language models with human intent via RL strategies, such as GRPO [243].

- • Referring Remote Sensing Image Segmentation (RRSIS). RRSIS focuses on aerial or satellite images, posing unique challenges by varying spatial scales, object orientations, and complex backgrounds distinct from natural images. To address these issues, several tailored methods [244], [245], [246], [247], [248] are proposed. Yuan et al. [244] introduce RRSIS task and RefSegRS benchmark, along with a LAVT-based [84] framework that integrates multi-scale features for segmenting small and scattered objects. RMSIN [246] addresses orientation variations using rotated convolutions to better capture multi-oriented object features.
- • Other. Beyond the core tasks, alternative settings have been explored [249], [250], [251], [252]. Zero-guidance segmentation [249] seeks to segment input images and label all segments using natural language. Grounded Conversation Generation [250] aims to generate natural language responses interleaved with object segmentation masks. Several methods [253], [254] extend referring segmentation capabilities to embodied intelligence settings. Additionally, PTQ4RIS [255] proposes a post-training quantization framework to address challenges for on-device RES inference.

5 REFERRING VIDEO OBJECT SEGMENTATION

5.1 Per-frame and Online Methods

- • Per-frame Methods. A natural approach to Referring Video Object Segmentation (RVOS) is to treat a video as a sequence of images and apply image-level referring segmentation to each frame independently. RefVOS [256] follows this paradigm by

fusing frame-level visual and language features using an imagebased RES network. Subsequent methods such as CMPC [108] and VLT [4] adopt similar per-frame pipelines, offering a simple extension of image-level RES to the video domain.

• Online Temporal Understanding. Per-frame methods process each frame independently and often suffer from temporal inconsistency due to the lack of temporal context. To address this, online methods process videos sequentially while maintaining memory across frames. For example, URVOS [40] uses a temporal memory attention module to improve intra-frame consistency, and OnlineRefer [257] introduces an online association framework to enhance temporal coherence and referring accuracy.

###### 5.2 Offline One-stage Methods

- • Offline Temporal Understanding. Although online methods leverage historical frame information, their inability to access future frames restricts their capacity to resolve motion-centric and temporally ambiguous expressions, e.g., “Elephant that goes towards then turns back”. To mitigate this limitation, many RVOS approaches employ offline processing, which enables global temporal reasoning by considering the full video sequence [110], [112], [113], [114], [129], [257], [258], [259], [260], [261], [262], [263], [264], [265]. For example, TempCD [260] employs global referent tokens and local object queries to perform video-level reasoning and frame-wise segmentation via a collection-distribution mechanism. HTML [262] applies hierarchical temporal sampling to capture multi-scale temporal interactions. LBDT [113] performs early-stage spatio-temporal alignment using language-guided encoding, while LOCATER [112] utilizes a finite-memory structure to dynamically gather relevant temporal context. ReferMo [266] employ motion-vectors to compress the motion in across the video. Most recent RVOS works follow this offline pipeline, with various designs optimizing different aspects of video-level reasoning.
- • Better Representations. Some RVOS methods [22], [118], [267], [268], [269], [270], [271] focus on learning better representations to enhance model performance. They improve either visual or textual feature representations to strengthen multi-modal understanding. On the visual side, several methods [267], [268], [269], [270], [271] aim to enhance visual feature quality. MLRL [268] introduces a multi-level representation learning framework that generates discriminative embeddings by integrating multi-frame temporal dynamics (video level), spatial context (frame level), and object-aware priors (object level). VD-IT [270] leverages visual features from pretrained text-to-video diffusion models, achieving improved temporal consistency across frames. On the textual side, methods such as LoSh [118] and DsHmp [22] focus on refining language representations. LoSh [118] jointly predicts with long and short expressions, using the short form to enhance appearance-based localization. To address the challenging motion expressions [1], DsHmp [22] decomposes referring expression understanding into static and motion perception components, with an emphasis on improving temporal language comprehension. Some other recent works explore using extra types of feature, such as spectrum [272] or flow map [117], to assist in representation.
- • Enhancing Multi-Modal Interaction. Effective integration of visual and textual features is essential for RVOS. Several methods [110], [111], [273], [274], [275], [276] have been proposed to enhance cross-modal interaction. OATNet [274] concatenates visual and textual features and processes them through a shared multimodal encoder to jointly model intra- and inter-modal relationships. EFCMA [111] introduces an encoder fusion network with

- gradual language guidance and a co-attention mechanism to enhance feature alignment. ReferFormer [275] designs a cross-modal feature pyramid network to enable multi-scale fusion of visionlanguage features. CMSA [110] introduces a gated multi-level fusion module to selectively integrate cross-modal features across visual hierarchies. YOFO [273] designs a meta-transfer module that injects target-specific linguistic cues into visual features while suppressing irrelevant language variations. SSA [277] addresses the gap between linguistic descriptions and video objects, as well as interference from background clutter.
- • Optimizing Mask Decoder. Vision foundation models (e.g., SAM [127], SAM2 [128]) have shown strong segmentation capabilities, inspiring their adaptation for RVOS. Recent methods [115], [123], [278], [279] leverage the precise mask generation of these models while extending them to handle referring expressions in video contexts. RefSAM [278] integrates multi-view information from diverse modalities and frames across time to adapt SAM for RVOS. SAMWISE [123] and MPG-SAM 2 [279] are two representative RVOS works built on SAM2 [128]. SAMWISE [123] injects temporal cues and multimodal signals via an adapter during feature extraction, while MPG-SAM 2 [279] employs mask priorbased dense prompts and multi-level global context fusion.
- • Improving Training Objectives. Some methods [105], [129] aim to improve training objectives in RVOS. For example, SOC [129] introduces a visual-linguistic contrastive loss that applies semantic supervision at the video level to align object representations across modalities. Mei et al. [105] propose a general self-supervised language-video pretraining framework that learns pixel-level features by using optical flow as an intermediate supervision signal during pretraining. Some methods [86], [122], [280], [281], [282], [283], [284] address multiple video segmentation tasks via unified frameworks or multi-task learning strategies. MUTR [282] adopts a DETR-style transformer to jointly handle various tasks, while AL-Ref-SAM 2 [284] explores a trainingfree paradigm to unify RVOS and AVS. UniMM [281] presents a unified model for both mask-referred (VOS) and languagereferred (RVOS) segmentation. UniVS [280] unifies all major video segmentation tasks within a single model. Sa2VA [122] integrates SAM2 with LLaVA to enable dense grounded understanding across text, image, and video in a shared LLM token space. Moreover, recent study such as VEGGIE [285] shows that generalist generative models can also be used for RVOS.

5.3 Two-Stage Methods

Similar to RES, many early RVOS methods adopt a two-stage paradigm. Some directly extend image-based methods, for example, Khoreva et al. [36] adapt image method MAttNet [60] to videos by applying frame-wise segmentation followed by postprocessing temporal smoothing. Others [64], [286] operate at the video level by generating object tracklets across the entire video sequence and selecting the one that best matches the referring expression. For example, Liang et al. [64] first detect tracklets and then perform language-tracklet matching to identify the target.

- • Resurgence for Long and Motion Expressions. Two-stage methods were initially in the minority due to the architectural complexity introduced by decoupling the segmentation and languagevision understanding stages. While this separation increases design complexity, it offers a key advantage: the generated tracklets can encode the entire temporal sequence in a compressed form, preserving both short-term and long-term motion cues for downstream multi-modal understanding in the second stage.

The importance of such global temporal modeling is highlighted by the challenging MeViS [1] benchmark, which emphasizes motion-centric expressions. Many expressions in MeViS describe long-term motions, requiring models to capture extended temporal dynamics. However, online methods lack access to global context, and offline one-stage models often rely on sparse frame sampling (e.g., 5 or 8 frames) to reduce computational cost, which can miss fine-grained or long-range motion patterns, leading to suboptimal performance. This challenge has motivated a resurgence of two-stage designs. Recent works [287], [288] first extract full-length mask tracklets using off-the-shelf video instance segmentation models, then align them with language in a separate stage. This enables comprehensive temporal modeling and more accurate understanding of motion-guided expressions.

###### 5.4 Other Task Settings

- • Weakly-Supervised RVOS. RVOS methods typically depend on densely annotated datasets like MeViS [1], which are expensive to create. To reduce annotation costs, weakly supervised methods [289], [290] have emerged. For example, SimRVO [289] introduces a weak supervision scheme where only the first frame has mask supervision, and subsequent frames use bounding boxes.
- • Video Reasoning Segmentation. This setting focuses on implicit expressions that require complex reasoning, e.g., “winner in the sprint race,” posing significant challenges in intent understanding and external knowledge. Driven by advances in LLMs/MLLMs [220], [221], [222], several methods [41], [42], [116], [228], [291], [292] have been proposed to address this setting. They typically input video frames and queries into LLMs/MLLMs to generate <SEG> tokens, which are then decoded into segmentation masks. VideoLISA [42] adopts a sparsedense sampling strategy to balance temporal context and spatial detail under limited computation. VRS-HQ [116] fuses spatial features into temporal tokens and employs SAM2 [128] for keyframe segmentation followed by mask propagation. JiT [293] proposes an online method using “digital twins” to help LLM better understand video context. GLUS [294] propose to migrate local and global reasoning with one transformer network.
- •Other Variants and Settings. Recent studies explore alternative RVOS variants [126], [295], [296], [297], [298]. SAMA [126], MoRA [299], and ViCaS [300] extend the visual questioning and answering to segmentation answers, enables multi-turn dialogue or fine-grained segmentation. VideoGLaMM [295] leverages LLMs for grounded conversations, requiring text responses anchored at the pixel level in video frames. FS-RVOS [297], [301] introduces a few-shot setting [302] to address scenarios with limited annotated samples. ActionVOS [298] focuses on egocentric videos, using action narrations as additional language prompts to segment active objects. Beyond general-scene video, RVOS can also be adapted to specific videos, such as medical or sugical videos [303], [304].

6 REFERRING AUDIO-VISUAL SEGMENTATION

###### 6.1 Audio-Visual Segmentation

- 6.1.1 Fully Supervised Learning

• Better Representation. Feature representations extracted from audio and visual encoders are critical to AVS. Recent works [25], [305], [306], [307], [308], [309] focus on enhancing these representations. Lin et al. [308] demonstrate that Vision Transformers serve as parameter-efficient audio-visual learners. TeSO [25] enhances audio guidance by generating scene-level descriptions with

LLMs and extracting sounding-object cues via Chain-of-Thought prompting. QDFormer [306] uses product quantization to disentangle multi-source semantics into noise-suppressed components and applies global-to-local distillation to refine frame-level audio features. COMBO [305] leverages foundation model priors for precise representations, while ECMVAE [310] factorizes features into shared and modality-specific components.

- • Enhancing Multi-Modal Interaction. Recent methods [305], [307], [311], [312], [313], [314] focus on improving audio-visual fusion and alignment. DeepAVFusion [311] performs early fusion using learnable tokens to integrate audio-visual patches in parallel or sequentially. GAVS [312] adopts an encoder–prompt–decoder framework to leverage SAM’s generalization capacity. Dolphin [315] achieves fine-grained spatial-temporal alignment via multi-scale adapters and interleaved fusion, projecting features into an LLM for joint understanding. RAVS [316] mitigates audio ambiguity by clustering visual features by semantic density, weighting them by audio responsiveness, and modeling uncertainty for rapid transitions. DDESeg [317] disentangles mixed audio into semantic cues, filters noise via visual context, and enhances alignment through modality-specific discriminability.
- •Temporal Processing. Several AVS methods [305], [318], [319], [320], [321] explicitly address temporal modeling. COMBO [305] enforces temporal coherence via an adaptive inter-frame consistency loss based on cosine similarity of adjacent masks. UFE [320] employs temporal partitioning strategy, using neighboring frames for motion guidance and distant ones to enhance data diversity.
- • Improving Training Objectives. Some methods [104], [130], [319], [322], [323], [324], [325] aim to improve AVS performance by optimizing training objectives. PIF [323] decomposes AVS into two subtasks: correlation learning, which aligns audio with visible individuals to provide positional priors, and segmentation refinement, which produces masks. While transformer-based methods [314] have advanced AVS, they suffer from cross-attention inefficiency and unstable bipartite matching. CPM [322] addresses these limitations by introducing a hybrid query strategy combining class-agnostic and class-conditional queries, and further enhances training with a prompt-based joint audio-visual contrastive objective. In contrast, TransAVS [104] improves audio query diversity using self-supervised losses at both query and mask levels. Other methods [324], [325] introduce silent-object-aware objectives to ensure the segmentation of all potential sounding objects. 6.1.2 Weakly Supervised/Unsupervised Methods
- • Weakly-Supervised AVS. WS-AVS [326] proposes a weaklysupervised method using instance-level annotations and a multiscale contrastive learning to improve cross-modal alignment.
- •Unsupervised AVS. Recent woks such as MoCA [327] leverage foundation models (e.g., SAM [127], ImageBind [328]) to achieve competitive AVS performance without manual annotations.

###### 6.2 Referring Audio-Visual Segmentation

Ref-AVS [96] extends referring segmentation to the audio-visual domain, aiming to segment specific objects in video using synchronized audio and free-form language expressions. To support this task, Wang et al. [96] introduce the first Ref-AVS benchmark with over 3,000 videos annotated with masks and text expressions. The proposed framework employs modality-specific encoders for audio, vision, and text, fused via a query-based decoder. Ref-AVS handles scenarios beyond the scope of traditional AVS or RVOS,

“saxophone” “bass+guitar” “bass+keyboard”

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

AVS

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

RefAVS

[Figure 108]

##### Expression:

“The one blown to create a soulful melody.”

[Figure 109]

[Figure 110]

[Figure 111]

Omni AVS

[Figure 112]

##### Expression:

“The instrument is played by someone who looks like <image: > and makes a similar sound in <sound: sound of bass >.”

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Fig. 8: Comparison of different audio-visual segmentation tasks.

such as “person singing bass in a cappella group”, by leveraging multimodal cues for fine-grained disambiguation. It also shows that adding language improves the discriminative power of audio cues [25]. Omni-R1 [329] further enhances performance on RefAVS using reinforcement learning to select keyframes and rewrite tasks, enabling efficient Ref-AVS with just one training epoch. STBridge [296] bridges the modality gap between speech and text, allowing RVOS models to effectively process noisy spoken input. SAM2-LOVE [330] introduces a multimodal fusion framework that integrates audio, text, and visual information into learnable tokens to prompt SAM2 [128] for Ref-AVS.

Despite its potential, Ref-AVS is still in its early stages. Challenges include temporal misalignment, underused spatial audio, and limited dataset diversity. Future work may explore spatialized audio, adaptive fusion, and broader benchmarks.

###### 6.3 Omnimodal Referring Audio-Visual Segmentation

Omnimodal Audio-Visual Segmentation (OmniAVS) [10] is a recently proposed task that extends traditional text-only referring audio-visual segmentation by introducing omnimodal expressions, flexibly combining text, speech, sound, and visual cues, see Fig. 8. This formulation yields 8 types of expressions: 1) Text, 2) Speech, 3) Text + Sound, 4) Speech + Sound, 5) Text + Image, 6) Speech + Image, 7) Text + Sound + Image, and 8) Speech + Sound + Image. To support this task, Ying et al. [10] introduce the first benchmark, OmniAVS, comprising 59,458 omnimodal expressions for 4,262 objects across 2,098 auditory videos. They propose Omnimodal Instructed Segmentation Assistant (OISA) as baseline, integrating a MLLM with a flexible mask head. OISA uses an audiovisual interleaving strategy to align content across modalities and generates target-aware [seg] tokens for segmentation, enabling accurate tracking and understanding in complex scenes.

While OISA demonstrates promising performance, several open challenges remain in OmniAVS: (i) Improving audio-visual fusion via more robust joint representations, rather than relying on post-hoc alignment; (ii) Disentangling overlapping sound sources when multiple objects emit sounds simultaneously; (iii) Effectively combining multiple modalities within expressions through unified representations or cross-modal fusion; (iv) Enhancing segmentation robustness in complex scenarios such as occlusion,

disappearance, and reappearance; (v) Pretraining on larger-scale audio-visual datasets to improve generalization to unseen settings.

7 REFERRING 3D SEGMENTATION

7.1 3D Referring Expression Segmentation 7.1.1 Fully Supervised Learning

- • Two-Stage Methods. Two-stage methods first generate object proposals via 3D instance segmentation, followed by languageguided identification of the target object. TGNN [61] is a pioneering method in this paradigm, extracting high-quality instance masks and features, then modeling instance–language relationships using a Graph Neural Network, followed by multimodal feature aggregation for final prediction. Following this, X-RefSeg3D [331] enhances cross-modal interaction by constructing a scene graph that integrates linguistic entities into visual representations, and employs dual-stream relation reasoning through textual and spatial modules to more effectively identify the target object.
- • One-Stage Methods. One-stage methods perform end-toend segmentation by directly fusing visual and linguistic features, offering improved efficiency over two-stage pipelines. 3DSTMN [332] introduces a superpoint-text matching (STM) mechanism to align superpoints with referring expressions, enabling faster inference and improved multimodal alignment. LESS [333], built on SPFormer [334], adopts a one-stage pipeline and introduces an area regularization loss and point-to-point contrastive loss to mitigate interference from surrounding objects and background clutter. RefMask3D [77] proposes a Linguistic Primitives Construction (LPC) module to learn fine-grained semantic primitives, enhancing vision-language alignment during decoding. RGSAN [335] improves spatial reasoning by integrating text-driven localization with rule-guided weak supervision to model interobject spatial relationships. IPDN [336] incorporates multi-view image features to enrich 3D representations and employs taskguided prompts to focus on language-relevant targets, addressing point cloud incompleteness and semantic ambiguity. While most methods focus on single-object segmentation, 3D-GRES [11] extends GRES [3] to support an arbitrary number of target objects in point clouds, enabling generalized 3D referring segmentation.
- • 3D Reasoning Segmentation. Motivated by the success of LLMs/MLLMs [221], [222] in 2D reasoning segmentation [21], recent studies have begun exploring their potential in 3D domain. SegPoint [9] and Reason3D [337] are early efforts that integrate LLMs/MLLMs to enable reasoning in 3D-RES, primarily focusing on single-object or single-category settings. To handle more complex scenarios, MORE3D [338] extends this capability to multiobject reasoning and additionally generates textual explanations alongside segmentation outputs. 3D-LLaVA [339] further introduces a unified vision-language framework for 3D tasks, jointly addressing question answering, dense captioning, and referring segmentation while achieving competitive performance.
- • Multi-Task Learning. Joint learning of 3D Referring Segmentation (3D-RES) and Referring Expression Comprehension (3DREC) is a natural extension of multi-task learning, as demonstrated in 2D vision [68], [181], where shared representations benefit related tasks. 3DRefTR [340] follows this paradigm by extending a 3D-REC model through reuse of query embeddings and visual tokens, adding a mask prediction branch to support 3D-RES with minimal overhead. MCLN [341] further explores task synergy by jointly optimizing separate branches for 3D-REC and 3D-RES, leveraging their semantic and structural alignment.

Beyond pairwise integration, recent efforts aim to develop unified models for a broad range of 3D vision-language tasks. Uni3DL [342] introduces a versatile framework that supports tasks such as semantic/instance segmentation, referring segmentation, captioning, retrieval, and object classification on raw point clouds via shared query modeling and dynamic task routing. Similarly, UniSeg3D [343] presents a unified architecture for six segmentation tasks, including open-vocabulary and referring segmentation, using shared encoders and decoders, and enhancing performance through inter-task knowledge distillation and contrastive learning.

- 7.1.2 Weakly Supervised/Unsupervised Methods To reduce reliance on manual labeling, recent studies explore limited-supervision settings. 3D-REST [344] addresses the semisupervised setting by generating high-quality pseudo-labels and reweighting low-confidence predictions based on reliability, enabling effective training with fewer annotations. MEN [345] tackles the weakly supervised setting without mask labels by integrating multimodal cues, e.g., global context, fine-grained attributes, and category priors, to guide accurate segmentation.

###### 7.2 Referring 3D Gaussian Splatting Segmentation

To advance research in Referring 3D Gaussian Splatting Segmentation (R3DGS), ReferSplat [5] introduces a framework that links 3D Gaussians with language via spatially aware referring fields, and proposes Ref-LERF, the first dedicated dataset for this task. Each Gaussian is assigned a learnable feature vector that interacts with textual queries to produce segmentation masks through similarity-based rendering.

Despite recent progress, several challenges remain in R3DGS. First, existing method [5] is limited to static scenes and lack temporal modeling. Integrating 4D representations, e.g., 4D Gaussian Splatting [346], could enable reasoning in dynamic environments. Second, current method emphasizes segmentation but overlook fine-grained grounding, such as precise localization and size estimation, which are critical for spatial reasoning. Finally, the limited scale and diversity of existing dataset restrict generalization. Advancing the field requires the development of large-scale, diverse datasets to bridge the gap with more mature 2D counterparts.

#### 8 GENERALIZED REFERRING EXPRESSION

Generalized Referring Expression tasks (GREx) [3], [11], [32], [347], [348] aim to segment or detect an arbitrary number of target objects in visual scenes based on free-form referring expressions. This flexibility makes GREx more applicable to real-world scenarios and highlights it as a promising direction for future research.

• Generalized Referring Expression Segmentation (GRES). Liu and Ding et al. [3] first introduce the GRES task, which extends classic RES to further support multi-target and no-target expressions. They also propose ReLA, a region-based framework that segments images into semantically meaningful sub-instance regions and explicitly models region–region and region–language dependencies. Following this formulation, numerous works [3], [47], [150], [174], [349], [350], [351], [352], [353], [354] have advanced GRES toward more practical and generalizable scenarios. Shah et al. [150] identify a limitation in some GRES methods, including ReLA, termed query collapse, where all queries produce identical mask predictions due to traditional RES’s single-mask constraint. To mitigate this, they propose dynamic query generation conditioned on expressions, along with a regularization

strategy to diversify query representations. DMMI [47] introduces a dual-branch decoder enabling bidirectional interaction: one branch guides visual localization using text, while the other reconstructs expressions from visual features to enforce semantic consistency. GSVA [174] extends the reasoning segmentation paradigm of LISA [21] to the GRES setting by leveraging MLLMs and adapting [SEG] tokens for multi-target references. It further introduces a [REJ] token to to explicitly handle no-target cases.

- • Generalized Referring Expression Comprehension (GREC). He and Ding et al. [32] introduce the GREC task, an extension of classic REC that allows expressions to refer to any number of target objects. They also propose new evaluation metrics tailored for this setting. As a representative method, HieA2G [33] presents a hierarchical multimodal semantic alignment framework that facilitates cross-modal interactions at word-object, phraseobject, and text-image levels. To accommodate varying numbers of targets, it incorporates an adaptive counting mechanism.
- • 3D Generalized Referring Expression Segmentation (3DGRES). 3D-GRES [11] extends GRES [3] to the 3D domain, aiming to segment an arbitrary number of target objects in point cloud scenes. The framework employs text-guided query generation and optimization to enable effective interaction between queries, point cloud features, and language, while maintaining semantic consistency across multiple targets.
- • 3D Generalized Referring Expression Comprehension (3DGREC). Zhang et al. [48] introduce the first 3D-GREC dataset, Multi3DRefer. They propose a CLIP-based method with multimodal contrastive learning to enable online rendering of proposal objects for generating 2D visual cues. 3D DOG [355] handles paragraph-level grounding of multiple objects, while GNL3D [356] proposes group-wise grounding across related 3D scenes with flexible target counts.

9 RELATED TASKS AND APPLICATIONS

- •Referring Expression Comprehension (REC). REC [34], [60], [68], [357], [358] aims to detect a target object in an image based on a text referring expression by predicting a bounding box around the described object. Unlike RES, REC focuses on target object detection by bounding box rather than pixel-level segmentation.
- • 3D Referring Expression Comprehension (3D-REC). 3DREC [45], [46], [359], [360] aims to detect a target object in a 3D scene based on a text referring expression. This task takes a 3D point cloud and a referring expression as input, and outputting a 3D bounding box around the described object.
- •Referring Video Object Tracking (RVOT). RVOT [361], [362],

- [363] aims to track a single target object throughout a video based on a text referring expression. The model takes as input a video and a natural language description, and produces a trajectory of bounding boxes for the target object across frames.

• Referring Multi-Object Tracking (RMOT). RMOT [121],

- [364] tracks multiple objects based on text expressions. TransRMOT [121] uses decoupled object queries for detection and tracking, while iKUN [364] adopts a two-stage framework with a language tracker and a Neural Kalman Filter for adaptive tracking.

• Referring Expression Generation (REG). REG [357], [365] aims to generate unambiguous textual referring expressions that uniquely identify specific objects in a visual scene. Some works [99], [131] leverage the synergies between RES and REG through multi-task learning approaches.

[Figure 118]

[Figure 119]

[Figure 120]

“Please give me my glasses.”

“Replace the cow in the back with a panda.”

[Figure 121]

[Figure 122]

Referring Grasp Synthesis

Step 1: Referring Expression Segmentation

Step 2: Grasp Pose Estimation

[Figure 123]

[Figure 124]

[Figure 125]

Output Mask

Output Grasp Parameters

(a) Referring Grasp Synthesis

(b) Referring Image Editing

Fig. 9: Illustration of Several Applications.

- •Phrase Grounding (PG). PG [366], [367] aims to ground each entity mentioned by a noun phrase in an image caption to its corresponding region in the image.
- • Panoptic Narrative Grounding (PNG). PNG [368], [369] focuses on segmenting both foreground objects and background stuff that are described in detailed narrative captions of an image. While similar to RES, PNG requires the model to identify and segment entities mentioned throughout the entire caption text.
- •Referring Expression Counting. This task [370], [371] aims to count object subsets specified by free-form expressions, requiring fine-grained grounding (e.g., “green grapes” vs. “purple grapes”).
- •Referring Grasp Synthesis (RGS). RGS [372], [373] predicts grasp poses for objects described by language, combining visionlanguage grounding with robotic control. It typically follows a two-stage pipeline: grounding and grasp prediction, see Fig. 9(a).
- •Referring Image Editing (RIE). RIE [12] aims to identify and modify specific objects in an image based on referring expression, see Fig. 9(b). Unlike general image editing [13], which often applies global changes, or region-based editing that relies on masks or bounding boxes, RIE focuses on editing objects grounded by referring expressions. This ensures precise region-level editing and also provides flexible and natural user interaction.

10 CONCLUSION AND DISCUSSION

This survey presents a comprehensive overview of multimodal referring segmentation across image, video, and 3D domains. We unify the taxonomy of task settings, summarize a metaarchitecture, and analyze representative methods and benchmarks from classic RES and RVOS to emerging tasks like GRES, OmniAVS, and reasoning segmentation, offering a holistic view of the field’s evolution. Recent trends reflect a clear shift toward more general, human-centric, and reasoning-driven models:

- • Omnimodal Understanding. OmniAVS demonstrates the need for models to flexibly interpret text, speech, sound, and visual cues. This reflects a broader trend toward promptable and generalist models capable of handling diverse multimodal inputs.
- • Generalization for Practical Scenarios. Generalized Referring Expression Segmentation (GRES) extends classical settings by supporting multi-target and no-target expressions. It promotes open-world segmentation and robustness to ambiguous language.
- • Motion-Centric Video Understanding. MeViS pushes the boundary of RVOS by introducing motion-centric expressions and complex temporal dependencies. It encourages models to perform holistic video reasoning and dynamic object tracking.
- • Foundation Models and Reasoning Segmentation. Large vision and language models, such as SAM/SAM2 and MLLMs

like LLaVA, have reshaped the landscape of referring segmentation. They enable prompt-based interfaces and spark progress in reasoning segmentation, where models must interpret abstract, indirect, or knowledge-intensive instructions.

Looking ahead, several research directions remain open: 1) Developing generalist segmentation agents that scale across modalities and tasks; 2) Enabling deeper commonsense and temporal reasoning under limited supervision; 3) Enhancing model robustness, efficiency, and interpretability for real-world deployment; and 4) Establishing unified benchmarks to evaluate crossmodal, open-world, and reasoning capabilities.

#### APPENDIX: PERFORMANCE COMPARISON

In this section, we present performance comparison of multimodal referring segmentation methods. For each field, the most widely used datasets, as outlined in Sec. 2, are selected for benchmarking. Given the large volume of publications in this area, we selectively report results from representative methods published in top-tier conferences and journals.

A.1 RES Performance Benchmarking Evaluation Metrics. For Referring Expression Segmentation (RES) evaluation, four primary metrics are commonly used:

- • Intersection over Union (IoU) measures the overlap between predicted (Mp) and ground truth (Mgt) masks: IoU = area(Mp ∩ Mgt)/area(Mp ∪ Mgt). It serves as a fundamental metric for evaluating model accuracy in identifying and segmenting regions.
- • Mean Intersection over Union (mIoU) is defined by the

average of all per-image Intersection-over-Unions (IoUs): mIoU = N i=1 IoUi/N, where N is the total number of data samples. This

metric provides a comprehensive evaluation of overall segmentation performance across the entire dataset.

• Cumulative Intersection over Union (cIoU) is defined by the cumulative intersection over the cumulative union: cIoU =

N i=1 area(Mpi ∩ Mgti )/ Ni=1 area(Mpi ∪ Mgti ). It is worth noting that cIoU is highly biased toward large-area objects and tends to fluctuate significantly, which can affect the reliability of performance assessments.

• Precision@X (Pr@X) measures the percentage of predictions that achieve an IoU score higher than a given IoU threshold X.

RES Benchmark Results. As shown in TABLE 2, we present the performance comparison of recent RES methods on three widely-used benchmarks: RefCOCO [8], RefCOCO+ [8], and RefCOCOg [374]. The field has witnessed substantial progress since 2016, with recent methods achieving remarkable improvements across all datasets. For conventional fully-supervised methods, OneRef-L [66] achieves the best performance among conventional fully-supervised methods with 75.68% mIoU on RefCOCOg validation set and 76.82% mIoU on the test set, followed closely by UNINEXT [194] with 74.70% and 76.40% mIoU respectively. These results demonstrate remarkable progress compared to early methods like LSTM-CNN [6] from 2016, which achieved only 34.06% mIoU on RefCOCOg validation set. For Weakly/SemiSupervised, Zero-shot and Generalist Methods, ADDP [178] achieves the best performance on RefCOCOg with 59.05% mIoU on the validation set and 59.60% on the test set, outperforming several conventional fully-supervised methods despite using less supervision. Among MLLM-based methods, SAM4MLLM [175] achieves the best performance with 75.50% mIoU on RefCOCOg val set and 76.40% on the test set, followed by POPEN [240] with

- TABLE 2: Performance Comparison on RES benchmarks. The results are evaluated on RefCOCO [8], RefCOCO+ [8], and RefCOCOg [374] (UMD partition) using mIoU. †: results post-processed with DenseCRF [375]. ∗: methods utilizing 30% mask annotations and 70% bounding box annotations. ‘ft’ denotes fine-tuning on referring expression segmentation datasets. ‡ indicates methods using only 5% labeled data. UNINEXT [194] utilizes val/test set masks during training, leading to mask information leakage.

RefCOCO RefCOCO+ RefCOCOg val test A test B val test A test B val test

Method Venue

- a. Conventional Fully-supervised methods. LSTM-CNN [6] [ECCV’16] - - - - - - 34.06 -

RMI† [81] [ICCV’17] 45.18 45.69 45.57 29.86 30.48 29.50 34.52 DMN [80] [ECCV’18] 49.78 54.83 45.13 38.88 44.22 32.29 - KWA [82] [ECCV’18] - - - - - - 36.92 -

RRN† [83] [CVPR’18] 55.33 57.26 53.93 39.75 42.15 36.11 36.45 MAttNet [60] [CVPR’18] 56.51 62.37 51.70 46.67 52.39 40.08 47.64 48.61 CMSA† [110] [CVPR’19] 58.32 60.61 55.09 43.76 47.60 37.89 39.98 LSCM† [169] [ECCV’20] 61.47 64.99 59.55 49.34 53.12 43.50 - -

ConvLSTM† [167] [TMM’20] 59.04 60.74 56.73 44.54 47.92 39.73 41.77 -

MCN [68] [CVPR’20] 62.44 64.20 59.71 50.62 54.99 44.69 49.22 49.40 BRINet [97] [CVPR’20] 60.98 62.99 59.21 48.17 52.32 42.11 - -

STEP (5-fold) [69] [NeurIPS’21] 60.04 63.46 57.97 48.19 52.33 40.41 46.40 Referring Transformer [120] [NeurIPS’21] 74.34 76.77 70.87 66.75 70.58 59.40 66.63 67.39

EFN [98] [CVPR’21] 62.76 65.69 59.67 51.50 55.24 43.01 51.93 BUSNet [168] [CVPR’21] 63.27 66.41 61.39 51.76 56.87 44.13 - -

LTS [176] [CVPR’21] 65.43 67.76 63.08 54.21 58.32 48.02 54.40 54.25 CMPC† [110] [TPAMI’21] 61.36 64.53 59.64 49.56 53.44 43.23 - -

SeqTR [181] [ECCV’22] 67.26 69.79 64.12 54.14 58.93 48.19 55.67 55.64 CoupAlign [156] [NeurIPS’22] 74.70 77.76 70.58 62.92 68.34 56.69 62.84 62.22

ReSTR [87] [CVPR’22] 67.22 69.30 64.45 55.78 60.44 48.27 - CRIS [91] [CVPR’22] 70.47 73.18 66.10 62.27 68.08 53.68 59.87 60.36

LAVT [84] [CVPR’22] 72.73 75.82 68.79 62.14 68.38 55.10 61.24 62.09 BVPR [145] [CVPR’22] 67.01 69.63 63.45 55.34 60.72 47.11 55.09 55.31

BKINet [146] [TMM’23] 73.22 76.43 69.42 64.91 69.88 53.39 64.21 63.77 M3Att [99] [TIP’23] 73.60 76.23 70.36 65.34 70.50 56.98 64.92 67.37 ETRIS [92] [ICCV’23] 70.51 73.51 66.63 60.10 66.89 50.17 59.82 59.91

VPD [139] [ICCV’23] 73.25 - - 62.69 - - 61.96 TRIS [205] [ICCV’23] 41.10 48.10 31.90 31.60 31.90 30.60 39.00 39.90 PolyFormer [85] [CVPR’23] 75.96 77.09 73.22 70.65 74.51 64.64 69.36 69.88 CGFormer [149] [CVPR’23] 76.93 78.70 73.32 68.56 73.76 61.72 67.57 67.83

VG-LAW [138] [CVPR’23] 75.62 77.51 72.89 66.63 70.38 58.89 65.63 66.08 UNINEXT [194] [CVPR’23] 82.20 83.40 81.30 72.50 76.40 66.20 74.70 76.40

ReLA [3] [CVPR’23] 73.82 76.48 70.18 66.04 71.02 57.65 65.00 65.97

VLT [4], [7] [TPAMI’23] 65.65 68.29 62.73 55.50 59.20 49.36 52.99 56.65 CM-MaskSD [102] [TMM’24] 74.89 77.54 71.28 67.47 71.80 59.91 66.53 66.63

ReMamber [151] [ECCV’24] 71.60 73.30 68.40 61.60 65.80 54.00 61.10 61.20 Barleria [161] [ICLR’24] 72.40 75.90 68.30 65.00 70.80 56.90 63.40 63.80

UniRES [50] [CVPR’24] 79.20 81.60 76.60 73.00 78.10 65.80 71.70 73.20 MagNet [155] [CVPR’24] 76.55 78.27 72.15 68.10 73.64 61.81 67.79 69.29

LQMFormer [150] [CVPR’24] 74.16 76.82 71.04 65.91 71.84 57.59 64.73 66.04 Prompt-RIS [172] [CVPR’24] 78.10 81.21 74.64 71.13 76.60 64.25 70.47 71.29 OneRef-L [66] [NeurIPS’24] 81.26 83.06 79.45 76.60 80.16 72.95 75.68 76.82

- b. Weakly/Semi- Supervised, Zero-shot and Generalist Methods. Weakly-RIS [201] [ICCV’23] 31.06 32.30 30.11 31.28 32.11 30.13 32.88 -

SaG [199] [ICCV’23] 44.60 50.10 38.40 35.50 41.10 27.60 - Global-Local CLIP [213] [CVPR’23] 26.20 24.94 26.56 27.80 25.64 27.84 33.52 33.67

X-Decoder [189] [CVPR’23] - - - - - - 64.60 Partial-RES∗ [180] [CVPR’23] 66.24 68.39 63.57 54.37 58.16 47.92 54.69 54.81

GTMS [208] [ECCV’24] 66.54 69.98 63.41 57.59 63.46 50.32 54.52 54.75 SAFARI∗ [212] [ECCV’24] 67.04 69.17 64.23 54.98 59.31 48.26 55.72 55.83 SemiRES‡ [209] [ICML’24] 61.31 66.64 55.94 47.00 54.42 38.74 47.61 50.11

PPT [198] [CVPR’24] 46.76 45.33 46.28 45.34 45.84 44.77 42.97 SEEM [190] [NeurIPS’24] - - - - - - 65.70 PCNet [206] [NeurIPS’24] 52.20 58.40 42.10 47.90 56.50 36.20 46.80 46.90

HybridGL [219] [CVPR’25] 49.48 53.37 45.19 43.40 49.13 37.17 51.25 51.59 ADDP [178] [ICLR’25] 69.14 70.27 67.46 57.58 61.65 51.76 59.05 59.60

- c. MLLMs based Methods. LISA-7B (ft) [21] [CVPR’24] 74.90 79.10 72.30 65.10 70.80 58.10 67.90 70.60 PixelLm-7B [223] [CVPR’24] 73.00 76.50 68.20 66.30 71.70 58.30 69.30 70.50

SESAME-7B [236] [CVPR’24] 74.70 - - 64.90 - - 66.10 AnyRef-7B (ft) [196] [CVPR’24] 76.90 79.90 74.20 70.30 73.50 61.80 70.00 70.70 PerceptionGPT-13B [230] [CVPR’24] 75.30 79.10 72.10 68.90 74.00 61.90 70.70 71.90

GSVA-7B (ft) [174] [CVPR’24] 77.20 78.90 73.50 65.90 69.60 59.80 72.70 73.30 GLaMM-7B [250] [CVPR’24] 79.50 83.20 76.90 72.60 78.70 64.60 74.20 74.90 CoReS-7B [231] [ECCV’24] 76.00 78.60 72.50 65.10 70.00 58.60 69.00 70.70

SAM4MLLM-8B [175] [ECCV’24] 79.80 82.70 74.70 74.60 80.00 67.20 75.50 76.40 M2SA-13B [237] [ICLR’25] 74.60 77.60 71.00 64.00 68.10 57.60 69.00 69.30 SegLLM-7B [67] [ICLR’25] 80.20 81.50 75.40 70.30 73.00 62.50 72.60 73.60

READ-7B [233] [CVPR’25] 78.10 80.20 73.20 68.40 73.70 60.40 70.10 71.40 POPEN-7B (ft) [240] [CVPR’25] 79.30 82.00 74.10 73.10 77.00 65.10 75.40 75.60

- TABLE 3: Performance Comparison on RVOS Benchmarks. The results are evaluated on MeViS [1], Ref-YouTube-VOS [40], and Ref-DAVIS17 [36] datasets with J and F metrics.

MeViS Ref-YouTube-VOS Ref-DAVIS17

Method Venue

J &F J F J &F J F J &F J F

URVOS [40] [ECCV’20] 27.80 25.70 29.90 47.23 45.27 49.19 51.63 47.29 55.96 CMPC [108] [TPAMI’21] - - - 47.48 45.64 49.32 - - MLRL [268] [CVPR’22] - - - 49.70 48.43 50.96 57.94 53.85 62.02 LBDT [113] [CVPR’22] 29.30 27.80 30.80 49.38 48.18 50.57 54.52 - MTTR [286] [CVPR’22] 30.00 28.80 31.20 55.32 54.00 56.64 - - -

ReferFormer [275] [CVPR’22] 31.00 29.80 32.20 62.90 61.30 64.60 61.10 58.10 64.10 EFCMA [111] [TPAMI’22] - - - 48.97 47.82 50.12 50.23 47.37 53.08 HTML [262] [ICCV’23] - - - 63.40 61.50 65.20 62.10 59.20 65.10

SgMg [272] [ICCV’23] - - - 65.70 63.90 67.40 63.30 60.60 66.00 TempCD [260] [ICCV’23] - - - 65.80 63.60 68.00 64.60 61.60 67.60 UniRef++ [86] [ICCV’23] - - - 66.90 64.80 69.00 67.20 63.40 70.90

OnlineRefer [257] [ICCV’23] 32.30 31.50 33.10 63.50 61.60 65.50 64.80 61.60 67.70

LMPM [1] [ICCV’23] 37.20 34.20 40.20 - - - - - LASTC [261] [TPAMI’23] - - - 49.30 48.15 50.45 54.45 - Locater [112] [TPAMI’23] - - - 56.50 54.80 58.10 - - -

VD-IT [270] [ECCV’24] - - - 66.50 64.40 68.50 69.40 66.20 72.60 VISA-13B [41] [ECCV’24] 44.50 41.80 47.10 63.00 61.40 64.70 70.40 67.00 73.80 SOC [129] [NeurIPS’24] - - - 67.30 65.30 69.30 65.80 62.50 69.10 VideoLISA-3.8B [42] [NeurIPS’24] 42.30 39.40 45.20 61.70 60.20 63.30 67.70 63.80 71.50

UniVS [280] [CVPR’24] - - - 58.00 - - 59.40 - -

LoSh [118] [CVPR’24] - - - 67.20 65.40 69.00 64.30 61.80 66.80 DsHmp [22] [CVPR’24] 46.40 43.00 49.80 67.10 65.00 69.10 64.90 61.70 68.10 ViLLa [291] [ArXiv’24] 49.40 46.50 52.30 67.50 64.60 70.40 74.30 70.60 78.00

SAMWISE [123] [CVPR’25] 48.30 45.40 51.20 67.20 65.20 69.30 68.50 65.60 71.50 VRS-HQ-13B [116] [CVPR’25] 50.90 48.00 53.70 71.00 69.00 73.10 74.40 71.00 77.90

GLUS [294] [CVPR’25] 51.30 48.50 54.20 67.30 65.50 69.00 - - Sa2VA-26B [122] [ArXiv’25] 46.20 - - 70.10 - - 77.00 - -

RGA3-7B [376] [ICCV’25] 50.10 47.40 52.80 68.50 66.80 70.10 72.80 68.30 77.30 MPG-SAM 2 [279] [ICCV’25] 53.70 50.70 56.70 73.90 71.70 76.10 72.40 68.80 78.00

75.40% and 75.60%, respectively, demonstrating the effectiveness of integrating MLLMs into referring segmentation.

A.2 RVOS Performance Benchmarking Evaluation Metrics. Three metrics are frequently used to in Referring Video-Object Segmentation (RVOS) evaluation:

- • Region Jaccard J is calculated by the intersection-overunion (IoU) between the predicted segmentation mask Mp and the ground-truth Mgt: J = |Mp ∩ Mgt|/|Mp ∪ Mgt|, which computes the number of pixels of the intersection between Mp and Mgt, and divides it by the size of the union.
- • Boundary Accuracy F is the harmonic mean of the boundary

precision Pc and recall Rc. The value of F reflects how well the predicted contours match the ground-truth contours. Usually, the value of Pc and Rc can be computed via bipartite graph matching [377], then the boundary accuracy F can be computed as: F = 2PcRc/(Pc + Rc).

- • J &F is computed as the average of region similarity and contour accuracy: J &F = (J + F)/2, providing a comprehensive evaluation of both region and boundary accuracy.

RVOS Benchmark Results. As shown in TABLE 3, we present the performance comparison of recent RVOS methods on three widely-used benchmarks: MeViS [1], Ref-YouTubeVOS [40], and Ref-DAVIS17 [36]. The field has witnessed substantial progress since 2020, with recent methods like GLUS [294] and VRS-HQ [116] achieving remarkable improvements on all benchmarks. On the MeViS dataset, which focuses on temporal motion understanding, GLUS achieves the best performance with 51.30% J &F, followed closely by VRS-HQ with 50.90%. For Ref-YouTube-VOS, VRS-HQ leads with 71.00% J &F, significantly outperforming earlier methods like URVOS [40] (47.23%).

TABLE 4: Performance Comparison on AVS Benchmarks. The results are evaluated on AVSBench [43] and AVSBenchSemantic [44] with J and F score metrics. ∗ : weakly-supervised.

S4 MS3 AVSS

Method Venue

J F J F J F

AVSBench [43] [ECCV’22] 78.74 87.90 54.00 64.50 29.77 35.20 ECMVAE [310] [ICCV’23] 81.74 90.10 57.84 70.80 - -

LAVISH [308] [CVPR’23] 80.10 - - - - -

PIF [323] [TMM’24] 81.40 90.00 58.90 70.90 - BAVS [325] [TMM’24] 82.68 89.75 59.63 65.89 33.59 37.52 C3N [313] [TMM’24] 83.11 90.80 61.72 72.20 - -

CPM [322] [ECCV’24] 81.37 90.47 59.80 71.00 34.53 39.57 Stepping Stones [378] [ECCV’24] 83.20 91.30 67.30 77.60 48.50 53.20

TESO [25] [ECCV’24] 83.27 93.30 66.02 80.10 38.96 45.10

WS-AVS∗ [326] [NeurIPS’24] 34.13 51.76 30.85 46.87 - QDFormer [306] [CVPR’24] 79.50 88.20 61.90 66.10 53.40 -

COMBO [305] [CVPR’24] 84.70 91.90 59.20 71.20 42.10 46.10

VPO [379] [CVPR’24] 85.77 92.86 62.39 73.62 44.70 57.76 DeepAVFusion [311] [CVPR’24] 89.94 92.34 52.05 58.29 - -

RAVS [316] [CVPR’25] 93.10 93.80 70.60 82.10 60.80 70.60 DDESeg [317] [CVPR’25] 92.40 95.90 72.30 83.40 63.40 72.30

On Ref-DAVIS17, VRS-HQ also demonstrates superior performance with 74.40% J &F, showing substantial improvement over previous state-of-the-art methods.

###### A.3 R-AVS Performance Benchmarking

Evaluation Metrics. Region Jaccard J , boundary accuracy F are also widely adopted for Audio-Visual Segmentation (AVS) performance evaluation.

AVS Benchmark Results. As shown in TABLE 4, we present the performance comparison of recent AVS methods on AVSBench [43] (S4 and MS3 subsets) and AVSBench-Semantic [44] (AVSS) using J and F metrics. The field has seen significant progress since 2022, with recent methods like DDESeg [317] and

- TABLE 5: Performance Comparison on Ref-AVS Benchmark. The results are evaluated on Ref-AVS [96] dataset with J and F metrics. S denotes the square root of the ratio between the predicted mask area and the background area, with lower values indicating better performance.

Method Venue

Seen Unseen Mix(S+U) Null J F J F J F S (↓)

AVSBench [43] [ECCV’22] 23.2 51.1 32.4 54.7 27.8 52.9 20.8 ReferFormer [275] [CVPR’22] 31.3 50.1 30.4 48.8 30.9 49.5 17.6 R2VOS [351] [ICCV’23] 25.0 41.0 27.9 49.8 26.5 45.4 18.3 AVGSegFormer [314] [AAAI’24] 33.5 47.0 36.1 50.1 34.8 48.6 17.1 GAVS [312] [AAAI’24] 28.9 49.8 29.8 49.7 29.4 49.8 19.0 RefAVS [96] [ECCV’24] 34.2 51.3 49.5 64.8 41.9 58.1 0.7 TSAM [380] [CVPR’25] 43.4 56.8 54.6 66.4 - - 1.7 SAM2-LOVE [330] [CVPR’25] 43.5 51.9 66.5 72.3 55.0 62.1 23.0 OISA-1B [10] [ICCV’25] 51.7 58.7 58.3 65.1 54.5 61.4 9.8

- TABLE 6: Performance Comparison on OmniAVS Benchmark. The results are evaluated on OmniAVS [10] dataset with J &F. All is the average result across 8 splits. MET.: METEOR.

Method All I II III IV V VI VII VIII MET.

LMPM [1] 25.8 31.2 28.7 20.0 22.7 21.3 20.9 30.0 31.4 EEMC [96] 29.6 34.4 32.6 19.6 26.0 28.0 24.7 35.6 36.0 MUTR [282] 32.3 35.4 33.3 28.4 29.8 26.5 22.8 41.6 40.5 LISA-7B [21] 33.6 33.3 31.2 29.2 32.7 28.6 27.3 43.4 43.1 11.6 LISA-13B [21] 36.1 36.4 32.1 30.4 35.7 31.6 30.2 46.7 45.7 16.5 OISA-1B [10] 41.1 40.1 38.5 34.9 38.5 35.9 35.2 52.6 53.0 21.7

RAVS [316] achieving remarkable improvements on all benchmarks. DDESeg achieves the best performance on S4 with 92.40% J and 95.90% F, as well as on MS3 with 72.30% J and 83.40% F. For the more challenging AVSS benchmark, DDESeg also leads with 63.40% J and 72.30% F, demonstrating the effectiveness in audio-visual semantic segmentation task.

Ref-AVS Benchmark Results. As shown in TABLE 5, we present the performance comparison of recent Ref-AVS methods on the Ref-AVS [96] dataset using J and F metrics across different scenarios. The field has seen remarkable progress since 2022, with recent methods like OmniAVS [10] achieving the best overall performance with 51.7% J and 58.7% F on the Seen set, and 58.3% J and 65.1% F on the Unseen set. TSAM [380] and SAM2-LOVE [330] also demonstrate competitive results, with TSAM achieving 43.4% J and 56.8% F on Seen, and SAM2LOVE reaching 66.5% J and 72.3% F on Unseen scenarios. These results highlight the significant advancement in audio-visual referring segmentation capabilities, substantially outperforming earlier methods like AVSBench [43] and ReferFormer [275].

OmniAVS Benchmark Results. As shown in TABLE 6, we present the performance comparison of recent methods on the OmniAVS [10] dataset using J &F metric across eight different splits and METEOR for caption evaluation. OISA-1B [10] achieves the best overall performance with 41.1% J &F across all splits and 21.7 METEOR score, demonstrating significant improvements over previous methods. MUTR [282] and LISA13B [21] also show competitive results with 32.3% and 36.1% overall performance respectively. The results across different splits show varying levels of difficulty, with splits VII and VIII generally achieving higher performance (e.g., OISA-1B’s 52.6% on VII) compared to splits III and IV (e.g., OISA-1B’s 34.9% on III), indicating the heterogeneous nature of the benchmark and the challenges in multi-modal audio-visual understanding.

- TABLE 7: Performance Comparison on 3D-RES Benchmarks. The results are evaluated on ScanRefer [45] dataset with Acc@K and mIoU metric. ∗ denotes weakly-supervised methods.

Method Venue

Unique (∼19%) Multiple (∼81%) Overall

mIoU Acc@0.25 Acc@0.5 Acc@0.25 Acc@0.5 Acc@0.25 Acc@0.5

TGNN [61] [AAAI’21] - - - - 37.50 31.40 28.80

X-RefSeg3D [331] [AAAI’24] - - - - 40.33 33.77 29.94 3D-STMN [332] [AAAI’24] 89.30 84.00 46.20 29.20 54.60 39.80 39.50 RefMask3D [77] [MM’24] 89.55 84.69 48.09 40.77 55.87 49.24 44.86

MDIN [11] [MM’24] 91.00 87.20 50.10 44.90 58.00 53.10 48.30 SegPoint [9] [ECCV’24] - - - - - - 41.70

MCLN [341] [ECCV’24] 89.57 78.22 53.28 45.88 58.70 50.70 44.72 RG-SAN∗ [335] [NeurIPS’24] 89.20 84.30 55.00 35.40 61.70 44.90 44.60 UniSeg3D [343] [NeurIPS’24] - - - - 41.50 28.00 29.60

LESS∗ [333] [NeurIPS’24] - - - - 53.23 29.88 33.74 Reason3D [337] [3DV’25] 88.40 84.20 50.50 31.70 57.90 41.90 42.00

3D-LLaVA [339] [CVPR’25] - - - - - - 43.30 IPDN [336] [AAAI’25] 91.50 88.00 53.10 47.00 60.60 54.90 50.20

- TABLE 8: Performance Comparison on GRES Benchmarks. The results are evaluated on gRefCOCO [3] dataset with the cIoU and mIoU metric. ‡ denotes zero-shot method.

Val testA testB cIoU gIoU cIoU gIoU cIoU gIoU

Method Venue

MattNet [60] [CVPR’18] 47.51 48.24 58.66 59.30 45.33 46.14 CRIS [91] [ICCV’21] 55.34 56.27 63.82 63.42 51.04 51.79 LTS [176] [CVPR’21] 52.30 52.70 61.87 62.64 49.96 50.42

LAVT [84] [CVPR’22] 57.64 58.40 65.32 65.90 55.04 55.83 ReLA [3] [CVPR’23] 62.42 63.60 69.26 70.03 59.88 61.02 VLT [4] [TPAMI’23] 52.51 52.00 62.19 63.20 50.52 50.88 LaSagnA‡ [235] [arXiv’24] 38.10 32.40 50.40 47.30 42.10 38.90 CoHD [354] [arXiv’24] 65.17 68.42 71.85 72.67 62.63 63.60 HDC [381] [arXiv’24] 65.42 68.23 71.60 72.52 62.79 63.85

MABP [352] [arXiv’24] 65.72 68.86 71.59 72.81 62.76 64.04 InstAlign [353] [arXiv’24] 68.94 74.34 73.22 74.51 63.88 65.74 LISA-13B [21] [CVPR’24] 63.96 65.24 71.00 69.99 62.29 62.11

LQMFormer [150] [CVPR’24] 64.98 70.94 - - - GSVA-13B [174] [CVPR’24] 66.38 70.04 72.79 73.29 63.20 65.45

HieA2G [33] [AAAI’25] 64.20 68.40 70.40 72.00 61.00 62.80 PSALM-G5 (ft) [348] [CVPR’25] 68.00 67.30 75.20 77.30 73.10 78.90

UniRES++ [52] [arXiv’25] 69.90 74.40 74.50 76.00 66.60 69.80 Segment Anyword [153] [ICML’25] 67.73 66.08 73.57 74.63 67.56 70.90

RAS [382] [arXiv’25] 70.48 74.64 76.99 77.45 67.90 69.42

###### A.4 3D-RES Performance Benchmarking

Evaluation Metrics. mIoU and Acc@X (i.e., Pr@X) are adopted for evaluating 3D-RES performance.

Benchmark Results. As shown in TABLE 7, we present the performance comparison of recent 3D-RES methods on ScanRefer [45]. The evaluation is conducted on both unique and multiple reference scenarios, with unique references constituting approximately 19% of the dataset and multiple references making up the remaining 81%. IPDN [336] achieves the best overall performance with 60.60% Acc@0.25 and 54.90% Acc@0.5, demonstrating significant improvements over earlier methods. RG-SAN [335], despite being weakly-supervised, shows competitive results with 61.70% Acc@0.25, though its Acc@0.5 performance (44.90%) is lower than fully-supervised methods like IPDN [336] and MDIN [11]. Most methods perform substantially better on unique references than multiple references, highlighting the challenge of disambiguating between similar objects in 3D scenes.

###### A.5 GREx Performance Benchmarking

Evaluation Metrics. In addition to cIoU, the evaluation metrics for GRES and GREC also include:

• Generalized IoU (gIoU): The widely-used cIoU tends to favor larger objects. Since multi-target samples have larger foreground areas in GRES, this bias can significantly impact the evaluation results. Similar to mean IoU, gIoU calculates the mean value of per-image IoU over all samples. For no-target samples, the IoU

TABLE 9: Performance Comparison on GREC Task. The results are evaluated on gRefCOCO dataset [3] with Pr@(F1=1, IoU≥0.5) and N-acc. metrics.

Val testA testB Pr N-acc. Pr N-acc. Pr N-acc. MCN [68] [CVPR’20] 28.0 30.6 32.3 32.0 26.8 30.3

Method Venue

MDETR [187] [ICCV’21] 42.7 36.3 50.0 34.5 36.5 31.0 VLT [4] [TPAMI’23] 36.6 35.2 40.2 34.1 30.2 32.5 UNINEXT [194] [CVPR’23] 58.2 50.6 46.4 49.3 42.9 48.2 Ferret [383] [ICLR’24] 54.8 48.9 49.5 45.2 43.5 43.8

SimVG [384] [NeurIPS’24] 62.1 54.7 64.6 57.2 54.8 57.2 Grounding Dino [173] [ECCV’24] - - 45.7 79.0 44.8 76.7

NGDINO [385] [arXiv’25] - - 46.1 83.2 45.6 78.1 HieA2G [33] [AAAI’25] 67.8 60.3 66.0 60.1 56.5 56.0

values of true positive no-target samples are regarded as 1, while IoU values of false negative samples are treated as 0.

- • N-acc. and T-acc.: These two metrics to assess model performance on no-target identification. N-acc. (No-target accuracy) evaluates how well a model identifies samples without targets by computing the ratio of true positives, i.e., predictions with no foreground pixels for no-target samples, to total no-target samples: Nacc. = TP/(TP + FN). Meanwhile, T-acc. (Target accuracy) measures the model’s ability to avoid misclassifying target-containing samples as no-target samples: T-acc. = TN/(TN + FP).
- • Precision@(F1=1, IoU>0.5): This metric computes the percentage of samples that achieve an F1 score of 1 with the IoU threshold set to 0.5. Given a sample and its predicted/groundtruth bounding boxes, a predicted bounding box is regarded as a TP if it has a matched (IoU>0.5) ground-truth bounding box. When multiple predicted bounding boxes match one ground-truth bounding box, only the one with the highest IoU is considered TP while others are FP. Ground-truth bounding boxes having no matched predicted bounding box are FN while predicted bounding boxes having no matched ground-truth are FP. The F1 score

is calculated by F1 = 2TP+2FNTP+FP . A sample is considered successfully predicted if the F1 score equals 1. For no-target samples, the F1 score is regarded as 1 if there is no predicted bounding box, otherwise 0.

GRES Benchmark Results. As shown in TABLE 8, we present the performance comparison of recent GRES methods on the gRefCOCO [3] dataset using cIoU and mIoU metrics. The field has seen significant progress since the GRES [3] task was proposed, with recent methods like InstAlign [353] achieving the best performance with 68.94% cIoU and 74.34% mIoU on the validation set. GSVA [174] also demonstrates strong performance, reaching 66.38% cIoU and 70.04% mIoU on the validation set. On the testA and testB sets, InstAlign continues to lead with 73.22% cIoU, 74.51% mIoU and 63.88% cIoU, 65.74% mIoU respectively. These improvements highlight the rapid advancement in generalized referring expression segmentation capabilities and challenges, significantly outperforming earlier methods like MattNet [60] and VLT [4] designed for only single-object tasks.

GREC Benchmark Results. As shown in TABLE 9, we present the performance comparison of recent GREC methods on the gRefCOCO [3] dataset using Pr@(F1=1, IoU≥0.5) and N-acc. metrics. The field has witnessed significant advancement since the introduction of GREC task, with recent methods like HieA2G [33] achieving the best overall performance with 67.8% Pr and 60.3% N-acc. on the validation set. SimVG [384] also demonstrates competitive results, reaching 62.1% Pr and 54.7% N-acc. on validation. On the testA set, SimVG leads with 64.6%

TABLE 10: Performance Comparison on ReasonSeg Benchmarks. The results are evaluated on the ReasonSeg [21] dataset with the mIoU and cIoU metrics. ‘ft’ denotes fine-tuning on the ReasonSeg dataset. For MLLMs training methods, SFT denotes Supervised Fine-Tuning, RL denotes Reinforcement Learning.

Val testA

Method Venue Training

Method mIoU cIoU mIoU cIoU w/o Multimodel Large Language Models (MLLMs).

ReLA [3] [CVPR’23] - 22.4 19.9 21.3 22.0 X-Decoder [189] [CVPR’23] - 22.6 17.9 21.7 16.3 OVSeg [386] [CVPR’23] - 28.5 18.6 26.1 20.8

SEEM [190] [NeurIPS’24] - 25.5 21.2 24.3 18.7 Grounded-SAM [171] [arXiv’24] - 26.0 14.5 21.3 16.4

###### w. Multimodel Large Language Models (MLLMs).

LISA-13B (ft) [21] [CVPR’24] SFT 65.0 72.9 61.3 62.2

LISA++-7B (ft) [387] [arXiv’23] SFT 64.2 68.1 57.0 59.5 GSVA-7B (ft) [174] [CVPR’24] SFT 50.5 56.4 - -

LLM-Seg-7B [234] [CVPR’24] SFT 52.3 47.5 - SAM4MLLM-8B (ft) [175] [ECCV’24] SFT 58.4 60.4 - CoReS-13B (ft) [231] [ECCV’24] SFT 68.1 - 65.5 -

SegLLM-7B [67] [ICLR’25] SFT 57.2 54.3 52.4 48.4 READ-7B (ft) [233] [CVPR’25] SFT 59.8 67.6 56.8 59.0

POPEN-7B [240] [CVPR’25] SFT+RL 60.2 64.5 - Seg-Zero-7B [239] [arXiv’25] RL 62.6 62.0 57.5 52.0

PixelThink-7B [241] [arXiv’25] RL 63.8 62.7 60.2 55.8

SAM-R1-7B [242] [arXiv’25] RL 64.0 55.8 60.2 54.3 VisionReasoner-7B [388] [arXiv’25] RL 66.3 - 63.6 -

Pr, while NGDINO [385] achieves the highest N-acc. at 83.2%. For testB evaluation, HieA2G maintains strong performance with 56.5% Pr and 56.0% N-acc.

A.6 ReasonSeg Performance Benchmarking Evaluation Metrics. mIoU and cIoU are adopted for evaluating Reasoning Segmentation performance.

ReasonSeg Benchmark Results. As shown in TABLE 10, we present the performance comparison of recent Reasoning Segmentation (ReasonSeg) methods on the ReasonSeg [21] dataset using gIoU and cIoU metrics. The results demonstrate a clear performance gap between methods with and without Multimodal Large Language Models (MLLMs). Traditional methods without MLLMs, such as ReLA [3], X-Decoder [189], and OVSeg [386], achieve modest performance with gIoU scores ranging from 22.4% to 28.5% on the validation set. In contrast, MLLM-based approaches show substantial improvements, with CoReS [231] achieving the best performance at 68.1% gIoU on the validation set, followed by VisionReasoner [388] at 66.3%. On the test set, CoReS leads with 65.5% gIoU, followed by VisionReasoner at 63.6%. When comparing 7B parameter MLLM models for fair comparison, VisionReasoner achieves the best performance with 66.3% gIoU on validation and 63.6% on test, followed by SAMR1 [242] and PixelThink [241] (both 60.2% gIoU on test). These results highlight the significant advantage of leveraging MLLMs for complex reasoning tasks in segmentation, with fine-tuned models consistently outperforming their non-MLLM counterparts by large margins across both validation and test sets.

#### REFERENCES

- [1] H. Ding, C. Liu, S. He, X. Jiang, and C. C. Loy, “MeViS: A large-scale benchmark for video segmentation with motion expressions,” in ICCV,

2023. 1, 2, 4, 6, 10, 11, 16, 17

- [2] H. Ding, C. Liu, S. He, K. Ying, X. Jiang, C. C. Loy, and Y.-G. Jiang, “MeViS: A multi-modal dataset for referring motion expression video segmentation,” IEEE TPAMI, 2025. 1, 2, 4
- [3] C. Liu, H. Ding, and X. Jiang, “GRES: generalized referring expression segmentation,” in CVPR, 2023. 1, 3, 4, 5, 6, 9, 12, 13, 15, 17, 18

- [4] H. Ding, C. Liu, S. Wang, and X. Jiang, “VLT: Vision-language transformer and query generation for referring segmentation,” IEEE TPAMI, 2023. 1, 2, 5, 6, 7, 8, 10, 15, 17, 18
- [5] S. He, G. Jie, C. Wang, Y. Zhou, S. Hu, G. Li, and H. Ding, “ReferSplat: Referring segmentation in 3d gaussian splatting,” in ICML, 2025. 1, 2, 4, 5, 13
- [6] R. Hu, M. Rohrbach, and T. Darrell, “Segmentation from natural language expressions,” in ECCV, 2016. 1, 5, 6, 7, 14, 15
- [7] H. Ding, C. Liu, S. Wang, and X. Jiang, “Vision-language transformer and query generation for referring segmentation,” in ICCV, 2021. 1, 5, 6, 8, 15
- [8] S. Kazemzadeh, V. Ordonez, M. Matten, and T. L. Berg, “Referitgame: Referring to objects in photographs of natural scenes,” in EMNLP, 2014. 1, 4, 14, 15
- [9] S. He, H. Ding, X. Jiang, and B. Wen, “SegPoint: Segment any point cloud via large language model,” in ECCV, 2024. 1, 2, 4, 5, 12, 17
- [10] K. Ying, H. Ding, G. Jie, and Y.-G. Jiang, “Towards omnimodal expressions and reasoning in referring audio-visual segmentation,” in ICCV, 2025. 1, 2, 4, 5, 6, 12, 17
- [11] C. Wu, Y. Liu, J. Ji, Y. Ma, H. Wang, G. Luo, H. Ding, X. Sun, and R. Ji, “3D-GRES: Generalized 3d referring expression segmentation,” ACM MM, 2024. 1, 4, 5, 12, 13, 17
- [12] C. Liu, X. Li, and H. Ding, “Referring image editing: Object-level image editing via referring expressions,” in CVPR, 2024. 1, 14
- [13] X. Shuai, H. Ding, X. Ma, R. Tu, Y.-G. Jiang, and D. Tao, “A survey of multimodal-guided image editing with text-to-image diffusion models,” arXiv, 2024. 1, 14
- [14] H.-S. Fang, C. Wang, M. Gou, and C. Lu, “Graspnet-1billion: A largescale benchmark for general object grasping,” in CVPR, 2020. 1
- [15] J. Lin, J. Chen, K. Peng, X. He, Z. Li, R. Stiefelhagen, and K. Yang, “Echotrack: Auditory referring multi-object tracking for autonomous driving,” IEEE TITS, 2024. 1
- [16] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR, 2015. 1, 6
- [17] K. He, G. Gkioxari, P. Doll´ar, and R. Girshick, “Mask r-cnn,” in ICCV,

2017. 1, 5, 7

- [18] H. Ding, X. Jiang, B. Shuai, A. Q. Liu, and G. Wang, “Context contrasted feature and gated multi-scale aggregation for scene segmentation,” in CVPR, 2018. 1
- [19] X. Li, H. Ding, H. Yuan, W. Zhang, J. Pang, G. Cheng, K. Chen, Z. Liu, and C. C. Loy, “Transformer-based visual segmentation: A survey,” IEEE TPAMI, 2024. 1
- [20] J. Wu, X. Li, S. Xu, H. Yuan, H. Ding, Y. Yang, X. Li, J. Zhang, Y. Tong, X. Jiang et al., “Towards open vocabulary learning: A survey,” IEEE TPAMI, 2024. 1
- [21] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, “Lisa: Reasoning segmentation via large language model,” in CVPR, 2024. 2, 4, 8, 9, 12, 13, 15, 17, 18
- [22] S. He and H. Ding, “Decoupling static and hierarchical motion perception for referring video segmentation,” in CVPR, 2024. 2, 7, 10, 16
- [23] Y. Wang, P. Sun, D. Zhou, G. Li, H. Zhang, and D. Hu, “Ref-avs: Refer and segment objects in audio-visual scenes,” ECCV, 2024. 2, 4, 5
- [24] A. Sokolov, S. Bhosale, and X. Zhu, “3d audio-visual segmentation,” arXiv, 2024. 2
- [25] Y. Wang, P. Sun, Y. Li, H. Zhang, and D. Hu, “Can textual semantics mitigate sounding object segmentation preference?” in ECCV, 2024. 2, 11, 12, 16
- [26] Y. Qiao, C. Deng, and Q. Wu, “Referring expression comprehension: A survey of methods and datasets,” IEEE TMM, 2020. 1
- [27] D. Liu, Y. Liu, W. Huang, and W. Hu, “A survey on text-guided 3d visual grounding: Elements, recent advances, and future directions,” arXiv, 2024. 1
- [28] Y. Shen, C. Li, F. Xiong, J.-O. Jeong, T. Wang, M. Latman, and M. Unberath, “Reasoning segmentation for images and videos: A survey,” arXiv, 2025. 1, 9
- [29] Y. Wei, D. Hu, Y. Tian, and X. Li, “Learning in audio-visual context: A review, analysis, and new perspective,” arXiv, 2022. 1
- [30] L. Xiao, X. Yang, X. Lan, Y. Wang, and C. Xu, “Towards visual grounding: A survey,” arXiv, 2024. 1
- [31] L. Ji, Y. Du, Y. Dang, W. Gao, and H. Zhang, “A survey of methods for addressing the challenges of referring image segmentation,” Neurocomputing, 2024. 1
- [32] S. He, H. Ding, C. Liu, and X. Jiang, “GREC: Generalized referring expression comprehension,” arXiv, 2023. 3, 4, 5, 7, 13
- [33] Y. Wang, H. Ding, S. He, X. Jiang, B. Wei, and J. Liu, “Hierarchical alignment-enhanced adaptive grounding network for generalized referring expression comprehension,” in AAAI, 2025. 3, 13, 17, 18

- [34] L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg, “Modeling context in referring expressions,” in ECCV, 2016. 4, 7, 13
- [35] C. Wu, Z. Lin, S. Cohen, T. Bui, and S. Maji, “Phrasecut: Languagebased image segmentation in the wild,” in CVPR, 2020. 4
- [36] A. Khoreva, A. Rohrbach, and B. Schiele, “Video object segmentation with language referring expressions,” in ACCV, 2018. 4, 6, 10, 16
- [37] K. Gavrilyuk, A. Ghodrati, Z. Li, and C. G. M. Snoek, “Actor and action video segmentation from a sentence,” in CVPR, 2018. 4, 5, 6
- [38] C. Xu, S.-H. Hsieh, C. Xiong, and J. J. Corso, “Can humans fly? action understanding with multiple classes of actors,” in CVPR, 2015. 4
- [39] H. Jhuang, J. Gall, S. Zuffi, C. Schmid, and M. J. Black, “Towards understanding action recognition,” in ICCV, 2013. 4
- [40] S. Seo, J.-Y. Lee, and B. Han, “Urvos: Unified referring video object segmentation network with a large-scale benchmark,” in ECCV, 2020. 4, 5, 6, 10, 16
- [41] C. Yan, H. Wang, S. Yan, X. Jiang, Y. Hu, G. Kang, W. Xie, and E. Gavves, “Visa: Reasoning video object segmentation via large language models,” in ECCV, 2024. 4, 11, 16
- [42] Z. Bai, T. He, H. Mei, P. Wang, Z. Gao, J. Chen, L. Liu, Z. Zhang, and M. Z. Shou, “One token to seg them all: Language instructed reasoning segmentation in videos,” in NeurIPS, 2024. 4, 11, 16
- [43] J. Zhou, J. Wang, J. Zhang, W. Sun, J. Zhang, S. Birchfield, D. Guo, L. Kong, M. Wang, and Y. Zhong, “Audio–visual segmentation,” in ECCV, 2022. 4, 6, 16, 17
- [44] J. Zhou, X. Shen, J. Wang, J. Zhang, W. Sun, J. Zhang, S. Birchfield, D. Guo, L. Kong, M. Wang et al., “Audio-visual segmentation with semantics,” IJCV, 2024. 4, 16
- [45] D. Z. Chen, A. X. Chang, and M. Nießner, “Scanrefer: 3d object localization in rgb-d scans using natural language,” in ECCV, 2020. 4, 5, 13, 17
- [46] P. Achlioptas, A. Abdelreheem, F. Xia, M. Elhoseiny, and L. Guibas, “Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes,” in ECCV, 2020. 4, 13
- [47] Y. Hu, Q. Wang, W. Shao, E. Xie, Z. Li, J. Han, and P. Luo, “Beyond one-to-one: Rethinking the referring image segmentation,” in ICCV,

2023. 4, 13

- [48] Y. Zhang, Z. Gong, and A. X. Chang, “Multi3drefer: Grounding text description to multiple 3d objects,” in ICCV, 2023. 4, 5, 13
- [49] R. Liu, C. Liu, Y. Bai, and A. L. Yuille, “Clevr-ref+: Diagnosing visual reasoning with referring expressions,” in CVPR, 2019. 4
- [50] W. Wang, T. Yue, Y. Zhang, L. Guo, X. He, X. Wang, and J. Liu, “Unveiling parts beyond objects: Towards finer-granularity referring expression segmentation,” in CVPR, 2024. 4, 15
- [51] R. Li, “Aeroreformer: Aerial referring transformer for uav-based referring image segmentation,” arXiv, 2025. 4
- [52] J. Liu, W. Wang, Y. Zhang, Y. Tang, X. He, L. Guo, T. Yue, and X. Wang, “Towards unified referring expression segmentation across omni-level visual target granularities,” arXiv, 2025. 4, 17
- [53] R. Hu, L. Zhu, Y. Zhang, T. Cheng, L. Liu, H. Liu, L. Ran, X. Chen, W. Liu, and X. Wang, “Groundingsuite: Measuring complex multigranular pixel grounding,” arXiv, 2025. 4
- [54] Y. Shen, C. Li, C. Fan, and M. Unberath, “Rvtbench: A benchmark for visual reasoning tasks,” arXiv, 2025. 4
- [55] D.-H. Kim, H. Song, and D. Kim, “Synres: Towards referring expression segmentation in the wild via synthetic data,” arXiv, 2025. 4
- [56] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung, “A benchmark dataset and evaluation methodology for video object segmentation,” in CVPR, 2016. 4
- [57] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in CVPR, 2017. 5
- [58] C. Yeshwanth, Y.-C. Liu, M. Nießner, and A. Dai, “Scannet++: A highfidelity dataset of 3d indoor scenes,” in ICCV, 2023. 5
- [59] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik, “Lerf: Language embedded radiance fields,” in ICCV, 2023. 5
- [60] L. Yu, Z. Lin, X. Shen, J. Yang, X. Lu, M. Bansal, and T. L. Berg, “Mattnet: Modular attention network for referring expression comprehension,” in CVPR, 2018. 5, 7, 10, 13, 15, 17, 18
- [61] P.-H. Huang, H.-H. Lee, H.-T. Chen, and T.-L. Liu, “Text-guided graph neural networks for referring 3d instance segmentation,” in AAAI, 2021. 5, 12, 17
- [62] R. Hu, M. Rohrbach, J. Andreas, T. Darrell, and K. Saenko, “Modeling relationships in referential expressions with compositional modular networks,” in CVPR, 2017. 5, 7
- [63] X. Liu, Z. Wang, J. Shao, X. Wang, and H. Li, “Improving referring expression grounding with cross-modal attention-guided erasing,” in CVPR, 2019. 5

- [64] C. Liang, Y. Wu, T. Zhou, W. Wang, Z. Yang, Y. Wei, and Y. Yang, “Rethinking cross-modal interaction from a top-down perspective for referring video object segmentation,” arXiv, 2021. 5, 10
- [65] C. Liu, X. Jiang, and H. Ding, “Primitivenet: decomposing the global constraints for referring segmentation,” Visual Intelligence, 2024. 5
- [66] L. Xiao, X. Yang, F. Peng, Y. Wang, and C. Xu, “Oneref: Unified one-tower expression grounding and segmentation with mask referring modeling,” in NeurIPS, 2024. 5, 6, 8, 14, 15
- [67] X. Wang, S. Zhang, S. Li, K. Kallidromitis, K. Li, Y. Kato, K. Kozuka, and T. Darrell, “Segllm: Multi-round reasoning segmentation,” in ICLR,

2025. 5, 9, 15, 18

- [68] G. Luo, Y. Zhou, X. Sun, L. Cao, C. Wu, C. Deng, and R. Ji, “Multi-task collaborative network for joint referring expression comprehension and segmentation,” in CVPR, 2020. 5, 7, 8, 12, 13, 15, 18
- [69] D.-J. Chen, S. Jia, Y.-C. Lo, H.-T. Chen, and T.-L. Liu, “See-throughtext grouping for referring image segmentation,” in ICCV, 2019. 5, 6, 8, 15
- [70] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in ECCV, 2020. 5, 6, 7
- [71] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016. 5
- [72] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2020. 5
- [73] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in ICCV, 2021. 5
- [74] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, “Learning spatiotemporal features with 3d convolutional networks,” in ICCV,

2015. 5, 6

- [75] J. Carreira and A. Zisserman, “Quo vadis, action recognition? a new model and the kinetics dataset,” in CVPR, 2017. 5, 6
- [76] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, “Video swin transformer,” in CVPR, 2022. 5
- [77] S. He and H. Ding, “RefMask3D: Language-guided transformer for 3d referring segmentation,” ACM MM, 2024. 5, 6, 7, 12, 17
- [78] B. Graham, M. Engelcke, and L. Van Der Maaten, “3d semantic segmentation with submanifold sparse convolutional networks,” in CVPR,

2018. 5

- [79] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in CVPR, 2017. 5
- [80] E. Margffoy-Tuay, J. C. P´erez, E. Botero, and P. Arbel´aez, “Dynamic multimodal instance segmentation guided by natural language queries,” in ECCV, 2018. 6, 7, 15
- [81] C. Liu, Z. Lin, X. Shen, J. Yang, X. Lu, and A. Yuille, “Recurrent multimodal interaction for referring image segmentation,” in ICCV,

2017. 6, 7, 15

- [82] H. Shi, H. Li, F. Meng, and Q. Wu, “Key-word-aware network for referring expression image segmentation,” in ECCV, 2018. 6, 7, 15
- [83] R. Li, K. Li, Y.-C. Kuo, M. Shu, X. Qi, X. Shen, and J. Jia, “Referring image segmentation via recurrent refinement networks,” in CVPR, 2018. 6, 7, 8, 15
- [84] Z. Yang, J. Wang, Y. Tang, K. Chen, H. Zhao, and P. H. Torr, “Lavt: Language-aware vision transformer for referring image segmentation,” in CVPR, 2022. 6, 8, 9, 15, 17
- [85] J. Liu, H. Ding, Z. Cai, Y. Zhang, R. K. Satzoda, V. Mahadevan, and R. Manmatha, “Polyformer: Referring image segmentation as sequential polygon generation,” in CVPR, 2023. 6, 7, 8, 15
- [86] J. Wu, Y. Jiang, B. Yan, H. Lu, Z. Yuan, and P. Luo, “Segment every reference object in spatial and temporal spaces,” in ICCV, 2023. 6, 10, 16
- [87] N. Kim, D. Kim, C. Lan, W. Zeng, and S. Kwak, “Restr: Convolutionfree referring image segmentation using transformers,” in CVPR, 2022. 6, 8, 15
- [88] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: pretraining of deep bidirectional transformers for language understanding,” in NAACL-HLT, 2019. 6
- [89] Y. Liu, “Roberta: A robustly optimized bert pretraining approach,” arXiv, 2019. 6
- [90] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021. 6, 9
- [91] Z. Wang, Y. Lu, Q. Li, X. Tao, Y. Guo, M. Gong, and T. Liu, “Cris: Clip-driven referring image segmentation,” in CVPR, 2022. 6, 7, 8, 15, 17

- [92] Z. Xu, Z. Chen, Y. Zhang, Y. Song, X. Wan, and G. Li, “Bridging vision and language encoders: Parameter-efficient tuning for referring image segmentation,” in ICCV, 2023. 6, 8, 15
- [93] S. Hershey, S. Chaudhuri, D. P. Ellis, J. F. Gemmeke, A. Jansen, R. C. Moore, M. Plakal, D. Platt, R. A. Saurous, B. Seybold et al., “Cnn architectures for large-scale audio classification,” in ICASSP, 2017. 6
- [94] S. Schneider, A. Baevski, R. Collobert, and M. Auli, “wav2vec: Unsupervised pre-training for speech recognition,” in INTERSPEECH, 2019. 6
- [95] Y. Gong, C.-I. Lai, Y.-A. Chung, and J. Glass, “Ssast: Self-supervised audio spectrogram transformer,” in AAAI, 2022. 6
- [96] Y. Wang, P. Sun, D. Zhou, G. Li, H. Zhang, and D. Hu, “Ref-avs: Refer and segment objects in audio-visual scenes,” in ECCV, 2024. 6, 11, 17
- [97] Z. Hu, G. Feng, J. Sun, L. Zhang, and H. Lu, “Bi-directional relationship inferring network for referring image segmentation,” in CVPR, 2020. 6, 7, 15
- [98] G. Feng, Z. Hu, L. Zhang, and H. Lu, “Encoder fusion network with co-attention embedding for referring image segmentation,” in CVPR,

2021. 6, 7, 15

- [99] C. Liu, H. Ding, Y. Zhang, and X. Jiang, “Multi-modal mutual attention and iterative interaction for referring image segmentation,” IEEE TIP,

- 2023. 6, 7, 13, 15

[100] S. Kim, M. Kang, D. Kim, J. Park, and S. Kwak, “Extending clip’s image-text alignment to referring image segmentation,” in NAACL-HLT,

- 2024. 6, 8

- [101] S. Yu, I. Jung, B. Han, T. Kim, Y. Kim, D. Wee, and J. Son, “A simple baseline with single-encoder for referring image segmentation,” arXiv,

2024. 6, 8

- [102] W. Wang, X. He, Y. Zhang, L. Guo, J. Shen, J. Li, and J. Liu, “Cmmasksd: Cross-modality masked self-distillation for referring image segmentation,” IEEE TMM, 2024. 6, 8, 15
- [103] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in ICML, 2021. 6
- [104] Y. Ling, Y. Li, Z. Gan, J. Zhang, M. Chi, and Y. Wang, “Transavs: Endto-end audio-visual segmentation with transformer,” in ICASSP, 2024. 6, 11
- [105] J. Mei, A. Piergiovanni, J.-N. Hwang, and W. Li, “Slvp: Self-supervised language-video pre-training for referring video object segmentation,” in WACV, 2024. 6, 10
- [106] H. Ding, C. Liu, S. He, X. Jiang, P. H. Torr, and S. Bai, “MOSE: A new dataset for video object segmentation in complex scenes,” in ICCV,

2023. 6

- [107] H. Ding, K. Ying, C. Liu, S. He, X. Jiang, Y.-G. Jiang, P. H. Torr, and S. Bai, “MOSEv2: A more challenging dataset for video object segmentation in complex scenes,” arXiv, 2025. 6
- [108] S. Liu, T. Hui, S. Huang, Y. Wei, B. Li, and G. Li, “Cross-modal progressive comprehension for referring segmentation,” IEEE TPAMI,

2021. 6, 8, 10, 16

- [109] S. Huang, T. Hui, S. Liu, G. Li, Y. Wei, J. Han, L. Liu, and B. Li, “Referring image segmentation via cross-modal progressive comprehension,” in CVPR, 2020. 6, 8
- [110] L. Ye, M. Rochan, Z. Liu, and Y. Wang, “Cross-modal self-attention network for referring image segmentation,” in CVPR, 2019. 6, 7, 10, 15
- [111] G. Feng, L. Zhang, J. Sun, Z. Hu, and H. Lu, “Referring segmentation via encoder-fused cross-modal attention network,” IEEE TPAMI, 2022. 6, 10, 16
- [112] C. Liang, W. Wang, T. Zhou, J. Miao, Y. Luo, and Y. Yang, “Localglobal context aware transformer for language-guided video segmentation,” IEEE TPAMI, 2023. 6, 10, 16
- [113] Z. Ding, T. Hui, J. Huang, X. Wei, J. Han, and S. Liu, “Languagebridged spatial-temporal interaction for referring video object segmentation,” in CVPR, 2022. 6, 10, 16
- [114] S. Cho, S. Lee, M. Lee, J. Lee, and S. Lee, “Find first, track next: Decoupling identification and propagation in referring video object segmentation,” arXiv, 2025. 6, 10
- [115] T. Liang, K.-Y. Lin, C. Tan, J. Zhang, W.-S. Zheng, and J.-F. Hu, “Referdino: Referring video object segmentation with visual grounding foundations,” arXiv, 2025. 6, 10
- [116] S. Gong, Y. Zhuge, L. Zhang, Z. Yang, P. Zhang, and H. Lu, “The devil is in temporal token: High quality video reasoning segmentation,” in CVPR, 2025. 6, 11, 16
- [117] W. Zhao, K. Wang, X. Chu, F. Xue, X. Wang, and Y. You, “Modeling motion with multi-modal features for text-based video segmentation,” in CVPR, 2022. 6, 10

- [118] L. Yuan, M. Shi, Z. Yue, and Q. Chen, “Losh: Long-short text joint prediction network for referring video object segmentation,” in CVPR,

2024. 6, 10, 16

- [119] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, “Masked-attention mask transformer for universal image segmentation,” in CVPR, 2022. 6
- [120] M. Li and L. Sigal, “Referring transformer: A one-step approach to multi-task visual grounding,” NeurIPS, 2021. 6, 8, 15
- [121] D. Wu, W. Han, T. Wang, X. Dong, X. Zhang, and J. Shen, “Referring multi-object tracking,” in CVPR, 2023. 6, 13
- [122] H. Yuan, X. Li, T. Zhang, Z. Huang, S. Xu, S. Ji, Y. Tong, L. Qi, J. Feng, and M.-H. Yang, “Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos,” arXiv, 2025. 6, 10, 16
- [123] C. Cuttano, G. Trivigno, G. Rosi, C. Masone, and G. Averta, “Samwise: Infusing wisdom in sam2 for text-driven video segmentation,” in CVPR,

2025. 6, 10, 16

- [124] X. Huang, G. Luo, C. Zhu, B. Tong, Y. Zhou, X. Sun, and R. Ji, “Deep instruction tuning for segment anything model,” in ACM MM, 2024. 6, 8
- [125] Y. Zhang, T. Cheng, R. Hu, L. Liu, H. Liu, L. Ran, X. Chen, W. Liu, and X. Wang, “Evf-sam: Early vision-language fusion for text-prompted segment anything model,” arXiv, 2024. 6
- [126] Y. Sun, H. Zhang, H. Ding, T. Zhang, X. Ma, and Y.-G. Jiang, “Sama: Towards multi-turn referential grounded video chat with large language models,” arXiv, 2025. 6, 11
- [127] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in ICCV, 2023. 6, 8, 9, 10, 11
- [128] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. R¨adle, C. Rolland, L. Gustafson et al., “Sam 2: Segment anything in images and videos,” in ICLR, 2025. 6, 10, 11, 12
- [129] Z. Luo, Y. Xiao, Y. Liu, S. Li, Y. Wang, Y. Tang, X. Li, and Y. Yang, “Soc: Semantic-assisted object cluster for referring video object segmentation,” in NeurIPS, 2024. 7, 10, 16
- [130] P. Sun, H. Zhang, and D. Hu, “Unveiling and mitigating bias in audio visual segmentation,” ACM MM, 2024. 7, 11
- [131] Y.-W. Chen, Y.-H. Tsai, T. Wang, Y.-Y. Lin, and M.-H. Yang, “Referring expression object segmentation with caption-aware consistency,” BMVC, 2019. 7, 8, 13
- [132] D. Liu, H. Zhang, F. Wu, and Z.-J. Zha, “Learning to assemble neural module tree networks for visual grounding,” in ICCV, 2019. 7
- [133] C. Liu, X. Jiang, and H. Ding, “Instance-specific feature propagation for referring segmentation,” IEEE TMM, 2023. 7
- [134] Z. Cheng, P. Jin, H. Li, K. Li, S. Li, X. Ji, C. Liu, and J. Chen, “Wico: Win-win cooperation of bottom-up and top-down referring image segmentation,” IJCAI, 2023. 7
- [135] Y. Jiao, Z. Jie, W. Luo, J. Chen, Y.-G. Jiang, X. Wei, and L. Ma, “Twostage visual cues enhancement network for referring image segmentation,” in ACM MM, 2021. 7
- [136] L. Xu, M. H. Huang, X. Shang, Z. Yuan, Y. Sun, and J. Liu, “Meta compositional referring expression segmentation,” in CVPR, 2023. 7
- [137] P. Yue, J. Lin, S. Zhang, J. Hu, Y. Lu, H. Niu, H. Ding, Y. Zhang, G. JIANG, L. Cao et al., “Adaptive selection based referring image segmentation,” in ACM MM, 2024. 7
- [138] W. Su, P. Miao, H. Dou, G. Wang, L. Qiao, Z. Li, and X. Li, “Language adaptive weight generation for multi-task visual grounding,” in CVPR,

2023. 7, 15

- [139] W. Zhao, Y. Rao, Z. Liu, B. Liu, J. Zhou, and J. Lu, “Unleashing textto-image diffusion models for visual perception,” in ICCV, 2023. 7, 15
- [140] S. Ha, C. Kim, D. Kim, J. Lee, S. Lee, and J. Lee, “Finding nemo: Negative-mined mosaic augmentation for referring image segmentation,” in ECCV, 2025. 7
- [141] M. Lee, S. Lee, S. Park, D. Han, B. Heo, and H. Shim, “Maskris: Semantic distortion-aware data augmentation for referring image segmentation,” arXiv, 2024. 7
- [142] T. L¨uddecke and A. Ecker, “Image segmentation using text and image prompts,” in CVPR, 2022. 7
- [143] J. Park, J. Lee, J. Song, S. Yu, D. Jung, and S. Yoon, “Know “no” better: A data-driven approach for enhancing negation awareness in clip,” arXiv, 2025. 7
- [144] J. Yang, L. Zhang, J. Sun, and H. Lu, “Spatial semantic recurrent mining for referring image segmentation,” arXiv, 2024. 7
- [145] I. Kesen, O. A. Can, E. Erdem, A. Erdem, and D. Y¨uret, “Modulating bottom-up and top-down visual processing via language-conditional filters,” in CVPR Workshop, 2022. 7, 15

- [146] H. Ding, S. Zhang, Q. Wu, S. Yu, J. Hu, L. Cao, and R. Ji, “Bilateral knowledge interaction network for referring image segmentation,” IEEE TMM, 2024. 7, 15
- [147] S. Ouyang, H. Wang, S. Xie, Z. Niu, R. Tong, Y.-W. Chen, and L. Lin, “Slvit: Scale-wise language-guided vision transformer for referring image segmentation.” in IJCAI, 2023. 8
- [148] Z. Wei, X. Chen, M. Chen, and S. Zhu, “Linguistic query-guided mask generation for referring image segmentation,” arXiv, 2023. 8
- [149] J. Tang, G. Zheng, C. Shi, and S. Yang, “Contrastive grouping with transformer for referring image segmentation,” in CVPR, 2023. 8, 15
- [150] N. A. Shah, V. VS, and V. M. Patel, “Lqmformer: Language-aware query mask transformer for referring image segmentation,” in CVPR,

2024. 8, 13, 15, 17

- [151] Y. Yang, C. Ma, J. Yao, Z. Zhong, Y. Zhang, and Y. Wang, “Remamber: Referring image segmentation with mamba twister,” ECCV, 2024. 8, 15
- [152] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in COLM, 2024. 8
- [153] Z. Liu, A. Saseendran, L. Tong, X. He, F. Yousefi, N. Burlutskiy, D. Oglic, T. Diethe, P. Teare, H. Zhou et al., “Segment anyword: Mask prompt inversion for open-set grounded segmentation,” in ICML, 2025. 8, 17
- [154] S.-A. Liu, Y. Zhang, Z. Qiu, H. Xie, Y. Zhang, and T. Yao, “Caris: Context-aware referring image segmentation,” in ACM MM, 2023. 8
- [155] Y. X. Chng, H. Zheng, Y. Han, X. Qiu, and G. Huang, “Mask grounding for referring image segmentation,” in CVPR, 2024. 8, 15
- [156] Z. Zhang, Y. Zhu, J. Liu, X. Liang, and W. Ke, “Coupalign: Coupling word-pixel with sentence-mask alignments for referring image segmentation,” NeurIPS, 2022. 8, 15
- [157] Y. Cho, H. Yu, and S.-J. Kang, “Cross-aware early fusion with stagedivided vision and language transformer encoders for referring image segmentation,” IEEE TMM, 2024. 8
- [158] M. Lee, D. Lee, J. Lee, S. Cho, H. Choi, I.-J. Kim, and S. Lee, “Synchronizing vision and language: Bidirectional token-masking autoencoder for referring image segmentation,” arXiv, 2023. 8
- [159] K. He, X. Chen, S. Xie, Y. Li, P. Doll´ar, and R. Girshick, “Masked autoencoders are scalable vision learners,” in CVPR, 2022. 8
- [160] W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, and F. Wei, “Image as a foreign language: BEIT pretraining for vision and vision-language tasks,” in CVPR, 2023. 8
- [161] Y. Wang, J. Li, X. Zhang, B. Shi, C. Li, W. Dai, H. Xiong, and Q. Tian, “Barleria: An efficient tuning framework for referring image segmentation,” in ICLR, 2024. 8, 15
- [162] J. Huang, Z. Xu, T. Liu, Y. Liu, H. Han, K. Yuan, and X. Li, “Densely connected parameter-efficient tuning for referring image segmentation,” in AAAI, 2025. 8
- [163] H. Yu, M. Li, A. Rezazadeh, Y. Yang, and C. Choi, “A parameterefficient tuning framework for language-guided object grounding and robot grasping,” in ICRA, 2025. 8
- [164] Z. Huang and S. Satoh, “Referring image segmentation via joint mask contextual embedding learning and progressive alignment network,” in EMNLP, 2023. 8
- [165] Y. Iioka, Y. Yoshida, Y. Wada, S. Hatanaka, and K. Sugiura, “Multimodal diffusion segmentation model for object segmentation from manipulation instructions,” in IROS, 2023. 8
- [166] Z. Yang, J. Wang, Y. Tang, K. Chen, H. Zhao, and P. H. Torr, “Semantics-aware dynamic localization and refinement for referring image segmentation,” in AAAI, 2023. 8
- [167] L. Ye, Z. Liu, and Y. Wang, “Dual convolutional lstm network for referring image segmentation,” IEEE TMM, 2020. 8, 15
- [168] S. Yang, M. Xia, G. Li, H.-Y. Zhou, and Y. Yu, “Bottom-up shift and reasoning for referring image segmentation,” in CVPR, 2021. 8, 15
- [169] T. Hui, S. Liu, S. Huang, G. Li, S. Yu, F. Zhang, and J. Han, “Linguistic structure guided context modeling for referring image segmentation,” in ECCV, 2020. 8, 15
- [170] S. Wu, S. Jin, W. Zhang, L. Xu, W. Liu, W. Li, and C. C. Loy, “F-lmm: Grounding frozen large multimodal models,” in CVPR, 2025. 8
- [171] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded sam: Assembling open-world models for diverse visual tasks,” arXiv, 2024. 8, 18
- [172] C. Shang, Z. Song, H. Qiu, L. Wang, F. Meng, and H. Li, “Promptdriven referring image segmentation with instance contrasting,” in CVPR, 2024. 8, 15
- [173] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in ECCV, 2024. 8, 18

- [174] Z. Xia, D. Han, Y. Han, X. Pan, S. Song, and G. Huang, “Gsva: Generalized segmentation via multimodal large language models,” in CVPR, 2024. 8, 9, 13, 15, 17, 18
- [175] Y.-C. Chen, W.-H. Li, C. Sun, Y.-C. F. Wang, and C.-S. Chen, “Sam4mllm: Enhance multi-modal large language model for referring expression segmentation,” in ECCV, 2024. 8, 9, 14, 15, 18
- [176] Y. Jing, T. Kong, W. Wang, L. Wang, L. Li, and T. Tan, “Locate then segment: A strong pipeline for referring image segmentation,” in CVPR,

2021. 8, 15, 17

- [177] Z. Cheng, K. Li, P. Jin, S. Li, X. Ji, L. Yuan, C. Liu, and J. Chen, “Parallel vertex diffusion for unified visual grounding,” in AAAI, 2024. 8
- [178] Z. Pang, X. Xu, and Y.-X. Wang, “Aligning generative denoising with discriminative objectives unleashes diffusion for visual perception,” in ICLR, 2025. 8, 14, 15
- [179] G. Luo, Y. Zhou, R. Ji, X. Sun, J. Su, C.-W. Lin, and Q. Tian, “Cascade grouped attention network for referring expression segmentation,” in ACM MM, 2020. 8
- [180] M. Qu, Y. Wu, Y. Wei, W. Liu, X. Liang, and Y. Zhao, “Learning to segment every referring object point by point,” in CVPR, 2023. 8, 9, 15
- [181] C. Zhu, Y. Zhou, Y. Shen, G. Luo, X. Pan, M. Lin, C. Chen, L. Cao, X. Sun, and R. Ji, “Seqtr: A simple yet universal network for visual grounding,” in ECCV, 2022. 8, 12, 15
- [182] M. Lan, C. Chen, Y. Zhou, J. Xu, Y. Ke, X. Wang, L. Feng, and W. Zhang, “Text4seg: Reimagining image segmentation as text generation,” in ICLR, 2025. 8
- [183] H. Tang, C. Xie, H. Wang, X. Bao, T. Weng, P. Li, Y. Zheng, and L. Wang, “Ufo: A unified approach to fine-grained visual perception via open-ended language interface,” arXiv, 2025. 8
- [184] S. Cheng, Y. Liu, X. He, S. Ourselin, L. Tan, and G. Luo, “Weakmcn: Multi-task collaborative network for weakly supervised referring expression comprehension and segmentation,” in CVPR, 2025. 8, 9
- [185] W. Kang, G. Liu, M. Shah, and Y. Yan, “Segvg: Transferring object bounding box to segmentation for visual grounding,” in ECCV, 2024. 8
- [186] J. Wang, H. Wang, W. Zhang, K. Ji, D. Huang, and Y. Zheng, “Progressive language-guided visual learning for multi-task visual grounding,” arXiv, 2025. 8
- [187] A. Kamath, M. Singh, Y. LeCun, G. Synnaeve, I. Misra, and N. Carion, “Mdetr-modulated detection for end-to-end multi-modal understanding,” in ICCV, 2021. 8, 18
- [188] X. Yang, L. Xu, H. Sun, H. Li, and S. Zhang, “Enhancing visual grounding and generalization: A multi-task cycle training approach for vision-language models,” arXiv, 2023. 8
- [189] X. Zou, Z.-Y. Dou, J. Yang, Z. Gan, L. Li, C. Li, X. Dai, H. Behl, J. Wang, L. Yuan et al., “Generalized decoding for pixel, image, and language,” in CVPR, 2023. 8, 15, 18
- [190] X. Zou, J. Yang, H. Zhang, F. Li, L. Li, J. Wang, L. Wang, J. Gao, and Y. J. Lee, “Segment everything everywhere all at once,” NeurIPS, 2023. 8, 15, 18
- [191] J. Wu, Y. Jiang, Q. Liu, Z. Yuan, X. Bai, and S. Bai, “General object foundation model for images and videos at scale,” in CVPR, 2024. 8
- [192] Z. Zhang, Y. Ma, E. Zhang, and X. Bai, “Psalm: Pixelwise segmentation with large multi-modal model,” in ECCV, 2024. 8
- [193] T.-J. Fu, Y. Qian, C. Chen, W. Hu, Z. Gan, and Y. Yang, “Univg: A generalist diffusion model for unified image generation and editing,” arXiv, 2025. 8
- [194] B. Yan, Y. Jiang, J. Wu, D. Wang, P. Luo, Z. Yuan, and H. Lu, “Universal instance perception as object discovery and retrieval,” in CVPR, 2023. 8, 14, 15, 18
- [195] T. Zhang, X. Li, Z. Huang, Y. Li, W. Lei, X. Deng, S. Chen, S. Ji, and J. Feng, “Pixel-sail: Single transformer for pixel-grounded understanding,” arXiv, 2025. 8
- [196] J. He, Y. Wang, L. Wang, H. Lu, J.-Y. He, J.-P. Lan, B. Luo, and X. Xie, “Multi-modal instruction tuned llms with fine-grained visual perception,” in CVPR, 2024. 8, 15
- [197] R. Strudel, I. Laptev, and C. Schmid, “Weakly-supervised segmentation of referring expressions,” arXiv, 2022. 8, 9
- [198] Q. Dai and S. Yang, “Curriculum point prompting for weaklysupervised referring image segmentation,” in CVPR, 2024. 8, 9, 15
- [199] D. Kim, N. Kim, C. Lan, and S. Kwak, “Shatter and gather: Learning referring image segmentation with text supervision,” in ICCV, 2023. 8, 9, 15
- [200] H. Li, M. Sun, J. Xiao, E. G. Lim, and Y. Zhao, “Fully and weakly supervised referring expression segmentation with end-to-end learning,” IEEE TCSVT, 2023. 8, 9

- [201] J. Lee, S. Lee, J. Nam, S. Yu, J. Do, and T. Taghavi, “Weakly supervised referring image segmentation with intra-chunk and interchunk consistency,” in ICCV, 2023. 9, 15
- [202] J. Xu, S. De Mello, S. Liu, W. Byeon, T. Breuel, J. Kautz, and X. Wang, “Groupvit: Semantic segmentation emerges from text supervision,” in CVPR, 2022. 9
- [203] X. Chen, Y. Luo, G. Luo, J. Ji, H. Ding, and Y. Zhou, “Dvin: Dynamic visual routing network for weakly supervised referring expression comprehension,” in CVPR, 2025. 9
- [204] A. Arbelle, S. Doveh, A. Alfassy, J. Shtok, G. Lev, E. Schwartz, H. Kuehne, H. B. Levi, P. Sattigeri, R. Panda et al., “Detector-free weakly supervised grounding by separation,” in ICCV, 2021. 9
- [205] F. Liu, Y. Liu, Y. Kong, K. Xu, L. Zhang, B. Yin, G. Hancke, and R. Lau, “Referring image segmentation using text supervision,” in ICCV, 2023. 9, 15
- [206] Z. Yang, Y. Liu, J. Lin, G. Hancke, and R. W. Lau, “Boosting weaklysupervised referring image segmentation via progressive comprehension,” NeurIPS, 2024. 9, 15
- [207] G. Feng, L. Zhang, Z. Hu, and H. Lu, “Learning from box annotations for referring image segmentation,” TNNLS, 2024. 9
- [208] H. Lyu, T. Zhong, and S. Zhao, “Gtms: A gradient-driven tree-guided mask-free referring image segmentation method,” in ECCV, 2024. 9, 15
- [209] D. Yang, J. Ji, Y. Ma, T. Guo, H. Wang, X. Sun, and R. Ji, “Sam as the guide: Mastering pseudo-label refinement in semi-supervised referring expression segmentation,” ICML, 2024. 9, 15
- [210] S. Yu, P. H. Seo, and J. Son, “Pseudo-ris: Distinctive pseudo-supervision generation for referring image segmentation,” in ECCV, 2024. 9
- [211] Y. Zang, R. Cao, C. Fu, D. Zhu, M. Zhang, W. Hu, L. Zhu, and T. Chen, “Resmatch: Referring expression segmentation in a semi-supervised manner,” Information Sciences, 2025. 9
- [212] S. Nag, K. Goswami, and S. Karanam, “Safari: Adaptive sequence transformer for weakly supervised referring expression segmentation,” ECCV, 2024. 9, 15
- [213] S. Yu, P. H. Seo, and J. Son, “Zero-shot referring image segmentation with global-local context features,” in CVPR, 2023. 9, 15
- [214] Y. Wang, J. Ni, Y. Liu, C. Yuan, and Y. Tang, “Iterprime: Zero-shot referring image segmentation with iterative grad-cam refinement and primary word emphasis,” AAAI, 2025. 9
- [215] R. Burgert, K. Ranasinghe, X. Li, and M. S. Ryoo, “Peekaboo: Text to image diffusion models are zero-shot segmentors,” in CVPR, 2023. 9
- [216] R. Wang and H. Zhang, “Resanything: Attribute prompting for arbitrary referring segmentation,” arXiv, 2025. 9
- [217] Y. Suo, L. Zhu, and Y. Yang, “Text augmented spatial-aware zero-shot referring image segmentation,” EMNLP, 2023. 9
- [218] S. Sun, R. Li, P. Torr, X. Gu, and S. Li, “Clip as rnn: Segment countless visual concepts without training endeavor,” in CVPR, 2024. 9
- [219] T. Liu and S. Li, “Hybrid global-local representation with augmented spatial guidance for zero-shot referring image segmentation,” in CVPR,

2025. 9, 15

- [220] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv, 2023. 9, 11
- [221] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” NeurIPS, 2024. 9, 11, 12
- [222] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in CVPR, 2024. 9, 11, 12
- [223] Z. Ren, Z. Huang, Y. Wei, Y. Zhao, D. Fu, J. Feng, and X. Jin, “Pixellm: Pixel reasoning with large multimodal model,” in CVPR, 2024. 9, 15
- [224] Y. Lu, J. Cao, Y. Wu, B. Li, L. Tang, Y. Ji, C. Wu, J. Wu, and W. Zhu, “Rsvp: Reasoning segmentation via visual prompting and multi-modal chain-of-thought,” in ACL, 2025. 9
- [225] M. Wahed, K. A. Nguyen, A. S. Juvekar, X. Li, X. Zhou, V. Shah, T. Yu, P. Yanardag, and I. Lourentzou, “Prima: Multi-image vision-language models for reasoning segmentation,” in AAAI, 2025. 9
- [226] M. Siam, “Pixfoundation: Are we heading in the right direction with pixel-level vision foundation models?” arXiv, 2025. 9
- [227] Y. Yuan, W. Li, J. Liu, D. Tang, X. Luo, C. Qin, L. Zhang, and J. Zhu, “Osprey: Pixel understanding with visual instruction tuning,” in CVPR,

2024. 9

- [228] C. Wei, Y. Zhong, H. Tan, Y. Zeng, Y. Liu, Z. Zhao, and Y. Yang, “Instructseg: Unifying instructed visual segmentation with multi-modal large language models,” arXiv, 2024. 9, 11
- [229] A. Zhang, Y. Yao, W. Ji, Z. Liu, and T.-S. Chua, “Next-chat: An lmm for chat, detection and segmentation,” in ICML, 2024. 9

- [230] R. Pi, L. Yao, J. Gao, J. Zhang, and T. Zhang, “Perceptiongpt: Effectively fusing visual perception into llm,” in CVPR, 2024. 9, 15
- [231] X. Bao, S. Sun, S. Ma, K. Zheng, Y. Guo, G. Zhao, Y. Zheng, and X. Wang, “Cores: Orchestrating the dance of reasoning and segmentation,” in ECCV, 2024. 9, 15, 18
- [232] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” in NeurIPS, 2022. 9
- [233] R. Qian, X. Yin, and D. Dou, “Reasoning to attend: Try to understand how <seg> token works,” in CVPR, 2025. 9, 15, 18
- [234] J. Wang and L. Ke, “Llm-seg: Bridging image segmentation and large language model reasoning,” in CVPR Workshop, 2024. 9, 18
- [235] C. Wei, H. Tan, Y. Zhong, Y. Yang, and L. Ma, “Lasagna: Languagebased segmentation assistant for complex queries,” arXiv, 2024. 9, 17
- [236] T.-H. Wu, G. Biamby, D. Chan, L. Dunlap, R. Gupta, X. Wang, J. E. Gonzalez, and T. Darrell, “See say and segment: Teaching lmms to overcome false premises,” in CVPR, 2024. 9, 15
- [237] D. Jang, Y. Cho, S. Lee, T. Kim, and D.-S. Kim, “Mmr: A largescale benchmark dataset for multi-target and multi-granularity reasoning segmentation,” in ICLR, 2025. 9, 15
- [238] D. Cai, X. Yang, Y. Liu, D. Wang, S. Feng, Y. Zhang, and S. Poria, “Pixel-level reasoning segmentation via multi-turn conversations,” arXiv, 2025. 9
- [239] Y. Liu, B. Peng, Z. Zhong, Z. Yue, F. Lu, B. Yu, and J. Jia, “Seg-zero: Reasoning-chain guided segmentation via cognitive reinforcement,” arXiv, 2025. 9, 18
- [240] L. Zhu, T. Chen, Q. Xu, X. Liu, D. Ji, H. Wu, D. W. Soh, and J. Liu, “Popen: Preference-based optimization and ensemble for lvlm-based reasoning segmentation,” in CVPR, 2025. 9, 14, 15, 18
- [241] S. Wang, G. Fang, L. Kong, X. Li, J. Xu, S. Yang, Q. Li, J. Zhu, and X. Wang, “Pixelthink: Towards efficient chain-of-pixel reasoning,” arXiv, 2025. 9, 18
- [242] J. Huang, Z. Xu, J. Zhou, T. Liu, Y. Xiao, M. Ou, B. Ji, X. Li, and K. Yuan, “Sam-r1: Leveraging sam for reward feedback in multimodal segmentation via reinforcement learning,” arXiv, 2025. 9, 18
- [243] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv, 2024. 9
- [244] Z. Yuan, L. Mou, Y. Hua, and X. X. Zhu, “Rrsis: Referring remote sensing image segmentation,” IEEE TGRS, 2024. 9
- [245] Y. Pan, R. Sun, Y. Wang, T. Zhang, and Y. Zhang, “Rethinking the implicit optimization paradigm with dual alignments for referring remote sensing image segmentation,” in ACM MM, 2024. 9
- [246] S. Liu, Y. Ma, X. Zhang, H. Wang, J. Ji, X. Sun, and R. Ji, “Rotated multi-scale interaction network for referring remote sensing image segmentation,” in CVPR, 2024. 9
- [247] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan, “Geochat: Grounded large vision-language model for remote sensing,” in CVPR, 2024. 9
- [248] X. Li, J. Ding, and M. Elhoseiny, “Vrsbench: A versatile visionlanguage benchmark dataset for remote sensing image understanding,” in NeurIPS, 2024. 9
- [249] P. Rewatbowornwong, N. Chatthee, E. Chuangsuwanich, and S. Suwajanakorn, “Zero-guidance segmentation using zero segment labels,” in ICCV, 2023. 9
- [250] H. Rasheed, M. Maaz, S. Shaji, A. Shaker, S. Khan, H. Cholakkal, R. M. Anwer, E. Xing, M.-H. Yang, and F. S. Khan, “Glamm: Pixel grounding large multimodal model,” in CVPR, 2024. 9, 15
- [251] H. Shi, W. Pan, Z. Zhao, M. Zhang, and F. Wu, “Unsupervised domain adaptation for referring semantic segmentation,” in ACM MM, 2023. 9
- [252] T. Zhang, X. Li, H. Fei, H. Yuan, S. Wu, S. Ji, C. L. Chen, and S. Yan, “Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding,” in NeurIPS, 2024. 9
- [253] C. Jiang, A. Luo, and M. Jagersand, “Robot manipulation in salient vision through referring image segmentation and geometric constraints,” in ICRA, 2025. 9
- [254] C. Jiang, Y. Yang, and M. Jagersand, “Clipunetr: Assisting humanrobot interface for uncalibrated visual servoing control with clip-driven referring expression segmentation,” in ICRA, 2024. 9
- [255] X. Jiang, H. Yang, K. Zhu, X. Qiu, S. Zhao, and S. Zhou, “Ptq4ris: Post-training quantization for referring image segmentation,” in ICRA,

2025. 9

- [256] M. Bellver, C. Ventura, C. Silberer, I. Kazakos, J. Torres, and X. Giro-i Nieto, “A closer look at referring expressions for video object segmentation,” Multimedia Tools and Applications, 2022. 9

- [257] D. Wu, T. Wang, Y. Zhang, X. Zhang, and J. Shen, “Onlinerefer: A simple online baseline for referring video object segmentation,” in ICCV, 2023. 10, 16
- [258] R. Yan, W. Guo, X. Liu, X. Liu, Y. Zhang, and X. Yuan, “Trackingforced referring video object segmentation,” in ACM MM, 2024. 10
- [259] B. Miao, M. Bennamoun, Y. Gao, M. Shah, and A. Mian, “Temporally consistent referring video object segmentation with hybrid memory,” IEEE TCSVT, 2024. 10
- [260] J. Tang, G. Zheng, and S. Yang, “Temporal collection and distribution for referring video object segmentation,” in ICCV, 2023. 10, 16
- [261] T. Hui, S. Liu, Z. Ding, S. Huang, G. Li, W. Wang, L. Liu, and J. Han, “Language-aware spatial-temporal collaboration for referring video segmentation,” IEEE TPAMI, 2023. 10, 16
- [262] M. Han, Y. Wang, Z. Li, L. Yao, X. Chang, and Y. Qiao, “Html: Hybrid temporal-scale multimodal learning framework for referring video object segmentation,” in ICCV, 2023. 10, 16
- [263] Z. Zhou, W. Xiong, L. Zhou, X. Li, Z. He, and Y. Wang, “Harnessing vision-language pretrained models with temporal-aware adaptation for referring video object segmentation,” arXiv, 2024. 10
- [264] H. Wang, C. Deng, F. Ma, and Y. Yang, “Context modulated dynamic networks for actor and action video segmentation with language queries,” in AAAI, 2020. 10
- [265] T. Hui, S. Huang, S. Liu, Z. Ding, G. Li, W. Wang, J. Han, and F. Wang, “Collaborative spatial-temporal modeling for language-queried video actor segmentation,” in CVPR, 2021. 10
- [266] T. Liang, H. Jiang, Y. Yang, C. Tan, S. Li, W.-S. Zheng, and J.-F. Hu, “Long-rvos: A comprehensive benchmark for long-term referring video object segmentation,” arXiv, 2025. 10
- [267] B. McIntosh, K. Duarte, Y. S. Rawat, and M. Shah, “Visual-textual capsule routing for text-based video segmentation,” in CVPR, 2020. 10
- [268] D. Wu, X. Dong, L. Shao, and J. Shen, “Multi-level representation learning with semantic alignment for referring video object segmentation,” in CVPR, 2022. 10, 16
- [269] W. Chen, D. Hong, Y. Qi, Z. Han, S. Wang, L. Qing, Q. Huang, and G. Li, “Multi-attention network for compressed video referring object segmentation,” in ACM MM, 2022. 10
- [270] Z. Zhu, X. Feng, D. Chen, J. Yuan, C. Qiao, and G. Hua, “Exploring pre-trained text-to-video diffusion models for referring video object segmentation,” in ECCV, 2024. 10, 16
- [271] H. Wang, C. Deng, J. Yan, and D. Tao, “Asymmetric cross-guided attention network for actor and action video segmentation from natural language query,” in ICCV, 2019. 10
- [272] B. Miao, M. Bennamoun, Y. Gao, and A. Mian, “Spectrum-guided multi-granularity referring video object segmentation,” in ICCV, 2023. 10, 16
- [273] D. Li, R. Li, L. Wang, Y. Wang, J. Qi, L. Zhang, T. Liu, Q. Xu, and H. Lu, “You only infer once: Cross-modal meta-transfer for referring video object segmentation,” in AAAI, 2022. 10
- [274] X. Yang, H. Wang, D. Xie, C. Deng, and D. Tao, “Object-agnostic transformers for video referring segmentation,” IEEE TIP, 2022. 10
- [275] J. Wu, Y. Jiang, P. Sun, Z. Yuan, and P. Luo, “Language as queries for referring video object segmentation,” in CVPR, 2022. 10, 16, 17
- [276] M. Gao, J. Yang, J. Han, K. Lu, F. Zheng, and G. Montana, “Decoupling multimodal transformers for referring video object segmentation,” IEEE TCSVT, 2023. 10
- [277] F. Pan, H. Fang, F. Li, Y. Xu, Y. Li, L. Benini, and X. Lu, “Semantic and sequential alignment for referring video object segmentation,” in CVPR, 2025. 10
- [278] Y. Li, J. Zhang, X. Teng, L. Lan, and X. Liu, “Refsam: Efficiently adapting segmenting anything model for referring video object segmentation,” arXiv, 2023. 10
- [279] F. Rong, M. Lan, Q. Zhang, and L. Zhang, “Mpg-sam 2: Adapting sam 2 with mask priors and global context for referring video object segmentation,” in ICCV, 2025. 10, 16
- [280] M. Li, S. Li, X. Zhang, and L. Zhang, “Univs: Unified and universal video segmentation with prompts as queries,” in CVPR, 2024. 10, 16
- [281] M. Sun, J. Xiao, E. G. Lim, C. Zhao, and Y. Zhao, “Unified multimodality video object segmentation using reinforcement learning,” IEEE TCSVT, 2023. 10
- [282] S. Yan, R. Zhang, Z. Guo, W. Chen, W. Zhang, H. Li, Y. Qiao, H. Dong, Z. He, and P. Gao, “Referred by multi-modality: A unified temporal transformer for video object segmentation,” in AAAI, 2024. 10, 17
- [283] X. Li, H. Yuan, W. Li, H. Ding, S. Wu, W. Zhang, Y. Li, K. Chen, and C. C. Loy, “OMG-Seg: Is one model good enough for all segmentation?” in CVPR, 2024. 10
- [284] S. Huang, R. Ling, H. Li, T. Hui, Z. Tang, X. Wei, J. Han, and S. Liu, “Unleashing the temporal-spatial reasoning capacity of gpt for

- training-free audio and language referenced video object segmentation,” in AAAI, 2025. 10
- [285] S. Yu, D. Liu, Z. Ma, Y. Hong, Y. Zhou, H. Tan, J. Chai, and M. Bansal, “Veggie: Instructional editing and reasoning of video concepts with grounded generation,” arXiv, 2025. 10
- [286] A. Botach, E. Zheltonozhskii, and C. Baskin, “End-to-end referring video object segmentation with multimodal transformers,” in CVPR,

2022. 10, 16

- [287] S. Kim, W. Jin, S. Lim, H. Yoon, H. Choi, and S. Kim, “Referring video object segmentation via language-aligned track selection,” arXiv, 2024. 11
- [288] H. Fang, R. Cong, X. Lu, X. Zhou, S. Kwong, and W. Zhang, “Decoupled motion expression video segmentation,” in CVPR, 2025. 11
- [289] W. Zhao, K. Nan, S. Zhang, K. Chen, D. Lin, and Y. You, “Learning referring video object segmentation from weak annotation,” arXiv,

2023. 11

- [290] C.-S. Lin, I. Liu, M.-H. Chen, C.-Y. Wang, S. Liu, Y.-C. F. Wang et al., “Groprompt: Efficient grounded prompting and adaptation for referring video object segmentation,” in CVPR Workshop, 2024. 11
- [291] R. Zheng, L. Qi, X. Chen, Y. Wang, K. Wang, Y. Qiao, and H. Zhao, “Villa: Video reasoning segmentation with large language model,” arXiv, 2024. 11, 16
- [292] J. Zhu, Z.-Q. Cheng, J.-Y. He, C. Li, B. Luo, H. Lu, Y. Geng, and X. Xie, “Tracking with human-intent reasoning,” arXiv, 2023. 11
- [293] Y. Shen, B. Liu, C. Li, L. Seenivasan, and M. Unberath, “Online reasoning video segmentation with just-in-time digital twins,” arXiv,

2025. 11

- [294] L. Lin, X. Yu, Z. Pang, and Y.-X. Wang, “Glus: Global-local reasoning unified into a single large language model for video segmentation,” in CVPR, 2025. 11, 16
- [295] S. Munasinghe, H. Gani, W. Zhu, J. Cao, E. Xing, F. S. Khan, and S. Khan, “Videoglamm: A large multimodal model for pixel-level visual grounding in videos,” in CVPR, 2024. 11
- [296] X. Li, J. Wang, X. Xu, M. Yang, F. Yang, Y. Zhao, R. Singh, and B. Raj, “Towards noise-tolerant speech-referring video object segmentation: Bridging speech and text,” in EMNLP, 2023. 11, 12
- [297] G. Li, M. Gao, H. Liu, X. Zhen, and F. Zheng, “Learning crossmodal affinity for referring video object segmentation targeting limited samples,” in ICCV, 2023. 11
- [298] L. Ouyang, R. Liu, Y. Huang, R. Furuta, and Y. Sato, “Actionvos: Actions as prompts for video object segmentation,” in ECCV, 2024. 11
- [299] A. Deng, T. Chen, S. Yu, T. Yang, L. Spencer, Y. Tian, A. S. Mian, M. Bansal, and C. Chen, “Motion-grounded video reasoning: Understanding and perceiving motion at pixel level,” in CVPR, 2025. 11
- [300] A. Athar, X. Deng, and L.-C. Chen, “Vicas: A dataset for combining holistic and pixel-level video understanding using captions with grounded segmentation,” in CVPR, 2025. 11
- [301] H. Liu, G. Li, M. Gao, X. Zhen, F. Zheng, and Y. Wang, “Few-shot referring video single- and multi-object segmentation via cross-modal affinity with instance sequence matching,” IJCV, 2025. 11
- [302] K. Ying, H. Hu, and H. Ding, “MOVE: Motion-guided few-shot video object segmentation,” in ICCV, 2025. 11
- [303] H. Liu, M. Gao, X. Luo, Z. Wang, G. Qin, J. Wu, and Y. Jin, “Resurgsam2: Referring segment anything in surgical video via credible long-term tracking,” arXiv, 2025. 11
- [304] R. Yuan, M. Chen, J. Xu, L. Zhou, Q. Li, Y. Zhang, R. Feng, T. Zhang, and S. Gao, “Text-promptable propagation for referring medical image sequence segmentation,” arXiv, 2025. 11
- [305] Q. Yang, X. Nie, T. Li, P. Gao, Y. Guo, C. Zhen, P. Yan, and S. Xiang, “Cooperation does matter: Exploring multi-order bilateral relations for audio-visual segmentation,” in CVPR, 2024. 11, 16
- [306] X. Li, J. Wang, X. Xu, X. Peng, R. Singh, Y. Lu, and B. Raj, “Qdformer: Towards robust audiovisual segmentation in complex environments with quantization-based semantic decomposition,” in CVPR, 2024. 11, 16
- [307] K. Li, Z. Yang, L. Chen, Y. Yang, and J. Xiao, “Catr: Combinatorialdependence audio-queried transformer for audio-visual video segmentation,” in ACM MM, 2023. 11
- [308] Y.-B. Lin, Y.-L. Sung, J. Lei, M. Bansal, and G. Bertasius, “Vision transformers are parameter-efficient audio-visual learners,” in CVPR,

2023. 11, 16

- [309] J. Li, S. Yu, Y. Wang, L. Wang, and H. Lu, “Selm: Selective mechanism based audio-visual segmentation,” in ACM MM, 2024. 11
- [310] Y. Mao, J. Zhang, M. Xiang, Y. Zhong, and Y. Dai, “Multimodal variational auto-encoder based audio-visual segmentation,” in ICCV,

2023. 11, 16

- [311] S. Mo and P. Morgado, “Unveiling the power of audio-visual early fusion transformers with dense interactions through masked modeling,” in CVPR, 2024. 11, 16
- [312] Y. Wang, W. Liu, G. Li, J. Ding, D. Hu, and X. Li, “Prompting segmentation with sound is generalizable audio-visual source localizer,” in AAAI, 2024. 11, 17
- [313] Z. Shi, Q. Wu, F. Meng, L. Xu, and H. Li, “Cross-modal cognitive consensus guided audio-visual segmentation,” IEEE TMM, 2024. 11, 16
- [314] S. Gao, Z. Chen, G. Chen, W. Wang, and T. Lu, “Avsegformer: Audiovisual segmentation with transformer,” in AAAI, 2024. 11, 17
- [315] Y. Guo, S. Ma, S. Ma, X. Bao, C.-W. Xie, K. Zheng, T. Weng, S. Sun, Y. Zheng, and W. Zou, “Aligned better, listen better for audio-visual large language models,” in ICLR, 2025. 11
- [316] C. Liu, P. Li, L. Yang, D. Wang, L. Li, and X. Yu, “Robust audio-visual segmentation via audio-guided visual convergent alignment,” in CVPR,

2025. 11, 16, 17

- [317] C. Liu, L. Yang, P. Li, D. Wang, L. Li, and X. Yu, “Dynamic derivation and elimination: Audio visual segmentation with enhanced audio semantics,” in CVPR, 2025. 11, 16
- [318] T. Chen, Z. Tan, T. Gong, Q. Chu, Y. Wu, B. Liu, L. Lu, J. Ye, and N. Yu, “Bootstrapping audio-visual segmentation by strengthening audio cues,” IEEE TCSVT, 2025. 11
- [319] D. Hao, Y. Mao, B. He, X. Han, Y. Dai, and Y. Zhong, “Improving audio-visual segmentation with bidirectional generation,” in AAAI,

2024. 11

- [320] J. Liu, Y. Liu, F. Zhang, C. Ju, Y. Zhang, and Y. Wang, “Audio-visual segmentation via unlabeled frame exploitation,” in CVPR, 2024. 11
- [321] S. Gong, Y. Zhuge, L. Zhang, Y. Wang, P. Zhang, L. Wang, and H. Lu, “Avs-mamba: Exploring temporal and multi-modal mamba for audiovisual segmentation,” IEEE TMM, 2025. 11
- [322] Y. Chen, C. Wang, Y. Liu, H. Wang, and G. Carneiro, “Cpm: Classconditional prompting machine for audio-visual segmentation,” ECCV,

2024. 11, 16

- [323] S. Xu, S. Wei, T. Ruan, L. Liao, and Y. Zhao, “Each perform its functions: Task decomposition and feature assignment for audio-visual segmentation,” IEEE TMM, 2024. 11, 16
- [324] C. Liu, P. P. Li, X. Qi, H. Zhang, L. Li, D. Wang, and X. Yu, “Audiovisual segmentation by exploring cross-modal mutual semantics,” in ACM MM, 2023. 11
- [325] C. Liu, P. Li, H. Zhang, L. Li, Z. Huang, D. Wang, and X. Yu, “Bavs: bootstrapping audio-visual segmentation by integrating foundation knowledge,” IEEE TMM, 2024. 11, 16
- [326] S. Mo and B. Raj, “Weakly-supervised audio-visual segmentation,” NeurIPS, 2024. 11, 16
- [327] S. Bhosale, H. Yang, D. Kanojia, J. Deng, and X. Zhu, “Unsupervised audio-visual segmentation with modality alignment,” in AAAI, 2025. 11
- [328] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “Imagebind: One embedding space to bind them all,” in CVPR, 2023. 11
- [329] H. Zhong, M. Zhu, Z. Du, Z. Huang, C. Zhao, M. Liu, W. Wang, H. Chen, and C. Shen, “Omni-r1: Reinforcement learning for omnimodal reasoning via two-system collaboration,” arXiv, 2025. 12
- [330] Y. Wang, H. Xu, Y. Liu, J. Li, and Y. Tang, “Sam2-love: Segment anything model 2 in language-aided audio-visual scenes,” in CVPR,

2025. 12, 17

- [331] Z. Qian, Y. Ma, J. Ji, and X. Sun, “X-refseg3d: Enhancing referring 3d instance segmentation via structured cross-modal graph neural networks,” in AAAI, 2024. 12, 17
- [332] C. Wu, Y. Ma, Q. Chen, H. Wang, G. Luo, J. Ji, and X. Sun, “3d-stmn: Dependency-driven superpoint-text matching network for end-to-end 3d referring expression segmentation,” in AAAI, 2024. 12, 17
- [333] X. Liu, X. Xu, J. Li, Q. Zhang, X. Wang, N. Sebe, and L. Ma, “Less: Label-efficient and single-stage referring 3d segmentation,” in NeurIPS,

2024. 12, 17

- [334] J. Sun, C. Qing, J. Tan, and X. Xu, “Superpoint transformer for 3d scene instance segmentation,” in AAAI, 2023. 12
- [335] C. Wu, J. Ji, H. Wang, Y. Ma, Y. Huang, G. Luo, H. Fei, X. Sun, R. Ji et al., “Rg-san: Rule-guided spatial awareness network for end-to-end 3d referring expression segmentation,” in NeurIPS, 2024. 12, 17
- [336] Q. Chen, C. Wu, J. Ji, Y. Ma, D. Yang, and X. Sun, “Ipdn: Imageenhanced prompt decoding network for 3d referring expression segmentation,” in AAAI, 2025. 12, 17
- [337] K.-C. Huang, X. Li, L. Qi, S. Yan, and M.-H. Yang, “Reason3d: Searching and reasoning 3d segmentation via large language model,” in 3DV, 2025. 12, 17

- [338] X. Jiang, L. Lu, L. Shao, and S. Lu, “Multimodal 3d reasoning segmentation with complex scenes,” arXiv, 2024. 12
- [339] J. Deng, T. He, L. Jiang, T. Wang, F. Dayoub, and I. Reid, “3d-llava: Towards generalist 3d lmms with omni superpoint transformer,” in CVPR, 2025. 12, 17
- [340] H. Lin, Y. Luo, X. Zheng, L. Li, F. Chao, T. Jin, D. Luo, Y. Wang, L. Cao, and R. Ji, “A unified framework for 3d point cloud visual grounding,” arXiv, 2023. 12
- [341] Z. Qian, Y. Ma, Z. Lin, J. Ji, X. Zheng, X. Sun, and R. Ji, “Multi-branch collaborative learning network for 3d visual grounding,” in ECCV, 2024. 12, 17
- [342] X. Li, J. Ding, Z. Chen, and M. Elhoseiny, “Uni3dl: Unified model for 3d and language understanding,” in ECCV, 2024. 13
- [343] W. Xu, C. Shi, S. Tu, X. Zhou, D. Liang, and X. Bai, “A unified framework for 3d scene understanding,” in NeurIPS, 2024. 13, 17
- [344] W. Chen, M. Qu, W. Kang, Y. Yan, Y. Zhao, and Y. Wei, “3drest: A strong baseline for semi-supervised 3d referring expression segmentation,” arXiv, 2025. 13
- [345] Y. Liu, C. Wu, X. Sun, J. Ji, Y. Ma, G. Luo, l. Cao, and R. Ji, “Weaklysupervised 3d referring expression segmentation,” arXiv, 2025. 13
- [346] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,” in CVPR, 2024. 13
- [347] H. Ding, C. Liu, S. He, X. Jiang, and Y.-G. Jiang, “GREx: Generalized referring expression segmentation, comprehension, and generation,” arXiv, 2025. 13
- [348] Y. Zong, Q. Zhang, D. An, Z. Li, X. Xu, L. Xu, Z. Tu, Y. Xing, and O. Dabeer, “Ground-v: Teaching vlms to ground complex instructions in pixels,” in CVPR, 2025. 13, 17
- [349] J. Wu, X. Li, X. Li, H. Ding, Y. Tong, and D. Tao, “Towards robust referring image segmentation,” IEEE TIP, 2024. 13
- [350] Y. Wu, Z. Zhang, C. Xie, F. Zhu, and R. Zhao, “Advancing referring expression segmentation beyond single image,” in ICCV, 2023. 13
- [351] X. Li, J. Wang, X. Xu, X. Li, B. Raj, and Y. Lu, “Robust referring video object segmentation with cyclic structural consensus,” in ICCV, 2023. 13, 17
- [352] W. Li, Z. Zhao, H. Bai, and F. Su, “Bring adaptive binding prototypes to generalized referring expression segmentation,” IEEE TMM, 2024. 13,

- 17

[353] E.-R. Nguyen, H. Le, D. Samaras, and M. Ryoo, “Instance-aware generalized referring expression segmentation,” arXiv, 2024. 13, 17,

- 18

- [354] Z. Luo, Y. Wu, T. Cheng, Y. Liu, Y. Xiao, H. Wang, X.-P. Zhang, and Y. Yang, “Cohd: A counting-aware hierarchical decoding framework for generalized referring expression segmentation,” arXiv, 2024. 13, 17
- [355] W. Huang, D. Liu, and W. Hu, “Dense object grounding in 3d scenes,” in ACM MM, 2023. 13
- [356] ——, “Advancing 3d object grounding beyond a single 3d scene,” in ACM MM, 2024. 13
- [357] V. K. Nagaraja, V. I. Morariu, and L. S. Davis, “Modeling context between objects for referring expression understanding,” in ECCV,

2016. 13

- [358] P. Wang, Q. Wu, J. Cao, C. Shen, L. Gao, and A. v. d. Hengel, “Neighbourhood watch: Referring expression comprehension via languageguided graph attention networks,” in CVPR, 2019. 13
- [359] M. Feng, Z. Li, Q. Li, L. Zhang, X. Zhang, G. Zhu, H. Zhang, Y. Wang, and A. Mian, “Free-form description guided 3d visual graph network for object grounding in point cloud,” in ICCV, 2021. 13
- [360] Y. Wu, X. Cheng, R. Zhang, Z. Cheng, and J. Zhang, “Eda: Explicit text-decoupling and dense alignment for 3d visual grounding,” in CVPR,

2023. 13

- [361] M. Guo, Z. Zhang, H. Fan, and L. Jing, “Divert more attention to visionlanguage tracking,” NeurIPS, 2022. 13
- [362] D. Ma and X. Wu, “Tracking by natural language specification with long short-term context decoupling,” in ICCV, 2023. 13
- [363] H. Fan, L. Lin, F. Yang, P. Chu, G. Deng, S. Yu, H. Bai, Y. Xu, C. Liao, and H. Ling, “Lasot: A high-quality benchmark for large-scale single object tracking,” in CVPR, 2019. 13
- [364] Y. Du, C. Lei, Z. Zhao, and F. Su, “ikun: Speak to trackers without retraining,” in CVPR, 2024. 13
- [365] J. Liu, W. Wang, L. Wang, and M.-H. Yang, “Attribute-guided attention for referring expression generation and comprehension,” IEEE TIP,

2020. 13

- [366] B. A. Plummer, L. Wang, J. C. Caicedo, J. Hockenmaier, and S. Lazebnik, “Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models,” IJCV, 2017. 14

- [367] T. Gupta, A. Vahdat, G. Chechik, X. Yang, J. Kautz, and D. Hoiem, “Contrastive learning for weakly supervised phrase grounding,” in ECCV, 2020. 14
- [368] C. Gonz´alez, N. Ayobi, I. Hern´andez, J. Hern´andez, J. Pont-Tuset, and P. Arbel´aez, “Panoptic narrative grounding,” in ICCV, 2021. 14
- [369] C. Gonz´alez, N. Ayobi, I. Hern´andez, J. Pont-Tuset, and P. Arbel´aez, “Piglet: Pixel-level grounding of language expressions with transformers,” IEEE TPAMI, 2023. 14
- [370] S. Dai, J. Liu, and N.-M. Cheung, “Referring expression counting,” in CVPR, 2024. 14
- [371] N. Amini-Naieni, T. Han, and A. Zisserman, “Countgd: Multi-modal open-world counting,” in NeurIPS, 2024. 14
- [372] M. Shridhar and D. Hsu, “Interactive visual grounding of referring expressions for human-robot interaction,” in RSS, 2018. 14
- [373] G. Tziafas, Y. Xu, A. Goel, M. Kasaei, Z. Li, and H. Kasaei, “Languageguided robot grasping: Clip-based referring grasp synthesis in clutter,” in CoRL, 2023. 14
- [374] J. Mao, J. Huang, A. Toshev, O. Camburu, A. L. Yuille, and K. Murphy, “Generation and comprehension of unambiguous object descriptions,” in CVPR, 2016. 14, 15
- [375] P. Kr¨ahenb¨uhl and V. Koltun, “Efficient inference in fully connected crfs with gaussian edge potentials,” in NeurIPS, 2011. 15
- [376] H. Wang, Q. Chen, C. Yan, J. Cai, X. Jiang, Y. Hu, W. Xie, and S. Gavves, “Object-centric video question answering with visual grounding and referring,” in ICCV, 2025. 16
- [377] D. R. Martin, C. C. Fowlkes, and J. Malik, “Learning to detect natural image boundaries using local brightness, color, and texture cues,” IEEE TPAMI, 2004. 16
- [378] J. Ma, P. Sun, Y. Wang, and D. Hu, “Stepping stones: A progressive training strategy for audio-visual semantic segmentation,” ECCV, 2024. 16
- [379] Y. Chen, Y. Liu, H. Wang, F. Liu, C. Wang, H. Frazer, and G. Carneiro, “Unraveling instance associations: A closer look for audio-visual segmentation,” in CVPR, 2024. 16
- [380] A. Radman and J. Laaksonen, “Tsam: Temporal sam augmented with multimodal prompts for referring audio-visual segmentation,” in CVPR,

2025. 17

- [381] Z. Luo, Y. Wu, Y. Liu, Y. Xiao, X.-P. Zhang, and Y. Yang, “Hdc: Hierarchical semantic decoding with counting assistance for generalized referring expression segmentation,” arXiv, 2024. 17
- [382] S. Cao, Z. Wei, J. Kuen, K. Liu, L. Zhang, J. Gu, H. Jung, L.-Y. Gui, and Y.-X. Wang, “Refer to anything with vision-language prompts,” arXiv preprint arXiv:2506.05342, 2025. 17
- [383] H. You, H. Zhang, Z. Gan, X. Du, B. Zhang, Z. Wang, L. Cao, S.-F. Chang, and Y. Yang, “Ferret: Refer and ground anything anywhere at any granularity,” in ICLR, 2024. 18
- [384] M. Dai, L. Yang, Y. Xu, Z. Feng, and W. Yang, “Simvg: A simple framework for visual grounding with decoupled multi-modal fusion,” NeurIPS, 2024. 18
- [385] Z. Sun, Y. Liu, H. Zhu, Y. Gu, Y. Zou, Z. Liu, G.-S. Xia, B. Du, and Y. Xu, “Refdrone: A challenging benchmark for referring expression comprehension in drone scenes,” arXiv preprint arXiv:2502.00392,

2025. 18

- [386] F. Liang, B. Wu, X. Dai, K. Li, Y. Zhao, H. Zhang, P. Zhang, P. Vajda, and D. Marculescu, “Open-vocabulary semantic segmentation with mask-adapted clip,” in CVPR, 2023. 18
- [387] S. Yang, T. Qu, X. Lai, Z. Tian, B. Peng, S. Liu, and J. Jia, “Lisa++: An improved baseline for reasoning segmentation with large language model,” arXiv, 2023. 18
- [388] Y. Liu, T. Qu, Z. Zhong, B. Peng, S. Liu, B. Yu, and J. Jia, “Visionreasoner: Unified visual perception and reasoning via reinforcement learning,” arXiv, 2025. 18

