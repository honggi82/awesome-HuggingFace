# arXiv:2403.09333v2[cs.CV]11Aug2025

## Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring

Yufei Zhan1,2, Shurong Zheng1,4, Yousong Zhu1, , Hongyin Zhao1, Fan Yang1,4, Ming Tang1,2, Jinqiao Wang1,2,3,4,

1 Foundation Model Research Center, Institute of Automation, Chinese Academy of Sciences, Beijing, China 2 School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China 3 Peng Cheng Laboratory, Shenzhen, China 4 Wuhan AI Research, Wuhan, China

{zhanyufei2021, zhengshurong2023, zhaohongyin2020, yangfan 2022}@ia.ac.cn {yousong.zhu, tangm, jqwang}@nlpr.ia.ac.cn

#### Abstract

Large Vision Language Models have achieved fine-grained object perception, but the limitation of image resolution remains a significant obstacle to surpassing the performance of task-specific experts in complex and dense scenarios. Such limitation further restricts the model’s potential to achieve nuanced visual and language referring in domains such as GUI Agents, counting, etc. To address this issue, we introduce a unified high-resolution generalist model, Griffon v2, enabling flexible object referring with visual and textual prompts. To efficiently scale up image resolution, we design a simple and lightweight down-sampling projector to overcome the input tokens constraint in Large Language Models. This design inherently preserves the complete contexts and fine details and significantly improves multimodal perception ability, especially for small objects. Building upon this, we further equip the model with visuallanguage co-referring capabilities through a plug-and-play visual tokenizer. It enables user-friendly interaction with flexible target images, free-form texts, and even coordinates. Experiments demonstrate that Griffon v2 can localize objects of interest with visual and textual referring, achieve state-of-the-art performance on REC and phrase grounding, and outperform expert models in object detection, object counting, and REG. Data and codes are released at https://github.com/jefferyZhan/Griffon.

#### 1. Introduction

Large Vision Language Models (LVLMs) [2, 15, 29, 56] show promising performance in region-level tasks like Referring Expression Comprehension (REC) after the break-

through in image-text understanding [3, 11] and reasoning [19]. In particular, models like Griffon [53] have demonstrated more compelling perception capability in object detection tasks. This has further spurred the development of flexible references of objects beyond only textual descriptions for better user interaction.

Despite these progresses, current LVLMs still meet the bottleneck in surpassing the task-specific experts in finegrained tasks, tripped by the image resolution. In other words, they can hardly capture the nuanced visual details, leading to a plethora of fact-conflicting hallucinations. It is particularly evident when dealing with low-resolution scenarios such as answering region-based questions without basis [38], failing in small words in characters-related tasks [33], or providing incorrect counting results [23].

To address this issue, recent works have explored resolution enhancement and flexible visual-language referring in LVLMs. On the one hand, previous methods [12, 13] adopt a progressive training approach to gradually improve the resolution. However, the maximum input length of Large Language Models (LLMs) imposes a constraint on achieving higher image resolutions. Additionally, some approaches [25, 28, 30] divide the image into smaller patches and encode them separately for zooming in. This divisionbased design proves sub-optimal multimodal perception ability, which loses the contexts and edge details of patches and increases computation complexity [25]. On the other hand, prior research has predominantly studied various forms of references [6, 9, 54] to improve the understanding of specific image content based on low-resolution images (e.g. 224 or 448). However, these methods often excel at perceiving prominent, and image-level objects, but fall short in accurately localizing and describing fine-grained local objects. Additionally, singular visual or language prompts

###### Cross-image Counting

###### Object Counting

Unified Textual Output

Ins: Find all objects similar to <region> and output their coordinates in the image

Ins: In this image find all objects similar to <region>

potted plant-[0.37, 0.335, 0.409, 0.498] & tv-[0.011, 0.394, 0.244, 0.617] & tv-[0.871, 0.491, 0.998,

and output their coordinates.

0.676] & chair-[0.561, 0.512, 0.648, 0.753] & chair-[0.454, 0.512, 0.551, 0.743] & chair-[0.646, 0.523, 0.693, 0.714] & chair-[0.496, 0.515, 0.53, 0.542] & person-[0.645, 0.37, 0.728, 0.694] & …

<region>:cross image

[Figure 1]

[Figure 2]

[Figure 3]

<region>: screenshot

+

| |
|---|

Griffon v2

[Figure 4]

[Figure 5]

[Figure 6]

Visual-Language Co-Referring

[x1, y1, x2, y2]

Screenshot target

Textual Descriptions Cross-image target

Language Visual

###### REC

Object Detection

###### Visual Grounding

###### REG

Ins: Examine the image for any objects from the category set. Report the coordinates of each detected object. The category set includes person, car…

Ins: Help me find all the hamburgers please.

Ins: What is in [0.077, 0.586, 0.589, 0.738] ?

Ins: Locate a chair with towel and seat in the image.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Figure 1. Overview of Griffon v2. Griffon v2 enables high-resolution input and seamless visual-language co-referencing for LVLMs, allowing users to refer to objects through coordinates, textual descriptions, screenshots, or cross-image modes. It excels in localizing arbitrary objects and generating descriptions with precise co-referencing across diverse scenarios.

alone either lack conversational abilities or are constrained by linguistic descriptions [17], failing to provide a comprehensive user-friendly interactive experience.

In this paper, we propose Griffon v2 to unveil the direct high-resolution design and endow it with locating any objects of interest with visual-language co-referring. Instead of partitioning the high-resolution image into smaller patches, we employ a high-resolution visual encoder to directly extract representations, and design a simple and lightweight down-sampling projector with strided convolution to compress the length of visual tokens. The compressed visual tokens are then trained to align with text tokens and fed into LLM for further fusion like LLaVA [29]. Compared to complex resampler structure[28] and image partitioning methods mode[25, 30], the proposed direct high-resolution pipeline preserves context and advances dense instance-level tasks a lot with competitive VQA performances. It is also parameter-efficient and computationally concise. More importantly, we build a visual-language co-referring paradigm to enhance the model’s fine-grained perception of high-resolution inputs, greatly expanding the model’s applicability. It supports local cropped images, texts, and coordinates prompting, and outputs coordinates of target objects or corresponding text descriptions, providing various interactive abilities, thereby mitigating the conversational deficiencies of singular visual prompting and the potential expressive ambiguity of textual prompting. Finally, we collected 12M publicly available localization data for pre-training and 900K instruction data for finetuning. We achieve advanced results in the REC task, phrase grounding task, and Referring Expression Genera-

tion (REG) task. Notably, our model outperforms several object detection and object counting expert models for the first time.

In summary, our main contributions are:

- 1. We propose a high-resolution multimodal perception model for better local understanding and less global context loss, better suited for challenging tasks such as intensive multi-object detection and counting in complex scenarios.
- 2. We introduce a visual-language co-referring structure, which broadens the model’s scope of application and provides various interaction modes.
- 3. We have conducted experiments on a wide range of localization-related tasks and demonstrated state-of-theart performance on REC, phrase grounding, and REG. We surpass expert models in object detection and object counting task quantitatively and qualitatively.

#### 2. Related Work

##### 2.1. LVLMs with Localization

With the significant advancements made in LVLMs[1, 2, 15, 29], object localization as an essential foundation visual task has been explored with LVLMs. One branch of these methods [26, 42, 49, 55] focuses on invoking the detection expert models through an API. When the LVLM requires detection, the LLM issues instructions to call the detection model and receives the results for further processing. This approach increases the complexity of the model and can not enjoy the unified structure with LVLMs. Another set of methods [4, 9, 13, 50] focuses on fine-grained single-object

|[Figure 12]<br><br>[Figure 13]<br><br>Visualization|
|---|

|Tokenizer<br><br>|𝑋𝑖𝑛𝑠1: In the image, could you<br><br>please describe the object in [0.951, 0.723, 0.982, 0.814] briefly?|
|---|
<br><br>0.360,0.989] & [0.356, 0.000,<br><br>| |
|---|---|
|𝑋𝑖𝑛𝑠2: In this image, can you<br><br>locate all the black pigeons?| |
|Language Referring Input| |

|Motorcycle-[0.000, 0.358, 0.272, 0.679] & Motorcycle-[0.261, 0.360, 0.521, 0.675]<br><br>Where are the motorcycle in the image?<br><br>Motorcycle on the far left.<br><br>What is [0.000, 0.358, 0.272, 0.679]?<br><br>Conversation|
|---|

[Figure 14]

𝑋𝑎 (for 𝑋𝑞 and 𝑋𝑖𝑛𝑠2): [0.034,0.101,0.328,0.989] & [0.166,0.276, 0 0.549, 0.989] & [0.518,0.231,0.878,0.989]…

[Figure 15]

|𝑋𝑎 (for 𝑋𝑖𝑛𝑠1): A black pigeon standing on the white ground.|
|---|

###### Large Language Model

- (b1) Language Referring

|Locate and count <region> in the image.<br><br>[Figure 16]<br><br>2: Motorcycle-[0.000, 0.358, 0.272, 0.679]<br><br>& Motorcycle-[0.261, 0.360, 0.521, 0.675]<br><br>2: Motorcycle-[0.000, 0.358, 0.272, 0.679] & Motorcycle-[0.261, 0.360, 0.521, 0.675]<br><br>Any objects similar to <region> in the image.<br><br>[Figure 17]<br><br>Conversation|
|---|

|[Figure 18]<br><br>[Figure 19]<br><br>Visualization|
|---|

- (b2) Visual Referring

Down-sampling Projector

Visual Tokenizer

Linear Projector

Linear Projector

Convolution Down Sampling

Region Encoder

High-Resolution Visual Encoder

[Figure 20]

[Figure 21]

|𝑋𝑞: Visual Referring Image|
|---|

|𝑋𝑣: High-resolution Input Image|
|---|

(a) Griffon v2’s High-Resolution Structure (b) Visual-Language Co-referring Examples

Figure 2. Structure of the proposed high-resolution Griffon v2 with visual-language co-referring.

localization, i.e. REC. These methods transform REC task data into instruction data and encode the coordinates with different representation approaches, enabling the LVLM itself to possess localization capabilities. However, methods like Ferret [50] fall short in more complex tasks like object detection and more challenging scenarios with lots of small objects under the low-resolution. Until the appearance of Griffon[53], which supports coarser granularity object detection, the object localization capacity of LVLM is extended to multi-object and arbitrary granularity. Our model further proposes a high-resolution structure that elevates the detection and object counting capability of LVLM beyond expert models, which has also proved more effective and efficient than existing division-based resolution-enhanced methods [25, 28]. By enhancing the object detection and enabling object counting, our work address a broader spectrum of granularity in comprehension, perception and localization.

##### 2.2. Object Referring in Multimodal Models

Since the application of LLMs [1, 44] in large multimodal models, textual descriptions have become the most straightforward method for object referring. Users naturally utilize this method to ask questions without even paying special attention, such as “What is in front of this cat?” Despite its simplicity, it struggles to distinctly refer to specific classes of objects that are difficult to describe in dense and complex scenes, such as the cell shown in Figure 1. The growing importance of region comprehension has sparked the exploration of spatial references in complex and dense scenes. Current works have proposed spatial referrals with textual coordinates [4, 8, 9], learnable embeddings [36], and Re-

gion of Interest features [54]. Some approaches [1, 6, 50] upgrade from the user-friendly perspective, supporting arrows, marks, and so on. Although these methods enhance the convenience of user interaction, they primarily focus on designing visual prompts to distinguish one specific object from others and are mostly applied in text-output VQA tasks. In contrast, our model uses visual referring to represent a class of objects, not limited to distinction of individual objects. Additionally, we introduce bounding box output to enable dense scenario understanding in LMMs for the first time. The proposed visual-language co-referring architecture empowers our model to handle various tasks while remaining user-friendly.

#### 3. Methodology

In this section, we start with a brief overview of our Griffon v2. As the key foundation, we first describe the highresolution structure design. Then, we represent the visuallanguage co-referring and the training pipeline.

##### 3.1. Overview

As depicted in Figure 2(a), Griffon v2 employs an autoregression paradigm, seamlessly integrating referring and grounding tasks within a unified language modeling objective. For a given input image Xv, we leverage a visual encoder, adapted to high-resolution by bilinear interpolation from the pre-trained EVA2-CLIP-L/14@336 [43] model to extract high-resolution visual features (e.g. over 1K). Simultaneously, the designed down-sampling projector transforms these image features into visual embedding tokens Hv. In cases where a user customizes a visual prompt Xq with a screenshot or an additional target image, region em-

beddings Hq are derived from the visual referring image using the visual tokenizer. Meanwhile, we inherently support language references, Xins, to identify objects, such as textual descriptions and coordinates, which are projected into text embeddings as Hins after tokenization. Subsequently, the visual prompt or text embeddings together with image embeddings undergo processing using the LLM, specifically Llama2-13B [44], resulting in the generation of the desired answer Xa. Notably, the representation of bounding boxes eschews the use of special tokens and specialized heads and instead incorporates textual numerical values, thereby maintaining simplicity.

##### 3.2. Efficient High-Resolution Scaling Design

High resolution has been empirically proven to enhance the accurate interpretation of nuanced visual features, thereby benefiting both visual tasks and vision-language tasks. In previous methods [25], images are commonly partitioned into smaller patches, enlarging their total resolution to 762 or 896. However, this approach faces limitations in terms of contextual coherence and the representation of edge details among patches. While progressive scaling methods can extend resolution to 784 (patch size=14) within the constraint of 4096 input tokens, they are unsuitable for handling outputs involving extensive long texts and significantly increase the computation complexity.

To address these challenges, we introduce a highresolution structure, incorporating a down-sampling projector capable of supporting a resolution of more than 1K. Upon receiving an input image of arbitrary dimensions, we resize it to a predefined input height and width, denoted as Xv ∈ RH×W×3. Instead of employing image division, we encode the image using a trainable high-resolution visual encoder adapted from a pre-trained model by bilinear interpolation, yielding visual features Zv = fV (Xv). In contrast to low-resolution fixed encoders, our approach excels in capturing finer details. As illustrated in the left part of Figure 2(a), we subsequently introduce a lightweight downsampling projector to abstract essential features with compression under the guidance of the training objective and connect visual features to word embedding space. Specifically, a strided convolution layer (Conv) is simply applied to down-sample the features by S, and a projection matrix W is utilized to convert Zv into visual embedding tokens:

Hv = W · Conv(Zv), with Zv = fV (Xv). (1)

It can be inferred that the final number of visual tokens is quadratically related to the input size and the convolution stride, i.e. H,W, and S. To demonstrate the effectiveness of our design, we employ the convolution layer with kernel size at 3 and stride at 2 and investigate several different resolutions in Figure 3. As depicted in Figure 3, comparing the results at resolutions of 700 and 448 which use two-

1400

| |mAP<br><br>Visual token Num.| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

30

1200

25

Numberofvisualtokens

1000

MSCOCOval2017mAP

20

800

15

600

10

400

5

200

0

0

448 Griffon v1

1022 Griffon v2 Resolution of input image

700 Griffon v2

Figure 3. Performance and visual token number comparison of different resolutions under the same training data. The 448-resolution model, i.e. Griffon v1, employs a two-layer linear projector, whereas the others utilize our down-sampling projector (Stride=2, Kernel Size=3). Our method is capable of extracting features while preserving details, thereby reducing token redundancy.

layer linear projection [29] without reduction, using the resolution of 700 with our structure shows a higher mAP and fewer tokens. This indicates that our method is capable of extracting features while preserving details, thereby reducing token redundancy. It is also observed that further increasing the resolution can lead to improved performance.

##### 3.3. Visual-Language Co-Referring

Recognizing the limitations of singular visual or language referring demonstrated before, lack of conversational ability, and potential ambiguity respectively, we capitalize on CLIP’s adeptness in extracting robust semantic features to facilitate visual-language co-referring. This approach mitigates these drawbacks and finds broader applicability under high-resolution conditions.

Visual Referring. While various existing methods propose diverse visual reference forms, ranging from marking on the raw image to object masking, either of them alters the raw image content [1] or requires a specially designed extractor [50]. To strike a balance between efficiency and referring accuracy, we opt for a straightforward method, using visual referring images to represent the target of interest. Specifically, as depicted in the middle of Figure2(a), given a user-provided referring image Xq, we employ a region encoder, i.e. EVA-CLIP-L/14@224 [43] pre-trained ViT to extract visual prompt features Zq. To align the LLM’s understanding of visual references with textual references, we project the region features into the word embedding space, yielding the visual prompt token Hq. Based on the robust semantic representation capability of [CLS] token, pre-trained on a dataset of billions of instances [39], we discard other tokens following previous

applications [39, 43] and utilize the [CLS] token to represent the specific object. This approach allows us to refer to the object at the category level, eliminating overly specific information from pixel patches and maintaining the stability of reference. To accommodate diverse conversation scenarios, we use “<region>” in instructions to represent the referred object as shown in Figure 2(b). Before inputting into the LLM, we replace this placeholder with the embedding of the referred object, seamlessly integrating it for optimal coherence.

Language Referring. Textual referring is an inherent capability of LLM-based multimodal models. Despite the simplicity of textual referring, it faces the challenge of resulting in wrong answers with ambiguous referring. This has naturally prompted us to consider whether associating texts with regions could allow LLMs to comprehend the corresponding object based on textual-region reference and thereby fulfill task instructions and keep the simplicity advantage. Therefore, we further support textual coordinate references as illustrated in Figure 2(b). In addition to the detailed description, an object can also be referred to by its top-left and bottom-right coordinates. In this case, it is processed the same as textual descriptions, involving tokenization and embedding into the word space as Hins, depicted at the right side of Figure 2(a). Through training with instruction-following data whose object references are transformed into textual coordinates, we demonstrate that LLMs can accurately refer to objects in complex and dense scenes based on the textual coordinates of objects. Through collaborative referring involving both text and vision, our approach achieves optimal referring performance and flexible interaction.

##### 3.4. Training Pipeline

To understand different users’ intents and finish diverse tasks with high-quality results, we adopt a three-stage endto-end training procedure for Griffon v2 as follows guided by [29, 53].

Stage Annotation Type Sources

- I Image-text LLaVA

- II

Object Detection Objects 365, MSCOCO REC/REG Visual Genome, RefCOCO series Visual Grounding V3Det, LVIS, Flickr30K Entities Object Counting CA-44, OpenImages, Self-collected Non-existing Judging LVIS

- III Instruction-following Build from stage 2

Table 1. Data statistics of different stages. We convert the data of different annotations to Prompt/Instruction-Answer form. We will release the self-collected and processed data in Stage II and III. More details are listed in the supplements.

###### Stage I: High-resolution Vision-Language Alignment.

Feature alignment before pretraining has been widely utilized to achieve better training efficiency. We adopt this strategy to connect the high-resolution visual encoder with the LLM using 558K image-text pairs[29]. The highresolution image encoder and LLM parameters remain frozen, with only the down-sampling projector trainable. Without visual prompts in these image-text pairs, the visual prompt tokenizer is not trained in Stage I.

###### Stage II: Co-referring Multi-Tasks Pre-training.

Building on the basic understanding of visual content achieved in Stage I, we further pre-train the entire model with a diverse dataset comprising 12M multi-task instances involving both visual and textual referring. This stage aims to enhance fine-grained perception and localization capabilities, and visual-language referring abilities. As detailed in the data composition in Table 1, our textual-referring data is curated from various task datasets covering over 70K object categories, such as Visual Genome [22] and V3Det[45]. Additionally, we build a dataset with public and self-collected object counting data from 10 different domains, spanning aerial photography, agriculture, industry, and more, to create our visual-referring data. The diverse textual categories and visual domains contribute to the model’s generalization capability. During this stage, we train the entire network while keeping the region encoder in the visual prompts tokenizer frozen.

###### Stage III: Intent-Enhanced Instruction Tuning. Fol-

lowing the pretraining of stage II, the model gains the ability to locate and describe objects of interest with free-form texts and flexibly obtained visual referring images. It can be customized and enhanced by users to achieve specific tasks as a foundation model. To further enhance its understanding ability of user’s intents, we finetune the model with nearly 900K instruction-following data built from stage II and more diverse instruction prompts for different tasks detailed in supplements. Moreover, to preserve the refined visual feature extraction and prevent forgetting, we keep the high-resolution visual encoder and region encoder frozen and train the LLM together with the projectors.

#### 4. Experiments

##### 4.1. Implementation Details

Following the widely used configuration of ResNet[16], we set the convolution layer of the down-sampling projector with a kernel size of 3, a stride of 2, and a padding of 1. For the chosen image resolution, we consider both the patch size of pre-trained CLIP and the token length of our data under the constraint of Llama2, supporting a maximum of 4096 input tokens. We follow the previous methods to utilize the L-size CLIP model, whose patch size is usually 14 or 16. As the average textual token length of counting data

###### Type Model Res. Epochs mAP AP50 AP75 APS APM APL

Faster RCNN-FPN[41] 1022 12 37.9 58.6 40.9 20.4 41.1 50.3 Faster RCNN-C4[41] 1022 12 35.6 55.7 37.8 17.0 40.6 50.3 DAB-DETR[31] 1333 12 38.0 60.3 39.8 19.2 40.9 55.4 Pix2Seq[10] 1333 300 43.0 61.0 45.6 25.1 46.9 59.4 DETR[7] 1333 500 42.0 62.4 44.2 20.5 45.8 61.1

Specialists

Griffon-13B[53] 448 1 24.8 40.6 25.1 5.9 25.5 48.7 Lumen[18] 448 - 35.3 53.2 35.8 - - Qwen2.5-VL-7B[5] dynamic - 16.2 25.0 16.7 5.9 16.8 33.8 InternVL2.5-8B[14] dynamic - 11.9 19.4 12.1 1.9 11.2 24.2

Generalist

###### Griffon v2 1022 1 38.5 54.3 41.2 19.4 43.2 57.6

Table 2. Object detection results on MSCOCO val2017 [27]. Griffon v2 shows a significant improvement over advanced LVLMs.

Type Model MAE(↓) NAE(↓)

FamNet[40] 68.5 2.3 FSDetView[48] 29.0 0.8 Counting-DETR[35] 23.5 0.6

Specialists

Qwen2.5-VL-7B[5] 33.0 0.9 InternVL2.5-8B[14] 28.5 0.9 Griffon v2 20.3 0.5

Generalist

- Table 3. Object counting results on the dense FSCD-LVIS unseen test set [35]. MAE stands for Mean Average Error, and NAE for Normalized Relative Error.

is approximately 2500, the maximum achievable resolution is 1022 (patch size = 14) or comparable 1024 (patch size = 16), corresponding to 1369 tokens or 1024 tokens respectively. As shown in Table 5, as the EVA-CLIP-L/14 has the best performance, we set the resolution to 1022. We initialize the visual encoder with adapted EVA2-CLIP-ViTL/14@336 by position embedding interpolation and LLM with Llama2, leaving the down-sampling projector and projector of visual tokenizer randomly initialized. More details are demonstrated in the supplements.

##### 4.2. Complex Detection and Counting

Object detection and counting are essential visual perception tasks, presenting significant challenges with multiple categories and dense objects. We evaluate models on these two tasks as the first LVLM and demonstrate our finegrained perception ability in complex and dense scenarios.

Object Detection. The object detection task is evaluated on MSCOCO val2017 [27] using textual references and Griffon-13B’s prompt. We input all test categories simultaneously for each image and calculate the confidence score for each prediction following [53]. As illustrated in Table 2, we outperform existing expert models, including Faster RCNN (C4 and FPN)[41] and DAB-DETR[31], with fewer

training epochs and lower input resolution. Moreover, we outperform the first pure LVLM generalist Griffon-13B under the same data and training settings as depicted in Figure 3, and also achieve substantial improvements across all metrics compared to the generalist Griffon-13B overall. Compared to Lumen[18], which uses a task-specific decoder to output detection results, our Griffon v2 surpasses by 3.2% on mAP score, showcasing the superior detection capability of ours. The latest Qwen2.5-VL[5], in contrast, faces challenges in object detection, as the partition-based highresolution structure may lose edge details and contexts of patches and is not the optimal solution for intensive multiobject localization under high-resolution.

Object Counting. Object counting is conducted with visual references and tested on the unseen test classes set of FSCD-LVIS [35] aiming for accurate counting and facilitating generalization comparisons. The visual reference is constructed by randomly selecting one example box from the set and screenshotting the region in the image. For Qwen2.5-VL-7B[5] and InternVL2.5-8B[14], we query the model to count the number of the target following their trained settings. As depicted in Table 3, we surpass existing classical expert models with lower MAE and NAE. Notably, our approach not only outputs the number but also provides the bounding boxes of detected objects. This marks the first time LVLMs achieve real expert-level detection and counting, showcasing the superiority of Griffon v2 and the generalization ability of our visual reference approach. By contrast, Qwen2.5-VL-7B and InternVL2.5-8B struggle to predict the accurate number of specific category of targets in dense scenarios, proving the effectiveness of our highresolution scaling strategy and co-referring mechanism.

##### 4.3. Evaluation on Referring and Grounding

Basic grounding and referring mainly include the visual grounding and REC tasks, typically involving one preexisting object or a limited number of multiple targets. Grif-

ODINW

Type Model

RefCOCO/+/g

AerialDrone Aquarium Rabbits

EgoHands Mushrooms

Packages PascalVOC

pistols pothole Raccoon Shellfish thermal Vehicles

AVG

AVG

MDETR[20] 0.6 1.7 66.5 5.9 39.8 63.6 5.6 15.9 12.7 50.6 8.1 4.5 13.4 22.2 83.4 G-DINO-L[32] 12.6 28.1 71.7 52.0 72.3 63.9 66.0 71.4 30.4 65.8 62.5 21.3 62.7 52.4 86.6

Spec.

Ferret-13B† [50] 0 4.3 59.8 1.5 6.1 40.1 35.2 41.5 3.9 49.5 29.5 36.5 44.4 27.1 85.6 Griffon-13B[53] - - - - - - - - - - - - - - 84.0 InternVL2.5-8B† [14] 0 6.9 38.5 0.2 26.7 16.4 37.0 29.2 1.1 46.6 28.5 3.8 27.1 20.2 87.6 Qwen2.5-VL-3B[5] 6.2 16.4 75.0 24.6 8.3 66.6 52.0 42.3 10.2 47.7 36.7 40.7 57.1 37.2 85.0 Qwen2.5-VL-7B† [5] 7.8 20.3 73.5 32.2 7.0 57.6 49.8 48.5 7.4 40.1 42.7 38.0 56.3 37.0 86.6

Gen.

Griffon v2 5.4 18.2 75.1 25.7 63.7 62.1 62.0 43.1 6.4 39.8 44.1 43.1 57.6 42.0 90.0

- Table 4. Visual Grounding and REC results. Spec. represents specialists, while Gen. represents generalists. RefCOCO/+/g utilizes the metric of ACC@0.5, while ODINW uses mAP metric. † indicates the results are reproduced to get results of all sets following the official Settings.

fon v2 is systematically compared with specialist and generalist models across these tasks.

As a fundamental task in visual grounding, the REC task has been extensively researched in the LVLMs as a basic localization task grounding a single object with textual description. We evaluate our model on the RefCOCO [51], RefCOCO+ [51], RefCOCOg [34] and it achieves an average score of 90.0. As illustrated in Table 4, our model demonstrates excellent performance, outperforming InternVL2.5-8B[14] and Qwen2.5-VL-7B[5].

To comprehensively assess the model’s capabilities to generalize detection of uncommon categories in diverse real-world scenarios, we also conduct experiments on a more intensive object grounding benchmark, Object Detection in the Wild (ODinW)[24]. ODinW contains more rare categories and multiple target locations in the wild.

As shown in Table 4, our model outperforms recent advanced LVLMs across multiple sets especially when the query category corresponds to multiple targets, such as Mushrooms and PascalVOC. Compared to the latest Qwen2.5-VL-7B[5], our model improved the average score by 5.0%, narrowing the gap with specialist models.

Beyond these two visual grounding tasks, we also evaluate phrase grounding and REG on Flickr30K Entities[37] and RefCOCOg [34] respectively. The detailed experimental setup and results are reported in Appendix 5.

##### 4.4. Ablation Studies

In the ablation studies below, by default, we mainly evaluate Griffon v2 on the object detection task on MSCOCO val2017 [27] and train the whole model with only the train2017 set after feature alignment.

Different pre-trained visual encoders. Existing methods utilize the visual encoder from the pre-trained CLIP

model, and the performance of pre-trained models varies with different optimization methods. We compare EVA2CLIP-L/14 [43], original CLIP-L/14 [39] and SAM-CLIP16 [21]. We apply bilinear interpolation to upscale the resolution from 336 to 1022, which closely matches the 1024 resolution of the pre-trained SAM-CLIP-16, due to the patch size difference (14 v.s.16). As shown in Table 5, the EVA2-CLIP-based visual encoder outperforms the other two models with an average improvement of 2%. Comparing the CLIP-based models using positional interpolation and the SAM-based model without interpolation, indicates that with our structure design, using positional interpolation to increase resolution can also work with even better performance.

High-resolution design. Recent works[25, 28, 30] incorporate image partitioning to achieve dynamic highresolution resolution. Though effective on multimodal benchmarks, it’s unknown how it performs on fine-grained localization tasks. To further validate the effectiveness of our direct high-resolution scaling strategy, we use the same train data to conduct a fair comparison with the partitionbased high-resolution approach. We use the multimodal data from LLaVA-NeXt[30] and compare with it, and use the COCO data to train Monkey[25] with similar solution for holistic comparison. As shown in Table 6, Griffon v2 outperforms LLaVA-NeXt in general VQA and largely exceeding Monkey by 6.7% in object detection. The results underscore the effectiveness of our high-resolution structure design, which can preserve the visual contexts and adapt to global understanding tasks.

Down-sampling structures. Previous methods apply the resampler proposed in Flamingo [2] to down-sample the visual features with learnable tokens while increasing the resolution. We compare this approach with our designed

Referring Expression Comprehension

Visual/Phrase Grounding Penguin Window Doughnut

Referring Expression Generation Question-Region: [0.545,0.084,0.869,0.540]

Grounding with referring expressions shown in images

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

| |
|---|

Object Detection with MSCOCO Categories

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Answer: a orange container holding a type of food that is shaped like a triangle

Question Region: [0.349,0.209,0.546,0.744]

[Figure 33]

Object Counting with Cross and Cropped Images

| |
|---|

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

[Figure 44]

[Figure 45]

Answer: a shirtless man with blue

Cross-image: Cross-image: Cross-image: Screenshot: Screenshot: Screenshot:

shorts skateboarding down a ramp

Figure 4. Visualization results of Griffon v2 across five vision and vision-language tasks.

Pretrained mAP AP50 AP75 APS APM APL AR100

CLIP 29.8 48.6 29.7 10.0 32.7 53.9 41.4 SAM 28.4 43.1 29.6 11.2 30.8 46.9 40.9 EVA2-CLIP 31.9 50.4 32.7 14.3 38.6 52.3 43.8

Table 5. Ablations on different pre-trained visual encoders.

Model VQAv2 ChartQA MMEP POPE mAP LLaVA-NeXt[30] 80.9 62.2 1575 86.3 -

Monkey[25] - - - - 17.6 Griffon v2 81.9 69.8 1581 87.5 24.3

Table 6. Comparison with partition-based high-resolution structure on VQAs and object detection. We use the same train data with LLaVA-NeXt and Monkey respectively to ensure fairness.

down-sampling projector in terms of the performance and memory of this module during training. As depicted in Table 7, our model achieves higher precision with less memory consumption, which is quite important for large-scale pre-training of LVLMs. While the resampler can extract semantic information for understanding tasks, it falls short in capturing fine-grained details for perception and localization tasks under the same training setting, necessitating more epochs [7].

##### 4.5. Qualitative Analysis

We further evaluate Griffon v2’s performance across five tasks by presenting visualization results. As depicted in Figure 4, Griffon v2 consistently demonstrates its ability to precisely locate objects of interest and generate accurate descriptions through visual-language co-referring. We pro-

Type mAP AP50 AP75 Mem. Resampler 9.6 18.4 8.8 416G Down-sample projector (ours) 28.4 43.1 29.6 461M

Table 7. Ablations on the different projectors with down-sampling. Mem. donates the memory consumption of this block including the parameters and forward/backward pass.

vide more results in Appendix 6.

#### 5. Conclusion

In this study, we present Griffon v2, an innovative highresolution multimodal model supporting resolutions up to 1K and facilitating visual-language co-referring. Our designed high-resolution structure directly extracts visual features and projects them into visual tokens with compression effectively and efficiently without division. Subsequently, we introduce a visual-language co-referring paradigm that accommodates locally cropped images, texts, and coordinates as prompts, offers diverse interactive capabilities, and mitigates the limitations of singular visual and textual prompting. Trained through our 3-stage end-to-end pipeline and the 12M multi-tasks and 900K instruction dataset, Griffon v2 surpasses expert models in object detection and counting tasks within a unified LVLM and demonstrates competitive performance across REC, REG, and phrase grounding tasks. Griffon v2 establishes a robust foundation for further exploration in the realm of intelligent multimodal systems with fine-grained perception and localization capabilities. We hope that the performance of Griffon v2 will instill confidence in the advancement of LVLMs.

#### 6. Acknowledgement

This work was supported by Beijing Natural Science Foundation (L247028) and was in part by the Beijing Municipal Science and Technology Project (Z231100007423004), and National Natural Science Foundation of China (No.62276260, No.62472423).

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2, 3, 4

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736,

2022. 1, 2, 7

- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425– 2433, 2015. 1
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 2, 3
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. 6, 7
- [6] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. arXiv preprint arXiv:2312.00784, 2023. 1, 3
- [7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 6, 8
- [8] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechu Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023. 3
- [9] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 1, 2, 3

- [10] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. arXiv preprint arXiv:2109.10852, 2021. 6
- [11] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 1
- [12] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565, 2023. 1
- [13] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, et al. Pali-3 vision language models: Smaller, faster, stronger. arXiv preprint arXiv:2310.09199, 2023. 1, 2
- [14] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. 6, 7
- [15] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 1, 2

- [16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 5
- [17] Qing Jiang, Feng Li, Tianhe Ren, Shilong Liu, Zhaoyang Zeng, Kent Yu, and Lei Zhang. T-rex: Counting by visual prompting. arXiv preprint arXiv:2311.13596, 2023. 2
- [18] Yang Jiao, Shaoxiang Chen, Zequn Jie, Jingjing Chen, Lin Ma, and Yu-Gang Jiang. Lumen: Unleashing versatile vision-centric capabilities of large multimodal models. Advances in Neural Information Processing Systems, 37: 81461–81488, 2025. 6
- [19] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017. 1
- [20] Aishwarya Kamath, Mannat Singh, Yann LeCun, Ishan Misra, Gabriel Synnaeve, and Nicolas Carion. Mdetr– modulated detection for end-to-end multi-modal understanding. arXiv preprint arXiv:2104.12763, 2021. 7, 3
- [21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-

- head, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 7
- [22] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 5
- [23] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A high-resolution multimodality model. arXiv preprint arXiv:2311.04219, 2023. 1
- [24] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10965–10975, 2022. 7
- [25] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607,

2023. 1, 2, 3, 4, 7, 8

- [26] Yaobo Liang, Chenfei Wu, Ting Song, Wenshan Wu, Yan Xia, Yu Liu, Yang Ou, Shuai Lu, Lei Ji, Shaoguang Mao, et al. Taskmatrix. ai: Completing tasks by connecting foundation models with millions of apis. arXiv preprint arXiv:2303.16434, 2023. 2
- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 6, 7
- [28] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 1, 2, 3, 7
- [29] Haotian Liu and et al. Visual instruction tuning. In NeurIPS,

2023. 1, 2, 4, 5

- [30] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 1, 2, 7, 8
- [31] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic anchor boxes are better queries for detr. arXiv preprint arXiv:2201.12329, 2022. 6
- [32] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 7
- [33] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 1

- [34] Varun K Nagaraja, Vlad I Morariu, and Larry S Davis. Modeling context between objects for referring expression understanding. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 792–807. Springer,

2016. 7, 3

- [35] Thanh Nguyen, Chau Pham, Khoi Nguyen, and Minh Hoai. Few-shot object counting and detection. In European Conference on Computer Vision, pages 348–365. Springer, 2022. 6
- [36] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 3
- [37] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015. 7, 3
- [38] Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, et al. Cogcom: Train large vision-language models diving into details through chain of manipulations. arXiv preprint arXiv:2402.04236, 2024. 1
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4, 5, 7
- [40] Viresh Ranjan, Udbhav Sharma, Thu Nguyen, and Minh Hoai. Learning to count everything. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 6
- [41] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015. 6
- [42] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580, 2023. 2
- [43] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 3, 4, 5, 7
- [44] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3, 4
- [45] Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. V3det: Vast vocabulary visual detection dataset. In The IEEE International Conference on Computer Vision (ICCV), 2023. 5
- [46] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu,

- Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. arXiv preprint arXiv:2308.01907, 2023. 3
- [47] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. arXiv preprint arXiv:2212.00280, 2022. 3
- [48] Yang Xiao, Vincent Lepetit, and Renaud Marlet. Few-shot object detection and viewpoint estimation for objects in the wild. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3090–3106, 2022. 6
- [49] Hui Yang, Sifu Yue, and Yunzhong He. Auto-gpt for online decision making: Benchmarks and additional opinions. arXiv preprint arXiv:2306.02224, 2023. 2
- [50] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704, 2023. 2, 3, 4, 7
- [51] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 7
- [52] Licheng Yu, Hao Tan, Mohit Bansal, and Tamara L Berg. A joint speaker-listener-reinforcer model for referring expressions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7282–7290, 2017. 3
- [53] Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon: Spelling out all object locations at any granularity with large language models. In European Conference on Computer Vision, pages 405–422. Springer, 2025. 1, 3, 5, 6, 7
- [54] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023. 1, 3
- [55] Yang Zhao, Zhijie Lin, Daquan Zhou, Zilong Huang, Jiashi Feng, and Bingyi Kang. Bubogpt: Enabling visual grounding in multi-modal llms. arXiv preprint arXiv:2307.08581,

2023. 2

- [56] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1

## Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring

### Supplementary Material

This supplementary document extends our main paper by providing more details about the dataset we have constructed, the unified representation, the implementation, the Phase Grounding results, the REG results, the result analysis on REC, and more qualitative analyses that are not included in the main paper due to the length limit. We will release the code and data upon publication of the paper.

#### 1. Dataset Details

As demonstrated in the Training Pipeline of the main paper, we have collected and processed 12M data to build our pre-training and instruction-following dataset, with visuallanguage co-referring. In the section, We detail the data construction and the main data processing as below.

##### 1.1. Pre-training Data Construction

To imbue the model with fine-grained perception and localization capabilities, and proficiency in visual-language co-referring, we curate a dataset of nearly 12 million localization-related instances with textual or visual reference. As illustrated in Table 8, we encompass six localization-related tasks, transforming their respective datasets into a conversational style using task-specific prompts. The data from the object counting task are utilized for visual reference, while the remaining datasets serve for textual reference. Alongside the utilization of publicly available datasets, we have derived a counting subset comprising 416K instances from OpenImages v4 and a selfcollected counting dataset comprising 266K instances. The counting subset filters out images lacking categories with instance numbers exceeding 5. The self-collected counting dataset integrates data from 11 domains, as depicted in Figure 5 and Figure 6. This broad domain coverage ensures the generalization of our model without succumbing to overfitting in any particular scenario.

##### 1.2. Instruction-following Data Construction

In contrast to the extensive data providing wide knowledge used in the pre-training phase, we leverage a smaller subset of the multi-task localization pre-training data with a greater diversity of instruction prompts, exampled in Table 9, to enhance the model’s understanding of intents. Instead of manually selecting subsets from various domains, we have opted for random sampling for both the visual grounding task and object counting task. We utilize the RefCOCO series for the REG/REC and MSCOCO for object detection.

Type Dataset Name Vol. REC/REG

Visual Genome 3.6M RefCOCO/+/g 288K

MSCOCO 118K Objects365 1.7M

Object Detection

LVIS 361K V3Det 638K Flickrs30K Entities 427K

Visual/Phrase Grounding

CA-44 22K OpenImages v4 416K Self-collected 266K

Object Counting

Non-existing Judging LVIS 96K

Table 8. The statistic of the composition of pre-training data.

[Figure 46]

Figure 5. Data distribution of the self-collected counting data.

The data of each task realize a relative balance in terms of quantity.

##### 1.3. Data Processing

As previously mentioned, we consolidate six tasks into a unified instruction-answer format. For REC, REG, and object detection, we adopt the processing methodology introduced by Griffon-13B [53], wherein raw annotations are directly transformed using randomly sampled instruction prompts. Regarding the visual grounding, object counting, and non-existing judging tasks, we initially convert detection-type annotations such as V3Det into instances, formulating one question for each category and enumer-

Task Example prompts chosen from the instruction set

Where is <expr> <image>? answer in [x0,y0,x1,y1] format.

REC

I am looking for the position of <expr> in <image>. Can you provide its coordinates?

Help me locate and determine the coordinates of <expr> in <image>.

Please generate a distinguishing description for the region <region> in the image <image>.

REG

Describe the area <region> in a unique way, given the picture <image>.

Create a one-of-a-kind description for the region <region> found in the picture <image>.

Identify and locate all the objects from the category set in the image<image>. Please provide the coordinates for each detected object. The category set includes <category set>.

Examine the image<image> for any objects from the category set. Report the coordinates of each detected object. The category set includes <category set>.

Object Detection

Locate and identify the objects from the category set in the image<image>. Output the coordinates of each detected object. The category set includes <category set>.

Would you kindly provide the coordinates of <expr> located in the picture <image>?

Visual Grounding

Find <expr> in <image> and share its coordinates with me.

In the given <image>, can you find <expr> and tell me its coordinates?

Detect and record the positions of objects that bear resemblance to <region> in this image. I want you to find all objects in the image<image> that closely match the characteristics of <region> and give me their coordinates. Can you identify any objects that look like <region> in this image<image>? Output their coordinates for closer inspection, analysis, and comparison.

Object Counting

- Table 9. Examples of task templates on different types of training data. The placeholders are explained as follows: “<image>” represents the input image, “<expr>” represents the expression describing the object, “<category set>” represents the categories to be detected, and “<region>” represents the textual coordinates of the region to be asked or the locally cropped image.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Animals

Retail Aerial Coins

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Biology Gaming Industrial

Human

[Figure 55]

[Figure 56]

[Figure 57]

Agriculture

Pharmaceutical

Transportation

Figure 6. Data samples of the self-collected counting data.

ating all annotated categories for each image. Notably, in the case of non-existing judging data, we leverage the “neg category ids” annotated for each image, indicating categories unequivocally absent in that image. Subsequently, these data are integrated with randomly selected instruction templates.

#### 2. Instruction Template Examples

In order to augment users’ intent comprehension, we employ a diverse training instruction set along with a random sampling strategy. Here, we present a selection of task prompts utilized by Griffon v2 in Table 9. Each task encompasses hundreds of prompts generated by GPT-4 with specific requirements and illustrative examples. It is important to emphasize that Griffon v2 does not impose restrictions on users, allowing them the flexibility to employ their preferred natural language expressions.

#### 3. Unified Representation with VL Coreferring

Griffon v2 employs an enhanced unified input/output representation, building upon the framework introduced in Griffon-13B, in which the input is task-specific instruction and each instance in the output is formulated as “expression-[x1, y2, x2, y2]”. The representation, as preliminarily illustrated in Figure 2 from a referring perspective, has been upgraded to accommodate REG and object counting in addition to the previously supported REC, object detection, and visual/phrase grounding tasks. In Figure 7, for the REG task, we refer to the question region with normalized 3-precision coordinates, “[x1, y1, x2, y2]” uniformly, seamlessly integrating it into the instruction, with the answer describing the region. Regarding object count-

Layer Parameters Value

Stride 2 Kernel 3 Padding 1 inchannel 1024 outchannel 5120

Convolution

inchannel 5120 outchannel 5120

Linear

- Table 10. Hyperparameters of the designed Down-sampling Projector.

Parameter Stage-1 Stage-2 Stage-3 batch size 256 128 128

lr 1e-3 2e-5 2e-5 lr schedule cosine decay

lr warmup ratio 0.03 weight decay 0

epoch 1

optimizer AdamW DeepSpeed stage 2

Max Length 2048 4096 4096

Table 11. Hyperparameters of the training paradigm.

ing with visual referring, we initially employ the placeholder “<region>” in the instruction to denote the target. During training, this region is randomly selected from the bounding box annotation set of a specific category within the image, subsequently cropped out, and represented by the extracted token. The output sequence is the coordinates of detected instances concatenated with “&”. During inference, it’s specified by the user with screenshots or target images.

- 4. Implementation Details

Model parameters and training hyperparameters constitute crucial aspects of the implementation. Beyond the fundamental settings introduced in the main paper, the comprehensive lists of parameters are provided in Table 10 and Table 11. The training hyperparameters predominantly adhere to the LLaVA configuration, with the maximum length extended to 4096 to accommodate higher-resolution images and longer texts. The total training time is about 40 NVIDIA A800 days similar to 100 patch-based Monkey [25] with more data.

- 5. Phase Grounding and REG results

Phrase Grounding. Phrase grounding task presents a greater challenge compared to the REC task and is evalu-

Type Model ANY MERGED

DDPN - 73.5 VisualBert 71.3 -

Spec.

MDETR 83.4 83.8

UniTAB - 79.6 Ferret-13B - 84.8 Shikra-13B - 78.4 Griffon-13B 84.2 82.8

Gen.

Griffon v2 84.8 83.1

Table 12. Phrase grounding results on Flickr30K Entities[37] test set. Spec. represents specialists, while Gen. represents generalists.

Type Model CIDEr Meteor

SLR[52] 66.2 15.9 ASM[46] 41.9 13.6 Grit[47] 71.6 15.2

Spec.

KOSMOS-2[36] 60.3 12.2

Gen.

Griffon v2 72.5 12.1

Table 13. REG resuls on RefCOCOg [34].

ated on Flickr30K Entities [37]. Two evaluation protocols [20] are employed, including the ANY-BOX protocol and MERGE-BOXES protocol. The ANY-BOX protocol focuses on the atomicity of each instance, while the MERGEBOXES protocol evaluates whether the model identifies all referred objects with a merged box. Existing LVLMs are typically limited to the single referent scenario, tending to predict only one box per phase, thereby employing the MERGED-BOXES protocol. As shown in Table 12, Griffon v2 achieves state-of-the-art results in the ANY-BOX protocol and surpasses most specialists and generalists in the MERGE-BOX protocol, with more fine-grained boxes.

REG. REG aims to generate concise descriptions for specified objects based on their region locations. We input textual coordinates for object referring and tests on the RefCOCOg [34] val set. As illustrated in Table 13, in contrast to KOSMOS-2 [36], which uses learnable embeddings for referring, we achieve superior performance in CIDEr, concentrating on the semantic similarity, while the Meteor focuses more on the accuracy of wording, making it less suitable for the open-ended description generation of LLMs.

#### 6. More Qualitative Results

To further demonstrate the performance of our Griffon v2, we provide more visualization results on object detection (Figure 8), visual grounding (Figure 9), and object counting (Figure 10), compared to some expert models.

###### REC/Visual Grounding/Non-existing Judging

Instruction: Where is < expression, e.g., the man wearing a light brown coat and scarf in the middle > in the image? Answer: the man wearing a light brown coat and scarf in the middle-[0.468, 0.409, 0.515, 0.647]

[Figure 58]

###### Object detection

Instruction: Examine the image for any objects from the category set, which includes < e.g., car, skateboard, bus, person, motorcycle, tie, sculpture, backpack, dog and surfboard ... >.

Answer: car-[0.799, 0.436, 1.000, 0.758] & motorcycle-[0.526, 0.513, 0.553, 0.605]& bus-[0.577, 0.146, 0.846, 0.687] & bus-[0.308, 0.259, 0.496, 0.641] & person-[0.237, 0.432, 0.327, 0.727] & person-[0.061, 0.422, 0.127, 0.706] & person-

[0.516, 0.442, 0.564, 0.574] & car-[0.557, 0.472, 0.58, 0.583] & car-[0.497, 0.441, 0.532, 0.476] & backpack-[0.251, 0.547,

0.276, 0.623]…

###### REG

[Figure 59]

Instruction: For the given image, can you provide a unique description of the < region, e.g., is [0.000, 0.358, 0.272, 0.679] >.

Answer: Motorcycle on the far left.

###### Object Counting

[Figure 60]

Instruction: For the given image, can you provide a unique description of the < region, e.g., >. Answer: [0.000, 0.358, 0.272, 0.679] & [0.261, 0.360, 0.521, 0.675]

- Figure 7. Examples of unified representation for each task. Object counting task utilizes visual referring without category information and directly outputs the object coordinates and corresponding number.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Grounding

DINO

Griffon v2

Grounding

Truth

- Figure 8. Comparison with Grounding DINO in object detection. Griffon v2 demonstrates a reduced occurrence of both missed detections (col. 2) and false positives (col. 1,3).

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Griffon v2

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Grounding DINO

- Figure 9. Comparison with Grounding DINO in visual grounding. Griffon v2 and Grounding DINO exhibit comparable visual grounding capabilities.

Griffon v2 T-Rex Visual Referring Image

[Figure 81]

[Figure 82]

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

- Figure 10. Comparison with T-Rex in object counting. Griffon v2 achieves counting proficiency with visual reference comparable to that of the expert model T-Rex.

