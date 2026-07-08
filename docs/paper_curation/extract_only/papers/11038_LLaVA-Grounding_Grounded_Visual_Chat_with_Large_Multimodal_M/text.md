## LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models

Hao Zhang♠∗3, Hongyang Li♢∗, Feng Li♠3, Tianhe Ren†, Xueyan Zou§3, Shilong Liu¶3, Shijia Huang♯, Jianfeng Gao‡2, Lei Zhang†2, Chunyuan Li‡1, Jianwei Yang‡1

♠ HKUST ♢ SCUT ‡ Microsoft Research, Redmond † IDEA § UW-Madison ¶ Tsinghua ♯ CUHK

∗ Equal Contribution 1. Directional Lead 2. Equal Advisory Contribution 3. Work performed during an internship at Microsoft

https://llava-vl.github.io/llava-grounding/

# arXiv:2312.02949v1[cs.CV]5Dec2023

### Abstract

Visual Chat

80

|Co|gVLM|-Chat| | | | | | |
|---|---|---|---|---|---|---|---|---|
|LLaV|A| |Shikr|a| | |LL|aVA-|
| | | | |Bu|boGP|T| |(Ours)|
| |m|iniGPT|v2| | | | | |
| | | | | | | | | |
| | | | |Cog|VLM-|Groun|ding| |

With the recent significant advancements in large multimodal models (LMMs), the importance of their grounding capability in visual chat is increasingly recognized. Despite recent efforts to enable LMMs to support grounding, their capabilities for grounding and chat are usually separate, and their chat performance drops dramatically when asked to ground. The problem is the lack of a dataset for grounded visual chat (GVC). Existing grounding datasets only contain short captions. To address this issue, we have created GVC data that allows for the combination of grounding and chat capabilities. To better evaluate the GVC capabilities, we have introduced a benchmark called Grounding-Bench. Additionally, we have proposed a model design that can support GVC and various types of visual prompts by connecting segmentation models with language models. Experimental results demonstrate that our model outperforms other LMMs on Grounding-Bench. Furthermore, our model achieves competitive performance on classic grounding benchmarks like RefCOCO/+/g and Flickr30K Entities.

70

G

60

50

40

30

0

Grounding

22 24 26 28 30 32 34 36 38

0

Figure 1. A comparison on the integrated ability of visual grounding and visual chat of open-source LMMs on Grounding-Bench. LLaVA-G achieves a good trade-off on both abilities simultaneously. For CogVLM [33], two different model checkpoints are released: CogVLM-Grounding is the grounding model and CogVLM-Chat is the chat model. Grounding and Visual Chat scores represent the F1 score and Chat scores of detailed descriptions in Table 4, respectively. Circle size indicates the model size.

ing grounding and referring capabilities for LMMs [2, 3, 10, 33, 40]. While these models have achieved performance comparable to specialized models [19, 21] on classic grounding benchmarks such as RefCOCO [8] and Flickr30K [29], they often treat grounding as a distinct task that requires customized prompts to initiate. Consequently, their text responses undergo significant changes when tasked with grounding. Most models, such as MiniGPT-v2 [2] and CogVLM-Grounding [33], can only generate short captions when performing grounding, as they are primarily trained on grounding caption data like Flickr30K. As illustrated in Fig.1, these earlier models struggle to excel simultaneously in both chat and grounding tasks. BuboGPT[47] maintains chat capability by leveraging an external grounding model for grounding, but this approach can be constrained by the performance of the language encoder in the grounding model. Shikra [3] engages in referential dialog, which includes grounded chat, but its performance is limited due to the

### 1. Introduction

With the success of large language models (LLMs) like GPT4 [25] and the open-sourced substitutes LLaMA [31], researchers are eager to leverage their strong language capabilities in the field of vision. This enthusiasm has led to a surge in the development of large multimodal models (LLMs). Previous LMMs, such as LLaVA [18] and miniGPT-4 [49], have demonstrated exceptional visual chat abilities by generating plausible responses based on images and user instructions. However, they often encounter challenges in providing responses that exhibit a fine-grained understanding of images, including specific regions and alignment with related image regions—this is often referred to as visual grounding.

Recognizing the significance of visual grounding for LMMs, recent research efforts have focused on develop-

scarcity of available data. All existing LMMs [2, 3, 33, 40] only support outputting coordinates as text, which restricts localization performance, and they do not support pixel-wise grounding and referring. In summary, previous LMMs struggle to perform grounded visual chat effectively due to the scarcity of grounded visual chat data and suboptimal model designs. Furthermore, they lack the capability for pixel-wise grounding and referring.

input output text click box mark text box mask mark

LLaVA [12] ✓ ✓ MiniGPT-4 [49] ✓ ✓ GPT4ROI [46] ✓ ✓ ✓ Shikra [3] ✓ ✓ Ferret [40] ✓ ✓ ✓ ✓ MiniGPTv2 [2] ✓ ✓ ✓ LLaVA1.5 [17] ✓ ✓ ✓ CogVLM-Grounding [33] ✓ ✓ ✓ LLaVA-G (Ours) ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

Table 1. A comparison of input referring and output grounding format of LMMs.

To address these challenges, we contribute to grounded visual chat in three key areas: data creation, network architecture, and benchmarking. When annotating grounding data, previous methods such as Kosmos-2 [28] and GPT4ROI [46] rely on pretrained grounding models or detection models to predict bounding boxes based on existing captions. In contrast, we label grounded visual chat data using humanlabeled object detection data [15].

LLaVA instruction tuning dataset.

- 2. We present an end-to-end model, named LLaVAGrounding (LLaVA-G for brevity), which connects a Large Multimodal Model (LMM) with a grounding model to facilitate grounded visual chat. Our model supports both object and pixel-level grounding, accommodating various visual prompts such as mark, click, box, and scribble. Table 1 demonstrates that our model offers a broader range of input and output prompt types compared to other LMMs.
- 3. We establish the Grounding-Benchbenchmark for evaluating grounded visual chat and propose an auto-evaluation pipeline aided by GPT-4. This benchmark assesses grounded visual chat capabilities and provides performance metrics for other state-of-the-art methods.
- 4. Through extensive experiments, we demonstrate that our model surpasses other grounding LMMs in terms of performance on Grounding-Bench, while also achieving competitive results on classic grounding benchmarks like RefCOCO/+/g and Flickr30K.

Our data creation process begins by leveraging GPT4 [25], following the data creation method used in LLaVA [18]. We provide GPT-4 with chat data and groundtruth instances, instructing it to match instances with noun phrases in the chat data. This approach benefits from the high quality of human-labeled instances and chat data generated by GPT-4, ensuring minimal noise in the data annotation pipeline. In total, we annotated 150K grounded visual chat data.

In terms of network architecture, we propose connecting the output features of the Language Model (LLM) with a grounding model to handle grounding tasks, relieving the language model from the burden of vision localization tasks. For this purpose, we use the open-set segmentation and detection model OpenSeeD [44] as the grounding model, enabling both box and pixel-level grounding simultaneously.

### 2. Method

To evaluate the capability of grounded visual chat, we introduce the Grounding Bench, a benchmark that assesses grounding and chat performances concurrently. Built upon the foundation of LLaVA bench, our benchmark evaluates chat and phrase grounding in three contexts: conversation, detailed description, and complex reasoning. Additionally, recognizing that grounded detailed description is the most challenging aspect of grounded visual chat, we propose grounded recall and precision metrics. Grounded recall measures the proportion of ground-truth instances correctly mentioned and grounded, while grounded precision measures the accuracy of groundings or predicted boxes. We also calculate the F1 score, a combination of precision and recall. To evaluate the correctness of semantic matching since the models generate free-form phrases, we rely on GPT-4.

#### 2.1. Overview

To advance the development of grounded visual chat for Large Multimodal Models (LMMs), we introduce a comprehensive pipeline for labeling grounded visual chat data, a tailored modeling approach designed for the grounded visual chat task, and a benchmark for evaluating grounded visual chat performance, as illustrated in Figure 2. We will provide further details on these three components in the following subsections.

|Metrics & Benchmark|
|---|

In summary, our contributions are as follows:

- 1. We introduce a data annotation pipeline to label highquality Grounded Visual Chat (GVC) data. Leveraging human-labeled object detection data [15] and harnessing the robust matching capability of GPT-4 [27], we have successfully labeled 150K GVC instances using the

|Data Creation Pipeline|
|---|

150K+ Data LLaVA-G

Figure 2. An overview of our main contributions. We use the data creation pipeline to create training and test data. The training data is used to train our LLaVA-G. The test data is used to build our Grounding-Bench.

###### Context type 1: Boxes (for data annotation)

- 1.person: [0.681, 0.242, 0.774, 0.694], 2.person: [0.63, 0.222, 0.686, 0.516], 3.person: [0.444, 0.233, 0.487, 0.34], 4.backpack: [0.384, 0.696, 0.485, 0.914],

[Figure 1]

- 5.backpack: [0.755, 0.413, 0.846, 0.692], 6.suitcase: [0.758, 0.413, 0.845, 0.69], 7.suitcase: [0.1, 0.497, 0.173, 0.579], 8.bicycle: [0.282, 0.363, 0.327, 0.442], 9.car: [0.786, 0.25, 0.848, 0.322], 10.car: [0.783, 0.27, 0.827, 0.335], 11.car: [0.86, 0.254, 0.891, 0.3], 12.car: [0.261, 0.101, 0.787, 0.626]

Context type 2: user responses (for data annotation) The image is an underground parking area with a black sport utility vehicle (SUV) parked. There are three people in the scene, with one person standing closer to the left side of the vehicle, another person in the middle, and the third person on the right side. They are all working together to pack their luggage into the SUV for a trip.

Response: grounded responses (for data annotation)

The image is an underground parking area with a (black sport utility vehicle) [10.car] (SUV) parked. There are (three people) [1.person, 2.person, 3.person] in the scene, with (one person) [3.person] standing closer to the left side of the vehicle, (another person) [2.person] in the middle, and (the third person) [1.person] on the right side. They are all working together to pack (their luggage) [4.backpack, 5.backpack,

- 6.suitcase, 7.suitcase] into the SUV for a trip.

###### Context type 3: predicted grounded responses (for evaluation) The depiction is of a below-ground parking facility, where a sleek, black vehicle [9.car] is situated. In the vicinity of this SUV, a trio of individuals [1.person, 2.person, 3.person] is engaged in an activity: the first person [1.person] is adjacent to the left side of the vehicle, the second [2.person] is situated centrally, and the third [3.person] is near the right side. They are collaboratively arranging their travel bags in the SUV, signaling the onset of an impending journey.

Response: TPpred and TPgt (for evaluation) "a sleek, black vehicle" [9.car] - Incorrectly referred.

"a trio of individuals" [1.person, 2.person, 3.person] - 3 Correctly referred. "the first person" [1.person] - Incorrectly referred. "the second" [2.person] - Correctly referred. "the third" [3.person] - Incorrectly referred.

There are 4 correct references (TPpred) and 3 correctly referred entities (TPgt).

Table 2. Illustrate the data annotation (top) and the evaluation (bottom) with language GPT4. The top table shows the contexts and responses for data annotation. The bottom table shows the contexts and responses for evaluating the recall and precision of grounded description. Note that the Context 1 for evaluation is same as that for data annotation. Note that the visual image is not used to prompt GPT4, we only show it here as a reference.

#### 2.2. Grounded Visual Chat Data Creation

To perform grounded visual chat (GVC) effectively, it is crucial to have high-quality data that encompasses both meaningful conversations and accurate grounding. We have constructed our dataset based on LLaVA instruction tuning data for two primary reasons. Firstly, the conversations within this dataset are generated by GPT-4, known for its high linguistic quality. Secondly, the images used are sourced from COCO, which contains human-annotated grounding box instances.

Our data annotation process aims to associate phrases from conversations with specific instances. To achieve this, we leverage the capabilities of GPT-4. As illustrated in Table 2, we provide GPT-4 with ground-truth (GT) boxes containing class labels and a sentence from the conversation. We task GPT-4 with matching noun phrases from the sentence to the GT instances. Once noun phrases are successfully grounded by GPT-4, we mark them with special start tokens, ⟨gs⟩ and ⟨ge⟩, followed by a token, ⟨seg⟩, which corresponds to the output feature used by the grounding model to segment the grounded region. An example of a question and its answer in the dataset is as follows:

Q: What is the man doing? A: ⟨gs⟩ The man ⟨ge⟩ ⟨seg⟩ is using ⟨gs⟩ a clothing iron ⟨ge⟩ ⟨seg⟩ on the back of ⟨gs⟩ a yellow taxi ⟨ge⟩ ⟨seg⟩.

For each ⟨seg⟩, we have a corresponding segmentation

mask. This annotated data forms the basis of our Grounded Visual Chat (GVC) dataset. Optionally, to support visual prompts in user instructions, we apply a similar annotation process to instances in the question itself. The resulting data appears as follows:

Q: What is the object ⟨obj⟩ doing? A: ⟨gs⟩ The man ⟨ge⟩ ⟨seg⟩ is using ⟨gs⟩ a clothing iron ⟨ge⟩ ⟨seg⟩ on the back of ⟨gs⟩ a yellow taxi ⟨ge⟩ ⟨seg⟩.

It’s important to note that we modify "the man" to "the object" in cases where the model might disregard the visual prompts. For each ⟨obj⟩ in the question, we provide a corresponding segmentation mask. This dataset is referred to as GVC-R (Grounded Visual Chat with Referring).

#### 2.3. Network Architectures

Since our network architecture is nearly identical to LLaVA, with the exception of the additional prompt encoder and grounding model, we will only introduce these two parts in this section. For the other components of our architecture, please refer to LLaVA [18].

Prompt encoder. For an input image Xv and a visual prompt Xp, we employ the pre-trained Semantic-SAM as the prompt encoder. This encoder extracts visual features based on the input image and visual prompts, denoted as Xp = h(Xv,Xp). To convert these prompt features into language embedding tokens Hp of the same dimensionality

instruction-following data is presented as follows:

Visual Grounding Module

Grounding Response

Box Mask

Language Response

Human : Xv < \n > Xq(Xp)<STOP> Assistant : Xa(Xg)<STOP>\n

[Figure 2]

[Figure 3]

(3)

[Figure 4]

Projection

Grounding model

[Figure 5]

[Figure 6]

###### Xa

In this representation, Xp and Xg are enclosed in brackets, indicating that they are optional. During training, the model is trained to predict the assistant’s answers, including the grounded instances and where to stop. Consequently, only the green sequence/tokens are used to compute the loss in the auto-regressive model.

f  Language Model

Hv

Hq

| | |
|---|---|
|P|rojectionW|
| | |
| |ision Encoder|
| | |

Zv

Xq

Projection Prompt Encoder

[Figure 7]

[Figure 8]

Language Instruction

[Figure 9]

[Figure 10]

Xv

Visual prompt

[Figure 11]

Stage 1: Pretraining for alignment. Stage 1 focuses on feature alignment for the visual encoder and granularity alignment for the grounding model.

Image

Visual Interaction Module

Figure 3. Network architecture of our LLaVA-Grounding contains a CLIP vision encoder, a LLM, a prompt encoder, a grounding model and the corresponding projection layers. LLaVA-Grounding expands LLaVA with two additional modules highlighted in blue blocks: the visual interaction module that accepts user drawing and visual grounding module that outputs object masks/boxes. The yellow tokens represents the visual prompt feature aligned to language embedding space. The light green output tokens represent the grounding features which are the last-layer hidden feature of the language model corresponding to ⟨seg⟩ tokens.

Feature alignment for vision encoder. As shown in Table 3, we utilize the RefCOCO/+/g, COCO 2017train, Visual Genome, LLaVA 585K image caption, and Flickr30K Entities datasets for Stage 1. Both LLaVA 585K and Flickr30K Entities datasets consist of image caption pairs and are used to train the projection layer W for feature alignment in the vision encoder. The conversation construction approach aligns with that of LLaVA, where a question is randomly selected from Table 17 as Xq, and the original caption is used as Xa. The learnable parameter for this part is denoted as θ = {W}.

as the word embedding space in the language model, we use a simple linear layer with a trainable projection matrix Wp:

##### Feature and granularity alignment for grounding model.

Hp = Wp · Xp, where Xp = h(Xv,Xp) (1)

To facilitate grounding, we need to align the features Xg output by the language model with the vocabulary space of the grounding model. For this purpose, we train on the RefCOCO/+/g, COCO 2017train, Visual Genome, and Flickr30K Entities datasets. The approach to construct instruction-following data is as follows:

This results in a sequence of visual tokens Hp. It’s worth noting that there are special tokens ⟨obj⟩ in Xq with word embeddings as placeholders, and visual tokens in Hp replace the word embeddings of ⟨obj⟩ in Hq.

Grounding model. In addition to the language response Xa, our model also produces features Xg for grounding. These features correspond to the last layer hidden features of the language model that align with the ⟨seg⟩ tokens. We initially map these features to a grounding space using a trainable projection matrix Wg. Subsequently, we employ a pretrained OpenSeeD model as the grounding model to generate bounding boxes B and masks M. This process can be defined as follows:

- 1. For RefCOCO/+/g and Visual Genome, the user instruc-

tion Xq is randomly selected from Table 16, and Xa consists only of the special token ⟨seg⟩. COCO 2017train follows the same approach as RefCOCO/+/g, but with a distinction: the class name of an instance serves as its referring text.

- 2. In contrast, the Flickr30K Entities dataset differs from the image caption data mentioned earlier. Here, the user instruction is followed by a suffix randomly chosen from Table 18. This suffix signals the model to produce a response in grounding format, as described in Section 2.2. The response Xa is then converted into the grounding format by inserting special tokens ⟨gs⟩, ⟨ge⟩, and ⟨seg⟩ into Xa to mark noun phrases.

B,M = s(Xv,Wg · Xg) (2) Here, s(·,·) represents the grounding model, which takes

the image Xv and the grounding features as input.

Given the instruction-following data, the last-layer hidden features of the language model corresponding to ⟨seg⟩ tokens Xg are mapped to the grounding vocabulary space by multiplying them with Wg. Additionally, since our grounding model is pretrained on COCO and Object365, which have different granularities compared to the Visual Genome

#### 2.4. Training

We propose a three-stage training strategy, as illustrated in Table 3. These stages are pretraining for alignment, instruction tuning for grounded visual chat, and extension to visual prompt. A unified representation of our

Grounding Grounding Seg Visual Chat Chat with VP

RefCOCO/+/g [8, 41] ✓ ✓ ✓ Visual Genome [9] ✓ ✓ COCO train2017 [15] ✓ ✓ LLaVA 585K [18] ✓ Flickr30K [29] ✓ ✓ ✓ LLaVA 150K [18] ✓ GVC 2.2 ✓ ✓ ✓ GVC-R 2.2 ✓

Table 3. Blue, green and red means the training data and tasks in the 1st, 2nd, and 3rd stages, respectively. "Grounding" means only predict boxes and "Grounding Seg" means predict masks. For Flickr30K, we use SAM to label pseudo GT masks. “Chat with VP" means chat with visual prompts.

and Flickr30K grounding data, we also train the grounding model to align these granularities.

In summary, the learnable parameters for Stage 1 are denoted as θ = {W,Wg,ϕg}.

- Stage 2: Instruction tuning for grounded visual chat. In the second training stage, we leverage the Grounded Visual Chat (GVC) data, excluding visual prompts, for instruction tuning. To also support chat without grounding, we incorporate LLaVA 158K instruction-following data. During this stage, we freeze the CLIP vision encoder and focus on finetuning the other components of the model. The learnable pa-

rameters in this stage are denoted as θ = {W,Wg,ϕ,ϕg}.

The data format consists of instruction data containing ⟨seg⟩ tokens in the answer, accompanied by several grounding annotations. The number of grounding annotations corresponds to the number of ⟨seg⟩ tokens present. In this stage, we calculate both language loss and grounding losses. The language loss is computed in the same manner as in LLaVA for the answer tokens and "STOP" tokens. The grounding losses encompass box, mask, and matching losses. Box and mask losses are utilized solely for training the grounding model, while the matching loss is propagated to the language model.

- Stage 3: Extension to visual prompt. In the third stage, we introduce support for visual prompts as an additional component by training only hϕ and the projection layer Wp. As detailed in Table 3, the training data includes RefCOCO/+/g, Visual Genome, and GVC-R. In contrast to Stage 1, for RefCOCO/+/g and Visual Genome, we provide visual prompts for the ground truth (GT) instances and instruct the model to predict captions. The text instruction

Xp is randomly selected from Table 19, where ⟨obj⟩ tokens serve as placeholders, and their input embeddings will be replaced by prompt features. The text answer Xa comprises the original referring expressions.

In this stage, the learnable parameters are represented as θ = {ϕp,Wp}, where ϕp is trained to output boxes and masks corresponding to visual prompts, and Wp is trained

to align visual prompt features with the language embedding space.

Set-of-Mark (SoM) prompts. (Optional) In addition to visual prompts (such as clicks and boxes) that can be handled through the prompt encoder, our model also supports marks as visual prompts, similar to the approach presented in [35]. These marks consist of alphanumerics and masks that are directly overlaid on the image. To illustrate, consider the data sample in Sec.2.2. Let’s assume we overlay marks labeled as ⟨1⟩, ⟨2⟩, and ⟨3⟩ on the "man," "iron," and "taxi" in the input image. This results in the Grounded and Referring Visual Chat (GRVC) data taking the form:

Q: What is the object ⟨1⟩ doing? A: The man ⟨1⟩ is using a clothing iron ⟨2⟩ on the back of a yellow taxi ⟨3⟩.

It’s important to note that both the question and answer consist of text only. Therefore, in order to support marks as visual prompts, we specifically fine-tune the language part of the model.

#### 2.5. Grounding-Bench

Benchmark Creation. We introduce a benchmark named Grounding-Bench to assess a model’s grounded visual chat capability. To evaluate both grounding and chat abilities concurrently, we build this benchmark on top of LLaVA Bench (COCO), which comprises chat data generated by GPT4 and instance annotations from MSCOCO. To enhance the robustness of Grounding-Bench, we expand our test dataset to include 1000 images with 7000 entities, all sourced from the MSCOCO 2014val split. These images are converted into grounded visual chat data using our data creation pipeline, forming the basis of our test dataset.

Task Definition. Grounded visual chat tasks involve taking an image XV and a user instruction I as input and generating a caption T accompanied by bounding boxes b, with each bounding box corresponding to a specific phrase.

Evaluate Chat Scores. Our benchmark evaluation encompasses two main aspects: chat scores and grounded response scores. We outline the evaluation process for GroundingBench in Algorithm 1. Chat scores are akin to those used in LLaVA Bench. However, in contrast, we instruct the model to produce grounded responses. Subsequently, we process the output to remove special tokens and boxes, yielding the pure-text response for evaluation.

Evaluate Grounded Response Scores. For grounded responses, we specifically evaluate the grounded detailed description task. Our evaluation includes metrics such as recall (R) for completeness, precision (P) for hallucination, and the F1 score (F1) to combine both aspects. R measures

CVPR

#5335

CVPR 2024 Submission #5335. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.

CVPR

#5335

- 256 “STOP" tokens. The grounding losses include box, mask and
- 257 matching losses. Box and mask losses are used to train the
- 258 grounding model only and the matching loss will be propa-
- 259 gated to the language model. Stage 3: Extensions to Visual
- 260 Prompt. After the previous stages, we can insert support
- 261 for visual prompt as an add-on component by only train hϕ
- 262 and the projection layer Wp. As shown in Table 3, the train-
- 263 ing data includes Refcoco/+/g, Visual Genome and GVC-R.
- 264 Different from the stage 1, we use Refcoco/+/g and Visual
- 265 Genome by giving the visual prompt of the GT instances
- 266 and let the model predict the caption. The text instruction
- 267 Xp is randomly picked from Table 17 where ⟨obj⟩ tokens
- 268 are placeholders whose input embeddings will be replaced
- 269 by prompt features. The text answer Xa is the original re-
- 270 ferring expressions. The learnable parameters in stage 3
- 271 are θ = {ϕp,Wp} where ϕp is trained to outptu boxes and
- 272 masks corresponding to visual prompts and Wp is trained to
- 273 align visual prompt features to language embedding space.
- 274 Set-of-Mark (SoM) prompts. (Optional) In addition to
- 275 visual prompts (e.g., clicks and boxes) that can be dealt with
- 276 via the prompt encoder, our model also supports marks as
- 277 visual prompts similar to [34]. Marks are alphanumerics and
- 278 masks directly overlayed on the image. Take the data sample
- 279 in Sec.2.2 as an example. Assume we overlay marks with
- 280 ⟨1⟩, ⟨2⟩ and ⟨3⟩ to the “man", “iron" and “taxi" of the input
- 281 image. The Grounded and Referring Visual Chat (GRVC)
- 282 data becomes Q: What is the object ⟨1⟩ doing? A: The man
- 283 ⟨1⟩ is using a clothing iron ⟨2⟩ on the back of a yellow taxi
- 284 ⟨3⟩. Note that both the question and answer are text only.
- 285 Therefore we only tune the language part to support marks
- 286 as visual prompts.
- 287 2.5. Grounding-Bench
- 288 Benchmark Creation: We propose a benchmark named
- 289 Grounding-Bench to measure a model’s capability of
- 290 grounded visual chat. In order to measure grounding and
- 291 chat ability at the same time, we build our benchmark on top
- 292 of LLaVA Bench (COCO) which contains GPT4-generated
- 293 chat data and instance annotations from MSCOCO. In order
- 294 to increase the robustness of Grounding-Bench, we enrich
- 295 our test dataset to contain 1000 images with 7000 entities,
- 296 all from MSCOCO 2014val split. We convert them into
- 297 grounded visual chat data with our data creation pipeline to
- 298 form our test dataset.
- 299 Task Definition: Grounded visual chat requires an image
- 300 XV and a user instruction I as the input, and output a cap-
- 301 tion T with some bounding boxes b. Each bounding box is
- 302 corresponding to a phrase.
- 303 Evaluate chat scores: We mainly evaluate two aspects
- 304 in our benchmark, which are chat scores and grounded re-
- 305 sponse scores. We demonstrate the evaluation of Grounding-
- 306 Bench in Algorithm 13. The chat scores are similar to that

Object365, and the interactive encoder is initialized from a Semantic-SAM Tiny model pretrained on COCO with three granularities.

Algorithm 1 Pseudo code for Grounding-Bench Evaluation

# Inputs: Grounded response (T, b); GT response and boxes (T,ˆ bˆ). # Outputs: Chat scores (S); Grounded response scores (P, R, F). # Functions: GPT_chat_eval(), match_gt_box(), construct_gd_response(), GPT_gd_eval(), calc_scores().

In the first training stage, we freeze the language model and train the grounding model, prompt encoder, and projection layers with a learning rate of 1 × 10−4. For the second stage, we train the language model and projection layers with a learning rate of 2 × 10−5, while training the grounding model with a learning rate of 1 × 10−4 while freezing the CLIP vision encoder and the prompt encoder.

- 1 def GroundingBenchEval(T, b,T,ˆ bˆ):

- 2 S = GPT_chat_eval(T,Tˆ) # Request GPT-4 to score T
- 3 N=len(bˆ), N =len(b)
- 4 b = []
- 5 for b in b:

- 6 gt_box, iou = match_gt_box(b,bˆ)
- 7 if iou ≥ 0.5:

- 8 b =b +gt_box

- 9 T = construct_gd_response(T,b )# Insert matched gt boxes into the response to form responses of Context type 3 in Table 2
- 10 TP ,TP = GPT_gd_eval(T ) # GPT help evaluate semantic matching
- 11 P, R, F = calc_scores(TP ,TP,N) # Calculate grounded response scores
- 12 output = S, P, R, F
- 13 return output

#### 3.2. Grounding-Bench

To demonstrate the effectiveness of our method in Grounded Visual Chat (GVC), we compare our method with other strong LMMs that support visual grounding on our benchmark. As shown in Table 4, the results in grounded response scores are presented in two parts for each grid. The left one is evaluated on the 1000 images of our Grounding-Bench, and the right one is on the 30 images in LLaVA Bench (COCO). All the numbers for grounding LMMs are evaluated using their official prompt templates for grounding to ensure the best performance. The results show that our method outperforms all open-source methods in both grounded response scores and chat scores on grounded responses, except for CogVLM-Chat and LLaVA, which are chat models. GPT4V achieves the best performance on grounded detailed description with the help of SoM, but it is a combination of two models. Among open-source methods, GogVLM is second only to ours in terms of the F1 score for grounded detailed description, but it has the lowest GPT evaluated scores. Shikra’s chat scores are second only to ours. We also annotated 30 images in LLaVA Bench (COCO) as grounded detailed description and reported phrase grounding performance of our model and Shikra for reference.

of LLaVA Bench. Differently, we let the model output 307 grounded responses. We then process the output to remove 308 special tokens and boxes to obtain the pure-text response for 309 evaluation. 310 Evaluate grounded response scores: For grounded re- 311 sponse, we evaluate on grounded detailed description task. 312 We evaluate recall R = TPgt/Ngt for completeness, pre- 313 cision P = TPpred/Npred for hallucination and F1 = 314 2×P ×R/(P +R) to combine the two. R is the proportion 315 of entities correctly mentioned and grounded in the descrip- 316 tion and P is the proportion of correct predicted groundings. 317 A grounding is correct only when the box is matched to a 318 GT box with IoU larger than 0.5 and their semantic is cor- 319 rectly matched. We replace the predicted box with the most 320 matched box in grounded response as the Context type 3. We 321 then provide Context type 1 and 3 to GPT4 resort to decide 322 TPpred and TPgt as shown in the bottom block in Table 2. 323 In the example, Npred = 7 and Ngt = 12. According to the 324 response from GPT4, TPpred = 4 and TPgt = 3. Therefore, 325 we have P = 0.57,R = 0.25,F1 = 0.35. 326

the proportion of entities correctly mentioned and grounded in the description, while P assesses the proportion of correctly predicted groundings. A grounding is deemed correct only when the box matches a ground truth (GT) box with an IoU greater than 0.5, and their semantics are accurately matched. To determine TPpred and TPgt for GPT4, we provide Context types 1 and 3, as shown in the bottom block in Table 2. For example, in the provided example, Npred = 7 and Ngt = 12. Based on GPT4’s response, we calculate TPpred = 4 and TPgt = 3. Consequently, we obtain P = 0.57, R = 0.25, and F1 = 0.35.

### 3. Experiments

In this section, we will first introduce our experimental settings. Then, we will compare our model with other state-ofthe-art models on our benchmark, Grounding-Bench. Next, we will evaluate our model against other grounding models on challenging Referring Expression Comprehension (REC) and Referring Expression Segmentation (RES) tasks on RefCOCO, RefCOCO+, and RefCOCOg. The results will demonstrate that our model outperforms other grounding LLMs with the same number of parameters on both REC and RES tasks, and ours is the only model capable of handling both REC and RES effectively. Afterward, we will conduct an evaluation of the support for various types of visual prompts. Finally, we will perform ablation studies on our modeling and data creation processes to validate our method.

#### 3.3. Traditional Grounding Benchmarks

3. Experiments 327 In this section, we first introduce our experimental settings. 328 Then, we compare our model with other state-of-the-art mod- 329 els on Grounding-Bench. Next, we compare our model with 330 other grounding models on challenging Referring Expression 331 Comprehension (REC) and Referring Expression Segmen- 332 tation (RES) tasks on Refcoco, Refcoco+ and Refcoco-g. 333 The results demonstrate that our model outperforms other 334 grounding LMMs with the same number of parameters on 335 both REC and RES and ours is the only one that can do both 336 REC and RES. Later, we do evaluation on the supporting 337 of various types of visual prompts. Finally, we do ablation 338

We also evaluate our model on classic grounding benchmarks, including RefCOCO/+/g for Referring Expression Comprehension (REC) and Referring Expression Segmentation (RES), and Flickr30K Entities for Phrase Grounding. For this experiment, we use the 7B language model with the grounding model using the Swin-Tiny backbone. Our model is trained for the first stage with RefCOCO/+/g, Visual Genome, and Flickr30K Entities. Our model stands out as the only LMM that can excel in both REC and RES tasks. On the REC task, our model outperforms all LMMs, except for CogVLM-Grounding, which utilizes a 4B vision model and a 6B connection module. On RES and Phrase grounding tasks, our model surpasses all LMMs. One advantage of our model is its ability to be trained on both box and mask data, allowing us to leverage Visual Genome to enhance our RES performance.

#### 3.1. Experimental Settings

To facilitate result reproduction, we provide detailed settings. Our language model is initialized from a pretrained Vicuna7b v1.3, the grounding model is initialized from the vision part of an OpenSeeD Tiny model pretrained on COCO and

5

#Vision Grounded Response Scores Chat Scores Phrase Model params(B) Recall Precision F1 Detail desc. Conv. Reasoning All grounding LLaVA [18] 0.30 - - - 69.1 82.0 92.6 81.2 Bubo-GPT [47] 2.00 26.2|25.7 37.2|31.3 30.7|28.2 65.0 75.9 93.4 78.2 Shikra [3] 0.30 21.1|21.6 39.8|38.4 27.6|27.7 64.7 75.4 86.4 75.5 64.29 Shikra∗ 0.30 22.0|28.7 44.6|48.6 29.4|36.1 41.8 - - - miniGPT v2 [2] 1.00 20.6|25.3 33.6|39.1 25.6|30.7 48.0 51.0 38.7 45.8 CogVLM-Grounding [33] 10.0 22.3|27.5 56.3|62.5 32.0|38.2 35.8 47.8 22.2 34.9 CogVLM-Chat 10.0 - - - 73.1 86.9 92.1 84.2 GPT4-V+SoM [26, 35] - − − |55.1 − − |73.5 − − |63.2 67.3 104.3 108.4 93.3 LLaVA-G (Ours) 0.35 28.6|36.3 52.7|53.4 37.1|43.2 67.2 78.7 91.1 79.3 81.6

- Table 4. A comparison on our Grounding-Bench. For each model, we use the prompt template recommended by the paper. The results in grounded response scores are two parts in each grid where the left one is evaluated on the 1000 images of our Grounding-Bench and the right one is on the 30 images in LLaVA Bench (COCO). ∗ denotes Shikra with a special prompt for grounded description recommended by the paper. We make GPT4-V+SoM grey because it uses external model to label marks.

Models

RefCOCO RefCOCO+ RefCOCOg Flickr30k Entities

REC RES REC RES REC RES ACC@0.5 mIoU cIoU ACC@0.5 mIoU cIoU ACC@0.5 mIoU cIoU val test

ReLA [16] – – 73.80 – – 66.00 – – 65.00 – – PolyFormer-L[19] – 76.94 75.96 – 72.15 69.33 – 71.15 69.20 – – UniTAB [36] 86.32 – – 78.70 – – 79.96 – – 78.76 79.58 MDETR [7] 86.75 – – 79.52 – – 81.64 – – 82.3 83.8 GLIP-T∗ [14] 50.42 – – 49.50 – – 66.09 – – – – GDINO-T [21] 89.19 – – 81.09 – – 84.15 – – – –

Kosmos-2∗ [28] 52.32 – – 45.48 – – 60.57 – – 77.80 78.70 LISA-7B [3] – – 74.9 – – 65.1 – – 67.9 – – MiniGPT v2-7B [3] 88.06 – – 79.58 – – 84.19 – – – – Shikra-7B [3] 87.01 – – 81.60 – – 82.27 – – 75.84 76.54 Ferret-7B [40] 87.49 – – 80.78 – – 83.93 – – 80.39 82.21 CogVLM-Grounding-17B [33] 93.40 – – 87.76 – – 93.02 – – – –

LLaVA-G-7B (Ours) 89.16 79.68 77.13 81.68 72.92 68.79 84.82 74.39 71.54 83.03 83.62

- Table 5. Performance comparison on the referring expression comprehension (REC) referring expression segmentation (RES) and phrase grounding tasks. We mark the best results with bold. ∗ denotes the zero-shot results are reported. Since CogVLM-Grounding is a larger model with 4B vision model and 6B connection module, we make it grey.

Model Ground type α Mark Size Mark val test Ours - - - 83.0 83.6 Ours Mark 0.4 / 0.4 20 72.1 73.7 Ours Mark 0.4 / 0.2 30 75.1 75.4 Ours Mark 0.2 / 0.2 30 76.6 77.9

- Table 6. The top1 accuracy of phrase grounding on Flickr30K. The first row is our original pipeline with grounding model to do phrase grounding.

LLava Shikra GPT4ROI PVIT Ours-T click Ours-T box 40 53.9 64 64.5 70.8 71.5

- Table 7. The comparison on COCO object classification accuracy. Numbers except for our method are from PVIT [1] paper. They evaluate llava by cropping the regions of GT boxes.

the masks. The table presents results for different α values of masks and sizes of marks. We also include the performance of Shikra-7B and Ours-7B with the grounding model for comparison. The results clearly indicate that our model effectively supports the use of marks for grounding.

Support for clicks and boxes as visual prompts. To further demonstrate our support for visual prompts, we compare object classification accuracy on COCO val2017 in Table 7. We report our performances achieved using clicks and boxes as visual prompts. Additionally, we include numbers from the PVIT paper for reference. The results illustrate that our referring accuracy surpasses that of previous works, with boxes yielding better accuracy than clicks.

#### 3.4. Visual Prompts

We demonstrate our support for various types of visual prompts, including marks, clicks, and boxes.

Support for marks as visual prompts. In Table 6, we showcase our phrase grounding performance on Flikr30K when using marks as visual prompts. It’s important to note that for this experiment, we overlay GT masks and marks onto the input image, positioning the marks at the centers of

#### 3.5. Visualizations

Compare with other models. We provide visual comparisons between our model and other grounding LMMs, including CogVLM-Grounding, Shikra, and MiniGPTv2, in Table 8, 20, and 21. These visualizations highlight that our model produces longer descriptions, detects more objects, and maintains high precision. Furthermore, our model demonstrates the ability to ground open-set concepts, such as "dragons," which do not exist in our grounded visual chat

###### Grounded description example 1:

User Describe the following image with grounding. LLaVA-G The image depicts a group of people sitting at a large, white table, engaged in activities. There are four people in the scene, with three of them sitting close to each other and one person sitting further to the right. They are all using laptop computers, likely working or collaborating on a project together. The table is well organized, featuring multiple laptops, a large TV screen on the left side, and a clock on the wall. There are also several books and a cell phone placed on the table, indicating a casual and comfortable environment. A chair is positioned near the left side of the table, and there are two bottles placed nearby, possibly containing drinks for the people working at the table.

[Figure 12]

User Can you provide a description of the image and include the coordinates [[x0,y0,x1,y1]] for each mentioned object? CogVLM-Grounding A group of people are sitting at a table with laptops.

[Figure 13]

User Provide a comprehensive description of the image <image> and specify the positions of any mentioned objects in square brackets. Shikra A group of people are in a meeting , some with laptops, some with laptops, and

[Figure 14]

some without .

User [grounding] Describe the scene in detail. MiniGPT v2 A group of people are sitting around a large table in a meeting room , with

[Figure 15]

laptops and phones on the table

- Table 8. An example of grounded detailed descriptions generated by LLaVA-G and other LMMs. We use the best prompt recommended by the paper for each model.

data. These visual comparisons underscore the strengths and capabilities of our model in the context of grounded visual chat.

##### Other functionalities.

- 1. Visual Chat with Visual Prompts (Table 9): Our model demonstrates its capability to engage in visual chat using visual prompts such as boxes or clicks.
- 2. Support for Marks as Visual Prompts (Table 10): After fine-tuning with mark data, our model effectively supports marks as visual prompts.
- 3. Debugging and Hallucination Removal (Table 11): Our model exhibits a potential for assisting in debugging by allowing users to identify and correct errors. In this exam-

ple, the initial output contains hallucination ("two small birds" grounded to the woman’s mouth), which the user detects through grounding and subsequently instructs the model to rectify.

#### 3.6. Ablation Studies

In this section, we provide insights into our visual chat capability and the design of the grounding model through various ablation studies.

Maintaining visual chat capability. We demonstrate that our model retains strong visual chat capabilities by comparing it with LLaVA on LLaVA Bench (Table 12). The results indicate that our model’s visual chat performance is

###### Visual prompt example:

User What is the object ⟨obj⟩ doing and why? LLaVA-G The cat is sitting in front of a laptop computer, likely curious about the content displayed on the screen. Cats are naturally inquisitive animals, and they are often attracted to anything that seems new, interesting, or engaging. The movement or visual elements on the laptop screen might have caught the cat’s attention, prompting it to sit and observe the screen. It could be merely a passing curiosity or the start of an interest in technology or screen time that the cat may develop further in the future.

[Figure 16]

User What is the object ⟨obj⟩ used for? LLaVA-G The laptop computer screen is used for various purposes, such as working, studying, entertainment, and communication. In this particular image, the laptop appears to have a Word document or a web page open, which suggests that the user might be working on a document, browsing the internet, or using the computer for other tasks.

[Figure 17]

- Table 9. An example of visual chat based on visual prompts where ⟨obj⟩ denotes the positions in the user instruction to insert the visual prompt features.

Example of marks as prompts: User What do you think is going on in this snapshot with grounding?

[Figure 18]

LLaVA-G In the snapshot, a little girl <2> is sitting on a rug with a guitar-shaped toy <1> in her hands. She appears to be playing or interacting with the toy guitar, which suggests that she might be interested in music or is simply having fun with her toy. This scene could indicate that the child is exposed to musical instruments or enjoys imitating the sounds she hears. It is also possible that the toy guitar is a part of her imaginative play, where she creates different scenarios and engages with her surroundings.

Table 10. An example of visual chat based on marks.

comparable to LLaVA, whether responding with or without grounding.

Number of queries in grounding model. Table 13 presents our model’s performance on Referring Expression Comprehension (REC) and Referring Expression Segmentation (RES) tasks with different numbers of queries. The results reveal that using 50 queries is sufficient for both tasks and achieves optimal performance. This finding highlights the efficiency of our approach in handling these tasks.

Detaching the grounding model. We investigate the impact of detaching the grounding model on both chat and grounding performance. Detaching the grounding model means stopping gradients from propagating from the grounding model to the Language Model (LLM). Table 14 compares the detached model’s performance with the original model. The results demonstrate that detaching the grounding model

leads to slightly improved chat performance but significantly compromises the grounding performance. This indicates the importance of the grounding model in maintaining highquality visual chat with grounding capabilities.

### 4. Related Work

#### 4.1. Large Multi-modal Models

With the recent surge in Large Language Models (LLMs), researchers have been actively exploring ways to extend the capabilities of these models beyond text to encompass other modalities, resulting in the development of several Large Multi-modal Models (LMMs). Notably, projects like LLaVA [12] and MiniGPT-4 [49] have undertaken the task of integrating visual instruction tuning data into their LMMs. They achieved this by utilizing GPT-4 or employing hand-

###### Example of reduce hallucination:

User What do you see happening in this image with grounding? LLaVA-G The image features a young woman standing in a field, surrounded by tall grass. She is holding a string or a ribbon in her hand, which seems to be attached to a bird . The bird is flying in the sky, with its position slightly to the right and above the woman. In addition to the woman and the bird, there are two small birds in the scene. One is located in the middle of the scene, close to the woman, while the other is slightly above the first bird. The scene has a peaceful, nature-oriented atmosphere.

[Figure 19]

User There are actually no small birds, so please describe the scene again with grounding.

[Figure 20]

LLaVA-G In the image, there is a young woman standing in a field of tall grass. She holds a string or a ribbon, which is connected to a bird . The bird is flying in the sky, towards the right and slightly above the woman. The scene portrays a sense of wonder and tranquility as the woman interacts with the bird and enjoys the company of nature.

Table 11. An example to show the potential of LLaVA-G to reduce hallucination.

LLaVA-Bench (COCO) LLaVA-Bench (In-the-Wild) GVC Conv. Detail Reasoning All Conv. Detail Reasoning All

LLaVA 82.0 69.1 92.6 81.2 42.6 51.9 68.9 57.1 LLaVA-G Yes 74.8 68.5 95.3 79.7 38.5 40.1 75.1 55.8 LLaVA-G No 79.3 71.2 92.8 81.2 47.7 44.6 70.0 57.2

- Table 12. A comparison on LLaVA-Bench. “GVC" is “No" means it outputs pure-text response without grounding.

RefCOCO RefCOCO+ RefCOCOg

#Q ACC cIoU mIoU ACC cIoU mIoU ACC cIoU mIoU 50 86.71 74.77 77.6 77.91 64.97 69.68 82.37 68.46 72.43 100 86.58 74.70 77.40 77.23 64.08 69.02 81.99 68.02 72.06 300 86.35 74.26 77.19 77.78 64.68 69.54 81.92 67.89 71.85

- Table 13. The comparison of performance when using different number of queries in the grounding model. “#Q" denotes the number of queries.

Detach Grounded detail description Chat scores Model GD Recall Precision F1 Detail desc. Conv. Reasoning All

Ours ✓ 25.1 58.2 35.1 61.6 86.3 94.9 81.2 Ours 36.3 53.4 43.2 67.2 78.7 91.1 79.3

- Table 14. Ablations on our benchmark. “Detach GD" means stop gradient from the grounding model to language model.

designed prompts, thereby enhancing the LMMs’ ability to follow instructions effectively.

In addition to these, there exist other noteworthy works in the field, including mPLUG-DocOwl [39], Otter [11], LLaMa-Adaptor [45], and InternGPT [22]. These projects have also contributed significantly to the advancement of LMMs by incorporating various techniques and methodologies.

Moreover, researchers have delved into the realm of finegrained understanding of LMMs, as exemplified by works like VisionLLM [32], GPT4RoI [46], and PVIT [1]. VisionLLM, for instance, employs a language-guided tokenizer

to extract vision features at specific granularities, whereas GPT4RoI and PVIT utilize bounding boxes to obtain relevant visual features.

#### 4.2. Visual Grounding Models

The visual grounding task [4, 7, 19, 23, 24, 34, 48] aims to pinpoint the location of objects within an image based on textual input. This challenge is fundamental in multimodal perception and has promising applications. It requires a deep understanding of both the image and the text, along with establishing correspondences between image regions and textual descriptions.

The GLIP model [13] takes a significant step in this direction by integrating various data formats, including detection and referring data. It demonstrates that grounded pretraining effectively enhances the localization capabilities of grounding models. Building upon GLIP, GLIPv2 [43] takes a further stride by unifying grounding and Visual-Language (VL) understanding tasks. Grounding-DINO [21], which leverages grounded pretraining and the DINO [42] detector, stands out for its superior performance in this domain.

In recent years, vision-and-language models have gained increasing attention in tasks related to visual recognition and perception. Models like CLIP [30] and ALIGN [6], through contrastive learning on large-scale image-text pair datasets at the image level, have achieved generalized and robust capabilities in image classification. Simultaneously, in more fine-grained recognition tasks like visual grounding [4, 5, 7, 19, 23, 24, 34, 48], which aims to locate specific regions based on textual inputs, researchers are exploring the

potential of conducting image and text contrastive learning at the region level.

Approaches such as MDETR [7], DetCLIP [37], DetCLIPv2 [38], GLIP [13], GLIPv2 [43], and GroundingDINO [21] strive to detect arbitrary categories by training with large-scale region-text data. For instance, MDETR [7] was trained on existing multimodal datasets with explicit alignment between text phrases and image objects, employing an end-to-end framework.

GLIP [13] advances this approach by re-formulating object detection as a grounding task and incorporating additional grounding data to perform grounded pretraining, enhancing semantic alignment between phrases and regions. GLIPv2 further demonstrates how grounded pretraining can improve VL understanding, leading to a unified model for localization and VL understanding.

Moreover, Grounding-DINO [21], by incorporating grounded pretraining with the DINO [42] detector, excels in this field. These advancements in vision-and-language models, particularly through contrastive learning on largescale text-region data, represent significant progress in finegrained recognition tasks, resulting in more precise and contextually aware visual understanding.

#### 4.3. Grounding Large Multi-modal Models

Based on their architectural characteristics and functionalities, Grounding LMMs can be classified into three distinct categories.

The first category involves models that predict box coordinates in text format. Notable models in this category include Kosmos-2 [28], Shikra [3], MiniGPT v2 [49], Ferret [40], and CogVLM [33]. For instance, Kosmos-2 introduced a comprehensive grounding caption dataset and trained a model with strong grounding capabilities, showcasing impressive zero-shot performance across various grounding benchmarks. Shikra, on the other hand, focused on building referral dialog data and training their model to support referral dialog with boxes as both input and output. MiniGPT v2 employed task tokens to activate different task-specific capabilities, including support for grounded output with boxes. Meanwhile, CogVLM leveraged a 10-billion parameter vision model to achieve state-of-the-art performance in various vision-language tasks, including grounding. It’s worth noting that many of these methods trained on low-quality grounding caption data, despite achieving significant progress in visual grounding. For instance, Shikra’s referential dialog data, although valuable, is relatively small, consisting of only 5,000 images.

The second category involves models that employ a separate grounding model for grounded chat, exemplified by BuboGPT [47] and LLaVA-PLUS [20]. However, these models often face performance limitations at the language encoder of the grounding model.

The third category adopts an approach where the output of a language model is fed into a grounding model to decode masks and boxes. LISA [10] is a representative model in this category, with a primary focus on various segmentation tasks rather than chat interactions.

In many previous works, there has been a trade-off between grounding and chat abilities, with data and evaluation metrics typically emphasizing one of these aspects. In contrast, our dataset and benchmark prioritize assessing the compositional abilities of both grounding and chat interactions, providing a unique perspective in this field.

### 5. Conclusion

This paper introduced LLaVA-Grounding, an AI assistant that combines visual chat and grounding capabilities. We began by creating a grounded visual chat dataset using a novel data creation pipeline. Subsequently, we proposed an end-to-end model architecture that integrates a grounding model with a Language Model (LM) for effective grounding. Additionally, we introduced Grounding-Bench as a comprehensive benchmark for evaluating grounded visual chat performance, covering both chat and grounding aspects. Our experiments demonstrated that LLaVA-Grounding consistently outperforms other open-source LM models in both chat and grounding tasks, showcasing its effectiveness. Furthermore, LLaVA-Grounding excelled in traditional grounding benchmarks, highlighting its versatility. However, we acknowledge that LLaVA-Grounding has limitations in terms of semantic scope, and future work could explore extending the dataset and data labeling methods to open-vocabulary settings.

### References

- [1] Chi Chen, Ruoyu Qin, Fuwen Luo, Xiaoyue Mi, Peng Li, Maosong Sun, and Yang Liu. Position-enhanced visual instruction tuning for multimodal large language models, 2023. 7, 10
- [2] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigptv2: Large language model as a unified interface for visionlanguage multi-task learning. arXiv:2310.09478, 2023. 1, 2, 7
- [3] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195,

2023. 1, 2, 7, 11

- [4] Jiajun Deng, Zhengyuan Yang, Tianlang Chen, Wengang Zhou, and Houqiang Li. Transvg: End-to-end visual grounding with transformers, 2022. 10
- [5] Shijia Huang, Feng Li, Hao Zhang, Shilong Liu, Lei Zhang, and Liwei Wang. A unified mutual supervision framework for referring expression segmentation and generation, 2022. 10

- [6] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. arXiv preprint arXiv:2102.05918, 2021. 10
- [7] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr – modulated detection for end-to-end multi-modal understanding, 2021. 7, 10, 11
- [8] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 1, 5
- [9] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Fei-Fei Li. Visual genome: Connecting language and vision using crowdsourced dense image annotations, 2016. 5
- [10] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692,

2023. 1, 11

- [11] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint

- arXiv:2305.03726, 2023. 10

[12] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-andvision assistant for biomedicine in one day. arXiv preprint

- arXiv:2306.00890, 2023. 2, 9

- [13] Liunian Harold Li*, Pengchuan Zhang*, Haotian Zhang*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In CVPR, 2022. 10, 11
- [14] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded languageimage pre-training. In CVPR, 2022. 7
- [15] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014. 2, 5
- [16] Chang Liu, Henghui Ding, and Xudong Jiang. Gres: Generalized referring expression segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23592–23601, 2023. 7
- [17] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 2
- [18] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 1, 2, 3, 5, 7

- [19] Jiang Liu, Hui Ding, Zhaowei Cai, Yuting Zhang, Ravi Kumar Satzoda, Vijay Mahadevan, and R. Manmatha. Polyformer: Referring image segmentation as sequential polygon generation, 2023. 1, 7, 10

- [20] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang, Jianfeng Gao, and Chunyuan Li. Llava-plus: Learning to use tools for creating multimodal agents, 2023. 11
- [21] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 1, 7, 10, 11
- [22] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, Kunchang Li, Zhe Chen, Xue Yang, Xizhou Zhu, Yali Wang, Limin Wang, Ping Luo, Jifeng Dai, and Yu Qiao. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language, 2023. 10
- [23] Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng, and Rongrong Ji. Multi-task collaborative network for joint referring expression comprehension and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 10
- [24] Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Yongjian Wu, Yue Gao, and Rongrong Ji. Towards language-guided visual recognition via dynamic convolutions. International Journal of Computer Vision, pages 1–19, 2023. 10
- [25] OpenAI. Gpt-4 technical report, 2023. 1, 2
- [26] OpenAI. Gpt-4v(ision) system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf,

2023. 7

- [27] OpenAI. Gpt-4 technical report, 2023. 2
- [28] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 2, 7, 11
- [29] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV, 2015. 1, 5
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021. 10
- [31] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [32] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, and Jifeng Dai. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks, 2023. 10
- [33] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming

- Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. 2023. 1, 2, 7, 11
- [34] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Zehuan Yuan, Ping Luo, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In CVPR, 2023. 10
- [35] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v, 2023. 5, 7
- [36] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Faisal Ahmed, Zicheng Liu, Yumao Lu, and Lijuan Wang. Unitab: Unifying text and box outputs for grounded visionlanguage modeling, 2022. 7
- [37] Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing Xu, and Hang Xu. Detclip: Dictionary-enriched visual-concept paralleled pretraining for open-world detection, 2022. 11
- [38] Lewei Yao, Jianhua Han, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, and Hang Xu. Detclipv2: Scalable openvocabulary object detection pre-training via word-region alignment, 2023. 11
- [39] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499, 2023. 10
- [40] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity, 2023. 1, 2, 7, 11
- [41] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions, 2016. 5
- [42] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection, 2022. 10, 11
- [43] Haotian* Zhang, Pengchuan* Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. arXiv preprint arXiv:2206.05836, 2022. 10, 11
- [44] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianfeng Gao, Jianwei Yang, and Lei Zhang. A simple framework for open-vocabulary segmentation and detection. arXiv preprint arXiv:2303.08131, 2023. 2
- [45] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention, 2023. 10
- [46] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023. 2, 10
- [47] Yang Zhao, Zhijie Lin, Daquan Zhou, Zilong Huang, Jiashi Feng, and Bingyi Kang. Bubogpt: Enabling visual grounding in multi-modal llms. arXiv preprint arXiv:2307.08581, 2023. 1, 7, 11

- [48] Chaoyang Zhu, Yiyi Zhou, Yunhang Shen, Gen Luo, Xingjia Pan, Mingbao Lin, Chao Chen, Liujuan Cao, Xiaoshuai Sun, and Rongrong Ji. Seqtr: A simple yet universal network for visual grounding. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXV, pages 598–615. Springer, 2022. 10
- [49] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1, 2, 9, 11

## LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models Supplementary Material

stage1 stage2

The man is sitting in a sofa on top of a green car.

|[Figure 21]<br><br>[Figure 22]|
|---|

lrllm 0.0 2e − 5 lrgd 1e − 4 1e − 4 bslang 32 128 bsgd 32 64 warm up ratio 0.03 0.03 weight decay 0.0 0.0 bf16 ✓ ✓ tf32 ✓ ✓ grad accumulate 2 1 nsteps 10000 8000

[Figure 23]

Language Model

[Figure 24]

What is the object <obj> doing

[Figure 25]

𝑃𝑟𝑜𝑗 Vision Enc

𝑃𝑟𝑜𝑗

[Figure 26]

🔥

[Figure 27]

🔥

Prompt Enc

Figure 4. Network architecture of our LLaVA-Grounding for supporting visual prompts. Snow flake denotes the part is frozen and fire denotes the part is trainable.

Table 15. The hyper-parmeters for stage1 and stage2.

### A. More details about visual prompt

LLaVA instruction tuning data to visual prompt data. The data has high-quality visual chat because it is generated by GPT4.

We support visual prompts such as clicks, boxes and scribbles with low training cost by only train the visual prompt encoder to align prompt features to the language space.

However, it is not good at distinguishing different instances in the image because it usually talks about a few main objects for each image. For many images, the data only contain discussions about one instance. This is not good for the model to distinguish between different instances. Therefore, we include RefCOCO to train visual prompt encoder for distinguishing instances. RefCOCO has several instance, text pairs denoted as (I,T) for each image. We convert them as visual chat data with the following template: Q:Please describe the object ⟨obj⟩ briefly. A: T where T is the referring text.

Training. We add the support of visual prompt seamlessly to the trained grounded visual chat model. We use a pretrained Semantic-SAM model as the prompt encoder. As shown in Fig. 4, a visual prompt will be encode as a prompt feature by prompt encoder and map to the language space with a projection layer. There is a special token ⟨obj⟩ in the text instruction, whose input embedding will be replaced by the visual prompt feature. To avoid any influence on the existing model, we propose to fix the other part of the model and only tune the prompt encoder and the projection layer of the prompt encoder.

### B. Implementation details

We provide more details of our experiment configuration for reproducing our method. We provide hyper-parameters for both stage1 and stage2 in Table 15.

Data. As demonstrated in Section 2.2, we created 150K grounded visual chat data with GPT4 to match none phrases in answers with GT instances. The data for visual prompts is annotated in the same way as the grounded visual chat data. As shown in the first blocks of Table 2, we simply change the sentence in Context 2 to questions and GPT4 can help to match entities in the question with GT instances. Then, we can sample various types of visual prompts based on the GT instances. An example data is as follows. Assume the input image is that in Fig. 4 and the original question is Q: What is the man doing?. With the help of GPT4, “the man" is matched to the red box in the image. Then we change the text prompt to Q: What is the object ⟨obj⟩ doing?. ⟨obj⟩ is a place holder. Its input embedding will be replaced by the prompt embedding of “the man". We labeled a part of 150K

### C. Instruction lists for different data format.

In this section, we give instruction lists used to construct instruction following data from different formats of data.

### D. More visualizations

Table 20 and 21 shows more comparison of LLaVA-G and other grounding LMMs in grounded description.

- • "Please segment ⟨phrase⟩."
- • "Can you segment ⟨phrase⟩?"
- • "Please provide the boxes and masks for ⟨phrase⟩."
- • "We need the boxes and masks for ⟨phrase⟩, please."
- • "Ensure that the boxes and masks for ⟨phrase⟩ are provided."
- • "Include the boxes and masks for ⟨phrase⟩ in your submission."
- • "Don’t forget to attach the boxes and masks for ⟨phrase⟩."
- • "It’s important to have the boxes and masks for ⟨phrase⟩."
- • "Please remember to include ⟨phrase⟩’s boxes and masks."
- • "The request is for the boxes and masks related to ⟨phrase⟩."
- • "Kindly submit the boxes and masks corresponding to ⟨phrase⟩."

Table 16. The list of instruction templates for referring expression tasks. ⟨phrase⟩ denotes the referring expression.

- • "Describe the image concisely."
- • "Provide a brief description of the given image."
- • "Offer a succinct explanation of the picture presented."
- • "Summarize the visual content of the image."
- • "Give a short and clear explanation of the subsequent image."
- • "Share a concise interpretation of the image provided."
- • "Present a compact description of the photo’s key features."
- • "Relay a brief, clear account of the picture shown."
- • "Render a clear and concise summary of the photo."
- • "Write a terse but informative summary of the picture."
- • "Create a compact narrative representing the image presented."

Table 17. The list of instructions for brief image description.

- • "with grounding"
- • "with boxes and masks"
- • "Please also provide the boxes and masks for the noun phrases in the response."
- • "Kindly ensure that the response includes the relevant boxes and masks for each noun phrase."
- • "Additionally, include the boxes and masks that match each noun phrase in the response."
- • "Please provide the boxes and masks that correspond to every noun phrase in your response."
- • "It’s important to have the boxes and masks that align with each noun phrase in the response."
- • "Make sure to include the appropriate boxes and masks for each noun phrase in your response."
- • "In your response, include the boxes and masks that pertain to each noun phrase."
- • "Also, supply the boxes and masks that are linked to each noun phrase in the response."
- • "Additionally, please furnish the boxes and masks that correspond to each noun phrase in the response."
- • "Don’t forget to provide the boxes and masks associated with each noun phrase in your response."
- • "Ensure that each noun phrase in the response has its respective boxes and masks."

Table 18. The list of suffixes of the instructions for grounding caption.

- • "Could you offer a concise description of the object ⟨obj⟩ ?"
- • "Please give a short summary of the object ⟨obj⟩ .",
- • "Kindly provide a succinct description of the object ⟨obj⟩ ."
- • "I’d appreciate a brief overview of the object ⟨obj⟩ ."
- • "Could you summarize the object ⟨obj⟩ in a few words?"
- • "Please provide a brief explanation of the object ⟨obj⟩ ."
- • "I’d like a quick description of the object ⟨obj⟩ , please."
- • "Could you quickly describe the object ⟨obj⟩ ?"
- • "Please offer a condensed description of the object ⟨obj⟩ ."
- • "Can you briefly characterize the object ⟨obj⟩ ?"
- • "Kindly provide a brief description of the object ⟨obj⟩ ."
- • "Would you mind giving a brief account of the object ⟨obj⟩ ?"
- • "Can you sketch a short description of the object ⟨obj⟩ ?"
- • "Please deliver a terse description of the object ⟨obj⟩ ."

Table 19. The list of instructions for region caption data.

[Figure 28]

LLaVA-G The image features a fantasy scene with a woman and a man standing close together, riding on the back of horses. The woman is on the left side of the image, and the man is on the right side. Both individuals appear to be enjoying their time on horseback, surrounded by a visually appealing fantasy setting. In the background, there are two dragons, one on the left side and the other on the right side of the image. The dragons are situated near the top, adding a sense of depth and intrigue to the scene.

User Can you provide a description of the image and include the coordinates [[x0,y0,x1,y1]] for each mentioned object? CogVLM-Grounding A man and a woman, both dressed in armor, are riding horses.

[Figure 29]

User Provide a comprehensive description of the image <image> and specify the positions of any mentioned objects in square brackets. Shikra A woman with blond-hair sitting on a white horse with a man standing next to

[Figure 30]

her.

User [grounding] Describe the scene in detail. MiniGPT v2 A woman in blue is riding a white horse ,a man in black is walking towards the

[Figure 31]

horse , and a dragon is in the background

- Table 20. An example of grounded detailed descriptions generated by LLaVA-G and other LMMs. We use the best prompt recommended by the paper for each model.

[Figure 32]

LLaVA-G The image features a dining table with three pizzas on separate open boxes placed in the center. The pizzas have various toppings, making them visually appealing. In addition to the pizzas, there are two bowls on the table, one located near the top left corner and the other at the top right. The table appears to be set for a meal, with the pizzas and bowls ready for guests to enjoy.

User Can you provide a description of the image and include the coordinates

[Figure 33]

[[x0,y0,x1,y1]] for each mentioned object? CogVLM-Grounding Three pizzas in a box.

User Provide a comprehensive description of the image <image> and specify the positions of any mentioned objects in square brackets. Shikra Three pizzas in pizza boxes with various toppings.

[Figure 34]

User [grounding] Describe the scene in detail. MiniGPT v2 Three pizza boxes with pizzas and toppings on them

[Figure 35]

- Table 21. An example of grounded detailed descriptions generated by LLaVA-G and other LMMs. We use the best prompt recommended by the paper for each model.

