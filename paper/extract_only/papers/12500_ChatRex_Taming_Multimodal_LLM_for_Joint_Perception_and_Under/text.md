# arXiv:2411.18363v3[cs.CV]11Mar2025

## ChatRex: Taming Multimodal LLM for Joint Perception and Understanding

Qing Jiang1,2 , Gen Luo1 , Yuqin Yang1,2 , Yuda Xiong1 , Zhaoyang Zeng1 Yihao Chen1 , Tianhe Ren1 , Lei Zhang1,2† 1International Digital Economy Academy (IDEA) 2South China University of Technology

mountchicken@outlook.com , leizhang@idea.edu.cn

##### Abstract

Perception and understanding are two pillars of computer vision. While multimodal large language models (MLLM) have demonstrated remarkable visual understanding capabilities, they arguably lack accurate perception abilities, e.g. the stage-of-the-art model Qwen2-VL only achieves a 43.9 recall rate on the COCO dataset, limiting many tasks requiring the combination of perception and understanding. In this work, we aim to bridge this perception gap from both model designing and data development perspectives. We first introduce ChatRex, an MLLM with a decoupled perception design. Instead of having the LLM directly predict box coordinates, we feed the output boxes from a universal proposal network into the LLM, allowing it to output the corresponding box indices to represent its detection results, turning the regression task into a retrieval-based task that LLM handles more proficiently. From the data perspective, we build a fully automated data engine and construct the Rexverse-2M dataset which possesses multiple granularities to support the joint training of perception and understanding. After a three-stage training approach, ChatRex demonstrates strong perception and understanding performance, and the combination of these two capabilities also unlocks many attractive applications, demonstrating their complementary roles in MLLM. Code is available at https://github.com/IDEA-Research/ ChatRex.

##### 1. Introduction

Perception and understanding are two fundamental human faculties within behavioral science. Humans initially perceive objects, with vision signals transmitted to the brain for understanding, and can then locate back to the objects during conversation. In pursuit of AGI, Multimodal Large

This work was done when Qing Jiang, Gen Luo and were interns at IDEA.

† Corresponding author.

Language Models (MLLMs) [3, 4, 6, 13, 14, 16, 18, 36, 60, 64, 79, 85] have exhibited remarkable capacities for visual understanding empowered by advancements in Large Language Models (LLMs) [1, 2, 5, 15, 22, 78, 80, 83, 84]. Despite showing strong visual understanding, we find through experiments that these models generally lack fine-grained perception capabilities, particularly in object detection.

We evaluate the performance of several generalpurpose [14, 85] and detection-focused [10, 61, 94] MLLMs on the COCO [46] dataset by prompting them to detect objects within the image. The state-of-the-art model Qwen2VL-7B [85] only achieves a 43.9 recall rate at an IoU threshold of 0.5. The results indicate that MLLMs still struggle with fundamental perception tasks despite their remarkable visual understanding capabilities. This shortfall in perception constrains them in numerous tasks requiring precise perception, such as autonomous driving and robotic navigation. Also, it hinders their interactivity by identifying objects during conversation. We argue that this performance gap between perception and understanding in MLLMs arises primarily from two factors: i) modeling conflicts between these two tasks, and ii) lack of data that seamlessly balances both perception and understanding.

For object detection, a common practice is to quantize [12] box coordinates into tokens within the vocabulary of LLM to fit the auto-regressive framework. Although this ensures compatibility with understanding tasks through next-token prediction, we argue this method is in conflict with accurately modeling perception for three reasons: i) Error propagation: representing a single box typically requires 9 tokens including digits, square brackets, and commas, where an error in any token can cause cascading errors, which become even worse in multi-object detection. In subsequent experiments, we find that this is one of the reasons for the low recall rate; ii) Ambiguity in prediction order: there is no inherent order among objects in object perception, yet the auto-regressive nature imposes a sequential order that the LLM must decide which object to predict first and iii) Quantization range limitation: quantization er-

Grounded Counting

Referring

Grounding & Detection

Q: Please detect man with a green hat…

Q: Please detect bottle, knife fork …

[Figure 1]

Q: How many dogs are there? A: <g>bottle</g><o><obj12></o>… A: There are 20 <g>dogs</g><o><obj6>…</o> A: <g>man with a green hat</g><o><obj4>…</o>

[Figure 2]

[Figure 3]

[Figure 4]

###### Grounded Image Caption

Grounded Conversation

[Figure 5]

[Figure 6]

Q: Provide a short description of the image and

Q: Answer the question in grounded format.

ground the objects that are mentioned.

What is unusual about this image?

A: The unusual aspect of this image is that it

A: A <g>young woman</g><o><obj5></o> with

features <g>dog</g><o><obj3>..<o>s

<g>long brown hair</g><o><obj7></o> is lying on her back, wearing a <g> white t-shirt

and a <g>cupcake</g><o><obj0>…</o>s, which is not a common sight. Typically,

</g><o><obj11></o> and <g> blue jeans

images of dogs are not associated with food

</g><o><obj2></o>, with <g> her hand </g><o><obj1></o> gently touching <g> her

items like cupcakes. The combination of a dog and a cupcake in the same image is

head </g><o><obj7></o>

unexpected and creates a unique and

intriguing visual experience.

Region Relationships

Region Description

###### Region QA

Q: What is <obj0> and <obj1> hodling respectively?

[Figure 7]

Q: what might be the species of <obj0>

Q: Provide one-sentence descriptions for

And what might be their relationship.

<obj0> and <obj1>, respectively.

A: The species of <obj0> appears to be a dog, likely a breed with black and white fur, such as a

A: <obj0> is holding a toy gun, while <obj1> is holding a toy balloon. They are standing next to each other,

A: <g>A majestic lion statue, intricately carved and adorned with a crown, stands proudly on a pedestal, exuding strength and regality</g><o><obj0></o>

Border Collie or a similar herding dog.

which suggests they might be companions or partners.

[Figure 8]

[Figure 9]

<g>A majestic bronze statue of a mounted figure, adorned with a crown and holding a

sword, stands proudly on a pedestal,

surrounded by a group of people</g><o><obj1></o>

Figure 1. Overview of the perception capabilities in ChatRex. We utilize a decoupled design for perception and understanding, allowing ChatRex to respond to questions while simultaneously grounding its answers to the referenced objects.

ror easily occurs when the image size is large.

vision encoder design to incorporate additional vision encoder [56] to provide high-resolution visual information for perception. For the second, we introduce a Universal Proposal Network (UPN), which leverages granularity-based prompt learning on a pre-trained open-set object detection model. This enables the generation of proposals that cover diverse granularities, categories, and domains, thereby ensuring robust box inputs for the LLM.

To address these inherent modeling conflicts, we adopt a decoupled model design and introduce ChatRex. For multimodal understanding tasks like image caption and image QA, we retain the auto-regressive text prediction framework. However, for perception, particularly object detection, we transform the task as a retrieval-based task inspired by Groma [61]. Specifically, instead of prompting the LLM to predict bounding box coordinates, the boxes are directly provided as inputs, each represented as an object token by combining its RoI feature with its positional embedding. When the LLM needs to reference an object, it outputs the index of the relevant box. This method represents each box as a single token without quantization, with the sequence order determined by the input boxes, effectively addressing prior modeling conflicts.

From the data perspective, current MLLMs are also limited by the lack of data that effectively balances both perception and understanding. To address this limitation, we developed a fully automated data engine to construct the Rexverse-2M dataset, which comprises image-region-text annotation triplets at varying levels of granularity. The data engine is composed of three primary modules. The first module generates image captions for input images [14], while the second aligns referenced objects or phrases using a grounding model [71]. The third module [22] refines region descriptions at multiple granularities.

However, this retrieval-based approach presents two key challenges for achieving optimal performance: the need for high-resolution visual input and a robust object proposal model. To address the first challenge, we adopt a dual

Experimental results show that ChatRex achieves

coarse-grained proposal

strong performance in object detection tasks, including COCO[46], LVIS[25], and RefCOCO/+/g [30, 62, 95], while also demonstrating competitive performance on multimodal benchmarks. Our findings highlight that both perception and understanding are fundamental capabilities for multimodal models, and their integration leads to performance gain with expansion on the scope of real-world applications, as illustrated in Fig. 1. To summarize, our contributions are threefold:

[Figure 10]

coarse-grained

UPN

fine-grained proposal

fine-grained

[Figure 11]

Added Prompts

Detection Queries

- • We reveal the performance gap in the perception of MLLMs and introduce a decoupled model ChatRex and a universal proposal network (UPN) to address the modeling conflict between perception and understanding.
- • We develop an automated data engine to create Rexverse2M, a comprehensive dataset supporting both perception and understanding tasks for model training.
- • Experimental results demonstrate that ChatRex exhibits strong perception and multimodal understanding capabilities, highlighting that these two complementary abilities are both essential for MLLM.

Figure 2. Overview structure of the Universal Proposal Network (UPN). UPN is a DETR-based model capable of detecting any object at two granularities.

rection employs additional decoders for perception. For instance, LISA [34], GLaMM [69], LLaVA-Grounding [105], PerceptionGPT [66], and VisionLLMv2 [87] use auxiliary detection or segmentation models for perception tasks. Groma [61] initially proposed re-framing detection as a box retrieval task, and we follow this method in this work.

##### 3. ChatRex Architecture

##### 2. Related Work

ChatRex employs a design that decouples perception from understanding. For perception, we train a universal proposal network to detect arbitrary objects, supplying box inputs to the LLM. For understanding, we adopt the standard LLaVA [50] structure with a dual vision encoder to facilitate high-resolution image encoding. We introduce each part in the following sections.

###### 2.1. General MLLMs

Leveraging breakthroughs in large language models within natural language processing, Multimodal Large Language Models (MLLMs) [3, 4, 6, 13, 14, 16, 18, 36, 60, 64, 79, 85] have demonstrated robust visual understanding capabilities. LLaVA [50] pioneered the paradigm of visual instruction tuning, inspiring a wave of subsequent work. Research on general-purpose MLLMs encompasses various directions, including: i) exploring the use of high-resolution image inputs to enhance model perceptual abilities, with models like LLaVA-Next [51], SPHINX [48], Monkey [43], InternLM-XComposer2 [20], LLaVA-UHD [89], NVLM [17] employing image slicing methods, and others like LLaVA-HR [59], Mini-Gemini [41], Eagle [76], and MG-LLaVA [108] utilizing high-resolution vision encoders for additional vision encoding; ii) investigating diverse approaches for pre-training [45, 63, 91] and fine-tuning data [37, 82], and iii) extending to multi image [28, 37] or video tasks [44, 90].

###### 3.1. Universal Proposal Network (UPN)

To ensure that the LLM can accurately retrieve the correct box, it is essential that the input boxes comprehensively encompass all objects within an image. This requires a proposal model with two key properties: i) robust generalization ability to generate proposal boxes for any object in any scenario, and ii) the proposed boxes should be comprehensive, including both instance-level and part-level objects.

To meet these requirements, a straightforward approach is to aggregate multiple detection datasets, merge their categories, and treat all object classes as a single foreground category for training. However, this strategy is suboptimal due to inconsistencies in object definitions across different datasets. For example, while datasets such as COCO [46] and O365 [75] annotate objects at the instance level, SA1B [32] annotate objects at part-level. These discrepancies in labeling can introduce ambiguities that compromise training stability. To mitigate this issue, we adopt a dualgranularity prompt tuning strategy.

###### 2.2. Perception MLLMs

While generic multimodal models demonstrate strong image-level understanding, they still lack fine-grained perception capabilities. Inspired by Pix2seq [12], several works such as Kosmos-2 [65], Shikra [10], Ferret [94, 104], CogVLM [86], Griffon [100, 101] and other generalized MLLMs [14, 63, 85] have transformed box regression into a quantized coordinate prediction task suited for LLM nexttoken prediction. SoM [93] uses a set of marks to prompt GPT4V for visual grounding tasks. Another research di-

Specifically, we utilize T-Rex2 [29] as our base model. T-Rex2 is a DETR-based [8] model trained on vast data and exhibits strong generalization, making it a suitable pretrained model for detecting any objects in varied scenes.

[Figure 12]

Vision Token Object Token

2

5 1

Text Token Object Index Token

<G> </G> <O> <obj2> <obj5> </O> <obj1>

There are 2 dog in this image.

7

Grounding Start Token Grounding End Token

<G> </G>

Object Start Token Object End Token

<O> </O>

###### LLM

- Task1: Detection and Grounding Q: Please detect the yellow dog in this image. A:
- Task2: Region Understanding Q: Give me a brief description of A:
- Task3: Grounded Image Conversation Q: What are the dogs doing? A:

Text Projection

Image Projection Object Projection

<G> the yellow dog </G> <O> <obj2> </O>

Last Scale

<obj0> <obj1>

<obj5>

Multi Scale

Universal Proposal

<G> Black dog with a stick </G> <O> <obj5> </O>

CLIP ConvNext

Tokenizer

ROI Align

Low Res. High Res.

[Figure 13]

How many dogs are

The <G> dogs </G> <O> <obj2> <obj5> </O> are running on a <G> beach </G> <O> <obj7> </O> while carrying

there in this image.

Give the number and

<G> a stick </G> <O> <obj1> </O> together

detect all of them.

- Figure 3. Overview of the proposed ChatRex model architecture and the workflow for modeling the detection output of the LLM from coordinates prediction task to input box indices retrieval task.

The model outputs object queries Qdec that pass through an MLP to predict bounding boxes. The classification of these bounding boxes is achieved via a dot product between the queries and the prompt embeddings E:

along the channel dimension, producing the same number of the low-resolution token count.

Object Encoder. We encode each output box from the universal proposal network to object tokens and feed them to the LLM. Assume K input boxes {Bi}Ki=1 from the UPN, let FH denote the multi-scale visual features produced by the high-resolution encoder, for each box Bi, we extract its content feature Ci using multi-scale RoI Align [26]:

Scls = E · QTdec : RC×D × RD×N → RC×N (1)

Where C is the number of classes, N represents the number of detection queries (default is 900), and D is the channel dimension of outputted queries. We extend T-Rex2 by introducing two additional learnable prompts, Pfine and Pcoarse, concatenated into Pconcat to classify boxes into fine-grained or coarse-grained categories:

Ci = RoIAlign(FH,Bi) (3)

Since the RoI feature does not contain positional information which is essential for referring tasks, we enhance each object feature with a positional embedding to capture spatial context. We encode each box coordinate through a sin-cos position embedding layer and add to the RoI feature:

Scls = Pconcat · QTdec : R2×D × RD×N → R2×N (2)

For training, we utilize SA-1B as the fine-grained dataset and other detection datasets (such as COCO and O365) as coarse-grained inputs. This dual-granularity prompt design effectively resolves labeling ambiguities between datasets, allowing the proposal model to accurately capture and characterize objects across varying levels of detail.

Vi = Ci + PE(Bi) (4)

LLM. We use two separate MLP projectors to map visual and object tokens to the text space. We also add an index token to each object token to inform the LLM about the index of each object token, which will be described in Sec. 3.3. These tokens are then concatenated with the text token and fed into the LLM for the next-token prediction task. We use Qwen2.5-7B [92] as our default LLM.

###### 3.2. MLLM Architecture

Dual Vision Encoders. An accurate perception system typically requires high-resolution inputs. To equip ChatRex with sufficient perception capabilities, we adopt an additional high-resolution vision encoder for image encoding. As illustrated in Fig. 3, we use the ViT [21] from CLIP [67] for low-resolution image encoding and ConvNeXt [56] from LAION [73] for high-resolution image encoding. To reduce the number of vision tokens fed into the LLM, we first adjust the input resolutions for both vision encoders to ensure they generate the same number of tokens at the last scale. We then directly concatenate these two tokens

###### 3.3. Task Formulations

We formulate the task of leveraging LLM for detection as an index selection process over input boxes. To do so, we first extend the vocabulary of LLM by incorporating specialized tokens, including object index tokens <obj0>, <obj1>, ..., <objN>, where N denotes the maximum number of input boxes and is set to 100 in this work; grounding start

[Figure 14]

The image shows an individual lying

The image shows an individual lying prone on grassy ground, aiming a

prone on grassy ground, aiming a

bolt-action rifle with their right eye close to the sight. He is wearing a

bolt-action rifle with their right eye

close to the sight. He is wearing a military-style uniform with a steel

military-style uniform with a steel

helmet with netting…

helmet with netting…

[Figure 15]

InternVL2

Spacy

Grounding DINO 1.5

Task: Grounding

Task: Noun phrase extraction

Task: Image description

+ Noun Phrases

+ Noun Phrases to Box Mapping

+ Image Caption

Region Crop

InternVL2

###### LLAMA3

|[Figure 16]<br><br>bolt-action rifle &|
|---|

Task: Phrase-conditioned Region Description

Task: Region Filtering & Rewriting

|[Figure 17]<br><br>steel helmet &|
|---|

|[Figure 18]<br><br>military-style uniform &|
|---|

+ Detailed Region Description

+ Referring Style Region Description

{

{

‘bolt-action rifle’: a bolt-action rifle

‘bolt-action rifle’: ‘bolt-action rifle

with a wooden stock and metal barrel, positioned on the ground in an outdoor setting.

|[Figure 19]<br><br>grassy ground &|
|---|

with wooden stock and metal barrel’

}

}

- Figure 4. Overview of the ChatRex data engine. There are three main components, including the image captioning module, the grounding module, and the region captioning module.

token <g>; grounding end token </g>; object start token <o>; and object end token </o>.

We then adopt a standard two-stage training methodology following LLaVA [50], enabling the model to preserve its perception capabilities while progressively acquiring multimodal understanding and dialog skills.

LLM Input Format. The input token sequence for the LLM are structured as follows:

###### 4.1. RexVerse-2M Data Engine

<image>\n<obj1><roi>...<objN><roi>\nQuestion

where <image> represents the visual tokens from the vision encoder, and <roi> denotes the object feature associated with each corresponding bounding box. Each <roi> token is prefixed by its respective object index token.

Our objective is to construct a dataset that can be effectively utilized for both perception and understanding tasks. To achieve this, our data pipeline focuses on generating an annotation triplet comprising image descriptions, region descriptions, and bounding boxes. As shown in Fig. 4, the data engine is structured around three core modules: image captioning, object grounding, and region captioning.

Decoupled Task Formulations. The detection results produced by the LLM are structured using the following combination of noun phrases and box indices:

Image Collection. We started by collecting images from COYO700M [7] dataset through a series of filtering processes including removing images with small resolution and NSFW tags. We also train an image classifier to filter out low-content web images with plain white backgrounds. Finally, we selected two million images as the dataset images.

<g>noun phrases</g><o><objm>...<objn><o>

where <objm> and <objn> refer to specific object index tokens, identifying the start (m) and end (n) of the sequence of detected objects linked to the noun phrases. This structured format enables a precise mapping between noun phrases and their corresponding bounding box indices.

Image Caption. We use InternVL2-8B [14] to generate image caption for each image. This image caption will refer to the main objects in the image by their category name or descriptive phrases.

With this input-output schema, ChatRex can handle various tasks such as detection, grounding, region comprehension, and grounded dialogue, in addition to generating plain text responses, as illustrated in Fig. 3.

Phrase Grounding. We then utilize SpaCy to extract noun phrases from generated image captions. Depending on the caption, SpaCy may identify category names, such as soldier or descriptive phrases (at least 3 words per region) like military-style uniform. We will also filter out some abstract nouns that might not be an object

##### 4. Data and Training

To equip ChatRex with robust perception and understanding capabilities, we build RexVerse-2M dataset with two million annotated images, featuring multi-granularity annotations generated through a fully automatic data engine.

https://spacy.io/

like image, background etc. Subsequently, we employ Grounding DINO 1.5 [71] to ground the filtered noun phrases. This process ultimately produces boxes associated with their category names or short phrase descriptions.

Phrase-Conditioned Region Caption. To support the training for understanding tasks, it is essential to generate detailed descriptions for each region rather than relying solely on category names or short phrases, which often provide limited information. A straightforward approach might involve cropping each region and feeding it into an MLLM model for image captioning. However, this method is prone to hallucinations when the cropped regions are too small or contain parts of other objects. To reduce such inaccuracies, we implemented a phrase-conditioned image description strategy. Specifically, we leverage the InternVL2-8B model [14] to generate image captions that are conditioned on predefined phrases related to each region. By guiding the model with these phrases, we ensure that the generated descriptions are more accurate and context-relevant, significantly reducing the likelihood of hallucinations and enhancing the quality of the region-specific captions.

Region Caption Filtering and Rewriting. Lastly, we employ LLaMA3-8B [22] to verify whether the generated captions accurately align with their original category names or short phrases, filtering out any remaining hallucinated outputs. Once validated, we then prompt it to refine these detailed captions into more concise referring expressions, thereby enhancing training for referring tasks.

Rexverse-2M consists of 2.1 million images with captions, 10.2 million regions annotated with category labels, 2.5 million regions labeled with short phrases, 2.5 million regions with detailed descriptions, and 2.4 million regions with referring descriptions. Additionally, we use this data engine to annotate 776K grounded conversation data from the ALLaVA-4V-Instruct [9] dataset for instruction tuning. Specifically, the conversation responses are treated as image captions, which are then passed through the engine.

###### 4.2. Training

UPN Training. We utilize two types of datasets with bounding boxes to train our UPN: coarse-grained datasets including O365 [75], OpenImages [33], Bamboo [107], COCO [46], LVIS [25], HierText [57], CrowdHuman [74], SROIE [27] and EgoObjects [109]; and fine-grained datasets SA-1B [32]. All dataset categories are defined as either coarse-grained or fine-grained, reducing the task to a binary classification problem. Following T-Rex2, Hungarian matching is used to match predictions with ground truth. We employ L1 Loss and GIOU Loss for box predictions, along with sigmoid focal loss for classification.

ChatRex Training Tasks. We adopt three main tasks to train ChatRex including i) Grounding: where the model outputs the indices of corresponding objects based on a given

|Stage|Task<br><br>|# Samples|Datasets|
|---|---|---|---|
|Stage1<br><br>|Image Caption<br><br>|976K|ALLAVA-4V-Caption [9]|
|Stage2<br><br>|Grounding & Region Understanding|2.07M|COCO [46], O365 [75], LVIS [25], RefCOCO/+/g [30, 62, 95], Rexverse-2M<br><br>|
|Stage3|Grounding & Counting & Region Understanding & Grounded Conversation Conversation & QA<br><br>|3.8M|Rexverse-2M, COCO, O365, LVIS, RefCOCO/+/g [30, 62, 95], PACO [68] MVDP [47], Osprey [97], CrowdHuman [74], VCR [99] , ALLAVA-4V-Instruct [9] , LLAVA-1.5 [49], LLaVA-Onevision [37]|

Table 1. Training data and tasks for each stage.

category name, phrase, or referring expression. ii) Region Understanding: where, given region indices, the model generates descriptions at varying levels of detail, including category names, short phrases, detailed descriptions, or referring descriptions, and iii) Grounded Image Conversation: The model needs to output indices of objects mentioned in its generated conversation output. We mix the ground truth boxes of the current image with the proposal boxes from UPN, and keep at most 100 boxes as input. We adopt a three-stage training process and the data for each stage are listed in Tab. 1

###### Stage-1: Alignment Training. In the first stage, the ob-

jective is to align visual features and object features with the text feature space. To achieve this, we train the image projection MLP, object projection MLP, as well as the input and output embeddings of the LLM, given that we have added special tokens to its vocabulary.

###### Stage-2: Perception Training. In this stage, we im-

prove the perception capability of ChatRex by training it on Rexverse-2M and other grounding data. All the parameters are trainable in this stage.

###### Stage-3: Joint Training. In this stage, we integrate

perception and understanding tasks into a unified training process, ensuring that ChatRex acquires both capabilities. This joint optimization equips the model with comprehensive multimodal abilities and enables mutual enhancement between perception and understanding.

##### 5. Experiments

###### 5.1. Perception Capability Evaluation

Evaluation Metrics. Mean Average Precision (mAP) [46] is a common metric for object detection, which measures the area under the precision-recall curve, reflecting both the precision and recall of the model. However, for MLLMs that predict coordinates as vocabulary tokens, computing AP can be challenging due to the lack of confidence scores for each predicted box. Therefore, we directly report recall and precision metrics instead. We provide all ground truth categories for the current test image and prompt the model to generate the corresponding coordinate boxes. The details of prompts used for each model are included in the Appendix. For ChatRex, we use fine-grained proposal boxes from UPN and their corresponding confidence scores as input, enabling us to compute precision, recall, and mAP.

|Method|Type|COCO-Val|LVIS-Mini Val|RefCOCO<br><br>|RefCOCO+<br><br>|RefCOCOg|
|---|---|---|---|---|---|---|
| | |P@0.5 R@0.5 mAP|P@0.5 R@0.5 mAP AP-R AP-C AP-F<br><br>|val testA testB<br><br>|val testA testB|val test|
|Faster-RCNN [70] DETR [8] Pix2Seq [12] DINO [102]<br><br>|Closed-set Detection Model<br><br>|- - 42.0<br><br>- - 43.3<br><br>- - 43.2<br><br>- - 49.4<br><br><br>|- - - - - -<br><br>- - - - - -<br><br>- - - - - -<br><br>- - - - - -<br><br><br>|- - -<br><br>- - -<br><br>- - -<br><br>- - -<br><br><br>|- - -<br>- - -<br>- - -<br>- - -<br>|- -<br><br>- -<br><br>- -<br><br>- -<br>|

|Florence2 [88] GLIP [39] T-Rex2 [29] Grounding DINO [52]|Open-set Detection Model<br><br>|- - 43.4<br>- - 49.8<br>- - 46.5<br>- - 48.4<br>|- - - - - -<br><br>- - 37.3 28.2 34.3 41.5<br><br>- - 47.6 45.4 46.0 49.5<br><br>- - 33.0 22.2 30.7 38.8<br><br><br>|- - -<br><br>- - -<br><br>- - -<br><br><br>89.2 91.9 86.0<br><br>|- - -<br>- - -<br>- - -<br><br><br>81.1 87.4 74.7|- -<br><br>- -<br><br>- -<br><br><br>84.2 84.9|
|---|---|---|---|---|---|---|
|Shikra-7B [10] Ferret-7B [94] Groma-7B [61] InternVL2-7B [14] Qwen2-VL-7B [85]|MLLM<br><br>|40.3 21.5 66.3 33.5 69.9 28.9 45.3 24.5 59.3 43.9 -<br><br>|52.8 14.5 - - - 72.9 25.2 - - - -<br><br>76.3 10.9 - - - 51.6 13.1 - - - -<br><br>77.0 34.7 - - - -<br><br><br>|87.0 90.6 80.2<br><br>- - 89.5 92.1 86.3<br><br>87.1 91.1 80.7 91.7 93.6 87.3<br>|81.6 87.4 72.1 - - 83.9 88.9 78.1 79.8 87.9 71.4 85.8 90.5 79.5<br><br>|82.3 82.2<br><br>-<br><br>86.4 87.0 82.7 82.7<br>87.3 87.8<br>|
|ChatRex-7B| |73.5 72.8 48.2<br><br>|80.3 58.9 42.6 44.6 48.4 37.2<br><br>|91.0 94.1 87.0<br><br>|89.8 91.9 79.3<br><br>|89.8 90.0<br><br>|

Table 2. Comparison of different models on object detection tasks on the COCO, LVIS, and RefCOCO/+/g datasets. For COCO and LVIS, we report the R@0.5 and P@0.5 metrics for MLLMs, representing recall and precision at an IoU threshold of 0.5, respectively. For RefCOCO/+/g, a prediction is considered correct if its overlap IoU with the ground truth is larger than 0.5.

Model MME MMB SEEDI MMStar MMVet MMMU AI2D OCRBench TextVQA POPE Hallusion BLIP-2 [38] 1293.8 - 49.7 - 22.4 - - - - 85.3 -

InstructBLIP [16] 1212.8 - - - - - - - - 78.9 Mini-Gemini-HD-8B [42] 1606.0 72.7 73.2 - - 37.3 73.5 47.7 70.2 - -

LLaVA-HR [58] 1554.0 - 64.2 - 31.2 - - - 67.1 87.6 -

LLaVA-NeXT-7B [51] 1498.0 68.7 72.2 38.4 42.2 35.3 69.0 531 64.6 86.7 29.1 Eagle-X5-7B [76] 1579.0 68.8 73.5 41.7 42.6 36.3 77.2 574 71.2 88.8 37.8 MM1.5-7B [103] 1514.9 - 73.4 - 42.2 41.8 72.2 635 76.5 88.6 -

Cambrian-8B [81] 1547.1 75.9 74.7 47.1 48.9 41.6 73.6 610 71.7 86.8 39.4 LLaVA-OV-7B [37] 1577.8 83.2 76.7 61.9 51.9 47.9 82.4 622 78.5 88.4 31.6 InternVL2-8B [14] 1639.7 81.7 75.4 61.5 54.2 49.8 83.0 794 77.4 84.2 45.0 Qwen2-VL-7B [85] 1639.2 83.0 76.0 60.7 62.0 54.1 83.0 845 84.3 88.4 50.6

ChatRex-7B 1544.0 81.1 74.4 57.5 41.5 46.7 79.1 626 69.1 87.6 39.1

Table 3. Comparison of different models on multimodal benchmarks.

Common Object Detection. As shown in Tab. 2, ChatRex achieves a 48.2 mAP on the COCO dataset, which is comparable to conventional object detectors like DINO [102], indicating that ChatRex possesses strong perception capabilities. In contrast, other MLLMs generally exhibit low recall rates. This discrepancy arises from the multi-object nature of COCO, where each image contains multiple categories with numerous instances. The low recall rate implies that current MLLMs face significant challenges in detecting multiple objects, which is a common requirement in real-world scenarios. Furthermore, we identified specific issues with general MLLMs such as InternVL2 and Qwen2-VL, which have a tendency to repeatedly generate the same coordinates until reaching the model’s maximum output length. A detailed analysis of these problems is provided in the Appendix, highlighting areas for potential improvement in future work.

Long-tailed Object Detection. We further evaluate ChatRex on the more challenging LVIS [25] dataset, which encompasses 1,203 object categories. ChatRex achieved a 42.6 mAP, surpassing open-set detection models like Grounding DINO [52] and GLIP [39] and is on par with T-Rex2 [29]. We attribute this performance to the strong

semantic understanding capabilities of the LLM. Within the ChatRex model structure, the LLM primarily functions to classify bounding boxes generated by the proposal model. By aligning visual features with the textual space through comprehensive training and data optimization, the LLM is able to accurately classify a broad spectrum of categories, thereby demonstrating its robustness in handling complex, long-tailed object detection scenarios.

Referring Object Detection. Referring object detection involves identifying an object based on a given description. We evaluate ChatRex on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks, which predominantly focus on single-object detection, where each expression generally corresponds to a single object. As shown in Tab. 2, ChatRex possesses strong referring capabilities, which are crucial for tackling complex perception tasks.

###### 5.2. Understanding Capability Evaluation

General Multimodal Benchmarks. We evaluate ChatRex on various academic multimodal benchmarks including MME [23], MMBench [53], SEEDI [35], MMstar [11], MM-Vet [96], MMMU [98], AI2D [31], OCRBench [54], TextVQA [77], POPE [40], and HallusionBench [24]. As

LVIS PACO

Method

SS S-IoU SS S-IoU

LLaVA-1.5 [50] 49.0 19.8 42.2 14.6 Kosmos-2 [65] 39.0 8.7 32.1 4.8 Shikra-7B [10] 49.7 19.8 43.6 11.4

GPT4RoI-7B [106] 51.3 12.0 48.0 12.1

Ferret-7B [94] 63.8 36.6 58.7 26.0 Osprey-7B [97] 65.2 38.2 73.1 52.7

VisionLLM v2-7B [87] 68.9 46.3 67.7 44.0 SPHINX-V-7B [48] 87.1 62.9 80.0 55.0 ChatRex-7B (Ours) 89.8 82.6 91.4 85.1

Table 4. Comparison on referring object classification task.

|Training With Stage-2<br><br>|Understanding|
|---|---|
| |MME MMB MMStar SEEDI MMMU POPE Hallusion|
|No Yes|1388.8 76.8 44.5 72.9 46.0 80.3 34.0 1439.1 76.7 46.2 73.0 52.0 86.0 37.6<br><br>|

Perception

|Training With Stage-2<br><br>|COOC mAP<br><br>LVIS mAP<br><br>Ref+ val<br><br>Ref+ testA<br><br>Ref+ testB<br><br>Refg test<br><br>Refg val|
|---|---|
|No Yes<br><br>|47.8 42.3 85.9 90.9 79.7 89.4 88.8<br>48.7 43.4 88.6 92.2 82.2 90.6 90.3<br>|

- Table 5. Ablation of the impact of Stage-2 training on the perception and understanding capabilities of ChatRex. We remove the LLaVA-Onevision data in the final stage for quick validation.

|Eval Stage<br><br>|COCO mAP<br><br>LVIS mAP<br><br>Ref+ val<br><br>Ref+ testA<br><br>Ref+ testB<br><br>Refg test<br><br>Refg val|
|---|---|
|Stage-2<br><br>Stage-3<br><br><br>|48.7 42.2 86.5 91.9 80.9 89.6 89.0 48.2 42.6 89.8 91.9 79.3 89.8 90.0|

- Table 6. Ablation of the perception capabilities of ChatRex at different stage checkpoints.

shown in Table 3, ChatRex demonstrates strong multimodal capabilities, though there remains a performance gap compared to state-of-the-art models like Qwen2-VL and InternVL2. We believe this performance gap can be narrowed with larger-scale data and improved visual representations, such as the AnyRes strategy [37]. Furthermore, ChatRex’s enhanced perception capabilities enable it to provide object grounding information during conversations, extending its applicability to a broader range of real-world scenarios.

Region Caption Benchmarks. In addition to imagelevel understanding, ChatRex demonstrates strong regionlevel understanding capabilities. Following Osprey [97], we evaluate the referring object classification task on the LVIS [25] and PACO [68] datasets. In this task, given the object index, the model is prompted to output the category name of the specified region. The evaluation metrics include Semantic Similarity (SS) and Semantic Intersection over Union (S-IOU) [72]. As shown in Tab. 4, ChatRex achieves state-of-the-art results, highlighting its robust region classification capabilities.

|Method<br><br>|Understanding|
|---|---|
| |MME MMB MMStar SEEDI MMMU POPE Hallusion|
|Baseline w/o ConvNeXt|1439.1 76.7 46.2 73.0 52.0 86.0 37.6 1341.9 70.4 42.7 65.4 43.3 83.8 24.4<br><br>|

Perception

|Method|COOC mAP<br><br>LVIS mAP<br><br>Ref+ val<br><br>Ref+ testA<br><br>Ref+ testB<br><br>Refg test<br><br>Refg val<br><br>|
|---|---|
|Baseline w/o ConNeXt|48.7 43.4 88.6 92.2 82.2 90.6 90.3 26.0 22.0 66.8 73.2 61.4 73.1 72.5<br><br>|

Table 7. Ablation of the dual vision encoder designing in ChatRex. We evaluate the impact of removing the ConvNeXt high-resolution encoder on perception and understanding tasks. We remove the LLaVA-Onevision data in Stage-3 for quick validation.

###### 5.3. Ablation Experiments

Mutual Benefits of Perception and Understanding. We conduct ablation experiments to analyze the mutual influence of perception and understanding in ChatRex. As shown in Tab. 5, incorporating perception training in Stage2 enhances the model’s multimodal understanding, demonstrating that stronger perception contributes to improved understanding. Additionally, training with Stage-2 also leads to a performance boost in perception, highlighting the effectiveness of the Rexverse-2M dataset in strengthening perception capabilities.

To further examine this interaction, we compare the performance of the perception between Stage-2 and Stage-3, as shown in Tab. 6. The results indicate that the perception performance of ChatRex improves with Stage-3 training, suggesting that multimodal data not only enhances understanding but also refines perception. This reinforces the interdependence of perception and understanding, demonstrating that their integration leads to mutual enhancement and a more robust multimodal model.

Ablation on ChatRex Architecture. In ChatRex, we adopt a dual vision encoder design, where object features are extracted from the high-resolution encoder to serve as object tokens. To assess the effectiveness of this design, we conduct an ablation study by removing the high-resolution encoder and instead extracting object features from the lowresolution CLIP encoder. As shown in Tab. 7, the results indicate that the removal of the high-resolution encoder leads to a decline in both perception and understanding performance, with a more significant drop observed in perception. We attribute this to the fact that perception tasks heavily rely on higher input image resolutions to capture finegrained details, which aligns with established findings in object detection community, where increasing image resolution generally improves detection accuracy and overall performance.

##### 6. Conclusions

In this work, we reveal the perception drawback in existing MLLMs due to their conflict modeling between perception and understanding, and the lack of data that effectively balance these two aspects. To address these issues, we

introduce ChatRex, a model with a decoupled architecture, along with Rexverse-2M, a multi-granularity dataset designed to balance these two aspects. ChatRex demonstrates strong perception abilities while also excelling in multimodal understanding and dialog capabilities. The synergy between perception and understanding allows ChatRex to be highly interactive by grounding mentioned objects within visual contexts during dialogue. We believe that perception and understanding are both critical for MLLMs, as their integration can significantly enhance model capabilities and unlock a wide range of novel applications.

##### References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219,

2024. 1

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [3] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Devendra Chaplot, Jessica Chudnovsky, Saurabh Garg, Theophile Gervet, Soham Ghosh, Am´elie H´eliou, Paul Jacob, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073,

2024. 1, 3

- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 35:23716–23736, 2022. 1, 3
- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966,

2023. 1, 3

- [7] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 5

- [8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer,

2020. 3, 7

- [9] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024. 6
- [10] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 1, 3, 7, 8, 2
- [11] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 7
- [12] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. arXiv preprint arXiv:2109.10852,

2021. 1, 3, 7

- [13] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. In ICLR, 2022. 1, 3
- [14] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with opensource suites. arXiv:2404.16821, 2024. 1, 2, 3, 5, 6, 7
- [15] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 1
- [16] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. In NeurIPS, 2023. 1, 3, 7
- [17] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuoling Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. arXiv preprint arXiv:2409.11402, 2024. 3
- [18] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-theart multimodal models. arXiv preprint arXiv:2409.17146,

2024. 1, 3

- [19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009. 1

- [20] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k HD. arXiv: 2404.06512, 2024. 3
- [21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 4
- [22] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1, 2, 6

- [23] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024. 7
- [24] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large vision-language models. arXiv preprint arXiv:2310.14566, 2023. 7
- [25] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5356–5364, 2019. 3, 6, 7, 8
- [26] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In ICCV, pages 2961–2969, 2017. 4
- [27] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019. 6
- [28] Dongfu Jiang, Xuan He, Huaye Zeng, Con Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multiimage instruction tuning. arXiv:2405.01483, 2024. 3
- [29] Qing Jiang, Feng Li, Zhaoyang Zeng, Tianhe Ren, Shilong Liu, and Lei Zhang. T-rex2: Towards generic object detection via text-visual prompt synergy. In European Conference on Computer Vision, pages 38–57. Springer, 2025. 3, 7, 1
- [30] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 3, 6
- [31] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is

- worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235– 251. Springer, 2016. 7
- [32] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chlo´e Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross B. Girshick. Segment anything. arXiv: 2304.02643, 2023. 3, 6
- [33] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 128(7): 1956–1981, 2020. 6
- [34] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692, 2023. 3
- [35] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 7
- [36] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixtureof-experts model. arXiv preprint arXiv:2410.05993, 2024. 1, 3
- [37] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 3, 6, 7, 8
- [38] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 7
- [39] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975, 2022. 7
- [40] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, pages 292–305,

2023. 7

- [41] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv: 2403.18814, 2024. 3
- [42] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 7

- [43] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai.

- Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607, 2023. 3
- [44] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 3
- [45] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024. 3
- [46] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, pages 740–755, 2014. 1, 3, 6
- [47] Weifeng Lin, Xinyu Wei, Ruichuan An, Peng Gao, Bocheng Zou, Yulin Luo, Siyuan Huang, Shanghang Zhang, and Hongsheng Li. Draw-and-understand: Leveraging visual prompts to enable mllms to comprehend what you want. arXiv preprint arXiv:2403.20271, 2024. 6
- [48] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 3, 8
- [49] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv: 2310.03744, 2023. 6
- [50] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 3, 5, 8
- [51] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3, 7
- [52] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 7
- [53] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv: 2307.06281, 2023. 7
- [54] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. 7
- [55] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, pages 10012–10022, 2021. 1
- [56] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976– 11986, 2022. 2, 4

- [57] Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. Towards end-to-end unified scene text detection and layout analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1049–1059, 2022. 6
- [58] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-ofresolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024. 7
- [59] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-ofresolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024. 3
- [60] Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, et al. Kosmos-2.5: A multimodal literate model. arXiv preprint arXiv:2309.11419, 2023. 1, 3
- [61] Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. arXiv preprint arXiv:2404.13013, 2024. 1, 2, 3, 7
- [62] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, pages 11–20, 2016. 3, 6
- [63] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, Anton Belyi, Haotian Zhang, Karanjeet Singh, Doug Kang, Ankur Jain, Hongyu H`e, Max Schwarzer, Tom Gunter, Xiang Kong, Aonan Zhang, Jianyu Wang, Chong Wang, Nan Du, Tao Lei, Sam Wiseman, Guoli Yin, Mark Lee, Zirui Wang, Ruoming Pang, Peter Grasch, Alexander Toshev, and Yinfei Yang. MM1: methods, analysis & insights from multimodal LLM pre-training. arXiv: 2403.09611, 2024. 3
- [64] OpenAI. Gpt-4v(ision) system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf,

2023. 1, 3

- [65] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 3, 8
- [66] Renjie Pi, Lewei Yao, Jiahui Gao, Jipeng Zhang, and Tong Zhang. Perceptiongpt: Effectively fusing visual perception into llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27124– 27133, 2024. 3
- [67] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 4
- [68] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, et al. Paco: Parts

- and attributes of common objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7141–7151, 2023. 6, 8
- [69] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024. 3
- [70] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. In NIPS, pages 91–99, 2015. 7
- [71] Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, et al. Grounding dino 1.5: Advance the” edge” of open-set object detection. arXiv preprint arXiv:2405.10300, 2024. 2, 6
- [72] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 658–666,

2019. 8, 1

- [73] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 35: 25278–25294, 2022. 4
- [74] Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, and Jian Sun. Crowdhuman: A benchmark for detecting human in a crowd. arXiv preprint arXiv:1805.00123, 2018. 6
- [75] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, pages 8430–8439, 2019. 3, 6
- [76] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024. 3, 7
- [77] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 7
- [78] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023. 1
- [79] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv: 2312.11805, 2023. 1, 3

- [80] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https:// github.com/InternLM/InternLM, 2023. 1
- [81] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2025. 7
- [82] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 3
- [83] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv: 2302.13971, 2023. 1
- [84] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv: 2307.09288, 2023. 1
- [85] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 3, 7, 2
- [86] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 3
- [87] Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Wenhai Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, et al. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. arXiv preprint arXiv:2406.08394,

2024. 3, 8

- [88] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024. 7
- [89] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an LMM perceiving any aspect ratio and high-resolution images. arXiv: 2403.11703, 2024. 3
- [90] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024. 3
- [91] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yu-

- tong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024. 3
- [92] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 4
- [93] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 3
- [94] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704, 2023. 1, 3, 7, 8, 2
- [95] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In ECCV, pages 69–85, 2016. 3, 6
- [96] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 7
- [97] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28202–28211, 2024. 6, 8
- [98] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502,

2023. 7

- [99] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 6
- [100] Yufei Zhan, Yousong Zhu, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon v2: Advancing multimodal perception with high-resolution scaling and visuallanguage co-referring. arXiv preprint arXiv:2403.09333,

2024. 3

- [101] Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon: Spelling out all object locations at any granularity with large language models. In European Conference on Computer Vision, pages 405–422. Springer, 2025. 3
- [102] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022. 7, 1
- [103] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, et al. Mm1. 5: Methods, anal-

- ysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566, 2024. 7
- [104] Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, et al. Ferretv2: An improved baseline for referring and grounding with large language models. arXiv preprint arXiv:2404.07973,

2024. 3

- [105] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Chunyuan Li, Jainwei Yang, et al. Llava-grounding: Grounded visual chat with large multimodal models. In European Conference on Computer Vision, pages 19–35. Springer, 2025. 3
- [106] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-ofinterest. arXiv: 2307.03601, 2023. 8
- [107] Yuanhan Zhang, Qinghong Sun, Yichun Zhou, Zexin He, Zhenfei Yin, Kun Wang, Lu Sheng, Yu Qiao, Jing Shao, and Ziwei Liu. Bamboo: Building mega-scale vision dataset continually with human-machine synergy. arXiv preprint arXiv:2203.07845, 2022. 6
- [108] Xiangyu Zhao, Xiangtai Li, Haodong Duan, Haian Huang, Yining Li, Kai Chen, and Hua Yang. Mg-llava: Towards multi-granularity visual instruction tuning. arXiv preprint arXiv:2406.17770, 2024. 3
- [109] Chenchen Zhu, Fanyi Xiao, Andr´es Alvarado, Yasmine Babaei, Jiabo Hu, Hichem El-Mohri, Sean Culatana, Roshan Sumbaly, and Zhicheng Yan. Egoobjects: A largescale egocentric dataset for fine-grained object understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20110–20120, 2023. 6

## ChatRex: Taming Multimodal LLM for Joint Perception and Understanding Supplementary Material

##### A. MLLM Evaluation Details on Detection Datasets

In this section, we explain our methodology for evaluating MLLMs on the object detection task, including the design of model-specific prompts, the visualization of output results, and a comprehensive analysis of the challenges and limitations encountered throughout the evaluation process.

###### A.1. Prompt for Each MLLM

For each MLLM evaluated, we utilized either the prompts used in the original paper or manually crafted optimized prompts to maximize performance. The specific prompts used in our evaluation are detailed in Tab. 8,

###### A.2. More Analysis on the Detection Results

The visualization results on the COCO dataset are presented in Fig. 5. The qualitative analysis indicates that most evaluated MLLMs suffer from a low recall rate, underscoring significant limitations in their object perception capabilities. Additionally, specific deficiencies are observed in certain models. For example, Shikra [10] exhibits a notable issue with coordinate offset, where the predicted bounding boxes fail to accurately enclose the target objects. This misalignment is primarily attributed to quantization errors inherent in the model. Moreover, during the evaluation phase, most models, excluding general-purpose MLLMs such as InternVL2 [14] and Qwen2-VL [85], frequently struggle to follow task-specific instructions. This observation highlights persistent challenges in the instruction-following capability within these models.

We also identified a distinct issue with general MLLMs during the evaluation process: a pattern of repeated output coordinates in their predictions. This repetition is not random but follows a systematic pattern. For example, in the case of Qwen2-VL-7B, the model consistently generates bounding box coordinates where each subsequent bounding box is derived by incrementally adjusting the preceding one. For example, the xmin and xmax values of the bounding boxes are increased by a fixed value, such as 24, as illustrated in the below block. We argue this behavior stems from the auto-regressive coordinate prediction mechanism. The reliance on previously predicted coordinates introduces a probabilistic bias, causing subsequent predictions to deviate systematically as regular modifications of earlier outputs. This limitation underscores the model’s inability to independently and accurately generate coordinates without cumulative error propagation.

Example of repeated coordinate prediction in Qwen2-VL-7B.

Response: [ {class: car, rect: [234, 186, 370, 283]},

{class: car, rect: [568, 214, 622, 283]}, {class: car, rect: [743, 186, 822, 300]}, {class: car, rect: [110, 199, 128, 240]}, {class: car, rect: [134, 200, 152, 240]}, {class: car, rect: [158, 200, 176, 240]}, {class: car, rect: [182, 200, 200, 240]}, {class: car, rect: [206, 200, 224, 240]}

]... Repeat until exceeds the max output length.

##### B. Model Details

###### B.1. Implementation Details of UPN

In alignment with the methodology of T-Rex2 [29], we employ the Swin Transformer Large model [55], pre-trained on ImageNet [19], as the vision backbone. During the Hungarian matching process, the optimization incorporates three types of losses: classification loss, box L1 loss, and generalized intersection over union (GIOU) loss [72], with respective weights of 2.0, 5.0, and 2.0. For the overall loss computation, we similarly utilize classification loss, box L1 loss, and GIOU loss, adjusting the corresponding weights to 1.0, 5.0, and 2.0. Consistent with the training strategy of DINO [102], we adopt contrastive denoising training (CDN) to enhance training stability and accelerate convergence. The pre-trained weights of T-Rex2-L are used for initialization, followed by full-parameter optimization on the universal proposal task.

###### B.2. Implementation Details for ChatRex

We utilize the CLIP pre-trained ViT-Large-14-336 model as the low-resolution visual encoder and the LAION pretrained ConvNext-Large-320 model as the high-resolution visual encoder. The input resolution is set to 336x336 for the low-resolution encoder and 768x768 for the highresolution encoder. During the pretraining stage, we employ a batch size of 32 per device, resulting in an aggregate batch size of 256 across all devices. For the instructiontuning stage, the batch size is reduced to 16 per device, with a total batch size of 128. The learning rate is initialized at 1e-3 for the pre-training stage and adjusted to 2e-5 during the instruction-tuning stage.

For perception and region-based question-answering tasks, we designed tailored prompts to effectively guide and instruct the models. Examples of these customized prompts

GT InternVL2-8B Qwen2-VL-7B Shikra-7B Ferret-7B Groma-7B

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

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Figure 5. Visualized prediction results on the COCO dataset, from general-purpose MLLM including Qwen2-VL [85], InternVL2 [14], and detection-focused MLLMs including Ferret [94], Shikra [10], and Groma [61] These models generally suffer from a low recall rate in multi-object scenes.

Method Prompt

In this picture, you are required to finish object detection for every instance of the category we provide. To complete the above mission, you need to provide me with the answers in the format of a Python list of dictionaries by the category provided above. Attention: No other category shall appear in the detection object attributes, except for the genre we offer. Bounding box format: [108(xmin), 210(ymin), 810(xmax), 640(ymax)], where xmin, ymin, xmax and ymax must be positive integers. If there is no object in the picture, please provide an empty list. Here is an example which you must follow in your responses. Example: If the question is as below: Category: [’person’, ’car’]. If there is an object of the category, The Answer should be: [{{”class”: ”person”, ”rect”: [0, 614, 220, 771]}}, {{”class”: ”person”, ”rect”: [638, 468, 784, 941]}}, {{”class”: ”car”, ”rect”: [110, 100, 500, 300]}}]. Else if no object of the category in the picture, the Answer should be:[]. Here is the question you shall answer: Category: {}

Qwen2-VL-7B & InternVL2-8B

Ferret-7B What is the location of all instances of categories {} in the image? Please answer me respectively. Shikra-7B Help me locate {} in <image>and give its coordinates, please.

[grounding] There are categories you need to describe with positions, only including <p>{}<p>. Give me a short description of the image and include the coordinates [[x0,y0,x1,y1]] for each instance of categories.

Groma-7B

Table 8. Prompt used by each MLLM for object detection.

are provided in Tab. 9.

##### C. Details for the Rexverse-2M Data Engine

###### C.1. Visualization of Rexverse-2M dataset

We visualize a portion of the Rexverse-2M dataset in Fig. 6, including image-level annotations and region-level annotations.

###### C.2. Prompt for Different Modules

In the Rexverse-2M data engine, we leverage both state-ofthe-art MLLMs and Large Language Models LLMs to construct the dataset. The prompts employed in each module are detailed in Tab. 10.

###### C.3. Effectiveness of Phrase-Conditioned Region Description

In the data engine, to generate region-level descriptions, we adopt a phrase-conditioned region captioning method. This approach utilizes a short phrase or category name, specified for each region during the grounding phase, as an additional prompt to guide a multimodal large language model (MLLM) in producing captions. This method differs from directly inputting regional images into the MLLM for captioning. As illustrated in Figure 7, the direct image captioning approach often suffers from hallucinations, particularly when regions contain distracting objects or are too small to be reliably recognized. In contrast, the phrase-conditioned method mitigates these issues by incorporating contextual input, leading to more accurate captions.

Task Prompt Templates

Grounding & Detection & Referring Please detect [OBJ] in this image. Answer the question with object indexes. Brief Image Caption with Grounding Please briefly describe this image and detect all the mentioned objects. Answer with grounded object indexes.

Detailed Image Caption with Grounding Please provide a detailed description of the image and detect all the mentioned objects. Answer the question with grounded object indexes. Region Caption in Category Name What is the category name of [OBJ]? Answer the question with its category name in free format. Region Caption in Short Phrase Can you provide me with a short phrase description of [OBJ]? Answer the question with short phrases. Region Caption Briefly Can you provide me with a brief description of [OBJ]? Answer the question with brief description. Region Caption in One Sentence Can you provide a one sentence description of [OBJ] in the image? Answer the question with one sentence description.

Grounded Counting How many [OBJ] are there in this image? Answer the question with the number of objects and locate them with object indexes. Grounded Conversation Answer the question in grounded format. Question:

Table 9. Example prompt for different perception tasks of ChatRex.

Task Model Prompt Image Description InternVL2-8B Please provide a one-sentence description for this image.

I will provide you with a short phrase description of an object and its image. You need to rewrite this short phrase description to a one sentence description by adding more details about this object based on the image. The rewritten description can only focus on this object according to the original description and should also be a one-sentence description. The original short phrase description is:

Phrase-Conditioned Region Descrption

InternVL2-8B

I will provide you with a one-sentence description of an object, and the category name of that object. Based on these two pieces of information, write a referring description of the object. This description should capture the most important and distinguishing features of the object, and should not describe anything that doesn’t exist in the description I’ve provided. Note that the referring object should be the category name provided. The rewritten referring description should be more than 5 words but less than 10 words. The referring description should be as short and concise as possible, without commas. Directly output the answer.

Region Filtering & Rewriting

LLAMA3-8B-Instruct

Table 10. Prompt for each module where MLLM or LLM is used.

##### D. ChatRex Capabilities and Qualitative Examples

In this section, we present the visualization results to demonstrate the capabilities of ChatRex.

- D.1. Common Object Detection

We show the results on the common object detection task in Fig. 8.

- D.2. Long-tailed Object Detection

We show the results on the long-tailed object detection task in Fig. 9.

- D.3. Short-Phrase Object Detection

We show the results on the short-phrase object detection task in Fig. 10.

- D.4. Referring Object Detection

We show the results on the referring object detection task in Fig. 11.

- D.5. Region Caption We show the results on the region caption task in Fig. 12.
- D.6. Region QA We show the results on the region QA task in Fig. 13.

- D.7. Brief Grounded Image Caption We show the results on the brief grounded image caption

- task in Fig. 14.

D.8. Detailed Grounded Image Caption

We show the results on the detailed grounded image caption

- task in Fig. 15.

D.9. Grounded Counting

We show the results on the grounded counting task in Fig.

- 16.

- D.10. Grounded Conversation

We show the results on the grounded conversation task in Fig. 17 and Fig. 18.

Image Caption & Region Grounding

[Figure 44]

The image features a unique combination of

[Figure 45]

Three white horses are galloping through a body of water creating splashes and sprays, against a

objects: a green, semi-precious stone ring with a polished and faceted surface, resting

on a large, irregularly shaped, clear crystal rock, surrounded by various other colorful

dramatic, sepia-toned sky.

gemstones and moss in the background,

creating a visually appealing and intricate

display

[Figure 46]

[Figure 47]

A high-speed Formula 1 race car,

The image captures two young girls

number 12, adorned with Verizon and Shell logos, navigates a turn on a green track with blurred spectators in the background,

deeply engrossed in playing violins, their

focused expressions reflecting their dedication to the music, with their hands

expertly positioned on the instruments

captured in motion by the sharp

and bows in motion, while seated in a

focus on the car.

classroom setting with a blurred

background featuring other students and

a conductor. The image shows a sleek black Mini Cooper

[Figure 48]

[Figure 49]

Three scuba divers are posing for a photo on a boat, wearing black

S car with the license plate "S.SB 2069" in

wetsuits with "Aventura" logos, yellow

motion on a road, featuring modern design

snorkels, and snorkel masks, with the

elements such as a prominent front grille,

ocean and clear blue sky in the

angular headlights, and aerodynamic

background

contours, captured with a motion blur effect to emphasize speed and dynamism against

a backdrop of blurred trees and a clear sky.

Region Caption

[Figure 50]

[Figure 51]

Detailed Caption: A colorful bracelet

Detailed Caption: A white paper cup with a

featuring a mix of pastel and metallic beads, showcasing a combination of green, pink, silver, and black tones.

green polka dot pattern, labeled "Fall Style

| |
|---|

Autumn Collection" and featuring a steaming

cup icon with "Tea House" branding,

accompanied by a tea leaf.

Referring Caption: Colorful mix of pastel and

metallic beads in green pink silver black

Referring Caption: White paper cup with green polka dots.

| |
|---|

[Figure 52]

[Figure 53]

Detailed Caption: a person's muscular

Detailed Caption: A vibrant, blue-tinged mushroom stands prominently against a

| |
|---|

shoulder, highlighting the contours and

definition of the upper arm muscles. Referring Caption: Muscular upper arm with defined contours.

blurred backdrop of lush greenery and warm

sunlight filtering through the foliage. Referring Caption: Vibrant blue-tinged

mushroom stands prominently against

greenery.

| |
|---|

Detailed Caption: A vibrant green sprig

[Figure 54]

[Figure 55]

Detailed Caption: a car trunk, featuring a

of mint delicately adorns a slice of

gray carpeted floor and a neatly arranged space designed for storage

| |
|---|

toasted garlic bread, adding a fresh burst of flavor.

Referring Caption: Gray carpeted storage

Referring Caption: Fresh green mint leaf

space for luggage and cargo.

on toasted garlic bread.

Figure 6. Visualization of the Rexverse-2M dataset.

[Figure 56]

[Figure 57]

Category Name: top

Category Name: top

| |
|---|

With Phrase-Conditioned: a V-neckline top adorned with a delicate necklace featuring a circular pendant.

With Phrase-Conditioned: a dark suit jacket with a white shirt underneath is prominently displayed in the image

| |
|---|

Without Phrase-Conditioned: a close-up of a

Without Phrase-Conditioned: a person

person wearing a burgundy-colored top and

wearing a black suit with a white shirt

a delicate necklace with a pendant.

underneath.

Figure 7. Visualization of the proposed phrase-conditioned region description method.

Common Object Detection

[Figure 58]

[Figure 59]

[Figure 60]

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

- Figure 8. Visualization on the Common Object Detection Task.

Long-tailed Object Detection

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

- Figure 9. Visualization on the Long-tailed Object Detection Task.

Short Phrase Object Detection

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

- Figure 10. Visualization on the Short-Phrase Object Detection Task.

Referring object detection

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

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

- Figure 11. Visualization on the Referring Object Detection Task.

Region Caption in Category Name

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Region Caption in Short Phrase

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Region Caption in Referring Style

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Region Caption in Detail

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Figure 12. Visualization on Region Caption Tasks.

[Figure 129]

Question: What is <obj0> wearing and what is he doing now? ChatRex: <obj0> is wearing a black jacket and a black beanie.

He is standing in a store, looking at something on the shelf.

[Figure 130]

Question: What are the positions of <obj0>, <obj1>, <obj2>, and <obj3> in the band, respectively?

ChatRex: <obj0> is the bassist, <obj1> is the guitarist, <obj2> is the vocalist, and <obj3> is the drummer.

[Figure 131]

Question: What is <obj0>? And how might <obj0> feels now? ChatRex: <obj0> is a cow, and it might feel scared or startled by

the action of the man on the horse.

[Figure 132]

Question: What is the price for <obj0> according to this image. ChatRex: The price for <obj0> is 2.49, as indicated by the sign in front of the cat.

[Figure 133]

Question: Describe the wearing of <obj0> in detail ChatRex: The dress of <obj0> is a bright yellow, which

stands out against the snowy backdrop. It appears to be a

winter coat, designed to provide warmth in cold weather. The coat is long-sleeved and reaches down to the person's

knees, suggesting it is made for cold temperatures. The coat's color is vibrant and contrasts with the muted colors

of the snowy environment.

Figure 13. Visualization on the Region QA Task.

[Figure 134]

A man in a white tuxedo

[Figure 135]

Three dolphins are leaping out of the water,

with a red bow tie is holding an Oscar statuette and

creating a dynamic

standing on a stage with a microphone, while a large,

splash as they dive and play in the ocean.

ornate Oscar statue is visible

in the background

[Figure 136]

A person is slicing fresh

[Figure 137]

The image is a vintage

cilantro a wooden cutting

advertisement for Fitzgerald & Co., featuring

board, surrounded by various ingredients including

a hand pouring a bottle of whiskey into a crescent

a bowl of chopped

tomatoes, a bowl of grated cheese a bowl of chopped

moon with the text

“Fitzgerald & Co. Dublin" prominently displayed.

herbs, a whole avocado and

a bowl of grated cheese.

A woman with glasses is

A woman with long brown

[Figure 138]

[Figure 139]

smiling and holding a white

hair is enjoying a cone of ice cream on a beach, with

envelope with a handwritten address , standing in a cozy

a backdrop of waves and seagulls in flight.

room with a window and a

bookshelf filled with books in the background.

[Figure 140]

[Figure 141]

The image features a charming

The image showcases a variety of

black dog-shaped keychain

ingredients laid out on a white surface, including a plate of raw beef,

with a textured surface, adorned with a pink tongue

a bowl of sliced onions, a plate of

and a small brown tag

fresh green lettuce, a plate of sliced cheese, a bowl of nuts, and a plate of

attached to a brown leather strap, all set against a white

bread.

background.

Figure 14. Visualization on the Brief Grounded Image Caption Task.

[Figure 142]

The image is a portrait-style photograph of a man. He has long, dark hair that falls over his forehead and is styled in a way that it

appears to be parted in the middle. His facial features include a straight nose, full lips, and a slightly furrowed brow. The man is

wearing a dark blue suit with a white shirt and a dark blue tie. The

background is dark and out of focus, which puts the emphasis on the man's face and upper body. There are no visible texts or logos

in the image. The lighting on the man's face is soft, with the light

source coming from the front, casting gentle shadows on his face. The overall tone of the image is professional and serious.

The image depicts a military parade or event. In the foreground,

[Figure 143]

there is a large, green military truck with a camouflage pattern , carrying what appears to be a missile system. The truck is

positioned on a road and there are several people standing nearby, some of whom are taking photographs.\n\nIn the

background, there is a large stage with a blue and white color

scheme, featuring a flag with a red background and a white emblem in the center. The flag is flanked by two portraits of a

man presumably a leader or figure of importance, given the

context of the event. The stage is adorned with red and white flags, and there is a banner with Arabic text and a graphic of a

missile.\n\nThe sky is clear, suggesting good weather conditions

for the event.

The image is a color photograph featuring a young man standing on

[Figure 144]

a sidewalk. He is wearing a plaid shirt with a blue and white checkered pattern, a white undershirt, and a blue backpack. He has

curly hair and is looking down at a smartphone he is holding in his

right hand. The man is wearing white headphones around his neck.\n\nIn the background, there is a large map on a wall which

appears to be a map of a city. The map is framed by a black border

and is mounted on a wall that has a window with a view of the street outside. The window is partially obscured by a red sign with white

text, but the text is not legible in the image.\n\nThe sidewalk where

the man is standing is lined with a metal railing, and there are other people visible in the background, suggesting a public space or a

busy area.

Figure 15. Visualization on the Detailed Grounded Image Caption Task.

#### Grounded Counting

[Figure 145]

There are 20

<g>pigeons</g><o><obj0><obj1><obj2><obj3>< obj4><obj5><obj6><obj7><obj9><obj10><obj11>

<obj12><obj14><obj15><obj17><obj19><obj22>

<obj25><obj34><obj38></o>.

[Figure 146]

There are 12 <g>person</g><o><obj0><obj1><obj2><obj3>

<obj4><obj5><obj6><obj11><obj12><obj21>

<obj22><obj83></o>.

[Figure 147]

There are 18

<g>bananas</g><o><obj2><obj4><obj5><obj6><obj7><

obj8><obj9><obj10><obj11><obj12><obj13><obj14><o bj15><obj16><obj17><obj18><obj19><obj25></o>.

[Figure 148]

There are 2 <g>dog</g><o><obj2><obj9></

o>, 1 <g>cat</g><o><obj0></o>.

Figure 16. Visualization on the Grounded Counting Task.

### Grounded Conversation1: Embodied AI Planning and Navigation

[Figure 149]

Question: Answer the question in grounded format. How would you organize this space? Sort and arrange

items on the surface based on their function or category, ensuring the area is tidy and visually pleasing. ChatRex: To organize the space, I would sort the items into categories such as stationery, electronics,

personal items, and accessories. For stationery, I would group the pen, pencil, and ruler together.

Electronics would include the laptop, mouse, and keyboard. Personal items would be grouped with the notebooks, wallet, and keys. Accessories would be separated from the other categories and include the

scissors, bobby pins, and tape. After sorting, the items would be arranged neatly on the surface, with similar items grouped together and the larger items placed at the edges to create a visually pleasing and

organized space.

[Figure 150]

Question: Answer the question in grounded format. Use the tools available on the table to prepare the pear

for consumption. Ensure any necessary cleaning, peeling, or cutting is performed appropriately, and leave the workspace tidy afterward.

ChatRex: To prepare the pear for consumption, I would first ensure that the pear is clean and free of any dirt

or blemishes. If necessary, I would use the knife to peel the pear, removing the skin in a smooth and even manner. After peeling, the pear would be placed in the yellow bowl for further preparation. If the pear is to

be cut, the knife would be used to make the necessary cuts, ensuring that the pear is sliced into even

portions. The glasses on the table could be used to rinse the pear under running water to remove any residual water or liquid. After the pear is prepared, the knife would be cleaned and stored properly, and the

table would be tidied up to maintain a clean and organized workspace.

- Figure 17. Visualization on the Grounded Counting Task.

[Figure 151]

Question: Answer the question in Grounded format. What kind of atmosphere does this room create, and who might it appeal to?

ChatRex: The room creates a warm, inviting, and sophisticated atmosphere. The rich colors, elegant furniture,

and artwork suggest a taste for classic and refined interior design. It might appeal to individuals who appreciate a blend of traditional and contemporary styles, with a preference for comfort and a touch of luxury.

The room could be particularly attractive to those who enjoy hosting guests or entertaining, given the ample

seating and the welcoming ambiance.

[Figure 152]

Question: Answer the question in grounded format. Where might this boy be according to his surroundings?

ChatRex: Based on the surroundings, the boy appears to be in a dental clinic or office. This is indicated by the

presence of a dental chair, the dental instruments being used, and the computer monitor in the background, which is typical of a dental office for patient records and monitoring equipment.

- Figure 18. Visualization on the Grounded Counting Task.

