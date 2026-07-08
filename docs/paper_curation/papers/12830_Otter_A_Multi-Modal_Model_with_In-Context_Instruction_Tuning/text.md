# Otter: A Multi-Modal Model with In-Context Instruction Tuning

Bo Li∗, Yuanhan Zhang∗, Liangyu Chen∗, Jinghao Wang∗, Fanyi Pu∗ Joshua Adrian Cahyono, Jingkang Yang, Chunyuan Li, Ziwei Liu

## arXiv:2305.03726v2[cs.CV]28Jul2025

Abstract—Recent advances in Large Multimodal Models (LMMs) have unveiled great potential as visual assistants. However, most existing works focus on responding to individual instructions or using previous dialogues for contextual understanding. There is little discussion on employing both images and text as in-context examples to enhance the instruction following capability. To bridge this gap, we introduce the Otter model to leverage both textual and visual in-context examples for instruction tuning. Specifically, Otter builds upon Flamingo with Perceiver architecture, and has been instruction tuned for general purpose multi-modal assistant. Otter seamlessly processes multi-modal inputs, supporting modalities including text, multiple images, and dynamic video content. To support the training of Otter, we present the MIMIC-IT (MultI-Modal In-Context Instruction Tuning) dataset, which encompasses over 3 million multi-modal instruction-response pairs, including approximately 2.2 million unique instructions across a broad spectrum of images and videos. MIMIC-IT has been carefully curated to feature a diverse array of in-context examples for each entry. Comprehensive evaluations suggest that instruction tuning with these in-context examples substantially enhances model convergence and generalization capabilities. Notably, the extensive scenario coverage provided by the MIMIC-IT dataset empowers the Otter model to excel in tasks involving complex video and multi-image understanding.

Index Terms—Instruction Tuning, In-context Learning, Multimodal Models

✦

#### 1 INTRODUCTION

Recent advancements in Large Language Models (LLMs), exemplified by GPT-2 [72] and GPT-3 [10], have highlighted their effectiveness as few/zero-shot learners trained on vast text corpora. The development of models like InstructionGPT [67] and ChatGPT [65] through instruction tuning has significantly enhanced their capacity for understanding and executing natural language instructions. This innovation integrates task-specific rules during fine-tuning, markedly improving user intent comprehension and response precision. Concurrently, there has been significant progress in multimodal models [3], [7], [8], [16], [19], [20], [28], [45], [47], [56], [58], [99], [106], reflecting a trend towards leveraging diverse modalities like images and text. This shift aims to provide a more comprehensive approach to understanding and engaging with the real world.

These evolutions are primarily attributed to the exploration of in instruction tuning [22], [24], [69], [80], [85], [86], [87] applied to both Large Language Models (LLMs) and Large Multimodal Models (LMMs). Instruction tuning, a process that refines models through diverse, high-quality instructions [24], [85], has notably enhanced zero-shot capabilities in natural language processing [87] and multimodal tasks [27], [31], [48], [57], [99], [106]. However, current instruction tuning concentrates on simple instruction towards single image or

- • ∗Equal Contribution.
- • Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Joshua Adrian Cahyono, Jingkang Yang and Ziwei Liu are with the S-Lab, Nanyang Technological University. E-mail: {libo0013, yuanhan002, liangyu.chen, jinghao003, fpu001, jo0001no, jingkang001, ziwei.liu}@ntu.edu.sg
- • Chunyuan Li is with the Microsoft Research, Redmond. E-mail: chunyuan.li@microsoft.com

leveraging preceding dialogues as contextual information. They may overlook the potential synergistic effect of utilizing both images and text as contextual information during the instruction tuning process.

Reflecting on multi-modal models like Flamingo, which effectively integrate image-text data, their training underscores the synergy between images and texts. This interplay motivates our adoption of a similar approach in instruction tuning, combining both modalities to in-context examples. Furthermore, studies emphasize the benefits of optimized contextual information in contrastively learned visual language models [104], [105]. Meanwhile, in language models, instruction tuning datasets such as Flan collection [87] offer rich, complex textual contexts per instance (e.g. Premise, Hypothesis, and Target), this approach would potentially improve instruction understanding and facilitate differentiation from other instructions or tasks.

Inspired by these insights, we introduce the Otter model, inherited from the Flamingo structure, designed to effectively leverage this multimodal synergy and instruction-tuned to function as a general-purpose multimodal assistant. The in-context instruction tuning on Otter aligns seamlessly with Flamingo’s upstream image-text interleaved pretraining paradigm. This approach enhances the model’s proficiency in comprehending and executing complex instructions and improving training efficacy.

To better facilitate the in-context instruction tuning of Otter model, we propose MIMIC-IT (MultI-Modal In-Context Instruction Tuning) dataset. This expansive dataset encompasses over 3 million multimodal instruction-response pairs, including approximately 2.2 million unique instructions that span a wide spectrum of images and videos. The MIMICIT dataset stands out for its thorough curation, featuring a

Pretrained Model

Instruct Model

Instruct Data

###### Billion-scale corpus of images interleaved with text.

###### WALNUT AND BLUE CHEESE STUFFED MUSHROOMS

[Figure 1]

[Figure 2]

🦩

|[Figure 3]|
|---|

|[Figure 4]|
|---|

Instruction Tuning

### MIMIC-IT

Pretraining

###### MMC4/M3W

Flamingo/ OpenFlamingo

OTTER

<image> “This Walnut and Blue Cheese Stuffed Mushrooms recipe is sponsored by Fisher

Nuts.”, <image> “The ideas for stuffing mushrooms are endless, so many combinations to play with, a couple of my personal favourites are these Mediterranean Stuffed Mushrooms and these Spinach and Toasted Pine Nut Stuffed Mushrooms.”,

###### Image-Text Interleaved Retrieved Examples as Contexts

###### Multiple Images – Conversations as Contexts

###### Task Description as Contexts

This task involves a wide assortment of question-answer pairs, where the model is expected to provide responses based on a given context, which could be an image, a hypothetical scenario, or a broad topic… The model is required to identify and describe objects or activities shown in a purported image, reflecting its capacity to understand and interpret visual information accurately… The format of the answers can vary, but they generally take the form of straightforward responses, detailed explanations, step-by-step process descriptions, or identified objects or activities.

Instruction: <image>…<image> How did the students advocate for immigration reform policies during the ceremony?

|[Figure 5]|[Figure 6]|
|---|---|
|[Figure 7]|[Figure 8]|

Instruction: <image> What factors contribute to the impressive performance of the dog in this image? Response: Several factors contribute to the impressive performance of the dog jumping high...

[Figure 9]

Response: The students who participated in the National DREAM Graduation ceremony advocated for immigration reform policies…

###### VST

LA-I2ILA-T2T

###### Videos – Conversations/Related clips as Contexts

Instruction: What was the purpose of the Dream Act diploma presented at the end of the ceremony?

Instruction: <image> What benefits might the dog's owner gain from this activity? Response: Engaging the dog in an activity like playing with a frisbee and running...

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

|Scene Navigation| |
|---|---|
| | |

Ego4D

TVShows

|[Figure 16]|[Figure 17]|[Figure 18]|
|---|---|---|

Instruction: <video> What task is the woman performing in the room Response: The woman is vacuuming the rug area between chairs and couches, suggesting… Instructions: What is the male character doing while sitting next to a window? Response: The male character is smoking a cigarette while sitting…

Instruction: <image>…<image> What are some options for keeping myself entertained while inside this room? <image> I want to organize a virtual art gallery tour in this room…

###### Instruction: <image>

Instruction: <video> Can you suggest any ways to brighten up the room I'm in? Response: According to my observations, there is a small potted plant and a black vase on the table, which could add some color and life to the room. Additionally, you could consider adding some colorful decor or a bright accent piece to the space. Would you like me to suggest some specific items?

[Figure 19]

| | |
|---|---|
| | |
| | |

What purpose does the large teddy bear dressed as Santa serve in this setting?

|[Figure 20]|[Figure 21]|[Figure 22]|[Figure 23]|[Figure 24]|
|---|---|---|---|---|

Response: The large teddy bear dressed as Santa serves the purpose of attracting…

DenseCaption

| |
|---|
|Spot Difference|

Instruction: <image> What does the teddy bear ornament on the Christmas tree signify? Response: The teddy bear ornament hanging on the Christmas tree conveys…

[Figure 25]

Instruction: <video> Is the camel being ridden by anyone else in the video? Response: There is no mention of anyone else riding the camel in the video, only the little girl.

Instruction: <image><image> What is the difference between the two images?

More…

Instructions: Is the little girl enjoying the experience? Response: Yes, the little girl appears to be enjoying the experience, as she smiles while riding the camel around the pen.

Fig. 1: Otter, built upon the Flamingo architecture, leverages in-context instruction tuning to align with Flamingo’s upstream image-text interleaved pretraining paradigm. The MIMIC-IT dataset consists of 3 million multimodal instruction-response pairs, where each instruction is paired with multimodal contexts. This enables LMMs trained on MIMIC-IT to exhibit strong capabilities as general-purpose assistants. The specific data format of MIMIC-IT is shown in Figure 2.

diverse array of multi-modal contextual information for each instance. MIMIC-IT’s broad and diverse scenario coverage enables Otter to excel in complex tasks involving video and multi-image comprehension. The dataset’s rich contextual variety forms the backbone of the model’s proficiency in these domains, highlighting its adaptability and wide-ranging applicability.

images and videos.

• Comprehensive Evaluations and Analysis: Our analysis reveals significant benefits of multi-modal in-context instruction tuning using the MIMIC-IT dataset, enhancing Otter’s training efficiency, generalization, and performance. MIMIC-IT’s high-quality and diversity improves Otter’s robustness in image and video evaluation benchmarks and its proficiency in diverse video and multi-image tasks, demonstrating its potential of serving as multi-mdoal general purpose assistant.

Also as evidenced in Sec. 6, in-context instruction tuning significantly boosts Otter’s training convergence and generalization capabilities. This is reflected in the our analysis and Otter’s superior performance across various benchmarks. We concretize our contributinos as follows:

2 RELATED WORK

- • Otter Model: We introduce the Otter, a multi-modality model crafted to leverage both textual and visual incontext examples during instruction tuning. Otter’s approach to in-context instruction tuning echoes and amplifies the image-text interleaved pretraining objectives common in multi-modality models, thereby enhancing the training convergence and augmenting model capacities.
- • MIMIC-IT Dataset: To complement Otter’s capabilities and advance multi-modality model research, we present the MIMIC-IT dataset. This comprehensive dataset encompasses over 3 million multimodal instructionresponse pairs, including approximately 2.2 million unique instructions, covering a wide array of real-world

2.1 In-Context Learning

In-context learning (ICL) in LLMs facilitates task adaptation through contextual input, leveraging inherent knowledge without the need for retraining [11], [23]. The mechanics of ICL, however, remain underexplored. Studies suggest that LLMs implicitly undergo meta-learning during this process [2], [26], [32], [62], [84], although evidence exists of performance resilience despite random label assignments in demonstrations. Instruction tuning improves the multitasking abilities of LLMs [24], [65], [68], [73], [86], [87], yet it is unclear whether these capabilities are intrinsic to pretrained models or acquired during tuning. Research indicates that noisy instances generated by pretrained LLMs

###### 3.2 Learning

can be effectively utilized in instruction tuning, suggesting inherent connections between ICL and instruction tuning [34], [86], [92].

We train Otter on visual instruction tuning data, denoted as D = {(Xv,Ti,Tr)} with token-level supervised finetuning [68], [82]. During the training phase, the perceiver resampler processes images Xv, converting them into visual tokens. These tokens are then mixed with the text modality within the language model’s layers. These visual tokens condition subsequent layers via cross-gated attention modules. The training paradigm aligns with the next-token prediction objective, characteristic of the GPT series models [11], [65], with the integration of visual and textual inputs. The likelihood of a targeted generated response Tr is formulated as:

Recent studies have concentrated on instruction tuning in multimodal models, enabling them to process and respond to multi-modal instructions [4], [7], [28], [56], [99]. In this domain, the integration of ICL with instruction tuning remains unexplored except that Flamingo [4] has exhibited robust few-shot ICL capabilities after pretraining on imagetext interleaved webpage data.

###### 2.2 Multi-modal Instruction Tuning Dataset

The concept of instruction tuning in multi-modal models was first presented in Multi-Instruct [90], covering a broad spectrum of tasks involving visual comprehension and multi-modal reasoning, including Visual Question Answering [33], [43], [108]. Mini-GPT4 [106] followed by creating an instruction-based dataset combining Conceptual Caption [13], [77], SBU [66], and LAION [75] with custom instruction templates. More recently, LLaVA-Instruct-158K [56] enhanced instruction tuning datasets by integrating selfinstruct and ChatGPT [65] methodologies with manually crafted seed instructions on COCO images [52]. Follow up works extend in various aspects such as text richness [101], counter-examples [53], multi-images [103], larger-scale [102], 3D modality [93], fine-grained detail [18] and including videos [64].

L

p(tl | Xv,Ti,Tr,<l). (1)

p(Tr | Ti,Xv) =

l=1

Here, Ti represents the instruction tokens, and Tr,<l signifies the sequence of response tokens preceding the current token tl. In the inference stage, the language decoder’s text tokenizer translates these tokens into coherent natural language. Based on our preliminary experiments and the aim for training stability, Otter is primarily categorized into two variants based on the differing inputs of Xv. The specifics of these two variants are detailed as follows:

###### Task Description

This task involves a wide assortment of question-answer pairs, where the model is expected to provide responses based on a given context, which could be an image, a hypothetical scenario, or a broad topic…

3 OTTER MODEL

###### In-Context Examples

The Otter model, inspired by the Flamingo framework [4], integrates pre-trained vision models (CLIP [70]) and language models (e.g. OPT [100], LLaMA [83]). Flamingo’s training utilizes interleaved image-text data from web sources [1], [107] and employs a perceiver resampler architecture for processing visual inputs. Despite Flamingo’s unavailability, an opensourced replicate, the OpenFlamingo [7] project reproduced its architecture and training methods and provided pretrained model weights.

<image> User: {instruction} GPT:<answer> {response} <|endofchunk|>

... <image> User: {instruction} GPT:<answer> {response} <|endofchunk|>

Query Example <image> User: {instruction} GPT:<answer> {response} <|endofchunk|>

Fig. 2: Instruction tuning template for the Otter model, where the targeted output in the decoding process is colored.

Otter-Image processes visual inputs Xv as tensors ([N,

T, C, H, W]), where N is the number of images, T the frame count for videos, and [C, H, W] the typical image dimensions ([3, 224, 224]). It enriches in-context instruction tuning by using the N dimension for contextual images, aligning with instruction-response pairs ( Figure 2). The model utilizes User and GPT role labels for improved dialogue interaction.

###### 3.1 Architecture We built Otter model based on a pretrained OpenFlamingo9B. The crux of Otter’s implementation1 is efficiently integration of two critical components: MPT-7B language decoder [81] and CLIP VIT-L/14 vision encoder [71]. These two modules are connected via the perceiver resampler and the cross-gated attention modules as adopted in Flamingoseries models [4]. Initially, the perceiver resampler module ingests a sequence of image or video features to produce a fixed set of visual tokens. Subsequently, these tokens condition the language layers through cross-gated attention modules, where these tokens act as keys and values while text from preceding layers serves as queries. Cross-gated attention enables that language layers is kept intact at initialization for improved stability and performance.

Otter-Video extends Otter-Image’s functionality to dynamic content by leveraging the T dimension for sequential frame processing. This feature allows Otter-Video to interpret video inputs as temporally coherent frames with added time embeddings.

#### 4 MIMIC-IT DATASET

The MIMIC-IT dataset consists of visual instruction tuning data, represented as D = (Xv,Ti,Tr). Each instance includes a set of N images, forming a tuple: (Tqi,Tqr,Xqv), where xNj=1 ⊆ Xqv. Here, Tqi denotes the q-th instruction, Tqr the response, and Xqv the associated images or videos 2.

To optimize efficiency and accelerate experimental iterations, we only train the perceiver resampler and cross-gated attention modules within the pretrained OpenFlamingo-9B, collectively encompassing 1.4B parameters.

1. When unspecified, Otter primarily refers to the Otter-9B model, with specific Image and Video versions indicated by additional suffixes.

2. Videos are seen as sequential images.

The primary objective within the Otter model framework (Sec. 3) is modeling the probability pθ(Tqr | (Tqi,Xqv)) with parameters θ. The model is tasked with generating the response Tqr for each query (Tqi,Xqv), exemplifying the standard instruction tuning procedure in a visual language model.

A set of in-context examples is defined as (Tki ,Tkr,Xkv)Mk=1, where M is the count of such sets.

The context function Cψ : (Tqi,Xqv)  → (Tki ,Xkv)Mk=1 represents these in-context examples in relation to the

current query. Consequently, the dataset format integrates query examples with corresponding in-context examples as follows:

dq = (Tqi,Tqr,Xqv,Cψ(Tqi,Xqv)), dq ∼ DMIMIC-IT (2) The visual language model, now incorporating in-context

examples, is denoted as pθ(Tqr|(Tqi,Xqv,Cψ(Tqi,Xqv))). The context function Cψ, being task-specific, necessitates different organizational strategies for in-context examples depending on the current query.

The construction of the MIMIC-IT dataset entails extracting data from source images/videos and existing annotations, employing the self-instruct [85] approach. This process tasks ChatGPT with generating diverse questions based on the provided information and formulating its own responses. We guide ChatGPT to assume different roles in assorted scenarios using System Messages and In-Context Examples, creating data tailored to specific situations. We have an automatic pipeline, named Syphus, that assists us of this process. However, due to space constraints in the main paper, the intricacies of this method are detailed in the appendix. Additionally, the MIMIC-IT dataset utilizes various methods for constructing these examples or contextual information within its subsets, which will be further explicated in the subsequent section.

###### 4.1 Data Exploration

To enrich large multi-modal models, we developed eight diverse subsets within the MIMIC-IT dataset. These encompass scenarios like responding to instructions with examples (LAT2T/I2I), analyzing image sequences (VST), and comparing images (GSD, SD). Additionally, they help in understanding TV show narratives (TVC) and enable a first-person view assistant (E4D) to interact using video inputs, improving user experience in daily tasks.

LLaVA-I2I/T2T (LA-I2I/T2T) Since LLaVA-Instruct [56] predominantly features single image-instruction-response pairs, for LA-I2I/T2T in MIMIC-IT, we aim to augment its utility for multi-modal models like Otter and Flamingo by generating in-context examples. We utilize CLIP similarity for instance matching with the following two approaches. For image-to-image matching within a MIMIC-IT dataset subset, we compute CLIP similarity scores and select the top-K similar images. Similarly, for instruction-based matching, the top-K examples are identified using text-to-text similarity. For a given query example dq, the in-context sets DI2I/T2T = {d1,d2,...,dK}, crafted for LA-I2I and LA-T2T, are composed of individual examples di = (Tci,Tcr,Xcv).

General Scene Difference (GSD). Learning to discern differences between images is vital for understanding real-world changes. MIMIC-IT encompasses two kinds of discerning difference tasks. The first type, General Scene Difference, involves creating a pair of images by determining the most similar one to the current image, utilizing image-to-image CLIP similarity from the COCO [52]. In GSD, we leverage the original image captions and object detection annotations to prompt ChatGPT with self-instruct strategy to generate instructions to ask about the difference between given two images.

Subtle Difference (SD). Besides general difference, we also create Subtle Difference (SD), features pairs of similar images with subtle distinctions sourced from the Spot-the-Diff dataset [38], extracted from surveillance footage. We prompt ChatGPT to generate the instruction-response pairs according to original descriptions provided by dataset, focusing on identifying differences between the paired images.

Visual Story Telling (VST). To enhance LMMs’ context comprehension and narrative creation, we design the task Visual Story Telling, where source images and annotations are from [35], comprising sequential images of an event (e.g. presidential selection) and corresponding inquiry questions. Annotations provide narratives and timelines not evident in the images alone. We direct ChatGPT to interpret these images and respond to related queries conversationally. The task also includes challenging questions to foster logical thinking and creativity. Notably, each instance comprises multiple image inputs and relevant instruction-response pairs, with prior conversations acting as contextual examples. Scene Navigation (SN). To demonstrate virtual assistants’ planning skills, we use 2D photos of indoor scenes, derived from RGB-D images in ScanNetv2 [25]. We generate firstperson view 2D representations of rooms and apply ScanRefer’s visual annotations [15]. ChatGPT then formulates instructions for human interaction in these settings. The process involves ChatGPT creating a room owner persona and plans compatible with both the persona and room layout, ensuring context-sensitive user assistance. Like VST, SN instances comprise image sequences and corresponding instruction-response pairs.

Dense Captions (DC). To enhance video comprehension, the DC subset integrates dense captions by Krishna et al. [41] with clips from longer videos, converted into 1FPS image frames. ChatGPT creates self-guided instructions based on questions covering video content aspects like visuals, human actions, event sequences, and causal links. Each DC instance includes sequential images with several instruction-response sets, serving as mutual in-context examples.

TV Show Captions (TVC). To improve LMMs’ social reasoning, TVC pairs TV show clips with captions, using resources from [46]. These clips facilitate the analysis of complex character interactions and social dynamics. LMMs are tasked with interpreting these narratives, investigating character motivations and relationships. This focus on social interactions and plot understanding prepares LMMs for realworld scenarios and varied user queries. Like VST, a TVC instance includes sequential images and associated instructionresponse pairs, each serving as contextual examples for the others.

Ego4D (E4D). E4D leverages egocentric video data to simu-

- TABLE 1: Comparative Analysis of MIMIC-IT and Other Multi-Modal Instruction Datasets. MIMIC-IT distinguishes itself with several notable characteristics: (1) Extensive size, encompassing a corpus of 3M instruction tuning examples specifically designed for LMMs; (2) Inclusion of video data; (3) Support for in-context instruction tuning; (4) Multilingual capabilities;

(5) Data sourced from seven diverse contexts, including indoor and outdoor environments, TV dramas, and egocentric perspectives. In this table, lang. refers to language, vis. denotes vision, and uni. inst. signifies unique instructions.

Dataset Source/Subset In-Context Modality Video #Clips/Images #Uni. Inst. #Instances Lang. MiniGPT-4 [106] CC [14] -/- - / 134M 4 5K English LLaVA [56] COCO [52] lang./- - / 81K 156K 156K English LLAVAR [101] LAION-5B [74] lang./- - / 16K 16K 16K English LRV [54] VisText [79] lang./- - / 35K 400K 400K English M3IT [50] Combined lang./- 37.7K / 2.0M 400 2.4M Multi. SVIT [102] Visual Genome [44] lang./- - / 108.1K 4.2M 4.2M English Video-ChatGPT [64] ActivityNet-200 [12] lang./- 13K / - 45K 100K English

LA-I2I/T2T lang./vis. - / 81K 261K 156K

GCD lang./vis. - / 81K 261K 141K SD lang./vis. - / 9K 10K 16K SN lang./vis. - / 0.5K 4.8K 6.6K DC lang./vis. 16K / 1M 40K 63K VST lang./vis. - / 16K 32K 34K TVC lang./vis. 86K / 577K 86K 89K E4D lang./vis. 400K / 6.4M 1.9M 2.5M

Multi.

MIMIC-IT

Total lang./vis. 502K / 8.1M 2.2M 3M

late the experience of Augmented Reality (AR) assistants interacting in real-life settings. Prompting ChatGPT for instructional generation based on visual cues, the aim is to mirror User/AR assistant interactions. The tasks and questions formulated are designed to elicit context-sensitive responses, enhancing the practicality of the LMMs in daily user-assistant interactions, underscoring the potential in providing actionable insights in day-to-day activities. The instance and in-context examples format is similar to DC and TVC.

###### 4.2 Dataset Statistics

To examine the characteristics and diversity of the instructions (refer to Figure 3 (a)) and responses (refer to Figure 3 (b)), we analyze the verb-noun structure present in them, refering to [85]. Specifically, we employ spaCy for parsing the instructions, extracting the verb closest to the root, and retrieving its first direct noun object3. We plot the top 20 most frequently occurring root verbs alongside their top 4 direct noun objects. Our findings reveal that the sentence structure of responses exhibits greater diversity compared to that of instructions. Moreover, we demonstrate diversity in terms of the length of instructions/responses, the number of images per instruction, and the number of in-context examples per instruction, as depicted in Figure 3 (c).

###### 4.3 Safety and Ethical Considerations

In the process of curating the MIMIC-IT dataset, several ethical, privacy, and potential bias concerns arise.

Personally Identifiable Information The MIMIC-IT dataset, primarily sourced from public datasets, has the potential to contain personally identifiable information (PII). The image

3. https://github.com/explosion/spacy-models/releases/tag/en_ core_web_md-3.5.0

and video sources from COCO, DC, and VIST are derived from online web searches. Consequently, the foundational dataset creators do not bear responsibility for personal information protection. In contrast, the TVC dataset comprises scenes from TV series. The E4D videos are either captured with informed consent in controlled environments or in public spaces, with faces and other PII obscured.

In essence, while MIMIC-IT may contain PII, it is primarily sourced from the public domain or obtained with consent, addressing concerns related to personal data protection.

Gender and Race Ratio Distribution In addressing potential ethical biases in image and video sources, we conducted a race and gender classification on our source images to assess their distribution. These findings, detailed in our paper, serve as guidelines for researchers utilizing our dataset and are presented in Table 3.

Specifically, we sampled a uniform 5% from each subset, proportional to the entire dataset’s volume. For each image and video frame, we employed the model from [37] for facial recognition and race/gender classification. The classification labels covered combinations of Asian, White, Black × Man, Woman. Images with multiple faces were counted accordingly; those without faces were excluded. Our race/gender classification results for the sampled data are outlined in Table 3.

To provide context, we also present reference distributions for race and gender, selecting the United States as a comparative framework due to its inclusive racial diversity, offering more balanced proportions than some Asian or European countries. The racial and gender compositions in the United States are presented in Tables 4 and 5, respectively.

The gender composition for reference in the United States is detailed in Table 5.

Satety Alignment in Instruction-Response Pairs In our utilization of ChatGPT for generating instruction-response

- TABLE 2: Performance comparison of Otter-Image with recent open-sourced LMMs, detailing trainable parameters and instruction/response data pairs. The best is represented in bold while the second-best in underline.

|Model<br><br>|Train Param. POPE MMEPer. MMECog. MMBench MMMUval. SEEDBench MM-Vet MathVista ScienceQA|
|---|---|
|OpenFlamingo-9B [6] Idefics-9BInstruct [45] mPLUG-OwlV [91] LLaMA-Adapter [97] InstructBLIP-7B [27] InstructBLIP-13B [27] Qwen-VL-7BChat [8] LLaVA-7B1.5 [55]<br><br>|1.3B - - - 5.7 28.7 24.8 24.8 - 9.0B 74.6 187.9 1351.8 45.5 - 44.5 23.7 19.8 9.6B - - - - 32.7 34.0 - - 37.9 1.8B - - - 39.5 29.8 - - - 85.2 0.2B - - - 36.0 - 53.4 26.2 - 60.5 0.2B 78.9 291.8 1212.8 33.9 - - 25.6 25.3 63.1 9.6B - 360.7 1487.5 61.8 35.9 58.2 - - 68.2 7.0B 85.9 - 1510.7 59.5 29.2 58.6 30.5 - 66.8<br><br>|
|Otter-Image-9B|1.3B 86.5 332.5 1525.8 62.1 32.2 57.5 30.6 27.5 69.3<br><br>|

ok

item

type

loc

n

co nte nt

o

h azard

type item

purpose

erso

b

atio n

object

information

answer

conversation

detail

p

impact

view

way

benefit

infor

option

shirt

d eter

see

object

have

recipe

- m atio
- n

min e

gear

suggest

identify

item

glove

provide

wear

jacket

place

object item

take

include

see

precaution

put

explain

place

affect

hand

play

notice

pick

difference

recommend

experience

notice

picture

book

performance

play

hand

game

capture

observe

take

photo

card

pick

person

observe

bottle

depict

- d

escrib

- e

piece

give

option

book

way

sho w

book

provid e

hold

use

ature

id e

item

face

game

sign

role

sign

object

use

role

h ave

hazard

do

ntify

action

person

kind

game

item

fe

scene

tool

moment

group

person

precaution

em

object

advice

card

tennis

instruction

m

infor

man

group

street

phone

scene

g

equipment

character

phone

object

chalenge

din

tool

type

e

place

wo man

person

group

plant

ere

nc

d fcuty

ssue

n

- m atio
- n

botle

risk

c o

u

ara

man

nv ers

h

ro

ok

ac v y

h a n d

p

p

phone

tro w el

im p act

os

e

u

o

tro u

o bject

knife

ure

su

bject

- m

a

- n

kin d

p

b

p

piece

atio n

gro

e

c

ble

cu

- m

a

- n

bra n d

m

p

p

h az

e

ns

p

- d
- e a

a

at

b

ard

- g
- h

tip

ty

o

ta

o

w

(a) Instructions (b) Responses

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

(c) Statistics of instructions and responses.

- Fig. 3: The data statistics of multi-modal in-context instruction-response pairs. (a) and (b), the root verb-noun pairs of instruction and responses, where the inner circle of the plot represents the root verb of the output response, and the outer circle represents the direct nouns. (c) Statistics of instructions and responses, retaining 25% of Ego4D instructions for a more balanced distribution. # Related instructions denotes the number of related instructions in an instance, given the same set of visual input data.

- TABLE 3: Race and Gender Distribution in MIMIC-IT.

Gender

Race

Asia White Black Total

|Woman|17.3% 19.5% 9.7% 46.5%<br><br>|
|---|---|
|Man|18.4% 22.5% 12.6% 53.5%<br><br>|
|Total|35.7% 42.0% 22.3% 100.0%|

- TABLE 4: Racial Composition in the United States.

Location White Black Hispanic Asian Others Total

United States 58.2% 11.6% 19.0% 5.7% 5.6% 100.0%

pairs, we stringently adhere to Microsoft Azure’s ChatGPT4 content policy. This policy is meticulously crafted to ensure the generation of content that is both safe and ethical. It

- 4. https://azure.microsoft.com/en-in/products/ai-services/

openai-service/

TABLE 5: Gender Composition in the United States

Location Biological Woman Biological Man United States 49.28% 50.72%

serves as a pivotal measure in mitigating potential biases, stereotypes, explicit content, or misinformation, thus aligning with broader initiatives in AI ethics and safety. By implementing ChatGPT’s rigorous safety and alignment strategies, we not only uphold high standards of content safety but also actively contribute to the responsible development and deployment of AI technologies. This dedication is consistently demonstrated in the quality and reliability of the content produced by ChatGPT for our datasets.

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

In-context examples

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

##### Cold Start

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

ChatGPT

ChatGPT

Step 1 System Message + visual annotation

Step 2 Generate instructionresponse pairs

Step 3 Filtering

Step 4 Translation

Prompt

- Fig. 4: Syphus overview. We employ a cold-start stage to identify the optimal system message and in-context example for querying instruction-response pairs in a given dataset. Subsequently, Syphus, spanning steps 1 to 4, generates high-quality instruction-response pairs in eight languages.
- 5 CHATGPT-ASSISTED DATASET GENERATION

|User: Can the black and white snowboard be used for beginners? GPT: Based on my observation, there is a black<br><br>and white snowboard visible, but I cannot determine its suitability for beginners. Snowboarding can be a challenging sport, and it’s essential to choose equipment that is appropriate for your skill level. Would you like me to find more information on snowboarding equipment?<br><br>|
|---|

###### 5.1 Generation Pipeline

We present Syphus (see Figure 4), an automated pipeline for generating high-quality instruction-response pairs in multiple languages. Building upon the framework proposed by LLaVA [56], we utilize ChatGPT to generate instructionresponse pairs based on visual content and original annotations.

To ensure the quality of the generated instructionresponse pairs, our pipeline incorporates system messages, visual annotations, and in-context examples as prompts for ChatGPT. System messages define the desired tone and style of the generated instruction-response pairs, while visual annotations provide essential image information such as bounding boxes and image descriptions. In-context examples assist ChatGPT in learning within the context. Since the quality of coreset impacts subsequent data collection process [17], we employ a cold-start strategy to enhance in-context examples before the large-scale query.

TABLE 6: Example of a correct dialogue.

dialogues and scenarios, an essential attribute for practical applications in diverse real-world settings.

###### 5.3 Dataset Cleaning

After obtaining the instruction-response pairs generated by ChatGPT, the majority of the data satisfies our requirements, adhering to the format, content, style, and pronoun usage provided in the in-context examples. However, some data deviates from our specified pattern due to unforeseen circumstances. For instance, in the E4D dataset, we expect the ChatGPT-generated response to adopt a style in which the AI, acting as a first-person AR headset assistant, helps users solve problems. Consequently, the questions should be phrased in the second person, simulating a user’s inquiry, while the responses should be in the first person, simulating the model’s reply to the user’s question. Tab. 6 demonstrates an example of a correct dialogue.

During the cold-start stage, in-context examples are collected by prompting ChatGPT solely through system messages and visual annotations, employing a heuristic approach. This stage concludes only when satisfactory in-context examples are identified. In step 4, once the instruction-response pairs are obtained, the pipeline expands them into Chinese (zh), Japanese (ja), Spanish (es), German (de), French (fr), Korean (ko), and Arabic (ar).

###### 5.2 Application Scenarios Demonstration

In Figures 5 to 8, we present a diverse array of sample scenarios selected from various subsets of our dataset. These samples vividly illustrate the versatility and adaptability of our model across a range of contexts and dialogues pertinent to specific scenarios.

Nevertheless, the content generated by ChatGPT may occasionally contain errors, such as including the user (you) as shown in Tab. 7, which might result from the information provided in our query.

Each scenario is carefully chosen to represent the distinct characteristics of its respective subset, demonstrating not only the breadth of our dataset but also the depth of the model’s contextual understanding and response accuracy. These examples collectively showcase the model’s robustness in interpreting and responding to a wide spectrum of

We cross-checked each dataset with two human individuals, each person spent 15 hours to manually check the error patterns in different datasets, and, for fixable errors (such as the one above, where user (you) needs to be replaced with you), employed regular expressions to match and replace these errors with the correct expressions. For other errors that

###### Dense Caption (DC)

###### TV Caption (TVC)

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Instruction: Who is the main subject of this video? Response: The main subject of this video is a woman wearing a yellow shirt. Instruction: What is the woman doing in the video? Response: In the video, the woman is pouring a liquid from a bottle into a bucket and then uses a mop rag to mop the tile on the floor. Instruction: What does the woman do with the mop rag? Response: The woman puts the mop rag into the bucket with the liquid and then uses it to mop the tile on the floor. Instruction: Is the woman alone in the room? Response: There is no mention of other people in the room, so it is unclear whether the woman is alone or not.

Instruction: What is the male character doing while sitting next to a window? Response: The male character is smoking a cigarette while sitting next to a window. Instruction: Where is the male character looking while smoking a cigarette? Response: The male character is looking out of the window while smoking a cigarette, which might suggest that he is deep in thought or simply enjoying the view. Instruction: What task is the woman performing in the room? Response: The woman is vacuuming the rug area between chairs and couches, suggesting that she is cleaning the living space.

Fig. 5: Demonstration examples in DC and TVC.

###### Visual Story Telling (VST)

###### Ego4D (E4D)

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

Instruction: Can you describe the atmosphere at the parties in these photos? Response: The atmosphere at these parties seems to be lively and energetic, with people enjoying themselves and socializing with others. There were various forms of entertainment, such as live music and dancing, which contributed to the festive ambiance. It also appears that the parties were held in a small space, as there were annotations mentioning the crowded room and the need to leave to get some sleep. Overall, the photos convey a sense of fun and excitement that comes with being surrounded by friends and enjoying a night out on the town.

Instruction: Is there a plant visible in the footage? Response: Yes, there are multiple plants visible in my observation. There is a small potted plant on the table, a green plant in a black pot, and a flower in a black vase. Do you require further assistance? Instruction: Is there a bottle of water visible in the footage? Response: Yes, there is a clear plastic bottle of water with a green lid visible in the footage. It is located on a table near a black vase and a white piece of paper. Can I help you with anything else? Instruction: Is there a magazine visible in the footage? Response: Yes, there is a magazine visible in the footage. It is located on a table with a green book with a cartoon character on it, a clear plastic cup, and a plant in a black pot. Is there anything else you would like me to help with?

Fig. 6: Demonstration examples in VST and E4D.

|User: Is there any equipment in the garden that I could use for digging? GPT: According to what I saw, the user (you) picked up a garden pick mattock and a sledgehammer. These could be used for digging, but I cannot guarantee their availability or suitability for your needs. Is there anything else you need help with?<br><br>|
|---|

6 EXPERIMENTS

6.1 Data Mixture

Consistent with the observations made in LLaVA-1.5 [55], we found that precisely controlling the response format in instruction tuning data significantly enhances benchmark performance. This improvement is particularly notable in tasks [30], which require binary yes/no answers. In addition to MIMIC-IT, our training dataset for Otter also includes selections from academic sources to further refine its capabilities. Data from these academic sources were predominantly sourced from the M3IT Collection [50]. For all datasets, Otter was exclusively trained on the train sets, enabling a focused evaluation on the validation and test sets.

TABLE 7: Example of a wrong dialogue.

For Otter-Image model, we compiled a data mixture from the following public datasets: VG [42], VQAv2 [5], GQA [36], OKVQA [61], OCRVQA [63], A-OKVQA [76], TextQA [78], RefCOCO [94], COCO-ITM [48], ImageNet [29], ImageNetParagraphs [40]. For these datasets, we only construct the Answer Format part in task description as the prefix for each instruction.

cannot be rectified en masse, we documented their locations and stored them in a dedicated invalid.json file. During the dataset iteration process, these samples were fed into ChatGPT to repeat the query and obtain accurate and usable data.

###### General Scene Difference (GSD)

###### Spot The Difference (SD)

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Instruction: What is the difference between the two images? Response: The person in white and the person in blue are no longer in the second image. Additionally, there are several people near the building in the background in the second image that are not present in the first image. Instruction: Is there anyone new in the second image? Response: Yes, there are several people near the building in the background in the second image that are not present in the first image. Instruction: Is the person in white still there in the second image? Response: No, the person in white is gone in the second image. Instruction: Has anything changed near the umbrella tables? Response: Yes, the person in white and the person in blue are both gone near the umbrella tables in the second image.

Instruction: How are the tennis players different in these two images? Response: In the first image, the tennis player is standing with one foot on the boundary line while in the second image, the tennis player is jumping up in the air.

Instruction: what is the difference between the tennis rackets in these two images? Response: In the first image, the tennis racket is shiny and new while in the second image, only the shadow of the tennis racket is visible.

Fig. 7: Demonstration examples in GSD and SD.

###### Scene Navigation (SN)

Instruction: I would like to hold a painting night in this room. What modifications can be made to create an inspiring and creative space? Response: For a creative space, you can set up the slender table with a white cloth on it as a designated workspace for the artists. You can also use the large beige-colored couch in the shape of the letter J as a comfortable seating area for the artists to relax and chat. Additionally, consider rearranging the furniture to provide enough open floor space for easels and art supplies. What do you think? Instruction: That's a great idea. What other elements can I incorporate into the room to enhance the painting experience? Response: Lighting is key for any painting activity. You can utilize the natural light that comes through the windows during the day. In the evening, you could set up some string lights around the bookshelf and the fireplace to provide a warm and cozy atmosphere. Additionally, you could provide some refreshments and snacks on the black TV stand to keep the artists energized and inspired. Instruction: How many artists can I accommodate in this space for a painting night? Response: Based on the size of the room and furniture, you could comfortably accommodate around 10-12 artists for a painting night, provided the layout is optimized for the activity. However, it's always a good idea to do a trial run with a smaller group to ensure that the space is suitable for your needs.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Instruction: Do you have any recommendations for how I can make the most of this room's potential for activities? Response: I have every confidence that Painting Night is a wonderful indoor activity. The open layout of the room and the natural lighting make it an ideal location for a painting night.

Fig. 8: Demonstration examples in SN.

TABLE 8: Performance comparison of Otter-Video with recent open-sourced LMMs on video question-answer tasks. The best is represented in bold while the second-best in underline.

For Otter-Video model, we trained it on the DC/TVC subsets from MIMIC-IT, with each video was initially sampled and stored into the dataset at 1 FPS. During batch-wise training, we uniformly extracted 64 frames for training (if a video had fewer frames, adjacent frames were repetitively sampled). We also used instruction tuning data from MSVD [21] and MSRVTT [89], where training data from these sources were limited to only 8 frames per clip.

|Model<br><br>|MSVD-QA ↑ MSRVTT-QA ↑ ActivityNet-QA ↑|
|---|---|
|LLaMA-AdapterV [98] Video-ChatGPT [64]<br><br>VideoChat [49] mPLUG-OwlV [91]|32.6 35.7 30.5 49.1 48.7 47.2 47.6 48.0 48.4 64.1 66.2 47.4<br><br>|
|Otter-Video|77.9 68.4 68.5|

###### 6.2 Benchmark Performance

Image Benchmarks In Tab. 2, we present a comprehensive comparison between Otter-Image and other state-of-theart LMMs across a variety of benchmarks. We present performance in accuracy on benchmarks including POPE [51], MM-Vet [95], MMBench [59], MathVista [60]. On MMBench, we report results on test set. For MME [30], we report the aggregated scores in cognitive and perception to follow its evaluation convention.

and ActivityNet-QA [96]. This evaluation employs a zeroshot approach using ChatGPT-assisted scores to evaluate model proficiency. The focus is on measuring the precision of the model’s predictive responses. Additionally, we include a few-shot evaluation results of MSRVTT-Caption, as well as the evaluation details in the appendix for comprehensive coverage.

Video Benchmarks In Tab. 8, we present a detailed quantitative assessment across several leading open-ended questionanswer video datasets: MSRVTT-QA [89], MSVD-QA [21],

- TABLE 9: Performance gains through task description enhancements.

Loss Curves/Base Models

w/TaskDesc

1.2

wo/TaskDesc

|Method<br><br>|MME-Cog ↑ MM-Vet ↑|
|---|---|
|Base Model|1212.9 23.7<br><br>|
|+ Question Coverage + Model Ability + Answer Format<br><br>|1315.2 24.5 1329.3 25.8 1414.5 22.3|
|Eval w/Full Desc.<br><br>|1415.7 27.0|

1.0

TrainingLoss

0 5000 10000 15000 20000 25000 30000

Loss Curves/Instruct Models

1.0

w/TaskDesc

###### 6.3 Benefiting from Contextual Information

wo/TaskDesc

From the above-described instruction template, it is evident that the multi-modal in-context instruction tuning (ICIT) process involves incorporating additional contextual information, namely task description and in-context examples, before each query example. This strategy, when compared to other instruction tuning data approaches, potentially enriches the context for the current query example and aids in steering the model towards generating the targeted response. To verify this intuition, we conducted a series of ablation studies on the effect of adding contextual information in following sections. Task Description For efficiency, our ablation study only uses the LLaVA-Instruct-158K dataset [56], categorizing it into three subsets: Detailed Description, Conversation, and Complex Reasoning. The task descriptions on each subset are constructed from three dimensions: (1) Question Coverage. (2) Model Ability. (3) Answer Format. The three parts would prepend to each instruction as prefix to aid model better comprehend the instructions.

0.5

0.0

0 1000 2000 3000

Batch Steps

- Fig. 9: Comparative analysis of training convergence with and without task descriptions.

0 5000 10000 15000 20000

Batch Steps

0.5

1.0

TrainingLoss

Loss Curves

w/2-shot In Context Examples

wo/In Context Examples

- Fig. 10: Comparative analysis of training convergence with and without 2-shot in-context examples, illustrating the impact on the model’s learning trajectory.

Figure 9 compares model convergence with and without task descriptions during training. The figure is split into upper and lower sections: the upper shows the loss curve during finetuning from an OpenFlamingo base model, with a focus on the impact of task descriptions. Initially, both methods show similar loss, but after 20K steps, they begin to diverge. Our initial hypothesis was that training loss differences would be minimal. However, an experimental pivot led us to further explore this using the better-performing ’instruct models’ from the first training phase. This revealed a growing loss divergence over time, with task description-inclusive models showing lower loss and improved convergence.

improvement in MM-Vet is derived from specifying the capabilities on which the model relies for answering questions, with a notable increase (23.7 → 25.8).

In-Context Examples As detailed in Sec. 4, apart from task descriptions, we incorporate contextual information through retrieval-based in-context examples. In our ablation study, we employed LA-Interleaved subsets from the MIMIC-IT dataset, with in-context examples retrieved using both textto-text (T2T) and image-to-image (I2I) similarity matching. Our findings, presented in Figure 10, highlight the variations in convergence speed when employing 2-shot in-context examples.

We also present ablation results across various benchmarks, focusing on response format optimization for benchmark performance. To aid the model learn to respond in certain answer format, two subsets were constructed: one (830 examples) from COCO [52] for object presence queries (Yes/No responses), and another (2000 examples) from RefCOCO [39] for identifying objects from options. Both subsets were created through rule-based construction from annotations or direct sampling from data samples.

We further demonstrate the performance improvements on benchmarks achieved through in-context instruction tuning, as shown in Table 10. The columns labeled 0, 2, and 4 shots in the table correspond to the number of in-context examples included during the evaluation phase. To circumvent information leakage in the MME test set, we crafted in-context examples from web images, aligning the instructions and answers with the dataset’s required format. The rows in the table represent the varying quantities of in-context examples used during the instruction tuning process.

We selected two benchmarks with different objectives for evaluation. MME-Cog is the Cognition subset of MME, featuring questions about art, celebrities, locations, etc., with answers limited to yes or no. MM-Vet focuses more on logical reasoning and basic mathematics, with a freeform answer format. The correctness of responses is assessed using GPT4. The model’s ablation results on these benchmarks are presented in Tab. 9. It is evident that controlling the Answer Format through task description during training particularly enhances performance on MME (1212.9 → 1414.5). Such effect is not observed in MM-Vet. However, the most significant

In few-shot evaluations, base models like Flamingo and OpenFlamingo, without instruction tuning, often show improved performance with more in-context examples due to ICL capabilities. Our ablation study, illustrated in Figure 11,

- TABLE 10: Ablations of in-context examples on MME-Cog.

###### In-Context Examples

###### Query Example

[Figure 95]

[Figure 96]

[Figure 97]

|Method<br><br>|MME-Cog ↑ 0 2 4|
|---|---|
|Train wo/ICIT|1212.9 1310.7 1173.5<br><br>|
|Train w/2-shot Train w/4-shot<br><br>|1227.7 1430.3 1397.8 1277.8 1387.9 1440.8|

Instruction: What’s written on this image? (wo/ICIT) R: “Think Different” is written on this image. (w/ICIT) R: Think Different.

Instruction: What’s written on this image? Response: pepsi, is pepsi okay?

Instruction: What’s written on this image? Response: SUBWAY, eat fresh.

Flamingo-9B

OpenFlamingo-9B Otter-9B (wo/ICIT) Otter-9B (w/ICIT)

110

- Fig. 12: Comparative demonstrations of Otter with and without in-context instruction tuning (ICIT).

Otter-Video/on DC

Instruction: Hey Otter, do you think it's cool? Response: Yes, I think it is cool that the man is playing video games while wearing a head-mounted display.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Otter-Video/on E4D

Instruction: Is it safe to walk on the floor while the woman is cleaning? Response: Based on what I saw, it's best to avoid walking on the floor

while it's being cleaned to prevent slipping or accidentally disrupting the cleaning process. Would you like me to suggest ways to stay safe while the floor is being cleaned

[Figure 104]

[Figure 105]

[Figure 106]

- Fig. 13: Demonstrations of Otter’s proficiency in different scenarios

CIDErScore

100

90

80

0 4 8

Number of example shots

Fig. 11: Few-shot performance on COCO Caption.

explores this phenomenon in multimodal instruct models. Notably, Otter (wo/ICIT) excel in 0-shot scenarios but falter as examples increase, implying a diminished ICL capacity when fine-tuned on single image-instruction-response pairs. Conversely, Otter (w/ICIT) markedly improve with more shots, underscoring ICIT’s importance in preserving ICL effectiveness in multimodal settings. More few-shot evaluation results are provided in appendix.

To clarify how ICIT impacts model behavior, Figure 12 demonstrates how referencing in-context examples influences the trained model’s text formatting control.

###### 6.4 Diverse Applications

MIMIC-IT extends Otter’s capabilities beyond image understanding to diverse applications.

Video Captioner and Assistant. MIMIC-IT’s extensive video collection, featuring dialogues on content and character relationships, enriches Otter’s functionality. This includes YouTube content and series like Friends in the DC and TVC subsets. Trained models act as video assistants, enhancing user interaction and enabling detailed video content annotation for research.

ChatGPT. These descriptions serve as prefixes placed before the instruction/response pairs for each task, and brings potential benefits during the mixed training process of large data mixture to assist the model in understanding the required capabilities and response formats for different tasks. In this section, we provide the detailed information in generating task descriptions for each subsets.

Egocentric Visual Assistant. Additionally, MIMIC-IT’s egocentric videos and images, particularly from SN and E4D scenarios, focus on indoor navigation and planning. These datasets improve Otter’s first-person scene interpretation and strategy formulation, especially for AR headset applications. The Otter model variant, tailored for egocentric views, marks a significant advancement in AR headset visual language modeling.

For each dataset, we select 20 to 50 data entries (totaling less than 2048 tokens) for ChatGPT5 to summarize. The summarization by ChatGPT is requested to focus on three dimensions: (1) Coverage, (2) Ability, and (3) Answer Format. The query prompt used and the generated task description on MIMIC-IT’s VST subset are as in Tab. 14

In Figure 13, we showcase Otter’s adaptability in both scenarios, demonstrating its capabilities of responding to various instructions.

###### 6.5 Further Analysis

Task Descriptions For different subsets of MIMIC-IT, we generated task-specific descriptions with the assistance of

5. Unless otherwise specified, the term ChatGPT throughout our paper refers to GPT-4-0613 model.

- TABLE 11: Summary of hyper-parameters of OtterImage/Video.

|Hyper-parameters|Otter-Image Otter-Video<br><br>|
|---|---|
|Batch Size/GPU|8 4<br><br>|
|LR|1e-5<br><br>|
|LR Schedule<br><br>|cosine decay|
|LR Warmup Ratio|0.03<br><br>|
|Epoch<br><br>|6|
|Sample Frames<br><br>|- 8|
|Optimizer<br><br>|AdamW|
|DeepSpeed|Zero2<br><br>|
|Peak GPU Mem.<br><br>|∼70G ∼76G|

- TABLE 12: Ablations on adding data from other sources.

Q: what is the gender of the athlete? A: the athlete is a male.

Prompt: You are a helpful AI assistant. Your task is to score the model response on a scale of 0 to 100 based on how well it matches the ground truth answer

to the question.

Question: what is the gender of the athlete? GT: the athlete is a male. Response: The athlete shown in the video is male. Score: 100

Question: what happened after the billiards? GT: they chatted after playing billiards. Response: After the man finishes explaining the game , the video shows clips of a man hitting the balls into the pockets on the table. Score: 0

|Data Sources<br><br>|MMBench MM-Vet|
|---|---|
|MIMIC-IT Subsets∗|41.5 19.7<br><br>|
|w/LLAVAR|43.2 23.2<br><br>|
|w/Academic Datasets<br><br>|62.1 30.6|

and complex instruction-response pairs across diverse scenarios. Empirical results demonstrate Otter’s superior training efficiency and performance in multimodal tasks.

Otter has significant potentials to serve as a generalpurpose assistant, yet it still has limitations in certain scenarios like fine-grained OCR tasks due to the relatively low-res (224×224) of the pretrained vision encoder. Future research [9], [47] on this, particularly in enabling models to more accurately perceive information from visual inputs, is highly anticipated.

Hyperparameters We provide hyper-parameters related information at Tab. 11. The Otter models for Image and Video fundamentally employ similar hyperparameters. However, due to the need to accommodate multiple images (frames) in the Video model, it is necessary to reduce the corresponding batch size. We didn’t conduct hyper-parameters search, the default configuration is sufficient to train good enough models for both image and video scenarios.

#### ACKNOWLEDGEMENT

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP20221-0012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

Video Evaluations In main paper, we provide the evaluation results on multiple Video-QA datasets. Here we elaborate on how we do the evaluations with ChatGPT as evaluator, as well as present results related to ChatGPT evaluation scores and other traditional metrics on Video-Caption tasks.

Directly calculating matching accuracy may involve misconceptions during evaluations and can not reflect models true abilities to provide correct answers the questions. Consequently, many recent benchmarks [59], [88], [95] proposed to use ChatGPT as an evaluator to score by comparing the model’s responses with the ground truth, and we have adopted this method as well in our evaluations for QA and Captioning tasks. In the following example, we demonstrate the query prompt used in the ChatGPT Evaluation process, as well as samples where ChatGPT scores are respectively 100 and 0. In the case of the 100-point model response, it precisely captures the information required in the Ground Truth (GT). In contrast, the 0-point response exhibits hallucination, resulting in a failure to score.

Caption Results Subsequently, in Tab. 13, we provide results with both traditional metrics (BLEU, METEOR and ROUGE_L) and ChatGPT scores for audience reference.

#### 7 CONCLUSION

We presented the Otter model, a multi-modal model with in-context instruction tuning, driven by the comprehensive MIMIC-IT dataset with multi-modal contextual information

TABLE 14: Task description example of MIMIC-IT’s VST subset.

Prompt: I am writing a task description for a dataset. Can you take a look at few data from this task and provide me with the task description? The task description is used to describe the data to a vision language model. You may give a general summarization from the three perspectives (1) what is the questions and answers coverage of this task? (2) what ability needed to provide answers of given instructions in this task? (3) what is the general answer format of this task. Remember you can only observe a few items of this dataset so do not assume

the data you observe is the whole, and keep your task description be general and summarized. You will be given multiple instances from the dataset in question/answer format. Please generate two versions

of task description, one is short (less than 100 words) and one is long (less than 500 words).

- TABLE 13: Comparative results for captioning tasks on MSVD and MSRVTT with both traditional metrics and ChatGPT score. B-1: BLEU-1, M: METEOR, R: ROUGE_L, G: ChatGPT Score

|Model<br><br>|MSVD-Cap MSVTT-Cap B-1/M/R G B-1/M/R G|
|---|---|
|Video-ChatGPT|12.9/15.8/21.4 46.4 10.5/20.5/33.7 33.1<br><br>|
|mPLUG-Owl-V|13.0/14.2/19.5 51.5 10.5/23.9/41.5 39.0<br><br>|
|Otter-Video|33.1/14.3/21.0 54.3 58.3/32.6/59.0 67.9|

Question: What color are the sheep? Answer: The sheep are white. Question: What are some of the toppings visible on the pizza?

Answer: Some of the toppings visible on the pizza include shrimp, basil, and possibly tomato sauce. ...

Short: This task involves answering questions based on detailed descriptions, observations, experiences,

or events that the participants in the dataset have encountered. The questions cover a wide range of

topics, from personal impressions of places and events to specific activities and historical significance. Answers require understanding of context and ability to provide informative responses

based on the described scenarios. Answers are in several formats, including direct responses, narrative experiences, and explanatory comments. Long: This task covers questions related to various experiences, observations, or events as relayed by individuals or inferred from provided details. To address each data item effectively, you need a thorough comprehension of nuanced context and the ability to summarize experiences or observations. Answers might need to encapsulate personal reflections, summarize events, or describe the purpose and significance of activities. The answer format varies: some are narrative, detailing personal experiences or encounters; some are descriptive, painting a vivid picture of a scene

or an event; and some are explanatory, providing reasons or background information about a subject. You should tailor each answer to the specifics of the question, ensuring it is informative and accurately reflects the gist of the provided data, whether it conveys a subjective opinion, outlines a factual account, describes an experience, or explains the purpose of an event or organization.

#### REFERENCES

- [18] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024. 3
- [19] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565, 2023. 1
- [20] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Soricut. Pali-3 vision language models: Smaller, faster, stronger, 2023. 1
- [21] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 9
- [22] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. 1
- [23] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. 2
- [24] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022. 1, 2
- [25] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richlyannotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 4
- [26] Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei. Why can gpt learn in-context? language models secretly perform gradient descent as meta optimizers. arXiv preprint arXiv:2212.10559, 2022. 2
- [27] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500,

2023. 1, 6

- [28] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. CoRR, abs/2305.06500, 2023. 1, 3
- [29] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li FeiFei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255. Ieee, 2009. 8
- [30] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394,

2023. 8, 9

- [31] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023. 1
- [32] Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022. 2
- [33] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 3
- [34] Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. Unnatural instructions: Tuning language models with (almost) no human labor. arXiv preprint arXiv:2212.09689, 2022. 3

- [1] Common crawl. https://commoncrawl.org/. Accessed: 2023-11-

11. 3

- [2] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661,

2022. 2

- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022. 1
- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022. 3
- [5] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015. 8
- [6] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023. 6
- [7] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390,

2023. 1, 3

- [8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 1, 6
- [9] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sag˘nak Tas¸ırlar. Introducing our multimodal models, 2023. 12
- [10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1
- [11] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems (NeuIPS), 33:1877–1901, 2020. 2, 3
- [12] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015. 5
- [13] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558– 3568, 2021. 3
- [14] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558– 3568, 2021. 5
- [15] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. 16th European Conference on Computer Vision (ECCV),

2020. 4

- [16] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 1
- [17] Liangyu Chen, Yutong Bai, Siyu Huang, Yongyi Lu, Bihan Wen, Alan L Yuille, and Zongwei Zhou. Making your first choice: To address cold start problem in vision active learning. arXiv preprint arXiv:2210.02442, 2022. 7

- [35] Ting-Hao K. Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Jacob Devlin, Aishwarya Agrawal, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2016), 2016. 4
- [36] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 8
- [37] Wondong Hyeon. Face Classification and Detection. https:// github.com/wondonghyeon/face-classification, 2023. Accessed: 2023-11-25. 5
- [38] Harsh Jhamtani and Taylor Berg-Kirkpatrick. Learning to describe differences between pairs of similar images. In EMNLP, pages 4024–4034. Association for Computational Linguistics, 2018. 4
- [39] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 10
- [40] Jonathan Krause, Justin Johnson, Ranjay Krishna, and Li FeiFei. A hierarchical approach for generating descriptive image paragraphs. In Computer Vision and Patterm Recognition (CVPR),

2017. 8

- [41] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715,

2017. 4

- [42] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. corr abs/1602.07332, 2016. 8
- [43] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 3
- [44] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 5
- [45] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. 1, 6
- [46] Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. Tvr: A largescale dataset for video-subtitle moment retrieval. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXI 16, pages 447–463. Springer,

2020. 4

- [47] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A high-resolution multi-modality model. arXiv preprint arXiv:2311.04219, 2023. 1, 12
- [48] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597,

2023. 1, 8

- [49] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chatcentric video understanding. arXiv preprint arXiv:2305.06355, 2023. 9
- [50] Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, et al. M3it: A large-scale dataset towards multi-modal multilingual instruction tuning. arXiv preprint arXiv:2306.04387, 2023. 5, 8
- [51] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large visionlanguage models. arXiv preprint arXiv:2305.10355, 2023. 9
- [52] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 3,

4, 5, 10

- [53] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 3
- [54] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning, 2023. 5
- [55] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 6, 8
- [56] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 1, 3, 4, 5, 7, 10
- [57] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 1
- [58] Shikun Liu, Linxi Fan, Edward Johns, Zhiding Yu, Chaowei Xiao, and Anima Anandkumar. Prismer: A vision-language model with an ensemble of experts. arXiv preprint arXiv:2303.02506, 2023. 1
- [59] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 9, 12
- [60] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 9
- [61] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195– 3204, 2019. 8
- [62] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837, 2022. 2
- [63] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, 2019. 8
- [64] Salman Khan Muhammad Maaz, Hanoona Rasheed and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. ArXiv 2306.05424, 2023. 3, 5, 9
- [65] OpenAI. Gpt-4 technical report. https://openai.com/research/ gpt-4, 2023. 1, 2, 3
- [66] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011. 3
- [67] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural

- Information Processing Systems, 35:27730–27744, 2022. 1

[68] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural

- Information Processing Systems, 35:27730–27744, 2022. 2, 3

- [69] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with GPT-4. arXiv preprint arXiv:2304.03277, 2023. 1
- [70] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [71] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), pages 8748–8763. PMLR,

2021. 3

- [72] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 1
- [73] Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207,

2021. 2

- [74] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294,

2022. 5

- [75] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 3
- [76] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pages 146–162. Springer, 2022. 8
- [77] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 3
- [78] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 8
- [79] Benny J Tang, Angie Boggust, and Arvind Satyanarayan. Vistext: A benchmark for semantically rich chart captioning. arXiv preprint arXiv:2307.05356, 2023. 5
- [80] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023. 1
- [81] MosaicML NLP Team. Introducing mpt-7b: A new standard for open-source, commercially usable llms, 2023. Accessed: 2023-05-05. 3
- [82] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [83] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [84] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pages 35151–35174. PMLR, 2023. 2
- [85] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Selfinstruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022. 1, 4, 5
- [86] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085– 5109, 2022. 1, 2, 3
- [87] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In ICLR. OpenReview.net, 2022. 1, 2
- [88] Binzhu Xie, Sicheng Zhang, Zitang Zhou, Bo Li, Yuanhan Zhang, Jack Hessel, Jingkang Yang, and Ziwei Liu. Funqa: Towards surprising video comprehension. arXiv preprint arXiv:2306.14899,

2023. 12

- [89] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017. 9
- [90] Zhiyang Xu, Ying Shen, and Lifu Huang. Multiinstruct: Improving multi-modal zero-shot learning via instruction tuning. arXiv preprint arXiv:2212.10773, 2022. 3

- [91] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chaoya Jiang, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality, 2023. 6, 9
- [92] Seonghyeon Ye, Hyeonbin Hwang, Sohee Yang, Hyeongu Yun, Yireun Kim, and Minjoon Seo. In-context instruction learning. arXiv preprint arXiv:2302.14691, 2023. 3
- [93] Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Xiaoshui Huang, Zhiyong Wang, Lu Sheng, Lei Bai, et al. Lamm: Language-assisted multi-modal instructiontuning dataset, framework, and benchmark. Advances in Neural Information Processing Systems, 36:26650–26685, 2023. 3
- [94] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 8
- [95] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 9, 12
- [96] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering, 2019. 9
- [97] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023. 6
- [98] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention, 2023. 9
- [99] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023. 1, 3
- [100] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. 3
- [101] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023. 3, 5
- [102] Bo Zhao, Boya Wu, and Tiejun Huang. Svit: Scaling up visual instruction tuning. arXiv preprint arXiv:2307.04087, 2023. 3, 5
- [103] Haozhe Zhao, Zefan Cai, Shuzheng Si, Xiaojian Ma, Kaikai An, Liang Chen, Zixuan Liu, Sheng Wang, Wenjuan Han, and Baobao Chang. Mmicl: Empowering vision-language model with multimodal in-context learning. arXiv preprint arXiv:2309.07915, 2023. 3
- [104] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1
- [105] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision (IJCV), 2022. 1
- [106] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1, 3, 5
- [107] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal C4: An open, billion-scale corpus of images interleaved with text. arXiv preprint arXiv:2304.06939, 2023. 3
- [108] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4995–5004, 2016. 3

[Figure 107]

Bo Li received the B.S. from Harbin Institute of Technology in 2020. He is currently working towards the PhD degree in the College of Computer and Data Science, Nanyang Technological University, Singapore. His research interests mainly include multimodal learning and foundation models..

Yuanhan Zhang is currently a Ph.D. student at MMLab@NTU, Nanyang Technological University, supervised by Prof. Ziwei Liu. His interests lie in computer vision and deep learning. In particular, He is focused on adapting foundation models—from vision to multi-modal—for real-world exploration. He has published several papers in ICCV, ECCV, CVPR, NeurIPS and IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). He also served as a reviewer for CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR,

[Figure 108]

IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), and International Journal of Computer Vision (IJCV).

[Figure 109]

Liangyu Chen received his B.Eng. from Nanyang Technological University, Singapore, in 2022. He worked at MMLab@NTU from 2022 to 2024. He is currently pursuing a Ph.D. in Computer Science at Stanford University. He researches multimodal foundation models and data-centric machine learning.

[Figure 110]

Fanyi Pu is an undergraduate student at Nanyang Technological University, Singapore, majoring in Data Science and Artificial Intelligence. His research focus is on multimodal large models.

[Figure 111]

Joshua Adrian is an undergraduate student at Nanyang Technological University of Singapore, majoring in Data Science and Artificial Intelligence. His research focuses on multimodal models and agent based models.

Jingkang Yang is a final-year PhD student at Nanyang Technological University (NTU), is working under the guidance of Professor Ziwei Liu. His research specializes in multimodal models and egocentric video understanding. Jingkang has published papers in top conferences such

[Figure 112]

- as CVPR, ICCV, ECCV, ICLR, and NeurIPS. Additionally, he has served as an outstanding reviewer for several top conferences, including CVPR, ICCV, ECCV, and as an area chair for ACL, EMNLP, and NAACL.

[Figure 113]

Chunyuan Li ’s recent focus is large multimodal models in vision-and-language. His contributions include the development of LLaVA and the series of model families, as well as earlier works include Oscar, GLIP, Grounding DINO, GLIGEN and Florence. He has worked with xAI, ByteDance, Microsoft Research, and obtained his PhD at Duke University.

[Figure 114]

Ziwei Liu is currently an Associate Professor

- at Nanyang Technological University, Singapore. His research revolves around computer vision, machine learning and computer graphics. He has published extensively on top-tier conferences and journals in relevant fields, including CVPR, ICCV, ECCV, NeurIPS, ICLR, SIGGRAPH, TPAMI, TOG and Nature - Machine Intelligence. He is the recipient of PAMI Mark Everingham Prize, CVPR Best Paper Award Candidate, Asian Young Scientist Fellowship, International Congress of Basic Sci-

ence Frontiers of Science Award and MIT Technology Review Innovators under 35 Asia Pacific. He serves as an Area Chair of CVPR, ICCV, ECCV, NeurIPS and ICLR, as well as an Associate Editor of IJCV.

