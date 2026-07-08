arXiv:2307.03601v5[cs.CV]12Jun2025

# GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest

Shilong Zhang1 ⋆, Peize Sun1∗ , Shoufa Chen1∗ , Min Xiao2 Wenqi Shao2 , Wenwei Zhang2 , Yu Liu3 , Kai Chen2 , and Ping Luo2

- 1 The University of Hong Kong
- 2 Shanghai AI Laboratory 3 Alibaba Group

Abstract. Visual instruction tuning large language model (LLM) on image-text pairs has achieved general-purpose vision-language abilities. However, the lack of region-text pairs limits their advancements to fine-grained multimodal understanding. In this paper, we propose spatial instruction tuning, which introduces the reference to the region-ofinterest (RoI) in the instruction. Before sending to LLM, the reference is replaced by RoI features and interleaved with language embeddings as a sequence. Our model GPT4RoI, trained on 7 region-text pair datasets, brings an unprecedented interactive and conversational experience compared to previous image-level models. (1) Interaction beyond language: Users can interact with our model by both language and drawing bounding boxes to flexibly adjust the referring granularity. (2) Versatile multimodal abilities: A variety of attribute information within each RoI can be mined by GPT4RoI, e.g., color, shape, material, action, etc. Furthermore, it can reason about multiple RoIs based on common sense. On the Visual Commonsense Reasoning (VCR) dataset, GPT4RoI achieves a remarkable accuracy of 81.6%, surpassing all existing models by a significant margin (the second place is 75.6%) and almost reaching human-level performance of 85.0%. The code, dataset, and demo can be found at https://github.com/jshilong/GPT4RoI.

Keywords: Visual Instruction Tuning · LLM · Multimodal

## 1 Introduction

Recent advancements of large language models (LLM) have shown incredible performance in solving natural language processing tasks in a human-like conversational manner, for example, commercial products [2,23,47,48] and community open-source projects [11,18,64,67–69]. Their unprecedented capabilities present a promising path toward general-purpose artificial intelligence models. Witnessing the power of LLM, the field of multimodal models [17,21,26,81] is developing a new technology direction to leverage LLM as the universal interface to build general-purpose models, where the feature space of a specific task is tuned to be aligned with the feature space of pre-trained language models.

⋆ Equal contribution.

Response

Response

Large Language Model

Large Language Model

[Figure 1]

[Figure 2]

| |
|---|

<region1>

… …

… Image tokens

… Image tokens

What is the woman in the upper left corner of the picture doing

What is <region1> doing

(a) Visual Instruction Tuning (b) Spatial Instruction Tuning

- Fig. 1: Comparison of visual instruction tuning on image-text pairs and spatial instruction tuning on region-text pairs. The bounding box and text description of each object are provided in region-text datasets. During training, the bounding box is from annotations, and in inference, it can be provided by the user or any off-the-shelf object detector.

- Table 1: Comparisons of vision-language models. Our GPT4RoI is an end-to-end model that supports region-level understanding and multi-round conversation.

Multi-Round End-to-End Dialogue Model

Model Image Region Multi-Region

Visual ChatGPT [75] MiniGPT-4 [100] LLaVA [37] InstructBLIP [15] MM-REACT [81] InternGPT [42] VisionLLM [72] CaptionAnything [70] DetGPT [52]

GPT4RoI

As one of the representative tasks, vision-and-language models align the vision encoder feature to LLM by instruction tuning on image-text pairs, such

- as MiniGPT-4 [100], LLaVA [37], InstructBLIP [15], etc. Although these works achieve amazing multimodal abilities, their alignments are only on image-text pairs [7,9,49,61,62], the lack of region-level alignment limits their advancements to more fine-grained understanding tasks such as region caption [29] and reasoning [88]. To enable region-level understanding in vision-language models, some works attempt to leverage external vision models, for example, MM-REACT [81], InternGPT [42] and DetGPT [52], as shown in Table 1. However, their nonend-to-end architecture is a sub-optimal choice for general-purpose multi-modal models.

Considering the limitations of previous works, our objective is to construct an end-to-end vision-language model that supports a fine-grained understanding on region-of-interest. Since there is no operation that can refer to specific regions in current image-level vision-language models [15,37,94,100], our key design is to incorporate references to bounding boxes into language instructions, thereby

upgrading them to the format of spatial instructions. For example, as shown in Figure 1, when the question is “what is <region1> doing?”, where the <region1> refers to a specific region-of-interest, the model will substitute the embedding of <region1> with the region feature extracted by the corresponding bounding box. The region feature extractor can be flexibly implemented by RoIAlign [25] or Deformable attention [101].

To establish fine-grained alignment between vision and language, we involve region-text datasets in our training, where the bounding box and the text description of each region are provided. The datasets are consolidated from publicly available ones including COCO object detection [36], RefCOCO [85], RefCOCO+ [85], RefCOCOg [45], Flickr30K entities [54], Visual Genome (VG) [29] and Visual Commonsense Reasoning (VCR) [88]. These datasets are transformed into spatial instruction tuning format. Moreover, we incorporate the LLaVA150K dataset [37] into our training process by utilizing an off-the-shelf detector to generate bounding boxes. This enhances our model’s ability to engage in multi-round conversations and generate more human-like responses.

The collected datasets are categorized into two types based on the complexity of the text. First, the plain-text data contains object category and simple attribute information. It is used for pre-training the region feature extractor without impacting the LLM. Second, the complex-text data often contains complex concepts or requires common sense reasoning. We conduct end-to-end fine-tuning of the region feature extractor and LLM for these data.

Benefiting from spatial instruction tuning, our model brings a new interactive experience, where the user can express the question to the model with language and the reference to the region-of-interest. This leads to new capacities beyond image-level understanding, such as region caption and complex region reasoning. As a generalist, our model GPT4RoI also shows its strong region understanding ability on popular benchmarks, including comprehensive image region understanding benchmark ViP-Bench [5], the region caption task on Visual Genome [29], the region reasoning task on Visual Commonsense Reasoning [88] (VCR). Especially noteworthy is the performance on the most challenging VCR dataset, where GPT4RoI achieves an impressive accuracy of 81.6%, 6 points ahead of the second-place and nearing the human-level performance benchmarked

- at 85.0%. In summary, our work makes the following contributions:

- – We introduce spatial instruction, combining language and the reference to region-of-interest into an interleave sequence, enabling accurate region referring and enhancing user interaction.
- – By spatial instruction tuning LLM with region-text datasets, our model can follow user instructions to solve diverse region understanding tasks, such as region caption and reasoning.
- – Our method, as a generalist, outperforms the previous state-of-the-art approach on a wide range of region understanding benchmarks.

## 2 Related Work

### 2.1 Large Language Model

The field of natural language processing (NLP) has achieved significant development by the high-capability large language model (LLM). The potential of LLM is first demonstrated by pioneering works such as BERT [16] and GPT [57]. Then scaling up progress is started and leads to a series of excellent works, for example, T5 [58], GPT-3 [3], Flan-T5 [13], PaLM [12], etc. With the growth of training data and model parameters, this scaling-up progress brings a phenomenal product, ChatGPT [47]. By generative pre-trained LLM and instruction tuning [50] on human feedback, ChatGPT shows unprecedented performance on conversations with humans, reasoning and planning tasks [4,46,78], etc.

### 2.2 Large Vision-Language Model

To utilize high-performance LLM to build up vision-language models, LLM as task coordinator is proposed. Given the user instruction, LLM parses the instruction and calls various external vision models. Some representative works are Visual ChatGPT [75], ViperGPT [65], MM-REACT [81], InternGPT [42], VideoChat [31], etc. Although these models largely expand the scope of multimodal models, they depend on external vision models and these non-end-to-end architectures are not the optimal choice for multi-modal models. To obtain end-to-end visionlanguage models, instruction tuning LLM on image-text pairs is proposed to align visual features with LLM and accomplish multimodal tasks in a unified way, for example, Flamingo [1], MiniGPT-4 [100], LLaVA [37], LLaMa-Adapter [94], InstructBLIP [15], MM-GPT [22], VPGTrans [92], etc. These models achieve amazing image-level multimodal abilities, while several benchmarks such as LVLM-eHub [77] and MMBench [41] find that these models still have performance bottlenecks when need to be under specific region reference. Our GPT4RoI follows the research line of visual instruction tuning and moves forward region-level multimodal understanding tasks such as region caption [29] and reasoning [88].

### 2.3 Region-Level Image Understanding

For region-level understanding, it is a common practice in computer vision to identify potential regions of interest first and then do the understanding. Object detection [6,60,87,101] tackles the search for potential regions, which are generally accompanied by a simple classification task to understand the region’s content. To expand the object categories, [28, 33, 39, 99] learn from natural language and achieve amazing open-vocabulary object recognition performance. Region captioning [27,76,80] provides more descriptive language descriptions in a generative way. Scene graph generation [34,66,79] analyzes the relationships between regions by the graph. The VCR [89] dataset presents many region-level reasoning cases and [32, 63, 82, 84] exhibit decent performance by correctly selecting the answers in the multiple-choice format. However, a general-purpose

region understanding model has yet to emerge. In this paper, by harnessing the powerful large language model [11,68], GPT4RoI uses a generative approach to handle all these tasks. Users can complete various region-level understanding tasks by freely asking questions.

### 2.4 Other Region Reference Format.

Many recent studies (concurrent works or follow-ups of GPT4RoI) integrate region-level data via different region reference formats into the training of MLLMs [8,51,53,55,83,86,96]. In particular, Kosmos-2 [51], Shikra [8], and CogVLM [71] directly quantize bounding boxes into discrete location tokens or numeric representations of positions. Regiongpt [24], Osprey [86], and Groma [44] use different feature extraction operations to extract the features of user reference, which may be partially influenced by our approach. Additionally, there has been research that explores a fusion of the two approaches, as shown in references [59,83,93].

## 3 GPT4RoI

GPT4RoI consists of a vision encoder, a projector for image-level features, a region feature extractor, and a large language model (LLM). Compared to previous works [37,100], GPT4RoI stands out for its ability to convert instructions that include spatial positions into an interleaved sequence of region features and text embeddings, as shown in Figure 2.

### 3.1 Model Architecture

We adopt the CLIP ViT-L/14 [56] as the vision encoder. We use the feature map of the penultimate transformer layer as the representation of the entire image and then map it to the language space using a single linear layer. A region feature extractor is used to obtain region features with the region references in the user instruction. Finally, we employ the Vicuna [97], an instruction-tuned LLaMA [68], to perform further processing.

We utilize widely adopted modules in the field of object detection to construct the region feature extractor. To ensure a robust feature representation for regions of varying scales, we construct a multi-level image feature pyramid [35] by selecting four layers from the CLIP vision encoder and fusing them with five lightweight scale shuffle modules [95]. These layers are located at the second-to-last, fifth-tolast, eighth-to-last, and eleventh-to-last positions, respectively. Additionally, we incorporate feature coordinates [38,73] for each level to address the problem of translation invariance in CNNs. This helps make the model sensitive to absolute position information, such as the description “girl on left” in Figure 3. Finally, we use RoIAlign to extract region-level features with an output size of 14×14 [25], which ensures that sufficient detailed information is preserved. Moreover, all four level features are involved in the RoIAlign operation and fused into a single embedding as the representation of the region of interest (RoI) [40].

[Figure 3]

- Fig. 2: GPT4RoI is an end-to-end vision-language model for processing spatial instructions that contain references to the region-of-interest, such as <region{i}>. During tokenization and conversion to embeddings, the embedding of <region{i}> in the instruction is replaced with the RoIAlign results from multi-level image features. Subsequently, such an interleaved region feature and language embedding sequence can be sent to a large language model (LLM) for further processing. We also utilize the entire image feature to capture global information.

### 3.2 Tokenization and Embedding

To enable users to refer to regions of interest in text inputs, we define a special token <region{i}>, which acts as the placeholder that will be replaced by the corresponding region feature after tokenization and embedding. One example is depicted in Figure 2. When a user presents a spatial instruction, “What was <region1> doing before <region3> touched him?”, the embedding of <region1> and <region3> are replaced by their corresponding region features. However, this replacement discards the pure text references to different regions. To allows LLM to maintain the original references (region1, region3) in the response sequence, the instruction is modified to “What was region1 <region1> doing before region3 <region3> touched him?”. Then, LLM can generate a reply like “The person in region1 was eating breakfast before the person in region3 touched them.”

Regardless of the user instruction, we incorporate a prefix prompt, “The <image> provides an overview of the picture.” The <image> is a special token that acts as a placeholder, the embedding of which would be replaced by image features of the vision encoder. These features enable LLM to receive comprehensive image information and obtain a holistic understanding of the visual context.

### 3.3 Spatial Instruction Tuning

Our model is trained using a next-token prediction loss [37,100], where the model predicts the next token in a given input text sequence.

We formulate annotations in instruction tuning format by creating a question that refers to the mentioned region for each region-text annotation. We partition

the available region-text data into two groups and use them separately in two distinct training stages. In the first stage, we attempt to align region feature with word embedding in language models using simple region-text pairs that contain color, position, or category description. The second stage is designed to handle more complex concepts, such as action, relationship, and common sense reasoning. Furthermore, we provide diverse instructions for these datasets to simulate chat-like input in this stage.

- Stage 1: Pre-training In this stage, we first load the weights of LLaVA [37] after its initial stage of training, which includes a pre-trained vision encoder, a projector for image-level features, and an LLM. We only keep the region feature extractor trainable and aim to align region features with language embedding by collecting short text and bounding box pairs. These pairs are from both normal detection datasets and referring expression detection datasets, which have short expressions. The objective is to enable the model to recognize categories and simple attributes of the region in an image, which are typically represented by a short text annotation (usually within 5 words). Specifically, we utilize COCO [36], RefCOCO [85], and RefCOCO+ [85] datasets in this stage.

As shown in Table 2, for COCO detection data, we first explain the task in the prompt and then convert the annotations to a single-word region caption task. For RefCOCO and RefCOCO+, we also give task definitions first and train the model to generate descriptions containing basic attributes of the region. Only the description of the region (in red color) will be used to calculate the loss.

[Figure 4]

Fig. 3: After Stage 1 training, GPT4RoI is capable of identifying the category of the region (elephant), simple attributes such as color (purple), and the position of the region (left).

After this training stage, GPT4RoI can recognize categories, simple attributes, and positions of regions in images, as shown in Figure 3.

- Stage 2: End-to-end fine-tuning In this stage, we only keep the vision encoder weights fixed and train the region feature extractor, image feature projector, and LLM weights. Our main focus is to enhance GPT4RoI’s ability to accurately follow user instructions and tackle complex single/multiple region understanding tasks.

As shown in Table 3, we tailor specific instructions for different tasks. For single region caption, we construct from Visual Genome (VG) region caption

- Table 2: Exampls for Stage 1 training data: For both tasks, we begin by providing a description of the task and the expected answer. Then, we concatenate all region-text pairs into a sequence. For detection data, the format is <region{i}> category_name. For referring expression comprehension, the format is <region{i}> description of region. Only the responses highlighted in red are used to calculate the loss.

Object Detection: In the conversation below, you simply answer the category name based on what you see in the imagery inside a particular region. I will give you only one region each time. Categories containing person, bicycle, car ... <region1> person, <region2> dog

[Figure 5]

Referring Expression Comprehension I will provide you with only one region containing only one object, although there may be other objects present in the image. It is recommended that you describe the object’s relative position with respect to other objects in the image and its basic attributes.

[Figure 6]

- <region1> red shirt girl
- <region2> guy in black
- <region3> right most person

part [29] and RefCOCOg [45]. For multiple region caption, Flicker30k [54] is converted to a multiple region caption task where the caption should include all visual elements emphasized by bounding boxes. To simulate user instruction, we create 20 questions for each caption task. For the region reasoning task, we modify Visual Commonsense Reasoning (VCR) [88] to meet the input format requirements and make it more similar to human input.

To improve the capability of GPT4RoI for multi-round conversation and generate more human-like responses, we also involve the LLaVA150k [37] visual instruction dataset in this stage. We employ an off-the-shelf LVIS detector [19] to extract up to 100 detection boxes per image. These boxes are then concatenated with the user instructions in the format “<region{i}> may feature a class_name”. LLaVA150k significantly improves the capability of GPT4RoI for multi-round conversation.

After completing this training stage, GPT4RoI is capable of performing complex region understanding tasks based on user instructions, including region caption and reasoning, as demonstrated in Figure 4 and Section 4.

- Table 3: Exampls for Stage 2 training data: Only the response in red color and stop string ### will be used to calculate the loss.

Region Caption ### Question: Can you provide me with a detailed description of the region marked by <region1> ? ### Answer: A man wearing a light blue T-shirt and jeans

with his arms extend.

[Figure 7]

### Question: Could you please give me a detailed description of areas <region1>, <region2>, <region3>, <region4>, <region5> ? ### Answer: A man in a white

shirt with a plate of food sits outside in a folding chair with a little girl who is writing

[Figure 8]

Region Reasoning ### Question: Is <region1> happy to be speaking with <region2> and <region3> ? ### Answer: No, person at region1 is bothered by the conversation. ### Question: What factors influenced your perspective? ### Answer: Person at region1 is standing with his hand on his hip in a defensive way.

[Figure 9]

- 4 Demostrations

In this section, we compare the differences between the visual instruction tuning model LLaVA [37] and our spatial instruction tuning model GPT4RoI. We demonstrate our new interactive approach and highlight its advanced capabilities in understanding multimodality.

As shown in Figure 4.A, when we try to make LLaVA focus on the center region of the image, it only sees the boy holding an umbrella and a bag, but it misses the book. As a result, LLaVA gives a wrong answer to the question “What is the boy doing” (Figure 4.A.①), and this leads to an incorrect conclusion that “the boy’s behavior is not dangerous” (Figure 4.A.②).

In comparison, as shown in Figure 4.B, our approach GPT4RoI efficiently recognizes visual details using the given bounding box. This allows it to accurately identify the action of “reading a magazine.” Furthermore, GPT4RoI demonstrates

[Figure 10]

- Fig. 4: GPT4RoI and LLaVA dialogue performance showcase. Figures A and C demonstrate the dialogue scenarios of LLaVA when referring to a single instance and multiple instances solely using natural language in the conversation. On the other hand, Figures B and D showcase how GPT4RoI utilizes bounding boxes as references to address the same scenarios.

its reasoning abilities by correctly inferring that the “boy’s behavior is dangerous”, and giving a reasonable reason that “the boy is reading a book while crossing the street”.

When there are multiple instances in the image (as depicted in Figure 4.C), we attempt to refer to the corresponding instances as “the right” and “the middle”. However, LLaVA provides incorrect information by stating that the right man is “looking at the women” (as shown in Figure 4.C.③). Even more concerning, LLaVA overlooks the actual women in the middle and mistakenly associates the women on the left as the reference, resulting in completely inaccurate information (as shown in Figure 4.C.④ & ⑤).

In comparison, as shown in Figure 4.D, GPT4RoI is able to understand the user’s requirements, such as identifying the person to call when ordering food, and accurately recognize that the person in region1 fulfills this criterion. Additionally, it correctly recognizes that the person in region3 is “looking at the menu”. Importantly, GPT4RoI can also infer relationships between the provided regions based on visual observations. For example, it deduces that the likely relationship between region2 and region3 is that of a “couple”, providing a reasonable explanation that they “are smiling and enjoying each other’s company”.

Table 4: Evaluation results on ViP-Bench.The assessed dimensions are Recognition (Rec), OCR, Knowledge (Know), Math, Relationship (Rel), and Language Generation (Lang). † means concurrent work.

Type Rec OCR Know Math Rel Lang All

Kosmos-2† 29.5 14.2 18.5 9.7 7.5 21.9 26.9 InstructBLIP 36.9 16.3 34.2 22.3 26.8 7.5 31.7 Shikra-7B† 40.2 10.0 28.0 3.5 18.9 20.6 33.7

GPT4RoI-7B 35.6 16.7 29.7 9.7 32.5 13.8 35.1

## 5 Experiments

We adopt several representative benchmarks to quantitatively evaluate the region understanding ability of GPT4RoI. First, we give the results on comprehensive LLM image region understanding benchmark ViP-Bench [5]. We also evaluate the region recognition ability through the open-vocabulary region recognition and region caption task. For region reasoning ability, we report the results on Visual Commonsense Reasoning [88].

### 5.1 Comprehensive Region Understanding

ViP-Bench [5] is a comprehensive benchmark for evaluating multimodal models’ region understanding capabilities, including recognition (Rec), OCR, knowledge (Know), math, relationship (Rel), and language generation (Lang). As shown in Table 4, we give the comparison of methods including InstructBLIP [15] with a pure language reference, Shikra [8] with textual coordinates as reference, Kosmos-2 [51] with extra position tokens. We can see that GPT4RoI surpasses by a clear margin with actually much less training data.

### 5.2 Region Recognization

We also conduct evaluations on several specific region-level understanding tasks to demonstrate the superior performance of GPT4RoI. The instruction template for these tasks is provided in Table 5.

Open-Vocabulary Recognition The results are obtained by calculating the semantic similarity between the generated region caption of GPT4RoI and the vocabulary lists of each dataset and then selecting the category with the highest similarity as the final result (as the study [86]). The input to the CLIP-SurgeryViT-L is the cropped region with a size of 512×512. As shown in Table 6, constrained by the resolution of 224×224, GPT4RoI exhibits slightly lower performance than CLIP-Surgery-ViT-L only at the instance level. Considering all metrics score on two datasets, GPT4RoI demonstrates robust and comprehensive region recognition capabilities compared to our concurrent works that employ alternative reference formats.

Table 5: Task prompt of three downstream tasks.

Region Caption Task

### Question: Can you give a description of the region mentioned by <region> ### Answer: A man wearing a light blue t-shirt and jeans with his arms extended

Region Reasoning Task on VCR Q → A

### Question: <region1>,<region2>,<region3>... refers to specific areas within the photo along with their respective identifiers. I need you to answer the question. Questions are multiple-choice; you only need to pick the correct answer from the given options (A), (B), (C), or (D).

How is 1 feeling ?

- (A),1 is feeling amused .
- (B),1 is upset and disgusted .
- (C),1 is feeling very scared .
- (D),1 is feeling uncomfortable with 3 ### Answer: (C)

##### QA → R

### Question: <region1>,<region2>,<region3>... refers to specific areas within the photo along with their respective identifiers. I give you a question and its answer, I need you to provide a rationale explaining why the answer is right. Both questions are multiple-choice; you only need to pick the correct answer from the given options (A), (B), (C), or (D).

"How is 1 feeling ?" The answer is "1 is feeling very scared." What’s the rationale for this decision?

- (A),1’s face has wide eyes and an open mouth .
- (B),When people have their mouth back like that and their eyebrows lowered they are usually disgusted by what they see .
- (C),3,2,1 are seated at a dining table where food would be served to them . people unaccustomed to odd or foreign dishes may make disgusted looks at the thought of eating it .
- (D),1’s expression is twisted in disgust . ### Answer: (A)

Region Caption We report the scores of BLEU, METEOR, ROUGE, and CIDEr for different methods on the validation set of Visual Genome [29]. As shown in Table 7, compared to the concurrent work Shikra [8] with textual coordinates as region reference, GPT4RoI shows clearly better region caption ability. After fine-tuning, GPT4RoI outperforms the previous state-of-the-art specialist model GRiT [76] by a significant margin, without any additional techniques or tricks. Additionally, we observe that the performance of GPT4RoI-7B and GPT4RoI-13B is comparable, suggesting that the bottleneck in performance lies in the design of the visual module and the availability of region-text pair data. These areas can be explored further in future work.

- Table 6: Recognition performance on panoptic segmentation (PQ), instance segmentation (AP) and semantic segmentation (mIoU) upon the validation sets of Cityscapes [14] and ADE20K [98]. † means concurrent work.

Method

Cityscapes ADE20K-150

PQ AP mIoU PQ AP mIoU CLIP-Surgery-ViT-L 27.24 28.35 21.92 26.55 29.70 21.42 Kosmos-2† 12.09 9.81 13.71 6.53 4.33 5.40 Shikra-7B† 17.80 11.53 17.77 27.52 20.35 18.24 GPT4RoI-7B 34.70 21.93 36.73 36.32 26.08 25.82

- Table 7: Compariation of region caption ability on the validation dataset on Visual Genome. All methods employ ground truth bounding boxes as input. † means concurrent work. ⋄ means the model is after fine-tuning on Visual Genome.

Model BLEU@4 METEOR ROUGE CIDEr

GRiT - 17.1 - 142.0 Shikra-7B† 8.9 15.2 30.3 115.8

GPT4RoI-7B 10.5 16.5 33.4 134.5 GPT4RoI-7B⋄ 11.5 17.4 35.0 145.2 GPT4RoI-13B⋄ 11.7 17.6 35.2 146.8

### 5.3 Region Reasoning

Visual Commonsense Reasoning [88] employs multiple-choice settings that require both recognition ability and commonsense reasoning to select the correct choice. Calculating the similarity of LLM’s output with the correct choice is quite a challenge. This is because it is hard to find a model that can capture the reasoning similarity. Therefore, we finetune GPT4RoI to align with the answer format, following conventional methods.

Visual Commonsense Reasoning Visual Commonsense Reasoning (VCR) offers a highly demanding scenario that necessitates advanced reasoning abilities, heavily relying on common sense. Given the question(Q), the model’s task is not only to select the correct answer(A) but also to select a rationale(R) that explains why the chosen answer is true. In the Q→A setup, a model is given a question and must select the correct answer from four choices. In the QA->R setup, a model is provided with a question and the correct answer, and it needs to justify the answer by selecting the most appropriate rationale from four choices. The performance of models is evaluated using the Q→AR metric, where accuracy is measured as the percentage of correctly answered questions along with the correct rationale.

As shown in Table 8, GPT4RoI shows significant improvements over the previous methods across all Q → A, QA → R, and Q → AR tasks. Notably, in the crucial Q → AR task, GPT4RoI-13B achieves a performance of 81.6%

#### Table 8: Accuracy scores on VCR. GPT4RoI achieves state-of-the-art accuracy among all methods.

Val Acc.(%) Test Acc.(%) Q → A QA → R Q → AR Q → A QA → R Q → AR

Model Code #Params

ViLBERT [43] Y 221M 72.4 74.5 54.0 73.3 74.6 54.8 Unicoder-VL [30] Y - 72.6 74.5 54.5 73.4 74.4 54.9 VLBERT-L [63] Y 383M 75.5 77.9 58.9 75.8 78.4 59.7 UNITER-L [10] Y 303M - - - 77.3 80.8 62.8 ERNIE-ViL-L [84] Y - 78.52 83.37 65.81 79.2 83.5 66.3 MERLOT [91] Y 223M - - - 80.6 80.4 65.1 VILLA-L [20] Y - 78.45 82.57 65.18 78.9 82.8 65.7 RESERVE-L [90] Y 644M - - - 84.0 84.9 72.0 VQA-GNN-L [74] Y 1B+ - - - 85.2 86.6 74.0

GPT4RoI-7B (ours) Y 7B+ 87.4 89.6 78.6 - - -

VLUA+@Kuaishou N - - - - 84.8 87.0 74.0 KS-MGSR@KDDI Research and SNAP N - - - - 85.3 86.9 74.3 SP-VCR@Shopee N - - - - 83.6 88.6 74.4 HunYuan-VCR@Tencent N - - - - 85.8 88.0 75.6 Human Performance [88] - - - - - 91.0 93.0 85.0

GPT4RoI-13B (ours) Y 13B+ - - - 89.4 91.0 81.6

accuracy, surpassing previous methods by over 6 points, even outperforming confidential commercial product. More importantly, this performance is almost reaching human-level performance of 85.0% accuracy, which shows that the multimodal ability of GPT4RoI is promising to be further developed to human intelligence. Furthermore, comparing GPT4RoI to previous methods, particularly observing the size of the language model used, also demonstrates the significant benefits of the Large Language Model (LLM) for visual reasoning tasks.

## 6 Conclusions

We present GPT4RoI, an end-to-end vision-language model that can execute user instructions to achieve region-level image understanding. Our approach employs spatial instruction tuning for the large language model (LLM), where we convert the reference to bounding boxes from user instructions into region features. These region features, along with language embeddings, are combined to create an input sequence for the large language model. We show that GPT4RoI enhances user interaction by accurately referring to regions and achieves impressive performance in region-level image understanding tasks.

## 7 Acknowledgement

This project is supported by the National Key R&D Program of China No.2022ZD0161000.

## References

- 1. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems 35, 23716–23736 (2022)
- 2. Anthropic: Claude. https://www.anthropic.com/index/introducing-claude

(2023)

- 3. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901 (2020)
- 4. Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y.T., Li, Y., Lundberg, S., et al.: Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712 (2023)
- 5. Cai, M., Liu, H., Mustikovela, S.K., Meyer, G.P., Chai, Y., Park, D., Lee, Y.J.: Making large multimodal models understand arbitrary visual prompts (2023)
- 6. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: Endto-end object detection with transformers. In: European Conference on Computer Vision. pp. 213–229. Springer (2020)
- 7. Changpinyo, S., Sharma, P., Ding, N., Soricut, R.: Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3558–3568 (2021)
- 8. Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., Zhao, R.: Shikra: Unleashing multimodal llm’s referential dialogue magic (2023)
- 9. Chen, X., Fang, H., Lin, T.Y., Vedantam, R., Gupta, S., Dollár, P., Zitnick, C.L.: Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325 (2015)
- 10. Chen, Y.C., Li, L., Yu, L., Kholy, A.E., Ahmed, F., Gan, Z., Cheng, Y., Liu, J.: Uniter: Universal image-text representation learning (2020)
- 11. Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., Stoica, I., Xing, E.P.: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality (March 2023), https://lmsys.org/ blog/2023-03-30-vicuna/
- 12. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., et al.: Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 (2022)
- 13. Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., et al.: Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022)
- 14. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: ECCV. pp. 3213–3223 (2016)
- 15. Dai, W., Li, J., Li, D., Tiong, A.M.H., Zhao, J., Wang, W., Li, B., Fung, P., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning (2023)
- 16. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805

(2018)

- 17. Driess, D., Xia, F., Sajjadi, M.S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al.: Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378 (2023)
- 18. Du, Z., Qian, Y., Liu, X., Ding, M., Qiu, J., Yang, Z., Tang, J.: Glm: General language model pretraining with autoregressive blank infilling. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 320–335 (2022)
- 19. Fang, Y., Sun, Q., Wang, X., Huang, T., Wang, X., Cao, Y.: Eva-02: A visual representation for neon genesis. arXiv preprint arXiv:2303.11331 (2023)
- 20. Gan, Z., Chen, Y.C., Li, L., Zhu, C., Cheng, Y., Liu, J.: Large-scale adversarial training for vision-and-language representation learning (2020)
- 21. Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra,

I.: Imagebind: One embedding space to bind them all. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15180– 15190 (2023)

- 22. Gong, T., Lyu, C., Zhang, S., Wang, Y., Zheng, M., Zhao, Q., Liu, K., Zhang, W., Luo, P., Chen, K.: Multimodal-gpt: A vision and language model for dialogue with humans (2023)
- 23. Google: Bard. https://bard.google.com/ (2023)
- 24. Guo, Q., De Mello, S., Yin, H., Byeon, W., Cheung, K.C., Yu, Y., Luo, P., Liu, S.: Regiongpt: Towards region understanding vision language model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13796–13806 (2024)
- 25. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: Proceedings of the IEEE international conference on computer vision. pp. 2961–2969 (2017)
- 26. Huang, S., Dong, L., Wang, W., Hao, Y., Singhal, S., Ma, S., Lv, T., Cui, L., Mohammed, O.K., Liu, Q., et al.: Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045 (2023)
- 27. Johnson, J., Karpathy, A., Fei-Fei, L.: Densecap: Fully convolutional localization networks for dense captioning (2015)
- 28. Kamath, A., Singh, M., LeCun, Y., Synnaeve, G., Misra, I., Carion, N.: Mdetr – modulated detection for end-to-end multi-modal understanding (2021)
- 29. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision 123, 32–73 (2017)
- 30. Li, G., Duan, N., Fang, Y., Gong, M., Jiang, D., Zhou, M.: Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training (2019)
- 31. Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., Qiao, Y.: Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355

(2023)

- 32. Li, L.H., Yatskar, M., Yin, D., Hsieh, C.J., Chang, K.W.: Visualbert: A simple and performant baseline for vision and language (2019)
- 33. Li*, L.H., Zhang*, P., Zhang*, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., Zhang, L., Hwang, J.N., Chang, K.W., Gao, J.: Grounded language-image pre-training. In: CVPR (2022)
- 34. Li, Y., Ouyang, W., Zhou, B., Wang, K., Wang, X.: Scene graph generation from objects, phrases and region captions (2017)
- 35. Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2117–2125 (2017)

- 36. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. pp. 740–755. Springer (2014)
- 37. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. arXiv preprint arXiv:2304.08485 (2023)
- 38. Liu, R., Lehman, J., Molino, P., Such, F.P., Frank, E., Sergeev, A., Yosinski, J.: An intriguing failing of convolutional neural networks and the coordconv solution

(2018)

- 39. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., Zhang, L.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection (2023)
- 40. Liu, S., Qi, L., Qin, H., Shi, J., Jia, J.: Path aggregation network for instance segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 8759–8768 (2018)
- 41. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281 (2023)
- 42. Liu, Z., He, Y., Wang, W., Wang, W., Wang, Y., Chen, S., Zhang, Q., Yang, Y., Li, Q., Yu, J., et al.: Internchat: Solving vision-centric tasks by interacting with chatbots beyond language. arXiv preprint arXiv:2305.05662 (2023)
- 43. Lu, J., Batra, D., Parikh, D., Lee, S.: Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks (2019)
- 44. Ma, C., Jiang, Y., Wu, J., Yuan, Z., Qi, X.: Groma: Localized visual tokenization for grounding multimodal large language models. arXiv preprint arXiv:2404.13013

(2024)

- 45. Mao, J., Huang, J., Toshev, A., Camburu, O., Yuille, A.L., Murphy, K.: Generation and comprehension of unambiguous object descriptions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 11–20 (2016)
- 46. Mu, Y., Zhang, Q., Hu, M., Wang, W., Ding, M., Jin, J., Wang, B., Dai, J., Qiao, Y., Luo, P.: Embodiedgpt: Vision-language pre-training via embodied chain of thought. arXiv preprint arXiv:2305.15021 (2023)
- 47. OpenAI: Chatgpt. https://openai.com/blog/chatgpt (2022)
- 48. OpenAI: Gpt-4 technical report (2023)
- 49. Ordonez, V., Kulkarni, G., Berg, T.: Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems 24

(2011)

- 50. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35, 27730–27744 (2022)
- 51. Peng, Z., Wang, W., Dong, L., Hao, Y., Huang, S., Ma, S., Wei, F.: Kosmos-2: Grounding multimodal large language models to the world (2023)
- 52. Pi, R., Gao, J., Diao, S., Pan, R., Dong, H., Zhang, J., Yao, L., Han, J., Xu, H., Zhang, L.K.T.: Detgpt: Detect what you need via reasoning. arXiv preprint arXiv:2305.14167 (2023)
- 53. Pi, R., Yao, L., Gao, J., Zhang, J., Zhang, T.: Perceptiongpt: Effectively fusing visual perception into llm (2023)
- 54. Plummer, B.A., Wang, L., Cervantes, C.M., Caicedo, J.C., Hockenmaier, J., Lazebnik, S.: Flickr30k entities: Collecting region-to-phrase correspondences for richer

- image-to-sentence models. In: Proceedings of the IEEE international conference on computer vision. pp. 2641–2649 (2015)
- 55. Pramanick, S., Han, G., Hou, R., Nag, S., Lim, S.N., Ballas, N., Wang, Q., Chellappa, R., Almahairi, A.: Jack of all tasks, master of many: Designing generalpurpose coarse-to-fine vision-language model (2023)
- 56. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)
- 57. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al.: Improving language understanding by generative pre-training. OpenAI (2018)
- 58. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research 21(1), 5485–5551 (2020)
- 59. Rasheed, H., Maaz, M., Shaji, S., Shaker, A., Khan, S., Cholakkal, H., Anwer, R.M., Xing, E., Yang, M.H., Khan, F.S.: Glamm: Pixel grounding large multimodal model. arXiv preprint arXiv:2311.03356 (2023)
- 60. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems 28, 91–99 (2015)
- 61. Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., Komatsuzaki, A.: Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114 (2021)
- 62. Sharma, P., Ding, N., Goodman, S., Soricut, R.: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 2556–2565 (2018)
- 63. Su, W., Zhu, X., Cao, Y., Li, B., Lu, L., Wei, F., Dai, J.: Vl-bert: Pre-training of generic visual-linguistic representations. arXiv preprint arXiv:1908.08530 (2019)
- 64. Sun, T., Xipeng, Q.: Moss. https://github.com/OpenLMLab/MOSS (2022)
- 65. Surís, D., Menon, S., Vondrick, C.: Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128 (2023)
- 66. Tang, K., Zhang, H., Wu, B., Luo, W., Liu, W.: Learning to compose dynamic tree structures for visual contexts (2018)
- 67. Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., Hashimoto, T.B.: Stanford alpaca: An instruction-following llama model. https: //github.com/tatsu-lab/stanford_alpaca (2023)
- 68. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)
- 69. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023)
- 70. Wang, T., Zhang, J., Fei, J., Ge, Y., Zheng, H., Tang, Y., Li, Z., Gao, M., Zhao, S., Shan, Y., et al.: Caption anything: Interactive image description with diverse multimodal controls. arXiv preprint arXiv:2305.02677 (2023)
- 71. Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., Xu, J., Xu, B., Li, J., Dong, Y., Ding, M., Tang, J.: Cogvlm: Visual expert for pretrained language models (2023)

- 72. Wang, W., Chen, Z., Chen, X., Wu, J., Zhu, X., Zeng, G., Luo, P., Lu, T., Zhou, J., Qiao, Y., et al.: Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. arXiv preprint arXiv:2305.11175 (2023)
- 73. Wang, X., Kong, T., Shen, C., Jiang, Y., Li, L.: Solo: Segmenting objects by locations. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVIII 16. pp. 649–665. Springer (2020)
- 74. Wang, Y., Yasunaga, M., Ren, H., Wada, S., Leskovec, J.: Vqa-gnn: Reasoning with multimodal semantic graph for visual question answering (2022)
- 75. Wu, C., Yin, S., Qi, W., Wang, X., Tang, Z., Duan, N.: Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671 (2023)
- 76. Wu, J., Wang, J., Yang, Z., Gan, Z., Liu, Z., Yuan, J., Wang, L.: Grit: A generative region-to-text transformer for object understanding (2022)
- 77. Xu, P., Shao, W., Zhang, K., Gao, P., Liu, S., Lei, M., Meng, F., Huang, S., Qiao, Y., Luo, P.: Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models (2023)
- 78. Yang, J., Tan, W., Jin, C., Liu, B., Fu, J., Song, R., Wang, L.: Pave the way to grasp anything: Transferring foundation models for universal pick-place robots. arXiv preprint arXiv:2306.05716 (2023)
- 79. Yang, J., Ang, Y.Z., Guo, Z., Zhou, K., Zhang, W., Liu, Z.: Panoptic scene graph generation (2022)
- 80. Yang, L., Tang, K., Yang, J., Li, L.J.: Dense captioning with joint inference and visual context. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (Jul 2017)
- 81. Yang, Z., Li, L., Wang, J., Lin, K., Azarnasab, E., Ahmed, F., Liu, Z., Liu, C., Zeng, M., Wang, L.: Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381 (2023)
- 82. Yao, Y., Chen, Q., Zhang, A., Ji, W., Liu, Z., Chua, T.S., Sun, M.: Pevl: Positionenhanced pre-training and prompt tuning for vision-language models. arXiv preprint arXiv:2205.11169 (2022)
- 83. You, H., Zhang, H., Gan, Z., Du, X., Zhang, B., Wang, Z., Cao, L., Chang, S.F., Yang, Y.: Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704 (2023)
- 84. Yu, F., Tang, J., Yin, W., Sun, Y., Tian, H., Wu, H., Wang, H.: Ernie-vil: Knowledge enhanced vision-language representations through scene graph (2021)
- 85. Yu, L., Poirson, P., Yang, S., Berg, A.C., Berg, T.L.: Modeling context in referring expressions. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14. pp. 69–85. Springer (2016)
- 86. Yuan, Y., Li, W., Liu, J., Tang, D., Luo, X., Qin, C., Zhang, L., Zhu, J.: Osprey: Pixel understanding with visual instruction tuning (2023)
- 87. Zang, Y., Li, W., Han, J., Zhou, K., Loy, C.C.: Contextual object detection with multimodal large language models (2023)
- 88. Zellers, R., Bisk, Y., Farhadi, A., Choi, Y.: From recognition to cognition: Visual commonsense reasoning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6720–6731 (2019)
- 89. Zellers, R., Bisk, Y., Farhadi, A., Choi, Y.: From recognition to cognition: Visual commonsense reasoning. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)

- 90. Zellers, R., Lu, J., Lu, X., Yu, Y., Zhao, Y., Salehi, M., Kusupati, A., Hessel, J., Farhadi, A., Choi, Y.: Merlot reserve: Multimodal neural script knowledge through vision and language and sound. In: CVPR (2022)
- 91. Zellers, R., Lu, X., Hessel, J., Yu, Y., Park, J.S., Cao, J., Farhadi, A., Choi, Y.: Merlot: Multimodal neural script knowledge models. In: Advances in Neural Information Processing Systems 34 (2021)
- 92. Zhang, A., Fei, H., Yao, Y., Ji, W., Li, L., Liu, Z., Chua, T.S.: Transfer visual prompt generator across llms. arXiv preprint arXiv:2305.01278 (2023)
- 93. Zhang, A., Zhao, L., Xie, C.W., Zheng, Y., Ji, W., Chua, T.S.: Next-chat: An lmm for chat, detection and segmentation. arXiv preprint arXiv:2311.04498 (2023)
- 94. Zhang, R., Han, J., Zhou, A., Hu, X., Yan, S., Lu, P., Li, H., Gao, P., Qiao, Y.: Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199 (2023)
- 95. Zhang, S., Wang, X., Wang, J., Pang, J., Lyu, C., Zhang, W., Luo, P., Chen, K.: Dense distinct query for end-to-end object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7329–7338 (June 2023)
- 96. Zhao, Y., Lin, Z., Zhou, D., Huang, Z., Feng, J., Kang, B.: Bubogpt: Enabling visual grounding in multi-modal llms (2023)
- 97. Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E., Stoica, I.: Judging llm-as-a-judge with mt-bench and chatbot arena (2023)
- 98. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ade20k dataset. In: CVPR. pp. 633–641 (2017)
- 99. Zhou, X., Girdhar, R., Joulin, A., Krähenbühl, P., Misra, I.: Detecting twentythousand classes using image-level supervision. In: ECCV (2022)
- 100. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023)
- 101. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159

(2020)

