# arXiv:2405.14129v2[cs.CL]23Nov2024

## AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability

Fei Zhao*, Taotian Pang*, Chunhui Li, Zhen Wu†, Junjie Guo, Shangyu Xing, Xinyu Dai National Key Laboratory for Novel Software Technology, Nanjing University

{zhaof,pangtt,lich,guojj,xsy}@smail.nju.edu.cn,{wuz,daixinyu}@nju.edu.cn

https://aligngpt-vl.github.io

### Abstract

POPE

VisWiz

SQA-I

86.0 54.2 68.5

Multimodal Large Language Models (MLLMs) are widely regarded as crucial in the exploration of Artificial General Intelligence (AGI). The core of MLLMs lies in their capability to achieve cross-modal alignment. To attain this goal, current MLLMs typically follow a two-phase training paradigm: the pre-training phase and the instructiontuning phase. Despite their success, there are shortcomings in the modeling of alignment capabilities within these models. Firstly, during the pre-training phase, the model usually assumes that all image-text pairs are uniformly aligned, but in fact the degree of alignment between different imagetext pairs is inconsistent. Secondly, the instructions currently used for finetuning incorporate a variety of tasks and different tasks usually require different levels of alignment capabilities, but previous MLLMs overlook these differentiated alignment needs. To tackle these issues, we propose a new multimodal large language model AlignGPT. In the pre-training stage, instead of treating all image-text pairs equally, we divide them into different groups according to the degrees of alignment of them. Then, the model is trained to learn the representations of different alignment levels. In the instruction-tuning phase, we adaptively combine these representations of alignment levels to meet the dynamic alignment needs of different tasks. Extensive experimental results show that our model achieves competitive performance on 12 benchmarks.

85.1

VQA-V2

79.1

MME

74.6 32.9

1527.4

60.9 1332.1

60.3

62.9

GQA

43.1

52.3

29.1

67.3

MMBench

66.5

30.8

SEED-I

InstructBLIP-7B

68.4

MiniGPT-v2-7B

Qwen-VL-Chat-7B

LLaVA_W

LLaVA-1.5-7B

AlignGPT-7B

MM-Vet

59.9

MMBench-CN

Figure 1. AlignGPT achieves competitive performances on a broad range of vision-language tasks compared with other generalist models. To facilitate observation, we only show the performance of MiniGPT-v2 and AlignGPT.

the pursuit of AGI, this cross-modal understanding and processing capability is essential, as it mimics how humans interact with the world and comprehend complex information through different senses, such as vision and language. The development of multimodal large language models not only advances the field of artificial intelligence but also provides machines with a way to process and understand information that is closer to human cognition.

Currently, MLLMs typically adhere to a unified training paradigm, which is divided into two key phases: the pretraining phase and the instruction-tuning phase [3, 5, 19, 25, 40, 41, 43, 46, 47]. The pre-training phase concentrates on aligning images with text, aiming to train the model to understand the relation between image contents and the respective textual descriptions of them. This alignment imbues the model with cross-modal comprehension abilities. The instruction-tuning phase further enhances its adaptability to specific tasks. This includes enabling the model to complete particular visual-language tasks based on given

### 1. Introduction

Multimodal Large Language Models (MLLMs) are considered a crucial step towards achieving Artificial General Intelligence (AGI) [1, 13, 30, 31]. The uniqueness of these models lies in their ability to integrate and understand various types of information, especially text and image data. In

*Equal contributions. †Corresponding author.

[Figure 1]

|0.186|
|---|

[Figure 2]

|0.220|
|---|

|0.256|
|---|

|0.297|
|---|

[Figure 3]

[Figure 4]

a ferrari 458 superfast at the geneva motor show

two road curves in the middle of a mountain range

cabin crew at an airport two ubuntu logo

Figure 2. Examples of image-text pairs in the pre-training dataset, where the numbers in each image represent the CLIP similarity.

instructions, such as generating textual descriptions from images, answering questions related to images, or even performing complex reasoning based on both text and images. This training paradigm equips multimodal pre-trained models with not only fundamental cross-modal understanding but also the flexibility to adapt to diverse task demands.

the higher scores indicate the higher degrees of alignment [17, 36]. For example, in Fig. 2, the degree of alignment of each image-text pair rises from left to right and the CLIP score of each pair also increases. Subsequently, we utilize these group labels as control signals to make the model to learn the representations of different alignment levels. During the instruction-tuning phase, the model is trained to dynamically combine these representations obtained by pre-training for the instructions of each task. In this process, we not only assign global alignment capabilities but also adaptively configure different local alignment capabilities according to the alignment needs for instructions of each task. The broad range of tests conducted demonstrates that our model achieves competitive performance across 12 benchmarks, as shown in Fig. 1.

Although current MLLMs have made great progress, the modeling of alignment capabilities of them is still insufficient for the following reasons:

- • The degree of alignment is inconsistent between different image-text pairs: During the pre-training phase, the model typically operates on a key assumption that all image-text pairs are consistently aligned. However, the degree of alignment in image-text pairs is not always uniform: in some image-text pairs, the text may describe the whole image (as shown in the rightmost of Fig. 2) while in others the text only describes a part of the image (as shown in the left three image-text pairs in Fig. 2) . If these differences are not differentiated during the pre-training phase, it could lead to a misunderstanding of the imagetext alignment relationships in the learning process.
- • The different tasks require different levels of alignment capabilities: The instructions currently used for finetuning cover a variety of tasks. Some of them, like image captioning [42], rely more on global image-text alignment capabilities. In contrast, other tasks, such as visual question answering (VQA) [2], typically require the model to answer questions based on specific parts of the image, which necessitates not only global image-text alignment but also local image-text alignment capabilities. However, previous work has neglected these differentiated alignment requirements.

Our contribution can be summarized as follows:

- • We propose a new multi-modal large language model AlignGPT to elevate and empower the alignment capabilities of MLLMs.
- • We propose a novel alignment strategy that learns different alignment levels in the pre-training stage, and then adaptively combines these alignment levels in the instruction-tuning stage to meet the needs of alignment capabilities for different tasks.
- • We conduct evaluations across multiple academic benchmarks and multimodal instruction-following benchmarks. Extensive experimental results show that our proposed AlignGPT achieves competitive performance. Further analysis verifies the effectiveness of the model.

### 2. Related Work

In this section, we review the existing studies on large language models and visual language models.

To effectively enhance the alignment capabilities, we propose a new multimodal large language model called AlignGPT. In the pre-training phase, we aim to make the model to understand different degrees of the image-text alignment relation. Specifically, instead of treating all image-text pairs equally, we divide image-text pairs into different groups according to the degrees of alignment of them and give an extra group label to each pair. This process is achieved by the help of CLIP scores [34], where

Large Language Models. In the field of natural language processing, BERT [11] and GPT-2 [33], as pioneering large pre-trained language models, marked a significant breakthrough in this technological direction. Their training on vast web text datasets demonstrated unprecedented language understanding and generation capabilities. Subsequently, the launch of GPT-3 [4] further accelerated the de-

velopment of this field, with its large model parameters and extensive training datasets showcasing exceptional abilities in few-shot learning, significantly enhancing task adaptability and flexibility. Following this, the introduction of InstructGPT and ChatGPT [32] focused on optimizing the efficiency and naturalness of interactions between models and humans, where InstructGPT enhanced the capability to execute precise instructions, and ChatGPT improved the conversational experience, making these models more fluent in human-computer communication. Meanwhile, as large language model (LLM) technology continued to evolve, emerging models like LLaMA [39] and GLM [14] began to make their mark. To equip these models with the ability to respond to human instructions similar to ChatGPT, research teams finetune LLaMA and GLM using high-quality instruction datasets, thereby further enhancing its capability to follow instructions, with representative projects such as Alpaca [38], Vicuna [9], and ChatGLM [45].

Although these models have made significant progress in interacting with humans through language, we recognize that human understanding and processing of complex information relies not only on language but also critically on visual and other sensory inputs. The observation has driven us to further explore more comprehensive visual-language models in order to more accurately simulate complex interactions between humans and the real world.

Visual Language Models. In recent years, multimodal large language models (MLLMs) have garnered increasing attention. The core of MLLMs lies in their ability to achieve cross-modal understanding and generalization. Most current models, such as LLaVA [25], MiniGPT-4 [47], mPLUG-Owl [43], Qwen-VL [3], MiniGPT-v2 [5], NExTGPT [41], InternLM-XComposer [46], CogVLM [40], and MM1 [29], utilize a standard training framework consisting of two primary phases: pre-training and instructiontuning. In the pre-training phase, the model utilizes image caption data to establish a rich understanding of crossmodal semantic knowledge. This training phase enables the model to comprehend and capture the correlation between images and text, establishing a solid foundation for subsequent stage. In the instruction-tuning phase, the model receives specific task instructions to optimize its performance on that task. Through this instruction-tuning phase, the model can further refine its understanding to execute specific tasks, enabling it to flexibly and accurately address various task requirements in practical applications.

Although current MLLMs have achieved promising results, they overlook two critical factors. First, the degree of alignment between different image-text pairs is inconsistent during the pre-training phase. Second, different tasks require different levels of alignment capabilities during the instruction-tuning phase. As a result, the modeling of align-

80000

70000

60000

50000

Samples

40000

30000

20000

10000

0

0.10 0.15 0.20 0.25 0.30

Figure 3. The distribution of CLIP similarity scores between images and texts in the pre-trained dataset.

ment capabilities in these models remains inadequate. To address these limitations, we propose a new multimodal large language model AlignGPT to effectively enhance the alignment capabilities of MLLMs.

### 3. Methodology

In this section, we initially present the fundamental structure of the visual-language model AlignGPT, followed by a demonstration of how to enhance the alignment capability of the model by our pre-training and instruction-tuning paradigms.

#### 3.1. Architecture

AlignGPT consists of four components: a visual backbone, a linear projection layer, a large language model, and an alignment module. Fig. 4 provides an overview of the AlignGPT architecture and its training process. The followings are the implementation details of these components:

Visual backbone. We utilize the pre-trained CLIP visual encoder ViT-L/14 [34] as our visual backbone. We train the model using an image resolution of 336×336.

Linear projection layer. We adopt a linear projection layer to map the representations of images from the vector space of the vision backbone to that of the language model.

Large language model. We choose the open-source model Vicuna [9] as our language model backbone, given its strong ability to follow instructions effectively in various language tasks.

Alignment module. We propose to add alignment embeddings to the inputs of MLLMs to enrich their alignment capabilities. These alignment embeddings are positioned

[Figure 5]

❄

Pre-training

Large Language Model

Caption

... ... ...

[Figure 6]

🔥 🔥

[Figure 7]

Projector

Indexing

Tuned

[Figure 8]

❄

Visual Encoder

Frozen Alignment Embedding Table

[Figure 9]

❄

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

🔥

...

Text

Image

Clip Score Bucketing

weak strong

[Figure 15]

[Figure 16]

🔥

Large Language Model

Instruction-tuning

Response

... ... ...

...

[Figure 17]

🔥

Gate

[Figure 18]

Projector 🔥

...

...

[Figure 19]

+

[Figure 20]

❄

Visual Encoder

Alignment Embedding Table

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

❄

[Figure 27]

...

Text

Image

...

[Figure 28]

[Figure 29]

weak strong

Figure 4. The architecture of AlignGPT.

scores correspond to texts that describe only partial regions of the image, indicating weaker alignment. In contrast, pairs with higher scores reflect texts that provide a more comprehensive description of the image, suggesting a stronger alignment between the text and the image [17, 36].

ahead of the image embeddings and text embeddings. In the subsequent sections, we will elaborate on the role of the alignment embeddings and the process to acquire them.

#### 3.2. Alignment of Image and Text

In our methodology, we utilize the similarity scores generated by the CLIP [34] model to evaluate the degree of alignment between images and text. As shown in Fig. 2, we present four image-text pairs and the CLIP scores of them. From left to right, the degree of alignment between each image-text pair rises, i.e., the text description becomes more comprehensive. Correspondingly, the CLIP score of each pair increases. The rationale of adopting CLIP similarity scores lies in that CLIP is pre-trained on a massive dataset of paired images and their corresponding textual descriptions, which enables it to effectively capture the relationship between visual and linguistic information. By employing contrastive learning techniques [8], CLIP minimizes the distance between representations of similar image-text pairs while maximizing the distance between those of different pairs. This training approach relies on 400 million data pairs, allowing the model to develop a nuanced understanding of image-text relationships.

#### 3.3. Alignment Level-aware Pre-training

As mentioned before, in the pre-training stage, the model usually assumes that all image-text pairs are uniformly aligned, and these pairs are used to train the model to comprehend the relations between images and their corresponding textual descriptions. However, in reality, the degree of alignment between these image-text pairs may vary considerably. Overlooking the difference could lead to a misunderstanding of the image-text alignment relations during the learning process.

Instead of treating all image-text pairs equally, we divide image-text pairs into different groups according to the degree of alignment of them and give each pair an extra group label. To achieve this, we leverage the similarity scores provided by CLIP. The higher the CLIP score is, the stronger the alignment is between image and text. Subsequently, we use these group labels as control signals to train the model, enabling it to understand the different alignment relations between different image-text pairs.

Beside, we also demonstrate the CLIP similarity distribution of image-text pairs in the pre-training dataset in Fig. 3. The results indicate that the CLIP similarity distribution varies significantly, suggesting a substantial difference in the alignment between images and texts. By jointly observing Fig. 2 and Fig. 3, we find that pairs with lower

More precisely, we start by computing the CLIP similarities s for all training image-text pairs. Subsequently, we rank all image-text pairs based on their similarity scores. Finally, we use a bucketing technique to divide them into N

discrete alignment levels. The process can be expressed as: l = bucket(s), l ∈ {1,2,...,N}, (1)

where bucket(·) denotes a bucketing function that assigns each pair into one of N equally spaced intervals and l is the alignment level (i.e., the group label) of an image-text pair. In this way, image-text pairs with lower CLIP similarity scores are assigned to buckets indicative of lower alignment levels, whereas those with higher CLIP similarity scores are grouped into buckets representing higher alignment levels.

Once the alignment level of each image-text pair is determined, we can regard it as a special token to express the alignment relation between the image and its textual description. This special token is placed ahead of the image and text tokens. During the pre-training phase, in addition to learning the mapping function in the linear projection layer, we also initialize this special token as an alignment embedding and continuously update its representation.

#### 3.4. Adaptive Alignment-based Instruction-tuning

Currently, the instructions used for finetuning cover vari-

- ous tasks such as image captioning, visual question answering, and visual grounding, etc. These tasks place different requirements on the alignment capabilities. For example, image captioning tasks mainly rely on global alignment between images and text, while VQA and visual grounding tasks require not only global alignment but also local alignment capabilities between images and text. To equip the model with the adaptive alignment capability, we propose an adaptive alignment-based instruction-tuning paradigm, which dynamically combine the alignment embeddings to meet the alignment needs for each task.

To this end, we first clarify how to represent the global and local alignment capabilities between image-text pairs. As mentioned in Section 3.3, after the pre-training stage, we obtain N alignment embeddings {H1,H2,...,HN} corresponding to N discrete alignment levels {1,2,...,N}. Among them, HN represents the highest level of alignment, i.e., HN indicates that the text provides very comprehensive description of an image. Here we regard it as a global alignment embedding. The embeddings below HN represent different degrees of alignment between the image and the text (i.e., {H1,H2,...,HN−1}), which means the text only describes a part of the information of the image from weak to strong. Thus, we regard them as local alignment embeddings of varying degrees.

Afterwards, we not only allocate global alignment capabilities to the instructions of each task, but also adaptively distribute varying degrees of local alignment capabilities based on the distinct alignment needs for each instruction. The reason behind this is that global alignment serves as the foundation for cross-modal understanding; only by mastering global alignment capabilities can a model truly

focus on enhancing local alignment abilities. Specifically, in addition to the global alignment embeddings, we assign different weights to the local alignment embeddings via a gate network. These weights are obtained based on input instructions and image, as the input instructions greatly influence the visual regions the model should focus on. The implementation of the gate network is as follows:

α = softmax(W(HI ⊗ HT) + b), (2)

where HI and HT denote the embeddings of the input instruction and the image, W and b are weight matrix and bias, α means the weights of local alignment embeddings. Finally, we aggregate the global alignment embedding and the local alignment embeddings with varying weights to ensure a more precise fulfillment of alignment requirements for instructions of each task:

N−1

αHi, (3)

Halign = HN +

i=1

where Halign means the final alignment embedding for each instruction during the instruction-tuning stage.

In general, we can regard the alignment embeddings obtained in the pre-training phase as foundational components, each of which has different alignment capabilities. During the instruction-tuning phase, we dynamically combine these components to meet the alignment needs for instructions of different tasks.

### 4. Experiments

#### 4.1. Experimental Settings

Datasets. For a fair comparison, we use the same pretraining and instruction dataset as the LLaVA-1.5 [24]. It mainly includes 558K caption pairs for modality alignment and 665K single- or multi-round conversations for instruction-tuning. Besides, we evaluate AlignGPT on a range of academic visual question answering (VQA) tasks and recent benchmarks designed specifically for MLLMs. This evaluation spans 12 benchmarks, including VQAV 2 [16], GQA [18], VizWiz [18], SQAI (ScienceQAIMG) [28], TextVQA [37], POPE [23], MME [15], MMB (MMBench), MMBCN (MMBench-Chinese) [26], SEEDI (SEED-Bench-IMG) [21], LLaVAW (LLaVA-Bench-inthe-Wild) [25], and MM-Vet [44] datasets.

Implementation Details. We adopt a ViT [12] model pretrained with CLIP [34] as a vision encoder to process visual inputs. On the language side, Vicuna [9] is utilized to handle multimodal features, ensuring a cohesive integration of text and visual data. In the pre-training phase, both the visual backbone and the large language model of AlignGPT

###### Method LLM Resolution POPE MME MMB MMBCN SEEDI LLaVAW MM-Vet

BLIP-2 Vicuna-13B 224 85.3 1293.8 - - 46.4 38.1 22.4 InstructBLIP Vicuna-7B 224 - - 36.0 23.7 53.4 60.9 26.2 InstructBLIP Vicuna-13B 224 78.9 1212.8 - - - 58.2 25.6 Shikra Vicuna-13B 224 - - 58.8 - - - IDEFICS-9B LLaMA-7B 224 - - 48.2 25.2 - - IDEFICS-80B LLaMA-65B 224 - - 54.5 38.1 - - MiniGPT-v2 LLaMA2-7B 448 85.1♢ 1332.1♢ 43.1♢ 29.1♢ 52.3♢ - Qwen-VL Qwen-7B 448 - - 38.2 7.4 56.3 - Qwen-VL-Chat Qwen-7B 448 - 1487.5 60.6 56.7 58.2 - LLaVA-1.5 Vicuna-7B 336 85.9 1510.7 64.3 58.3 66.2 63.4 30.5 LLaVA-1.5 Vicuna-13B 336 85.9 1531.3 67.7 63.6 68.2 70.7 35.4

AlignGPT Vicuna-7B 336 86.0 1527.4 67.3 59.9 66.5 68.4 30.8 AlignGPT Vicuna-13B 336 86.2 1572.0 69.5 63.7 67.8 75.2 35.6

- Table 1. Results on multimodal instruction-following benchmarks. For the baseline methods, the results on the SEEDI dataset are obtained from [7], and the other results are retrieved from [24].

Method LLM Resolution

Sample Size

VQAv2 GQA VisWiz SQAI TextVQA

Pre-train Finetune

BLIP-2 Vicuna-13B 224 129M - 41.0 41.0 19.6 61.0 42.5 InstructBLIP Vicuna-7B 224 129M 1.2M - 49.2 34.5 60.5 50.1 InstructBLIP Vicuna-13B 224 129M 1.2M - 49.5 33.4 63.1 50.7 Shikra Vicuna-13B 224 600K 5.5M 77.4 - - - IDEFICS-9B LLaMA-7B 224 353M 1M 50.9 38.4 35.5 - 25.9 IDEFICS-80B LLaMA-65B 224 353M 1M 60.0 45.2 36.0 - 30.9 MiniGPT-v2 LLaMA2-7B 448 - - 74.6♢ 60.3 32.9 60.9♢ 28.0♢ Qwen-VL Qwen-7B 448 1.4B 50M 78.8 59.3 35.2 67.1 63.8 Qwen-VL-Chat Qwen-7B 448 1.4B 50M 78.2 57.5 38.9 68.2 61.5 LLaVA-1.5 Vicuna-7B 336 558K 665K 78.5 62.0 50.0 66.8 58.2 LLaVA-1.5 Vicuna-13B 336 558K 665K 80.0 63.3 53.6 71.6 61.3

AlignGPT Vicuna-7B 336 558K 665K 79.1 62.9 54.2 68.5 58.4 AlignGPT Vicuna-13B 336 558K 665K 80.0 63.6 56.4 70.3 60.2

- Table 2. Performance comparison on multiple academic benchmarks. For the baselines, the results with ♢ are obtained by running the code released by the authors, and the other results are retrieved from [5, 24]. Best results are in bold.

remain frozen, with only the parameters of the linear projection layer and alignment embeddings being trained. During instruction-tuning phase, we freeze the alignment embeddings and visual backbone, while adjusting the parameters of the linear projection layer, large language model, and the gate network. The global batch sizes for the two phases are set at 256 and 128 respectively, with DeepSpeed [35] using ZeRO2 and ZeRO3 strategies accordingly. Regarding our training methodology, we conduct a single epoch of optimization for all models using the AdamW [27] optimizer coupled with a cosine learning schedule. Moreover, we initiate pre-training and instruction-tuning with learning rates of 1e-3 and 2e-5, respectively. The framework is trained on 8 A800 GPUs with 80GB memory.

#### 4.2. Compared Methods

We chose a diverse set of representative MLLMs as our baselines, including BLIP-2 [22], InstructBLIP [10], Shikra [6], IDEFICS [20], MiniGPT-v2 [5], Qwen-VL [3], Qwen-VL-Chat [3], and LLaVA-1.5 [24].

### 5. Results and Analysis 5.1. Main Results

MLLM-oriented Multi-modal Benchmarks. We apply AlignGPT to seven recent popular multimodal benchmarks, as shown in Tab. 1. We discover that, apart from LLaVA1.5-13B, AlignGPT-7B surpassed all previous multimodal models. This shows that our model has strong general-

###### Method Alignment Level VQAv2 GQA VisWiz SQAI TextVQA POPE MME MMB SEEDI

AlignGPT Number=4 79.0 62.9 52.3 68.7 58.3 86.2 1463.8 67.2 66.5 AlignGPT Number=6 79.0 62.7 51.2 68.9 58.3 85.8 1436.3 67.3 66.2 AlignGPT Number=8 79.1 62.9 54.2 68.5 58.4 86.0 1527.4 67.3 66.5 AlignGPT Number=10 79.1 62.6 53.0 67.8 58.4 86.2 1481.4 66.4 66.7

Table 3. Influence of different number of alignment level.

Settings Average Local Global VQAv2 GQA VisWiz SQAI TextVQA POPE MME MMB SEEDI

- (a) ✘ ✔ ✘ 79.1 62.7 53.3 67.9 58.6 85.9 1467.1 66.9 66.3

- (b) ✘ ✘ ✔ 79.1 62.9 52.6 68.3 58.4 85.9 1502.9 66.3 66.2

- (c) ✔ ✘ ✔ 79.0 62.8 52.5 68.6 58.4 85.6 1492.5 67.0 66.0

- (d) ✘ ✔ ✔ 79.1 62.9 54.2 68.5 58.4 86.0 1527.4 67.3 66.5

Table 4. Influence of local and global alignment.

POPE VisWiz

ization ability. Additionally, compared to LLaVA-1.5-13B, AlignGPT-13B has shown improvements on most datasets, particularly achieving good advancements on the MME, MMB, and LLaVAW benchamrks. This further validates the efficacy of both global and local alignment capabilities.

54.2

TextVQA

VQA-V2

86.0

85.7

79.1

79.0

58.4

58.2 68.2

49.1

SQA-I

68.5

62.6

62.9

GQA

Visual Question Answering. We evaluate AlignGPT using five popular academic benchmarks, as detailed in Tab. 2. Despite using less training data, the AlignGPT7B demonstrates competitive performance, surpassing other generalist models including InstructBLIP-13B, Shikra-13B, and IDEFICS-80B on most datasets, except for LLaVA-1.513B. These results verify the rationality of the structural design of our model. Moreover, considering that AlignGPT utilizes the same training dataset as LLaVA-1.5, it is evident that AlignGPT-7B outperforms LLaVA-1.5-7B across all evaluation datasets, and AlignGPT-13B also surpasses LLaVA-1.5-13B on the majority of datasets. This demonstrates that our approach effectively enhances the alignment capabilities of multimodal large language models. The fly in the ointment is that AlignGPT-13B does not perform as well as Qwen-VL on the TextVQA dataset. This may stem from the fact that TextVQA is a text-centric QA task, as it requires identifying text in images to answer questions. AlignGPT is tailored to boost multimodal alignment and might not exhibit strong results in text-focused scenarios.

1448.1

66.4

66.5

59.4

1527.4

65.5

MME

SEED-I

67.3

AlignGPT-7B (Random)

AlignGPT-7B

59.9

MMBench MMBench-CN

Figure 5. The performance comparison of AlignGPT (random) and AlignGPT in downstream datasets.

both methods have the same number of parameters, we can clearly assess whether the parameter number of the alignment embedding table influences the improvement in model performance. The experimental results are shown in Fig. 5. We find that the model using the randomly initialized alignment embedding table (referred to as AlignGPT (random)) shows a performance gap compared to AlignGPT. These results further confirm that the alignment information learned during pre-training is indeed the key factor in enhancing model performance, rather than the parameter number.

#### 5.2. Ablation Study

Without loss of generality, we choose AlignGPT-7b for the ablation study to analyze the impact of various components.

Impact of Number of Alignment Levels. To investigate the effect of the number of alignment levels N on AlignGPT, we vary the value of N in the range of [4, 10] with a step size of 2. Tab. 3 shows the performance of AlignGPT with different N on nine datasets. Actually, AlignGPT can achieve good results at N = 4, and their performance remains stable as the number of alignment levels

Impact of Alignment Embedding Table. We design an experiment to validate the role of the alignment embedding table. In the fine-tuning phase, we use a randomly initialized alignment embedding table instead of the embedding table obtained during the pre-training phase. Since

###### Method LLM Resolution VQAv2 GQA SQAI TextVQA POPE MMB SEEDI

AlignGPT Vicuna-7B 336 79.1 62.9 68.5 58.4 86.0 67.3 66.5 AlignGPT Vicuna-7B 672 79.7 63.3 68.3 60.3 86.8 67.2 66.5 AlignGPT Vicuna-7B 1008 79.8 63.4 68.2 60.3 86.8 67.2 66.6

- Table 5. Influence of different input image resolutions.

Method LLM Resolution VQAv2 GQA SQAI MME MMB MMBCN SEEDI

- AlignGPT LLaMA2-7B-Chat 336 79.1 62.9 65.9 1500.8 66.6 57.9 66.4 AlignGPT Vicuna-v1.5-7B 336 79.1 62.9 68.5 1527.4 67.3 59.9 66.5

- AlignGPT LLaMA3-8B-Base 336 79.6 63.1 70.4 1539.7 72.0 67.7 68.2

- Table 6. Influence of different large language models.

increases. Depending on the trajectory of the curve, their performance has an initial upward trend and then flattens

- out. These observations indicate that AlignGPT can improve the alignment capabilities of multi-modal large language models based on a small number of alignment levels. Finally, according to the trend of the curve, we set N to 8.

Impact of Local and Global Alignment. During the instruction-tuning phase, we assign global and local alignment capabilities to the instructions of each task. Among them, “Local” refers to the local alignment capabilities derived by assigning different weights to various local alignment embeddings using a gate network. “Global” denotes the global alignment capabilities, and “Average” represents the local alignment capabilities obtained by assigning equal weights to each local alignment embedding. The performance of these four strategies (settings a-d) is presented in Tab. 4. As we can see, setting (a) and setting (b) demonstrate divergent performances in downstream tasks, which can be attributed to the different demands these tasks place on global and local alignment capabilities. It is worth noting that setting (a) and setting (b) perform worse than our final approach (setting d) on most datasets, which verifies the necessity of the combination of global and local alignment capabilities. Moreover, the performance of setting (c) is inferior to that of setting (d), which may be due to the dynamically changing demands for local alignment capabilities across different downstream tasks.

#### 5.3. Discussion

Impact of different input image resolutions. Image resolution plays a crucial role in vision-language tasks as higher resolutions help reduce image blurring and enhance the understanding of image-text alignment. To evaluate the impact of resolution changes on the performance of multimodal tasks, we increase the image resolution from 336 to 1008, with the resulting performance changes detailed in

Tab. 5. The study results show that higher image resolutions can improve model performance on most multimodal tasks. For example, the score for VQAv2 increased from 79.1 to 79.8, while the score for TextVQA rose from 58.4 to 60.3. Meanwhile, the performance of the POPE improve by 0.8. These results highlight that appropriately increasing image resolution is an effective strategy for enhancing performance in studies of multimodal large language models.

Impact of different large language models. We also explore the impact of the large language model on the performance of AlignGPT, specifically testing three models: LLaMA-2-7B-Chat, Vicuna-v1.5-7B, and the latest LLaMA-3-8B-Base. The results are shown in Tab. 6. Initially, we observe that LLaMA-3-8B-Base achieves the best performance, followed by Vicuna-v1.5-7B, with LLaMA2-7B-Chat performing the worst, which is reasonable given LLaMA-3-8B-Base’s larger parameter size and richer training data. Besides, we observe that Vicuna-v1.5-7B achieves superior performance over LLaMA-2-7B-Chat on multimodal benchmarks such as MME, MMB, and SEEDI, while showing comparable results on VQA tasks. This advantage might be due to Vicuna-v1.5-7B undergoing supervised instruction-tuning with ShareGPT data, which contains background knowledge relevant to downstream tasks.

#### 5.4. Qualitative Results

Figure 6 presents a comparative analysis of our model with MiniGPT-v2 [5] and LLaVA-1.5 [24]. When a user submits an image alongside the instruction “Can you see the bright blue feathers on the bird’s wing?”, MiniGPT-v2 and LLaVA-1.5 both return an incorrect answer “Yes”. In contrast, our model produces accurate result “No”, thereby demonstrating that AlignGPT can effectively enhance the model’s alignment capability. In Figure 7, we further demonstrate the responses of AlignGPT under different levels of alignment capability. We find that with lower align-

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Can you see the bright blue feathers on the bird’s wing?

Can you see the bright blue feathers on the bird’s wing?

Can you see the bright blue feathers on the bird’s wing?

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Yes, I can see the bright blue feathers on the bird’s wing.

No, the bird in the image does not have bright blue feathers on its wing. It is a large bird of prey, possibly a hawk, with brown and white feathers.

Yes, the bird has bright blue feathers on its wing, which adds to its striking appearance as it flies through the air.

(a) MiniGPT-v2

(b) LLaVA-1.5

(c) AlignGPT

Figure 6. Comparison with other models.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Count the number of apples in the image

Count the number of apples in the image

Count the number of apples in the image

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

There are three apples in the image

There are four apples in the image

There are four apples in the image

(a) alignment level=1

(b) alignment level=4

(c) alignment level=7

Figure 7. The responses of AlignGPT under different levels of alignment capability.

ment levels, the model may only focus on certain regions of the image, resulting in an undercount of the total number of apples; whereas with higher alignment levels, the model considers the entire image area, thus achieving accurate apple quantity estimation. This finding once again underscores the necessity of enhancing the alignment capability of MLLMs.

on the specific demands of each instruction. Results from numerous experiments indicate that our AlignGPT achieves better performance than other state-of-the-art MLLMs.

### Limitations

The current study has two limitations: (1) This paper involves two modalities, i.e., text and image, while achieving AGI should also encompass video and audio, which requires us to do further research and exploration; (2) We propose a new perspective to enhance the alignment capability of MLLMs. However, there may be other methods to achieve this goal, which merit consideration in the future.

### 6. Conclusion

In this paper, we propose AlignGPT, a novel multimodal large language model designed to bolster the alignment capabilities of MLLMs. Our approach involves utilizing the alignment level of data as a control signal during pretraining to effectively handle the varying degrees of alignment in image-text pairs. Subsequently, in the instructiontuning phase, we begin by exploiting these control signals to shape different levels of alignment capabilities. Continuing from this, we go beyond assigning global alignment capabilities to instructions of each task; we also dynamically configure distinct local alignment capabilities based

### References

[1] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James

Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Ana¨ıs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023. 1

- [2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: visual question answering. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pages 2425–2433. IEEE Computer Society, 2015. 2
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. CoRR, abs/2308.12966, 2023. 1, 3, 6
- [4] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. 2
- [5] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. CoRR, abs/2310.09478,

2023. 1, 3, 6, 8

- [6] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. CoRR, abs/2306.15195, 2023. 6
- [7] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. CoRR, abs/2311.12793, 2023. 6
- [8] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, pages 1597–1607. PMLR,

2020. 4

- [9] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See

https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6,

2023. 3, 5

- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10

- 16, 2023, 2023. 6

- [11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics, 2019. 2
- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 5
- [13] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, pages 8469–8488. PMLR, 2023. 1
- [14] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335, 2022. 3
- [15] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. CoRR, abs/2306.13394, 2023. 5
- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 6325–6334. IEEE Computer Society, 2017. 5
- [17] Chanda Grover, Indra Deep Mastan, and Debayan Gupta. Contextclip: Contextual alignment of image-text pairs on CLIP visual representations. In Proceedings of the Thirteenth Indian Conference on Computer Vision, Graphics and

- Image Processing, ICVGIP 2022, Gandhinagar, India, December 8-10, 2022, pages 51:1–51:10. ACM, 2022. 2, 4
- [18] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 3608–3617. Computer Vision Foundation / IEEE Computer Society, 2018. 5
- [19] Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, and Zhuowen Tu. BLIVA: A simple multimodal LLM for better handling of text-rich visual questions. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, ThirtySixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 2256–2264. AAAI Press, 2024. 1
- [20] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. OBELICS: an open webscale filtered dataset of interleaved image-text documents. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10

- 16, 2023, 2023. 6

- [21] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. CoRR, abs/2307.16125, 2023. 5
- [22] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, pages 19730–

19742. PMLR, 2023. 6

- [23] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 292–305. Association for Computational Linguistics,

2023. 5

- [24] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. CoRR, abs/2310.03744, 2023. 5, 6, 8
- [25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. 1, 3, 5
- [26] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? CoRR, abs/2307.06281, 2023. 5

- [27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6
- [28] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28

- December 9, 2022, 2022. 5

- [29] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, Anton Belyi, Haotian Zhang, Karanjeet Singh, Doug Kang, Ankur Jain, Hongyu H`e, Max Schwarzer, Tom Gunter, Xiang Kong, Aonan Zhang, Jianyu Wang, Chong Wang, Nan Du, Tao Lei, Sam Wiseman, Guoli Yin, Mark Lee, Zirui Wang, Ruoming Pang, Peter Grasch, Alexander Toshev, and Yinfei Yang. MM1: methods, analysis & insights from multimodal LLM pretraining. CoRR, abs/2403.09611, 2024. 3
- [30] Seungwhan Moon, Andrea Madotto, Zhaojiang Lin, Tushar Nagarajan, Matt Smith, Shashank Jain, Chun-Fu Yeh, Prakash Murugesan, Peyman Heidari, Yue Liu, Kavya Srinet, Babak Damavandi, and Anuj Kumar. Anymal: An efficient and scalable any-modality augmented language model. CoRR, abs/2309.16058, 2023. 1
- [31] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774,

2023. 1

- [32] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. 3
- [33] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 2
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8748–

8763. PMLR, 2021. 2, 3, 4, 5

- [35] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2020, Virtual Event / Atlanta, Georgia, USA, November 9-19, 2020, page 20. IEEE/ACM, 2020. 6

- [36] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION400M: open dataset of clip-filtered 400 million image-text pairs. CoRR, abs/2111.02114, 2021. 2, 4
- [37] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 8317–8326. Computer Vision Foundation / IEEE, 2019. 5
- [38] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpaca: A strong, replicable instructionfollowing model. Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6):7, 2023. 3
- [39] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023. 3
- [40] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. CoRR, abs/2311.03079, 2023. 1, 3
- [41] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and TatSeng Chua. Next-gpt: Any-to-any multimodal LLM. CoRR, abs/2309.05519, 2023. 1, 3
- [42] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron C. Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, pages 2048–2057. JMLR.org, 2015. 2
- [43] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality. CoRR, abs/2304.14178, 2023. 1, 3
- [44] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. CoRR, abs/2308.02490, 2023. 5
- [45] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. GLM-130B: an open bilingual pre-trained model. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5,

2023. OpenReview.net, 2023. 3

- [46] Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang

Zhang, Haodong Duan, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. CoRR, abs/2309.15112, 2023. 1, 3

[47] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. CoRR, abs/2304.10592, 2023. 1, 3

