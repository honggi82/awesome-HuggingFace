arXiv:2404.12803v3[cs.CV]11Jun2025

# TextSquare: Scaling up Text-Centric Visual Instruction Tuning

Jingqun Tang1∗, Chunhui Lin1∗, Zhen Zhao2∗, Shu Wei1, Binghong Wu1, Qi Liu1, Yangfan He4,

Kuan Lu5, Hao Feng1, Yang Li1, Siqi Wang1, Lei Liao1, Wei Shi1, Yuliang Liu3, Hao Liu1, Yuan Xie2,Xiang Bai3, Can Huang1

1ByteDance Inc. 2 East China Normal University 3Huazhong University of Science and Technology 4University of Minnesota 5 Cornell University tangjingqun@bytedance.com, can.huang@bytedance.com

## Abstract

Text-centric visual question answering (VQA) has made great strides with the development of Multimodal Large Language Models (MLLMs), yet open-source models still fall short of leading models like GPT4V and Gemini, partly due to a lack of extensive, high-quality instruction tuning data. To this end, we introduce a new approach for creating a massive, high-quality instruction-tuning dataset, Square-10M, which is generated using closed-source MLLMs. The data construction process, termed Square, consists of four steps: Self-Questioning, Answering, Reasoning, and Evaluation. Our experiments with Square-10M led to three key findings: 1) Our model, TextSquare, considerably surpasses open-source previous state-of-the-art Text-centric MLLMs and sets a new standard on OCRBench (62.2%). It even outperforms top-tier models like GPT4V and Gemini in 6 of 10 text-centric benchmarks. 2) Additionally, we demonstrate the critical role of VQA reasoning data in offering comprehensive contextual insights for specific questions. This not only improves accuracy but also significantly mitigates hallucinations. Specifically, TextSquare scores an average of 75.1% across four general VQA and hallucination evaluation datasets, outperforming previous state-of-the-art models. 3) Notably, the phenomenon observed in scaling text-centric VQA datasets reveals a vivid pattern: the exponential increase of instruction tuning data volume is directly proportional to the improvement in model performance, thereby validating the necessity of the dataset scale and the high quality of Square-10M.

## 1 Introduction

Recent research on multimodal large language models (MLLMs) has achieved significant advancements in the text-centric visual question-answering(VQA) domain Ye et al. [2023a], Feng et al.

- [2023b], Liu et al. [2024d], Feng et al. [2023a], Zhao et al. [2025], Lu et al. [2024], Zhao et al.
- [2024b,a], with several closed-source state-of-the-art (SOTA) models leading the way. Two representative examples are GPT4V OpenAI [2023] and Gemini DeepMind [2023], which have demonstrated remarkable performance and have even surpassed human-level capabilities in certain aspects. Nevertheless, as illustrated in Figure 1, the performance of open-source models still lags significantly behind that of pioneering closed-source models. This phenomenon can be attributed to various factors, including model architecture, the scale of model parameters, image resolution, the volume of pretraining and instruction tuning data, and training strategies, among others.

∗ Equal contribution. Corresponding author.

Preprint. Under review.

AI2D InfoVQA

AI2D InfoVQA

79.0

79.0

75.2

51.5

78.7

78.2 75.1

TextVQA

ChartQA

TextVQA

ChartQA

50.4

79.4

73.9

79.4

76.1

78.0

78.5

70.3 45.1 69.5

72.7

74.6

44.5

74.1

70.5

66.8

68.8

67.6

66.1

68.4

67.3

WTQ

DocVQA

WTQ

DocVQA

66.8 49.7

84.3

88.4

49.7

65.1

88.1

47.5

81.6

57.9

84.3

36.1

78.2

45.5

49.6

32.9

39.8

49.3

66.5

51.5

30.2

59.6

28.7

32.3 67.9

25.3 49.8

573

514

TabFact

OCRBench

TabFact

OCRBench

34.6 42.8

511

597

84.2

69.3

622

645

659

622

84.2

80.4

68.9

62.3

51.7

578

41.2

26.4

38.7

34.2

36.7

46.5

41.9

43.5

49.3

78.5

43.2

48.5 79.7

51.8

48.4

79.6

84.5

48.9

48.3

83.4

80.3

86.6

86.6

52.2

85.9

53.2

53.2

SROIE

POPE

SROIE

POPE

58.9

68.0

60.7

61.2

64.5

62.3

64.5

71.2

63.8

64.9

78.0

71.8

71.8

67.1

64.5

80.3

81.8

77.2

POIE

GQA

POIE

GQA

83.7

78.0

71.4

71.4

VizWiz VQAv2

VizWiz VQAv2

(a) Comparison with closed-source models

(b) Comparison with open-source models

SquareText (2.2) GPT4V (2.4) Gemini Pro (3.5) DocOwl 1.5 (4.2) Llava Next 34B (4.7) Cogagent (5.9) Xcomposer2 (6.0) Monkey (7.1)

- Figure 1: The performance of TextSquare in various VQA tasks compared to existing models. (a) shows the comparison with state-of-the-art closed-source models (Gemini DeepMind [2023] and GPT4V OpenAI [2023]), and (b) shows the comparison with the leading open-source models. The numbers in parentheses after the model names in the legend indicate the average performance ranking across 10 text-centric multimodal benchmarks. TextSquare is marginally superior to GPT4V. Best viewed on screen.

Many pioneering studies Chen et al. [2024], Nayak et al. [2024], Chen et al. [2023], Zhang et al. [2023], Wang et al. [2025a,b], Li et al. [2024b], Sun et al. [2025] have recently conducted data-centric research into the challenges of insufficient instruction tuning data. For instance, Monkey Li et al. [2023c] initially employed expert models to generate descriptions of different aspects of images, which were then summarized by GPT-4 to produce high-quality and detailed image caption data. For better text-based knowledge injection, For better text-based knowledge injection, LLaVAR Zhang et al. [2023] and TG-Doc Wang et al. [2023] used GPT-4 to generate conversations for text-rich images by integrating OCR Liu et al. [2023b], Tang et al. [2022a,b], Zhao et al. [2024a], Tang et al. [2023, 2024], Feng et al. [2023c, 2024], Fu et al. [2024] results into the instructions. In order to improve the image caption ability for MLLMs, ShareGPT4V Chen et al. [2023] constructs a high-quality image caption dataset through GPT4V. While these efforts have achieved remarkable success, they have also left some challenges unresolved. Image caption data and VQA data belong to different domains, with inconsistencies in the granularity and scope of image content presentation. Furthermore, the scale of synthetic data remains relatively small, preventing MLLMs from fully realizing their potential. The exploration of methods that leverage large-scale text-centric VQA data for instruction tuning of existing open-source models remains limited.

To bridge the gap, this paper proposes a strategy termed Square for obtaining massive, high-quality text-centric VQA data from sophisticated and versatile closed-source MLLMs, resulting in the construction of a dataset (Square-10M) comprising tens of millions of instances for instruction tuning. Specifically, the method consists of four steps: Self-Questioning, Answering, Reasoning, and Evaluation. The self-questioning step involves utilizing the MLLM’s capabilities in text-image analysis and understanding to generate questions related to the textual content of images. The answering step involves answering these generated questions, leveraging various prompting techniques such as Chain-of-Thought and few-shot prompting. The reasoning step entails probing the model for the reasoning behind its answers, leveraging the powerful reasoning abilities of MLLMs. The evaluation step involves evaluating the question-answer pairs, assessing the validity of the questions and their relevance to the textual content of the images, as well as the correctness of the answers, thereby improving data quality and mitigating hallucinations. Overall, Square comprehensively leverages the capabilities of MLLMs in various aspects, significantly enhancing the data quality.

Besides, enriching the diversity of images is also crucial. We collect a diverse set of text-rich images from various public sources, including natural scenes, charts, tables, receipts, books, slides, PDFs, documents, products, and web images. Subsequently, deduplication is performed on this collection. By applying the Square method to these images, Square-10M is constructed.

Based on Square-10M, we achieve several remarkable results with extensive and rigorous experiments. First, as shown in Figure 1, our model (TextSquare) achieves comparable or superior performance to advanced closed-source models and substantially outperforms recent state-of-the-art open-source models on various benchmarks. It is notable that the image resolution of TextSquare is 700 and the parameters are 8.6B. Second, our experiments validate the beneficial impact of reasoning data on VQA tasks, demonstrating its ability to enhance model performance while mitigating hallucinations. With reasoning data for instruction tuning, TextSquare has a strong reasoning capability to provide elaborate explanations for VQA scenarios. Last but not least, by leveraging the dataset’s massive scale, we unveil the relationships between instruction tuning data scale, training convergence loss, and model performance. Whereas a few instruction tuning data can motivate MLLM well, it is not sufficient. Large amounts of high-quality data can still significantly reduce convergence loss and improve performance. The performance of TextSquare grows, and the loss of convergence decreases while continuously scaling up the instruction tuning data, which also demonstrates the effectiveness of our dataset.

In summary, the main contributions of this paper can be categorized into four points:

- • A high-quality dataset (Square-10M) comprising tens of millions of instances for text-centric VQA instruction tuning is constructed by comprehensively collecting text-rich images from various scenarios and employing the Square (Self-Questioning, Answering, Reasoning, and Evaluation) strategy on closed-source MLLMs.
- • Leveraging Square-10M, TextSquare achieves a significant outperformance of existing open-source models and even comparable or superior performance to SOTA closed-source models on various benchmarks, e.g., +0.9% on ChartQA, +2.1% on WTQ, +4.3% on SROIE. Notably, TextSquare outperforms GPT4V in overall rankings across ten text-centric benchmarks (ranking 2.2 v.s. 2.4).
- • Reasoning data is demonstrated to be beneficial in improving model performance and mitigating hallucinations in text-centric VQA scenarios, as it can deliver rich questionspecific contextual information.
- • Through extensive experiments, we reveal the relationships between data scale, convergence loss, and model performance for text-centric VQA instruction tuning, which demonstrates the effectiveness and necessity of Square-10M.

## 2 Related Work

### 2.1 Multi-modal Large Language Models

Recent work has increasingly focused on introducing visual knowledge into LLMs Zhu et al. [2023], Bai et al. [2023], Dai et al. [2024]. General attempts connect a visual encoder and an LLM with intermediate modules like Projector Liu et al. [2024c], Q-Former Li et al. [2023a], Perceiver Resampler Alayrac et al. [2022], etc, and go through pre-training alignment and instruction fine-tuning for vision-language understanding.

Recently, several researches Ye et al. [2023a], Feng et al. [2023b,a], Yu et al. [2023], Wei et al. [2023], Wan et al. [2024], Luo et al. [2024], Liu et al. [2024a] propose to enhance MLLMs’ capabilities in understanding textual elements (OCR, text-centric VQA, etc). Among them, mPLUG-DocOwl Ye et al. [2023a] creates novel instruction-following datasets to enhance the tuning process. TextMonkey Dai et al. [2024] adopts shifted window attention and filters out significant tokens. DocPedia Feng et al. [2023a] and HRVDA Liu et al. [2024a] enlarges input resolution to bridge the gap between MLLMs and visual document understanding.

Despite the extraordinary progress of existing open-source MLLMs, they still suffer from the huge gap against SOTA closed-source models like GPT4V OpenAI [2023] and Gemini Pro DeepMind [2023]. In this paper, we propose to mitigate this gap by training with large-scale and high-quality instruction-following data.

### 2.2 Text-Centric Visual Question Answering

Text-Centric Visual Question Answering aims to understand the interactions between the textual and the visual elements in the image. Donut Kim et al. [2022] first proposes an end-to-end training method based on a Transformer without OCR. Pix2Struct Lee et al. [2023] introduces a variableresolution input representation to adapt to document images. DoCo Li et al. [2024a] enhances the visual representation of the image encoder in LVLMs by aligning the document object of multi-modal inputs. BLIVA Hu et al. [2024b] enlarges the input token space by concatenating learned query embeddings and encoded patch embeddings. Several studies Feng et al. [2023b], Wang et al. [2023], Zhang et al. [2023] have performed data-centric attempts in this regard. UniDoc Feng et al. [2023b] constructs 600k document-oriented image-text pairs from PowerPoint presentations. LLaVAR Zhang et al. [2023] and TG-Doc Wang et al. [2023] prompt text-only GPT-4 to generate conversations for text-rich images by integrating OCR results into the instructions. These researches are restricted to small-scale annotations or generation based on unimodal inputs.

### 2.3 Generating Instruction-Tuning Data via LLMs

The success of LLMs has inspired recent work to employ them as training data generators Chen et al. [2023, 2024], Wang et al. [2022], Shao et al. [2023]. In this regard, we anchor on generating instruction-following data. Self-Instruct Wang et al. [2022] took the initial step towards synthesizing instructions via language models and improving the instruction-following capabilities. Llama-GPT4 Peng et al. [2023] uses GPT-4 to generate instruction-following data for LLM fine-tuning. Synthetic Prompting Shao et al. [2023] leverages a few handcrafted examples to prompt LLMs to generate more examples. Bonito Nayak et al. [2024] converts unannotated text into task-specific training datasets for instruction tuning. Recently, ALLAVA Chen et al. [2024] employs GPT4V to generate reasoning instructions and detailed answers from unlabeled images. All of the above attempts suffer from the low quality of the generated data and are typically performed on a small scale. In contrast, we collect massive text-centric images (i.e., tens of millions) and devise comprehensive generating methods and filtering rules to ensure the quantity and quality of the instruction tuning dataset.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Charts Document Slides Table Screenshot Receipt StreetView WebImage e-commerce

(a) Image Collection of Square-10M

Average QA tokens

Number of Images (K)

16.2 16.8 16.3 15.9

1000

1000

13.3 12.7

600 600

12.4 11.9

400

11.5

100

80

10

10

Street View

ecommerce

Chart PDF Slide Table Screen

Web Image

Street View

ecommerce

Chart PDF Slide Table Screen

Shot Receipt

Web Image

Receipt

Shot

40

60

300

500

900

40.9

44.2 43.1 44.4 46.2

1800 1900

49.8

55.6 56.8 55.4

3100

3500

Average Reasoning tokens

Number of QA pairs (K)

(c) Average QA and Reasoning Tokens

(b) Image and QA Distribution

- Figure 2: Overview of Square-10M: the distribution of images, the average tokens of the QAs, etc.

## 3 Square-10M: A Massive and High-quality Text-Centric VQA Instruction Tuning Dataset

Square-10M is synthesized by our proposed Square pipeline, i.e., Self-Questioning, Answering, Reasoning, and Evaluation.

### 3.1 Overview of Square

Figure 3 presents an overview of our proposed Square. Square generally consists of three stages for synthesizing high-quality instruction tuning data for text-centric VQA: (1) Data Collection for collecting large-scale images with textual elements of diverse properties. (2) Data Generation involves self-questioning, answering, and reasoning of the collected data. In this phase, the MLLM is prompted to generate VQA pairs based on the given image, as well as the reasoning behind its answers. (3) Data Filtering for self-evaluation of the generated content, aiming to discard meaningless questions and erroneous answers by employing the evaluation capabilities of MLLMs.

The above procedures result in our Square-10M dataset, standing out with its massive and high-quality text-centric VQA pairs and reasoning context. To be more specific, a total of 3.8 million images with rich textual elements are collected from diverse sources. After that, 20 million question-answer pairs are obtained from Data Generation. Finally, 9.1 million QA pairs as well as the reasoning context, are distilled with our Square strategy. A more precise analysis of Square-10M is depicted in Figure 2.

### 3.2 Data Collection

The data collection strategy is driven by the primary objective of encompassing a broad range of real-world text-rich scenarios. To this end, we collect 3.8 million unlabeled text-rich images (Figure

- 2). These images exhibit diverse properties. For instance, Chart and Table focus on textual elements with intense statistical information; Slide, Screenshot, and WebImage are designed for the interaction between text and prominent visual messages; Document/PDF, Receipt, and e-commerce contain images with fine and dense text; Street-View is derived from natural scenes. The collected images form a mapping of the textual elements in the real world and constitute the foundation of our research on text-centric VQA.

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Charts Document Slides Table Screenshot Web Image StreetView Receipt e-commerce

##### Questioning

- Q1: Who wrote this book ?
- Q2: Tell me the main idea of the book based on the textual content.
- Q3: What is the title of this book ?

Play an image content analysis expert. First, analyze all the image contents in a comprehensive manner. Then, generate several meaningful and distinct questions about the textual content of the image.

- A1: DR. BRIAN WILMOVSKY.
- A2: The main idea of the book is how to live a healthy, balanced life in an unbalanced world.
- A3: DREAM HEALTH.

Answering

Answer the following question based on all the contents in the image. The answer should be few words or phrases.

Reasoning

- R1: The text in the bottom of this image tells the author of this book.
- R2 & R3: ...

[Figure 19]

Based on the image and question-answer pair, think step by step and provide a detailed explanation for the answer.

Gemini

self-evaluation

Evaluation

##### self-evaluation

- E1: Yes. The answer (A1) correctly addresses the question (Q1).
- E2 & E3: ... multi-prompt & multi-context consistency

Play the role of a visual question answering expert. Judge the correctness of answers to questions based on image content and world knowledge. "Question": ... "Answer": ...

##### multi-prompt & multi-context consistency

Question: (Q1) Who wrote this book ? Based on varied context, answer the Question respectively.

- 1. Answer: DR. BRIAN WILMOVSKY.
- 2. Answer: BRIAN WILMOVSKY.
- 3. Answer: DR. BRIAN WILMOVSKY is the author.

- 1. The text in the bottom of this image tells the author of this book. (Reasoning)
- 2. Based on the example presented below ... (In-Context)
- 3. Respond to the subsequent question utilizing the image ... (Varied-Instructions) Judge whether the answers express consistent meanings.

E1: Yes. The answers express consistent meanings.

Figure 3: Pipeline for the proposed Square strategy. Gemini’s versatile multi-modal comprehension capabilities are utilized to synthesize Square-10M, which consists of four stages: self-questioning, answering, reasoning, and evaluation.

### 3.3 Data Generation: Self-Questioning, Answering, and Reasoning

We build our Square-10M dataset by employing the multi-modal understanding capabilities of Gemini Pro, one of the most advanced LLMs. For each image selected from a specific data source, Gemini Pro is instructed to generate VQA pairs and reasoning context through the subsequent three stages:

- Stage 1: Self-Questioning. In this stage, Gemini Pro is prompted to generate profound, meaningful, and non-trivial questions about the given image. We ask Gemini Pro to first comprehensively analyze the image and then raise questions based on its understanding, as shown in Figure 3. Considering that advanced MLLMs typically have a weaker understanding of the textual elements than visual elements, we also prepend the extracted text to the prompt by employing expert OCR models.
- Stage 2: Answering. Gemini Pro is then instructed to give appropriate answers to the generated questions. We leverage various prompting techniques to enrich the contextual information and improve the reliability of the generated answers, such as Chain-of-Thought and few-shot prompting. Figure 3 shows an example prompt for generating answers to a given question.
- Stage 3: Reasoning. We require Gemini Pro to elaborate on the detailed reasons behind its answers. Such an effort enforces Gemini Pro to think more about the connections between the questions and the visual elements, thus reducing hallucinations and providing accurate answers. Moreover, the generated reasons could serve as extra contextual information specific to individual questions, favoring possible research on the mechanism behind in-context learning. We present an example prompt for self-reasoning in Figure 3.

3.4 Data Filtering: Self-Evaluation and Answering Consistency

Despite the effectiveness of Self-Questioning, Answering, and Reasoning, the generated image-text pairs could face hallucinatory content, meaningless questions, and erroneous answers. We thus devise filtering rules based on the Evaluation capabilities of LLMs to select high-quality VQA pairs. The whole filtering system is established upon three aspects.

Self-Evaluation of MLLMs. We prompt Gemini Pro as well as other advanced MLLMs to judge whether the generated questions are meaningful and whether the answers are good enough to correctly address the questions.

Figure 3 depicts an example prompt for self-evaluation.

Multi-Prompt Consistency. Besides direct evaluation of the generated content, we manually augment the prompt and context space in Data Generation. A correct and meaningful VQA pair should be semantically consistent when provided with different prompts. Specifically, in the stage of Answering we provide Gemini Pro with different but semantically similar prompts to answer the given question. Then we discard the VQA pairs if the generated answers are not stable in semantics. An example is given in Figure 3.

Multi-Context Consistency. Similar to Multi-Prompt Consistency, we further validate the VQA pairs by prepending the question with varied context information. Given the generated question, three types of answers are produced by Gemini Pro with different contexts: (1) Answering with reasoning. Gemini Pro answers the question with a detailed explanation prepended (i.e., content generated in the stage of Reasoning). (2) In-Context answering. Gemini Pro answers the question with chain-of-thought or few-shot prompts prepended. (3) Naive answering. Gemini Pro answers the question with no extra context. We then discard the VQA pairs if the generated answers are not semantically consistent.

- 4 TextSquare: A Text-Centric Multimodal Large Language Model

### 4.1 Model Architecture

The model architecture of TextSquare follows the paradigm established by InternLM-Xcomposer2 Dong et al. [2024], including three integral components: (1) A Vision Encoder modified from OpenAI CLIP ViT-L-14-336 Radford et al. [2021], where the resolution is increased to 700 for improved performance. (2) A LLM based on InternLM-2 Cai et al. [2024], utilizing InternLM2-7B-ChatSFT as the practical variant. (3) A Projector, which semantically aligns the vision token and the text token.

### 4.2 Supervised Fine-Tuning with Square-10M

TextSquare is achieved by performing Supervised Fine-Tuning (SFT) with Square-10 M. The SFT process comprises three stages: In the first stage, we unfreeze all three components (i.e., the Vision Encoder, the LLM, and the Projector) and train the model in a resolution of 490. In the second stage, the input resolution is increased to 700, and only the Vision Encoder is trained to adapt to the resolution change. In the third stage, we further perform full-parameter fine-tuning in a resolution of 700. TextSquare demonstrates that with our Square-10M dataset, a model with 8B parameters and normal-size image resolution can achieve extraordinary performance on text-centric VQA, surpassing most available MLLMs and even the closed-source SOTA models.

5 Experiment

### 5.1 Implementation Details

The training data contains Square-10M and in-domain datasets (consistent with Monkey’s SFT data). The training process is divided into three phases, using the same data and the AdamW Loshchilov and Hutter [2017] optimizer with 64 A100-80G GPUs. In the first phase, we fine-tune InternLMXcomposer2 with full parameters, and the learning rate decreases from 1e-5 to 1e-6, taking about 9520 GPU hours; In the second phase we scale up the image resolution to 700, and train only VIT, with the learning rate decreasing from 1e-4 to 1e-5, taking about 7280 GPU hours; In the third stage, we perform full-parameter fine-tuning at 700 image resolution, and the learning rate drops from 1e-5 to 1e-6, spending about 12350 GPU hours.

### 5.2 Benchmark Evaluation

We report the results on Scene Text-centric VQA, Document-oriented VQA, Table VQA, Text-centric KIE, OCRBench, and General VQA for a comprehensive comparison of the performance of our model with existing models. The metrics of each benchmark are listed in Table 6 in the Supplementary Material.

- Table 1: Quantitative comparison of TextSquare with existing MLLMs on various text-centric benchmarks. “Res.” denotes image resolution. “*” denotes the results obtained through the opensource checkpoint or API of the closed-source model. The best results of each benchmark are bolded. The best results, except for closed-source models (GPT4V and Gemini Pro), are underlined.

|Method|Res.<br><br>|OCRBench|Document-Oriented DocVQA ChartQA InfoVQA<br><br>|Scene Text-Centric AI2D TextVQA<br><br>|Table VQA WTQ TabFact|KIE SROIE POIE<br><br>|
|---|---|---|---|---|---|---|
|UReader Ye et al. [2023b] Qwen-VL Bai et al. [2023] TextMonkey Liu et al. [2024d] Monkey Li et al. [2023c] Cogagent Hong et al. [2023] DocOwl 1.5 Hu et al. [2024a] Llava Next 34B Liu et al. [2024b] GPT4V OpenAI [2023] Gemini Pro DeepMind [2023] Xcomposer2 Dong et al. [2024]<br><br>|896 448 896 896 1120 1344 672 490|506 558 514<br><br>578∗ 597<br><br>573∗ 645 659 511<br><br>|65.4 59.3 42.2<br><br>65.1 65.7 73.0 67.1 -<br><br>66.5 65.1 36.1 81.6 68.4 44.5 81.6 70.5 50.4 78.2 67.3 45.1∗ 88.4 78.5 75.1 88.1 74.1 75.2 59.6 72.7 32.9<br><br><br>|- -<br>- 63.8<br><br><br>44.7 65.6 57.9∗ 67.6 49.6∗ 76.1<br><br>49.3 68.8 70.3 69.5 78.2 78.0 73.9 74.6 78.7 66.1|- -<br><br>- -<br><br><br>37.9 53.6 25.3∗ 49.8 30.2∗ 51.7∗<br><br>39.8 80.4 47.5∗ 68.9∗ 45.5∗ 69.3∗ 32.3∗ 67.9∗<br><br>28.7 62.3<br><br>|- -<br>- -<br><br><br>46.2 32.0 41.9 19.9 - 48.3 51.8 43.2∗ 46.5∗ 48.9∗ 41.2∗ 38.7∗ 34.6∗<br><br>34.2 49.3|
|TextSquare (ours)<br><br>|700|622<br><br>|84.3 79.4 51.5<br><br>|79.0 66.8<br><br>|49.7 84.2<br><br>|53.2 71.8<br><br>|

Document-Oriented Benchmark. While the documents have a clean background, dense text and complex typography pose distinct challenges. To effectively evaluate our model, we select representative benchmarks including DocVQA Mathew et al. [2021], ChartQA Masry et al. [2022], and InfographicVQA Mathew et al. [2022]. The results, detailed in Table 1, show that TextSquare outperforms all the open-source models in these three document-oriented VQA tasks with an average improvement of 3.5%, specifically, DocVQA 84.3% vs. 81.6% (Cogagent and mPLUG-DocOwl 1.5), ChartQA 79.4% vs. 72.7% (Intern-Xcomposer2), InfographicVQA 51.5% vs. 50.4% (mPLUGDocOwl 1.5). On the ChartQA dataset, TextSquare outperforms GPT4V and Gemini Pro by a slight margin. Note that TextSquare employs an image resolution of 700, which is smaller than most document-oriented MLLMs. Our model relies on comprehensive, high-quality VQA information specific to the text in the document, improving its ability to recognize and understand various document elements such as text, diagrams, infographics, and so on. If the image resolution is further

increased, it is believed that the model performance will be further improved, as demonstrated by Monkey et al.

Scene Text-centric Benchmark. The ability to answer text-based questions in images becomes an important aspect of the answering task as textual information is usually present in real-world scenes. In the evaluation, we utilize two datasets: TextVQA Singh et al. [2019] and AI2D Kembhavi et al. [2016]. As shown in Table 1, in this scenario, although TextSquare achieves SOTA performance on the AI2D dataset, there is no major improvement over our baseline Intern-Xcomposer2, which may be due to the fact that Intern-Xcomposer2 has been adequately optimized with high-quality in-domain data.

Table VQA Benchmark. Due to the complex structure of tables and the dense text, the understanding of the content of tables remains a challenging issue. In order to evaluate the performance of the comprehension of table content and structure, we choose two widely utilized datasets, Wiki Table Questions (WTQ) Pasupat and Liang [2015] and Table Fact (TabFact) Chen et al. [2019], as shown in Table 1. On the Table VQA benchmarks, TextSquare achieves optimal performance among the leading models with an average 3.0% improvement. This demonstrates that our model has reached a new level of table understanding, where high-quality generated table VQA and reasoning data play a key role.

Text-centric KIE Benchmark. Text-centric key information extraction tasks are frequently encountered in the information processing of various types of products, certificates, and receipts. We select a receipt information extraction dataset (SROIE) Huang et al. [2019] and a product information extraction dataset (POIE) Kuang et al. [2023], and the KIE task is converted to the VQA task. TextSquare achieves optimal performance in both datasets, with a major average lift of 14.8% (shown in Table 1). It is worth noting that there is no training set of POIE added to the training set, and there is not much data in the domain of product scenarios. This illustrates the extensive textual comprehension capabilities of our model.

OCRBench. OCRBench Liu et al. [2023a] is a comprehensive benchmark consisting of 29 OCR-related assessments, with text recognition, formula recognition, text-centric VQA, KIE, etc. TextSquare achieves optimal performance in OCRBench except for the closed-source models and becomes the first MLLM that exceeds 600 points with about 10B parameters. It indicates that the model performs well in both text-centric perception and comprehension tasks, especially in text recognition, where little in-domain data is included in the training set.

- Table 2: Quantitative comparison of our model with existing MLLMs on representative General VQA and hallucination evaluation benchmarks. VizWiz and POPE are relevant to both VQA and hallucination. Following Cogagent, we evaluate the adversarial part of POPE.

General VQA and Hallucination Evaluation VizWiz VQAv2 GQA POPEadv Average Qwen-VL Bai et al. [2023] 35.2 79.5 59.3 - Monkey Li et al. [2023c] 61.2 80.3 60.7 80.3∗ 70.6 Cogagent Hong et al. [2023] 36.7∗ 83.7 62.3∗ 85.9 67.2 DocOwl 1.5 Hu et al. [2024a] 43.5∗ 68.0∗ 48.5∗ 79.7∗ 59.9 Llava Next 34B Liu et al. [2024b] 63.8 83.7 67.1 83.4 74.5 GPT4V OpenAI [2023] 64.9∗ 77.2 48.4∗ 79.6∗ 67.5 Gemini Pro DeepMind [2023] 42.8∗ 71.2 52.2∗ 84.5∗ 62.7 Xcomposer2 Dong et al. [2024] 58.9∗ 81.8 64.5 78.5 70.9 TextSquare (ours) 71.4 78.0 64.5 86.6 75.1

Method

General VQA and Hallucination Evaluation Benchmark. General VQA requires the ability to learn both visual and textual information and a deep understanding of their inter-relationships. For general VQA, we validate on four benchmarks: VizWiz Gurari et al. [2018], VQAv2 Goyal et al. [2017], GQA Hudson and Manning [2019], and POPE Li et al. [2023b]. The VizWiz and POPE benchmarks are also relevant for hallucination evaluation. The results are shown in Table 2. On VQAv2 and GQA, TextSquare does not have a significant degradation compared to InternLMXcomposer2 and still maintains comparable performance. TextSquare exhibits superior capabilities in VizWiz and POPE, outperforming the closest competing method by an average of 3.6%. These results highlight the effectiveness of our approach, which is also able to mitigate model hallucinations in

particular with large-scale instruction tuning. We observe that it is partly attributed to the high-quality reasoning data that provides detailed explanations for VQA.

- 5.3 Qualitative Analysis

As illustrated in Figure 4, TextSquare has a formidable capability to provide plausible explanations of the answers to questions in a variety of text-centric VQA scenarios. Figure 4(a) shows that TextSquare has simple arithmetic capabilities. Figure 4(b) shows the ability to understand textual content and provide an approximate location in dense text. Figure 4(c) shows the comprehension of table structure and the ability to extract contextual information relevant to the question.

Answer : 2.13. Reason: To find the ratio of the people who approve and those who dont about Putin's handling of Corruption, we need to divide the number of people who approve by the number of people who dont. In this case, the number of people who approve is 62 and the number of people who don't is 29. Therefore, the ratio is 62:29, which simplifies to 2.13.

What is the ratio of the people who approve and those who don't about Putin's handling of Corruption? First answer the question and then give the reason. Answer: XV. Reason: It is the

[Figure 20]

page number shown at the bottom of the image.

What is the page number? First answer the question and then give the reason.

[Figure 21]

Answer: 1,146,000. Reason: From the image, we can see that the total number of deaths in prisons and camps is 69,000

+ 210,000 + 220,000 + 266,000 + 381,000 = 1,146,000.

what are the total number of deaths in prisons and camps? First answer the question and then give the reason.

[Figure 22]

[Figure 23]

|[Figure 24]|
|---|

[Figure 25]

|[Figure 26]|
|---|

(a) (b)

(c)

[Figure 27]

|[Figure 28]|
|---|

[Figure 29]

[Figure 30]

[Figure 31]

Figure 4: Qualitative results of VQA and reasoning for various text-centric scenarios.

Table 3: Ablation study on Incorporating Square-10M for Instruction Tuning.

|Model<br><br>|OCRBench|DocVQA ChartQA InfoVQA WTQ SROIE<br><br>|Average|
|---|---|---|---|
|Xcomposer2∗ TextSquare|571 622<br><br>|74.8 73.2 41.6 40.3 44.7 84.3 79.4 46.2 49.7 53.2|54.9 62.6<br><br>|

- 5.4 Ablation Study

The Effect of Incorporating Square-10M for Instruction Tuning.

In order to verify the effectiveness of Square-10M, we fine-tune the baseline model InternLMXcomposer2 on the public text-centric VQA instruction tuning dataset (consistent with Monkey’s training data). As shown in the Table, TextSquare substantially outperforms Xcomposer2∗ (finetuned) on various text-centric VQA benchmarks by 7.7%, which corroborates that Square-10M can fully exploit MLLM’s ability in text-centric VQA scenarios and that a large amount of high-quality instruction tuning data has a major improvement in performance.

Table 4: Ablation study on the evaluation step in the Square strategy.

Table 5: Ablation study on the VQA Reasoning data of Square-10M.

|Evaluation|DocVQA ChartQA WTQ<br><br>|
|---|---|
|w/ w/o<br><br>|84.3 79.4 49.7 81.7 77.2 46.9|

|Reasoning Data|DocVQA ChartQA POPEadv WizViz<br><br>|
|---|---|
|w/ w/o<br><br>|84.3 79.4 86.5 71.4 82.9 78.1 83.8 68.2|

The Effect of the Evaluation Step of the Square Strategy. As shown in Table 4, there is a distinct improvement in model performance after incorporating the evaluation of the generated VQA data, which verifies that the evaluation step of the Square strategy improves the quality of VQA instruction tuning data.

The Effect of VQA Reasoning Data on Model Performance and Hallucination Evaluation. From Table 5, we can find that VQA Reasoning data is helpful in both improving VQA performance and mitigating hallucinations. Specifically, in terms of enhancing VQA performance, there is a 1.4% and 1.3% gain on DocVQA and ChartQA. In terms of mitigating hallucinations, there is a 2.7% and 3.2% gain on POPE and WizViz.

(a) (b)

(c) (d)

Figure 5: The relationship between instruction tuning dataset scale, convergence loss, and model performance in text-centric VQA scenarios. Figure (a) and Figure (b) show the relationship between data scale and convergence loss, distinguished by a scaling of the horizontal coordinate of Figure (b) with log10. Figures (c) and (d) show the relationship between data scale and model performance, distinguished by a scaling of the horizontal coordinate of figure (e) with log10.

### 5.5 Relationships between Instruction Tuning Data Scale, Convergence Loss, and Model Performance

To explore the relationship between instruction tuning data scale, convergence loss, and model performance based on the merged large-scale Square-10M and the in-domain instruction tuning dataset, we conduct 10 sets of experiments for different data scales. The average performance of the models is evaluated on DocVQA, ChartQA, InfoVQA, WTQ, and SROIE. As shown in Figure 5(a)(b), the convergence loss of the model continues to decrease as the data scale grows, whereas the rate of decrease becomes progressively slower. The relationship between the convergence loss and the instruction tuning data scale approximately conforms to a logarithmic function. Similarly, from Figure 5(c)(d), it can be seen that as the instruction tuning data grows, the model performs better and better, but the rate of growth continues to slow down. Their relationship is also approximately in accordance with a logarithmic function. Holistically, there is a corresponding scaling law in the instruction tuning phase in text-centric VQA scenarios, where model performance is proportional to the logarithm of the scale of data. It can guide the construction of potentially larger datasets and predict model performance.

## 6 Limitation

Although our approach achieves remarkable results in various scenarios, there are some limitations. Firstly, large-scale data requires plenty of GPUs for long-term training, which greatly increases the training consumption. Second, while the Square strategy improves the quality of synthetic data, it still cannot reach the human level.

## 7 Conclusion

In this paper, we present the Square strategy for constructing a high-quality text-centric instruction tuning dataset(Square-10M). Leveraging this dataset, TextSquare significantly surpasses recent opensource models and even achieves performance comparable to GPT4V across various benchmarks. Furthermore, we derive the relationship between instruction tuning dataset scale, convergence loss, and model performance in order to pave the way for constructing even larger datasets. Our approach provides a data-centric perspective that revisits the role of instruction-tuning data in text-centric VQA, confirming that both the quantity and quality of data are crucial to model performance. We believe that there is a promising direction on how to further improve the data quantity and quality for closing the gap between open-source models and the leading ones.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.

Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024.

Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.

Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. Tabfact: A large-scale dataset for table-based fact verification. arXiv preprint arXiv:1909.02164, 2019.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36, 2024.

DeepMind. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. 2023. Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang,

Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.

Hao Feng, Qi Liu, Hao Liu, Wengang Zhou, Houqiang Li, and Can Huang. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding. arXiv preprint arXiv:2311.11810, 2023a.

Hao Feng, Zijian Wang, Jingqun Tang, Jinghui Lu, Wengang Zhou, Houqiang Li, and Can Huang. Unidoc: A universal large multimodal model for simultaneous text detection, recognition, spotting and understanding.

- arXiv preprint arXiv:2308.11592, 2023b.

Hao Feng, Zijian Wang, Jingqun Tang, Jinghui Lu, Wengang Zhou, Houqiang Li, and Can Huang. Unidoc: A universal large multimodal model for simultaneous text detection, recognition, spotting and understanding.

- arXiv preprint arXiv:2308.11592, 2023c.

Hao Feng, Qi Liu, Hao Liu, Jingqun Tang, Wengang Zhou, Houqiang Li, and Can Huang. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding. Science China Information Sciences, 67(12):1–14, 2024.

Ling Fu, Biao Yang, Zhebin Kuang, Jiajun Song, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, Mingxin Huang, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. arXiv preprint arXiv:2501.00321, 2024.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914, 2023.

Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895, 2024a.

Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, and Zhuowen Tu. Bliva: A simple multimodal llm for better handling of text-rich visual questions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 2256–2264, 2024b.

Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.

Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision, pages 498–517. Springer, 2022.

Jianfeng Kuang, Wei Hua, Dingkang Liang, Mingkun Yang, Deqiang Jiang, Bo Ren, and Xiang Bai. Visual information extraction in the wild: practical dataset and end-to-end solution. In International Conference on Document Analysis and Recognition, pages 36–53. Springer, 2023.

Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In International Conference on Machine Learning, pages 18893–18912. PMLR, 2023.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023a.

Xin Li, Yunfei Wu, Xinghua Jiang, Zhihao Guo, Mingming Gong, Haoyu Cao, Yinsong Liu, Deqiang Jiang, and Xing Sun. Enhancing visual document understanding with contrastive learning in large visual-language models. arXiv preprint arXiv:2402.19014, 2024a.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023b.

Zeyuan Li, Yangfan He, Lewei He, Jianhui Wang, Tianyu Shi, Bin Lei, Yuchen Li, and Qiuwu Chen. Falcon: Feedback-driven adaptive long/short-term memory reinforced coding optimization system. arXiv preprint arXiv:2410.21349, 2024b.

Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607, 2023c.

Chaohu Liu, Kun Yin, Haoyu Cao, Xinghua Jiang, Xin Li, Yinsong Liu, Deqiang Jiang, Xing Sun, and Linli Xu. Hrvda: High-resolution visual document assistant. arXiv preprint arXiv:2404.06918, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024b.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024c.

Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023a.

Yuliang Liu, Jiaxin Zhang, Dezhi Peng, Mingxin Huang, Xinyu Wang, Jingqun Tang, Can Huang, Dahua Lin, Chunhua Shen, Xiang Bai, et al. Spts v2: single-point scene text spotting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023b.

Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473, 2024d.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Jinghui Lu, Haiyang Yu, Yanjie Wang, Yongjie Ye, Jingqun Tang, Ziwei Yang, Binghong Wu, Qi Liu, Hao Feng, Han Wang, et al. A bounding box is worth one token: Interleaving layout and text in a large language model for document understanding. arXiv preprint arXiv:2407.01976, 2024.

Chuwei Luo, Yufan Shen, Zhaoqing Zhu, Qi Zheng, Zhi Yu, and Cong Yao. Layoutllm: Layout instruction tuning with large language models for document understanding. arXiv preprint arXiv:2404.05225, 2024.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.

Nihal V Nayak, Yiyang Nan, Avi Trost, and Stephen H Bach. Learning to generate instruction tuning datasets

for zero-shot task adaptation. arXiv preprint arXiv:2402.18334, 2024. OpenAI. Gpt-4v(ision) system card. 2023. Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. arXiv preprint

arXiv:1508.00305, 2015. Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Synthetic prompting: Generating chain-of-thought demonstrations for large language models. In International Conference on Machine Learning, pages 30706–30775. PMLR, 2023.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.

Wenhao Sun, Xue-Mei Dong, Benlei Cui, and Jingqun Tang. Attentive eraser: Unleashing diffusion model’s object removal potential via self-attention redirection guidance. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 20734–20742, 2025.

Jingqun Tang, Wenming Qian, Luchuan Song, Xiena Dong, Lan Li, and Xiang Bai. Optimal boxes: boosting end-to-end scene text recognition by adjusting annotated bounding boxes via reinforcement learning. In European Conference on Computer Vision, pages 233–248. Springer, 2022a.

Jingqun Tang, Wenqing Zhang, Hongye Liu, MingKun Yang, Bo Jiang, Guanglong Hu, and Xiang Bai. Few could be better than all: Feature sampling and grouping for scene text detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4563–4572, 2022b.

Jingqun Tang, Weidong Du, Bin Wang, Wenyang Zhou, Shuqi Mei, Tao Xue, Xing Xu, and Hai Zhang. Character recognition competition for street view shop signs. National Science Review, 10(6):nwad141, 2023.

Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri Faiz Bin Mahmood, Hao Feng, Zhen Zhao, et al. Mtvqa: Benchmarking multilingual text-centric visual question answering. arXiv preprint arXiv:2405.11985, 2024.

Jianqiang Wan, Sibo Song, Wenwen Yu, Yuliang Liu, Wenqing Cheng, Fei Huang, Xiang Bai, Cong Yao, and Zhibo Yang. Omniparser: A unified framework for text spotting, key information extraction and table recognition. arXiv preprint arXiv:2403.19128, 2024.

An-Lan Wang, Bin Shan, Wei Shi, Kun-Yu Lin, Xiang Fei, Guozhi Tang, Lei Liao, Jingqun Tang, Can Huang, and Wei-Shi Zheng. Pargo: Bridging vision-language with partial and global views. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7491–7499, 2025a.

Jianhui Wang, Zhifei Yang, Yangfan He, Huixiong Zhang, Yuxuan Chen, and Jingwei Huang. Mari: Material retrieval integration across domains. arXiv preprint arXiv:2503.08111, 2025b.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560, 2022.

Yonghui Wang, Wengang Zhou, Hao Feng, Keyi Zhou, and Houqiang Li. Towards improving document understanding: An exploration on text-grounding via mllms. arXiv preprint arXiv:2311.13194, 2023.

Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. arXiv preprint arXiv:2312.06109, 2023.

Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499, 2023a.

Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126, 2023b.

Yuechen Yu, Yulin Li, Chengquan Zhang, Xiaoqiang Zhang, Zengyuan Guo, Xiameng Qin, Kun Yao, Junyu Han, Errui Ding, and Jingdong Wang. Structextv2: Masked visual-textual prediction for document image pre-training. arXiv preprint arXiv:2303.00289, 2023.

Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023.

Weichao Zhao, Hao Feng, Qi Liu, Jingqun Tang, Binghong Wu, Lei Liao, Shu Wei, Yongjie Ye, Hao Liu, Wengang Zhou, et al. Tabpedia: Towards comprehensive visual table understanding with concept synergy. Advances in Neural Information Processing Systems, 37:7185–7212, 2025.

Zhen Zhao, Jingqun Tang, Chunhui Lin, Binghong Wu, Can Huang, Hao Liu, Xin Tan, Zhizhong Zhang, and Yuan Xie. Multi-modal in-context learning makes an ego-evolving scene text recognizer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15567–15576, 2024a.

Zhen Zhao, Jingqun Tang, Binghong Wu, Chunhui Lin, Shu Wei, Hao Liu, Xin Tan, Zhizhong Zhang, Can Huang, and Yuan Xie. Harmonizing visual text comprehension and generation. arXiv preprint arXiv:2407.16364, 2024b.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

- 8 Supplementary Material 8.1 Summary of the Evaluation Benchmarks We summarize the evaluation benchmarks used in this paper in Table 6.

Table 6: Summary of the evaluation benchmarks.

|Benchmark<br><br>|Description|Split<br><br>|Metric|
|---|---|---|---|
|DocVQA<br><br>|VQA on document images<br><br>|test|ANLS|
|ChartQA<br><br>|VQA on charts with visual and logical reasoning<br><br>|test|Relaxed Accuracy|
|InfoVQA|VQA on infographic images<br><br>|test|ANLS|
|AI2D<br><br>|Multiple choice VQA on science diagrams<br><br>|test|Accuracy|
|TextVQA|VQA involving reading and reasoning about text<br><br>|val|VQA Score|
|WTQ|VQA on semi-structured HTML tables sourced from Wikipedia<br><br>|test|Accuracy|
|TabFact|’Yes’ or ’No’ choice VQA about tables<br><br>|test|Accuracy|
|SROIE<br><br>|Key information extraction from receipts|test|Accuracy|
|POIE|Key information extraction on product images<br><br>|test<br><br>|Accuracy|
|VizWiz|Answering visual questions from blind people|val<br><br>|VQA Score|
|VQAV2|Open-ended VQA about natural images|val<br><br>|VQA Score|
|GQA<br><br>|Real-world visual reasoning and compositional question answering<br><br>|test-dev|Accuracy|
|POPE|Yes-or-No VQA to assess the object hallucination problem<br><br>|test(adversarial)|F1 Score|

