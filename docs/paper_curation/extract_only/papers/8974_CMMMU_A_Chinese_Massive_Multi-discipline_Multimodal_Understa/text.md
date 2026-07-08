# arXiv:2401.11944v4[cs.CL]4Nov2024

## CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark

###### Ge Zhang ,1,2∗Xinrun Du ∗Bei Chen9∗ Yiming Liang3,4 Tongxu Luo1 Tianyu Zheng ,9 Kang Zhu Yuyang Cheng1,5 Chunpu Xu6 Shuyue Guo9 Haoran Zhang1 Xingwei Qu Junjie Wang1,7 Ruibin Yuan ,1 Yizhi Li ,8 Zekun Wang ,9 Yudong Liu9 Yu-Hsuan Tsai9 Fengji Zhang9 Chenghua Lin ,8 Wenhao Huang ,9† Jie Fu1†

Multimodal Art Projection Research Community 1Hong Kong University of Science and Technology 2University of Waterloo 3Institute of Automation, Chinese Academy of Sciences 4School of Artificial Intelligence, University of Chinese Academy of Sciences 5Peking University 6The Hong Kong Polytechnic University 7Waseda University 8University of Manchester 901.AI

{zhangge,huangwenhao,duxinrun,chenbei}@01.ai jiefu@ust.hk

https://cmmmu-benchmark.github.io/

### Abstract

As the capabilities of large multimodal models (LMMs) continue to advance, evaluating the performance of LMMs emerges as an increasing need. Additionally, there is an even larger gap in evaluating the advanced knowledge and reasoning abilities of LMMs in non-English contexts such as Chinese. We introduce CMMMU, a new Chinese Massive Multi-discipline Multimodal Understanding benchmark designed to evaluate LMMs on tasks demanding college-level subject knowledge and deliberate reasoning in a Chinese context. CMMMU is inspired by and strictly follows the annotation and analysis pattern of MMMU (Yue et al., 2023). CMMMU includes 12k manually collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines. These questions span 30 subjects and comprise 39 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures. CMMMU focuses on complex perception and reasoning with domain-specific knowledge in the Chinese context. We evaluate 11 open-source LLMs and one proprietary GPT-4V(ision). Even GPT-4V only achieves accuracy of 43%, indicating a large space for improvement. CMMMU aims to enhance the development of next-generation LMMs for expert AI and support LMM democratization through offering varied language contexts.

### 1 Introduction

Large Multimodal Models (LMMs) have exhibited impressive problem-solving skills in many tasks, e.g., zero-shot image/video classification, zero-shot image/video-text retrieval, and multimodal question answering. But Yue et al. (2023); Lu et al. (2023); Deng et al. (2023) reveals a significant gap between advanced LMMs and multimodal expert AI, notably in complex perception and reasoning within specialized knowledge areas. To close this gap, college-level exams for different disciplines are a natural starting point for evaluating whether a Large Language Model (LLM) or an LMM can perform like an expert adult Yue et al. (2023); Hendrycks et al. (2021); Zhong et al. (2023); Zhang et al. (2023).

Additionally, with benchmarks a void, the development of bilingual LMMs has no sense of direction. We fill the gap by proposing CMMMU, a new comprehensive Chinese benchmark

∗These authors contribute equally to the work. †Corresponding Authors.

designed to evaluate LMMs on massive multi-discipline tasks, guiding the development of bilingual LMMs towards a path toward expert-level artificial intelligence.

As in Fig. 1, CMMMU, including 12k manually collected Chinese multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering, is one of the most comprehensive benchmarks for evaluating LMMs’ complex reasoning and perception abilities. Each question in CMMMU is further annotated with detailed subfields and image types to investigate which types of questions are difficult for LMMs.

We provide a comprehensive error analysis of 150 samples, which GPT-4V(ision) answers incorrectly, evenly distributed among 30 subjects, and covering most cases leading the most advanced LMMs to astray. By evaluating top-performing LMMs, e.g., QwenVL-Plus and GPT-4V, on CMMMU, we argue that there is still a long way to go towards an expert-level bilingual LMM. Even the most advanced closed-source LMMs, GPT-4V and Qwen-VL-Plus, only achieve accuracies of 43% and 36%, respectively, indicating significant room for improvement. We further reveal that the gap between LMMs released by the open-source community and the most powerful closed-source LMMs in a Chinese context is much smaller than in English, as demonstrated in MMMU.

For example, the most powerful opensource LMM, i.e., Yi-VL-34B, achieves an accuracy of 36%, with a 7% gap compared to GPT-4V, while the gap in English is 11%. In light of the insights obtained while developing CMMMU and benchmarking existing open-source LMMs, we observe that only Yi-VL-6B1, Yi-VL-34B2, and Qwen-VLChat perform notably better compared to a random choice setting and are close to GPT4V, while other open-source LMMs perform similarly to the random choice setting. Surprisingly, Yi-VL-34B even narrows the gap between open-source LMMs and GPT-4V on CMMMU to 7%.

Humanities & Social Sci.(9%)

[Figure 1]

Arts & Design(10%)

Technology & Engineering(27%)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Business(14%)

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Health & Medicine(17%)

Science(23%)

Figure 1: Disciplines of CMMMU.

We believe CMMMU can benefit the ongoing LMM research and development efforts, and promote the democratization of LMMs. Our contributions are summarized as follows:

- • We introduce CMMMU, the first Chinese Massive Multi-discipline Multimodal Understanding benchmark.
- • We reveal that existing LMMs, even including GPT-4V, perform poorly on complex reasoning and understanding in a Chinese context.
- • We examine the gap between open-source bilingual LMMs and closed-source LMMs in Chinese, finding it notably narrower than in English contexts.

### 2 Related Work

- 2.1 Multimodal Benchmark

Traditionally, multimodal benchmarks are task-oriented, thus not designed to evaluate LMMs.and benchmarking relies on tasks that align and utilize representations from various modalities, such as visual question answering (VQA) (Antol et al., 2015b), image captioning (Vinyals et al., 2014), and information retrieval (Wei et al., 2023; Wu et al., 2024). The success of building such multimodal tasks and benchmarks heavily relies on large-scale annotated datasets like MSCOCO (Lin et al., 2014) and Flickr30k (Plummer et al., 2015).

- 1https://huggingface.co/01-ai/Yi-VL-6B
- 2https://huggingface.co/01-ai/Yi-VL-34B

###### 科学 Science

###### 艺术设计 Arts & Design

Subcategory: 化学 Subfield: 有机化学，化学 性质，解离 Distribution: 考研题 Image Type: 化学结构 Difficulty Level: Middle <图片 1>

Question: 判断下面陈述对错：根据<图片 1>下面两个化 合物的pKa值，场效应起主要影响。

Subcategory: 音乐 Subfield: 旋律发展手法 Distribution: 本科 Image Type: 乐谱 Difficulty Level: Middle

Question: 下列谱例<图片 1>中的旋律发展手法是() Option:

[Figure 22]

[Figure 23]

Answer: 对

- (A) 时值减缩
- (B) 时值扩大

- (C) 倒影
- (D) 逆行 <图片 1>

###### 技术与工程 Technology & Engineering

###### 商业 Business

Subcategory: 电子学 Subfield: 电子技术，电路 分析 Distribution: 本科 Image Type: 电路图 Difficulty Level: Easy

Subcategory: 金融 Subfield: 金融管理，风险 管理

Question: 根据久期变化图<图片 1>可以看出收益率同等 变化幅度下，债券价格增加的幅度要超过债权减少的幅度， 该债券价格波动符合__特征

Question: 下图所示电路中，Y恒为0的图是（ ）

[Figure 24]

[Figure 25]

[Figure 26]

###### Option:

###### Option:

- (A) <图片 1>
- (B) <图片 2>

- (C) <图片 3>
- (D) <图片 4>

Distribution: 本科 Image Type: 图表

- (A) 反凸性
- (B) 正凸性

- (C) 极大值
- (D) 极小值

- <图片 1> <图片 3>
- <图片 2> <图片 4>

[Figure 27]

[Figure 28]

<图片 1> Difficulty Level: Middle

###### 人文社会科学 Humanities & Social Sci.

###### 健康与医学 Health & Medicine

Question:针刺不同穴位的镇痛效果如图<图片 1>，假设 H0：三穴位镇痛效果的分布相同， H1 ：三学位镇痛效果 的不同或不全相同，a=0.05，计算H=___。

Subcategory: 文献学 Subfield: 古代汉语，古文 字学

Subcategory: 公共卫生 Subfield: 卫生统计学，医 疗信息学，病理生理学

Question: 请根据下面汉字的演变过程，写出该字的楷书 形式（ ）甲骨文<图片 1>金文<图片 2>篆书<图片 3>

Answer: 妥

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Answer: 2.212

Distribution: 本科 Image Type: 书法 Difficulty Level: Middle

Distribution: 本科 Image Type: 表格

<图片 1> <图片 2> <图片 3>

<图片 1> Difficulty Level: Hard

Figure 2: CMMMU examples sampled from each discipline. The pictures include music scores, tables, chemical structures, curves, circuit diagrams and other types of pictures, and the difficulty of the questions requires expert-level knowledge to understand and reason.

Some work also evaluates the cross-modal alignment ability with VQA data derived from general knowledge bases (Marino et al., 2019; Schwenk et al., 2022).

A recent line of research attempts to design benchmarks tailored to evaluating LMMs. For example, we can examine the models by requiring them to perceive and learn the complicated knowledge from the given data distribution, e.g., in the scientific domain (Lu

- et al., 2022; Wu et al., 2024). To construct benchmarks compatible with generative LMMs, MME (Fu et al., 2023) uses yes-no problems, and MMBench (Liu et al., 2023) is based on the multi-choice format. Some recent studies propose examining whether models can perceive and interpret information produced in more challenging scenarios like math reasoning (Lu et al., 2023), website interaction Deng et al. (2023), or comprehensive collegelevel knowledge reasoning (Yue et al., 2023). Though promising progress in this field of multimodal benchmarking has been made, a dominant ratio of the dataset is in English, which makes it an urgent gap to build a comprehensive and challenging benchmark in other frequently used languages like Chinese.

2.2 Bilingual Large Multimodal Models

Different from the development trace of the benchmarks, many of the existing multimodal models support both English and Chinese due to the integrated bilingual large language models(LLMs). Although such a statement is established in different models, the cases may vary on nuanced features. While multimodal models aim to go beyond the textual data by adapting the language models with cross-modality alignment methods, some language models pre-trained with Chinese-English bilingual corpus are selected as the component for text modeling (Hu et al., 2023; Bai et al., 2023b; Ding et al., 2021; Du et al., 2022; LinkSoulAI, 2023). Although some interesting insights are explored, only a few of the models are evaluated on Chinese multimodal tasks. For instance, it is revealed by Hu et al. (2023) that multimodal models trained only with English instruction tuning data work well in Chinese even in the zero-shot setting. Another set of models selects the language models adapted to Chinese with efficient tuning (Cui et al., 2023). Given the proper alignment architecture designs and training data selection, these models still show strong performances on bilingual multimodal tasks (Ye et al., 2023; Sun et al., 2023; Chen et al., 2023; Wang et al., 2023; Hong

- et al., 2023; LinkSoul-AI, 2023). Moreover, even though the closed-source GPT-4 (Achiam et al., 2023) does not provide architecture-relevant details, it is a worth mentioning baseline

for Chinese multimodal benchmarks, given it achieves visual understanding tasks in English close to human-level.

Regardless of the choice of language models and the training data, many multimodal models show the capability for Chinese tasks at a certain level in practical use. In this work, we aim to quantitatively measure the ability boundaries of the models with comprehensive and challenging Chinese multimodal tasks, as most of them have been only assessed with English tasks.

### 3 The CMMMU Benchmark

We introduce the Chinese Massive Multi-discipline Multimodal Understanding (CMMMU) benchmark, a manually curated benchmark covering college-level knowledge to evaluate LMMs’ expert-level multimodal understanding capability across a broad scope of tasks. CMMMU is the first multimodal question-answering benchmark in a Chinese context and one of the few existing multimodal benchmarks investigating LMMs’ complex understanding and reasoning capacities.

Image Type #Num Image Type #Num Image Type #Num

历史时间线 Historical Timeline

广告 Advertisement

人体扫描 Body Scan

9 电力学符号 Electrical Symbols

6

4

DNA序列 DNA Sequence

数学符号 Mathematical Symbols

21 标志和品牌形象 Logos and Brand Identity

10

13

3D渲染图 3D Rendering

风景画 Landscape Painting

22

24 天文图像 Astronomical Images

23

其他 Other

图标和符号 Icons and Symbols

31

31

39 海报 Poster

树形图 Tree Diagram

雕塑 Sculpture

67 书法 Calligraphy

47

54

地图 Map

有向图 Directed Graph

85 建筑设计图 Architectural Design Drawing

72

82

机械工程图 Mechanical Engineering Drawings

病理图像 Pathology Images

107 流程图 Flowchart

94

99

乐谱 Sheet Music

系统框图 System Diagram

174 漫画和卡通 Cartoons and Comics

128

137

肖像 Portrait

绘画作品 Artwork

209

235

286

屏幕截图 Screenshot

几何形状 Geometric Shapes

机械结构图 Mechanical Structure Diagram

301

346 显微镜图像 Microscope Image

339

工程结构图 Engineering Structural Diagram

医学图像 Medical Images

517 电路图 Circuit Diagram

416

491

图表 Charts

化学结构 Chemical Structures

851 照片 Photographs

557

676

草图 Sketches

表格 Table

1680

2480

3180

Table 1: Image type and corresponding number.

- 3.1 Data Curation Process

Data Collection: We carefully design a three-stage data collection procedure. In Stage 1, annotator organizers (mainly the authors) collect sources satisfying license requirements in the format of website links or book titles. The annotator organizers are well instructed to adhere to copyright and license regulations, avoiding data from sites prohibiting copying and redistribution. We collect at least 20 annotation sources, i.e., websites or books, for each subject in each discipline. In Stage 2, annotator organizers forward the annotation sources to the crowdsourcing annotators for further annotation. All annotators are undergraduate students or have higher degrees to ensure they can verify the annotated questions and related explanations. During the annotation process, we ask the annotators to strictly follow several key principles to filter out unqualified questions with images: (1) Questions

Dataset Size Images Format Source Answer

VQA (Antol et al., 2015a) > 1M V I+T Annotated Open GQA (Hudson & Manning, 2019) > 1M V I+T Synthesized Open VizWiz (Gurari et al., 2018) 32K V I+T Annotated Open TextVQA (Ganz et al., 2023) 45K OC I+T Annotated MC OKVQA (Marino et al., 2019) 14K V+OC I+T Annotated Open SEED (Li et al., 2023) 19K V+OC I+T Annotated MC MMBench (Liu et al., 2023) 3K V+OC I+T Repurposed MC MM-Vet (Yu et al., 2023) 0.2K V+OC I+T Annotated Open ScienceQA (Lu et al., 2022) 6K 5 Types I+T Textbooks MC MathVista (Lu et al., 2023) 6K V+OC I+T Synthesized MC/Open

Textbooks Internet Annotated

Open MC

MMMU (Yue et al., 2023) 11.5K 30 Types Interleaved

Textbooks Internet Annotated

Open MC T/F

CMMMU 12K 39 Types Interleaved

Table 2: Comparison with other benchmarks. V: visual input, OC: optical characters, I+T: images and text, Open: open questions, MC: multiple choice questions, FIB: fill in the blank questions, T/F: true or false questions.

that can be answered without the images should be filtered out. (2) Questions that use the same image should be filtered out as much as possible. (3) Questions not requiring expert knowledge to answer should be filtered out as much as possible. (4) The number of questions that are about the same specific knowledge point and have similar question angles should not exceed 10. We also ask annotators to follow the data annotation protocol in the Appendix.G of Yue et al. (2023). In Stage 3, annotator organizers additionally supplement questions to subjects that lack questions, e.g., Arts, Diagnostics, and Economics, to balance the datasets.

Data Quality Control: To further improve the data quality of CMMMU, we follow a strict data quality control protocol. First, each question is manually verified by at least one of the paper’s authors. We carefully filter out questions with answers that are too hard to extract from the responses generated by LMMs. During the process, we also carefully filter out all the questions that are not up to college-level examinations. Second, given the concern of data contamination, we filter out all the questions that can be correctly solved by GPT-4, Qwen-7B, Deepseek-7B, and Yi-7B simultaneously without the assistance of OCR. Some example questions are shown in Fig. 2.

- 3.2 Comparison with Existing Benchmarks

We compare CMMMU with existing multimodal benchmarks in Tab. 2.From the input image type, the common image formats in the benchmark can be roughly divided into three simple categories: visual input (V), optical characters (OC) and V+OC. In addition, there are 5 types of image formats in the ScienceQA benchmark. CMMMU benchmark has 39 types as in Tab. 1, involving charts, tables, diagrams, chemical structures, photos, paintings, geometric shapes, musical scores, and medical images. Concerning the input format, existing benchmarks generally exhibit a relatively independent relationship between the input images and text(I+T). In the CMMMU benchmark, images and text are interleaved, establishing a markedly tighter connection. In terms of question types, most of the common benchmarks are open questions (Open) or multiple choice questions (MC). CMMMU not only contains open-ended questions and multiple choice questions, but also adds judgment questions to enrich the question types. In terms of knowledge depth, previous benchmarks typically require common sense or simple physical or temporal reasoning. In contrast, our proposed CMMMU benchmark requires thoughtful reasoning with university-level subject knowledge.

Statistics Number

Statistics Number Total Questions 12012 Disciplines/Subjects/Subfields 6/30/4165 Image Types 39

Multiple-choice Questions 7738 (64.41%) Fill in the blank Questions 2998 (24.95%) True or false Questions 1276 (10.62%)

Dev:Validation:Test 112:900:11000 Easy: Medium: Hard 30%:58%:12%

Questions with an Explanation 247 (2.05%) Image in the Question 11760 (84.42%) Image in Options 2169 (15.57%) Example with Multiple Images 597 (4.97%)

Average question length 51.12 Average option length 8.76 Average explanation length 78.29

Table 3: Statistics of CMMMU

- 3.3 Statistics of CMMMU

CMMMU covers 6 disciplines, including Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering, spanning over 30 subjects. As Fig.

- 3, CMMMU consists of 12K questions, divided into few-shot development set, validation set, and test set. The few-shot development set comprises 5 questions for each topic, the validation set aids in hyperparameter selection with 900 questions, and the test set includes 11K questions.

The pictures include 39 types such as pathological diagrams, musical scores, circuit diagrams, and chemical structure diagrams. We categorized the data as Easy (30%), Medium (58%), and Hard (12%) by logical difficulty rather than intellectual difficulty. According to the question type, there are 7738 multiple choice questions, 2998 fill-in-the-blank questions, and 1276 judgment questions. Of these examples, 11,760 are images in the question, 2169 are images in the option, and 597 are images with multiple images. The average question length is approximately 51 words, the average option length is about 9 words, and the average explanation length is around 78 words.

Validation Overall

Test Overall

Art & Design

Health & Medicine

Human. & Social Sci.

Tech & Eng.

Business Science

(900) (11,000) (1,091) (1,538) (2,494) (1,865) (1,038) (2,974)

Random Choice 21.6 21.6 32.9 9.1 18.8 23.8 23.8 23.9 Frequent Choice 24.1 26.0 36.2 11.8 23.9 30.2 28.5 27.7

###### LMMs: Text + Image as Input

mPLUG-Owl2 20.8 22.2 30.4 13.3 19.6 25.2 24.7 23.4 VisCPM 25.2 22.7 37.7 11.3 19.1 26.1 24.0 23.7 Chinese-LLaVA 25.5 23.4 34.4 11.7 21.6 25.5 26.3 24.7 Emu2-Chat 23.8 24.5 35.3 11.7 22.1 25.5 28.0 27.1 CogAgent-Chat 24.6 23.6 33.8 14.1 20.6 26.3 24.8 25.3 Qwen-VL-Chat 30.7 31.3 52.6 18.5 26.9 33.4 34.1 31.4 InternVL-Chat-V1.1 34.7 34.0 56.7 19.7 28.6 39.2 39.6 32.3 Yi-VL-6B 35.8 35.0 58.0 19.9 32.3 39.3 40.6 32.1 Yi-VL-34B 36.2 36.5 62.9 19.1 31.5 42.1 42.5 34.5

Qwen-VL-Plus 39.5 36.8 61.5 23.2 32.8 40.5 43.4 33.3 GPT-4V 42.5 43.7 61.0 36.3 40.9 46.8 44.2 41.5

###### LLMs: Only Text as Input

DeepSeek-7B 22.3 21.9 41.3 11.2 18.3 23.5 24.7 21.3 Baichuan-7B 26.0 24.3 42.7 12.6 19.6 28.0 27.8 23.9 Qwen-7B 24.7 25.1 43.8 12.6 20.7 30.5 26.9 24.5 Yi-6B 25.6 24.2 26.3 15.0 23.4 29.1 27.0 24.7

DeepSeek-7B + OCR 25.2 23.2 41.2 13.2 19.4 26.1 26.5 21.8 Baichuan-7B + OCR 25.3 24.7 40.2 15.2 21.0 27.9 30.7 22.8 Qwen-7B + OCR 27.0 26.1 44.6 14.3 22.1 29.3 29.8 25.4 Yi-6B + OCR 28.4 26.8 33.4 16.9 24.8 32.3 33.2 25.5

- Table 4: Overall results of open-source and closed-source models on the CMMMU validation and test set. bold results in LMMs indicate the best results for all models, and the blue results indicate the best results among the open-source models.

### 4 Experiments

We perform a comprehensive evaluation of various models, including LLMs and LMMs, with considering both closed-source and open-source implementations. The evaluation process employs zero-shot settings, rather than fine-tuning or few-shot settings, to examine the raw ability of the model to generate accurate answers on multimodal tasks. For models with corresponding task prompts, we use the default prompts for either multiple-choice or open-ended question-answering tasks. As for models without corresponding task prompts, we use the same task prompts, which are hand-picked on the validation set. In addition, we also test the results of some models on few-shot settings, which are documented in the Appendix. All these experiments are performed on NVIDIA A100 GPUs.

- 4.1 Baselines

LMMs. We consider the current mainstream Chinese-English bilingual large multimodal models. We use each model’s official API (closed-source) or official checkpoint (opensource) published on the huggingface website. Baselines includes: (1) mPLUG-Owl2 (Ye et al., 2023) employs a modular network design with a language decoder as a common interface for managing different modalities, effectively exploiting modal collaboration to improve performance in textual and multimodal tasks. (2) VisCPM (Hu et al., 2023) is trained based on the large language model CPM-Bee with 10B parameters, fusing visual encoder (Q-Former) and visual decoder (Diffusion-UNet) to support visual inputs and outputs. (3) Chinese-LLaVA (LinkSoul-AI, 2023) uses Chinese Llama2 as the language model base, plus image understanding capabilities. The work follows the structure of LLaVA with a two-stage training using Chinese data. (4) Emu2 (Sun et al., 2023) is a generative multimodal model with 37 billion parameters that performs well in few-shot Settings. (5) CogAgent (Hong et al., 2023) is a 180 billion-parameter Vision-Language Model designed for GUI comprehension and navigation. (6) Qwen-VL (Bai et al., 2023b) uses Qwen-7B as the initialization of the LLM, and Openclip ViT-bigG as the initialization of the visual encoder. And connects them with a randomly initialized cross-attention layer. We choose QWen-VL-Chat and QWen-VL-plus. (7) InternVL (Chen et al., 2023) scales up the Vision Transformer (ViT) to 6B parameters and aligns it with LLM. There are multimodal models with varying sizes of language models within the InternVL series, including InternVL-Chatvit-6B-Vicuna-7B, InternVL-Chat-vit-6B-Vicuna-13B, InternVL-Chat-vit-6B-Llama2-13B, and InternVL-Chat-V1.1. (8) GPT-4V 3 is a closed-source large multimodal model from OpenAI that accepts image and text inputs and emits text outputs, demonstrating human-level performance on a variety of professional and academic benchmarks. (9) Yi-VL-6B and Yi-VL-34B are our multimodal models, providing image understanding capabilities to large language models. In these models, Vit is the Openclip 224, and the language model is either Yi-6B-Chat or Yi-34B-Chat.

Question Type Question Difficulty

Models

Overall MC FIB T/F Easy Medium Hard

mPLUG-Owl2 22.9 7.0 53.8 25.5 20.8 20.7 22.2 VisCPM 24.5 5.4 52.8 26.8 21.1 20.1 22.7 Chinese-LLaVA 25.6 5.4 52.7 25.5 26.3 24.7 23.4 Emu2 28.4 2.9 51.4 28.0 22.4 25.1 24.5 CogAgent 25.9 5.9 51.9 27.7 21.7 22.7 23.6 InternVL-Chat-V1.1 36.7 14.4 63.5 41.8 30.8 29.4 34.0 Yi-VL-6B 40.8 11.7 54.9 43.3 31.6 30.3 35.0 Yi-VL-34B 42.5 10.4 61.6 45.6 32.6 31.9 36.5 Qwen-VL-Plus 42.9 15.7 49.4 46.7 32.9 29.9 36.8 GPT-4V 46.4 27.4 66.0 51.5 40.7 38.3 43.7

- Table 5: Combined result decomposition across question type and difficulty level. MC: multiple choice questions, FIB: fill in the blank questions, T/F: true or false questions.

3https://openai.com/research/gpt-4v-system-card

Text-only LLMs. We evaluate the performance of LLMs (e.g., GPT44, Qwen-7B (Bai et al., 2023a), Deepseek-7B (DeepSeek-AI et al., 2024), Yi-6B5) when dealing with plain text, and Baichuan-7B on multimodal data. In addition, to verify whether external image tools can enhance the performance of LLMs on multimodal data, we deploy OCR by Mathpix 6 processing images to convert certain image information into textual forms.

Evaluation. We build a systematic and rule-based evaluation pipeline. Robust regular expressions are built to extract answers from the model responses. Specifically, for multiplechoice questions, we directly use options as keywords to extract model responses, and take the one with the highest number of options in the model response as the answer. If there is no valid answer in the model’s response, random selection is performed for multiplechoice questions. For the judgment and open-ended question answering questions, we utilize specific rules to extract some segments where the answer may occur, and then detect whether the answer occurs in them. We add random selection and frequent selection as baselines: the former randomly selects an option, while the latter selects the most frequent option for each specific topic in the validation set based on its frequency of occurrence in that topic. Finally, we adopt micro-average accuracy as the evaluation metric. The prompts we use are in Appendix A.

- 4.2 Results of CMMMU

In this section, we present the main result and detailed ablation studies of different LMMs’ and their performances on the CMMMU benchmark. Results are shown in Tab. 4, 5 and A1. We emphasize our key observations as follows:

- • CMMMU is much more challenging than MMMU, while MMMU is already very challenging. GPT-4V only achieves an accuracy of 41.7% while it achieves an accuracy of 55.7% in an English context. It reveals that existing cross-linguistic generalization is not good enough even for the most advanced closed-source LMMs.
- • The disparity between representative open-source models and GPT-4V is relatively smaller in a Chinese context compared to MMMU. The disparity between Qwen-VL-Chat and GPT-4V on CMMMU is 13.3% while the disparity between BLIP2-FLAN-T5- XXL and GPT-4V on MMMU is 21.9%. Surprisingly, Yi-VL-34B even shortens the disparity between open-source bilingual LMMs and GPT-4V on CMMMU to 7.5%, meaning that open-source bilingual LMMs hold a candle to GPT-4V in a Chinese context, which is a promising progress in the open-source community.
- • The key disparity between open-source LMMs and GPT-4V is the capacity to calculate and reason given complex conditions. Notably, the performance disparity between Open-source LMMs and GPT-4V of Business, Science, and Tech & Eng is larger compared to other disciplines. More questions require complex reasoning in the three disciplines, which reveals that open-source LMMs cannot calculate and reason given complex conditions.
- • The game of pursuing expert Chinese Multimodal Artificial General Intelligence (AGI) has just begun in the open-source community. We point out that all the bilingual LMMs from the open-source community only achieve comparable accuracies with the frequent choice setting referring to MMMU, except recently released Qwen-VL-Chat, Yi-VL-6B, and Yi-VL-34B. These three LMMs trigger the first shot for the race of Chinese Multimodal AGI.

We conduct result decomposition across question difficulties, as shown in Tab. ??. Notably, there is a larger gap between the best open-source LMM, i.e. Yi-VL-34B, and GPT-4V when facing the medium and hard questions. This is further strong evidence of the observation that the key disparity between open-source LMMs and GPT-4V is the capacity to calculate and reason given complex conditions.

- 4https://openai.com/research/gpt-4
- 5https://huggingface.co/01-ai/Yi-6B-Chat
- 6https://mathpix.com/

We conduct result decomposition across question types, as shown in Tab. ??. We notice that Qwen-VL-Plus does not well on True or False questions, indicating that Qwen-VL-Plus may not understand the prompt for answering True or False questions. It might be a free lunch for Qwen-VL-Plus to improve its performance on CMMMU. We further point out that the disparity between Yi-VL Series, Qwen-VL-Plus, and GPT-4V is mainly because of their capacity difference for answering Multiple-choice questions.

- 4.3 Error Analysis

This section carefully analyzes over 150 examples of GPT-4V’s incorrect answers. As shown in the error distribution Fig. 3, several main types of errors are found, such as perceptual errors, lack of knowledge, reasoning errors, rejection to answer, and annotation errors. Analyzing these error types is key to understanding the capabilities and limitations of current LMMs, and can also guide future improvements in designing and training models. The 75 examples of correct responses and 150 examples of incorrect responses are detailed in Appendix B, and the characteristics of each error type are described next.

Perceptual Errors (26%): Perceptual errors are one of the primary reasons for the generation of erroneous examples by GPT-4V. On one hand, when the model fails to comprehend arrows and symbols in the image, misinterprets the sequence from top to bottom and left to right, it introduces deviations in the basic perception of the image, leading to incorrect responses. On the other hand, when the model encounters ambiguity in domain-specific knowledge, hidden meanings, or unclear formulas, it tends to exhibit perceptual errors specific to that domain. In such cases, GPT-4V tends to rely more on answering based on textual information (i.e., the question and options), prioritizing textual information over visual input, causing a bias in understanding multimodal data.

Resoning Errors (26%): Reasoning Error is another major factor contributing to the generation of erroneous examples by GPT-4V. On the one hand, reasoning errors arise when the model receives incorrect information, often stemming from the perceptual errors mentioned earlier, such as in the illustration of Fig. B149, where the model fails to perceive the hidden meaning of symbols, leading to erroneous inferences and outputs. On the other hand, even if the model correctly perceives the meaning conveyed by the image and text, errors in the reasoning process can occur when solving problems that require complex logical and mathematical reasoning. Typically, such errors result from the model’s weaker logical and mathematical reasoning capabilities.

Lack of Knowledge (22%): The lack of expertise is also one of the reasons why GPT-

- 4V generates erroneous responses. The example in Fig. B150 shows GPT-4V producing incorrect answers due to the lack of corresponding physics knowledge. Since CMMMU is for evaluating expert AGI of LMMs, expert-level knowledge in different disciplines and subfields is required. So, injecting expert-level knowledge into LMMs is also one of the directions that can be worked towards AGI.

Rejection (12%): The phenomenon of the model refusing to answer, resulting in incorrect responses, is also a common occurrence. Through analysis, we have identified several reasons for the model’s refusal to answer: (i) The model fails to perceive information from the image, and the textual information in the question is insufficient, causing the model to wait for more information. (ii) Questions involving religious matters or personal real-life information lead the model to refrain from answering, adhering to human values. (iii) When questions involve gender and subjective matters, the model avoids providing accurate responses.

[Figure 33]

Figure 3: GPT-4V error response distribution.

Other Errors: The remaining errors are text comprehension errors (7%), annotation errors (2%), and answer extraction errors (5%). These errors stem from factors like complex instruction comprehension, intricate text logic understanding, response generation limits, data annotation inaccuracies, and issues in extracting answer matches.

### 5 Conclusion

CMMMU represents a significant stride in developing AGI. The CMMMU’s design is tailored to rigorously evaluating the latest LMMs, and testing elementary perceptual skills, intricate logical reasoning, and profound expertise in specific domains. We reveal the disparity between the reasoning capacity of the most advanced bilingual LMMs in a Chinese context and an English context by comparing LMMs’ performance on CMMMU and MMMU. Such an exhaustive assessment is pivotal for delineating the trajectory towards achieving AGI that parallels the proficiency of seasoned professionals in various fields.

### Ethics Policy

In developing the CMMMU benchmark, we strictly adhere to ethical and legal standards, ensuring that our data collection and usage comply fully with pertinent ethical guidelines and legal regulations. Our dedication to promoting fairness, inclusivity, and diversity in our dataset is critical, aiming to reduce biases that might exacerbate societal disparities. We emphasize the importance of protecting privacy and intellectual property rights, highlighting our commitment to responsible and lawful data management. This methodology reflects our steadfast commitment to ethical integrity and legal compliance in the pursuit of advancing research in multimodal understanding and reasoning.

### Limitations

We recognize the inherent limitations of our study. Although the CMMMU benchmark is comprehensive, it does not encompass the entire range of human knowledge and cognitive skills. It is primarily focused on college-level content, which, despite its complexity, covers only a portion of human expertise. Additionally, our evaluation metrics, despite their robustness, might not completely grasp the sophisticated understanding and reasoning abilities of advanced AI systems. These limitations highlight the iterative process of our research, motivating ongoing refinement and expansion of our benchmarks to more accurately reflect the complexities of human cognition and learning. Moreover, the challenges of conducting manual experiments with contracted experts for question-answering tasks have precluded us from offering a comprehensive score for human experiments to date. Nonetheless, we are contemplating the inclusion of such scores in future updates.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. ArXiv preprint, abs/2303.08774, 2023. URL https://arxiv.org/ abs/2303.08774.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: visual question answering. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pp. 2425–2433. IEEE Computer Society, 2015a. doi: 10.1109/ICCV.2015.279. URL https://doi.org/10.1109/ICCV.2015.279.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: visual question answering. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pp. 2425–2433. IEEE Computer Society, 2015b. doi: 10.1109/ICCV.2015.279. URL https://doi.org/10.1109/ICCV.2015.279.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. ArXiv preprint, abs/2309.16609, 2023a. URL https://arxiv.org/abs/2309.16609.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. ArXiv preprint, abs/2308.12966, 2023b. URL https://arxiv.org/abs/2308.12966.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visuallinguistic tasks. ArXiv preprint, abs/2312.14238, 2023. URL https://arxiv.org/abs/ 2312.14238.

Yiming Cui, Ziqing Yang, and Xin Yao. Efficient and effective text encoding for chinese llama and alpaca. ArXiv preprint, abs/2304.08177, 2023. URL https://arxiv.org/abs/ 2304.08177.

DeepSeek-AI, :, Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, A. X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu, Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv: 2401.02954, 2024.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. ArXiv preprint, abs/2306.06070, 2023. URL https://arxiv.org/abs/2306.06070.

Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-to-image generation via transformers. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (eds.), Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 19822–19835, 2021. URL https://proceedings.

###### neurips.cc/paper/2021/hash/a4d92e2cd541fca87e4620aba658316d-Abstract.html.

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. GLM: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 320–335, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.26. URL https://aclanthology.org/2022.acl-long.26.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv preprint, abs/2306.13394, 2023. URL https://arxiv.org/abs/2306.13394.

Roy Ganz, Oren Nuriel, Aviad Aberdam, Yair Kittenplon, Shai Mazor, and Ron Litman. Towards models that can see and read. IEEE International Conference on Computer Vision,

2023. doi: 10.1109/ICCV51070.2023.01985.

Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pp. 3608–3617. IEEE Computer Society, 2018. doi: 10.1109/CVPR.2018.00380. URL http://openaccess.thecvf.com/content_ cvpr_2018/html/Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.html.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 37, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for gui agents, 2023.

Jinyi Hu, Yuan Yao, Chongyi Wang, Shan Wang, Yinxu Pan, Qianyu Chen, Tianyu Yu, Hanghao Wu, Yue Zhao, Haoye Zhang, Xu Han, Yankai Lin, Jiao Xue, Dahai Li, Zhiyuan Liu, and Maosong Sun. Large multilingual models pivot zero-shot multimodal learning across languages. 2023.

Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pp. 6700–6709. Computer Vision Foundation / IEEE, 2019. doi: 10.1109/CVPR.2019.00686. URL http: //openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_ for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv: 2307.16125, 2023.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.

###### LinkSoul-AI. Chinese llava. https://github.com/LinkSoul-AI/Chinese-LLaVA, 2023.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? ArXiv preprint, abs/2307.06281, 2023. URL https://arxiv.org/abs/ 2307.06281.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. ArXiv preprint, abs/2310.02255, 2023. URL https://arxiv.org/abs/2310.02255.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A visual question answering benchmark requiring external knowledge. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pp. 3195–3204. Computer Vision Foundation / IEEE, 2019. doi: 10.1109/CVPR.2019.00331. URL http://openaccess.thecvf.com/content_CVPR_ 2019/html/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_ External_Knowledge_CVPR_2019_paper.html.

Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pp. 2641–2649. IEEE Computer Society, 2015. doi: 10.1109/ICCV.2015.303. URL https://doi.org/10.1109/ICCV.2015.303.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pp. 146–162. Springer, 2022.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. ArXiv preprint, abs/2312.13286, 2023. URL https://arxiv.org/abs/2312.13286.

Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. corr abs/1411.4555 (2014). arXiv preprint arXiv:1411.4555, 2014.

- Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2023.

Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers. ArXiv preprint, abs/2311.17136, 2023. URL https://arxiv.org/abs/2311.17136.

Siwei Wu, Yizhi Li, Kang Zhu, Ge Zhang, Yiming Liang, Kaijing Ma, Chenghao Xiao, Haoran Zhang, Bohao Yang, Wenhu Chen, Wenhao Huang, Noura Al Moubayed, Jie Fu, and Chenghua Lin. Scimmir: Benchmarking scientific multi-modal information retrieval, 2024.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. ArXiv preprint, abs/2311.04257, 2023. URL https://arxiv.org/ abs/2311.04257.

- Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv: 2308.02490, 2023.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. ArXiv preprint, abs/2311.16502, 2023. URL https://arxiv.org/abs/2311.16502.

Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. Evaluating the performance of large language models on gaokao benchmark. ArXiv preprint, abs/2305.12474, 2023. URL https://arxiv.org/abs/2305.12474.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. ArXiv preprint, abs/2304.06364, 2023. URL https://arxiv.org/abs/ 2304.06364.

### A Appendix

In experiments, the prompts we use and their corresponding question types are as follows: Multiple-choice questions: 请回答以下多项选择题，并选出正确选项。这些题目可能包 括单选和多选题型。如果所提供的信息不足以确定一个明确的答案，那么请根据可用的 数据和你的判断来选择最可能正确的选项。 (Please answer the following multiple-choice questions and select the correct options. These questions may include both single-choice and multiple-choice formats. If the provided information is not sufficient to determine a definite answer, please choose the option that is most likely correct based on the available data and your judgment.)7

True/False questions: 请回答以下判断题，并根据题目描述和所给的信息来判断问题中 陈述的对错。如果信息不完整或不足以作出绝对判断，请运用你的逻辑推理和现有信息 来做出最可能的判断。 (Please answer the following true/false questions and determine the correctness of the statements based on the question descriptions and the provided information. If the information is incomplete or insufficient for an absolute judgment, please use your logical reasoning and available information to make the most likely judgment.)

Fill-in-the-blank questions: 请回答以下填空题，并根据题目的要求和所提供的信息来给出 最恰当的答案。如果信息不足以确切回答，那么请依据现有的数据和你的推理能力来填写 最合理的答案。 (Please answer the following fill-in-the-blank questions and provide the most appropriate answer based on the question requirements and the provided information. If the information is insufficient for an exact answer, please fill in the most reasonable response based on the available data and your reasoning abilities.)

Fig. A1 shows the proportion of 6 disciplines and 30 subjects in CMMMU, and Tab. A1 shows the result decomposition across image types.

会计 Accounting (524, 4.36%)

###### 历史 History (331,

[Figure 34]

健康与医学 Health & Medicine

科学 Science

技术与工程

- 2.75%) 人文历史，中国古代史， 地理历史,军事史……

文献学 Literature (98, 0.81%)

文化史，考古学，古代 文学,科学史……

心理学 Psychology (382,

- 3.18%) 实验心理学，信号检测 论，认知心理学……

基础会计，财务分析,债 务资本……

Technology & Engineering

[Figure 35]

艺术 Arts (219, 1.82%)

[Figure 36]

基础医学 Basic Med. (422, 3.51%)

农业 Agriculture (283, 2.35%)

绘画作品，电影艺术，戏 剧艺术，诗歌鉴赏……

药用植物栽培，食品科学，病虫害防 治……

生物实验，细胞形态，细胞 形态，人体结构……

生物 Biology (473, 3.93%)

[Figure 37]

[Figure 38]

经济 Economics (267,

植物生物学，生物化学， 微生物结构……

[Figure 39]

- 2.22%) 宏观经济学，管理经济 学，微观经济……

金融 Finance (386,

- 3.21%) 公司金融，股票管理， 投资管理……

[Figure 40]

建筑学 Architecture (650, 5.41%)

艺术理论 Art Theory (442, 3.67%)

诊断学 Diagnostic (159, 1.32%)

[Figure 41]

土木工程，水利结构，结构力学……

[Figure 42]

[Figure 43]

生物医学，断层解剖学，病 理切片，医学检查……

构图手法，西方美学，中 国美术，心理学艺术……

化学 Chemistry (561,

[Figure 44]

- 4.67%) 分析化学，有机化学，聚 合物化学……

地理 Geography (647,

- 5.38%) 区域地理，气候特征，地 质地貌学原理……

[Figure 45]

计算机科学 Computer Sci. (463, 3.85%)

[Figure 46]

计算机网络，操作系统,计算机图形 学……

临床医学 Clinical Med. (374, 3.11%)

[Figure 47]

[Figure 48]

设计 Design (246, 2.04%)

临床病理学，骨骼结构，临 床神经学，免疫学……

管理 Management (294, 2.44%)

平面设计，建筑设计，影 像处理，绘图效果……

[Figure 49]

电子学 Electronics (387, 3.22%)

[Figure 50]

电路分析，影像电子学，电子技 术……

[Figure 51]

运筹学，企业经营，战 略管理……

[Figure 52]

能源和电力 Energy & Power (422,

###### 社会学 Sociology (323, 2.68%)

[Figure 53]

- 3.51%) 热力系统，核电知识，电气工程……

材料 Materials (504, 4.19%)

结构工程，材料力学，材料表征技 术……

机械工程 Mechanical Eng. (540,

- 4.49%) 机械原理，工程动力学，机械设计理 论……

制药 Pharmacy (462, 3.84%)

音乐 Music (283, 2.35%)

社会学史，社会经济学， 技术与社会……

数学 Mathematics (574, 4.77%)

[Figure 54]

生物制药，药物化学，酸碱 理论,生物药剂学……

营销 Marketing (209, 1.73%)

传统乐器，声乐演唱，乐 理知识，合唱艺术……

[Figure 55]

微积分，经济数学，概率 论与数理统计……

[Figure 56]

市场营销，市场预测， 数据分析……

[Figure 57]

公共卫生 Public Health (619, 5.15%)

人文社会科学 Humanities & Social Sci.

艺术设计 Arts & Design

商业 Business

物理 Physics (468, 3.89%)

流行病学，传染病控制， 健康风险评估……

电磁基础，刚体力学基础， 经典力学……

Figure A1: The proportion of 6 disciplines and 30 subjects in the CMMMU. The multimodal samples in 30 subjects uniformly cover the relevant expert-level domain knowledge.

7The English version is not part of the input to the models.

Chemical Structures

Circuit Diagram

Engineering Diagram

Medical Images

Microsc. Images

Sketches Table Photos Charts

Overall

Models

mPLUG-Owl2 21.8 15.7 29.0 22.5 24.6 14.3 20.2 26.7 21.4 22.2 VisCPM 22.4 14.0 31.2 23.8 20.6 16.2 21.7 28.9 21.2 22.7 Chinese-LLaVA 24.2 15.5 29.8 21.8 24.1 15.0 24.0 26.9 20.9 23.4 Emu2 25.6 15.8 30.3 22.9 27.3 16.6 26.1 26.9 21.2 24.5 CogAgent 23.7 16.1 30.5 24.5 22.4 20.3 22.9 28.0 23.5 23.6 InternVL-Chat-V1.1 29.9 21.3 51.0 33.1 31.1 20.3 23.9 44.0 35.7 34.0 Yi-VL-6B 30.6 21.3 52.6 35.1 34.8 19.9 26.5 42.3 36.0 35.0 Yi-VL-34B 31.2 20.8 55.5 35.0 34.9 26.2 22.3 47.4 36.8 36.5

Qwen-VL-Plus 30.2 24.2 53.1 41.1 35.1 23.0 27.8 43.4 37.6 36.8 GPT-4V 35.7 38.8 56.8 44.9 44.0 37.7 25.1 44.3 35.2 43.7

Table A1: Result decomposition across images type. bold results in LMMs indicate the best results for all models, and the Blue results indicate the best results among the open-source models.

### B Case Study

The appendix is our sample analysis of GPT-4V, including an analysis of 150 error examples and 75 correct examples.

##### List of Case Study Figures

- 1 Art 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 2 Art 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 3 Art 3: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 4 Art 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 5 Art 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 6 Art 6: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 7 Art 7: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 8 Art 8: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- 9 Art Theory 1: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- 10 Design 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- 11 Design 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- 12 Design 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- 13 Design 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- 14 Design 5: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- 15 Design 6: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- 16 Design 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- 17 Design 8: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- 18 Design 9: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
- 19 Music 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- 20 Music 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- 21 Music 3: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- 22 Music 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- 23 Music 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- 24 Music 6: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- 25 Music 7: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- 26 Basic Medicine 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- 27 Basic Medicine 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
- 28 Basic Medicine 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- 29 Basic Medicine 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
- 30 Basic Medicine 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
- 31 Basic Medicine 6: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- 32 Basic Medicine 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- 33 Basic Medicine 8: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
- 34 Basic Medicine 9: Answer Extraction Error . . . . . . . . . . . . . . . . . . . . . . 53
- 35 Basic Medicine 10: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . 54
- 36 Clinical Medicine 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
- 37 Clinical Medicine 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
- 38 Clinical Medicine 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
- 39 Clinical Medicine 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . 58
- 40 Clinical Medicine 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- 41 Clinical Medicine 6: Annotation Error . . . . . . . . . . . . . . . . . . . . . . . . . 60
- 42 Clinical Medicine 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . 61
- 43 Clinical Medicine 8: Textual Understanding . . . . . . . . . . . . . . . . . . . . . . 62
- 44 Clinical Medicine 9: Answer Extraction Error . . . . . . . . . . . . . . . . . . . . . 63
- 45 Clinical Medicine 10: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . 64
- 46 Diagnostics and Lab Medicine 1: Correct Case . . . . . . . . . . . . . . . . . . . . 65
- 47 Diagnostics and Lab Medicine 2: Correct Case . . . . . . . . . . . . . . . . . . . . 66
- 48 Diagnostics and Lab Medicine 3: Correct Case . . . . . . . . . . . . . . . . . . . . 67
- 49 Diagnostics and Lab Medicine 4: Reject to Answer . . . . . . . . . . . . . . . . . . 68
- 50 Diagnostics and Lab Medicine 5: Reject to Answer . . . . . . . . . . . . . . . . . . 69
- 51 Diagnostics and Lab Medicine 6: Reasoning Error . . . . . . . . . . . . . . . . . . 70

- 52 Diagnostics and Lab Medicine 7: Lack of Knowledge . . . . . . . . . . . . . . . . 71
- 53 Diagnostics and Lab Medicine 8: Lack of Knowledge . . . . . . . . . . . . . . . . 72
- 54 Pharmacy 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
- 55 Pharmacy 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
- 56 Pharmacy 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75
- 57 Pharmacy 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76
- 58 Pharmacy 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
- 59 Pharmacy 6: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 78
- 60 Pharmacy 7: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
- 61 Pharmacy 8: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 80
- 62 Pharmacy 9: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
- 63 Public Health 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 82
- 64 Public Health 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 83
- 65 Public Health 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84
- 66 Public Health 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85
- 67 Public Health 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86
- 68 Public Health 6: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
- 69 Public Health 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . 88
- 70 Public Health 8: Textual Understanding . . . . . . . . . . . . . . . . . . . . . . . . 89
- 71 Public Health 9: Answer Extraction Error . . . . . . . . . . . . . . . . . . . . . . . 90
- 72 Public Health 10: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . 91
- 73 Accounting 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
- 74 Accounting 2: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
- 75 Accounting 3: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 94
- 76 Economics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
- 77 Economics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 96
- 78 Economics 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 97
- 79 Economics 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 98
- 80 Economics 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 99
- 81 Economics 6: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
- 82 Economics 7: Textual Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . 101
- 83 Economics 8: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
- 84 Economics 9: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 103
- 85 Finance 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
- 86 Finance 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105
- 87 Finance 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
- 88 Finance 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 107
- 89 Finance 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
- 90 Finance 6: Annotation Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
- 91 Finance 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110
- 92 Finance 8: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 111
- 93 Finance 9: Textual Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . 112
- 94 Finance 10: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 113
- 95 Management 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 114
- 96 Management 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 115
- 97 Management 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 116
- 98 Management 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
- 99 Management 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . 118
- 100Management 6: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . 119
- 101Management 7: Textual Understanding . . . . . . . . . . . . . . . . . . . . . . . . 120
- 102Management 8: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . 121
- 103Management 9: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . 122
- 104Management 10: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . 123
- 105Marketing 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 124
- 106Marketing 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 125
- 107Marketing 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 126
- 108Marketing 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 127
- 109Marketing 5: Textual Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . 128
- 110Marketing 6: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 129

- 111Marketing 7: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 130
- 112Marketing 8: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 131
- 113Marketing 9: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 132
- 114Marketing 10: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . 133
- 115Biology 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 134
- 116Biology 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 135
- 117Biology 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 136
- 118Biology 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 137
- 119Biology 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 138
- 120Biology 6: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 139
- 121Biology 7: Answer Extraction Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 140
- 122Biology 8: Answer Extraction Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 141
- 123Biology 9: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 142
- 124Biology 10: Lack of Knowledge, Reject to Answer . . . . . . . . . . . . . . . . . . 143
- 125Chemistry 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 144
- 126Chemistry 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 145
- 127Chemistry 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 146
- 128Chemistry 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 147
- 129Chemistry 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 148
- 130Chemistry 6: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 149
- 131Chemistry 7: Annotation Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 150
- 132Chemistry 8: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 151
- 133Geography 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 152
- 134Geography 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 153
- 135Geography 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 154
- 136Geography 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 155
- 137Geography 5: Perceptual Error, Lack of Knowledge . . . . . . . . . . . . . . . . . 156
- 138Geography 6: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 157
- 139Geography 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 158
- 140Geography 8: Textual Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . 159
- 141Geography 9: Answer Extraction Error . . . . . . . . . . . . . . . . . . . . . . . . . 160
- 142Geography 10: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . 161
- 143Mathematics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 162
- 144Mathematics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 163
- 145Mathematics 3: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . 164
- 146Physics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 165
- 147Physics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 166
- 148Physics 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 167
- 149Physics 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 168
- 150Physics 5: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 169
- 151Physics 6: Lack of Knowledge, Answer Extraction Error . . . . . . . . . . . . . . . 170
- 152History 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 171
- 153History 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 172
- 154History 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 173
- 155History 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 174
- 156History 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 175
- 157Literature 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 176
- 158Literature 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 177
- 159Literature 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 178
- 160Literature 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 179
- 161Literature 5: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 180
- 162Psychology 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 181
- 163Psychology 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 182
- 164Psychology 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 183
- 165Psychology 4: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 184
- 166Psychology 5: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 185
- 167Psychology 6: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 186
- 168Psychology 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 187
- 169Sociology 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 188

- 170Sociology 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 189
- 171Sociology 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 190
- 172Sociology 4: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 191
- 173Sociology 5: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 192
- 174Sociology 6: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 193
- 175Sociology 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 194
- 176Agriculture 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 195
- 177Agriculture 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 196
- 178Agriculture 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 197
- 179Agriculture 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 198
- 180Agriculture 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 199
- 181Agriculture 6: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . 200
- 182Agriculture 7: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . 201
- 183Architecture 1: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . 202
- 184Architecture 2: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . 203
- 185Architecture 3: Reject to Answer, Perceptual Error . . . . . . . . . . . . . . . . . . 204
- 186Architecture 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 205
- 187Architecture 5: Perceptual Error, Reject to Answer . . . . . . . . . . . . . . . . . . 206
- 188Architecture 6: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 207
- 189Architecture 7: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 208
- 190Architecture 8: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 209
- 191Architecture 9: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 210
- 192Computer Science 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . 211
- 193Computer Science 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . 212
- 194Computer Science 3: Textual Understanding . . . . . . . . . . . . . . . . . . . . . 213
- 195Computer Science 4: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . 214
- 196Electronics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 215
- 197Electronics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 216
- 198Electronics 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 217
- 199Electronics 4: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 218
- 200Electronics 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 219
- 201Electronics 6: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 220
- 202Energy and Power 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . 221
- 203Energy and Power 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . 222
- 204Energy and Power 3: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . 223
- 205Energy and Power 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . 224
- 206Energy and Power 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . 225
- 207Energy and Power 6: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . 226
- 208Energy and Power 7: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . 227
- 209Energy and Power 8: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . 228
- 210Energy and Power 9: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . 229
- 211Materials 1: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 230
- 212Materials 2: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 231
- 213Materials 3: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 232
- 214Materials 4: Perceptual Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 233
- 215Materials 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 234
- 216Materials 6: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 235
- 217Mechanical Engineering 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . 236
- 218Mechanical Engineering 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . 237
- 219Mechanical Engineering 3: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . 238
- 220Mechanical Engineering 4: Perceptual Error, Lack of Knowledge . . . . . . . . . . 239
- 221Mechanical Engineering 5: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . 240
- 222Mechanical Engineering 6: Textual Understanding . . . . . . . . . . . . . . . . . . 241
- 223Mechanical Engineering 7: Answer Extraction Error . . . . . . . . . . . . . . . . . 242
- 224Mechanical Engineering 8: Answer Extraction Error . . . . . . . . . . . . . . . . . 243
- 225Mechanical Engineering 9: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . 244

[Figure 58]

#### 艺术与设计:艺术

Question: 如<图片 1>，北宋山水画《溪山行旅图》运用的空间表现方法是法 Option:

- (A) 平远法
- (B) 阔远法
- (C) 深远法
- (D) 高远法

|[Figure 59]|
|---|

<图片 1>

GPT-4V: 这幅画展现了中国古典山水画的特点，其中包括了山峰、云雾和树木。根据这些元素的排列 和表现手法，可以推断画家可能使用了一种特定的空间表现方法来营造深度和远近感。 选项中的：

- (A) 平远法 - 通常指的是通过平行排列的山峰、水面等元素来表现深远的空间感。
- (B) 阔远法 - 通过宽阔的水面或平原来表现空间的深远。
- (C) 深远法 - 通过层层叠加的山峰、云雾等元素来表现空间的深度和远近关系。
- (D) 高远法 - 通过高耸的山峰和视角的抬升来表现空间的深远。 从图片中可以看出，画中的山峰高耸，云雾环绕，给人一种高远和深邃的感觉。 因此，最可能正确的选项是： 正确答案：(D) 高远法

Ground Truth: (D) 高远法

Figure B1: A sample correct case of Art and Design (subcategory: Art). Back to List of figures

[Figure 60]

#### 艺术与设计:艺术

Question: 如<图片 1>,该作品(局部)的名称是( )。 Option: (A) 《踏歌图》 (B) 《溪山行旅图》 (C) 《虢国夫人游春图》 (D) 《朝元仙仗图》

|[Figure 61]|
|---|

<图片 1>

GPT-4V: 正确答案是：(C) 《虢国夫人游春图》 这幅画是中国北宋时期画家张择端创作的《虢国夫人游春图》，描绘了一位高级官员的夫人出游的场 景。

Ground Truth: (C) 《虢国夫人游春图》

Figure B2: A sample correct case of Art and Design (subcategory: Art). Back to List of figures

[Figure 62]

#### 艺术与设计:艺术

Question: <图片 1>该线描类型属于（ ）。 Option: (A) 铁线描 (B) 钉头鼠尾描 (C) 高古游丝描 (D) 柳叶描

|[Figure 63]|
|---|

<图片 1>

GPT-4V: 这张图片展示的是一种线描艺术，但由于我无法看到实际的线条宽窄变化和线条的具体风格， 因此无法准确判断它属于哪一种线描类型。不过，根据线条的流畅性和简洁性，我可以尝试猜测：选 项 (A) 铁线描 通常指的是线条坚硬如铁，强调力度和结构感。

- 选项 (B) 钉头鼠尾描 描述的是线条开始时粗而末端细，类似钉头和鼠尾。
- 选项 (C) 高古游丝描 是指非常细腻和复杂的线条，如游丝般轻柔。
- 选项 (D) 柳叶描 则是指线条细长如柳叶，柔美流畅。 根据这张图片，线条看起来比较简洁流畅，但没有足够的信息来确定它是否符合上述任何一种特定的 线描风格。如果必须选择，我可能会倾向于选择 (D) 柳叶描，因为线条看起来较为流畅，有点像柳 叶的形态。然而，这只是一个基于可见信息的猜测，而不是一个确定的答案。 正确答案：(D) 柳叶描（猜测）

Error Category: Perceptual Error Error Reason: GPT-4 attempts to guess possible answers by characterizing the type of line drawing provided by each option. However, this method is very limited in its accuracy and validity in the absence of direct observation and analysis of the picture. Without access to the specifics of the pictures, the GPT-4 is unable to effectively analyze or provide accurate answers. In addition, GPT-4 made it clear in its response that it could not “see” the actual lines.

Ground Truth: (B) 钉头鼠尾描

Figure B3: A sample error case of Art and Design (subcategory: Art). Back to List of figures

[Figure 64]

#### 艺术与设计:艺术

Question: <图片 1>下图中的书法用笔是（）运笔。

Option: (A) 正锋 (B) 中锋 (C) 侧锋 (D) 悬锋

|[Figure 65]|
|---|

<图片 1>

GPT-4V: 正确答案是：(C) 侧锋 在书法中，侧锋运笔是指笔尖偏离中心线，用笔的一侧进行书写。从提供的图片中可以看出，笔尖并 没有垂直纸面，而是有一定的倾斜角度，这表明是在使用侧锋运笔的方式。

Error Category: Perceptual Error Error Reason: GPT-4V incorrectly interpreted the angle of the calligraphy brush when processing the image. In this case, the correct answer needs to be based on the angle of the tip of the calligraphy brush relative to the surface of the paper. In the correct case, the tip of the brush should be perpendicular to the surface of the paper, i.e., a "center-front" stroke, but the model incorrectly recognizes that the tip of the brush is not perpendicular, and thus incorrectly chooses a "side-front" stroke as the answer. GPT-4V在处理图像时错误地解读了书法用笔的角度。在这个问题中，正确的答 案需要依据书法笔尖与纸面的相对角度来确定。正确的情况应该是笔尖垂直于纸面，也就是“中锋” 运笔，但模型错误地识别为笔尖并非垂直，从而错误地选择了“侧锋”运笔作为答案。

- Ground Truth: (B) 中锋

Figure B4: A sample error case of Art and Design (subcategory: Art). Back to List of figures

[Figure 66]

#### 艺术与设计:艺术

Question: 如<图片 1>，该作品的作者 Option: (A) 德拉克洛瓦 (B) 席里科 (C) 米勒 (D) 罗丹

|[Figure 67]|
|---|

<图片 1>

GPT-4V: 这幅画是由法国浪漫主义画家德拉克洛瓦创作的，名为《马赛的女性》。 因此，正确答案是：(A) 德拉克洛瓦

Error Category: Perceptual Error Error Reason: Since GPT-4V cannot directly process and recognize image content, it cannot accurately identify the artwork and its author in the image. As a result, the model is unable to validate its assumptions or assertions about the artwork, leading to the misattribution of the author of The Raft of Medusa.

- Ground Truth: (B) 席里科

Figure B5: A sample error case of Art and Design (subcategory: Art). Back to List of figures

[Figure 68]

#### 艺术与设计:艺术

Question: 下图中的绘画作品是（）<图片 1> Option: (A) 《洛神赋图》 (B) 《女史箴图》 (C) 《步辇图》 (D) 《历代帝王图》

|[Figure 69]|
|---|

<图片 1>

GPT-4V: 在其正确答案是：(B) 《女史箴图》 这幅图是张萱的《女史箴图》，是中国唐代著名的绘画作品。

Error Category: Perceptual Error Error Reason: GPT-4V needs to recognize the name of a painting based on the picture provided. The answer given by the model is "(B) The Picture of the Lady of History". However, the correct answer should be "(A) The Picture of Luoshen Fu". The model's error may be due to a misunderstanding of the content of the image or a lack of knowledge in ancient Chinese paintings. Both "The Picture of a Woman in the Palace of History" and "Luoshen Fu" are famous paintings in ancient China, and there may be similarities in style and subject of these two paintings, which may have led to the modeling confusion. Although both "The Picture of a Woman in the Palace of History" and "Luoshen Fu" belong to ancient Chinese figure paintings, they differ in terms of figure modeling, costumes, and backgrounds.

- Ground Truth: (A) 《洛神赋图》

Figure B6: A sample error case of Art and Design (subcategory: Art). Back to List of figures

[Figure 70]

#### 艺术与设计:艺术

Question: <图片 1>黄筌《写生珍禽图》的创作时代是（ ）。

Option: (A) 唐代 (B) 北宋 (C) 南宋 (D) 五代

|[Figure 71]|
|---|

<图片 1>

GPT-4V: 抱歉，我无法直接识别图片中的艺术作品及其创作时代。但根据您提供的信息，黄筌是南宋 时期的画家，因此《写生珍禽图》的创作时代应该是南宋。 所以正确答案是：(C) 南宋

Error Category: Lack of Knowledge Error Reason: GPT-4V is unable to recognize the content of the image directly, and is unable to determine what kind of artwork it is. Moreover, in the answer, the model mentions that "Huang Chyuan was a painter in the Southern Song Dynasty", which indicates that it relies on the historical information in its training data. However, the model's training data may not contain accurate information about the specific era in which Huang Chyuan's Rare Bird Drawings were created, or this information may not be correctly learned and recalled by the model. This causes the model to exhibit knowledge deficiencies in its responses.

Ground Truth: (D) 五代

Figure B7: A sample error case of Art and Design (subcategory: Art). Back to List of figures

[Figure 72]

#### 艺术与设计:艺术

Question: 如<图片 1>，黄公望在作品《富春山居图》(局部)中主要使用的是哪种装法? Option: (A) 折带皴 (B) 斧劈 (C) 披麻皴 (D) 荷叶

|[Figure 73]|
|---|

<图片 1>

GPT-4V: 黄公望在其作品《富春山居图》中使用了多种绘画技巧，但根据这幅画的局部图像，我们可 以看到的是一种细腻而有层次感的线条技法，这种技法在中国画中被称为皴法。在提供的选项中，最 符合这种线条特征的是(A) 折带皴。

- 因此，正确答案是：(A) 折带皴

Error Category: Lack of Knowledge Error Reason: GPT-4V did not accurately identify or understand these specialized painting terms. The model's choice of (A) Folded Band Chafing suggests that it has a limited understanding of Chinese painting terminology and techniques and is unable to accurately differentiate between different chafing techniques. Additionally, the model's responses appear to be based on general descriptions of painting styles rather than a precise understanding of specific terms.

- Ground Truth: (C) 披麻皴

Figure B8: A sample error case of Art and Design (subcategory: Art). Back to List of figures

[Figure 74]

#### 艺术与设计:艺术理论

Question: 如图<图片 1>陶器产生与什么时代（） Option: (A) 原始时期 (B) 新石器时代 (C) 旧石器时代 (D) 作为旧石器时代和新石器时代的划分标识

|[Figure 75]|
|---|

<图片 1>

GPT-4V: 陶器的产生通常与新石器时代相关联。因此，正确答案是：(B) 新石器时代

Error Category: Lack of Knowledge Error Reason: Although it is true that pottery was widely used starting in the Neolithic Age, the GPT-4V model's recognition of this fact is not incorrect in itself. However, the model failed to accurately grasp the focus of the question, which is that the emergence of pottery is an important distinguishing marker between the Paleolithic and Neolithic periods. This suggests that the model lacks comprehensive knowledge in considering the characteristics of historical periods and archaeological divisions.

- Ground Truth: (D) 作为旧石器时代和新石器时代的划分标识

Figure B9: A sample error case of Art and Design (subcategory: Art Theory). Back to List of figures

[Figure 76]

#### 艺术与设计:设计

Question: 如图所示的地面图案形成了近大远小的效果，这是通过（ ）变形操作生成的。<图片 1> Option:

(A) 【编辑】/【变换】/【放缩】 (B) 【编辑】/【变换】/【斜切】 (C) 【编辑】/【变换】/【透视】 (D) 【编辑】/【变换】/【变形】

|[Figure 77]|
|---|

<图片 1>

GPT-4V: 正确答案是：(C) 【编辑】/【变换】/【透视】 图中的地面图案显示了一种透视效果，其中图案的大小随着距离的增加而减小，模拟了在三维空间中 的视觉现象。这种效果通常是通过透视变形操作来实现的。

- Ground Truth: (C) 【编辑】/【变换】/【透视】

Figure B10: A sample correct case of Art and Design (subcategory: Design). Back to List of figures

[Figure 78]

#### 艺术与设计:设计

Question: 将图B完整地拷贝到图A中形成“图层1”，在【图层】面板中将“图层1”的【混合模式】 更改为（ ）可以得到图C所示的效果，即白色背景自动透明。 <图片 1>

Option:

- (A) 【滤色】
- (B) 【亮光】
- (C) 【叠加】
- (D) 【正片叠底】

|[Figure 79]|
|---|

<图片 1>

GPT-4V: 根据提供的图片和描述，要实现图C所示的效果，即将图B的白色背景变为透明，同时保留图 B中的红色线条，并叠加到图A上，最合适的混合模式是【正片叠底】。 正确答案是：(D) 【正片叠底】

- Ground Truth: (D) 【正片叠底】

Figure B11: A sample correct case of Art and Design (subcategory: Design). Back to List of figures

[Figure 80]

#### 艺术与设计:设计

Question: 如果要绘制出<图片 1>中所示的逐渐淡出的笔触,需要在【画笔】控制面板中设置的一项 重要参数是( ）。

Option: (A) 【抖动】 (B) 【湿边】 (C) 【颜色动态】 (D) 【渐隐】

|[Figure 81]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是一系列笔触，它们似乎从清晰逐渐变得模糊和淡出。在数字绘画软件中， 要实现这种效果，通常需要调整的是画笔的渐隐效果，这样笔触就会随着时间或笔触的长度而逐渐减 少其不透明度，从而实现淡出的效果。 正确答案是：(D) 【渐隐】

Ground Truth: (D) 【渐隐】

Figure B12: A sample correct case of Art and Design (subcategory: Design). Back to List of figures

[Figure 82]

#### 艺术与设计:设计

Question: 判断下面陈述对错：<图片 1>的日历部分是由于使用了“仿制图章工具”而变花。 Option:

- (A) 正确
- (B) 错误

|[Figure 83]|
|---|

<图片 1>

GPT-4V: 对不起，我无法提供关于图片内容的帮助。

Error Category: Reject to Answer Error Reason: In this response, the GPT-4V rejects to answet the question. This choice may be due to a misunderstanding of its image processing capabilities. In fact, GPT-4V has some image analysis capabilities, and is able to perform basic visual content analysis on supplied images, such as identifying objects, colors, layouts, etc. In this question, the user asked about the image processing capabilities of GPT-4V. In this question, the user is asking about the impact of image editing techniques ("faux stamp tools"), which is something that GPT-4V could theoretically answer by analyzing the visual characteristics of the image. In addition, the model may not have properly understood the context of the question and mistakenly assumed that it was a request involving the direct modification or generation of image content, a feature that GPT-4V does not currently have. In reality, however, the question is only asking to analyze the image content, not to modify the image.

- Ground Truth: (B) 错误

Figure B13: A sample error case of Art and Design (subcategory: Design). Back to List of figures

[Figure 84]

#### 艺术与设计:设计

Question: 下面所写出下图<图片 1>所示对象的名称、朝代、地点_ Option: (A) 临潼姜寨原始社会氏族村落遗址平面 (B) 陕西长安县客省庄原始社会半地穴住宅遗址平面 (C) 偃师二里头商代宫殿遗址平面 (D) 岐山凤雏西周建筑遗址平面

|[Figure 85]|
|---|

<图片 1>

GPT-4V: 很抱歉，我无法直接识别图片中的遗址平面图并将其与特定的历史遗址相匹配。这需要专业 的考古或历史知识，以及对特定遗址的详细了解。如果没有更多的上下文信息或数据，我无法确定正 确答案。如果您能提供更多的背景信息或细节，我会尽力帮助您。

Error Category: Reject to Answer Error Reason: GPT-4V chooses to refuse to answer directly rather than attempting to reason logically or provide historical knowledge based on the options given in the question, reflecting a rejecting attitude. In this case, even though it is not possible to analyze the pictures directly, the model can still use its extensive knowledge base to provide contextual information about the options, thus assisting the user in reasoning about the possible correct answers. For example, describing the historical sites or eras mentioned in each option, or providing some of the common features and historical context of these sites.

Ground Truth: (D) 岐山凤雏西周建筑遗址平面

Figure B14: A sample error case of Art and Design (subcategory: Design). Back to List of figures

[Figure 86]

#### 艺术与设计:设计

Question: 对于下面所示的效果是通过什么来实现的？<图片 1>（） Option: (A) 蒙板功能的使用 (B) 菜帧并帧动画的使用 (C) 补间动画的使用 (D) 以上A B C都可以来实现

|[Figure 87]|
|---|

<图片 1>

GPT-4V: 抱歉，由于技术限制，我无法查看图片。但是，根据你提供的选项，我可以提供一些关于这 些选项的信息，这可能会帮助你确定正确答案。

- (A) 蒙板功能的使用 - 蒙板功能可以用来在动画中创建特定形状的可视区域，使得只有在蒙板形状 内的部分可见。
- (B) 帧并帧动画的使用 - 帧并帧动画是一种传统动画技术，其中每一帧都是单独绘制的，以创建动 画效果。
- (C) 补间动画的使用 - 补间动画是一种动画技术，其中动画的开始和结束点是定义的，而中间的帧 是由软件自动生成的。
- (D) 以上A B C都可以来实现 - 这个选项表明所有上述技术都可以用来实现所描述的效果。 由于我无法看到图片，我无法确定哪种技术被用来创建特定的效果。但是，如果效果涉及形状的可视 区域变化，可能使用了蒙板功能；如果效果看起来非常手工和详细，可能使用了帧并帧动画；如果效 果涉及平滑的过渡，可能使用了补间动画。如果效果看起来可能由多种技术实现，那么选项(D)可能 是正确的。

Error Category: Perceptual Error Error Reason: GPT-4V explicitly shows that it is unable to view images and therefore cannot directly analyze their content in order to answer questions. This demonstrates the model's limitations in processing image-related information, even if it possesses some degree of image processing capability. The model attempts to compensate for the lack of perception of image content by providing a general explanation of each option, but this approach cannot accurately answer questions specific to image content. While the model's explanations are technically correct, they are not directly applicable to a specific image, as this requires direct analysis of the image content.

- Ground Truth: (A) 蒙板功能的使用

Figure B15: A sample error case of Art and Design (subcategory: Design). Back to List of figures

[Figure 88]

#### 艺术与设计:设计

Question: 在【渐变编辑器】对话框中有如图所示的颜色条，位于色条上部和下部的小方块分别用来 调节渐变色的（ ）属性。<图片 1>

Option: (A) 【颜色】和【位置】 (B) 【颜色】和【不透明度】 (C) 【不透明度】和【颜色】 (D) 【位置】和【平滑度】

|[Figure 89]|
|---|

<图片 1>

GPT-4V: 在渐变编辑器中，位于颜色条上部的小方块通常用来调节渐变色的【颜色】属性，而位于颜 色条下部的小方块则用来调节【不透明度】。

- 因此，正确答案是：(B) 【颜色】和【不透明度】

Error Category: Reasoning Error Error Reason: The error made by GPT-4V may stem from a misunderstanding of the function of the small square of the color bar in the Gradient Editor. In graphic editing software, the design and functionality of the gradient editor may vary from software to software. The model may infer answers based on general knowledge from its training data or the functionality of a particular software without accurately grasping the details of a particular situation in the question.

- Ground Truth: (C) 【不透明度】和【颜色】

Figure B16: A sample error case of Art and Design (subcategory: Design). Back to List of figures

[Figure 90]

#### 艺术与设计:设计

Question: 请选择图中截面附合构造要求的混凝土基础。（） Option:

(A) <图片 1> (B) <图片 2> (C) <图片 3> (D) <图片 4>

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

<图片 1>

<图片 2> <图片 3> <图片 4>

GPT-4V: 根据提供的图片，我们需要选择一个符合构造要求的混凝土基础截面。在土木工程中，混凝 土基础的设计需要考虑承载力、稳定性和结构的耐久性。基础通常需要有足够的底面积来分散上部结 构的荷载，并且有适当的形状以防止不均匀沉降。从提供的四个选项中：

- - 图片 1 显示了一个带有扩大底部的基础，这有助于分散荷载。
- - 图片 2 显示了一个类似的基础，但底部没有扩大。
- - 图片 3 显示了一个基础，底部扩大得非常平坦，可能不利于荷载的传递。
- - 图片 4 显示了一个基础，底部扩大且形状介于图片 1 和图片 3 之间。 通常，基础的底部扩大是为了提供更大的承载面积，减少基础下的土压力，从而提高结构的稳定性。 因此，我们通常会选择那些底部有适当扩大的基础设计。 在这四个选项中，图片 1 的基础底部扩大看起来最为合理，因为它提供了一个均匀的扩展，有助于 稳定和分散荷载。 因此，正确答案是：(A) <图片 1>

Error Category: Reasoning Error Error Reason: GPT-4V incorrectly applied its understanding of civil engineering principles to image analysis when analyzing image content. The model attempted to infer the characteristics of each image based on its general knowledge of concrete foundation design, but did not properly apply this knowledge to the specific image analysis. This may be due to the model's lack of sufficient expertise to accurately identify and evaluate the construction requirements of concrete foundations.

- Ground Truth: (B) <图片 2>

Figure B17: A sample error case of Art and Design (subcategory: Design). Back to List of figures

[Figure 95]

#### 艺术与设计:设计

Question: 如<图片 1>所示,图中文字部分的文字透视效果是通过“变形文本”中的哪种样式产生的？

Option: (A) 贝壳 (B) 旗帜 (C) 下弧 (D) 扇形

|[Figure 96]|
|---|

<图片 1>

GPT-4V: 图中文字部分的文字透视效果是通过“变形文本”中的“下弧”样式产生的。 正确答案：(C) 下弧

Error Category: Lack of Knowledge Error Reason: The source of the error is the lack of expertise in GPT-4V, i.e., the lack of proper recognition and understanding of the specific text style in the image. Although the text in the image shows an arc-like visual effect, the correct answer should be "(D) Fan". This suggests a knowledge limitation in the model's ability to understand and differentiate between different text morphing styles. The model relies on the information in its training data when answering such questions. If there is insufficient information about specific text styles (e.g., fan, downward arc, etc.) in the training data, the model may not be able to accurately recognize and understand the characteristics of these styles and application scenarios.

- Ground Truth: (D) 扇形

Figure B18: A sample error case of Art and Design (subcategory: Design). Back to List of figures

[Figure 97]

#### 艺术与设计:音乐

Question: <图片 1>是民族管弦乐队的位置图，①所在的区域通常是哪种乐器？（） Option:

(A) 拉弦乐器 (B) 打击乐器 (C) 弹拨乐器 (D) 吹管乐器

|[Figure 98]|
|---|

<图片 1>

GPT-4V: 正确答案是：(A) 拉弦乐器 在传统的管弦乐队布局中，①所在的区域通常是为第一小提琴和第二小提琴预留的位置，它们属于拉 弦乐器。这个区域通常位于指挥的左手边，靠近观众。

Ground Truth: (A) 拉弦乐器

Figure B19: A sample correct case of Art and Design (subcategory: Music). Back to List of figures

[Figure 99]

#### 艺术与设计:音乐

Question: 关于下图<图片 1>京剧《沙家浜》中阿庆嫂的角色属于什么角色（）

Option: (A) 旦行 (B) 生行 (C) 丑行 (D) 净行

|[Figure 100]|
|---|

<图片 1>

GPT-4V: 阿庆嫂是京剧《沙家浜》中的一个角色，根据她的服装和角色性别，她属于旦行。旦行是京 剧中专指扮演女性角色的行当。 正确答案是：(A) 旦行

Ground Truth: (A) 旦行

Figure B20: A sample correct case of Art and Design (subcategory: Music). Back to List of figures

[Figure 101]

#### 艺术与设计:音乐

Question: 下列哪些调式音阶与音列<图片 1>的结构相似() Option: (A) D雅乐较大调 (B) D和声大调式 (C) D清乐调式 (D) D自然小调式

|[Figure 102]|
|---|

<图片 1>

GPT-4V: 很抱歉，由于图片中的音符和谱号无法被我直接解读，我无法提供一个确切的答案。但是， 我可以根据您提供的选项来解释每个调式的特点，这样您可以根据图片中的音列来判断哪个调式与之 相似。

- (A) D雅乐较大调 - 这个选项可能是指D大调的一个变种，但“雅乐较大调”不是一个标准的音乐术 语。
- (B) D和声大调式 - D和声大调式以D为主音，其特点是在第七音上升半音形成升高的七度音，以形成 一个完全的主和弦。
- (C) D清乐调式 - 这个选项可能是指D旋律小调式或D自然小调式的一个变种，但“清乐调式”不是一 个标准的音乐术语。
- (D) D自然小调式 - D自然小调式以D为主音，其音阶结构为全半全全半全全（全表示全音，半表示半 音）。 如果您能够根据音符的升降号和它们的间隔来判断音阶的结构，您就可以选择与图片中音列结构最相 似的调式。 预测答案为：ABCD

Error Category: Reject to Answer Error Reason: GPT-4V failed to accurately recognize and analyze the provided music sheet image, thus it rejected to answer the question. However, the model gave the appropriate tonal analyses for the four options given in texts, indicating that the model actually possesses a certain amount of knowledge in the fundamentals of music theory, but lacks the appropriate perceptual and interpretive analyzing ability for the current image.

- Ground Truth: (A) D雅乐较大调, (C) D清乐调式, (D) D自然小调式1V

Figure B21: A sample error case of Art and Design (subcategory: Music). Back to List of figures

[Figure 103]

#### 艺术与设计:音乐

Question: 关于下图<图片 1>戏曲《刘大哥讲话理太偏》属于____。

Option: (A) 越剧 (B) 京剧 (C) 豫剧 (D) 黄梅戏

|[Figure 104]|
|---|

<图片 1>

GPT-4V: 这张图片展示了一群穿着华丽戏服的演员，他们的服饰和化妆风格可以帮助我们判断他们可 能正在表演的戏曲类型。然而，由于中国的戏曲种类繁多，且不同戏曲之间的服饰和化妆有时候可能 相似，因此仅凭图片来确定确切的戏曲类型可能会有一定的挑战性。在这张图片中，演员们的服饰色 彩鲜艳，头饰也非常精致，这些特点在多种地方戏中都有可能出现。没有更多的文化或地域线索，很 难确定是哪一种戏曲。因此，我无法仅凭这张图片提供一个确切的答案。如果你有更多关于这部戏曲 的信息，比如它的起源、流行地区或者特定的表演风格，那么我可能能够帮助你更准确地判断。在没 有额外信息的情况下，我建议选择最可能正确的选项。 正确答案：无法确定。

Error Category: Perceptual Error Error Reason: GPT-4V was unable to correctly recognize and interpret the information in the images to answer the questions. This suggests a perceptual limitation in the model's ability to process image input. Although the model was able to recognize some basic elements in the image (e.g., actors' costumes and make-up styles), it failed to accurately associate these elements with specific types of theatre. This may be due to the fact that the model lacks a sufficiently in-depth understanding of the visual features and stylistic details of various types of Chinese theater.

- Ground Truth: (C) 豫剧

Figure B22: A sample error case of Art and Design (subcategory: Music). Back to List of figures

[Figure 105]

#### 艺术与设计:音乐

Question: 关于下图<图片 1>这段主题旋律出自于那部歌曲（） Option: (A) 《波莱罗》 (B) 《第五（命运）交响曲》 (C) 《卡门序曲》 (D) 《摇篮曲》

|[Figure 106]|
|---|

<图片 1>

GPT-4V: 抱歉，我无法直接识别乐谱中的旋律对应的歌曲。但是，如果你能够演奏或哼唱这段旋律， 或者提供更多的信息，我可能能够帮助你确定这段旋律出自哪部歌曲。

Error Category: Perceptual Error Error Reason: GPT-4V model has the ability to recognize images, but in this particular case it did not correctly utilize this ability to recognize the sheet music in the image and thus the corresponding song. The model's response, "I cannot directly recognize the song that corresponds to the melody in the sheet music," suggests that the model is not aware of its ability to process image input, especially in interpreting sheet music. This may be due to the model's strategies or limitations in processing image input that are not adapted to this particular query.

- Ground Truth: (B) 《第五（命运）交响曲》

Figure B23: A sample error case of Art and Design (subcategory: Music). Back to List of figures

[Figure 107]

#### 艺术与设计:音乐

Question: 在图中键盘<图片 1>中，从右面开始的第四个黑键的音名是什么 Option: (A) 降D (B) E (C) 降E (D) 降F

|[Figure 108]|
|---|

<图片 1>

GPT-4V: 在钢琴键盘上，黑键位于两个白键之间，可以表示为这两个白键的升或降音。从右侧开始数， 第四个黑键位于E和F之间。因此，这个黑键可以被称为E的升音（升E）或F的降音（降F）。在音乐理 论中，这两个名称指的是同一个音。 正确答案是：(C) 降E (D) 降F

Error Category: Lack of Knowledge Error Reason: GPT-4V demonstrates a lack of knowledge in certain music theories. On a piano keyboard, each black key is indeed an ascending or descending tone of two white keys, but they have specific tone names. In this case, the model failed to accurately provide the correct tone name for the black key, but instead gave general descriptions, which demonstrates a lack of knowledge in music theory. The model should have enough knowledge to accurately identify and name each key on the piano keyboard.

Ground Truth: (A) 降D

Figure B24: A sample error case of Art and Design (subcategory: Music). Back to List of figures

[Figure 109]

#### 艺术与设计:音乐

Question: 关于下图<图片 1>戏歌《唱脸谱》的旋律汲取了什么元素（） Option: (A) 京剧元素 (B) 豫剧元素 (C) 黄梅戏元素 (D) 昆曲元素

|[Figure 110]|
|---|

<图片 1>

GPT-4V: 这张图片展示了五个戴着中国传统戏剧脸谱的人物。脸谱通常与中国不同地区的戏剧形式相 关联，如京剧、豫剧、黄梅戏和昆曲等。然而，仅凭这张图片，我无法确定《唱脸谱》这首歌的旋律 具体汲取了哪种戏剧元素，因为图片中没有提供音乐或旋律的信息。 如果没有其他信息，我无法准确回答这个问题。因此，基于图片提供的信息，我无法选择一个正确的 答案。如果有关于这首歌的更多背景信息或者旋律的描述，我可能会做出更准确的判断。

Error Category: Lack of Knowledge Error Reason: In this response, the image content does provide enough information to determine the source of the melodic elements of the song Singing Faces, and the model's failure to recognize and utilize this information demonstrates its lack of domain-specific knowledge.

Ground Truth: (A) 京剧元素

Figure B25: A sample error case of Art and Design (subcategory: Music). Back to List of figures

[Figure 111]

#### 健康与医学:基础医学

Question: 图<图片 1>中的组织细胞多层，由表层到基底层细胞形状依次为扁平形、多边形、矮柱状， 是（） Option: (A) 复层扁平上皮 (B) 假复层纤毛柱状上皮 (C) 单层柱状上皮 (D) 其余选项均不对

|[Figure 112]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是一种组织切片，但由于我无法进行实际的显微镜检查，我只能根据您提供 的描述和图片的视觉信息来推测。根据您的描述，这种组织细胞是多层的，表层细胞为扁平形，中间 层细胞为多边形，而基底层细胞为矮柱状。这种细胞排列和形状的描述符合复层扁平上皮的特征。 因此，正确答案是： (A) 复层扁平上皮

Ground Truth: (A)复层扁平上皮

- Figure B26: A sample correct case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 113]

#### 健康与医学:基础医学

Question: <图片 1>中细胞扁平，排列紧密的肌组织是（）

|[Figure 114]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是一种肌组织的显微镜图像。根据细胞扁平、排列紧密的描述，这很可能是 平滑肌组织。平滑肌细胞在显微镜下通常呈现为长条形，两端尖细，中间较宽，且核呈椭圆形或豆形， 排列紧密但没有横纹。因此，<图片 1>中细胞扁平，排列紧密的肌组织是平滑肌。

Ground Truth: 平滑肌

- Figure B27: A sample correct case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 115]

#### 健康与医学:基础医学

Question: <图片 1>中是血涂片，5是（）

|[Figure 116]|
|---|

<图片 1>

GPT-4V: 图片中标记为5的细胞是一种白细胞，根据其形态特征，它可能是一种淋巴细胞。在血涂片 中，淋巴细胞的特点是较大的圆形或略呈椭圆形的核，以及较少的细胞质。因此，正确答案是：5是 淋巴细胞。

Ground Truth: 淋巴细胞

- Figure B28: A sample correct case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 117]

#### 健康与医学:基础医学

Question: <图片 1>中是透明软骨组织，框中圈出的是（） Option: (A) 肾上腺皮质 (B) 淋巴小结生发中心 (C) 甲状腺滤泡 (D) 同源细胞群

|[Figure 118]|
|---|

<图片 1>

GPT-4V: 很抱歉，我无法提供有关此图像的医学或生物学分析。如果您有其他类型的问题或需要其他 帮助，请告诉我！

Error Category: Reject to Answer Error Reason: Due to ethical considerations and other factors, GPT-4V refuses to provide medical or biological analyses.

Ground Truth: (D) 同源细胞群

- Figure B29: A sample error case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 119]

#### 健康与医学:基础医学

Question: <图片 1>图示箭头所指的细胞名称（） Option: (A) 血小板 (B) 血小板聚集 (C) 巨大血小板 (D) 吞噬细胞

|[Figure 120]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是染色后的血液涂片，箭头指向的细胞由于图片分辨率和缺乏特定的细胞 学特征，难以准确判断。但是，我们可以根据一些基本的细胞学知识来尝试推断： (A) 血小板 - 通 常比较小，不含核，这些细胞看起来较大且可能含有核，因此不太可能是血小板。 (B) 血小板聚集 - 血小板聚集通常是指多个血小板粘在一起，这些细胞看起来是单个的，不是聚集在一起的。 (C) 巨大血小板 - 巨大血小板是异常大的血小板，但是这些细胞的大小和形态与典型的巨大血小板不太 相符。 (D) 吞噬细胞 - 吞噬细胞，如巨噬细胞，通常较大，含有较大的核和丰富的细胞质。箭头指 向的细胞可能是吞噬细胞，但没有更多的信息很难确定。 由于缺乏足够的信息，无法确定一个明确 的答案。但是，根据细胞的大小和形态，最可能的选项是(D) 吞噬细胞。

Error Category: Perceptual Error Error Reason: GPT-4V correctly understood the question, but it made an error in image perception. The arrow points to a group of cells clustered together, whereas GPT-4V identified it as a single cell, leading to an error.

Ground Truth: (B)血小板聚集

- Figure B30: A sample error case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 121]

#### 健康与医学:基础医学

Question: <图片 1>图示箭头所指的细胞名称（） Option: (A) 血小板 (B) 血小板聚集 (C) 巨大血小板 (D) 血小板卫星现象

|[Figure 122]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是一个血液涂片，箭头指向的细胞可能是血小板聚集，因为它看起来是由多 个较小的细胞组成的一团，这与血小板聚集的特征相符。血小板（选项A）通常是单个的小细胞，而 巨大血小板（选项C）是单个但异常大的血小板。血小板卫星现象（选项D）通常是指血小板围绕着白 细胞。根据这张图片，最可能正确的选项是： (B) 血小板聚集

Error Category: Perceptual Error Error Reason: GPT-4V correctly understood the question, but it made an error in image perception. The arrow should be pointing to a large cell (compared to normal platelets), whereas GPT-4V identified it as a group of smaller cells clustered together, leading to an error.

Ground Truth: (C)巨大血小板

- Figure B31: A sample error case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 123]

#### 健康与医学:基础医学

Question: <图片 1>为离心沉淀后的血液，这三层从上至下为（） Option:

(A) 血清,白细胞和血小板,红细胞 (B) 血清,红细胞,白细胞和血小板 (C) 血浆,白细胞和血小板,红细胞 (D) 血浆,红细胞,白细胞和血小板

|[Figure 124]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是离心沉淀后的血液样本。从上至下，血液通常分为三层：最上面是血浆或 血清，中间是白细胞和血小板形成的白色或淡黄色层，称为白细胞层或白膜，最下面是红细胞。 根 据这个描述，正确答案是： (A) 血清,白细胞和血小板,红细胞

Error Category: Reasoning Error Error Reason: GPT-4V correctly understood the question and the image, determining that the blood sample after centrifugation separates into three layers. However, in the subsequent reasoning, GPT4V confused plasma with serum.

Ground Truth: (C) 血浆,白细胞和血小板,红细胞

- Figure B32: A sample error case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 125]

#### 健康与医学:基础医学

Question: <图片 1>图示箭头所指的细胞名称____ Option: (A) 蓝细胞（涂抹细胞） (B) 脂肪细胞 (C) 淋巴细胞 (D) 其余选项均不对

|[Figure 126]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是一个显微镜下的血液涂片。箭头所指的细胞由于图片的分辨率和缺乏特定 的染色细节，难以确定其确切类型。然而，根据其相对较大的尺寸和不规则的形状，这个细胞可能是 一种白细胞。在给出的选项中，蓝细胞（涂抹细胞）通常不是血液中的一个组成部分，脂肪细胞在血 液涂片中也不常见。淋巴细胞是血液中的一种常见白细胞，但是由于缺乏详细信息，无法确定这是否 是淋巴细胞。 因此，根据可用信息，最可能正确的选项是： (C) 淋巴细胞 但请注意，没有更多的 上下文信息和特定的染色细节，这个答案并不是确定无疑的。

Error Category: Reasoning Error Error Reason: GPT-4V correctly understood the question and the image, deducing that the image is a blood smear under the microscope. However, it excluded the option based on blue cells (smear cells), typically not being a component of blood. In usual circumstances, smear cells refer to damaged white blood cells during the process of making a blood smear.

Ground Truth: (A)蓝细胞（涂抹细胞）

- Figure B33: A sample error case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 127]

#### 健康与医学:基础医学

Question: 图<图片 1>中位于肝小叶相连出，有三个管，单层立方上皮的小叶间胆管、小叶间动、静 脉的是什么？_____

|[Figure 128]|
|---|

<图片 1>

GPT-4V: 这个结构是肝三角。在肝脏的组织学切片中，肝三角（也称为门区或门管区）是位于肝小叶 之间的区域，其中包含了三个主要结构：小叶间胆管、小叶间动脉和小叶间静脉。这些结构共同构成 了肝脏的血管和胆汁排泄系统的基本单元。小叶间胆管负责收集肝细胞分泌的胆汁，而小叶间动脉和 静脉则分别负责将血液输送到肝小叶和从肝小叶中带走血液。

Error Category: Answer Extraction Error Error Reason: GPT-4V correctly understood the question and the image, inferring from the image features that it is the liver hilum (also known as the porta hepatis). However, during the Extract & Match process, it did not match with GPT-4V's actual answer.

Ground Truth: 门管区

- Figure B34: A sample error case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 129]

#### 健康与医学:基础医学

Question: 图<图片 1>中核车轮状偏向一侧，核旁有淡染区的是（）

|[Figure 130]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是某种细胞的显微镜图像。根据您的描述，图中的细胞核车轮状偏向一侧， 核旁有淡染区，这可能是某种特定类型的白细胞。在血液学中，这样的描述可能指向某种特殊的白细 胞形态，如嗜中性粒细胞的带状核或其他类型的白细胞。然而，由于我无法提供专业的医学诊断，所 以不能确切地告诉您这是哪种细胞。通常，这种类型的分析需要由具有相应专业知识的医疗专业人员 来完成。

Error Category: Lack of Knowledge Error Reason: GPT-4V inferred from the question and the image that the cells in the picture might be a type of white blood cell. However, lacking specific knowledge, it couldn't deduce the cells to be plasma cells based on their features.

Ground Truth: 浆细胞

- Figure B35: A sample error case of Health and Medicine (subcategory: Basic Medicine). Back to List of figures

[Figure 131]

#### 健康与医学:临床医学

Question: 关于骨折的晚期综合征中，容易造成<图片 1>图中病理征象的是（） Option: (A) 缺血性骨坏死 (B) 缺血性肌挛缩 (C) 关节僵硬 (D) 损伤性骨化

|[Figure 132]|
|---|

<图片 1>

GPT-4V: 图片显示的是一只手，手指呈现异常弯曲的姿态。这种病理征象可能与多种病理状态有关， 但根据提供的选项，我们可以尝试确定最可能的原因。 (A) 缺血性骨坏死 - 这通常指的是由于血液 供应不足导致的骨组织死亡，可能会影响关节的功能，但不一定直接导致手指的异常弯曲。 (B) 缺 血性肌挛缩 - 这是由于肌肉血液供应不足导致的肌肉收缩，可能会导致手指或其他肢体的异常姿态。 (C) 关节僵硬 - 这是指关节失去正常活动范围，可能是由于长时间不动、炎症或其他病理原因。关 节僵硬可能导致手指无法正常伸展。 (D) 损伤性骨化 - 这是指在软组织中形成骨组织，通常是由于 损伤或炎症反应。这可能导致关节周围的异常硬化，但不一定直接导致手指的异常弯曲。 根据图片 中手指的异常弯曲姿态，最可能的选项是(B) 缺血性肌挛缩，因为这种情况下肌肉收缩可能导致手指 保持在不正常的位置。因此，正确答案是(B) 缺血性肌挛缩。

Ground Truth: (B)缺血性肌挛缩

- Figure B36: A sample correct case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 133]

#### 健康与医学:临床医学

Question:食管癌最好发的部位是图<图片 1>中哪个部位？ Option: (A) A (B) B (C) C (D) D

|[Figure 134]|
|---|

<图片 1>

GPT-4V:食管癌最常见的发生部位是食管中下段，因此在这幅图中，最可能正确的选项是： (C) C 这个部位大致对应于食管的中下段，是食管癌最好发的区域。

Ground Truth: (C) C

- Figure B37: A sample correct case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 135]

#### 健康与医学:临床医学

Question:患者，女，近期经常烦躁不安，心悸，且出现怕光、视力以及记忆力减退等症状，因此来 医院就诊，体查：患者眼裂增宽，眼球外突，如图<图片 1>所示。患者最可能患有（） Option:

(A)单纯性甲状腺肿 (B) 甲状腺功能亢进 (C) 甲状腺功能减退 (D) 甲状腺癌

|[Figure 136]|
|---|

<图片 1>

GPT-4V: 根据题目描述，患者出现了烦躁不安、心悸、怕光、视力减退和记忆力减退的症状，体查发 现眼裂增宽和眼球外突。这些症状和体征与甲状腺功能亢进相关的格雷夫斯病（Graves' disease） 相符，这是一种自身免疫性疾病，其特征之一就是突眼（exophthalmos）。因此，最可能的答案是： (B) 甲状腺功能亢进

Ground Truth: (B)甲状腺功能亢进

- Figure B38: A sample correct case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 137]

#### 健康与医学:临床医学

Question:患者，女，57岁，因子宫脱垂来医院就诊，具体情况如图<图片 1>所示，患者子宫脱垂的 程度为（）

Option: (A) I度轻型 (B) I度重型 (C) II度轻型 (D) II度重型

|[Figure 138]|
|---|

<图片 1>

GPT-4V:很抱歉，我无法提供医学诊断或评估。子宫脱垂的程度通常需要通过临床检查来确定，而不 是仅仅通过图片。在实际医疗情况中，医生会根据子宫下降的位置、是否超过阴道口以及相关症状来 评估脱垂的程度。通常，子宫脱垂分为I度（轻度）、II度（中度）和III度（重度）。由于我无法进 行实际的医学评估，建议咨询专业的医疗人员以获得准确的诊断。

Error Category: Reject to Answer Error Reason: Out of ethical considerations, GPT-4V refuses to provide medical-related diagnostic assessments.

Ground Truth: (D) II度重型

- Figure B39: A sample error case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 139]

#### 健康与医学:临床医学

Question:患者，男，40 岁，高空坠落摔伤半小时，图中部位发生疼痛，伴轻度恶心、心慌、出冷汗。 查体：血压 75/50mmHg，心率 130 次/分，<图片 1>图中部位胸壁皮肤挫裂伤且轻度压痛，反跳痛不 明显，局限性腹肌紧张，有移动性浊音，腹部抽出不凝血。该患者最有可能的诊断是（）

Option:

- (A)脾破裂
- (B)肝破裂
- (C)肠穿孔
- (D)胰破裂

|[Figure 140]|
|---|

<图片 1>

GPT-4V:根据题目描述，患者是一名40岁男性，经历了高空坠落摔伤，出现了疼痛、轻度恶心、心慌 和出冷汗的症状。查体发现血压偏低（75/50mmHg），心率加快（130次/分），胸壁皮肤挫裂伤且轻 度压痛，反跳痛不明显，局限性腹肌紧张，有移动性浊音，腹部抽出不凝血。这些症状和体征表明患 者可能有内脏器官损伤和内出血。 在提供的选项中，我们需要根据图中标注的部位和临床表现来判 断最可能的诊断： (A) 脾破裂 - 脾脏位于左上腹部，脾破裂可能导致左上腹痛、休克症状（如低血 压和心率加快），以及腹部抽出不凝血。 (B) 肝破裂 - 肝脏位于右上腹部，肝破裂可能导致右上腹 痛、休克症状，以及腹部抽出不凝血。 (C) 肠穿孔 - 肠穿孔通常会导致急性腹痛、腹膜炎症状（如 反跳痛和腹肌紧张），但不一定伴有移动性浊音。 (D) 胰破裂 - 胰腺位于上腹部中央，胰破裂可能 导致上腹痛、背痛，以及消化酶相关的症状。 考虑到患者的症状和体征，以及图中标注的部位，最 有可能的诊断是脾破裂或肝破裂。但由于图中标注的部位位于左侧，这使得脾破裂的可能性更高。因 此，正确答案是： (A) 脾破裂 请注意，这是基于题目描述和图示信息的推断，实际诊断需要通过进 一步的医学检查，如超声波检查、CT扫描等来确定。

Error Category: Perceptual Error Error Reason: GPT-4V lacks an understanding of the perspective in images. While from an external viewpoint, the wound appears to be on the upper left abdomen, from the patient's perspective, the wound is actually on the upper right abdomen. GPT-4V's failure to comprehend the image's perspective resulted in an error.

Ground Truth: (B)肝破裂

- Figure B40: A sample error case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 141]

#### 健康与医学:临床医学

Question:患者，女，60 岁，今日晨独自上卫生间摔倒，伤后感下肢剧烈疼痛，不能站立行走，通过 <图片 1>图中判断该患者最有可能发生了（）

Option: (A)股骨干骨折 (B)髋关节脱位 (C)胫腓骨干骨折 (D)股骨颈骨折

|[Figure 142]|
|---|

<图片 1>

GPT-4V:根据图片中的描绘，我们可以看到患者躺在床上，其中一条腿外展和外旋的姿势。这种姿势 通常与髋部的问题相关，尤其是在老年人中，股骨颈骨折是一个常见的诊断，尤其是在跌倒后无法承 重的情况下。正确答案是： (D) 股骨颈骨折 这个结论是基于图中患者的腿部姿势和年龄，以及描述 的症状。然而，最终的诊断应该由医疗专业人员通过适当的临床评估和影像学检查来确定。

Error Category: Annotation Error Error Reason: GPT-4V accurately interpreted the image where the patient is lying in bed with one leg in an abducted and externally rotated position. The model made the correct inference, but an error occurred in our annotations. The Ground Truth should be a (D)股骨颈骨折

Ground Truth: (C)胫腓骨干骨折

- Figure B41: A sample error case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 143]

#### 健康与医学:临床医学

Question:患者，男，32 岁，腰椎间盘突出入院，行髓核摘除术，术后护士指导患者进行<图片 1>图 示锻炼，其目的是为了预防：

Option:

(A)肌肉萎缩 (B)切口感染 (C)形成血肿 (D)神经根粘连

|[Figure 144]|
|---|

<图片 1>

GPT-4V:这张图片展示了一个人躺在背上进行腿部锻炼的示意图。根据这个锻炼的姿势，它似乎是为 了加强腿部和腰部的肌肉。在腰椎间盘突出手术后，进行适当的锻炼可以帮助恢复肌肉力量，预防肌 肉萎缩，同时也有助于促进腰部的稳定性和灵活性，减少神经根粘连的风险。因此，根据提供的选项， 正确答案可能是： (A) 肌肉萎缩 (D) 神经根粘连 这两个选项都与术后恢复锻炼的目的相关。然而， 如果必须选择一个最可能的答案，我会选择： (A) 肌肉萎缩 因为这种类型的锻炼首先和直接地关联 到肌肉力量的维持和增强。

Error Category: Reasoning Error Error Reason: GPT-4V accurately analyzed the image: a person lying on their back for leg exercises and inferred the possible correct answers in both (A) and (D). However, it overlooked that the patient was admitted due to lumbar disc herniation, and the exercise was intended to prevent (D) nerve root adhesion after lumbar disc herniation surgery, leading to an reasoning error.

Ground Truth: (D)神经根粘连

- Figure B42: A sample error case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 145]

#### 健康与医学:临床医学

Question:某病人，现在出现呼吸困难，通过<图片 1>图示，可以判定病人出现了哪种疾病？（）

Option: (A)脓胸 (B) 张力性气胸 (C) 开放性气胸 (D) 闭合性气胸

|[Figure 146]|
|---|

<图片 1>

GPT-4V:这张图片展示了两种不同的胸部情况。左边（1）显示的是正常的胸腔，没有明显的异常。 右边（2）显示的胸腔中，肺部有明显的塌陷，这是气胸的特征。气胸是指空气进入胸膜腔，导致肺 部不能充分扩张的情况。 在这四个选项中，脓胸（A）通常是由感染引起的脓液积聚在胸膜腔内，而 不是空气。开放性气胸（C）是指胸壁有开放性伤口，使得外界空气可以进入胸膜腔。闭合性气胸（D） 是指胸壁没有明显的开放性伤口，但胸膜腔内仍然有空气积聚。张力性气胸（B）是一种严重的气胸， 其中空气持续进入胸膜腔并不能逸出，导致压力增加，影响心脏和肺的功能。 由于图片中没有显示 胸壁的开放性伤口，也没有足够的信息来判断是否为张力性气胸，因此最可能的选项是闭合性气胸 （D）。然而，需要注意的是，仅凭一张图示并不能确诊，通常需要结合患者的临床表现和其他诊断 手段，如X光片等，来确定确切的疾病类型。 正确答案：(D) 闭合性气胸

Error Category: Textual Understanding Error Reason: GPT-4V misunderstood the question, where the left side (1) represents inhalation and the right side (2) represents exhalation, both depicting pathological images. GPT-4V incorrectly interpreted the left side (1) as a normal thoracic cavity.

Ground Truth: (C)开放性气胸

- Figure B43: A sample error case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 147]

#### 健康与医学:临床医学

Question:患者男，50 岁。确诊为 COPD，现要求护士指导患者做呼吸功能锻炼，如图片<图片 1>所 示，下列描述正确的是：()

Option:

(A)指导患者缩唇吸气，用鼻缓慢呼气 (B) 指导患者用鼻吸气，缩唇快速呼气

(C) 吸气和呼气之比为 1：2 (D) 吸气和呼气之比是 2：1

|[Figure 148]|
|---|

<图片 1>

GPT-4V:根据提供的图片和描述，正确答案是： (A) 指导患者缩唇吸气，用鼻缓慢呼气 图片展示了 一个人在进行缩唇呼吸练习，这是一种常用于COPD（慢性阻塞性肺疾病）患者的呼吸技巧。该技巧涉 及通过鼻子吸气和通过缩小的嘴唇缓慢呼气，以增加气道压力，减少气道塌陷，帮助改善呼吸。选项 (B)中提到的“快速呼气”是不正确的，因为缩唇呼吸的目的是要缓慢呼气。选项(C)和(D)关于吸气 和呼气比例的描述没有直接体现在图片中，但通常在缩唇呼吸中，呼气时间会比吸气时间长，所以(C)

可能更接近正确的呼吸比例。然而，没有足够的信息来确定确切的比例，因此最准确的答案是(A)。

Error Category: Answer Extraction Error Error Reason: GPT-4V correctly interpreted the image and made an accurate judgment. However, in its subsequent analysis, the inclusion of keywords such as “正确" led to errors in the Extract & Match process, resulting in the extraction of answers (A) and (C).

Ground Truth: (A)指导患者缩唇吸气，用鼻缓慢呼气

- Figure B44: A sample error case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 149]

#### 健康与医学:临床医学

Question:患儿，男，13 岁，患有化脓性骨髓炎，经全身抗生素治疗，效果不佳，给予手术治疗，进 行引流脓液，如图<图片 1>所示，请问①和②高度分别为？

Option:

(A) 40cm～50cm和30cm (B) 50cm～60cm和40cm (C) 60cm～70cm和50cm (D) 70cm～80cm和60cm

|[Figure 150]|
|---|

<图片 1>

GPT-4V: 很抱歉，但我无法提供精确的测量结果，因为图片中没有提供尺度参考或任何可以用来确定 实际高度的标记。因此，我无法确定①和②的确切高度。如果这是一个理论或实践考试问题，通常会 有一些背景知识或标准实践来指导你选择最合适的答案。在没有这些信息的情况下，我无法提供正确 答案。

Error Category: Lack of Knowledge Error Reason: GPT-4V correctly interpreted the image and identified both ① and ②, but due to a lack of practical knowledge, it could not determine the height of ① and ②.

Ground Truth: (C) 60cm～70cm和50cm

- Figure B45: A sample error case of Health and Medicine (subcategory: Clinical Medicine). Back to List of figures

[Figure 151]

#### 健康与医学: 诊断学与实验室医学

Question:如<图片 1>所示为常见的热型中的_____。 Option:

(A) 间歇热 (B) 弛张热 (C) 回归热 (D) 稽留热

|[Figure 152]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一张体温图，上面有体温随时间变化的记录。从图中可以看出，体温在一 段时间内保持正常，然后突然升高，之后又迅速下降到正常水平，这种模式重复出现。这种体温变化 模式符合间歇热的特点，间歇热是指体温在发热期间升高，然后在非发热期间恢复正常。

正确答案是： (A) 间歇热

Ground Truth: (A)间歇热

- Figure B46: A sample correct case of Health and Medicine (subcategory: Diagnostics and Lab Medicine).

Back to List of figures

[Figure 153]

#### 健康与医学: 诊断学与实验室医学

Question:识别<图片 1>切片为_ Option: (A) 肝细胞脂肪变性 (B) 上皮细胞脂肪变性 (C) 淋巴细胞脂肪变性 (D) 其余选项均不对

|[Figure 154]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一个显微镜下的组织切片。在这个切片中，我们可以看到许多含有大量空 泡的细胞，这些空泡可能代表脂肪滴。这种现象通常与脂肪变性有关，这是一种细胞内脂肪积累的病 理过程。

在给出的选项中，我们需要确定哪种细胞类型发生了脂肪变性。肝细胞脂肪变性（选项A）是一种常 见的病理现象，通常与酒精性或非酒精性肝病有关。上皮细胞脂肪变性（选项B）和淋巴细胞脂肪变 性（选项C）则不是常见的术语或诊断。

根据图像中的组织结构和细胞类型，这个切片最有可能是来自肝脏的。因此，最可能正确的选项是： (A) 肝细胞脂肪变性 请注意，没有临床信息和特定的染色细节，这只是一个基于图像特征的假设性判断。实际的诊断应由 专业的医疗人员在综合所有相关信息后做出。

Ground Truth: (A)肝细胞脂肪变性

- Figure B47: A sample correct case of Health and Medicine (subcategory: Diagnostics and Lab Medicine).

Back to List of figures

[Figure 155]

#### 健康与医学: 诊断学与实验室医学

Question:请根据以下PAgT的实验结果<图片 1>判断可能的疾病。病例1：____________；病例2： ___________；病例3：___________；病例4：___________。

Option:

- (A) 血小板聚集功能正常；血管性血友病或巨大血小板综合征；血小板释放功能障碍或贮存池病；血 小板无力症
- (B) 血管性血友病或巨大血小板综合征；血小板释放功能障碍或贮存池病；血小板无力症；血小板聚 集功能正常
- (C) 血小板释放功能障碍或贮存池病；血小板无力症；血小板聚集功能正常；血管性血友病或巨大血 小板综合征
- (D) 血管性血友病或巨大血小板综合征；血小板无力症；血小板释放功能障碍或贮存池病；血小板聚 集功能正常

|[Figure 156]|
|---|

<图片 1>

GPT-4V:根据图片中提供的PAgT（血小板聚集试验）的实验结果，我们可以尝试对每个病例进行分析：

- 病例1：所有的刺激剂（ADP、Epinephrine、Collagen、Arachidonate、Ristocetin）都显示正常聚 集（N），这表明病例1的血小板聚集功能正常。
- 病例2：除了Ristocetin刺激下显示异常聚集（A）外，其他刺激剂下的聚集都是正常的。Ristocetin 异常通常与血管性血友病或巨大血小板综合征相关。
- 病例3：ADP、Epinephrine、Collagen刺激下显示异常聚集（A），而Ristocetin刺激下显示正常聚集 （N）。这种模式可能表明血小板释放功能障碍或贮存池病。
- 病例4：所有刺激剂（除了Ristocetin）下都显示异常聚集（A），这可能表明血小板无力症。 因此，根据这些信息，正确答案是：

- 病例1：血小板聚集功能正常；
- 病例2：血管性血友病或巨大血小板综合征；
- 病例3：血小板释放功能障碍或贮存池病；
- 病例4：血小板无力症。

所以正确答案是 (A) 血小板聚集功能正常；血管性血友病或巨大血小板综合征；血小板释放功能障 碍或贮存池病；血小板无力症。

Ground Truth: (A)

- Figure B48: A sample correct case of Health and Medicine (subcategory: Diagnostics and Lab Medicine).

Back to List of figures

[Figure 157]

#### 健康与医学: 诊断学与实验室医学

Question:<图片 1>图示可见的异常细胞名称（） Option:

(A) 分化好腺癌细胞 (B) 分化差腺癌细胞 (C) 分化好纤维性鳞癌细胞 (D) 放疗反应细胞

|[Figure 158]|
|---|

<图片 1>

GPT-4V:很抱歉，但我无法提供医学诊断或识别特定类型的细胞。这张图片显示的是染色后的细胞， 但是没有足够的上下文信息或专业知识来确定这些细胞的确切类型。在医学领域，正确识别和分类细 胞通常需要专业的训练和知识，以及详细的临床信息。因此，我无法选择正确答案。如果这是一个医 学相关的问题，建议咨询具有相关专业知识的医疗专业人员。

Error Category:Reject to Answer Error Reason:The security mechanisms of the GPT series make GPT-4V extremely conservative in critical areas such as healthcare, hence it refuses to answer.

Ground Truth: (D)放疗反应细胞

- Figure B49: A sample error case of Health and Medicine (subcategory: Diagnostics and Lab Medicine).

Back to List of figures

[Figure 159]

#### 健康与医学: 诊断学与实验室医学

Question:<图片 1>中何者不属于胼胝体的分部（） Option:

(A) 胼胝体嘴 (B) 胼胝体膝 (C) 胼胝体干 (D) 胼胝体体部

|[Figure 160]|
|---|

<图片 1>

GPT-4V:抱歉，由于图片中的文字和标记不足以提供关于胼胝体分部的确切信息，我无法直接从图片 中确定哪个选项是不属于胼胝体的分部。然而，根据我所知，胼胝体的主要分部包括胼胝体前部（包 括嘴部）、膝部、干部和后部（有时称为体部）。因此，所有提供的选项（A）胼胝体嘴、（B）胼胝 体膝、（C）胼胝体干和（D）胼胝体体部都是胼胝体的一部分。没有提供不属于胼胝体分部的选项。 如果必须选择一个答案，我建议重新检查题目和选项，因为可能存在误解或信息不足。

Error Category:Reject to Answer Error Reason:GPT-4V refused to answer due to a lack of relevant knowledge, but the question can lead to the conclusion D.

Ground Truth: (D)胼胝体体部

- Figure B50: A sample error case of Health and Medicine (subcategory: Diagnostics and Lab Medicine).

Back to List of figures

[Figure 161]

#### 健康与医学: 诊断学与实验室医学

Question:下面是某基因中的一个片段：<图片 1>指出转录的方向_______，转录模板是______。 Option:

(A) 从左向右；负链 (B) 从右向左；正链 (C) 从右向左；负链 (D) 从左向右；正链

|[Figure 162]|
|---|

<图片 1>

GPT-4V:在DNA中，转录是由RNA聚合酶催化的过程，它沿着DNA模板链合成RNA。RNA聚合酶在DNA模板 链上移动的方向是从3'到5'，因为RNA是按照5'到3'的方向合成的。在提供的DNA片段中，我们需要找 到哪条链作为模板，并确定转录的方向。

在这个例子中，我们没有直接的信息来确定哪条链是模板链，因为转录可以在两条链上的任何一条进 行，取决于基因的具体位置和方向。然而，通常情况下，我们可以假设图像中提供的信息是为了回答 问题而设计的。因此，我们可以假设转录是沿着3'到5'方向的链进行的，这样RNA聚合酶就可以在5' 到3'方向上合成RNA。

根据这个逻辑，如果转录是沿着上面的链（负链）从5'到3'方向进行的，那么RNA聚合酶将会沿着下 面的链（正链）从3'到5'方向移动，这意味着转录的方向是从右向左，转录模板是正链。

因此，正确答案是： (B) 从右向左；正链",

Error Category:Reasoning Error Error Reason:Transcription indeed proceeds from right to left, therefore it occurs along the 3' to 5' direction of the template (negative) strand. The strand mentioned above (negative strand) is the transcription template. GPT-4V overextended in its deduction, leading to an incorrect conclusion.

Ground Truth: (C)血小板聚集功能正常；血管性血友病或巨大血小板综合征；血小板释放功能障碍 或贮存池病；血小板无力症

- Figure B51: A sample error case of Health and Medicine (subcategory: Diagnostics and Lab Medicine).

Back to List of figures

[Figure 163]

#### 健康与医学: 诊断学与实验室医学

Question:<图片 1>图示为外周血涂片，中心区细胞名称（） Option: (A) 浆细胞 (B) 淋巴细胞 (C) 巨核细胞 (D) 反应性淋巴细胞

|[Figure 164]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一个血涂片中的细胞，具有较大的细胞体积、圆形或略呈不规则形状的细 胞核，以及宽广的蓝色胞浆。根据这些特征，这个细胞最有可能是一个淋巴细胞。淋巴细胞通常具有 圆形或略呈不规则形状的细胞核和较宽的胞浆。因此，正确答案是：

(B) 淋巴细胞",

Error Category:Lack of Knowledge Error Reason:GPT-4V overlooked or lacked some knowledge. In fact, lymphocytes do indeed possess the characteristics mentioned by GPT-4V. However, the cell in the image not only has blue cytoplasm but also has a very irregular shape, characteristic of reactive lymphocytes. Similarly, reactive lymphocytes also possess the aforementioned characteristics.

Ground Truth: (D)反应性淋巴细胞

- Figure B52: A sample error case of Health and Medicine (subcategory: Diagnostics and Lab Medicine).

Back to List of figures

[Figure 165]

#### 健康与医学: 诊断学与实验室医学

Question:判断下面陈述对错：下列是DNA的一段碱基序列<图片 1>，DNA聚合酶以上面的DNA片段为模 板，复制出的DNA碱基序列为5'-CTAATGCAACGTTGCAAGCT-3'。

|[Figure 166]|
|---|

<图片 1>

GPT-4V:在DNA复制中，DNA聚合酶会根据模板链的碱基序列合成一个互补的新链。在这个过程中，腺 嘌呤（A）与胸腺嘧啶（T）配对，而鸟嘌呤（G）与胞嘧啶（C）配对。因此，我们可以根据所给的模 板链序列来确定复制出的新链的序列。

所给的模板链序列为5'-AGCTTGCAACGTTGCATTAG-3'。根据碱基配对原则，复制出的新链应该是模板链 的互补链，方向相反，即从3'到5'。所以，复制出的新链应该是3'-TCGAACGTTGCAATGTAATC-5'。但是， 通常我们按照5'到3'的方向来写出序列，因此，正确的新链序列应该是5'-GATTAACGTGCAACGTTGAC-3'。

题目中给出的复制出的DNA碱基序列为5'-CTAATGCAACGTTGCAAGCT-3'。与我们根据模板链推导出的序 列相比较，可以看出题目中给出的序列与正确的互补序列不一致。

因此，根据题目描述和所给的信息，陈述是错误的。

Error Category:Lack of Knowledge Error Reason:The text given in the question is actually the reverse complementary sequence of the image DNF, so it's correct that the former is generated by the latter through DNA polymerase. GPT4V lacks knowledge about reverse complementary sequences, leading to an incorrect judgment.

Ground Truth: (正确)

- Figure B53: A sample error case of Health and Medicine (subcategory: Diagnostics and Lab Medicine).

Back to List of figures

[Figure 167]

#### 健康与医学: 制药

Question: 如图<图片 1>，少部分药物小剂量时以一级速率转运，而在大剂量时以零级速率转运。因 此描述这类药物的消除速率需要将两种速率类型结合起来，通常以_描述。

Option: (A) 罗﹣雷氏方程式 (B) 唐﹣曼氏方程式 (C) 米﹣曼氏方程式描述 (D) 其余选项均不对

|[Figure 168]|
|---|

<图片 1>

GPT-4V:正确答案是：

(C) 米﹣曼氏方程式描述

这个图表展示了药物消除速率随药物浓度变化的两种不同模式：小剂量时的一级速率转运和大剂量时 的零级速率转运。米-曼氏方程式（Michaelis-Menten equation）通常用于描述这种情况，它结合了 一级动力学和零级动力学，用于描述在不同浓度下酶促反应的速率，这在药物动力学中也适用。

Ground Truth: (C)米﹣曼氏方程式描述

- Figure B54: A sample correct case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 169]

#### 健康与医学: 制药

Question: 如图<图片 1>，下列化合物经硅胶柱层析分离时，用氯仿—甲醇系统作洗脱剂，其洗脱顺 序（）

Option:

(A) 丙>丁>甲>乙 (B) 乙>甲>丁>丙 (C) 丁>丙>甲>乙 (D) 乙>甲>丙>丁

|[Figure 170]|
|---|

<图片 1>

GPT-4V:在硅胶柱层析中，化合物的洗脱顺序通常取决于它们与洗脱剂的相互作用强度。洗脱剂的极 性越高，与化合物的相互作用也越强，因此极性较低的化合物会先洗脱出来。

在这个问题中，我们需要考虑化合物的极性。化合物的极性通常与它们的官能团有关，如羟基（-OH） 和羧基（-COOH）是极性官能团，而甲氧基（-OCH3）和甲基（-CH3）是非极性官能团。

根据图中的化合物结构，我们可以看到：

- - 甲化合物有三个羟基和一个甲氧基。
- - 乙化合物有一个羟基、一个羧基和两个酮基。
- - 丙化合物有三个羟基和一个甲基。
- - 丁化合物有一个羟基和两个甲基。

羧基（-COOH）是这里最极性的官能团，因此乙化合物的极性最高，应该最后洗脱。接下来是羟基（OH），甲化合物和丙化合物都有三个羟基，但甲化合物还有一个甲氧基，这使得甲化合物的极性略低 于丙化合物。最后，丁化合物有最少的羟基和最多的甲基，使其极性最低，应该最先洗脱。

因此，洗脱顺序应该是： 丁（最非极性，先洗脱）> 甲 > 丙 > 乙（最极性，最后洗脱）

正确答案是 (C) 丁>丙>甲>乙。

Ground Truth: (C)丁>丙>甲>乙

- Figure B55: A sample correct case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 171]

#### 健康与医学: 制药

Question: 给某患者静脉注射一单室模型药物,剂量为100.0mg,测得不同时刻血药浓度的数据如下表< 图片 1>所示。该药物的消除速率常数(单位$h^{-1}$)是__

|[Figure 172]|
|---|

<图片 1>

GPT-4V:为了计算消除速率常数（k），我们通常使用一阶动力学方程，其中血药浓度随时间的下降 可以表示为：

C = 𝐶𝐶0* 𝑒𝑒−𝑘𝑘𝑘𝑘 其中 C 是在时间 t 时的血药浓度，𝐶𝐶0是初始血药浓度，k是消除速率常数，t是时间。 从所提供的数据中，我们可以通过取两个时间点的浓度值来估算消除速率常数k，我们通常会取对数 值来将指数方程转换为线性方程，这样可以通过线性回归来估算 k。 ln(C) = ln(𝐶𝐶0) - kt 我们可以选择任意两个时间点的数据来计算 k。例如，我们可以选择 t=1h 时的浓度和 t=6h 时的浓 度。首先，我们需要计算每个时间点的 ln(C)。 对于 t=1h，C=8.40 μg/mL，ln(8.40) = 2.128 对于 t=6h，C=1.48 μg/mL，ln(1.48) = 0.392 现在我们有两个点 (1, 2.128) 和 (6, 0.392)。我们可以使用这两个点来计算斜率（即消除速率常 数 k）。

k = (ln(𝐶𝐶0) - ln(𝐶𝐶𝑘𝑘)) / t k = (2.128 - 0.392) / (6 - 1) k = 1.736 / 5 k = 0.3472 ℎ−1

因此，该药物的消除速率常数大约是 0.3472 ℎ−1。",

Ground Truth: (0.3465)

- Figure B56: A sample correct case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 173]

#### 健康与医学: 制药

Question:下图<图片 1>药物中，分子中含有氨甲酸乙酯结构的药物是__ Option:

(A) 班布特罗 (B) 地匹福林 (C) 特非那定 (D) 氯雷他定

|[Figure 174]|
|---|

<图片 1>

GPT-4V:氨甲酸乙酯结构是一个含有以下官能团的结构：-O-C(=O)-NHR，其中R可以是任何烷基或芳 香基。 在提供的选项中，我们需要找到包含这个官能团的分子。通过观察图中的分子结构：

- - (A) 班布特罗：没有氨甲酸乙酯结构。 - (B) 地匹福林：没有氨甲酸乙酯结构。 - (C) 特非那定：分子结构中含有氨甲酸乙酯结构。
- - (D) 氯雷他定：没有氨甲酸乙酯结构。 因此，正确答案是 (C) 特非那定。

Error Category:Perceptual Error Error Reason:The molecule in option C contains an amino (-NH-) group and an ether (-O-) structure. These two functional groups might be confused with components of ethyl carbamate. While these two parts contain nitrogen and oxygen atoms, respectively, they do not form an ethyl carbamate structure. Moreover, the compound in option C includes an ether oxygen (-O-) situated between a phenyl ring and an ethoxy group, which led to the incorrect selection by GPT-4

Ground Truth: (D)氯雷他定

- Figure B57: A sample error case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 175]

#### 健康与医学: 制药

Question:制药反应设备中的反应器类型很多，特点不一，可按不同方式进行分类。若按结构分类， 如<图片 1>所示为（）反应器。

Option: (A) 板式塔 (B) 填料塔 (C) 鼓泡塔 (D) 喷雾塔

|[Figure 176]|
|---|

<图片 1>

GPT-4V:这张图片展示的是一个塔式反应器的简化示意图。根据图中显示的特征，我们可以看到塔体 内部似乎填充了大量的小颗粒，这通常是填料塔的特点。填料塔内部填充了填料，以增加气液之间的 接触面积，从而提高传质效率。因此，最可能正确的选项是：

(B) 填料塔

Error Category:Perceptual Error Error Reason:GPT-4V incorrectly interpreted the image, mistaking the edge outline of the water mist from the nozzle as the body of a tower filled with material. This led to the mistaken identification of option (D), the spray tower, as option (B), the packed tower.

Ground Truth: (D)喷雾塔

- Figure B58: A sample error case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 177]

#### 健康与医学: 制药

Question:夹套是制药的搅拌釜式反应器的传热元件，如<图片 1>所示为（）。 Option:

(A) 环形夹套 (B) 螺旋挡板夹套 (C) 半管螺旋夹套 (D) 带扰流喷嘴的环形夹套

|[Figure 178]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一个夹套的简化图。根据图中所示的结构，这个夹套是一个环绕在反应器 外部的通道，用于流体介质的流动，以实现传热的目的。图中并没有显示出螺旋挡板或者半管螺旋的 特征，也没有带扰流喷嘴的特征。因此，最可能正确的选项是：

(A) 环形夹套 正确答案是 (A) 环形夹套。

Error Category:Perceptual Error Error Reason:The image clearly shows a turbulence nozzle, but GPT-4V might not have recognized it. Due to insufficient ability to read images or weaker recognition capabilities, it failed to correctly obtain all the true information from the image, which further led to the wrong selection.

Ground Truth: (D)带扰流喷嘴的环形夹套

- Figure B59: A sample error case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 179]

#### 健康与医学: 制药

Question:下列化合物，何者酸性最强的是（） Option:

(A) <图片 1> (B) <图片 2> (C) <图片 3> (D) <图片 4>

|[Figure 180]|
|---|

|[Figure 181]|
|---|

<图片 1>

<图片 2>

|[Figure 182]|
|---|

|[Figure 183]|
|---|

<图片 3> <图片 4>

GPT-4V:正确答案是 (A) 环形夹套。在这个问题中，我们需要比较四个化合物的酸性强度。酸性强 度可以通过羟基（-OH）的能力来释放质子（H+）来评估。通常，当羟基连接的芳香环上的取代基能 够稳定负电荷时，酸性会增强。这是因为取代基可以通过共轭效应或诱导效应来稳定脱质子后的负电 荷。

- (A) 化合物具有一个甲氧基（-OCH3）和一个羟基（-OH）在芳香环上，甲氧基是吸电子基团，会减弱 酸性。
- (B) 化合物具有两个甲氧基和一个羟基，甲氧基的数量更多，因此会进一步减弱酸性。
- (C) 化合物具有一个羟基和一个甲氧基，但羟基位于与甲氧基不同的环上，这可能会影响其酸性。
- (D) 化合物具有两个羟基和一个甲氧基，羟基的数量更多，这可能会增强酸性。

在这些选项中，羟基的数量和位置以及甲氧基的效应都会影响酸性。羟基是提供酸性的基团，而甲氧 基是吸电子基团，会减弱酸性。因此，具有更多羟基和较少甲氧基的化合物通常酸性更强。\n\n根据 这些信息，(D) 选项中的化合物具有两个羟基和一个甲氧基，这使得它比其他选项中的化合物具有更 强的酸性。因此，正确答案是： 正确答案：(D)

Error Category:Perceptual Error Error Reason:Obviously, GPT-4V incorrectly counted the number of hydroxyl groups in the figure. In reality, option C has the same number of hydroxyl groups as D, and there are two methoxy groups on the same ring as the hydroxyl groups, which would greatly enhance the acidity of the compound. Therefore, this is a case of misreading the diagram.

Ground Truth: (C)<图片 3>

- Figure B60: A sample error case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 184]

#### 健康与医学: 制药

Question:下图<图片 1>为药物与受体的亲和力及内在活性对量-效曲线的影响图,下列描述正确的是() Option:

(A) a、b、c三药和受体的亲和力相等 (B) a、b、c三药的内在活性相等 (C) a药的亲和力最大 (D) c药的内在活性最大

|[Figure 185]|
|---|

<图片 1>

GPT-4V:根据所提供的量-效曲线图，我们可以对药物与受体的亲和力及内在活性进行分析：

- (A) a、b、c三药和受体的亲和力相等 - 这个选项不正确。在量-效曲线中，药物与受体的亲和力通 常由EC50（或此图中的pD2）表示，即产生50%最大效应所需的药物浓度。曲线越向左，亲和力越高。 从图中可以看出，三条曲线的EC50（pD2）不同，因此亲和力不相等。
- (B) a、b、c三药的内在活性相等 - 这个选项不正确。内在活性是指药物达到的最大效应，即E(%)的 最大值。从图中可以看出，药物a、b、c达到的最大效应不同，因此内在活性不相等。
- (C) a药的亲和力最大 - 这个选项不正确。从图中可以看出，药物c的曲线最靠左，表明其EC50（pD2） 最小，因此c药的亲和力最大。
- (D) c药的内在活性最大 - 这个选项正确。从图中可以看出，药物c的曲线达到了最高的E(%)值，即 100%，表明其内在活性最大。 因此，正确答案是 (D)c药的内在活性最大。

Error Category:Perceptual Error Error Reason:GPT-4V needs to improve its understanding of the affinity and intrinsic activity of drugs with receptors. KD represents the affinity of a drug for a receptor, and its value is equal to EC50. The negative logarithm of KD (-lgKD) is the affinity index (pD2), which is directly proportional to the affinity. The effect of a drug binding to a receptor depends not only on its affinity but also on its intrinsic activity. When two drugs have equal affinity, their effect depends on their intrinsic activity. When two drugs have equal intrinsic activity, the effect depends on their affinity. From the diagram, it can be seen that the EC50 of drugs a, b, and c are equal, meaning their affinity indices (pD2) are equal, but their maximum effects are a > b > c, indicating different intrinsic activities.

Ground Truth: (A)a、b、c三药和受体的亲和力相等

- Figure B61: A sample error case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 186]

#### 健康与医学: 制药

Question:如图<图片 1>，此生物碱结构属于（） Option: (A) 吲哚类 (B) 莨菪烷类 (C) 喹啉类 (D) 萜类

|[Figure 187]|
|---|

<图片 1>

GPT-4V:这个结构是一个吲哚类生物碱的结构。吲哚类生物碱的特征是含有一个吲哚环，即一个六元 的苯环与一个五元含氮杂环相融合。在您提供的结构中，左侧的部分是一个典型的吲哚结构。因此， 正确答案是：

(A) 吲哚类

Error Category:Lack of Knowledge Error Reason:.GPT-4V believed that the left part of the image was a typical indole structure, but in reality, it is only structurally similar. The left part of the image has an additional nitrogen atom and is not a benzene ring, indicating that GPT-4V did not precisely define the indole structure. Therefore, it incorrectly chose option A. Option B, on the other hand, consists of a tropane structure formed by a seven-membered ring, a six-membered ring, and a five-membered ring sharing one carbon atom and two nitrogen atoms. Hence, option B is the correct choice.

Ground Truth: (B)莨菪烷类

- Figure B62: A sample error case of Health and Medicine (subcategory: Pharmacy). Back to List of figures

[Figure 188]

#### 健康与医学:公共卫生

Question:甲、乙两人群中几种特殊部位癌肿报告新病例的相对数<图片 1>：据此推论甲人群较乙人 群更易患肺癌、乳腺癌、子宫颈癌，该推论（） Option: (A) 不正确，因为未区分发病率或死亡率 (B) 不正确，因为未用率指标测量 (C) 不正确，因为未设置对照组 (D) 不正确，因为未进行率的标化

|[Figure 189]|
|---|

<图片 1>

GPT-4V: 很抱歉，由于图片中的数据不足以提供完整的信息来确定甲人群是否比乙人群更易患肺癌、 乳腺癌、子宫颈癌，因此无法直接得出结论。但是，根据所提供的选项，我们可以分析哪个选项最可 能是正确的。 选项(A)提到了未区分发病率或死亡率，这是一个重要的考虑因素，因为仅有的新病例 数并不能直接反映人群的风险，需要知道这些新病例是在多大的人群中发生的。 选项(B)指出了未用 率指标测量，这同样是一个关键点，因为没有考虑到人群的大小，单纯的新病例数不能反映真实的发 病风险。 选项(C)提到了未设置对照组，这对于比较研究来说是重要的，但在这个情况下，甲和乙两 个人群本身就是相互比较的，所以这个选项可能不是最合适的。 选项(D)提到了未进行率的标化，这 是一个关键的统计过程，用于调整不同人群中的年龄、性别等结构差异，以便更准确地比较发病率。 根据这些信息，最可能正确的选项是(B)不正确，因为未用率指标测量。这是因为没有提供足够的信 息来计算标准化的发病率或死亡率，这是比较不同人群风险的关键。

Ground Truth: (B)不正确，因为未用率指标测量

- Figure B63: A sample correct case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 190]

#### 健康与医学:公共卫生

Question:识别<图片 1>为（）的成像 Option: (A) 钩虫卵 (B) 带绦虫卵 (C) 血吸虫卵 (D) 受精蛔虫卵

|[Figure 191]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是在显微镜下观察到的一个卵。根据卵的形状、大小和可能的特征，我们可 以尝试识别它。然而，由于我不是医学专家，我不能提供一个确切的诊断。通常，钩虫卵较小且呈椭 圆形，带绦虫卵具有非常特殊的形状和内部结构，血吸虫卵通常有一个特征性的尖端或突起，而受精 蛔虫卵则呈圆形或椭圆形，表面可能有不规则的外壳。 根据这张图片，我们可以看到卵是圆形的， 并且有一个明显的外壳。然而，没有更多的特定信息，很难确定这是哪种寄生虫的卵。如果必须猜测， 我会说这个卵最可能是(D) 受精蛔虫卵，因为它看起来比较接近这种类型的卵的描述。但请注意，这 只是一个基于可见特征的猜测，而不是一个专业的医学诊断。 正确答案：(D) 受精蛔虫卵（这是一 个基于图像和一般知识的猜测，而非确切答案）

Ground Truth: (D)受精蛔虫卵

- Figure B64: A sample correct case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 192]

#### 健康与医学:公共卫生

Question:为了评价水痘疫苗的流行病学效果，其随访结果见<图片 1>，该疫苗的保护率是（） Option: (A) 0.8 (B) 0.833 (C) 0.9 (D) 0.925

|[Figure 193]|
|---|

<图片 1>

发病率

接种组 发病率

GPT-4V: 保护率（Efficacy）可以通过以下公式计算： [𝐸𝐸 = (1 −

) × 100%]

对照组

首先，我们需要计算接种组和对照组的发病率。 接种组发病率 = 发病人数 / 接种人数 = 10 / 400 = 0.025 对照组发病率 = 发病人数 / 对照人数 = 90 / 600 = 0.15 然后，我们将这些值代入 保护率公式: [𝐸𝐸 = (1 − 0.0250.15 ) × 100%][𝐸𝐸 = (1 − 16) × 100%][𝐸𝐸 = (1 − 0.1667) × 100%][𝐸𝐸 = 0.8333 × 100%][𝐸𝐸 = 83.33%]

所以，保护率最接近的选项是 (B) 0.833。 Ground Truth: (B)0.833

- Figure B65: A sample correct case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 194]

#### 健康与医学:公共卫生

Question:识别<图片 1>内容为（）虫卵 Option: (A) 肝吸虫卵 (B) 肺吸虫卵 (C) 鞭虫卵 (D) 姜片虫卵

|[Figure 195]|
|---|

<图片 1>

GPT-4V: 很抱歉，我无法提供医学诊断或识别具体的寄生虫卵。图片显示的是在显微镜下观察到的一 个物体，但是没有专业的医学背景和具体的上下文信息，我无法确定这是哪种寄生虫的卵。正确识别 寄生虫卵通常需要专业的知识和实验室测试。如果需要此类信息，请咨询医学专业人士。

Error Category: Reject to Answer Error Reason: Due to ethical considerations and other factors, GPT-4V refuses to provide medical diagnoses or identify specific parasitic eggs.

Ground Truth: (B)肺吸虫卵

- Figure B66: A sample error case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 196]

#### 健康与医学:公共卫生

Question: 如图<图片 1>，雌蚊和雄蚊的轮毛有什么区别？ Option: (A) 雌蚊的轮毛短而密，雄蚊的轮毛长而疏 (B) 雌蚊的轮毛长而疏，雄蚊的轮毛短而密 (C) 无区别 (D) 雌蚊的轮毛短而疏，雄蚊的轮毛长而密

|[Figure 197]|
|---|

<图片 1>

GPT-4V: 正确答案是：(A) 雌蚊的轮毛短而密，雄蚊的轮毛长而疏 从提供的图片中可以看出，雌蚊 （c）的轮毛相对较短且分布较密集，而雄蚊（d）的轮毛则较长且分布较疏。

Error Category: Perceptual Error Error Reason: GPT-4V made an error in image perception, incorrectly identifying the sparse setae on the left (image c) of the female mosquito as dense, and the dense setae on the right (image d) of the male mosquito as sparse.

Ground Truth: (D)雌蚊的轮毛短而疏，雄蚊的轮毛长而密

- Figure B67: A sample error case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 198]

#### 健康与医学:公共卫生

Question: 为评价一种家用“简便”妊娠试验药盒的效果，其试验结果与医院常规的妊娠诊断试验结 果如图<图片 1>，该药盒的敏感度是（） Option:

(A) 42.65% (B) 50.88% (C) 42.68% (D) 35%

|[Figure 199]|
|---|

<图片 1>

GPT-4V: 敏感度（Sensitivity）是指在所有真实为阳性的样本中，测试正确识别为阳性的比例。根 据所提供的图表，我们可以计算敏感度： 敏感度 = 真阳性 / (真阳性 + 假阴性) 从图表中，我们 可以看到真阳性（即家用试验药盒和医院诊断都为阳性）的数量是29，假阴性（即医院诊断为阳性但 家用试验药盒为阴性）的数量是39。 敏感度 = 29 / (29 + 39) 敏感度 = 29 / 68 敏感度 = 0.4265 或者 42.65% 所以正确答案是 (A) 42.65%。

Error Category: Perceptual Error Error Reason: GPT-4V made an error in image perception. While it correctly inferred the falsenegative scenario as "diagnosed positive in the hospital but negative in the home test kit," it misread the false-negative count as 39, when the correct count for false negatives is 28.

Ground Truth: (B)50.88%

- Figure B68: A sample error case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 200]

#### 健康与医学:公共卫生

Question: 如图<图片 1>，请选择正确的名称及其与对应寄生虫之间的关系（） Option: (A) 荸荠，肝吸虫的第一中间宿主 (B) 茭白，姜片虫的第一中间宿主 (C) 荸荠，姜片虫的传播媒介 (D) 茭白，姜片虫的传播媒介

|[Figure 201]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是荸荠。荸荠是一种水生植物的块茎，它并不是寄生虫的宿主。因此，选项 (A)和(C)都是不正确的。茭白是另一种水生植物，与荸荠不同，所以选项(B)和(D)中提到的茭白也是 不正确的。由于题目中的信息不足以确定荸荠与任何寄生虫之间的关系，正确答案应该是没有列出的 选项，即这些选项都不正确。

Error Category: Reasoning Error Error Reason: GPT-4V correctly identified the plant in the image as water chestnut. However, it made an incorrect inference by ruling out the correct option (C), as water chestnut is not a host for parasites.

Ground Truth: (C)荸荠，姜片虫的传播媒介

- Figure B69: A sample error case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 202]

#### 健康与医学:公共卫生

Question:为评价水痘疫苗的流行病学效果，其随访结果如图<图片 1>，该疫苗的效果指数是（） Option:

(A) 6 (B) 7 (C) 8 (D) 9

|[Figure 203]|
|---|

<图片 1>

GPT-4V: 为了计算疫苗的效果指数，我们需要使用以下公式来计算疫苗效力（VE，Vaccine Efficacy）： VE = (1 - RR) × 100% 其中RR是相对风险（Relative Risk），可以通过以下公式计 算： RR = (接种疫苗后发病率) / (未接种疫苗的发病率) 根据图中的数据，我们可以计算出接种疫 苗后和未接种疫苗的发病率： 接种疫苗后发病率 = 接种疫苗后发病人数 / 接种疫苗人数 = 10 / 400 = 0.025 未接种疫苗的发病率 = 未接种疫苗的发病人数 / 未接种疫苗人数 = 90 / 600 = 0.15 然后计算相对风险（RR）： RR = 0.025 / 0.15 = 1/6 ≈ 0.1667 最后计算疫苗效力（VE）： VE = (1 - 0.1667) × 100% ≈ 83.33% 在所给的选项中，并没有一个选项直接对应到83.33%，但是我们 可以选择最接近的选项。在这种情况下，正确答案是： (D) 9 因为83.33%最接近于选项(D) 9，如果 9代表的是疫苗效果指数的百分比形式，即90%。

Error Category: Textual Understanding Error Reason: GPT-4V misinterpreted the question, confusing "efficacy index" with "vaccine efficacy." The correct formula for the efficacy index is: Efficacy Index = Incidence Rate in Control Group / Incidence Rate in Vaccinated Group = 0.15 / 0.025 = 6.

Ground Truth: (A)6

- Figure B70: A sample error case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 204]

#### 健康与医学:公共卫生

Question: 如图<图片 1>，是1938～1952年某地区医院提供两性婴幼儿某病的发病率资料，X2=0.02， P＞0.10。 上述资料的结果的含意是（） Option: (A)婴幼儿的性别因素与某病的发病率有差别 (B) 婴幼儿的性别因素与某病发病率无统计学的差别 (C) 某病与婴幼儿性别有相关 (D) 性别是该病的病因因素

|[Figure 205]|
|---|

<图片 1>

GPT-4V: 根据所提供的信息，我们可以看到卡方检验的结果是 X2 =0.02，并且 P>0.10。这意味着在 统计学上，婴幼儿的性别因素与某病的发病率之间没有显著差异，因为P值大于0.10，通常认为P值小 于0.05时结果才具有统计学意义。 因此，正确答案是： (B)婴幼儿的性别因素与某病发病率无统计 学的差别 其他选项： (A)错误，因为统计结果表明没有显著差异。 (C)错误，因为没有统计学相关 性。(D)错误，因为性别不是该病的病因因素，至少在统计学上没有证据表明有相关性。

Error Category: Answer Extraction Error Error Reason: GPT-4V correctly understood the image and made an accurate judgment. However, in its subsequent analysis, it also examined each incorrect option, leading to errors in the Extract & Match process, resulting in the extraction of answers (A), (B), (C), and (D).

Ground Truth: (B)婴幼儿的性别因素与某病发病率无统计学的差别

- Figure B71: A sample error case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 206]

#### 健康与医学:公共卫生

Question:识别<图片 1>内容为（）虫卵 Option:

(A) 肺吸虫卵 (B) 血吸虫卵 (C) 肝吸虫卵 (D) 姜片虫卵

|[Figure 207]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是在显微镜下观察到的一个卵。由于我无法进行实际的实验室测试或分析， 我不能提供一个确切的答案。然而，根据形状和外观，这个卵可能是某种寄生虫的卵。在选项中，肺 吸虫、血吸虫和肝吸虫都是寄生虫，而姜片虫并不是一个常见的寄生虫名称，这可能是一个干扰项。 通常，肺吸虫卵、血吸虫卵和肝吸虫卵在显微镜下有特定的形状和大小，但是需要专业知识来准确识 别。 如果必须猜测，我会建议选择最可能正确的选项，但请注意，这不是一个基于实验室分析的确 切答案。因此，正确答案可能是： (A) 肺吸虫卵 (B) 血吸虫卵 (C) 肝吸虫卵 由于缺乏具体信息， 我无法确定具体是哪一个。如果这是一个实际的医学或科学问题，建议咨询专业人士进行准确的鉴定。

Error Category: Lack of Knowledge Error Reason: GPT-4V correctly understood the image and inferred that the answer lies within the approximate range of (A), (B), and (C) based on the image. However, due to a lack of domainspecific knowledge, it couldn't provide a specific answer.

Ground Truth: (A)肺吸虫卵

- Figure B72: A sample error case of Health and Medicine (subcategory: Public Health). Back to List of figures

[Figure 208]

#### 商业:会计

Question: 如果要绘制出<图片 1>中所示的逐渐淡出的笔触,需要在【画笔】控制面板中设置的一项 重要参数是( ）。

Option:

(A) 平衡持有量 (B) 最佳持有量 (C) 最低持有量 (D) 最高持有量

|[Figure 209]|
|---|

<图片 1>

GPT-4V: 这张图展示了一个典型的库存成本模型，其中包括固定成本、变动成本和总成本。图中的总 成本曲线在某一点达到最低值，这个点对应的现金水平是企业持有现金的最佳水平。因此，当两种成 本之和，也即总成本达到最小值时，企业所持有的现金水平为最佳持有量。

正确答案是： (B) 最佳持有量

- Ground Truth: (B) 最佳持有量

Figure B73: A sample correct case of Business (subcategory: Accounting). Back to List of figures

[Figure 210]

#### 商业:会计

Question: 某企业为了进行主要资产项目的比重分析,特根据资产负债表表制定了主要资产项目比重 及其差异表，如下：<图片 1>，由此可见非流动资产中固定资产比重__

Option: (A) 上升 (B) 降低 (C) 保持不变 (D) 无法判断

|[Figure 211]<br><br>[Figure 212]|
|---|

<图片 1>

GPT-4V: 根据提供的图片中的数据，我们可以看到“固定资产”这一行的比重从上一期的14.7%上升 到了本期的14.4%，差异为-0.3%。这表明固定资产的比重实际上是降低了。 因此，正确答案是： (B) 降低

Error Category: Perceptual Error Error Reason: GPT-4V incorrectly identified the line 'inventory' as fixed assets and recognized a 0.3% difference as a change in the proportion of fixed assets. However, in reality, the proportion of fixed assets changed by +8.2%. This error was due to a misperception by GPT-4V while reading the image.

Ground Truth: (A) 上升

Figure B74: A sample error case of Business (subcategory: Accounting). Back to List of figures

[Figure 213]

#### 商业:会计

Question: 分析资料:某企业生产丙产品,其产品单位成本简表如下表<图片 1>，用差额计算法分析单 位工时和小时工资率变动对单位直接人工的影响为__

Option:

(A) 材料成本增加 (B) 材料成本降低 (C) 人工成本降低 (D) 人工成本增加

|[Figure 214]|
|---|

<图片 1>

GPT-4V: 为了分析单位工时和小时工资率变动对单位直接人工的影响，我们需要查看表格中的相关数 据。 从表格中可以看到，直接人工的单位成本在标准情况下是20元，而在实际情况下是27元。这意 味着单位直接人工成本实际上增加了。

同时，我们可以看到标准工时为12小时，实际工时为11小时，单位工时成本从标准的10元降低到了实 际的10元。尽管工时成本没有变化，但是工时减少了，这意味着每个单位产品的工时效率提高了。 然而，尽管工时效率提高，直接人工的总成本还是增加了，这表明小时工资率的增加对成本的影响超 过了工时效率的提高。

因此，我们可以得出结论，小时工资率的增加导致了单位直接人工成本的增加。 正确答案是： (D) 人工成本增加

Error Category: Reasoning Error Error Reason: GPT-4V, in answering the question, only qualitatively estimated the direct labor unit cost and used this as a basis to conclude that labor costs increased. However, in reality, the increase in labor expenses for a company is mainly due to the rise in hourly wage rates, and there has been no decline in the company's production efficiency. Improved efficiency has actually reduced the labor cost per unit product by 2 yuan. Here, the model did not engage in quantitative reasoning, and therefore, it constitutes a Reasoning Error.

- Ground Truth: (C) 人工成本降低

Figure B75: A sample error case of Business (subcategory: Accounting). Back to List of figures

[Figure 215]

#### 商业: 经济

Question:根据<图片 1>，市场的均衡价格和数量是( ) Option: (A) 18美元和15单位 (B) 5美元和15单位 (C) 10美元和30单位 (D) 5美元和35单位

|[Figure 216]|
|---|

<图片 1>

GPT-4V:根据提供的图表，市场均衡价格和数量是供给和需求曲线的交点。从图中可以看出，这两条 曲线在价格为10美元和数量为30单位的点上相交。因此，市场的均衡价格是10美元，均衡数量是30单 位。

正确答案是：(C) 10美元和30单位。

Ground Truth: (C)10美元和30单位

Figure B76: A sample correct case of Business (subcategory: Economics). Back to List of figures

[Figure 217]

#### 商业: 经济

Question:如图<图片 1>洛伦兹曲线反映了收入分配的不平等程度，弯曲程度与收入分配不平等成（） 比。

Option: (A) 正 (B) 反 (C) 不相关 (D) 以上都对

|[Figure 218]|
|---|

<图片 1>

GPT-4V:洛伦兹曲线是一个用来表示收入或财富分配平等程度的图形。在这个曲线上，完全平等的分 配会被表示为一条从原点（0,0）开始到（100%, 100%）的直线，这条直线被称为“平等线”。洛伦 兹曲线越是弯曲，表示收入分配的不平等程度越高。因此，洛伦兹曲线的弯曲程度与收入分配不平等 成正比。

正确答案是：

- (A) 正

Ground Truth: (A)正

Figure B77: A sample correct case of Business (subcategory: Economics). Back to List of figures

[Figure 219]

#### 商业: 经济

Question:请<图片 1>是中国学者最早创造性倡导的__理论。 Option:

(A) 区域经济成长一阶段 (B) 区域经济成长二阶段 (C) 区域经济成长三阶段 (D) 区域经济成长四阶段

|[Figure 220]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一个关于区域经济成长的理论模型图，图中标注了四个阶段：I - 产业结 构高度集中，II - 产业结构分散，III - 产业结构再集中，IV - 产业结构高度分散。根据这个信息， 可以推断这是一个描述区域经济成长四个阶段的理论。因此，正确答案是：

(D) 区域经济成长四阶段

- Ground Truth: (D)区域经济成长四阶段

Figure B78: A sample correct case of Business (subcategory: Economics). Back to List of figures

[Figure 221]

#### 商业: 经济

Question:如图<图片 1>所示，厂商的理性决策应在（） Option:

(A) 3≤L≤8 (B) 3≤L≤5 (C) 0≤L≤5 (D) 5≤L≤8

|[Figure 222]|
|---|

<图片 1>

GPT-4V:这张图显示了平均产品（AP）和边际产品（MP）随劳动力（L）的变化情况。理性的厂商决 策通常是在边际产品（MP）开始下降之前或当它等于平均产品（AP）时，因为这代表了生产效率的最 大化。在图中，MP曲线在L=3时开始下降，并且在L=5时与AP曲线相交。因此，理性的决策应该是在MP 开始下降之前，即在L=3时，直到MP等于AP，即L=5时。所以正确的选项是：

- (B) 3≤L≤5

Error Category:Reject to Answer Error Reason:In economics, firms produce in the second stage, which in the diagram refers to L=5 to L=8. Therefore, the correct answer is D. The main reason GPT-4V got this wrong is due to a Lack of Knowledge.

Ground Truth: (D)5≤L≤8

Figure B79: A sample error case of Business (subcategory: Economics). Back to List of figures

[Figure 223]

#### 商业: 经济

Question:在<图片 1>所示的博弈中，纳什均衡是 ( ） Option:

(A) 上，左 (B) 上，右 (C) 下，左 (D) 下，右

|[Figure 224]|
|---|

<图片 1>

GPT-4V:在这个博弈中，我们可以通过检查每个玩家在对方策略给定的情况下的最佳响应来找到纳什 均衡。 首先，我们看玩家A（行玩家）的选择。如果玩家B选择“左”，玩家A会选择“下”（因为4大于1）。 如果玩家B选择“右”，玩家A同样会选择“下”（因为7大于3）。 接下来，我们看玩家B（列玩家）的选择。如果玩家A选择“上”，玩家B会选择“右”（因为2大于 0）。如果玩家A选择“下”，玩家B会选择“右”（因为4大于5）。 结合这两个玩家的最佳响应，我们可以看到纳什均衡发生在玩家A选择“下”，玩家B选择“右”的情 况下。因此，正确答案是： (D) (下，右)

Error Category:Reasoning Error Error Reason:When player A chooses "Down" and player B chooses "Right," GPT-4V considers this a Nash equilibrium. However, in this situation, if player B changes their choice to "Left," they will gain a higher payoff. Therefore, the combination of "Down, Right" is not a Nash equilibrium. GPT4V made a Reasoning Error in this case.

Ground Truth: (C)下，左

Figure B80: A sample error case of Business (subcategory: Economics). Back to List of figures

[Figure 225]

#### 商业: 经济

Question:考虑在双头垄断市场上存在两个企业生产同类产品。为了在竞争中占得先机，双方都考虑 是否采取降价的策略。他们决策的不同组合，导致收益矩阵如<图片 1>所示。假设两家企业同时决策， 那么该博弈的纳什均衡是_____。

Option: (A) 企业1降价，企业2不降价 (B) 企业1不降价，企业2降价 (C) 企业1降价，企业2降价 (D) 企业1不降价，企业2不降价

|[Figure 226]|
|---|

<图片 1>

GPT-4V:很在这个收益矩阵中，我们可以看到企业 1 和企业 2 的收益取决于他们是否选择降价。为 了找到纳什均衡，我们需要找到这样一种策略组合，即在给定对方策略的情况下，任何一方改变自己 的策略都不会使自己更好。

我们可以通过检查每个企业在对方选择不同策略时的最佳反应来确定纳什均衡：

- - 当企业 2 选择降价时，企业 1 的最佳反应是降价（18 > 4）。
- - 当企业 2 选择不降价时，企业 1 的最佳反应是不降价（23 > 16）。
- - 当企业 1 选择降价时，企业 2 的最佳反应是降价（18 > 4）。
- - 当企业 1 选择不降价时，企业 2 的最佳反应是不降价（20 > 16）。 因此，纳什均衡发生在两个企业都选择降价的情况下，因为在这种情况下，没有一方有激励单方面改 变策略。 正确答案是：

- (C) 企业 1 降价，企业 2 降价

Error Category:Reasoning Error Error Reason:GPT-4V has already listed the best responses for each firm when the other chooses different strategies, and knows that a Nash equilibrium occurs when no party has the incentive to unilaterally change their strategy. Up to this point, there are no issues. However, the error occurred when reasoning which situation belongs to "no party having the incentive to unilaterally change their strategy." When both firms 1 and 2 reduce prices, firm 2's change in strategy actually leads to a greater profit. Therefore, the Nash equilibrium does not occur in option A but rather in option C.

- Ground Truth: (A)企业1降价，企业2不降价

Figure B81: A sample error case of Business (subcategory: Economics). Back to List of figures

[Figure 227]

#### 商业: 经济

Question:如图<图片 1>微笑曲线中，处于中间环节的制造附加值___。

Option: (A) 最高 (B) 最低 (C) 较高 (D) 较低

|[Figure 228]|
|---|

<图片 1>

GPT-4V:根据所提供的图片，微笑曲线展示了不同环节的附加值变化。在这个曲线中，制造环节位于 中间的低谷位置，这表示制造环节的附加值相对较低。因此，正确答案是：

- (D) 较低

Error Category:Textual Understanding Error Reason:GPT-4V correctly understood the image and identified the key point that 'the manufacturing link is located in the middle valley position', but it failed to correctly differentiate between 'lowest' and 'relatively low' in the question options. It mistook the globally lowest position for being relatively low, which led to GPT-4V answering this question incorrectly.

- Ground Truth: (B)最低

Figure B82: A sample error case of Business (subcategory: Economics). Back to List of figures

[Figure 229]

#### 商业: 经济

Question:图<图片 1>所示弧弹性为_ Option:

(A) 无限弹性 (B) 富有弹性 (C) 单位弹性 (D) 缺乏弹性

|[Figure 230]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一个完全无弹性的需求曲线，因为无论数量(Q)如何变化，价格(P)保持不 变。因此，正确答案是： (D) 缺乏弹性

Error Category:Lack of Knowledge Error Reason:Lack of elasticity refers to the sensitivity of the response to price changes. When the price changes, the amount of change in demand is always zero, which is infinite elasticity. Therefore, the incorrect answer by GPT-4V is due to the lack of this knowledge point.

- Ground Truth: (A)无限弹性

Figure B83: A sample error case of Business (subcategory: Economics). Back to List of figures

[Figure 231]

#### 商业: 经济

Question:假设某消费者的均衡如<图片 1>所示。已知商品1的价格P1=2元。则消费者的收入为（）。 Option:

(A) 50 (B) 40 (C) 70 (D) 60

|[Figure 232]|
|---|

<图片 1>

GPT-4V:这张图片显示了一个消费者的均衡点在两种商品（X1 和 X2）的消费组合上。均衡点在预算 线上，预算线的方程是 I=P1X1+P2X2，其中 I 是收入，P1 是商品1的价格，P2 是商品2的价格，X1 和 X2 分别是商品1和商品2的数量。

根据图中的信息，我们可以看到消费者选择了15单位的商品1（X1=15）。已知商品1的价格P1是2元， 我们可以计算出消费者在商品1上的支出是 2元/单位 * 15单位 = 30元。

然而，我们没有商品2的价格P2，因此不能直接计算出消费者的总收入。但是，我们可以观察到预算 线在X2轴上的截距。这个截距代表了如果消费者将所有收入都用于购买商品2时能够购买的最大数量。 从图中可以看出，这个截距是20单位的X2。

如果我们假设消费者将所有收入都用于购买商品2，那么我们可以用总收入除以商品2的数量来得到商 品2的价格P2。但是，由于我们没有商品2的价格，我们无法使用这种方法来确定确切的收入。

不过，我们可以使用图中的信息来估计收入。由于消费者在商品1上花费了30元，如果他们在商品2上 也花费了30元，那么总收入将是60元。但是，如果商品2的价格低于2元，那么消费者在商品2上的花 费将少于30元，总收入将低于60元。同样，如果商品2的价格高于2元，总收入将高于60元。

由于没有足够的信息来确定商品2的价格，我们无法准确计算出总收入。因此，我们无法从提供的选 项中确定一个明确的答案。在这种情况下，我们可以选择一个可能的答案，但请注意这只是一个基于 可用信息的猜测。

基于上述分析，正确答案是无法确定，因为我们没有商品2的价格信息。

Error Category:Lack of Knowledge Error Reason:GPT-4V made some deductions in the early stage, but the final conclusion failed to yield the correct answer. This is due to GPT-4V's lack of specific problem-solving knowledge. Therefore, the reason for GPT-4V's error here is 'Lack of Knowledge'.

- Ground Truth: (C)70

Figure B84: A sample error case of Business (subcategory: Economics). Back to List of figures

[Figure 233]

#### 商业: 金融

Question:某股票收益分布的可能情况如下表<图片 1>，试计算该股票的年预期收益率( )

Option: (A) 0.20 (B) 0.24 (C) 0.21 (D) 0.18

|[Figure 234]|
|---|

<图片 1>

GPT-4V: 为了计算股票的年预期收益率，我们需要将每种可能情况的收益率与其相应的概率相乘， 然后将这些乘积相加。根据提供的表格，我们可以进行以下计算： 预期收益率 = (A的收益率 * A的 概率) + (B的收益率 * B的概率) + (C的收益率 * C的概率) 预期收益率 = (0.4 * 0.25) + (0.4 * 0.20) + (0.2 * 0.15) 预期收益率 = (0.1) + (0.08) + (0.03) 预期收益率 = 0.21 正确答案：(C) 0.21。

Ground Truth: (C) 0.21

Figure B85: A sample correct case of Business (subcategory: Finance). Back to List of figures

[Figure 235]

#### 商业: 金融

Question:根据久期变化图<图片 1>可以看出收益率同等变化幅度下，债券价格增加的幅度要超过债 权减少的幅度，该债券价格波动符合__特征

Option:

(A) 反凸性 (B) 正凸性 (C) 极大值 (D) 极小值

|[Figure 236]|
|---|

<图片 1>

GPT-4V: 这张图显示了债券价格与收益率之间的关系。从图中可以看出，当收益率下降时（即从y* 向左移动到y1），债券价格P上升的幅度比收益率上升时（即从y*向右移动到y2）债券价格下降的幅 度要大。这种价格变动的非对称性表现出债券价格对于利率变动的正凸性（convexity）特征。 正确答案： (B) 正凸性

- Ground Truth: (B) 正凸性

Figure B86: A sample correct case of Business (subcategory: Finance). Back to List of figures

[Figure 237]

#### 商业: 金融

Question:<图片 1>在债券收益率继续上升，债券价格下降时，黄色虚线部分代表的含权债券价格降 幅小于不含权债券，并且回售价格就是下降的最底线，形成了一个__保护

Option:

(A) 地板价格 (B) 平均价格 (C) 最高价格 (D) 以上皆非

|[Figure 238]|
|---|

<图片 1>

GPT-4V: 根据所提供的图片和描述，黄色虚线代表的含权债券（Putable bond）价格降幅小于不含 权债券（Non putable bond），并且回售价格（Put price）就是下降的最底线，这形成了一个价格 保护。在金融术语中，这种价格保护通常被称为“地板价格”（Floor price），因为它为债券价格 提供了一个最低限度。 正确答案： (A) 地板价格

Ground Truth: (A) 地板价格

- Figure B87: A sample correct case of Business (subcategory: Finance). Back to List of figures

|商业: 金融|
|---|

[Figure 239]

|Question: 已知某年度非金融企业部门的实物投资为11849．8亿元，当年该部门的可支配收入为 6324．1亿元。请根据下表<图片 1>给出的非金融企业部门的资金流量数据，证券市场对促进企业融 资的作用非常有限，只占_%。<br><br>GPT-4V:为了计算证券市场在非金融企业部门融资中的比例，我们需要查看表格中与证券市场相关的 融资数据。从提供的图片中，我们可以看到“从证券市场融资”这一项的数据。然而，图片中并没有 直接显示这一数据，因此我们无法直接计算出证券市场在非金融企业部门融资中的具体比例。 由于 缺乏具体的证券市场融资数据，我们无法提供一个确切的答案。如果有更多信息或数据提供，我们可 以进行计算。在目前情况下，我们只能说证券市场对非金融企业部门融资的作用的确切比例未知。<br><br>Ground Truth: 0.48<br><br>Error Category: Reject to Answer Error Reason:GPT-4V believes that the image does not directly display the data on "financing from the securities market," making it impossible to directly calculate the specific proportion of financing from the securities market in the non-financial corporate sector. In reality, the "net borrowing from the securities market" in the image is the key information needed for an answer. GPT-4V considers that the lack of specific data prevents it from providing a response.<br><br><图片 1><br><br>|[Figure 240]|
|---|
|
|---|

- Figure B88: A sample error case of Business (subcategory: Finance). Back to List of figures

|商业: 金融|
|---|

[Figure 241]

|Question:某厂生产的三种产品的有关资料如图<图片 1>，产量变化使总成本减少___%，减少的绝对 量为___元。<br><br>Option:<br><br>(A) 1.67；680<br>(B) 1.67；820<br>(C) 2.51；680<br>(D) 2.51；820<br><br><br>GPT-4V:为了解决这个问题，我们需要计算总成本的变化。首先，我们来计算原始的总成本和变化后 的总成本。 原始总成本 = 单位成本 * 产量 变化后总成本 = (单位成本 + 单位成本变动) * (产量 + 产量变动) 产品A的原始总成本 = 110元/件 * 600件 = 66000元 产品A的变化后总成本 = (110 + 10)元/件 * (600 + 120)件 = 120 * 720 = 86400元 产品B的原始总成本 = 120元/件 * 570件 = 68400元 产品B的变化后总成本 = (120 + 60)元/件 * (570 + 180)件 = 180 * 750 = 135000元 产 品C的原始总成本 = 14元/件 * 50件 = 700元 产品C的变化后总成本 = (14 + 1)元/件 * (50 + 9) 件 = 15 * 59 = 885元 原始的总成本 = 66000 + 68400 + 700 = 135100元 变化后的总成本 = 86400 + 135000 + 885 = 224285元 总成本的减少量 = 原始总成本 - 变化后总成本 总成本的减少 量 = 135100 - 224285 = -89185元 因为总成本实际上增加了，所以没有减少的绝对量。这意味着所 有选项都不正确，因为它们都假设了成本的减少。然而，如果问题是关于成本的增加，我们可以计算 增加的百分比： 增加的百分比 = (总成本的增加量 / 原始总成本) * 100% 增加的百分比 = (89185 / 135100) * 100% ≈ 66.01% 但这不是题目要求的，所以根据题目所给的信息和计算结果，没有一 个选项是正确的。<br><br>Ground Truth: (D) 2.51；820<br><br>Error Category: Perceptual Error Error Reason:GPT-4V provides the formula for calculating the total cost; however, it incorrectly interprets the information in the chart. Taking Product A as an example: it mistakenly considers the base period output as the unit cost and attributes the output of Product B as its own, resulting in a misalignment of calculation data and consequently leading to errors in the computation.<br><br>|<图片 1><br><br>[Figure 242]|
|---|
<br><br><图片 1>|
|---|

- Figure B89: A sample error case of Business (subcategory: Finance). Back to List of figures

|商业: 金融|
|---|

[Figure 243]

|Question: 某股票的收益分布的可能情况如下表<图片 1>，试计算该股票的年预期收益率为（ ）。 Option:<br><br>(A) 0.13<br>(B) 0.14<br>(C) 0.15<br>(D) 0.16<br><br><br>GPT-4V:为了计算该股票的年预期收益率，我们需要将每种可能情况的收益率乘以其相应的概率，然 后将这些乘积相加。根据提供的表格，我们可以进行以下计算： 预期收益率 = (概率A * 收益率A) + (概率B * 收益率B) + (概率C * 收益率C) 预期收益率 = (0.6 * 0.15) + (0.3 * 0.20) + (0.1 * 0.10) 预期收益率 = (0.09) + (0.06) + (0.01) 预期收益率 = 0.16 因此，该股票的年预期收益 率为0.16 正确答案： (D) 0.16<br><br>Ground Truth: (C) 0.15<br><br>Error Category: Annotation Error Error Reason:In this case, GPT-4V accurately grasped the meaning of the question and provided a rigorous formulation and calculation. The computed results were completely correct. However, due to an issue with annotating the standard answer, the correct response should be D 0.16, leading to an Annotation Error.<br><br><图片 1><br><br>|[Figure 244]|
|---|
|
|---|

- Figure B90: A sample error case of Business (subcategory: Finance). Back to List of figures

|商业: 金融|
|---|

[Figure 245]

|Question: 某人投资了三种股票，这三种股票的方差一协方差矩阵如下表<图片 1>，矩阵第(i，j)位 置上的元素为股票i与j的协方差，已知此人投资这三种股票的比例分别为0.3，0.3，0.4，则该股票 投资组合的风险是( )。<br><br>Option:<br><br>(A) 8.1<br>(B) 5.1<br>(C) 6.2<br>(D) 9.2<br><br><br>GPT-4V:为了计算股票投资组合的风险，我们需要使用投资组合的方差公式。对于三种股票的投资组 合，其方差可以表示为：𝜎𝜎𝑝𝑝2 = 𝑤𝑤𝐴𝐴2 ⋅ 𝜎𝜎𝐴𝐴2 + 𝑤𝑤𝐵𝐵2 ⋅ 𝜎𝜎𝐵𝐵2 + 𝑤𝑤𝐶𝐶2 ⋅ 𝜎𝜎𝐶𝐶2 + 2 ⋅ 𝑤𝑤𝐴𝐴 ⋅ 𝑤𝑤𝐵𝐵 ⋅ 𝐶𝐶𝐶𝐶𝐶𝐶 𝐴𝐴,𝐵𝐵 + 2 ⋅ 𝑤𝑤𝐴𝐴 ⋅ 𝑤𝑤𝐶𝐶 ⋅ 𝐶𝐶𝐶𝐶𝐶𝐶 𝐴𝐴,𝐶𝐶 + 2 ⋅ 𝑤𝑤𝐵𝐵 ⋅ 𝑤𝑤𝐶𝐶 ⋅ 𝐶𝐶𝐶𝐶𝐶𝐶 𝐵𝐵,𝐶𝐶 其中， 𝑤𝑤𝐴𝐴,𝑤𝑤𝐵𝐵 和 𝑤𝑤𝐶𝐶 是各股票的投资比例， 𝜎𝜎𝐴𝐴2,𝜎𝜎𝐵𝐵2 和 𝜎𝜎𝐶𝐶2<br><br>是各自的方差，而 𝐶𝐶𝐶𝐶𝐶𝐶 𝐴𝐴,𝐵𝐵 ,𝐶𝐶𝐶𝐶𝐶𝐶 𝐴𝐴,𝐶𝐶 和 𝐶𝐶𝐶𝐶𝐶𝐶 𝐵𝐵,𝐶𝐶 。根据题目，我们有： 𝑤𝑤𝐴𝐴 = 0.3,𝑤𝑤𝐵𝐵 = 0.3,𝑤𝑤𝐶𝐶 = 0.4 ,𝜎𝜎𝐴𝐴2 = 25,𝜎𝜎𝐵𝐵2 = 36,𝜎𝜎𝐶𝐶2 = 49 ,𝐶𝐶𝐶𝐶𝐶𝐶 𝐴𝐴,𝐵𝐵 = 30, 𝐶𝐶𝐶𝐶𝐶𝐶 𝐴𝐴,𝐶𝐶 = 35,𝐶𝐶𝐶𝐶𝐶𝐶 𝐵𝐵,𝐶𝐶 = 42。 将 这 些 值 代<br><br>入上述公式中，我们得到：𝜎𝜎𝑝𝑝2 = 0.3 2 ⋅ 25 + 0.3 2 ⋅ 36 + 0.4 2 ⋅ 49 + 2 ⋅ 0.3 ⋅ 0.3 ⋅ 30 + 2 ⋅ 0.3 ⋅ 0.4 ⋅ 35 + 2 ⋅ 0.3 ⋅ 0.4 ⋅ 42 计 算 这 个 表 达 式 ： 𝜎𝜎𝑝𝑝2 = 0.09 ⋅ 25 + 0.09 ⋅ 36 + 0.16 ⋅ 49 + 0.18 ⋅ 30 + 0.24 ⋅ 35 + 0.24 ⋅ 42 ，𝜎𝜎𝑝𝑝2 = 2.25 + 3.24 + 7.84 + 5.4 + 8.4 + 10.08，𝜎𝜎𝑝𝑝2<br><br>= 37.21所以，该股票投资组合的风险（方差）是 37.21。然而，这个结果并不在所给的选项中。可 能是计算过程中出现了错误，或者题目中的选项有误。根据所给的选项，没有一个与计算结果相符， 因此无法确定正确答案。如果必须选择一个最接近的选项，那么选项 (D) 9.2 是最接近计算结果的 平方根（投资组合风险的标准差），但这并不是方差的正确答案。"D",<br><br>Ground Truth: (C) 6.2<br><br>Error Category: Reasoning Error Error Reason:GPT-4V correctly presented the variance formula for an investment portfolio during the response: 𝜎𝜎𝑝𝑝2 = 𝑤𝑤𝐴𝐴2 ⋅ 𝜎𝜎𝐴𝐴2 + 𝑤𝑤𝐵𝐵2 ⋅ 𝜎𝜎𝐵𝐵2 + 𝑤𝑤𝐶𝐶2 ⋅ 𝜎𝜎𝐶𝐶2 + 2 ⋅ 𝑤𝑤𝐴𝐴 ⋅ 𝑤𝑤𝐵𝐵 ⋅ 𝐶𝐶o𝐶𝐶 𝐴𝐴,𝐵𝐵 + 2 ⋅ 𝑤𝑤𝐴𝐴 ⋅ 𝑤𝑤𝐶𝐶 ⋅ 𝐶𝐶𝐶𝐶𝐶 𝐴𝐴,𝐶𝐶<br><br>+ 2 ⋅ 𝑤𝑤𝐵𝐵 ⋅ 𝑤𝑤𝐶𝐶 ⋅ 𝐶𝐶𝐶𝐶𝐶 𝐵𝐵,𝐶𝐶<br><br>.Furthermore, it successfully utilized this formula to compute the portfolio's variance. However, during subsequent reasoning, GPT-4V did not accurately calculate the standard deviation. Instead, it opted for an answer closest to the variance, resulting in a reasoning error.<br><br><图片 1><br><br>|[Figure 246]|
|---|
|
|---|

- Figure B91: A sample error case of Business (subcategory: Finance). Back to List of figures

|商业: 金融|
|---|

[Figure 247]

|Question: A、B 和 C 三个项目，已知各期现金流情况如<图片 1>。如用回收期法来选择项目目标回 收期为3 年，选择___项目<br><br>Option:<br><br>(A) A和B<br>(B) A和C<br>(C) B和C<br>(D) A和B和C<br><br><br>GPT-4V:根据图片中提供的现金流信息，我们可以计算每个项目的回收期。 项目A的现金流为：第一 年+1000元，第二年+1000元，第三年+3000元。累计现金流为第一年1000元，第二年2000元，第三年 5000元。因此，项目A在第三年回收了初始投资，满足3年的目标回收期。 项目B的现金流为：第一年 0元，第二年+1000元，第三年+2000元。累计现金流为第一年0元，第二年1000元，第三年3000元。因 此，项目B在第三年回收了初始投资，满足3年的目标回收期。 项目C的现金流为：第一年+1000元， 第二年+2000元，第三年+3000元。累计现金流为第一年1000元，第二年3000元，第三年6000元。因此， 项目C在第三年也回收了初始投资，满足3年的目标回收期。 根据回收期法，所有三个项目A、B和C都 在3年内回收了初始投资， 正确答案：(D) A和B和C<br><br>Ground Truth: (B) A和C<br><br>Error Category: Reasoning Error Error Reason:In this case, GPT-4V accurately calculated the situations for projects A and C. However, when calculating project B, it made an incorrect inference. With a cash flow of 3000 in the third year, which is less than 5000, project B does not meet the requirement for recovering the initial investment. The correct answer should be A and C.<br><br><图片 1><br><br>|[Figure 248]|
|---|
|
|---|

- Figure B92: A sample error case of Business (subcategory: Finance). Back to List of figures

|商业: 金融|
|---|

[Figure 249]

|Question: 你可以投资如<图片 1>两个完全负股票，目标收益率为16%，那么这个投资组合的方差为 （ ）<br><br>Option:<br><br>(A) 0.0225<br>(B) 0.0324<br>(C) 0.0357<br>(D) 0.0431<br><br><br>GPT-4V:为了计算两个股票组合的方差，我们需要知道每个股票的预期收益率、各自的标准差以及它 们之间的相关系数。然而，从提供的信息中，我们没有股票之间的相关系数，这是计算组合方差的关 键要素之一。 不过，我们可以尝试使用给定的信息来估计组合的方差。首先，我们需要确定组合中 每个股票的权重，这样才能达到目标收益率。假设股票A的权重为w，股票B的权重为1-w，我们可以建 立以下方程来解决w： w ∗ 12% + 1 − w ∗ 20% = 16% 解这个方程，我们得到：0.12w + 0.20 −<br><br>0.20w = 0.16 −0.08w = −0.04 w = 0.5 这意味着股票A和股票B在组合中的权重都是50%。然而，由于 我们没有股票之间的相关系数，我们无法准确计算组合的方差。如果我们假设两个股票之间的相关系 数为0（即它们是完全不相关的），那么组合的方差可以使用以下公式计算： 组合方差 = 𝑤𝑤2 ∗<br><br>方差𝐴𝐴 + 1 − 𝑤𝑤 2 ∗ 方差𝐵𝐵 其中方差A是股票A的标准差的平方，方差B是股票B的标准差的平方。将 给定的标准差转换为方差： 方差𝐴𝐴 = 15% 2 = 0.0225 方差𝐵𝐵 = 45% 2 = 0.2025 现在我们可以 计算组合的方差： 组合方差 = 0.52 ∗ 0.0225 + 0.52 ∗ 0.2025 组合方差 = 0.25 ∗ 0.0225 + 0.25 ∗<br><br>0.2025 组合方差 = 0.005625 + 0.050625 组合方差 = 0.05625 这个结果并不在提供的选项中，这可能 是因为我们没有考虑股票之间的相关系数。在实际情况中，如果两个股票不是完全不相关的，那么组 合的方差将会受到股票之间相关系数的影响，从而导致不同的结果。 由于我们无法确定正确答案， 我们不能选择任何一个选项作为正确答案。在现实世界的投资决策中，投资者需要考虑所有相关的风 险因素，包括资产之间的相关性。<br><br>正确答案：AB<br><br>Ground Truth: (A) 0.0225<br><br>|[Figure 250]|
|---|
<br><br>Error Category: Textual Understanding Error Error Reason:GPT-4V incorrectly interpreted the question prompt. The prompt explicitly stated that stocks A and B are two completely negatively correlated stocks, providing information about their correlation coefficient. However, GPT-4V overlooked this crucial information, leading to other errors in subsequent inference calculations and resulting in multiple answers, including AB.<br><br><图片 1>|
|---|

- Figure B93: A sample error case of Business (subcategory: Finance). Back to List of figures

|商业: 金融|
|---|

[Figure 251]

|Question: <图片 1>资产i的必要报酬率Ri与β的函数形成的直线称为__（证券/资本）市场线。<br><br>GPT-4V:资产i的必要报酬率Ri与β的函数形成的直线称为资本市场线。 正确答案：资本市场线。<br><br>Ground Truth: 证券<br><br>Error Category: Lack of Knowledge Error Reason: The response provided by GPT-4V is inaccurate. In the field of finance, the line formed by the relationship between the required rate of return (Ri) for asset i and its beta (β) is typically referred to as the Security Market Line (SML), not the Capital Market Line. GPT-4V's error stems from a lack of knowledge in the relevant domain.<br><br>|<图片 1><br><br>[Figure 252]|
|---|
<br><br><图片 1>|
|---|

- Figure B94: A sample error case of Business (subcategory: Finance). Back to List of figures

[Figure 253]

#### 商业: 管理

Question: 市场上A、B、C、D四支证券的期望收益率和标准差如<图片 1>，面对四支证券，理性的投 资者该如何选择__

Option: (A) A (B) B (C) C (D) D

|[Figure 254]|
|---|

<图片 1>

GPT-4V: 在选择证券时，理性的投资者通常会考虑收益率和风险（标准差）。理性的投资者会寻找风 险和收益的最佳组合，即寻找风险调整后收益最高的证券。 从提供的信息来看：

- - A证券的期望收益率为10%，标准差为12%。
- - B证券的期望收益率为18%，标准差为20%。
- - C证券的期望收益率为8%，标准差为24%。
- - D证券的期望收益率为20%，标准差为10%。 比较这些证券，D证券有最高的期望收益率，同时它的 标准差是最低的，这意味着它提供了最高的收益与最低的风险。因此，根据这些数据，理性的投资者 最有可能选择D证券。 正确答案： (D) D

- Ground Truth: (D) D

Figure B95: A sample correct case of Business (subcategory: Management). Back to List of figures

[Figure 255]

#### 商业: 管理

Question: 在如<图片 1>所示行业，格兰仕占了一半以上的市场份额，财源滚滚而入。 根据波士顿咨 询集团分析法，如<图片 1>所示是格兰仕的（）。

Option: (A) 问号类战略业务单位 (B) 明星类战略业务单位 (C) 现金牛类战略业务单位 (D) 狗类战略业务单位

|[Figure 256]|
|---|

<图片 1>

GPT-4V: 图片显示了一个格兰仕品牌的微波炉。根据题目描述，格兰仕在这个行业中占有一半以上的 市场份额，并且财源滚滚而入。根据波士顿咨询集团（BCG）矩阵的分析法，这样的业务单位具有高 市场份额并且产生了大量现金流，通常被分类为“现金牛”（Cash Cow）。 正确答案： (C) 现金牛类战略业务单位

- Ground Truth: (C) 现金牛类战略业务单位

Figure B96: A sample correct case of Business (subcategory: Management). Back to List of figures

[Figure 257]

#### 商业: 管理

Question: 某服装生产企业实施差异化战略，向消费者提供与众不同的产品，获得竞争优势。该企业 为了降低原材料采购成本，进入纺织行业，自主生产和供应服装加工所需面料。该企业以许可经营的 形式积极拓展海外市场，允许国外企业使用该企业的专利、商标、设计款式，扩大企业的国际声誉。 同时，该企业积极进行新产品开发，不断推出新款服装。拟推出的新款服装共有甲产品、乙产品、丙 产品、丁产品四种开发方案可供选择；每种产品方案均存在着市场需求高、市场需求一般、市场需求 低三种可能的市场状态，但各种状态发生的概率难以测算。在市场调查的基础上，该服装生产企业对 四种备选方案的损益值进行了预测，在不同市场状态下损益值如下表<图片 1>所示（单位：百万元）。 若采用后悔值原则进行新款服装的决策，该企业应选择的方案为开发（ ）

Option:

- (A) 丁产品
- (B) 乙产品
- (C) 甲产品
- (D) 丙产品

|[Figure 258]|
|---|

<图片 1>

GPT-4V: 后悔值原则，又称为最小最大后悔原则，是一种决策准则，用于在不确定性条件下做出决策。 它考虑的是如果不选择最佳方案会产生的损失（后悔值），并试图最小化这种潜在的损失。 为了应 用这个原则，我们需要计算每种市场状态下不选择最佳方案的后悔值。首先，我们找出每种市场状态 下的最大损益值，然后计算其他方案与这个最大值之间的差距，即后悔值。 根据提供的表格，我们 可以计算出每种市场状态下的最大损益值： - 市场需求高时，最大损益值为280（丙产品）。 - 市 场需求一般时，最大损益值为150（丁产品）。 - 市场需求低时，最大损益值为30（乙产品）。 现 在，我们计算每个方案的后悔值： - 甲产品的后悔值为：市场需求高时（280-270=10），市场需求 一般时（150-110=40），市场需求低时（30-10=20）。 - 乙产品的后悔值为：市场需求高时（280265=15），市场需求一般时（150-100=50），市场需求低时（30-30=0）。 - 丙产品的后悔值为：市 场需求高时（280-280=0），市场需求一般时（150-140=10），市场需求低时（30-(-10)=40）。 丁产品的后悔值为：市场需求高时（280-250=30），市场需求一般时（150-150=0），市场需求低时 （30-20=10）。 接下来，我们找出每个方案的最大后悔值： - 甲产品的最大后悔值为40。 - 乙产 品的最大后悔值为50。 - 丙产品的最大后悔值为40。 - 丁产品的最大后悔值为30。 根据后悔值原 则，我们应该选择最大后悔值最小的方案，即丁产品。 正确答案： (A) 丁产品。

- Ground Truth: (A) 丁产品

- Figure B97: A sample correct case of Business (subcategory: Management). Back to List of figures

|商业: 管理|
|---|

[Figure 259]

|Question: 某企业拟购置一套大型设备，甲、乙、丙三个供应商的报价相同，设备性能也一样，只是 使用过程中需要的维修费不同，预计情况如<图片 1>，根据以上数据，企业应购买哪个(些)供应商的 设备?（）<br><br>option:<br><br>(A) 甲或丙<br>(B) 乙<br>(C) 甲<br>(D) 丙<br><br><br>GPT-4V:根据提供的图片信息，我们可以看到三个供应商在不同年份的设备维修费用和发生故障的概 率。为了确定哪个供应商的设备最经济，我们需要计算每个供应商设备的预期维修成本。 预期维修 成本可以通过将每年的维修费用乘以相应年份发生故障的概率来计算，然后将所有年份的结果相加得 到总的预期维修成本。 甲供应商的预期维修成本计算如下： - 第一年：40万元 × 40% = 16万元 第二年：10万元 × 60% = 6万元 - 第三年：30万元 × 40% = 12万元 - 第四年：20万元 × 60% = 12万元 - 第五年：50万元 × 30% = 15万元 - 第六年：10万元 × 70% = 7万元 总计：16 + 6 + 12 + 12 + 15 + 7 = 68万元 乙供应商的预期维修成本计算如下： - 第一年：40万元 × 10% = 4万 元 - 第二年：10万元 × 30% = 3万元 - 第三年：30万元 × 20% = 6万元 - 第四年：20万元 × 50% = 10万元 - 第五年：50万元 × 10% = 5万元 - 第六年：10万元 × 60% = 6万元 总计：4 + 3 + 6 + 10 + 5 + 6 = 34万元 丙供应商的预期维修成本计算如下： - 第一年：40万元 × 20% = 8万 元 - 第二年：10万元 × 50% = 5万元 - 第三年：30万元 × 10% = 3万元 - 第四年：20万元 × 40% = 8万元 - 第五年：50万元 × 20% = 10万元 - 第六年：10万元 × 30% = 3万元 总计：8 + 5 + 3 + 8 + 10 + 3 = 37万元 比较三个供应商的总预期维修成本，乙供应商的总预期维修成本最低， 为34万元。因此，企业应该选择乙供应商的设备。 正确答案： (B) 乙<br><br>Ground Truth: (A) 甲或丙<br><br>Error Category: Perceptual Error Error Reason:GPT-4V misunderstood the meaning of the data in the image. The data in the second row should represent annual maintenance costs, and the data in the third row corresponds to the respective probabilities. Each column represents a different supplier, and it does not involve changes between different years. However, GPT mistakenly interpreted the data in different columns as changes over different years<br><br>|<图片 1><br><br>[Figure 260]|
|---|
<br><br><图片 1>|
|---|

- Figure B98: A sample error case of Business (subcategory: Management). Back to List of figures

|商业: 管理|
|---|

[Figure 261]

|Question: 除冰剂是冬季北方城市用于去除道路冰雪的常见产品。下表<图片 1>显示了五种除冰剂的 各项特征：以下哪项对上述五种除冰剂的特征概括最为准确?（）<br><br>Option:<br><br>(A) 融冰速度较慢的除冰剂在污染土壤和污染水体方面的风险都低<br>(B) 没有一种融冰速度快的除冰剂三个方面的风险都高<br>(C) 若某种除冰剂至少在两个方面风险低，则其融冰速度一定较慢<br>(D) 若某种除冰剂三方面风险都不高，则其融冰速度一定也不快<br><br><br>GPT-4V:根据提供的表格信息，我们可以对每个选项进行分析： (A) 融冰速度较慢的除冰剂在污染 土壤和污染水体方面的风险都低 这个选项不准确。从表格中可以看出，融冰速度较慢的除冰剂（III 和V）在污染土壤方面的风险都是低的，但在污染水体方面，III的风险是低的，而V的风险是中等的。 (B) 没有一种融冰速度快的除冰剂三个方面的风险都高 这个选项是准确的。从表格中可以看出，融 冰速度快的除冰剂（I和IV）没有一种在三个方面（对植物的危害、污染土壤、污染水体）的风险都 是高的。 (C) 若某种除冰剂至少在两个方面风险低，则其融冰速度一定较慢 这个选项不准确。从表 格中可以看出，除冰剂II在两个方面风险低（对植物的危害和污染水体），但其融冰速度是中等的。 (D) 若某种除冰剂三方面风险都不高，则其融冰速度一定也不快 这个选项不准确。从表格中可以看 出，除冰剂II在三个方面的风险都不是高的（对植物的危害是低的，污染土壤和污染水体是中等的）， 但其融冰速度是中等的，不是不快。 正确答案： (B) 没有一种融冰速度快的除冰剂三个方面的风险都高。<br><br>Ground Truth: (C) 若某种除冰剂至少在两个方面风险低，则其融冰速度一定较慢<br><br>Error Category: Reasoning Error Error Reason:In this case, there were some issues with the reasoning of GPT-4V. For instance, in option B, GPT-4V incorrectly concluded that de-icing agents with fast ice-melting rates (I and IV) do not pose high risks in all three aspects. This contradicts the actual situation where option I has high risks in all three aspects. Similarly, errors in reasoning were observed in the analysis of the remaining options, leading to incorrect judgments.<br><br><图片 1><br><br>|[Figure 262]|
|---|
|
|---|

- Figure B99: A sample error case of Business (subcategory: Management). Back to List of figures

|商业: 管理|
|---|

[Figure 263]

|Question: 某公司计划对某一项目进行投资，投资额为300万元，期限为5年，每年净现金流量分别为 150万元、200万元、200万元、100万元、200万元。假设资本成本率为10％。该项目的净现金流量及 复利现值系数如下表<图片 1>所示。如果不考虑资金时间价值，该项目的投资回收期为（）<br><br>option:<br><br>(A) 1.75年<br>(B) 2年<br>(C) 2.75年<br>(D) 2.83年<br><br><br>GPT-4V:题目要求不考虑资金的时间价值，因此我们只需要计算何时净现金流量累计达到投资额300 万元即可。 第一年后，净现金流量为150万元； 第二年后，累计净现金流量为150 + 200 = 350万元。 在第二年结束时，累计净现金流量已经超过了投资额300万元。因此，投资回收期小于2年。由于第一 年结束时还没有回收完， 正确答案： (B) 2年<br><br>Ground Truth: (A) 1.75年<br><br>Error Category: Reasoning Error Error Reason:during the reasoning process, GPT-4V failed to perform precise calculations for the task and instead obtained a vague result of 'investment payback period less than 2 years' through estimation. This led to a reasoning error. The correct calculation should be: 1 + (300 - 150) / 200 = 1.75 years.<br><br><图片 1><br><br>|[Figure 264]|
|---|
|
|---|

- Figure B100: A sample error case of Business (subcategory: Management). Back to List of figures

|商业: 管理|
|---|

[Figure 265]

|Question: 有A 、B 、C、D四个零件，均先在第一台设备上加工，再到第二台设备上加工，各个零件 在这两台设备上的加工时间如下表<图片 1>：按约翰逊法则，四个零件最优的加工顺序应为（） option:<br><br>(A) B-A-C-D<br>(B) C-D-A-B<br>(C) B-C-D-A<br>(D) D-A-C-B<br><br><br>GPT-4V:约翰逊法则是一种用于确定在两台设备上进行作业的零件加工顺序的方法，目的是最小化总 的加工时间。根据约翰逊法则，应该首先比较各个零件在两台设备上的最短加工时间，并按照这个时 间从小到大排序。如果最短时间在第一台设备上，那么该零件应该尽可能早地进行加工；如果最短时 间在第二台设备上，那么该零件应该尽可能晚地进行加工。 根据提供的图片信息，我们可以看到： - 零件A在第一台设备上的加工时间为9，在第二台设备上的加工时间为10。 - 零件B在第一台设备上 的加工时间为3，在第二台设备上的加工时间为2。 - 零件C在第一台设备上的加工时间为6，在第二 台设备上的加工时间为4。 - 零件D在第一台设备上的加工时间为8，在第二台设备上的加工时间为9。 按照约翰逊法则，我们应该首先找出最短的加工时间，然后按照上述规则进行排序。 最短的加工时 间是零件B在第二台设备上的2小时，所以B应该放在最前面。接下来是零件C在第二台设备上的4小时， C应该放在B之后。然后是零件A在第一台设备上的9小时和零件D在第一台设备上的8小时，因为D的时 间更短，所以D应该放在C之后。最后是零件A。 因此，最优的加工顺序应为 B-C-D-A。 正确答案： (C) B-C-D-A<br><br>Ground Truth: (D) D-A-C-B<br><br>Error Category: Textual Understanding Error Reason:GPT-4V misunderstood the meaning of the task. During the reasoning process, it overlooked the prerequisite that the processing should first occur on the first device before moving to the second device. This oversight led to errors in the reasoning process<br><br><图片 1><br><br>|[Figure 266]|
|---|
|
|---|

- Figure B101: A sample error case of Business (subcategory: Management). Back to List of figures

|商业: 管理|
|---|

[Figure 267]

|Question: 已知有两家互不相关的公司，其有关资料如下<图片 1>，计算出B公司的息税前资产利润率 为_%<br><br>GPT-4V:息税前资产利润率（EBIT Asset Return Rate）是一个财务指标，用来衡量公司在扣除利息 和税项前，通过其资产赚取利润的能力。计算公式为： 息税前利润（EBIT）/ 总资产 = 息税前资产 利润率 根据图片中提供的B公司的数据，我们可以计算出B公司的息税前资产利润率： EBIT（息税前 利润）= 36000 总资产 = 8000 息税前资产利润率 = (EBIT / 总资产) * 100% 息税前资产利润率 = (36000 / 8000) * 100% 息税前资产利润率 = 4.5 * 100% 息税前资产利润率 = 450% 因此，B公司 的息税前资产利润率为450%。<br><br>Ground Truth: 22.86<br><br>Error Category: Lack of Knowledge Error Reason:GPT-4V mistakenly provided an incorrect formula for calculating the Earnings Before Interest and Taxes (EBIT) to Total Assets ratio. The correct formula should be Earnings Before Interest and Taxes (EBIT) divided by the average total assets employed. Additionally, GPT-4V exhibited deviation in processing image data, inaccurately reading the values of Earnings Before Interest and Taxes (EBIT) and total assets<br><br><图片 1><br><br>|[Figure 268]|
|---|
|
|---|

- Figure B102: A sample error case of Business (subcategory: Management). Back to List of figures

|商业: 管理|
|---|

[Figure 269]

|Question: 振宁公司现有A、B、C三个投资中心，2006年各投资中心的有关资料如表所示<图片 1>：计 算A中心的剩余收益为_<br><br>GPT-4V:为了计算A中心的剩余收益，我们需要使用以下公式： 剩余收益 = 营业收入 - (资本成本 率 × 资本投资额) 根据图片中提供的数据，A中心的营业收入为270万元，资本成本率为18%，资本 投资额为150万元。将这些数值代入公式中计算： 剩余收益 = 270 - (18% × 150) 剩余收益 = 270 - 27 剩余收益 = 243万元 因此，A中心的剩余收益为243万元。<br><br>Ground Truth: 4.2万元<br><br>Error Category: Lack of Knowledge Error Reason:GPT-4V provided an incorrect calculation method for residual income, stating that it is equal to profit minus (capital cost rate × capital investment). In the case of Center A, with a profit of 285,000, an average operating asset of 1,350,000, and a capital cost rate of 18%, the correct result should be 285,000 - (1,350,000 × 18%) = 42,000.<br><br><图片 1><br><br>|[Figure 270]|
|---|
|
|---|

- Figure B103: A sample error case of Business (subcategory: Management). Back to List of figures

|商业: 管理|
|---|

[Figure 271]

|Question: 某企业某部门运用一一对比法对所属的4名员工进行绩效考核，考核情况如下表所<图片 1> 示:由此可知，绩效最差的员工是（）<br><br>option:<br><br>(A) 张XX<br>(B) 王XX<br>(C) 李XX<br>(D) 赵XX<br><br><br>GPT-4V:根据提供的表格，我们可以看到每个员工的绩效考核结果是通过与其他员工的一对一对比来 确定的。表格中的“+”表示胜过对方，“-”表示不如对方，“0”表示自己，不进行比较。 从表格 中我们可以看出： - 张XX胜过李XX和赵XX，但输给了王XX。 - 王XX胜过所有其他三人。 - 李XX输 给了所有其他三人。 - 赵XX胜过李XX，但输给了张XX和王XX。 因此，绩效最差的员工是李XX，因为 他在与其他所有员工的对比中都是负面的。 正确答案：(C) 李XX。<br><br>Ground Truth: (D) 赵XX<br><br>Error Category: Lack of Knowledge Error Reason:GPT-4V lacks relevant knowledge about the 'one-to-one comparison method,' leading to incorrect processing during comparative ranking. It mistakenly interprets higher comparisons as lower ones, resulting in erroneous inferences<br><br>|<图片 1><br><br>[Figure 272]|
|---|
<br><br><图片 1>|
|---|

- Figure B104: A sample error case of Business (subcategory: Management). Back to List of figures

[Figure 273]

#### 商业: 营销

Question: 营销环境不以某个营销组织或个人的意志为转移,它有自己的运行规律和特点。企业的营销 活动只能主动地适应和利用客观环境,不能改变或违背。客观、公正地分析营销环境因素才能减少营 销决策的盲目和失误,赢得营销活动的成功。由此可知该图<图片 1>为__

Option:

(A) 交易市场营销环境 (B) 交易市场广告环境 (C) 企业市场广告环境 (D) 企业市场营销环境

|[Figure 274]|
|---|

<图片 1>

GPT-4V: 这张图片展示的是一个营销环境分析模型，中心写着“企业”，周围分为不同的部分，包括 “政治法律”、“社会文化”、“中间商”、“供应商”、“顾客”、“竞争对手”、“公众”和“ 技术环境”。这个模型用于帮助分析和理解一个企业在进行营销活动时需要考虑的外部环境因素。 正确答案： (D) 企业市场营销环境

- Ground Truth: (D) 企业市场营销环境

Figure B105: A sample correct case of Business (subcategory: Marketing). Back to List of figures

[Figure 275]

#### 商业: 营销

Question:某公司需要对某新产品生产批量做出决策,各种批量在不不同的自然状态下的收益情况如 下表(收益矩阵)<图片 1>,用悲观决策方法,对上述生产经营问题做出决策方案，选择________方案。

Option:

(A) S1 (B) S2 (C) S3 (D) S2和S3

|[Figure 276]|
|---|

<图片 1>

GPT-4V: 悲观决策方法，又称为Maximin准则，是一种在不确定条件下的决策方法。这种方法考虑的 是每个策略在最坏情况下的最大收益。具体操作是先找出每个策略下最坏情况的收益，然后从中选择 收益最大的策略。 根据提供的收益矩阵，我们可以看到： -对于策略S1，最坏情况下的收益是-6。 -对于策略S2，最坏情况下的收益是-2。 -对于策略S3，最坏情况下的收益是5。 在这三个策略中， S3在最坏情况下的收益最高，为5。因此，根据悲观决策方法，我们应该选择S3。 正确答案： (C) S3

Ground Truth: (C) S3

Figure B106: A sample correct case of Business (subcategory: Marketing). Back to List of figures

[Figure 277]

#### 商业: 营销

Question:现有四种证券资料如下:<图片 1>无风险报酬率为8%,市场上所有证券的组合报酬率为14%。 要求:计算上述四种证券各自的必要报酬率 𝑅𝑅 𝐴𝐴 = ______;𝑅𝑅 𝐵𝐵 = ______;𝑅𝑅 𝐶𝐶 =______; 𝑅𝑅 𝐷𝐷 =______;

Option:

- (A) 10.4%；14%；17%；23%
- (B) 17%；14%；10.4%；23%
- (C) 14%；17%；23%；10.4%
- (D) 17%；23%；14%；10.4%

|[Figure 278]|
|---|

<图片 1>

GPT-4V:为了计算每种证券的必要报酬率，我们可以使用资本资产定价模型（Capital Asset Pricing Model, CAPM），该模型的公式为： 𝑅𝑅𝑖𝑖 = 𝑅𝑅𝑓𝑓 + 𝛽𝛽𝑖𝑖 𝑅𝑅𝑚𝑚 − 𝑅𝑅𝑓𝑓 其中： 𝑅𝑅𝑖𝑖 是证券 i 的必要报 酬率 𝑅𝑅𝑓𝑓是无风险报酬率 𝛽𝛽𝑖𝑖 是证券 i 的风险系数（Beta）, 𝑅𝑅𝑚𝑚 是市场组合的报酬率 根据题目， 我们知道 𝑅𝑅𝑓𝑓 = 8% 和 𝑅𝑅𝑚𝑚 = 14% 。现在我们可以使用每种证券的 Beta 值来计算它们的必要报酬 率。 对于证券 A： 𝑅𝑅𝐴𝐴 = 8% + 1.5 × 14% − 8% = 8% + 1.5 × 6% = 8% + 9% = 17% 对 于证券 B：𝑅𝑅𝐵𝐵 = 8% + 1.0 × 14% − 8% = 8% + 6% = 14%.对于证券 C：𝑅𝑅𝐶𝐶 = 8% + 0.4

× 14% − 8% = 8% + 0.4 × 6% = 8% + 2.4% = 10.4% 对于证券 D： 𝑅𝑅𝐷𝐷 = 8% + 2.5 × 14% − 8% = 8% + 2.5 × 6% = 8% + 15% = 23%。 因此，每种证券的必要报酬率分别为：

𝑅𝑅𝐴𝐴 = 17%，𝑅𝑅𝐵𝐵 = 14% ，𝑅𝑅𝐶𝐶 = 10.4% ，𝑅𝑅𝐷𝐷 = 23% 正确答案是 (B) 17%；14%；10.4%；23%。

- Ground Truth: (B) 17%；14%；10.4%；23%

- Figure B107: A sample correct case of Business (subcategory: Marketing). Back to List of figures

|商业: 营销|
|---|

[Figure 279]

|Question: 某高校在校本科生40000人,分优、良、中、差4个层次,其他资料如表所示<图片 1>。当 n=400 人时,试按分层最佳比例抽样法确定优、良、中、差各层抽样数_____________。<br><br>Option:<br><br>(A) 45 、116 、176 、63<br>(B) 35 、136 、156 、73<br>(C) 55 、116 、166 、83<br>(D) 25 、116 、176 、83<br><br><br>GPT-4V:为了确定每个层次的抽样数，我们需要使用分层抽样的公式。分层抽样的目的是确保每个层 次在样本中的代表性与其在总体中的比例相符。在分层最佳比例抽样法中，每个层次的抽样数是根据 层内总数(Ni)和层内标准差(Si)的乘积的比例来确定的。 首先，我们需要计算每个层次的NiSi值， 然后计算所有层次NiSi值的总和。之后，每个层次的抽样数可以通过以下公式计算：抽样数 𝑛𝑛𝑛𝑛<br><br>= 𝑁𝑁𝑁𝑁𝑁𝑁𝑁𝑁 总𝑁𝑁𝑁𝑁𝑁𝑁𝑁𝑁<br><br>∗ 总抽样数 𝑛𝑛 根据图片中的表格，我们可以计算出每个层次的NiSi值，然后用上述公式 计算出每个层次的抽样数。 优层的𝑁𝑁𝑛𝑛𝑆𝑆𝑛𝑛 = 6000 ∗ 4 = 24000 良层的𝑁𝑁𝑛𝑛𝑆𝑆𝑛𝑛 = 14000 ∗ 5<br><br>= 70000 中层的𝑁𝑁𝑛𝑛𝑆𝑆𝑛𝑛 = 16000 ∗ 5 = 80000 差层的𝑁𝑁𝑛𝑛𝑆𝑆𝑛𝑛 = 4000 ∗ 6 = 24000 总𝑁𝑁𝑛𝑛𝑆𝑆𝑛𝑛 = 24000<br><br>+ 70000 + 80000 + 24000 = 198000 现在我们可以计算每个层次的抽样数： 优层的抽样数<br><br>= (24000 / 198000) ∗ 400 ≈ 48.48 ≈ 48（四舍五入） 良层的抽样数 = (70000 / 198000) * 400 ≈ 141.41 ≈ 141（四舍五入） 中层的抽样数 = (80000 / 198000) * 400 ≈ 161.62 ≈ 162（四 舍五入） 差层的抽样数 = (24000 / 198000) * 400 ≈ 48.48 ≈ 48（四舍五入） 总抽样数 = 48 + 141 + 162 + 48 = 399 由于总抽样数应该是400，我们需要对某个层次的抽样数进行调整，以确保 总数为400。通常，我们会增加最大层次的抽样数，因此中层的抽样数应该是163。 最终的抽样数为 ： 优层：48 良层：141 中层：163 差层：48 这些数值并不完全匹配所给的选项，但是根据计算过 程，我们可以选择最接近的选项。在这种情况下，没有一个选项完全匹配我们的计算结果，但是选项 (A) 45、116、176、63是最接近的，尽管它与我们的计算结果有所差异。如果必须选择一个选项，我 们将选择(A)作为最接近的答案。然而，需要注意的是，这个答案并不精确，可能是由于四舍五入或 其他计算差异导致的。<br><br>Ground Truth: (D) 25 、116 、176 、83<br><br>Error Category: Perceptual Error Error Reason:GPT-4V has no issues in understanding and inferring the textual content of the question. However, it encountered difficulties in interpreting images, mistakenly treating the 𝐶𝐶𝑁𝑁 as Si. This resulted in problems throughout the subsequent calculation process, ultimately leading to an erroneous inference.<br><br><图片 1><br><br>|[Figure 280]|
|---|
|
|---|

- Figure B108: A sample error case of Business (subcategory: Marketing). Back to List of figures

|商业: 营销|
|---|

[Figure 281]

|Question: 某企业按五年计划规定,某产品最后一年产量应达到200万吨,各年实际生产情况如表所示:< 图片 1>计算该产品五年计划完成程度______和提前完成五年计划指标的时间_______。<br><br>Option:<br><br>(A) 92.3%；延后一个季度<br>(B) 97.4%；延后一个季度<br>(C) 107.5%；提前一个季度<br>(D) 112.6%；提前一个季度<br><br><br>GPT-4V:为了回答这个问题，我们需要先计算五年总产量，然后将其与计划产量进行比较，以确定完 成程度。同时，我们需要查看实际产量达到计划产量200万吨的时间点。 根据提供的图片，我们可以 看到各年的产量如下： - 第一年：160万吨 - 第二年：165万吨 - 第三年：165万吨 - 第四年：40 万吨 + 45万吨 = 85万吨 - 第五年：45万吨 + 50万吨 + 55万吨 + 60万吨 = 210万吨 五年总产量 = 160 + 165 + 165 + 85 + 210 = 785万吨 计划总产量 = 200万吨 * 5年 = 1000万吨 完成程度 = (实际产量 / 计划产量) * 100% = (785 / 1000) * 100% = 78.5% 然而，这个计算与所给选项不匹 配，这可能是因为图片中的数据没有被正确解读。请确保图片中的数据是准确的，并且没有遗漏任何 信息。 现在，我们需要确定提前完成计划的时间。根据图片中的数据，我们可以看到在第五年的每 个季度产量都有所增加。我们需要计算在哪个季度时，累计产量首次达到或超过200万吨。 累计产量 达到200万吨的时间点： - 第一年末：160万吨 - 第二年末：160 + 165 = 325万吨 - 第三年末： 325 + 165 = 490万吨 - 第四年末：490 + 85 = 575万吨 - 第五年第一季度末：575 + 45 = 620万 吨 - 第五年第二季度末：620 + 50 = 670万吨 - 第五年第三季度末：670 + 55 = 725万吨 - 第五 年第四季度末：725 + 60 = 785万吨 由于在第五年的每个季度结束时，累计产量都没有达到计划的 200万吨，因此我们无法确定提前完成计划的具体时间。这意味着我们无法根据所给的选项来确定正 确答案。 由于所给的选项与计算结果不符，我们可能需要更多信息或者重新检查题目和图片中的数 据。如果图片中的数据是准确的，那么根据上述计算，没有一个选项是正确的。<br><br>Ground Truth: (C) 107.5%；提前一个季度<br><br>Error Category: Textual Understanding Error Reason:GPT-4V incorrectly interpreted the text information in the question. The content of the five-year plan should be "the final year's production of a certain product should reach 2 million tons." However, GPT-4V mistakenly assumed that the annual production for each of the five years should be 2 million tons. Therefore, the correct approach is to consider only the production situation in the last year when providing an answer.<br><br><图片 1><br><br>|[Figure 282]|
|---|
|
|---|

- Figure B109: A sample error case of Business (subcategory: Marketing). Back to List of figures

|商业: 营销|
|---|

[Figure 283]

|Question: 从一行业中随机抽取5家企业，所得产品产量与生产费用的数据如下：<图片 1>利用最小二 乘法求出估计的回归方程____________。<br><br>Option:<br><br>(A) 𝑦𝑦  = 124.248 + 0.567𝑥𝑥<br>(B) 𝑦𝑦  = 124.248 + 0.732𝑥𝑥<br>(C) 𝑦𝑦  = 111.314 + 0.732𝑥𝑥<br>(D) 𝑦𝑦  = 111.314 + 0.567𝑥𝑥<br><br><br>GPT-4V:很抱歉，我无法直接计算出最小二乘法的回归方程，因为这需要进行数学计算，而我目前无 法执行这样的操作。不过，我可以告诉你如何手动计算它。 最小二乘法的回归方程通常采用以下形 式： 𝑦𝑦  = 𝑎𝑎 + 𝑏𝑏𝑥𝑥其中， y是因变量的估计值，𝑥𝑥 是自变量，𝑏𝑏 是斜率，𝑎𝑎是截距。斜率 𝑏𝑏可以通过以下<br><br>公式计算： 𝑏𝑏 = 𝑛𝑛(𝑛𝑛∑𝑥𝑥𝑦𝑦)−(∑𝑥𝑥2 −∑∑𝑥𝑥)(𝑥𝑥∑2𝑦𝑦),截距𝑎𝑎可以通过以下公式计算：𝑎𝑎 = ∑𝑦𝑦−𝑏𝑏(𝑛𝑛 ∑𝑥𝑥) 其中，𝑛𝑛 是观测值的 数量。 你需要使用上述公式和提供的数据来计算𝑎𝑎和𝑏𝑏的值，然后将它们代入回归方程中。如果你能 提供计算结果，我可以帮你验证哪个选项是正确的。<br><br>Ground Truth: (D) 𝒚𝒚 = 111.314 + 𝟎𝟎.𝟓𝟓𝟓𝟓𝟓𝟓𝒙𝒙<br><br>Error Category: Lack of Knowledge Error Reason:GPT-4V provided the formula for calculating the regression equation in response to the request to use the least squares method for the regression equation. However, it seems that GPT-4V lacks the direct capability to compute the regression equation using the least squares method, and therefore, it refused to provide an answer.<br><br><图片 1><br><br>|[Figure 284]|
|---|
|
|---|

- Figure B110: A sample error case of Business (subcategory: Marketing). Back to List of figures

|商业: 营销|
|---|

[Figure 285]

|Question:已知观察期数据资料如表所示<图片1>,建立一元线性回归方程模型，计算相关系数 r=______，标准误差𝑆𝑆y_______=。<br><br>Option:<br><br>(A) 0.987543；6<br>(B) 0.994343；8<br>(C) 0.987543；8<br>(D) 0.994343；6<br><br><br>GPT-4V:很抱歉，我无法直接计算相关系数r和标准误差𝑆𝑆y。这需要进行一些统计计算，包括求和、 平方和、乘积和等，然后使用这些值来计算相关系数和标准误差。如果你能提供这些计算的结果或者 更多的上下文信息，我可以帮助你理解这些概念和它们是如何计算的。<br><br>Ground Truth:(B) 0.994343；8<br><br>Error Category: Lack of Knowledge Error Reason:GPT-4V cannot directly calculate the correlation coefficient and standard error . This requires some statistical calculations, including summation, sum of squares, product sum, and so on.<br><br><图片 1><br><br>|[Figure 286]|
|---|
|
|---|

- Figure B111: A sample error case of Business (subcategory: Marketing). Back to List of figures

|商业: 营销|
|---|

[Figure 287]

|Question: 某家用电器社会购买力(十万元)与该市家庭人均货币收入(元)的资料如表所示<图片 1>， 建立一元线性回归方程模型，试预测该市1994年的购买力区间________元。（α=0.05 ）<br><br>Option:<br><br>(A) 458.36~485.06<br>(B) 466.54~485.06<br>(C) 458.36~493.19<br>(D) 466.54~493.19<br><br><br>GPT-4V:很抱歉，我无法直接计算或建立一元线性回归方程模型，因为这需要专门的统计软件或工具<br><br>来进行计算。不过，我可以告诉你一般的步骤： 1. 首先，你需要使用家庭人均货币收入作为自变量 （X），家用电器社会购买力作为因变量（Y），并收集这些数据点。 2. 然后，你可以使用最小二乘 法来计算线性回归方程的参数，即斜率（b）和截距（a）。 3. 一旦有了线性回归方程，你可以使用 它来预测1994年的家用电器社会购买力。 4. 最后，你需要计算预测值的置信区间，这通常涉及到查 找t分布表和计算标准误差。 由于我无法执行这些计算，我建议你使用统计软件或求助于有统计学背 景的专业人士来完成这个任务。如果你有进一步的数据或需要帮助理解这个过程，请告诉我。<br><br>Ground Truth: (B) 466.54~485.06<br><br>Error Category: Lack of Knowledge Error Reason:GPT-4V provides a detailed explanation of the specific steps to establish a simple linear regression model, and there are no errors in the information. However, GPT-4V cannot directly compute or establish a simple linear regression model as it requires specialized statistical software or tools for the calculations.<br><br><图片 1><br><br>|[Figure 288]|
|---|
|
|---|

- Figure B112: A sample error case of Business (subcategory: Marketing). Back to List of figures

|商业: 营销|
|---|

[Figure 289]

|Question: 七奇公司在成长过程中拟追加筹资4000万元,现有A、B、C三个追加筹资方案可供选择,有关 资料经整理列示如下表<图片 1>:测算三个公司的边际资本成本率，A方案_______；B方案_________ ；C方案_______。<br><br>Option:<br><br>(A) 10.5%；10.51%；10.75%<br>(B) 10.75%；10.5%；10.51%<br>(C) 10.5%；10.75%；10.51%<br>(D) 10.51%；10.5%；10.75%<br><br><br>GPT-4V:为了计算边际资本成本率，我们需要根据提供的资料计算每个方案的加权平均资本成本（ WACC）。边际资本成本率是指最后一笔资金的成本，通常是指在现有资本结构下，公司为了筹集额外 资金而必须支付的资本成本。根据图片中的表格，我们可以计算出每个方案的边际资本成本率。计算 方法是找出每个方案中最后一笔资金的成本，即最高的资本成本百分比。方案A的最后一笔资金成本 是14%，方案B的最后一笔资金成本也是14%，而方案C的最后一笔资金成本是14%。 因此，正确答案是 ： (A) 10.5%；10.51%；10.75% 但是，我们注意到这个答案与我们的计算结果不符。所有方案的最 后一笔资金成本都是14%，所以我们需要重新审视选项。由于没有一个选项与我们的计算结果完全匹 配，我们需要选择最接近的选项。 根据我们的计算，所有方案的边际资本成本率都应该是14%，但是 这个选项并没有在提供的选项中。因此，我们无法根据提供的信息和选项来确定一个明确的答案。在 这种情况下，我们可以选择一个包含14%的选项，即使它不完全正确。最接近的选项是：(D) 10.51% ；10.5%；10.75% 请注意，这个答案并不准确，但根据我们的计算和提供的选项，这是最接近的选择 。<br><br>Ground Truth: (C) 10.5%；10.75%；10.51%<br><br>Error Category: Lack of Knowledge Error Reason:GPT-4V lacks relevant knowledge in calculating the marginal cost of capital and mistakenly interprets it as "finding the cost of the last capital in each scenario, i.e., the highest percentage of capital cost." The correct approach should involve calculating the proportion of each fundraising amount relative to the total fundraising amount and using these proportions as weights to compute the marginal cost of capital<br><br>|<图片 1><br><br>[Figure 290]|
|---|
<br><br><图片 1>|
|---|

- Figure B113: A sample error case of Business (subcategory: Marketing). Back to List of figures

|商业: 营销|
|---|

[Figure 291]

|Question: 某行业的市场是完全竞争性的,现在市场上有100家成本相同的企业,所有企业的固定成本和 可变成本如下,成本单位(元):<图片 1>现在的市场均衡价格是9元，每个企业生产_____，市场供给量 是______。<br><br>Option:<br><br>(A) 3；300<br>(B) 4；400<br>(C) 5；500<br>(D) 6；600<br><br><br>GPT-4V:在完全竞争市场中，企业会在边际成本等于市场价格的情况下生产，因为这样可以最大化利 润。根据提供的成本数据表，我们需要找到边际成本（MC）等于市场均衡价格9元的产量水平。 从图 表中，我们可以看到： - 当产量为1时，边际成本是5元。 - 当产量为2时，边际成本是7元。 - 当 产量为3时，边际成本是10元。 - 当产量为4时，边际成本是19元。 由于市场均衡价格是9元，我们 需要找到边际成本刚好等于或者刚好超过9元的产量。在这个情况下，边际成本在产量为2时是7元， 在产量为3时是10元。因此，每个企业会选择产量为2的情况，因为这是边际成本最接近但不超过市场 价格的点。 既然每个企业都会生产2单位的产品，而市场上有100家企业，市场供给量将是每个企业 的产量乘以企业数量，即2 * 100 = 200。 这个选择题中没有提供“200”的选项，所以可能是题目 有误或者选项有误。根据提供的信息和数据，最合理的答案是每个企业生产2单位，市场供给量是200 单位，但这不符合任何给定的选项。<br><br>Ground Truth: (B) 4；400<br><br>Error Category: Lack of Knowledge Error Reason:GPT-4V lacks relevant knowledge in calculating marginal costs and mistakenly interprets variable costs from the chart as marginal costs. However, the true meaning of marginal cost should be the change in total cost for a one-unit change in production. Therefore, when producing 4 units, the marginal cost would be 19-10=9, at which point the marginal cost equals the market equilibrium price.<br><br><图片 1><br><br>|[Figure 292]|
|---|
|
|---|

- Figure B114: A sample error case of Business (subcategory: Marketing). Back to List of figures

[Figure 293]

#### 科学:生物

Question: <图片 1>图示为被子植物的生活史，A处为（）

Option: (A) 合子 (B) 配子 (C) 精子 (D) 孢子体

|[Figure 294]|
|---|

<图片 1>

GPT-4V: 这张图片展示的是被子植物的生活史，其中A处标识的应该是生活史中的一个特定阶段。根 据被子植物的生活史，A处位于受精过程之后，因此A处表示的是合子（zygote），这是由精子和卵子 结合后形成的单个细胞。所以正确答案是：(A) 合子

- Ground Truth: (A) 合子

Figure B115: A sample correct case of Science (subcategory: Biology). Back to List of figures

[Figure 295]

#### 科学:生物

Question: 如图<图片 1>可看出盐胁迫对根伸长的影响。盐胁迫通过影响细胞周期减少根分生组织的 细胞数量来抑制根的伸长。无盐环境下的幼苗转移到两种介质中,一种含盐,另一种不含盐。在无盐环 境下,根分生组织的细胞数量不变,根伸长程度的趋势不变。而在含盐的环境中__

Option:

(A) 幼苗根分生组织的细胞数量增加，根伸长受到促进。 (B) 幼苗根分生组织的细胞数量减少，根伸长受到促进。 (C) 幼苗根分生组织的细胞数量增加，根伸长受到抑制。 (D) 幼苗根分生组织的细胞数量减少，根伸长受到抑制。

|[Figure 296]|
|---|

<图片 1>

GPT-4V: 根据题目描述和所提供的图片信息，我们可以得出以下结论：题目描述了盐胁迫对根伸长的

影响，指出盐胁迫通过影响细胞周期减少根分生组织的细胞数量来抑制根的伸长。在无盐环境下，根 分生组织的细胞数量不变，根伸长程度的趋势不变。而在含盐的环境中，可以推断根分生组织的细胞 数量会减少，根伸长受到抑制。因此，正确答案是：(D) 幼苗根分生组织的细胞数量减少，根伸长受 到抑制。

Ground Truth: (D) 幼苗根分生组织的细胞数量减少，根伸长受到抑制。

Figure B116: A sample correct case of Science (subcategory: Biology). Back to List of figures

[Figure 297]

#### 科学:生物

Question: <图片 1>是根的发育模式中的哪一阶段？（） Option: (A) 初生分生组织 (B) 初生结构 (C) 形成层的发生 (D) 形成层环的形成

|[Figure 298]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是植物根的横截面图。根据图中显示的结构，可以看到中心柱、初生木质部、 初生韧皮部、皮层和表皮等部分。这些都是根的初生结构的组成部分。因此，正确答案是：(B) 初生 结构

- Ground Truth: (B) 初生结构

Figure B117: A sample correct case of Science (subcategory: Biology). Back to List of figures

[Figure 299]

#### 科学:生物

Question: <图片 1>是噬菌体侵染细菌的示意图,侵染细菌的正确顺序应是（） Option:

(A) BDAEC (B) ABCDE (C) BDEAC (D) DBAEC

|[Figure 300]|
|---|

<图片 1>

GPT-4V: 这张图片展示了噬菌体侵染细菌的不同阶段。根据噬菌体的生命周期，正确的顺序应该是：

- 噬菌体附着到细菌细胞壁（A）- 注入遗传物质（B）- 细胞内合成噬菌体组分（C）- 组装成熟噬 菌体（D）- 细胞裂解释放噬菌体（E）因此，正确的顺序是 ABCDE。正确答案是：(B) ABCDE

Error Category: Perceptual Error Error Reason: GPT-4V has knowledge about the process of bacteriophage infection in bacteria, understanding that the sequence of infection comprises the bacteriophage attaching itself to the bacterial cell wall, injecting genetic material, synthesizing bacteriophage components within the cell, assembling mature bacteriophages, and cell lysis to release the phages. However, it fails to accurately identify images depicting each stage, leading to incorrect matches between the images and their corresponding names.

- Ground Truth: (A) BDAEC

Figure B118: A sample error case of Science (subcategory: Biology). Back to List of figures

[Figure 301]

#### 科学:生物

Question: 如图<图片 1>为CDK-cyclin复合体在细胞周期中的功能。__和__都是被特定的CDK-cyclin 复合体调控的。这些复合体在细胞周同期的适当时间点合成。

Option: (A) G1期细胞生长；S期DNA复制 (B) M期纺锤体形成；G1期细胞生长 (C) S期DNA复制；M期纺锤体形成 (D) G2期准备分裂；G1期细胞生长

|<image 1><br><br>[Figure 302]|
|---|

<图片 1>

GPT-4V:根据提供的图片，我们可以看到CDK-cyclin复合体在细胞周期的不同阶段起作用。图片中标

注了G1期、S期、G2期和M期，并且在G1期和S期旁边有CDK-cyclin复合体的标记。因此，我们可以推 断CDK-cyclin复合体在G1期和S期有特定的功能。G1期通常与细胞生长有关，而S期与DNA复制有关。 因此，正确答案是：(A) G1期细胞生长；S期DNA复制

Error Category: Reasoning Error Error Reason: GPT-4V accurately identified the stages in the cell cycle depicted in the image as G1, S, G2, and M phases. It also detected the CDK-cyclin marker adjacent to the S phase. It correctly inferred that CDK-cyclin was involved in DNA replication during the S phase, yet incorrectly presumed that CDK-cyclin also plays a role in cell growth during the G1 phase. Moreover, it overlooked the CDK-cyclin adjacent to the M phase. These misjudgments led to the incorrect answer.

- Ground Truth: (C) S期DNA复制；M期纺锤体形成

Figure B119: A sample error case of Science (subcategory: Biology). Back to List of figures

[Figure 303]

#### 科学:生物

Question: <图片 1>图为进行性肌营养不良遗传病的系谱图，该病为隐性伴性遗传病，7号的致病基 因是由（）号传下来的

Option:

(A) 1 (B) 2 (C) 3 (D) 4

|[Figure 304]|
|---|

<图片 1>

GPT-4V: 这是一个关于隐性伴性遗传病的系谱图。在隐性伴性遗传中，女性需要两个隐性基因才会表 现出疾病，而男性只需要一个隐性基因就会表现出疾病，因为男性是XY性染色体，女性是XX性染色体。 在这个系谱图中，7号是一个表现疾病的男性，所以他的X染色体上带有致病基因。他只能从他的母亲 那里继承X染色体，因为他从父亲那里继承的是Y染色体。因此，7号的致病基因是由他的母亲（3号） 传下来的。 正确答案是： (C) 3

Error Category: Reasoning Error Error Reason: GPT-4V accurately identified Individual 7 as a diseased male and correctly deduced that his pathogenic gene was inherited from his mother. However, it misplaced the identification number of his mother. She should be labeled as Individual 5, not Individual 3.

- Ground Truth: (B) 2

Figure B120: A sample error case of Science (subcategory: Biology). Back to List of figures

[Figure 305]

#### 科学:生物

Question: <图片 1>所示的是生物信息学中常使用的__

|[Figure 306]|
|---|

<图片 1>

GPT-4V: <图片 1>所示的是生物信息学中常使用的聚类分析结果图。

Error Category: Answer Extraction Error Error Reason: GPT-4V correctly answer the question, but the function to extract the answer has the limitation, which led to the wrong answer being extracted.

Ground Truth: 聚类分析

Figure B121: A sample error case of Science (subcategory: Biology). Back to List of figures

[Figure 307]

#### 科学:生物

Question: 淀粉是植物细胞中以贮藏状态存在的糖,是贮存能量的多糖,<图片 1>链不分支的淀粉为__ Option: (A) 初生分生组织 (B) 初生结构 (C) 形成层的发生 (D) 形成层环的形成

|[Figure 308]|
|---|

<图片 1>

GPT-4V: 链不分支的淀粉为直链淀粉，也称为淀粉直链部分或淀粉的直链分子，学术上称为淀粉素 （Amylose）。

Error Category: Answer Extraction Error Error Reason: GPT-4V correctly answer the question, but the function to extract the answer has the limitation, which led to the wrong answer being extracted.

Ground Truth: 直链淀粉

Figure B122: A sample error case of Science (subcategory: Biology). Back to List of figures

[Figure 309]

#### 科学:生物

Question:下图<图片 1>是一台离心萃取机，正确的叙述是 Option:

(A) A是重液出口，B是重液入口，C是轻液入口，D是轻液出口 (B) A是轻液出口，B是轻液入口，C是重液入口，D是重液出口 (C) A是重液出口，B是轻液入口，C是重液入口，D是轻液出口 (D) A是重液入口，B是轻液出口，C是重液出口，D是轻液入口

|[Figure 310]|
|---|

<图片 1>

GPT-4V: 很抱歉，我无法直接识别和验证图纸上的标记与实际离心萃取机的对应关系，因为这需要具

体的设备设计知识和说明。通常，离心萃取机的工作原理是基于密度差异，轻液和重液会在离心力的 作用下分离。轻液和重液的入口和出口位置取决于具体的机型和设计。如果没有具体的设备手册或者 设计说明，我无法确定哪个选项是正确的。在实际情况中，您应该参考设备的用户手册或者联系制造 商来获取准确的信息。如果这是一个理论测试题目，通常会提供足够的信息来确定正确答案，或者假 设你已经有了相关背景知识。在这种情况下，如果没有额外的信息，我无法提供一个确定的答案。

Error Category: Lack of Knowledge Error Reason: GPT-4V cannot determine the answer due to a lack of knowledge about the centrifugal extractor.

- Ground Truth: (C) A是重液出口，B是轻液入口，C是重液入口，D是轻液出口

Figure B123: A sample error case of Science (subcategory: Biology). Back to List of figures

[Figure 311]

#### 科学:生物

Question: 如图<图片 1>最早的能归人现存生物分类单位的真核生物是一个保存在加拿大北部12亿年 前的岩石中的__,该化石源自潮间沉积物。这个种类非常像暗紫红毛菜,一类现存的在淡水和海水交界 处生长的藻类

|[Figure 312]|
|---|

<图片 1>

GPT-4V: 对不起，我无法提供具体的化石名称，因为这需要专业的古生物学知识和对具体化石研究的 了解。不过，根据您提供的信息，这种化石非常像现存的暗紫红毛菜，这是一种藻类。因此，最合理 的答案可能是这个化石属于某种古老的藻类，但是没有更多的信息，我无法提供一个确切的名称。

Error Category: Lack of Knowledge, Reject to Answer Error Reason: GPT-4V refuses to answer the question due to a lack of professional paleontological knowledge and understanding of specific fossil studies.

Ground Truth: 海生红藻化石

Figure B124: A sample error case of Science (subcategory: Biology). Back to List of figures

[Figure 313]

#### 科学:化学

Question: 如图<图片 1>，比较下列化合物酸性强弱_。 Option:

(A) B>A>C (B) C>B>A (C) A>C>B (D) A>B>C

|[Figure 314]|
|---|

<图片 1>

GPT-4V: 根据所提供的图片，我们可以看到三个化合物：A: 含有一个氨基（-NH2）的化合物。B: 含 有一个羟基（-OH）的化合物。C: 含有一个羧基（-COOH）的化合物。在这三个官能团中，羧基是最 酸性的，因为它可以释放质子（H+），形成稳定的羧酸根离子。羟基次之，而氨基通常是碱性的，不 容易释放质子。因此，酸性强弱顺序应该是 C（羧基）> B（羟基）> A（氨基）。正确答案是 (D) A>B>C。

- Ground Truth: (D) A>B>C

Figure B125: A sample correct case of Science (subcategory: Chemistry). Back to List of figures

[Figure 315]

#### 科学:化学

Question: <图片 1>中,哪一点代表双链DNA的Tm值?（） Option:

(A) A (B) B (C) C (D) D

|[Figure 316]|
|---|

<图片 1>

GPT-4V: 在这张图片中，我们看到的是一个DNA熔解曲线，它显示了随着温度（T）的升高，DNA吸收 度（OD260）的变化。双链DNA的熔解温度（Tm）是指DNA从双链状态转变为单链状态时的温度，这通 常对应于吸收度曲线上最陡峭的部分，即曲线的中点。在这个图中，点C位于曲线的中间部分，这个 位置代表了吸收度急剧上升的地方，这是因为双链DNA解链成单链，导致吸收度增加。因此，点C最有 可能代表双链DNA的Tm值。正确答案是：(C) C

Ground Truth: (C) C

Figure B126: A sample correct case of Science (subcategory: Chemistry). Back to List of figures

[Figure 317]

#### 科学:化学

Question: 用y=ax+b的形式来表达米氏方程时,其表达式如<图片 1>所示，当在反应体系中加入非竟争 性抑制剂时, a值将_

Option: (A) 增大 (B) 减小 (C) 保持不变 (D) 不确定

|[Figure 318]|
|---|

<图片 1>

GPT-4V: 这 张 图 片 显 示 的 是 米 氏 方 程 （ Michaelis-Menten equation ） 的 线 性 形 式 ， 也 就 是 Lineweaver-Burk plot的方程式。在这个方程中，V1是y轴上的值， [S]1 是x轴上的值，斜率a等于VKm

， 截距b等于V 1

max

。当加入非竞争性抑制剂时，Vmax会减小，因为抑制剂会降低酶的最大反应速率。然 而，Km（米氏常数）保持不变，因为非竞争性抑制剂不影响酶与底物结合的亲和力。由于a值等于

max

Km

Vmax ，Vmax的减小会导致a值增大。因此，正确答案是： (A) 增大

- Ground Truth: (A) 增大

Figure B127: A sample correct case of Science (subcategory: Chemistry). Back to List of figures

[Figure 319]

#### 科学:化学

Question: 如图<图片 1>，将下列化合物按脱水反应速率由大到小排列_。 Options:

(A) A>B>C (B) B>C>A (C) C>A>B (D) A>C>B

|[Figure 320]|
|---|

<图片 1>

GPT-4V: 抱歉，我无法直接提供化学问题的答案。但是，我可以告诉你如何分析这个问题。在比较脱 水反应速率时，我们通常考虑以下因素：1. 分子中羟基（-OH）的位置：羟基所在的碳原子的杂化类 型（sp3, sp2, sp）会影响脱水速率。2. 羟基附近的稳定化因素：例如，若脱水后能形成稳定的碳 正离子，则该脱水反应更容易进行。3. 分子内部可能的稳定化作用：例如，若脱水后能通过共轭或 超共轭效应稳定化，则该脱水反应速率也会更快。根据这些原则，你可以分析每个化合物的结构，确 定脱水反应的相对速率。通常，若脱水能够导致稳定的共轭体系的形成，那么该脱水反应的速率会更 快。在这个特定的问题中，你需要考虑每个化合物脱水后可能形成的碳正离子的稳定性，以及是否有 共轭体系的形成。请根据这些信息和你对化学反应机理的了解来选择最可能正确的答案。

Error Category: Reject to Answer Error Reason: GPT-4V refuses to directly answer the question.

- Ground Truth: (B) B>C>A

Figure B128: A sample error case of Science (subcategory: Chemistry). Back to List of figures

[Figure 321]

#### 科学:化学

Question: 下列装置的设计能达到实验目的的是（ ） Option:

(A) <图片 1> (B) <图片 2> (C) <图片 3> (D) <图片 4>

|[Figure 322]<br><br>[Figure 323]<br><br>[Figure 324]<br><br>[Figure 325]<br><br><图片 1> <图片 2> <图片 3> <图片 4>|
|---|

GPT-4V: 由于题目没有给出具体的实验目的，我无法直接给出正确答案。不过，我可以根据每张图片

所展示的实验装置的特点来分析它们可能的用途：(A) 这个装置看起来像是用于蒸馏的实验装置，其 中包括了蒸馏瓶、冷凝管和接收瓶。(B) 这个装置是一个伏打电池的示意图，其中包含了锌和铜两种 金属电极，以及盐桥。(C) 这个装置可能用于展示不溶于水的液体（如四氯化碳CCl4）与水分离的实 验。(D) 这个装置可能用于收集气体（如氢碘酸HI）的实验，其中包括了发生瓶和收集瓶。如果你能 提供实验的目的或者更多的上下文信息，我可以帮助你更准确地选择正确的选项。

Error Category: Perceptual Error Error Reason: GPT-4V does not understand the text in each image and refuses to answer.

- Ground Truth: (C) <图片 3>

Figure B129: A sample error case of Science (subcategory: Chemistry). Back to List of figures

[Figure 326]

#### 科学:化学

Question: <图片 1>以下化合物最容易与H2SO4反应的是___

- (A) A
- (B) B
- (C) C
- (D) D

|[Figure 327]|
|---|

<图片 1>

GPT-4V: 很抱歉，由于图片中的化合物结构不完整，我无法看到完整的分子结构，因此无法准确判断 哪个化合物最容易与硫酸（ H2SO4 ）反应。通常情况下，硫酸会与醇、醚或含有活泼氢的化合物反应， 但是需要知道化合物的完整结构才能做出准确的判断。如果您能提供更多信息或者化合物的完整名称， 我或许能够帮助您找到正确答案。

Error Category: Perceptual Error Error Reason: GPT-4V does not understand the figure in each option.

- Ground Truth: (C) C

Figure B130: A sample error case of Science (subcategory: Chemistry). Back to List of figures

[Figure 328]

#### 科学:化学

Question: <图片 1>该化合物含有的杂环结构是____

|[Figure 329]|
|---|

<图片 1>

GPT-4V: 正确答案：嘧啶和咪唑

Error Category: Annotation Error Error Reason: The annotated answer is D, which is incorrect.

- Ground Truth: D

Figure B131: A sample error case of Science (subcategory: Chemistry). Back to List of figures

[Figure 330]

#### 科学:化学

Question:普伐他汀是一种调节血脂的药物，其结构简式如图1所示(未表示出其空间构型)。下列关于 普伐他汀的化学性质描述正确的是（）

Option: (A) 能发生加成、取代、消去反应 (B) 不能使酸性KMnO4溶液褪色 (C) 能与FeCl3溶液发生显色反应 (D) 1mol该物质最多可与1mol NaOH反应

|[Figure 331]|
|---|

<图片 1>

GPT-4V: 根据所提供的普伐他汀的结构简式，我们可以分析其化学性质：(A) 普伐他汀分子中含有碳 碳双键、羟基和羧基，这些官能团可以参与加成反应（如碳碳双键）、取代反应（如羟基的取代）和 消去反应（如羧基的脱水反应）。因此，选项(A)是正确的。(B) 碳碳双键可以与酸性KMnO4溶液发生 反应，使溶液褪色，因为KMnO4是一种强氧化剂，可以氧化双键。因此，选项(B)是错误的。(C) 普伐 他汀分子中没有酚羟基，而是醇羟基，通常FeCl3溶液与酚类化合物发生显色反应。因此，选项(C)是 错误的。(D) 普伐他汀分子中只有一个羧基，这个羧基可以与1mol的NaOH反应生成盐和水。因此，选 项(D)是正确的。综上所述，正确答案是(A)和(D)。

Error Category: Reasoning Error Error Reason: For option D, GPT-4V accurately assessed that the carboxyl group can react with 1 mol of NaOH. However, it overlooked the presence of -COOC- in pravastatin, which can also react with 1 mol of NaOH. Therefore, 1 mol of this substance can react with up to 2 mol of NaOH.

Ground Truth: (A) 能发生加成、取代、消去反应

Figure B132: A sample error case of Science (subcategory: Chemistry). Back to List of figures

[Figure 332]

#### 科学:地理

Question:如下为福建师范大学地理学院地理专业学生利用暑期旅游考察所画的地貌景观素描<图片 1>，学生乙说他是全班同学中到达的海拔最高的一个，他绘制的最有可能是_（序号）

|[Figure 333]|
|---|

<图片 1>

GPT-4V: 学生乙说他是全班同学中到达的海拔最高的一个，他绘制的最有可能是表示海拔较高地区的 地貌。在提供的五幅素描中，C和E看起来像是山脉，通常山脉的海拔会比较高。但是，C图中的山脉 更加陡峭和高耸，更有可能代表海拔最高的地区。因此，学生乙最有可能绘制的是C图。 正确答案：C

Ground Truth: C

Figure B133: A sample correct case of Science (subcategory: Geography). Back to List of figures

[Figure 334]

#### 科学:地理

Question:如下<图片 1>C船推迟转向的决定是航行中偏离航线的船舶通过_来纠正偏船 Option: (A)增加航速 (B)减小航速 (C)调整转向时机 (D)更换导航设备

|[Figure 335]|
|---|

<图片 1>

GPT-4V: 根据所提供的图片，我们可以看到三艘船的航线。C船的航线显示了它在转向时有所延迟， 导致偏离了原定航线。为了纠正这种偏离，船舶通常需要调整其转向时机，而不是仅仅改变航速或更 换导航设备。因此，最合适的选项是：(C) 调整转向时机 所以正确答案是 (C)。

Ground Truth: C

Figure B134: A sample correct case of Science (subcategory: Geography). Back to List of figures

[Figure 336]

#### 科学:地理

Question: 此为河流袭夺示意<图片 1>，甲为袭夺河，乙为断头河或不适称河，乙'为被袭夺河，则2 为_

Option: (A) 已上溯的袭夺裂点 (B) 断头河 (C) 被袭夺河 (D) 改向河或反向河

|[Figure 337]|
|---|

<图片 1>

GPT-4V: 正确答案是：(A) 已上溯的袭夺裂点

在河流袭夺的过程中，袭夺河（甲）通过侵蚀作用逐渐向上游延伸，最终会切断另一条河流（乙'） 的部分流程，形成一个袭夺点。在示意图中，2号位置标示的是这个袭夺点，它是甲河流袭夺乙'河流 的结果。因此，2号位置是已上溯的袭夺裂点。

Ground Truth: (A) 已上溯的袭夺裂点

Figure B135: A sample correct case of Science (subcategory: Geography). Back to List of figures

[Figure 338]

#### 科学:地理

Question: 读下面的我国渤海西岸某地的地表物质分布示意<图片 1>，将正确答案填在括弧内。该地 区在①沙丘、②海蚀崖、③三角洲平原和④风蚀蘑菇气等几种可能的地貌类型中，主要是（）

Option: (A) ①② (B) ②③ (C) ①③ (D) ②④

|[Figure 339]|
|---|

<图片 1>

GPT-4V: 由于图片中的文字和图例不是很清晰，我无法提供一个确定的答案。但是，根据常见的地貌 特征，沙丘通常与风成沉积有关，海蚀崖与海洋侵蚀有关，三角洲平原与河流沉积有关，而风蚀蘑菇 状地貌是由风蚀作用形成的。在这个情况下，我会根据图中的图例和分布特征来猜测最可能的地貌类 型组合。如果您能提供更清晰的图例或者对图中的标记进行解释，我可以帮助您更准确地确定答案。

Error Category: Reject to Answer Error Reason: GPT-4V refuses to answer this question without further explanations for the figure.

Ground Truth: (C) ①③

Figure B136: A sample error case of Science (subcategory: Geography). Back to List of figures

[Figure 340]

#### 科学:地理

Question: 如<图片 1>所示，甲的绝对高度为_米

|[Figure 341]|
|---|

<图片 1>

GPT-4V: 根据图片中的信息，我们可以看到点甲与海平面的相对高度是1500米，而点乙与海平面的相 对高度是500米。点乙到点丙的相对高度差是800米，因此点丙的相对高度是500米加上800米，等于 1300米。点丙到点甲的相对高度差是1000米，所以点甲的绝对高度是点丙的相对高度加上这个高度差， 即1300米加上1000米，等于2300米。 所以，甲的绝对高度为2300米。

Error Category: Perceptual Error, Lack of Knowledge Error Reason: The GPT-4V lacks specific knowledge of the concept of “absolute height”, failing to comprehend that it refers to the height relative to sea level. Furthermore, GPT-4V misinterprets the dotted line in the image indicating sea level and the point labeled “乙“ and incorrectly calculates the height difference between point “甲“ and “丙“ .

Ground Truth: 2500

Figure B137: A sample error case of Science (subcategory: Geography). Back to List of figures

[Figure 342]

#### 科学:地理

Question: 如<图片 1>所示,A线为稳定情况下的水位流量关系曲线,则涨洪情况的水位流量关系曲线一 般为_（A/B/C）线

|[Figure 343]|
|---|

<图片 1>

GPT-4V: 正确答案：B线

在涨洪情况下，水位流量关系曲线通常会表现出更高的水位对应更大的流量，这是因为洪水增加了河 流的流量，同时也会导致水位的上升。在提供的图像中，B线相对于A线在水位（Z轴）和流量（Q轴） 上都有所增加，这符合涨洪情况下水位流量关系的变化趋势。因此，涨洪情况下的水位流量关系曲线 一般为B线。

Error Category: Reasoning Error Error Reason: GPT-4V successfully understands the images and text and correctly recalls the knowledge that a higher water level corresponds to a larger flow rate in a flood situation. However, the model directly select the answer without fixing the variables. Actually, the model should fix one variable and compare the values of the other variable.

Ground Truth: C

Figure B138: A sample error case of Science (subcategory: Geography). Back to List of figures

[Figure 344]

#### 科学:地理

Question: <图片 1>若图中所示的是水循环示意图，则能体现东亚夏季风的环节是( ) Option:

(A) ① (B) ② (C) ③ (D) ④

|[Figure 345]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是一个水循环的示意图，我们可以看到有蒸发、降水等过程。东亚夏季风通 常与海洋上的蒸发和随后的降水过程有关，因为夏季风带来的是来自海洋的湿润空气，这些湿润空气 上岸后会上升并冷却，导致水蒸气凝结成云，最终形成降水。 在这张图中，①号标记的过程可能代 表水蒸气的水平运动，②号标记的过程代表蒸发，③号标记的过程代表降水，④号标记的过程代表水 蒸气上升。 根据这些信息，能体现东亚夏季风环节的是蒸发和降水过程，即选项 (B) ② 和 (C) ③。 然而，由于题目要求选择单个选项，我们需要确定哪一个环节更能体现东亚夏季风的特点。通常情况 下，东亚夏季风更为人们所熟知的是其带来的降水，因此我会选择 (C) ③ 作为最可能正确的选项。

Error Category: Reasoning Error Error Reason: GPT-4V accurately identified the processes within the diagram of water cycle, including evaporation, precipitation, and recognized that summer winds bring moist air from the ocean. However, it made mistakes while reasoning the various stages of the water cycle. The correct stages should be: ① Precipitation, ② Surface Runoff, ③ Evaporation, and ④ Vapor Transportation. As the East Asian summer monsoon is a southeastern wind originating from the ocean, the vapor transport in the ④ comes from the ocean. Therefore, the stage that reflects the characteristics of the East Asian summer monsoon is ④.

- Ground Truth: (D) ④

Figure B139: A sample error case of Science (subcategory: Geography). Back to List of figures

[Figure 346]

#### 科学:地理

Question: <图片 1>中M、N两地降水的共同特点是_ Option: (A) 全年少雨 (B) 全年多雨 (C) 季节性降水 (D) 全年均衡降水

|[Figure 347]|
|---|

<图片 1>

GPT-4V:根据提供的图片，我们可以看到一个气候图，它显示了降水量（以毫米为单位）和温度（以 摄氏度为单位）随时间（月份）的变化。从图中可以看出，降水量在某些月份特别高，而在其他月份 则相对较低。这表明降水具有季节性的特点。因此，根据图表所示的数据，M和N两地的共同特点是： C) 季节性降水 所以正确答案是 (C) 季节性降水。

Error Category: Textual Understanding Error Reason: The question inquired about the precipitation characteristics of areas M and N, however, GPT-4V's response pertained to the rainfall patterns of area P, which consequently resulted in an incorrect answer.

- Ground Truth: (A) 全年少雨

Figure B140: A sample error case of Science (subcategory: Geography). Back to List of figures

[Figure 348]

#### 科学:地理

Question: 在两个相同的长方形塑料盒子的一端用潮湿的土壤分别搭起一座相同坡度的“小山”，从 盒子的另一端倒入等量的水，形成“湖泊”；不同的是在其中一个盒子内，将一块微湿的海绵压入 “小山”底端的土壤里（如图<图片 1>）。分别用喷水壶，往两个“小山”上洒等量的水，模拟降水。 观察、比较两个盒子内水面发生的变化。该实验模拟的地理事实是（）

Option:

- (A) 湖泊能够调节空气湿度
- (B) 坡度变化会改变地表径流量
- (C) 湿地能够吸纳地表径流
- (D) 降水变化能够影响自然环境

|[Figure 349]|
|---|

<图片 1>

GPT-4V: 根据实验的描述和图示，这个实验主要是为了观察和比较有无海绵（代表湿地）对地表径流 的影响。在这个实验中，海绵的作用是吸收和减缓水流，模拟湿地对地表径流的吸纳作用。因此，正 确答案是：(C) 湿地能够吸纳地表径流 其他选项与实验的直接目的不相关。选项 (A) 湖泊能够调节空气湿度，(B) 坡度变化会改变地表径 流量，和 (D) 降水变化能够影响自然环境，虽然是正确的地理事实，但它们并不是这个实验设计的 主要观察点。

Error Category: Answer Extraction Error Error Reason: GPT-4V answered the question correctly, but the function to extract the answer has the limitation, which led to the wrong answer being extracted.

Ground Truth: (C) 湿地能够吸纳地表径流

Figure B141: A sample error case of Science (subcategory: Geography). Back to List of figures

[Figure 350]

#### 科学:地理

Question: 判断下面陈述对错：如<图片 1>为幼年期河谷

|[Figure 351]|
|---|

<图片 1>

GPT-4V: 这个陈述是正确的。所提供的图片显示了一个河流在其河谷中蜿蜒流动，河谷的侧壁较为陡 峭，河流的弯曲程度较大，这些特征符合幼年期河谷的典型特征。幼年期河谷通常具有较为陡峭的河 岸和较多的河流弯曲，因为河流正在侵蚀其床岩，形成深而狭窄的河谷。随着时间的推移，河流会继 续侵蚀和拓宽河谷，进入成熟期和老年期。

Error Category: Lack of Knowledge Error Reason: The model understands to assess the development stage of a valley based on the steepness of the valley's sidewalls and the winding nature of the river. However, it has confused the characteristics of mature valleys with those of youthful ones. Rivers in youthful valleys are rather straight, instead of winding.

Ground Truth: 错

Figure B142: A sample error case of Science (subcategory: Geography). Back to List of figures

[Figure 352]

#### 科学: 数学

Question: 二维离散型随机变量(X,Y)的分布律为<图片 1>，F(x,y)为(X,Y)的联合分布函数,则 F(1.5,1.5）等于

Option:

(A) 1/6 (B) 1/2 (C) 1/3 (D) 1/4

|[Figure 353]|
|---|

<图片 1>

GPT-4V: 要找到F(1.5, 1.5)，我们需要计算所有X≤1.5和Y≤1.5的概率值之和。根据给定的分布律 表，我们可以看到：

- - 当X=1且Y=0时，概率是1/6。
- - 当X=1且Y=1时，概率是1/3。

由于1.5大于表中的X和Y的最大值1，我们不需要考虑X=2或Y=2的情况，因为这些值超出了我们计算 F(1.5, 1.5)时考虑的范围。

因此，F(1.5, 1.5)是这两个概率值的和： F(1.5, 1.5) = 1/6 + 1/3 = 1/6 + 2/6 = 3/6 = 1/2 所以正确答案是 (B) 1/2。

- Ground Truth: (B) 1/2

Figure B143: A sample correct case of Science (subcategory: Mathematics). Back to List of figures

[Figure 354]

#### 科学: 数学

Question: 某电子生产企业2003年和2002年三种主要产品的单位成本和产量资料如<图片 1>，则三种 产品的产值总指数及产值增减总额为_元。

|[Figure 355]|
|---|

<图片 1>

GPT-4V: 为了计算三种产品的产值总指数及产值增减总额，我们需要先计算每种产品2002年和2003年 的产值，然后计算产值增减额和产值总指数。 首先，我们计算每种产品的产值（单位成本乘以产量）： 电能电池：

- 2002年产值 = 900元/千 * 8.5千 = 7650元
- 2003年产值 = 1000元/千 * 9.0千 = 9000元 电路板： 2002年产值 = 500元/块 * 55.0块 = 27500元 2003年产值 = 500元/块 * 58.5块 = 29250元 家电机具：

- 2002年产值 = 700元/台 * 100.0台 = 70000元
- 2003年产值 = 800元/台 * 115.0台 = 92000元

接下来，我们计算每种产品的产值增减额： 电能电池增减额 = 9000元 - 7650元 = 1350元 电路板增减额 = 29250元 - 27500元 = 1750元 家电机具增减额 = 92000元 - 70000元 = 22000元

然后，我们计算产值增减总额： 产值增减总额 = 1350元 + 1750元 + 22000元 = 25100元

最后，我们计算产值总指数。产值总指数是通过将各个产品的2003年产值与2002年产值进行比较得出 的。我们使用的是简单的产值总指数公式，即：

产值总指数 = (2003年各产品产值之和 / 2002年各产品产值之和) * 100%

- 2002年产值总和 = 7650元 + 27500元 + 70000元 = 105150元
- 2003年产值总和 = 9000元 + 29250元 + 92000元 = 130250元

产值总指数 = (130250元 / 105150元) * 100% ≈ 123.88% 综上所述，三种产品的产值总指数为123.88%，产值增减总额为25100元。

Ground Truth: 25100

Figure B144: A sample correct case of Science (subcategory: Mathematics). Back to List of figures

[Figure 356]

#### 科学: 数学

Question: 随机抽取了10个家庭,调查了他们的家庭月收入x(单位:百元)和月支出y(单位:百元),记录 于<图片 1>，y与x_(填“是”或“否”)存在线性关系_。

|[Figure 357]|
|---|

<图片 1>

GPT-4V: 根据所提供的图片信息，我们可以看到家庭月收入x和月支出y的数据。为了确定这两个变量 之间是否存在线性关系，我们需要分析数据点是否大致排列在一条直线上，或者使用统计方法如计算 相关系数来判断。

从图片中提供的数据来看，没有明显的模式表明x和y之间存在直接的线性关系。数据点没有显示出一 致的趋势，即随着x的增加，y并没有显示出一致的增加或减少。因此，根据这些数据点，我们不能确 定存在线性关系。

正确答案：否

Error Category: Perceptual Error Error Reason: GPT-4V mistakenly believes that there is no linear correlation between x and y, but we can calculate it based on the Pearson correlation coefficient. The answer is r=0.909, which means there is a strong linear correlation.

Ground Truth: 是

Figure B145: A sample error case of Science (subcategory: Mathematics). Back to List of figures

[Figure 358]

#### 科学: 物理

Question: 如<图片 1>，单色平行光垂直入射到双缝上。观察屏上P点到两缝的距离分别为𝑟𝑟1和𝑟𝑟2。设 双缝和屏之间充满折射率为𝑛𝑛的媒质,则P点处二相干光线的光程差为_。

Option: (A) 𝑟𝑟2 − 𝑟𝑟1 (B) 𝑛𝑛(𝑟𝑟2 − 𝑟𝑟1) (C) 2𝑛𝑛(𝑟𝑟2 − 𝑟𝑟1) (D) 2(𝑟𝑟2 − 𝑟𝑟1)

|[Figure 359]|
|---|

<图片 1>

GPT-4V:正确答案是 (B) 𝑛𝑛(𝑟𝑟2 − 𝑟𝑟1)。 光程差是指两束光在相同时间内走过的路径长度的差。当光在不同介质中传播时，其速度会改变，

因此光程差也会受到介质折射率的影响。在这个问题中，两束光通过的媒质折射率为𝑛𝑛，因此光程差 不仅取决于两束光在空气中的路径长度差（𝑟𝑟2 − 𝑟𝑟1），还要乘以媒质的折射率𝑛𝑛。所以，P点处二相 干光线的光程差为𝑛𝑛(𝑟𝑟2 − 𝑟𝑟1)。

Ground Truth: (B)

Figure B146: A sample correct case of Science (subcategory: Physics). Back to List of figures

[Figure 360]

#### 科学: 物理

Question: <图片 1>一质量为𝑀𝑀的均匀细杆，可绕光滑水平轴转动，一质量为𝑚𝑚的小球以速度𝑉𝑉水平飞 来，与杆端做完全非弹性碰撞，则小球与杆组成的系统(图1),满足（）

Option: (A)动量守恒，角动量守恒； (B) 动量不守恒，角动量守恒； (C) 动量不守恒，角动量不守恒； (D) 动量守恒，角动量不守恒。

|[Figure 361]|
|---|

<图片 1>

GPT-4V:在这个问题中，我们需要考虑动量守恒和角动量守恒的条件。

动量守恒定律指出，如果没有外力作用，系统的总动量保持不变。在这个问题中，小球与杆的碰 撞是一个内力作用，但由于杆固定在一个轴上，这个轴对杆的作用力提供了一个外力，因此系统的线 动量不守恒。所以动量不守恒。

角动量守恒定律指出，如果没有外力矩作用，系统的总角动量保持不变。在这个问题中，由于杆 绕一个固定轴旋转，且轴对杆的作用力通过轴心，不产生力矩，因此系统的角动量守恒。因此，正确 答案是 (B) 动量不守恒，角动量守恒。

Ground Truth: (B)

Figure B147: A sample correct case of Science (subcategory: Physics). Back to List of figures

[Figure 362]

#### 科学: 物理

Question:判断下面陈述对错：如右图所示<图片 1>，两个半径分别为R1和𝑅𝑅2, 𝑅𝑅1 < 𝑅𝑅2 的均带电球 面同心放置，其间充满介电常数为𝜀𝜀 的各向同性均电介质，已知两个球面的带电量均为 𝑄𝑄。那么内球 面以内的空间中的电场能量为0。

|[Figure 363]|
|---|

<图片 1>

GPT-4V:根据电磁学的基本原理，当两个带电球面同心放置并且带有相同的电荷量 𝑄𝑄 时，内球面以 内的空间实际上是一个电场为零的区域。这是因为内球面上的电荷会在其表面产生一个向外的电场， 而外球面上的电荷会在内球面以内的区域产生一个等大小但方向相反的电场，两者在内球面以内的区 域相互抵消，导致该区域的电场强度为零。

由于电场能量密度与电场强度的平方成正比（电场能量密度 𝑢𝑢 = 1/2 𝜀𝜀 𝐸𝐸2，其中 𝜀𝜀 是介电常数， 𝐸𝐸 是电场强度），如果电场强度为零，那么该区域的电场能量也将为零。

因此，根据题目描述和电磁学的基本原理，陈述“内球面以内的空间中的电场能量为0”是正确 的。正确答案是：正确。

Ground Truth: 正确

Figure B148: A sample correct case of Science (subcategory: Physics). Back to List of figures

[Figure 364]

#### 科学: 物理

Question:波长为𝜆𝜆的单色光垂直照射<图片 1>所示的透明薄膜。膜厚度为𝑒𝑒,两束反射光的光程差𝛿𝛿 =_

|[Figure 365]|
|---|

<图片 1>

GPT-4V:在这个问题中，我们需要计算两束反射光之间的光程差。光程差是由于光在不同介质中传播 时速度不同而产生的。在这个情况下，光在透明薄膜中来回传播一次，因此光程差𝛿𝛿可以由下面的公 式计算得出：𝛿𝛿 = 2𝑛𝑛𝑒𝑒 − 𝜆𝜆/2

这里的𝜆𝜆/2是因为从𝑛𝑛1到𝑛𝑛2的界面反射的光会产生半个波长的相位变化（即𝜋𝜋的相位差），而从 𝑛𝑛2到𝑛𝑛3的界面反射的光不会产生这样的相位变化。因此，我们需要从总的光程差中减去半个波长来考 虑这个相位变化。

所以，正确答案是：𝛿𝛿 = 2𝑛𝑛2𝑒𝑒 − 𝜆𝜆/2 其中，𝑛𝑛2是薄膜的折射率，𝑒𝑒是薄膜的厚度。由于题目中没有给出具体的𝜆𝜆和𝑒𝑒的数值，所以我们

不能给出一个具体的数值答案，只能提供上述表达式。

Error Category: Perceptual Error Error Reason: The model did not fully understand the image, especially failing to recognize the numerical information in the image. Therefore, it produced an incorrect answer.

Ground Truth: 2.60

Figure B149: A sample error case of Science (subcategory: Physics). Back to List of figures

[Figure 366]

#### 科学: 物理

Question: 在𝑝𝑝 − 𝑉𝑉图<图片 1>上有两条曲线𝑎𝑎𝑏𝑏𝑎𝑎和𝑎𝑎𝑎𝑎𝑎𝑎,由此可以得出以人下结论（） Option:

(A)其中一条是绝热线,另一条是等温线 (B) 两个过程吸收的热量相同 (C) 两个过程中系统对外作的功相等 (D) 两个过程中系统的内能变化相同

|[Figure 367]|
|---|

<图片 1>

GPT-4V:在𝑝𝑝 − 𝑉𝑉图上，两条曲线𝑎𝑎𝑏𝑏𝑎𝑎和𝑎𝑎𝑎𝑎𝑎𝑎表示了两个不同的热力学过程。根据热力学的第一定律， 系统的内能变化等于系统吸收的热量与系统对外做的功之差。在这个问题中，我们没有足够的信息来 确定这两个过程的具体类型（如绝热、等温、等压等），因此我们无法直接确定选项(A)是否正确。

- 对于选项(B)，由于我们不知道这两个过程的具体细节，我们无法确定两个过程吸收的热量是否

相同。

- 对于选项(C)，在𝑝𝑝 − 𝑉𝑉图上，系统对外做的功可以通过计算曲线下的面积来确定。由于两条曲线

𝑎𝑎𝑏𝑏𝑎𝑎和𝑎𝑎𝑎𝑎𝑎𝑎在图中的面积看起来不同，我们可以推断两个过程中系统对外做的功不相等。

- 对于选项(D)，系统的内能变化取决于吸收的热量和对外做的功。由于我们无法确定吸收的热量，

我们也无法确定内能变化是否相同。

因此，根据图中的信息和热力学原理，最可能正确的选项是(C) 两个过程中系统对外作的功不相 等。

Error Category: Lack of Knowledge Error Reason:The conclusion in the options can be deduced and demonstrated from the physical processes expressed in the image, but the model did not answer using specific knowledge.

Ground Truth: (D)两个过程中系统的内能变化相同

Figure B150: A sample error case of Science (subcategory: Physics). Back to List of figures

[Figure 368]

#### 科学:物理

Question: <图片 1>在𝑥𝑥𝐶𝐶𝑦𝑦面上倒扣着半径为𝑅𝑅的半球面上电荷均匀分布,面电荷密度为𝜎𝜎，A点的坐标 为(0,𝑅𝑅/2),B 点的坐标为(3𝑅𝑅/2,0),AB之间电势为_____

Option:

(A) 𝑅𝑅×𝜎𝜎2×𝜀𝜀 (B) 𝑅𝑅×𝜎𝜎3×𝜀𝜀 (C) 𝑅𝑅×𝜎𝜎4×𝜀𝜀 (D) 𝑅𝑅×𝜎𝜎6×𝜀𝜀

|[Figure 369]|
|---|

<图片 1>

GPT-4V: 很抱歉，但我无法直接提供这个物理问题的答案，因为它需要特定的物理知识和计算来解决， 而这些信息没有在问题中给出。通常，计算两点间的电势差需要应用电场和电势的物理定律，如高斯 定律或库仑定律，并且可能需要积分来处理连续的电荷分布。

然而，如果你能提供相关的物理公式或者更多的背景信息，我可以帮助你理解这些概念，或者指 导你如何使用这些公式来解决问题。

Error Category: Lack of Knowledge, Answer Extraction Error Error Reason: The model did not apply relevant knowledge in answering, it is obvious that GPT-4V lacked related physical knowledge in this round of answer.

Ground Truth: (D) 𝑹𝑹×𝝈𝝈𝟓𝟓×𝜺𝜺

Figure B151: A sample error case of Science (subcategory: Physics). Back to List of figures

[Figure 370]

#### 人文社会科学: 历史

Question: 如图<图片 1>，为加强中央对地方的管理，它推行_____制度。

|[Figure 371]|
|---|

<图片 1>

GPT-4V: 根据您的问题描述，如果是关于中国历史上加强中央对地方的管理的制度，一个可能的答 案是“郡县制”制度。这是中国古代一个重要的行政管理制度，始于秦朝，用以加强中央集权。

Ground Truth: 郡县

- Figure B152: A sample correct case of Humanities and Social Sciences (subcategory: History). Back to List of figures

[Figure 372]

#### 人文社会科学: 历史

Question: 为下表<图片 1>选取表名，最恰当的是() option:

- (A)京津冀地区晚清民族企业简表
- (B)北京近代民族企业简表
- (C)洋务运动时期北方企业简表
- (D)近代民族资本主义企业简表

|<图片 1><br><br>[Figure 373]|
|---|

<图片 1>

GPT-4V:根据图片中提供的信息，表格列出了不同年份成立的企业以及它们的名称和地点。所有列出 的企业都是在晚清时期成立的，并且地点包括了中国北方的城市，如北京和天津。因此，最合适的表 名应该是与晚清时期的北方地区和民族企业相关的。

- 选项(A) "京津冀地区晚清民族企业简表"提到了京津冀地区，这是包括北京和天津在内的地区，与表 中的信息相符合。
- 选项(B) "北京近代民族企业简表"只提到了北京，而表中包含了天津的企业，因此不够全面。
- 选项(C) "洋务运动时期北方企业简表"虽然提到了北方企业，但没有特指民族企业，且洋务运动的时 间范围比晚清更狭窄。
- 选项(D) "近代民族资本主义企业简表"虽然提到了近代民族资本主义企业，但没有指明地区，且不特 定于晚清时期。 综上所述，最合适的选项是(A) "京津冀地区晚清民族企业简表"。

Ground Truth: (A) 京津冀地区晚清民族企业简表

- Figure B153: A sample correct case of Humanities and Social Sciences (subcategory: History). Back to List of figures

[Figure 374]

#### 人文社会科学: 历史

Question: 读如图<图片 1>，下列表述正确的是 ①东南地区通商口岸贸易网络最为密集②“五口通商” 中的香港成为贸易中心③天津的口岸贸易在北方居于核心地位④最早开埠的汉口是南北口岸贸易枢纽

option:

- (A)①②
- (B)③④
- (C)①③
- (D)②④

|<图片 1><br><br>[Figure 375]|
|---|

<图片 1>

GPT-4V:根据提供的图片，我们可以看到中国东南地区的贸易网络图。图中显示了不同口岸之间的贸 易联系，以及它们的相对位置和连接情况。

- ① 图中东南地区的贸易网络确实看起来比较密集，有多个连接点和线路。
- ② 图中并没有明确指出“五口通商”中的香港，因此无法确定香港是否成为贸易中心。
- ③ 天津位于图中北方，有多条线路连接，看起来确实在北方口岸贸易中占据核心地位。
- ④ 图中并没有提供足够的信息来判断汉口是否是最早开埠的，也无法确定它是否是南北口岸贸易的 枢纽。 根据以上分析，最可能正确的选项是： (C) ①③ n正确答案是 (C) ①③。

Ground Truth: (C) ①③

- Figure B154: A sample correct case of Humanities and Social Sciences (subcategory: History). Back to List of figures

[Figure 376]

#### 人文社会科学: 历史

Question: 结合所学，分析<图片 1>，对此解读正确的是（ ） Option: (A) 重矿业和交通业的发展始终处于领先地位 (B) 以轻工业为主的民族工业发展相对较快 (C) 传统经济形式已然淡出历史舞台 (D) 独立完整的工业体系已经形成

|[Figure 377]|
|---|

<图片 1>

GPT-4V: 根据提供的图片，我们可以看到一个表格，其中列出了1895-1911年间不同时间段的工业产 值（万元）。表格分为轻工业、重工业和交通业三个部分。通过分析这些数据，我们可以尝试确定正 确的选项。

- (A)重矿业和交通业的发展始终处于领先地位 - 从表格中可以看出，重工业和交通业的产值在各个时 间段都是最高的，因此这个选项是正确的。
- (B)以轻工业为主的民族工业发展相对较快 - 表格中的数据显示轻工业的产值增长也很快，但是没有 具体的比较数据来证明它的增长速度相对于其他行业是更快的。因此，这个选项没有足够的信息 来证明其正确性。
- (C)传统经济形式已然淡出历史舞台 - 表格中没有提供关于传统经济形式的信息，因此无法判断这个 选项的正确性。
- (D)独立完整的工业体系已经形成 - 表格中的数据显示工业产值在增长，但是没有足够的信息来证明 是否已经形成了一个独立完整的工业体系。

综上所述，根据表格中的数据，最可能正确的选项是 (A) 重矿业和交通业的发展始终处于领先地位。 其他选项要么没有足够的信息来证明其正确性，要么与表格中的数据不完全相关。

Error Category: Perceptual Error Error Reason: GPT-4V model has a problem understanding of the picture, it does not understand the speed of development of light industry.

Ground Truth: (B) 以轻工业为主的民族工业发展相对较快

- Figure B155: A sample error case of Humanities and Social Sciences (subcategory: History). Back to List of figures

[Figure 378]

#### 人文社会科学: 历史

Question: <图片 1>图示铜镜流行的时期为_____时期

Option: (A) 武昭 (B) 宣平 (C) 新莽 (D) 魏晋

|[Figure 379]|
|---|

<图片 1>

GPT-4V: 这些铜镜的图案风格和制作工艺可能暗示了它们流行的时期。然而，由于我无法直接识别 这些铜镜的确切时期，我将根据中国历史上铜镜的发展来推测。

- (A) 武昭 - 这不是一个广为人知的历史时期名称，可能是一个特定朝代的年号。
- (B) 宣平 - 这同样可能是一个年号，而不是一个广为人知的历史时期。
- (C) 新莽 - 新莽是指王莽篡夺西汉政权后建立的政权，时间大约是公元9年到公元23年。
- (D) 魏晋 - 这指的是魏晋南北朝时期，大约是公元220年到公元589年。 根据中国历史上铜镜的发展，魏晋南北朝时期是铜镜非常流行的时期，特别是具有精美图案和复杂设 计的铜镜。因此，如果必须选择最可能的选项，我会选择： 正确答案：(D) 魏晋

Error Category: Reasoning Error Error Reason: The knowledge and requirements of the problem were correctly understood, but errors were made during reasoning. Only through the bronze mirror popular era, rather than through the style of the bronze mirror to infer the era.

Ground Truth: (B) 宣平

- Figure B156: A sample error case of Humanities and Social Sciences (subcategory: History). Back to List of figures

[Figure 380]

#### 人文社会科学: 文献学

Question: <图片 1>该皇帝在位时命令修撰的“世界有史以来最大的百科全书”名为《____》

|[Figure 381]|
|---|

<图片 1>

GPT-4V: 这位皇帝是明朝的成祖朱棣，他在位时命令修撰的“世界有史以来最大的百科全书”名为 《永乐大典》。

Ground Truth: 永乐大典

- Figure B157: A sample correct case of Humanities and Social Sciences (subcategory: Literature).

Back to List of figures

[Figure 382]

#### 人文社会科学: 文献学

Question: <图片 1>图示的图书装帧方式是____ Option:

(A) 蝴蝶装 (B) 卷轴装 (C) 经折装 (D) 包背装

|[Figure 383]|
|---|

<图片 1>

GPT-4V: 这张图片展示的是一种古代书籍的装帧方式。根据图示的结构特点，这种装帧方式有封面、 书脊和折页，看起来像是书页被折叠起来并且装订在一起。这种装帧方式与传统的蝴蝶装相似，因为 蝴蝶装是将单张纸对折，使得印刷面在内，折痕在外，然后将对折的纸页捆扎成册，形成书籍。

正确答案是： (A) 蝴蝶装

Ground Truth: (A) 蝴蝶装

- Figure B158: A sample correct case of Humanities and Social Sciences (subcategory: Literature).

Back to List of figures

[Figure 384]

#### 人文社会科学: 文献学

Question: <图片 1>所示的文献基本检索输入方式是序号检索 Option:

(A) 刊名检索 (B) 分类检索 (C) 序号检索 (D) 高级检索

|[Figure 385]|
|---|

<图片 1>

GPT-4V: 根据图片中显示的界面，这是一个高级检索的界面，可以看到有多个检索条件组合的选 项，如“主题”、“篇名”、“作者”等，并且可以通过逻辑运算符（如AND）来组合这些条件。图 片中没有直接显示“序号检索”的选项，因此根据图片提供的信息，最可能正确的选项是：

(D) 高级检索"

Ground Truth: (D) 高级检索

- Figure B159: A sample correct case of Humanities and Social Sciences (subcategory: Literature).

Back to List of figures

[Figure 386]

#### 人文社会科学: 文献学

Question: <图片 1>所示的文献基本检索输入方式是__ Option:

(A) 刊名检索 (B) 分类检索 (C) 单位检索 (D) 高级检索

|[Figure 387]|
|---|

<图片 1>

GPT-4V: 根据图片中的信息，我们可以看到搜索框中输入了“中国农业大学TP309”，并且搜索结果 显示了与这个关键词相关的文献列表。这表明进行的是特定关键词的检索，但图片中没有直接显示这 些关键词是用于刊名、分类、单位还是其他高级检索条件。

然而，由于“中国农业大学”是一个单位名称，而“TP309”可能是一个分类号，这暗示了使用了单 位名称和分类号进行检索。在没有更多信息的情况下，我们可以假设这是一个结合了单位和分类的检 索方式，但这种组合在选项中并未直接给出。

因此，根据图片中的信息，最可能正确的选项是： (C) 单位检索

这是因为“中国农业大学”是一个单位名称，而搜索结果与这个单位相关。尽管“TP309”看起来像 一个分类号，但在提供的选项中没有明确的“分类检索”选项，所以我们只能选择最符合图片信息的 选项。

Error Category: Perceptual Error Error Reason:

- 1. In the picture, it is "中图分类号TP309" instead of "中国农业大学TP309";
- 2. The search method in pictures allows users to accurately find documents by combining multiple search conditions, so it is an advanced search.

Ground Truth: (D) 高级检索

- Figure B160: A sample error case of Humanities and Social Sciences (subcategory: Literature).

Back to List of figures

[Figure 388]

#### 人文社会科学: 文献学

Question: <图片 1>所示的信息检索中的__ Option:

(A) 正向索引 (B) 倒排索引 (C) 作者检索 (D) 刊名检索

|[Figure 389]|
|---|

<图片 1>

GPT-4V: 正确答案是：(B) 倒排索引

这张图片展示的是一个倒排索引的结构。在信息检索系统中，倒排索引是一种索引方法，它将文档的 内容映射到包含这些内容的文档ID。这种结构使得搜索引擎能够快速找到包含特定词汇的所有文档。 图中显示了不同的文档ID对应到包含的词汇，这是倒排索引的典型表示方式。

Error Category: Perceptual Error Error Reason:

- 1. Inverted Index: Maps keywords to a list of documents containing the keywords;
- 2. Forward index: Maps documents to the keywords they contain. The index method in the picture is the mapping of document ID to word, so it is a forward index.

Ground Truth: (A) 正向索引

- Figure B161: A sample error case of Humanities and Social Sciences (subcategory: Literature).

Back to List of figures

[Figure 390]

#### 人文社会科学: 心理学

Question: <图片 1>所示的心理学效应是__ option:

(A)半途效应 (B)边际效益递减效应 (C)比马龙效应 (D)安泰效应

|[Figure 391]|
|---|

<图片 1>

GPT-4V: 这张图片展示了随着消费数量的增加，对每一单位额外消费的满足度或幸福感逐渐降低的现 象。这种效应在心理学和经济学中被称为边际效用递减。因此，正确答案是： (B) 边际效益递减效应

Ground Truth: (B) 边际效益递减效应

- Figure B162: A sample correct case of Humanities and Social Sciences (subcategory: Psychology).

Back to List of figures

[Figure 392]

#### 人文社会科学: 心理学

Question: <图片 1>所示的心理学效应是是__ option:

(A)富兰克林效应 (B)鲶鱼效应 (C)南风效应 (D)损失厌恶效应

|[Figure 393]|
|---|

<图片 1>

GPT-4V: 正确答案是：(D) 损失厌恶效应 这张图片描绘了一个天平，一边有标记为“LOSS”的较重的部分，另一边有标记为“GAIN”的较轻的 部分。这代表了损失厌恶效应，即人们对损失的感受比对同等大小的收益的感受要强烈得多。

Ground Truth: (D) 损失厌恶效应

- Figure B163: A sample correct case of Humanities and Social Sciences (subcategory: Psychology).

Back to List of figures

[Figure 394]

#### 人文社会科学: 心理学

Question: <图片 1>所示的消费者心理是__ option:

(A)侥幸心理 (B)占便宜心理 (C)从众心理 (D)攀比心理

|<图片 1><br><br>[Figure 395]|
|---|

<图片 1>

GPT-4V: 这张图片展示了两位女士，一位看起来心情沮丧，心形气泡中写着“虚荣心”，而另一位看 起来很高兴，手里拿着购物袋。根据这些信息，我们可以推断出这张图片可能在表达关于消费者心理 的某种对比。 选项中： (A)侥幸心理通常指的是人们希望能够幸运地获得好处或避免坏事的心理，这与图片内容不符。 (B)占便宜心理是指人们喜欢获取比正常更有利的交易或优惠，这在图片中没有明显的迹象。 (C)从众心理是指人们倾向于跟随大众或朋友的选择和行为，图片中没有显示这种心理。 (D)攀比心理是指人们在社会地位、财富、消费等方面与他人比较，希望不落后于人甚至超越他人。 图片中的“虚荣心”气泡和两位女士的不同表情可能暗示了这种心理。 因此，最可能正确的选项是： (D) 攀比心理

Ground Truth: (D) 攀比心理

- Figure B164: A sample correct case of Humanities and Social Sciences (subcategory: Psychology).

Back to List of figures

[Figure 396]

#### 人文社会科学: 心理学

Question: 判断下面陈述对错：<图片 1>所示人物是吉布森

|[Figure 397]|
|---|

<图片 1>

GPT-4V: 抱歉，我无法确认图片中的人物是否是吉布森，因为我不能提供真实人物的身份信息。如 果您有其他问题或需要帮助，请告诉我。

Error Category: Reject to Answer Error Reason: The model declined to answer questions about personal information.

Ground Truth: 错

- Figure B165: A sample error case of Humanities and Social Sciences (subcategory: Psychology).

Back to List of figures

[Figure 398]

#### 人文社会科学: 心理学

Question: <图片 1>所示的心理学效应是__ option:

(A)半途效应 (B)责任分散效应 (C)旁观者效应 (D)安泰效应

|[Figure 399]|
|---|

<图片 1>

GPT-4V: 这张图片描绘了两个人在挖隧道，但他们之间只有一薄层土壤，而其中一个人已经放弃，如 果他再坚持一会儿就能到达另一边。这个图通常用来说明坚持和努力的重要性，以及在接近成功时不 要放弃的道理。然而，这并不直接对应于所给选项中的任何心理学效应。根据图片内容，没有足够的 信息来确定这与所给选项中的哪个心理学效应最相关。因此，我无法基于这张图片选择一个正确的答 案。如果这张图片是用来象征性地表示某种心理学效应，那么这种联系可能是隐喻性的，而不是直接 的。如果你能提供更多的上下文或解释图片是如何与这些效应相关联的，我可能能够提供更多帮助。

Error Category: Reject to Answer Error Reason: There is no obvious correlation between the image and the text, or the image does not have an obvious opt-out option, the model refuses to answer.

Ground Truth: (A)半途效应

- Figure B166: A sample error case of Humanities and Social Sciences (subcategory: Psychology).

Back to List of figures

[Figure 400]

#### 人文社会科学: 心理学

Question: <图片 1>该人认为认为构成意识的元素是（）", option:

(A)感觉、知觉、情感 (B)感觉、知觉、表象 (C)感觉、表象、情感 (D)感觉、记忆、表象

|[Figure 401]|
|---|

<图片 1>

GPT-4V:很抱歉，但我无法根据图片提供的信息来确定这个人的观点是什么。如果您能提供更多关于 这个人的背景信息或者他们的理论，我可能能够帮助您确定正确的答案。

Error Category: Perceptual Error Error Reason: The model doesn't understand who the person in the picture is.

Ground Truth: (C) 感觉、表象、情感

- Figure B167: A sample error case of Humanities and Social Sciences (subcategory: Psychology).

Back to List of figures

[Figure 402]

#### 人文社会科学: 心理学

Question: <图片 1>成人与儿童对一幅图画的知觉有很大的差别，年龄较小的儿童只能说出图画中主 要的部分，而成人能把握整个图画的意义，这说明了（） option:

(A)知觉的选择性 (B)知觉的理解性 (C)知觉的恒常性 (D)知觉的整体性

|[Figure 403]|
|---|

<图片 1>

GPT-4V: 这张图片展示了一个成人和一个儿童，成人似乎在向儿童展示或读一本书。根据题目描述， 成人能够理解图画的整体意义，而儿童只能说出图画中的主要部分。这表明成人的知觉更加整体和完 整。因此，正确答案是： (D) 知觉的整体性

Error Category: Reasoning Error Error Reason: The model correctly understood the knowledge and requirements of the problem, but it made mistakes in reasoning.

Ground Truth: (B) 知觉的理解性

- Figure B168: A sample error case of Humanities and Social Sciences (subcategory: Psychology).

Back to List of figures

[Figure 404]

#### 人文社会科学: 社会学

Question: 某股票收益分布的可能情况如下表<图片 1>，试计算该股票的年预期收益率( )

Option: (A) 0.20 (B) 0.24 (C) 0.21 (D) 0.18

|[Figure 405]|
|---|

<图片 1>

GPT-4V: 为了计算股票的年预期收益率，我们需要将每种可能情况的收益率与其相应的概率相乘， 然后将这些乘积相加。根据提供的表格，我们可以进行以下计算： 预期收益率 = (A的收益率 * A的 概率) + (B的收益率 * B的概率) + (C的收益率 * C的概率) 预期收益率 = (0.4 * 0.25) + (0.4 * 0.20) + (0.2 * 0.15) 预期收益率 = (0.1) + (0.08) + (0.03) 预期收益率 = 0.21 正确答案：(C) 0.21。

Ground Truth: (C) 0.21

- Figure B169: A sample correct case of Humanities and Social Sciences (subcategory: Sociology).

Back to List of figures

[Figure 406]

#### 人文社会科学: 社会学

Question: <图片 1>所示人物的经典条件反射学说内容：他在研究消化现象时，观察了狗的__分泌， 即对食物的一种反应特征。

|[Figure 407]|
|---|

<图片 1>

GPT-4V: 正确答案：唾液

Ground Truth: 唾液

- Figure B170: A sample correct case of Humanities and Social Sciences (subcategory: Sociology).

Back to List of figures

[Figure 408]

#### 人文社会科学: 社会学

Question: <图片 1>图中人物首次提出社会学这一概念的时间是（） option:

(A)1918年 (B)1928年 (C)1838年 (D)1948年

|[Figure 409]|
|---|

<图片 1>

GPT-4V: 正确答案是：(C) 1838年 社会学这一概念最早是由奥古斯特·孔德在1838年提出的

Ground Truth: (C)1838年

- Figure B171: A sample correct case of Humanities and Social Sciences (subcategory: Sociology).

Back to List of figures

[Figure 410]

#### 人文社会科学: 社会学

Question：如图<图片 1>所示是华联商厦某个月甲、乙、丙三种品牌彩电的销售量统计图，则甲、丙 两种品牌彩电该月的销售量之和为（）

option:

(A)50台 (B)65台 (C)75台 (D)95台

|[Figure 411]|
|---|

<图片 1>

GPT-4V: 根据提供的统计图，我们可以看到甲品牌彩电的销售量大约是40台，而丙品牌彩电的销售量 大约是35台。将这两个数值相加，我们得到甲、丙两种品牌彩电该月的销售量之和为40台加35台，等 于75台。 因此，正确答案是： (C) 75台

Ground Truth: (C) 75台

- Figure B172: A sample correct case of Humanities and Social Sciences (subcategory: Sociology).

Back to List of figures

[Figure 412]

#### 人文社会科学: 社会学

Question: 判断下面陈述对错：<图片 1>宗教是一种很重要的社会控制方式，应大力提倡。

|[Figure 413]|
|---|

<图片 1>

GPT-4V:这个问题的陈述包含了一个主观的观点，即宗教是一种很重要的社会控制方式，并且应该大 力提倡。关于宗教的作用和是否应该提倡，不同的人和社会有不同的看法。有些人可能认为宗教对于 维持社会秩序和提供道德指导是重要的，而另一些人可能认为宗教不应该用于社会控制或者应该保持 宗教信仰的个人自由。 由于这个问题涉及个人和文化价值观的差异，没有客观的“正确”或“错误”答案。因此，我无法给 出一个绝对的判断，但可以说这个问题的答案取决于个人的价值观和信仰。

Error Category: Reject to Answer Error Reason: When it comes to religion, the model refuses to make a judgment.

Ground Truth: 错

- Figure B173: A sample error case of Humanities and Social Sciences (subcategory: Sociology).

Back to List of figures

[Figure 414]

#### 人文社会科学: 社会学

Question:根据<图片 1>图中人物概括，男女在各方面的差别有（） option: (A)在语言表达能力方面， 男性胜于女性 (B)在视觉、 平衡觉能力方面， 女性胜于男性 (C)在逻辑思维、 数学能力方面， 男性较高于女性 (D)在触觉方面， 女性比男性敏感， 反应更快

|[Figure 415]|
|---|

<图片 1>

GPT-4V:很抱歉，但我无法根据图片内容来回答关于男性和女性在不同方面能力差异的问题。这些差 异通常是基于广泛的科学研究和统计数据，而不是单个图片中的人物。此外，性别能力差异的话题非 常复杂，涉及生物学、社会学、心理学等多个领域，并且存在大量争议和不同的研究结果。 如果你有关于图片中人物的其他问题，或者需要关于性别差异的一般信息，请告诉我，我会尽力帮助 你。

Error Category: Reject to Answer Error Reason: When it comes to gender, the model refuses to make a judgment.

Ground Truth: (C) 在逻辑思维、 数学能力方面， 男性较高于女性 (D) 在触觉方面， 女性比男性敏感， 反应更快

- Figure B174: A sample error case of Humanities and Social Sciences (subcategory: Sociology).

Back to List of figures

[Figure 416]

#### 人文社会科学: 社会学

Question: <图片 1>图示家庭结构为（） option:

(A)联合家庭 (B)主干家庭 (C)核心家庭 (D)完全家庭

|[Figure 417]|
|---|

<图片 1>

GPT-4V:这张图片展示的是一个家庭结构图。根据图中的信息，我们可以看到有两个成年人（通常表 示为较大的符号）和两个孩子（表示为较小的符号）。这个结构通常代表了一个父母和子女组成的家 庭单元。 选项中： (A)联合家庭通常指的是多代同堂的家庭结构，这里没有显示。 (B)主干家庭可能指的是有直系亲属和旁系亲属共同生活的家庭，这里也没有显示。 (C)核心家庭通常指的是最基本的家庭结构，即父母和子女。 (D)完全家庭通常指的是有父亲、母亲和子女的家庭结构。 根据这些定义，最符合这张图片的描述是(C) 核心家庭。 正确答案：(C) 核心家庭

Error Category: Reasoning Error Error Reason: The model misses the point in the figure, leading to incorrect inference. A joint family refers to a family with two or more couples in any generation, such as a family consisting of a parent and two or more generations of married children, or a family in which siblings do not separate after marriage.

Ground Truth: (A) 联合家庭

- Figure B175: A sample error case of Humanities and Social Sciences (subcategory: Sociology).

Back to List of figures

[Figure 418]

#### 技术与工程: 农业

Question: 如<图片 1>所示，该作物属于（） Option:

(A) 长日照作物 (B) 短日照作物 (C) 中日照作物 (D) 定日照作物

|[Figure 419]|
|---|

<图片 1>

GPT-4V: 图片显示的是小麦，它是一种典型的长日照作物。长日照作物是指那些需要较长白昼时间 才能开花结实的作物。因此，正确答案是： (A) 长日照作物 Ground Truth: (A) 长日照作物

- Figure B176: A sample correct case of Technology and Engineering (subcategory: Agriculture).

Back to List of figures

[Figure 420]

#### 技术与工程: 农业

Question: 下列关于<图片 1>中的植物产生的一种成花刺激物的叙述,错误的是() Option: (A) 它是一种小分子量蛋白质 (B) 它在叶片中合成 (C) 它的基因表达不受日照长度影响 (D) 它通过韧皮部运输到茎尖

|[Figure 421]|
|---|

<图片 1>

GPT-4V: 很抱歉，但我无法直接从图片中确定植物产生的成花刺激物的具体特性。然而，我可以提供 一些关于植物成花刺激物（例如弗洛里根）的一般信息，这可能有助于推断正确答案。

- (A) 成花刺激物的确切化学本质尚未完全确定，但有研究表明它可能是一种蛋白质或与蛋白质有关的 分子。
- (B) 成花刺激物通常在叶片中合成，这是因为叶片是植物感受外界光周期变化的主要部位。
- (C) 成花刺激物的合成通常受到日照长度的影响，这是因为植物的开花时间往往与光周期有关，这一 点在长日植物和短日植物的研究中得到了证实。
- (D) 成花刺激物被认为是通过植物的韧皮部从叶片运输到其他部位，如茎尖，以诱导开花。

根据这些信息，错误的选项最可能是 (C) 它的基因表达不受日照长度影响，因为成花刺激物的合成 通常是受到日照长度影响的。

Ground Truth: (C) 它的基因表达不受日照长度影响

- Figure B177: A sample correct case of Technology and Engineering (subcategory: Agriculture).

Back to List of figures

[Figure 422]

#### 技术与工程: 农业

Question: 下列构成蛋白质的氨基酸中,R 基团能够产生如<图片 1>所示反应的是（） Option: (A) 甘氨酸 (B) 酪氨酸 (C) 苯丙氨酸 (D) 甲硫酸

|[Figure 423]|
|---|

<图片 1>

GPT-4V: 图片显示的是ATP（三磷酸腺苷）向ADP（二磷酸腺苷）转化的过程，这个过程涉及到一个磷 酸基团的转移。在蛋白质的氨基酸中，有些氨基酸的R基团可以参与磷酸化反应，即接受磷酸基团。 在提供的选项中，酪氨酸（B选项）是已知可以被磷酸化的氨基酸之一。酪氨酸残基的羟基（-OH）可 以作为磷酸基团的受体，进行磷酸化反应。因此，正确答案是： (B) 酪氨酸

Ground Truth: (B) 酪氨酸

- Figure B178: A sample correct case of Technology and Engineering (subcategory: Agriculture).

Back to List of figures

[Figure 424]

#### 技术与工程: 农业

Question: <图片 1>所示的是__农业

Option: (A) 原始 (B) 传统 (C) 现代 (D) 观光

|[Figure 425]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是一个精心规划和设计的农业区域，具有整齐的田地和花坛，以及现代化的

建筑。这不太可能是原始或传统农业，因为这些通常不会有这么精心的布局和现代设施。同时，这个 场景也可能是为了吸引游客而设计的，因此可能具有观光农业的特点。因此，最可能正确的选项是：

(C) 现代 (D) 观光

正确答案是 (C) 现代 和 (D) 观光。

Error Category: Perceptual Error Error Reason: Modern agriculture pays more attention to production efficiency and technology application, aiming to improve food production and quality, while tourism agriculture focuses more on providing agricultural experience and education, adding economic value to agriculture. There are obviously many tourists in the picture, so choose D instead of C.

Ground Truth: (D) 观光

- Figure B179: A sample error case of Technology and Engineering (subcategory: Agriculture). Back to List of figures

[Figure 426]

#### 技术与工程: 农业

Question: <图片 1>所示的6是__

Option: (A) 瓣胃 (B) 皱胃 (C) 网胃 (D) 瘤胃

|[Figure 427]|
|---|

<图片 1>

GPT-4V: 这张图片展示的是牛的消化系统的示意图。在牛的胃部分，有四个室：瘤胃、网胃、瓣胃和 皱胃。根据图中的数字6的位置，它指向的是牛的第二个胃室，即网胃。 正确答案是： (C) 网胃

Error Category: Reasoning Error Error Reason: The omasum is the third stomach of ruminants. The regurgitated food will directly enter the omasum for further digestion. The flaps on the inner wall of the omasum are folded into book pages and contain digestive enzymes that can break down plant fibers into glucose and transport them to the intestines for absorption.

Ground Truth: (A) 瓣胃

- Figure B180: A sample error case of Technology and Engineering (subcategory: Agriculture). Back to List of figures

[Figure 428]

#### 技术与工程: 农业

Question: 豌豆象<图片 1>属于()昆虫。 Option:

(A) 单食性 (B) 寡食性 (C) 多食性 (D) 腐食性

|[Figure 429]|
|---|

<图片 1>

GPT-4V: 这张图片展示了一种昆虫，但是仅凭图片无法确定这种昆虫的食性。豌豆象是一种甲虫，但

是没有足够的信息来判断它是单食性、寡食性、多食性还是腐食性。通常，豌豆象属于害虫，它们可 能会专门攻击某些植物，如豌豆植物，这可能意味着它们是单食性或寡食性。然而，为了给出一个准 确的答案，需要更多的生态和生物学背景信息。在没有这些信息的情况下，我无法确定正确答案。

Error Category: Lack of Knowledge Error Reason: Pea weevils are monophagous insects, mainly because of their food specialization, specializing in the seeds of pea plants.

Ground Truth: (A) 单食性

- Figure B181: A sample error case of Technology and Engineering (subcategory: Agriculture). Back to List of figures

[Figure 430]

#### 技术与工程: 农业

Question: <图片 1>所示的是减轻作物病虫草的轮作方式中的__ Option:

(A) 复种轮作 (B) 水旱轮作 (C) 作物轮作 (D) 换茬式轮作

|[Figure 431]|
|---|

<图片 1>

GPT-4V: 这张图片显示了两种不同的作物种植在同一块土地上，一种作物似乎是玉米，而另一种作物 是低矮的植物，可能是豆类或其他类型的作物。这种种植方式看起来像是一种轮作方式，旨在通过种 植不同类型的作物来打断病虫害的生命周期，提高土壤的肥力，并减少对化肥的依赖。

根据选项和图片信息，最合适的答案是： (C) 作物轮作

这是因为作物轮作通常涉及在不同的季节或年份更换不同类型的作物，以达到上述的农业效益。其他 选项如复种轮作、水旱轮作和换茬式轮作没有足够的信息来直接从图片中确认。因此，最可能正确的 选项是(C) 作物轮作。

Error Category: Lack of Knowledge Error Reason: Multiple cropping: The practice of growing two or more crops on the same land in one year. In the picture, two crops are grown on dry land at the same time, so the correct answer is A.

Ground Truth: (A) 复种轮作

- Figure B182: A sample error case of Technology and Engineering (subcategory: Agriculture). Back to List of figures

[Figure 432]

#### 技术与工程:建筑学

Question: <图片 1>体系的几何组成为（） Option: (A) 几何不变无多余约束 (B) 几何不变有多余约束 (C) -1 (D) -2

|[Figure 433]|
|---|

<图片 1>

GPT-4V: 很抱歉，但我无法直接提供这个特定的机械系统的几何组成类型，因为这需要对机械工程的 专业知识进行分析，而这张图片没有提供足够的信息来确定其几何组成。通常，这类问题需要对机构 的自由度、约束类型以及其运动能力进行分析。如果你能提供更多关于这个系统的背景信息或者其工 作原理的描述，我或许能够帮助你更好地理解它的分类。

Error Category: Reject to Answer Error Reason: While GPT-4V does possess the capability of image recognition, in this specific context, it did not fully leverage this ability to analyze the image content and provide an answer. The model chose to avoid directly answering the question and instead emphasized the need for more background information. This might be due to the model's strategy of being overly cautious when dealing with image-based multiple-choice questions, thus avoiding making direct judgments about the image content.

Ground Truth: (B) 几何不变有多余约束

- Figure B183: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 434]

#### 技术与工程:建筑学

Question: <图片 1>体系的几何组成为（） Option: (A) 几何不变，无多余联系 (B) 几何不变，有多余联系 (C) 瞬变 (D) 常变

|[Figure 435]|
|---|

<图片 1>

GPT-4V: 很抱歉，但我无法直接提供这个特定机械系统的几何组成类型。通常，这类问题需要对机械 系统的运动学和动力学有深入的了解，以及对图示中结构的约束和自由度进行分析。在没有具体分析 的情况下，我无法确定正确答案。如果你能提供更多关于这个系统的背景信息或其工作原理的细节， 我可能能够帮助你更好地理解这个问题。

Error Category: Reject to Answer Error Reason:GPT-4V chose not to answer this question. The model indicated that it cannot directly provide the specific geometric composition types of a mechanical system because this typically requires a deep understanding of the kinematics and dynamics of the mechanical system, as well as an analysis of the constraints and degrees of freedom of the structures depicted in the illustration. However, this response approach demonstrates the limitations of the model when dealing with image-based questions.

Ground Truth: (D) 常变

- Figure B184: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 436]

#### 技术与工程:建筑学

Question: <图片 1>体系为：（） Option: (A) 几何不变无多余约束 (B) 几何不变有多余约束 (C) 几何常变 (D) 几何瞬变

|[Figure 437]|
|---|

<图片 1>

GPT-4V: 抱歉，由于图片中的信息不足以确定一个明确的答案，我无法直接提供正确选项。但是，根 据图片中显示的结构，它似乎是一个机械系统或结构，可能是一个机构或框架。通常，这类问题涉及 到机械工程领域，特别是机构理论和运动学。

在机械工程中，"几何不变"通常指的是一个机构在运动过程中保持其几何形状不变。如果一个机构在 运动时保持某些几何特性不变，但又有多余的约束使得其运动受限，那么它可能被描述为"几何不变 有多余约束"。如果机构的几何特性在运动中会发生变化，那么它可能被描述为"几何常变"。"几何瞬 变"则可能指的是机构在特定瞬间的几何变化。

由于我无法看到图片中的详细信息，我无法确定这个结构是否有多余的约束或者它的几何特性是否会 随时间变化。因此，我无法确定正确答案。如果你能提供更多关于这个结构的背景信息或者它的运动 特性，我可能能够帮助你更准确地判断。

Error Category: Reject to Answer, Perceptual Error Error Reason: Firstly, the model stated that it couldn't determine the correct answer due to the inability to view detailed information in the image. This suggests that the model believes it cannot access or interpret image content correctly, leading to a choice of not answering the question. In reality, GPT-4V has image recognition capabilities, so it should have been able to analyze image content and provide a response based on it. This indicates a perceptual limitation in the model when dealing with image-related questions.

Secondly, because the model did not correctly interpret the image content, it couldn't engage in reasonable inference based on the image information. The model's response primarily relied on the understanding of the text in the question, without incorporating key information from the image. This reflects that in tasks involving a combination of text and images, the model failed to effectively integrate these two sources of information, leading to a perceptual error.

Ground Truth: (A) 几何不变无多余约束

- Figure B185: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 438]

#### 技术与工程:建筑学

Question: <图片 1>平面体系的计算自由度等于_。

|[Figure 439]|
|---|

<图片 1>

GPT-4V: 这个问题涉及到机械工程领域中的结构分析，特别是平面刚架结构的自由度计算。计算平面 刚架结构的自由度通常使用库兹曼方程（Kutzbach criterion），该方程为： 自由度 F = 3n - 3j - c 其中：

- - n 是结构中的节点数
- - j 是结构中的杆件数
- - c 是结构中的约束反应数（支座反力数）

在提供的图片中，我们需要数一下节点、杆件和约束。但由于图片的分辨率较低，我无法清晰地识别 具体的节点数、杆件数和约束数。如果你能提供这些信息，我可以帮你计算自由度。如果没有这些信 息，我无法提供一个确切的答案。

Error Category: Perceptual Error Error Reason: In its response, GPT-4V mentioned that it couldn't clearly identify the number of nodes, members, and constraints due to the low image resolution. This indicates that the model attempted to answer the question based on visual information, but due to the limitations of its visual recognition capabilities, it was unable to accurately interpret the content in the image. This is a typical perception error, where the model's interpretation of visual information is either inaccurate or insufficient.

Ground Truth: -2

- Figure B186: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 440]

#### 技术与工程:建筑学

Question: <图片 1>体系是（） Option: (A) 无多余联系的几何不变体系 (B) 几何可变体系 (C) 有多余联系的几何不变体系 (D) 瞬变体系

|[Figure 441]|
|---|

<图片 1>

GPT-4V: 很抱歉，由于图片的内容是技术性质的，并且需要专业知识来解释，我无法直接看到图片内 容。因此，我无法提供一个确定的答案。通常，这类问题涉及到结构分析，其中包括判断一个结构是 否是几何不变的、是否有多余的联系，或者是否是瞬变体系。这需要对结构力学和工程原理有深入的 了解。

在没有具体分析这个结构的能力的情况下，我建议咨询一个结构工程师或者使用相关的结构分析软件 来确定正确的答案。如果你能提供更多的背景信息或者详细描述这个结构的特点，我或许能够帮助你 更好地理解这些选项。

Error Category: Perceptual Error, Reject to Answer Error Reason: First, regarding the perception error, GPT-4V actually has image recognition capabilities, but in this response, it incorrectly stated that it cannot see or understand image content. This indicates that the model did not correctly use its image processing abilities to parse the content of the image in the question, resulting in an incomplete understanding of the question. Secondly, regarding the refusal to answer, after confirming that it cannot understand the image content, the model chose the path of not providing a specific answer. While exercising caution in situations where it cannot fully comprehend a question is reasonable, in such cases, the model should have made an attempt to use the information in its knowledge base to provide an answer that closely aligns with the question's request, or at least provide a reasonable guess based on textual descriptions.

Ground Truth: (A) 无多余联系的几何不变体系

- Figure B187: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 442]

#### 技术与工程:建筑学

Question: 如<图片 1>两块平板玻璃构成的空气劈尖，左边为棱边，用单色平行光垂直入射。若上面 的平板玻璃慢慢地向上平移，则干涉条纹（）

Option:

(A) 向棱边方向平移，条纹间隔变小 (B) 向棱边方向平移，条纹间隔变大 (C) 向棱边方向平移，条纹间隔不变 (D) 向远离棱边的方向平移，条纹间隔不变

|[Figure 443]|
|---|

<图片 1>

GPT-4V: 在这个问题中，我们考虑的是两块平板玻璃构成的空气劈尖，其中一块玻璃相对于另一块慢 慢向上平移。由于是单色平行光垂直入射，干涉条纹的形成是由于光波在劈尖的上下表面反射后发生 干涉。当上面的玻璃板向上平移时，劈尖的厚度会在棱边附近增加，这意味着光波在劈尖内部的光程 差会增加。

由于光程差增加，原本在某一位置的明条纹（或暗条纹）会向劈尖的薄部分移动，即向远离棱边的方 向平移。条纹间隔（即干涉条纹的间距）是由劈尖的角度和光的波长决定的，而在这个过程中，劈尖 的角度和光的波长都没有改变，因此条纹间隔不会变化。

因此，正确答案是： (D) 向远离棱边的方向平移，条纹间隔不变

Error Category: Reasoning Error Error Reason: The GPT-4V model exhibited a reasoning error in analyzing the question. In this problem, the key is to understand how interference fringes change when a slab of glass, surrounded by air, is displaced vertically under monochromatic parallel light with perpendicular incidence. The model's response incorrectly suggested that as the upper glass slab is moved upwards, the thickness of the wedge near the edge increases, leading to an increase in optical path difference, causing the interference fringes to shift away from the edge. This is an incorrect inference.

Ground Truth: (C) 向棱边方向平移，条纹间隔不变

- Figure B188: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 444]

#### 技术与工程:建筑学

Question: 如<图片 1>所示，已知F=100N,q=5N/m,R=5cm,OA=10cm,BC=5cm。则该力系对I点的合力矩 为（）。

Option: (A) M1=1000N·cm（顺时针） (B) M1=1000N·cm（逆时针） (C) M1=500N·cm（逆时针） (D) M1=500N·cm（顺时针）

|[Figure 445]|
|---|

<图片 1>

GPT-4V: 由于我无法直接进行物理实验或测量，我将基于图像中提供的信息和物理学原理来解答这个 问题。 首先，我们需要计算力F和力q对点I的力矩。

对于力F，它的力矩是力乘以力臂。力臂是力的作用线到转动轴（这里是点I）的垂直距离。在这个情 况下，力F的作用线是沿着OA延伸的，所以力臂是OA的长度加上AB的长度（因为I点在B点的垂直上 方）。OA是10cm，AB是BC的一半，即2.5cm。所以力臂是12.5cm。因此，力F对点I的力矩是100N * 12.5cm = 1250N·cm。由于力F是向右的，它会使物体绕I点逆时针旋转。

对于均布力q，它的合力可以看作是作用在其作用线中点的单一力。在这个情况下，均布力的作用线 是BC，所以合力作用在B和C的中点，即2.5cm/2 = 1.25cm处。均布力的大小是q乘以BC的长度，即 5N/m * 5cm = 25N。因此，力q对点I的力矩是25N * 1.25cm = 31.25N·cm。由于力q是向下的，它会 使物体绕I点顺时针旋转。

将两个力矩相加，我们需要注意它们的旋转方向。力F产生的是逆时针力矩，而力q产生的是顺时针力 矩。因此，合力矩是两者的差值，即1250N·cm - 31.25N·cm = 1218.75N·cm（逆时针）。

这个结果不在所给的选项中，但是根据题目的选项，我们可以选择最接近的答案。因此，正确的选项 是：

(B) M1=1000N·cm（逆时针） 请注意，这个答案是基于所给信息的近似计算，实际的答案可能略有不同。

Error Category: Reasoning Error Error Reason: The GPT-4V model made a reasoning error when analyzing this physics problem. The problem requires calculating the resultant moment of the given force system about point I. The model incorrectly calculated the moment of force F about point I by adding the length of OA to the length of AB for the lever arm.

Ground Truth: (D) M1=500N·cm（顺时针）

- Figure B189: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 446]

#### 技术与工程:建筑学

Question: 梁的弯矩图如<图片 1>所示，则梁的最大剪力是（）

Option: (A) 0.5F (B) F (C) 1.5F (D) 2F

|[Figure 447]|
|---|

<图片 1>

GPT-4V: 在这个问题中，我们需要根据弯矩图来确定梁的最大剪力。弯矩图显示了梁上各点的弯矩值， 而剪力图可以通过弯矩图的斜率来确定。在弯矩图中，弯矩的最大增加或减少发生在剪力为零的位置， 而剪力的最大值通常出现在弯矩图的最大或最小点附近。

从给定的弯矩图中，我们可以看到在点C和点D处弯矩为零，这意味着在这些点处剪力会改变符号。在 A到C的区间内，弯矩线性增加，而在C到D的区间内，弯矩线性减少。在D到B的区间内，弯矩再次线性 增加。

为了确定最大剪力，我们需要找到弯矩变化率最大的区间。在A到C区间，弯矩从0增加到0.5Fa，变化 率为0.5Fa/a。在C到D区间，弯矩从0.5Fa减少到0，变化率为-0.5Fa/a。在D到B区间，弯矩从0增加到 1.5Fa，变化率为1.5Fa/a。

因此，最大的弯矩变化率出现在D到B的区间，这意味着最大剪力也出现在这个区间。由于弯矩变化率 为1.5Fa/a，我们可以推断最大剪力为1.5F。

正确答案是： (C) 1.5F

Error Category: Reasoning Error Error Reason:GPT-4V correctly pointed out that the moment diagram can be used to determine the shear force diagram and that the maximum shear typically occurs near the maximum or minimum points of the moment diagram. However, there was a deviation in the model's reasoning when analyzing specific changes in the moment variation rate.

Ground Truth: (D) 2F

- Figure B190: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 448]

#### 技术与工程:建筑学

Question: <图片 1>体系的几何组成为（） Option: (A) 几何不变无多余约束 (B) 几何不变有多余约束 (C) 瞬变体系 (D) 可变体系

|[Figure 449]|
|---|

<图片 1>

GPT-4V: 这张图片显示的是一个结构体系，但由于我无法进行实际的力学分析，我不能确定它是否具 有多余约束。通常，一个结构体系是否具有多余约束取决于其杆件和节点的配置以及支撑条件。结构 工程师会使用静力学的方法来确定结构是否是静定的、超静定的还是不稳定的。

在这个情况下，如果结构的杆件和节点配置导致内力可以在不改变结构几何形状的情况下唯一确定， 那么它可能是几何不变无多余约束的（选项A）。如果结构的杆件和节点配置导致内力的计算需要额 外的条件，那么它可能是几何不变有多余约束的（选项B）。如果结构可以在外力作用下改变其形状 而不引起内力的变化，那么它可能是瞬变体系（选项C）。如果结构可以在外力作用下改变其形状并 引起内力的变化，那么它可能是可变体系（选项D）。

由于缺乏具体的力学分析，我无法给出一个确定的答案。然而，如果你能提供更多的上下文或细节， 我可能能够帮助你更好地理解这个结构体系的性质。

Error Category: Reasoning Error Error Reason: GPT-4V demonstrated a certain level of domain knowledge when reasoning about possible options, but it failed to accurately apply this knowledge to specific questions. The model correctly described the general characteristics of several structural systems but couldn't effectively judge specific options based on image information. This disconnect between understanding and application may be due to the model lacking sufficient contextual information or because its knowledge base lacks extensive expertise in this area.

Ground Truth: (B) 几何不变有多余约束

- Figure B191: A sample error case of Technology and Engineering (subcategory: Architecture).

Back to List of figures

[Figure 450]

#### 技术与工程: 计算机科学

Question: 下列给定的关键字输入序列中，不能生成如下二叉排序树的是<图片 1> Option:

(A) 4,5,2,1,3 (B) 4,5,1,2,3

(C) 4,2,5,3,1 (D) 4,2,1,3,5

|[Figure 451]|
|---|

<图片 1>

GPT-4V: 抱歉，由于我的限制，我无法直接查看图片。但是，我可以帮助你分析这个问题。 二叉排序树（也称为二叉搜索树）是一种特殊的二叉树，它满足以下性质：

- - 每个节点包含一个键。
- - 每个节点的左子树只包含小于该节点键的节点。
- - 每个节点的右子树只包含大于该节点键的节点。
- - 左右子树也必须分别是二叉排序树。 根据这些性质，我们可以尝试根据给定的关键字序列构建二叉排序树，并检查它们是否能生成与题目中图片相 同的树结构。

- (A) 4,5,2,1,3

- - 4 是根节点。
- - 5 大于 4，所以它是 4 的右子节点。 - 2 小于 4，所以它是 4 的左子节点。
- - 1 小于 4 且小于 2，所以它是 2 的左子节点。 - 3 大于 2 且小于 4，所以它是 2 的右子节点。

- (B) 4,5,1,2,3

- - 4 是根节点。
- - 5 大于 4，所以它是 4 的右子节点。
- - 1 小于 4，所以它是 4 的左子节点。
- - 2 大于 1 且小于 4，所以它是 1 的右子节点。
- - 3 大于 2 且小于 4，所以它是 2 的右子节点。

- (C) 4,2,5,3,1

- - 4 是根节点。
- - 2 小于 4，所以它是 4 的左子节点。
- - 5 大于 4，所以它是 4 的右子节点。
- - 3 大于 2 且小于 4，所以它是 2 的右子节点。
- - 1 小于 2 且小于 4，所以它是 2 的左子节点。

- (D) 4,2,1,3,5

- - 4 是根节点。
- - 2 小于 4，所以它是 4 的左子节点。
- - 1 小于 2 且小于 4，所以它是 2 的左子节点。
- - 3 大于 2 且小于 4，所以它是 2 的右子节点。
- - 5 大于 4，所以它是 4 的右子节点。 根据上述分析，我们可以看到选项 (B) 生成的二叉排序树与其他选项不同，因为在 (B) 中，数字 2 是数字 1 的右子节点，而在其他选项中，数字 3 是数字 2 的右子节点。因此，如果图片中的二叉排序树结构与 (A)、 (C) 和 (D) 描述的结构相匹配，那么 (B) 就不能生成题目中的二叉排序树。正确答案是 (B)。

Ground Truth: (B) 4,5,1,2,3

- Figure B192: A sample correct case of Technology and Engineering (subcategory: Computer Science).

Back to List of figures

[Figure 452]

#### 技术与工程: 计算机科学

Question: 某网络拓扑如<图片 1>所示，路由器R1只有到达子网 192.168.1.0/24 的路由。为使R1可 以将 IP 分组正确地路由到图中所有子网，则在R1中需要增加的一条路由（目的网络，子网掩码，下 一跳）是

Option: (A) 192.168.2.0,255.255.255.128,192.168.1.1 (B) 192.168.2.0,255.255.255.0,192.168.1.1 (C) 192.168.2.0,255.255.255.128,192.168.1.2 (D) 192.168.2.0,255.255.255.0,192.168.1.2

|[Figure 453]|
|---|

<图片 1>

GPT-4V: 根据提供的网络拓扑图，我们可以看到路由器R1已经有到达子网192.168.1.0/24的路由。 为了让R1能够将IP分组正确地路由到图中的所有子网，我们需要考虑如何到达子网192.168.2.0/25和 192.168.2.128/25。

从图中可以看出，这两个子网都是通过路由器R2来访问的，而R2的接口IP地址是192.168.1.2。因 此， R1需要添加一条到192.168.2.0网络的路由，并且下一跳应该是R2的接口IP地址，即

- 192.168.1.2。 选项中，(A)和(C)的子网掩码是255.255.255.128，这意味着它们只能覆盖到192.168.2.0/25或
- 192.168.2.128/25中的一个子网，而不是两个。而(B)和(D)的子网掩码是255.255.255.0，这可以覆 盖到192.168.2.0/24，即包括了192.168.2.0/25和192.168.2.128/25两个子网。

因此，我们需要选择一个子网掩码为255.255.255.0，下一跳为192.168.1.2的选项。这样的选项是 (D) 192.168.2.0，255.255.255.0，192.168.1.2。

正确答案：(D) 192.168.2.0，255.255.255.0，192.168.1.2。 Ground Truth: (D) 192.168.2.0,255.255.255.0,192.168.1.2

- Figure B193: A sample correct case of Technology and Engineering (subcategory: Computer Science).

Back to List of figures

[Figure 454]

#### 技术与工程: 计算机科学

Question: 使用Weiler-Atherton算法讲行多边形裁减，顶点按逆时针顺序表示的多边形<图片 1>P的 各个顶点中，哪些属于进点（灰色为裁减窗口）？（ ）

Option: (A)P1, P2, P3 (B)1,3 (C)2,4 (D)1,2,3,4

|[Figure 455]|
|---|

<图片 1>

GPT-4V: 在Weiler-Atherton算法中，进点是指多边形的边从外部进入裁剪窗口的顶点。根据提供的 图片，我们可以看到多边形P的顶点P1, P2, P3, 和P4，以及它们与裁剪窗口的关系。 从图片中可以看出：

- - 顶点P1是从外部进入裁剪窗口的，因此它是一个进点。
- - 顶点P2是从内部离开裁剪窗口的，因此它不是进点。
- - 顶点P3是从外部进入裁剪窗口的，因此它也是一个进点。
- - 顶点P4是从内部离开裁剪窗口的，因此它不是进点。 因此，属于进点的顶点是P1和P3。 正确答案是：(B) 1, 3

Error Category: Textual Understanding Error Reason: GPT-4V does not understand that vertices are represented in counterclockwise order, and instead answers according to clockwise order.

Ground Truth: (C) 2,4

- Figure B194: A sample error case of Technology and Engineering (subcategory: Computer Science).

Back to List of figures

[Figure 456]

#### 技术与工程: 计算机科学

Question: 设有一个递归算法如下<图片 1>，下面叙述正确的是（） Option: (A) 计算fact（n）需要执行n次递归 (B) fact（7）=5040 (C) 此地归算法最多只能计算到fact（8） (D) 以上结论都不对

|[Figure 457]|
|---|

<图片 1>

GPT-4V: 这是一个计算阶乘的递归算法。根据算法的定义，我们可以评估每个选项：

- (A) 计算fact（n）需要执行n次递归 - 这是正确的。每次调用函数都会减少n的值，直到n为0，因此 会执行n次递归。
- (B) fact（7）=5040 - 这也是正确的。7的阶乘是7 * 6 * 5 * 4 * 3 * 2 * 1，等于5040。
- (C) 此递归算法最多只能计算到fact（8） - 这是错误的。递归算法理论上可以计算任何正整数的阶 乘，只要系统的栈空间足够大。没有足够的信息来确定这个算法是否有栈溢出的风险，或者是否有其 他限制。
- (D) 以上结论都不对 - 由于(A)和(B)是正确的，(D)选项自然是错误的。 正确答案是：(A) 计算fact（n）需要执行n次递归 和 (B) fact（7）=5040。

Error Category: Lack of Knowledge Error Reason: (A) Computing fact(n) requires performing n+1 recursions, because in addition to the original call and each call for decrementing n, there is also a base case call when n reaches 0.

Ground Truth: (B) fact（7）=5040

- Figure B195: A sample error case of Technology and Engineering (subcategory: Computer Science).

Back to List of figures

[Figure 458]

#### 技术与工程: 电子学

Question:在<图片 1>所示的稳压电路中，已知 𝑈𝑈𝑍𝑍 = 1.6𝑉𝑉，则 𝑈𝑈0 为： Option:

(A) 9V (B) 1.6V (C) 7.4V (D) 10.6V

|[Figure 459]|
|---|

<图片 1>

GPT-4V: 在这个稳压电路中，W7809是一个固定输出电压的稳压器，其输出电压通常为9V。然而，电 路中还包括了一个稳压二极管Dz，其稳定电压为$U_{Z}$=1.6V。稳压二极管Dz与稳压器的输出并联， 这意味着输出电压 𝑈𝑈0 将是稳压器的输出电压与稳压二极管的稳定电压之和。

因此， 𝑈𝑈0 = 9V（稳压器输出）+ 1.6V（稳压二极管稳定电压）= 10.6V。 正确答案是： (D) 10.6V

Ground Truth: (D) 10.6V

- Figure B196: A sample correct case of Technology and Engineering (subcategory: Electronics).

Back to List of figures

[Figure 460]

#### 技术与工程: 电子学

Question:已知某电路的真值表如<图片 1>所示，该电路逻辑表达式为 Option: (A) F=C (B) F=ABC (C) F=AB+C (D) 都不是

|[Figure 461]|
|---|

<图片 1>

GPT-4V: "根据提供的真值表，我们可以观察到输出F在以下情况为1：

- 1. 当A=0, B=0, C=1时（第二行）
- 2. 当A=0, B=1, C=1时（第四行）
- 3. 当A=1, B=0, C=1时（第六行）
- 4. 当A=1, B=1, C=0时（第七行）
- 5. 当A=1, B=1, C=1时（第八行） 我们可以看到，只有C为1时，F才可能为1，但是C为1并不是F为1的充分条件，因为当A=1, B=0, C=1 时（第五行），F为0。所以选项(A) F=C是错误的。 同时，我们也可以看到，当A、B和C同时为1时，F为1，但是这也不是唯一的情况，因此选项(B) F=ABC也是错误的。 选项(C) F=AB+C，我们可以验证一下：

- - 当A=0, B=任意, C=1时，F=1，符合真值表的第二行和第六行。
- - 当A=1, B=1, C=任意时，F=1，符合真值表的第七行和第八行。 因此，根据真值表，最可能正确的逻辑表达式是(C) F=AB+C。所以正确答案是(C) F=AB+C

Ground Truth: (C) F=AB+C

- Figure B197: A sample correct case of Technology and Engineering (subcategory: Electronics).

Back to List of figures

[Figure 462]

#### 技术与工程: 电子学

Question:半加器逻辑符号如图 <图片 1>所示， 当 A “1”， B “1”时， C 和 S 分别为： Option:

(A)C=0 S=0 (B) C=0 S=1 (C) C=1 S=0 (D) C=1 S=1

|[Figure 463]|
|---|

<图片 1>

GPT-4V: 抱半加器是一种电路，它可以对两个一位二进制数进行相加，产生一个和（S）和一个进位 （C）。在半加器中，当两个输入A和B都为1时，和（S）将为0，进位（C）将为1。 因此，正确答案是： (C) C=1 S=0。

Ground Truth: (C) C=1 S=0

- Figure B198: A sample correct case of Technology and Engineering (subcategory: Electronics).

Back to List of figures

[Figure 464]

#### 技术与工程: 电子学

Question: 如<图片 1>所示，二极管为同一型号的理想元件，电阻R=4kΩ，电位𝑈𝑈𝐴𝐴 =1V， 𝑈𝑈B =3V，则 电位uF等于（）。 Option:

(A) 1V (B) 3V (C) 12V (D) 5V

|[Figure 465]|
|---|

<图片 1>

GPT-4V:在这个电路中，我们有两个理想二极管和一个电阻。理想二极管的特性是在正向偏置时导通， 而在反向偏置时截止。由于𝑈𝑈𝐴𝐴 = 1𝑉𝑉，𝑈𝑈𝐵𝐵 = 3𝑉𝑉，我们可以推断出连接到𝑈𝑈𝐵𝐵的二极管将导通，因为它 有更高的正向偏置电压，而连接到𝑈𝑈𝐴𝐴的二极管将截止。 因此，电位𝑢𝑢𝐹𝐹将等于𝑈𝑈𝐵𝐵的电位，因为这是通过导通的二极管所连接的电位。所以𝑢𝑢𝐹𝐹 = 3𝑉𝑉。 正确答案是： (B) 3V

Error Category: Reasoning Error Error Reason: GPT-4V correctly understood the requirements of the question and the configuration of the circuit but made a mistake in the reasoning process. It erroneously assumed that the conducting diode was the one connected to the higher potential 𝑈𝑈𝐵𝐵, ignoring that the direction of current through a diode is determined by the potential difference, not just the absolute value of potential. In this circuit, since both 𝑈𝑈𝐴𝐴 and 𝑈𝑈𝐵𝐵 are positive and diodes only conduct when forwardbiased, the diode connected to the lower potential 𝑈𝑈𝐴𝐴 will conduct, causing the diode connected to 𝑈𝑈𝐵𝐵 to be cut-off. Therefore, the potential 𝑢𝑢𝐹𝐹 should be equal to the potential of 𝑈𝑈𝐴𝐴, which is 1V. Hence, the correct answer is (A) 1V.

Ground Truth: (A) 1V

- Figure B199: A sample error case of Technology and Engineering (subcategory: Electronics). Back to List of figures

[Figure 466]

#### 技术与工程: 电子学

Question: 求图<图片 1>所示电路中的电流 𝐼𝐼1 _A。

|[Figure 467]|
|---|

<图片 1>

GPT-4V:为了求解电路中的电流𝐼𝐼1，我们可以使用基尔霍夫电流定律（KCL），该定律指出，任何电路 节点进入和离开的电流之和等于零。 首先，我们可以观察节点C。根据KCL，进入节点C的电流等于离开节点C的电流。我们有：10A 进入节 点C，5A 和 𝐼𝐼1离开节点C。所以我们可以写出方程：10A = 5A + 𝐼𝐼1 解这个方程，我们得到： 𝐼𝐼1 = 10A - 5A 𝐼𝐼1 = 5A 因此，电流𝐼𝐼1是5A。

Error Category: Reasoning Error Error Reason: The solution correctly identifies the need to use Kirchhoff's Current Law (KCL) to find the current 𝐼𝐼1 but makes a reasoning error in the application of the law to node C. The given currents entering and leaving the node were not correctly accounted for, leading to an incorrect calculation of 𝐼𝐼1 The correct approach would involve writing and solving the KCL equation considering all currents entering and leaving the node, not just the 10A and 5A currents.

Ground Truth: 9

- Figure B200: A sample error case of Technology and Engineering (subcategory: Electronics). Back to List of figures

[Figure 468]

#### 技术与工程: 电子学

Question: 图<图片 1>所示电路中B点的电位_V。

|[Figure 469]|
|---|

<图片 1>

GPT-4V:为了确定电路中B点的电位，我们需要分析电路并应用基本的电路定律，如欧姆定律和基尔 霍夫电压定律。然而，由于我无法直接测量或看到电路的实际值，我将提供一个基于电路图的理论分 析。 在这个电路中，我们可以看到三个电阻R1、R2和R3，以及两个电压源，一个是+50V，另一个是-50V。 B点是R2电阻的上端。为了找到B点的电位，我们需要知道流过R2的电流I2的方向和大小。 由于电路图中没有提供足够的信息来直接计算电流，我们需要做一些假设。如果假设电路处于稳态条 件，那么流过R1和R3的电流（I1和I3）将分别从+50V和-50V的电源流出，并在B点汇合。由于电流在 电路中的任何节点的总流入电流等于总流出电流，我们可以假设流过R2的电流I2是I1和I3的代数和。 但是，没有具体的电流值或其他电路参数，我们无法准确计算B点的电位。如果我们有电流的值，我 们可以使用欧姆定律（V = IR）来计算流过R2的电压降，然后根据基尔霍夫电压定律确定B点的电位。 由于缺乏具体信息，正确答案是：无法确定B点的电位，因为需要更多的电路信息。

Error Category: Reasoning Error Error Reason: The error occurred because GPT-4V correctly understood the components and structure of the circuit from the image but failed to apply the correct reasoning process to determine the voltage at point B. It should have used the principles of electrical circuits, such as Ohm's law and Kirchhoff's voltage law, to establish equations based on the given values of resistances and voltages. By solving these equations, the voltage at point B could have been determined. Instead, GPT-4V claimed it could not determine the voltage at point B without additional information, which was a reasoning error since the necessary information was present in the circuit diagram.

Ground Truth: -14.286

- Figure B201: A sample error case of Technology and Engineering (subcategory: Electronics). Back to List of figures

[Figure 470]

#### 技术与工程: 能源和电力

Question:如<图片 1>所示,电源电动势E=60V,电源内阻R=2Ω,负载电阻R=18Ω,则电源内阻消耗功率 为___W。

|[Figure 471]|
|---|

<图片 1>

GPT-4V:为了计算电源内阻消耗的功率，我们首先需要计算电路中的总电流。电路的总电阻是电源内 阻𝑅𝑅0和负载电阻R的和，即𝑅𝑅𝑘𝑘𝑡𝑡𝑘𝑘𝑡𝑡𝑡𝑡 = 𝑅𝑅0 + 𝑅𝑅 = 20𝛺𝛺

根据欧姆定律，电流 I 可以通过电动势 E 除以总电阻𝑅𝑅𝑘𝑘𝑡𝑡𝑘𝑘𝑡𝑡𝑡𝑡来计算：

𝐸𝐸 𝑅𝑅𝑘𝑘𝑡𝑡𝑘𝑘𝑡𝑡𝑡𝑡

𝐼𝐼 =

= 3𝐴𝐴

R0

现在我们知道了电流，我们可以计算电源内阻 消耗的功率 P，功率可以通过电流的平方乘以电阻 来计算：

𝑃𝑃 = 𝐼𝐼2 × 𝑅𝑅 = 18𝑊𝑊 所以，电源内阻消耗的功率是 18W。

Ground Truth: (18)

- Figure B202: A sample correct case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 472]

#### 技术与工程: 能源和电力

Question:判断下面陈述对错：图<图片 1>是强制循环间接加热水箱-水罐系统，系统采用温差循环控 制原理控制热水水温

|[Figure 473]|
|---|

<图片 1>

GPT-4V:这张图片展示的是一个技术图纸，它描述了一个加热水箱-水罐系统的结构和工作原理。从 图中可以看到，系统包括热交换器、水泵、控制阀门、水箱和其他相关的管道和控制设备。根据图纸 上的信息和符号，可以推断这是一个用于加热和循环水的系统。

题目中的陈述是“图<图片 1>是强制循环间接加热水箱-水罐系统，系统采用温差循环控制原理控制 热水水温”。从图纸上可以看到，有一个循环泵（可能是标号2的部件），这表明系统确实使用了强 制循环来移动水。此外，系统中似乎包含了温度控制的元素，如温度传感器和控制阀，这些可能用于 根据温差来控制水温。

因此，根据图纸和题目描述，可以合理推断题目中的陈述是正确的。

Ground Truth: (对)

- Figure B203: A sample correct case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 474]

#### 技术与工程: 能源和电力

Question:图<图片 1>是_____，适用于屋面等高位设置贮热水箱的场所 Option: (A) 强制循环间接加热双罐系统 (B) 强制循环间接加热水箱-水罐系统 (C) 自然循环系统 (D) 分体式强制循环系统

|[Figure 475]|
|---|

<图片 1>

GPT-4V:这张图片展示的是一个太阳能热水系统的示意图。根据图中的元件布置和流向，可以看出这 是一个有泵的循环系统，因为有明显的泵（泵的符号）在系统中。自然循环系统通常不包含泵，因为 它依赖于热水的自然对流来循环。此外，系统中包含至少两个储存热水的容器，这表明它可能是一个 双罐系统。

因此，根据提供的选项和图中的信息，正确答案是： (A) 强制循环间接加热双罐系统 这个选项最符合图中所示的系统特点，即使用泵进行强制循环，并且有两个储热水箱。",

Error Category:Perceptual Error Error Reason:GPT-4V observed that the system contains at least two containers for storing hot water, and the process is a forced circulation indirect heating mode. However, in the picture, there is one water tank and one water jug, while GPT-4 incorrectly identified both as water jugs, thus leading to an erroneous selection.

Ground Truth: (B) 强制循环间接加热水箱-水罐系统

- Figure B204: A sample error case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 476]

#### 技术与工程: 能源和电力

Question:图<图片 1>为__安装在平屋面上的布置示例 Option: (A) 一体式太阳能集热器 (B) 分体式太阳能集热器 (C) 平板太阳能集热器 (D) 真空管太阳能集热器

|[Figure 477]|
|---|

<图片 1>

GPT-4V:这张图片展示的是一些安装在屋顶上的设备，它们看起来像是太阳能集热器。由于图片中的 集热器是平板状的，并且没有显示出真空管的特征，因此可以排除选项(D) 真空管太阳能集热器。至 于是一体式还是分体式，图片没有提供足够的信息来确定这一点。然而，由于集热器通常是平板状的， 我们可以合理推测这些是平板太阳能集热器。

正确答案是： (C) 平板太阳能集热器

Error Category:Perceptual Error Error Reason:GPT-4V correctly identified the basic types of items in the picture, but failed to capture the more critical information in the image: there are two solar collectors. Therefore, GPT-4V merely recognized the solar collectors and was unable to provide the correct answer: split-type solar collectors.

Ground Truth: (B)分体式太阳能集热器

- Figure B205: A sample error case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 478]

#### 技术与工程: 能源和电力

Question:图<图片 1>是“湿”式真空管集热器截面图，在“湿”式真空管中，水在由___组成的热虹 吸管中循环。

Option: (A)外玻璃管和储热水箱 (B) 外玻璃管和储热器 (C) 真空管和集热器 (D) 真空管和储热器

|[Figure 479]|
|---|

<图片 1>

GPT-4V:这张图片展示的是一个湿式真空管集热器的截面图。在这种系统中，水通常在集热器内部的 热虹吸管中循环。根据图示和常识，热虹吸管是位于真空管内部的，因此水是在真空管和集热器之间 循环的。

正确答案是： (C) 真空管和集热器

Error Category:Reasoning Error Error Reason:In this specific 'wet' vacuum tube solar collector design, water is heated inside the vacuum tube by absorbing solar energy, and this heat is directly transferred to a storage heater connected to the vacuum tube. However, GPT-4V did not consider this, which is a typical Reasoning Error.

Ground Truth: (D) 真空管和储热器

- Figure B206: A sample error case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 480]

#### 技术与工程: 能源和电力

Question:图<图片 1>是重水堆核电厂示意图，采用___作为中子慢化剂。其对热中子的吸收概率远小 于轻水，使得重水堆可以使用天然铀作为燃料。

Option: (A) 氘水 (B) 氚水 (C) 重水 (D) 轻水

|[Figure 481]|
|---|

<图片 1>

GPT-4V:这个问题的正确答案是 (A) 氘水。在核电领域，重水通常指的是氘水（D2O），它含有较重 的氘原子，而不是普通的氢原子。重水作为中子慢化剂的特性包括对热中子的吸收概率远小于轻水 （普通水，H2O），这使得重水堆可以使用未经浓缩的天然铀作为燃料。

Error Category:Reasoning Error Error Reason:GPT-4V possesses knowledge in this area and has already answered that "heavy water reactors can use unenriched natural uranium as fuel". However, in the final answer, it responded with (A) heavy water, which is a Reasoning Error.

Ground Truth: (C)重水

- Figure B207: A sample error case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 482]

#### 技术与工程: 能源和电力

Question:<图片 1>所示的某种能源的开采设备，这种能源是__

Option: (A) 原煤 (B) 石油 (C) 天然气 (D) 以上都不是

|[Figure 483]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一台巨大的机械设备，它是一种隧道掘进机（TBM），通常用于挖掘隧道， 而不是直接用于能源开采。因此，根据所提供的选项，正确答案是： (D) 以上都不是

Error Category:Lack of Knowledge Error Reason:The device in the picture is a model of a coal mining machine. GPT-4V did not recognize it, indicating that it may lack this kind of knowledge. Therefore, this error is categorized as a Lack of Knowledge.

Ground Truth: (A) 原煤

- Figure B208: A sample error case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 484]

#### 技术与工程: 能源和电力

Question:<图片 1>所示的某种能源的开采设备，这种能源是__

Option: (A) 原煤 (B) 石油 (C) 天然气 (D) 以上都不是

|[Figure 485]|
|---|

<图片 1>

GPT-4V:正确答案是：(B)石油

Error Category:Lack of Knowledge Error Reason:The device in the image is a classic offshore natural gas drilling platform. GPT-4V did not recognize it, indicating that it may lack this kind of knowledge. Therefore, this error is due to a lack of knowledge.

Ground Truth: (C)天然气

- Figure B209: A sample error case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 486]

#### 技术与工程: 能源和电力

Question:<图片 1>所示的设备的形态是是__ Option:

(A) Y形 (B) T形 (C) I形 (D)落地罐式

|[Figure 487]|
|---|

<图片 1>

GPT-4V:正确答案是： (B) T形 这张图片显示的设备由三个垂直的柱状物体组成，它们通过一个横向的结构连接在一起，形成了一个 类似于字母"T"的形状。

Error Category:Lack of Knowledge Error Reason:The object in the picture is an "I-type circuit breaker". It is obvious that GPT-4V does not recognize this kind of item, so it made a wrong choice based on its shape inference.

Ground Truth: (C) I形

- Figure B210: A sample error case of Technology and Engineering (subcategory: Energy and Power).

Back to List of figures

[Figure 488]

#### 技术与工程:材料

Question: 如图<图片 1>，由___正方点阵可以构成一个面心正方点阵。

Option: (A) 简单 (B) 底心 (C) 体心 (D) 以上均不对

|[Figure 489]|
|---|

<图片 1>

GPT-4V: 很抱歉，由于图片的内容无法直接告诉我这是一个简单、底心还是体心的正方点阵，我无法 直接给出答案。但是，我可以提供一些信息帮助你判断：

- - 简单正方点阵（simple cubic lattice）是最简单的点阵类型，它的原子或者点位于每个立方体的 角上。
- - 底心正方点阵（base-centered cubic lattice）在简单正方点阵的基础上，在每个立方体的底面 中心增加了一个原子或点。
- - 体心正方点阵（body-centered cubic lattice）在简单正方点阵的基础上，在立方体的中心增加 了一个原子或点。
- - 面心正方点阵（face-centered cubic lattice）在简单正方点阵的基础上，在每个立方体的每个 面的中心增加了一个原子或点。

根据这些信息，你可以观察图片中的点阵结构，看看它们是如何排列的，然后选择最符合的选项。如 果图片显示的点阵结构不足以确定答案，那么可能需要更多的信息才能作出选择。

Error Category: Reject to Answer Error Reason: The GPT-4V model chose to not answer the question directly (Reject to Answer) and instead provided descriptive information about different types of square lattices. This is because the model lacks the ability to extract and understand crucial information from the image to determine which type of lattice is being displayed. The model adopted a cautious strategy of providing relevant information rather than a direct answer to avoid giving an incorrect response.

Ground Truth: (C) 体心

- Figure B211: A sample error case of Technology and Engineering (subcategory: Materials). Back to List of figures

[Figure 490]

#### 技术与工程:材料

Question: 如图<图片 1>，为热分析法测___相图。

Option: (A) 匀晶 (B) 共晶 (C) 偏晶 (D) 共析

|[Figure 491]|
|---|

<图片 1>

GPT-4V: 很抱歉，由于图片中的信息不足以确定一个明确的答案，我无法直接给出正确选项。但是， 根据图中的热分析曲线，我们可以看到有一个明显的平台区域，这通常表示在该成分比例下有一个相 变发生，这可能是一个共晶或共析反应。因此，最可能正确的选项是：

(B) 共晶 (D) 共析

如果这是一个典型的共晶或共析相图，那么在特定的成分比例下，温度会保持不变，直到相变完成。 然而，没有更多的上下文信息，很难确定是共晶还是共析。通常需要更多的材料科学背景知识或者是 对特定材料体系的了解来准确判断。

Error Category: Perceptual Error Error Reason: According to the question description, the issue pertains to the interpretation of a phase diagram obtained through a thermal analysis method. However, GPT-4V's response demonstrates its inability to accurately identify and interpret the crucial information in the image. The model incorrectly interpreted the features in the image as possibly indicative of eutectic or eutectoid reactions, whereas the correct answer is "eutectic."

Ground Truth: (A) 匀晶

- Figure B212: A sample error case of Technology and Engineering (subcategory: Materials). Back to List of figures

[Figure 492]

#### 技术与工程:材料

Question: 如图<图片 1>，为Al-Si系的___范围。 Option: (A) 共晶区 (B) 伪共晶区 (C) 固溶体区 (D) 共析区

|[Figure 493]|
|---|

<图片 1>

GPT-4V: 这张图是Al-Si（铝硅）合金的相图。在这个相图中，可以看到不同成分比例和温度下材料 的相态。根据图中的标记和相图的典型特征，图中标记的垂直线所在的位置是铝硅合金的共晶组成， 即铝含量大约为88.3%，硅含量为11.7%的位置。在这一组成比例下，合金在冷却过程中会从液态（L） 直接变为两个固态相的混合物（α+Si），而不经历固液共存阶段。这种转变发生在共晶温度，即相 图中的水平线所表示的温度。

正确答案是： (A) 共晶区

Error Category: Perceptual Error Error Reason: GPT-4V incorrectly interpreted the phase diagram in the image as representing a "eutectic region," whereas the actual correct answer is the "pseudo-eutectic region." Several factors may have contributed to this error. Firstly, despite GPT-4V's capability for image recognition and interpretation, it may face challenges when analyzing specific types of scientific images, especially complex phase diagrams. These images often contain rich scientific information and require domain-specific knowledge for correct interpretation. In this case, the model may not have fully identified all the crucial features in the image or may not have correctly understood the specific significance of these features in materials science.

Ground Truth: (B) 伪共晶区

- Figure B213: A sample error case of Technology and Engineering (subcategory: Materials). Back to List of figures

[Figure 494]

#### 技术与工程:材料

Question: 图示<图片 1>轴向受力杆件,杆内最大拉力为()

Option: (A) 8kN (B) 4kN (C) 5kN (D) 3kN

|[Figure 495]|
|---|

<图片 1>

GPT-4V: 根据所提供的图片，我们可以看到一个轴向受力的杆件，其中标注了三个力：3kN、8kN和 5kN。为了确定杆内最大拉力，我们需要分析这些力的作用方式。 在杆的左侧，有一个向左的力，大小为3kN。在杆的右侧，有一个向右的力，大小为5kN。在杆的中间， 有一个向右的力，大小为8kN。 为了保持平衡，杆内的力必须相互抵消。因此，我们可以推断出：

- - 杆的左侧部分必须承受一个向右的力，以抵消3kN的向左力，因此左侧部分的拉力为3kN。
- - 杆的右侧部分必须承受一个向左的力，以抵消5kN的向右力，因此右侧部分的拉力为5kN。
- - 杆的中间部分承受的是8kN的向右力，这是杆内的最大力。 因此，杆内最大的拉力是8kN。

正确答案是： (A) 8kN

Error Category: Perceptual Error Error Reason: Due to a misinterpretation of arrow directions, the GPT-4V model arrived at an incorrect conclusion. Properly understanding the direction of arrows is crucial for solving statics problems as it directly impacts the analysis of force equilibrium. In this image, the 3kN force is to the left, and the 5kN force is actually also to the left, not to the right as initially explained by the model, which is in contradiction.

Ground Truth: (D) 3kN

- Figure B214: A sample error case of Technology and Engineering (subcategory: Materials). Back to List of figures

[Figure 496]

#### 技术与工程:材料

Question: 如<图片 1>所示为晶核的半径r与∆𝐺𝐺间的关系，现有不同温度的三条曲线，温度最低的是 （）

Option:

(A) T1 (B) T2 (C) T3 (D) 都一样

|[Figure 497]|
|---|

<图片 1>

GPT-4V: 这张图片显示了晶核的半径 r 与自由能变化 ∆𝐺𝐺 之间的关系。根据成核理论，自由能变化 曲线的峰值代表了成核的能垒。温度越低，成核能垒越高，因此曲线的峰值越高。 在这张图片中，我们可以看到 T1 曲线的峰值最高，这意味着它代表的是最低的温度。因此，正确答 案是： (A) T1

Error Category: Reasoning Error Error Reason: While explaining the image, GPT-4V correctly referenced general principles from nucleation theory but made an error in applying this principle for inference. In the question, the image depicted the relationship between the nucleation radius 'r' and the change in free energy 'ΔG,' and it included three curves representing different temperatures. In theory, the height of the free energy change curve is inversely proportional to temperature, meaning that a lower curve represents a higher temperature.

However, when analyzing this image, GPT-4V incorrectly assumed that the curve with the highest peak represented the lowest temperature, neglecting a crucial piece of information mentioned in the question: "G1>G2>G3, T1>T2>T3 when r is the same." This implies that, for the same nucleation radius, a larger change in free energy '∆𝐺𝐺' corresponds to a lower temperature.

Ground Truth: (C) T3

- Figure B215: A sample error case of Technology and Engineering (subcategory: Materials). Back to List of figures

[Figure 498]

#### 技术与工程:材料

Question: 如图<图片 1>，是晶体局部滑移产生___位错。

Option: (A) 刃型 (B) 螺型 (C) 混合 (D) 无法确定

|[Figure 499]|
|---|

<图片 1>

GPT-4V: 这两幅图展示了晶体中的位错移动。在左侧的图中，晶体的一部分似乎在一个特定的平面上 发生了滑移，而右侧的图显示了滑移后的结果。根据这些图像，我们可以看到一个额外的半平面插入 到晶体中，这是刃型位错的特征。因此，正确答案是：

(A) 刃型

Error Category: Reasoning Error Error Reason: The GPT-4V model mentioned "additional half-planes inserted into the crystal," which is a typical characteristic of an edge dislocation. However, based on the question description and the correct answer, the image should be depicting a screw dislocation. This indicates that the model deviated in its logical reasoning about the content of the image, possibly due to incorrectly associating the image features with an edge dislocation.

Ground Truth: (B) 螺型

- Figure B216: A sample error case of Technology and Engineering (subcategory: Materials). Back to List of figures

[Figure 500]

#### 技术与工程: 机械工程

Question: <图片 1>机械识图的三维图形研究从图来理解物，通过看平面图形，想象物体的__。 Option: (A) 空间大小 (B) 颜色 (C) 位置 (D) 空间形状

|[Figure 501]|
|---|

<图片 1>

GPT-4V: 正确答案是： (D) 空间形状 机械识图的三维图形研究通常涉及从二维图像中理解和想象物体的三维形状。因此，通过看平面图形 ，我们可以想象物体的空间形状。

Ground Truth: (D)空间形状

- Figure B217: A sample correct case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

[Figure 502]

#### 技术与工程: 机械工程

Question: 试计算<图片 1>机构的自由度F=__。

|[Figure 503]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一个机械连杆系统，我们可以使用库兹巴赫方程来计算其自由度（F）。 库兹巴赫方程是： F = 3(n - 1) - 2j - h 其中： n = 连杆数（包括地面） j = 铰链数（完全约束的转动副） h = 高阶副数（例如，滑块副） 从图片中，我们可以看到：

- - 有4个连杆（包括地面作为一个参考连杆）
- - 有4个铰链（A, B, C, D）
- - 没有高阶副 因此： n = 4 j = 4 h = 0 代入库兹巴赫方程： F = 3(4 - 1) - 2*4 - 0 F = 3*3 - 8 F = 9 - 8 F = 1 所以，这个机构的自由度 F = 1。 Ground Truth: 1

- Figure B218: A sample correct case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

[Figure 504]

#### 技术与工程: 机械工程

Question: 设有一弹簧、质量、阻尼器机械系统，如<图片 1>所示，以外力f(t)为输入量，位移y(t) 为输出量的运动微分方程式可以对图中系统进行描述，那么这个微分方程的阶次是（） Option:

(A) 1; (B) 2; (C) 3; (D) 4;

|[Figure 505]|
|---|

<图片 1>

GPT-4V:这个系统是一个典型的二阶机械系统，因为它包含有质量（m）、弹簧（K，弹性系数）和阻 尼器（B，阻尼系数）。在这样的系统中，位移（y(t)）是受力（f(t)）的函数，而这个关系可以通 过一个二阶微分方程来描述，因为系统的动态行为涉及到位移的加速度（即位移的二阶导数）。 因此，正确答案是： (B) 2

Ground Truth: (B)2

- Figure B219: A sample correct case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

|技术与工程: 机械工程|
|---|

[Figure 506]

|Question: 如<图片 1>所示，甲、乙、丙是软铁片，电键闭合后，则（）。 Option:<br><br>(A)甲的左端出现N级<br>(B) 丙的左端出现N级<br>(C) 乙的左端出现N级<br>(D) 乙的右端出现N级<br><br><br>GPT-4V:抱歉，由于图片中的信息不足，我无法提供一个确切的答案。但是，根据电磁学的基本原理， 当电流通过线圈时，线圈会产生磁场，使得附近的软铁片磁化。磁场的方向遵循右手定则，即如果将 右手的四指指向电流的方向，拇指指出的方向就是磁场的北极。 在这个问题中，我们需要知道电流的方向以及线圈的绕法，才能确定软铁片的哪一端会出现N极（北 极）。由于这些信息在图片中没有提供，我无法确定正确答案。如果你能提供更多的背景信息或者电 流的方向，我可能能够帮助你确定正确的选项。<br><br>Ground Truth: (A)甲的左端出现N级 (D) 乙的右端出现N级<br><br>Error Category: Perceptual Errorack of Knowledge Error Reason: GPT-4V failed to accurately understand or analyze the information in the image, such as the direction of the electric current, the winding of the coil, and the related magnetic field effects. This error occurred because GPT-4V cannot directly parse details in images, such as circuit diagrams or symbols, and therefore could not accurately determine the magnetic pole changes in the soft iron pieces A, B, and C after the electrical key is closed. In this case, GPT-4V did not provide a definitive answer but explained based on general principles of electromagnetism, indicating its inability to process specific visual information in the image.<br><br><图片 1><br><br>|[Figure 507]|
|---|
|
|---|

- Figure B220: A sample error case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

|技术与工程: 机械工程|
|---|

[Figure 508]

|Question: 取一机器的主轴为等效构件，已知主轴平均转速n=1000r/min，在一个稳定运动循环（2 ） 中的等效阻力矩Mer如<图片 1>所示，等效驱动力矩Med为常数。若不计机器中各构件的转动惯量，试 求：当主轴运转不均匀系数 ＝0.05时，应在主轴上加装的飞轮的转动惯量J=_。<br><br>GPT-4V:由于题目中没有提供足够的信息来直接计算飞轮的转动惯量J，我们需要依据现有的数据和 一些基本的物理原理来进行推理。首先，我们知道主轴的不均匀系数是0.05，这个系数定义为速度波<br><br>动的最大值与平均角速度的比值。不均匀系数可以表示为：𝛿𝛿 = 𝜔𝜔∆𝜔𝜔<br><br>𝑎𝑎𝑎𝑎𝑎𝑎<br><br>其中，∆𝜔𝜔 是角速度的波动范围，<br><br>𝜔𝜔𝑎𝑎𝑎𝑎𝑎𝑎是平均角速度。由于不计机器中各构件的转动惯量，飞轮的作用是平衡等效阻力矩Mer的波动， 使得主轴的转速尽可能稳定。飞轮的转动惯量J与其能够储存的动能量成正比，而动能量与角速度的 波动范围有关。在稳定运动循环中，等效驱动力矩Med是一个常数，因此我们可以假设它等于等效阻 力矩Mer的平均值。从图中可以看出，Mer在一个周期内是一个三角形波动，我们可以计算出这个三角 形的面积来代表一个周期内的总阻力矩工作量，然后除以周期时间得到平均阻力矩。然而，由于没有 给出Med的具体数值，我们无法直接计算出飞轮的转动惯量J。在实际应用中，J的计算需要通过动能 的变化量和角速度的波动范围来确定。这通常涉及到对系统动力学的详细分析，包括计算等效阻力矩 Mer和等效驱动力矩Med之间的差值在一个运动周期内的时间积分，以及应用能量守恒原理。因此，没 有具体的数值和更多的系统参数，我们无法给出一个确切的答案。在实际工程问题中，这通常需要详 细的系统分析和计算。<br><br>Ground Truth: 0.51<br><br>Error Category: Reasoning Error Error Reason: GPT-4V made a reasoning error in its response. The question asked for the calculation of the moment of inertia of a flywheel that should be added to the main shaft, but GPT4V failed to perform the necessary reasoning and calculations to arrive at the answer. While it provided information about the non-uniformity coefficient and angular velocity fluctuation, it did not apply this information to the calculation of the flywheel's moment of inertia. The correct approach involves calculating the maximum angular velocity fluctuation and then applying it to the definition of the non-uniformity coefficient. Due to GPT-4V's failure to perform such reasoning and calculations, it resulted in an incorrect answer.<br><br><图片 1><br><br>|[Figure 509]|
|---|
|
|---|

- Figure B221: A sample error case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

|技术与工程: 机械工程|
|---|

[Figure 510]

|Question: 指出<图片 1>下列定位元具体限制自由度方向___ Option:<br><br>(A) 𝑥𝑥̅<br>(B) 𝑦𝑦 <br>(C) 𝑥𝑥̅ ，𝑦𝑦 <br>(D) 𝑥𝑥̅ ，𝑧𝑧̅<br><br><br>GPT-4V:这张图片显示了一个机械零件的剖面图，其中包含了一个定位元。根据图示，定位元似乎限 制了零件在垂直于定位元接触面的方向上的移动。在这种情况下，定位元限制了零件沿着z轴的移动 （ 𝑧𝑧̅ ）。然而，由于图中没有明确显示定位元是否同时限制了x轴或y轴方向上的移动，我们无法确定 是否还有其他限制。 根据图示，我们可以确定的是定位元限制了沿z轴的移动。因此，正确答案是： (D) 𝑥𝑥̅ ， 𝑧𝑧̅ 请注意，这个答案是基于图示和常见的机械设计原则做出的最佳判断。如果图中提供了更多信息或者 有具体的设计背景，可能会有不同的答案<br><br>Ground Truth: (C) 𝒙𝒙 , 𝒚𝒚<br><br>Error Category: Textual Understanding Error Reason: GPT-4V had a problem with understanding the text of the question. The question asked to specify the specific directions in which the locating element restricts degrees of freedom, but GPT-4V incorrectly interpreted the question, leading to an incorrect answer choice. It chose (D) 𝑥𝑥̅ 𝑧𝑧̅ , but the actual correct answer should be(C) 𝑥𝑥̅ ，𝑦𝑦  . This is because the question did not mention the z-axis, and GPT-4V incorrectly included the z-axis in its answer.<br><br><图片 1><br><br>|[Figure 511]|
|---|
|
|---|

- Figure B222: A sample error case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

|技术与工程: 机械工程|
|---|

[Figure 512]

|Question: 试求<图片 1>机构的自由度F=__。<br><br>GPT-4V:这张图片显示的是一个平面机构，看起来像是一个四连杆机构。为了计算这个机构的自由度 （F），我们可以使用库兹巴赫公式（Kutzbach criterion）来进行计算，该公式适用于平面机构： F = 3(n - 1) - 2j 其中，n 是机构中的杆件数（包括地面），j 是铰接点（即转动关节）的数量。 从图片中，我们可以看到有四个杆件（包括地面作为一个杆件），以及四个铰接点。因此，我们可以 将这些值代入公式中： n = 4 （三个移动杆件加上地面） j = 4 （四个铰接点） 代入公式得： F = 3(4 - 1) - 2*4 F = 3*3 - 8 F = 9 - 8 F = 1 所以，这个机构的自由度 F = 1。这意味着机构有一个独立的运动自由度。<br><br>Ground Truth: 3<br><br>Error Category: Answer Extraction Error Error Reason: Incorrect extraction and matching. In GPT-4V's response, it utilized the Kutzbach criterion formula to calculate the degree of freedom of the mechanism, but it made a mistake during the calculation process. Although it correctly determined the number of links and joints in the mechanism, it encountered an error in calculating the degree of freedom.<br><br><图片 1><br><br>|[Figure 513]|
|---|
|
|---|

- Figure B223: A sample error case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

|技术与工程: 机械工程|
|---|

[Figure 514]

|Question: 图<图片 1>中是哪种联接方式？__ Option:<br><br>(A) 平键联接<br>(B) 无键联接<br>(C) 滑键联接<br>(D) 花键联接<br><br><br>GPT-4V:这张图片显示了一个齿轮和一个轴，轴的一端有一系列的凸起，这些凸起与齿轮内部的槽相 匹配。这种设计通常用于花键联接，它允许轴和齿轮之间既可以传递扭矩，也可以相对滑动，如果需 要的话。 正确答案是： (D) 花键联接<br><br>Ground Truth: (B)无键联接<br><br>Error Category: Answer Extraction Error Error Reason: The answer extracted by the code is not the correct answer for GPT-4V. The question asked to identify the type of connection shown in the image, and GPT-4V incorrectly chose (D) Keyed Connection, while the actual connection shown in the image is a Keyless Connection. This indicates that GPT-4V made an error in extracting the correct answer from the text and image, selecting an inappropriate answer.<br><br><图片 1><br><br>|[Figure 515]|
|---|
|
|---|

- Figure B224: A sample error case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

#### 技术与工程: 机械工程

[Figure 516]

Question: 如<图片 1>所示中的螺纹联接为（） Option:

(A) 螺栓联接 (B) 螺柱联接 (C) 螺钉联接 (D) 圆柱联接

|[Figure 517]|
|---|

<图片 1>

GPT-4V:这张图片显示的是一个螺纹联接的示意图。根据图中所示的组件特征，可以看到有一个带螺 纹的杆状部件穿过一个孔，并且在两端有螺母固定。这种类型的联接通常是螺栓联接，因为它包括一 个螺栓和两个螺母，螺栓穿过被联接的部件，并且在两端用螺母固定。 正确答案是： (A) 螺栓联接

Error Category: Lack of Knowledge Error Reason: Lack of the knowledge required to answer this question or a hallucination regarding the knowledge needed to answer this question. The question asks for the identification of the type of threaded connection, and GPT-4V incorrectly identified it as a bolted connection, while the correct answer is a screw connection. This error may be due to GPT-4V lacking sufficient knowledge to distinguish between different types of threaded connections, leading to an incorrect answer choice.

Ground Truth: (C) 螺钉联接

- Figure B225: A sample error case of Technology and Engineering (subcategory: Mechanical Engineering).

Back to List of figures

