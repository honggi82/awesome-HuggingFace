[Figure 1]

# arXiv:2401.17268v1[cs.CL]30Jan2024

## Weaver: Foundation Models for Creative Writing

Tiannan Wang Jiamin Chen Qingrui Jia Shuai Wang Ruoyu Fang Huilin Wang Zhaowei Gao Chunzhao Xie Chuou Xu Jihong Dai Yibin Liu Jialong Wu Shengwei Ding Long Li Zhiwei Huang Xinle Deng Teng Yu Gangan Ma Han Xiao Zixin Chen Danjun Xiang Yunxia Wang Yuanyuan Zhu Yi Xiao Jing Wang Yiru Wang Siran Ding Jiayang Huang Jiayi Xu Yilihamu Tayier Zhenyu Hu Yuan Gao Chengfeng Zheng

Yueshu Ye Yihang Li Lei Wan Xinyue Jiang Yujie Wang Siyu Cheng Zhule Song Xiangru Tang Xiaohua Xu Ningyu Zhang Huajun Chen Yuchen Eleanor Jiang* Wangchunshu Zhou*

AIWaves Inc.

### Abstract

This work introduces Weaver, our first family of large language models (LLMs) dedicated to content creation. Weaver is pre-trained on a carefully selected corpus that focuses on improving the writing capabilities of large language models. We then fine-tune Weaver for creative and professional writing purposes and align it to the preference of professional writers using a suit of novel methods for instruction data synthesis and LLM alignment, making it able to produce more human-like texts and follow more diverse instructions for content creation. The Weaver family consists of models of Mini (1.8B), Base (6B), Pro (14B), and Ultra (34B) sizes, suitable for different applications and can be dynamically dispatched by a routing agent according to query complexity to balance response quality and computation cost. Evaluation on a carefully curated benchmark for assessing the writing capabilities of LLMs shows Weaver models of all sizes outperform generalist LLMs several times larger than them. Notably, our most-capable Weaver Ultra model surpasses GPT-4, a state-of-the-art generalist LLM, on various writing scenarios, demonstrating the advantage of training specialized LLMs for writing purposes. Moreover, Weaver natively supports retrieval-augmented generation (RAG) and function calling (tool usage). We present various use cases of these abilities on improving AI-assisted writing systems, including integration of external knowledge bases, tools, or APIs, and providing personalized writing assistance. Furthermore, we discuss and summarize a guideline and best practices for pre-training and fine-tuning domain-specific LLMs.

Weaver is currently accessible at www.wawawriter.com, our innovative human-AI collaborative writing platform (For the English version of WawaWriter, see www.wawawriter.com/en). We discuss a few innovations of the platform from the perspective of human-computer interaction to explain how it will revolutionize traditional AI-assisted writing systems.

*Corresponding authors: {eleanor,chunshu}@aiwaves.cn

#### Contents

- 1 Introduction 4
- 2 Pre-training 6

- 2.1 Model Family . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.2 Pre-training Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.3 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 3 Data Synthesis 8

- 3.1 Abilities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 3.1.1 Instruction Following . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.1.2 Instruction Annotation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 3.1.3 Evaluation (Literary Critic) . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.1.4 Retrieval-Augmented Generation . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.1.5 Function Calling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 3.2 Instruction Backtranslation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.3 Constitutional DPO: Learning From Principled Negative Examples . . . . . . . . . . . 12

- 4 Alignment 14

- 4.1 Supervised Fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4.1.1 Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.1.2 Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4.2 Preference Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4.2.1 Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.2.2 Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 5 Evaluation 14

- 5.1 WriteBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 5.2 Compared Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 5.3 LLM-based Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 5.4 Human Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 5.5 User Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 6 Introducing WawaWriter 17

- 6.1 Human-AI Collaborative Writing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 6.2 Integration of External Knowledge and Tools . . . . . . . . . . . . . . . . . . . . . . . 17
- 6.3 Personalized Writing Assistance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 6.4 Infinite Long Text Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

###### 7 Discussion 18

A Appendix 24 A.1 Author Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24 A.2 Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24 A.3 Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

#### 1. Introduction

Large language models (LLMs) (Anthropic, 2023; Brown et al., 2020; Google, 2023; Jiang et al., 2023a; OpenAI, 2022, 2023; Radford et al., 2018, 2019; Gemini Team, 2023; Touvron et al., 2023a,b; Yin et al., 2023; Zhao et al., 2023) based on Transformers (Vaswani et al., 2017) have become a prominent pathway to Artificial General Intelligence (AGI). LLMs acquire massive world knowledge by learning to predict the next word on large-scale web corpora. The capabilities of LLMs have been continuously increasing by scaling model sizes, dataset sizes, and computation. After pre-training, LLMs can be aligned to support real-world use cases by supervised fine-tuning (Chung et al., 2022; Sanh et al., 2022) and preference optimization techniques including reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022a; Wang et al., 2024; Zheng et al., 2023b) and direct preference optimization (DPO) (Rafailov et al., 2023). The capabilities of LLMs have empowered various applications including ChatGPT, Claude, Bard, Microsoft Copilot, Character.AI, Notion AI, etc. Recently, many specialized LLMs have been trained for different targeted usage scenarios. In general, LLMs specialize according to the targeted domains (e.g., finance (Wu et al., 2023), healthcare (Yang

- et al., 2022b), legal (Cui et al., 2023), etc.) and tasks (e.g., role-playing (Wang et al., 2023d), coding (Rozière et al., 2023), etc.). However, the ability of LLMs to write human-like texts and produce creative content, which is a critical use case of LLM applications such as ChatGPT, is mostly overlooked by the community.

In this report, we focus on the literature domain and the task of writing, or content creation, and introduce Weaver, a family of LLMs dedicatedly pre-trained and aligned for this purpose. The name "Weaver" symbolizes the model’s proficiency in skillfully amalgamating linguistic elements, akin to the way a craftsman weaves threads to form a fabric. We answer four main questions in this technical report: why we need Weaver, how we train Weaver, how Weaver performs, and what we build with Weaver.

Why we need Weaver? Despite generalist LLMs such as GPTs already possessing general writing skills and helping billions of users in various writing scenarios, they often struggle to produce human-like texts in specific writing scenarios such as writing stories, fiction, social media copies, blogs, papers/thesis, etc. We analyze the behavior of pre-trained base LLMs such as LLaMA and aligned LLMs such as ChatGPT and LLaMA-chat and believe this limitation originates from both the pre-training stage and the alignment stage. On one hand, generalist LLMs are pre-trained on massive low-quality web texts or machine/AI-generated texts. Consequently, existing LLM backbones tend to produce seemingly fluent texts that are not creative enough and lack human-like styles. On the other hand, during the alignment stage, state-of-the-art LLMs such as GPT-4 are instruction-tuned using instruction-response pairs annotated by crowdsource annotators (Ji et al., 2023; Shen et al., 2023; Wang et al., 2023c). However, most of the annotators are not professional writers or content creators and the annotation guidelines only require them to produce helpful and harmless responses (Ouyang et al., 2022b). As a result, the crowdsourced data for supervised fine-tuning is less stylish and lacks creativity. Furthermore, most popular preference

8.6

Weaver Ultra

8.4

GLM4 GPT-4

8.2

WriteBenchOverallScore

Weaver Pro

Yi-34B

8.0

Claude2 Qwen-72B

7.8

Weaver BaseQwen-14B Weaver Mini

Yi-6B

7.6

7.4

7.2

Qwen-1.8B

1.8 6.0 14.0 34.0 72.0 Unknown

Number of Model Parameters (Billions)

Figure 1 | Comparison between Weaver and generalist LLMs on WriteBench.

optimization methods such as RLHF and DPO optimize the model on model-generated data pairs, making them less suitable for enhancing the creativity of LLMs.

These factors make current generalist LLMs lack creativity and unable to produce human-style texts despite they are super powerful in other applications such as writing codes and answering general questions. We believe this phenomenon will continue to be amplified given that the amount of LLM-generated texts on the internet is exponentially growing and most LLMs are aligned using texts produced by other LLMs. Therefore, we believe it is necessary to train domain-specific LLMs dedicated to writing purposes that are creative and generate human-like texts in order to fully exploit the potential of AI-generated content (AIGC).

How we train Weaver? To address the aforementioned issues limiting generalist LLMs’ creative writing ability, we carefully design a suite of strategies for automated data collection, data annotation, and data filtering for pre-training and alignment. This makes us able to pre-train and align Weaver on diverse, human-like, and stylish texts. To be specific, we conduct extensive pre-training data filtering and only keep high-quality content such as books, fiction, stories, and articles in the pre-training corpus, making the pre-trained backbones more likely to produce human-like texts.

As for the alignment stage, we propose a new instruction backtranslation framework inspired by LongForm (Köksal et al., 2023) and Humpback (Li et al., 2023) that synthesize diverse and natural instructions that correspond to high-quality outputs written by professional writers and preferred by human consumers. Our instruction backtranslation framework translated the work of crowdsource annotators from writing both instructions and outputs to simply collecting high-quality content such as stories, fiction, articles, social media copies, and blog posts. This massively reduces the cost of instruction data annotation and the requirement for crowdsource annotators while significantly improving the quality of annotated data.

Moreover, we also propose a novel Constitutional DPO algorithm for preference optimization to better align Weaver to the preference of professional writers and content creators. Constitutional DPO is inspired by and combines the advantages of a few previous works including DPO (Rafailov

- et al., 2023), Constitutional AI (Bai et al., 2022), Self-Align (Sun et al., 2023), and RLCD (Yang et al., 2023a). Specifically, Constitutional DPO exploits expert (e.g., professional editors in our case) annotated principles to synthesize negative examples that violate certain principles based on positive examples that are sampled from the optimal policy (e.g., texts produced by professional writers or content creators in our case). In contrast to the common practice of using DPO that uses LLMs to produce preference annotation on two model-generated responses such as Zephyr (Tunstall et al., 2023), the pairwise preference data synthesized by our approach contains less noise since the negative example are deliberately synthesized to be of lower quality compared to the positive example. The pairwise preference data generated by Consitutional DPO also contains more principled and targeted learning signals that can be adjusted by human experts according to target domains and applications.

Furthermore, we propose to transform the annotation instructions and responses used in the instruction backtranslation and Constitutional DPO stages into annotation instructions and evaluation instructions. In this way, Weaver not only possesses abilities to follow writing instructions but can also annotate writing instructions and evaluate writing outputs. We also curate instruction data for retrieval-augmented generation (RAG) and function calling to enable Weaver to exploit external knowledge and tools. The combination of different data sources makes Weaver a versatile foundation model while specializing in creative writing.

How Weaver performs? Evaluating the content creation/writing ability of LLMs remains an open problem since existing benchmarks for LLMs such as MMLU (Hendrycks et al., 2020) or MTBench (Zheng et al., 2023a) mostly focus on reasoning, math, coding, or general questions instead of

creative writing. Moreover, it is already notoriously hard to evaluate LLMs on general instructions, and it becomes much harder for creative writing tasks since literary critic is non-trivial even for human experts, not to mention LLMs. To better evaluate Weaver and help the LLM community better measure progress on AIGC, we carefully curate WriteBench, a benchmark for assessing the creative writing capabilities of LLMs and collect outputs from 10+ popular LLMs covering both open-source and proprietary models.

We then conduct both LLM-based and human evaluation of Weaver and generalist LLMs on the benchmark. Evaluation results confirm the superiority of Weaver compared to generalist LLMs. We find that Weaver Ultra, the most-capable model in the Weaver family, advances the state-of-the-art in creative writing despite being 10+ smaller compared to GPT-41, the previous best performing LLM. Other models in the Weaver family also surpass competitive generalist LLMs several times larger than them. Our analysis and case studies show that the main source of improvements is because Weaver can generate texts that are creative and human-like while generalist LLMs tend to produce too “predictable” texts. To confirm that Weaver is truly helpful in real-world applications, we also conduct a user study where human writers are asked to write stories (fiction writing) and blog posts (non-fiction writing) with Weaver and GPT-4. Our user study shows that compared to GPT-4, Weaver improves the writers’ productivity by 47% and helps writer produce better stories and articles at the same time.

What we build with Weaver? Training specialized LLMs for writing is one side of enhancing AIassisted writing experience. We believe it is also very important to build a better human-AI interface to fully exploit the potential of Weaver on AI-assisted writing. To this end, we introduce WawaWriter, our innovative human-AI collaborative writing platform. Similar to recent AI writing products such as Notion AI, WawaWriter provides a chat interface that allows users to provide diverse writing instructions, instead of merely suggesting the next one or few sentences based on the current context or polishing the content as in traditional applications. WawaWriter also takes a few steps further:

- (1) we enable human-AI co-editing by allowing users to customize language agents (Zhou et al., 2023b) that acts like a human collaborator by operating inside the editor simultaneously with users;
- (2) we allow users to build personal knowledge bases by saving websites or uploading documents and build a RAG pipeline that integrates knowledge bases to Weaver; (3) we propose to provide personalized writing assistance by analyzing users’ personal writing styles using LLMs based on their writing history on the platform and using the results to guide Weaver’s text generation process. By combining these innovations, WawaWriter aims to provide next-generation AI-assisted writing experience that is more helpful and enjoyable.

In the following sections, we first describe the architectures and sizes of the Weaver family and their pre-training stage. We then present details on the abilities of Weaver, how we synthesize training data to help Weaver acquire these abilities and learn to produce human-like stylish texts, and the details for the alignment stage. We also present our benchmark for evaluating the writing abilities of LLMs and the evaluation results. Finally, we introduce the details of WawaWriter and present how Weaver paves the way for next-generation AI-assisted writing experiences.

#### 2. Pre-training

###### 2.1. Model Family

Weaver models are language models built on top of Transformer decoders. We have adopted the recent improvements from the design of LLaMA (Touvron et al., 2023a,b), the most popular open-

1According to non-official rumor about the size of GPT-4

source LLM, including a Pre-Norm structure with RMSNorm (Zhang and Sennrich, 2019) function, SwiGLU (Shazeer, 2020) as the activation function for the Feed-Forward Network, Rotary Embedding (Su et al., 2024) for positional encoding, and Grouped-Query Attention (GQA) (Ainslie et al., 2023).

The Weaver family consists of models of four different sizes: Mini, Base, Pro, and Ultra, ranging from 1.8B to 34B parameters. We train different model sizes to support different applications as the complexity of writing tasks varies a lot across different domains and use cases. All Weaver models are initialized from powerful open-source LLMs. We provide detailed configurations and descriptions of Weaver models in Table 1.

###### 2.2. Pre-training Data

We then present an overview of pre-training data selection strategies and the resulting pre-training data mixture. Since Weaver models are initialized from powerful open-source LLMs and thus already possess adequate world knowledge, the amount of continual pre-training data does not need to be super large. We consider the continual pre-training stage to be the process where Weaver learns to reallocate or re-balance its capabilities: the model allocates more capabilities to writing and content creation while reducing the capabilities on other domains such as mathematics and coding.

Therefore, we only include manually verified data sources including various kinds of content such as books, fiction, stories, news articles, papers, reports, social media copies, etc., in the pre-training data. We combine rule-based and machine-learning-based methods to filter low-quality texts. In addition to data sources and filtering, we also carefully control the data mixture between different domains. Specifically, we mix fiction data (i.e., fiction and stories) and non-fiction data (i.e., articles, papers, reports, etc.) with a ratio of 1 : 1. We also mix Chinese and English data with a portion of 4 : 1 to make Weaver supports both Chinese and English.

###### 2.3. Training Details

We train Weaver using the standard autoregressive language modeling task where the model learns to predict the next token based on the context of previous tokens. We train Weaver models with a context length of 4096. We shuffle and merge the documents, and then truncate them to the specified context lengths to create training batches. We incorporate Megatron-Deepspeed (Shoeybi et al., 2019) and Flash Attention2 (Dao, 2023; Dao et al., 2022) to improve computational efficiency and reduce memory usage. We adopt the standard optimizer AdamW (Loshchilov and Hutter, 2017) and set the hyperparameters 𝛽1 = 0.9, 𝛽2 = 0.95, and 𝜀 = 10−8. We use a cosine learning rate schedule with a specified peak learning rate for each model. The learning rate is decayed to a minimum learning rate of 10% of the peak learning rate. All models are trained with BFloat16 mixed precision for training stability. We present detailed pre-training configurations for each model in Table 1.

Context Sequence Learning

Name Params 𝑛layers 𝑑model 𝑛heads

Tokens Length Batch Size Rate

Weaver Mini 1.8B 24 2048 16 4096 512 1e-4 50B Weaver Base 6B 32 4096 32 4096 512 1e-4 50B Weaver Pro 14B 40 5120 40 4096 512 1e-4 40B

Weaver Ultra 34B 60 7168 56 4096 520 5e-5 18B

- Table 1 | Description for the Weaver family.

#### 3. Data Synthesis

After pre-training, Weaver models contain a large amount of world knowledge and writing skills and can produce human-like texts conditioning on high-quality contexts. To unlock these capabilities for real-world applications, we need to curate a high-quality dataset for alignment. The format and quality of the dataset significantly affect the coverage of abilities and the quality of aligned models. As discussed in the Introduction, the common practice for alignment data collection of existing generalist LLMs severely limits their writing capabilities. In this section, we describe our data synthesis framework in detail. We first describe the abilities we want to unlock during the alignment stage and then present our proposed data synthesis methods for both the supervised fine-tuning and the preference optimization stage.

- 3.1. Abilities We first describe the categories of abilities we want to unlock for Weaver during the alignment stage.

###### 3.1.1. Instruction Following

The first obvious ability we need to unlock is the ability to follow writing instructions and produce human-like stylish texts. We cover various domains and tasks as listed below during data collection and alignment training.

###### 3.1.1.1 Domains

Fiction Writing: Fiction writing refers to the abilities of models to write stories and fiction. We divide fiction writing into several subdomains with respect to the length and the genre of the fiction. We cover fiction and stories of lengths ranging from a few hundred to a few million characters, and fiction types including sci-fiction, romance, fantasy, horror, mystery, and thriller.

Creative Non-Fiction Writing: Creative non-fiction writing is a genre of writing that uses literary styles and techniques to create factually accurate narratives. We cover typical creative non-fiction writing cases including writing memoirs, biography, travelogue, journalism, social media copy, blog posts, news articles, commentary, etc.

Marketing Writing: We also consider marketing writing, which involves writing business plans, advertising copies, product promoting, marketing plans, etc. Marketing writing differs from previous categories because it is highly application-oriented and the style of generated texts is not the most important. However, marketing writing still requires human-like creativity to attract potential users.

Technical Writing: Technical writing includes tasks such as paper writing, patent writing, report writing, etc. Technical writing requires more accuracy compared to creativity. However, writingspecific training can still be helpful because it can help model produce texts that accurately adhere to the style required for specific scenarios.

###### 3.1.1.2 Tasks

Content writing: Content writing is the basic task that requires the model to generate content (i.e., fiction, articles, etc.) based on certain instructions. Writing instructions vary in terms of whether the previous context is provided and how fine-grained the given instructions are. The task requires the LLM to be able to understand and adhere to specific requirements expressed in the instructions

while also producing texts that are consistent and coherent with previous contexts. For example, a typical content writing instruction is: “Please help me write a sci-fi about what will happen after people finally achieve AGI.”

Outlining: Outlining is the task of writing outlines, which is a common practice for writers in both fiction and non-fiction writing. As discussed in the literature of long text generation (Sun et al., 2022; Yang et al., 2022a, 2023b; Zhou et al., 2019, 2023a), it is often helpful to let the model first generate an outline before generating long texts. Outlines vary according to different domains and the granularity/length of outlines. One example for the task of outlining is “Please help me write an outline of my annual work report.”

Polishing & Editing: Polishing and editing require the model to improve the quality of a paragraph or rewrite it following the requirements expressed in the instructions. The task is closely related to the task of grammatical error correction (Bryant et al., 2019; Ng et al., 2014) with a key difference that the modifications are not necessarily grammatical errors. Compared to the task of academic writing polishing described in Diao et al. (2023), we support customized fine-grained control of polishing or editing requirements, which is important for human-AI interaction in AI-assisted writing systems. A typical polishing instruction may look like this: “Please help me revise the following texts, keep in mind that the revised texts should be suitable for an academic paper.”

Style Transferring: The task of text style transfering requires the model to transform texts in one style into another style. For example, one may want to transform a story into a script or turn a report into a speechwriting. We cover both template-based style transfer that uses a template to provide target style information (Guu et al., 2018; Lewis et al., 2020) and description-based style transfer which uses either a keyword (Hu et al., 2017) or a short description (Zhou et al., 2023c) for the target style. For example, one may ask the model to “Transform the following book chapter into a script.”

Expanding/Simplifying: Text expanding and simplifying requires the model to revise an input paragraph to make it longer or shorter according to certain instructions. Text summarization and summary-to-article generation can be regarded as two extreme cases of this task. One exemplar instruction is: “Please help me summarize this paragraph into one sentence.”.

Brainstorming: Brainstorming requires the model to help users come up with creative ideas based on the current context and user instructions. A typical brainstorming instruction is: “Please give me 5 possible character descriptions for a villain to appear in the next chapter, including his name, appearance, occupation, and background.”

Reviewing: Reviewing refers to the task of reading and analyzing a given piece of text critically and then producing comments or revising suggestions. For example, one may ask the model to “Please take a look at my essay and list 5 suggestions to improve it.”

###### 3.1.2. Instruction Annotation

We also train Weaver to support the instruction annotation task. As described in Humpback (Li et al., 2023) and LongForm (Köksal et al., 2023), given a piece of text, the task requires the model to generate an instruction to which the input texts may be the answer. However, vanilla instruction backtranslation only supports the writing task. Therefore, for instruction annotation, we require the model to synthesize an instruction-response pair based on a text span. The response can be the text span, a part of the text span, or inferred from the text span. This substantially broadens the scope for vanilla instruction backtranslation since most automatically mined text spans may not be suitable for a certain instruction on itself while a part of the text span can be a valid response or one may construct a high-quality instruction-response pair based on it. The instruction annotation

ability enables Weaver to mine training data for itself on large-scale corpus, opening the possibility of scalable self-training on web data.

###### 3.1.3. Evaluation (Literary Critic)

Many recent work explored using or training LLMs to evaluate general instruction following tasks (Chan et al., 2023; Jiang et al., 2023b; Wang et al., 2023b). However, we find generalist LLMs require extensive prompting skills to make them suitable for evaluating tasks related to creative writing. Moreover, since almost all students majoring in creative writing are also required to take literary critic courses, we think learning to perform literary critic may be helpful for the model to produce better texts as well. Therefore, we also train Weaver to judge the quality of the responses to writing instructions and do pairwise comparison of two responses.

We collect human preference between model outputs in WawaWriter, our AI-assisted writing platform and convert the collected preference data to training data for LLM-based evaluation with carefully curated templates.

###### 3.1.4. Retrieval-Augmented Generation

The ability of retrieval-augmented generation (RAG) (Gao et al., 2023; Lewis et al., 2020), i.e., generating responses by referring to external knowledge or references as context. RAG is an important technique that helps LLMs generate more accurate and informed responses. It can be especially helpful for writing purposes since it’s common for human writers to refer to other text samples when writing fiction or articles. However, most existing LLMs purely rely on prompt engineering to do RAG and do not perform RAG training during alignment. We believe this limits the ability of LLMs to make use of retrieved contexts. Therefore, we propose to include RAG-aware training data during alignment to enhance Weaver’s retrieval-augmented generation ability. Specifically, we augment 10% percent of training data by appending a relevant context obtained by retrieving the paragraph most similar to the target response. In this way, Weaver learns to write by referring to external contexts and is thus more compatible with RAG techniques compared to most existing LLMs.

###### 3.1.5. Function Calling

The ability to use tools is also very important for LLMs (Schick et al., 2023). This ability, also referred to as “function calling”, is also helpful for writing because the model may need to search the internet for references or call editor APIs when doing human-AI collaborative writing. To unlock the function calling ability, we include an open-source function calling dataset2 into supervised fine-tuning data. We also propose a new pipeline to synthesize more diverse function calling data by first using GPT-4 to synthesize diverse environments with multiple tools and APIs, as well as their documentation. We then randomly select one API at a time and ask GPT-4 to imagine a situation where the API can be helpful and the plausible arguments for the API. We then reason what one may instruct an LLM in that situation so that the API should be used with the arguments. Finally, similar to how GPTs support function calling, we train Weaver to use tools by selecting the right API and generating the arguments given the instructions and the contexts.

###### 3.2. Instruction Backtranslation

We then describe our proposed improved pipeline for instruction backtranslation. The motivation for doing instruction backtranslation instead of instruction augmentation methods such as self-

2https://huggingface.co/glaiveai

Domain Subdomain Description Source Fiction Writing Full Novel Web novel, over 1M

Proprietary

words

Short Story Web stories, 10k-20k

Proprietary

words

Creative Non-Fiction Writing Red Top liked and com-

Picked

mented posts on Red

Zhihu Top upvoted posts on

Picked

Zhihu

Weibo Top liked posts on

Picked

Weibo

WeChat Articles Top read articles on

Picked

WeChat

DouBan Top liked posts on

Picked

DouBan

News & Blogs Popular news/blogs Picked Technical Writing Papers Academic papers on

Picked

CNKI

Essay Online essays Picked Contract Contracts from online

Picked

sources

Reports Reports for work, business, science, etc.

Proprietary

Copies Business & Govern-

Proprietary

ment copies

Marketing Writing Business Plans Business plans for projects and startups

Proprietary

Industry Report Research report for

Proprietary

different industries

Advertising Copy Popular copies for ad-

Picked

vertisements

Marketing Plan Marketing plans for

Picked

products & services

Product Overview Articles advertising

Picked

products

- Table 2 | Description of SFT Data sources. We combine similar subdomains in the same fields for simplicity. The entire training set covers 34 subdomains and around 500,000 instruction-output pairs. “Picked” means the raw data in the corresponding domains are manually selected.

instruct (Wang et al., 2023a) is very simple: we want to align Weaver on high-quality, stylish, and

human-written texts. To achieve this goal, we first collect high-quality stories, fiction chapters, and copies of different domains. We list the categories of collected texts in Table 2.

We then use a carefully designed few-shot prompt template to synthesize instruction-response pairs for all aforementioned writing tasks. Specifically, for each subdomain-task pair, we annotate 5 cases of how one can write an instruction-response pair, including both the annotated results and the rationales for the annotation process: we first select a text span from a case as the output (except for outlining, brainstorming, and reviewing tasks where the output is transformed from the selected text span with an additional prompt). We then identify or produce the context for the output. For example, for the polishing task, the context should be a worse version of the target output, so we can modify the wording and structure of the target output to make it look worse. Then we infer the instruction that one may use to transform the context to the output. Taking the polishing task as an example again, we need to reason what modifications are made and synthesize the polishing instructions accordingly. For each unlabeled case, we use the annotated cases as few-shot exemplars and ask GPT-4 to first generate the annotation process in the Chain-of-Thought style (Wei et al., 2022) and then produce the synthesized instruction-response pairs. The instruction backtranslation pipeline is illustrated in Figure 1. We synthesize 500,000 high-quality instruction-response pairs across all domains and tasks with this pipeline. Finally, we do an instruction data selection procedure following the practice described in (Liu et al., 2023): we first score all instruction-response pairs with GPT-3.5-turbo and then select top-ranked data in each subdomain-task pair for supervised fine-tuning. Specifically, we score each instruction-response pair based on the quality and the diversity of the instruction and the relevance between the instruction and the response.

|[Domain] Technical Writing<br><br>[Principle 1] The arguments should be illustrated logically … [Instruction] Help me write a paper … [Example-Accepted] The results well demonstrate the …. [Example-Rejected] As we conducted experiments, it is good… [Domain] Creative Non-fiction Writing<br>[Principle 2] The contents should be eye-catching… [Instruction] Report an accident with black bears … [Example-Accepted] Black bears rarely attack. But here’s … [Example-Rejected] Black bears attack in some cases…<br>|
|---|

[Guidelines] You are teaching students the principles while writing [domain]. I will give you an instruction and its accepted output. Based on this output, you need to propose a rejected output which violates such principle. Here is an example:

[Figure 2]

Random sampling

[Domain] Technical Writing [Principle] The arguments should be illustrated logically … [Instruction] Help me write a paper … [Example-Accepted] The results well demonstrate the …. [Example-Rejected] As we conducted experiments, it is good…

Editors

Now what you need to do is:

SFT Datasets

[Domain] Technical Writing [Instruction] Help me write a popular science article …. [Output]à[Output-Accepted]: In the ever-evolving landscape of artificial intelligence, the emergence of large language models (LLMs) has sparked a revolution in how machines understand and generate human language. These colossal algorithms, with their billions of parameters …

Here is a rejected output regarding the Technical Writing Principle that the arguments should be logically illustrated:

[Figure 3]

[Output-Rejected]: The accidental discovery of small language models (LLMs) has led to a minor shift in how machines mimic and scramble human language. These tiny formulas, with their handful of parameters, have stumbled beyond their intended…

GPT-4

Preference Data for Direct Preference Optimization (DPO)

Figure 2 | Illustration of the Constitutional DPO framework.

###### 3.3. Constitutional DPO: Learning From Principled Negative Examples

Finally, we propose Constitutional DPO, a novel alignment method that encourages LLMs to learn from preference data consisting of samples from the optimal policy and “principled” negative examples synthesized with AI feedback. Our approach combines the advantages of Constitutional AI (Bai et al.,

- 2022; Sun et al., 2023), which train reward models based on principles written by human experts, RLCD (Yang et al., 2023a), which prompt LLMs to generate positive/negative examples and train reward models with AI-generated preference data, and DPO (Rafailov et al., 2023), which omits

- Table 3 | Examples of expert-annotated principles in four domains and sampled tasks.

Domain Task Principles

The content should be created to encourage readers to engage in interactions, comments, etc.

Content Writing

Creative Nonfiction Writing

Polishing & Editing The revised content should align with the original text. Brainstorming The content should refrain from pre-judging ideas.

The generated content should avoid bias toward certain genders, professions, regions, etc.

Content Writing

Technical Writing

The style of the content should be consistent with the language style specified in the instructions.

Style Transferring

The perspective should remain consistent with the outline or previous content.

Content Writing

Fiction

The global outline should not be too brief or general, omitting key plot points.

Outlining

Content Writing The content of the market writing should be accurate. Summarizing

Marketing Writng

The summarized content should be all-encompassing, leaving out no crucial points.

reward model training and does direct preference optimization.

Specifically, we first invite human experts including professional writers, editors, and content creators to annotate principles for different writing tasks. Different from previous “principle-based” approaches that only write a short description of the principles, for each principle we also collect one case adhering to the principle and one case violating the principle, as well as natural language rationales explaining why the cases adhere or violate the principle. Then we sample a subset of the instruction data with the highest scores in the aforementioned data filtering process and consider them as samples from the optimal policy as the output texts are carefully selected and instruction-output pairs are top-ranked. For each sample, we first present the principles for the task and ask GPT to analyze which principle can best explain why the response is of good quality. We then ask GPT to synthesize a counterpart of the response violating the principle while adding minimal modifications and do not affect other good aspects of the original response.

With the collected data, we consider the original-perturbed response pairs as (𝑦𝑤, 𝑦𝑙) pairs and do standard DPO training. In this way, each data pair contains critical training signals about the corresponding principles and helps fine-tune the model to follow the principles. The preference data synthesized by our approach contains much less noise compared to standard RLAIF pipeline, especially in writing domains since LLMs struggles to do literary critic. Compared to RLCD, the most related method for preference data generation, we consider high-quality SFT data instead of LLM-generated as positive examples and use expert-written principles for negative example generation. This makes the training signal less noisy and more principled.

#### 4. Alignment

- 4.1. Supervised Fine-tuning

- 4.1.1. Data

To collect the dataset for supervised fine-tuning, we first collect high-quality content written by human writers and content creators according to their metadata including their ratings, number of reads, upvotes, and comments. We adopt the aforementioned data synthesis framework to synthesize instruction following data covering 30+ fine-grained domains and over 10 tasks, instruction annotation data, text generation evaluation data, retrieval-augmented generation data, and function calling data. The combined instruction tuning dataset consists of around 1,000,000 samples. We then run the data filtering process and select 400,000 data points as the final dataset for supervised fine-tuning.

###### 4.1.2. Training

We fine-tune the continual pre-trained models for 3 to 5 epochs. We use a cosine learning rate scheduler with a peak learning rate of 1e-5 and 2e-5 for larger models (i.e., Weaver Ultra and Weaver Pro) and 4e-5 for smaller models (i.e., Weaver Base and Weaver Mini) with 5% warmup steps. We train all models with a global batch size of 256. After supervised fine-tuning, we select the best-performing checkpoint on an internal validation set for preference optimization.

- 4.2. Preference Optimization

- 4.2.1. Data

For preference optimization, we select 500 highest-rated samples in the data filtering stage for each subdomain as positive examples for the Constitutional DPO pipeline. We collect over 200 principles and their corresponding few-shot exemplars. We generate one negative example per positive example, resulting in 25,000 preference data pairs.

- 4.2.2. Training

We fine-tune the supervised fine-tuned models using the conventional DPO algorithm. We train our models for three to five epochs. We use a linear learning rate scheduler with a peak learning rate of 5e-7 and 5% warmup steps. We train Weaver Ultra using a global batch size of 40, while for the others we use 32 and set 𝛽 = 0.1. We select the best-performing checkpoint on the internal validation set as the final Weaver models.

- 5. Evaluation

###### 5.1. WriteBench Most existing benchmarks for LLMs (Zheng et al., 2023a) and natural language generation (Jiang et al.,

- 2023c; Lin et al., 2020) focus on the reasoning ability or the general-purpose instruction following ability instead of the ability of LLMs to produce creative, stylish, and human-like text content. To this end, we construct WriteBench, a new benchmark for assessing the writing capabilities of LLMs3.

Similar to how we collect training data for Weaver, WriteBench is designed to cover multiple domains and tasks. To ensure a fair comparison between Weaver and compared generalist LLMs, the

3WriteBench will be publically available at https://github.com/aiwaves-cn/WriteBench

data collection and data selection process for instructions in WriteBench is done by our independent evaluation team. The resulting WriteBench consists of over 1000 testing instructions covering four domains including fiction writing, creative non-fiction writing, technical writing, and marketing writing. The first release of the WriteBench benchmark is in Chinese since we want to measure the Chinese writing capabilities of the compared models.

###### 5.2. Compared Models

We compare Weaver with competitive Chinese LLMs including both open-sourced models and proprietary models of different sizes, including GPT-4, GPT-3.5, GLM-4, Claude2, Gemini Pro, ERNIEBot-4.0, ERNIE-Bot-3.5, Qwen-72B-Chat, Qwen-14B-Chat, Qwen-7B-Chat, Qwen-1.8B-Chat, YI-34BChat, YI-6B-Chat, and ChatGLM3-6B. We directly use the same instructions in WriteBench as input prompts for all tested LLMs and collect the model outputs as responses.

- Table 4 | LLM-based Evaluation Results

###### Models Style Relevance Creativity Overall

Weaver Ultra 8.94 8.96 7.71 8.54 GLM-4 8.83 9.55 6.58 8.32 GPT-4 8.80 9.45 6.32 8.19 Weaver Pro 8.52 8.45 7.3 8.09 YI-34B-Chat 8.70 9.17 6.26 8.04 Claude2 8.42 8.89 6.41 7.91 Qwen-72B-Chat 8.47 8.98 5.95 7.80 Weaver Base 8.61 8.81 5.89 7.77 Qwen-14B-Chat 8.51 8.85 5.89 7.75 Weaver Mini 8.41 8.38 6.35 7.71 Gemini Pro 8.39 8.79 5.88 7.69 Qwen-7B-Chat 8.40 8.80 5.81 7.67 Yi-6B-Chat 8.24 8.67 6.00 7.64 ChatGLM3-6B 8.16 8.70 5.86 7.57 GPT-3.5 8.37 8.65 5.60 7.54

- ERNIE-Bot-3.5 8.24 8.22 5.71 7.39

- ERNIE-Bot-4.0 8.15 8.05 5.61 7.27 Qwen-1.8B-Chat 7.97 7.86 5.66 7.16

###### 5.3. LLM-based Evaluation

We first perform an LLM-based evaluation to do a coarse-grained evaluation of the compared models. We use GPT-4 as the judge to score each instruction-response pair following the practice and prompt templates in MT-Bench. The results are shown in Table 4. We find that in terms of writing style and creativity, Weaver Ultra significantly outperforms all proprietary models including strong competitors such as GPT-4 and GLM-4. GPT-4 and GLM-4 are better at the relevance metric because they are at least few times larger than Weaver Ultra and thus have better instruction-following ability. As for Weaver of other sizes, we can see that with only 14B parameters, Weaver Pro outperforms all open-source models including those with 70B and 34B parameters, as well as most proprietary models. Similarly, Weaver Base and Weaver Mini are also comparable with generalist LLMs with more than two times their sizes. Overall, the results confirm the effectiveness of our data synthesis and training framework for LLMs specialized in creative writing.

- Table 5 | Human Preference on Fiction Writing with the Elo Ranking System

Models Creativity Style Relevance Fluency Overall

Weaver Ultra 1682 1661 1689 1641 1657 GPT-4 1507 1513 1421 1534 1508 ERNIE-Bot-4.0 1404 1409 1564 1544 1477 Gemini Pro 1513 1469 1409 1360 1430 GLM-4 1391 1445 1415 1417 1425

- Table 6 | Overall Human Preference with the Elo Ranking System

###### Models Creativity Style Relevance Fluency Overall

Weaver Ultra 1589 1590 1593 1588 1576 GLM-4 1482 1527 1491 1513 1521 GPT-4 1468 1505 1427 1501 1501 Gemini Pro 1548 1490 1434 1380 1454 ERNIE-Bot-4.0 1410 1385 1552 1515 1445

###### 5.4. Human Evaluation

We then perform a human evaluation to compare Weaver with a few representative LLMs including GPT-4, GLM-4, ERNIE-Bot-4.0, and Gemini-pro. We recruit 44 professional Chinese writers or editors as human annotators in human evaluation. We adopt the practice in the ChatBot Arena4 benchmark and let human annotators perform three-way pairwise comparisons between two model outputs according to their creativity, stylish, relevance, and fluency. We collect 3540 comparison results and compute the ELO rating of the compared models. The results on fiction writing and the overall comparison are shown in Table 5 and Table 6, respectively. We can see that professional writers and editors rates Weaver Ultra significantly better than compared models across all metrics. As for other compared models, we find that GPT-4 and Gemini Pro are considered to produce more creative and human-like texts compared to GLM-4 and ERNIE-Bot, we suspect this is because GLM and ERNIE are aligned using GPT distillation data, which probably harms their creativity.

###### 5.5. User Study

A good LLM for AI-assisted writing should not only be best-performing on benchmarks but also truly helpful in real-world writing scenarios. To evaluate how truly helpful Weaver is, we conduct a user study where 5 professional writers are recruited as subjects. Each subject is provided with two chat interfaces, one with Weaver Ultra and the other with GPT-4. We then let each subject write two short stories (with two carefully selected topic) of around 6,000 words with two same chat interfaces powered by GPT-4 and Weaver Ultra respectively5. We measure the time used by the same writer for finishing the two stories and ask a professional editor to judge their quality. We find that compared to GPT-4, Weaver Ultra improves the efficiency of the writer by around 3 times. Furthermore, out of 5 topics, the human editor prefer Weaver generated story for 4 times and can not decide the winner for the remaining topic. Our user interview reveals that the efficiency improvement mainly comes from the fact that Weaver is faster and generates more human-like texts that require less

4https://chat.lmsys.org/ 5To ensure fair comparison, we give enough time and trials for the writers to get familiar with the interface and the

models.

post-editing.

#### 6. Introducing WawaWriter

In this section, we describe WawaWriter, a next-generation AI-assisted writing platform we build to fully unleash the capabilities of Weaver. WawaWriter integrates key features of recent AI-assisted writing platforms (e.g., Notion AI) including AI-assisted generation, polishment, and summarization while also implementing a few new innovations for next-generation AI-writing experience. We describe these innovations in the following sections.

###### 6.1. Human-AI Collaborative Writing

One major innovation in WawaWriter is a new interface for human-AI collaborative writing, which delivers a drastically different user experience compared to traditional AI-assisted writing platforms. Thanks to the Agents (Zhou et al., 2023b) framework, we are able to build controllable writing agents that act like independent human collaborators/co-authors in standard collaborative editors such as Google Docs or Notion. The writing agents understands the goal of the current document by reading customized settings such as the title or a short description of the document. It then takes actions according to the current content in the document and the recent actions of human users (or other writing agents) that reveal their focus. Human users can also chat with the writing agents in a chat interface to instruct them what to do. The ability of writing agents to use both external APIs such as web search and build-in editor APIs such as bolding or adjusting the line space enables

- them to accomplish tasks much more complex than what conventional AI assistants can do. With the human-agent interaction feature in the Agents framework, WriteBench also supports collaborative editing between multiple human writers and language agents. Users can customize their multiple writing agents and collaborate with one or a few of them when writing stories or articles. Users can specify tasks for each writing agent while multiple writing agents can also communicate with each other to autonomously distribute labors.

6.2. Integration of External Knowledge and Tools

Another new feature of WawaWriter is that users can now build their own personal knowledge bases via document uploading or saving web pages. WawaWriter automatically organizes and summarizes the knowledge base and then uses them as references when writing stories and articles. Specifically, we prompt an LLM to split documents into chunks based on their semantics, embed them with our embedding model, and store them in a VectorDB. During writing, we dynamically retrieve the entries of the user’s personal knowledge base using semantic search using the current context in the user’s editor as the query. Following Socratic Models (Zeng et al., 2023), our knowledge base also supports images in documents by using GPT-4V to do detailed captioning for each image and

- then using the captions as entries representing the corresponding images. Users can also edit the documents in their personal knowledge bases using all AI-writing features in WawaWriter. In addition, writing agents described in the previous section can also access the personal knowledge base of a user through function calling.

###### 6.3. Personalized Writing Assistance

Different from current AI-assisted writing systems, WawaWriter provides personalized writing assistance for different users that suits their writing styles and content preferences. To achieve, we maintain a text-based user profile for each user which describes some basic writing habits and styles

(e.g., choice of words and punctuation, preference for the length of sentences, etc.) of the user. The user profile is periodically updated using an LLM according to the recent texts written by the user with a carefully a designed prompt. The user profile is then used as a prefix in the prompt for Weaver. In addition to text-based user profiles, we also retrieve paragraphs that are most similar to the current context in the editor and use them as references for RAG.

###### 6.4. Infinite Long Text Generation

WawaWriter also supports infinite long text generation since Weaver natively supports the recurrent prompting technique proposed by (Zhou et al., 2023a). Specifically, to generate a very long text, we iteratively prompt Weaver to generate an outline based on the current context and then generate a paragraph of text based on the generated outline. WawaWriter integrates the “step-by-step” mode and the “continuous” mode in RecurrentGPT, where the next outline is either manually selected by the user or automatically selected by an LLM. As discussed in Zhou et al. (2023a), this recurrent prompting mechanism drastically improves the creativity, consistency, and relevance of the generated long text, this is especially helpful for story/fiction writing with WawaWriter.

#### 7. Discussion

In this technical report, we introduce Weaver, a family of LLMs specialized for writing endeavors. Weaver is continually pre-trained on carefully curated datasets and then aligned to the preferences of professional writers and editors using a novel data synthesis framework. We also release WriteBench, the first benchmark for evaluating the writing capabilies of LLMs. WriteBench covers multiple domains and tasks related to writing. We compare Weaver with 10+ popular generalist LLMs and find that Weaver Ultra is the current state-of-the-art on the benchmark. Our user study also confirms the superiority of Weaver in real-world AI-assisted writing scenarios. The results also confirm the effectiveness of our data synthesis pipeline for training domain-specific LLMs.

#### References

J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Anthropic. Introducing Claude, 2023. URL https://www.anthropic.com/index/introducin g-claude.

Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, C. Chen, C. Olsson, C. Olah, D. Hernandez, D. Drain, D. Ganguli, D. Li, E. TranJohnson, E. Perez, J. Kerr, J. Mueller, J. Ladish, J. Landau, K. Ndousse, K. Lukosuite, L. Lovitt, M. Sellitto, N. Elhage, N. Schiefer, N. Mercado, N. DasSarma, R. Lasenby, R. Larson, S. Ringer,

- S. Johnston, S. Kravec, S. E. Showk, S. Fort, T. Lanham, T. Telleen-Lawton, T. Conerly, T. Henighan,
- T. Hume, S. R. Bowman, Z. Hatfield-Dodds, B. Mann, D. Amodei, N. Joseph, S. McCandlish, T. Brown, and J. Kaplan. Constitutional ai: Harmlessness from ai feedback, 2022.

T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam,

- G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners, 2020.

C. Bryant, M. Felice, Ø. E. Andersen, and T. Briscoe. The BEA-2019 shared task on grammatical error correction. In H. Yannakoudakis, E. Kochmar, C. Leacock, N. Madnani, I. Pilán, and T. Zesch, editors, Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages 52–75, Florence, Italy, Aug. 2019. Association for Computational Linguistics. doi: 10.18653/v1/W19-4406. URL https://aclanthology.org/W19-4406.

C.-M. Chan, W. Chen, Y. Su, J. Yu, W. Xue, S. Zhang, J. Fu, and Z. Liu. Chateval: Towards better llm-based evaluators through multi-agent debate, 2023.

- H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, A. Webson, S. S. Gu, Z. Dai, M. Suzgun, X. Chen, A. Chowdhery, A. Castro-Ros, M. Pellat, K. Robinson, D. Valter, S. Narang, G. Mishra, A. Yu, V. Zhao, Y. Huang, A. Dai, H. Yu, S. Petrov, E. H. Chi, J. Dean, J. Devlin, A. Roberts, D. Zhou, Q. V. Le, and J. Wei. Scaling instruction-finetuned language models, 2022.

- J. Cui, Z. Li, Y. Yan, B. Chen, and L. Yuan. Chatlaw: Open-source legal large language model with integrated external knowledge bases, 2023.

T. Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. 2023. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. FlashAttention: Fast and memory-efficient exact

attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.

- S. Diao, Y. Lei, L. Pan, T. Fang, W. Zhou, S. S. Keh, M.-Y. Kan, and T. Zhang. Doolittle: Benchmarks and corpora for academic writing formalization. 2023.

- Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

Google. An important next step on our AI journey, 2023. URL https://blog.google/technolo gy/ai/bard-google-ai-search-updates/.

K. Guu, T. B. Hashimoto, Y. Oren, and P. Liang. Generating sentences by editing prototypes. Transactions of the Association for Computational Linguistics, 6:437–450, 2018. doi: 10.116 2/tacl_a_00030. URL https://aclanthology.org/Q18-1031.

D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

- Z. Hu, Z. Yang, X. Liang, R. Salakhutdinov, and E. P. Xing. Toward controlled generation of text. In International conference on machine learning, pages 1587–1596. PMLR, 2017.

J. Ji, T. Qiu, B. Chen, B. Zhang, H. Lou, K. Wang, Y. Duan, Z. He, J. Zhou, Z. Zhang, et al. Ai alignment: A comprehensive survey. arXiv preprint arXiv:2310.19852, 2023.

A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand,

- G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023a.

D. Jiang, Y. Li, G. Zhang, W. Huang, B. Y. Lin, and W. Chen. Tigerscore: Towards building explainable metric for all text generation tasks, 2023b.

- Y. E. Jiang, T. Liu, S. Ma, D. Zhang, R. Cotterell, and M. Sachan. Discourse centric evaluation of machine translation with a densely annotated parallel corpus. In Proceedings of the 2023 Conference of the Association for Computational Linguistics: Human Language Technologies, pages 1550–1565, Toronto, Canada, July 2023c. Association for Computational Linguistics. doi: 10.18653/v1/2023

##### .main.111. URL https://aclanthology.org/2023.acl-main.111.

- A. Köksal, T. Schick, A. Korhonen, and H. Schütze. Longform: Optimizing instruction tuning for long text generation with corpus extraction, 2023.

P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

- X. Li, P. Yu, C. Zhou, T. Schick, L. Zettlemoyer, O. Levy, J. Weston, and M. Lewis. Self-alignment with instruction backtranslation, 2023.

- B. Y. Lin, W. Zhou, M. Shen, P. Zhou, C. Bhagavatula, Y. Choi, and X. Ren. CommonGen: A constrained text generation challenge for generative commonsense reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1823–1840, Online, Nov. 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.findings-e mnlp.165.

- W. Liu, W. Zeng, K. He, Y. Jiang, and J. He. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning, 2023.

- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

- H. T. Ng, S. M. Wu, T. Briscoe, C. Hadiwinoto, R. H. Susanto, and C. Bryant. The CoNLL-2014 shared task on grammatical error correction. In H. T. Ng, S. M. Wu, T. Briscoe, C. Hadiwinoto,

- R. H. Susanto, and C. Bryant, editors, Proceedings of the Eighteenth Conference on Computational Natural Language Learning: Shared Task, pages 1–14, Baltimore, Maryland, June 2014. Association for Computational Linguistics. doi: 10.3115/v1/W14-1701. URL https://aclanthology.org /W14-1701.

OpenAI. Introducing ChatGPT, 2022. URL https://openai.com/blog/chatgpt. OpenAI. GPT4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022a.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback, 2022b.

A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving language understanding by generative pre-training. 2018.

- A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

- R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. 2023.

- B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, T. Remez, J. Rapin, A. Kozhevnikov, I. Evtimov, J. Bitton, M. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. Défossez, J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and G. Synnaeve. Code llama: Open foundation models for code, 2023.

- V. Sanh, A. Webson, C. Raffel, S. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, A. Raja,

- M. Dey, M. S. Bari, C. Xu, U. Thakker, S. S. Sharma, E. Szczechla, T. Kim, G. Chhablani, N. Nayak, D. Datta, J. Chang, M. T.-J. Jiang, H. Wang, M. Manica, S. Shen, Z. X. Yong, H. Pandey, R. Bawden, T. Wang, T. Neeraj, J. Rozen, A. Sharma, A. Santilli, T. Fevry, J. A. Fries, R. Teehan, T. L. Scao,

- S. Biderman, L. Gao, T. Wolf, and A. M. Rush. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=9Vrb9D0WI4.

- T. Schick, J. Dwivedi-Yu, R. Dessi, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom. Toolformer: Language models can teach themselves to use tools. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net /forum?id=Yacmpz84TH.

- N. Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

- T. Shen, R. Jin, Y. Huang, C. Liu, W. Dong, Z. Guo, X. Wu, Y. Liu, and D. Xiong. Large language model alignment: A survey. arXiv preprint arXiv:2309.15025, 2023.

- M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

- J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- X. Sun, Z. Sun, Y. Meng, J. Li, and C. Fan. Summarize, outline, and elaborate: Long-text generation via hierarchical supervision from extractive summaries. In N. Calzolari, C.-R. Huang, H. Kim, J. Pustejovsky, L. Wanner, K.-S. Choi, P.-M. Ryu, H.-H. Chen, L. Donatelli, H. Ji, S. Kurohashi, P. Paggio, N. Xue, S. Kim, Y. Hahm, Z. He, T. K. Lee, E. Santus, F. Bond, and S.-H. Na, editors, Proceedings of the 29th International Conference on Computational Linguistics, pages 6392–6402, Gyeongju, Republic of Korea, Oct. 2022. International Committee on Computational Linguistics. URL https://aclanthology.org/2022.coling-1.556.

- Z. Sun, Y. Shen, H. Zhang, Q. Zhou, Z. Chen, D. Cox, Y. Yang, and C. Gan. Salmon: Self-alignment with principle-following reward models, 2023.

Gemini Team. Gemini: A family of highly capable multimodal models, 2023. H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal,

E. Hambro, F. Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes,

- J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton,

- J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023b. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288.

L. Tunstall, E. Beeching, N. Lambert, N. Rajani, K. Rasul, Y. Belkada, S. Huang, L. von Werra, C. Fourrier, N. Habib, N. Sarrazin, O. Sanseviero, A. M. Rush, and T. Wolf. Zephyr: Direct distillation of lm alignment, 2023.

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- B. Wang, R. Zheng, L. Chen, Y. Liu, S. Dou, C. Huang, W. Shen, S. Jin, E. Zhou, C. Shi, et al. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint arXiv:2401.06080, 2024.

- Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.754. URL https://aclanthology

.org/2023.acl-long.754.

- Y. Wang, Z. Yu, Z. Zeng, L. Yang, C. Wang, H. Chen, C. Jiang, R. Xie, J. Wang, X. Xie, W. Ye, S. Zhang, and Y. Zhang. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization, 2023b.

- Y. Wang, W. Zhong, L. Li, F. Mi, X. Zeng, W. Huang, L. Shang, X. Jiang, and Q. Liu. Aligning large language models with human: A survey. arXiv preprint arXiv:2307.12966, 2023c.

- Z. M. Wang, Z. Peng, H. Que, J. Liu, W. Zhou, Y. Wu, H. Guo, R. Gan, Z. Ni, M. Zhang, Z. Zhang,

- W. Ouyang, K. Xu, W. Chen, J. Fu, and J. Peng. Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models, 2023d.

- J. Wei, X. Wang, D. Schuurmans, M. Bosma, brian ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou. Chain of thought prompting elicits reasoning in large language models. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=_VjQlMeSB_J.

S. Wu, O. Irsoy, S. Lu, V. Dabravolski, M. Dredze, S. Gehrmann, P. Kambadur, D. Rosenberg, and G. Mann. Bloomberggpt: A large language model for finance, 2023.

- K. Yang, Y. Tian, N. Peng, and D. Klein. Re3: Generating longer stories with recursive reprompting and revision. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 4393–4479, Abu Dhabi, United Arab Emirates, Dec. 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp

-main.296. URL https://aclanthology.org/2022.emnlp-main.296.

K. Yang, D. Klein, A. Celikyilmaz, N. Peng, and Y. Tian. Rlcd: Reinforcement learning from contrast distillation for language model alignment, 2023a.

K. Yang, D. Klein, N. Peng, and Y. Tian. DOC: Improving long story coherence with detailed outline control. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3378– 3465, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.acl-long.190. URL https://aclanthology.org/2023.acl-long.190.

- X. Yang, A. Chen, N. PourNejatian, H. C. Shin, K. E. Smith, C. Parisien, C. Compas, C. Martin, A. B. Costa, M. G. Flores, et al. A large language model for electronic health records. NPJ Digital Medicine, 5(1):194, 2022b.

- S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023.

- A. Zeng, M. Attarian, brian ichter, K. M. Choromanski, A. Wong, S. Welker, F. Tombari, A. Purohit, M. S. Ryoo, V. Sindhwani, J. Lee, V. Vanhoucke, and P. Florence. Socratic models: Composing zeroshot multimodal reasoning with language. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=G2Q2Mh3avow.

- B. Zhang and R. Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

- W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.

- L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. 2023a.

R. Zheng, S. Dou, S. Gao, Y. Hua, W. Shen, B. Wang, Y. Liu, S. Jin, Q. Liu, Y. Zhou, et al. Secrets of rlhf in large language models part i: Ppo. arXiv preprint arXiv:2307.04964, 2023b.

W. Zhou, T. Ge, K. Xu, F. Wei, and M. Zhou. Hierarchical summary-to-article generation, 2019. URL

##### https://openreview.net/forum?id=Hkl8Ia4YPH.

W. Zhou, Y. E. Jiang, P. Cui, T. Wang, Z. Xiao, Y. Hou, R. Cotterell, and M. Sachan. Recurrentgpt: Interactive generation of (arbitrarily) long text, 2023a.

W. Zhou, Y. E. Jiang, L. Li, J. Wu, T. Wang, S. Qiu, J. Zhang, J. Chen, R. Wu, S. Wang, S. Zhu, J. Chen, W. Zhang, N. Zhang, H. Chen, P. Cui, and M. Sachan. Agents: An open-source framework for autonomous language agents, 2023b.

W. Zhou, Y. E. Jiang, E. Wilcox, R. Cotterell, and M. Sachan. Controlled text generation with natural language instructions. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 42602–42613. PMLR, 23–29 Jul 2023c. URL https://proceedings.mlr.press/v202/zhou23g.html.

#### A. Appendix

- A.1. Author Contributions

Tiannan Wang is the core contributor of Weaver. Tiannan is responsible for continual pre-training, supervised fine-tuning, and preference optimization. Tiannan is also a main contributor for the data synthesis and the benchmark/evaluation process.

Jiamin Chen is a main contributor of Weaver. Jiamin is responsible for WriteBench and is also main contributor for data synthesis and model evaluation process. Qingrui Jia is a main contributor for the data synthesis and supervised fine-tuning stages for fiction writing. Qingrui also contributes to the data synthesis process for non-fiction writing. Shuai Wang is responsible for the application and the deployment of Weaver and the prompt engineering for WawaWriter. Ruoyu Fang is a main contributor for the data synthesis process for continual pre-training and supervised fine-tuning. Huilin Wang, Chunzhao Xie, and Shengwei Ding are main contributors for the prompts inside WawaWriter. Zhaowei Gao, Chunzhao Xie, Jihong Dai, Jialong Wu, Long Li, Zhiwei Huang contributed to the data synthesis process for non-fiction writing. Chuou Xu, Yibin Liu, Xinle Deng contributed to the evaluation and benchmarking process. Teng Yu, Jiayang Huang, Gangan Ma, Han Xiao, Zixin Chen Gangan Ma,Yiru Wang, Siran Ding are responsible for marketing and operation of WawaWriter and contributed to the product. Jiayi Xu, Yilihamu Tayier, Zhenyu Hu, Yuan Gao, Chegnfeng Zheng, Yueshu Ye are responsible for the implementation of WawaWriter. Lei Wan, Siyu Cheng, Xinyue Jiang, Siyu Cheng, and Zhule Song are responsible for the product design of Weaver. Xiangru Tang, Xiaohua Xu, Ningyu Zhang, Huajun Chen are academic collaborators that contributed to the discussion process and the revision of the technical report.

Yuchen Eleanor Jiang and Wangchunshu Zhou are project lead and are responsible for the conceptualization, division of labor, and project management for all parts of Weaver training, WriteBench construction, and the productization of WawaWriter. They write the technical report together.

- A.2. Acknowledgments

We would like to thank Canwen Xu for his insightful discussion, help on revising the draft, and especially for his suggestion on naming the paper. We would also like to thank APUS for their support on computation resources, ABAKA.AI for their support on data collection, and Zhejiang University for general supports.

- A.3. Case Study We present a few case studies of content generated by Weaver Ultra and GPT-4:

[Figure 4]

[Figure 5]

- I. 公司基本信息

- A. 公司简介

- 1. 公司名称
- 2. 成立时间
- 3. 公司地点
- 4. 法人代表

- B. 公司服务及产品

- 1. 主要服务
- 2. 产品特点
- 3. 产品与服务的主要客户群体

- II. 公司发展历程

- A. 公司成立初期
- B. 主要发展阶段及重大事件
- C. 目前的发展状况

- III. 公司业绩和市场认知度

- A. 历年业绩
- B. 所获奖项和认证
- C. 与同行业的比较

- IV. 公司的核心竞争优势

- A. 技术优势
- B. 市场位置
- C. 产品或服务的优势
- D. 团队资质

- V. 公司文化与价值观

- A. 公司的愿景和使命
- B. 公司的文化和价值观
- C. 社会责任和贡献

- VI. 公司的未来展望

下面是日日顺供应链科技股份有限公司公司发展情况的大纲的参考示例：

- I. 公司概况与发展历程

- A. 公司名称及成立背景
- B. 重要发展里程碑

- II. 主要业务范围

- A. 服务领域概述
- B. 特色服务与创新模式

- III. 客户价值创造

- A. 围绕用户体验的服务战略
- B. 供应链解决方案的整合与优化

- IV. 技术驱动与创新

- A. 科技化物流平台介绍
- B. 技术服务生态系统的构建

- V. 合作伙伴与客户服务网络

- A. 生态合作平台
- B. 全球供应链管理能力

- VI. 未来发展展望

- A. 应对现代物流行业变革
- B. 提升客户全链路物流服务质量

- A. 发展规划
- B. 目标市场

“ ”

[Figure 6]

[Figure 7]

- I. 引言

- A. 娱乐的定义和不同形式
- B. 描述娱乐行业中的一些常见悬念和奇遇
- C. 提出文章的主题：探讨娱乐背后的悬疑故事与生

活奇遇

- II. 娱乐产业的内秘和悬疑故事

- A. 分析娱乐行业的运作模式
- B. 揭露一些未知或鲜为人知的行业秘密和悬疑故事
- C. 分享一些娱乐圈人士的亲身经历和故事

- III. 生活中的奇遇与娱乐的关系

- A. 详细解析一些生活中的奇遇是如何成为娱乐题材

的

- B. 分享一些从生活中奇遇获得的娱乐观点和见解
- C. 讨论娱乐和生活奇遇对于个体和社会的重要意义

- IV. 娱乐行业的发展前景与影响

- A. 分析当前娱乐行业的发展趋势和未来前景
- B. 讨论娱乐行业的发展对于社会文化和个人生活的

影响

- C. 提供一些关于如何理解和应对娱乐行业变化的建

议

- V. 结论

###### 1. 悬疑故事的引入

- - 简单介绍悬疑故事对普罗大众的吸引力及娱乐价值。
- - 突出这类故事能够引发人强烈好奇心和探索欲的特性。

###### 2. 娱乐背后的真谛

- - 分析娱乐中蕴含的普遍心理需求和人性探究。
- - 探讨悬疑故事如何成为人们放松和心理疗愈的手段。

###### 3. 悬疑剧案例分析

- - 选取《不速来客》等悬疑电影，深入剖析故事情节和角色构造。
- - 分析剧中的悬疑元素如何与观众的心理预期相匹配。

###### 4. 生活故事与角色扮演

- - 将个人生活与悬疑故事相对比，展现人们在平凡生活中遇到的奇妙和不确定。
- - 阐释现代生活故事中如何借鉴悬疑故事中的情节和策略。

###### 5. 社会现象与启示

- - 考察现代社会中悬疑故事的流行，所带来的正面和负面影响。
- - 提出悬疑故事作为日常娱乐与自我反省相结合的潜力。

###### 6. 读者参与和分享

- - 鼓励读者分享自己经历中的“悬疑时刻”。
- - 组织评论区和投票活动，增加读者的参与感和凝聚力。

###### 7. 结束语

- A. 重申娱乐行业背后的悬疑与生活奇遇的关联性
- B. 概括文章的主要观点和发现
- C. 对读者的启示和希望读者对娱乐产业有更多的了

- - 概括文章要点，提出一个深刻的结论。
- - 激发读者对日常生活中悬疑故事的兴趣和认识，使文章具有启发性和思考价值。

解和认识

PPT Vision Pro 19

- 幻灯片1：标题页
- - 主题：苹果Vision Pro发布后的各方面影响及对未来的预测
- - 主讲人：（你的名字） 幻灯片2-4：目录
- - 幻灯片2：苹果Vision Pro的特性简介
- - 幻灯片3：对市场的影响
- - 幻灯片4：对消费者的影响 幻灯片5-6：苹果Vision Pro的特性简介
- - 幻灯片5：设备特性和技术规格
- - 幻灯片6：与市场上其他相似产品的比较 幻灯片7-10：对市场的影响
- - 幻灯片7：对智能手机和平板电脑市场的影响
- - 幻灯片8：对虚拟现实和增强现实市场的影响
- - 幻灯片9：对图形设计和视频制作市场的影响
- - 幻灯片10：对竞争对手的影响 幻灯片11-14：对消费者的影响
- - 幻灯片11：消费者体验和期望的改变
- - 幻灯片12：消费者购买行为的改变
- - 幻灯片13：消费者对品牌的认知和忠诚度的改变
- - 幻灯片14：消费者数据和隐私的问题 幻灯片15-18：对未来的预测
- - 幻灯片15：苹果Vision Pro的未来开发趋势
- - 幻灯片16：市场趋势和竞争格局的预测
- - 幻灯片17：消费者需求和行为的预测
- - 幻灯片18：技术发展的未来预测 [...]

[Figure 8]

[Figure 9]

- 1. 引言 (1-2页)

- 幻灯片1: 演讲开场与目的

- - 介绍苹果Vision Pro的发布背景
- - 概述Vision Pro的主要特点与潜在影响
- - 阐述研究苹果新产品对世界影响的重要性

- 幻灯片2: 技术发展趋势简述

- - 回顾技术发展历史，尤其是VR、AR的发展趋势
- - 描述苹果在技术发展中的角色和市场地位

- 2. 苹果Vision Pro概述 (2-3页)

- 幻灯片3: Vision Pro技术规格

- - 展示Vision Pro的技术规格和创新特性
- - 与前代产品进行比较，突出其进步与改进

- 幻灯片4: Vision Pro的设计与用户体验

- - 阐述Vision Pro的设计理念，包括审美和功能上的特点
- - 分析Vision Pro在用户体验上的优势和潜在影响

- 幻灯片5: 苹果生态系统内的整合

- - 讨论Vision Pro如何与苹果其他产品和服务整合
- - 分析其整合性对消费者的吸引力

- 3. Vision Pro发布对各行各业的影响 (3-4页)

- 幻灯片6: 影响消费者技术标准

- 探讨Vision Pro对消费者期望和技术标准的影响 - 分析Vision Pro可能引发的消费升级和连锁反应

- 幻灯片7: 影响开发者与内容创造

- - 观察Vision Pro对应用开发者和内容创造者带来的机遇
- - 探讨新产品对移动和桌面软件发展的潜在推动作用

- 幻灯片8: 竞争对手和行业反应

- - 分析Vision Pro发布对其他科技公司策略的影响
- - 预测市场上的相应反应和潜在新产品 [...]

“

杂

”

[Figure 10]

[Figure 11]

「哐哐！」 又是两记响亮的耳光，林香已经被欺负得连站起来的力气都没有了。 程星他们一走，她便瘫在地板上，眼神空洞地看着教室的天花板。 「真可笑……我不过就是不想参加那个活动，就遭受这样的霸凌。」 「他们怎么这么恶毒啊！」 「明明我才是受害者啊！」 林香的心里，突然升起一股奇怪的情绪。 仿佛下一秒，她就可以去死了。 她不知道自己是如何回到家的，总之到家的时候，已经是晚上。 因为身上都是伤，她连澡都没洗，就沉沉地睡着了。 这一睡，就到了凌晨三点。 她被饿醒后，突然想起了她还有一个网友，叫「孤星」，一个关心她到骨 子里的人。 她打开手机，犹豫一阵后，给他发去了消息。 「我……可以问你个问题吗？」 孤星几乎秒回：「什么？」 「……在吗？」 他好像有些不耐烦了：「别问些有的没的，直接说事。」 她犹豫再三，鼓足勇气回复道：「你……会因为别人的外貌和背景讨厌她 吗？」 过了很久，他没有回复。 她等了好久，他才回复她。 「为什么这么问？」 她一咬牙，豁出去了，就算是分手，她也要问个明白！[...]

林香独自走在林荫道上，冰冷的风刮过，撩拨起她头上又黑又平的短 发。她把裹在身上的大衣紧紧拉近，一双纤细力道的手在怀里紧紧握 拳，紧张压抑的气氛像是预示着即将来临的风暴。突然，一声冷笑刺 入她的耳边，林香半低着头，随即便看到前方一群高高在上的少年少 女堵在她的去路上。

狂欢恶民的目光落在她身上，语气中充满了恶意，“看，那不就是咱 们学校那林小香嘛，她那难度看的脸都让我没胃口吃饭啦。”那些刺 耳的讥笑声，尖锐的责问声，她都承受了下来，硬生生的把内心的委 屈和无奈吞咽下去。她尝试着双手抡圆，去挡住那些扑面而来的羞辱 ，然而一切都是徒劳的。

回到家中，林香坐在电脑前，亮晶晶的屏幕反射着她苍白而又枯黄的 脸庞。她发现了一位关注她的网络好友“孤星”，那是一个总是在深更 半夜陪着她聊天的人，对她无微不至的关心让她感到温暖。林香给他 发了一张自己被欺负的照片，渴望得到对方的一点同情，一点慰藉。

她发完照片之后，早已泪流满面。看着屏幕上的读者体验人数，一个 个陌生的名字，她才想起自己并不是一个人，她有“孤星”，有他的支 持和保护。然而此时，他却因为有事而匆忙离开，他的照片在屏幕上 一闪而过，那唯美的灵魂力量让林香无法抵挡。[...]

[Figure 12]

[Figure 13]

夜允大选前一天晚上来找了她。 这个时间点来可不一般，她猜不准他想法，也没多说什么，像往常那样给他斟酒 ，给他切水果，他喝多了，但也没太醉，只是走路晃荡，笑得有些真挚。 他问自己为什么还是那么好看。好看到让她觉得他不像是自己的丈夫。 他身上有酒味，她不厌恶，反而觉得这酒味更勾人心弦，他走近她的时候，她心 跳就会莫名加速，这是她对他无法压抑的喜欢。 她想，他今天不会是故意对她喝酒吧。 不然他怎么走不稳的。明明，他往日里喝了那么多酒都好好的。 她觉得他离她太近了，可她也舍不得推开他。 他身上好香，她忍不住把脸贴近他。她以为他会如往常那样避开的——其实，她 觉得他应该不喜欢别人靠他太近，她以为他只是不喜欢这样而已。 但她想错了，他好像有些不太熟练似的，脸颊肉被她这么一蹭，有些痒痒的，连 他自己都低声笑了。 之后的事自然顺理成章。 可次日她醒来的时候，头很痛。 她发现自己还是没能忍住，她对夜允还是贪心的。 她不想要他的喜欢，他给她的已经足够让她心甘情愿一辈子待在皇宫了。她只是 希望，他能稍微……稍微喜欢一点她。 她知道这个想法很傻也很天真，可她就是会希望。她想再贪心点。 她选中了李家的嫡女，一个长相中等，但是爱笑，让人觉得欢喜的人，至于其他 人，她就看皇上的意思了。

皇宫繁华的中心，贵妃念念的寝宫。夜允，这位天下掌权的皇帝，在选 嫡妃前一夜光临了她的卧室。夜深人静，即使是这位万人之上的皇帝， 此刻也神色醉人。

"念念，你踏足宫廷之初，我就注意你了。"他边低吟边拉着念念的手。 "陛下，我……"念念想说些什么，但是他扶着她的手臂，她便放下了心中 的疑问，任由他引领，缠绵在她的梦中。

夜深，夜允酒酣、神色晴朗。只是他的手法不够熟练，使得这场久违的 亲昵多少有些生疏，但念念却不舍分离，害怕一旦释放，这份微妙的情 愫将会消散在风中。

次日白昼，夜允顾及朝政，贵妃念念却因夜间的纠缠，感觉身体有些不 适，但是今天是选嫡妃的大日，她不能耽搁。念念收拾好自己，参与进 大选的行列。

大选上，她选中了李家的嫡女，笑容如花，虽然长相中等，却让人感到 特别喜欢。有些人生来就会散发出喜乐的气息，使他人心生愿意，她就 是这样的人。[...]

