## YAYI 2: Multilingual Open-Source Large Language Models

Yin Luo∗, Qingchao Kong∗, Nan Xu∗, Jia Cao, Bao Hao, Baoyu Qu, Bo Chen, Chao Zhu Chenyang Zhao, Donglei Zhang, Fan Feng, Feifei Zhao, Hailong Sun, Hanxuan Yang, Haojun Pan Hongyu Liu, Jianbin Guo, Jiangtao Du, Jingyi Wang, Junfeng Li, Lei Sun, Liduo Liu, Lifeng Dong Lili Liu, Lin Wang, Liwen Zhang, Minzheng Wang, Pin Wang, Ping Yu, Qingxiao Li, Rui Yan, Rui Zou Ruiqun Li, Taiwen Huang, Xiaodong Wang, Xiaofei Wu, Xin Peng, Xina Zhang, Xing Fang, Xinglin Xiao Yanni Hao, Yao Dong, Yigang Wang, Ying Liu, Yongyu Jiang, Yungan Wang, Yuqi Wang Zhangsheng Wang, Zhaoxin Yu, Zhen Luo, Wenji Mao, Lei Wang∗, Dajun Zeng∗ Beijing Wenge Technology Co., Ltd. Institute of Automation, Chinese Academy of Sciences

# arXiv:2312.14862v1[cs.CL]22Dec2023

### Abstract

As the latest advancements in natural language processing, large language models (LLMs) have achieved human-level language understanding and generation abilities in many realworld tasks, and even have been regarded as a potential path to the artificial general intelligence. To better facilitate research on LLMs, many open-source LLMs, such as Llama 2 and Falcon, have recently been proposed and gained comparable performances to proprietary models. However, these models are primarily designed for English scenarios and exhibit poor performances in Chinese contexts. In this technical report, we propose YAYI 2, including both base and chat models, with 30 billion parameters. YAYI 2 is pre-trained from scratch on a multilingual corpus which contains 2.65 trillion tokens filtered by our pre-training data processing pipeline. The base model is aligned with human values through supervised fine-tuning with millions of instructions and reinforcement learning from human feedback. Extensive experiments on multiple benchmarks, such as MMLU and CMMLU, consistently demonstrate that the proposed YAYI 2 outperforms other similar sized open-source models.

### 1 Introduction

Large language models (LLMs) (Vaswani et al., 2017; Kaddour et al., 2023) have shown strong capabilities in language understanding and comprehension (Brown et al., 2020), as well as in commonsense Q&A, programming and logical reasoning (Lightman et al., 2023). Since the launch of ChatGPT, a large number of LLMs have been proposed by different institutions and companies around the

*Corresponding authors: {yin.luo, nan.xu, lei.wang} @wenge.com, {qingchao.kong, dajun.zeng}@ia.ac.cn.

world, which mostly serve as intelligent personal assistants through a chat interface, and excel at creative writing, summarizing texts, planning activities, etc. Due to the comprehensive capabilities, LLMs are even regarded as a potential path towards the artificial general intelligence (AGI).

Terabytes of training data and expensive computing resources have become the main bottlenecks restricting the development of LLMs. Several representative LLMs-based products such as ChatGPT and Claude (Bai et al., 2022) are closed-source models. To make it more accessible for researchers, many open-source LLMs have been proposed. For example, BLOOM (Workshop et al., 2022) is the first multilingual LLM with 175 billion parameters trained on the ROOTS corpus. Llama (Touvron et al., 2023a,b) series models have achieved comparable performances with GPT-3.5 and Palm 2 (Anil et al., 2023) by training on more text tokens with better quality. Besides the ROOTs corpus, more datasets such as RedPajama (Computer, 2023) and RefinedWeb (Penedo et al., 2023) are open-sourced to further facilitate LLMs training. However, these open-source datasets contain only a small portion of Chinese text and lack the common knowledge and culture about China, which severely limits the applications of open-source LLMs in Chinese-related scenarios. To fill this gap, several Chinese-based LLMs are proposed, including ChatGLM (Zeng et al., 2023), Baichuan 2 (Yang et al., 2023) and Qwen (Bai et al., 2023).

In this technical report, we propose a series of multilingual LLMs, denoted as YAYI ( 雅意 ) 2, including base and chat models, both with 30 billion parameters. YAYI 2 models are trained on 2.65 trillions tokens on a computing cluster of over 1000 A800 GPUs. For pre-training dataset, we collect

[Figure 1]

Figure 1: Training procedure of YAYI 2 base and chat models.

over 240 terabytes of texts, including news, books, Wikipedia, code, etc., of which 41.5% are Chinese. In addition, we design a rigorous pre-training data processing pipeline, consisting of normalizing, heuristic cleaning, multi-level deduplication, and toxicity filtering. To speed up the training and inference speed, the FlashAttention 2 (Dao, 2023) and multi-query attention (MQA) (Shazeer, 2019) are adopted. We elaborate the training details and optimizing techniques to improve the training efficiency. We align YAYI 2 base model through supervised fine-tuning (SFT) with millions of instruction-output pairs and reinforcement learning from human feedback (RLHF), with better support for long instructions and multi-turn conversations. The training procedure of YAYI 2 base and chat models are shown in Figure 1. We conduct comprehensive experiments to evaluate the effectiveness of the proposed base model. The experimental results show that the proposed model outperforms other similar-sized open-source LLMs on benchmarks covering knowledge understanding, math reasoning, and programming, and even demonstrates superiority on some benchmarks over the LLM with much larger parameters.

### 2 Pre-Training

This section provides details on the pre-training process from four aspects: pre-training data, tokenization, model architecture, and training details. We first summarize the sources of pre-training data and propose a self-developed data processing pipeline. Leveraging high-quality cleaned data, we construct YAYI 2 multilingual tokenizer. Next, we elaborate the model architecture and parameter settings. Finally, we introduce the computing cluster configuration and training strategies along with some model training tricks.

2.1 Pre-Training Data 2.1.1 Data Distribution

The objective of pre-training is to accumulate a wide range of knowledge from all over the world and acquire various professional proficiency such as math, coding, and logical reasoning, which should give the model’s responsiveness to multilingual scenarios and diverse data formats. In pursuit of the above goals, a large amount of internet data is used to train the language understanding and expression capabilities, supplemented by curated general data and domain-specific data to further enhance professional skills of the model. Figure 3&4 show the distributions of data categories and languages, respectively. Details of the data distribu-

[Figure 2]

Figure 2: Pre-training data processing pipeline.

[Figure 3]

[Figure 4]

Figure 4: Language distribution.

Figure 3: Data distribution.

tion are as follows:

- • Internet data primarily comprises private data consisting of social media, internet web documents and high-quality open-source datasets. In our selection of data sources, we exclude certain open-source datasets, such as OSCAR (Ortiz Suárez et al., 2019), that can contain harmful information.
- • Curated general data covers a wide range of categories including books (e.g., textbooks, novels), codes, encyclopedias, forums, academic papers, authoritative news, laws and regulations.
- • Domain-specific data encompasses popular fields such as finance, taxation, media and

publicity, public opinion, and traditional Chinese medicine.

#### 2.1.2 Preprocessing

We establish a comprehensive data processing pipeline to enhance data quality in all aspects. This pipeline comprises four modules: normalizing, heuristic cleaning, multi-level deduplication, and toxicity filtering. Through comprehensive performance optimization, the pipeline significantly reduces the response time for processing terabytescale data to a few hours. Figure 2 illustrates the complete pre-training data processing pipeline. 240 terabytes of raw data are collected for pre-training, and only 10.6 terabytes of high-quality data remain after preprocessing.

Normalizing Through normalization, all raw data are formatted as JSON with keys such as data

YAYI 2 ChineseAlpaca 2 ChatGLM Baichuan 1 XVERSE Vocab Size 81920 55296 64794 64000 100534

Chinese-English bilingual 0.480 ± 0.209 0.502 ± 0.230 0.482 ± 0.227 0.502 ± 0.239 0.640 ± 0.278 Multilingual 0.476 ± 0.215 0.642 ± 0.434 0.551 ± 0.294 0.570 ± 0.288 0.669 ± 0.286

Table 1: Comparison of compression ratio.

source, identifier, and content. Additionally, a language detector model is employed for language detection.

Heuristic Cleaning We introduce a heuristic multi-level cleaning strategy, building a collaborative filtering mechanism based on chapters, lines, words, and characters. For dozens of data categories such as encyclopedias, Q&A, news, books, and codes, we devise over a thousand heuristic cleaning rules, tackling issues in formats, contents and encoding. At the chapter level and line level, the strategy concentrates on semantic issues such as garbled characters, logical confusion, and lowquality lines. At the word level, the emphasis is on eliminating advertising trigger words, while at the character level, the strategy scrutinizes cases of redundant and missing characters.

Multi-level Deduplication To filter various duplication patterns, we adopt a multi-level collaborative deduplication strategy, including the chapterlevel deduplication based on URL and simHash, the paragraph-level deduplication based on cosine similarity, and the sentence-level deduplication based on prefix-suffix matching.

Toxicity Filtering The Internet contains a substantial amount of harmful and false information, including but not limited to pornography, violence, prejudice, discriminatory speech, personal attacks, and illegal activities. To alleviate this problem, we propose a dual-filtering mechanism, which uses a Yayi 2 Checker model based on sensitive words for screening at the first stage and employs a classification model based on quantum heuristic language to complete secondary filtering.

#### 2.2 Tokenization

In the international landscape, most LLMs are centered around English, limiting their generalization ability in other languages. Similarly, LLMs released in China tend to focus on bilingual scenarios (Chinese and English), lacking a multilingual training corpus. To enhance the model’s comprehension and analytical capabilities across various languages,

the YAYI 2 models employ a well-trained multilingual tokenizer.

Training Data The tokenizer of YAYI 2 is trained on a 500GB high-quality multilingual corpus, which covers over ten commonly used languages including Chinese, English, French, Russian, etc. The diverse sources of training data encompass web pages, social media, books, newspapers, academic papers, etc.

Vocab Size To support minor languages while maintaining the proficiency in Chinese and English, the YAYI 2 tokenizer expands the vocabulary size to 80,000. Moreover, to harness the tensor parallelization technology and tensor cores efficiently, the vocabulary size needs to be divisible by 128. Thus, we adopt 81,920 as the final vocab size.

Normalization The YAYI 2 tokenizer adopts a unique approach by directly utilizing raw text for training without undergoing normalization. This strategy ensures the model’s adeptness in handling general scenarios.

Algorithm By training using the Byte-Pair Encoding (BPE) algorithm (Shibatay et al., 1999) from the Sentence-Piece library (Kudo and Richardson, 2018), the YAYI 2 tokenizer exhibits a robust approach. During training, each digit of a number is intelligently split to facilitate mathematical reasoning. The manually curated vocabulary includes an array of HTML identifiers, common punctuation to enhance segmentation accuracy, and 200 reserved slots for potential applications like adding identifiers during SFT. As a byte-level segmentation algorithm, the YAYI 2 tokenizer excels in handling unknown characters.

Evaluation The performance of the tokenizer is measured by the compression ratio, which is defined as follows:

Ltoken Lorigin

r =

(1)

where r denotes the compression ratio, Ltoken and Lorigin denote the lengths of the tokenized text and

original text, respectively. The lower compression ratio signifies a higher efficiency performance of the tokenizer.

To comprehensively evaluate the YAYI 2 tokenizer’s multilingual performance, we sample data from the SlimPajama (Shen et al., 2023) dataset and internal data with a length of 10,000 tokens for each, covering Chinese, English, and various minor languages. The results presented in Table 1 reveal that, in both bilingual (CH-EN) and multilingual scenarios, the YAYI 2 tokenizer outperforms other Chinese models such as Baichuan 1 (Baichuan, 2023), ChatGLM (Zeng et al., 2023), Chinese Alpaca 2 (Cui et al., 2023), XVERSE (XVERSE, 2023), boasting a lower compression ratio indicative of superior training and inference efficiency.

#### 2.3 Model Architectures

The YAYI 2 models are based on the Transformer architecture (Vaswani et al., 2017), embracing the decoder-only structure and training in the autoregressive manner. This architecture, adopted by most prominent LLMs like GPT (Brown et al., 2020), BLOOM (Workshop et al., 2022), LLaMA (Touvron et al., 2023a,b) and Baichuan (Yang et al., 2023), offers advantages such as efficient computation, lower memory usage, and good generalization.

#### 2.3.1 Positional Embeddings

Due to the exceptional extrapolation capability, currently there are two popular position encoding methods leveraged by LLMs, i.e., the Rotary Position Embedding (RoPE) (Su et al., 2023), which generates position codes dynamically for the distance between each pair of elements by learning relative position information, and the Attention with Linear Biases Enables Input Length Extrapolation (ALiBi) (Press et al., 2022), which applies a preset offset matrix to the attention score based on the distance between tokens. We empirically find that RoPE shows better adaptation to the accelerate frameworks like Flashattention 2 (Dao, 2023) and xFormers (Lefaudeux et al., 2022). Thus, we opt for RoPE as the chosen positional encoding method.

#### 2.3.2 Attention Mechanism

The YAYI 2 models incorporate a distinctive MultiQuery Attention (MQA) (Shazeer, 2019) mechanism to implement Self-Attention, which involves sharing the WK and WV weight matrices among

heads and concatenating the results. MQA plays a pivotal role in significantly reducing the size of tensors and lowering memory bandwidth requirements for incremental decoding. To enhance the efficiency of calculating the attentions, we leverage the Flashattention 2 (Dao, 2023) framework during training to implement the MQA calculation.

#### 2.3.3 Activations and Normalizations

Our model incorporates SwiGLU (Shazeer, 2020) as the activation function due to its superior performance and faster convergence. In terms of the regularization method, we employ RMSNorm (Zhang and Sennrich, 2019), which only focuses on the rescaling invariance and performs regularization to the summed inputs simply based on the root mean square. Compared to the commonly used Layer Normalization (Ba et al., 2016), RMSNorm can approximately reduce computation time by 7%-64%.

2.4 Model Training

#### 2.4.1 Computing Cluster

YAYI 2 models are trained on a cluster comprising over 1000 A800 GPU servers. This cluster’s nodes are interconnected through an InfiniBand (IB) network, facilitating high-speed direct memory access and data transfer. GPUs within each node are linked through high-bandwidth, low-latency NVLink connections. To optimize cluster management of code, data, and model checkpoints, an SSD hard drive is implemented as the shared storage for the whole cluster using the Network File System (NFS). Addressing common challenges in large-scale cluster management, such as resource allocation, job scheduling, and scalability, we enhance the SLURM (Simple Linux Utility for Resource Management) system for resource management and job scheduling. Additionally, an anomaly alert module is also added to monitor the real-time running status of the cluster in case of hardware failures and unhandled program exceptions.

#### 2.4.2 Training Strategy

Distributed Training To keep a balance between GPU memory utilization and communication efficiency, the Zero Redundancy Optimizer (ZeRO) (Rajbhandari et al., 2020) stage 3 is applied, which works in conjunction with gradient checkpointing, significantly improving the GPU memory utilization. As expected, the average processing speed of GPUs reaches 600 tokens/s, with tensor core utilization rate of 65%, showcasing superior

[Figure 5]

Figure 5: The training loss of YAYI 2-30B.

performances in large-scale clusters (Touvron et al., 2023a).

Optimizer The AdamW (Loshchilov and Hutter, 2017) is used for training. Unlike Adam (Kingma and Ba, 2015), AdamW achieves higher computational efficiency, superior generalization, and faster convergence speed. For parameters of AdaW, the β1 and β2 are set be to 0.9 and 0.95, respectively. The weight decay is 0.1. The model training is warmed up with the learning rate from 5 × 10−5 to 1 × 10−4 for the first 2000 steps.

Figure 5 shows the final training loss of YAYI230B.

#### 2.4.3 Training Tricks

Data Pre-allocation Maintaining a stable data distribution is pivotal for improving model performances. Large jitter in data distribution can be harmful to model convergence. To precisely control the data distribution, we design a data preallocation mechanism based on file indices. This mechanism builds a global file index table and allocates data files to each GPU before training, guaranteeing consistent data distribution across training steps. According to whether the quantity of data is fixed, pre-training data can be divided into static data and dynamic data. The quantity of static data does not change with time, and mainly includes knowledge data such as books, ancient poetry, textbooks, academic papers, encyclopedia knowledge bases, etc. The quantity of static data is limited but of high quality, whereas dynamic data exhibits a vast quantity but with lower quality. The quantity of dynamic data continues to grow over time, mainly including current news data such as web pages, newspapers, social media, etc. To reduce model hallucinations, we upsample static data and downsample dynamic data by increasing and decreasing file indices, respectively.

Lazy Loading When loading binary model checkpoints, since each GPU in one node needs to pre-load the model weights from the node’s CPU memory into its GPU memory, the CPU memory may overflow under different configurations of computing clusters. By introducing a lazy loading strategy, i.e. allowing different GPUs to start the pre-loading process sequentially, we reduce the peak memory usage during the model loading phase and effectively avoid CPU memory overflow.

Training Restart With the expansion of the computing cluster, training tasks are prone to be interrupted due to various software and hardware issues. To minimize idle time of the training cluster and restart training from intermediate checkpoint, we optimize preventive measures for common problems such as GPU crash, disk space exhaustion, deadlocks, etc. Specifically, we perform automated interventions from three aspects: logging, exception alerts, and automatic restarts.

- • Logging. We maintain detailed logs of the current training task status, including model training outputs and data usage states.
- • Exception alerts. By monitoring GPU utilization and the update timestamp of log files, we establish an auto-alert mechanism via instant messaging. The types of malfunctions is also detected and notified.
- • Automatic restarts. Based on the type of malfunction, the training cluster adopts corresponding restart strategies. For example, when some GPU crashes, problematic nodes are removed, and standby nodes are incorporated into the training cluster before restarting the training process.

### 3 Alignment

The alignment process for YAYI2-30B consists of two crucial stages: Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).

3.1 Supervised Fine-Tuning 3.1.1 Instruction Dataset

Instruction data for YAYI encompasses manually annotated high-quality instruction data and opensource SFT datasets. We strictly review the instruction data in terms of format and content. For data

###### Task Type Description Weight

Text Generation Generate articles, outlines, schemes, etc. 30% Reading Comprehension Answer questions based on given context. 18% Open QA Knowledge, common sense, and other questions. 10% Creative Inspiration Write poetry, design, naming, script creation, etc. 10% Information Extraction Extract content from context, output in a specified format. 8% Chit-chat Role Play Daily consultations, chat, and role-playing. 5% Text Rewriting Change style, change language, etc. 5% Abstraction & Summarization Summarize titles or abstracts, etc. 4% Text Classification Text classification task. 3% Text Translation Multilingual translation tasks. 2% Code Capability Code generation, completion, commenting, etc. 2% Logical Reasoning Mathematical and reasoning tasks. 2% Other Tasks Tasks not classified into the above categories. 1%

Table 2: General tasks and descriptions.

format, we check for missing line breaks. For the content, we: (1) examine the completeness of content (i.e., truncated answers, incomplete information, and repeated generations); (2) ensure the consistency of language in instructions and answers, except for translation tasks; (3) confirm that the generated answers follow the given instructions; (4) ensure that the generated responses are free from hallucination; (5) verify that the answers comply with laws and regulations; (6) scrutinize the human values in the instruction-answer pairs.

For the data format, content completeness, and language consistency, a classification model is trained to evaluate the open-source instruction data and the auto-generated data. For the instruction compliance and the hallucination issue, we systematically inspect data in batches through manual verification. Data sources within each batch are consistent. A batch of data is dismissed if it displays poor compliance with instructions or has many hallucination issues. For safety concerns, see Section 5.2.

After filtering and review, we identify highquality data to ensure balanced sampling for training. To promise the data diversity for SFT, we sample data across different task types, language categories, context lengths, and data sources, where the distribution of general task types is outlined in Table 2.

Following OpenAI’s Chat Markup Language (ChatML) format, the SFT for YAYI adheres to a structured multi-turn conversation involving three roles: system, human, and Yayi. The system defines global role information and identity settings, the human represents the user, and YAYI symbolizes the large model. Identifiers for these roles are denoted as "<system>", "<human>", and "<yayi>"

for clarity.

- 3.1.2 Training Details Aligned with the pre-training stage, the YAYI 2 models employ a distributed training framework for SFT. The training utilizes BF16 floating-point precision to enhance efficiency and employs the

AdamW optimizer with β1 set as 0.9, β2 set as 0.96, and ϵ set as 1e-8. The learning rate warmup steps constitute approximately 10% of the total steps, gradually increasing to a peak learning rate of 5e-6. To prevent overfitting, the weight decay is set as 1e-3.

To accommodate various instruction lengths during training, including short, long, single-turn conversation, and multi-turn conversation instructions, we progressively adjust the context window from 2,048 to 4,096 and ultimately 8,192. The computing cluster is the same as Section 2.4.1.

- 3.1.3 Long Instructions To bolster the model’s capability in handling lengthy texts, a batch of long SFT data is built, encompassing both the long input type and long output type. The long input data includes tasks like long text summarization, reading comprehension, and other complex instructions. The long output data involves generating long articles, multi-level outlines, and research reports, etc.
- 3.1.4 Multi-Turn Conversation We build multi-turn conversation data from two dimensions:

• Context Relevance Dimension: including context-relevant and context-irrelevant multiturn conversation data. Context-relevant conversations involve human questions related to the previous content, while context-irrelevant

[Figure 6]

Figure 6: Dimensions of multi-turn conversation.

conversations comprise questions unrelated to the ongoing conversation.

• Role Information Dimension: including multiturn conversations for general tasks (without role information) and role-played multi-turn conversations.

In the course of instruction data generation, we applied a nuanced approach, segmenting the data into distinct role information dimensions. This segmentation encompassed multi-turn conversations tailored for general tasks and role-played multiturn conversations, strategically intertwined with contextual relevance for data synthesis. In the realm of general tasks, multi-turn conversations featured instances both relevant and irrelevant to the ongoing context. In contrast, role-played multiturn conversations, distinguished by their succinct and context-driven nature, exclusively factored in context-relevant scenarios. This conceptualization is succinctly depicted in Figure 6.

• Context-relevant multi-turn conversation data in general tasks: In this regard, we devise a meticulous algorithm for data generation. Commencing with a randomly sampled human question data-seeding from the instruction database, the model generates an initial

response. Subsequently, leveraging the extant context, we systematically formulate related questions and amalgamate the contextual content to prompt the model for generating successive rounds of responses. This iterative process results in the creation of context-relevant multi-turn conversation data tethered to the original data-seeding.

- • Context-irrelevant multi-turn conversation data in general tasks: In this regard, we independently draw random batches of task-typesimilar and task-type-irrelevant single-turn data. Through statistical scrutiny, it emerges that human queries in a single multi-turn conversation exhibit a propensity for thematic similarity. Guided by this observation, we sample and concatenate analogous task-type data or devised mixed-sample data, mirroring scenarios where humans pose multiple queries related to the same task (e.g., prompting the model to generate poetry repeatedly) or varied tasks within a single multi-turn conversation (e.g., initiating poetry generation followed by mathematical problem-solving).
- • Role-played multi-turn conversations: Prompt generation begins by randomly assigning roles to the YAYI model, encompassing task-centric

[Figure 7]

[Figure 8]

Figure 9: Computation of multi-turn conversation loss.

Figure 7: Multi-turn conversation data format.

struct a series of domain-specific data for SFT, aiming to hone the model’s prowess in navigating authentic business challenges.

[Figure 9]

3.2 Reinforcement Learning from Human Feedback

Despite the commendable performances of supervised fine-tuning across various tasks, the efficacy of the proposed model is contingent on the quality of annotated data and is susceptible to overfitting. To overcome these limitations and further elevate the YAYI models’ capacity for generalization, we turn to reinforcement learning from human feedback (Ouyang et al., 2022). This methodology aims to align the models’ generated content more closely with human preferences. Specifically, a reward model is trained to predict human preferences, and the Proximal Policy Optimization (PPO) (Schulman et al., 2017) algorithm is employed to reinforce the YAYI model, guiding it toward generating responses that better resonate with human expectations.

Figure 8: Roleplay data format.

roles (e.g., traditional Chinese medicine practitioner, lawyer, financial analyst) and specific character roles (e.g., Confucius, Sun Wukong, Zhang Fei). Based on the speaking style and personality traits inherent in these roles, we simulate multi-turn conversations involving human participants. Following rigorous quality assessments, it assumes the role of the model’s multi-turn conversation training dataset.

The format of the multi-turn conversation training data aligns with that of the single-turn instruction task training. It commences with globally defined role information, alternates between the user and the YAYI model, and concludes each YAYI model’s reply with an end-of-sequence token "</s>." The format is illustrated below.

#### 3.2.1 Reward Model

To collect high-quality and well-balanced instructions, a meticulous instruction sampling strategy is implemented. Initially, a semantic deduplication system is utilized for a vast instruction set. Subsequently, a two-tier task subdivision is employed with a dynamic weighted sampling strategy to maintain instructional equilibrium within each category.

During model training, we only compute the loss for the output of each turn in multi-turn conversations, as depicted in Figure 9. This strategic approach steers the model toward generating highquality conversation content, circumventing unnecessary calculations for irrelevant content. This targeted methodology significantly augments training efficiency.

Given a prompt, the YAYI 2 chat model generates two responses, employing distinct sampling strategies. Expert annotators evaluate these responses across four dimensions: format correctness, content completeness, content accuracy, and instruction compliance. These evaluations are employed for the continuous refinement of the reward model’s performance.

#### 3.1.5 Domain Tasks

The YAYI large model is meticulously tailored to real-world business scenarios, encapsulating a diverse spectrum of hundreds of tasks spanning finance, media, law, healthcare, and beyond. Through manual annotation and review, we con-

The reward model is trained starting with the YAYI chat model after SFT. Notably, for performance stability, a reward token is appended to each data point. The embedding features of this token

are then utilized to predict the ultimate reward. Throughout training, the reward model exhibits an escalating trend in discriminative accuracy as the quality gap between the two responses widens.

#### 3.2.2 Reinforcement Learning via PPO

The PPO algorithm is adopted for reinforcement learning, encompassing four models: the policy model (responsible for response generation, requiring optimization), the reference model (providing a fixed-parameter reference for the policy model), the reward model (assessing response quality with fixed parameters), and the value model (learning token values and requiring optimization). The value model undergoes a warm-up stage of 50 training steps during the training process. Both the value model and policy model are updated using the standard PPO algorithm. To maintain training stability, clipping and normalization techniques are also applied.

4 Inference

#### 4.1 Long-Context Reasoning

The YAYI 2 models have significantly enhanced their capacity for processing lengthy texts and multi-turn conversations by leveraging an extended context window. While mainstream proprietary models, like GPT-4-Turbo, have extended their context length to 128K, open-source LLMs, such as Llama, typically support a context length of 4K. In this technical report, we augment the YAYI 2 models’ ability to handle extensive contextual information by extending its extrapolation capabilities based on scalable features of the RoPE position embedding method.

Current research in RoPE extrapolation primarily explores two directions: methods based on sliding windows and methods based on adjusting rotation angles. Given the loss of global low-frequency information associated with sliding windows, recent studies concentrate more on adjusting the encoding rotation angle. The YAYI 2 models adopt the YaRN method (Peng et al., 2023) for RoPE extrapolation, integrating NTK with sliding window methods to mitigate the collapses in ultra-long contexts.

Figure 10 shows that YAYI2-30B with YaRN has significantly lower perplexity and is more stable, which demonstrates that the effectiveness of NTK with sliding window for extrapolation.

[Figure 10]

Figure 10: Perplexity of different configurations for extrapolation.

#### 4.2 Diverse Hardware Inference Adaptation

In addition to NVIDIA GPUs, the YAYI 2 models have been adapted for efficient inference on the Huawei Ascend 910B hardware. To address the challenges posed by the large parameter count for a 30B model during single-GPU inference, we adopt a distributed inference solution, which involves using multi-GPUs for inference. This process entails compiling the target strategy network to obtain the distributed strategy file. Thus based on the splitting strategy, the model weights are partitioned and loaded onto each GPU for the following inference procedure.

5 Safety

#### 5.1 Pre-training Stage

The pre-training data-preparation phase prioritizes and strictly adheres to data security protocols to ensure the integrity and compliance of the data. A comprehensive strategy is deployed, incorporating a robust data security filtering and classification system.

This system’s primary objective is to identify and exclude potentially harmful and security-related content, preventing the model from learning and reproducing inappropriate information. The specific categories of safety-related information include:

• Sensitive information. Confidential internal corporate data, such as undisclosed financial reports and research materials, is filtered out to prevent intellectual property infringement issues. Other sensitive information includes personal privacy data, including but not lim-

[Figure 11]

Figure 11: Results of similar sized LLMs on 10 benchmarks.

ited to personal identity information, contact details, bank accounts, and medical records.

- • Inappropriate content. Inappropriate content includes hate speech, violence, discrimination (e.g. ethnic and gender discrimination), extremist speech, pornography and other indecent content.
- • Content Violating Laws and Regulations. Copyrighted materials are removed to ensure that protected works under copyright are not used illegally or included in training data.
- • Misleading and Inaccurate Information. Misinformation includes fake news, rumors, and other potentially misleading content. Antiscientific, pseudoscientific, and inaccurate medical health information are also removed to curb the spread of misinformation.

These strategies are implemented during the data source selection and data processing steps. Initially, data source selection involves strict adherence to reputable channels to sidestep intellectual property disputes. In the preliminary screening of text data, a Deterministic Finite Automaton (DFA)-based sensitive word filtering mechanism is applied. For Chinese text, the segmentation system is expanded, incorporating a specialized segmentation library to enhance filtering accuracy.

Furthermore, an efficient text classification model is developed and trained to identify and eliminate text containing inappropriate content. The training set covers categories such as pornography, violence, discrimination, hate speech, per-

sonal safety infringement, and illegal content. To broaden the model’s recognition capabilities, sample translation between Chinese and English is conducted, diversifying training samples. Medical professional data is specifically included in training samples to prevent medical-related text misclassification. These above two critical steps significantly enhance the security of the training data and lay a robust foundation for subsequent model training.

#### 5.2 Fine-tuning Stage

A safety auditing algorithm combining regular matching with machine learning models is designed for safety classification in the fine-tuning stage. Among the SFT data, safety instruction data is categorized into positive guidance and refusalto-answer:

- • Positive guidance category: Questions containing statements misaligned with human values or contradicting objective facts require the model to correct inappropriate content and provide a positive guidance response in line with human values.
- • Refusal-to-answer category: Questions involving illegal issues or those contravening relevant policies and laws prompt the model to express apologies, inform users of inappropriate content, and refuse to provide an answer.

Joint training of safety-enhanced instruction data with general tasks and domain tasks is carried out to prevent catastrophic forgetting and enhance the model’s security.

C-Eval(val) MMLU AGIEval CMMLU GAOKAO-Bench

Model

5-shot 5-shot 3/0-shot 5-shot 0-shot

MPT-30B – 46.9 33.8 – – Falcon-40B – 55.4 37.0 – – LLaMA2-34B – 62.6 43.4 – – Baichuan2-13B 59.0 59.5 37.4 61.3 45.6 Qwen-14B 71.7 67.9 51.9 70.2 62.5 InternLM-20B 58.8 62.1 44.6 59.0 45.5 Aquila2-34B 98.5 76.0 43.8 78.5 37.8 Yi-34B 81.8 76.3 56.5 82.6 68.3 Qwen-72B 83.3 77.3 61.8 83.6 86.0

YAYI2-30B 80.9 80.5 62.0 84.0 64.4

- Table 3: Evaluation results on knowledge and language understanding. The best results are in bold and the second are underlined.

### 6 Evaluations

#### 6.1 Baseline Models

In this section, we evaluate the YAYI 2 base model’s performance against a series of opensource models with similar parameter sizes on standard benchmark datasets. The evaluation dimensions encompass knowledge and language understanding, mathematical reasoning, and programming ability. Comparative base models include MPT-30B (MosaicML et al., 2023), Falcon-40B (Almazrouei et al., 2023), LLaMA 2-34B (Touvron et al., 2023b), Baichuan 2-13B (Yang et al., 2023), Qwen-14B&72B (Bai et al., 2023), InternLM-20B (InternLM, 2023), Aquila 2-34B (BAAI, 2023) and Yi-34B (01-AI, 2023).

#### 6.2 Evaluation Results

We use accuracy as the primary metric and, if available, report the results of comparative models evaluated by OpenCompass (OpenCompass, 2023), taken from the leaderboard of the OpenCompass official website1. The reported results of YAYI230B model are also evaluated by the source code at the OpenCompass Github repo. For the models that have not been evaluated by the OpenCompass, including MPT-30B, Falcon-40B and LLaMA 234B, we use the results reported by Touvron et al. (2023b). Note that on some benchmarks there can be some slight differences between the evaluations conducted by OpenCompass and Touvron et al. (2023b). See Figure 11 for the overall comparison

1https://opencompass.org.cn/leaderboard-llm, evaluation results reference date: Dec. 15, 2023.

with three similar sized LLMs, including InternLM20B, Aquila2-34B and Yi-34B.

6.2.1 Knowledge and Language Understanding

The evaluations regarding knowledge cover various benchmarks, including MMLU (Hendrycks et al., 2021a), C-Eval validation set (Huang et al., 2023), CMMLU (Li et al., 2023), AGIEval (Zhong et al., 2023) and GAOKAO-Bench (Zhang et al., 2023).

- • MMLU: English interdisciplinary knowledge evaluation benchmark, covering multiple choice questions from 57 disciplines in STEM, humanities, social sciences, and other fields.
- • C-Eval: Chinese comprehensive exam evaluation benchmark, consisting of 13,948 multiple choice questions, with four different levels of difficulty, covering knowledge across 52 disciplines.
- • AGIEval: Benchmark for knowledge reasoning ability in both Chinese and English, including questions in various fields such as SAT, college entrance examination, and judicial exam.
- • CMMLU: Chinese benchmark assessing knowledge reasoning ability, including 67 single-choice questions across various themes in natural science, humanities and social sciences, and everyday life.
- • GAOKAO-Bench: Chinese benchmark for knowledge reasoning ability, including major

GSM8K MATH BBH 8/4-shot 4-shot 3-shot

Model

MPT-30B 15.2 3.1 38.0 Falcon-40B 19.6 5.5 37.1 LLaMA2-34B 42.2 6.2 44.1 Baichuan2-13B 52.6 10.1 49.0 Qwen-14B 61.6 25.2 53.7 InternLM-20B 52.6 7.9 52.5 Aquila2-34B 50.0 17.8 42.5 Yi-34B 67.9 15.9 66.4 Qwen-72B 77.6 35.1 63.7

YAYI2-30B 71.2 14.8 54.5

- Table 4: Evaluation results on mathematical reasoning.

HumanEval MBPP 0-shot 3-shot

Model

MPT-30B 25.0 32.8 Falcon-40B 0.6 29.8 LLaMA2-34B 22.6 33.0 Baichuan2-13B 17.1 30.8 Qwen-14B 32.3 39.8 InternLM-20B 25.6 35.6 Aquila2-34B 0.0 41.0 Yi-34B 26.2 38.2 Qwen-72B 33.5 51.6

YAYI2-30B 53.1 45.8

Table 5: Evaluation results on programming.

questions from national college entrance exams from 2010 to 2022, from which objective questions are selected to evaluate the model.

We report the 3-shot (for MPT-30B, Falcon-40B and LLaMA 2-34B) or zero-shot (for other models) evaluation results on AGIEval and GAOKAOBench, and 5-shot results on MMLU, C-Eval and CMMLU. Table 3 shows the detailed results of our proposed model in the comparative experiments on these benchmarks. Our model outperforms other models on MMLU, AGIEval and CMMLU benchmarks, even surpassing the Qwen-72B with a much larger parameter size.

#### 6.2.2 Math and Logic Reasoning

In the domain of mathematics and reasoning, our model is evaluated on three prominent benchmarks: GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b) and BBH (Suzgun et al., 2022). We use accuracy as the principal evaluation metric.

- • GSM8K: A benchmark dataset designed for mathematical reasoning, encompassing 1,319 elementary math word questions.
- • MATH: Comprising 5,000 challenging mathematical questions spanning diverse domains such as linear algebra, geometry, and probability.
- • BBH: A subset of the BIG-Bench dataset, featuring 23 highly challenging tasks encompassing logic reasoning, common-sense understanding, and mathematics. Its objective is to challenge the model with more intricate reasoning and language-understanding tasks.

We report the 8-shot (for MPT-30B, Falcon-40B and LLaMA 2-34B) or 4-shot (for other models) evaluation results on GSM8K, 4-shot results on MATH, and 3-shot results on BBH. Upon examination of Table 4, the YAYI 2 base model has achieved the best performance on the GSM8K benchmark among models with comparable parameter sizes.

#### 6.2.3 Programming

In the evaluation of programming capabilities, the evaluation benchmarks include HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021)

- • HumanEval: A dataset comprising 164 programming questions, each featuring a function signature, documentation string, subject code, and an average of 7.7 unit tests. Covering aspects of language understanding, reasoning, algorithms, and basic mathematics, it serves as a comprehensive assessment of the model’s proficiency in code generation.
- • MBPP: A coding benchmark consisting of 500 beginner-level Python programming questions.

The primary evaluation metric is pass@1, indicating the model’s success rate in generating the correct code on the first attempt.

Following the evaluation method of OpenCompass, we report the zero-shot results on HumanEval and 3-shot results on MBPP. Table 5 demonstrates our model’s standing as the pinnacle performer among models with comparable parameter sizes, and even significant superiority over the much larger Qwen-72B on the HumanEval benchmark.

In summary, our model showcases remarkable competence across knowledge understanding, mathematical reasoning, and programming benchmarks, validating the effectiveness of our model.

### 7 Conclusions

In this technical report, we propose the multilingual YAYI2-30B LLMs with a specific focus on Chinese-related applications. We introduce the distributions of the pre-training dataset, as well as the preprocessing pipeline. The YAYI2-30B models follow the popular decoder-only model architecture, and adopt FlashAttention 2 and MQA to speed up training and inference. We also reveal the pre-training details, including computing clusters, training strategies and tricks, which we believe will greatly benefit the industry practitioners. We further show how to build the instruction dataset for instruction tuning, and the YAYI 2 models’ support for long instructions, multi-turn conversations and domain-specific applications. The RLHF process is further applied to better align with human values and ensure safety. The YAYI 2 base model is evaluated on three types of benchmarks, including knowledge and Language understanding, math and logic reasoning, and programming. Extensive experimental results show that the proposed model achieves superior performances over similar-sized open-source LLMs on multiple benchmarks, including MMLU, AGIEval, CMMLU, GSM8K, HumanEval and MBPP. Especially on the MMLU, AGIEval, CMMLU and HumanEval benchmarks, our model can even outperform the larger-sized Qwen-72B with considerable margins.

Although we have adopted various methods to ensure safety and reduce hallucinations, the YAYI 2 models still can produce harmful content or fabricate "facts", so the model users are highly encouraged to review the answers, especially in safetycritical situations. The model users are also advised to prevent the misuse of the YAYI 2 models and abide by related laws and regulations. The YAYI 2 models are still under active development, and all suggestions and feedback are welcomed.

### References

01-AI. 2023. Yi: A series of large language models trained from scratch by developers at 01-ai. https: //github.com/01-ai/Yi.

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojo-

caru, Merouane Debbah, Etienne Goffinet, Daniel Heslow, Julien Launay, Quentin Malartic, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023. Falcon-40B: An open large language model with state-of-the-art performance. https:

//huggingface.co/tiiuae/falcon-40b.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer normalization.

BAAI. 2023. Aquila2 series proposed by BAAI. https: //github.com/FlagAI-Open/Aquila2.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional AI: Harmlessness from AI feedback.

Baichuan. 2023. A large-scale 7B pretraining language model developed by baichuan Inc. https://github. com/baichuan-inc/Baichuan-7B.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems.

Together Computer. 2023. RedPajama: An open dataset for training large language models. https://github.com/togethercomputer/ RedPajama-Data.

Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for Chinese LLaMA and Alpaca.

Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the MATH dataset. In Conference on Neural Information Processing Systems Track on Datasets and Benchmarks.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, et al. 2023. C-Eval: A multi-level multi-discipline chinese evaluation suite for foundation models.

InternLM. 2023. InternLM: A multilingual language model with progressively enhanced capabilities. https://github.com/InternLM/InternLM.

Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. 2023. Challenges and applications of large language models.

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71.

Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. 2022. xformers: A modular and hackable transformer modelling library. https: //github.com/facebookresearch/xformers.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. 2023. CMMLU: Measuring massive multitask language understanding in chinese.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. In International Conference on Learning Representations.

MosaicML et al. 2023. MPT-30B: Raising the bar for open-source foundation models. https://www. mosaicml.com/blog/mpt-30b.

OpenCompass. 2023. OpenCompass: A universal evaluation platform for foundation models. https: //github.com/open-compass/opencompass.

Pedro Javier Ortiz Suárez, Benoît Sagot, and Laurent Romary. 2019. Asynchronous pipelines for processing huge corpora on medium to low resource infrastructures. In Workshop on the Challenges in the Management of Large Corpora.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The RefinedWeb dataset for Falcon LLM: Outperforming curated corpora with web data, and web data only.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. Yarn: Efficient context window extension of large language models.

Ofir Press, Noah Smith, and Mike Lewis. 2022. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1– 16. IEEE.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms.

- Noam Shazeer. 2019. Fast transformer decoding: One write-head is all you need.
- Noam Shazeer. 2020. Glu variants improve transformer.

Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, and Eric Xing. 2023. SlimPajama-DC: Understanding data combinations for LLM training.

Yusuke Shibatay, Takuya Kiday, Shuichi Fukamachiz, Masayuki Takeday, Ayumi Shinoharay, Takeshi Shinoharaz, and Setsuo Arikaway. 1999. Byte pair encoding: A text compression scheme that accelerates

pattern matching. Technical Report DOI-TR-161, Department of Informatics, Kyushu University.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2023. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, page 127063.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. LLaMA: Open and efficient foundation language models.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. LLaMA 2: Open foundation and fine-tuned chat models.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems.

BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al. 2022. Bloom: A 176Bparameter open-access multilingual language model.

XVERSE. 2023. XVERSE-13B: A multilingual large language model developed by XVERSE Technology Inc. https://github.com/xverse-ai/ XVERSE-13B.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, et al. 2023. Baichuan 2: Open largescale language models.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2023. GLM-130B: An open bilingual pre-trained model. In International Conference on Learning Representations.

Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. In Advances in Neural Information Processing Systems.

Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. 2023. Evaluating the performance of large language models on GAOKAO benchmark.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. AGIEval: A human-centric benchmark for evaluating foundation models.

