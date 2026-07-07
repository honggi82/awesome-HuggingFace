# arXiv:2408.15079v1[cs.CL]27Aug2024

## BaichuanSEED: Sharing the Potential of ExtensivE Data Collection and Deduplication by Introducing a Competitive Large Language Model Baseline

Guosheng Dong1* Da Pan1 Yiding Sun1,2† Shusen Zhang1 Zheng Liang1 Xin Wu1 Yanjun Shen1 Fan Yang1 Haoze Sun1 Tianpeng Li1 Mingan Lin1 Jianhua Xu1 Yufan Zhang1 Xiaonan Nie1 Lei Su1 Bingning Wang1 Wentao Zhang3 Jiaxin Mao2 Zenan Zhou1* Weipeng Chen1 1Baichuan Inc. 2Gaoling School of Artificial Intelligence, Renmin University of China 3Peking University

Correspondence: {dongguosheng, zhouzenan}@baichuan-inc.com

### Abstract

The general capabilities of Large Language Models (LLM) highly rely on the composition and selection on extensive pretraining datasets, treated as commercial secrets by several institutions. To mitigate this issue, we open-source the details of a universally applicable data processing pipeline and validate its effectiveness and potential by introducing a competitive LLM baseline. Specifically, the data processing pipeline consists of broad collection to scale up and reweighting to improve quality. We then pretrain a 7B model BaichuanSEED with 3T tokens processed by our pipeline without any deliberate downstream task-related optimization, followed by an easy but effective supervised fine-tuning stage. BaichuanSEED demonstrates consistency and predictability throughout training and achieves comparable performance on comprehensive benchmarks with several commercial advanced large language models, such as Qwen1.5 and Llama3. We also conduct several heuristic experiments to discuss the potential for further optimization on downstream tasks, such as mathematics and coding.

### 1 Introduction

Large Language Models (LLMs) represented by ChatGPT and GPT-4 (OpenAI, 2023) have demonstrated exceptional performance across various domains, benefiting from the general capabilities through extensive pre-training on diverse datasets. Pre-training a foundation model is costly, typically requiring training on terabytes of tokens over millions of GPU hours, making the process expensive and difficult to reproduce (Brown et al., 2020; Chowdhery et al., 2023; Zhu et al., 2024; Yang et al., 2023; Bai et al., 2023; Bi et al., 2024; Dubey

† This work is finished during internship in Baichuan Inc.

* Corresponding Authors. https://baichuanseed.github.io/

et al., 2024). Many existing works focus on data selection, selecting high-quality data from massive datasets, to decrease the loss, thereby reducing computational costs and even enhancing downstream task performance (Zhao et al., 2023; Albalak et al., 2024). These methods mainly include heuristic rule-based selection (Sun et al., 2024; Chen et al., 2024a), model-based selection (Xie et al., 2023; Engstrom et al., 2024), dependent or relevant document detection and concatenation (Chen et al., 2023; Shi et al., 2024).

Although data selection shows considerable potential, the high computational cost makes it challenging to scale up. Additionally, the preliminary steps of data selection, data collection and reweighting are often treated as commercial secrets by companies (Touvron et al., 2023b; Bai et al., 2023), and even many open-source models that have released checkpoints do not disclose the details, hindering research advancement in this area. Additionally, many commercial models even over-optimize to excel in specific benchmarks (Zhou et al., 2023; Xu et al., 2024c; Deng et al., 2024), thus masking their true capabilities. Therefore, we believe that training a transparent LLM without any specific optimization, and publicly sharing all its details will be beneficial for the community to understand the potential of data. To emphasize, BaichuanSEED is not a SOTA model. What really matters to us is that understanding of a pure model helps in recognizing its actual strengths and weaknesses, the preliminary for evaluating the real impact of various optimization strategies.

In this technical report, we first introduce a 7B foundation model, BaichuanSEED, with the similar model architecture of Baichuan2 (Yang et al., 2023) on 3T bilingual tokens from scratch and then supervised fine-tune it to obtain BaichuanSEEDSFT, a chat model. Following Falcon (Penedo et al., 2023), the pre-training data processing procedure

was defined to involve extensive collection to scale up, followed by reweighting data points to set the sampling probability during pre-training. Specifically, we collect high-quality data widely available on the internet while intentionally excluding synthetic and downstream benchmarks to ensure model purity. Subsequently, we design a global multi-granularity deduplication algorithm to adjust the sampling weight of each data point during training. Here, we avoid fine-grained data selection intentionally, not to undermine its benefits but to focus more on exploring the achievable limit through data collection and reweighting. We then conduct a straightforward yet effective fine-tuning process to endow the model with instruction-following capabilities.

In evaluation, BaichuanSEED exhibits consistency and predictability, indicating the robustness of its training process. Consistency is reflected in the uniform trends observed across pre-training and fine-tuning benchmarks as the pre-training. Predictability refers to the ability to forecast the model’s future performance based on early checkpoints. We also evaluate our model against a series of LLMs of similar scale on comprehensive benchmarks and several downstream tasks. The experimental results indicate that without excessive optimization, our model can already achieve performance comparable to advanced commercial models like Llama3 and Qwen-1.5 on several comprehensive benchmarks, while our model still has room for improvement on some downstream tasks, especially on math. Finally, we explore the extensive potential of our model by utilizing several common optimization methods, such as adjusting the ratio of high-knowledge-density data and optimizing for mathematical and coding abilities. We leave integrating these optimization methods into our model BaichuanSEED to construct a highly robust LLM for future work.

Our main contributions are two-fold: (1) We propose a data processing pipeline, including broad collection to scale up and reweighting to deduplicate and improve the data quality. (2) We train a competitive 7B LLM baseline from scratch with 3T data processed by the aforementioned pipeline, followed by a simple yet effective supervised finetuning. Our model is consistent and predictable, and achieves comparable performance on comprehensive benchmarks with cutting-edge commercial LLMs without any deliberate optimization.

### 2 Model Architecture

BaichuanSEED first pre-train from scratch, followed by a Supervised Fine-Tuning (SFT) stage for alignment. Our model follows a Transformer decoder stack architecture similar to our previous version model Baichuan2 (Yang et al., 2023) and Llama (Touvron et al., 2023a,b). Specifically, our model comprises 32 layers with 32 attention heads. The hidden dimension size is 4,096, and the feedforward layer size is 11,008. SwiGLU (Shazeer, 2020) is used as the activation function, while RMSNorm (Zhang and Sennrich, 2019) is employed to enhance training stability. Rotary Positional Embedding (RoPE) (Su et al., 2024) is employed to model relative position dependencies.

### 3 Pre-training

In this section, we first provide a detailed overview of our efforts on pre-training data. We follow the pipeline of first broad collection from trusted sources to scale up, then re-weighting the data to obtain diverse and high-quality pre-training data. We then introduce the details of the training setups.

#### 3.1 Pre-training Data

The principle for constructing our pre-training data encompasses two aspects: diversity and highquality. In terms of diversity, we argue that pretraining data should cover a wide range of topics, linguistic styles and formats to ensure the model can adapt to diverse application scenarios, helping it learn comprehensive world knowledge and linguistic patterns. Regarding high-quality, we base on the guideline that documents with high production costs typically have higher quality, which may generally curated after a rigorous process of human inspection and correction. Additionally, we address that a healthy data type distribution across the entire dataset should be ensured, while reducing information redundancy and the proportion of low-quality data, to increase the knowledge density.

To achieve the goal of diversity and high quality, our approach focuses on both scaling up and reweighting the documents. Specifically, we first collect all accessible data from trusted sources to scale up via our self-constructed pipeline. Then, we employ a global multi-granularity data deduplication strategy to re-weight the documents, enhancing the data quality while filtering out personal identifiable information (PII) and low-quality data. Following

STEM

Humanities

Web Pages

Books

Papers

Social

Others

30.9% 1.7%

19.1% 0.9%

18.6%

8.6%

36.5%

30.0%

35.6%

31.8%

14.9%

71.4%

Figure 1: The detail data proportion of subjects ranging from STEM, mathematics, social science, and others, with respect to the web pages, books, and papers.

the aforementioned principles, we obtain over 10T tokens of pre-training data consequently.

#### 3.1.1 Collection

Our collected data mostly originate from the public internet, including web pages, knowledge intensive data, code, and vertical domain data. To continuously collect the latest data for model iterations, a scraping and extracting pipeline is constructed to fetch up-to-date data automatically, enhancing the generalizability of our model in practice. In this section, we will introduce the details of data collection and extraction.

Web Pages. We collect 94 publicly available batches of CommonCrawl spanning the past decade. We construct our WARC extraction and processing pipeline, considering the effectiveness and cost of the WET and WARC formats. Trafilatura (Barbaresi, 2021), a Python package and command-line tool designed to gather text from the web, is opted for further processing, in terms of the performance and cost among open-source web page parsing tools. CLD3 is utilized for document language identification.

Knowledge Intensive Data. Apart from web pages, we further focus on Knowledge Intensive Data (KID), including books, academic papers, and technical reports. KID originates from diverse domains, and possesses a more balanced distribution compared to web pages. As illustrated in Figure 1, most web pages are in the domain of social science, with STEM comprising only 30%, whereas STEM in books and papers accounts for 35% and 72%, respectively. However, processing KID is costly, due to its typical PDF format and the high requirements for document parsing. Consequently, we employ a hybrid approach, utilizing both the open-source tool Nougat (Blecher et al., 2023) and business ser-

https://commoncrawl.org/ https://github.com/google/cld3

vices, to maintain the original reading order and ensure the recognition and extraction accuracy of text, tables, and formulas. Tables and formulas are represented in LaTeX. We further discuss the effect of KID in Section 6.1.

Code. Our code data consist of StarCoder (Li et al., 2023b), GitHub filtered by license permissions, and StackExchange, in an interleaving form of code fragments and discussion texts. Both document-level and repository-level organization methods are adopted to ensure data diversity. For document-level, we apply the Minhash deduplication (Broder, 1997) and then use heuristic rules to select high-quality code. For repository-level, following Deepseek-coder (Guo et al., 2024), we calculate the dependencies among files within the same repository, perform topological sorting and concatenate the files. One repository is treated as a single sample and perform Minhash deduplication to preserve repository integrity. Repository-level data also undergoes heuristic rule-based pipeline to remove low-quality code and the majority of HTML, XML, JSON, and YAML files. We also prioritize sampling Github repositories with top stars. Repository-level data constitutes about 62% of the total 750B code mentioned in Table 2. Pivot experiments demonstrate that the mixture of repositorylevel and document-level data yields better model performance compared to using either one alone.

#### 3.1.2 Reweighting

Reweighting is a crucial stage in our pre-training data processing strategy, affecting the sampling proportion of each data point. We first introduce the deduplication operator, deciding whether to downsample a data point to zero. Then a reasonable data mixture is confirmed, to balance the distribution of data from each domain. The details of deduplication and data mixture are introduced in the following parts.

Deduplication. Deduplication is the first step of reweighting. Taking web pages as an example, we empirically found data with higher frequency is more likely tend to lead to a negative impact on foundation models. Therefore, we propose a global multi-granularity deduplication strategy, which will reduce the amount of raw data drastically. Via this strategy, we filter 88% of the tokens of our CommonCrawl (CC) dataset. We present the details of our global multi-granularity deduplication strategy as follows, including document-level deduplication, sentence-level deduplication across documents, PII

and harmful content filtering.

- • Document-level Deduplication Globally. We take both the frequency and quality into account. Specifically, existing works split CC into several batches, deduplicate in each batch, and finally merge. While we perform document-level deduplication globally, minimizing the effect of data points with extremely high frequency. We perform exact match deduplication by utilizing the MD5 value as the key of each document. Then MinHash method is adopted for similar neighbourhood document deduplication. We keep the document with the longest text length for a cluster of similar documents.
- • Sentence-level Deduplication across Documents. Similar to document-level, we empirically remove sentences with extremely high frequency across documents, which mostly are meaningless junk data. For example, it may be website template breadcrumbs, standard article openings and closings, and quoted passages. These frequently recurring fixed patterns may harm the model empirically. To eliminate biases in the raw data while ensuring diversity, we first split the documents into sentences, and then utilize Minhash with searched optimal hyperparameters at the sentence level across documents, making it more suitable for deduplication of short neighboring sentences. Several cases after deduplication in sentencelevel can be found in Appendix B.
- • PII and Harmful Content Filtering. PII and harmful content recognition and filtering is essential for reducing the model’s toxicity. Recent works employ heuristic rule-based filtering strategies. However, these methods require rule curation manually and suffer from poor performance on scalability, precision, and recall, making them unsuitable for building robust, and automated data processing pipelines. We adopt a combined approach of heuristic rules and model-based methods. Specifically, the former primarily includes fixed site templates, URL blacklists, regular expressions, word count and repeated n-gram character ratios, while the latter follows a iterative distillation method. We first obtain a set of harmful data by GPT-4 and human annotation as seeds to train a FastText (Joulin

w/o dedup

w/ doc-level dedup

2.70

w/ mult-level dedup

2.65

2.60

TrainingLoss

2.55

2.50

2.45

10 11 12 13 14 15 16 17 18 19

Training Steps (M)

Figure 2: The training loss with respect to different deduplication strategies on 2B models.

et al., 2017) model for classification, marking the PII, advertisement, copyright statements, and meaningless special characters as negative samples. Hard samples are mined to iterative improve the previous model and results in a generalized harmful content classification model ultimately. The probability that data points classified as negative is taken as a proxy of quality scores for PII an harmful content filtering. Additionally, data with low language scores is filtered, mostly meaningless texts.

Table 1: Loss of different deduplication strategies for 2B foundation models on held-out set.

Loss (ZH) Loss (EN)

w/o dedup 3.136 2.893 w/ document-level dedup 3.083 2.877 w/ multi-level dedup 3.006 2.874

We conduct ablation studies with different deduplication strategies by continued pre-training several 2B models with the same architecture as BaichuanSEED. The result can be found in Table 1. With the same training tokens, our global multi-granularity deduplication strategy achieves the lowest loss compared with the strategy “without deduplication” on both Chinese and English test set by -4.15% and -0.66% respectively, improving the model compression ability significantly (Delétang et al., 2023). Notably, our global multi-granularity deduplication strategy increases the training loss, as shown in Figure 2. We argue that our deduplication strategy leads to better diversity, making the prediction of the foundation model harder in the pre-training stage.

Data Mixture. We conduct several empirical experiments to determine the data mixture of each

Table 2: The data mixture of our pre-training data.

Domain #Tokens #Token/Doc Weight

Web / Chinese 405B 361 13.5% Web / Engligh 945B 752 31.5% KID / Chinese 300B 10,362 10% KID / English 450B 16,172 15% Code 750B 3,157 25% Others 150B 409 5%

Pre-training Data 3000B 4,558 100%

domain, with some details that can be found in Section 6.1. We then train our BaichuanSEED with the fixed data mixture from scratch as shown in Table 2.

#### 3.1.3 Other Principles

Synthetic Data. Synthetic data is widely used in SFT to enhance the difficulty, complexity, and diversity of constructed instructions (Wang et al., 2023; Xu et al., 2024a). However, some studies demonstrate that while synthetic data can improve model performance on coding and mathematics benchmarks, it may lead to a loss of instruction following ability after SFT (Chen et al., 2024b), making the impact of synthetic data unclear and introducing additional confounding variables. Our technical report focuses on training a completely pure baseline to explore the limits of model capability influenced by data collection and deduplication. Therefore, we did not incorporate synthetic data when training BaichuanSEED. We plan to explore the impact of synthetic data in future work.

Fine-grained Curation of Data. We directly prompt GPT-4 to obtain a set of knowledge category labels of training data and train a model to classify the full dataset, enhancing our fine-grained understanding. These labels are used to further explore the potential on mathematics of our model, as discussed in Section 6.2. They can also be applied to more detailed curriculum learning exploration. Additionally, we make effort to ensure that the evaluation datasets are excluded from the training data of BaichuanSEED.

#### 3.2 Training Details

We train on 3T tokens with a sequence length of 16K and a global batch size of 224. A cosine learning rate scheduler is used, with an initial learning rate of 3e−4 and a final learning rate of 3e−5.

### 4 Supervised Fine-tuning

In this section, we focus on the details of the stage of SFT, including our meticulously designed instruction construction and our training settings.

- 4.1 SFT Data

During the stage of SFT, we utilize a selfconstructed dataset for training. This comprehensive dataset includes mathematics, logical reasoning, coding, creative writing, brainstorming, and multi-turn dialogues, with approximately 450K samples. Based on prior research (Bai et al., 2024; Touvron et al., 2023b), we make optimizations to the SFT data, primarily including clustering massive data to enhance the diversity, synthesizing the instructions to increase the complexity. Additionally, we employ human annotations for part of data to improve the quality.

- 4.2 Training Details

We use a constant learning rate schedule with an initial learning rate of 2e−5, a batch size of 40, and a sequence length of 16K tokens. Each sample includes three parts: system, which is optimal, prompt, and answer. To enhance training efficiency and ensure sequences were filled, we employ sample packing for data concatenation and utilize Flash Attention 2 (Dao, 2023) to accelerate the training.

Ultimately, we selected a 6-epoch checkpoint for validation. This is more than the usual number of epochs for SFT training, comparing to Llama2, which used only 2 epochs. Empirically, more epochs yields better results for small size models, especially when the amount of pre-training token is insufficient. This might not be the optimal solution, as the ideal number of epochs can vary with different checkpoints. Nevertheless, this does not affect our evaluation and comparison of the alignment performance of the foundation models. We discuss the SFT evaluation results in Section 5.1.

### 5 Evaluation

In this section, we first explore the scaling law of BaichuanSEED’s downstream task performance with respect to the amount of training tokens. We elucidate the consistency and predictability during the training of BaichuanSEED. Furthermore, we evaluate our model and a series of 7B LLM baselines on several comprehensive benchmarks. We also assess the model’s generalization ability across a series of carefully selected down-

2.00

MMLU

MBPP

SFT-FB5-HSR SFT-FB5-SSR

SFT-SCM6 Loss (Val.)

70

CMMLU

GSM8K

1.95

60

1.90

50

Performance

Loss

1.85

40

1.80

30

1.75

20

10

1.70

0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 Training Token Amount (T Tokens)

Figure 3: The performance scaling curve of BaichuanSEED with respect to the amount of training tokens.

stream task benchmarks in coding, mathematics, common-sense reasoning, and reading comprehension, demonstrating the robustness of our model.

#### 5.1 Scaling Curves

Since we emphasize that appropriate data collection and deduplication can lead to comparable performance, which requires significantly less manual effort and computational cost, compared to meticulous data selection during the pre-training phase. However, the absence of data selection may result in inevitable noise. The process might raise concerns about whether training on a larger yet potentially lower-quality dataset can truly yield benefits or even have a negative impact such as drastic training fluctuation, and worse downstream performance. Consequently, we demonstrate the robustness of BaichuanSEED from two dimensions: Consistency and Predictability.

Consistency. We define consistency as the consistent trend of model capability growth with the number of training tokens before and after SFT. Specifically, model capability is reflected by metrics including the loss on test sets, comprehensive benchmark performance of base models, and the SFT benchmark performance of SFT models. We illustrate trends of the metrics with respect to the amount of training tokens during pre-training, as

- shown in Figure 3. The comprehensive benchmarks consist of MMLU (Hendrycks et al., 2021a), CMMLU (Li et al., 2023a), MBPP (Austin et al., 2021), GSM8K (Cobbe et al., 2021). While the SFT benchmarks encompass FollowBench (Jiang

- et al., 2023) and SuperCLUE-Math6 (Xu et al., 2024b). The SFT benchmark results are evaluated on different checkpoints of the foundation model

60

Train

Val.

58

Fit Curve

56

MMLUPerformance(Acc.)

54

52

50

48

46

44

0.5 1.0 1.5 2.0 2.5 3.0 Training Token Amount (T tokens)

Figure 4: The curve fit by the first half performance of BaichuanSEED with respect to the amount of training tokens on MMLU.

after the same SFT process. BaichuanSEED exhibits remarkable consistency across all evaluation benchmarks, indicating the model’s generalization capability. It indicates that BaichuanSEED has not been specialized to optimize for a specific task or evaluation benchmark, retaining the potential to transfer to specific downstream tasks.

Notably, some models may incorporate excessive synthetic data during the pre-training phase, creating a bubble impression of strong generalization capabilities and hindering the potential to improve during the SFT phase, even affecting the instruction-following capability (Chen et al., 2024b). We argue that pre-training and SFT are both indispensable, and excessively exploiting SFT may hinder the community from being aware of the foundational models’ capabilities. In contrast, our model exhibits consistent gains across both pre-training and SFT evaluation benchmarks, validating the decision to minimize using synthetic data during the pre-training phase. Additionally, the consistently increasing trend on several benchmarks validates the stability of our training strategy. Stability is highly essential in LLM development, where many approaches often sacrifice training accuracy to ensure training stability (Zhang et al., 2022; Zeng et al., 2023).

Predictability. We define predictability as the ability to forecast the model’s capabilities at later checkpoints based on its performance during the early stages of training. Predictability is especially crucial for developers, as having a forward-looking understanding of the model’s performance trends during the early stages of training can facilitate rapid iteration and minimize unnecessary resource

costs. Taking BaichuanSEED’s performance on MMLU as an example, we employ a logarithmic curve to fit the performance on MMLU with respect to the amount of training tokens, allowing us to predict the potential of our model. Specifically, we take points from the first half for extrapolation and fit for the points in the latter half perfectly, demonstrating the predictability ability of our model, as

- shown in Figure 4.
- 5.2 Comprehensive Benchmarks

We present the performance of BaichuanSEED and a series of 7B models as baselines across a comprehensive set of evaluation benchmarks.

Baselines. The selected baselines are all cuttingedge open-sourced LLMs, including the base and chat or instruct version of Llama family (AI@Meta, 2024), Baichuan series (Yang et al., 2023), Qwen series (Bai et al., 2023), MAP-Neo (Zhang et al., 2024), and OLMo (Groeneveld et al., 2024).

Benchmarks. The benchmarks encompass MMLU, CMMLU, AGIEval (Zhong et al., 2023), C-Eval, MMLU-Pro (Wang et al., 2024), and LiveBench (White et al., 2024). We also carefully select representative benchmarks for downstream task performance evaluation, including MBPP, HumanEval (Chen et al., 2021) for coding, GSM8K, MATH (Hendrycks et al., 2021b) for mathematics, TriviaQA (Joshi et al., 2017) for commonsense reasoning, and HellaSwag (Zellers et al., 2019) for reading comprehension. These benchmarks typically cover a wide range of subjects, prompting the LLMs to directly answer questions, which may be in the form of single-choice or short-answer. We take accuracy as the evaluation metric.

Overall Performance. The overall performance is shown in Table 3, while the details across various subjects can be found in Appendix A. Compared to the Llama family models, BaichuanSEED shows significant gains in Chinese evaluation benchmarks, such as CMMLU and C-Eval. We suppose that the incorporation of knowledge intensive Chinese data, such as high-quality textbooks and academic papers, leading to the improvement. Moreover, our foundation model surpasses Baichuan2-7B and Baichuan2-13B in comprehensive English benchmarks such as MMLU (+10.2%, +0.7%), AGIEval (+7.3%, +29.1%), and MMLUPro (+22.7%, -0.1%), and is comparable to Qwen1.5-7B and MAP-Neo. Although our model still lags behind Llama3-8B, we argue that training with only one-fifth of Llama’s tokens may

contribute to the result, which suggests the potential for comparable or even superior performance at the same scale. LiveBench, a benchmark with high timeliness, effectively assesses the model’s generalization ability, preventing potential benchmark leakage. Our model also outperforms Baichuan2-7B-Chat (+42.1%), Qwen1.57B-Chat (+9.2%), OLMo-7B-SFT (+108.2%), and MAP-Neo-7B-SFT (+27.7%) on LiveBench, likely due to its focus on acquiring extensive world knowledge and language patterns, not limited to overfit on specific optimization tasks. To conclude, our BaichuanSEED achieve competitive performance without any elaborate data selection and benchmark optimazation, even compared to the cutting-edge commercial LLMs.

Downstream task Performance. LLMs are typically expected to exhibit strong generalization capabilities across multiple downstream tasks. The detained performance is shown in Table 4. Our fine-tuned model BaichuanSEED-SFT consistently achieve second best in code tasks, due to the rich logical reasoning found in high-quality data. However, our model underperforms in mathematics, trailing Llama3 and MAP-Neo by nearly 10 points on MATH and 25 points on GSM8K, respectively. For example, MAPNeo achieved only 5% accuracy on MATH and 21% on GSM8K before annealing, while significant improvement after annealing with high ratio of mathematical data (Zhang et al., 2024). To emphasize, BaichuanSEED tries to be a completely pure model, without training on synthetic data or up-sampling any downstream task data in pre-training, and utilizing general instructions during SFT. Therefore, our foundation model still has significant untapped potential, especially in code and mathematical tasks, discussed in Section 6.2. This can be achieved through further pre-training with an increased proportion of downstream task data (Shao et al., 2024; DeepSeek-AI et al., 2024b) or high-quality data annealing training (Hu et al., 2024; Dubey et al., 2024). Our foundation model is comparable to Qwen1.5 and Llama on HellaSwag, and performs best after fine-tuning. Notably, our model shows no significant changes after alignment, except for a slight improvement in math benchmarks, attributed to the enhanced instruction following and comprehension abilities gained during SFT. In contrast, Qwen1.5 experienced a decline in MMLU-Pro and downstream code and math tasks, highlighting the importance of a completely pure training process to ensure the

Table 3: The performance on comprehensive benchmarks.

Training Tokens MMLU CMMLU AGIEval C-Eval MMLU-Pro LiveBench

(5-shot) (5-shot) (0-shot) (5-shot) (5-shot) (0-shot)

Baichuan2-7B 2.6T 54.65 56.95 28.95 56.19 21.65 Baichuan2-13B 2.6T 59.83 61.32 24.07 58.10 26.59 Qwen1.5-7B 3T 62.19 71.84 39.46 73.64 30.30 Llama3-8B 15T 66.57 50.68 26.74 49.89 35.30 OLMo-7B 2.5T 28.40 25.55 19.89 27.27 13.05 MAP-Neo-7B 4.5T 58.18 55.06 33.87 57.50 26.89 -

BaichuanSEED 3T 60.25 62.09 31.07 61.58 26.57 Baichuan2-7B-Chat 2.6T 54.35 55.36 35.29 55.09 25.11 12.89 Baichuan2-13B-Chat 2.6T 57.28 61.32 30.15 58.04 28.03 13.04 Qwen1.5-7B-Chat 3T 61.49 68.02 39.29 68.96 16.29 16.78 Llama3-8B-Instruct 15T 67.10 51.66 38.37 50.71 41.88 25.91 OLMo-7B-SFT 2.5T 47.49 35.49 29.12 35.43 17.99 8.80 MAP-Neo-7B-SFT 4.5T 58.31 55.24 37.98 55.58 30.24 14.35 BaichuanSEED-SFT 3T 60.15 60.84 32.62 59.41 29.63 18.32

- Table 4: The performance on code, mathematics, commonsense reasoning, and reading comprehension benchmarks.

Training Tokens MBPP HumanEval MATH GSM8K TriviaQA HellaSwag

(3-shot) (0-shot) (4-shot) (4-shot) (0-shot) (0-shot)

Baichuan2-7B 2.6T 25.40 17.68 5.94 25.02 53.73 67.56 Baichuan2-13B 2.6T 30.88 17.07 10.68 52.08 58.73 71.09 Qwen1.5-7B 3T 36.60 53.05 21.08 54.74 50.92 72.64 Llama3-8B 15T 44.60 26.22 13.44 50.11 65.23 74.54 OLMo-7B 2.5T 21.00 11.59 1.72 2.00 49.81 70.31 MAP-Neo-7B 4.5T 25.90 7.93 15.14 53.90 54.80 67.85

BaichuanSEED 3T 34.12 21.34 9.84 38.81 45.92 70.20 Baichuan2-7B-Chat 2.6T 22.40 15.24 8.70 32.37 44.65 69.18 Baichuan2-13B-Chat 2.6T 26.30 18.90 8.62 56.79 53.47 72.32 Qwen1.5-7B-Chat 3T 12.58 29.27 13.12 56.10 10.22 72.81 Llama3-8B-Instruct 15T 52.17 21.34 25.62 78.17 63.37 71.45 OLMo-7B-SFT 2.5T 25.16 19.51 2.52 17.66 42.87 72.62 MAP-Neo-7B-SFT 4.5T 33.66 29.27 30.86 70.28 53.82 68.48 BaichuanSEED-SFT 3T 37.60 23.17 14.06 53.98 43.92 73.03

consistency after alignment.

### 6 Discussion

In this section, we conduct a series of experiments to explore the effectiveness of strategies employed in the training of BaichuanSEED. Firstly, thanks to the precise categorization of subjects as mentioned in Section 3.1.3 to adjust the proportion of mathematics, we validate the potential of our model in mathematical tasks under the continued pre-training strategy. Secondly, we investigate the optimal ratio of high-density knowledge data during the pre-training phase using both cold start and continued pre-training methods on a 2B samearchitecture model.

#### 6.1 Effect of Knowledge Intensive Data

Knowledge Intensive Data refers to data that contains massive knowledge per token. We argue that LLMs trained on KID can achieve better performance in the same training steps (Hu et al., 2024; Li et al., 2023c). We extensively collect public Chinese and English academic papers and books as stated in Section 3.1.1. However, the introduction of KID inevitably down-sampling others, which may hinder the model from possessing comprehensive world knowledge or language patterns. Therefore, determining the appropriate proportion of KID is crucial.

We explore the optimal proportion of KID in two settings: training from scratch and continued pre-training. Considering the computational cost, we conduct experiments on a 2B model with the

- 52

- 53

- 54

- 55

- 56

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 22

- 23

- 24

- 25

- 26

- 27

- 28

- 29

baseline

baseline

baseline

2% math

2% math

2% math

10% math

10% math

10% math

GSM8KPerformance

MMLUPerformance

MATHPerformance

1.34 1.44 1.49 1.59 1.69 1.74

1.34 1.44 1.49 1.59 1.69 1.74

1.34 1.44 1.49 1.59 1.69 1.74

Training Token Amount (T tokens)

Training Token Amount (T tokens)

Training Token Amount (T tokens)

Figure 5: The performance of different checkpoints continued pre-training from a 1.44T token BaichuanSEED checkpoint with different proportion of mathematics-related data evaluated on MMLU, GSM8K, and MATH.

same architecture as BaichuanSEED, training from scratch or continued pre-training 100B tokens from a model that trained on 1.5T tokens and have already obtained emergent abilities, for two settings respectively. We search for the sampling proportion of KID from 0% to 50%, while keeping the remaining distribution of others consistent with Table 2. We evaluate the language modeling capabilities on a private, high-quality corpus that constructed by a wide range of domains such as encyclopedias, web pages, papers, and news.

- Table 5: The performance of models trained with different proportion of knowledge intensive data.

Train from Scratch Continued Pretrain KID Prop. 0% 25% 50% 0% 25% 50% PPL 14.24 13.68 13.86 12.21 11.73 11.80

The results are shown in Table 5. The model trained with 25% KID achieved the lowest perplexity, i.e. best language modeling capabilities. This suggests that the inclusion of KID accelerates the model’s acquisition of language modeling capabilities. Consequently, we adopted 25% as the final proportion of knowledge intensive data for training BaichuanSEED based on the scaling law.

#### 6.2 Potential on Mathematics

Taking mathematics reasoning as an example, we explore the potential of BaichuanSEED on a wide range of downstream tasks. Specifically, we first employ a classification model to perform a finer categorization of high-density knowledge data on subjects to better utilize this data. We then conduct continued pre-training experiments by altering the data proportions to verify the potential of

our model. The experimental results indicate that a higher proportion of mathematics data significantly enhances the model’s performance in mathematical downstream tasks, such as GSM8K and MATH, without compromising the comprehensive evaluation benchmarks.

To address the details, we increase the proportion of mathematics-related data to 2% and 10%, conducting 300B token continued pre-training on a checkpoint with 1.44T tokens of BaichuanSEED, as illustrated in Figure 5. At the 2% proportion, we do not observe significant gains on the mathematics benchmarks. However, when we aggressively raise the proportion to 10%, notable improvements are evident. To be specific, the 1.74T checkpoint shows a 7.4% and 65.6% increase on GSM8K and MATH respectively, with only a 0.3% fluctuation on MMLU, compared to the baseline. Therefore, we argue that knowledge intensive data has substantial potential, especially when enhancing a language model’s mathematical abilities. By searching optimal data proportions and employing curriculum learning strategies, the value of these data can be further captured. In addition, it validates the immense potential of our model in downstream tasks.

### 7 Related Works

Several institutions have published detailed technical reports and open-sourced the weights of their cutting-edge models, to promote the development of the LLM field. Recently, numerous models made efforts on meticulous data filtering (AI@Meta, 2024; Penedo et al., 2023; Maini et al., 2024; Soldaini et al., 2024), innovative architectural exploration (Ainslie et al., 2023; Jiang et al., 2024; Gu

and Dao, 2023), and optimization for vertical downstream tasks (DeepSeek-AI et al., 2024b; Shao

- et al., 2024). As a baseline model, BaichuanSEED focus on the exploration of the limits of model capabilities through data collection and deduplication, avoiding any specific optimization. Nevertheless, our model achieves competitive performance on comprehensive benchmarks, comparable to many open-source commercial models of similar scales. These representative models include earlier versions of Baichuan (Yang et al., 2023), the Llama family (Touvron et al., 2023b; Dubey et al., 2024), the Qwen series (Bai et al., 2023), the Deepseek series (Dai et al., 2024; DeepSeek-AI et al., 2024a), and the Phi series (Li et al., 2023c; Abdin et al., 2024). Moreover, there are several initiatives dedicated to disclosing technical details, such as training datasets and code, including MAPNeo (Zhang et al., 2024), OLMo (Groeneveld et al., 2024), Pythia (Biderman et al., 2023), and Amber (Liu et al., 2023). 8 Conclusions

This technical report introduces BaichuanSEED, a pure yet competitive 7 billion parameter large language model trained from scratch to explore its limits without elaborate data selection and specific downstream task optimization. We pre-train the model on 3 trillion tokens of high-quality data and supervised fine-tune it with a straightforward yet effective process. Guided by the pretraining principles that collection to scale up and reweighting to improve data quality, our model achieves comparable performance with leading commercial models on several comprehensive benchmarks. We also conduct heuristic experiments to explore the model’s potential for further optimization in downstream tasks. All training details are publicly available, with the hope that our technical report would benefit the community to further unveil the skyline of data on large language models.

### References

Marah I Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat S. Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Caio César Teodoro Mendes, Weizhu Chen, Vishrav Chaudhary, Parul Chopra, Allie Del Giorno, Gustavo de Rosa, Matthew Dixon, Ronen Eldan, Dan Iter, Amit Garg, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Jamie Huynh, Mojan Javaheripi, Xin Jin, Piero Kauffmann, Nikos Karampatziakis, Dongwoo Kim, Mahoud Khademi, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Chen Liang, Weishung Liu, Eric Lin, Zeqi Lin, Piyush Madan, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Xia Song, Masahiro Tanaka, Xin Wang, Rachel Ward, Guanhua Wang, Philipp Witte, Michael Wyatt, Can Xu, Jiahang Xu, Sonali Yadav, Fan Yang, Ziyi Yang, Donghan Yu, Chengruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. 2024. Phi-3 technical report: A highly capable language model locally on your phone. CoRR, abs/2404.14219.

AI@Meta. 2024. Llama 3 model card.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. GQA: training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 4895–4901. Association for Computational Linguistics.

Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, Colin Raffel, Shiyu Chang, Tatsunori Hashimoto, and William Yang Wang. 2024. A survey on data selection for language models. CoRR, abs/2402.16827.

Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. 2021. Program synthesis with large language models. CoRR, abs/2108.07732.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian

Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen technical report. Preprint, arXiv:2309.16609.

Yuelin Bai, Xinrun Du, Yiming Liang, Yonggang Jin, Ziqiang Liu, Junting Zhou, Tianyu Zheng, Xincheng Zhang, Nuo Ma, Zekun Wang, Ruibin Yuan, Haihong Wu, Hongquan Lin, Wenhao Huang, Jiajun Zhang, Wenhu Chen, Chenghua Lin, Jie Fu, Min Yang, Shiwen Ni, and Ge Zhang. 2024. COIG-CQIA: quality is all you need for chinese instruction fine-tuning. CoRR, abs/2403.18058.

Adrien Barbaresi. 2021. Trafilatura: A web scraping library and command-line tool for text discovery and extraction. In Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL 2021 - System Demonstrations, Online, August 1-6, 2021, pages 122–131. Association for Computational Linguistics.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, Alex X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu, Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. 2024. Deepseek LLM: scaling open-source language models with longtermism. CoRR, abs/2401.02954.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 2397–2430. PMLR.

Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. 2023. Nougat: Neural opti-

cal understanding for academic documents. CoRR, abs/2308.13418.

Andrei Z. Broder. 1997. On the resemblance and containment of documents. In Compression and Complexity of SEQUENCES 1997, Positano, Amalfitan Coast, Salerno, Italy, June 11-13, 1997, Proceedings, pages 21–29. IEEE.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Daoyuan Chen, Yilun Huang, Zhijian Ma, Hesen Chen, Xuchen Pan, Ce Ge, Dawei Gao, Yuexiang Xie, Zhaoyang Liu, Jinyang Gao, Yaliang Li, Bolin Ding, and Jingren Zhou. 2024a. Data-juicer: A one-stop data processing system for large language models. In Companion of the 2024 International Conference on Management of Data, SIGMOD/PODS 2024, Santiago AA, Chile, June 9-15, 2024, pages 120–134. ACM.

Jie Chen, Yupeng Zhang, Bingning Wang, Wayne Xin Zhao, Ji-Rong Wen, and Weipeng Chen. 2024b. Unveiling the flaws: Exploring imperfections in synthetic data and mitigation strategies for large language models. CoRR, abs/2406.12397.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. CoRR, abs/2107.03374.

Mayee F. Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, and Christopher Ré.

2023. Skill-it! A data-driven skills framework for understanding and training language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2023. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res., 24:240:1– 240:113.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. CoRR, abs/2401.06066.

Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691.

DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, Hao Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang,

Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, Tao Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, and Xiaowen Sun. 2024a. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. CoRR, abs/2405.04434.

DeepSeek-AI, Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y. Wu, Yukun Li, Huazuo Gao, Shirong Ma, Wangding Zeng, Xiao Bi, Zihui Gu, Hanwei Xu, Damai Dai, Kai Dong, Liyue Zhang, Yishi Piao, Zhibin Gou, Zhenda Xie, Zhewen Hao, Bingxuan Wang, Junxiao Song, Deli Chen, Xin Xie, Kang Guan, Yuxiang You, Aixin Liu, Qiushi Du, Wenjun Gao, Xuan Lu, Qinyu Chen, Yaohui Wang, Chengqi Deng, Jiashi Li, Chenggang Zhao, Chong Ruan, Fuli Luo, and Wenfeng Liang. 2024b. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence. CoRR, abs/2406.11931.

Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. 2023. Language modeling is compression. CoRR, abs/2309.10668.

Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan. 2024. Investigating data contamination in modern benchmarks for large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8698–8711.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen,

Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic,

Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield,

Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Logan Engstrom, Axel Feldmann, and Aleksander Madry. 2024. Dsdm: Model-aware dataset selection with datamodels. CoRR, abs/2401.12926.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi. 2024. Olmo: Accelerating the science of language models. CoRR, abs/2402.00838.

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. CoRR, abs/2312.00752.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. 2024. Deepseek-coder: When the large language model meets programming - the rise of code intelligence. CoRR, abs/2401.14196.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zhen Leng Thai, Kai Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. Minicpm: Unveiling the potential of small language models with scalable training strategies. CoRR, abs/2404.06395.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, MarieAnne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of experts. CoRR, abs/2401.04088.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. 2023. Followbench: A multi-level fine-grained constraints following benchmark for large language models. CoRR, abs/2310.20410.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, pages 1601–1611. Association for Computational Linguistics.

Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomás Mikolov. 2017. Bag of tricks for efficient text classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2017, Valencia, Spain, April 3-7, 2017, Volume 2: Short Papers, pages 427– 431. Association for Computational Linguistics.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. 2023a. CMMLU: measuring massive multitask language understanding in chinese. CoRR, abs/2306.09212.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason T. Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh,

Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023b. Starcoder: may the source be with you! Trans. Mach. Learn. Res., 2023.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023c. Textbooks are all you need II: phi-1.5 technical report. CoRR, abs/2309.05463.

Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Yonghao Zhuang, Guowei He, Haonan Li, Fajri Koto, Liping Tang, Nikhil Ranjan, Zhiqiang Shen, Xuguang Ren, Roberto Iriondo, Cun Mu, Zhiting Hu, Mark Schulze, Preslav Nakov, Tim Baldwin, and Eric P. Xing. 2023. LLM360: towards fully transparent open-source llms. CoRR, abs/2312.06550.

Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, and Navdeep Jaitly. 2024. Rephrasing the web: A recipe for compute and data-efficient language modeling. CoRR, abs/2401.16380.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300.

Noam Shazeer. 2020. GLU variants improve transformer. CoRR, abs/2002.05202.

Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Xi Victoria Lin, Noah A. Smith, Luke Zettlemoyer, Wen-tau Yih, and Mike Lewis. 2024. In-context pretraining: Language modeling beyond document boundaries. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Raghavi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian

Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. 2024. Dolma: an open corpus of three trillion tokens for language model pretraining research. CoRR, abs/2402.00159.

Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Yiding Sun, Feng Wang, Yutao Zhu, Wayne Xin Zhao, and Jiaxin Mao. 2024. An integrated data processing framework for pretraining foundation models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024, pages 2713–2718. ACM.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 13484–13508. Association for Computational Linguistics.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. CoRR, abs/2406.01574.

Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Siddartha Naidu, et al. 2024. Livebench: A challenging, contamination-free llm benchmark. arXiv preprint arXiv:2406.19314.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. 2023. Doremi: Optimizing data mixtures speeds up language model pretraining. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. 2024a. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Liang Xu, Hang Xue, Lei Zhu, and Kangkang Zhao. 2024b. Superclue-math6: Graded multi-step math reasoning benchmark for llms in chinese. CoRR, abs/2401.11819.

Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. 2024c. Benchmarking benchmark leakage in large language models. CoRR, abs/2404.18824.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, Juntao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, and Zhiying Wu. 2023. Baichuan 2: Open large-scale language models. CoRR, abs/2309.10305.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4791–4800. Association for Computational Linguistics.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023. GLM-130B: an open bilingual pre-trained model. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 12360–12371.

Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng, Jie Liu, Qunshu Lin, Raven Yuan, Tuney Zheng, Wei Pang, Xinrun Du, Yiming Liang, Yinghao Ma, Yizhi Li, Ziyang Ma, Bill Y. Lin, Emmanouil Benetos, Huan Yang, Junting Zhou, Kaijing Ma, Minghao Liu, Morry Niu, Noah Wang, Quehry Que, Ruibo Liu, Sine Liu, Shawn Guo, Soren Gao, Wangchunshu Zhou, Xinyue Zhang, Yizhi Zhou, Yubo Wang, Yuelin Bai, Yuhan Zhang, Yuxiang Zhang, Zenith Wang, Zhenzhu Yang, Zijian Zhao, Jiajun Zhang, Wanli Ouyang, Wenhao Huang, and Wenhu Chen. 2024. Map-neo: Highly capable and transparent bilingual large language model series. CoRR, abs/2405.19327.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. OPT: open pre-trained transformer language models. CoRR, abs/2205.01068.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. CoRR,

- abs/2303.18223.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A human-centric benchmark for evaluating foundation models. CoRR,

- abs/2304.06364.

Kun Zhou, Yutao Zhu, Zhipeng Chen, Wentong Chen, Wayne Xin Zhao, Xu Chen, Yankai Lin, Ji-Rong Wen, and Jiawei Han. 2023. Don’t make your LLM an evaluation benchmark cheater. CoRR, abs/2311.01964.

Yutao Zhu, Kun Zhou, Kelong Mao, Wentong Chen, Yiding Sun, Zhipeng Chen, Qian Cao, Yihan Wu, Yushuo Chen, Feng Wang, Lei Zhang, Junyi Li,

Xiaolei Wang, Lei Wang, Beichen Zhang, Zican Dong, Xiaoxue Cheng, Yuhan Chen, Xinyu Tang, Yupeng Hou, Qiangqiang Ren, Xincheng Pang, Shufang Xie, Wayne Xin Zhao, Zhicheng Dou, Jiaxin Mao, Yankai Lin, Ruihua Song, Jun Xu, Xu Chen, Rui Yan, Zhewei Wei, Di Hu, Wenbing Huang, ZeFeng Gao, Yueguo Chen, Weizheng Lu, and Ji-Rong Wen. 2024. Yulan: An open-source large language model. CoRR, abs/2406.19853.

### A Detailed Evaluation Results

The detailed performance of MMLU, CMMLU, and AGIEval can be found in Table 6. The detailed performance of each subjects of MMLU-Pro and LiveBench can be found in Table 7 and Table 8, respectively. All bold numbers indicate the best performance, while the underlined stand for the second.

### B Case Study on Global Multi-granularity Deduplication

To display the performance of our proposed global multi-granularity deduplication intuitively in Section 3.1.2, we provide several cases before and after the deduplication, as shown in Figure 6.

In terms of sentence deduplication, we filter from two dimensions: frequency and quality. On the one hand, reducing the frequency of repeated sentences, which may mitigate the repetition generation. On the other hand, it can avoid low-quality sentences being learnt by the model. These two terms are indispensable and complementary to each other. We conclude the characteristics of the removed sentences as follows, while filtering out these patterns may improve the accuracy and recall of harmful and toxic contents:

- • Advertisements. SEO strategies are adopted by advertisements to increase visibility, such as creating spam sites and utilizing various trending long-tail keywords to mislead search engines. The normal popular contents are interleaved with the advertisements, resulting in lots of template sentences. Without deduplication, relying solely on the quality score is not sufficient to filter them out.
- • Texts for dissemination and interactivity. It is widely utilized in web pages, which is one of the most common patterns. While it is usually unrelated to the main idea of the documents.
- • Copyrights and PII. These similar patterns mostly appear in news, books. It may address privacy concerns, and aggravate the LLM repetition generation issue.
- • Residual HTML formats and watermarks. Inappropriate custom parsing website templates may address this issue.

|Freq.|Highlighted Sentence Quality Score|Document Before Deduplication|Document After Deduplication|Comments|
|---|---|---|---|---|
|Over 10M|0.071|我依然爱你,我只是不喜欢你了。\n再好的 ⽯头放在⼼头也会难受,⼼⾥的⽯头终有放 下的时候。\n将⽂章分享到朋友圈。\n将⽂ 章分享到朋友圈。\n或许她就像故事中的⼥<br><br>孩⼀样,或许我应该喜欢她的,我确实是喜欢 她的,可为什么有那么多原因,那么多理由让 我不能选择爱你。\n将⽂章分享到朋友圈。<br><br>\n将⽂章分享到朋友圈。快来留⾔吧!|我依然爱你,我只是不喜欢你了。 \n再好的⽯头放在⼼头也会难受, ⼼⾥的⽯头终有放下的时候。或 许她就像故事中的⼥孩⼀样,或许 我应该喜欢她的,我确实是喜欢她 的,可为什么有那么多原因,那么多 理由让我不能选择爱你。|• Common sentences for the dissemination and interactivity|
|Among 3M and 4M|0.816|Online Lottery OL suggested that, White was thrilled to sign with the Rockets, ‘I knew I was on the radar, my agent told me there were a couple of teams that had been watching me, so I worked hard and honed myself to be more wellrounded. Pay 1 dollar to gain bonus discount for further lottery payment.|[REMOVED]|• Trending long-tail sentence for betting websites SEO<br>• But high sentence quality score<br>|
|Among 0.1M and 1M|0.110|Disclaimer: The content of this article and pictures are from the network. if there is a problem, please contact us to delete. Generally when the server hangs, crashes or has low performance, it is necessary to capture the server‘s thread stack (Thread Dump) for subsequent analysis. [Omit Details] OS command to get Thread Dump ps -ef | grep java kill -3 <pid> The code has been copied to the clipboard Note: Be careful, one wrong step can kill the server process. kill -9 will kill the process.|Generally when the server hangs, crashes or has low performance, it is necessary to capture the server‘s thread stack (Thread Dump) for subsequent analysis. [Omit Details] OS command to get Thread Dump ps -ef | grep java kill -3 <pid> Note: Be careful, one wrong step can kill the server process. kill -9 will kill the process.|• Website fixed patterns<br>• Advertisements<br>• Website watermarks<br>• Related recommendations<br>|
|Among 0.1K and 10K|0.074|这是9⽉18⽇在台北市动物园拍摄的努⼒挣 扎着翻⾝的“圆仔”,当时体重4702克,但是成 功率不⾼,时常累得仰⾯朝天睡了起来。新 华社发(台北市动物园提供)\n2013年7⽉6⽇,<br><br>⼤陆赠台⼤熊猫“团团”“圆圆”的⾸只幼崽在 台北市动物园出⽣。新华社发(台北市动物 园提供）⼤熊猫宝宝“圆仔”出⽣⾄今从未公 开露⾯,台北市动物园⼯作⼈员拍摄的⼀张 张照⽚全程记录着它成⻓的点滴。作者:记 者 [Interview Name] 编辑/剪辑: [Editor Name] 温馨提示|这是9⽉18⽇在台北市动物园拍摄 的努⼒挣扎着翻⾝的“圆仔”,当时 体重4702克,但是成功率不⾼,时常 累得仰⾯朝天睡了起来。2013年7 ⽉6⽇,⼤陆赠台⼤熊猫“团团”“圆 圆”的⾸只幼崽在台北市动物园出 ⽣。⼤熊猫宝宝“圆仔”出⽣⾄今 从未公开露⾯,台北市动物园⼯作 ⼈员拍摄的⼀张张照⽚全程记录 着它成⻓的点滴。|• Sources<br>• Author or editor PII<br>• Breadcrumb contents<br>|
|Under 0.1K|0.920|• 肺癌晚期的患者采⽤保守治疗为主,其中 中药偏⽅能够帮助患者提⾼免疫⼒,减轻 不适症状,⽐如使⽤⽩茅根,仙鹤草,蛇莓, 猪肺,⽣姜,蜜枣等做成茅根仙鹤草蛇莓猪 肺汤。<br>• 如果龋病得不到及时有效的治疗,还可能 引发⽛髓病和根尖周病,导致患者⽛痛难 忍,甚⾄不得不拔⽛。<br>• 甲状腺炎性结节属于甲状腺结节的⼀种, 主要是因为病毒感染或者细菌感染所致。<br>|[KEPT for further reweighting]|• Good document examples|

###### Figure 6: The cases with different frequency before and after our global multi-granularity deduplication. We provide the frequency, the quality score of highlighted sentences, and our comments on each case. The contents highlighted in blue are removed after deduplication.

Table 6: The detailed performance on MMLU, CMMLU, and AGIEval.

Models MMLU CMMLU AGIEval Avg. HUM SOC STEM OTH Avg. CHI SOC STEM HUM OTH Avg. ZH EN Gaokao

Baichuan2-7B 54.65 60.17 62.25 45.03 56.31 56.95 59.31 62.10 43.42 61.05 61.19 28.95 31.17 25.99 35.24 Baichuan2-13B 59.83 65.56 68.54 48.54 62.55 61.32 62.59 65.93 47.36 66.79 65.63 24.07 35.96 8.21 39.29 Llama3-8B 66.57 70.38 75.93 56.33 68.97 50.68 44.78 52.67 43.54 52.96 53.88 26.74 31.45 20.46 34.81 Qwen1.5-7B 62.19 65.73 70.02 53.49 64.15 71.84 73.01 74.09 63.07 76.01 74.84 39.46 53.08 21.31 59.26 OLMo-7B 28.40 25.13 30.01 29.99 28.52 25.55 24.97 25.27 25.27 26.58 25.36 19.89 18.84 21.29 19.44 MAP-Neo-7B 58.18 60.88 66.83 59.86 49.73 55.06 55.88 59.44 45.58 62.14 53.24 33.87 33.91 33.83 37.92

BaichuanSEED 60.25 63.78 68.85 50.22 61.39 62.09 61.36 66.20 47.57 66.13 65.89 31.07 34.70 21.17 40.28 Baichuan2-7B-Chat 54.35 60.27 62.37 43.48 57.10 55.36 56.24 58.75 43.61 61.05 61.19 35.29 37.98 21.71 42.70 Baichuan2-13B-Chat 57.28 62.58 65.65 46.72 59.67 59.74 61.10 64.54 46.37 65.10 63.19 30.15 41.02 15.65 46.43 Llama3-8B-Instruct 67.10 70.75 76.88 57.08 69.29 51.66 45.84 52.92 44.21 52.49 57.53 38.37 38.30 38.46 41.53 Qwen1.5-7B-Chat 61.49 66.59 69.12 52.65 62.26 68.02 68.58 71.21 59.31 71.83 69.92 39.29 39.87 38.52 46.26 OLMo-7B-SFT 47.49 48.48 55.78 51.23 39.02 35.49 33.60 37.81 30.81 33.84 38.82 29.12 27.37 31.46 30.42 MAP-Neo-7B-SFT 58.31 60.44 66.22 58.23 51.90 55.24 55.73 57.51 47.25 63.43 53.86 37.98 38.22 37.66 43.69 BaichuanSEED-SFT 60.15 65.10 67.82 49.68 62.70 60.84 60.28 64.76 49.07 64.36 65.46 32.62 40.28 20.42 47.44

- Table 7: The detailed performance on MMLU-Pro.

Models Avg. BIO BUS CHEM CS ECO ENG HEA HIST LAW MATH OTH PHIL PHY PSY Baichuan2-7B 21.65 33.61 19.90 14.31 21.95 27.01 12.80 24.33 23.10 16.17 14.80 25.22 22.65 15.09 32.21 Baichuan2-13B 26.59 41.84 26.36 15.28 27.32 33.77 16.10 30.93 27.82 15.53 19.62 31.28 29.46 19.71 37.22 Llama3-8B 35.30 56.40 32.07 24.82 33.66 46.68 25.49 43.28 36.22 19.52 30.42 40.48 31.41 53.26 41.45 Qwen1.5-7B 30.30 43.51 28.01 23.59 32.93 39.57 21.98 32.89 28.08 18.35 30.42 32.36 28.26 22.40 41.85 OLMo-7B 13.05 14.92 12.67 9.72 12.93 17.42 10.22 15.89 12.07 11.35 11.55 12.99 14.83 12.47 13.66 MAP-Neo-7B 26.89 46.44 26.24 19.70 24.63 33.89 22.08 26.04 26.25 14.26 25.46 27.60 25.65 23.33 34.96 BaichuanSEED 26.57 40.03 25.73 17.14 23.90 33.29 19.30 33.01 24.93 14.90 21.02 30.52 27.86 18.40 41.98 Baichuan2-7B-Chat 25.11 44.49 19.52 15.81 24.39 33.89 17.03 30.32 24.67 16.80 14.80 27.27 27.25 17.17 38.10 Baichuan2-13B-Chat 28.03 45.47 25.86 16.78 25.37 38.51 15.17 30.68 31.23 18.71 19.62 32.47 30.06 19.25 43.23 Llama3-8B-Instruct 41.88 66.53 37.14 32.24 41.46 53.20 31.37 47.68 40.16 25.34 35.16 33.05 40.48 33.49 58.02 Qwen1.5-7B-Chat 16.29 28.45 21.80 11.13 16.34 13.86 10.42 20.78 11.29 11.08 20.50 13.64 23.05 10.24 15.54 OLMo-7B-Chat 17.99 26.64 14.20 13.25 19.02 22.39 12.69 23.59 14.96 15.35 12.51 21.75 15.83 13.01 26.69 MAP-Neo-7B-Chat 30.24 49.51 30.80 21.55 29.76 38.86 19.50 29.34 32.02 16.26 34.64 30.52 25.25 25.87 39.47 BaichuanSEED-SFT 29.63 49.93 24.84 14.93 30.00 37.20 18.27 34.47 31.50 21.25 21.69 31.60 30.66 20.40 48.12

- Table 8: The detailed performance on LiveBench.

Models Avg. Code DA IF Language Math Reasoning Baichuan2-7B-Chat 12.89 2.00 13.78 40.13 5.15 6.96 9.33 Baichuan2-13B-Chat 13.04 2.00 11.20 44.72 4.70 6.94 8.67 Llama3-8B-Instruct 25.91 16.95 25.33 56.84 19.40 15.59 21.33 Qwen1.5-7B-Chat 16.78 6.63 16.23 44.12 6.18 12.86 14.67 Qwen2-7B-Chat 26.23 29.21 28.75 44.74 10.21 25.83 18.67 OLMo-7B-SFT 8.80 0.00 7.33 32.76 3.53 5.81 3.33 MAP-Neo-7B-SFT 14.35 7.95 4.27 36.84 7.48 11.59 18.00 BaichuanSEED-SFT 18.32 5.00 15.75 52.91 7.12 16.46 12.67

