# Open-FinLLMs: Open Multimodal Large Language Models for Financial Applications

arXiv:2408.11878v3[cs.CL]7Jun2025

Jimin Huang1, Mengxi Xiao1, Dong Li1, Zihao Jiang1, Yuzhe Yang4, Yifei Zhang5, Lingfei Qian1, Yan Wang1, Xueqing Peng1, Yang Ren1, Ruoyu Xiang1, Zhengyu Chen1, Xiao Zhang1, Yueru He3, Weiguang Han2, Shunian Chen4, Lihang Shen3, Daniel Kim6, Yangyang Yu8, Yupeng Cao8, Zhiyang Deng8, Haohang Li8, Duanyu Feng9, Yongfu Dai9, VijayaSai Somasundaram10, Peng Lu11, Guojun Xiong14, Zhiwei Liu7, Zheheng Luo7, Zhiyuan Yao8, Ruey-Ling Weng1, Meikang Qiu14, Kaleb E Smith15, Honghai Yu5, Yanzhao Lai1, Min Peng2, Jian-Yun Nie11, Jordan W. Suchow8, Xiao-Yang Liu3*, Benyou Wang4*, Alejandro Lopez-Lira10*, Qianqian Xie1,2*, Sophia Ananiadou7,16,17, Junichi Tsujii 16

1The Fin AI 2Wuhan University 3Columbia University 4The Chinese University of Hong Kong, Shenzhen 5Nanjing University 6Rensselaer Polytechnic Institute 7The University of Manchester 8Stevens Institute of Technology 9National University of Singapore 10University of Florida 11University of Montreal 12Yale University 13New York University 14Harvard University 15NVIDIA 16Artificial Intelligence Research Centre 17Archimedes/Athena Research Centre xl2427@columbia.edu, wangbenyou@cuhk.edu.cn, alejandro.lopez-lira@warrington.ufl.edu, xqq.sincere@gmail.com * Corresponding Authors

Abstract

Financial LLMs hold promise for advancing financial tasks and domain-specific applications. However, they are limited by scarce corpora, weak multimodal capabilities, and narrow evaluations, making them less suited for real-world application. To address this, we introduce Open-FinLLMs, the first open-source multimodal financial LLMs designed to handle diverse tasks across text, tabular, time-series, and chart data, excelling in zero-shot, few-shot, and fine-tuning settings. The suite includes FinLLaMA, pre-trained on a comprehensive 52billion-token corpus; FinLLaMA-Instruct, finetuned with 573K financial instructions; and FinLLaVA, enhanced with 1.43M multimodal tuning pairs for strong cross-modal reasoning. We comprehensively evaluate Open-FinLLMs across 14 financial tasks, 30 datasets, and 4 multimodal tasks in zero-shot, few-shot, and supervised fine-tuning settings, introducing two new multimodal evaluation datasets. Our results show that Open-FinLLMs outperforms advanced financial and general LLMs such as GPT-4, across financial NLP, decision-making, and multi-modal tasks, highlighting their potential to tackle real-world challenges. To foster innovation and collaboration across academia

and industry, we release all codes and models 1 under OSI-approved licenses.

## 1 Introduction

The advancements of financial AI have been significantly driven by the progress of natural language processing (NLP) techniques, particularly large language models (LLMs) (Brown et al., 2020; Bubeck et al., 2023). Commercial LLMs like OpenAI’s GPT-4 (Achiam et al., 2023) and open-source LLMs such as Meta AI’s LLaMA series (Touvron et al., 2023a,b) have set new benchmarks in NLP tasks and vertical-domain tasks like medicine, owing to their impressive text understanding and generation capabilities. However, these generalpurpose LLMs face limitations in the financial domain due to the knowledge gap. These models are primarily pretrained on general texts and lack an understanding of financial terminology, regulations, and market nuances (Xie et al., 2023a; Wu et al., 2023a; Xie et al., 2023b). Additionally, they are unable to effectively handle non-text data, such as tabular and time-series data, which are critical components of financial knowledge.

1https://huggingface.co/collections/TheFinAI/ open-finllms-66b671f2b4958a65e20decbe

To bridge this gap, researchers have developed specialized financial LLMs through pretraining from scratch (BloombergGPT (Wu et al., 2023b)), continual pre-training (FinTral (Bhatia et al., 2024)), or instruction tuning (PIXIU (Xie et al., 2023b), FinGPT (Liu et al., 2023, 2024c; Yang et al., 2023)) using domain-specific data (Table 1). However, these models still face notable challenges (Nie et al., 2024): First, they rely on limited domain-specific corpora for continual pretraining and instruction tuning, restricting their ability to fully capture the complexity of financial knowledge, language, and data types. For example, FinTral uses only 20 billion tokens for continual pre-training. Second, they show limited multimodal capabilities, lacking support for tabular, time-series, and chart data. Most models, like BloombergGPT, focus solely on text, missing critical aspects of real-world tasks such as portfolio optimization and trend analysis. Third, evaluations are conducted on narrow scenarios, mainly instruction-tuned financial NLP tasks. Zero/fewshot performance, multimodal reasoning, and financial decision-making tasks remain underexplored, limiting real-world applicability.

To address these limitations, in this paper, we introduce Open-FinLLMs, a series of financial large language models tailored for various financial tasks. We begin with FinLLaMA, a groundbreaking foundational model pre-trained on a massive 52-billiontoken corpus comprising text, tabular, and timeseries data from high-quality financial sources such as reports, papers, and market data for the first time. This innovative pre-training strategy incorporating extensive data modalities equips FinLLaMA with deep financial insights and analytical capabilities. We further develop FinLLaMA-Instruct by finetuning the model with expanded datasets of 573K diverse and high-quality financial instructions. Enriched instruction-tuning datasets enhance not only the model’s ability to follow instructions but also coverage of financial domain knowledge, leading to better performance for a wide range of downstream tasks. To handle multimodal financial data, we present FinLLaVA, leveraging 1,430K financial multimodal instruction pairs, including images, text, charts, and tabular data. Unlike traditional methods that primarily focus on standard imagetext pairs, our fine-tuning process is the first to incorporate chart image-text pairs and images of tabular layouts. This innovation enables the model to effectively interpret and process complex finan-

cial data with improved precision and versatility.

We comprehensively evaluate our models across 14 financial tasks and 30 datasets. FinLLaMA is tested on 19 datasets (9 tasks) in zero-shot and 4 datasets (3 tasks) in few-shot settings. It outperforms other financial and general LLMs including LLaMA3-8B, LLaMA3.1-8B, and BloombergGPT in almost all tasks (including broad financial NLP, reasoning, asset trading and decision making tasks) in the zero-shot and few-shot settings, showing strong generalization . FinLLaMA-Instruct outperforms other financial LLMs on 4 out of 6 domainspecific tasks like sentiment analysis, NER, numeric understanding, summarization, stock prediction, and credit scoring, also surpassing GPT-4 on 3 tasks. While FinLLaVA outperforms other open multimodal LLMs on 4 multimodal tasks (1 general, 3 financial), and even surpasses advanced close source models including GPT-4o and Gemini1.5-pro on the tabular task, Together, these highlight Open-FinLLMs as powerful tools for realworld financial applications.

In summary, our key contributions are2: (1) We introduce Open-FinLLMs, a groundbreaking suite of financial LLMs trained on three comprehensive datasets tailored for different training stages. This structured training pipeline ensures a robust understanding of financial terminology, numerical data, and complex financial contexts. (2) By pioneering the integration of tabular and time-series data, Open-FinLLMs are the first financial foundation models with advanced multimodal capabilities. This multimodal approach bridges the gap between textual and structured financial data, enabling a more comprehensive capture of financial knowledge from diverse data types. (3) Extensive experiments demonstrate Open-FinLLMs’ superior performance across a range of financial and multimodal benchmark tasks, including zero-shot and few-shot settings. Notably, the models excel in both traditional NLP tasks and finicial tasks in real applications, showcasing their readiness to tackle complex challenges understanding multimodal finicial data. (4) We have open-sourced the training code, datasets, and models, providing a valuable reference for future research on LLMs in the financial domain. The shared code and training methodologies can also support advancements in other fields that require processing time-series data or

2We will publicly share the training code, datasets, and models.

Table 1: Comparison of key elements between Open-FinLLMs with other financial LLMs. Abbreviations: PT for pre-training from scratch, CPT for continual pre-training, and IFT for instruction fine-tuning.

Evaluation Zero-shot Few-shot Instruction-tuned Multimodal Trading

Model Backbone Size PT CPT IFT Tabular Time Chart

BloombergGPT (Wu et al., 2023b) BLOOM 50B 363B × × × × × × ✓ × × × PIXIU (Xie et al., 2023b) LLaMA 7/30B × × 128K × × × × × ✓ × × FinGPT (Liu et al., 2023) LLaMA2 7B × × 205.3K × × × × × ✓ × × FinTral (Bhatia et al., 2024) Mistral 7B × 20B 226.3K × × ✓ × × ✓ ✓ × Open-FinLLMs LLaMA3 8B × 52B 573K ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

understanding multimodal information.

## 2 Related Work

### 2.1 Financial Large Language Models

Recently, several financial LLMs have been developed to address domain-specific challenges. BloombergGPT (Wu et al., 2023a) pioneered financial LLMs with pretraining on 363 billion financial tokens; however, it remains closed-source, limiting its accessibility. On the open-source front, PIXIU (Xie et al., 2023b) and FinGPT (Liu et al., 2023, 2024c; Yang et al., 2023) fine-tuned LLaMA models with hundreds of thousands of financial instructions, focusing primarily on text-based tasks. FinTral (Bhatia et al., 2024) introduced multimodal capabilities for tabular data, but limited by the small size of its domain-specific data and its evaluation scope. Fin-o1 (Qian et al., 2025), developed through instruction tuning on reasoning-enhanced data, demonstrates that domain-enhanced reasoning abilities significantly improve the model’s performance in reasoning tasks.

### 2.2 Domain Specialization of Large Language Models

Domain-specific LLMs are developed using three main strategies: pre-training from scratch (PT), continued pre-training (CPT), and instruction finetuning (SFT) (Wu et al., 2024). CPT adapts existing LLMs by further training on domain data, like Code LLaMA (Roziere et al., 2023), which improves code generation. SFT tailors models for specific tasks with domain instructions, requiring less data and compute, as demonstrated by PIXIU (Xie et al., 2023b) and FinGPT (Liu et al., 2024c).

## 3 Open-FinLLMs: Open Multimodal Financial LLMs

In this section, we introduce the Open-FinLLMs model family as shown in Figure 1, including FinLLaMA for foundational financial knowledge, FinLLaMA-Instruct for instruction-following tasks,

Table 2: Comparison of financial corpora used for pre-training BloombergGPT (Wu et al., 2023b), FinTral (Bhatia et al., 2024), and Open-FinLLMs.

Data Types BloombergGPT FinTral Open-FinLLMs Financial Papers - - 4B Conference Calls - - 5B Financial Reports 9B 1.55M 5B Indicators - - 12B News+Social Media 43B 5.657B 7B Historical Data - - 13B SEC Filings 14B 2.55B 6B Web Data 298B 11.75B Total 363B 20.0B 52B

and FinLLaVA for multimodal financial applications.

3.1 FinLLaMA: Specializing LLaMA3 for Finance with Continual Pre-training

Curation of Continual Pre-training Corpus. To facilitate effective continual pretraining, we construct a comprehensive financial corpus comprising 52 billion tokens sourced from seven diverse financial domains. As shown in Table 2, our continual pre-training corpus encompasses a wide array of data sources to ensure comprehensive coverage of financial knowledge: (1) Financial papers: 4 billion tokens from academic papers and research articles, offering a strong foundation in financial concepts and theories. These papers, sourced from SSRN3 and open-source conference proceedings, span from 2000 to 2023 and cover topics such as market analysis, financial modeling, and economic theory. (2) Conference calls: 5 billion tokens of open-source transcripts from earnings calls, analyst meetings, and investor briefings, collected from 09/08/2004 to 12/17/2021, providing real-time insights into corporate performance and strategies. (3) Financial reports: This component consists of 5 billion tokens from annual and quarterly reports, covering the period from 2005 to 2020, crucial for assessing a company’s financial health and market positioning. (4) Technical indicators: 12 billion

3https://www.ssrn.com/index.cfm/en/

[Figure 1]

Figure 1: Overview of Open-FinLLMs.

tokens of economic indicators and financial ratios sourced from company filings and open market data spanning from 2009 to 2023, essential for macroeconomic analysis and investment decisions. (5) News and social media: 7 billion tokens from financial news outlets and social media platforms collected from 1999 to 2021, offering timely updates on market trends and public opinion. (6) Historical data: 13 billion tokens of historical stock prices, trading volumes, and market data from 1999 to 2022, vital for quantitative analysis and algorithmic trading. (7) SEC filings: 6 billion tokens from U.S. SEC filings, such as 10-K and 10-Q reports spanning from 1994 to 202, providing comprehensive insights into corporate activities and performance. Unlike existing models, we chose not to use web data due to its higher noise level compared to other data sources. See Appendix A for more details of the pre-training data.

Mixing Financial Data with General Data. To prevent catastrophic forgetting, as highlighted in previous studies (Wu et al., 2023a; Gupta et al.), we integrate a subset of general-domain data with our financial-domain corpus. We use the Fineweb dataset (Penedo et al., 2024), which contains over 15 trillion tokens of cleaned and deduplicated English web data from CommonCrawl. Using the approach outlined in DoReMi (Xie et al., 2024b), we determine the optimal mixture ratio of financial to general-domain data to be around 3:1. Accordingly, we sampled a total of 18 billion tokens from the general-domain corpus. This mixture helps our models retain previously learned general knowledge while being specifically fine-tuned for financial tasks.

Pretraining Details. We use LLaMA3-8B

as the backbone, optimizing the token sequence likelihood with a standard language modeling objective. The model maximizes L(Θ) =

k i log PΘ(xi|x1,...,xi−1) to predict token sequences. Distributed training is performed using DeepSpeed on 64 A100 80GB GPUs across 8 nodes, requiring 250 hours for 1 epoch. The learning rate is set to 1×10−5 with a cosine schedule, weight decay of 0.00001, and a 0.05 warm-up ratio. The batch size is 2 per device, with a maximum sequence length of 8,192 tokens.

3.2 FinLLaMA-Instruct: Domain Task Optimization through Instruction Tuning

Building on the foundation of FinLLaMA, we propose FinLLaMA-Instruct-8B, developed through instruction fine-tuning to enhance the model’s instruction-following capabilities and optimize performance on downstream domain tasks.

Financial Instruction Dataset To optimize FinLLaMA for downstream domain tasks and instruction following ability, we have assembled an extensive instruction-tuning dataset, totaling 573K samples, specifically tailored for financial applications, as shown in Table 3. The dataset is sourced from four open data sources: (1) FLUPE (Xie et al., 2023b), with 123K samples covering key financial NLP tasks; (2) finred (Liu et al., 2023), with approximately 32.67K examples focused on financial report and document comprehension; (3) MathInstruct (Yue et al., 2023), with 262K examples from 13 distinct mathematical rationale datasets; and (4) Sujet-Finance-Instruct-177k, which integrates data from 18 different financial NLP datasets. To ensure the uniqueness and quality of the data, we performed rigorous deduplication and filtering, ad-

dressing potential overlaps between datasets like FLUPE and Sujet-Finance-Instruct-177k. See Appendix B for more details on the instruction tuning data and data processing.

Table 3: Overview of instruction datasets used in FinLLaMA-Instruct and comparison with FinTral’s data.

Source Fintral FinLLaMA-Instruct

ChanceFocus/FLUPE 123.0k 123.0k FinGPT/Hingpt-finred 32.67k 32.67k TIGER-Lab/MathInstruct 26.2k 262k sujet-ai/Sujet-Finance-Instruct-177k - 177k

Total after deduplication 226.3k 573k

Instrution-tuning Details. For instruction tuning, we utilize FinLLaMA as the backbone model and conduct training on 8 NVIDIA A100 80GB GPUs for 6 hours. The model is optimized using Qlora (Dettmers et al., 2024) via AutoTrain 4, configured with a block size and model maximum length of 4096. We train the model over 2 epochs with a batch size of 1 and a learning rate of 0.0002. Parameter-efficient tuning is achieved with LoRA settings of r = 64, α = 128, and no dropout, using INT4 quantization. All linear modules are targeted, with right-aligned padding. The AdamW optimizer (Loshchilov and Hutter), coupled with a cosine scheduler, is used for optimization, along with gradient accumulation set to 4.

- 3.3 FinLLaVA: Enabling Multimodal Capabilities via Multimodal Instruction Tuning

Based on FinLLaMA, we build its financial multimodal extension FinLLaVA to address multimodal financial tasks by leveraging multimodal instruction tuning based on the LLaVA-1.5 (Liu et al.,

- 2024a) framework. Multimodal Instruction Data. We curated a

diverse multi-modal dataset comprising image, tabular, chart, and text data to ensure comprehensive coverage of various data formats, as shown in Table 4. For image data, we utilized three vision instruction datasets: ALLaVA-4V (Chen et al.,

- 2024b), LLaVA-v1.5-mix665k (Liu et al., 2024b), and OCR-VQA (Wang et al., 2023). For chart data, we integrate subsets from multiple sources: UniChart (Masry et al., 2023), with 5k chart imagetext pairs; Chart2Text (Obeid and Hoque, 2020), with 30k chart image-text pairs; and ChartQA

4https://huggingface.co/autotrain

Table 4: Statistics of multimodal instruction dataset. SFT stands for supervised fine-tuning. Asterisk (*) indicates the dataset only contains textual data.

Stage Dataset Source Instructions

ALLaVA-4V (Chen et al., 2024b) 468k OCR-VQA (Wang et al., 2023) 79k

Multimodal Alignment

SynthTabNet (Nassar et al., 2022) 20k UniChart (Masry et al., 2023) 5k ChartQA (Masry et al., 2022) 20k

Chart2Text (Obeid and Hoque, 2020) 30k SFT

LLaVA-v1.5-mix665k (Liu et al., 2024b) 665k Evol-Instruct * (Chen et al., 2024b) 143k

Total 1430k

(Masry et al., 2022), featuring 20k chart images and their associated question-answering (QA) pairs. We used GPT-4o to evaluate these datasets and filtered out images most relevant to the financial domain. Further details are provided in Appendix C.2. Different from previous work (Bhatia et al., 2024), which only image and chart data, we selected a subset of the SynthTabNet (Nassar et al., 2022) dataset, consisting of 20k annotated images of data in tabular layouts. Further details are provided in Appendix C.1. Additionally, we included EvolInstruct (Chen et al., 2024b), a dataset of 143k pure text instructions, to enhance the model’s generalization capabilities and reduce the risk of hallucination.

Multimodal Instruction Finetuning. We utilize CLIP (Radford et al., 2021) as our visual encoder in conjunction with the FinLlaMA language decoder, fine-tuning the model on our multimodal instruction dataset. Our approach follows the training framework established by LLaVA1.5 (Liu et al., 2024b), implementing a two-stage instruction-tuning process. Multimodal Alignment: In this initial stage, we aim to align the vision encoder’s output with the language model’s embedding space. During this phase, both the vision encoder and LLM weights remain frozen. The key objective is to train a two-layer MLP projector to bridge the gap between the vision encoder’s features and the LLM’s embedding. For each input, consisting of an image Xv, instructions Xinstruct that may involve single-turn or multi-turn conversations, and the target answer Xa, the vision encoder processes the image data to generate a vision feature: Zv = g(Xv). The MLP projector then maps Zv into the embedding space of the language model: Hv = fMLP(Zv;θ), where θ represents the trainable parameters of the projector. The training objective is to maximize the auto-regressive likelihood: Li Pθ(Xi | Xv,Xinstruct,<i,Xa,<i), where L is the sequence length of the target an-

swer Xa, and Xinstruct,<i and Xa,<i are the tokens of instructions and answers preceding the current prediction Xi. Supervised Fine-tuning: In the second stage, we continue updating the parameters of both the language model and the MLP projector, while keeping the vision encoder’s parameters frozen. We maintain the same autoregressive training objective as the previous stage but apply it to a different dataset. As shown in Table 4, we utilize LLaVA-v1.5-mix665k and Evol-Instruct as our primary training data sources.

Training Details: In the multi-modal alignment stage, we set the global batch size to 128, the learning rate to 1 × 10−3, with a warm-up ratio of 0.03 and cosine decay. Training uses bf16 and tf32 precision for stability and acceleration. Weight decay is set to 0.0, and the model’s maximum length is 2048 tokens. We train on eight NVIDIA HGX H20 GPUs, completing one epoch in approximately 30 hours. In the SFT stage, we set the global batch size to 256, the learning rate to 2 × 10−5, with a warmup ratio of 0.05 and cosine decay. The model’s maximum length is increased to 8192 tokens for longer sequences. Weight decay remains at 0.0, and training runs for one epoch with the same precision settings for efficiency and performance.

## 4 Experiments

We conducted broad evaluations on general and financial tasks. Unlike FinTral (Bhatia et al., 2024), which only reported instruction-tuned performance, we assess: 1) zero- and few-shot performance of the FinLLaMA base model, 2) FinLLaMA-Instruct performance, 3) trading performance, and 4) multimodal capabilities.

### 4.1 Performance of FinLLaMA

In this section, we evaluate the continual pretrained FinLLaMA model using 26 datasets spanning 11 critical financial tasks, categorized into financialNLP tasks (Sentiment Analysis, Classification, QA, and Named Entity Recognition) and financialspecific tasks (Misinformation, Math, Credit Scoring, Fraud Detection, Financial Distress, Claim Analysis, Decision Making, Auditing and Risk & Compliance). We compare its performance against competitor models by reproducing their results where possible or reporting the best publicly available scores.

Zero-shot Performance In zero-shot scenarios, as shown in Table 8, our evaluation uses 19 datasets

covering both financial-NLP and financial-specific tasks, A detailed description of evaluation tasks and detailed prompts for each dataset are available in Appendix E.4.1 and Appendix E.4.2.

Overall, Table 5 (and Figure 5 in Appendix E.4.2) demonstrates that FinLLaMA outperformed the baseline models on most financial tasks, highlighting its robustness and versatility in zeroshot settings. It surpasses its backbone model, LLaMA3-8B, on all tasks, highlighting the effectiveness of continual pre-training with large-scale domain-specific data in enhancing financial knowledge. Additionally, FinLLaMA exceeds the performance of BloombergGPT, despite its larger model size of 50B, and also outperforms the current most capable open-source LLM, LLaMA3.1-8B on most tasks.

FinLLaMA shows exceptional performance in sentiment analysis and classification tasks, demonstrating its proficiency in fundamental financial operations. Its improved accuracy in fact-checking, as shown by the FinFact dataset, highlights its ability to comprehend and evaluate financial information, enabling precise judgments on claims. In math problem-solving tasks, our model shows improvement across all datasets (MC, KnowledgeMath, and DocMath-Eval) compared to LLaMA3-8B.

In credit scoring tasks, performance varies significantly across datasets. Upon manual inspection of prediction results, we found that on the German dataset, both our model and LLaMA3-8B predicted all cases as one category, while LLaMA3.1-8B predicted the opposite. On the Australian dataset, our model’s superior performance demonstrates the benefits of continuous pre-training, even with anonymized features. In the LendingClub dataset, our model outperformed LLaMA3-8B, though not LLaMA3.1-8B, likely due to its larger scale training data. Additionally, FinLLaMA excelled in fraud detection, financial distress identification, and claim analysis, showcasing its robust capabilities across diverse financial tasks.

Few-shot Performance For the few-shot evaluation setting, we use four datasets covering three financial-NLP tasks, as shown in Table 9. These tasks are aligned with the BloombergGPT evaluation settings to ensure consistency and comparability. Detailed descriptions of the evaluation tasks can be found in Appendix E.4.2.

Overall, as shown in Fig. 6, FinLLaMA consistently outperforms baseline models across a wide range of financial tasks, demonstrating its robust-

- Table 5: Performance of FinLLaMA and baseline models on benchmark tasks, ranging from 0 to 100. We boldface the best performance in each benchmark task. (-) indicates N/A.

Category Task Dataset LLaMA3-8B LLaMA3.1-8B BloombergGPT FinLLaMA

Financial Sentiment Analysis TSA 75.00 67.00 - 81.00 (zero-shot) Classification FOMC 41.00 47.00 - 50.00

Classification FinArg-AUC 51.00 51.00 - 55.00 Classification MA 34.00 51.00 - 70.00 Classification SC 69.00 73.00 - 86.00

Misinformation FinFact 29.48 32.07 - 34.62 Math MC 15.00 19.3 - 18.00 Math KnowledgeMath 2.3 2.7 - 2.5 Math DocMath-Eval 1.8 3.5 - 3.1

Credit Scoring German 34.00 66.02 - 34.00 Credit Scoring Australian 27.60 26.00 - 49.80 Credit Scoring LendingClub 9.30 38.00 - 22.60

Fraud Detection ccf 50.10 50.06 - 50.10 Fraud Detection ccfraud 49.20 48.02 - 50.25

Financial Distress polish 47.65 50.00 - 50.00 Financial Distress taiwan 45.80 47.75 - 50.00

Claim Analysis ProtoSeguro 48.95 49.35 - 49.55 Claim Analysis travelinsurance 50.00 50.00 - 50.55

QA ConvFinQA 31.95 32.55 43.41 51.41 Risk& Compliance RegulationQA 16.72 17.35 17.34

Auditing Abbreviation 14.68 15.99 18.92 (5-shots) Sentiment Analysis FPB 69.65 13.08 51.07 70.25

Sentiment Analysis FiQA-SA 52.29 65.39 75.05 75.34

Classification Headlines 80.59 59.95 82.20 85.54 (20-shots) Named Entity Recognition NER 39.18 49.04 60.82 82.10 Agent Decision Making Single-asset trading 45.50 45.50 43.50 67.67

ness and versatility in few-shot settings. In the NER task, FinLLaMA achieves a remarkable F1 score of 82.10, significantly surpassing its backbone model LLaMA3-8B (39.18), LLaMA3.18B (49.04), and BloombergGPT (60.82). This highlights the substantial improvement in entity recognition due to continual pre-training. For the FPB dataset, FinLLaMA achieves an F1 score of 70.25 in few-shot settings, outperforming LLaMA3-8B (69.65), LLaMA3.1-8B (13.08), and BloombergGPT (51.07). Similarly, in the FiQA-SA dataset, FinLLaMA scored 75.34, surpassing LLaMA3-8B (52.29), LLaMA3.18B (65.39), and BloombergGPT (75.05). In the Headlines dataset for classification, FinLLaMA achieves a score of 85.54, outperforming LLaMA3-8B (80.59), LLaMA3.1-8B (59.95), and BloombergGPT (82.20). These results demonstrate FinLLaMA’s strong ability to classify financial texts accurately with minimal examples.

Agent We further assess the performance of FinLLaMA on a financial-specific task (DecisionMaking) using the FinMem agent framework (Yu et al., 2024), tested on a single-asset trading dataset comprising multi-sourced financial data from August 15, 2021, to April 25, 2023. The results are presented in Table 12. This task assesses the LLM’s proficiency in single-asset trading, with Cumulative Return and Sharpe Ratio as the key performance metrics. For more tasks and datasets, please refer to Appendix F. As shown in Table 12, FinLLaMA outperforms other LLMs with positive Cumulative Return and Sharpe Ratio metrics, demonstrating

profitability in dynamic trading environments. It achieves the highest Sharpe Ratio (SR) of over 1, indicating a superior risk-return balance. Additionally, FinLLaMA maintains investment stability with an Annual Volatility of 0.4766 and a Maximum Drawdown of 0.2693, both lower than other models. This combination of high Sharpe Ratio and low volatility highlights FinLLaMA’s ability to deliver consistent returns with minimized risk, making it highly reliable for trading strategies. These results highlight the significant impact of continual pre-training in enhancing FinLLaMA’s performance. For detailed results, please refer to Appendix G.

### 4.2 Performance of FinLLaMA-Instruct

Table 10 provides detailed information on the datasets and tasks used for the evaluation of instruction fine-tuned models. We align our evaluations with Fintral’s (Bhatia et al., 2024) settings for consistency and comparability, reporting the average performance across datasets in each task. The evaluation includes 6 tasks and 15 datasets used in the FinBEN paper (Xie et al., 2024a): (1) sentiment analysis (SA) task, using FiQA-SA, FOMC, FPB, and Headlines datasets; (2) named entity recognition (NER) task, using Finer-Ord and NER datasets; (3) number understanding (NU) task, using ConvFinQA and FinQA datasets; (4) text summarization (TS) task, using ECTSUM and EDTSUM datasets; (5) stock movement prediction (SMP) task, using ACL18, BigData22, and CIKM18 datasets; (6) credit scoring (CS) task, us-

ing Australia and German datasets.

Overall, Table 7 shows that FinLLaMA-Instruct outperforms other financial LLMs on 4 out of 6 financial tasks, including surpassing GPT-4 on

- 3 tasks, underscoring its effectiveness and applicability in the financial domain. FinLLaMAInstruct also exceeds the performance of all other specialized financial LLMs, including PalmyraFin-70B-32, which is significantly larger, in 4 out of 6 financial tasks. In the numerical understanding task, FinLLaMA-Instruct achieves the best performance with an average accuracy score of 0.69, even outperforming GPT-4. This highlights the effectiveness of our instruction tuning using large-scale math reasoning data in enhancing the model’s numeric understanding ability. Furthermore, FinLLaMA-Instruct consistently outperforms Mistral-7B-Instruct on all tasks and surpasses ChatGPT in five out of six tasks, with text summarization being the exception. This underscores its superior performance compared to general LLMs. FinLLaMA-Instruct achieves better performance compared with GPT-4 on three key financial analysis tasks, demonstrating the robustness of the FinLLaMA backbone model and the effectiveness of our fine-tuning approach and datasets.

- Table 6: Performance of FinLLaVA and baseline models on zero-shot multi-modal benchmark evaluations. Asterisks (*) indicate results on the MMMU test dataset (Yue et al., 2024). Daggers (†) indicate results on our own benchmarks.

Method Backbone

MMMUOverall*

MMMUBussniess*

ChartBench† TableBench†

Closed Source

Gemini-1.5-pro - 49.3 49.8 61.4 58.2 GPT-4o - 55.7 64.3 66.3 66.7 Qwen-VL-MAX - 46.8 39.8 56.0 55.4

Open Source

- LLaVA-1.5 Vicuna-7B 32.0 26.3 43.4 56.0

- LLaVA-1.5 Vicuna-13B 33.6 29.0 49.1 69.1
- LLaVA-1.6 Vicuna-7B 32.3 25.6 41.7 42.9

- LLaVA-1.6 Vicuna-13B 34.0 29.1 50.3 59.3

Deepseek-VL-7B-Chat DeepSeek-LLM-7B-Base 34.2 28.6 51.4 57.3 Qwen-VL-Chat Qwen-7B 32.0 26.2 52.6 48.2 FinLLaVA FinLLaMA 36.3 30.7 52.9 72.4

- Table 7: Performance of FinLLaMA-Instruct and baselines on benchmark tasks (0-1 range). Best performances are bolded, and second-best are underlined. FinTral results are cited from its paper due to unavailable evaluation code or methodology.

Model SA NER NU TS SMP CS

Mistral-7B-Instruct-v0.11(Jiang et al., 2023) 0.49 0.00 0.00 0.30 0.49 0.48 ChatGPT (gpt-3.5-turbo) 0.70 0.53 0.58 0.59 0.53 0.31 GPT-4 (gpt-4-0613) (OpenAI, 2023) 0.79 0.80 0.63 0.65 0.54 0.70 Fintral (Bhatia et al., 2024) 0.81 0.40 0.02 0.40 0.53 0.61 Palmyra-Fin-70B-32K2 0.69 0.08 0.21 0.07 0.54 0.53 FinMA-7B-full3(Xie et al., 2023b) 0.78 0.35 0.12 0.35 0.51 0.29 FinLLaMA-instruct 0.82 0.57 0.69 0.37 0.56 0.56

- 1 https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
- 2 https://huggingface.co/Writer/Palmyra-Fin-70B-32K
- 3 https://huggingface.co/TheFinAI/finma-7b-full

### 4.3 Performance of FinLLaVA

Multimodal Tasks We evaluate our model on four multimodal understanding tasks as shown in Table 11. The MMMU-Overall dataset, with 10,500 instances, assesses general multimodal capabilities, while the MMMU-Business dataset, with 1,428 instances, evaluates performance in financial domains such as accounting and marketing. In addition to existing datasets, in this paper, we build two new financial multi-modal evaluation tasks. First, ChartBench tests chart interpretation skills with 350 financially relevant instances selected using the ChartInstructionData. We employed GPT-4o to assign a finance-relevance score, selecting instances with scores of 9 or above (out of 10). These were categorized into seven groups, as detailed in Appendix E.3, with 50 randomly chosen instances per category forming the ChartBench benchmark. The TableBench assesses multimodal capabilities using tabular images, offering a realistic testbed for handling complex financial data. Our dataset includes 450 questions split between comparison and data retrieval tasks, essential for extracting data points and comparing metrics, reflecting key decision-making processes in finance.

Performance As shown in Table 6, our multimodal model, FinLLaVA, our multimodal model, FinLLaVA, achieves the best performance across all tasks among open-source models with 7B and 13B model sizes. FinLLaVA even outperforms larger models like LLaVA-1.5 and LLaVA-1.6, both of which use Vicuna-13B. On TableBench, FinLLaVA achieves the best performance and outperforms SOTA commercialized LLMs GPT-4 and Gemini-1.5-pro, which proves the effectiveness of our multimodal extension. These results highlight the robustness and promising performance of FinLLaVA On the TableBench dataset, FinLLaVA not only achieves the best performance but also surpasses state-of-the-art commercialized LLMs like GPT-4 and Gemini-1.5-pro. This success demonstrates the effectiveness of our multimodal extension in enhancing the model’s capability to process and analyze complex financial data. These results highlight its potential for widespread application in the financial domain, offering an efficient solution for interpreting and managing multimodal financial data. Our evaluation differs from FinTral’s in using out-of-domain data, showcasing our model’s robustness and superior performance. We enhance its ability to interpret financial tables by integrat-

ing extensive OCR data during alignment. With entirely image-based input, users only need to provide table or chart images, making it convenient for real-world applications like financial reporting and auditing. This simplicity allows financial professionals to efficiently use our model, streamlining their workflow and improving productivity without requiring extensive technical knowledge.

## 5 Conclusion

In this paper, we present Open-FinLLMs, an innovative suite of open-source financial language models specifically designed to address the limitations of LLMs in financial applications. Our contributions include FinLLaMA, a foundational model built on the continual pre-training of LLaMA3 8B with an extensive multimodal domainspecific datasets, FinLLaMA-Instruct, fine-tuned with diverse instructions for improved instructionfollowing and conversational capabilities, and FinLLaVA, the multimodal extension capable of handling vision, tabular, and charts. Our comprehensive evaluation, which covers 14 financial tasks with 30 datasets and 4 multimodal tasks, demonstrates that Open-FinLLMs models outperform existing financial LLMs and even better performance to GPT-4 and GPT-4o despite their smaller size, on financial NLP, reasoning, decision-making and multimodal tasks in zero-shot, few-shot, and supervised fine-tuning settings. The results underscore the potential of Open-FinLLMs in advancing Financial AI applications through their robust performance in various financial tasks.

## Limitation

Our Open-FinLLMs, including the foundational FinLLaMA, the instruction-tuned FinLLaMAInstruct, and the multimodal FinLLaVA, demonstrate promising capabilities but have several limitations. First, the models are currently limited to a size of 8B parameters, and future work should explore both smaller models for efficiency and larger models for enhanced performance. Second, our experiments focused exclusively on English, highlighting the need to expand to multilingual settings to better serve global financial markets. Third, the tasks we addressed, such as trading-related scenarios, represent only a subset of potential applications, and future research should investigate broader industrial use cases like financial auditing, risk management, and regulatory compliance.

Finally, the multimodal capabilities of FinLLaVA are restricted to handling charts and tabular data; incorporating other data types.

## Ethical Statement

The development and dissemination of the Open FinLLMs by the authors carry full responsibility for any potential violation of rights or arising legal issues. All raw data we used are publicly available and do not contain any personal information. Diligent efforts have been undertaken to ensure the construction of the Open FinLLMs respects privacy and conforms to established ethical guidelines. The datasets compiled within Open FinLLMs are shared under the MIT license, with the expectation that users agree to adhere to its conditions.

This manuscript, including any associated large language models, source codes, datasets, and appendices ("Material"), is intended solely for academic research and educational purposes. Users must recognize that the Material is not designed to provide financial, legal, medical, or other professional advice, nor should it serve as a substitute for expert judgment or decision-making in any domain.

While the authors have taken reasonable measures to ensure the accuracy and reliability of the Material, no guarantees—explicit or implied—are provided regarding its completeness, accuracy, or fitness for specific purposes. The authors and their affiliated institutions disclaim responsibility for any outcomes, whether direct or indirect, arising from the use, misuse, or reliance upon the Material. Users are strongly advised to consult relevant professionals when making decisions based on outputs generated by this Material.

By accessing or utilizing this Material, individuals agree to indemnify, defend, and hold harmless the authors, as well as their affiliated organizations, from any claims, liabilities, or damages resulting from the use or interpretation of the Material, including outputs produced by the large language model. Users bear the responsibility for ensuring the appropriate and ethical use of this Material in accordance with applicable laws and regulations.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Julio Cesar Salinas Alvarado, Karin Verspoor, and Timothy Baldwin. 2015. Domain adaption of named entity recognition to support credit risk assessment. In Proceedings of the Australasian Language Technology Association Workshop 2015, pages 84– 90.

Gagan Bhatia, El Moatez Billah Nagoudi, Hasan Cavusoglu, and Muhammad Abdul-Mageed. 2024. Fintral: A family of gpt-4 level multimodal financial large language models. arXiv preprint arXiv:2402.10986.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, and 1 others. 2023. Sparks of artificial general intelligence: Early experiments with GPT-4. arXiv preprint arXiv:2303.12712.

Chung-Chi Chen, Chin-Yi Lin, Chr-Jr Chiu, Hen-Hsen Huang, Alaa Alhamzeh, Yu-Lieh Huang, Hiroya Takamura, and Hsin-Hsi Chen. 2023. Overview of the ntcir-17 finarg-1 task: Fine-grained argument understanding in financial analysis. In Proceedings of the 17th NTCIR Conference on Evaluation of Information Access Technologies, Tokyo, Japan, pages 12–15.

Daoyuan Chen, Yilun Huang, Zhijian Ma, Hesen Chen, Xuchen Pan, Ce Ge, Dawei Gao, Yuexiang Xie, Zhaoyang Liu, Jinyang Gao, Yaliang Li, Bolin Ding, and Jingren Zhou. 2024a. Data-juicer: A one-stop data processing system for large language models. In International Conference on Management of Data.

Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. 2024b. ALLaVA: Harnessing GPT4V-synthesized data for a lite vision-language model. Preprint, arXiv:2402.11684.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021. Finqa: A dataset of numerical reasoning over financial data. Proceedings of EMNLP 2021.

Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering. Preprint, arXiv:2210.03849.

Keith Cortis, André Freitas, Tobias Daudert, Manuela Huerlimann, Manel Zarrouk, Siegfried Handschuh, and Brian Davis. 2017. Semeval-2017 task 5: Finegrained sentiment analysis on financial microblogs

and news. In Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017), pages 519–535.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2024. Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36.

Duanyu Feng, Yongfu Dai, Jimin Huang, Yifang Zhang, Qianqian Xie, Weiguang Han, Zhengyu Chen, Alejandro Lopez-Lira, and Hao Wang. 2024. Empowering many, biasing a few: Generalist credit scoring through large language models. Preprint, arXiv:2310.00566.

Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats Leon Richter, Quentin Gregory Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. Continual pre-training of large language models: How to re-warm your model? In Workshop on Efficient Systems for Foundation Models@ ICML2023.

Hans Hofmann. 1994. Statlog (German Credit Data). UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5NC77.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, and 1 others. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024b. Visual instruction tuning. Advances in Neural Information Processing Systems, 36.

Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, and Daochen Zha. 2023. Data-centric FinGPT: Democratizing internet-scale data for financial large language models. Workshop on Instruction Tuning and Instruction Following, NeurIPS.

Xiao-Yang Liu, Jie Zhang, Guoxuan Wang, Weiqing Tong, and Anwar Walid. 2024c. FinGPT-HPC: Efficient pretraining and finetuning large language models for financial applications with high-performance computing. arXiv preprint arXiv:2402.13533.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations.

Macedo Maia, Siegfried Handschuh, Andre Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: Financial opinion mining and question answering.

Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala. 2014. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4):782–796.

Dominique Mariko, Hanna Abi Akl, Estelle Labidurie, Stephane Durfort, Hugues De Mazancourt, and Mahmoud El-Haj. 2020. Financial document causality detection shared task (fincausal 2020). arXiv preprint arXiv:2012.02505.

Ahmed Masry, Parsa Kavehzadeh, Enamul Hoque, Shafiq Joty, and 1 others. 2023. Unichart: A universal vision-language pretrained model for chart comprehension and reasoning. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263– 2279, Dublin, Ireland. Association for Computational Linguistics.

Rajdeep Mukherjee, Abhinav Bohra, Akash Banerjee, Soumya Sharma, Manjunath Hegde, Afreen Shaikh, Shivani Shrivastava, Koustuv Dasgupta, Niloy Ganguly, Saptarshi Ghosh, and 1 others. 2022. Ectsum: A new benchmark dataset for bullet point summarization of long earnings call transcripts. arXiv preprint arXiv:2210.12467.

Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, and Peter Staar. 2022. Tableformer: Table structure understanding with transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4614–4623.

Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M Mulvey, H Vincent Poor, Qingsong Wen, and Stefan Zohren. 2024. A survey of large language models for financial applications: Progress, prospects and challenges. arXiv preprint arXiv:2406.11903.

Jason Obeid and Enamul Hoque. 2020. Chart-to-text: Generating natural language descriptions for charts by adapting the transformer model. In International Conference on Natural Language Generation, pages 138–147.

OpenAI. 2023. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Guilherme Penedo, Hynek Kydlíˇcek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, and 1 others. 2024. The FineWeb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557.

Lingfei Qian, Weipeng Zhou, Yan Wang, Xueqing Peng, Jimin Huang, and Qianqian Xie. 2025. Fino1: On the transferability of reasoning enhanced llms to finance. arXiv preprint arXiv:2502.08127.

Ross Quinlan. Statlog (Australian Credit Approval). UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C59012.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. Preprint, arXiv:2103.00020.

Aman Rangapur, Haoran Wang, and Kai Shu. 2023. Fin-fact: A benchmark dataset for multimodal financial fact checking and explanation generation. arXiv preprint arXiv:2309.08793.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, and 1 others. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950.

Agam Shah, Suvan Paturi, and Sudheer Chava. 2023a. Trillion dollar words: A new financial dataset, task & market analysis. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6664– 6679, Toronto, Canada. Association for Computational Linguistics.

Agam Shah, Ruchit Vithani, Abhinav Gullapalli, and Sudheer Chava. 2023b. Finer: Financial named entity recognition dataset and weak-supervision model. arXiv preprint arXiv:2302.11157.

Ankur Sinha and Tanmay Khandait. 2021. Impact of news on the commodity market: Dataset and results. In Advances in Information and Communication: Proceedings of the 2021 Future of Information and Communication Conference (FICC), Volume 2, pages 589–601. Springer.

Yejun Soun, Jaemin Yoo, Minyong Cho, Jihyeong Jeon, and U Kang. 2022. Accurate stock movement prediction with self-supervised learning from sparse noisy tweets. In 2022 IEEE International Conference on Big Data (Big Data), pages 1691–1700. IEEE.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, and 1 others. 2023b. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288.

Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. 2023. To see is to believe: Prompting gpt-4v for better visual instruction tuning. Preprint, arXiv:2311.07574.

Huizhe Wu, Wei Zhang, Weiwei Shen, and Jun Wang. 2018. Hybrid deep sequential modeling for social text-driven stock prediction. In Proceedings of the 27th ACM international conference on information and knowledge management, pages 1627–1630.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023a. Bloomberggpt: A large language model for finance. Preprint, arXiv:2303.17564.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023b. BloombergGPT: A large language model for finance. Preprint, arXiv:2303.17564.

Tongtong Wu, Linhao Luo, Yuan-Fang Li, Shirui Pan, Thuy-Trang Vu, and Gholamreza Haffari. 2024. Continual learning for large language models: A survey. arXiv preprint arXiv:2402.01364.

Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, and 1 others. 2024a. The finben: An holistic financial benchmark for large language models. arXiv preprint arXiv:2402.12659.

Qianqian Xie, Weiguang Han, Yanzhao Lai, Min Peng, and Jimin Huang. 2023a. The wall street neophyte: A zero-shot analysis of chatgpt over multimodal stock movement prediction challenges. Preprint, arXiv:2304.05351.

Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023b. Pixiu: A large language model, instruction data and evaluation benchmark for finance. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 33469–33484.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. 2024b. Doremi: Optimizing data mixtures speeds up language model pretraining. Advances in Neural Information Processing Systems, 36.

Yumo Xu and Shay B Cohen. 2018. Stock movement prediction from tweets and historical prices. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1970–1979.

Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. 2023. FinGPT: Open-source financial large language models. FinLLM at IJCAI.

Linyi Yang, Eoin M Kenny, Tin Lok James Ng, Yi Yang, Barry Smyth, and Ruihai Dong. 2020. Generating plausible counterfactual explanations for deep transformers in financial text classification. arXiv preprint arXiv:2010.12512.

Yangyang Yu, Haohang Li, Zhi Chen, Yuechen Jiang, Yang Li, Denghui Zhang, Rong Liu, Jordan W Suchow, and Khaldoun Khashanah. 2024. FinMem: A performance-enhanced llm trading agent with layered memory and character design. In AAAI Symposium Series, volume 3, pages 595–597.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and 3 others. 2024. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653.

Yilun Zhao, Hongjun Liu, Yitao Long, Rui Zhang, Chen Zhao, and Arman Cohan. 2023a. Knowledgemath: Knowledge-intensive math word problem solving in finance domains. arXiv preprint arXiv:2311.09797.

Yilun Zhao, Yitao Long, Hongjun Liu, Linyong Nan, Lyuhao Chen, Ryo Kamoi, Yixin Liu, Xiangru Tang, Rui Zhang, and Arman Cohan. 2023b. Docmath-eval: Evaluating numerical reasoning capabilities of llms in understanding long documents with tabular data. arXiv preprint arXiv:2311.09805.

Zhihan Zhou, Liqian Ma, and Han Liu. 2021. Trade the event: Corporate events detection for news-based event-driven trading. arXiv preprint arXiv:2105.12825.

## A Curation of Continual Pre-training Corpus

### A.1 Dataset Details

Our continual pre-training corpus is designed to ensure comprehensive coverage of financial knowledge by integrating a diverse range of data sources. This appendix provides a detailed overview of each data source:

- • Financial Papers: This subset includes 4 billion tokens extracted from academic papers and research articles, offering a strong foundation in financial concepts and theories. The papers span a period from 2000 to 2023. These documents are sourced from SSRN and open-source conference proceedings, providing in-depth insights into both foundational and cutting-edge financial research.
- • Conference Calls: Comprising 5 billion tokens, this dataset includes open-source transcripts from earnings calls, analyst meetings, and investor briefings, collected from 09/08/2004 to 12/17/2021. These transcripts provide real-time insights into corporate performance and strategic directions, allowing for a nuanced understanding of company operations and market positioning. Sources include major corporations across various industries, reflecting a diverse set of perspectives and strategies.
- • Financial Reports: This component consists of 5 billion tokens from annual and quarterly reports, covering the period from 2005 to

2020. These reports are crucial for assessing a company’s financial health, market positioning, and strategic outlook. They include balance sheets, income statements, and management discussions, providing a comprehensive view of corporate financial performance.

- • Technical Indicators: With 12 billion tokens, this dataset includes open-source economic indicators and financial ratios sourced from company filings and open market data, spanning from 2009 to 2023. These indicators are essential for macroeconomic analysis and investment decision-making, covering metrics such as GDP, inflation rates, interest rates, and key financial ratios.

- • News and Social Media: This subset includes 7 billion tokens from financial news outlets and social media platforms, collected from 1999 to 2021. This data provides timely updates on market trends, public opinion, and emerging issues, reflecting the dynamic nature of financial markets. Sources include leading financial news websites, and financial forums, capturing both traditional media and real-time public sentiment.
- • Historical Data: Encompassing 13 billion tokens, this dataset includes historical stock prices, trading volumes, and market data from 1999 to 2022. This data is vital for quantitative analysis and algorithmic trading, providing historical context and trend analysis capabilities. The data is sourced from Yahoo Finance, offering a robust foundation for time-series analysis and predictive modeling.
- • SEC Filings: This section includes 6 billion tokens from U.S. SEC filings, such as 10-K and 10-Q reports, spanning from 1994 to 2020. These filings provide comprehensive insights into corporate activities, financial conditions, and risk factors. They are sourced from the U.S. Securities and Exchange Commission’s EDGAR database, ensuring official and up-todate corporate information.

### A.1.1 Data Processing & Cleaning

In our data preprocessing pipeline for training Open-FinLLMs, we utilize Data-Juicer (Chen et al., 2024a) to clean and standardize the datasets. For each corpus, we remove email addresses and URLs to enhance privacy and reduce noise, ensuring the focus remains on the textual content. We address unicode inconsistencies by standardizing characters across the dataset, which maintains uniformity and aids in accurate text representation. Punctuation is normalized to provide consistency in text parsing, while excess whitespace is removed to improve readability and structure. For tabular and time-series data, we first split them into rows into samples of approximately 2,048 tokens each, formatting each block in HTML and ensuring each includes the table header for context. We then combine all datasets and further chunk the entire dataset into 8,192 token blocks, readying the data for efficient processing by the model.

### A.2 Tabular and Time-series Data Format

|<table> <thead> <tr><th>Column Header 1</th><th>Column Header 2</th></tr> </thead> <tbody> <tr><td>Data Row 1, Cell 1</td><td>Data Row 1, Cell 2</td></tr> </tbody> </table>|
|---|

## B Financial Instruction Dataset

### B.1 Data Sources

Our financial instruction dataset is a compilation of diverse and specialized datasets designed to enhance the capabilities of Open-FinLLMs. Below, we provide detailed descriptions of each dataset and the specific tasks they cover:

- • ChanceFocus/FLUPE (Xie et al., 2023b): The FLUPE dataset, is instrumental in improving financial natural language processing capabilities. It includes tasks such as financial sentiment analysis, news headline classification, and named entity recognition (NER). These tasks involve analyzing financial texts to identify sentiment, classify financial headlines, and recognize entities within financial documents. By providing diverse examples, FLUPE helps models refine their understanding of financial language and context, supporting improved task performance across various financial NLP applications. FinGPT/Fingptfinred (Liu et al., 2023): This dataset focuses on financial report and document comprehension, with approximately 32.67k examples. It is designed to enhance the model’s ability to interpret complex financial documents. The dataset enables models to improve their analytical skills and decision-making abilities based on comprehensive document analysis.
- • TIGER-Lab/MathInstruct (Yue et al., 2023): Comprising 262k examples, the MathInstruct dataset is built from 13 distinct mathematical rationale datasets. It includes tasks such as arithmetic operations, algebraic reasoning, probability calculations, statistical analysis, and calculus-based problem solving. The dataset employs methods like chain-ofthought (CoT) and program-of-thought (PoT) rationales to provide intermediate reasoning capabilities across these mathematical fields.

This is crucial for financial tasks that require precise calculations and quantitative insights, enabling models to tackle mathematical problems effectively within financial contexts.

• sujet-ai/Sujet-Finance-Instruct-177k5: The Sujet-Finance-Instruct-177k dataset is a comprehensive collection of financial textual data, designed for fine-tuning language learning models for specialized financial tasks. It integrates data from 18 different datasets, providing a total of 177,597 entries. The dataset covers a wide range of financial tasks, including:

- – Sentiment Analysis: 44,209 entries focused on categorizing financial texts into sentiments such as positive, negative, neutral, bearish, or bullish.
- – Question Answering (QA): 38,801 entries for direct-answer financial questions that do not require additional context.
- – QA with Context: 40,475 entries where financial questions require contextual understanding for accurate answers.
- – QA Conversation: 15,613 entries involving conversational interactions between a user and an LLM assistant.
- – Yes/No Questions: 20,547 entries focused on questions necessitating a simple yes or no answer.
- – Topic Classification: 16,990 entries for classifying financial texts into specific finance-related categories.
- – NER Sentiment Analysis: 962 entries for conducting sentiment analysis at the entity level within texts.

### B.2 Data Processing

In our instruction dataset, we identified overlapping task samples between the FLUPE and Sujet-Finance-Instruct-177k datasets, particularly in tasks such as sentiment analysis and NER. To address this, we manually excluded these redundant samples, which resulted in the removal of approximately 30,000 samples. This step was crucial to ensure that each task is represented uniquely and effectively in the dataset, avoiding any biases that could arise from duplicated entries.

5https://huggingface.co/datasets/sujet-ai/ Sujet-Finance-Instruct-177k

## C Multimodal Instruction Data

### C.1 Table

Our table data is sourced from the Fintabnet and Marketing categories of SynthTabNet, featuring real financial and marketing tables with diverse layouts. Each table includes parsed bounding boxes, allowing us to reconstruct structure-aware prompts, which are more accurate than image-based descriptions.

To ensure OCR quality, we limit tables to a maximum size of 10 × 10 to avoid resolution-related cell blurring. In the SFT stage, we design seven financial-specific tasks (Appendix E.3), randomly sampled in the prompts. Figure 2 shows an example table and how we align and generate SFT data.

### C.2 Chart

Our chart dataset is derived from Unichart, Chart2Text, and ChartQA, covering real financial data, marketing trends, and varied visual styles. We focus on numerical and financial charts to support robust quantitative analysis. An example questionanswer pair is shown in Figure 3.

During SFT, we develop seven chart-specific tasks (Appendix E.3) to enhance the model’s ability to interpret financial charts and extract insights. Figure 4 illustrates an example chart used in SFT data generation.

|You are a data analyst reviewing a table from a financial report. Your task is to understand the data and its location in the table. Based on the dataset’s table structure and content, interpret what the table represents.<br><br>Cell Information: Each cell in the dataset is described with four main attributes:<br><br>- bbox: Bounding box coordinates indicating the position of the cell within the table.<br>- tokens: The actual content of the cell, which might include text or numerical data.<br>- is_header: A boolean value indicating whether the cell is a header cell or not.<br>- span: Additional information about the cell’s span, such as colspan and rowspan. Details of the cells: {cells_str} Tasks for Pretraining Data:<br>- Examine the table’s content to understand what information it conveys and how it is structured.<br>- Generate a descriptive question that encourages a deep dive into the table’s displayed data and its organizational framework.<br>- Avoid using HTML tags in your response.<br>- Ensure your response contains numerical data.<br>- Provide detailed information about the table’s content and structure, highlighting specific values from the table (excluding headers).<br>- Provide spatial information about the cells in this table without including any bounding box information. For example, you can describe the column or row layout. Tasks for SFT Data:<br>- Understand the content of the table, including the spatial position of each cell, numerical information, headers, etc.<br>- Based on the specific content of the table, propose a question related to {selected_task} for a deeper analysis of the table.<br>- If the table content is related to data, analyze the data and use the numerical values provided to support your analysis.<br>- If you don’t know the answer to the question, you should further analyze the table to provide a more detailed examination of its content. Note: You should not create or invent any numerical values; you should only analyze using the values provided in the table.<br>|
|---|

|Based on the provided chart image, generate a detailed caption or description that thoroughly explains the chart’s content, including its numerical data and organizational structure.<br><br>Tasks for Pretraining Data:<br><br>- Examine the chart’s content to understand what information it conveys and how it is structured.<br>- Generate a comprehensive description or caption that covers all aspects of the chart’s numerical data and organizational framework. Random Questions for Pretraining Data:<br><br><br>• What key information does the chart present?<br>• Can you detail the main data points covered in the chart?<br>• What topics or categories are addressed within the chart?<br>• Identify the major types of data provided by the chart.<br>• What essential insights does the chart offer about its subject matter?<br>• What are the primary themes or elements highlighted in the chart?<br>• Describe the type of information that the chart provides.<br>• What are the main conclusions or facts presented in the chart?<br>• Outline the core pieces of information contained in the chart.<br>• Summarize the main subjects or topics included in the chart.<br>• What does the chart specifically focus on in its data presentation?<br>|
|---|

- Figure 2: Example table image. Description: Q: What financial metrics are represented in the table, and how are they organized? A: The table presents various financial metrics organized into rows and columns, with headers indicating the type of data. The first column lists categories such as “Corporate”, “For the year ended December 31”, and “General and administrative expenses”. The corresponding numerical values in the adjacent columns include 628,856, 518,246, 222,216, and 793,595, respectively. The data appears to be structured in a way that highlights different financial activities and their associated values, with totals and segment information also included. For instance, the “Total segment” row shows a value of 886,238, while “Cash and cash equivalents at beginning of period” has a value of 946,929. The table also indicates that the data is presented in millions of dollars, emphasizing the scale of the financial figures.

Instruction (Make Comparisons): Q: Which segment had the highest “Total net revenues” and what was the amount? A: The segment with the highest “Total net revenues” is “Millions of dollars” with an amount of 972,582.

[Figure 2]

|Your task is to analyze a financial chart represented in an image. Follow these detailed steps to understand the data and its location in the chart, and generate a high-quality SFT dataset. Tasks for SFT Data:<br><br>1. Interpret the Chart’s Content:<br><br>• Identify the spatial position of each cell in the chart.<br>• Only extract and note the numerical information which is 100% convincing to you from the chart.<br>• Identify and record headers and labels from the chart.<br><br><br>2. Formulate a Question for Deeper Analysis:<br><br>• Based on the specific content of the chart, propose a question related to {selected_task}.<br><br>3. Analyze the Data:<br><br>• If the chart content includes data, analyze it.<br>• Use the numerical values provided in the chart to support your analysis.<br><br><br>4. Further Examination:<br><br>• If you are unsure of the answer to the question, perform a deeper analysis of the table to provide a more detailed examination of its content.<br><br>Note: Do not create or invent any numerical values; only use the values provided in the table. Example Chart Analysis Workflow:<br><br>1. Interpret the Chart’s Content:<br><br>• Examine the image to identify the layout of the chart, such as rows and columns.<br>• Record numerical values and their corresponding labels or headers.<br><br><br>2. Formulate a Question for Deeper Analysis:<br><br>• Example: “What is the trend of quarterly sales over the past year?”<br><br>3. Analyze the Data:<br><br>• Example: “The quarterly sales over the past year show an increasing trend, starting from $50,000 in Q1 to $70,000 in Q4. This indicates consistent growth in sales, with Q2 and Q3 showing moderate increases to $55,000 and $60,000 respectively.”<br><br>4. Further Examination:<br><br>• If further detail is required, delve deeper into the data and provide more specific insights.|
|---|

- Figure 3: Example chart image 1. Description: Q: Describe the type of information that the chart provides. A: The chart shows year-over-year growth rates for various countries in 2014 and 2015. Each country has two bars representing growth rates for 2014 (in dark blue) and 2015 (in light blue). The growth rates are measured in percentages and are displayed on the horizontal axis, which ranges from -7.5% to 20%. The countries listed on the left axis include China, Russia, Germany, India, South Korea, United States, Brazil, France, Australia, Canada, United Kingdom, Spain, Netherlands, Italy, and Japan. Key information includes: China’s growth rates were 17.8% in 2014 and 16.5% in 2015. Russia had a negative growth rate of -5.4% in 2014 and a positive growth rate of 13.6% in 2015. Germany’s growth rates were 7% in 2014 and 10.6% in 2015. India’s growth rates were 2.1% in 2014 and 7.6% in 2015. South Korea had growth rates of 5.9% in 2014 and 5.8% in 2015. The United States had growth rates of 5.9% in 2014 and 6.8% in 2015. Brazil had growth rates of 5.9% in 2014 and 12% in 2015.

[Figure 3]

- Figure 4: Example chart image 2. Instruction (Make Comparisons): Q: How has the unemployment rate in the given chart changed from 1999 to 2020? A: The unemployment rate in the chart shows significant fluctuations between 1999 and 2020. Starting at 9.64% in 1999, it slightly increased to 9.65% in 2000. There was a sharp decline to 6.81% in 2003 and further to 4.83% in

2005. The rate then fluctuated, peaking at 5.46% in 2009 before decreasing to 4.09% in 2011. It rose again to 5.26% in 2016 and slightly varied around 4.62% in 2018. By 2020, the rate was at 4.81%. Overall, the chart indicates a general downward trend in the unemployment rate over the two decades with occasional increases.

[Figure 4]

Table 8: Datasets used for zero-shot evaluation tasks. The dataset with * is newly created in this paper. Abbreviations: EM Accuracy means exact match accuracy. MCC means Matthews correlation coefficient.

Task Dataset Test Size Metrics License

Sentiment analysis TSA (Cortis et al., 2017) 561 F1 CC BY-NC-SA 4.0 Classification FOMC (Shah et al., 2023a) 496 F1 CC BY-NC 4.0 Classification FinArg-AUC (Chen et al., 2023) 969 F1 CC BY-NC-SA 4.0 Classification MA (Yang et al., 2020) 500 F1 Public Classification SC (Mariko et al., 2020) 8,630 Entity F1 CC BY 4.0 Misinformation FinFact (Rangapur et al., 2023) 3,369 Weighted F1 Public Math *MC 50 F1 Public Math KnowledgeMath (Zhao et al., 2023a) 1000 EM Accuracy MIT License Math DocMath-Eval (Zhao et al., 2023b) 3200 EM Accuracy MIT License Credit scoring German (Hofmann, 1994) 1,000 MCC CC BY 4.0 Credit scoring Australian (Quinlan) 690 MCC CC BY 4.0 Credit scoring LendingClub (Feng et al., 2024) 2,690 MCC CC0 1.0 Fraud detection ccf (Feng et al., 2024) 2,278 MCC DbCL v1.0 Fraud detection ccfraud (Feng et al., 2024) 2,097 MCC Public Distress polish (Feng et al., 2024) 1,736 MCC CC BY 4.0 Distress taiwan (Feng et al., 2024) 1,364 MCC CC BY 4.0 Claim analysis ProtoSeguro (Feng et al., 2024) 2,381 MCC Public Claim analysis travelinsurance (Feng et al., 2024) 3,800 MCC ODbL v1.0 Question answering ConvFinQA (Chen et al., 2022) 1,490 EM Accuracy MIT License

## D Prompts for Generating Finance-Relevance Score

|You are a financial analyst specialized in evaluating the relevance of a conversation to the financial domain. Your goal is to assess how closely the conversation content pertains to financial topics.<br><br>Conversation: Human: {conversation[0][“value”]} GPT: {conversation[1][“value”]} Tasks:<br><br>You need to complete the following tasks: - Review the conversation between the human and the GPT model, and determine the relevance of the conversation to financial matters. - Provide a relevance score between 0 and 10, where 0 indicates no relevance and 10 indicates high relevance to finance. - Ensure the result is formatted correctly in JSON for further analysis.<br><br>Output Format: {“relevance_score": “<Insert relevance score here>"}|
|---|

## E Experiments

### E.1 Datasets used for different experiments

Details of the dataset used for evaluation are listed in Table 8, 9, 10, 11.

Table 9: Datasets used for few-shot evaluation tasks.

Task Dataset Test Size Metrics License

Sentiment analysis FPB (Malo et al., 2014) 970 F1 CC BY-SA 3.0 Sentiment analysis FiQA-SA (Maia et al., 2018) 235 F1 Public Classification Headlines (Sinha and Khandait, 2021) 2,283 Avg F1 CC BY-SA 3.0 Named entity recognition NER (Alvarado et al., 2015) 980 Entity F1 CC BY-SA 3.0

### E.2 Details of Agent evaluation results

Results of different models in agent evaluation setting is lised in E.2.

- Table 10: Datasets used for evaluating instruction finetuned models.

Task Dataset Test size Metrics License

Sentiment analysis

FiQA-SA (Maia et al., 2018) 235

Accuracy

Public

FOMC (Shah et al., 2023a) 496 CC BY-NC 4.0 FPB (Malo et al., 2014) 970 CC BY-SA 3.0 Headlines (Sinha and Khandait, 2021) 2,283 CC BY-SA 3.0

Named entity recognition

Finer-Ord (Shah et al., 2023b) 1,080

Entity-F1

CC BY-NC 4.0 NER (Alvarado et al., 2015) 980 CC BY-SA 3.0

Number understanding

ConvFinQA (Chen et al., 2022) 1,490

EM Accuracy

MIT License FinQA (Chen et al., 2021) 1,147 MIT License

Text summarization

ECTSUM (Mukherjee et al., 2022) 495

Rouge-score

Public EDTSUM (Zhou et al., 2021) 2,000 Public

Stock movement prediction

ACL18 (Xu and Cohen, 2018) 3.720

Accuracy

MIT License

BigData22 (Soun et al., 2022) 1,470 Public CIKM18 (Wu et al., 2018) 1,140 Public

Credit scoring

Australian (Quinlan) 690

Accuracy

CC BY 4.0 German (Hofmann, 1994) 1,000 CC BY 4.0

- Table 11: Datasets used for multimodal evaluation tasks.

Dataset Test Size Metrics License

MMMU (Yue et al., 2024) 10,500 Accuracy Public MMMU-Business (Yue et al., 2024) 1,428 Accuracy Public ChartBench 350 Accuracy our data TableBench 450 Accuracy our data

- Table 12: Trading performance of FinLLaMA and baseline models.

Metric Asset Buy & Hold LLaMA3.1-8B LLaMA3-8B FinLLaMA

Cumulative Return TSLA -0.1633 -0.1829 -0.1552 0.5573 COIN -0.0562 -0.3515 -0.0319 0.1743 GOOG -0.0562 0.1651 0.1631 0.1098 NIO -0.3530 -0.2677 -0.3526 0.4645 Overall -0.1571 -0.1592 -0.0942 0.3265

Sharpe Ratio TSLA -0.4769 -0.5557 -0.8486 2.4532 COIN -0.1041 -0.7911 -0.0666 0.5778 GOOG -0.1041 0.9464 1.0128 0.6927 NIO -0.8678 -0.6951 -1.0313 1.9113 Overall -0.3823 -0.2739 -0.2334 1.4088

Normalized Sharpe Ratio TSLA 42.05 40.72 35.86 90.89 COIN 48.27 36.82 48.89 59.63 GOOG 48.27 82.44 83.88 61.55 NIO 35.54 38.41 32.81 81.86 Overall 43.63 45.43 45.92 73.48

Annual Volatility TSLA 0.7015 0.6741 0.3748 0.4654 COIN 1.1047 0.9103 0.9823 0.6181 GOOG 1.1047 0.3574 0.3300 0.3249 NIO 0.8332 0.7891 0.7005 0.4979 Overall 0.9360 0.6827 0.5969 0.4766

Max Drawdown TSLA 0.5532 0.5248 0.2662 0.1883 COIN 0.6019 0.5449 0.4056 0.4142 GOOG 0.6019 0.1983 0.2092 0.1741 NIO 0.4498 0.4213 0.5172 0.3008 Overall 0.5517 0.4223 0.3496 0.2693

### E.3 Multimodal Task Categories

We chose these seven categories for our evaluations because they represent the most critical and common tasks in financial analysis. By focusing on these areas, we ensure a comprehensive and thorough assessment of our model’s capabilities in handling financial data.

• Make Comparisons: This category involves comparing different financial metrics or data points across various time periods, companies, or financial instruments. For example, comparing quarterly revenues of different companies to determine market performance trends. This is crucial for financial analysts who need

to benchmark performance and identify trends over time. Accurate comparisons help in making informed decisions about investments, cost management, and strategic planning.

- • Find Correlations: This involves identifying relationships between different financial variables. For example, determining if there’s a correlation between interest rates and stock prices, which can help in predictive financial modeling. Understanding correlations is essential for risk management and portfolio diversification, as it allows analysts to predict how changes in one variable might affect another.
- • Data Retrieval: This category focuses on extracting specific data points from financial tables or charts. For example, retrieving the net income values from an annual financial report for analysis. Efficient data retrieval is fundamental for compiling reports, conducting audits, and performing detailed financial analysis. It ensures that all necessary data can be quickly accessed and utilized.
- • Find Extremum: This involves identifying the maximum or minimum values within financial datasets. For example, finding the highest stock price over a given period or the lowest expense in a budget report. Identifying extremum points helps in spotting significant events or trends that might require further investigation or immediate action. This is particularly useful in scenarios like peak revenue analysis or cost-cutting strategies.
- • Find Clusters: This category entails grouping financial data into clusters based on similarities. For example, clustering companies based on similar financial performance indicators like revenue, profit margins, and market share. Clustering helps in market segmentation, identifying peer groups, and understanding competitive positioning. It is valuable for comparative analysis and strategic planning.
- • Characterize Distributions: This involves describing the distribution of financial data points. For example, analyzing the distribution of daily returns of a stock to understand its volatility and risk. Characterizing distributions aids in risk assessment, financial forecasting, and identifying patterns that could

influence decision-making processes. It provides a statistical foundation for understanding variability and risk.

• Find Anomalies: This focuses on detecting outliers or unusual patterns in financial data. For example, identifying unexpected spikes in expenses that could indicate fraud or errors in financial reports. Detecting anomalies is crucial for maintaining the integrity of financial data, preventing fraud, and ensuring accurate financial reporting. It helps in early detection of issues that might otherwise go unnoticed.

E.4 Details of Continual Pretrained Model Evaluation

E.4.1 Descriptions of Zero-shot Evaluation Tasks

- • Sentiment analysis focuses on extracting sentiment information (positive, negative, or neutral) from financial texts, using the TSA dataset.
- • Classfication: 1) Hawkish-Dovish classification aims to classify sentences from monetary policy texts as ’hawkish’ or ’dovish’ focusing on the nuanced language and economic implications of financial texts, using the FOMC dataset. 2) Argument unit classification categorizes sentences as claims or premises, using the FinArg AUC dataset. 3) Deal completeness classification predicts if mergers and acquisitions events are "completed" or remain "rumors" based on news and tweets, employing the MA dataset.
- • Causal Classification discerns whether sentences from financial news and SEC filings convey causality, using the SC dataset.
- • Misinformation detection is formulated as a three-classification task, verifying financial misinformation (True/False/Not Enough Information). The input is textual claim information. The aim is to let the model deliver accurate results, which requires LLMs to identify fraudulent financial content and verify the claim’s authenticity.
- • Mathematical Computation is structured as a generation task, specifically designed to compute financial metrics based solely on questions about company financial statements.

The input for this task consists of 50 questions, each focusing on different financial metrics such as revenue, turnover, and other measurable outcomes. The objective is for the model to generate accurate financial analyses, such as capital expenditures or financial ratios, from the information presented in the questions alone, without direct access to the financial sheets. This task assesses the model’s ability to infer and calculate key financial indicators crucial for evaluating the financial health and performance of the company.

- • KnowledgeMath is formulated as a math word problem-solving task, predicting the value of the final answer. The input is a math word problem in finance domains, the aim is to let the model perform math reasoning to predict the final answer of the math word problem.
- • DocMath-Eval is formulated as the document question answering task, predicting the value of the final answer. The input comprises a financial document and a question, the aim is to let the model perform information extraction and math reasoning to predict the final answer of the question.
- • Credit Scoring is a vital process employed by financial institutions to evaluate a borrower’s creditworthiness. It assesses financial information provided in loan applications to determine eligibility, interest rates, and loan terms to predict credit risk.
- • Fraud Detection is a task closely aligned with credit scoring, focusing on identifying genuine versus fraudulent loan applications. This process is essential for safeguarding financial systems and shielding institutions from financial losses. The datasets of this task are often imbalanced, a characteristic common to fraud detection, with genuine fraud cases constituting a small fraction of total applications.
- • Financial Distress Identification aims to predict the likelihood of a company experiencing bankruptcy, leveraging publicly accessible data. This process is crucial for stakeholders to assess the financial health and stability of a company.

- • Claim Analysis is a critical task for insurance companies, involving the analysis of claims to detect fraudulent activity. Fraudulent claims are illegitimate attempts to obtain payment under false pretenses, while legitimate claims represent valid requests for payment due to losses covered by an insurance policy. This distinction is vital for preventing financial losses due to fraud and ensuring that only rightful claims are reimbursed. Most datasets of this task are often imbalanced, meaning that fraudulent claims are significantly less frequent than legitimate ones, a common scenario in real-world insurance claim analysis.
- • Question Answering focuses on answering financial questions based on the provided information. We use the ConFinQA dataset, which includes multi-turn question-and-answer pairs over earnings reports.

### E.4.2 Descriptions of Few-shot Evaluation

Tasks In our few-shot evaluations, we use three financial NLP tasks:

- • Sentiment Analysis: Extracting sentiment information from financial texts using the FPB and FiQA-SA datasets, which focus on determining sentiment polarity (positive, negative, or neutral) in financial sentences.
- • Classification: Evaluating the model’s capability to classify financial texts. The Headlines dataset is used to classify news headlines related to financial events.
- • Named Entity Recognition: Extracting entities such as persons, organizations, and locations from financial texts. We use the NER dataset with manually annotated four entities for three financial agreements.

For detailed prompts for evaluation datasets, please see Table 13 for 0-shot, and Table 14 for few-shots. The results are listed in Figure 5 and Figure 6.

E.5 Prompts for Multimodal Benchmark Evaluation

In our multimodal benchmark evaluation, we introduce two main stages. First, we generate answers from LLMs using three different templates of prompts. Second, we utilize gpt-4o-mini to extract the correct JSON format from the LLM’s response.

Table 13: 0-shot task datasets prompt overview.

Data Prompt

“Given the following financial text, return a sentiment score for Ashtead as a floating-point number ranging from -1 (indicating a very negative or bearish sentiment) to 1 (indicating a very positive or bullish sentiment), with 0 designating neutral sentiment. Return only the numerical score first, follow it with a brief reasoning behind your score."

TSA

“Examine the excerpt from a central bank’s release below. Classify it as HAWKISH if it advocates for a tightening of monetary policy, DOVISH if it suggests an easing of monetary policy, or NEUTRAL if the stance is unbiased. Your response should return only HAWKISH, DOVISH, or NEUTRAL."

FOMC

“Analyze sentences from earnings conference calls and identify their argumentative function. Each sentence is either a premise, offering evidence or reasoning, or a claim, asserting a conclusion or viewpoint. Return only premise or claim."

FinArg - ACC

“In this task, you will be given Mergers and Acquisitions news articles or tweets. Your task is to classify each article or tweet based on whether the mentioned deal was completed or remained a rumour. Your response should be a single word - either ’complete’ or ’rumour’ representing the outcome of the deal mentioned in the provided text."

MA

“In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either ’causal’ or ’noise’ based on whether or not it indicates a causal relationship between financial events. Please return only the category ’causal’ or ’noise’."

SC

FinFact “ Determine if the following claim is 0. True or 1. False or 2. NEI (Not Enough Information). Please directly answer and do not explain. Claim: " MC “ Please answer following questions: "

“ You are a financial expert, you are supposed to answer the given question. You need to first think through the problem step by step, documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as “Therefore, the answer is {final answer}”. The final answer should be a numeric value. Question: {question} Let’s think step by step to answer the given question. "

KnowledgeMath

“ You are a financial expert, you are supposed to answer the given question. You need to first think through the problem step by step, documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as “Therefore, the answer is {final answer}”. The final answer should be a numeric value. {Document context} Let’s think step by step to answer the given question. "

DocMath-Eval

“Assess the creditworthiness of a customer using the following table attributes for financial status. Respond with either ’good’ or ’bad’. And the table attributes including 13 categorical attributes and 7 numerical attributes are as follows:"

German

“Assess the creditworthiness of a customer using the following table attributes for financial status. Respond with either ’good’ or ’bad’. And the table attributes including 13 categorical attributes and 7 numerical attributes and values have been changed to meaningless symbols to protect confidentiality of the data. :"

Australian

“Assess the client’s loan status based on the following loan records from Lending Club. Respond with only ’good’ or ’bad’, and do not provide any additional information. For instance, ’The client has a stable income, no previous debts, and owns a property.’ should be classified as ’good’."

LendingClub

“Detect the credit card fraud using the following financial table attributes. Respond with only ’yes’ or ’no’, and do not provide any additional information. Therein, the data contains 28 numerical input variables V1, V2, ..., and V28 which are the result of a PCA transformation and 1 input variable Amount which has not been transformed with PCA. The feature ’Amount’ is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. For instance, ’The client has attributes:{category}"

ccf

“Detect the credit card fraud with the following financial profile. Respond with only ’good’ or ’bad’, and do not provide any additional information. For instance, ’The client is a female, the state number is 25, the number of cards is 1, the credit balance is 7000, the number of transactions is 16, the number of international transactions is 0, the credit limit is 6.’ should be classified as ’good’."

ccfraud

“Predict whether the company will face bankruptcy based on the financial profile attributes provided in the following text. Respond with only ’no’ or ’yes’, and do not provide any additional information."

polish

“Predict whether the company will face bankruptcy based on the financial profile attributes provided in the following text. Respond with only ’no’ or ’yes’, and do not provide any additional information."

taiwan

“Identify whether or not to files a claim for the auto insurance policy holder using the following table attributes about individual financial profile. Respond with only ’yes’ or ’no’, and do not provide any additional information. And the table attributes that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation."

Porto-Seguro

“Identify the claim status of insurance companies using the following table attributes for travel insurance status. Respond with only ’yes’ or ’no’, and do not provide any additional information. And the table attributes including 5 categorical attributes and 4 numerical attributes are as follows:{category}"

travelinsurace

The results in Table 6 display only the highest scores attained by any of the three prompts for each evaluation metric. This approach highlights the optimal performance our model can achieve,

leveraging the strengths of each prompt. By doing so, we provide a clear and concise summary of our model’s capabilities, illustrating its versatility and robustness across different evaluation scenarios.

Table 14: Few-shot task datasets prompt overview.

Data Prompt

“Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral. For instance, ’The company’s stocks plummeted following the scandal.’ would be classified as negative."

FPB

“What is the sentiment of the following financial {category}: Positive, Negative, or Neutral?"

FiQA-SA

“Consider whether the headline mentions the price of gold. Is there a Price or Not in the gold commodity market indicated in the news headline? Please answer Yes or No."

Headlines

“In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person (’PER’), an organization (’ORG’), or a location (’LOC’). The required answer format is: ’entity name, entity type’. For instance, in ’Elon Musk, CEO of SpaceX, announced the launch from Cape Canaveral.’, the entities would be: ’Elon Musk, PER; SpaceX, ORG; Cape Canaveral, LOC’"

NER

[Figure 5]

Figure 5: Zero-shot performance of FinLLaMA and baseline models.

[Figure 6]

Figure 6: Few-shot performance of FinLLaMA and baseline models.

### E.6 Prompts for Generating LLM Answers

In our evaluation, we utilized three distinct prompts to assess performance comprehensively:

- Prompt 0: Our custom-designed prompt is tai-

lored specifically for the unique characteristics of our dataset and objectives.

- Prompt 1: The standard prompt corresponds to

the MMbench benchmark, allowing for a standardized comparison with existing models.

- Prompt 2: The prompt associated with

FinTral (Bhatia et al., 2024) , ensuring relevance and applicability in financial domain-specific tasks.

The rationale behind employing these three prompts is to capture a broad spectrum of scenarios and requirements.

By including our customized prompt (Prompt 0), we can fine-tune our model to the specific nuances of our dataset. The inclusion of MMbench (Prompt 1) provides a standardized baseline for comparison, facilitating a fair assessment against other models in the field. Lastly, FinTral’s prompt (Prompt 2) ensures that the evaluation encompasses domain-specific challenges, particularly in the financial sector.

#### E.6.1 Prompt 0

|<image>\n Given the following multiple-choice financial question and the image that contains relevant information, generate the correct answer based on the image content:<br><br>Question: {question}<br><br>The output format should be: {“answer": “<Insert the correct option here>"}|
|---|

- E.6.2 Prompt 1

|<image>\n {question} Answer with the option’s letter from the given choices directly.|
|---|

- E.6.3 Prompt 2

|You are a financial expert specializing in the nuanced analysis of financial statements and a wide array of data-driven financial tasks. For each prompt you are given, think step by step. Sometimes, you must extract relevant information to proceed with the problem.<br><br>Instructions: If any options are specified, ensure that your answer is one of the options specified. Do not explain why you think the answer is correct.<br><br>Context:<image>\n Question:{question} Answer:|
|---|

### E.7 Example of LLM Answers

Question: How does the stock index value on January 15, 2023, compare to the value on January 30, 2023?A. January 15 is lower B. January 15 is equal C. January 30 is higher D. January 15 is higher

- Answer of Prompt 0:: {“answer": “A. January

15 is lower"}

- Answer of Prompt 1:A
- Answer of Prompt 2:Identify Key Dates and

Values: The graph shows a downward trend from early January. On January 15 (indicated by “202301-17"), the stock index was around 4000 points. By January 30 (indicated as “2023-01-29"), the stock index had decreased significantly below 3000 points at approximately 3300 points.

Compare Values: As per the data: Value on January 15 = 4000; Value on January 30 = 3300

Conclusion Based on Comparison: Since 4000 > 3300, it’s clear that the stock index value on January 15, 2023, is higher compared to the value on January 30, 2023.

Therefore, based on this analysis:Answer: D. January 15 is higher

This conclusion aligns with option D because the values indicate an increase over time before reaching their peak again later into the month.

Figure 7: Example input.

[Figure 7]

### E.8 Prompts for Evaluating LLM Answers

For each prompt, we conducted separate evaluations to assess the performance of our LLM. To ensure the consistency and relevance of the LLM’s generated answers, we employed GPT-4o-mini to filter and determine the accuracy of the outputs for each prompt individually. This filtering process helps in aligning the generated answers with the intended prompt format and maintaining output quality. Here is our prompt.

|You are an AI model tasked with answering a multiple-choice question based on the provided additional information. Please answer the multiple-choice question based barely on the provided additional information.<br><br>Question and options: {query}<br><br>Additional Information: {LLM’s output}<br><br>Please analyze the question and options carefully and provide the answer in the following JSON format: {“answer": “<Insert the correct option here (A, B, C, or D)>"}<br><br>Note:<br><br>- Remember that you should only give me the JSON format without any additional information.<br>- You should answer the question with only A, B, C, or D, not the full answer text.<br>- If the additional information tells the answer, then you should follow his options.<br>- If you don’t know the answer, the output should be:{“answer": “E"}<br>|
|---|

## F Details for Trading in the Decision Making Task

In this task, at each time step, the LLM receives a feed of memories retrieved from the memory module. Based on this information, the LLM must

make an investment decision—choosing to buy, sell, or hold—while providing its reasoning for the decision and specifying the index of the supporting information within the memory module. This process evaluates the model’s ability to analyze financial data, make informed decisions, and justify its choices by referencing specific pieces of stored information in a dynamic trading environment.

Data: We evaluated the model’s performance on the Decision Making task using the following datasets:

- • OHLCV data: Open-High-Low-Close prices and trading volume data for COIN, GOOG, NIO, and TSLA, obtained from Yahoo Finance.
- • News data: Collected from Alpaca News API for COIN, GOOG, NIO, and TSLA.
- • Form 10-Q data: Extracted from SEC EDGAR for COIN, GOOG, and TSLA.
- • Form 10-K data: Retrieved from SEC EDGAR for COIN, GOOG, and TSLA.

Figure 8 and 9 present our prompts for trading in the Decision Making task.

## G Trading Comparison in the Decision Making Task

Figure 10 through 12 illustrate the overall cumulative returns over time for the Decision Making task using different models. For TSLA and NIO stocks, we can see that FinLLaMA consistently outperforms other models across all time periods. On the COIN stock, FinLLaMA exhibits a more stable and consistent upward trend in cumulative returns compared to other models, particularly before February 2023. In contrast, for GOOG stock, while FinLLaMA’s cumulative return is slightly lower than that of LLAMA3-8B and Palmyra-Fin70B-32K across various time periods, it remains superior to the Buy & Hold strategy.

Initialize Profile

- 1. Operations:

- - Provide a performance overview of the trading stock based on available data.
- - Set up the risk inclination as the key character of the trading agent.

- 2. Range: Financial information such as the financial sectors, historical performance, and previous stock trends of the trading stock.
- 3. Prompts: You are an experienced trading manager and investment firm. Your task is to make informed decisions on the given stock based on the provided information. Under Self-Adaptive Risk Character Setting: When historical momentum is positive, you are a risk-seeking investor. But when historical momentum is negative, you are a risk-averse investor.
- 4. General background setting: You have accumulated a lot of information about the following sectors, so you are especially good at trading them: 1)Electric Vehicles (Automotive Sector). 2) Energy Generation and Storage...From year 2021 to 2022 September, Tesla’s continued growth and solid financial performance over the defined period ... Summarize

- 1. Operations:

- - Summarize different types of input information.
- - Distribute them to corresponding layers of the long-term memory database.

- 2. Range: Daily market news, Long Documents such as company 10-K and 10-Q reports
- 3. Prompts:

- - (1). Summarize the contents: Summarize the following documents into 1000 words.
- - (2). Comprehend the investment sentiment of news insights: The positive, neutral and negative scores are for understanding the investment sentiments, opinions, or emotions. For example, positive news about a company can lift investor sentiment, encouraging more buying activity, which in turn can push stock prices higher...

- 4. Outputs:

- (1). To Shallow Memory Layer:

- - [News (ID: 261)] Here’s How Much You Would Have Made Owning Tesla Stock In The Last 10 Years Tesla (NASDAQ:TSLA) has outperformed the market over the past 10 years by 50.69% on an annualized basis producing an average annual return of 60.76%. Currently, Tesla has a market capitalization of $683.54 billion.... The sentiment is {positive}.
- - [News (ID: 278)] Tesla Q3 Earnings Are Imminent. Can Nio Foreshadow What’s To Come? What To Know Before The Print Tesla Inc (NASDAQ: TSLA) shares were trading down slightly Wednesday afternoon ahead of the automaker´s third-quarter report, but the stock is up 6% over the last five sessions... The sentiment is {positive}.
- - ...

- (2). To Intermediate Memory Layer:

- - [Form 10-Q (ID: 222)] Tesla Q3 2022 revenues were $21.5 billion, up 56% year-over-year. Automotive sales revenue grew 56% to $17.8 billion driven by higher Model 3/Y and Model S/X deliveries. Gross automotive margin declined to 27.9% due to cost inflation and factory ramps. Net income was $3.3 billion, up 102% year-over-year. Positive free cash flow was $6.1 billion...
- - [News (ID: 275)] Tesla Q3 Earnings Highlights: Record Revenue, Operating Margin And Free Cash Flow, Tesla Semi Deliveries Coming In December Electric vehicle leader Tesla Inc (NASDAQ: TSLA) reported third-quarter financial results after market close Wednesday...The sentiment is {neutral}.
- - [News (ID: 274)] Tesla Preps For 2023 Cybertruck Launch, Will Make Battery Packs In California The Cybertruck is one of Tesla Inc. (NASDAQ: TSLA) most hotly anticipated, but also most delayed, products. -

...The sentiment is {negative}.

- (3). To Deep Memory Layer:

- - [News (ID: 161)] Tesla Whale Trades Spotted A whale with a lot of money to spend has taken a noticeably bearish stance on Tesla. Looking at the options history for Tesla (NASDAQ:TSLA) we detected 477 strange trades. The sentiment is {positive}.
- - [Self-reflection (ID: 226)] Given the short-term positive news score in the market for TSLA and a positive cumulative return, there is a high probability of continued growth in the short term. However, investor should be aware of potential threats in the mid-term market with competitors like General Motors, and Nio...

Observe

- 1. Operations: Access and interpret market indicators such as current stock prices and historical momentum data.
- 2. Range: Stock’s daily adjusted closing price, historical momentum in the past k days (k = 3 in this case), etc.
- 3. Prompts:

- The information below provides a summary of stock price fluctuations over the previous few days, which is the "momentum" of a stock. It reflects the trend of a stock. Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.

- 4. Outputs:

- - (1). The daily adjusted closing price of TSLA on {2022-10-25} is {$222.42}.
- - (2). Train: On {2022-10-25}, the momentum of TSLA, indicated by the price difference between the current and the next trading

day, is {$2.22}. Test: On {2022-10-25}, the historical momentum of TSLA, as measured by its cumulative logarithmic

returns in the past {3} days, was {7.05%}.

Figure 8: First section of FINMEM’s workflow for perceiving and processing multi-sourced information from market environment.

Reflect

- 1. Operations: Train: - Infer the reasoning from the retrieved memories insights for the recommended trading actions and the increase or decrease of the trading stock price. Test: - Generate recommended trading actions. - Infer the reasoning for making such trading action. Train & Test: - Track the memory insight IDs that contributed to the current trading decision.
- 2. Range: Interpretation of observed market indicators and top k insights from each layer of the long-term memory database.
- 3. Prompts: Train: Given the following information, can you explain why the financial market fluctuation from the current day to the next day behaves like this? Summarize the reason for the decision. You should provide summary information and the IDs of the information to support your summary. Test: Given the information, can you make an investment decision? Just summarize the reason for the decision. Consider only the available short-term information, the mid-term information, the long-term information, and the reflection-term information. Consider the momentum of the historical stock price. Also, consider how much share of the stock the investor holds now. You should provide exactly one of the following investment decisions: buy or sell. You could go with the ’hold’ option if necessary. You also need to provide the IDs of the information to support your decision.
- 4. Outputs: Train:

- - Recommended Action Reason: The recommended trading action is ’Buy’. This can be attributed to positive market sentiment driven by a range of favorable factors. Notably, Tesla’s strong battery supply chain, effective market expansion plans in Germany, and its ability to maintain market strength despite potential competitive threats positively influenced investor confidence, contributing to the stock price rise.
- - Reflected Contributed Insights IDs: {261, 278, 275, 274, 161,...} Test:
- - Trading Action: {Buy}
- - Recommended Action Reason: Considering the short-term positive revenue growth, higher Model 3/Y and Model S/X deliveries, positive free cash flow, and increased cash and cash equivalents, Tesla appears to have strong financial health. The momentum being positive, it also encourages a risk-seeking strategy. The significant positive signs from news such as the imminent Q3 earnings add to the favorable view of the investment. Despite the mid-term neutral sentiment, long-term information shows only marginal drops and the stock is still looking strong overall. These, along with positive reflection-term information, form the reasoning behind the ’buy’ decision...
- - Reflected Contributed Insights IDs: {261, 278, 275, 274, 161,...}

Figure 9: Second section of FINMEM’s workflow for generating trading action, reasoning and reflection.

###### Figure 10: Comparison of CRs over time: FinLLaMA vs. other LLMs in TSLA trading with FINMEM.

[Figure 8]

###### Figure 11: Comparison of CRs over time: FinLLaMA vs. other LLMs in COIN trading with FINMEM.

[Figure 9]

###### Figure 12: Comparison of CRs over time: FinLLaMA vs. other LLMs in GOOG trading with FINMEM.

[Figure 10]

###### Figure 13: Comparison of CRs over time: FinLLaMA vs. other LLMs in NIO trading with FINMEM.

[Figure 11]

