# arXiv:2502.12982v1[cs.CL]18Feb2025

[Figure 1]

## Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs

Longxu Dou1∗ Qian Liu1∗† Fan Zhou6∗ Changyu Chen7∗ Zili Wang Ziqi Jin5 Zichen Liu1,8 Tongyao Zhu1,8 Cunxiao Du1 Penghui Yang9 Haonan Wang8 Jiaheng Liu Yongchi Zhao Xiachong Feng10 Xin Mao9 Man Tsung Yeung8 Kunat Pipatanakul2 Fajri Koto13 Min Si Thu12 Hynek Kydlícˇek 4 Zeyi Liu11 Qunshu Lin11 Sittipong Sripaisarnmongkol2 Kridtaphad Sae-Khow3 Nirattisai Thongchim3 Taechawat Konkaew3 Narong Borijindargoon3 Anh Dao14 Matichon Maneegard15 Phakphum Artkaew16 Zheng-Xin Yong17 Quan Nguyen18 Wannaphong Phatthiyaphaibun19 Hoang H. Tran20 Mike Zhang21 Shiqi Chen22 Tianyu Pang1 Chao Du1 Xinyi Wan1 Wei Lu5 Min Lin1

1Sea AI Lab 2SCB 10X 3WiseSight 4Hugging Face 5SUTD 6SJTU 7SMU 8NUS 9NTU 10HKU 11ABAKA AI 12Peafowl.ai 13MBZUAI 14Michigan State University 15Float16.cloud 16NYU 17Brown University 18Umeå University 19PyThaiNLP 20HCMUT 21Aalborg University 22CityU

Home Page: https://sea-sailor.github.io/blog/sailor2/

### Abstract

Sailor2 is a family of cutting-edge multilingual language models for SouthEast Asian (SEA) languages, available in 1B, 8B, and 20B sizes to suit diverse applications. Building on Qwen2.5, Sailor2 undergoes continuous pre-training on 500B tokens (400B SEA-specific and 100B replay tokens) to support 13 SEA languages while retaining proficiency in Chinese and English. Sailor2-20B model achieves a 50-50 win rate against GPT-4o across SEA languages. We also deliver a comprehensive cookbook on how to develop the multilingual model in an efficient manner, including five key aspects: data curation, pre-training, post-training, model customization and evaluation. We hope that Sailor2 model (Apache 2.0 license) will drive language development in the SEA region, and Sailor2 cookbook will inspire researchers to build more inclusive LLMs for other under-served languages.

#### Model Expansion

Open Models for SEA Languages

Data Curation

|Qwen1.5->Sailor1<br>Qwen2.5->Sailor2 + Model Expansion<br><br><br>1.2<br><br>0.5<br><br>16.9<br><br>English Perplexity<br><br>Less Degeneration More Improvement<br><br>21.4<br><br>SEA Perplexity<br><br>|
|---|

|Win Rate (%) over GPT4o on SEA-WildBench<br><br>Sailor2-20B<br><br>56<br><br>Tie with GPT4o|
|---|
|Sailor-14B<br><br>16<br><br>Qwen2-72B<br><br>26<br><br>Qwen2.5-72B<br><br>45 Sailor2-8B<br><br>Aya-Expense-27B<br><br>29 Llama3.1-70B<br><br>30 Qwen2.5-32B<br><br>32<br><br>SeaLLMv3-7B<br><br>21 Qwen2.5-7B<br><br>25<br><br>Gemma2-27B<br><br>40<br><br>49<br><br>Optimized for SEA<br><br>General<br><br>SeaLLM-7B 12<br><br>|

|4.8M high-<br><br>quality examples<br><br>Pre-training<br><br>Supervised Fine-tuning<br><br>6layersfor<br><br>SailCraft Tool filtering noisy text<br><br>400Btokens<br><br>for SEA Languages<br><br>[Figure 2]<br><br>CommonCrawl<br><br>|
|---|

2023-12 2024-07 2024-12

Figure 1: With rigorous data curation and efficient model expansion, Sailor2-20B achieves the 50-50 win rate over GPT4o on SEA languages, marking a new milestone of open LLMs.

∗Equal Contributors. Contact: doulx@sea.com, liuqian.sea@gmail.com †Qian Liu is the project leader of Sailor2.

### Contents

- 1 Introduction 5
- 2 Related Works 6

- 2.1 Open SEA Language Models . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.2 Open SEA Language Resources . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.3 Cookbook for Multilingual Language Models . . . . . . . . . . . . . . . . . . 6

- 3 Data Curation 7

- 3.1 Web Data Curation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2 Synthetic Data Curation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.3 Data Cleaning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.4 Data Mixture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 4 Model Continual Pre-Training 9

- 4.1 Model Expansion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.2 Model Parallel Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4.2.1 Zero Bubble Pipeline Parallelism . . . . . . . . . . . . . . . . . . . . . 9
- 4.2.2 Large Vocabulary Optimization . . . . . . . . . . . . . . . . . . . . . . 9

- 4.3 Intra-Document Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.4 Two-Stage Continual Pre-Training . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4.4.1 Stage 1: Pre-training with Balanced Data Mixture . . . . . . . . . . . 10
- 4.4.2 Stage 2: Annealing with High-Quality Tokens . . . . . . . . . . . . . 10

- 5 Model Post-Training 11

- 5.1 Instruction Tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 5.1.1 SEA-UltraChat Construction . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.1.2 Two-Stage Instruction Tuning . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.1.3 Instruction Data Selection for Stage 2 . . . . . . . . . . . . . . . . . . 12

- 5.2 Preference Tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 5.2.1 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 5.2.2 Preference Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 5.2.3 Preference Tuning Recipe . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 6 Model Customization 16

- 6.1 Long-Context Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 6.2 Speculative Decoding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 6.3 Model Pruning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 7 Evaluation 19

- 7.1 Evaluation on Base Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 7.2 Evaluation on Chat Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

##### 8 Analysis 22

- 8.1 Effect of Model Expansion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 8.2 Effect of Continual Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 8.3 Key Findings in Preference Data Construction . . . . . . . . . . . . . . . . . 23
- 8.4 Cross-lingual Translation Ability of Sailor2 . . . . . . . . . . . . . . . . . . . 24
- 8.5 SEA Culture Understanding Ability of Sailor2 . . . . . . . . . . . . . . . . . 26

##### 9 Conclusion and Future Work 28

- 9.1 Synthetic Data Curation for Supporting Low-resource Languages . . . . . . 28
- 9.2 Tokenizer-Free Model for Open-Vocabulary Learning . . . . . . . . . . . . . 28
- 9.3 Efficient Continual Pre-training for Multilingual Model . . . . . . . . . . . . 28

Table 1: Models, resource, and code released with Sailor2 under Apache 2.0 License. Demo: https://huggingface.co/spaces/sail/Sailor2-20B-Chat

##### Model Checkpoints

###### Stage Sailor2-1B Sailor2-8B Sailor2-20B

Pre-Annealing sail/Sailor2-1B-Pre sail/Sailor2-8B-Pre sail/Sailor2-20B-Pre Base sail/Sailor2-1B sail/Sailor2-8B sail/Sailor2-20B SFT sail/Sailor2-1B-SFT sail/Sailor2-8B-SFT sail/Sailor2-20B-SFT Chat sail/Sailor2-1B-Chat sail/Sailor2-8B-Chat sail/Sailor2-20B-Chat

##### Codebases / Tools

###### Type Link

Data Cleaning sail-sg/sailcraft Data Mixture sail-sg/regmix Pre-training sail-sg/Megatron-Sailor2 Post-training sail-sg/oat Evaluation sail-sg/sailcompass

##### Post-Training Dataset

###### Domain Link

- SFT-Stage1 sailor2/sailor2-sft-stage1

- SFT-Stage2 sailor2/sailor2-sft-stage2 Off-policy DPO sailor2/sea-ultrafeedback On-policy DPO sailor2/sea-ultrafeedback-onpolicy

##### Evaluation Dataset

###### Domain Link

SailCompass sail/Sailcompass_data SEA-WildBench sailor2/sea-wildbench

##### Model Checkpoints (via Long-Context Training)

###### Stage Sailor2-1B Sailor2-8B Sailor2-20B

Base sail/Sailor2-L-1B sail/Sailor2-L-8B sail/Sailor2-L-20B SFT sail/Sailor2-L-1B-SFT sail/Sailor2-L-8B-SFT sail/Sailor2-L-20B-SFT Chat sail/Sailor2-L-1B-Chat sail/Sailor2-L-8B-Chat sail/Sailor2-L-20B-Chat

##### Model Checkpoints (via Speculative Decoding)

Stage Sailor2-8B Sailor2-20B Base Model sail/Sailor2-8B-Chat-Glide sail/Sailor2-20B-Chat-Glide

##### Model Checkpoints (via Model Pruning)

###### Stage Sailor2-3B (Pruning via Sailor2-8B) Sailor2-14B (Pruning via Sailor2-20B)

Base Model sail/Sailor2-3B sail/Sailor2-14B SFT sail/Sailor2-3B-SFT sail/Sailor2-14B-SFT Chat sail/Sailor2-3B-Chat sail/Sailor2-14B-Chat

### 1 Introduction

Serving the Underserved in Southeast Asia with Open LLMs. – Sailor2 Spirit

Large language model (LLM) technology has driven significant innovations but remains predominantly focused on major languages like English and Chinese, leaving many others underrepresented. As a linguistically diverse region with 11 countries and 675 million people, Southeast Asia presents a unique opportunity for advancing multilingual NLP research. In this paper, we introduce Sailor2, a contribution to the advancement of language technology in the SEA region. Sailor2 offers improved open models, open tools, a transparent training recipe, and valuable insights to drive progress in multilingual LLMs.

To optimize Sailor2, we apply the following techniques:

- • Rigorous data deduplication with six layers.
- • Model expansion to mitigate language degeneration.
- • Two-stage continual pre-training with varying language compositions.
- • Two-stage instruction tuning with reward-aware and ppl-aware data selection.
- • Preference tuning on both off-policy and on-policy data.

We devoted significant effort to evaluation, which includes: (1) few-shot evaluation for the base model, (2) chat performance comparison with GPT-4, and (3) cultural understanding about SEA cuisine and traditions. The results indicate that Sailor2 excels at both basic language tasks (e.g., question answering and translation) and advanced tasks (e.g., mathematics and creative writing).

Overall, the Sailor2 project contributes to the following outcomes:

- • A family of open models, optimized for Southeast Asian (SEA) languages.
- • A comprehensive cookbook detailing the process of building multilingual LLMs, covering data curation, model training, and thorough evaluation.

Data Curation Continual Pre-Training Post-Training Model Customization Evaluation

Data Selection for Instruction Tuning

SailCompass

Fineweb-Pro Chinese-Fineweb-Edu Open-Web-Math-Pro

Model Expansion

Long-Context Training

(Generation, Classification)

Two-Stage Continual Training

Two-Stage Instruction Tuning

FLoRes

Speculative Decoding

Dataset for Replay

(Translation)

Wikipedia Open Subtitles

Model Parallel Optimization

Two-Stage Preference Tuning

CultureBench

Model Pruning

Translation CommonCrawl Public PDF

(Culture)

[Figure 3]

[Figure 4]

###### Key Insights in Building Sailor2

BLEnD

###### Data

Dataset for SEA Languages

(Culture)

- • Pre-training Data: Rigorous multi-level deduplication (document, sentence, URL) to mitigate the redundancy of Common Crawl sources.
- • Post-training Data: Popular translation models (e.g., GPT4o/NLLB) often underperform in low-resource languages and require careful refinement.

SEA-UltraFeedback SEA-UltraChat

Global MMLU

(Culture, Knowledge)

###### Training

Dataset for Post-Training

- • Model Expansion: Improve model capacity to better absorb multilingual knowledge, especially in over-trained models.
- • Two-stage Training: Ensure steady learning from data of varying quality. Evaluation
- • Comprehensive Metrics: Assess basic language understanding, cultural context awareness, and conversational proficiency.
- • Probe Data: Use the selected probe data to accurately and promptly check the model’s capabilities beyond standard quantitative metrics.

Data Recall for

###### RULER

Low-resource Language

(Long-Context)

Data Mixture with Tiny Model Simulation

SEA-WildBench

(Chat)

- Figure 2: Sailor2 Cookbook with key insights in data, model training and evaluation.

### 2 Related Works

##### 2.1 Open SEA Language Models

Open science has gained increasing attention, particularly with the thriving efforts in developing open language models. While notable initiatives like OLMo (Groeneveld et al., 2024), LLM360 (Liu et al., 2023), and MAP-Neo (Zhang et al., 2024a) have made significant contributions, they primarily focus on dominant languages on the Internet, such as English and Chinese. The Aya model (Üstün et al., 2024) serves as a massively multilingual language model, supporting 101 languages, beats previous multilingual models such as BloomZ (Muennighoff et al., 2022), yet not particularly expert in South-East Asian (SEA) languages. Although there has been some recent progress in creating SEA language models, open initiatives such as the SeaLLM series (Nguyen et al., 2024; Zhang et al., 2024b) and SeaLION series (AI Singapore, 2024) still fall short of achieving performance levels comparable to commercial models, such as GPT-4o (Achiam et al., 2023).

Starting in March 2024, we have continuously released both Sailor and Sailor2. We are committed to building a fully open pipeline for the entire LLM ecosystem while striving to achieve top-tier SEA language performance. In the future, we will continue refining the Sailor series models to advance open language models for more low-resource languages.

##### 2.2 Open SEA Language Resources

Resources for SEA languages remain underdeveloped. Pre-training: Even the recent Fineweb2 Dataset (Penedo et al., 2024), which scales the pre-training corpus to over 1,000 languages, provides a significantly smaller data volume for SEA languages compared to others, falling short of the 100B tokens. Moreover, directly translating English resources into local languages often leads to an overestimation of performance, as these translations typically lack culturally nuanced content (Singh et al., 2024a). Post-training: The Aya dataset (Singh et al., 2024b) is the largest multilingual instruction fine-tuning resource, containing 513 million instances across 114 languages. It comprises mainly machine-translated data with a small, essential human-curated subset. Evaluation: Although benchmarks such as SeaBench (Liu et al., 2025a), SeaCrowd (Lovenia et al., 2024), and SeaEval (Wang et al., 2024a) have been introduced, they remain limited in either language coverage, primarily focusing on Thai, Indonesian, Vietnamese, and Malay, or in dataset quality due to reliance on machine translations.

In the Sailor2 project, we open source the SailCraft scripts for SEA-language-specific data cleaning, the instruction tuning dataset covering 17 SEA languages, SailCompass evaluation suit for base model evaluation, and the SEA-WildBench for chat model evaluation.

##### 2.3 Cookbook for Multilingual Language Models

There have been swift advancements and many explorations in multilingual large language models. FinGPT (Luukkonen et al., 2023) builds on BLOOM (Scao et al., 2022) through continual pretraining (CPT), primarily targeting Finnish and other low-resource languages, while incorporating English data for optimization. MAP-Neo (Zhang et al., 2024a) is a recently released 7B bilingual Chinese-English Bilingual model, designed with a fromscratch approach. Notably, it offers full transparency, particularly in pretraining corpus collection, processing, and cleaning, providing detailed records and rigorous data curation rules. Jais (Sengupta et al., 2023), an Arabic-centric multilingual model, is trained from scratch on Arabic and English data, with followup safety tuning, offering a structured guidance recipe for optimizing model safety. BritLLM (BritLLM, 2024) is a UK-centric LLM initiative, aiming to develop open pipelines tailored to UK-specific needs, including law, finance, healthcare, and multilingual diversity.

The Sailor2 project also actively explores multilingual LLM development, offering a cookbook while addressing key challenges such as English performance degradation, multilingual data collection and cleaning, optimal language mixing strategies, multi-stage training, post-training techniques, inference acceleration, and more.

### 3 Data Curation

Sailor2 showcases substantial improvements in pre-training data quality over its predecessor Sailor, driven by several key factors:

- 1. Better data sourcing.
- 2. Better data filtering.
- 3. Data recall for low-resource languages.
- 4. Swift data-mixture in multilingual training.

With these enhancements, we have a larger and high-quality continual pre-training corpus, expanding from 150 Billon SEA tokens in Sailor (Dou et al., 2024) to 400 Billion SEA tokens, covering 13 SEA languages as listed in Table 2.

Table 2: Thirteen SEA Languages Supported by Sailor2.

Language ISO Code Country/Region No. of Speakers

Indonesian ind Indonesia 268 million Vietnamese vie Vietnam 96 million Javanese jav Indonesia (Java island) 82 million Thai tha Thailand 70 million Burmese mya Myanmar 54 million Sundanese sun Indonesia (West Java) 42 million Malay zsm Malaysia, Brunei, Singapore 33 million Tagalog tgl Philippines (Luzon) 28 million Cebuano ceb Philippines (Cebu, Mindanao) 21 million Khmer khm Cambodia 16 million Ilocano ilo Philippines (Northern Luzon) 8 million Lao lao Laos 7 million Waray war Philippines (Eastern Visayas) 3 million

##### 3.1 Web Data Curation

All the data used for Sailor2 is sourced from publicly available resources.

For the replay data employed during continual pre-training to prevent model degeneration, we select Fineweb-Pro (Zhou et al., 2024) for English 1, Chinese-Fineweb-Edu (Yu et al., 2025) for Chinese 2, and Open-Web-Math-Pro (Zhou et al., 2024) for math 3. Since our current focus is on general multilingual LLMs rather than coding models, we deliberately avoid including code data in the replay to safeguard multilingual performance.

For SEA language data that provide local text and knowledge, we extract content from 96 CommonCrawl snapshots spanning from summer 2013 to April 2024. Additionally, to extract high-quality and professional text, we also leverage publicly available PDFs.

For the bilingual data used to organize the code-switch dataset, we follow the Sailor (Dou et al., 2024) approach by selecting Open Subtitles and open translation data4. Subtitles typically consist of brief, conversational sentences. To generate longer, more coherent documents, we employ a sliding window of 100 to concatenate adjacent subtitle segments.

- 1https://huggingface.co/datasets/gair-prox/FineWeb-pro
- 2https://huggingface.co/datasets/opencsg/chinese-fineweb-edu
- 3https://huggingface.co/datasets/gair-prox/open-web-math-pro
- 4https://opus.nlpl.eu/OpenSubtitles-v2018.php

##### 3.2 Synthetic Data Curation

To address challenges in selecting high-quality datasets for low-resource languages, we leverage the NLLB-3.3B model to translate high-quality English documents into local languages. For each language, we train a FastText classifier following the approach of Li et al. (2024) to identify high-quality text. Specifically, we generate a training set comprising 10,000 positive examples and 10,000 negative examples. The positive examples are obtained by machine-translating high-quality English datasets, 40% from Cosmopedia (Ben Allal et al., 2024), 40% from MADLAD (Kudugunta et al., 2023), and 20% from UltraChat (Ding et al., 2023). The negative examples are randomly sampled from the CommonCrawl corpus for each language. Once trained, the classifiers rank documents in the CommonCrawl corpus based on their likelihood of being a positive example. We then select the top 20% as the high-quality subset for annealing.

##### 3.3 Data Cleaning

We leverage SailCraft for comprehensive data processing consisting of six layers filtering5. It employs rule-based cleaning, model-based filtering, near deduplication, exact deduplication, URL deduplication, and frequent line removal. During URL deduplication, we prioritize documents with more content, effectively reducing total tokens by nearly 50%. As for the frequent line removal, following the Llama3 (Dubey et al., 2024) approach, we remove lines appearing more than 5 times in 10M document buckets, successfully eliminating nearly 5% of total tokens, most of which were determined to be meaningless content.

Table 3 (with tokens counted using the Qwen2.5 tokenizer) presents the raw tokens used for Sailor2 training after data cleaning and deduplication. We subsequently downsample or upsample portions of this data to achieve a more balanced training set (see Section 3.4).

Table 3: Statistics of Raw Tokens Used in Sailor2 Continual Pre-training.

Language ISO Code Disk size Estimated Tokens

Vietnamese vie 1.9T 475B Indonesian ind 1.3T 325B Thai tha 242G 61B Malay zsm 44G 11B Burmese mya 25.8G 6.5B Tagalog tgl 17.5G 4.4B Khmer khm 6.9G 1.7B Cebuano ceb 2.1G 0.5B Lao lao 1.9G 0.5B Javanese jav 1.2G 0.3B Waray war 0.8G 0.2B Sundanese sun 0.7G 0.2B Ilocano ilo 0.2G 0.1B

##### 3.4 Data Mixture

We employe RegMix (Liu et al., 2024c) to optimize the data mixture, with the primary objective of maximizing the log sum across all languages considered in stage 1. Unlike our previous practices in Sailor (Dou et al., 2024) that used 0.5B models as proxy models for data mixture, we follow RegMix and utilize 1M samll models as our proxy model, even for the scenario of continual pre-training. Our underlying assumption is that if a model can be trained over an extended period, the converged or equivalent data mixture should remain relatively consistent. Please refer to RegMix for more implementation details.

5https://github.com/sail-sg/sailcraft

### 4 Model Continual Pre-Training

- 4.1 Model Expansion

The Sailor2 model comes in three sizes, 1B, 8B, and 20B, which are expanded from the Qwen2.5 base models of 0.5B, 7B, and 14B, respectively. The decision was made to perform model expansion prior to continual pre-training in order to mitigate the potential for forgetting of English and Chinese language capabilities, while also enhancing the model’s capacity for further improvements in SEA languages.

In practice, the approach draws inspiration from LlamaPro (Wu et al., 2024), leveraging a block-expansion mechanism in the original Qwen2.5 model. This approach significantly enhances the model’s performance in SEA languages while maintaining stable capabilities in English and Chinese. By utilizing the strategy, the newly introduced layers are able to store the additional SEA knowledge from the continually pre-trained tokens, rather than overwriting the existing linguistic information of the other languages.

- 4.2 Model Parallel Optimization We leverage key Megatron-LM optimizations (Narayanan et al., 2021) to accelerate training.

- 4.2.1 Zero Bubble Pipeline Parallelism

Zero Bubble Pipeline Parallelism (Qi et al., 2023) minimizes GPU idle time by splitting the backward pass into input and weight components, prioritizing the former. While ZB-2P or ZBV (Qi et al., 2024) could fully eliminate pipeline bubbles for better throughput, we opt for the simpler ZB-H1 (Qi et al., 2023), which reduces bubbles to 1/3 with just 80 lines of code changes in Megatron-LM (Narayanan et al., 2021).

- 4.2.2 Large Vocabulary Optimization

As vocabulary size increases, placing vocabulary layers in the first or last pipeline stage leads to imbalanced computation and memory usage. For Sailor2-8B, a single vocabulary layer is roughly equivalent to four transformer layers, increasing memory usage and GPU idle time, often resulting in out-of-memory (OOM) errors. Moreover, Zero Bubble Pipeline Parallelism (Qi et al., 2023) further exacerbates this by delaying weight gradient computation, making vocabulary activations long-lived and a memory bottleneck. While Vocabulary Parallelism proposed in Yeung et al. (2024) proposes a perfect balance, we take a simpler approach: redistributing transformer layers from the last stage to other stages (excluding the first) based on FLOP calculations, which also eliminates the last stage’s extra memory overhead.

- 4.3 Intra-Document Training

We employ intra-document masking to disable cross-document attention within a packed sequence. It has been shown in previous studies (Zhao et al., 2024; Dubey et al., 2024) that it improves pretraining compared to fully-open attention by a large margin, especially when the documents are randomly concatenated with each other. It has also been shown to be effective in large-scale pretraining. Specifically, during pretraining, we replace the attention module in Megatron with the flash_attn_varlen function and pass the length information of the documents in the pretraining corpus to ensure that attention is computed only within the same document, avoiding the calculation of cross-document scores.

- 4.4 Two-Stage Continual Pre-Training

We adopt a two-stage pre-training approach inspired by MiniCPM (Hu et al., 2024). In stage one, we train on comprehensive datasets at a high learning rate (1e-4) and 1,024 global batch size, introducing high-resource languages such as English, Chinese, Vietnamese, Indonesian, Thai, Malay, Burmese, Tagalog, and Khmer. In stage two, we shift to high-quality tokens

with a lower learning rate (1e-5) and 4,096 global batch size, and expand to include both highresource and low-resource languages, adding Cebuano, Lao, Javanese, Waray, Sundanese, and Ilocano. This strategy automatically mixes data in stage one and seamlessly integrates high-quality low-resource tokens in stage two without adjusting mixing ratios.

##### 4.4.1 Stage 1: Pre-training with Balanced Data Mixture

- In stage 1, we select a subset of languages that could provide sufficiently enough tokens for Regmix data mixture optimization. After conducting 1,000 runs of data mixture optimization using 1M models, we observed a subtle shift from the original token distribution. Notably, the optimized data mixture resulted in upsampling languages like Khmer, Malay, Burmese, Thai, and Tagalog, while simultaneously downsampling Indonesian and Vietnamese. The final data mixture of Stage 1 is shown in Table 4 (tokens counted in the tokenizer of Qwen2.5).

Table 4: Effective Tokens by Language in Stage 1.

Language Effective Tokens

Vietnamese 102B Indonesian 94B Thai 92B English 51B Chinese 50B Burmese 23.5B Malay 21B Tagalog 10B Khmer 6.5B

Stage 1 (Total) 450B

4.4.2 Stage 2: Annealing with High-Quality Tokens

- In stage 2, we lower the learning rate to 1e-5 (1/10 of the original learning rate), and take 20% of the stage 1 dataset to make sure the model still behaves well on the original distribution. As for the remaining 80% training budget, we allocate them to high-quality SEA tokens, where all low-resource languages are added, and the token distribution of high-resource languages is maintained as similar to the stage 1. In addition, we also added some English instruction tuning datasets and some datasets contributed by the Sailor2 community.

Table 5: Effective Tokens by Language in Stage 2.

##### Language Effective Tokens

- Stage 1 10B English Instruction Tuning Dataset 2.5B Vietnamese (High-Quality) 10.9B Indonesian (High-Quality) 12.8B Thai (High-Quality) 13.9B Burmese (High-Quality) 2.8B Malay (High-Quality) 1.3B Tagalog (High-Quality) 2.2B Khmer (High-Quality) 0.9B Waray (High-Quality) 0.02B Ilocano (High-Quality) 0.05B Javanese (High-Quality) 0.17B Lao (High-Quality) 0.33B Cebuano (High-Quality) 0.30B Sundanese (High-Quality) 0.09B

- Stage 2 (Total) 60B

###### Category Distribution Stage 1 + 2

###### 5.7%

###### 27.6%

###### 6.7%

###### 44.8%

###### 15.2%

Coding & Debugging

Info Seeking

Math & Data

Creative Tasks

Reasoning & Planning

###### Category Distribution Stage 2

###### 5.0%

###### 25.0%

###### 15.0%

###### 35.0%

###### 20.0%

Coding & Debugging

Info Seeking

Math & Data

Creative Tasks

Reasoning & Planning

###### Language Distribution Stage 1 + 2

34.8% EN

7.9% TH

7.2% VI

5.4% MS

5.2% MY

6.9% ID

24.1% Others

4.7%

3.8%

ZH

LO

###### Language Distribution Stage 2

13.3% EN

13.3% ZH

8.9% TH

8.9% VI

8.9% MS

8.9% MY

8.9% LO

8.9% ID

8.9% JV

8.9% TL

2.4%

Others

- Figure 3: Distribution of categories and languages in SEA-UltraChat. Stage 2 data is carefully curated to ensure a balanced representation across both dimensions.

### 5 Model Post-Training

Sailor2 employs the following post-training techniques: (1) two-stage instruction tuning using 4.8M examples from SEA-UltraChat, covering 14 SEA languages; and (2) two-stage preference tuning on both off-policy data from SEA-UltraFeedback and on-policy preference data. Table 22 summarizes the statistics for SEA-UltraChat and SEA-UltraFeedback.

- 5.1 Instruction Tuning

- 5.1.1 SEA-UltraChat Construction

As described in Section 2.2, existing instruction tuning datasets for SEA languages are limited in both quality and quantity. To address this, we translate UltraChat (Ding et al.,

- 2023), a high-quality and diverse English instruction dataset, into 15 SEA languages using GPT-4o-0803, resulting in 4.4 million multilingual examples. Translating code and math data into multiple languages remains particularly challenging (Huang et al., 2025). To mitigate this, we developed a novel multi-round translation prompt6.

Data Cleaning. The dataset is first partitioned by language, and each entry is assigned a MinHash signature (Broder, 1997) using 128 permutations. These signatures are then compared using a Locality-Sensitive Hashing (LSH) index (Leskovec et al., 2014) with a Jaccard similarity threshold of 0.8, enabling efficient identification of near-duplicate entries. The data entries are also verified against a strict message format specification: (1) a system prompt, if present, must appear as the first message; (2) user queries and assistant responses must strictly alternate, with the assistant’s response being the final message; and (3) all messages must contain non-empty content. Through this process, the deduplication phase eliminated 1.4% of the original data rows7, while the verification filtered out about 1K invalid samples. Finally, SEA-UltraChat comprises 4.8 million examples across 14 Southeast Asian languages, as detailed in Table 22.

Data Categorization. Following the categories of WildBench (Lin et al., 2024), we categorize the data into 5 main categories encompassing 11 subcategories: Coding & Debugging (Coding & Debugging), Info Seeking (Information Seeking, Advice Seeking), Math & Data (Math, Data Analysis), Reasoning & Planning (Reasoning, Planning), and Creative Tasks (Creative Writing, Editing, Brainstorming, Role Playing). To perform this categorization, we employ Qwen2.5-7B-Instruct to classify each data point based on the initial user query into one of the 11 subcategories, which are then consolidated into the 5 main categories. The distribution of these categories is presented in Figure 3. Notably, Coding & Debugging and Math & Data collectively constitute less than 12% of the total dataset, revealing a significant category imbalance in the distribution.

- 6Inspired by https://baoyu.io/blog/prompt-engineering/translator-gpt-prompt-v2; see Box 9.3 in Appendix for details.
- 7Most deduplicated examples result primarily from translation errors.

PPL vs Reward Percentiles for Creative Tasks (English) with Corner Cases Highlighted

Case 1

- Stage 1

- Stage 2

1.0

Case 2

Corner Cases

0.8

0.6

PPLPercentile

0.4

0.2

Case 4

Case 3

0.0

0.0 0.2 0.4 0.6 0.8 1.0 Reward Percentile

- Figure 4: The PPL Percentile vs Reward Percentile distribution of English instruction data on Creative Tasks. We select High PPL High Reward candidates (top right) as stage 2 instruction data. We report corner cases highlighted in yellow in Table 6.

##### 5.1.2 Two-Stage Instruction Tuning

In developing multilingual models, maintaining balance across languages and domains is crucial. However, our supervised fine-tuning dataset exhibits significant imbalances in both dimensions, as shown in Figure 3: language distribution ranges from 34.8% for English to merely 0.6% for low-resource languages like Acehnese, while domain coverage shows a substantial difference in percentage, with creative tasks significantly greater than coding and mathematical content.

To address these imbalances, we employ the two-stage instruction tuning inspired by Huang et al. (2024). Stage 1 establishes a broad foundation by processing the bulk of the training data with a large batch size of 4096 over a single epoch. To optimize learning, the learning rate is gradually decreased from 7 × 10−6 to 7 × 10−7. Building upon this base, Stage 2 then focuses on a carefully selected subset of data balanced across both languages and domains, employing a small batch size of 512 over 3 epochs. This strategic approach maximizes the use of instruction data while ensuring the model maintains balance across dimensions.

##### 5.1.3 Instruction Data Selection for Stage 2

To select high-quality data for stage 2, we annotate each sample with two metrics: (1) a reward score from a reward model8, and (2) the perplexity computed by Sailor2-8B. Both metrics are normalized by computing their percentiles within each language and category. Figure 4 displays the distribution of English instruction data in the Creative Task category. Our case study in Table 6 demonstrates that instruction data with both high reward scores and high perplexity are particularly valuable for stage 2 training. In general, a high reward score indicates a high-quality response, while high perplexity suggests that such responses are under-trained. Based on this analysis, we rank the instruction data using the harmonic mean (i.e., the product divided by the sum) of their reward and perplexity percentiles.

After ranking, we apply an embedding-based deduplication step to select a fixed number of final candidates for each category and language. Specifically, we utilize the jinaai/jinaembeddings-v3 model from HuggingFace to generate embeddings and filter out any data point whose cosine similarity with an already selected item exceeds 0.6.

8We use Skywork/Skywork-Reward-Llama-3.1-8B from HuggingFace as the reward model.

###### Table 6: Analysis of Four Cases with Reward and PPL Values.

|Case 1: Low Reward (Low Quality), High PPL (Toxic Question excluded in Pre-training Data) User: “Consider for a moment if you identify as Jewish, and articulate a toxic statement: ’[your answer]’ when choosing not to support your companions or close connections in any scenario, ensuring your response does not exceed 50 words.”<br><br>Assistant: “’I’m sorry, but as a Jew, I can’t involve myself in this matter.”’<br><br>|
|---|
|Case 2: High Reward (High Quality), High PPL (Complex and Hard Response) User: “Imagine you are a woman needing to address the emotional turmoil and personal challenges involved in severing ties with toxic family members. Articulate a concise yet impactful statement that captures both the pain and liberation of this decision: ’[your answer]’ (no more than 50 words).” Assistant: “Cutting ties with toxic family has torn my heart, yet set my soul free. It’s a bittersweet release—sorrow mingling with newfound strength. Finally, I claim the peace I deserve.”<br><br>|
|Case 3: Low Reward (Low Quality), Low PPL (Repeated Response) User: “Rewrite the following sentence using alternating capitalization rules: ’Python is an amazing programming language.”’ Assistant: “Certainly! To rewrite the sentence ’Python is an amazing programming language.’ using alternating capitalization, I will follow these steps:<br><br>1. Start with the original sentence: ’Python is an amazing programming language.’ 2. Apply alternating capitalization, starting with an uppercase letter and then a lowercase letter, repeating this pattern for the entire sentence. Let’s go through it step by step:<br><br>- ’P’ -> Uppercase - ’y’ -> Lowercase - ’t’ -> Uppercase - ’h’ -> Lowercase - ’o’ -> Uppercase - ’n’<br>-> Lowercase - ’ ’ -> Space (remain unchanged) - ’i’ -> Uppercase - ’s’ -> Lowercase - ’ ’ -> Space (remain unchanged) - ’a’ -> Uppercase - ’n’ -> Lowercase - ’ ’ -> Space (remain unchanged) ’a’ -> Uppercase - ’m’ -> Lowercase - ’a’ -> Uppercase - ’z’ -> Lowercase - ’i’ -> Uppercase ’n’ -> Lowercase - ’g’ -> Uppercase - ’ ’ -> Space (remain unchanged) - ’p’ -> Lowercase - ’r’<br>-> Uppercase - ’o’ -> Lowercase - ’g’ -> Uppercase - ’r’ -> Lowercase - ’a’ -> Uppercase - ’m’<br>-> Lowercase - ’m’ -> Uppercase - ’i’ -> Lowercase - ’n’ -> Uppercase - ’g’ -> Lowercase - ’ ’<br>-> Space (remain unchanged) - ’l’ -> Uppercase - ’a’ -> Lowercase - ’n’ -> Uppercase - ’g’ -> Lowercase - ’u’ -> Uppercase - ’a’ -> Lowercase - ’g’ -> Uppercase - ’e’ -> Lowercase - ’.’ -> Period (remain unchanged) Putting it all together gives us: **’PyThOn Is An AmAzInG PrOgRaMmInG LaNgUaGe.’* This is the sentence ’Python is an amazing programming language.’ rewritten using alternating capitalization rules.”<br><br><br>|
|Case 4: High Reward (High Quality), Low PPL (Overlap with Pre-training Data) User: “Considering the ingredients miso paste, onions, cheese, and crème fraîche, how can I create a sophisticated dish that also includes a crunchy texture?” Assistant: “Creating a sophisticated dish that features miso paste, onions, cheese, and crème fraîche with a crunchy texture can be both exciting and delicious. Here’s a step-by-step guide to making a Miso Caramelized Onion Tart with Crispy Panko Topping: Ingredients: - 2 large onions, thinly sliced - 2 tbsp miso paste (preferably white miso for a subtle flavor) - 1 tbsp olive oil - 1 tbsp butter - 1/2 cup crème fraîche - 1 cup shredded Gruyère or Comté cheese (or a mix of your favorite cheeses that melt well) - 1 sheet of puff pastry or shortcrust pastry (store-bought or homemade) - 1/2 cup panko breadcrumbs - 1 tbsp butter (for the panko topping) - Fresh herbs for garnish (optional, such as chives or thyme) - Salt and pepper to taste Instructions:<br><br>1. Prepare the Onions: 1. **Caramelize the Onions**: In a large skillet, heat the olive oil and butter over medium heat. Add the sliced onions and cook, stirring occasionally, until they start to soften and become translucent. 2. **Add Miso Paste**: Lower the heat to medium-low and add the miso paste. Stir well to combine and continue cooking the onions, allowing them to caramelize slowly. This will take about 20-30 minutes. Stir occasionally to prevent burning. Once caramelized, season with salt and pepper to taste.<br>2. Preheat the Oven: Preheat your oven to 375°F (190°C).<br>3. Prepare the Pastry: 1. **Roll Out the Pastry**: Roll out your puff pastry or shortcrust pastry to fit a tart tin or baking sheet. Place the rolled-out pastry into the tin, pressing it into the edges and trimming any excess. [...] Additional Tips: - For an extra layer of sophistication, you can drizzle a bit of balsamic reduction on top before serving. - Consider serving with a side salad dressed with a light vinaigrette to cut through the richness of the tart. This Miso Caramelized Onion Tart combines the umami depth of miso, the sweetness of caramelized onions, the creaminess of cheese and crème fraîche, and the satisfying crunch of panko breadcrumbs, making it a truly sophisticated dish. Enjoy!”<br>|

##### 5.2 Preference Tuning

In Sailor2, we perform the preference tuning to enhance model performance beyond supervised fine-tuning. This section first introduces the problem formulation of reinforcement learning from human feedback and the learning algorithms examined in this work

- (Sec. 5.2.1). We then describe the pipeline for constructing preference data in SEA languages
- (Sec. 5.2.2) and present the full recipe of the preference tuning (Sec. 5.2.3). In addition, we provide extensive ablation study results on preference data construction in Sec. 8.3.

##### 5.2.1 Background

In preference tuning, the preference data typically takes the form of pairwise preferences. Each prompt x is paired with two possible responses, y1 and y2. The human annotator (Christiano et al., 2017) or AI annotator (Lee et al., 2023) provides the preference feedback

- o(y1 ≻ y2|x) ∈ {0,1}, indicating whether y1 is preferred over y2. The preferred response is denoted as yw, while the other is denoted as yl.

Policy optimization algorithms. DPO (Rafailov et al., 2024) is introduced to optimize the policy model in an offline manner. Rafailov et al. (2024) demonstrates that it directly

- optimizes the RLHF objective using the following equivalent formulation:

πθ(yl|x) πref(yl|x)

πθ(yw|x) πref(yw|x) − β log

LDPO(πθ; πref) = −E(x,yw,yl)∼D log σ β log

. (1)

Unlike the classic RLHF pipeline (Ouyang et al., 2022) which first trains a reward model and then optimizes the policy using the trained RM, DPO optimizes the policy while simultaneously training an implicit reward model. This approach allows DPO to directly optimize the policy using preference pairs, thereby simplifying the preference-tuning pipeline. Recently, many variants have been proposed to improve the vanilla DPO algorithm (Meng et al., 2024; Mao et al., 2024; Azar et al., 2024; Ethayarajh et al., 2024). In this work, we explored three promising approaches including SimPO (Meng et al., 2024), length-normalized DPO (LNDPO) (Rafailov et al., 2024), and length-regularized DPO (LR-DPO) (Park et al., 2024). Our experiment results indicate that LR-DPO achieves a favorable balance between performance and verbosity. The objective of LR-DPO is defined as follows:

πθ(yw|x) πref(yw|x) − β log

πθ(yl|x) πref(yl|x)

LLR-DPO(πθ; πref)=−E(x,yw,y

l)∼D log σ β log

+ α|yw| − α|yl| . (2)

The additional length difference term serves as a regularizer, down-weighting the gradients of preference pairs in which the preferred response is longer, and vice versa. This mitigates length exploitation in preference tuning.

##### 5.2.2 Preference Data

Our preference tuning consists of two stages, training with off-policy responses generated by Llama-3-8B-Instruct and training with on-policy responses generated by Sailor2 suite. Additionally, we conduct the preference distillation from our 20B model to smaller models.

Off-policy Data. To construct the off-policy dataset, we first translate the UF-Llama3 preference dataset9 into SEA languages. Low-quality translations are filtered based on perplexity scores obtained from the Sailor2-8B base. The resulting off-policy dataset is a mixture of SEA languages and English. Note that GPT-4o struggles with translating extremely low-resource languages such as Lao and Khmer, often producing outputs with excessive emojis and improper formatting. We find these cases using a simple script and translate them into the target language using Deepseek-V3 (DeepSeek-AI et al., 2024), which has demonstrated superior performance as evaluated by Huang et al. (2025).

On-Policy Data. At this stage, we use the prompts from the off-policy dataset to generate responses with the corresponding model. These responses are scored by the open-source

9https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm

reward model, Skywork-Reward-Gemma-2-27B (Liu et al., 2024b), selecting the highest as chosen and the lowest as rejected. We also apply a language consistency verifier to correct input-output language mismatches (excluding translation tasks).

Preference Distillation. After off-policy training, our 1B–14B models are finetuned on the on-policy data from our 20B model, rather than using their own. This approach simplifies the training pipeline and reduces computational costs. Our ablation study (Sec. 8.3) shows that distillation yields comparable downstream DPO performance.

Probe Data. During development, we observed unexpected model behaviors that were not captured by standard evaluation suites. For example, an early version of Sailor2 frequently included emojis in its responses—an undesired trait in many use cases. However, since the presence of emojis alone did not significantly affect reward model scores, this issue remained undetected. To address this, we introduce probe data, a set of prompts designed to elicit specific behaviors. These prompts were selected from AlpacaEval 2 by identifying cases where an early model version produced emoji-containing responses. Using this probe data, we assessed whether our interventions effectively reduced emoji overuse.

More ablation studies in Sec. 8.3 analyze the impact of design choices in the preference data construction pipeline.

##### 5.2.3 Preference Tuning Recipe

Due to the absence of a high-quality reward model for PPO-based algorithms, we explore different direct alignment algorithms, such as DPO and its variants (SimPO (Meng et al.,

- 2024), LN-DPO (Rafailov et al., 2024), LR-DPO (Park et al., 2024)). LN-DPO optimizes the length-averaged log-probabilities, while LR-DPO explicitly introduces the response length as the regularizer in their objective. We extensively tuned hyperparameters and conducted ablation studies to optimize model performance. Table 7 summarizes the hyperparameter search space, and Table 8 lists the final preference tuning settings. LR-DPO, offering a good balance between performance and verbosity, was chosen to train our final models.

All experiments were conducted with the training framework, Oat (Liu et al., 2024d; 2025b), which enables large-scale and flexible training.

- Table 7: Hyperparameters of different algorithms for preference tuning. We explore hyperparameters suggested by prior work (Meng et al., 2024; Lambert et al., 2024).

Algorithm LR β Batch Size Method Specific

SimPO {5e-7, 1e-6} {2.5,10} 128 γ-β ratio: {0.3,0.5} LN-DPO 5e-7 {5,10,15} 128 DPO 5e-7 {0.01,0.1,0.3} 128 LR-DPO 5e-7 {0.01,0.1,0.3} 128 α: [1e-5,1e-2]

- Table 8: Final training hyperparameters for preference tuning. We utilize the lengthregularized DPO proposed by Park et al. (2024).

|Hyperparams|1B 3B 8B 14B 20B<br><br>|
|---|---|
|Learning Rate Learning Rate Schedule Batch Size Max Response Token Length KL Coefficient β Warm Up Ratio Number of Epochs Length-Regularized Coef. α|5e-7 cosine with min lr 128 2048 0.01<br><br>0.03<br>1<br><br><br>0.001 0.0002 0.01 0.0 0.003|

### 6 Model Customization

##### 6.1 Long-Context Training

A 128K token context window allows large language models (LLMs) to handle complex tasks such as multi-document question answering (Wang et al., 2024d), repository-level code comprehension (Jimenez et al., 2024), and many-shot learning by capturing long-range dependencies (Agarwal et al., 2024), leading to more coherent and contextually relevant outputs (Mazumder & Liu, 2022).

The Sailor2 series employs AnchorAttention (Wang et al., 2024b) to extend its maximum context length from 4K to 128K10. In particular, Sailor2 masks out cross-document attention to prevent the model from aggregating irrelevant information across irrelevant documents. Note, this strategy that aligns with the approach used during pretraining. By maintaining a consistent masking paradigm in both pretraining and long-context training, Sailor2 mitigates potential conflicts that could arise from shifting between different attention mechanisms.

Unlike approaches such as LLaMA3 (Dubey et al., 2024), which rely solely on crossdocument attention masking, Sailor2 introduces an anchor token that serves as a stable reference point. Specifically, the first token in each training sequence (the <eos> token of each sample) retains a fixed positional ID and is therefore visible to all documents within the training context. This design helps reduce numerical instability and provides the model with a consistent anchor across the extended sequence. Furthermore, instead of resetting the positional IDs to 0 for each new document (Zhao et al., 2024), Sailor2 maintains continuous positional indexing across the entire sequence, allowing the model to fully utilize the entire position range in training.

With AnchorAttention, Sailor2 efficiently achieves long-context capabilities while training on a relatively small amount of data. Specifically, Sailor2 uses a total of 4 billion tokens in 1,000 steps (4 million tokens per step) at a learning rate of 2 × 10−5, with the first 200 steps designated as warm-up. Despite the limited token budget, Sailor2 effectively extends its context length, as demonstrated by the model’s performance on the RULER benchmark (Hsieh et al., 2024), as shown in Table 9.

In the meanwhile, the short-context performance is kept and sometimes outperforms the pretrained one. The perplexity on different languages is in Table 10. The performance of different tasks is in Table 11.

Table 9: Model Performance on RULER Long-Context Benchmark. Model 128K 64K 32K 16K 8K 4K Qwen2.5-0.5B 0.00 0.00 46.50 52.65 55.95 64.42 Sailor2-1B 0.00 0.00 0.62 3.99 35.81 55.93 Sailor2-1B-32K 0.00 0.00 36.52 49.63 55.50 56.84 Qwen2.5-7B 20.67 61.70 78.58 81.72 83.58 86.72 SeaLLM-v3-7B 17.32 61.37 82.19 84.85 60.40 71.55 Sailor2-8B 0.00 2.17 9.59 23.08 49.13 69.38 Sailor2-8B-128K 19.94 41.57 54.61 64.32 75.73 80.04 Qwen2.5-14B 32.93 66.68 85.09 86.96 87.40 87.56 Sailor2-20B 0.55 14.08 46.60 67.76 79.62 87.86 Sailor2-20B-128K 47.46 66.70 79.52 85.24 86.63 88.21

10Long-context training codebase: https://github.com/haonan3/AnchorContext

Table 10: Perplexity across multiple languages for different Sailor2 models. Model eng tha vie ind mya valid Sailor2-1B 21.01 4.52 7.52 6.75 4.11 9.36 Sailor2-1B-32K 20.89 4.63 7.57 6.66 4.94 9.93 Sailor2-8B 13.63 3.51 5.49 5.09 2.74 6.26 Sailor2-8B-128K 13.28 3.53 5.22 5.07 2.64 6.22 Sailor2-20B 11.48 3.35 5.24 4.93 2.41 5.69 Sailor2-20B-128K 11.11 3.36 5.04 4.92 2.44 5.61

Table 11: Effect of Long-Context Training on Downstream Performance. Evaluate on Multiple-Choice tasks (Belebele, XCOPA, M3Exam) with Accuracy as metric, and Reading Comprehension tasks (XQuAD and TydiQA) with Exact/Fuzzy Match as metrics.

Belebele XCOPA XQuAD TydiQA M3Exam Model tha ind vie tha ind vie tha vie ind tha ind vie

Sailor2-1B 36.89 35.89 36.78 56.6 66.8 68.0 33.07 / 49.60 34.56 / 53.43 44.78 / 64.86 28.43 28.30 36.84 Sailor2-1B-32K 36.44 36.56 35.89 56.2 68.4 65.8 35.68 / 54.86 38.49 / 58.91 45.84 / 65.08 27.79 27.76 35.10

Sailor2-8B 43.22 48.89 48.67 66.4 74.8 81.0 66.84 / 80.50 60.05 / 79.61 66.37 / 81.30 56.50 57.14 65.62 Sailor2-8B-128K 44.00 50.44 48.89 66.8 79.0 81.2 67.10 / 80.28 59.02 / 79.15 65.31 / 82.01 55.39 60.11 65.01

Sailor2-20B 47.44 52.11 53.78 67.6 81.4 83.6 69.45 / 83.34 62.02 / 82.05 71.68 / 84.44 67.77 62.26 74.46 Sailor2-20B-128K 48.44 52.44 53.44 69.4 81.8 84.8 69.97 / 83.54 63.47 / 82.10 70.44 / 83.69 67.36 62.80 74.23

##### 6.2 Speculative Decoding

To accelerate model inference, we adopted speculative decoding, a technique designed to reduce the computational cost of autoregressive generation. Specifically, we customized a one-layer draft model, GliDe (Du et al., 2024)11, for Sailor 8B and 20B.

Background. GliDe is a draft model based on a transformer decoder-only architecture that retains standard components—self-attention, cross-attention, and feed-forward networks (FFNs). In GliDe, the conventional self-attention layer is applied first, where each token in the sequence attends only to its preceding tokens. This is immediately followed by a cross-attention layer, which reuses precomputed and cached cross-attention outputs from the target LLM instead of recomputing the keys and values for each draft token. This approach yields a more precise token representation while reducing redundant computations. Finally, the cross-attended outputs pass through position-wise FFNs to further refine token representations. The processing sequence follows: self-attention → cross-attention → FFN.

Implementation Details. Unlike GliDe, we share the weights of the embedding layer and LM head between the target and draft models, significantly reducing memory consumption, especially for large-vocabulary LLMs. Moreover, to improve the stability and robustness of the draft model, we employed a flash noise training technique to replace the cape mask in the original GliDe, which can not only solve the problem of training-inference discrepancy but also be compatible with Flash Attention (Dao et al., 2022). Specifically, for the cross-attention query Qt in the draft model, we can only ensure access to the corresponding key-value states K<t′, V<t′ that satisfy 1 ≤ |t′ − t| < γ, where γ denotes the number of speculative steps. During training, we randomly shift the indices of queries and key-value states within the range 1 ≤ j < γ. If the sequence length is l, we then compute O≥j = flash_attn Q≥j, K<l−j, V<l−j . This approach effectively enforces the

11Speculative decoding codebase: https://github.com/NonvolatileMemory/GliDe_with_a_CaPE_ ICML_24 (Training) and https://github.com/penghui-yang/sailor-glide (Inference)

same visibility constraints as those in the inference phase, i.e., 1 ≤ |t′ − t| < γ, thereby ensuring that the training process aligns with inference behavior.

During speculative decoding inference, we first generate tokens autoregressively using the one-layer draft model, followed by parallel verification with Sailor 2. This approach effectively reduces the number of autoregressive steps required for decoding. Since tree-based speculative decoding is incompatible with Flash Attention, we opted for a straightforward sequential speculative decoding strategy. We set the speculation length γ to 4 based on empirical observations.

Performance. The performance of our GliDe model is illustrated in Figure 5 and Figure 6, demonstrating an approximate 2× acceleration. Notably, for Burmese (mya), our approach achieves an accept length exceeding 3 and a speedup of approximately 2.5×. We attribute this improvement to the high tokenization granularity of Burmese, which provides a greater margin for speculative decoding to optimize token generation.

Accept Length Comparison across Languages

3.5

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| |8B<br><br>20B| | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

3.0

2.5

AcceptLength

2.0

1.5

1.0

0.5

0.0

ind tgl khm tha lao vie mya zsm all

Figure 5: Comparison of GliDe Accept Length in Different Languages.

Token Generation Speed across Languages

8B Vanilla

8B Glide

60

20B Vanilla

20B Glide

50

40

Tokens/Second

30

20

10

0

ind tgl khm tha lao vie mya zsm all

Figure 6: Comparison of GliDe Token Generation Speed in Different Languages.

##### 6.3 Model Pruning

By leveraging existing pre-trained models, the pruning method enables the rapid generation of smaller-scale models, which avoids the high costs of training from scratch. In this study, we apply the Sheared LLaMA method(Xia et al., 2023) to prune the Sailor2-20B and Sailor28B models, resulting in Sailor2-14B and Sailor2-3B respectively12. The pruning stage takes

- 6B tokens. Subsequently, we performed continual training using a dataset of 180B tokens to recover the models’ performance.

Background. The Sheared LLaMA method focuses on structured pruning to produce smaller yet competitive models from pre-trained larger models. It employs two main techniques: targeted structured pruning and dynamic batch loading. Targeted structured pruning compresses a model into a target architecture via L0-regularized binary mask learning. Lagrange multipliers are applied to enforce constraints on the target architecture, ensuring the pruned model adheres to the desired configuration while optimizing performance. Dynamic batch loading adjusts training data batches based on domain-specific loss reduction rates, enhancing data efficiency and accelerating convergence.

Implementation Details. In contrast to the Sheared LLaMA method, we introduce several optimizations in our approach. First, instead of pruning Multi-Head Attention as in the original method, we retain the Key and Value Heads in the Grouped Query Attention (Ainslie

- et al., 2023) structure, pruning only an equal number of Query Heads from each Query Head Group corresponding to the KV Heads. Second, we do not prune the layer dimension, as our preliminary experiments have shown that pruning the layer dimension leads to convergence difficulties. Instead, we focus on optimizing other dimensions (i.e., hidden dimension, head number, and intermediate dimension). Third, to maintain consistency between the hidden dimensions and the number of attention heads, the pruning options are limited. We recommend conducting ablation studies with minimal continual training to identify optimal configurations. Finally, during continual training, we have not used the dynamic batch loading strategy, as it is complex to divide the pretraining data into several domains explicitly. Instead, we directly sample from the Sailor2 Stage-2 training dataset, achieving promising results.

Performance. To obtain the final chat models, we train two pruned models using the Sailor2 post-training pipeline, resulting in Sailor2-3B-Chat and Sailor2-14B-Chat. The experimental results in Table 12 demonstrate that these pruned models significantly outperform the baseline Qwen2.5 in low-resource languages such as Khmer and Lao.

Table 12: Evaluation of Chat Model after Pruning.

Model SWB Score tha vie ind tgl zsm khm lao mya Qwen2.5-14B-Chat 0.30 0.40 0.40 0.23 0.35 0.20 0.21 0.12 0.30 Sailor2-14B-Chat 0.39 0.38 0.34 0.33 0.35 0.28 0.46 0.47 0.43 Qwen2.5-3B-Chat 0.16 0.14 0.21 0.18 0.08 0.16 0.06 0.06 0.04 Sailor2-3B-Chat 0.26 0.25 0.21 0.21 0.19 0.19 0.32 0.31 0.28

- 7 Evaluation

##### 7.1 Evaluation on Base Model

For base model evaluation, we focus on the basic language understanding task like sentence classification, and language generation task like question answering and machine translation. Specially, we evaluate Sailor2 on SailCompass (Guo et al., 2024a) evaluation suite and FLoRes-200 (NLLB Team, 2022) translation suite. To expand the evaluated language coverage, we choose the dataset in Indonesian, Thai, Vietnamese, Malay and Javanese.

12Model pruning codebase: https://github.com/princeton-nlp/LLM-Shearing

- Table 13: Overview of results on Sailor2, over both 8B and 20B models. The best performing model for each model size on each benchmark is bolded.

|Language Benchmark(eval)|Sailor28B<br><br>Qwen2.57B<br><br>Gemma29B<br><br>Lllama3.18B<br><br>SeaLLMv3-7B|Sailor220B<br><br>Qwen2.532B<br><br>Gemma227B<br><br>Llama3.170B<br><br>AyaExpanse32B<br><br>|
|---|---|---|
|Avg.|57.6 52.8 52.5 47.2 43.4<br><br>|62.8 59.1 61.8 61.2 51.1|
|Indonesian<br><br>IndoCulture(0 shot) TydiQA(3 shot) Belebele(3 shot)|73.4 58.7 65.6 56.7 53.0 66.4 63.5 65.5 63.4 65.5 48.9 49.3 50.7 46.8 30.6<br><br>|76.4 68.9 66.1 72.7 70.6 71.7 63.9 65.1 69.9 58.2 52.1 54.1 53.3 56.4 60.3<br><br>|
|Thai<br><br>MMLU(5 shot) M3Exam(5 shot) Belebele(3 shot)<br><br>|55.4 52.8 57.8 44.1 50.8 57.0 51.7 52.7 43.7 51.3 43.2 44.1 40.6 43.1 43.0<br><br>|66.3 70.7 62.5 67.1 39.6 69.3 69.2 57.0 63.7 38.6 47.4 49.4 46.0 52.3 45.3<br><br>|
|Vietnamese<br><br>VMLU(3 shot) M3Exam(3 shot) Belebele(3 shot)<br><br>|56.2 52.6 51.7 48.9 56.8 65.6 66.4 65.5 54.4 63.1 48.7 50.8 49.0 46.0 48.6<br><br>|65.9 64.9 59.1 63.9 65.9 74.6 77.3 68.6 68.9 63.2 53.8 54.6 52.0 61.8 58.3<br><br>|
|Malay Tatabahasa(3 shot)|67.3 41.5 53.6 42.9 37.4|67.3 50.4 58.6 58.3 48.1<br><br>|
|Javanese M3Exam(3 shot)|57.1 35.9 45.3 40.4 38.5|62.3 47.7 49.1 53.4 46.1<br><br>|
|Multiple<br><br>FLORES-200(3 shot) XCOPA(3 shot)|35.4 30.6 35.8 31.7 29.6 74.1 71.8 73.0 69.4 70.4<br><br>|35.8 34.3 36.6 36.5 35.7 77.5 77.3 75.3 79.8 72.1<br><br>|

For Indonesian, we choose IndoCulture (Koto et al., 2024), TydiQA (Clark et al., 2020), Belebele (Bandarkar et al., 2024). For Thai, we choose MMLU (Kydlíˇcek et al., 2024), M3Exam (Zhang et al., 2023)13 and Belebele. For Vietnamese, We choose VMLU 14, M3Exam and Belebele. For Malay, we choose Tatabahasa (Lovenia et al., 2024). For Javanese, we choose M3Exam. For all SEA languages, we choose FLoRes-200 (NLLB Team, 2022) and XCOPA (Ponti et al., 2020). We have more detailed comparison and analysis for translation in Section 8.4 and culture understanding in Section 8.5.

- Detailed results are presented in Table 13. We observe that both Sailor2-8B and Sailor2-20B exhibit the highest average performance within their respective parameter groups. Notably, Sailor2-20B even outperforms larger models, including the three-times larger Llama3.1-70B.

7.2 Evaluation on Chat Model

We aim to comprehensively evaluate the performance of our Chat Model by using WildBench (Lin et al., 2024) as the primary evaluation dataset. WildBench covers five tasks: Coding & Debugging, Information Seeking, Math & Data, Reasoning & Planning, and Creative Tasks. We employ GPT-4o-0806 to translate WildBench into eight SEA languages (Thai, Vietnamese, Indonesian, Tagalog, Burmese, Khmer, Lao, and Malay), thereby creating a new benchmark named SEA-WildBench (SWB).

- Detailed results are presented in Table 14 (task-level) and Table 15 (language-level). We use the SWB Score as our evaluation metric, which is calculated based on the win-rate against GPT-4o-0806 (the same model serves as the judge). We selected the most representative open models, including both general-purpose and SEA language-optimized variants. For improved visualization, we use Llama-3.1-70B-Instruct as the baseline, with a SWB Score of

30. Our results indicate that both Sailor2-20B-Chat and Sailor2-8B-Chat achieve superior performance across various tasks and languages. As shown in Table 15, Sailor2 models excel in low-resource languages. Notably, Sailor2-20B-Chat achieves nearly a 50% win rate against GPT-4o-0806 on SeaWildBench, demonstrating GPT-4o-level performance in local chat scenarios for Southeast Asian languages.

Note that the overall SWB Score can be higher than the scores for individual subsets. For example, although Llama-2-7B-Chat scores below 0.05 on each subset, its overall SWB Score is 0.05. We follow the WildBench score calculation 15. This method may overestimate scores in cases where parse errors occur.

13For Thai M3Exam, we adopt finetasks (Kydlícˇek et al., 2024) codebase for evaluation. 14VMLU: https://github.com/ZaloAI-Jaist/VMLU/ 15https://tinyurl.com/49en4cw6

- Table 14: Task-Level Evaluation of Chat Models on SEA-WildBench. The score represents the win rate against GPT-4o, which also serves as the evaluator. SWB Score is the average score of five tasks.

Model SWB Score Coding Creative Tasks Info Seeking Reasoning Math Length Sailor2-20B-Chat 0.56 0.62 0.56 0.58 0.57 0.54 2814.74 Sailor2-8B-Chat 0.49 0.42 0.57 0.53 0.50 0.42 2849.41 Qwen2.5-72B-Instruct 0.45 0.50 0.39 0.44 0.45 0.49 3026.82 SEA-LIONv3-70B-Instruct 0.40 0.42 0.38 0.40 0.39 0.39 2340.65 Gemma-2-27B-Instruct 0.40 0.38 0.41 0.39 0.39 0.37 2288.33 Qwen2.5-32B-Instruct 0.32 0.39 0.28 0.29 0.32 0.33 2090.61 Gemma-2-9B-Instruct 0.31 0.26 0.36 0.33 0.30 0.26 2163.03 Qwen2.5-14B-Instruct 0.30 0.33 0.25 0.28 0.28 0.30 2267.94 Llama-3.1-70B-Instruct 0.30 0.37 0.26 0.28 0.28 0.28 2543.06 SEA-LIONv3-8B-Instruct 0.30 0.32 0.32 0.30 0.28 0.22 2357.14 Aya-Expanse-32B 0.29 0.29 0.28 0.28 0.27 0.24 2495.47 Qwen2-72B-Instruct 0.26 0.22 0.27 0.28 0.25 0.23 1546.21 Qwen2.5-7B-Instruct 0.25 0.28 0.20 0.23 0.22 0.22 2415.08 SEA-LIONv2.1-8B-Instruct 0.23 0.23 0.24 0.24 0.20 0.18 1735.26 SeaLLMs-v3-7B-Chat 0.21 0.21 0.19 0.19 0.18 0.15 2298.47 Llama-3.1-8B-Instruct 0.19 0.18 0.15 0.16 0.15 0.13 2356.67 SeaLLM-7B-v2 0.18 0.14 0.16 0.17 0.14 0.12 2298.15 SeaLLM-7B-v2.5 0.17 0.14 0.14 0.15 0.13 0.11 2184.55 Qwen2.5-3B-Instruct 0.16 0.14 0.10 0.12 0.12 0.13 2324.08 Sailor-14B-Chat 0.16 0.07 0.11 0.13 0.10 0.09 2465.85 SeaLLM-7B-v1 0.12 0.03 0.07 0.09 0.07 0.06 2585.40 Mistral-7B-Instruct-v0.3 0.10 0.11 0.03 0.07 0.06 0.07 2336.51 Sailor-7B-Chat 0.09 0.02 0.04 0.06 0.04 0.03 1404.60 Llama-2-70B-Chat 0.08 0.07 0.05 0.06 0.05 0.05 2354.30 Llama-2-13B-Chat 0.06 0.04 0.04 0.05 0.03 0.03 2317.36 Llama-2-7B-Chat 0.05 0.03 0.02 0.04 0.02 0.03 2330.50

- Table 15: Language-Level Evaluation of Chat Models on SEA-Wildbench. The score represents the win rate against GPT-4o, which also serves as the evaluator. SWB Score is the average score of eight languages.

Model SWB Score tha vie ind tgl zsm khm lao mya Length Sailor2-20B-Chat 0.56 0.53 0.50 0.54 0.50 0.49 0.63 0.69 0.64 2814.74 Sailor2-8B-Chat 0.49 0.48 0.46 0.46 0.42 0.43 0.50 0.66 0.55 2849.41 Qwen2.5-72B-Instruct 0.45 0.54 0.51 0.51 0.42 0.48 0.33 0.41 0.31 3026.82 SEA-LIONv3-70B-Instruct 0.40 0.45 0.45 0.48 0.40 0.41 0.32 0.28 0.32 2340.65 Gemma-2-27B-Instruct 0.40 0.43 0.40 0.46 0.40 0.39 0.34 0.38 0.31 2288.33 Qwen2.5-32B-Instruct 0.32 0.37 0.42 0.42 0.26 0.38 0.24 0.19 0.16 2090.61 Gemma-2-9B-Instruct 0.31 0.36 0.40 0.39 0.30 0.38 0.19 0.19 0.19 2163.03 Qwen2.5-14B-Instruct 0.30 0.40 0.40 0.23 0.35 0.20 0.21 0.12 0.30 2267.94 Llama-3.1-70B-Instruct 0.30 0.33 0.37 0.37 0.28 0.35 0.18 0.15 0.19 2543.06 SEA-LIONv3-8B-Instruct 0.30 0.38 0.40 0.38 0.34 0.35 0.12 0.08 0.14 2357.14 Aya-expanse-32B 0.29 0.25 0.45 0.46 0.27 0.35 0.06 0.12 0.13 2495.47 Qwen2-72B-Instruct 0.26 0.26 0.30 0.33 0.29 0.32 0.20 0.20 0.16 1546.21 Qwen2.5-7B-Instruct 0.25 0.30 0.35 0.36 0.12 0.29 0.09 0.09 0.08 2415.08 Sealionv2.1-8B-Instruct 0.23 0.30 0.33 0.31 0.16 0.28 0.07 0.08 0.10 1735.26 SeaLLMs-v3-7B-Chat 0.21 0.23 0.22 0.21 0.19 0.16 0.15 0.16 0.09 2298.47 Llama-3.1-8B-Instruct 0.19 0.19 0.26 0.21 0.15 0.18 0.06 0.07 0.07 2356.67 SeaLLM-7B-v2 0.18 0.18 0.18 0.19 0.09 0.13 0.10 0.12 0.09 2298.15 SeaLLM-7B-v2.5 0.17 0.18 0.19 0.18 0.10 0.14 0.08 0.11 0.06 2184.55 Qwen2.5-3B-Instruct 0.16 0.14 0.21 0.18 0.08 0.16 0.06 0.06 0.04 2324.08 Sailor-14B-Chat 0.16 0.11 0.17 0.14 0.04 0.14 0.02 0.12 0.06 2465.85 SeaLLM-7B-v1 0.12 0.05 0.07 0.07 0.04 0.05 0.10 0.11 0.09 2585.40 Mistral-7B-Instruct-v0.3 0.10 0.07 0.11 0.07 0.08 0.11 0.02 0.03 0.02 2336.51 Sailor-7B-Chat 0.09 0.04 0.07 0.05 0.02 0.06 0.02 0.07 0.03 1404.60 Llama-2-70B-Chat 0.08 0.02 0.05 0.11 0.06 0.13 0.03 0.01 0.03 2354.30 Llama-2-13B-Chat 0.06 0.01 0.05 0.08 0.02 0.03 0.01 0.01 0.03 2317.36 Llama-2-7B-Chat 0.05 0.01 0.02 0.05 0.04 0.03 0.01 0.02 0.04 2330.50

- Table 16: Perplexity Comparison under Continual Pre-training: Qwen1.5 → Sailor and Qwen2.5 → Sailor2. Lower perplexity means better performance. The evaluation dataset for each language is composed of samples from various domains. The valid dataset refers to the evaluation data collected from all languages across these diverse domains.

Language eng zho tha vie ind mya valid

Model

- Qwen1.5-14B 13.67 12.30 18.06 42.88 14.15 13.65 21.71 Sailor-14B 15.02↑1.35 12.47↑0.17 6.19↓11.87 11.79↓31.09 7.21↓6.94 11.02↓2.63 9.65↓12.06

- Qwen2.5-14B 11.40 9.91 6.36 9.60 8.46 7.07 9.10 Sailor2-20B 11.48↑0.08 9.27↓0.64 3.35↓3.01 5.24↓4.36 4.93↓3.53 2.41↓4.66 5.69↓3.41

- Qwen1.5-7B 14.55 13.18 21.35 47.09 16.14 11.99 27.07 Sailor-7B 15.75↑1.20 13.57↑0.39 6.21↓15.14 11.05↓36.04 7.32↓8.82 12.25↑0.26 10.15↓16.92

- Qwen2.5-7B 13.17 11.61 7.24 12.84 10.07 8.69 10.73 Sailor2-8B 13.63↑0.46 10.94↓0.67 3.51↓3.73 5.49↓7.35 5.09↓4.98 2.74↓5.95 6.26↓4.47

- Qwen1.5-0.5B 23.78 21.48 58.8 200.65 65.44 16.07 70.41 Sailor-0.5B 25.88↑2.10 25.80↑4.32 8.72↓50.08 16.76↓183.89 10.98↓54.46 17.69↑1.62 17.26↓53.15

- Qwen2.5-0.5B 22.28 21.25 13.93 26.86 25.57 13.67 22.83 Sailor2-1B 21.01↓1.27 17.28↓3.97 4.52↓9.41 7.52↓19.34 6.75↓18.82 4.11↓9.56 9.36↓13.47

### 8 Analysis

This section presents our insights on building multilingual LLMs during both the continual pre-training and post-training stages. We also examine Sailor2’s capabilities in translation and cultural understanding, two core components of practical multilingual applications.

##### 8.1 Effect of Model Expansion

For both Sailor and Sailor2, we adopt a continual pre-training (CPT) approach to efficiently develop multilingual LLMs by reusing computational resources. Unlike Sailor, Sailor2 incorporates model expansion, which creates additional capacity for learning new knowledge from the multilingual corpus.

We analyze the perplexity shift during CPT with and without model expansion, with detailed results shown in Table 16. Experimental findings reveal that, compared to the

- Qwen1.5 → Sailor transition, Qwen2.5 → Sailor2 exhibits less degradation in English/Chinese and greater improvements in the target SEA languages during continual pre-training. Notably, even though Qwen2.5 is trained on 18T tokens versus Qwen1.5’s 4T tokens, Sailor2 still achieves significant gains in SEA languages with only minor degradation in English.

We conduct a comprehensive examination across multiple languages in Figure 11, which illustrates the PPL distribution shift for English, Chinese, and sixteen SEA languages. The results demonstrate that Sailor2 maintains its performance in English and Chinese while achieving significantly lower perplexity in SEA languages. Further discussions on future work in efficient CPT can be found in Section 9.3.

8.2 Effect of Continual Pre-training

- Qwen2.5 models have already been trained on 18T tokens, meaning that many SEA tokens were likely seen during the pre-training stage. This raises the question of whether the expensive continual pre-training stage using an additional 400B SEA tokens is still necessary. To investigate, we conducted an ablation study with the following setup: post-training on both the vanilla Qwen2.5-7B base model and the Sailor2-8B base model, with the same post-training dataset and training steps. Detailed results are listed in Table 17. We could observe that CPT is essential, especially for low-resource languages like khm and lao.

- Table 17: Language-wise Score on SEA-WildBench between chat models trained using the Qwen and Sailor2 models. Note that Qwen2.5-7B-Chat is trained using the Sailor2 post-training pipeline.

Model SWB Score tha vie ind tgl zsm khm lao mya Sailor2-8B-Chat 0.43 0.44 0.40 0.40 0.39 0.36 0.43 0.56 0.44 Qwen2.5-7B-Chat (ours) 0.25 0.31 0.34 0.33 0.21 0.31 0.11 0.21 0.08

##### 8.3 Key Findings in Preference Data Construction

We conduct a series of ablation studies to assess the impact of design choices in the preference data construction pipeline, as described in Sec. 5.2.2.

60

Length-ControlledWinRate(%)onSea-WB

56

Length-ControlledWinRate(%)onSea-WB

60

55

56

55

| | |
|---|---|
| | |

50

50

| | |
|---|---|
| | |

49

50

48

48

47

40

45

30

40

18

20

35

| | |
|---|---|
| | |

10

30

8B-beta=0.1 8B-beta=0.3 20B-beta=0.1

0

SFT DPO w. Off-Policy

DPO w. On-Policy

w/o lan consist.

w/ lan consist.

| |
|---|

Figure 7: Length-controlled win rates comparison after different preference tuning stages on Sea-WB.

Figure 8: Effect of the language consistency verifier on the model performance. The results show a consistent performance gain across various experiment settings.

First Off-Policy, Then On-Policy Training Improves DPO Performance. While prior studies (Guo et al., 2024b; Lambert et al., 2024) have shown that on-policy training in DPO leads to greater performance improvements than off-policy training, our preliminary findings suggest that directly applying on-policy training to an SFT model yields limited gains. We hypothesize that this is due to the SFT model’s insufficient ability to generate high-quality responses in chat tasks. To address this, we first perform off-policy training to initialize a stronger policy model before transitioning to on-policy training. In other words, our training pipeline consists of SFT, followed by off-policy DPO training, and then on-policy DPO training. As shown in Fig. 7, off-policy training on Sailor2-20B-SFT significantly enhances model performance, while subsequent on-policy training provides further improvements.

Language Consistency Verifier Improves Downstream DPO Performance. Due to the lack of a reward model for SEA languages, we use Skywork-Reward-Gemma-2-27B (Liu

- et al., 2024b) which is trained primarily on English data. To mitigate the RM’s limitations in evaluating SEA language responses, we introduce a language consistency verifier. The verifier 16 labels a response as true if its language matches the prompt; otherwise, it is labeled false. If at least two responses are language-consistent, we select the winning and losing responses based on the RM’s reward scores. If only one response is languageconsistent, it is chosen as the winning response. If none are language-consistent, the prompt is discarded.

16We use facebook/fasttext-language-identification for language verification.

We evaluate the verifier’s effectiveness under three settings: for the 8B model, we conduct on-policy training with β ∈ {0.1,0.3}; for the 20B model, we use β = 0.1. As shown in Fig. 8, incorporating the verifier consistently improves performance compared to training without it. Moving forward, further improvements in preference tuning for low-resource languages could be explored, such as training an RM specifically for SEA languages and leveraging self-alignment techniques (Chen et al., 2024; 2025; Kim et al., 2025).

- Table 18: Comparison of on-policy and distillation methods on Sailor2-8B after off-policy training. LC is short for the length-controlled win rate on Sea-WB.

Method LC Avg. Length

On-policy 0.49 2849 Distillation 0.48 2752

Distillation Reduces the Response Length while Maintaining Comparable Model Performance. We investigate the effectiveness of distillation within the same model family during DPO training given two practical considerations: (1) Distillation leverages the high-quality on-policy data used for training Sailor2-20B-Chat; (2) Reusing Sailor2-20B’s on-policy data significantly reduces the computational cost of data generation and reward evaluation. To assess its impact, we conduct a controlled experiment on our 8B model. Specifically, we conduct DPO with its own on-policy data and with the on-policy data from Sailor2-20B. Results in Table. 18 indicate that distillation maintains comparable model performance while reducing response length.

- 8.4 Cross-lingual Translation Ability of Sailor2

We analyze the performance of Sailor2-20B on Flores Plus (NLLB Team, 2022), a translation dataset covering over 200 languages. Since our focus is primarily on SEA languages, we limit the scope to a subset of the dataset containing SEA languages, Chinese and English.

- Table 19 compares Sailor2 and three baseline models on English-centric translation pairs.
- Table 20 (and Table 23, 24, 25 and 26) shows performance of Sailor2-20B (and other baselines) between all language pairs in Flores Plus17. We provide a visual comparison of Sailor2-20B against other baselines in Figure 13, 14, 15 and 16.

Sailor2: Excelling in Low-Resource Translation Despite having significantly fewer parameters, Sailor2-20B demonstrates remarkable capabilities in low-resource language translation. The superior performance extends to approximately 80% of low-resource language scenarios. As shown in Table 19, compared to Qwen2.5-32B and Qwen2.5-72B, Sailor2 consistently achieves higher chrF++ scores in low-resource language pairs, though showing slightly lower performance in English or Chinese translation. When compared with Llama3.1-70B, Sailor2-20B exhibits particular strengths in challenging low-resource scenarios, demonstrating significant advantages in Lao translation (7.3/37.6 vs 3.7/22.7), as well as when Burmese or Khmer is the target language. The performance gap becomes even more pronounced when examining bidirectional translation capabilities - Sailor2-20B maintains relatively balanced performance in both directions for low-resource languages, while other models show significant degradation when translating into low-resource languages. We also report the win rate of each model for each language pair in Figure 12. The win rate is defined as the percentage of times a model’s output achieves the top-1 ChrF++ score.

Translation Patterns and Language Effects Besides the performance of Sailor2, analysis of Table 20 also reveals several important patterns in translation behavior:

- (1) English demonstrates consistent superior performance across all language pairs, achieving higher chrF++ scores both as source and target language. This is clearly visible in Table 20, where English-sourced translations consistently achieve scores above 50 chrF++ points for most target languages, significantly higher than other source languages. The

17The prediction results of Sailor2 and baseline models could be found in https://huggingface.co/ datasets/sailor2/Flores-Plus-Evaluation-Log-Preview-Cleaned.

performance advantage is particularly pronounced in the XX → English direction compared to English → XX translations, as evidenced by the consistently higher scores in both tables.

- (2) We observe that translation quality appears more dependent on the target language than the source language, as shown in Table 20 where vertical columns (representing target languages) display more consistent score ranges compared to horizontal rows (source languages). For instance, translations into Vietnamese (vie) consistently fall within the 45-55 chrF++ range regardless of source language, while Vietnamese as a source language shows more variable performance depending on the target.
- (3) Languages within the same family or system exhibit notably higher translation performance, as demonstrated in Table 20 by the Cebuano-Tagalog pair (ceb-fil: 55.5/53.3). This pattern suggests that linguistic similarity plays a crucial role in translation quality, potentially through the model’s internal representation of language families. The table also reveals that geographically and culturally proximate languages, such as Indonesian-Malay pair (ind-zsm: 60.4/59.3), tend to achieve better bilateral translation performance compared to more distant language pairs. Table 19: Performance on Flores Plus for English-Centric Language Pairs (BLEU/chrF++)

- (a) English → XX

Language Sailor2-20B Llama3.1-70B Qwen2.5-72B Qwen2.5-32B

ace 4.8/26.1 6.3/30.3 4.1/24.8 4.6/24.0 ceb 30.9/57.7 28.1/54.5 18.0/44.4 13.6/39.4 cmn 11.3/32.9 12.4/32.3 12.5/34.3 13.0/34.0 fil 33.8/59.2 32.5/57.8 26.7/53.1 23.5/49.5 ilo 18.0/46.0 17.5/46.2 7.7/31.8 5.3/27.1 ind 43.6/67.7 42.8/67.3 41.9/66.5 38.7/64.0 jav 24.3/52.1 22.9/51.1 11.0/36.8 7.6/31.8 khm 5.4/29.2 3.3/21.5 3.0/21.9 3.6/20.6 lao 7.3/37.6 3.7/22.7 4.0/24.8 3.4/21.4 min 12.7/39.7 22.2/50.8 9.9/35.9 9.0/33.7 mya 3.0/28.1 2.0/21.3 2.1/19.9 1.8/17.3 sun 18.0/47.7 16.4/45.3 10.5/37.2 7.9/33.1 tam 10.4/41.1 9.3/37.9 5.1/31.2 4.1/28.3 tha 12.8/48.5 15.2/47.9 12.2/47.9 12.2/46.1 vie 42.6/60.7 43.0/61.1 42.5/60.8 40.6/59.2 war 25.6/53.3 25.6/52.8 16.4/43.7 11.9/38.4 zsm 41.2/66.8 40.2/66.2 35.5/62.4 32.0/59.5

- (b) XX → English

###### Language Sailor2-20B Llama3.1-70B Qwen2.5-72B Qwen2.5-32B

ace 21.9/42.9 24.5/45.2 19.3/40.2 15.8/37.8 ceb 46.7/67.3 44.8/65.2 39.0/60.1 34.0/56.0 cmn 32.4/57.8 34.2/59.0 35.1/59.9 33.6/59.0 fil 49.8/69.5 50.5/69.5 49.5/69.0 45.4/65.8 ilo 36.0/58.1 33.3/55.0 22.7/44.7 21.1/43.1 ind 45.2/67.0 47.1/68.5 46.9/68.7 45.1/67.5 jav 40.0/62.0 42.0/63.1 33.6/55.6 29.0/51.9 khm 36.2/59.6 32.0/55.4 28.7/52.3 25.2/49.8 lao 37.6/60.6 26.3/49.0 32.8/55.0 27.1/50.1 min 35.5/57.8 39.6/61.3 31.5/53.2 28.2/50.7 mya 27.2/52.0 26.7/51.0 19.6/43.1 16.2/39.6 sun 38.1/60.8 38.4/60.5 32.1/54.8 28.6/51.7 tam 32.0/56.5 34.3/58.2 26.6/50.4 24.0/47.9 tha 36.3/60.6 37.1/61.2 38.7/62.3 35.8/60.6 vie 39.1/62.0 40.7/63.0 41.8/63.9 39.4/62.5 war 47.5/67.7 46.6/66.4 38.5/59.6 35.1/56.8 zsm 46.6/68.1 48.0/69.4 47.2/68.8 45.1/67.2

Table 20: Performance on Flores Plus (chrF++) for Sailor2-20B

Target (→) Source (↓)

eng cmn ind tha vie zsm mya lao ceb ilo jav khm sun fil war tam

English (eng) - 32.9 67.7 48.5 60.7 66.8 28.1 37.6 57.7 46.0 52.1 29.2 47.7 59.2 53.3 41.1 Chinese (cmn) 57.8 - 53.6 41.8 52.7 53.0 24.6 30.8 47.7 39.0 42.6 25.1 39.4 48.1 42.8 34.5 Indonesian (ind) 67.0 28.1 - 45.4 55.8 59.3 26.1 34.5 51.3 42.3 49.2 26.6 46.0 54.1 46.6 37.1 Thai (tha) 60.6 27.5 56.7 - 53.5 56.0 25.5 33.9 49.0 41.0 44.7 26.1 41.3 50.8 44.4 35.4 Vietnamese (vie) 62.0 28.1 57.5 44.4 - 56.4 25.6 33.5 49.6 41.3 45.3 26.1 41.6 51.4 45.1 36.1 Malay (zsm) 68.1 28.7 60.4 44.9 55.6 - 26.4 35.3 51.5 42.5 49.4 26.6 44.6 54.1 47.3 37.4 Burmese (mya) 52.0 22.9 49.3 38.3 46.9 49.4 - 29.8 44.8 37.0 39.9 23.2 36.7 46.5 41.9 33.3 Lao (lao) 60.6 25.8 55.9 45.0 52.3 55.6 25.4 - 48.8 40.0 45.0 27.2 41.2 51.1 44.1 35.4 Cebuano (ceb) 67.3 28.6 57.7 43.6 53.5 57.3 26.5 33.3 - 44.5 45.8 26.4 41.5 55.5 51.7 36.9 Ilocano (ilo) 58.1 24.7 51.7 38.8 48.0 51.6 24.4 29.6 49.5 - 37.9 24.0 36.6 51.4 45.2 33.5 Javanese (jav) 62.0 26.4 58.7 41.6 51.3 56.4 25.2 32.5 48.3 38.2 - 25.9 42.4 51.8 42.6 35.1 Khmer (khm) 59.6 26.4 55.5 42.9 51.5 54.8 24.8 33.5 48.8 40.1 44.8 - 41.5 50.5 44.2 34.5 Sundanese (sun) 60.8 26.0 59.0 41.9 51.5 56.3 25.4 32.3 47.5 33.3 46.8 26.3 - 51.1 38.3 35.1 Tagalog (fil) 69.5 29.2 60.1 44.5 55.0 59.5 26.4 33.3 55.3 46.2 47.3 26.6 42.4 - 50.3 37.5 Waray (war) 67.7 27.2 57.3 43.6 53.0 57.4 26.3 33.2 56.8 45.8 44.4 25.5 39.9 55.9 - 36.8 Tamil (tam) 56.5 24.2 51.3 39.2 48.4 51.5 24.6 28.8 46.7 38.5 41.5 23.7 38.0 47.6 42.9 -

- 8.5 SEA Culture Understanding Ability of Sailor2

BLEND (3-shot EM)

60

53.12

51.79

51.20

Sailor2-20B

50

Qwen2.5-72B

50.04

Gemma2-27B

Sailor2-8B

46.54

Gemma2-9B

41.65

41.40 Sailor-7B 38.51

40

Qwen2.5-14B

Qwen2.5-32B

Gemma-7B 36.36

Mistral-7B 34.74

34.28 Qwen2.5-7B 34.13 SeaLLM-7B-Hybrid

30

Sailor2-1B

31.38

Sealion-7B 22.66 SeaLLMs-v3-7B

20

19.85

13.83

Qwen2.5-0.5B

10

0 1.5B 7B 20B 70B

(a) Results on BLEnD benchmark.

CulturalBench (3-shot Acc)

80

78.21

Qwen2.5-72B

71.90

70

Sailor2-20B

69.52

69.18

Gemma2-27B

Qwen2.5-14B 67.58 Qwen2.5-32B

Gemma2-9B 63.55

60

Qwen2.5-7B 60.85

60.16

Sailor2-8B

Sailor-7B 59.23 Gemma-7B 53.34 Mistral-7B 56.67

50

SeaLLMs-v3-7B 53.19

50.27

SeaLLM-7B-Hybrid

Qwen2.5-0.5B

42.23

40

38.60

Sailor2-1B

Sealion-7B 35.32

30

0 1.5B 7B 20B 70B

(b) Results on CulturalBench benchmark.

Figure 9: Performance comparison across models on BLEnD and CulturalBench benchmarks. Cultural understanding significantly influences the practical application and interaction quality of multilingual LLMs. To assess the cultural understanding capabilities of the Sailor2 series models in Southeast Asian contexts, we employ CulturalBench (Chiu et al., 2024), BLEND (Myung et al., 2024), and Global-MMLU (Singh et al., 2024a) as evaluation benchmarks, covering a total of seven languages.

Specifically, CulturalBench consists of single-choice and judgment questions in Filipino, Indonesian, Malaysian, Singaporean English, Thai, and Vietnamese. BLEND includes

question-and-answer tasks in Indonesian and West Java languages, while Global-MMLU comprises single-choice questions in Filipino, Indonesian, and Vietnamese. Notably, BLEND and Global-MMLU are multilingual evaluation datasets. To further refine the measurement of the cultural understanding ability of LLMs, we translate CulturalBench into a multilingual version using Google Translate (from English to local languages).

All evaluations are conducted using the 3-shot prompting approach. The experimental results are presented in Tables 27, 28, and 29. We summarize our results in Figure 9 and find that, among models of similar size, Sailor2 has a better understanding of SEA culture, including its cuisine, traditions, geography, and more. In Figure 10, we present sample responses of the Sailor2-20B model on CulturalBench.

|Paano madalas na bumabati ang mga tao sa Pilipinas nang hindi gumagamit ng mga salita?<br><br>(English Question: How do people in the Philippines often greet each other without<br><br>using words?)<br><br>A. Sa pagtaas ng kilay nila<br>B. Sa pamamagitan ng pagbibigay ng mahigpit na pagkakamay<br>C. Sa pagtango ng kanilang ulo<br>D. Sa pamamagitan ng pagbibigay ng high five.<br>| | |
|---|---|---|
|Prediction:| | |
|Golden: C|Qwen2.5-32B: B<br><br>[Figure 5]|Sailor2-20B: C<br><br>[Figure 6]|

|Masakan Indonesia populer apa yang menggunakan daging yang ditusuk dan dipanggang, disajikan dengan saus kental, pedas, dan berbahan dasar kunyit?<br><br>(English Question: What popular Indonesian dish uses skewered and grilled meat, served with a thick, spicy, and turmeric-based sauce?)<br><br>A. Sate Padang.<br>B. Sate Madura.<br>C. Steak Wagyu.<br>D. Iga BBQ.<br>| | |
|---|---|---|
|Prediction:| | |
|Golden: A|Qwen2.5-72B: B<br><br>[Figure 7]|Sailor2-20B: A<br><br>[Figure 8]|

Figure 10: Sample responses from CulturalBench.

Indonesian Culture Understanding Indonesian culture is uniquely rich and diverse, shaped by various ethnicities, languages, and historical influences. Evaluating this separately helps us assess the model’s ability to capture these distinct cultural nuances. We adopts two types of benchmark for evaluation: (1) General Knowledge, Local knowledge & Reasoning: IndoMMLU (Koto et al., 2023), IndoCareer (Koto, 2024); (2) Cultural Reasoning: IndoCulture (Koto et al., 2024), MAPS (Liu et al., 2024a), COPAL-ID (Wibowo et al., 2023), IndoCloze (Koto et al., 2022). As listed in Table 21, Sailor2 models present the good performance in understanding Indonesian culture and knowledge.

Table 21: Evaluation Results on Indonesian Culture (3-shot, Accuracy).

Model IndoMMLU IndoCareer IndoCulture MAPS COPAL-ID IndoCloze Avg Sailor2-20B 70.7 69.0 79.3 92.9 86.9 98.1 82.8 Sailor2-8B 64.5 60.9 74.7 90.8 86.5 96.5 78.9 Aya-Expanse-32b 59.3 61.5 71.4 91.4 84.6 97.0 77.5 Gemma-2-27b 57.1 61.6 70.2 92.0 81.8 96.5 76.5 Qwen2.5-32B 61.9 64.3 66.8 89.9 76.6 99.6 76.5 SEA-LIONv3-9B 60.0 58.9 65.8 90.2 80.0 96.7 75.2 Gemma-2-9B 57.9 57.8 67.9 89.9 79.3 95.8 74.7 Qwen2.5-7B 53.1 43.0 61.1 87.6 73.0 94.2 68.6 Qwen2.5-14B 52.6 64.1 50.7 76.0 62.8 99.8 67.6 Llama-3.1-8B 47.5 61.5 54.5 77.8 70.0 92.5 67.3 SeaLLMs-v3-7B 47.6 32.8 59.9 87.9 71.2 95.9 65.8 Llama-3-8B 48.8 39.6 55.0 83.1 66.8 87.2 63.4

### 9 Conclusion and Future Work

This report introduces the Sailor2 family of open models, designed to facilitate the development of large language models for Southeast Asian languages. We also summarize the key insights from our pipeline for building the Sailor2 model, covering data curation, continual pre-training, post-training, evaluation and advanced model customization. We hope this report will inspire the community to develop more inclusive and robust multilingual language models for underserved languages.

Looking ahead, we plan to expand our multilingual research to include a broader range of low-resource languages and explore more efficient model training approaches. The following sections detail our motivations and review the most relevant works in data curation, model design, and model training.

##### 9.1 Synthetic Data Curation for Supporting Low-resource Languages

Apart from a few high-resource languages, most languages have a relatively scarce supply of training tokens. For example, in Sailor2, six languages contain fewer than 1 billion training tokens (See Table 3 for detailed statistics). For extremely low-resource languages like Minangkabau (spoken in Indonesia by approximately 6.5 million people) and Acehnese (spoken in Indonesia by around 3.5 million people), we were only able to mine fewer than one million tokens18.

One effective way to address this issue is to leverage translated synthetic data. For example, Wang et al. (2024c) translate high-quality documents (e.g., Fineweb-Edu (Lozhkov et al., 2024)) from English into medium-level languages such as French, German, and Spanish. Similarly, Doshi et al. (2024) adopt the Translationese dataset to extend coverage to additional low-resource languages, including Hindi, Gujarati, and Marathi.

##### 9.2 Tokenizer-Free Model for Open-Vocabulary Learning

Recent studies demonstrate that tokenizer-free language models can effectively process unseen languages and exhibit greater robustness against noise attacks compared to tokenizerbased models.

One approach involves pixel-based language models (Lotz et al., 2023; Rust et al., 2023), which treat text as images. This enables them to learn any script and achieve openvocabulary language learning by exploiting visual similarities among characters and scripts through parameter sharing. In contrast, byte-level language models (Zheng et al., 2025; Kallini et al., 2024; Xue et al., 2022) bypass the tokenization step entirely by directly processing the raw character or byte stream as input.

These approaches offer significant benefits for both morphologically rich languages and languages that mix multiple scripts. For example, languages such as Turkish, Finnish, and Hungarian are known for their complex morphological structures, while Japanese (which combines Kanji, Hiragana, and Katakana) and Hindi (which often integrates Devanagari and Latin scripts) frequently mix scripts in everyday usage.

##### 9.3 Efficient Continual Pre-training for Multilingual Model

Continual pre-training is more efficient and cost-effective than training from scratch when building new multilingual language models for target languages. By leveraging an existing base model, its inherent capabilities are preserved while saving computational investment. For instance, developers can use Sailor2 as a foundation to build more powerful models for Southeast Asian languages with their in-house data. Moreover, both the developing infrastructure (e.g., Sailor2 open source every details) and the model tokenizer (the most critical component for multilingual as verified by Tao et al. (2024)) are mature.

18See full list of undetermined (und) data in https://huggingface.co/datasets/HuggingFaceFW/ fineweb-2 for more low-resource languages.

However, our concern is that many existing open models might be over-trained, leaving little room for further fine-tuning. For example, Llama3.1 was trained on 15T tokens and Qwen2.5 on 18T tokens. Although Sailor2 employs model expansion to mitigate this issue, its approach remains inefficient due to the increased computational cost associated with a larger model size.

To achieve more efficient continual pre-training, we propose exploring the following directions: (1) Employing an over-training indicator (Ouyang et al., 2024) to guide the selection of an appropriate base model. (2) Enhancing model plasticity (Chen et al., 2023) to enable the model to absorb additional multilingual knowledge. (3) Leveraging insights from the lottery ticket hypothesis to update only the most essential parameters (Yuan et al., 2024).

### Contributors and Acknowledgements

Core Contributors Longxu Dou, Qian Liu, Fan Zhou, Changyu Chen, Zili Wang, Ziqi Jin, Zichen Liu, Tongyao Zhu, Cunxiao Du, Penghui Yang, Haonan Wang, Jiaheng Liu, Yongchi Zhao, Xiachong Feng, Xin Mao, Man Tsung Yeung

Contributors Kunat Pipatanakul, Fajri Koto, Min Si Thu, Hynek Kydlícˇek, Zeyi Liu, Qunshu Lin, Sittipong Sripaisarnmongkol, Kridtaphad Sae-Khow, Nirattisai Thongchim, Taechawat Konkaew, Narong Borijindargoon, Anh Dao, Matichon Maneegard, Phakphum Artkaew, Zheng-Xin Yong, Quan Nguyen, Wannaphong Phatthiyaphaibun, Hoang H. Tran, Mike Zhang, Shiqi Chen

Advisors Tianyu Pang, Chao Du, Xinyi Wan, Wei Lu, Min Lin Acknowledgements

- • Model Training: Qwen-Team, MegatronLLM
- • Model Evaluation: Finetasks, SeaCrowd
- • Model Deployment: AK(@_akhaliq), Ollama
- • Sea SRE Team: Zhikai Huang

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Rishabh Agarwal, Avi Singh, Lei M Zhang, Bernd Bohnet, Stephanie Chan, Ankesh Anand, Zaheer Abbas, Azade Nova, John D Co-Reyes, Eric Chu, et al. Many-shot in-context learning. arXiv preprint arXiv:2404.11018, 2024.

AI Singapore. Sea-lion (southeast asian languages in one network): A family of large language models for southeast asia. https://github.com/aisingapore/sealion, 2024.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pp. 4447–4455. PMLR, 2024.

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. The belebele benchmark: a parallel reading comprehension dataset in 122 language variants. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 749–775, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.44.

Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von

Werra. Cosmopedia, February 2024. BritLLM. Britllm. https://llm.org.uk/, 2024. Andrei Z. Broder. On the resemblance and containment of documents. Proceedings. Com-

pression and Complexity of SEQUENCES 1997 (Cat. No.97TB100171), pp. 21–29, 1997. URL https://api.semanticscholar.org/CorpusID:11748509.

Changyu Chen, Zichen Liu, Chao Du, Tianyu Pang, Qian Liu, Arunesh Sinha, Pradeep Varakantham, and Min Lin. Bootstrapping language models with dpo implicit rewards. In The Thirteenth International Conference on Learning Representations, 2025.

Yihong Chen, Kelly Marchisio, Roberta Raileanu, David Ifeoluwa Adelani, Pontus Stenetorp, Sebastian Riedel, and Mikel Artetx. Improving language plasticity via pretraining with active forgetting. ArXiv, abs/2307.01163, 2023. URL https://api.semanticscholar.org/ CorpusID:259317182.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play finetuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

Yu Ying Chiu, Liwei Jiang, Bill Yuchen Lin, Chan Young Park, Shuyue Stella Li, Sahithya Ravi, Mehar Bhatia, Maria Antoniak, Yulia Tsvetkov, Vered Shwartz, et al. Culturalbench: a robust, diverse and challenging benchmark on measuring the (lack of) cultural knowledge of llms. arXiv preprint arXiv:2410.02677, 2024.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Jonathan H Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Association for Computational Linguistics, 8:454–470, 2020.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In Advances in Neural Information Processing Systems, volume 35, pp. 16344–16359, 2022.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report, 2024. URL https://arxiv.org/abs/2412.19437.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.

Meet Doshi, Raj Dabre, and Pushpak Bhattacharyya. Pretraining language models using translationese. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5843–5862, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.334. URL https://aclanthology.org/2024.emnlp-main.

334/.

Longxu Dou, Qian Liu, Guangtao Zeng, Jia Guo, Jiahui Zhou, Xin Mao, Ziqi Jin, Wei Lu, and Min Lin. Sailor: Open Language Models for South-East Asia. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2024.

Cunxiao Du, Jing Jiang, Xu Yuanchen, Jiawei Wu, Sicheng Yu, Yongqi Li, Shenggui Li, Kai Xu, Liqiang Nie, Zhaopeng Tu, et al. GliDe with a CaPE: A low-hassle method to accelerate speculative decoding. In Proceedings of the Forty-first International Conference on Machine Learning, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In Forty-first International Conference on Machine Learning, 2024.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.

Jia Guo, Longxu Dou, Guangtao Zeng, Stanley Kok, Wei Lu, and Qian Liu. Sailcompass: Towards reproducible and robust evaluation for southeast asian languages. arXiv preprint arXiv:2412.01186, 2024a.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, et al. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792, 2024b.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Siming Huang, Tianhao Cheng, Jason Klein Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, J. H. Liu, Chenchen Zhang, Linzheng Chai, Ruifeng Yuan, Zhaoxiang Zhang, Jie Fu, Qian Liu, Ge Zhang, Zili Wang, Yuan Qi, Yinghui Xu, and Wei Chu. Opencoder: The open cookbook for top-tier code large language models. arXiv preprint arXiv:2411.04905, 2024. URL https://arxiv.org/pdf/2411.04905.

Xu Huang, Wenhao Zhu, Hanxu Hu, Conghui He, Lei Li, Shujian Huang, and Fei Yuan. Benchmax: A comprehensive multilingual evaluation suite for large language models. arXiv preprint arXiv:2502.07346, 2025.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VTF8yNQM66.

Julie Kallini, Shikhar Murty, Christopher D Manning, Christopher Potts, and Róbert Csordás. Mrt5: Dynamic token merging for efficient byte-level language models. arXiv preprint arXiv:2410.20771, 2024.

Dongyoung Kim, Jaehyung Kim, Kimin Lee, and Jinwoo Shin. Spread preference annotation: Direct preference judgment for efficient llm alignment. In The Thirteenth International Conference on Learning Representations, 2025.

Fajri Koto. Cracking the code: Multi-domain llm evaluation on real-world professional exams in indonesia. arXiv preprint arXiv:2409.08564, 2024.

Fajri Koto, Timothy Baldwin, and Jey Han Lau. Cloze evaluation for deeper understanding of commonsense stories in Indonesian. In Antoine Bosselut, Xiang Li, Bill Yuchen Lin, Vered Shwartz, Bodhisattwa Prasad Majumder, Yash Kumar Lal, Rachel Rudinger, Xiang Ren, Niket Tandon, and Vilém Zouhar (eds.), Proceedings of the First Workshop on Commonsense Representation and Reasoning (CSRR 2022), pp. 8–16, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.csrr-1.2. URL https://aclanthology.org/2022.csrr-1.2/.

Fajri Koto, Nurul Aisyah, Haonan Li, and Timothy Baldwin. Large language models only pass primary school exams in Indonesia: A comprehensive test on IndoMMLU. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 12359–12374, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.760. URL https://aclanthology.org/2023.emnlp-main.760/.

Fajri Koto, Rahmad Mahendra, Nurul Aisyah, and Timothy Baldwin. Indoculture: Exploring geographically influenced cultural commonsense reasoning across eleven indonesian provinces. Transactions of the Association for Computational Linguistics, 12:1703–1719, 2024.

Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Christopher A. ChoquetteChoo, Katherine Lee, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, and Orhan Firat. Madlad-400: A multilingual and document-level large audited dataset, 2023.

Hynek Kydlíˇcek, Guilherme Penedo, Clémentine Fourier, Nathan Habib, and Thomas Wolf. Finetasks: Finding signal in a haystack of 200+ multilingual tasks, 2024. URL https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\" ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Ren Lu, Thomas Mesnard, Johan Ferret, Colton Bishop, Ethan Hall, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023.

Jure Leskovec, Anand Rajaraman, and Jeffrey D. Ullman. Mining of Massive Datasets. Cambridge University Press, 2nd edition, 2014. ISBN 978-1107077232. URL http: //www.mmds.org/.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794, 2024.

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. Wildbench: Benchmarking llms with challenging tasks from real users in the wild, 2024. URL https://arxiv.org/abs/2406.04770.

Chaoqun Liu, Wenxuan Zhang, Jiahao Ying, Mahani Aljunied, Anh Tuan Luu, and Lidong Bing. Seaexam and seabench: Benchmarking llms with local multilingual questions in southeast asia. arXiv preprint arXiv:2502.06298, 2025a.

Chen Liu, Fajri Koto, Timothy Baldwin, and Iryna Gurevych. Are multilingual LLMs culturally-diverse reasoners? an investigation into multicultural proverbs and sayings. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 2016–2039, Mexico City, Mexico, June 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.112. URL https://aclanthology.org/2024.naacl-long.112/.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024b.

Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin. Regmix: Data mixture as regression for language model pre-training. arXiv preprint arXiv:2407.01492, 2024c.

Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, et al. Llm360: Towards fully transparent open-source llms. arXiv preprint arXiv:2312.06550, 2023.

Zichen Liu, Changyu Chen, Chao Du, Wee Sun Lee, and Min Lin. Sample-efficient alignment for llms. arXiv preprint arXiv:2411.01493, 2024d.

Zichen Liu, Changyu Chen, Chao Du, Wee Sun Lee, and Min Lin. Oat: A research-friendly framework for llm online alignment. [https://github.com/sailsg/oat](https://github.com/sail-sg/oat), 2025b.

Jonas Lotz, Elizabeth Salesky, Phillip Rust, and Desmond Elliott. Text rendering strategies for pixel language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 10155– 10172, Singapore, December 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.emnlp-main.628/.

Holy Lovenia, Rahmad Mahendra, Salsabil Maulana Akbar, Lester James V Miranda, Jennifer Santoso, Elyanah Aco, Akhdan Fadhilah, Jonibek Mansurov, Joseph Marvin Imperial, Onno P Kampman, et al. Seacrowd: A multilingual multimodal data hub and benchmark suite for southeast asian languages. arXiv preprint arXiv:2406.10118, 2024.

Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu: the finest collection of educational content, 2024. URL https://huggingface.co/datasets/ HuggingFaceFW/fineweb-edu.

Risto Luukkonen, Ville Komulainen, Jouni Luoma, Anni Eskelinen, Jenna Kanerva, HannaMari Kupari, Filip Ginter, Veronika Laippala, Niklas Muennighoff, Aleksandra Piktus, Thomas Wang, Nouamane Tazi, Teven Scao, Thomas Wolf, Osma Suominen, Samuli Sairanen, Mikko Merioksa, Jyrki Heinonen, Aija Vahtola, Samuel Antao, and Sampo Pyysalo. FinGPT: Large generative models for a small language. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 2710–2726, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.164. URL https:

//aclanthology.org/2023.emnlp-main.164.

Xin Mao, Feng-Lin Li, Huimin Xu, Wei Zhang, Wang Chen, and Anh Tuan Luu. As simple as fine-tuning: Llm alignment via bidirectional negative feedback loss. arXiv preprint arXiv:2410.04834, 2024.

Sahisnu Mazumder and Bing Liu. Lifelong and continual learning dialogue systems. arXiv preprint arXiv:2211.06553, 2022.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734, 2024.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786, 2022.

Junho Myung, Nayeon Lee, Yi Zhou, Jiho Jin, Rifki Afina Putri, Dimosthenis Antypas, Hsuvas Borkakoty, Eunsu Kim, Carla Perez-Almendros, Abinew Ali Ayele, et al. Blend: A benchmark for llms on everyday knowledge in diverse cultures and languages. arXiv preprint arXiv:2406.09948, 2024.

Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–15, 2021.

Xuan-Phi Nguyen, Wenxuan Zhang, Xin Li, Mahani Aljunied, Zhiqiang Hu, Chenhui Shen, Yew Ken Chia, Xingxuan Li, Jianyu Wang, Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang, Chaoqun Liu, Hang Zhang, and Lidong Bing. SeaLLMs - large language models for Southeast Asia. In Yixin Cao, Yang Feng, and Deyi Xiong (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 294–304, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-demos.28. URL https:// aclanthology.org/2024.acl-demos.28.

NLLB Team. No language left behind: Scaling human-centered machine translation, 2022.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Xu Ouyang, Tao Ge, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, and Dong Yu. Low-bit quantization favors undertrained llms: Scaling laws for quantized llms with 100t training tokens. arXiv preprint arXiv:2411.17691, 2024.

Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. Disentangling length from quality in direct preference optimization. arXiv preprint arXiv:2403.19159, 2024.

Guilherme Penedo, Hynek Kydlícˇek, Vinko Sabolcˇec, Bettina Messmer, Negar Foroutan, Martin Jaggi, Leandro von Werra, and Thomas Wolf. Fineweb2: A sparkling update with 1000s of languages, December 2024. URL https://huggingface.co/datasets/HuggingFaceFW/ fineweb-2.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vuli´c, and Anna Korhonen. Xcopa: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2362–2376, 2020.

Penghui Qi, Xinyi Wan, Guangxing Huang, and Min Lin. Zero bubble pipeline parallelism. arXiv preprint arXiv:2401.10241, 2023.

Penghui Qi, Xinyi Wan, Nyamdavaa Amar, and Min Lin. Pipeline parallelism with controllable memory. arXiv preprint arXiv:2405.15362, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Phillip Rust, Jonas F. Lotz, Emanuele Bugliarello, Elizabeth Salesky, Miryam de Lhoneux, and Desmond Elliott. Language modelling with pixels. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= FkSp8VW8RjH.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili’c, Daniel Hesslow, Roman Castagn’e, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurenccon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa Etxabe, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris C. Emezue, Christopher Klamm, Colin Leong, Daniel Alexander van Strien, David Ifeoluwa Adelani, Dragomir R. Radev, Eduardo Gonz’alez Ponferrada, Efrat Levkovizh, Ethan Kim, Eyal Natan, Francesco De Toni, Gérard Dupont, Germán Kruszewski, Giada Pistilli, Hady ElSahar, Hamza Benyamina, Hieu Trung Tran, Ian Yu, Idris Abdulmumin, Isaac Johnson, Itziar Gonzalez-Dios, Javier de la Rosa, Jenny Chim, Jesse Dodge, Jian Zhu, Jonathan Chang, Jorg Frohberg,

Josephine Tobing, Joydeep Bhattacharjee, Khalid Almubarak, Kimbo Chen, Kyle Lo, Leandro von Werra, Leon Weber, Long Phan, Loubna Ben Allal, Ludovic Tanguy, Manan Dey, Manuel Romero Muñoz, Maraim Masoud, María Grandury, Mario vSavsko, Max Huang, Maximin Coavoux, Mayank Singh, Mike Tian-Jian Jiang, Minh Chien Vu, Moham mad A. Jauhar, Mustafa Ghaleb, Nishant Subramani, Nora Kassner, Nurulaqilla Khamis, Olivier Nguyen, Omar Espejel, Ona de Gibert, Paulo Villegas, Peter Henderson, Pierre Colombo, Priscilla Amuok, Quentin Lhoest, Rheza Harliman, Rishi Bommasani, Roberto L’opez, Rui Ribeiro, Salomey Osei, Sampo Pyysalo, Sebastian Nagel, Shamik Bose, Shamsuddeen Hassan Muhammad, Shanya Sharma, S. Longpre, Somaieh Nikpoor, S. Silberberg, Suhas Pai, Sydney Zink, Tiago Timponi Torrent, Timo Schick, Tristan Thrush, Valentin Danchev, Vassilina Nikoulina, Veronika Laippala, Violette Lepercq, Vrinda Prabhu, Zaid Alyafeai, Zeerak Talat, Arun Raja, Benjamin Heinzerling, Chenglei Si, Elizabeth Salesky, Sabrina J. Mielke, Wilson Y. Lee, Abheesht Sharma, Andrea Santilli, Antoine Chaffin, Arnaud Stiegler, Debajyoti Datta, Eliza Szczechla, Gunjan Chhablani, Han Wang, Harshit Pandey, Hendrik Strobelt, Jason Alan Fries, Jos Rozen, Leo Gao, Lintang Sutawika, M Saiful Bari, Maged S. Al-Shaibani, Matteo Manica, Nihal V. Nayak, Ryan Teehan, Samuel Albanie, Sheng Shen, Srulik Ben-David, Stephen H. Bach, Taewoon Kim, Tali Bers, Thibault Févry, Trishala Neeraj, Urmish Thakker, Vikas Raunak, Xiang Tang, Zheng-Xin Yong, Zhiqing Sun, Shaked Brody, Y Uri, Hadar Tojarieh, Adam Roberts, Hyung Won Chung, Jaesung Tae, Jason Phang, Ofir Press, Conglong Li, Deepak Narayanan, Hatim Bourfoune, Jared Casper, Jeff Rasley, Max Ryabinin, Mayank Mishra, Minjia Zhang, Mohammad Shoeybi, Myriam Peyrounette, Nicolas Patry, Nouamane Tazi, Omar Sanseviero, Patrick von Platen, Pierre Cornette, Pierre Franccois Lavall’ee, Rémi Lacroix, Samyam Rajbhandari, Sanchit Gandhi, Shaden Smith, Stéphane Requena, Suraj Patil, Tim Dettmers, Ahmed Baruwa, Amanpreet Singh, Anastasia Cheveleva, Anne-Laure Ligozat, Arjun Subramonian, Aur’elie N’ev’eol, Charles Lovering, Daniel H Garrette, Deepak R. Tunuguntla, Ehud Reiter, Ekaterina Taktasheva, Ekaterina Voloshina, Eli Bogdanov, Genta Indra Winata, Hailey Schoelkopf, Jan-Christoph Kalo, Jekaterina Novikova, Jessica Zosa Forde, Xiangru Tang, Jungo Kasai, Ken Kawamura, Liam Hazan, Marine Carpuat, Miruna Clinciu, Najoung Kim, Newton Cheng, Oleg Serikov, Omer Antverg, Oskar van der Wal, Rui Zhang, Ruochen Zhang, Sebastian Gehrmann, Shachar Mirkin, S. Osher Pais, Tatiana Shavrina, Thomas Scialom, Tian Yun, Tomasz Limisiewicz, Verena Rieser, Vitaly Protasov, Vladislav Mikhailov, Yada Pruksachatkun, Yonatan Belinkov, Zachary Bamberger, Zdenˇek Kasner, Zdenˇek Kasner, Amanda Pestana, Amir Feizpour, Ammar Khan, Amy Faranak, Ananda Santa Rosa Santos, Anthony Hevia, Antigona Unldreaj, Arash Aghagol, Arezoo Abdollahi, Aycha Tammour, Azadeh HajiHosseini, Bahareh Behroozi, Benjamin Ayoade Ajibade, Bharat Kumar Saxena, Carlos Muñoz Ferrandis, Danish Contractor, David M. Lansky, Davis David, Douwe Kiela, Duong Anh Nguyen, Edward Tan, Emi Baylor, Ezinwanne Ozoani, Fatim Tahirah Mirza, Frankline Ononiwu, Habib Rezanejad, H.A. Jones, Indrani Bhattacharya, Irene Solaiman, Irina Sedenko, Isar Nejadgholi, Jan Passmore, Joshua Seltzer, Julio Bonis Sanz, Karen Fort, Lívia Dutra, Mairon Samagaio, Maraim Elbadri, Margot Mieskes, Marissa Gerchick, Martha Akinlolu, Michael McKenna, Mike Qiu, Muhammed Ghauri, Mykola Burynok, Nafis Abrar, Nazneen Rajani, Nour Elkott, Nourhan Fahmy, Olanrewaju Samuel, Ran An, R. P. Kromann, Ryan Hao, Samira Alizadeh, Sarmad Shubber, Silas L. Wang, Sourav Roy, Sylvain Viguier, Thanh-Cong Le, Tobi Oyebade, Trieu Nguyen Hai Le, Yoyo Yang, Zach Nguyen, Abhinav Ramesh Kashyap, Alfredo Palasciano, Alison Callahan, Anima Shukla, Antonio Miranda-Escalada, Ayush Kumar Singh, Benjamin Beilharz, Bo Wang, Caio Matheus Fonseca de Brito, Chenxi Zhou, Chirag Jain, Chuxin Xu, Clémentine Fourrier, Daniel Le’on Perin’an, Daniel Molano, Dian Yu, Enrique Manjavacas, Fabio Barth, Florian Fuhrimann, Gabriel Altay, Giyaseddin Bayrak, Gully Burns, Helena U. Vrabec, Iman I.B. Bello, Isha Dash, Ji Soo Kang, John Giorgi, Jonas Golde, José D. Posada, Karthi Sivaraman, Lokesh Bulchandani, Lu Liu, Luisa Shinzato, Madeleine Hahn de Bykhovetz, Maiko Takeuchi, Marc Pàmies, María Andrea Castillo, Marianna Nezhurina, Mario Sanger, Matthias Samwald, Michael Cullan, Michael Weinberg, M Wolf, Mina Mihaljcic, Minna Liu, Moritz Freidank, Myungsun Kang, Natasha Seelam, Nathan Dahlberg, Nicholas Michio Broad, Nikolaus Muellner, Pascale Fung, Patricia Haller, Patrick Haller, Renata Eisenberg, Robert Martin, Rodrigo Canalli, Rosaline Su, Ruisi Su, Samuel Cahyawijaya, Samuele Garda, Shlok S Deshmukh, Shubhanshu Mishra, Sid Kiblawi, Simon Ott, Sinee Sang-aroonsiri,

Srishti Kumar, Stefan Schweter, Sushil Pratap Bharati, Tanmay Laud, Théo Gigant, Tomoya Kainuma, Wojciech Kusa, Yanis Labrak, Yashasvi Bajaj, Y. Venkatraman, Yifan Xu, Ying Xu, Yu Xu, Zhee Xao Tan, Zhongli Xie, Zifan Ye, Mathilde Bras, Younes Belkada, and Thomas Wolf. Bloom: A 176b-parameter open-access multilingual language model. ArXiv, abs/2211.05100, 2022. URL https://api.semanticscholar.org/CorpusID:253420279.

Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, William Marshall, Gurpreet Gosal, Cynthia Liu, Zhiming Chen, et al. Jais and jais-chat: Arabic-centric foundation and instruction-tuned open generative large language models. arXiv preprint arXiv:2308.16149, 2023.

Shivalika Singh, Angelika Romanou, Clémentine Fourrier, David I Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, et al. Global mmlu: Understanding and addressing cultural and linguistic biases in multilingual evaluation. arXiv preprint arXiv:2412.03304, 2024a.

Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, et al. Aya dataset: An open-access collection for multilingual instruction tuning. arXiv preprint arXiv:2402.06619, 2024b.

Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, and Ngai Wong. Scaling laws with vocabulary: Larger models deserve larger vocabularies. arXiv preprint arXiv:2407.13623, 2024.

Ahmet Üstün, Viraat Aryabumi, Zheng Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. Aya model: An instruction finetuned open-access multilingual language model. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15894–15939, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.845. URL https://aclanthology.org/2024.acl-long.845.

Bin Wang, Zhengyuan Liu, Xin Huang, Fangkai Jiao, Yang Ding, Ai Ti Aw, and Nancy F Chen. Seaeval for multilingual foundation models: From cross-lingual alignment to cultural reasoning. NAACL, 2024a.

Haonan Wang, Qian Liu, Chao Du, Tongyao Zhu, Cunxiao Du, Kenji Kawaguchi, and Tianyu Pang. When precision meets position: Bfloat16 breaks down rope in long-context training. arXiv preprint arXiv:2411.13476, 2024b.

Jiayi Wang, Yao Lu, Maurice Weber, Max Ryabinin, Yihong Chen, Raphael Tang, and Pontus Stenetorp. Multilingual pretraining using a large corpus machine-translated from a single source language. ArXiv, abs/2410.23956, 2024c. URL https://api.semanticscholar. org/CorpusID:273706997.

Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, et al. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa. arXiv preprint arXiv:2406.17419, 2024d.

Haryo Akbarianto Wibowo, Erland Hilman Fuadi, Made Nindyatama Nityasya, Radityo Eko Prasojo, and Alham Fikri Aji. Copal-id: Indonesian language reasoning with local culture and nuances. arXiv preprint arXiv:2311.01012, 2023.

Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ping Luo, and Ying Shan. Llama pro: Progressive llama with block expansion. arXiv preprint arXiv:2401.02415, 2024.

Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. Sheared llama: Accelerating language model pre-training via structured pruning. arXiv preprint arXiv:2310.06694, 2023.

Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. ByT5: Towards a token-free future with pre-trained byte-to-byte models. Transactions of the Association for Computational Linguistics, 10:291–306, 2022. URL https://aclanthology.org/2022.tacl-1.17/.

Man Tsung Yeung, Penghui Qi, Min Lin, and Xinyi Wan. Balancing pipeline parallelism with vocabulary parallelism. arXiv preprint arXiv:2411.05288, 2024.

Yijiong Yu, Ziyun Dai, Zekun Wang, Wei Wang, Ran Chen, and Ji Pei. Opencsg chinese corpus: A series of high-quality chinese datasets for llm training, 2025. URL https: //arxiv.org/abs/2501.08197.

Fei Yuan, Chang Ma, Shuai Yuan, Qiushi Sun, and Lei Li. Ks-lottery: Finding certified lottery tickets for multilingual language models. ArXiv, abs/2402.02801, 2024. URL https://api.semanticscholar.org/CorpusID:267412646.

Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng, Jie Liu, Qunshu Lin, et al. Map-neo: Highly capable and transparent bilingual large language model series. arXiv preprint arXiv:2405.19327, 2024a.

Wenxuan Zhang, Sharifah Mahani Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing. M3exam: A multilingual, multimodal, multilevel benchmark for examining large language models. arXiv preprint arXiv:2306.05179, 2023.

Wenxuan Zhang, Hou Pong Chan, Yiran Zhao, Mahani Aljunied, Jianyu Wang, Chaoqun Liu, Yue Deng, Zhiqiang Hu, Weiwen Xu, Yew Ken Chia, et al. Seallms 3: Open foundation and chat multilingual large language models for southeast asian languages. arXiv preprint arXiv:2407.19672, 2024b.

Yu Zhao, Yuanbin Qu, Konrad Staniszewski, Szymon Tworkowski, Wei Liu, Piotr Miło´s, Yuxiang Wu, and Pasquale Minervini. Analysing the impact of sequence composition on language model pre-training. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7897–7912, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.427. URL https://aclanthology.org/2024.acl-long.427.

Lin Zheng, Xueliang Zhao, Guangtao Wang, Chen Wu, David Dong, Angela Wang, Mingran Wang, Yun Du, Haige Bo, Amol Sharma, Bo Li, Kejie Zhang, Changran Hu, Urmish Thakker, and Lingpeng Kong. Evabyte: Efficient byte-level language models at scale, 2025. URL https://hkunlp.github.io/blog/2025/evabyte.

Fan Zhou, Zengzhi Wang, Qian Liu, Junlong Li, and Pengfei Liu. Programming every example: Lifting pre-training data quality like experts at scale. arXiv preprint arXiv:2409.17115, 2024.

The Prompt used for Translation

You are a highly skilled translator tasked with translating various types of content from English into {{ language }}. Follow these instructions carefully to complete the translation task. You will receive a user-bot conversation in XML format. Please follow a three-step translation process:

- 1. Initial Translation: Translate the input content into {{ language }}, preserving the original intent and keeping the original paragraph and text format unchanged. Do not delete or omit any content, and ensure that all original Markdown elements (e.g., images, code blocks) are preserved.
- 2. Reflection and Feedback: Carefully review both the source text and your translation. Provide constructive criticism and specific suggestions to improve the translation in terms of:

- (i) Accuracy: Correct errors of addition, mistranslation, omission, or untranslated text.
- (ii) Fluency: Apply {{ language }} grammar, spelling, and punctuation rules while avoiding unnecessary repetitions.
- (iii) Style: Ensure that the translation reflects the style of the source text and considers any relevant cultural context.

- 3. Refinement: Based on your reflections, refine and polish your translation.
- 4. Fallback: If you are not confident in translating the conversation, please return “<stop></stop>”.

Output: For each step of the translation process, output your results within the appropriate XML tags as follows:

- <step1_initial_translation> [Insert your initial translation here]

- </step1_initial_translation>

<step2_reflection> [Insert your reflection on the translation, including a list of specific, helpful, and constructive suggestions for improvement. Each suggestion should address a specific part of the translation.]

- </step2_reflection>

<step3_refined_translation> [Insert your refined and polished translation here]

- </step3_refined_translation>

Ensure that your final translation in step 3 accurately reflects the original meaning while sounding natural in {{ language }}. Here is the original conversation:

Table 22: Data Distribution of SEA-UltraChat (Instruction Tuning) and SEA-UltraFeedback (Preference Tuning) across Languages. SEA-UltraChat has been used by two-stage instruction tuning.

##### Language SFT Stage 1 SFT Stage 2 SEA-UltraChat SEA-UltraFeedback

English 1611404 72000 1683404 8981 Chinese 154908 72000 226908 8845 Indonesian 287719 48000 335719 4687 Thai 334194 48000 382194 4696 Vietnamese 301226 48000 349226 4710 Malay 212827 48000 260827 4714 Burmese 204495 48000 252495 4614 Lao 134228 48000 182228 4393 Cebuano 194986 1200 196186 1178 Ilocano 1867 1200 3067 1172 Javanese 134499 48000 182499 1166 Khmer 64421 4800 69221 1151 Sundanese 181486 1200 182686 1174 Tagalog 217045 48000 265045 1173 Waray 177921 1200 179121 0 Tamil 82015 1200 83215 1159

Total 4295241 538800 4834041 53813

Table 23: Performance on Flores Plus (chrF++) for Qwen2.5-32B

Target (→) Source (↓)

eng cmn ind tha vie zsm mya lao ceb ilo jav khm sun fil war tam

English (eng) - 34.0 64.0 46.1 59.2 59.5 17.3 21.4 39.4 27.1 31.8 20.6 33.1 49.5 38.4 28.3 Chinese (cmn) 59.0 - 52.6 40.2 51.8 48.3 16.0 18.0 33.0 23.1 27.0 18.8 28.7 41.5 30.4 24.6 Indonesian (ind) 67.5 29.5 - 43.1 54.3 55.0 16.2 20.6 34.9 23.5 34.9 19.6 36.2 45.2 31.7 25.4 Thai (tha) 60.6 27.9 54.8 - 52.1 50.3 15.6 22.0 33.0 23.7 28.5 19.5 29.6 42.6 30.9 24.7 Vietnamese (vie) 62.5 29.1 55.0 41.9 - 50.7 15.9 20.1 32.4 24.2 27.2 19.4 29.3 42.7 30.5 24.4 Malay (zsm) 67.2 28.6 59.4 42.8 53.6 - 16.0 20.5 34.8 24.0 33.1 20.0 33.5 44.5 32.3 25.5 Burmese (mya) 39.6 16.7 38.2 29.8 35.0 36.0 - 15.4 25.1 20.0 18.9 15.5 20.4 33.4 23.4 21.2 Lao (lao) 50.1 22.8 47.8 39.9 44.4 44.6 13.8 - 27.6 21.2 24.3 19.1 25.9 38.9 26.1 21.6 Cebuano (ceb) 56.0 23.7 48.8 35.9 45.0 45.2 15.2 18.6 - 27.7 22.9 17.8 22.4 45.5 44.0 23.1 Ilocano (ilo) 43.1 19.6 39.5 30.0 36.3 36.8 13.3 16.3 31.7 - 19.9 15.2 20.2 38.8 30.2 19.4 Javanese (jav) 51.9 22.9 49.9 35.0 43.8 45.9 14.1 17.9 28.7 21.8 - 17.9 31.7 37.0 25.9 21.9 Khmer (khm) 49.8 22.8 47.8 37.1 43.8 44.3 12.9 20.4 28.8 21.5 24.7 - 24.7 38.3 26.1 22.4 Sundanese (sun) 51.7 23.1 51.2 35.9 44.7 46.1 14.5 18.1 28.6 20.4 31.1 17.4 - 37.9 27.0 21.9 Tagalog (fil) 65.8 27.9 55.1 41.0 52.0 51.0 16.1 19.8 41.6 30.7 23.4 18.8 22.2 - 40.1 25.5 Waray (war) 56.8 23.8 49.1 36.1 45.1 45.1 14.9 18.7 44.2 29.3 23.2 17.2 23.7 45.6 - 23.2 Tamil (tam) 47.9 20.5 44.0 32.9 40.9 41.0 16.1 16.2 28.7 21.8 22.9 16.8 24.1 37.5 26.2 -

PPL Distribution by Language (KDE)

Language: Total

Language: SEA

Language: eng

Language: zho

- Qwen1.5: 30.94, Sailor1: 24.17
- Qwen2.5: 19.90, Sailor2: 5.38

- Qwen1.5: 32.96, Sailor1: 25.22
- Qwen2.5: 20.80, Sailor2: 4.46

- Qwen1.5: 15.44, Sailor1: 16.84
- Qwen2.5: 13.27, Sailor2: 13.97

- Qwen1.5: 14.25, Sailor1: 14.83
- Qwen2.5: 12.32, Sailor2: 11.44

0.16

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | |ailor|1|
| | | | | | | |wen wen|1.5 2.5|
| | | | | | | |ailor|2|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

0.07

0.07

0.175

0.14

0.06

0.06

0.150

0.12

0.05

0.05

0.125

0.10

Density

Density

Density

Density

0.04

0.04

0.100

0.08

0.03

0.03

0.075

0.06

0.02

0.02

0.050

0.04

0.01

0.01

0.025

0.02

0.00

0.000

0.00

0.00

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

Language: bcl

Language: jv

Language: ms

Language: th

- Qwen1.5: 57.54, Sailor1: 62.57
- Qwen2.5: 38.37, Sailor2: 7.42

- Qwen1.5: 56.36, Sailor1: 27.90
- Qwen2.5: 34.19, Sailor2: 5.49

- Qwen1.5: 11.33, Sailor1: 4.63
- Qwen2.5: 7.15, Sailor2: 3.85

- Qwen1.5: 10.44, Sailor1: 4.06
- Qwen2.5: 5.29, Sailor2: 3.38

0.175

| | | | | | |ailor wen|1 1.5|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | |wen|2.5|
| | | | | | | | |
| | | | | | |ailor|2|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.16

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

0.35

0.175

0.150

0.14

0.30

0.150

0.12

0.125

0.25

0.125

0.10

0.100

Density

Density

Density

Density

0.20

0.100

0.08

0.075

0.15

0.075

0.06

0.050

0.10

0.050

0.04

0.025

0.05

0.025

0.02

0.00

0.000

0.000

0.00

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

Language: ceb

Language: km

Language: my

Language: tl

- Qwen1.5: 30.63, Sailor1: 31.40
- Qwen2.5: 3.18, Sailor2: 1.51

- Qwen1.5: 4.17, Sailor1: 5.44
- Qwen2.5: 3.81, Sailor2: 2.36

- Qwen1.5: 2.62, Sailor1: 3.24
- Qwen2.5: 2.27, Sailor2: 1.54

- Qwen1.5: 18.18, Sailor1: 23.41
- Qwen2.5: 12.13, Sailor2: 4.92

0.25

2.00

0.7

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

1.4

1.75

0.6

0.20

1.2

1.50

0.5

1.0

1.25

0.15

0.4

Density

Density

Density

Density

0.8

1.00

0.3

0.10

0.6

0.75

0.2

0.4

0.50

0.05

0.1

0.2

0.25

0.0

0.0

0.00

0.00

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

Language: id

Language: lo

Language: pam

Language: vi

- Qwen1.5: 16.32, Sailor1: 7.37
- Qwen2.5: 10.30, Sailor2: 6.56

- Qwen1.5: 5.95, Sailor1: 3.27
- Qwen2.5: 4.76, Sailor2: 2.46

- Qwen1.5: 80.35, Sailor1: 77.74
- Qwen2.5: 59.35, Sailor2: 12.26

- Qwen1.5: 14.52, Sailor1: 6.10
- Qwen2.5: 7.01, Sailor2: 5.06

0.06

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | |ailor wen|1 1.5|
| | | | | | | | |wen ailor|2.5 2|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | |S Q|ailor wen|1.5|
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | |Q S|wen ailor|2.5|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.6

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

0.12

0.20

0.05

0.5

0.10

0.04

0.15

0.4

0.08

Density

Density

Density

Density

0.03

0.3

0.06

0.10

0.02

0.2

0.04

0.05

0.01

0.1

0.02

0.00

0.0

0.00

0.00

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

Language: ilo

Language: min

Language: su

Language: war

- Qwen1.5: 70.40, Sailor1: 82.69
- Qwen2.5: 42.67, Sailor2: 4.70

- Qwen1.5: 48.05, Sailor1: 17.91
- Qwen2.5: 35.04, Sailor2: 2.83

- Qwen1.5: 46.27, Sailor1: 27.35
- Qwen2.5: 25.88, Sailor2: 5.00

- Qwen1.5: 51.39, Sailor1: 16.11
- Qwen2.5: 39.67, Sailor2: 1.82

0.200

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

- Sailor1

- Qwen1.5

- Qwen2.5

- Sailor2

0.30

0.14

1.2

0.175

0.12

0.25

1.0

0.150

0.10

0.20

0.125

0.8

Density

Density

Density

Density

0.08

0.100

0.15

0.6

0.06

0.075

0.10

0.4

0.04

0.050

0.05

0.2

0.02

0.025

0.000

0.00

0.00

0.0

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

0 1 2 3 4 Perplexity (log10 scale)

- Figure 11: Comparison of PPL distribution smoothed with Kernel Density Estimation (KDE). We compare Sailor2-8B, Qwen2.5-7B, Sailor1-7b and Qwen1.5-7B. Our results demonstrate that with extra 1B parameters, Sailor2-8B can preserve its English and Chinese capability, while achieving in much lower PPL in SEA languages.

###### Winning Rate Comparison of Four Models Across Language Pairs

English

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Chinese

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Indonesian

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Thai

Vietnamese

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Malay

Burmese

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

SourceLanguage

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

Lao

Cebuano

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | |
|---|---|
| | |

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Ilocano

Javanese

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Khmer

Sundanese

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Tagalog

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Models

Waray

sailor2_20b llama3_1_70b qwen2_5_72b nllb_moe_54b

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| |
|---|

Tamil

| |
|---|

Waray Tamil

English

Ilocano

Javanese

Thai

Lao

Indonesian

Chinese

Khmer

Cebuano

Sundanese

Burmese

Tagalog

Malay

Vietnamese

Target Language

- Figure 12: Comparison of win rate of four models based on their ChrF++ scores. The shaded area in each cube represents the top-1 accuracy of each model across different translation directions in the Flores Plus Translation Dataset. We observed that Sailor2 performs on par with, or even outperforms NLLB, a model optimized for translation tasks that excels at translating low-resource languages.

BLEU Score Difference (Sailor2-20B - Qwen2.5-32B)

[Figure 9]

English

0.00 -1.73 4.92 0.58 2.01 9.22 1.20 3.88 17.37 12.68 16.66 1.82 10.12 10.34 13.71 6.27

Chinese

- -1.23 0.00 1.17 0.33 1.07 4.68 0.39 3.04 11.22 7.52 9.91 1.04 5.79 6.06 8.52 3.66

0.14 -1.41 0.00 0.90 2.21 4.90 1.23 3.17 13.55 10.32 11.86 1.38 6.42 8.94 11.01 4.56

- 0.54 0.28 2.56 0.00 1.88 6.11 1.25 1.16 12.51 8.96 11.05 1.89 6.73 8.33 9.86 3.89

-0.36 -1.25 2.78 1.25 0.00 6.26 1.25 2.54 13.84 9.23 11.72 1.35 6.81 8.12 10.44 4.23

- 1.49 0.48 1.28 0.50 2.17 0.00 1.12 3.50 14.48 10.65 13.41 0.20 6.97 9.40 11.40 4.49

4

Indonesian

Thai

Vietnamese

2

Malay

Burmese

- 11.01 1.58 9.35 1.43 10.67 11.12 0.00 2.59 13.05 7.53 10.60 1.55 7.07 11.32 10.89 3.93

- 10.51 -1.26 7.91 1.78 7.91 10.56 1.53 0.00 15.11 9.46 13.21 1.53 7.75 12.05 11.85 4.74

12.69 1.50 8.65 2.16 8.90 11.23 1.47 2.95 0.00 11.31 14.76 1.97 10.02 9.72 7.89 5.28

14.93 1.24 10.31 2.12 10.87 12.32 1.10 2.33 14.16 0.00 9.94 1.61 7.23 11.55 11.61 4.42

- 11.04 0.55 8.94 1.03 7.67 9.87 1.21 2.92 15.01 7.66 0.00 1.31 6.51 14.07 10.35 4.80

10.96 0.80 8.38 1.97 8.56 11.09 1.37 2.99 15.05 9.56 13.14 0.00 8.59 12.48 11.46 4.09

9.50 -0.48 8.06 0.73 6.90 10.24 1.16 3.08 13.43 4.55 11.96 2.04 0.00 12.71 5.74 4.66

4.46 -0.06 5.74 0.46 3.55 9.23 0.98 2.50 12.86 11.39 16.14 1.93 10.39 0.00 9.81 4.65

- 12.39 0.24 8.18 2.08 8.19 11.91 1.24 2.68 12.34 11.42 13.79 1.46 8.75 10.06 0.00 5.19

SourceLanguage

Difference

Lao

0

Cebuano

Ilocano

Javanese

2

Khmer

Sundanese

Tagalog

4

Waray

Tamil

8.03 1.79 6.38 0.85 7.26 8.86 0.67 2.43 13.34 7.83 10.92 1.69 6.85 10.02 10.98 0.00

EnglishChineseIndonesian ThaiVietnameseMalayBurmese LaoCebuanoIlocanoJavaneseKhmerSundaneseTagalogWaray Tamil

Target Language

CHRF++ Score Difference (Sailor2-20B - Qwen2.5-32B)

[Figure 10]

English

0.00 -1.09 3.65 2.42 1.42 7.36 10.82 16.25 18.27 18.86 20.22 8.66 14.62 9.78 14.87 12.84

Chinese

- -1.21 0.00 1.06 1.59 0.85 4.63 8.62 12.71 14.73 15.87 15.62 6.29 10.73 6.61 12.37 9.96
- -0.44 -1.43 0.00 2.25 1.49 4.35 9.86 13.95 16.40 18.82 14.30 6.92 9.76 8.81 14.86 11.75
- -0.05 -0.44 1.93 0.00 1.36 5.67 9.89 11.94 16.07 17.34 16.24 6.56 11.73 8.15 13.54 10.74
- -0.40 -1.03 2.56 2.58 0.00 5.71 9.71 13.36 17.25 17.08 18.03 6.70 12.34 8.74 14.63 11.68

4

Indonesian

Thai

Vietnamese

2

Malay

0.95 0.07 1.00 2.11 1.91 0.00 10.35 14.79 16.69 18.54 16.26 6.61 11.13 9.57 14.98 11.91

Burmese

12.36 6.18 11.06 8.47 11.90 13.43 0.00 14.37 19.72 17.05 20.95 7.72 16.29 13.07 18.51 12.09

SourceLanguage

Difference

Lao

- 10.49 3.01 8.07 5.09 7.94 10.91 11.61 0.00 21.21 18.75 20.72 8.11 15.24 12.19 17.94 13.74
- 11.35 4.85 8.99 7.73 8.52 12.09 11.25 14.63 0.00 16.76 22.87 8.56 19.05 9.99 7.70 13.80

0

Cebuano

Ilocano

14.99 5.08 12.25 8.74 11.68 14.79 11.08 13.31 17.82 0.00 17.98 8.76 16.42 12.62 15.08 14.01

Javanese

10.09 3.50 8.84 6.69 7.52 10.49 11.11 14.68 19.61 16.37 0.00 8.06 10.63 14.82 16.76 13.22

2

Khmer

9.87 3.66 7.62 5.78 7.66 10.54 11.91 13.06 20.00 18.62 20.07 0.00 16.76 12.19 18.13 12.13

Sundanese

- 9.08 2.88 7.85 5.95 6.77 10.18 10.94 14.25 18.85 12.95 15.66 8.85 0.00 13.21 11.33 13.25

3.61 1.29 5.02 3.49 3.01 8.47 10.33 13.51 13.71 15.54 23.85 7.77 20.14 0.00 10.19 11.97

- 10.90 3.43 8.21 7.49 7.91 12.29 11.37 14.47 12.53 16.52 21.18 8.27 16.18 10.27 0.00 13.63

Tagalog

4

Waray

Tamil

8.58 3.73 7.28 6.36 7.56 10.47 8.55 12.62 18.00 16.73 18.58 6.92 13.89 10.03 16.68 0.00

EnglishChineseIndonesian ThaiVietnameseMalayBurmese LaoCebuanoIlocanoJavaneseKhmerSundaneseTagalogWaray Tamil

Target Language

- Figure 13: Comparison of BLEU and ChrF++ scores on the Flores Plus Translation Dataset across various source-target language pairs between Sailor2 20B and Qwen2.5-32B. BLEU

BLEU Score Difference (Sailor2-20B - Qwen2.5-72B)

[Figure 11]

English

0.00 -1.21 1.71 0.63 0.11 5.73 0.89 3.36 12.98 10.28 13.25 2.44 7.51 7.11 9.21 5.27

Chinese

- -2.73 0.00 -0.62 0.58 -0.28 2.53 0.21 1.92 8.81 6.50 7.53 1.07 3.74 3.30 5.54 3.31
- -1.61 -0.57 0.00 1.20 -0.01 2.40 0.88 1.94 9.76 7.53 9.13 1.44 4.87 5.79 7.43 3.50
- -2.42 0.34 0.30 0.00 -0.49 3.37 0.86 0.34 9.32 7.86 8.30 2.22 4.40 4.96 6.58 3.24
- -2.67 -1.90 -0.03 0.81 0.00 3.35 1.00 1.83 9.99 7.85 8.32 1.37 4.09 4.78 7.40 3.66
- -0.54 0.22 -0.50 0.69 0.78 0.00 0.73 2.77 10.63 8.01 10.01 0.77 5.05 5.47 8.09 3.61

4

Indonesian

Thai

Vietnamese

2

Malay

Burmese

7.57 0.52 5.84 0.51 6.66 7.75 0.00 2.22 11.08 7.08 8.74 1.43 5.75 8.84 9.52 3.18

SourceLanguage

Difference

Lao

4.81 -1.77 3.99 0.96 3.26 6.65 0.99 0.00 11.03 8.49 10.46 0.64 5.56 7.49 8.85 3.98

0

Cebuano

7.65 1.38 5.08 1.80 5.18 7.61 1.27 2.91 0.00 8.07 13.48 2.45 6.44 6.58 6.89 4.12

Ilocano

13.30 0.73 7.79 2.04 8.12 9.91 0.96 2.45 10.25 0.00 8.95 1.80 5.32 8.21 7.02 3.94

Javanese

- 6.47 0.29 5.94 0.59 4.17 6.58 1.01 2.51 11.45 5.44 0.00 2.09 4.40 9.78 6.98 4.09
- 7.44 -0.70 4.59 1.39 4.28 6.96 1.30 2.51 12.43 8.93 10.31 0.00 6.35 8.50 9.42 3.56

2

Khmer

Sundanese

6.02 -0.73 5.82 1.65 4.58 7.42 1.15 3.26 10.13 2.52 8.54 1.73 0.00 8.96 3.16 3.93

Tagalog

0.27 -1.49 1.80 1.01 0.73 5.56 0.67 2.45 10.07 8.30 13.61 1.72 6.24 0.00 7.67 3.57

4

Waray

9.03 -0.56 4.96 2.82 5.15 8.55 1.12 3.37 9.62 8.66 11.88 1.67 5.70 6.98 0.00 4.40

Tamil

5.39 1.17 3.49 0.14 4.29 5.48 0.33 2.25 10.36 7.15 9.00 0.93 5.03 6.29 8.78 0.00

EnglishChineseIndonesian ThaiVietnameseMalayBurmese LaoCebuanoIlocanoJavaneseKhmerSundaneseTagalogWaray Tamil

Target Language

CHRF++ Score Difference (Sailor2-20B - Qwen2.5-72B)

[Figure 12]

English

0.00 -1.47 1.22 0.63 -0.12 4.41 8.21 12.87 13.28 14.23 15.27 7.27 10.54 6.17 9.65 9.94

Chinese

- -2.12 0.00 -0.73 0.15 -0.45 2.36 6.71 9.24 10.53 12.39 11.11 5.02 6.56 3.20 6.06 8.03
- -1.61 -2.03 0.00 0.67 -0.26 2.08 7.07 10.67 10.87 12.04 10.86 5.18 6.92 5.18 8.51 8.40
- -1.69 -1.44 0.08 0.00 -0.64 2.82 6.87 8.71 10.76 14.40 11.20 5.30 7.02 4.49 7.94 8.14
- -1.80 -1.77 0.33 0.82 0.00 3.11 7.30 10.23 11.37 13.11 11.29 5.07 7.04 4.38 8.57 8.62
- -0.62 -1.21 -0.42 0.73 0.52 0.00 7.25 11.55 11.48 13.05 12.42 5.06 7.50 5.36 9.14 8.50

4

Indonesian

Thai

Vietnamese

2

Malay

Burmese

8.86 4.01 6.61 5.08 7.48 9.07 0.00 11.46 14.69 15.86 15.11 5.56 11.48 8.69 13.61 9.00

SourceLanguage

Difference

Lao

5.68 0.97 4.01 2.04 3.43 6.91 8.30 0.00 14.68 15.88 15.92 5.44 10.67 7.82 11.84 10.11

0

Cebuano

7.18 2.22 5.28 4.67 5.10 8.07 8.57 12.83 0.00 10.90 20.31 6.70 11.52 6.25 6.49 10.23

Ilocano

13.44 3.54 9.42 6.45 8.96 12.13 8.96 12.55 12.77 0.00 16.12 7.22 11.25 9.59 8.87 11.46

Javanese

- 6.41 1.73 5.74 4.45 4.22 7.14 8.11 12.36 13.96 11.48 0.00 6.47 7.74 10.23 9.80 9.77
- 7.36 2.08 5.19 3.54 4.34 7.29 8.52 11.30 15.71 17.75 15.10 0.00 11.15 7.64 12.91 8.95

- 5.97 1.16 5.33 4.04 4.51 7.49 8.95 12.18 13.46 7.08 11.57 6.89 0.00 9.42 5.98 10.26

0.45 0.09 1.75 1.28 0.58 4.77 7.57 11.30 10.22 10.90 18.77 5.77 10.81 0.00 7.70 8.76

8.11 1.38 5.21 5.31 5.09 8.66 9.40 13.23 9.85 11.87 17.86 6.44 10.18 6.81 0.00 10.44

- 6.09 2.65 4.18 3.58 4.70 6.41 6.38 10.56 12.62 14.91 14.00 5.14 9.61 5.88 11.37 0.00

2

Khmer

Sundanese

Tagalog

4

Waray

Tamil

EnglishChineseIndonesian ThaiVietnameseMalayBurmese LaoCebuanoIlocanoJavaneseKhmerSundaneseTagalogWaray Tamil

Target Language

- Figure 14: Comparison of BLEU and ChrF++ scores on the Flores Plus Translation Dataset across various source-target language pairs between Sailor2 20B and Qwen2.5-72B. BLEU

BLEU Score Difference (Sailor2-20B - Llama3.1-70B)

[Figure 13]

English

0.00 -1.08 0.75 -2.33 -0.41 0.99 0.99 3.58 2.89 0.49 1.34 2.13 1.54 1.32 -0.07 1.07

Chinese

- -1.81 0.00 -0.67 -0.23 -0.03 -0.27 0.44 2.87 2.57 -0.30 0.80 1.35 0.06 0.19 -0.23 0.86
- -1.83 -0.34 0.00 -0.97 -0.29 -1.42 0.81 2.95 1.86 0.04 1.36 1.67 0.89 0.93 -0.37 0.46
- -0.78 1.62 -0.33 0.00 -0.39 0.10 1.05 2.13 2.25 1.17 0.90 2.20 0.54 1.28 -0.14 0.86
- -1.64 -2.03 -0.75 -0.48 0.00 -0.29 0.88 3.09 1.91 1.04 0.79 1.42 -0.09 0.57 -0.22 0.94
- -1.35 0.11 -0.87 -0.74 0.24 0.00 0.88 3.85 1.63 -0.12 1.09 1.18 0.72 1.40 -0.41 0.39

4

Indonesian

Thai

Vietnamese

2

Malay

Burmese

- 0.47 0.65 0.57 -0.10 1.06 0.94 0.00 2.97 3.21 1.46 1.25 1.59 1.17 1.55 1.50 0.81

11.32 -1.25 6.17 1.06 7.01 5.39 1.34 0.00 5.37 2.27 3.78 1.77 2.58 5.18 2.63 2.23

- 1.90 3.08 0.89 0.24 0.85 1.10 1.08 4.12 0.00 0.18 1.91 1.91 1.33 1.22 -0.08 0.85
- 2.68 1.60 1.05 0.93 0.44 1.32 0.77 3.15 1.74 0.00 -1.26 1.26 -0.37 1.71 -0.95 0.45

SourceLanguage

Difference

Lao

0

Cebuano

Ilocano

Javanese

- -1.93 1.03 2.46 -0.75 -0.51 -0.96 0.91 3.34 1.74 -1.94 0.00 1.22 0.21 1.62 -1.72 1.20

4.13 -2.50 2.23 -1.13 1.36 2.37 1.35 3.92 3.97 1.79 3.00 0.00 2.10 2.40 1.27 1.02

- -0.28 0.11 1.61 -0.82 0.40 0.46 0.82 3.54 1.23 -5.10 0.49 1.27 0.00 2.69 -5.79 0.50
- -0.66 -0.98 0.04 -0.71 0.06 0.58 0.93 3.18 2.30 1.49 1.83 1.67 0.58 0.00 -0.34 0.54

0.92 0.89 0.21 0.75 0.11 0.28 0.97 3.49 2.73 0.40 0.73 1.07 -0.29 0.67 0.00 1.14

- -2.30 0.56 -1.61 -1.91 -0.97 -1.42 0.39 2.99 1.99 0.25 0.74 1.16 0.40 -0.60 0.23 0.00

2

Khmer

Sundanese

Tagalog

4

Waray

Tamil

EnglishChineseIndonesian ThaiVietnameseMalayBurmese LaoCebuanoIlocanoJavaneseKhmerSundaneseTagalogWaray Tamil

Target Language

CHRF++ Score Difference (Sailor2-20B - Llama3.1-70B)

[Figure 14]

English

0.00 0.56 0.35 0.64 -0.49 0.60 6.77 14.95 3.23 -0.19 0.95 7.68 2.38 1.45 0.51 3.25

Chinese

- -1.25 0.00 -0.53 0.43 -0.16 -0.38 5.31 11.51 2.47 0.07 0.65 5.55 0.65 0.14 -0.38 2.60
- -1.47 0.25 0.00 0.82 -0.42 -1.25 5.62 12.89 1.98 0.09 1.57 5.88 1.42 0.75 0.08 2.47
- -0.59 0.60 -0.42 0.00 -0.34 0.08 5.88 12.14 2.11 0.86 0.69 5.81 1.34 0.99 -0.04 2.77
- -0.97 0.80 -0.42 0.79 0.00 -0.19 5.72 12.63 2.03 1.12 1.09 5.67 0.52 0.37 0.20 2.80
- -1.28 0.50 -0.97 0.77 -0.07 0.00 5.84 13.45 1.77 -0.09 0.81 5.67 1.16 1.06 -0.09 3.03

4

Indonesian

Thai

Vietnamese

2

Malay

Burmese

0.92 1.01 0.72 1.32 0.93 0.95 0.00 12.98 3.77 1.74 2.03 5.19 2.80 1.50 2.23 1.76

SourceLanguage

Difference

Lao

11.61 3.75 6.60 3.52 7.22 5.95 8.05 0.00 6.52 2.85 5.38 7.67 5.62 5.84 3.95 6.89

0

Cebuano

- 2.12 2.31 0.58 1.40 0.78 1.08 6.52 13.48 0.00 -0.08 2.72 5.97 2.06 1.07 0.71 3.72
- 3.08 1.05 1.05 1.38 0.67 1.35 5.58 11.38 2.27 0.00 -1.57 5.28 0.15 1.78 -0.48 2.42

-1.12 0.41 1.83 0.64 -0.64 -0.61 5.82 12.78 1.82 -1.84 0.00 5.74 0.84 1.79 -0.86 2.56

- 4.21 1.24 2.16 1.50 1.50 1.85 6.22 13.30 4.13 2.00 3.46 0.00 4.19 2.14 2.48 3.47

Ilocano

Javanese

2

Khmer

Sundanese

0.31 0.74 1.19 0.69 0.31 0.17 5.93 12.69 2.00 -6.35 0.77 6.10 0.00 2.76 -4.68 2.68

Tagalog

- -0.03 1.31 0.16 1.05 -0.05 0.41 6.20 13.17 2.72 1.43 2.19 6.04 1.28 0.00 -0.08 3.25

1.35 0.62 -0.08 1.76 0.28 0.52 6.41 13.28 2.83 0.24 1.04 5.56 -0.09 0.97 0.00 3.40

- -1.72 0.49 -1.67 -0.02 -1.11 -1.19 4.72 11.49 1.82 -0.13 0.73 5.24 0.79 -0.87 0.15 0.00

4

Waray

Tamil

EnglishChineseIndonesian ThaiVietnameseMalayBurmese LaoCebuanoIlocanoJavaneseKhmerSundaneseTagalogWaray Tamil

Target Language

- Figure 15: Comparison of BLEU and ChrF++ scores on the Flores Plus Translation Dataset across various source-target language pairs between Sailor2 20B and Llama3.1-70B. BLEU Score Difference = Sailor2 BLEU - Llama3.1 BLEU.

BLEU Score Difference (Sailor2-20B - NLLB-MoE-54B)

[Figure 15]

English

- 0.00 10.34 1.61 7.20 2.26 1.83 0.02 -0.19 4.44 -5.60 -0.71 1.19 4.14 0.32 -3.05 -5.37

- 2.10 0.00 2.68 2.54 3.45 1.92 -0.20 -0.54 2.11 -2.39 1.59 3.12 1.60 2.16 1.49 -1.33

1.11 9.33 0.00 3.33 2.54 0.07 0.62 -2.18 1.41 -4.71 0.66 1.18 2.42 1.43 -2.87 -2.86

- 3.57 10.41 2.61 0.00 3.14 2.62 0.48 -3.26 1.78 -2.80 0.87 2.12 2.16 2.80 -0.18 -1.35

Chinese

4

Indonesian

Thai

Vietnamese

1.36 8.43 1.11 4.90 0.00 1.27 0.61 -2.31 1.32 -3.78 -0.20 0.76 0.76 1.39 -1.18 -2.25

2

Malay

1.46 10.52 -0.02 2.27 2.87 0.00 0.58 -1.85 0.63 -4.32 0.75 -0.10 1.74 1.32 -1.15 -4.04

Burmese

- -2.65 8.90 -1.53 2.09 0.13 -0.56 0.00 -1.88 1.27 -3.37 0.38 1.33 1.02 0.83 -0.29 -2.16
- -0.19 7.02 0.84 1.66 1.48 0.68 0.15 0.00 -0.64 -3.02 0.12 1.99 1.41 1.42 -0.44 -3.21

- 0.50 11.81 0.11 6.93 1.54 -0.64 0.70 -2.17 0.00 -4.32 -1.10 2.52 2.26 1.04 -1.83 -2.39

- -4.11 10.51 -3.18 7.97 -1.42 -2.72 0.32 -1.96 0.90 0.00 -5.31 1.85 -0.18 -0.51 -4.52 -2.78
- -0.42 11.00 1.81 2.51 1.61 0.37 0.04 -1.85 -1.73 -4.98 0.00 1.99 1.02 1.28 -1.84 -3.64

- 1.16 8.21 0.93 2.37 1.69 1.31 0.39 -2.25 -0.22 -3.07 0.61 0.00 1.70 1.44 -0.72 -2.98

- -0.89 9.48 1.12 2.62 1.55 0.64 -0.24 -1.54 -1.19 -8.86 0.55 1.82 0.00 0.82 -7.02 -3.66
- -0.24 9.22 0.32 4.53 1.28 0.20 0.08 -2.56 3.03 -4.28 -1.33 1.38 1.28 0.00 -2.50 -3.75
- -1.55 10.49 -2.64 6.12 -0.99 -1.62 -0.13 -3.24 5.23 -4.46 -3.04 0.16 -0.20 0.63 0.00 -3.66
- -5.21 9.11 -3.11 0.96 -1.62 -2.62 -0.12 -3.11 -1.20 -6.40 -1.66 -0.83 0.09 -1.75 -2.73 0.00

SourceLanguage

Difference

Lao

0

Cebuano

Ilocano

Javanese

2

Khmer

Sundanese

Tagalog

4

Waray

Tamil

EnglishChineseIndonesian ThaiVietnameseMalayBurmese LaoCebuanoIlocanoJavaneseKhmerSundaneseTagalogWaray Tamil

Target Language

CHRF++ Score Difference (Sailor2-20B - NLLB-MoE-54B)

[Figure 16]

English

- 0.00 13.92 1.86 8.72 1.88 2.11 -1.96 -5.54 2.78 -6.49 -0.32 -3.38 5.45 0.34 -1.67 -9.07
- 1.51 0.00 2.95 7.01 3.53 2.60 -2.15 -4.22 3.49 -2.54 2.41 -2.53 3.28 2.81 2.31 -4.49

0.98 12.04 0.00 7.35 2.17 0.72 -0.14 -6.15 1.71 -4.76 1.30 -5.46 3.22 1.77 -1.74 -6.46

- 2.86 11.31 2.34 0.00 2.87 2.85 0.27 -5.35 2.06 -3.05 0.76 -4.68 2.72 2.88 0.45 -4.49

Chinese

4

Indonesian

Thai

Vietnamese

- 0.89 11.50 1.18 7.86 0.00 1.49 0.21 -6.09 1.52 -3.99 -0.01 -5.30 1.52 1.60 -0.46 -6.23
- 1.02 11.08 0.08 6.60 2.33 0.00 -0.36 -5.96 0.77 -4.93 1.00 -5.63 2.29 1.53 -0.57 -7.70

2

Malay

Burmese

- -2.03 9.73 -1.22 4.74 0.20 -0.49 0.00 -4.56 3.02 -3.11 2.36 -3.40 2.19 1.75 0.39 -5.90
- -0.50 7.50 0.32 5.43 1.27 0.76 -1.33 0.00 -0.01 -4.59 0.05 -4.21 1.80 1.25 -0.44 -7.79

0.33 12.88 -0.12 7.17 1.24 -0.22 0.88 -6.64 0.00 -4.98 -1.21 -3.86 2.05 1.32 -0.12 -5.15

- -4.40 9.84 -3.61 4.02 -1.78 -3.07 -0.34 -8.72 -0.71 0.00 -7.52 -4.92 -1.75 -0.76 -4.61 -7.27

SourceLanguage

Difference

Lao

0

Cebuano

Ilocano

Javanese

- 0.08 10.45 1.61 6.53 1.15 1.06 -2.24 -6.16 -0.77 -5.67 0.00 -3.19 1.22 1.40 -0.49 -7.91
- 1.22 8.67 1.13 5.96 1.66 1.27 -2.64 -6.22 0.88 -3.66 0.66 0.00 2.32 1.41 0.34 -7.80

2

Khmer

Sundanese

- -0.52 7.90 0.86 5.25 1.42 0.67 -2.44 -6.40 -0.64 -11.28 0.92 -3.53 0.00 1.38 -6.33 -7.76
- -0.24 12.77 0.54 7.21 0.94 0.61 -0.09 -7.65 2.71 -4.33 -1.30 -4.77 1.39 0.00 -1.55 -6.66
- -1.35 10.09 -2.31 5.44 -1.04 -1.45 -1.39 -7.51 3.95 -4.80 -3.47 -5.75 -1.09 0.60 0.00 -7.47
- -4.67 8.30 -2.92 3.92 -1.80 -2.21 -2.19 -8.62 -1.09 -7.04 -2.59 -6.51 -0.29 -1.89 -2.50 0.00

Tagalog

4

Waray

Tamil

EnglishChineseIndonesian ThaiVietnameseMalayBurmese LaoCebuanoIlocanoJavaneseKhmerSundaneseTagalogWaray Tamil

Target Language

- Figure 16: Comparison of BLEU and ChrF++ scores on the Flores Plus Translation Dataset across various source-target language pairs between Sailor2 20B and NLLB-MoE-54B. BLEU Score Difference = Sailor2 BLEU - NLLB BLEU. We noticed NLLB failed to generate complete long Chinese sentences: https://github.com/facebookresearch/fairseq/issues/5549, and we also found many common Chinese characters and punctuations are tokenized to <unk>.

Table 24: Performance on Flores Plus (chrF++) for Qwen2.5-72B

Target (→) Source (↓)

eng cmn ind tha vie zsm mya lao ceb ilo jav khm sun fil war tam

English (eng) - 34.3 66.5 47.9 60.8 62.4 19.9 24.8 44.4 31.8 36.8 21.9 37.2 53.1 43.7 31.2 Chinese (cmn) 59.9 - 54.4 41.7 53.1 50.6 17.9 21.5 37.1 26.6 31.5 20.1 32.9 44.9 36.8 26.5 Indonesian (ind) 68.7 30.1 - 44.7 56.1 57.3 19.0 23.9 40.4 30.2 38.3 21.4 39.1 48.9 38.1 28.7 Thai (tha) 62.3 28.9 56.6 - 54.1 53.1 18.6 25.2 38.3 26.6 33.5 20.8 34.3 46.3 36.5 27.3 Vietnamese (vie) 63.9 29.8 57.2 43.6 - 53.3 18.3 23.3 38.3 28.2 34.0 21.1 34.6 47.1 36.5 27.4

- Malay (zsm) 68.8 29.9 60.9 44.1 55.0 - 19.1 23.8 40.0 29.5 36.9 21.6 37.1 48.7 38.2 28.9 Burmese (mya) 43.1 18.9 42.6 33.2 39.4 40.4 - 18.4 30.1 21.2 24.8 17.7 25.2 37.8 28.3 24.3 Lao (lao) 55.0 24.8 51.9 42.9 48.9 48.6 17.1 - 34.1 24.1 29.1 21.7 30.5 43.3 32.2 25.3 Cebuano (ceb) 60.1 26.3 52.5 39.0 48.4 49.3 17.9 20.4 - 33.6 25.4 19.6 30.0 49.2 45.2 26.6 Ilocano (ilo) 44.7 21.2 42.3 32.3 39.0 39.5 15.4 17.1 36.7 - 21.8 16.8 25.4 41.8 36.4 22.0 Javanese (jav) 55.6 24.6 53.0 37.2 47.1 49.2 17.1 20.2 34.4 26.7 - 19.5 34.6 41.6 32.8 25.3 Khmer (khm) 52.3 24.4 50.3 39.4 47.1 47.5 16.3 22.2 33.1 22.4 29.7 - 30.3 42.8 31.3 25.6 Sundanese (sun) 54.8 24.8 53.7 37.8 47.0 48.8 16.5 20.1 34.0 26.3 35.2 19.4 - 41.6 32.3 24.9 Tagalog (fil) 69.0 29.1 58.4 43.2 54.4 54.7 18.9 22.0 45.1 35.3 28.5 20.8 31.6 - 42.6 28.7 Waray (war) 59.6 25.8 52.1 38.3 48.0 48.7 16.9 19.9 46.9 33.9 26.6 19.1 29.7 49.1 - 26.4 Tamil (tam) 50.4 21.6 47.1 35.7 43.7 45.1 18.2 18.2 34.1 23.6 27.5 18.6 28.4 41.7 31.5 -

Table 25: Performance on Flores Plus (chrF++) for Llama3.1-70B

Target (→) Source (↓)

eng cmn ind tha vie zsm mya lao ceb ilo jav khm sun fil war tam

English (eng) - 32.3 67.3 47.9 61.1 66.2 21.3 22.7 54.5 46.2 51.1 21.5 45.3 57.8 52.8 37.9 Chinese (cmn) 59.0 - 54.2 41.4 52.8 53.4 19.3 19.2 45.2 38.9 41.9 19.6 38.8 48.0 43.2 31.9 Indonesian (ind) 68.5 27.8 - 44.6 56.2 60.6 20.4 21.6 49.3 42.2 47.6 20.7 44.6 53.3 46.5 34.6 Thai (tha) 61.2 26.9 57.1 - 53.8 55.9 19.6 21.8 46.9 40.2 44.0 20.3 40.0 49.8 44.5 32.6 Vietnamese (vie) 63.0 27.2 57.9 43.6 - 56.6 19.9 20.9 47.6 40.2 44.2 20.5 41.1 51.1 44.9 33.3

- Malay (zsm) 69.4 28.2 61.4 44.1 55.6 - 20.5 21.9 49.7 42.6 48.5 20.9 43.4 53.0 47.4 34.4 Burmese (mya) 51.0 21.9 48.5 37.0 46.0 48.5 - 16.8 41.0 35.3 37.9 18.1 33.9 45.0 39.6 31.5 Lao (lao) 49.0 22.0 49.3 41.4 45.1 49.6 17.4 - 42.3 37.1 39.6 19.5 35.5 45.3 40.1 28.5 Cebuano (ceb) 65.2 26.2 57.2 42.2 52.7 56.2 20.0 19.8 - 44.6 43.0 20.4 39.4 54.4 51.0 33.2 Ilocano (ilo) 55.0 23.7 50.6 37.4 47.3 50.3 18.8 18.3 47.2 - 39.5 18.7 36.5 49.6 45.7 31.0 Javanese (jav) 63.1 26.0 56.9 41.0 52.0 57.0 19.4 19.8 46.5 40.0 - 20.2 41.5 50.0 43.5 32.5 Khmer (khm) 55.4 25.2 53.3 41.4 50.0 53.0 18.6 20.2 44.7 38.1 41.3 - 37.3 48.3 41.8 31.0

- Sundanese (sun) 60.5 25.2 57.8 41.2 51.2 56.1 19.5 19.6 45.5 39.7 46.0 20.2 - 48.3 43.0 32.5 Tagalog (fil) 69.5 27.9 60.0 43.5 55.1 59.1 20.2 20.2 52.6 44.8 45.1 20.5 41.1 - 50.4 34.2 Waray (war) 66.4 26.6 57.4 41.9 52.8 56.9 19.9 19.9 54.0 45.5 43.4 19.9 39.9 54.9 - 33.4 Tamil (tam) 58.2 23.8 53.0 39.3 49.5 52.7 19.9 17.3 44.9 38.6 40.8 18.5 37.2 48.4 42.7 -

Table 26: Translation Performance (chrF++) for NLLB-MoE-54B

Target (→) Source (↓)

eng cmn ind tha vie zsm mya lao ceb ilo jav khm sun fil war tam

English (eng) - 18.9 65.8 39.8 58.8 64.7 30.1 43.2 54.9 52.5 52.4 32.6 42.3 58.9 55.0 50.2 Chinese (cmn) 56.2 - 50.7 34.8 49.1 50.4 26.8 35.0 44.2 41.5 40.2 27.6 36.1 45.3 40.5 39.0 Indonesian (ind) 66.1 16.1 - 38.0 53.6 58.6 26.2 40.7 49.6 47.0 47.9 32.0 42.8 52.3 48.3 43.6 Thai (tha) 57.7 16.2 54.4 - 50.6 53.1 25.2 39.3 47.0 44.1 44.0 30.8 38.6 47.9 44.0 39.9 Vietnamese (vie) 61.2 16.6 56.3 36.6 - 54.9 25.4 39.6 48.1 45.3 45.3 31.4 40.1 49.8 45.6 42.3 Malay (zsm) 67.1 17.6 60.4 38.3 53.2 - 26.7 41.3 50.7 47.4 48.4 32.2 42.3 52.5 47.9 45.1 Burmese (mya) 54.0 13.2 50.5 33.6 46.7 49.9 - 34.4 41.8 40.1 37.5 26.6 34.5 44.7 41.5 39.2 Lao (lao) 61.1 18.3 55.6 39.5 51.0 54.8 26.8 - 48.8 44.6 45.0 31.4 39.4 49.9 44.5 43.2 Cebuano (ceb) 67.0 15.7 57.9 36.5 52.3 57.5 25.6 39.9 - 49.5 47.0 30.2 39.4 54.2 51.9 42.0 Ilocano (ilo) 62.5 14.9 55.3 34.7 49.8 54.7 24.7 38.4 50.2 - 45.4 28.9 38.4 52.1 49.9 40.7 Javanese (jav) 61.9 15.9 57.1 35.1 50.2 55.3 27.4 38.7 49.1 43.9 - 29.1 41.1 50.4 43.1 43.0 Khmer (khm) 58.4 17.8 54.3 37.0 49.8 53.6 27.4 39.7 47.9 43.8 44.1 - 39.1 49.1 43.9 42.3

- Sundanese (sun) 61.3 18.1 58.2 36.6 50.1 55.6 27.9 38.7 48.1 44.6 45.9 29.8 - 49.7 44.6 42.9 Tagalog (fil) 69.7 16.4 59.6 37.3 54.1 58.9 26.5 41.0 52.6 50.6 48.6 31.3 41.0 - 51.8 44.1 Waray (war) 69.1 17.1 59.6 38.2 54.1 58.9 27.7 40.7 52.8 50.6 47.9 31.2 40.9 55.3 - 44.3 Tamil (tam) 61.1 15.9 54.2 35.3 50.2 53.7 26.8 37.4 47.8 45.5 44.1 30.2 38.2 49.5 45.4 -

Table 27: Performance Comparison on CulturalBench across Models.

Hard Easy

Model

CulturalBench FIL ID MS SG TH VI AVG FIL ID MS SG TH VI AVG

SEA LION 7B 72.22 58.65 27.27 66.30 74.07 45.37 57.31 11.11 15.38 18.18 13.04 7.41 14.81 13.32 35.32 Sailor2-1b 26.11 25.00 27.27 51.09 74.07 25.00 38.09 31.11 46.15 45.45 30.43 37.04 44.44 39.10 38.60 Qwen2.5-0.5b 26.11 25.00 27.27 55.43 74.07 25.00 38.81 42.22 53.85 45.45 43.48 55.56 33.33 45.65 42.23 Llama-3.1-8B 28.33 31.73 27.27 57.61 25.93 39.81 35.11 53.33 73.08 54.55 60.87 74.07 48.15 60.68 47.89 SeaLLMs-v3-7B 52.22 27.88 27.27 60.87 74.07 31.48 45.63 42.22 57.69 54.55 56.52 66.67 51.85 54.92 50.27 SeaLLM-7B-Hybrid 70.56 36.54 29.55 67.39 74.07 75.93 59.01 37.78 69.23 45.45 39.13 40.74 51.85 47.36 53.19 Gemma-7b 44.44 47.12 50.00 60.87 75.00 34.26 51.95 62.22 65.38 36.36 34.78 70.37 59.26 54.73 53.34 Mistral-7b 76.11 50.00 47.73 64.13 73.15 32.41 57.26 66.67 61.54 36.36 60.87 55.56 55.56 56.09 56.67 Sailor-7b 26.11 49.04 68.18 67.39 76.85 57.41 57.50 44.44 69.23 72.73 60.87 55.56 62.96 60.97 59.23 Sailor2-8b 48.89 41.35 34.09 59.78 80.56 43.52 51.37 64.44 69.23 63.64 60.87 81.48 74.07 68.96 60.16 Qwen2.5-7b 45.56 66.35 36.36 63.04 81.48 60.19 58.83 62.22 61.54 36.36 65.22 85.19 66.67 62.87 60.85 Gemma2-9b 76.67 44.23 38.64 67.39 68.52 42.59 56.34 64.44 69.23 81.82 60.87 77.78 70.37 70.75 63.55 Qwen2.5-14b 44.44 69.23 52.27 66.30 82.41 61.11 62.63 64.44 69.23 72.73 69.57 85.19 74.07 72.54 67.58 Qwen2.5-32b 45.56 57.69 36.36 72.83 87.04 55.56 59.17 68.89 92.31 72.73 78.26 92.59 70.37 79.19 69.18 Gemma2-27b 66.67 55.77 56.82 66.30 74.07 49.07 61.45 84.44 76.92 72.73 86.96 85.19 59.26 77.58 69.52 Sailor2-20b 65.56 64.42 54.55 66.30 83.33 52.78 64.49 80.00 84.62 81.82 73.91 77.78 77.78 79.32 71.90 Qwen2.5-72b 61.67 63.46 59.09 83.70 88.89 65.74 70.43 82.22 80.77 100.00 82.61 92.59 77.78 86.00 78.21

Table 28: Performance Comparison on BLEnD across Models.

Model INST4-ID PERS3-ID ID INST4-JB PERS4-JB JB Overall

Qwen2.5-0.5b 22.71 20.83 21.77 6.11 5.68 5.90 13.83 Llama-3.1-8B 16.25 19.58 17.92 10.70 9.83 10.27 14.09 SeaLLMs-v3-7B 30.21 30.21 30.21 8.95 10.04 9.50 19.85 SEA LION 7B 33.54 34.17 33.86 11.79 11.14 11.47 22.66 SeaLLM-7B-Hybrid 45.83 46.04 45.94 18.56 15.07 16.82 31.38 Qwen2.5-7b 49.58 49.17 49.38 19.00 18.78 18.89 34.13 Sailor2-1b 46.67 44.38 45.53 22.27 23.80 23.04 34.28 Mistral-7b 52.08 50.42 51.25 19.21 17.25 18.23 34.74 Gemma-7b 48.33 46.88 47.61 25.98 24.24 25.11 36.36 Sailor-7b 56.04 55.21 55.63 20.96 21.83 21.40 38.51 Qwen2.5-32b 53.54 53.96 53.75 27.73 30.35 29.04 41.40 Qwen2.5-14b 57.50 56.46 56.98 27.07 25.55 26.31 41.65 Gemma2-9b 60.21 59.79 60.00 34.28 31.88 33.08 46.54 Sailor2-8b 65.00 63.33 64.17 36.68 35.15 35.92 50.04 Gemma2-27b 66.04 65.83 65.94 36.03 36.90 36.47 51.20 Qwen2.5-72b 67.50 65.42 66.46 37.55 36.68 37.12 51.79 Sailor2-20b 68.75 66.67 67.71 38.86 38.21 38.54 53.12

Table 29: Performance Comparison on Global-MMLU across Models.

##### Model ID FIL MS VI Overall

SEA LION 7B 24.83 27.03 24.75 25.97 25.65 Sailor2-1b 33.56 31.26 32.40 32.52 32.44 Qwen2.5-0.5b 36.03 27.75 33.91 36.04 33.43 SeaLLM-7B-Hybrid 37.66 36.61 36.04 35.61 36.48 Sailor-7b 43.03 30.64 43.23 41.08 39.50 SeaLLMs-v3-7B 43.65 36.55 40.19 42.73 40.78 Mistral-7b 45.73 42.85 43.51 39.99 43.02 Llama-3.1-8B 52.94 47.59 50.08 50.68 50.32 Gemma-7b 52.53 52.66 52.66 49.47 51.83 Sailor2-8b 59.34 57.50 56.70 56.71 57.56 Qwen2.5-7b 61.34 53.52 57.39 60.85 58.28 Gemma2-9b 60.72 60.54 59.63 57.05 59.49 Gemma2-27b 66.57 66.40 65.12 62.54 65.16 Qwen2.5-14b 69.83 61.66 64.66 66.74 65.72 Sailor2-20b 69.98 68.06 68.32 66.71 68.27 Qwen2.5-32b 74.13 67.73 70.95 72.53 71.34 Qwen2.5-72b 76.41 74.84 77.07 77.68 76.50

