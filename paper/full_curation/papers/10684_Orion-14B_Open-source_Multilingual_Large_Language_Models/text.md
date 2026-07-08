# arXiv:2401.12246v1[cs.CL]20Jan2024

## Orion-14B: Open-source Multilingual Large Language Models

OrionStar Inc.∗

### Abstract

In this study, we introduce Orion-14B, a collection of multilingual large language models with 14 billion parameters. We utilize a data scheduling approach to train a foundational model on a diverse corpus of 2.5 trillion tokens, sourced from texts in English, Chinese, Japanese, Korean, and other languages. Additionally, we fine-tuned a series of models tailored for conversational applications and other specific use cases. Our evaluation results demonstrate that Orion-14B achieves state-of-the-art performance across a broad spectrum of tasks. We make the Orion14B model family and its associated code publicly accessible1, aiming to inspire future research and practical applications in the field.

### 1 Introduction

Three hundreds years ago, Gottfried Wilhelm Leibniz’s insightful declaration that "Language is the mirror of the mind" profoundly resonates in the contemporary exploration of language. This thought provides a philosophical foundation for understanding the intricate relationship between language and intelligence. In the 20th century, language modeling (LM) became a fundamental approach in artificial intelligence, forming the cornerstone of natural language processing (NLP). The goal of language modeling is to learn the probability distribution of word sequences. Desipite its simple modeling procedure, it encapsulates substantial information about languages. Given that a language contains N words, the potential number of word sequences of the length of L is NL. However, the actual number of sentences commonly used in the language is far fewer than NL. This discrepancy highlights how language models effectively encode linguistic information.

Traditionally, statistical methods were employed to model word frequency in a language. Among these, the N-gram model has been widely adopted, determining the probability of a word based on the previous N − 1 words. Though straightforward and efficient, the method suffers from the data sparsity problem. With the advancement of neural networks, a paradigm shift occurred towards employing neural networks for language modeling. There are many variations of neural language models, such as multi-layer perceptron (MLP) (Bengio et al., 2000), recurrent neural networks (RNN) (Mikolov et al., 2010; Yao et al., 2013), and transformer (Devlin et al., 2019; Vaswani et al., 2017). In recent years, the increase of model sizes and the scale of training data have significantly boosted the capability of language models (Devlin et al., 2019; Peters et al., 2018; Radford et al., 2018). Large language models (LLMs) have exhibited remarkable promise in many traditional NLP tasks, such

- as dialogue system, machine translation, information retrieval. Moreover, LLMs have proven adept
- at complex tasks such as reasoning, code generation, creative writing. These advancements have inspired both the academic and industrial sectors to further investigate the underlying principles and potential applications of LLMs.

The launch of ChatGPT/GPT-3.5 (OpenAI, 2022a) in 2022 captured tremendous attention from the public, pushing the boundaries of what AI can achieve and motivating researchers and engineers to

∗Authors are listed in Appendix A. 1https://github.com/OrionStarAI/Orion

delve deeper into their potential. While GPT-3.5 and its successor, GPT-4 (OpenAI, 2022b), are prime examples of LLMs, their specific model architectures and training methodologies remain undisclosed. In contrast, Meta’s release of LLaMA (Touvron et al., 2023a) and LLaMA 2 (Touvron et al., 2023b) have established a widely-recognized LLM architecture within the open-source community, with numerous libraries adopting these models. Despite LLaMA’s impressive performance, its primary focus on English limits its applicability to other languages. Recently, there has been a surge in the release of multilingual LLMs such as ChatGLM (THUDM, 2023), Baichuan (Baichuan, 2023a,b), Qwen (Bai et al., 2023a), InternLM (InternLM, 2023), XVERSE (Yuanxiang, 2023), Skywork (Wei et al., 2023) and Yi (01-ai, 2023). These models, trained on mainly English and Chinese datasets, have shown promising results in tasks involving both English and Chinese. Additionally, there has been a growing trend of LLMs specifically designed to enhance performance in other languages, such as Japanese (Kojima, 2023; Lee et al., 2023b; Preferred Networks, 2023; Sasaki et al., 2023)and Korean (Kim et al., 2021; Ko et al., 2023a).

In this technical report, we present Orion-14B, a family of multilingual language models with 14 billion parameters. The foundation model is trained on a diverse dataset of 2.5 trillion (2.5T) tokens, containing languages such as English, Chinese, Japanese, Korean, and others. It has demonstrated superior performance across a broad spectrum of tasks in multilingual settings.

We also provides a series of fine-tuned models built upon the foundation model, each trained to different focuses such as conversation, long-context text handling, quantization, and specific application requirements.

The remainder of this report describes our data preparation (Section 2), pretraining methodology (Section 3), fine-tuning methodology (Section 4), evaluation results (Section 5), extension works (Section 6), and conclusions (Section 7).

### 2 Data

In the training framework of LLMs, the role of data is crucial in determining the efficacy and performance of these models. Effective data processing for pretraining is essential for achieving the desired outcomes. This involves acquiring data from diverse sources, ensuring the high quality of the data through thorough filtering, and removing any redundant information. This section will discuss these processes in detail, outlining the necessary steps to prepare and refine data to suit the stringent requirements of LLM training.

#### 2.1 Data Source

Pretraining of LLM needs tremendous amounts of data. Hoffmann et al. (2022) offered guildlines regarding the optimal quantity of training data for models of varying sizes. For example, an LLM with 10 billion parameters requires 205 billion tokens for pretraining. However, recent work (Baichuan, 2023b; Touvron et al., 2023b; Wei et al., 2023) in training 10 billion parameter models have utilized

- 2.5 to 3 trillion tokens, substantially exceeding the recommended data volume. These efforts have yielded notably impressive results, demonstrating the efficacy of training with significantly larger datasets than those proposed in the aforementioned study.

In order to obtain such a large amount of data, it is imperative to collect data from multitude of sources with diversity and high quality. Our dataset incorporates texts in multiple languages, with English and Chinese being predominant, constituting over 90% of the entire dataset. Particular efforts are also made to include Japanese and Korean texts, which account for more than 5% of the dataset. The remaining portion comprises texts in various other languages, such as Spanish, French, German, Arabic, and more.

In terms of content and style, the dataset primarily composes of written language, with spoken language constituting only a minor portion. The dataset spans a broad spectrum of topics, including web pages, news articles, encyclopedic entries, books, source code, and academic publications, among others. The diversity within the dataset is a crucial factor in achieving superior performance across a range of tasks. The detailed distribution of the data sources is shown in Fig. 1. We believe that different types of corpora exert varying influences on the model training process; for instance, some may be more effective to language understanding, while others better facilitate knowledge reasoning. Unlike many existing studies that typically employ random shuffling of training examples,

we strategically feeds the model with varied data sources across different training stages. We believe this method leads to more efficient data usage. The details of this approach will be elaborated in Section 3.

[Figure 1]

Figure 1: Data sources distribution.

#### 2.2 Data Quality

Data quality is essential in the training of LLMs. To assure high-quality data, we have implemented a series of measures for data filtering, detailed as follows:

- • Text normalization: The datasets contain a large number of texts from various sources, such as web pages and ebooks. These texts are often accompanied by HTML, special characters, or other format tags, which are not useful for LLM training. We employ a series of tools, such as regular expressions and format parsers, to effectively eliminate them.
- • Harmful content removal: The Internet contains harmful and spam content. Our approach to mitigate this involves a two-stage process: the initial stage utilizes keywords and regular expressions matching, followed by a deep learning-based model designed to identify and remove such content. It is important to note that entirely eliminating harmful text from the training dataset could lead to a scenario where the trained model lacks the ability to identify and appropriately response to harmful information (Touvron et al., 2023b). Therefore, we intentionally retain a minimal amount of harmful text in the dataset. This approach ensures that the model remains capable of recognizing and effectively addressing such content.
- • Personal information removal: Some of the text data includes personal details like names, phone numbers, and addresses. We utilize rule-based methods for detection and either substitute these with placeholders or remove them entirely.
- • Quality filtering: This part is crucial in data processing. We first apply a set of rules to filter the data, such as removing texts with excessive repetition. Additionally, we use N-gram perplexity models to exclude texts with anomalously high perplexity. Lastly, our proprietary data quality models are employed to select high-quality data. We emphasize that while high quality is essential for LLM training, achieving a balance between quality and quantity of training data is a non-trivial task. Our models are optimized to retain as much data as possible while maintaining high data quality, which is one of the key factors in the successful training of LLMs.

#### 2.3 Deduplication

Given that the training data for LLMs is sourced from a variety of origins, there is a significant likelihood of encountering duplicate data. Duplicate data can detrimentally affect the training process, potentially leading to a model biased towards certain data sources and a decline in performance (Lee et al., 2021; Nunes et al., 2023; Penedo et al., 2023). To address this, we develop a deduplication procedure to eliminate redundant data.

In this process, we extract key words and phrases from each document and compute their corresponding embedding vectors and SimHash vectors (Charikar, 2002; Indyk and Motwani, 1998). These vectors are then compared to those in our database. If a vector in the database shows similarity within a certain threshold, the document is considered a duplicate and is subsequently discarded.

Importantly, we note that while LLMs have shown significant advancements in numerous NLP tasks, some studies (Golchin and Surdeanu, 2023; Wei et al., 2023; Yang et al., 2023) indicate that part of this improvement might be attributed to unintentional inclusion of evaluation data in the training datasets, potentially leading to overestimated results. To address this, we adopt our deduplication approach for all evaluation datasets to prevent the pretraining dataset from containing texts in the evaluation sets, thereby enhancing the integrity and reliability of our model’s evaluation results. We will further discuss the data contamination in detail in Section 5.3.

### 3 Pretraining

#### 3.1 Tokenizer

A tokenizer is a basic component of an LLM, which need to represent the text distribution in the language while maintaining an favorable vocabulary size for training. For a multilingual tokenizer, statistical methods are typically employed to generate word-level or subword-level tokens from texts in multiple languages. We utilize the byte-pair encoding (BPE) algorithm (Shibata et al., 1999), implemented via SentencePiece (Kudo and Richardson, 2018). Our configuration ensures a character coverage of 99.99%, with rare characters defaulting to UTF-8 bytes. To build a diverse corpus and align with our training data distribution, we curate a broad spectrum of text types from our training corpus. This includes English, Simplified Chinese, Traditional Chinese, Japanese, Korean, a few other languages, as well as rare characters. In Table 1, we provide a detailed comparison of our tokenizer with other open-source tokenizers. This comparison includes vocabulary size and compression ratio (CR), the latter calculated by the ratio of the size of the original data to the size of the tokenized data.

- Table 1: Tokenizer comparison with other open-source LLMs. We compare vocabulary sizes and compression ratios for simpifiled Chinese (zh_cn), tranditional Chinese (zh_cn), and English, respectively.

Model Vocab Size CR (zh_cn) CR (zh_tw) CR (en) LLaMA2 32,000 1.377 1.589 1.153 Yi 64,000 0.606 0.785 1.084 Baichuan2 125,696 0.554 0.783 1.077 ChatGLM3 65,024 0.582 0.703 1.081 Skywork 65,519 0.672 0.846 1.153 Orion-14B 84,608 0.549 0.656 1.067

#### 3.2 Architecture

Given that LLaMA2 has achieved superior performance, its architecture has been widely adopted by many open-source LLM. In our approach, we adhere to the LLaMA2 architecture while implementing several modifications. These include extending the number of tokens to 84,608 and enlarging the dimensions of the feed-forward network (FFN) to 15,360. We employ rotary positional embeddings (RoPE) (Su et al., 2021) for positional encoding to accommodate context lengths of up to 4096 tokens. The model uses 40 transformer layers with 40 attention heads each. The total parameter of the model is 14.4 billion, slightly exceeding that of LLaMA2-13B. Detailed model parameters is provided in Table 2.

- Table 2: A comparison of model architecture. The table shows comparison of our model and several open-source model with similar model size.

Model seq_len position embedding hidden size FFN size # layers # heads LLaMA2-13B 4096 RoPE 5,120 13,824 40 40 Skywork-13B 4096 RoPE 5,120 12,288 52 36 Baichuan2-13B 4096 AliBi 5,120 13,696 40 40 Qwen-14B 2048 RoPE 5,120 13,696 40 40 InternLM-20B 4096 RoPE 5,120 13,824 60 40 Orion-14B 4096 RoPE 5,120 15,360 40 40

#### 3.3 Infrastructure

For the training of Orion-14B, we employed Megatron-LM (Shoeybi et al., 2020) on a cluster comprising 11 servers, each equipped with 8 NVIDIA H800 GPUs. To optimize training efficiency, we integrated FlashAttention2 (Dao, 2023) and APEX (NVIDIA, 2023) into Megatron-LM framework, achieving a training speed of 4,000-5,000 tokens/GPU/second.

#### 3.4 Training Process

To train Orion-14B, we initiate the model training with a learning rate warm-up stage spanning 2000 iterations, during which we linearly increase the learning rate to the maximal learning rate of 3e-4. We then apply a cosine schedule to gradually decrease the learning rate to 3e-5 throughout the training processing. We employ the AdamW (Loshchilov and Hutter, 2018) optimizer, setting β1 to 0.9 and β2 to 0.95, respectively. In addition, we apply a weigh decay factor of 0.1 and enforce a gradient clipping threshold of 1.0 to ensure the stability of the training process. The model is trained using BF16/FP32 mixed precision, with a batch size of 1408, corresponding to approximately 5.7 million tokens per step.

#### 3.5 Data Scheduling

Training large language models requires hundreds of billions to trillions of tokens. It is an interesting area to explore scaling laws in LLM training and literature from Kaplan et al. (2020) through Hoffmann et al. (2022) to Touvron et al. (2023b) suggests that model training tends to favor an increase in the number of tokens over model sizes. We use a 2.5T token training dataset for our 14B parameter model, aiming a balance between computational efficiency and cost.

On the other side, while numerous theoretical and empirical studies have examined the interplay between model size and training data volume, there is no universally accepted methodology for scheduling training data. Considering that humans acquire knowledge in a deliberate order (Evanson et al., 2023), it is plausible that language models might also benefit from a structured learning order when processing training data. Curriculum learning (Bengio et al., 2009) has been suggested as a method to organize the learning process by progressively increasing the complexity of the training data. However, most prior studies have concentrated on sample-level data and smaller datasets. Chen et al. (2023) employed a skills-based framework for training data selection and continuous pretraining with a 3B-parameter language model. This approach achieved greater accuracy compared to the baseline method of uniform data source sampling, suggesting the potential efficacy of strategic data scheduling.

In training the Orion-14B model, we intentionally develop a data scheduling strategy that organizes training data to incrementally increase its complexity. We divide the training data into several stages based on the data sources and their complexity. These stages are differentiated by the mix ratios of data sources. Initial stages primarily include data with common knowledge, such as web pages and news articles. In the subsequent stages, we gradually augment the proportion of data containing more complex knowledge, including textbooks, academic papers, and source code. Additionally, the linguistic diversity of the training data is expanded progressively from English and Chinese to Japanese and Korean. The brief structure of our training data schedule is depicted in Table 3.

To assess the effectiveness of the data scheduling approach, we monitor the loss on a validation set throughout the training process. This validation set consists of 5,000 documents, each unseen in the

- Table 3: Training data schedule for Orion-14B. Primary data sources and languages refer to data that totally account for more than 90% of the whole training data in each stage.

Stages Tokens (B) Primary data sources Primary languages Early stages 0 ~600 web pages, news articles English, Chinese Middle stages 600 ~1100 web pages, news articles, textbooks,

academic papers

English, Chinese, Others Final stages 1100 ~2000 web pages, news articles, textbooks,

academic papers, source code

English, Chinese, Others

training set. It includes a diverse collection of English and Chinese texts sourced from a variety of data sources. As shown in Fig. 2, there are significant reduction in validation loss aligning with shifts in the training data distribution at 600B and 1,100B tokens. Additionally, the validation loss exhibits initial fluctuations, stabilizing progressively with continued training. This trend indicates that the model increasingly adapts to the diversity of data types as training progresses.

[Figure 2]

Figure 2: Validation loss during the training process. The validation set consists of 5,000 documents including a diverse collection of English and Chinese texts sourced from a variety of data sources.

To our knowledge, the training of most prior LLMs utilized fully shuffling the training data, which was then fed into the model in a random sequence. Orion-14B is the first LLM trained with a specific data scheduling strategy. The evaluation results indicate that this model demonstrates impressive performance in language understanding tasks at its early stages and rapidly enhances its capabilities in reasoning and academic tasks in later stages, aligning with our data scheduling policy. Notably, Orion-14B, trained on 2.5T tokens, achieves comparable performance to other open-source models trained on 2.6T to 3T tokens, thereby illustrating the efficiency of our data utilization approach.

- 4 Fine-tuning

During the pretraining stage, an LLM is trained to predict the next token at each step. However, in many applications, the model needs to generate a desired response to a given prompt. Thus, in the next stage, LLMs typically undergo further fine-tuning using supervised learning, where the training data consists of paired input and output text sequences. Further, to enhance the quality and safety, approaches like Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017; Ouyang et al., 2022) or Direct Preference Optimization (DPO) (Rafailov et al., 2023) can be employed. In this work, our primary focus is on the supervised fine-tuning (SFT) stage, leaving RLHF and DPO for future exploration.

#### 4.1 SFT Data

High-quality, diverse data has been proven to be crucial to supervised fine-tuning in previous studies (Touvron et al., 2023b; Zhou et al., 2023). To construct our SFT training data, we draw from two primary sources: a human-labeled dataset and an open-source filtered dataset.

For a high-quality human-labeled dataset, we assemble a team of expert annotators who spend weeks creating precisely annotated data. To ensure data quality, all annotators adhere to three key principles—helpfulness, truthfulness, and harmlessness—as outlined in InstructGPT (Ouyang

- et al., 2022) and LLaMA2 (Touvron et al., 2023b). In total, we produce approximately 220,000 human-labeled SFT data entries.

While the human-labeled dataset is of high quality, its volume is insufficient for training a highperformance LLM. Therefore, we also construct a large-scale, open-source filtered dataset. The original SFT data includes collections from various open-source datasets, such as COIG (Zhang

- et al., 2023a), WildChat (Wenting Zhao, 2023), OpenOrca (Lian et al., 2023), and UltraChat (Ding et al., 2023). Given the variable quality and occasional presence of inappropriate content in these open-source datasets, we implement a cleaning process inspired by methods from Platypus (Lee et al., 2023a) and MoDS (Du et al., 2023), comprising the following steps:

- • Rule-based filtering: We use regular expressions and keywords for simple filtering to remove personal information, temporal-sensitive data, etc.
- • Quality filtering: A large NLP model scores the data quality on a scale from 1 to 10, retaining only data with a score of 7 or higher.
- • Semantic deduplication: Text embeddings are used for semantic deduplication, considering texts with a similarity greater than 0.98 as duplicates.

Using this approach, we construct an open-source filtered dataset of 630,000 samples. Combined with the human-labeled data, we assemble an SFT dataset of 850,000 training pairs for supervised fine-tuning.

#### 4.2 Training details

To fine-tune a pretrained LLM, we prepend <human> and <assistant> as headers to the prompt text and the response text, respectively. The training process employs the AdamW optimizer, with hyperparameters configured as follows: β1 is set to 0.9, β2 to 0.95, and ϵ to 1e − 8. We limit the sequence length to 4096 and use a batch size of 128. Our training regimen spanned three epochs, involving over 500k samples; the learning rate was incrementally increased over the first 1,500 steps to a maximum of 1e − 5. To prevent overfitting, we apply a weight decay of 0.1, a dropout rate of 0.1, and a gradient clipping threshold of 1.0.

5 Evaluation

#### 5.1 Standard Evaluation

To effectively evaluate the LLM, we categorize the standard evaluation sets into the examinations and professional knowledge, and language understanding and common knowledge. We select the most common evaluation sets in each category as follows:

#### Professional Knowledge and Reasoning

- • C-Eval (Huang et al., 2023): A comprehensive Chinese evaluation benchmark consisting of more than 10,000 multi-choice questions.
- • CMMLU (Li et al., 2023): A general evaluation benchmark specifically designed to evaluate the knowledge and reasoning abilities of LLMs within the context of Chinese language and culture.
- • MMLU (Hendrycks et al., 2020): A widely used benchmark designed to measure knowledge acquired during pretraining by evaluating models.
- • AGIEval (Zhong et al., 2023): A human-centric benchmark crafted to assess the general capabilities of foundation models in tasks aligned with human cognition and problemsolving.
- • Gaokao (Zhang et al., 2023b): A dataset leverages questions from China’s national college entrance examination to test LLMs.

- • BBH (Suzgun et al., 2022): A challenging subset of the Big-Bench suite, covering a wide array of themes, such as linguistics, mathematics, common sense reasoning, biology, physics, software development, and more.

#### Language Understanding and Common Knowledge

- • RACE (Lai et al., 2017): A comprehensive reading comprehension dataset comprising over 28,000 passages and nearly 100,000 questions. It contains reading and comprehension materials for both middle school (middle) and high school (high) academic levels.
- • HellaSwag (Zellers et al., 2019): A challenge dataset for evaluating commonsense language inference that is particularly difficult for state-of-the-art models.
- • PIQA (Bisk et al., 2020): A dataset introducing the task of physical commonsense reasoning and a corresponding benchmark dataset.
- • Lambada (Paperno et al., 2016): A collection of narrative passages where human subjects can guess the last word if exposed to the whole passage, but not if they only see the last sentence preceding the target word.
- • WSC (Levesque et al., 2012): A pronoun disambiguation task, which requires determining the noun that the pronoun refers to according to the context.

For comparison, we select the most popular LLMs with a parameter range of 10-20 billion: LLaMA 2-13B (Touvron et al., 2023b), Skywork-13B (Wei et al., 2023), Baichuan 2-13B (Baichuan, 2023b), Qwen-14B (Bai et al., 2023a), InternLM (InternLM, 2023).

To ensure consistent comparisons, we employ open-source LLM evaluation frameworks such as OpenCompass (Contributors, 2023) and LM-Eval-Harness (Gao et al., 2021) for a unified performance evaluation of LLMs. For the models we compared, we refer to the published scores from OpenCompass or their official reports.

Table 4: LLM evaluation results on examination and professional knowledge. Bold font denotes the best score in each category, a convention followed in all subsequent tables throughout this paper.

|Model<br><br>|C-Eval CMMLU MMLU AGIEval Gaokao BBH|
|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B|41.4 38.4 55.0 30.9 18.2 45.6 59.1 61.4 62.7 43.6 56.1 48.3 59.0 61.3 59.5 37.4 45.6 49.0 71.7 70.2 67.9 51.9 62.5 53.7 58.8 59.0 62.1 44.6 45.5 52.5<br><br>|
|Orion-14B|72.9 70.6 69.9 54.7 62.1 56.5|

The evaluation results in Table 4 highlight Orion-14B’s superior performance across various examinations and professional knowledge evaluation sets, compared to other LLMs. Orion-14B achieves the highest scores in C-Eval, CMMLU, MMLU, AGIEval, and BBH, indicating its strong capabilities in understanding and reasoning within professional contexts. While it excels in most benchmarks, it is slightly outperformed by Qwen-14B in the Gaokao evaluation. These results position Orion-14B as a highly competitive and robust model for complex and professional tasks.

Table 5: LLM evaluation results on language understanding and common knowledge.

|Model<br><br>|RACE-middle RACE-high HellaSwag PIQA Lambada WSC|
|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B|63.0 58.9 77.5 79.8 76.5 66.3 87.6 84.1 73.7 78.3 71.8 66.3 68.9 67.2 70.8 78.1 74.1 65.4 93.0 90.3 80.2 79.8 71.4 66.3 86.4 83.3 78.1 80.3 71.8 68.3<br><br>|
|Orion-14B<br><br>|93.2 91.3 78.5 79.5 78.8 70.2|

As shown in Table 5, Orion-14B showcases robust performance in language understanding and common knowledge tasks, outperforming other models in RACE (mid and high), Lambada, and WSC benchmarks, highlighting its exceptional comprehension and reasoning abilities. However, for

HellaSwag, PIQA, and WSC tasks, it is slightly outperformed by Qwen-14B and InternLM-20B. Overall, the results indicate Orion-14B’s strong capabilities across a spectrum of natural language understanding benchmarks.

For a comprehensive evaluation, we also utilize all test sets used in OpenCompass leaderboard (Contributors, 2023) to assess performance. In OpenCompass leaderboard, the evaluation sets are organized into five categories. The summarized results for each category are shown in Table 6, where Orion-14B leads with an average score of 64.4%. Notably, it outperforms other models across four categories, including Examination, Language, Understanding, and Reasoning, indicating its excellent analytical and problem-solving abilities. These results demonstrate Orion-14B’s robust capabilities in a wide range of cognitive and language tasks. Detailed results for each testset are included in the Appendix B.

Table 6: LLM evaluation results of OpenCompass testsets

|Model|Average|Examination Language Knowledge Understanding Reasoning<br><br>|
|---|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B|47.3 53.6 49.4 62.4 59.4<br><br>|45.2 47.0 58.3 50.9 43.6<br><br>61.1 51.3 52.7 64.5 45.2 51.8 47.5 48.9 58.1 44.2 71.3 52.7 56.1 68.8 60.1<br>62.5 55.0 60.1 67.3 54.9<br>|
|Orion-14B<br><br>|64.3|71.4 55.0 60.0 71.9 61.6|

Note that, evaluation scores are not the definitive standard for assessing an LLM. Given the vast amount of training data, there is a high likelihood that the dataset includes elements of the evaluation set. To avoid this, we purposely deduplicate the evaluation datasets from our pretraining corpus, thereby ensuring that our model’s performance genuinely reflects its capabilities. Overlooking this critical step could lead to training a model that is overfitted to the evaluation set, resulting in artificially high scores. We will delve into this topic more deeply in Section 5.3.

#### 5.2 Multilingual

In our training approach, while the majority of the data is in English and Chinese, we also incorporate additional languages to enhance multilingual performance. Notably, Japanese and Korean texts are specifically added after surpassing 600B tokens in the training process. The total amounts of Japanese and Korean texts are approximately 100B and 50B tokens, respectively. Despite the lower quantity of Japanese and Korean tokens compared to English and Chinese, the model exhibits superior performance in these languages. This indicates a significant transfer of knowledge from the more dominant languages during the training of the LLM.

To assess the model’s multilingual capabilities, we benchmark it against other models trained on English+Japanese (Kojima, 2023; Lee et al., 2023b; Preferred Networks, 2023; Sasaki et al., 2023), English+Korean (Kim et al., 2021; Ko et al., 2023b), or multilingual datasets (01-ai, 2023; Bai et al., 2023a; Baichuan, 2023b; Touvron et al., 2023b). We employ the datasets from Gao et al. (2021) and Kim et al. (2022) for evaluation of Japanese and Korean, respectively.

- Table 7: Comparison of LLM performances on Japanese testsets. The header of each column stands for Japanese CommonsenseQA, Japanese NLI, MARC in Japanese, Japanese SQUAD, Japanese QKET_v2, XLSUM in Japanese, XWinograd in Japanese, MGSM, respectively.

|Model|Average<br><br>|JCQA JNLI MARC JSQD JQK XLS XWN MGSM|
|---|---|---|
|PLaMo-13B WebLab-10B ELYZA-jp-7B StableLM-jp-7B|52.3 50.7 48.8 51.1<br><br>|56.7 42.8 95.8 70.6 71.0 8.70 70.5 2.40 66.6 53.7 82.1 62.9 56.2 10.0 72.0 2.40 71.7 25.3 86.6 70.8 64.1 2.50 62.1 7.20 33.4 43.3 96.7 70.6 78.1 10.7 72.8 2.80|
|LLaMA 2-13B Baichuan 2-13B Qwen-14B Yi-34B|46.3 57.1 65.8 67.1<br><br>|75.0 47.6 38.8 76.1 67.7 18.1 63.2 10.4 73.7 31.3 91.6 80.5 63.3 18.6 72.2 25.2 85.9 60.7 97.0 83.3 71.8 18.8 70.6 38.0 83.8 61.2 95.2 86.1 78.5 27.2 69.2 35.2<br><br>|
|Orion-14B<br><br>|69.1<br><br>|88.2 75.8 94.1 75.7 85.1 17.3 78.8 38.0|

- Table 8: Comparison of LLM performances on Korean testsets. n = 0 and n = 5 stand for n-shot prompts used in the evaluation. The testsets are originally in English and have been translated to Korean by Kim et al. (2022).

Average HellaSwag COPA BooIQ SentiNeg

Model n=0 n=5 n=0 n=5 n=0 n=5 n=0 n=5 n=0 n=5 KoGPT 53.0 70.1 55.9 58.3 73.5 72.9 45.1 59.8 37.5 89.4 Polyglot-ko-13B 69.6 73.7 59.5 63.1 79.4 81.1 48.2 60.4 91.2 90.2 LLaMA 2-13B 46.7 63.7 41.3 44.0 59.3 63.8 34.9 73.8 51.5 73.4 Baichuan 2-13B 52.1 58.7 39.2 39.6 60.6 60.6 58.4 61.5 50.3 72.9 Qwen-14B 53.8 73.7 45.3 46.8 64.9 68.9 33.4 83.5 71.5 95.7 Yi-34B 54.2 72.1 44.6 44.7 58.0 60.6 65.9 90.2 48.3 92.9 Orion-14B 74.5 79.6 47.0 49.6 77.7 79.4 81.6 90.7 92.4 98.7

Table 9: Multilingual evaluation.

|Model|Train Lang<br><br>|Japanese Korean Chinese English|
|---|---|---|
|PLaMo-13B Weblab-10B ELYZA-jp-7B StableLM-jp-7B|En,Jp En,Jp En,Jp En,Jp<br><br>|52.3 * * *<br><br>50.7 * * * 48.8 * * *<br>51.1 * * *<br>|
|KoGPT-6B Polyglot-ko-13B|En,Ko En,Ko<br><br>|* 70.1 * *<br>* 70.7 * *<br>|
|Baichuan2-13B Qwen-14B LLaMA2-13B Yi-34B|Multi Multi Multi Multi<br><br>|57.1 58.7 50.8 57.1 65.8 73.7 64.5 65.4 46.3 63.7 41.4 55.3 67.1 72.2 58.7 68.8|
|Orion-14B|Multi|69.1 79.5 67.9 67.3|

As shown in Tables 7 and 8, Orion-14 consistently achieves the highest scores across the majority of the test sets. On average, it outperforms all other LLMs in Japanese and Korean datasets, surpassing even those models with a greater number of parameters.

To gain a clearer insight into the multilingual capabilities, we compute the average scores for the evaluation sets in Japanese, Korean, Chinese, and English for comparison. The scores for Japanese and Korean are derived directly from Tables 7 and 8. For the Chinese and English datasets, we calculate the average scores using the OpenCompass dataset, excluding the mathematics and programming testsets.

- Table 9 demonstrates Orion-14B’s impressive performance in multilingual evaluations. It leads with top scores in Japanese, Korean, and Chinese, surpassing other multilingual models. In English, Orion-14B is marginally outperformed by Yi-34B, which is an LLM with a significantly higher number of parameters. This data highlights Orion-14B’s robust proficiency in multiple languages.

#### 5.3 Analysis of Data Contamination

The rise of the LLM has led to a surge in the performance of evaluation tasks. Their superior performance is primarily attributed to the massive data consumed by these billion/trillion-parameter LLMs during training. However, recent work (Golchin and Surdeanu, 2023; Wei et al., 2023; Yang et al., 2023) has shown that the performance of LLM on many downstream tasks may be inflated due to data contamination, i.e., the presence of test data from these downstream tasks in the pretraining data of LLMs.

As mentioned above, to prevent the pretraining dataset from containing answers to the evaluation datasets, we apply our deduplication approach using all evaluation datasets. This process removes text similar to the evaluation data from the training corpus. On the other hand, to understand the influence of such data, we also experiment with training a model using previously deduplicated data. Specifically, we select those data that had been removed due to deduplication with the evaluation set but we do not contain data with the exact same texts as in the evaluation texts. In other words, this approach allows us to use data that may be semantically or partially related to the evaluation set while

excluding the exact text from it. We compile a smaller dataset of 200B tokens, which includes these selected data alongside the regular training data. We then continue the pretraining process with this 200B token dataset, resulting in a new pretrained model named Orion-14B-Exam. As illustrated in the accompanying table, Orion-14B-Exam demonstrates significantly higher scores on the evaluation set compared to the baseline.

Table 10: Evaluation for data contamination and overfitting.

|Model<br><br>|C-Eval CMMLU MMLU Lambada HellaSwag|
|---|---|
|GPT-4 Qwen-72B Yi-34B<br><br>|69.9 71 83 65.5 91.4 83.3 61.8 77.3 76.1 85.4 81.8 82.6 76.3 73.1 82|
|Orion-14B Orion-14B-Exam<br><br>|72.9 70.6 69.9 78.8 78.5 92.7 82.9 85.4 78.5 85.8|

The results in Table 10 reveal that manipulating training data can easily lead to fitting the evaluation dataset and achieve very high scores. We conduct an additional experiment to gauge the extent of overfitting. Specifically, we gather a collection of recent texts from the Internet, ensuring they are unseen in any model’s training set. We then calculate the loss on this new dataset Lunseen and compare it to the loss on texts drawn from the evaluation sets Leval mentioned in Tables 4 and 5, including C-Eval, MMLU, HellaSwag, and others. The loss differential between these two sets serves as an indicator of overfitting—the smaller the difference, the lower the likelihood of overfitting to the evaluation set. The results of this analysis are presented in Table 11. This table illustrates that with the inclusion of the new training dataset, there is a significant reduction in the loss on the evaluation set, decreasing from 1.87 to 1.44, clearly showing the overfitting on the evaluation set. On the other hand, the original Orion-14B model demonstrates consistent losses on both datasets, with a minimal difference as expected, indicating little overfitting levels.

Table 11: Overfitting analysis of the loss of each model.

|Model<br><br>|Lunseen Leval ∆(Lunseen − Leval)|
|---|---|
|Baichuan2-13B Qwen-14B Qwen-72B<br><br>|2.23 1.93 0.30 2.19 1.73 0.46 2.05 1.54 0.51|
|Orion-14B|2.15 1.87 0.28|
|Orion-14B-Exam|2.18 1.44 0.74|

In light of these performance, it is crucial to examine the evaluation methods used in the community of LLM. Since it is possible to achieve high scores through specific training tactics, such scores may not accurately reflect the true capabilities of an LLM. An overemphasis on top leaderboard positions can be misleading and does not guarantee actual model proficiency. The principal goal should be to develop robust, effective LLMs that prove their utility in a wide range of real-world applications.

#### 5.4 Fine-tuned Model Evaluations

The above evaluation utilizes standard evaluation datasets to test the performance of the pretrained foundation model (base-model). On the other hand, evaluating the performance of the fine-tuned model (chat-model) differs from that of the base-model. This is because the chat-model is designed to generate responses to given prompts, and determining the goodness of these responses can be subjective and dependent on the specific task. To comprehensively evaluate the chat-model’s performance, we conduct tests using three different approaches: 1) standard evaluation sets, similar to those used in the base-model evaluation; 2) subjective datasets based on GPT-4 scoring; and 3) human evaluation.

For the standard evaluation, we use widely recognized benchmarks, including CMMLU, MMLU, BBH, HellaSwag, PIQA, and WSC. As indicated in 12, Orion-14B-chat maintains strong performance in HellaSwag, BBH, PIQA, and WSC. However, there is a slight decline in performance on CMMLU and MMLU compared to the base model in Tabels 4 and 5. This is likely due to the evaluation prompts being more designed for the base model than the chat model. Therefore, incorporating subjective evaluation methods alongside standard metrics could provide a more comprehensive assessment of

Table 12: Standard evaluation for chat models. Model CMMLU MMLU BBH HellaSwag PIQA WSC Baichuan2-13B-Chat 58.4 57.0 49.9 66.9 77.6 71.2 Qwen-14B-Chat 70.0 66.4 58.0 65.2 74.0 66.3 LLaMA2-13B-Chat 38.7 54.6 40.2 78.2 78.8 68.3 InternLM-20B-Chat 52.2 52.5 35.3 69.2 76.7 61.5 Orion-14B-Chat 63.9 61.7 49.1 76.7 78.4 71.2

the model’s capabilities. We utilize MT-Bench (Zheng et al., 2023) and AlignBench (Liu et al., 2023) for English and Chinese, respectively.

Table 13: Subjective evaluation of MT-Bench.

|Model<br><br>|First-Turn Second-Turn<br><br>|Average|
|---|---|---|
|Baichuan2-13B-Chat Qwen-14B-Chat LLaMA2-13B-Chat InternLM-20B-Chat<br><br>|7.05 6.47 7.30 6.62 7.10 6.20 7.03 5.93<br><br>|6.76 6.96 6.65 6.48|
|Orion-14B-Chat<br><br>|7.68 7.07<br><br>|7.37|

Table 14: Subjective evaluation of AlignBench. The header of each column stands for Mathematics, Logic, Basic tasks, Chinese understanding, Comprehensive Q&A, Writing, Role-playing, and Professional tasks, and Average scores.

|Model|Math. Logi. Basic. Chi. Comp. Writ. Role. Prof.<br><br>|Avg.|
|---|---|---|
|Baichuan2-13B-Chat Qwen-14B-Chat LLaMA2-13B-Chat InternLM-20B-Chat|3.76 4.07 6.22 6.05 7.11 6.97 6.75 6.43<br><br>4.91 4.71 6.90 6.36 6.74 6.64 6.59 6.56<br><br><br>3.05 3.79 5.43 4.40 6.76 6.63 6.99 5.65 3.39 3.92 5.96 5.50 7.18 6.19 6.49 6.22<br><br>|5.25 5.72 4.70 4.96|
|Orion-14B-Chat|4.00 4.24 6.18 6.57 7.16 7.36 7.16 6.99<br><br>|5.51|

The results presented in Tables 13 and 14 highlight Orion-14B-Chat’s performance in subjective evaluations. In MT-Bench evaluation, Orion-14B-Chat significantly outperforms other models, achieving the highest scores in both First-Turn and Second-Turn evaluations, with an average score of 7.37. In the AlignBench evaluation, Orion-14B-Chat excels notably in Chinese understanding, Writing, Role-Playing, and Professional tasks. The results demonstrate competitive performance across diverse conversational contexts.

As the chat model is designed to generate responses to prompts, human evaluation is a critical measure of its effectiveness. Adopting an approach similar to the arena method used in Chatbot Arena (LMSYS, 2023), we engage human annotators to assess responses from two models in a randomized head-to-head format. Specifically, for a given prompt, responses generated by two anonymized models are presented to the annotators, who then rate them as "Win," "Tie," or "Loss" based on their preference. We have 14 human annotators evaluate a total of 3562 questions. The models compared in this arena battle are Orion-14B-Chat, Qwen-14B-Chat, and Baichuan2-13B-Chat. As indicated in Table 15, Orion-14B-Chat received the highest number of "win" votes, highlighting its exceptional performance in human evaluations.

### 6 Extension Works

In practical applications, LLMs have a variety of needs, including extended context handling, minimizing inference resource requirement, and adapting to specific applications. To address these challenges, we conduct extension works and develop several specialized models. Below are the extensions we have implemented:

- • Orion-14B-Long: This model is optimized for long context lengths more than 200,000 tokens and demonstrates performance comparable to proprietary models on long context evaluation sets (Bai et al., 2023b; Dacheng Li and Zhang, 2023).

Table 15: Human arena evaluation for chat models. Model Win Tie Loss Orion-14B-Chat 1172 1491 899 Qwen-14B-Chat 1101 1592 869 Baichuan2-13B-Chat 728 1601 1233

- • Orion-14B-INT4: A quantized model utilizing 4-bit integer weights. It significantly reduces the model size by 70% and increases the inference speed by 30% while incurring a minimal performance loss of only 1%.
- • Orion-14B-RAG: A chat-model fine-tuned on a custom retrieval augmented generation dataset, achieving superior performance in retrieval augmented generation tasks.
- • Orion-14B-PlugIn: A chat-model specifically tailored for plugin and function calling tasks, ideal for agent-related scenarios where the LLM acts as a plugin and function call system.

Due to time constraints, this technical report does not cover the training details and evaluations of these models. We make all the above models available for public use. For more information, please refer to our open-source library https://github.com/OrionStarAI/Orion.

### 7 Conclusion

In this study, we present Orion-14B, a diverse suite of multilingual large language models with 14 billion (14B) parameters. This family includes a pretrained base model and a fine-tuned chat model, as detailed in this technical report. Additionally, we offer several extensions to Orion-14B, such as a long context model, a quantized model, and several application-oriented models, enhancing its versatility and applicability. These models have demonstrated competitive performance against existing open-source models in the field of LLMs, positioning Orion-14B as a potential strong baseline for future LLM research.

Training a large language model like Orion-14B poses considerable challenges. Throughout this endeavor, we faced numerous obstacles and overcame significant hurdles. We responsibly provide open access to the Orion-14B family and documented our experiences and insights in this technical report, aiming to assist and inspire other researchers in the community.

The journey of LLMs is more than a technological advancement; it is a continuous dialogue between human intelligence and artificial intelligence, constantly evolving and pushing the boundaries of what’s possible. As Ludwig Wittgenstein insightfully remarked, "The limits of my language mean the limits of my world." (Wittgenstein, 1922) This interplay of language and machine learning does more than just reflect our existing world; it unlocks pathways to previously uncharted realms of understanding.

### References

01-ai. https://github.com/01-ai/Yi, 2023. Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge,

Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023a.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508, 2023b.

Baichuan. https://github.com/baichuan-inc/Baichuan-13B, 2023a. Baichuan. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023b.

URL https://arxiv.org/abs/2309.10305. Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. A neural probabilistic language model. Advances in neural information processing systems, 13, 2000.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

Moses S Charikar. Similarity estimation techniques from rounding algorithms. In Proceedings of the thiry-fourth annual ACM symposium on Theory of computing, pages 380–388, 2002.

Mayee F Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, and Christopher Ré. Skill-it! a data-driven skills framework for understanding and training language models. arXiv preprint arXiv:2307.14430, 2023.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models.

##### https://github.com/open-compass/opencompass, 2023.

Anze Xie Ying Sheng Lianmin Zheng Joseph E. Gonzalez Ion Stoica Xuezhe Ma Dacheng Li, Rulin Shao and Hao Zhang. How long can open-source llms truly promise on context length?, June 2023. URL https://lmsys.org/blog/2023-06-29-longchat.

Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4171–4186, 2019.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.

Qianlong Du, Chengqing Zong, and Jiajun Zhang. Mods: Model-oriented data selection for instruction tuning, 2023.

Linnea Evanson, Yair Lakretz, and Jean-Rémi King. Language acquisition: do children and language models follow similar learning stages? arXiv preprint arXiv:2306.03586, 2023.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021. URL https://doi.org/10.5281/zenodo.5371628.

Shahriar Golchin and Mihai Surdeanu. Time travel in LLMs: Tracing data contamination in large language models. arXiv preprint arXiv:2308.08493, 2023.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322, 2023.

Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: towards removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM symposium on Theory of computing, pages 604–613, 1998.

InternLM. Internlm: A multilingual language model with progressively enhanced capabilities.

##### https://github.com/InternLM/InternLM-techreport, 2023.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Dohyeong Kim, Myeongjun Jang, Deuk Sin Kwon, and Eric Davis. Kobest: Korean balanced evaluation of significant tasks, 2022. URL https://arxiv.org/abs/2204.04541.

Ildoo Kim, Gunsoo Han, Jiyeon Ham, and Woonhyuk Baek. Kogpt: Kakaobrain korean(hangul) generative pre-trained transformer. https://github.com/kakaobrain/kogpt, 2021.

Hyunwoong Ko, Kichang Yang, Minho Ryu, Taekyoon Choi, Seungmu Yang, jiwung Hyun, and Sungho Park. A technical report for polyglot-ko: Open-source large-scale korean language models, 2023a.

Hyunwoong Ko, Kichang Yang, Minho Ryu, Taekyoon Choi, Seungmu Yang, Sungho Park, et al. A technical report for polyglot-ko: Open-source large-scale korean language models. arXiv preprint arXiv:2306.02254, 2023b.

Takeshi Kojima. https://huggingface.co/matsuo-lab/weblab-10b, 2023. Taku Kudo and John Richardson. Sentencepiece: A simple and language independent subword

tokenizer and detokenizer for neural text processing. CoRR, abs/1808.06226, 2018. URL http: //arxiv.org/abs/1808.06226.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017.

Ariel N. Lee, Cole J. Hunter, and Nataniel Ruiz. Platypus: Quick, cheap, and powerful refinement of llms, 2023a.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris CallisonBurch, and Nicholas Carlini. Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499, 2021.

Meng Lee, Fujiki Nakamura, Makoto Shing, Paul McCann, Takuya Akiba, and Naoki Orii. Japanese stablelm base alpha 7b, 2023b. URL [https://huggingface.co/stabilityai/ japanese-stablelm-base-alpha-7b](https://huggingface.co/stabilityai/ japanese-stablelm-base-alpha-7b).

Hector Levesque, Ernest Davis, and Leora Morgenstern. The winograd schema challenge. In Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning. Citeseer, 2012.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023.

Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". Openorca: An open dataset of gpt augmented flan reasoning traces. https://https:// huggingface.co/Open-Orca/OpenOrca, 2023.

Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, Xiaohan Zhang, Lichao Sun, Hongning Wang, Jing Zhang, Minlie Huang, Yuxiao Dong, and Jie Tang. Alignbench: Benchmarking chinese alignment of large language models, 2023.

LMSYS. Chatbot arena leaderboard, 2023. URL https://lmsys.org/blog/

##### 2023-05-25-leaderboard/.

Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. 2018.

Tomas Mikolov, Martin Karafiát, Lukas Burget, Jan Cernock`y, and Sanjeev Khudanpur. Recurrent neural network based language model. In Interspeech, volume 2, pages 1045–1048. Makuhari, 2010.

Igor Nunes, Mike Heddes, Pere Vergés, Danny Abraham, Alex Veidenbaum, Alex Nicolau, and Tony Givargis. Dothash: Estimating set similarity metrics for link prediction and document deduplication. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1758–1769, 2023.

NVIDIA. https://github.com/NVIDIA/apex, 2023. OpenAI. Introducing ChatGPT. 2022a. OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2022b.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin, Germany, August 2016. Association for Computational Linguistics. URL http://www.aclweb.

##### org/anthology/P16-1144.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only, 2023.

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL), 2018.

Inc Preferred Networks. Plamo-13b, 2023. URL https://huggingface.co/pfnet/plamo-13b. Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language

understanding by generative pre-training. 2018.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.

Akira Sasaki, Masato Hirakawa, Shintaro Horie, and Tomoaki Nakamura. Elyza-japanese-llama-2-7b,

##### 2023. URL https://huggingface.co/elyza/ELYZA-japanese-LLaMA-2-7b.

Yusuxke Shibata, Takuya Kida, Shuichi Fukamachi, Masayuki Takeda, Ayumi Shinohara, Takeshi Shinohara, and Setsuo Arikawa. Byte pair encoding: A text compression scheme that accelerates pattern matching. 1999.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

##### THUDM. https://github.com/THUDM/ChatGLM3, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of the Conference on Neural Information Processing Systems (NIPS 2017), pages 5998–6008, 2017.

Tianwen Wei, Liang Zhao, Lichang Zhang, Bo Zhu, Lijie Wang, Haihua Yang, Biye Li, Cheng Cheng, Weiwei Lü, Rui Hu, et al. Skywork: A more open bilingual foundation model. arXiv preprint arXiv:2310.19341, 2023.

Jack Hessel Claire Cardie Yejin Choi Yuntian Deng. Wenting Zhao, Xiang Ren. (inthe)wildchat:

650k chatgpt interaction logs in the wild, 2023. Ludwig Wittgenstein. Tractatus logigo-philosphicus. 1922. Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E Gonzalez, and Ion Stoica. Rethinking

benchmark and contamination for language models with rephrased samples. arXiv preprint arXiv:2311.04850, 2023.

Kaisheng Yao, Geoffrey Zweig, Mei-Yuh Hwang, Yangyang Shi, and Dong Yu. Recurrent neural

networks for language understanding. In Interspeech, pages 2524–2528, 2013. Yuanxiang. https://github.com/xverse-ai/XVERSE-13B, 2023. Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine

really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Ge Zhang, Yemin Shi, Ruibo Liu, Ruibin Yuan, Yizhi Li, Siwei Dong, Yu Shu, Zhaoqun Li, Zekun Wang, Chenghua Lin, Wenhao Huang, and Jie Fu. Chinese open instruction generalist: A preliminary release, 2023a.

Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. Evaluating the performance of large language models on gaokao benchmark. 2023b.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, 2023.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models, 2023.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. Lima: Less is more for alignment, 2023.

Appendix

- A Contributions

All contributors sorted alphabetically by last name. Core Contributors: Du Chen, Yi Huang, Xiaopu Li, Yongqiang Li, Yongqiang Liu, Haihui Pan, Leichao Xu, Dacheng Zhang, Zhipeng Zhang. Contributors: Yang Fan, Xuefeng Li, Yuxiang Liu, Haonan Tan, Bingcheng Zhang, Enmao Zhang, Yinglou Zhao.

Human Annotators: Lixiu Chen, Zhenwei Hu, Ningting Luo, Zikang Ma, Jiali Pan, Yuping Qin, Qin Shu, Qin Tu, Haiyan Wu, Jiamin Wu, Jingping Wu, Jing Xia, Simiao Xu, Zhiyong Xue, Chonghuan Yang, Tao Zhu.

Science and Engineering Leadership: Kun Han. We thank the executive team for their support: Sheng Fu, Mingyan Sun, Ting Li.

- B Detailed evaluation results of OpenCompass

- Table 16: Evaluation results of OpenCompass in the examination category

|Model|Average<br><br>|C-Eval CMMLU MMLU AGIEval GaoKao ARC-c ARC-e|
|---|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B<br><br>|45.2 61.1 51.8 71.3 62.5|41.4 38.4 55.0 30.9 18.2 60.3 71.8 59.1 61.4 62.7 43.6 56.1 65.4 79.5 59.0 61.3 59.5 37.4 45.6 38 61.9 71.7 70.2 67.9 51.9 62.5 84.4 90.1 58.8 59.0 62.1 44.6 45.5 81.7 86.1<br><br>|
|Orion-14B<br><br>|71.4<br><br>|72.9 70.6 69.9 54.7 62.1 80.7 88.9|

- Table 17: Evaluation results of OpenCompass in the language category

|Model<br><br>|Average|WiC CHID AFQMC WSC TyDiQA Flores|
|---|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B<br><br>|47.0 51.3 47.5 52.7 55.0|53.3 53.0 69.0 66.3 33.2 7.20 51.1 88.1 69.0 66.3 27.9 5.40<br><br>60.2 83.2 38.0 66.3 30.8 6.40<br><br>50.9 84.7 69.0 66.3 39.8 5.30<br><br>61.8 81.7 69.0 68.3 43.2 6.00<br><br><br>|
|Orion-14B|55.0<br><br>|60.0 90.1 69.0 70.2 32.7 8.13|

- Table 18: Evaluation results of OpenCompass in the knowledge category

|Model|Average<br><br>|BoolQ CommonSenseQA TriviaQA NaturalQuestions|
|---|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B|58.3 52.7 48.9 56.1 60.1<br><br>|82.4 66.7 59.4 24.8 80.9 64.6 48.1 17.2<br><br>67 65.6 46.6 16.3<br><br>86.1 70.1 48.4 19.8<br>87.5 70.6 57.3 25.2<br>|
|Orion-14B<br><br>|60.0<br><br>|84.9 65.7 77.2 12.4|

- Table 19: Evaluation results of OpenCompass in the understanding category

|Model|Average<br><br>|C3 RACE-middle RACE-high OpenbookQA|
|---|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B<br><br>|50.9 64.5 58.1 68.8 67.3|46.1 63.0 58.9 65.0<br><br>64.9 87.6 84.1 83.4<br><br>65.6 68.9 67.2 65.0<br><br><br>90.8 93.0 90.3 94.8 73.7 86.4 83.3 87.6<br><br>|
|Orion-14B|71.9<br><br>|80.2 93.2 91.3 89.8|

Table 20: Evaluation results of OpenCompass in the understanding category (cont’)

|Model|CSL LCSTS XSum EPRSTMT Lambada<br><br>|
|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B<br><br>|58.8 7.80 23.4 58.8 76.5 60.0 17.7 22.6 88.1 71.8 63.1 6.30 25.2 87.5 74.1 54.4 12.5 24.7 86.9 71.4 65.6 12.7 35.5 89.4 71.8|
|Orion-14B<br><br>|62.5 28.9 38.2 83.8 78.8|

Table 21: Evaluation results of OpenCompass in the reasoning category

|Model<br><br>|Average|CMNLI OCNLI AXb AXg RTE COPA ReCoRD<br><br>|
|---|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B|43.6 45.2 44.2 60.1 54.9<br><br>|41.4 34.1 58.3 50.6 47.3 70.0 11.6 32.8 30.0 59.0 53.4 56.3 72.0 1.40 32.7 30.0 59.7 50.6 44.8 71.0 20.7 62.1 58.2 49.5 80.9 71.5 93.0 42.3 43.0 42.5 62.1 75.0 57.8 83.0 63.6|
|Orion-14B<br><br>|61.6|72.6 68.3 71.2 86.5 83.0 82.0 87.8|

Table 22: Evaluation results of OpenCompass in the reasoning category (cont’)

|Model<br><br>|HellaSwag PIQA SIQA MATH GSM8K DROP HumanEval MBPP BBH|
|---|---|
|LLaMA 2-13B Skywork-13B Baichuan 2-13B Qwen-14B InternLM-20B<br><br>|77.5 79.8 54.8 5.00 29.6 46.4 18.9 26.8 45.6 73.7 78.3 70.4 9.80 54.3 41.7 15.9 25.4 48.3 70.8 78.1 44.3 10.1 52.6 45.0 17.1 30.8 49.0 80.2 79.8 78.1 25.2 61.6 53.6 32.3 39.8 53.7<br>78.1 80.3 72.8 7.90 52.6 46.0 25.6 35.6 52.5<br>|
|Orion-14B<br><br>|78.5 79.5 69.4 7.78 51.9 40.8 20.7 29.0 56.5|

