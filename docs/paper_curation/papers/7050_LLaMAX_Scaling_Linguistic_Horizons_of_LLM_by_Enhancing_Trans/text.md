## LLaMAX: Scaling Linguistic Horizons of LLM by Enhancing Translation Capabilities Beyond 100 Languages

### Yinquan Lu1, Wenhao Zhu1,2, Lei Li3, Yu Qiao1, Fei Yuan1∗

1 Shanghai AI Laboratory, 2 Nanjing University, 3 Carnegie Mellon University {luyinquan,yuanfei}@pjlab.org.cn, zhuwh@smail.nju.edu.cn, leili@cs.cmu.edu

# arXiv:2407.05975v2[cs.CL]12Oct2024

### Abstract

Large Language Models (LLMs) demonstrate remarkable translation capabilities in highresource language tasks, yet their performance in low-resource languages is hindered by insufficient multilingual data during pre-training. To address this, we conduct extensive multilingual continual pre-training on the LLaMA series models, enabling translation support across more than 100 languages. Through a comprehensive analysis of training strategies, such as vocabulary expansion and data augmentation, we develop LLaMAX. Remarkably, without sacrificing its generalization ability, LLaMAX achieves significantly higher translation performance compared to existing opensource LLMs (by more than 10 spBLEU points) and performs on-par with specialized translation model (M2M-100-12B) on the Flores-101 benchmark. Extensive experiments indicate that LLaMAX can serve as a robust multilingual foundation model. The code 1 and the models 2 are publicly available.

### 1 Introduction

Large Language Models (LLMs; Brown et al., 2020; Zhang et al., 2022; Chowdhery et al., 2022; OpenAI, 2023; Touvron et al., 2023a,b) exhibit excellence performance in translation tasks involving high-resource languages (Vilar et al., 2023; Zhu et al., 2024b), yet their effectiveness in lowresource translation is suboptimal (Hendy et al., 2023; Bang et al., 2023; Zhu et al., 2024b). Figure 1 illustrates the number of translation directions with performance exceeding 10 spBLEU (Goyal et al., 2022) score on Flores-101 (Goyal et al., 2022). It is evident the majority of models are clustered around the origin point for Arabic-centric translations, demonstrating a significant disparity when compared to their English-centric performance.

∗Corresponding author.

- 1https://github.com/CONE-MT/LLaMAX/.
- 2https://huggingface.co/LLaMAX/.

100

Aya-23-8B

MADLAD-7B

| |
|---|

LLaMA3-8B LLaMA2-7B LLaMAX2-Alpaca-7B

PolyLM-13B

TowerInstruct-7B

| | |
|---|---|
| | |

80

#spBELU(LGX)>10

Yayi2-30B

60

40

20

0

0 20 40 60 80 100 # spBLEU (X LG) > 10

Figure 1: We assess translations in both directions, X→LG and LG→X, across various models using Flores101 test, with X representing all 101 languages included in Flores-101. The results are visualized in a figure where different markers represent various models, a red marker indicates that the language (LG) is Arabic, while a blue marker indicates English. We count the number of translation directions that achieve a spBLEU score higher than 10. The findings indicate that modest LLMs demonstrate strong support for English-centric translation, but underperform in Arabic-centric translation.

This discrepancy is primarily due to the lack of pre-training data for these languages (Wei et al.,

- 2023; Yuan et al., 2024b; Alves et al., 2024). Many researchers are actively working to address this issue. Guo et al. (2024) enhance the LLMs’ ability by translating low-resource languages after learning textbooks. Zhu et al. (2024b) find cross-lingual examples that can provide better task guidance for low-resource translation. In addition to the efforts focus on the fine-tuning stage, some studies have attempted to train a multilingual LLM from scratch (Wei et al., 2023), or to train a languagespecific LLM (Faysse et al., 2024; Alves et al.,
- 2024; Cui et al., 2024). However, the languages covered by these works are not extensive (Wei et al., 2023; Alves et al., 2024; Luo et al., 2023), and the translation performance is still unsatisfactory (Wei

et al., 2023; Alves et al., 2024; Luo et al., 2023).

To tackle this discrepancy, we conduct a massive multilingual continual pre-training for non-English languages. Firstly, we present a comprehensive analysis of critical technical designs, including vocabulary extension (Section 2.1) and data augmentation (Section 2.2). These analyses establish the groundwork for the training procedure, directly influencing the efficacy and, ultimately, the performance of the LLMs. Subsequently, we apply those strategies in continual pre-training using both parallel and monolingual data to enhance the translation performance of LLMs across the 102 languages covered by Flores-101, particularly for lowresource languages.

A primary challenge in expanding language support lies in determining the appropriate vocabulary (Cui et al., 2024; Fujii et al., 2024). After assessing the impact of adding language-specific tokens from various angles: tokenization granularity, embedding quality, and the model’s inner distribution, we find that introducing a small number of new tokens significantly degrades existing LLM performance, while a larger new token set increases training complexity and data requirements. Surprisingly, adhering to the original vocabulary of LLMs emerges as the most cost-effective strategy for extending LLMs to 102 languages.

Another great challenge in extending language support is the scarcity of data for low-resource languages (Chang et al., 2023; Guo et al., 2024). To alleviate the scarcity of training data, we delve into dictionary-based data augmentation (Pan et al., 2021; Lu et al., 2023) and conduct a comprehensive analysis of various augmentation strategies. This analysis takes into consideration different dictionaries and data sources (monolingual or parallel data). We find that the optimal approach for data augmentation involves using parallel data, with the choice of dictionary correlated to the number of target language entities it covers.

Finally, we leverage the above discussed techniques to perform large-scale, multilingual continual pre-training on LLaMA series models (Touvron et al., 2023b; AI@Meta, 2024), resulting in LLaMAX series models (LLaMAX2 and LLaMAX3). The LLaMAX2, trained over 60 days using 24 A100 GPUs, significantly enhances translation capabilities and achieves comparable performance (evaluated on Flores-101) to the specialized translation model M2M-100-12B (Fan et al.,

2021). Specifically, our method demonstrates an average improvement of more than 10 spBLEU compared to baseline models in low-resource-centric translation, as shown in Table 3. Furthermore, when extending our evaluation to Flores-200 (Team

- et al., 2022), it shows significant performance enhancements even for languages not included in the training set. All these translation performance improvements do not compromise general task performance. Interestingly, enhancing translation capabilities also establishes a robust multilingual base model foundation. When comparing results of supervised fine-tuning using task-specific English data on the X-CSQA (Lin et al., 2021a), XNLI (Conneau et al., 2018), and MGSM (Shi
- et al., 2023) tasks, we observe an average improvement of 5 points over LLaMA2. Our main contributions can be summarized as follows:

- • A series of open-sourced LLaMAX models enhance the translation performance across more than 100 languages.
- • Comprehensive analysis of the key techniques in multilingual continual pre-training, including vocabulary extension and data augmentation.
- • Extensive experiments on key technique design, comprehensive translation benchmark evaluation across various models, general task testing, and supervised fine-tuning on task-specific data demonstrate the superiority of LLaMAX.

### 2 Key Technique Design

Existing Pipeline. Exploring adapting pretrained LLMs to new languages without starting from scratch seems to have a concise pipeline, resulting in ChineseLLaMA2 (Cui et al., 2024), Swallow (Fujii et al., 2024), and so on. This pipeline comprises three crucial steps: 1) vocabulary expansion: extending the vocabulary of LLMs by adding new tokens specific to that language and initializing these new tokens as the average of embeddings from the existing tokens (Dobler and de Melo, 2023). 2) continual pre-training: continual pretraining LLM on a large corpus of text data from the target language. 3) instruction tuning: aligning the model with specific tasks or instructions, enhancing its performance. Instead of simply following the pipeline, we analyze primarily two key challenges related to the extension of language support: determining an appropriate vocabulary (in Section 2.1) and improving the effectiveness of data augmentation (in Section 2.2). For a more

# New Token

Romanian (ro) Bengali (bn)

fertility cosine R@1 shift distance # shift token spBLEU fertility cosine R@1 shift distance # shift token spBLEU 0 2.25 0.39 0.37 0.4708 112 32.50 8.62 0.17 0.01 0.4689 112 20.12

100 2.19 0.36 0.34 0.4720 112 28.75 4.96 0.14 0.02 0.4680 113 14.02 800 2.02 0.35 0.36 0.4682 113 27.78 3.21 0.13 0.02 0.4706 113 10.18

1600 1.93 0.34 0.34 0.4690 113 26.40 2.78 0.13 0.02 0.4695 113 1.82

6,400 1.74 0.31 0.31 0.4694 113 22.66 2.15 0.12 0.02 0.4712 113 1.96 12,800 1.63 0.29 0.29 0.0205 1 21.95 1.95 0.12 0.02 - 0 1.84 25,600 1.53 0.27 0.28 - 0 19.72 1.80 0.12 0.02 - 0 2.58 51,200 1.45 0.26 0.25 0.0203 1 17.79 1.70 0.12 0.03 - 0 1.14

- Table 1: Building upon LLaMA2, we add varying numbers of languages-specific new tokens, fully fine-tune LLaMA2, and test the translation performance of en→ro (bn) using Flores-101 test. Furthermore, we assess the effect of new tokens using several metrics: fertility, the cosine similarity with English sentence embeddings, the performance in the English language retrieval translation task (R@1), and the distribution shift of the original embedding vector. Our experiments demonstrate that the inclusion of new words significantly complicates the learning process, underscoring that the integration of new words is a complex task.

detailed analysis, refer to the discussions on the selection of multi-hop translation in the lexicon (see Appendix E) and the format of parallel data during continual pre-training (see Appendix F).

#### 2.1 Existing Vocabulary is Adequate.

Setting. We conduct a series of analytical experiments on the LLaMA2 vocabulary. Our initial focus is on examining the correlation between fertility and the quality of token representation. Here, fertility refers to the ratio of the length of the token sequence produced by the LLaMA2 tokenizer to the length of the input sentence when split by spaces (Chinese and Japanese is split by character). Furthermore, we carry out experiments using 10,000 en→ro and en→bn bilingual sentence pairs from Lego-MT dataset. For new tokens, the BBPE algorithm is executed on language-specific data from MADLAD-400 to produce a vocabulary of 100,000 tokens. Within this vocabulary, language-specific tokens are arranged based on their frequency in the corpus. Subsequently, we identify the top-k tokens (where k is determined by the corresponding “#New Token” in Table 1) that are absent in the original LLaMA vocabulary and incorporate them as new tokens into the LLaMA vocabulary. In each experiment, we introduce a varying number of language-specific new tokens and evaluate each model on the Flores-101.

Research Question 1: Why is adding new tokens considered a straightforward method for extending language support? We assess the quality of representation by en→X translation task. This task identifies the translated result that best aligns with the corresponding English sentence within an extensive target dataset, and evaluates with Recall at top 1, denoted as R@1 (Kabir and Carpuat, 2021).

A higher R@1 value signifies a more robust quality of the representation. Concurrently, we present the cosine similarity of representations generated by LLaMA2 for identical sentences in English and other languages. On experiments across 102 languages, more details in Appendix C, there exists a strong correlation between fertility and the quality of representation, evidenced by a Spearman correlation coefficient of approximately -0.88 for each assessed quality metric.

- Research Question 2: Does adding new tokens to reduce fertility yield prompt performance improvements? Extending vocabulary is a common method to reduce fertility. However, while adding new tokens indeed reduces fertility, it does not necessarily enhance its ability to capture and generalize linguistic patterns across multiple languages. As shown in Table 1, the more new tokens added, the worse the translation performance.
- Research Question 3: What is the impact of adding new tokens on model performance? As demonstrated in Table 1, even the addition of a small number (100) of new language-specific tokens can have a significant impact on the multilingual performance of LLMs. In addition, we conduct a further analysis on the original tokens (32k) embedding distribution and the token number before and after adding new tokens by KSLottery (Yuan et al., 2024a). For more details on KS-Lottery, refer to Appendix D. As the experimental result of “shift distance” and “# shift token” in Tabel 1, fine-tuning the entire model with limited new tokens follows a similar pattern to that with the original vocabulary. However, an excessive number of new tokens can shift the model’s training focus. This holds true regardless of whether the

spBLEU # entity similarity MUSE PanLex ∆ MUSE PanLex ∆ ratio MUSE PanLex ∆

Setting

en→ta 3.74 3.45 -0.29 139,134 91,652 -47,482 0.66 0.08 0.04 -0.04 en→th 5.45 6.14 0.69 21,567 297,573 276,006 13.80 0.20 0.06 -0.14 en→fr 44.03 43.85 -0.18 139,134 568,428 429,294 4.09 0.31 0.35 0.04 en→zh 14.65 16.64 1.99 139134 1,333,762 1,194,628 9.59 0.14 0.09 -0.05 en→es 26.98 27.36 0.38 142,780 433,468 290,688 3.04 0.28 0.32 0.04

- Table 2: Evaluate a specific data augmentation technique with different dictionaries. We measure translation performance (spBLEU), the number of target language entities in the dictionary (# entity), and average cosine similarity of entities (similarity), revealing a strong correlation between performance and “# entity”.

language (ro) is well-supported by the model or not (bn). The influence of these additional tokens is substantial, indicating that the process of enhancing the multilingual capabilities of LLMs is not as straightforward as simply expanding the vocabulary and training with more multilingual data.

Finding: The original vocabulary suffices to present the multilingualism of LLMs. The LLaMA tokenizer, which utilizes the Byte-level Byte Pair Encoding (BBPE; Wang et al., 2019) algorithm, is the foundation for multilingual language processing tasks. Its universal compatibility across all languages, in conjunction with the absence of the requirement for an “unknown” token, optimizes vocabulary sharing (Yuan et al., 2024b) and improves its robustness. It allows the model to understand/generate responses in various languages using the same vocabulary. Meanwhile, studies have shown that LLMs trained on unbalanced English-centric datasets, often use English as an internal pivot language. This helps LLMs to map the inputs closer to English in internal space before generating the output (Zhu et al., 2024a; Huang et al., 2024b; Yoon et al., 2024). Maintaining the original vocabulary helps to preserve this behavior, which also benefits for improving the multilingual capability.

#### 2.2 Data Augmentation

Setting. Given a parallel dataset subset (DP) from DparaA that contains translations in all directions for 6 languages (en,fr,es,zh,ta,th) and a monolingual subset (DM) from DmonoA for the same 6 languages. We then perform non-repetitive sampling 12,500 sentence pairs from DP in each direction to generate two subsets of parallel corpus data DP1 and DP2, respectively. Consequently, we preserve DP1 and evaluate the effect of augmentation on parallel data DP2 or monolingual data DM, resulting in two new dataset, DP′ 2 and DM′ ,

post-augmentation. To assess both the in-domain and out-of-domain capabilities of the model, we perform inference on it using 10 languages (en, fr, es, pt, de, zh, ta, th, is, zu), utilizing the Flores-101.

Finding: The choice of dictionary is related to the number of entities in the dictionary. As shown in Table 2, there is no clear dictionary preference is observed for en/ta/th/zh-centric translation, with optimal performance randomly distributed across the two dictionaries. Furthermore, we conduct an in-depth analysis of the MUSE and PanLex dictionary for translation from en to another 5 languages. We compare the end-to-end translation performance (spBLEU), the number of target language entities in the dictionary (# entity), and the similarity of entities embedding (simple average with entity token embeddings) extracted from the trained model. And find a clear correlation between the translation performance and #entity.

### 3 Training Data Construction

To build powerful LLMs that support translation across a hundred languages, it is crucial to collect and construct a sufficient amount of data.

#### 3.1 Components of Training Data

During the continual pertaining stage, the collected training data covering 102 languages (refer to A, which are all languages supported by Flores-101), mainly consists of two parts: monolingual (DmonoA ) and parallel (DparaA ) data. For languages with limited data availability, we generated a pseudo-parallel dataset (Daug) with multilingual dictionaries: MUSE (Lample et al., 2018) and PanLex (Wang et al., 2022). The whole continual pretraining utilizes over 64 billion tokens. More details on supported languages, dataset description, and data statistics can be found in the Appendix A.

Monolingual Data (DmonoA ). Our monolingual training data includes 94 languages supported by

Algorithm 1: Illustration of the Training Data Construction Process During a Single Training Epoch

Input: A: all language list. DmonoA : monolingual data for all languages. DEn: an English monolingual data. DparaA : a parallel data for all translation directions. Notably, DmonoA DEn = ∅. x: a single data point. g(x; φ): A translation model with parameter φ. In a parallel sentence pair, s represents the language of the source sentence, and t represents the language of the target sentence. f(x; θ): a large language model with parameter θ. h(x, z): augmentation function h enhances input sentence x using the dictionary z.

Output: Dtrain: a training dataset for current training epoch. Dtrain = {} for s ∈ A do

Dmonos ⊂ DmonoA // Extract a s-specific monolingual subset for t ∈ A do

Dpara ← Dparas→t ∪ Dparat→s Dparas ⊂ Dpara // Extract the s-centric parallel subset if |Dparas | < 25, 000 then

// The quantity of 25,000 determined by the machine’s memory capacity DEns ⊂ DEn, s.t. |DEns | = 25, 000 − |Dparas | // Extract an English subset for s language DEns→t ← g(x; φ) or DEnt→s ← g(x; φ), where x ∈ DEns Daugs→t ← h(x, z), where x ∈ DEns→t, or Daugt→s ← h(x, z), where x ∈ DEnt→s Daugs ← Daugs→t ∪ Daugt→s

end Dtrain ← Dtrain ∪ Dmonos ∪ Dparas ∪ Daugs

end

Flores-101 from MC4 (Xue et al., 2021) and MADLAD (Kudugunta et al., 2024), totaling 40,000,000 sentences. To ensure efficient handling and processing of the data, we use a strategy in which each piece of monolingual data is split into multiple entries, with a block size of 512.

Parallel Data (DparaA ). Our parallel data from Lego-MT (Yuan et al., 2023) encompasses 102 languages, forming a total 4,737 language pairs and 9,474 translation directions. For each translation direction, denoted as source language (s) to target language (t), we concatenate each translation pair, merely using a space as a delimiter, to form a single entry for training data. For each language pair, the probability of occurrence for each translation direction, for example, s → t and t → s is set as 50%. During the training stage, the gradient is computed for the entire data entry, rather than only for the target sentence. For language pairs that have fewer than 25,000 (bound by machine resources) sentence pairs, we replicate the original data thrice (Muennighoff et al., 2023).

Data Generated Through Augmentation (Daug). The way which is followed by Pan et al. (2021), to obtain code-switch data consists of two steps: 1) build multilingual lexicons; 2) construct pseudoparallel data. We show the data augmentation process in Figure 2.

Step 1: Building multilingual lexicons. The existing multilingual dictionaries, MUSE and PanLex, encompass multiple bilingual dictionaries, such as

Parallel Data

|Hello,|
|---|

|today|
|---|

|good|
|---|

|Hallo,|
|---|

|heute|
|---|

|gut|
|---|

[source]: :

day. [target]: 你好，今天是个好日子。

is a

[source]:

day. [target]: 你好，今天是个好日子。

H

h is a

Monolingual Data

|你好，|
|---|

|aujourd'hui|
|---|

|好的|
|---|

好 day. [target]: Hello, today is a good day.

，

day. [source]:

is a

|Hello,|
|---|

|today|
|---|

|good|
|---|

is a

Figure 2: A case illustrating the detailed process of constructing pseudo-parallel data using multilingual dictionary from monolingual or parallel data sources.

en-fr, en-de, en-zh bilingual dictionaries. A dictionary comprises numerous entries, each being a word or a term defined, usage, and provided with other relevant information. We iterate through each entry in the bilingual dictionary, reformat all entries, and create entries in the format of {entity}_{language}. For instance, the English word “hello” as translation in three bilingual dictionaries (en-fr, en-de, en-zh), leading us to construct a multilingual lexicons entry as hello_en, Bonjour_fr, Hallo_de, 你好_zh.

Step 2: Constructing pseudo-parallel data. The foundational data for construction can be based on either parallel or monolingual data, as shown in Figure 2. For each sentence, we convert it to lowercase and subsequently divide it into multiple words using spaces (for Chinese sentences, the Jieba tokenizer is utilized). In parallel data processing, words in a source sentence are randomly replaced

en-X zh-X de-X ne-X ar-X az-X ceb-X System Size

COMET BLEU COMET BLEU COMET BLEU COMET BLEU COMET BLEU COMET BLEU COMET BLEU Encoder-Decoder Models

M2M-100∗ (Fan et al., 2021) 418M 63.76 17.26 61.41 10.13 61.62 14.10 46.98 4.03 59.97 11.52 45.75 4.17 44.23 6.13 M2M-100∗ (Fan et al., 2021) 1.2B 70.00 21.54 67.29 13.13 67.62 17.73 56.04 7.14 62.62 12.57 52.39 6.06 52.79 9.46 M2M-100∗ (Fan et al., 2021) 12B 74.19 24.74 71.56 14.91 72.07 20.34 62.19 9.68 68.91 16.36 54.78 6.24 60.09 12.48 Lego-MT∗ (Yuan et al., 2023) 1.2B 69.49 24.96 68.23 16.28 69.20 21.42 68.37 16.98 65.57 18.38 65.69 13.51 58.21 16.83 NLLB-200 (Team et al., 2022) 1.3B 81.69 31.77 78.05 19.61 79.49 25.99 81.63 23.65 78.66 24.32 78.46 19.18 76.50 23.71 MADLAD-400 (Kudugunta et al., 2024) 7B 77.79 29.19 74.07 18.23 74.73 23.15 72.74 17.74 74.53 22.14 61.29 9.92 64.44 15.29 Aya-101 (Üstün et al., 2024) 13B 77.26 24.30 75.29 15.50 76.17 20.86 77.78 18.65 74.82 18.44 75.36 15.46 71.90 18.76

LLM based Decoder-Only Models LLaMA2 (Touvron et al., 2023b) 7B 43.95 4.21 44.62 0.91 45.26 2.14 38.22 0.39 39.43 0.54 47.43 0.68 33.50 1.49

- LLaMA2 (Touvron et al., 2023b) 13B 31.37 0.24 34.91 0.25 31.22 0.10 35.32 0.21 32.34 0.11 36.03 0.17 30.84 0.17

- LLaMA3 (AI@Meta, 2024) 8B 45.04 3.84 45.14 3.50 42.11 3.27 44.15 2.65 39.36 2.36 43.00 1.86 36.06 2.43 LLaMA2-Alpaca (Taori et al., 2023) 7B 52.83 9.44 51.29 3.80 51.47 6.82 46.59 1.31 46.76 2.84 48.63 1.36 41.02 2.69

- LLaMA2-Alpaca (Taori et al., 2023) 13B 57.16 11.85 53.93 6.25 54.70 9.42 51.47 3.11 50.73 5.23 50.68 2.74 47.86 4.96

- LLaMA3-Alpaca (Taori et al., 2023) 8B 67.97 17.23 64.65 10.14 64.67 13.62 62.95 7.96 63.45 11.27 60.61 6.98 55.26 8.52 PolyLM (Wei et al., 2023) 13B 45.16 5.72 52.41 1.42 47.89 3.59 38.00 0.45 45.82 1.04 38.65 0.57 29.74 0.77 Yayi2 (Luo et al., 2023) 30B 54.13 7.80 55.23 4.38 56.48 4.72 47.88 0.92 49.45 1.73 53.06 1.23 36.75 1.87 TowerInstruct (Alves et al., 2024) 7B 58.69 9.41 57.75 4.15 58.31 6.79 51.42 2.07 50.76 3.35 48.01 1.79 41.69 3.36 Aya-23 (Aryabumi et al., 2024) 8B 57.91 11.18 56.65 7.20 55.69 9.30 51.78 3.50 55.49 8.00 51.45 3.27 44.14 4.24 Qwen2-Instruct (Bai et al., 2023) 7B 59.64 9.61 59.70 6.84 57.44 7.69 58.62 4.40 57.22 6.35 54.49 3.83 49.61 3.76 ChineseLLaMA2-Alpaca (Cui et al., 2024) 7B - - 49.72 2.31 - - - - - - - - - -

LLaMAX2-Alpaca 7B 76.66 23.17 73.54 14.17 73.82 18.96 74.64 14.49 72.00 15.82 70.91 11.34 68.67 15.53 LLaMAX3-Alpaca 8B 75.52 22.77 73.16 14.43 73.47 18.95 75.13 15.32 72.29 16.42 72.06 12.41 68.88 15.85

X-en X-zh X-de X-ne X-ar X-az X-ceb System Size

COMET BLEU COMET BLEU COMET BLEU COMET BLEU COMET BLEU COMET BLEU COMET BLEU Encoder-Decoder Models

M2M-100∗ (Fan et al., 2021) 418M 68.47 21.19 62.15 10.34 60.19 14.25 40.43 1.30 63.33 11.53 49.74 2.44 47.80 4.85 M2M-100∗ (Fan et al., 2021) 1.2B 73.06 26.26 67.91 12.94 67.78 19.33 42.60 1.40 60.28 8.57 55.86 4.58 55.87 6.83 M2M-100∗ (Fan et al., 2021) 12B 74.45 28.01 69.27 13.35 70.17 21.31 45.50 2.85 69.94 15.15 61.36 6.44 57.07 8.77 Lego-MT∗ (Yuan et al., 2023) 1.2B 75.44 30.71 71.41 16.42 70.75 23.75 59.66 15.02 70.73 18.21 66.73 11.88 59.28 15.06 NLLB-200 (Team et al., 2022) 1.3B 84.22 38.60 76.75 15.27 79.50 25.71 73.70 21.84 79.85 21.80 80.02 15.55 69.05 24.72 MADLAD-400 (Kudugunta et al., 2024) 7B 83.05 38.14 78.49 20.48 77.50 26.79 61.94 13.93 77.84 22.25 75.41 13.85 51.33 4.24 Aya-101 (Üstün et al., 2024) 13B 80.72 31.92 78.51 22.49 77.37 15.43 69.69 17.13 77.90 16.54 78.70 13.51 67.76 21.58

LLM Based Decoder-Only Models LLaMA2 (Touvron et al., 2023b) 7B 55.46 11.80 43.50 0.55 43.10 3.22 34.41 0.42 39.13 0.25 43.98 0.59 41.64 1.16

- LLaMA2 (Touvron et al., 2023b) 13B 38.25 0.75 37.06 0.22 31.73 0.25 30.13 0.15 33.68 0.06 33.47 0.08 37.49 0.20

- LLaMA3 (AI@Meta, 2024) 8B 67.66 19.81 42.52 1.37 49.42 6.61 33.38 0.52 34.12 0.49 37.27 0.79 37.97 1.41 LLaMA2-Alpaca (Taori et al., 2023) 7B 65.85 16.44 56.53 4.46 56.76 9.01 34.96 1.03 44.10 2.18 40.67 0.63 45.69 1.73

- LLaMA2-Aplaca (Taori et al., 2023) 13B 68.72 19.69 64.46 8.80 62.86 12.57 38.88 2.16 52.08 4.48 41.18 0.87 48.47 2.51

- LLaMA3-Alpaca (Taori et al., 2023) 8B 77.43 26.55 73.56 13.17 71.59 16.82 46.56 3.83 66.49 10.20 58.30 4.81 52.68 4.18 PolyLM (Wei et al., 2023) 13B 50.98 7.75 42.60 1.20 43.95 3.69 33.69 0.36 42.27 1.67 40.24 0.44 39.29 0.96 Yayi2 (Luo et al., 2023) 30B 68.06 19.37 57.81 6.07 53.82 5.62 40.95 0.48 46.61 0.52 49.29 0.71 45.50 1.71 TowerInstruct (Alves et al., 2024) 7B 65.37 18.87 64.26 10.37 60.73 12.81 38.80 0.62 44.72 0.39 47.17 0.71 47.15 2.24 Aya-23 (Aryabumi et al., 2024) 8B 67.53 20.57 66.11 11.20 63.09 14.09 44.33 2.69 63.59 11.84 46.97 1.19 45.17 2.29 Qwen2-Instruct (Bai et al., 2023) 7B 73.25 19.04 72.52 13.52 64.61 11.33 41.41 2.27 64.94 8.50 47.96 1.66 55.45 3.00 ChineseLLaMA2-Alpaca (Cui et al., 2024) 7B - - 55.06 6.15 - - - - - - - - - -

LLaMAX2-Alpaca 7B 80.55 30.63 75.52 13.53 74.47 19.26 67.36 15.47 75.40 15.32 72.03 10.27 65.05 16.11 LLaMAX3-Alpaca 8B 81.28 31.85 78.34 16.46 76.23 20.64 65.83 14.16 75.84 15.45 70.61 9.32 63.35 12.66

- Table 3: Comparison with different architecture, including encoder-decoder and decoder-only models, on Flores101 dataset, where X refers to any language in 101 languages. ∗ refers to that model comparisons are restricted to 85 languages, denoted as |X| = 85. We make this choice because the M2M-100 baselines cover only 86 languages, as reported in the work by Flores-101 (Goyal et al., 2022; Yuan et al., 2023). This table compares our instructionaligned LLaMAX2 model (LLaMAX2-Alpaca) with the instruction-aligned LLaMA2 model (LLaMA2-Alpaca) to demonstrate the benefits of our multilingual continual pre-training. Additionally, we compare LLaMAX with other open-source multilingual-focus LLMs to highlight the impressive multilingual capabilities.

System Size

TED (en-X) TED (X-en) TICO (en-X) WMT23 (en-X) WMT23 (X-en)

COMET BLEU COMET BLEU COMET BLEU COMET BLEU COMET BLEU LLaMA2 (Touvron et al., 2023b) 7B 52.15 3.34 61.54 8.66 39.63 3.45 51.55 2.96 65.68 14.87

- LLaMA2 (Touvron et al., 2023b) 13B 34.66 0.17 40.87 0.49 31.65 0.42 33.74 0.43 41.18 0.85

- LLaMA3 (AI@Meta, 2024) 8B 44.72 2.09 53.56 6.04 40.02 4.82 47.44 2.61 55.18 7.84 LLaMA2-Alpaca (Taori et al., 2023) 7B 62.04 9.15 68.62 12.67 44.73 8.60 73.17 17.23 75.82 24.97

- LLaMA2-Alpaca (Taori et al., 2023) 13B 65.62 11.40 70.74 14.54 48.64 10.79 77.93 21.60 77.90 28.67

- LLaMA3-Alpaca (Taori et al., 2023) 8B 73.20 14.13 75.03 16.83 56.73 14.49 80.05 24.11 79.22 29.76 PolyLM (Wei et al., 2023) 13B 50.18 5.53 55.16 7.28 40.36 7.17 62.67 10.62 69.15 19.09 Yayi2 (Luo et al., 2023) 30B 61.53 8.54 70.92 14.09 47.02 7.91 65.69 10.76 75.60 20.47 TowerInstruct (Alves et al., 2024) 7B 64.83 8.22 70.91 15.29 50.48 10.14 74.03 18.42 80.08 30.03 Qwen2-Instruct (Bai et al., 2023) 7B 66.68 8.84 71.83 13.37 55.16 11.47 75.11 18.86 77.48 25.61 Aya-23 (Aryabumi et al., 2024) 8B 68.06 10.69 72.87 16.44 52.44 12.98 83.29 27.15 82.00 31.21

LLaMAX2-Alpaca 7B 75.58 16.12 76.18 17.81 68.33 19.79 80.17 23.91 79.55 30.30 LLaMAX3-Alpaca 8B 74.95 15.15 76.99 18.47 67.71 20.06 79.96 24.49 79.88 30.34

- Table 4: Benchmarking results on WMT23, TED and TICO dataset. X denotes various languages across different translation benchmarks; detailed information is available in Appendix A. Evaluation results across these benchmarks further validate the strong multilingual translation capabilities of LLaMAX.

Knowledge Commonsense Reasoning Math Reasoning Code

Avg.

MMLU BBH NQ HellaSwag Winogrande GSM8K Math HumanEval MBPP

LLaMA2-Alpaca 44.22 37.95 24.32 31.12 61.09 14.03 3.82 14.63 27.63 28.76 LLaMAX2-Alpaca 44.60 38.25 23.21 33.75 61.48 12.21 3.74 12.20 25.29 28.30

- Table 5: Evaluation results, assessed by OpenCompass (Contributors, 2023), on monolingual general benchmarks.

with translation from a different language using the multilingual dictionary created in Step 1. During the training, the loss is computed solely on the target sentence. In monolingual data processing, each word is individually replaced with a randomly chosen word from the multilingual dictionary. If no suitable replacement word in another language is found, the original word remains unchanged. Consequently, the modified sentence and the original sentence can form pseudo-parallel data. During the training, the loss is computed solely on both the source and the target sentence.

#### 3.2 Training Algorithm.

Given an LLM f(x;θ) on a collected training data {x(i)}ni=1, where θ is the pre-trained parameters, our objective is to obtain an LLM through continual pre-training, denoted as f(x;θ′). Here, θ′ indicates the updated parameters. The target of f(x;θ′) is to preserve the general capabilities of the model in high-resource languages while simultaneously enhancing the translation performance across all translation directions among 102 languages. The process of constructing training data is outlined in Algorithm 1. We gather monolingual data for each of the languages and parallel data for every translation direction. In particular, there is no augmentation for translations involving highresource languages. Instead, we solely augment the translation data that is insufficient by utilizing a trained translation model, Lego-MT model. Then we train the f(x;θ), the loss function is:

arg max

θ

Ti

n

log f(x(ti)|x<t(i);θ) (1)

t=1

i=1

where T is the total decoding time step.

After continual pre-training, we perform instruction tuning on LLaMAX using Alpaca (Taori et al.,

- 2023), a dataset comprising 52,000 English instruction examples. This process enhances the model’s capability to comprehend and follow instructions without introducing additional multilingual information, resulting in LLaMAX-Alpaca. We are currently using Alpaca to enhance the model’s capacity for instruction following. In the future, we

will release a more robust instruction model finetuned with a multilingual instruction dataset.

### 4 Benchmarking Results

In this section, we present multilingual benchmarking results to comprehensively demonstrate the potential of LLaMAX2. We evaluate translation quality with spBLEU (Goyal et al., 2022) and COMET22 (Rei et al., 2020) for both LLMs and translation models. See Appendix B for training details on LLaMAX2 and description of baseline models.

We significantly enhances the multilingual translation capabilities of the base LLaMA2 model through massive multilingual continual pretraining. The benefits of our continual pretraining is enhancing the base LLM’s multilingual translation capabilities. Evaluation results on Flores-101 benchmark are shown in Table 3. By comparing our multilingual-enhanced model with the base LLaMA2 model in instruction-tuned versions (LLaMAX2-Alpaca vs. LLaMA2-Alpaca), we consistently observe a significant performance improvement on both English-centric and nonEnglish-centric translation. In addition to Flores101, we also make evaluation on a range of diverse translation benchmarks (Table 4). The performance enhancement brought by our multilingual continual pre-training is consistent across these benchmarks.

LLaMAX outperforms other open-source decoder-only LLMs on multilingual translation by a large margin. Next, we compare LLaMAX2-Alpaca model with other open-source decoder-only LLMs built for multilingual purposes (Table 3, Table 4). Compared to other from-scratch trained LLMs, such as PolyLM, Yayi2, LLaMAX2 consistently shows better performance across various multilingual translation benchmarks, indicating that the LLaMA2 base model provides a strong foundation for language extension. Furthermore, when compared to other LLaMAbased continual pre-trained models, such as TowerInstruct, LLaMAX2 also achieves superior performance, demonstrating the effectiveness of our optimized continual pre-training pipeline.

90

| |50.93<br><br>70.62<br><br>38.32<br><br>55.11<br><br>76.22<br><br>43.70<br><br>LLaMA2-Task<br><br>LLaMAX2-Task| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

80

70

Accuracy

60

50

40

30

X-CSQA XNLI MGSM

Figure 3: Comparison results between instructiontuning our multilingual enhanced model and the base model with specialized instruction data. We take XCSQA, XNLI, MGSM as three examples tasks.

| |9.44<br><br>6.04<br><br>16.44<br><br>11.89<br><br>23.17<br><br>14.44<br><br>30.63<br><br>22.14<br><br>LLaMA2-Alpaca<br><br>LLaMAX2-Alpaca| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

30

25

20

spBLEU

15

10

5

0

En X (seen) En X (unseen) X En (seen) X En (unseen)

Figure 4: Comparison results between LLaMAX2Alpaca and LLaMA2-Alpaca on Flores-200. Some nonEnglish languages are not covered in Flores-200, but LLaMAX2 also boosts its translation performance.

LLaMAX benefits unseen long-tail low-resource languages as well. A significant challenge in multilingual enhancement is that the substantial cost of collecting scarce multilingual resources makes it prohibitive to cover massive languages. While our multilingual pre-training corpus already covers 102 languages, we acknowledge that there remains a large group of long-tail, low-resource languages that are not well covered. To assess the generalization capability of LLaMAX2, we evaluate it on Flores-200 dataset and observe its performance on these unseen languages (Figure 4). We find that for languages not encountered during training, LLaMAX2 still achieves significant improvements, demonstrating the generalization capability of our massive continual pre-training.

LLaMAX is closing the performance gap between open-source LLM translator and specialized encoder-decoder translation systems. While LLaMAX2 has achieved the state-of-theart translation performance among open-source decoder-only LLMs, the next critical question is whether we can close the gap between LLMs and specialized encoder-decoder translation systems. Table 3 provides a comprehensive comparison, reveals LLaMAX2 has reached the level of the M2M100-12B model. Future work will be needed to optimize the language extension framework to match the performance of advanced translation systems.

LLaMAX provides a better starting point for specialized instruction-tuning on English task data. In the end, we demonstrate the usage of our continual pre-trained model (LLaMAX2) on tasks beyond translation. While in previous experiments we use basic Alpaca instruction data to teach LLM to follow translation instructions,

we now show that our released checkpoint can be enpowered to handle more multilingual tasks beyond translation. Figure 3 presents three example tasks where we use specialized instruction data to unlock LLaMAX2’s abilities on specific tasks, such as math reasoning and common sense reasoning. We find that the instruction-tuned LLaMAX2 model outperforms its LLaMA2 model counterpart in non-English performance across all three tasks, demonstrating that provides a better starting point for instruction-tuning with task-specific data.

LLaMAX circumvents catastrophic forgetting issue. A common concern with continual pretraining on additional multilingual corpus is that the process might disturb the parametric knowledge and working pattern of the original model, a phenomenon known as catastrophic forgetting (Goodfellow et al., 2013). Furthermore, we compare LLaMAX2 with LLaMA2 on popular English benchmarks that measure a diverse set of core capabilities of LLMs. Experiment results in Table 5 show that the two models achieve very similar performance on these benchmarks (More details about these benchmarks are in Appendix A.), demonstrating that our continual pre-training does not compromise the general capability of the base model.

Beyond the English-centric translation is more efficient and effective. We further investigate the necessity and feasibility of multilingual augmentation for an English-centric LLM. We can effectively transform a translation task (src→trg) from the source language (src) to the target language (trg) into src→en and en→trg, which allows us to leverage the power of English as a central language, facilitating seamless communication and comprehension across various language

spBLEU COMET LLaMA3-Alpaca LLaMAX2-Alpaca LLaMA3-Alpaca LLaMAX2-Alpaca src→trg src→en→trg src→trg src→en→trg src→trg src→en→trg src→trg src→en→trg

Direct

zh→X 10.14 11.34 14.17 15.54 64.65 66.61 73.54 74.74 X→zh 13.17 15.37 13.53 15.11 73.56 75.66 75.52 77.21 de→X 13.62 14.24 18.96 19.38 64.67 65.79 73.82 74.36 X→de 16.82 18.08 19.26 20.71 71.59 73.11 74.47 76.04 ar→X 11.27 12.60 15.82 17.10 63.45 65.33 72.00 73.17 X→ar 10.20 10.88 15.32 16.00 66.49 69.54 75.40 76.32 ne→X 7.96 10.29 14.49 16.16 62.95 67.87 74.64 76.86 X→ne 3.83 7.08 15.47 16.86 46.56 58.89 67.36 69.47 az→X 6.98 9.52 11.34 13.54 60.61 65.16 70.91 73.60 X→az 4.81 6.96 10.27 11.44 58.30 67.52 72.03 75.60

ceb→X 8.52 10.69 15.53 16.98 55.26 60.71 68.67 70.76 X→ceb 4.18 7.17 16.11 18.94 52.68 59.55 65.05 66.52

Avg. 9.29 11.19 15.02 16.48 61.73 66.31 71.95 73.72

- Table 6: We can convert a translation task from the source language (src) to the target language (trg), represented as src→trg, to src→en→trg. The experimental results indicate that the performance of English as a powerful pivot falls short compared to LLaMAX2-Alpaca (LLaMA3 pivot translation vs. LLaMAX2-Alpaca). Furthermore, conducting similar pivot translation experiments on LLaMAX2-Alpaca can further improve translation performance.

pairs. We refer to this experimental setup as a pivot translation experiment. As shown in Table 6, the experimental results demonstrate that the pivot translation experiments effectively leverage the power of English to enhance translation performance (compared src→en→trg to src→trg on the same model), although it still falls short of the results obtained from large-scale multilingual continual pre-trained models (LLaMA3-Alpaca src→en→trg vs. LLaMAX2-Alpaca src→trg). Interestingly, conducting pivot translation experiments based on LLaMAX2-Alpaca reveals the potential for significant improvements in translation performance (LLaMAX2-Alpaca src→en→trg vs. LLaMAX2-Alpaca src→trg).

### 5 Related Work

Multilingual Large Language Models. Large Language Model (LLMs; OpenAI, 2023; Zhang

- et al., 2022; Brown et al., 2020; Chowdhery et al., 2022; Touvron et al., 2023a,b) trained with Englishcentric data can also solve various non-English tasks (Hendrycks et al., 2021a,b; Srivastava et al., 2022; Kwiatkowski et al., 2019; Hendrycks et al., 2021c), but the performance between non-English and English is significantly large (Yuan et al.,

2024b). Efforts to develop more multilingual LLMs in two different ways: retraining LLMs with diverse multilingual data from scratch (Wei

- et al., 2023); or continuous training of pre-trained models using language-specific data with the option to expand the vocabulary (Zhao et al., 2024a; Cui et al., 2024; Faysse et al., 2024; Alves et al.,
- 2024). Instead of training from scratch, continual

pre-training aims at updating pre-trained models with new data, making the process more efficient and cost-effective (Gupta et al., 2023; Alves et al., 2024; Xie et al., 2023).

Multilinguality in LLMs. Recent research has shed light on the multilingual capabilities of LLMs. A comprehensive survey by Huang et al. (2024a) discusses various aspects of multilingualism in LLMs, including training and inference methods, model security, multi-domain with languages culture, and emphasizes the need for language-fai technology. Yuan et al. (2024b) analysis multilingualism of LLMs from the vocabulary sharing aspect. Zhao et al. (2024b) delve into the architecture of LLMs to find how LLMs handle multilingualism. Recently, Li et al. (2024) quantify the multilingual performance of LLMs. These studies provide valuable insights into the multilingual capabilities of LLMs, and the key technical design of continual pre-training for LLaMAX.

### 6 Conclusion

In this work, we enhance the series models of LLaMA translation performance for 102 languages through continual pre-training, creating LLaMAX. We compare LLaMAX ’s translation capabilities with other decoder-only LLMs and encoderdecoder models across multiple benchmarks. LLaMAX is also assessed on general tasks and finetuned with task-specific instructions. Our results indicate that LLaMAX improves translation quality while maintaining general capabilities and can serve as a powerful foundation model for downstream multilingual applications.

### References

AI@Meta. 2024. Llama 3 model card.

Duarte M Alves, José Pombal, Nuno M Guerreiro, Pedro H Martins, João Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, et al. 2024. Tower: An open multilingual large language model for translation-related tasks. arXiv preprint arXiv:2402.17733.

Antonios Anastasopoulos, Alessandro Cattelan, ZiYi Dou, Marcello Federico, Christian Federmann, Dmitriy Genzel, Franscisco Guzmán, Junjie Hu, Macduff Hughes, Philipp Koehn, Rosie Lazar, Will Lewis, Graham Neubig, Mengmeng Niu, Alp Öktem, Eric Paquin, Grace Tang, and Sylwia Tur. 2020. TICO-19: the translation initiative for COvid-19. In Proceedings of the 1st Workshop on NLP for COVID-19 (Part 2) at EMNLP 2020, Online. Association for Computational Linguistics.

Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, David Cairuz, Hangyu Lin, Bharat Venkitesh, Madeline Smith, Kelly Marchisio, Sebastian Ruder, et al. 2024. Aya 23: Open weight releases to further multilingual progress. arXiv preprint arXiv:2405.15032.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 632–642, Lisbon, Portugal. Association for Computational Linguistics.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Mauro Cettolo, Christian Girardi, and Marcello Federico. 2012. WIT3: Web inventory of transcribed and translated talks. In Proceedings of the 16th Annual conference of the European Association for Machine Translation, pages 261–268, Trento, Italy. European Association for Machine Translation.

Tyler A. Chang, Catherine Arnett, Zhuowen Tu, and Benjamin K. Bergen. 2023. When is multilinguality a curse? language modeling for 250 high- and lowresource languages.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. Xnli: Evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

OpenCompass Contributors. 2023. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/ opencompass.

Yiming Cui, Ziqing Yang, and Xin Yao. 2024. Efficient and effective text encoding for chinese llama and alpaca.

Konstantin Dobler and Gerard de Melo. 2023. FOCUS: Effective embedding initialization for monolingual specialization of multilingual models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13440–13454, Singapore. Association for Computational Linguistics.

Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, et al. 2021. Beyond english-centric multilingual machine translation. Journal of Machine Learning Research (JMLR).

Manuel Faysse, Patrick Fernandes, Nuno M. Guerreiro, António Loison, Duarte M. Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro H. Martins, Antoni Bigata Casademunt, François Yvon, André F. T. Martins, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Croissantllm: A truly bilingual french-english language model.

Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Hiroki Iida, Masanari Ohi, Kakeru Hattori, Hirai Shota, Sakae Mizuki, Rio Yokota, and Naoaki Okazaki. 2024. Continual pre-training for cross-lingual llm adaptation: Enhancing japanese language capabilities.

Ian J Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. 2013. An empirical investigation of catastrophic forgetting in gradient-based neural networks. arXiv preprint arXiv:1312.6211.

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, PengJen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions of the Association for Computational Linguistics, 10:522–538.

Ping Guo, Yubing Ren, Yue Hu, Yunpeng Li, Jiarui Zhang, Xingsheng Zhang, and Heyan Huang. 2024. Teaching large language models to translate on lowresource languages with textbook prompting. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 15685–

15697, Torino, Italia. ELRA and ICCL.

Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L. Richter, Quentin Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. 2023. Continual pre-training of large language models: How to (re)warm your model?

Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt.

2021a. Aligning ai with shared human values. Proceedings of the International Conference on Learning Representations (ICLR).

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021b. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR).

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021c. Measuring mathematical problem solving with the math dataset. NeurIPS.

Amr Hendy, Mohamed Abdelrehim, Amr Sharaf, Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita, Young Jin Kim, Mohamed Afify, and Hany Hassan Awadalla. 2023. How good are gpt models at machine translation? a comprehensive evaluation.

Kaiyu Huang, Fengran Mo, Hongliang Li, You Li, Yuanchi Zhang, Weijian Yi, Yulong Mao, Jinchen Liu, Yuzhuang Xu, Jinan Xu, Jian-Yun Nie, and Yang Liu. 2024a. A survey on large language models with multilingualism: Recent advances and new frontiers.

Zixian Huang, Wenhao Zhu, Gong Cheng, Lei Li, and Fei Yuan. 2024b. Mindmerger: Efficient boosting llm reasoning in non-english languages. arXiv preprint arXiv:2405.17386.

Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, H’erve J’egou, and Tomas Mikolov. 2016. Fasttext.zip: Compressing text classification models. arXiv preprint arXiv:1612.03651.

Tasnim Kabir and Marine Carpuat. 2021. The UMD submission to the explainable MT quality estimation shared task: Combining explanation models with sequence labeling. In Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems, pages 230–237, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Tom Kocmi and Christian Federmann. 2023. GEMBAMQM: Detecting translation quality error spans with GPT-4. In Proceedings of the Eighth Conference on Machine Translation, pages 768–775, Singapore. Association for Computational Linguistics.

Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, and Orhan Firat. 2024. Madlad-400: A multilingual and document-level large audited dataset. Advances in Neural Information Processing Systems (NeurIPS).

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: a benchmark for question answering research. Transactions of the Association of Computational Linguistics.

Guillaume Lample, Alexis Conneau, Ludovic Denoyer, and Marc’Aurelio Ranzato. 2018. Unsupervised machine translation using monolingual corpora only. In International Conference on Learning Representations.

Zihao Li, Yucheng Shi, Zirui Liu, Fan Yang, Ninghao Liu, and Mengnan Du. 2024. Quantifying multilingual performance of large language models across languages.

Bill Yuchen Lin, Seyeon Lee, Xiaoyang Qiao, and Xiang Ren. 2021a. Common sense beyond english: Evaluating and improving multilingual language models for commonsense reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 1274– 1287. Association for Computational Linguistics.

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona T. Diab, Veselin Stoyanov, and Xian Li. 2021b. Few-shot learning with multilingual language models. CoRR, abs/2112.10668.

Hongyuan Lu, Haoran Yang, Haoyang Huang, Dongdong Zhang, Wai Lam, and Furu Wei. 2023. Chain-of-Dictionary Prompting Elicits Translation in Large Language Models. arXiv e-prints, page arXiv:2305.06575.

Yin Luo, Qingchao Kong, Nan Xu, Jia Cao, Bao Hao, Baoyu Qu, Bo Chen, Chao Zhu, Chenyang Zhao, Donglei Zhang, et al. 2023. Yayi 2: Multilingual open-source large language models. arXiv preprint arXiv:2312.14862.

Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. 2023. Scaling data-constrained language models.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2022. Crosslingual generalization through multitask finetuning.

OpenAI. 2023. Gpt-4 technical report.

Xiao Pan, Mingxuan Wang, Liwei Wu, and Lei Li. 2021. Contrastive learning for many-to-many multilingual neural machine translation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint

Conference on Natural Language Processing (Volume 1: Long Papers), pages 244–258, Online. Association for Computational Linguistics.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vuli´c, and Anna Korhonen. 2020. XCOPA: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2362–2376, Online. Association for Computational Linguistics.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A neural framework for MT evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2685–2702, Online. Association for Computational Linguistics.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlmüller, Andrew Dai, Andrew La, Andrew Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakas¸, B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bartłomiej Bojanowski, Batuhan Özyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, César Ferri Ramírez, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian Voigt, Christopher D. Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Moseguí González, Danielle Perszyk,

Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodola, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A. Chi, Ethan Dyer, Ethan Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Martínez-Plumed, Francesca Happé, Francois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germán Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Wang, Gonzalo JaimovitchLópez, Gregor Betz, Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Shevlin, Hinrich Schütze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac Noble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fernández Fisac, James B. Simon, James Koppel, James Zheng, James Zou, Jan Koco´n, Jana Thompson, Jared Kaplan, Jarema Radom, Jascha Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Joan Waweru, John Burden, John Miller, John U. Balis, Jonathan Berant, Jörg Frohberg, Jos Rozen, Jose Hernandez-Orallo, Joseph Boudeman, Joseph Jones, Joshua B. Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan, Katerina Ignatyeva, Katja Markert, Kaustubh D. Dhole, Kevin Gimpel, Kevin Omondi, Kory Mathewson, Kristen Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao, Li Zhang, Liam Dugan, Lianhui Qin, Lidia ContrerasOchando, Louis-Philippe Morency, Luca Moschella, Lucas Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros Colón, Luke Metz, Lütfi Kerem ¸Senel, Maarten Bosma, Maarten Sap, Maartje ter Hoeve, Maheen Farooqi, Manaal Faruqui, Mantas Mazeika, Marco Baturan, Marco Marelli, Marco Maru, Maria Jose Ramírez Quintana, Marie Tolkiehn, Mario Giulianelli, Martha Lewis, Martin Potthast, Matthew L. Leavitt, Matthias Hagen, Mátyás Schubert, Medina Orduna Baitemirova, Melody Arnaud, Melvin McElrath, Michael A. Yee, Michael Cohen, Michael Gu, Michael Ivanitskiy, Michael Starritt, Michael Strube, Michał Sw˛edrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Mo Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, Mukund Varma T, Nanyun Peng, Nathan Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas Roberts, Nick Doiron, Nikita Nangia, Niklas Deckers, Niklas Muennighoff, Nitish Shirish Keskar, Niveditha S. Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar

Agha, Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter Chang, Peter Eckersley, Phu Mon Htut, Pinyu Hwang, Piotr Miłkowski, Piyush Patil, Pouya Pezeshkpour, Priti Oli, Qiaozhu Mei, Qing Lyu, Qinlang Chen, Rabin Banjade, Rachel Etta Rudolph, Raefer Gabriel, Rahel Habacker, Ramón Risco Delgado, Raphaël Millière, Rhythm Garg, Richard Barnes, Rif A. Saurous, Riku Arakawa, Robbe Raymaekers, Robert Frank, Rohan Sikand, Roman Novak, Roman Sitelew, Ronan LeBras, Rosanne Liu, Rowan Jacobs, Rui Zhang, Ruslan Salakhutdinov, Ryan Chi, Ryan Lee, Ryan Stovall, Ryan Teehan, Rylan Yang, Sahib Singh, Saif M. Mohammad, Sajant Anand, Sam Dillavou, Sam Shleifer, Sam Wiseman, Samuel Gruetter, Samuel R. Bowman, Samuel S. Schoenholz, Sanghyun Han, Sanjeev Kwatra, Sarah A. Rous, Sarik Ghazarian, Sayan Ghosh, Sean Casey, Sebastian Bischoff, Sebastian Gehrmann, Sebastian Schuster, Sepideh Sadeghi, Shadi Hamdan, Sharon Zhou, Shashank Srivastava, Sherry Shi, Shikhar Singh, Shima Asaadi, Shixiang Shane Gu, Shubh Pachchigar, Shubham Toshniwal, Shyam Upadhyay, Shyamolima, Debnath, Siamak Shakeri, Simon Thormeyer, Simone Melzi, Siva Reddy, Sneha Priscilla Makini, Soo-Hwan Lee, Spencer Torene, Sriharsha Hatwar, Stanislas Dehaene, Stefan Divic, Stefano Ermon, Stella Biderman, Stephanie Lin, Stephen Prasad, Steven T. Piantadosi, Stuart M. Shieber, Summer Misherghi, Svetlana Kiritchenko, Swaroop Mishra, Tal Linzen, Tal Schuster, Tao Li, Tao Yu, Tariq Ali, Tatsu Hashimoto, Te-Lin Wu, Théo Desbordes, Theodore Rothschild, Thomas Phan, Tianle Wang, Tiberius Nkinyili, Timo Schick, Timofei Kornev, Timothy Telleen-Lawton, Titus Tunduny, Tobias Gerstenberg, Trenton Chang, Trishala Neeraj, Tushar Khot, Tyler Shultz, Uri Shaham, Vedant Misra, Vera Demberg, Victoria Nyamai, Vikas Raunak, Vinay Ramasesh, Vinay Uday Prabhu, Vishakh Padmakumar, Vivek Srikumar, William Fedus, William Saunders, William Zhang, Wout Vossen, Xiang Ren, Xiaoyu Tong, Xinran Zhao, Xinyi Wu, Xudong Shen, Yadollah Yaghoobzadeh, Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen, Yonatan Belinkov, Yu Hou, Yufang Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang, Zirui Wang, and Ziyi Wu. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford_alpaca.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti,

John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling humancentered machine translation. ArXiv.

Alexey Tikhonov and Max Ryabinin. 2021. It’s all in the heads: Using attention heads as a baseline for cross-lingual transfer in commonsense reasoning.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models.

Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cristian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288.

Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, WeiYin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, et al. 2024. Aya model: An instruction finetuned open-access multilingual language model. arXiv preprint arXiv:2402.07827.

David Vilar, Markus Freitag, Colin Cherry, Jiaming Luo, Viresh Ratnakar, and George Foster. 2023. Prompting PaLM for translation: Assessing strategies and performance. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15406– 15427, Toronto, Canada. Association for Computational Linguistics.

Changhan Wang, Kyunghyun Cho, and Jiatao Gu. 2019. Neural machine translation with byte-level subwords.

Xinyi Wang, Sebastian Ruder, and Graham Neubig. 2022. Expanding pretrained models to thousands more languages via lexicon-based adaptation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 863–877, Dublin, Ireland. Association for Computational Linguistics.

Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, et al. 2023. Polylm: An open source polyglot large language model. arXiv preprint arXiv:2307.06018.

Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122. Association for Computational Linguistics.

Yong Xie, Karan Aggarwal, and Aitzaz Ahmad. 2023. Efficient continual pre-training for building domain specific large language models.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics.

Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat, and Minjoon Seo. 2024. Langbridge: Multilingual reasoning without multilingual supervision.

Fei Yuan, Yinquan Lu, Wenhao Zhu, Lingpeng Kong, Lei Li, Yu Qiao, and Jingjing Xu. 2023. Lego-MT: Learning detachable models for massively multilingual machine translation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 11518–11533, Toronto, Canada. Association for Computational Linguistics.

Fei Yuan, Chang Ma, Shuai Yuan, Qiushi Sun, and Lei Li. 2024a. Ks-lottery: Finding certified lottery tickets for multilingual language models. arXiv preprint arXiv:2402.02801.

Fei Yuan, Shuai Yuan, Zhiyong Wu, and Lei Li. 2024b. How vocabulary sharing facilitates multilingualism in LLaMA? In Findings of the Association for Computational Linguistics ACL 2024, pages 12111–12130, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pretrained transformer language models.

Jun Zhao, Zhihao Zhang, Luhui Gao, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024a. Llama beyond english: An empirical study on language capability transfer.

Yiran Zhao, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, and Lidong Bing. 2024b. How do large language models handle multilingualism?

Wenhao Zhu, Shujian Huang, Fei Yuan, Shuaijie She, Jiajun Chen, and Alexandra Birch. 2024a. Question translation training for better multilingual reasoning. arXiv preprint arXiv:2401.07817.

Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2024b. Multilingual machine translation with large language models: Empirical results and analysis. In Findings of the Association for Computational Linguistics: NAACL 2024.

### Limitations

This work focuses on the discussion of some key technologies, such as the use of vocabulary lists and the determination of data augmentation schemes. However, it does not delve into further processing of the quality of open-source data. We acknowledge a gap in the literature regarding the thorough evaluation of open-source data quality, suggesting an opportunity for future research to improve data preprocessing methods for better model training outcomes.

### Acknowledgments

Authors of this paper would like to thank Zixian Huang, Qiushi Sun, Fangzhi Xu, Hanxu Hu, Chuanyang Jin, Yichao Du, and Zichen Ding for giving many helpful comments on previous versions of this paper. We deeply express our gratitude to Shanghai AI Laboratory. This work is partially supported by the National Key R&D Program of China (NO.2022ZD0160100).

### Outline

- • Section A: The comprehensive details of the training data, including monolingual and parallel data, and the evaluation benchmark (Table 7).
- • Section B: The detailed information of different models, including open-source Large Language Models (Section B.1) and well-trained translation models (Section B.2).
- • Section C: Analysis the correlation between embedding quality of LLaMA2 and fertility using Flores-101 test (Figure 5).
- • Section D: A detailed introduction to the KSLottery method.
- • Section E: Selection about multi-hop translation (Table 9 and Table 10).
- • Section F: The selection of the appropriate format for parallel data during training (Table 11).
- • Section G: The comparison of translation performance across all seven languages between LegoMT and GPT-4 (Figure 6).
- • Section H: Comparison results between LLaMAX2-Alpaca with language-specific enhanced LLMs (Table 13).

• Section I: We present comprehensive instructions utilized for all LLMs (Table 14).

### A Data Information

In this section, we will introduce the sources of our training data (Section A.1), the evaluation benchmarks (Section A.2). For translation tasks, we apply beam search to each model with beam size=4.

#### A.1 Training Dataset

The dataset was compiled from three distinct opensource datasets, with details on supported languages presented in Table 7 and continual pretraining data statistics in Table 7 and Table 8.

MC4 (Xue et al., 2021) is a multilingual variant of the C4 dataset, comprising natural text in 101 languages sourced from the Common Crawl web scrape. It was introduced to support the training of massively multilingual pre-trained text-to-text transformers like mT5.

MADLAD-400 (Kudugunta et al., 2024) is a manually audited, general domain monolingual dataset based on CommonCrawl, encompassing 419 languages and designed for document-level analysis. It is notable for its extensive language coverage and the rigorous auditing process involved in its creation.

Lego-MT (Yuan et al., 2023) is a benchmark for massively multilingual machine translation, featuring a detachable model built upon an efficient training recipe. It includes a comprehensive translation benchmark with data from OPUS, covering 433 languages and 1.3 billion parallel data points.

#### A.2 Evaluation Benchmark

Flores-101 (Goyal et al., 2022) is a benchmark for machine translation evaluation, comprising a multi-way dataset derived from English Wikipedia and produced by professional translators.

Flores-200 (Team et al., 2022) is an extension of Flores-101 dataset and also serves as a benchmark for machine translation. This dataset contains parallel sentences for 200 languages, with each language identified by its ISO 639-3 code ( (e.g. eng)) and an additional code (e.g., "eng_Latn",) that describes the script.

WMT-23 (Kocmi and Federmann, 2023) is also a comprehensive translation evaluation benchmark, proposed in 2023. We incorporate this dataset into

Family ISO Language # Mono. # Para. # Direct. Family ISO Language # Mono. # Para. # Direct.

|Afro-Asiatic|ha om so<br><br>am ar he mt<br><br>|Hausa Oromo Somali<br><br>Amharic<br><br>Arabic Hebrew Maltese|420,964 3,147,704 96<br><br>18,895 191,319 96 697,864 3,804,551 97 269,171 4,031,552 97 716,063 9,940,756 97 300,000 3,928,938 96 671,716 1,518,533 94<br><br>| |Indo-European<br><br>|ne or pa sd ur fa<br><br>ku ps tg<br><br>ast ca es fr gl<br><br>it oc pt ro<br><br>|Nepali<br><br>Odia Punjabi<br><br>Sindhi Urdu Persian Kurdish Pashto Tajik<br><br>Asturian Catalan Spanish<br><br>French Galician<br><br>Italian Occitan<br><br>Portuguese Romanian<br><br>|702,334 8,907,527 97 100,530 812,235 97 513,987 3,737,780 97 472,217 821,996 95 711,354 4,137,619 97 721,307 4,111,536 97 517,239 3,597,863 97 588,340 3,717,480 97 700,237 4,131,709 97<br><br>0 1,535,714 96 724,597 4,145,004 97 706,307 4,258,477 98 787,316 4,290,003 99 726,512 3,131,730 96 846,107 4,233,108 96<br><br>36,379 1,752,951 95 795,818 4,258,604 97 702,002 4,219,414 97|
|---|---|---|---|---|---|---|---|---|
|Austroasiatic|km vi<br><br>|Khmer Vietnamese|687,690 4,044,652 97 760,472 4,112,089 97<br><br>| | | | | |
|Austronesian|jv id<br><br>ms mi<br><br>ceb tl<br><br>|Javanese<br><br>Indonesian Malay Maori<br><br>Cebuano Tagalog<br><br>|505,619 2,799,761 97 707,962 4,243,235 97 711,895 4,121,713 97 180,678 3,437702 97 418,058 2,217,926 91<br><br>0 3,927,576 97| | | | | |
|Dravidian|te kn ml<br><br>ta|Telugu Kannada Malayalam Tamil<br><br>|708,459 4,219,702 97 712,832 3,592,636 97 715,387 4,516,012 97 711,863 4,444,734 97| | | | | |
| | | | | |Japonic<br><br>|ja|Japanese<br><br>|726,455 4,207,728 97|
|Indo-European<br><br>|hy<br><br>lt lv be bg bs cs hr<br><br>mk pl ru<br><br>sk<br>sl sr<br><br><br>uk cy ga<br><br>is sv da no<br><br>af de en lb nl el<br><br>bn as gu<br><br>hi mr|Armenian Lithuanian Latvian Belarusian Bulgarian Bosnian Czech Croatian Macedonian Polish Russian Slovak Slovenian Serbian Ukrainian Welsh Irish Icelandic Swedish Danish Norwegian<br><br>Afrikaans German English<br><br>Luxembourgish Dutch Greek<br><br>Bengali Assamese Gujarati Hindi Marathi<br><br>|712,835 3,677,780 97 718,382 3,946,735 96 700,889 4,011,628 97 708,288 4,169,719 95 711,500 4,131,053 97 300,000 2,953,912 97 711,179 4,135,944 97 300,000 4,106,335 97<br><br>702,035 4,009,787 97<br><br>792,829 4,200,001 98 853,407 4,204,365 97 715,540 4,100,272 98 731,613 4,073,213 97 711,535 4,033,130 97<br><br>714,181 4,070,250 97<br><br>703,507 3,777,953 97 693,460 2,814,912 96 704,159 4,088,886 97 726,893 4,213,939 97 721,543 4,194,587 97 721,715 4,045,571 97<br><br>703,546 4,143,358 98<br><br>881,553 10,273,597 97 846,712 19,548,583 100 574,166 1,035,619 94 769,778 4,199,773 96 707,751 4,081,607 97 707,099 4,560,978 97<br><br>33,825 1,656,861 97<br><br>704,619 3,761,401 97<br><br><br>715,691 4,186,127 97<br><br><br><br><br>702,382 4,295,708 97| |Kartvelian|ka<br><br>|Georgian|703,515 4,182,651 97|
| | | | | |Koreanic|ko<br><br>|Korean<br><br>|711,406 4,234,653 97|
| | | | | |Kra–Dai|lo th|Lao Thai<br><br>|357,758 2,642,799 97 707,719 4,437,476 97<br><br>|
| | | | | |Mongolic|mn<br><br>|Mongolian<br><br>|701,304 3,894,353 97|
| | | | | |Niger–Congo<br><br>|wo ln ns lg<br><br>ny sn<br><br>sw<br><br>umb xh yo zu ig<br><br>kam ff|Wolof Lingala Northern Sotho Luganda Nyanja Shona Swahili Umbundu Xhosa<br><br>Yoruba Zulu Igbo<br><br>Kamba Fulani<br><br>|871 802,521 97 3,325 159,684 96 0 96,288 88<br><br>13,030 216,135 95 226,940 3,104,349 92 386,588 3,140,063 97 700,422 3,775,394 97<br><br>0 54 2 122,720 3,955,426 97<br><br>98,281 3,364,040 96 470,403 2,899,738 97 147,319 3,314,731 96<br><br>0 8 1 26 313,870 97<br><br>|
| | | | | |Nilo-Saharan<br><br>|luo|Dholuo|0 91 6|
| | | | | |Portuguese<br><br>|kea|Kabuverdianu<br><br>|0 0 0|
| | | | | |Sino-Tibetan|zh zhtrad my|Chinese Chinese<br><br>Burmese<br><br>|726,112 14,215,583 96 0 3,747,297 96 579,160 3,887,841 97|
| | | | | |Turkic<br><br>|uz kk ky az<br><br>tr|Uzbek Kazakh Kyrgyz<br><br>Azerbaijani Turkish<br><br>|723,096 2,344,375 95 701,849 3,836,259 97 704,438 3,725,583 97 712,947 8,080,151 97 727,711 4,169,259 97<br><br>|
| | | | | |Uralic<br><br>|et fi<br><br>hu|Estonian Finnish Hungarian<br><br>|706,720 4,056,200 97 719,416 40,76,885 97 731,479 4,154,132 97<br><br>|

- Table 7: The detailed information of the collected monolingual and parallel datasets includes the translation directions for each supported language. Specifically, the “# Para.” represents the count of language-centric sentence pairs, while “# Mono.” denotes the number of individual monolingual sentences.

our evaluation to mitigate the risk of data leakage in LLMs. Based on benchmark, we evaluate the English-centric translation task performance, including de→en, en→cs, en→de, en→he, en→ja, en→ru, en→uk, en→zh, he→en, ja→en, ru→en, uk→en, zh→en.

TICO (Anastasopoulos et al., 2020) dataset represents a joint translation effort targeting COVID19 materials, developed in collaboration with academic, industry stakeholders, and Translators without Borders. It comprises translation memories, a glossary of translated COVID-19 terms, and functions as a benchmark for translation-related evalua-

tions. The all evaluated translation is en→{am, bn, din, fa, fuv, hi, km, ku, ln, ms, ne, om, ps, ru, so, ta, ti_ER, tl, zh, ar, ckb, es_LA, fr, ha, id, kr, lg, mr, my, nus, prs, pt_BR, rw, sw, ti, ti_ET, ur, zu}.

TED (Cettolo et al., 2012) is a massively multilingual dataset derived from TED Talk transcripts, covering 60 languages with parallel arrays of language and text. It is designed for natural language processing tasks and filters out missing or incomplete translations. We also evaluate the Englishcentric translation performance. The translation direction covers all 60 languages, including en↔{af, am, ar, arq, art-x-bork, as, ast, az, be, bg, bi, bn, bo,

###### ISO # Para. # Mono. ISO # Para. # Mono. ISO # Para. # Mono. ISO # Para. # Mono.

af 201,367,199 360,215,552 hi 593,592,809 366,433,792 mn 332,967,182 359,067,648 tg 347,063,556 358,521,344 am 903,470,803 137,815,552 hr 212,708,153 153,600,000 mr 609,131,394 359,619,584 th 597,728,017 362,352,128

- ar 1,054,714,212 366,624,256 hu 232,631,392 374,517,248 ms 234,113,298 364,490,240 tl 244,687,985 0

- as 313,146,729 17,318,400 hy 579,250,350 364,971,520 mt 102,804,684 343,918,592 tr 272,252,613 372,588,032

- ast 70,949,987 0 id 232,953,602 362,476,544 my 1,002,285,410 296,529,920 uk 218,572,425 365,660,672 az 654,492,231 365,028,864 ig 242,306,836 75,427,328 ne 1,237,255,500 359,595,008 umb 3,170 0 be 306,891,318 362,643,456 is 251,875,378 360,529,408 nl 193,189,558 394,126,336 ur 557,337,279 364,213,248 bg 229,686,547 364,288,000 it 195,146,279 433,206,784 no 190,141,837 369,518,080 uz 148,867,813 370,225,152 bn 755,297,957 362,034,688 ja 292,857,869 371,944,960 ns 6,056,515 0 vi 372,555,263 389,361,664 bs 155,671,162 153,600,000 jv 150,347,166 258,876,928 ny 194,642,682 116,193,280 wo 45,422,689 445,952 ca 196,058,689 370,993,664 ka 627,397,650 360,199,680 oc 91,504,042 18,626,048 xh 242,467,614 62,832,640

ceb 135,958,864 214,045,696 kam 477 0 om 13,239,275 9,674,240 yo 282,242,956 50,319,872 cs 218,791,438 364,123,648 kea 0 0 or 289,074,437 51,471,360 zh 878,523,029 371,769,344 cy 247,455,922 360,195,584 kk 299,995,454 359,346,688 pa 1,088,815,314 263,161,344 zhtrad 252,942,548 0 da 201,340,176 369,430,016 km 1,266,785,006 352,097,280 pl 223,440,053 405,928,448 zu 189,932,839 240,846,336 de 456,147,707 451,355,136 kn 1,198,503,370 364,969,984 ps 482,900,652 301,230,080 el 629,383,799 362,368,512 ko 415,419,459 364,239,872 pt 189,507,878 407,458,816 en 523,902,024 433,516,544 ku 494,346,376 264,826,368 ro 224,472,825 359,425,024

- es 193,760,704 361,629,184 ky 284,261,983 360,672,256 ru 213,581,742 436,944,384

- et 223,902,240 361,840,640 lb 58,408,912 293,972,992 sd 107,023,879 241,775,104 fa 505,307,774 369,309,184 lg 12,860,033 6,671,360 sk 232,485,422 366,356,480 ff 16,917,593 13,312 ln 8,942,304 1,702,400 sl 211,807,076 374,585,856 fi 242,982,346 368,340,992 lo 932,379,487 183,172,096 sn 196,567,944 197,933,056 fr 198,627,139 403,105,792 lt 231,673,345 367,811,584 so 255,665,827 357,306,368 ga 190,006,560 355,051,520 luo 4,996 0 sr 217,789,020 364,305,920 gl 145,312,272 371,974,144 lv 261,558,146 358,855,168 sv 190,891,437 372,169,216 gu 1,157,006,948 360,764,928 mi 234,795,047 92,507,136 sw 218,972,852 358,616,064 ha 185,399,766 215,533,568 mk 230,161,774 359,441,920 ta 805,830,274 364,473,856 he 401,537,464 153,600,000 ml 773,141,254 366,278,144 te 1,387,859,988 362,731,008

- Table 8: The detailed information about the tokens used in the continual pre-training stage. The “# Para.” shows the total tokens in the parallel dataset, and “# Mono.” represents the total tokens in the monolingual dataset.

bs, ca, ceb, cnh, cs, da, de, el, eo, es, et, eu, fa, fi, fil, fr, fr-ca, ga, gl, gu, ha, he, hi, hr, ht, hu, hup, hy, id, ig, inh, is, it, ja, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, ltg, lv, mg, mk, ml, mn, mr, ms, mt, my, nb, ne, nl, nn, oc, pa, pl, ps, pt, pt-br, ro, ru, rup, sh, si, sk, sl, so, sq, sr, srp, sv, sw, szl, ta, te, tg, th, tl, tlh, tr, tt, ug, uk, ur, uz, vi, zh, zh-cn, zh-tw}

X-CSQA (Lin et al., 2021a) is a multilingual extension of the Commonsense Question Answering (CSQA) dataset, designed for commonsense reasoning research. It facilitates the evaluation and improvement of multilingual language models in commonsense reasoning tasks.

XStoryCloze (Lin et al., 2021b) is a benchmark dataset that comprises the professionally translated English StoryCloze dataset (Spring 2016 version) into 10 non-English languages. It is designed to evaluate the zero- and few-shot learning capabilities of multilingual language models.

XCOPA (Ponti et al., 2020) is a benchmark dataset that assesses machine learning models’ ability to transfer commonsense reasoning across languages. It is an extension of the English COPA dataset and includes 11 languages from diverse language families and geographical regions.

XWinograd (Muennighoff et al., 2022; Tikhonov and Ryabinin, 2021) s a benchmark dataset that consists of a multilingual collection of Winograd Schemas, designed for the evaluation of crosslingual commonsense reasoning capabilities covering six languages.

XNLI (Conneau et al., 2018) is a crosslingual extension of the SNLI (Bowman et al., 2015)/MultiNLI (Williams et al., 2018), consisting of a subset of English examples translated into 14 different languages. It is used for evaluating textual entailment and classification tasks, where the goal is to determine if one sentence implies, contradicts, or is neutral to another sentence

MGSM (Shi et al., 2023) a dataset of gradeschool math problems, each translated into 10 languages by human annotators. It is derived from the GSM8K (Cobbe et al., 2021) dataset and is designed to support question answering on basic mathematical problems that require multi-step reasoning.

MMLU (Hendrycks et al., 2021a,b) is a benchmark for evaluating language models’ capabilities in language comprehension and reasoning across diverse domains. It consists of about 16,000 multiple-choice questions spanning 57 academic

| |ro 2.25<br><br>el 6.52<br><br>bn 8.62<br><br>Fertility Cosine Similarity Recall@1<br><br>| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

0.6

80

0.5

EmbeddingQuality

60

0.4

Fertility

0.3

40

0.2

20

0.1

0

0.0

esptnlafnssvluocebffbslbjvplidkamigisetlglvsntgazkksdhielmrkapaguknthkm

Languages

- Figure 5: Correlation between embedding quality and fertility. The embedding quality of LLaMA2 is measured by cosine similarity and Recall@1 on Flores-101 test. Fertility refers to the ratio of the length of a sentence after tokenization compared to its length before tokenization. A high fertility may result in a poor quality of embedding.

subjects, designed to measure knowledge acquired during pretraining in zero-shot and few-shot settings.

BBH (Srivastava et al., 2022) is a subset of the BIG-Bench, focusing on 23 challenging tasks that current language models struggle to perform, where they do not outperform the average humanrater. It serves as a rigorous evaluation suite to test the limits of language models’ capabilities.

HellaSwag (Zellers et al., 2019) s a dataset designed to evaluate advanced natural language understanding and common sense reasoning, which introduces more complexity and diversity, challenging AI models to predict the ending of incomplete narratives.

WinoG (Sakaguchi et al., 2021) is a large-scale dataset containing 44k problems inspired by the Winograd Schema Challenge, designed to improve the scale and hardness of coreference resolution tasks. It presents fill-in-the-blank questions with binary options, testing the model’s ability to understand nuanced human language.

NQ (Kwiatkowski et al., 2019) is a dataset for question answering research, containing over 300,000 examples each consisting of a real user query and a corresponding Wikipedia page. It is designed to train and evaluate automatic question answering systems by emulating how people search for information.

HumanEval (Chen et al., 2021) is designed to evaluate the code generation capabilities of large

language models, featuring 164 hand-crafted programming challenges that include function signatures, docstrings, bodies, and unit tests. On average, each problem is accompanied by 7.7 tests to assess functional correctness.

MBPP (Austin et al., 2021) comprises approximately 1,000 crowd-sourced Python programming problems, aimed at entry-level programmers and covering programming fundamentals and standard library functionality. Each problem includes a task description, code solution, and three automated test cases.

GSM8K (Cobbe et al., 2021) consists of 8.5K high-quality, linguistically diverse grade school math word problems created by human problem writers. It is designed to support question answering on basic mathematical problems that require multi-step reasoning.

Math (Hendrycks et al., 2021c) is a collection of 12,500 intricate problems derived from competition mathematics. Every problem within the Math dataset includes a comprehensive solution with step-by-step guidance, which serves as a resource for training models to produce detailed answer justifications and explanations.

### B Model Information

Model details about the baseline models for comparison, including decode-only large language models (LLMs) in Section B.1 as well as translation models in Section B.2 with an encoder-decoder structure.

en-centric ta-centric th-centric zh-centric en→X X→en ta→X X→ta th→X X→th zh→X X→zh

Setting Aug

LLaMA2 ✗ 18.31 23.61 0.99 0.49 4.83 1.15 10.02 7.35 DP1 ✗ 19.06 25.98 3.20 0.91 7.66 3.13 11.32 7.83 DP1+DP2 ✗ 19.46 26.40 4.17 1.76 7.28 3.02 11.65 8.82 DP1+DM ✗ 19.22 25.91 3.51 1.34 7.64 2.83 11.56 7.99 DP1+DP2+DM ✗ 19.36 26.47 4.35 1.82 7.78 3.49 11.44 9.14 DP1+DP′ 2 ✓ 19.47 26.65 4.54 1.83 7.66 3.13 11.89 9.17 DP1+DM′ ✓ 18.59 25.98 3.61 1.36 6.72 2.35 10.81 6.45 DP1+DP′ 2+DM ✓ 19.70 26.71 4.68 1.82 8.21 3.65 12.05 9.28 DP1+DP2+DM′ ✓ 19.17 26.58 4.57 1.95 7.12 3.12 11.52 7.73 DP1+DP′ 2+DM′ ✓ 18.80 26.56 4.78 1.79 7.31 3.18 11.35 7.28 Setting Dictionary

##### en-centric ta-centric th-centric zh-centric en→x x→en ta→x x→ta th→x x→th zh→x x→zh

- DP1+DP′ 2+DM′ MUSE: 1-hop 18.80 26.56 4.78 1.79 7.31 3.18 11.35 7.28

- DP1+DP′ 2+DM′ MUSE: 2-hop 18.70 26.50 4.47 1.83 7.08 3.26 10.74 6.68 DP1+DP′ 2+DM′ PanLex: 1-hop 19.33 26.54 4.40 1.83 7.57 3.31 10.86 8.08

- Table 9: A comprehensive analysis of data augmentation sources reveals that using a dictionary to augment parallel data alone improves translation performance. Each cell in the table refers to the average spBLEU score. “Aug” is a boolean representing whether a dictionary is used for augmentation. Meanwhile, we select a specific data augmentation technique and evaluate various dictionary configurations, including 1-hop and 2-hop, as well as different dictionaries.

1-hop translation 2-hop translation Direction Example Direction Example

en→fr dog → chien

en→fr→de dog → chien → Hund fr→de chien → Hund

Table 10: Case of 1-hop and 2-hop translations.

#### B.1 Large Language Models

LLaMA2 (Touvron et al., 2023b) is a decoderonly language model that predicts the next token based on the input sequence of ordered tokens, with a collection of pre-trained and fine-tuned models ranging from 7 billion to 70 billion parameters. The LLaMA2 7B model serves as our foundational model. Unless otherwise specified, any reference to LLaMA or LLaMA2 is the LLaMA2 7B model. The model leverages a Byte-level Byte Pair Encoding (BBPE; Wang et al., 2019) tokenizer, an efficient subword tokenizer that tokenizes at the byte level, allowing it to handle any language and be robust to noise in the data. The BBPE tokenizer is particularly useful for languages with large vocabularies and many rare words.

LLaMAX2 follows the model architecture of LLaMA2 without vocabulary extension. We utilize 24 A100 80GB GPUs and extended the pre-training on the amassed data for over 60 days. We set per device training batch size to 32, the learning rate to 2e-5, and the epoch number to 1.0.

PolyLM (Wei et al., 2023) is an open-source multilingual Large Language Model (LLM) trained on 640 billion tokens, available in two model sizes: 1.7B and 13B. It boasts proficiency in 15 major non-English languages, employing advanced training techniques to enhance its language processing capabilities.

Yayi2 (Luo et al., 2023) is a multilingual opensource Large Language Model pre-trained from scratch on a corpus containing 2.65 trillion tokens. It is aligned with human values through supervised fine-tuning and reinforce ment learning from human feedback.

TowerInstruct (Alves et al., 2024) is a 7B parameter language model fine-tuned on translationrelated tasks, supporting multiple languages including English, Portuguese, Spanish, French, and others. It is designed for tasks such as machine translation, automatic post-editing, and paraphrase generation. In our paper, we evaluate the instruction-tuned model TowerInstruct-7B-v0.2.

Aya-23 (Aryabumi et al., 2024) is an open weights research release of an instruction finetuned decoder-only model with advanced multilingual capabilities, serving 23 languages. It pairs a performant pre-trained Command family of models with the Aya Collection for robust language processing tasks.

Translation Tasks General Tasks Multilingual Tasks

Setting

ceb→x x→ceb QNLI QQP MRPC XStoryCloze XCOPA XWinograd splited-parallel + mono 3.36 2.74 49.46 36.82 68.38 59.20 56.82 73.72

connected-parallel + mono 4.45 3.68 49.46 36.82 68.38 59.10 56.80 74.07

Setting ceb→ca ceb→de ceb→en ceb→es ceb→fr ceb→it ceb→pt ceb→ru splited-parallel + mono 10.32 8.94 23.19 13.30 15.96 10.01 12.66 8.05

connected-parallel + mono 10.97 11.37 27.06 14.91 18.04 12.03 15.55 10.26

Setting ca→ceb de→ceb en→ceb es→ceb fr→ceb it→ceb pt→ceb ru→ceb splited-parallel + mono 5.90 4.91 7.44 5.14 6.02 5.54 6.12 4.24

connected-parallel + mono 7.62 6.92 9.88 6.41 7.39 6.91 7.62 6.54

- Table 11: Design for the utilization of parallel data, we take ceb-centric data as an example, apply two distict approaches, and discover that treating parallel data as two independent monolingual datasets harms to translation performance.

|[Figure 1]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

4

-2.0 -3.0 -2.6 -5.6 -2.5 -4.5

enzhdenearazceb TargetLanguage

2

- -6.7 -5.8 -3.1 -6.2 -2.8 -4.3
- -9.7 -5.6 -4.8 -8.0 -4.0 -7.0

+4.0 +4.0 +4.7 +2.7 +2.4 +2.9

- -6.3 -2.9 -4.4 -2.0 -2.4 -3.2
- -0.1 +0.5 +0.2 +1.8 -0.5 +0.2

0

2

4

6

8

+0.1 +0.2 +0.4 -1.3 -0.3 -1.2

en zh de ne ar az ceb Source Language

- Figure 6: The spBLEU gap between LLaMAX2 and GPT-4. Positive scores mean the result of LLaMAX2 is better than GPT-4. Empirical evidence demonstrates that while LLaMAX2 trails GPT-4 in high-resource translation scenarios, it outperforms in low-resource translation contexts.

ChineseLLaMA2-Alpaca (Cui et al., 2024) is founded on LLaMA2 and enhanced with an extensive Chinese vocabulary that concentrates on Chinese languages. This is a fine-tuned version of ChineseLLaMA2 using Alpaca (Taori et al., 2023) data.

LLaMA2-SFT (Taori et al., 2023) is a finetuned version of LLaMA2 model, leveraging a set of 52,000 diverse English instructions in Alpaca (Taori et al., 2023) to enhance the instructionfollowing capabilities of the model.

Qwen2-7B-Instruct (Bai et al., 2023) is part of the Qwen2 series, which is a instruction-tuned language models. It demonstrates competitiveness against proprietary models across multilingual benchmarks.

Swallow (Fujii et al., 2024) is a large language model which enhances Japanese capability based on LLaMA2. It achieves this by extending the vocabulary with Japanese characters and conducting continued pre-training on a Japanese corpus, resulting in superior performance compared to other LLMs in both English and Japanese tasks. In our paper, we evaluate the instruction-tuned model Swallow-7B-Instruct-v0.1. B.2 Translation Models

M2M-100 (Fan et al., 2021) encompasses multilingual machine translation models designed to translate between any pair of 100 languages directly, without the need for English as an intermediary. The M2M-100 series includes models of varying sizes, specifically 418M, 1.2B, and 12B parameters. These models are part of a groundbreaking approach in the field of machine translation, aiming to enhance direct translation efficiency across a wide array of languages.

Lego-MT (Yuan et al., 2023) is a novel approach to massively multilingual machine translation, featuring detachable models with individual branches for each language or group of languages. This design supports plug-and-play training and inference, enhancing flexibility and efficiency in language processing tasks.

MADLAD-400 (Kudugunta et al., 2024) is a multilingual machine translation model that leverages the T5 architecture and has been trained on a vast corpus of 250 billion tokens, covering over 450 languages.

Aya-101 (Aryabumi et al., 2024) is an opensource, massively multilingual generative language model that operates on the mT5 (Xue et al., 2021)

LLaMA2-Alpaca ChineseLLaMA2-Alpaca LLaMAX2-Alpaca

LLaMA2-Alpaca ChineseLLaMA2-Alpaca LLaMAX2-Alpaca

X

X

Rzh RX Rzh RX Rzh RX Rzh RX Rzh RX Rzh RX

af 0.20 28.36 31.32 0.10 0.30 79.84 ln 0.30 0.00 66.40 0.00 0.00 0.00 am 1.09 40.12 67.29 21.15 0.00 89.23 lo 1.38 32.71 89.03 0.10 0.00 58.30

- ar 2.17 81.23 72.92 24.70 0.00 99.80 lt 1.09 14.13 50.69 24.31 0.20 96.34

- as 8.40 0.59 84.39 0.30 0.00 76.78 luo 5.83 0.00 87.65 0.00 1.38 0.00

- ast 0.30 0.20 18.77 0.10 0.00 33.20 lv 0.30 15.51 52.67 15.42 0.20 97.73 az 0.20 18.87 39.23 4.25 0.00 96.44 mi 0.49 0.00 59.58 0.00 0.00 0.00 be 0.10 49.11 2.96 2.87 0.00 99.70 mk 0.40 17.19 7.31 21.94 0.00 99.31 bg 2.37 44.66 29.74 30.24 0.30 98.62 ml 8.20 12.15 79.55 7.51 0.49 51.88 bn 3.95 44.96 78.75 17.79 0.10 99.60 mn 1.58 17.49 85.67 1.48 0.00 99.51 bs 0.40 2.17 8.10 1.98 0.10 4.25 mr 0.40 19.86 31.42 1.58 0.00 99.01 ca 0.30 90.12 5.14 79.84 0.00 98.91 ms 0.59 5.93 20.36 3.95 0.00 43.18

ceb 0.20 21.94 6.72 16.01 0.00 95.55 mt 0.20 63.44 29.15 25.00 0.00 97.13 cs 0.20 54.55 24.90 38.14 0.30 94.76 my 1.78 47.33 38.74 29.74 0.00 99.90 cy 0.30 19.66 20.55 44.66 0.00 98.81 ne 0.49 35.77 71.64 3.06 0.00 98.72 da 0.30 49.01 22.73 39.72 0.49 91.80 nl 0.30 65.81 4.55 65.22 0.10 94.76 de 0.79 70.55 10.97 75.69 0.30 96.94 no 0.99 32.21 22.53 28.06 0.20 88.74 el 0.69 21.25 52.67 28.26 0.00 100.00 ns 0.20 0.00 38.74 0.00 0.10 0.00 en 0.00 100.00 0.30 99.70 0.00 100.00 ny 0.59 0.00 60.08 0.00 0.20 0.00

- es 0.10 96.94 4.74 93.08 0.00 99.51 oc 0.10 0.79 20.55 0.30 0.40 59.39

- et 2.27 8.50 75.49 2.96 0.10 96.34 om 0.20 0.00 38.04 0.00 0.20 0.00 fa 0.40 45.95 34.49 57.61 0.00 98.12 or 1.28 37.35 62.65 1.78 0.00 99.80 ff 0.49 0.00 73.81 0.00 0.59 0.00 pa 1.28 49.41 39.62 5.43 0.00 100.00 fi 3.95 55.43 65.22 17.59 0.30 97.13 pl 0.20 64.33 12.55 58.50 0.00 98.42 fr 0.10 94.17 3.46 92.98 0.00 98.72 ps 0.99 20.16 39.03 0.49 0.00 97.83 ga 0.20 19.37 8.70 6.82 0.00 93.08 pt 0.30 84.39 5.34 79.84 0.10 98.42 gl 0.20 0.89 26.19 0.10 0.20 83.99 ro 0.10 19.57 26.98 42.39 0.20 87.15 gu 0.59 36.96 45.65 29.74 0.00 99.60 ru 0.69 79.74 46.64 37.06 0.10 99.01 ha 0.79 0.00 67.98 0.00 0.10 0.00 sd 0.89 7.41 41.70 0.20 0.00 95.16 he 1.68 58.70 65.51 31.03 0.00 100 sk 0.40 20.26 25.40 3.56 0.10 97.23 hi 0.79 50.79 55.83 23.81 0.00 98.91 sl 1.19 37.25 49.60 16.21 0.69 91.90 hr 0.49 41.60 20.95 20.36 0.10 69.66 sn 0.49 0.00 34.58 0.00 0.10 0.00 hu 0.40 64.33 27.47 38.74 0.10 97.13 so 0.30 8.70 58.70 0.20 0.10 57.71 hy 4.74 47.13 79.15 12.15 0.00 99.60 sr 0.59 12.45 17.89 18.87 0.20 48.02 id 0.49 81.92 16.21 60.38 0.00 95.85 sv 0.10 47.33 46.94 25.00 0.10 96.94 ig 0.20 0.00 51.48 0.00 0.10 0.00 sw 0.20 39.23 36.86 22.73 0.00 94.66 is 0.40 35.08 40.02 28.46 0.20 92.98 ta 1.48 24.41 55.24 34.09 0.00 98.62 it 0.49 79.55 3.36 77.57 0.10 98.42 te 1.38 38.93 69.47 28.56 0.00 99.60 ja 48.02 16.70 28.36 70.95 6.62 92.00 tg 1.28 2.77 44.86 7.61 0.20 97.04 jv 0.20 0.00 13.83 0.00 0.00 64.62 th 1.28 58.60 71.25 28.56 0.00 100.00 ka 3.56 31.72 70.06 4.74 0.00 99.80 tl 0.20 66.7 32.91 45.75 0.00 98.91

kam 0.99 0.00 65.51 0.00 1.58 0.00 tr 0.89 37.94 48.02 31.42 0.00 95.65 kea 0.59 0.00 35.47 0.00 0.40 0.00 uk 0.49 71.54 10.38 28.06 0.49 98.62 kk 0.99 45.95 37.06 29.45 0.00 98.32 umb 0.59 0.00 54.94 0.00 0.30 0.00

- km 1.58 29.25 58.89 28.26 0.00 100.00 ur 1.68 19.86 75.49 14.82 0.10 96.54

- kn 3.16 38.24 75.59 14.72 0.00 100.00 uz 0.20 30.24 58.99 2.77 0.10 89.92

- ko 3.85 71.94 75.69 23.52 0.00 98.02 vi 0.10 92.69 13.44 81.13 0.00 99.70 ku 0.10 14.13 31.72 0.00 0.40 75.20 wo 0.30 0.00 56.62 0.00 0.49 0.00 ky 1.19 25.99 48.62 4.35 0.00 99.11 xh 0.20 0.00 40.51 0.00 0.10 0.00 lb 0.10 24.21 30.73 0.40 0.59 89.53 yo 0.10 3.56 57.91 0.40 0.10 15.81 lg 10.57 0.00 79.35 0.00 6.13 0.00 zhtrad 98.12 0.00 98.42 0.00 99.51 0.00

zu 0.20 0.00 45.55 0.00 0.10 0.00

- Table 12: Using langdetect (Joulin et al., 2016), we individually identify the language of the translation output in zh→X (where X represents any of the 101 languages included in Flores-101) for the LLaMA2-Alpaca, ChineseLLaMA2-Alpaca, and LLaMAX2-Alpaca models on the Flores-101 devtest. Rzh refers to the proportion of sentences in the zh→X translation output where the top predicted language is Chinese. RX, on the other hand, denotes the proportion where the top prediction corresponds to the target translated language.

architecture, covering 101 languages and designed to bridge the performance gap in non-dominant languages. It incorporates a 13B parameter base and has undergone instruction-finetuning to achieve high performance across its extensive language range.

### C The correlation between fertility and representation quality.

We conduct experiments on Flores-101. Fertility is defined as the ratio of the Ls to the LT, where Ls is the number of words for space-separated languages and characters for others and LT is the number of tokens after applying LLaMA2 tokenizer. The quality estimation of LLaMA on Flores-101 test.

Cosine similarity focuses on the similarity in the expressions of LLaMA across sentence representation of the same sentence in English and other languages. Recall@1 is often used in the context of information retrieval, which measures the quality of representation. The experimental results, as shown in Figure 5, indicate fertility has a high correlation with the representation quality.

### D Introduction to KS-Lottery.

KS-Lottery is a technique designed to identify a small, highly effective subset of parameters within LLMs for multilingual capability transfer. The core concept of this method involves utilizing the Kolmogorov-Smirnov Test to examine the distribu-

LLaMA2-Alpaca Swallow LLaMAX2-Alpaca

LLaMA2-Alpaca Swallow LLaMAX2-Alpaca

X

X

Rja RX Rja RX Rja RX Rja RX Rja RX Rja RX

af 0.20 35.28 72.23 0.00 0.59 75.69 lo 0.30 37.85 75.89 0.10 0.00 54.55

am 0.20 61.96 77.67 0.10 0.69 90.91 lt 4.74 32.41 70.85 4.55 3.66 94.76 ar 0.69 93.97 64.72 13.93 0.00 99.90 luo 0.49 0.00 71.25 0.00 0.89 0.00 as 3.66 1.38 74.01 0.00 0.10 73.22 lv 1.09 39.92 66.80 5.53 1.68 95.36 ast 0.20 1.48 71.44 0.00 0.20 34.19 mi 0.20 0.00 61.46 0.00 0.20 0.00 az 0.20 26.58 69.57 5.53 0.30 97.43 mk 0.30 17.98 78.46 0.00 0.49 98.81 be 0.40 60.18 72.92 0.00 0.20 99.11 ml 1.28 36.17 74.41 1.68 0.49 70.75 bg 1.09 60.28 77.67 0.30 0.89 98.02 mn 0.59 35.18 75.59 1.48 0.00 99.31 bn 1.78 64.62 75.69 1.78 0.00 99.90 mr 0.59 35.87 76.88 0.00 0.10 99.01 bs 0.69 1.38 73.52 0.00 1.98 3.16 ms 0.10 5.53 61.86 0.20 0.00 39.92 ca 0.40 89.92 65.02 11.07 0.49 98.12 mt 0.40 60.08 68.38 3.16 0.69 94.07

ceb 0.10 33.30 44.57 3.56 0.00 95.06 my 1.68 56.03 78.85 1.48 0.10 99.90 cs 1.19 61.46 72.13 5.24 1.68 93.38 ne 0.20 50.00 70.45 0.00 0.00 99.01 cy 0.20 30.83 66.90 2.47 0.20 98.52 nl 0.40 76.78 61.36 22.33 0.20 92.09 da 0.79 57.51 70.06 4.64 0.59 91.80 no 1.38 44.47 69.57 3.16 0.69 86.66 de 1.28 83.40 57.41 29.25 1.28 94.17 ns 1.58 0.00 62.55 0.00 1.38 0.00 el 1.09 42.00 75.20 7.41 0.00 100.00 ny 0.49 0.00 72.53 0.00 0.79 0.00 en 0.00 100.00 67.29 32.41 0.00 100.00 oc 0.20 1.09 68.97 0.00 0.59 58.10 es 0.40 97.04 57.81 20.26 0.10 99.21 om 0.30 0.00 72.53 0.00 2.57 0.00 et 0.69 14.03 68.48 8.70 4.35 89.13 or 0.69 61.86 79.45 0.00 1.09 98.52 fa 0.30 83.89 75.79 4.35 0.00 98.42 pa 0.40 77.67 72.04 1.78 0.79 98.91 ff 0.69 0.00 73.12 0.00 11.96 0.00 pl 0.79 73.32 71.54 8.40 0.49 98.02 fi 3.36 74.11 66.01 17.39 2.37 96.25 ps 0.20 43.28 75.40 0.00 0.00 98.22 fr 0.49 97.04 52.47 34.29 0.00 99.70 pt 1.09 90.71 63.14 8.20 0.20 98.22 ga 0.20 26.98 64.23 2.96 0.00 94.07 ro 0.30 45.95 68.97 4.25 0.30 89.53 gl 0.10 1.58 63.34 3.56 0.20 83.30 ru 0.30 83.10 71.44 12.45 0.20 99.41 gu 0.30 67.59 77.47 0.99 1.48 96.64 sd 0.89 2.47 74.31 0.00 0.00 92.59 ha 0.59 0.00 70.06 0.00 0.99 0.00 sk 0.49 27.27 65.42 7.81 0.59 94.57 he 1.78 76.19 63.34 16.60 0.00 100.00 sl 0.79 58.79 61.66 3.56 1.38 91.11 hi 0.69 70.75 67.98 7.91 0.00 99.90 sn 0.40 0.00 68.18 0.00 1.58 0.00 hr 0.89 54.55 69.37 1.28 1.19 66.60 so 0.10 7.71 74.31 0.20 0.99 59.19 hu 0.40 69.96 71.44 10.67 0.30 93.87 sr 1.48 15.22 75.49 1.48 1.98 44.07 hy 0.69 77.08 79.55 1.09 0.00 99.90 sv 2.57 49.90 66.01 13.34 1.68 95.16 id 0.20 84.98 70.65 7.61 0.00 97.04 sw 0.20 48.32 67.49 0.99 0.59 94.76 ig 0.10 0.00 74.80 0.00 0.20 0.00 ta 0.30 53.46 74.31 1.98 0.00 99.80 is 0.30 55.34 58.20 19.76 0.20 95.06 te 0.20 73.12 75.79 2.47 0.00 99.80 it 0.59 85.47 55.24 24.11 0.00 97.63 tg 0.69 6.23 74.01 0.00 0.40 97.33 jv 1.38 0.10 66.90 0.00 0.89 67.79 th 0.00 84.39 70.75 12.15 0.00 100.00 ka 1.28 63.14 65.91 16.01 0.00 100.00 tl 0.20 73.62 62.94 6.72 0.10 99.31 kam 0.30 0.00 73.22 0.00 3.56 0.00 tr 0.79 42.39 67.69 11.86 0.40 95.26 kea 0.20 0.00 71.25 0.00 0.99 0.00 uk 0.59 89.53 74.31 3.36 0.49 98.12 kk 0.10 55.93 76.48 0.49 0.10 99.21 umb 0.69 0.00 68.68 0.00 1.38 0.00

- km 0.40 53.66 80.34 0.69 0.00 99.90 ur 1.19 25.49 76.19 2.77 0.30 97.92

- kn 3.06 49.60 78.56 1.09 0.10 99.90 uz 0.40 32.71 74.51 0.20 1.78 86.36

- ko 1.58 94.17 60.57 21.84 0.10 99.51 vi 0.00 95.85 56.42 13.24 0.10 99.70 ku 0.20 28.06 60.28 0.49 2.77 72.73 wo 1.09 0.00 73.32 0.00 2.96 0.00 ky 0.40 40.71 75.79 0.00 0.10 99.41 xh 0.20 0.00 70.55 0.00 0.59 0.00 lb 0.69 31.23 66.11 0.00 2.27 87.75 yo 0.10 3.95 67.00 0.00 0.10 13.93 lg 1.38 0.00 74.11 0.00 12.65 0.00 zh 23.22 70.16 37.15 35.67 5.93 93.08 ln 0.30 0.00 71.84 0.00 0.79 0.00 zhtrad 32.41 0.00 43.87 0.00 7.31 0.00

zu 0.10 0.00 67.39 0.00 1.38 0.00

- Table 13: We utilize langdetect to identify the translation outputs from ja→X of LLaMA2-Alpaca, Swallow and LLaMAX2-Alpaca models on Flores-101 benchmark. Rja represents the ratio of sentence in the translation predicted result where the top predicted language is Japanese. Conversely, RX refers to the proportion where the top predicted language aligns with the target translated language.

tion shift of parameters before and after fine-tuning. This approach helps in pinpointing the “winning tickets” or the most impactful parameters that contribute significantly to the model’s performance in multilingual tasks.

### E 1-hop translation in data augmentation is enough.

Given a parallel dataset subset (DP) from DparaA that contains translations in all directions for 6 languages (en,fr,es,zh,ta,th) and a monolingual sub-

set (DM) from DmonoA for the same 6 languages. We then perform non-repetitive sampling 12,500

sentence pairs from DP in each direction to generate two subsets of parallel corpus data DP1 and DP2, respectively. Consequently, we preserve DP1 and evaluate the effect of augmentation on parallel data DP2 or monolingual data DM, resulting in two new dataset, DP′ 2 and DM′ , post-augmentation. To assess both the in-domain and out-of-domain capabilities of the model, we perform inference on it using 10 languages (en, fr, es, pt, de, zh, ta, th, is,

###### Model Templates

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: Translate the following sentences from English to Chinese Simpl ### Input: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added. ### Response:他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”

LLaMAXAlpaca

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: Translate the following sentences from English to Chinese Simpl ### Input: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added. ### Response:他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”

LLaMA Series Models

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: Translate the following sentences from English to Chinese Simpl ### Input: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added. ### Response:他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”

yayi2

"We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added. Translate this sentence English to Chinese Simpl. 他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”

polylm

<|im_start|>user Translate the following text from English into Chinese. English: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added. Chinese:<|im_end|> <|im_start|>assistant 他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”

TowerInstruct

aya23 <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following sentences from English to Chinese: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>他补充道：“我们 现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”<|END_OF_TURN_TOKEN|>

system You are a helpful assistant. user Translate the following sentences from English to Chinese Simpl: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added. assistant 他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”

Qwen2 instruct

[INST] «SYS» You are a helpful assistant. 你是一个乐于助人的助手。 «/SYS» Translate the following sentences from English to Chinese Simpl: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added. [/INS T] 他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”

ChineseAlpaca-2

[INST] «SYS» あなたはで秀な日本人のアシスタントです。 «/SYS» Translate the following sentences from Japanese to Chinese Simpl: 「我々がっている生後4か月のマウスはかつて糖尿病でしたが在は糖尿病ではない、」 と彼は付け加えました。 [/INST] 「他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”」

Swallow

Madlad ’<2zh> "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.’ 他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它

们曾经得过该病。”

- Table 14: Examples of instruction templates utilized for all evaluated LLMs, with the translation result, 他补充 道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”, using the reference instead of the model’s output.

zu), utilizing the Flores-101.

We use two different multilingual dictionaries MUSE provided by Lample et al. (2018) 3, and PanLex (Wang et al., 2022). In the context of a multilingual dictionary, we can use “1-hop” and “2-hop” to characterize the translation relationship among different languages, an example shown in Table 9.

We use the MUSE dictionary to perform data augmentation on both parallel DP2 and monolingual DM data, utilizing 1-hop and 2-hop translations. As shown in Table 9, using different hop translation for augmentation does not significantly impact the final translation performance. Multihop translation sometimes can even result in poorer performance.

3https://github.com/facebookresearch/MUSE.

### F Design of parallel format

The Usage of Parallel Data. Parallel data can be utilized in two distinct ways: split-parallel or connected-parallel. Split-Parallel: Consider the source language data and target language data involved in parallel data as two distinct monolingual datasets, which are randomly shuffled throughout the entire training set. Connected-Parallel: In the training process, we treat each pair of source and target language sentences from the parallel dataset as a single data point by concatenating them.

Based on different forms of parallel data, supervised fine-tuning (SFT) is conducted separately on ceb-centric using both parallel and monolingual datasets. As indicated in Table 11, we observed that the form of parallel data primarily impacts translation performance, with no significant difference in

|| | |
|---|---|
| | |
| | |
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

- Figure 7: Significant improvements in languagespecific-centric translation are observed with LLaMAX2-Alpaca compared to LLaMA2-7B-Alpaca, ChineseLLaMA2-7B-Alpaca, and Swallow, as demonstrated in the translation performance analysis on all translation directions in the Flores-101 dataset.

general tasks and cross-lingual general tasks; however, the disparity in translation is pronounced. We specifically highlighted some high-resource translation directions and found that such gaps are quite significant.

### G Comparison Results Between Our Model and GPT-4

In Figure 6, we compare the performance gap between our model and GPT-4. Considering the API cost of evaluating GPT-4, we only evaluate the mutual translation performance among seven languages (en, zh, de, ne, ar, az, ceb). Experiment results show that while our model lags behind in high-resource translation directions, it achieves onpar or even superior performance in low-resource translation.

### H Comparsion between LLaMAX2-Alpaca and language-specific LLMs.

The comparison between LLaMAX2-Alpaca, ChineseLLaMA2-Alpaca, and Swallow (a Japanese-specific LLM) explores the difference between the traditional pipeline for enhancing specific language capabilities based on existing pre-trained models and our proposed recipe. As shown in Figure 7, we evaluate language-specific LLMs to translate from the enhanced language to any of the 101 languages on Flores-101 and find that their performance is not significantly different from the original LLaMA2 model, but there exists a notable performance gap compared to LLaMAX2-Alpaca. As we described in Sec-

tion 2.1, excessively adding new language-specific tokens can shift the focus of training the LLM.

In addition, we conduct a deeper analysis of translation output to identify the factors contributing to the limited improvement in translation performance. The experimental results in Table 12 indicate that the language-specific LLM obtained through the traditional pipeline tends to output specific languages, while LLaMAX2 can accurately produce the answer with the corresponding language.

We perform further comparisons between LLaMAX2-Alpaca and Japanese-specific LLMsSwallow. After using LLaMAX2-Alpaca and Swallow to generate translations from Japanese (ja) to any language in Flores-101, we apply langdetect to determine the language of each translation result and calculate the proportion of Japanese and target translated language respectively. The experimental result, as shown in Table 13, indicates that the Japanese-specific LLM tends to output Japanese, whereas LLaMAX2-Alpaca performs more accurately in producing the target language.

### I Prompt Templates

We offer a comprehensive collection of prompt instruction templates, as illustrated in Table 14, which are utilized for all evaluated LLMs. These templates are meticulously designed based on existing LLMs, playing a crucial role in obtaining accurate model results and ensuring fairness in comparisons. Our goal in providing these templates is to promote transparency and make it easier to reproduce our findings.

