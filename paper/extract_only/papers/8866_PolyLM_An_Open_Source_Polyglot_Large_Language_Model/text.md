# arXiv:2307.06018v1[cs.CL]12Jul2023

## POLYLM: AN OPEN SOURCE POLYGLOT LARGE LANGUAGE MODEL

Xiangpeng Wei∗, Haoran Wei∗, Huan Lin∗, Tianhao Li∗, Pei Zhang∗, Xingzhang Ren∗, Mei Li∗ Yu Wan∗, Zhiwei Cao†, Binbin Xie†, Tianxiang Hu†, Shangjie Li†, Binyuan Hui, Bowen Yu Dayiheng Liu‡, Baosong Yang‡, Fei Huang, Jun Xie

DAMO Academy, Alibaba Group

ABSTRACT

Large language models (LLMs) demonstrate remarkable ability to comprehend, reason, and generate following nature language instructions. However, the development of LLMs has been primarily focused on high-resource languages, such as English, thereby limiting their applicability and research in other languages. Consequently, we present POLYLM, a multilingual LLM trained on 640 billion (B) tokens, avaliable in two model sizes: 1.7B and 13B. To enhance its multilingual capabilities, we 1) integrate bilingual data into training data; and 2) adopt a curriculum learning strategy that increases the proportion of non-English data from 30% in the first stage to 60% in the final stage during pre-training. Further, we propose a multilingual self-instruct method which automatically generates 132.7K diverse multilingual instructions for model fine-tuning. To assess the model’s performance, we collect several existing multilingual tasks, including multilingual understanding, question answering, generation, and translation. Extensive experiments show that POLYLM surpasses other open-source models such as LLaMA and BLOOM on multilingual tasks while maintaining comparable performance in English. Our models, alone with the instruction data and multilingual benchmark, are available at: https://modelscope.cn/models/damo/nlp_ polylm_13b_text_generation.

1 INTRODUCTION

Large language models (LLMs) are trained on vast amounts of data in a self-supervised fashion, which has shown promising performance in a variety of zero-shot and few-shot tasks (Brown et al., 2020; Chowdhery et al., 2022). Fine-tuning these models on a diverse set of tasks allows them to handle unseen tasks following natural language instructions (Ouyang et al., 2022; Longpre et al., 2023; Taori et al., 2023; Anand et al., 2023). These properties have attracted significant attention from the Artificial Intelligence community and offering a potential path towards artificial general intelligence. Unfortunately, most LLMs are developed for English, such as LLaMA (Touvron et al., 2023), BLOOM (Scao et al., 2022), Chinchilla (Hoffmann et al., 2022), OPT (Zhang et al., 2022). A main reason stems from recent findings that model performance is closely related to the scale of the training dataset (Kaplan et al., 2020; Rae et al., 2021; Biderman et al., 2023; Touvron et al., 2023), leading to predominant focus on resource-rich languages, particularly English.

The relatively high concentration of studies on English limits the research and usage of LLMs in other languages. For instance, Thai and Indonesian have over 300 million (M) speakers, yet the size of these two languages in common crawl-based dataset such as mC4 (Xue et al., 2020) is only 80 billion (B) tokens, comprising a mere 3% of the English data. Due to the insufficient high-quality internet data, LLM capabilities on low-resource languages fail to be easily improved through expanding their data size like English (Kaplan et al., 2020; Rae et al., 2021; Biderman et al., 2023). As a result, existing open-source LLMs such as XGLM (Lin et al., 2022), BLOOM (Scao

∗Major contributors. †Contribution during internship at Alibaba DAMO Academy. ‡Corresponding authors: {liudayiheng.ldyh, yangbaosong.ybs}@alibaba-inc.com

et al., 2022), and LLaMA (Touvron et al., 2023) perform relatively poor on these languages, some of which are entirely overlooked. It is crucial to explore multilingual LLMs to bridge this gap and achieve academic and social significance.

Our goal is to enhance the exploration and utilization of LLMs for non-native English speakers. In this work, we fill three significant gaps in this field: 1) the absence of an open-source multilingual LLM; 2) the inadequate availability of multilingual instruction data; and 3) the lack of a unified evaluation benchmark for multilingual settings.

Concretely, we first develop an open-source multilingual LLM from scratch, called Polyglot Large Language Model (POLYLM, Section 3). Contrary to existing open-source multilingual LLMs that lack 13B model, we release POLYLM-13B and POLYLM-1.7B to facilitate its usage. To construct POLYLM, we leverage a massive dataset of 640B tokens, culled from publicly available sources such as Wikipedia, mC4 (Xue et al., 2020), CC-100 (Conneau et al., 2019). This dataset contains over 30% of non-English languages, specifically covering 18 of the most commonly spoken languages.1 To alleviate the problem of insufficient data for low-resource languages, we propose a curriculum learning strategy. The training schedule increases the amount of data available for training in English during the initial phases, then ramping up the ratio of high-quality, low-resource languages as training progresses. We expect the method to enable the transfer of general knowledge from English to other languages, leading to significant improvements in overall performance.

In light of the supervised fine-tuning (SFT) stage, we construct a multilingual instruction dataset termed MULTIALPACA with 132,701 samples (Section 4). At present, there is a dearth of highquality open-source multilingual SFT datasets. On the one hand, extant multilingual SFT datasets, e.g. xP3-MT (Muennighoff et al., 2022), are acquired via machine translation, which potentially yields a style of translationese, a lack of cultural nuances, as well as translation errors. On the other hands, manually annotating instructions is a laborious and costly process that does not lend itself well to the incorporation of creative flourishes. Drawing inspiration from recent advances in self-instruct (Wang et al., 2022; Taori et al., 2023), we devise a multilingual self-instruct method to automatically generate instruction data. Utilizing 175 English seeds as a starting point, our method leverage multilingual seed translation, instruction generation, and filtering mechanisms to deliver high quality multilingual instruction data.

In order to assess the multilingual capabilities of LLM, we curate a benchmark derived from existing multilingual tasks (Section 5.1), including QA (Clark et al., 2020), understanding (Conneau et al., 2019; Yang et al., 2019; Tikhonov & Ryabinin, 2021; Ponti et al., 2020), generation (Chen et al., 2021), and cross-lingual machine translation (Barrault et al., 2020). The benchmark is constructed with meticulously prompting and finally covers 10 tasks across 15 languages. Extensive experiments (Section 6) demonstrate that our pretrained model outperforms open-source models of comparable model size (e.g. BLOOM, LLaMA, etc.) in non-English languages. Through in-depth analyses, we identify finding that the proposed curriculum training strategy boosts the multilingual performance while maintain the English proficiency. In addition, the use of multilingual instruction data markedly enhances the ability of POLYLM to tackle multilingual zero-shot tasks.

- 2 PRELIMINARY

In this section, we begin with a review of the background on language modeling. We then examine previous research on knowledge transferring, and instruction learning of pre-trained LLMs, with a focus on their relevance to POLYLM. Finally, we outline our rationale for training POLYLM.

Language Modeling refers to the process of estimating the probability of a sequence of tokens, i.e. p(x) = p(x1,x2,...,xT) = Tt=1 p(xt|x<t). This is also commonly referred to as autoregressive sequence modeling, as it involves predicting the future token at each time-step based on the preceding context. The initial language models were predominantly n-gram models that evaluate the likelihood of a sequence of tokens based on the frequency of its occurrence in a training corpus. Over the last two decades, neural networks have proven to be effective in the task of language modeling, including feed-forward models (Mikolov et al., 2010) and recurrent neural networks (Bengio et al.,

1According to https://www.ethnologue.com/insights/most-spoken-language/. Some languages with interchangeable and more widely used official languages are not given priority, such as Hindi, Wu Chinese, and Cantonese.

2000). More recently, Transformer (Vaswani et al., 2017), a self-attention based neural network, has shown unparalleled language model performance (Devlin et al., 2019; Radford et al., 2018), and become the de facto backbone of LLMs emerged in the past three years, such as GPT3 (Brown et al., 2020), Gopher (Rae et al., 2021), PaLM (Anil et al., 2023), BLOOM (Scao et al., 2022), Chinchilla (Hoffmann et al., 2022), GLM (Zeng et al., 2022) and LLaMA (Touvron et al., 2023).

Transfer Learning is a rapidly evolving field of research that has garnered significant interest in recent years. In this scenario, models are initially trained on extensive unlabeled data, and then their acquired knowledge is applied to various downstream tasks through fine-tuning. Some of the most prominent works in this area include the ELMo (Peters et al., 2018), BERT (Devlin et al., 2019) and GPT (Radford et al., 2018) have demonstrated remarkable success. These developments subsequently prompt work (Raffel et al., 2020; Radford et al., 2019; Xue et al., 2020) on better results by adopting larger scale data and parameters to further improve model performance. Although pretraing-then-finetuning is still effective in achieving high performance with limited labeled data, recent advancements has shown that language models with extremely large scale parameters can perform tasks without further optimization. The most exemplary model is GPT3 (Brown et al., 2020), which utilizes a contextualized approach by incorporating multiple input-output demonstrations and presenting them alongside the query. This effectively stimulates the model to generate accurate predictions, showcasing encouraging outcomes in zero/few-shot situations.

Instruction Learning aims to bring together various natural language processing tasks by framing them as question-answering exercises that operate over a given context. This approach enhances the value of LLMs by leveraging their existing knowledge. With the success of language models, there has been a growing interest in exploring their potential to comprehend and execute instructions. Several advanced researches (Ouyang et al., 2022; Wei et al., 2022; Peng et al., 2023; Ye et al., 2023; Zhou et al., 2023) have demonstrated a remarkable ability to generalize to new zero-shot tasks. However, they rely heavily on human-generated instruction data, which is frequently constrained in terms of quantity, diversity, and creativity, which is very time-consuming and labor-intensive. Wang et al. (2022) make an effort to construct a self-Instruct framework for improving the instructionfollowing capabilities of LLMs. Similarly, Xu et al. (2023) propose an evol-instruct framework to automatically rewrite simple human-written instructions step by step into more complex ones, to further improve instruction-followed LLMs.

In this paper, we propose POLYLM to address the following blanks and limitations in current LLM research, offering a comprehensive and innovative solution to advance this field.

- • We provide a 13B scale model that is proficient in the major non-English languages spoken worldwide, such as Spanish, Russian, Arabic, Japanese, Korean, Thai, Indonesian, and Chinese etc. It is a perfect complement to the existing open-source models, including: (1) LLaMA, English is predominant among the whole dataset. (2) BLOOM, lack of 13B version and fail to address languages spoken by significant populations, such as Japanese, Korean and Thai. (3) XGLM (Lin et al., 2022), the maximum version is 7B. (4) mGPT (Shliazhko et al., 2022), only 1.3B version is available.
- • We suggest an advanced curriculum learning approach that facilitates the transfer of commonsense knowledge, acquired mainly in English, to diverse non-English languages and specific NLP downstream tasks such as machine translation.
- • We propose MULTIALPACA to complement ALPACA (Taori et al., 2023) and CHINESEALPACA (Cui et al., 2023), making LLMs better follow multilingual instructions, particularly those coming from non-native English speakers.

- 3 POLYLM: A POLYGLOT LARGE LANGUAGE MODEL

In this section, we present the design of POLYLM, which includes a detailed description of its training dataset (Section 3.1), architecture (Section 3.2), and training process (Section 3.3).

3.1 DATASET

The composition of the pre-training dataset used for POLYLM is shown in Table 1. Our pre-training dataset contains 640B tokens in total, of which English data accounts for 68%. To develop POLYLM

Source Fraction Tokens Type

mC4 49.95% 321.7B Web-text (Multilingual) CC-100 32.31% 208.1B Web-text (Multilingual) The Pile16.41% 105.7B Web-text & books (English) GitHub 1.17% 7.5B Code OPUS 0.16% 1.0B Parallel Multilingual Data

Sum - 638B Table 1: The composition of the POLYLM pre-training dataset.

#### Language Tokens (B) Percentage (%) Language Tokens (B) Percentage (%)

En 424.96 67.56 Vi 4.13 0.66 Zh 139.29 22.14 Id 3.91 0.62 Ru 7.61 1.21 Pl 3.84 0.61 Es 5.62 0.89 Nl 3.52 0.56 De 5.56 0.88 Ar 3.48 0.55 Fr 5.10 0.81 Tr 3.42 0.54 It 4.31 0.69 Th 2.89 0.46 Pt 4.27 0.68 He 2.10 0.33 Ja 4.19 0.67 Ko 0.84 0.13

Table 2: Language distribution of the training data (excluding code and multilingual parallel data).

with multilingual capabilities, the pre-training dataset has about 32% non-English multilingual data, which is a higher percentage of non-English data than most previous open-sourced large language models (Biderman et al., 2023; Zhang et al., 2022; Touvron et al., 2023; Penedo et al., 2023). To be concrete, the English data contains documents with 425B tokens from multiple sources, such as The Pile (Gao et al., 2020), mC4 (Xue et al., 2020), and Wikipedia. While the 204B multilingual data tokens come from CC-100 (Conneau et al., 2019), mC4 (Xue et al., 2020), Wikipedia. The multilingual data mainly covers the following languages: zh, ar, es, fr, de, it, nl, ru, id, pl, pt, ja, th, tr, he, ko, vi, with the distribution given in Table 2. To enable the model ability of code understanding and generation, we also incorporate code data of 7.5B tokens from GitHub with permissioned licenses into our pre-training dataset. In order to further improve the cross-lingual and multilingual ability of the POLYLM, similar to PaLM2 (Anil et al., 2023), we employ parallel multilingual data of 1B tokens into our pre-training dataset.

To build the pre-training dataset, we also develop a comprehensive data pre-processing pipeline that implements multiple techniques for data cleaning and filtering. The pipeline consists of the following stages:

- 1) Language identification. We classify documents according to their primary languages and remove those with low confidence in classification, leveraging inexpensive n-gram models (e.g., fastText (Joulin et al., 2016)).
- 2) Rule-based filtering. Following Rae et al. (2021); Scao et al. (2022), we eliminate irrelevant or low-quality content using various rules and heuristics, including repetition removal (the document with the excessive line, paragraph, or n-gram repetitions is removed), document-wise filtering (removing outlier documents by overall length, symbol-to-word ratio, the ratio of ellipsis, invisible characters, numbers, and dates, etc.), and line-wise corrections (such as URL filtering, long words removal, and whitespace standardization).
- 3) ML-based quality filtering. We further filter low-quality multilingual documents using several small n-gram-based language models (e.g., KenLM (Heafield, 2011)) for different languages trained on their gold-standard corpora. In addition, similar to Raffel et al. (2020); Smith et al. (2022), we also train a 2-gram fastText (Joulin et al., 2016) classifier to filter the low-quality English documents. This classifier uses Wikipedia, and Books from The Pile (Gao et al., 2020) as the positive samples

6

CompressionRate

4

- 2

0

GPT-2 GPT-4 LLaMA BLOOM PolyLM

| |
|---|
| |

Th Ru Ko He Vi Ar Zh Ja Tr Pl Id De Nl Pt It Es Code Fr En

Figure 1: The compression rate of different tokenizers. We take XLM-R (Conneau et al., 2019) tokenizer as the baseline, and set the compression rate of XLM-R tokenizer to 1.

Hyperparameter (↓) POLYLM-1.7B POLYLM-13B Architecture hyperparameters

Number of parameters 1,722M 13,003M Precision bfloat16 Number of layers 24 40 Hidden dimension 2048 5120 Attention heads 16 40 Vocab size 256,000 Sequence length 2048 Activation GELU Position embedding Absolute

Pretraining hyperparameters

Global Batch Size 512 2048 Learning rate peak 1 × 10−4 6 × 10−5 Total training tokens 638B Gradient clipping 1.0 Weight decay 0.1

Multilingul Self-instruction finetuning hyperparameters

Global Batch Size 32 64 Sequence strategy The length is 2048 with packing Learning rate 1e-5 Total training tokens 16M tokens

Table 3: POLYLM Architecture and Training Hyperparameters.

and CommonCrawl web documents as the negative samples. To sum up, about 28.3% data are filtered with Rule-based filtering and ML-based quality filtering.

- 4) Deduplication. In line with Raffel et al. (2020), we remove similar documents to reduce data redundancy with MinHashLSH-based fuzzy deduplication technology, where 23.1% English documents and 18.6% non-English documents are removed.

Based on the POLYLM multilingual pre-training dataset, we derived a vocabulary with 256K token entries using Byte-Pair Encoding (BPE) (Sennrich et al., 2015) with the implementation from SentencePiece (Kudo & Richardson, 2018). To enhance the mathematical capabilities of our model, we follow Touvron et al. (2023) to split all numbers into individual digits. The unknown characters are fallback to byte encoding of UTF-8 to guarantee the coverage of rare words (e.g., emoji, and special symbols). For tokenizer training, we sample multilingual documents with a similar distribution as Conneau et al. (2019) used to increase the number of vocabulary tokens associated with low-resource languages and alleviate the bias towards high-resource languages. We compare the compression rate on different language corpora of different tokenizers. We use XLM-R (Conneau et al., 2019) tokenizer, which supports 100 languages, as the baseline (the compression rate of XLM-R tokenizer is set to 1). As shown in Figure 1, POLYLM has achieved significantly better compression rates in most covered languages, while maintaining the compression rate in English as

| |[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]|[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | |[Figure 20]| | | | | | | | | | |
| | | | | | | | | | | | | |

[Figure 21]

[Figure 22]

12

8

4

0

0 1k 2k 3k 4k 5k 6k 7k 8k 9k 10k

(a) Loss value

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

120

[Figure 37]

[Figure 38]

80

[Figure 39]

[Figure 40]

40

[Figure 41]

[Figure 42]

0

[Figure 43]

[Figure 44]

0 1k 2k 3k 4k 5k 6k 7k 8k 9k 10k

(b) Gradient norm

|[Figure 45]|[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>|[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>spike occurs<br><br>final choice| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| |[Figure 69]| | | | | | | | | | | |
| | | | | | | | | | | | | |

[Figure 70]

[Figure 71]

1e-4

8e-5

6e-5

4e-5

2e-5

0

0 1k 2k 3k 4k 5k 6k 7k 8k 9k 10k

(c) Learning rate

Figure 2: Training curves over iterations for the 13B model with learning rate as 1 × 10−4.

|[Figure 72]|[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

[Figure 97]

|[Figure 98]|[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]|[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]| | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | |[Figure 115]<br><br>[Figure 116]| | | | | | | |
| | | | | | | | | | |

[Figure 117]

[Figure 118]

| |[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]|[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

[Figure 147]

2.38

1.96

- 12

2.36

1.92

8

1.88

2.34

1.84

4

2.32

1.8

0

2.3

1.76

10k 20k 30k 40k 50k 60k

0 2k 4k 6k 8k 10k 12k 14k

0 4k 8k 12k 16k 20k 24k

(a) Loss with bfloat16 precision.

(b) Loss with mixed-precision.

(c) Loss on curriculum learning.

Figure 3: Training curves over iterations for the 13B model with learning rate as 6 × 10−5.

BLOOM (Scao et al., 2022), LLaMA (Touvron et al., 2023), GPT-2 (Radford et al., 2019), and GPT-

- 4 (OpenAI, 2023). Note that some open source models that are not friendly to language extensions, for example, LLaMA (Touvron et al., 2023) only contain a 32K size vocabulary mostly composed of English tokens, which is not friendly to non-Latin languages. When improving a certain non-Latin language ability, the vocabulary needs to be expanded like Chinese-LLaMA (Cui et al., 2023). On the contrary, POLYLM allows researchers to improve the model’s ability in a covered language by simply continuing monolingual pre-training without expanding the vocabulary.

- 3.2 ARCHITECTURE

It has become apparent that the computational cost of exploring different architectural designs for LLMs is prohibitive. Therefore, we present the distinctive design options of POLYLM2 in this section.

Following some endeavours on large language models, we develop a decoder-only autoregressive Transformer architecture detailed in Radford et al. (2019). To stabilize the training, we adopt PreLN (Xiong et al., 2020), i.e. y = x + LayerNorm(f(x)) (where f(·) indicates the layer function) for layer normalization, and apply the Xavier normal initialization (Glorot & Bengio, 2010) with bias terms are initialized to zero. To improve FFNs in Transformer, we replace ReLU with GeLU activation (Hendrycks & Gimpel, 2016).

In this paper we present two Transformer language models with 1.7 billion and 13 billion parameters, respectively. The architectural details are displayed in Table 3.

- 3.3 TRAINING

We train all models with a 2048 token context window, using the Adam (β1 = 0.9, β2 = 0.95) optimizer. We warm-up the learning rate from 1e−7 to the maximum learning rate over the first 2000 steps, and then decay it to 10% of the maximal learning rate using a cosine schedule. We use a weight decay of 0.1 and gradient clipping of 1.0.

2Recent research indicates that Rotary Position Encoding (RoPE) (Su et al., 2021) yields superior performance. Accordingly, we will switch to the latest Megatron-LM branch and promptly release 13B and 1.7B versions featuring RoPE.

100

Large-scale Pretraining

High-quality Curriculum Learning

Proportion(%)

10

- 0.1
- 1

En Zh Ru Es De Fr It Pt Ja Ko Vi Id Ar Th Nl Pl He Tr

- Figure 4: The proportion of multilingual data in curriculum learning significantly exceed that in the pretraining phrase.

mC4, 2.28%

CC-100, 35.88%

GitHub, 7.76%

OPUS, 4.76%

Books, 4.65%

Wikipedia, 17.58%

Arxiv, 14.79%

StackExchange, 12.31%

High-quality Curriculum Learning

mC4, 49.95%

CC-100, 32.31%

GitHub, 1.17%

OPUS, 0.16%

The Pile, 16.41%

Large-scale Pretraining

- Figure 5: The proportion of high-quality and multilingual source is raised in the curriculum learning dataset.

POLYLM was trained using Megatron-LM 3 on a cluster of 32 A100 GPU (8×80G) servers. We apply tensor model parallelism within a single node, setting tensor-model-parallel-size as 8. When training a 13B-parameter model, our code processes around 1170 tokens/sec/GPU, thus training over our dataset containing 640B tokens takes approximately 29 days. However, we faced numerous unforeseen spikes and deviations in losses, which prolonged the entire training process to a duration of two months. There are several possible conditions that result in training collapses, and our unique choices to enhance training stability.

Lower Maximal Learning Rate. Learning rate is an important hyperparameter in neural network models that controls the magnitude of parameter updates. In our first few attempts, we drew inspiration from previous research which indicated that smaller models tend to benefit from higher learning rates. As such, we opted to set the learning rate to 1×10−4. Without exception, all attempts to train POLYLM-13B have resulted in loss spikes with this choice in early stage, which tend to occur more frequently as the training progresses, as illustrated in Figure 2a. We have noticed that the gradient norm shows significant fluctuations during the warm-up phase, when the learning rate is increasing linearly (see Figure 2b).

The fundamental issue with instability during training is that a large learning rate can cause the gradient to grow too large, surpassing the model’s capacity and resulting in a gradient explosion that prevents parameter updates. The problem is handled via reducing learning rate to 6 × 10−5, i.e. a proper learning rate located before the step where the initial spike in loss occurs (Cf. Figure 2c).

Mixed-Precision. Despite the potential instabilities associated with training models using half precision (float16) activations and model parameters that arise from the limited numerical range, it has been proposed that the numbers represented by bfloat16 allow for training of models and can avoid performance degradation compared to full float32 training. Thus, we incorporate the bfloat16 numerical format to reduce memory and increase training efficiency. However, similar to OPT-175B (Zhang et al., 2022), BLOOM-176B (Scao et al., 2022) and GLM-130B (Zeng

3https://github.com/NVIDIA/Megatron-LM

et al., 2022), the training of POLYLM-13B still faces frequent loss spikes while lowering learning rate. We attempted to address such challenge via manually skipping data and restart the straining, it unfortunately tends to become increasingly severe as the training does on (Cf. Figure 3a).

After conducting two weeks of investigation, we have come to the realization that the instabilities we are encountering may not be due to the training data under the mutlilingual scenario (with the vocabulary up to 256,000), but rather due to the model itself. Specifically, we suspect that there could be a risk of overflow in the attention or residual connectivity layers. Taking this into account, we have configured the residual connection and attention layers to have a numerical precision of float32 to ensure optimal performance, resulting in a highly stable training process (Cf. Figure 3b).

Curriculum Learning. Optimizing LLMs to learn knowledge encoded in multiple languages simultaneously is a significant challenge. We concretely formulate this problem as transferring general knowledge to low-resource languages while maintaining the advantage of high-resource language in the model. To address this issue, we adopt a curriculum learning strategy (Bengio et al., 2009; Kumar et al., 2010; Jaegle et al., 2021) that ramps up the ratio of high-quality and low-resource languages during training. Specifically, the training process is divided into two stages. In the first stage, we use the whole pre-training dataset to train a base model yields commonsense generalization ability. In the second stage, we transition to a subset of the pre-training dataset that boasts superior quality and a greater proportion of multilingual content, to further strengthen the model’s multilingual capabilities. Figure 4 compares the language distribution of training data in two stages, indicating that the proportion of most low-resource languages has been increased in the sub-dataset.

To build the sub-dataset for curriculum learning, we first manually evaluate the quality of publicly available data source in the pre-training dataset, and sample about 97B tokens from the high-quality sources while increasing the proportion of languages other than Chinese and English. We also enhance the proportion of parallel data (OPUS) to facilitate the modeling of cross-lingual representation. The detail of the sub-dataset are illustrated in Figure 5. According to our established setup, the curriculum training process is highly stable (Cf. Figure 3c).

- 4 MULTIALPACA: A MULTILINGUAL SELF-INSTRUCTION DATASET

Fine-tuning LLMs with instruction-based tasks has been proven effective in practice (Ouyang et al., 2022; Wei et al., 2022; Peng et al., 2023; Ye et al., 2023). By providing accurate task instructions during the SFT phase, LLMs can not only learn to understand the requirements of each task via the instruction part, but also show extensive abilities to cope with other types of tasks which are even unseen during training (Wei et al., 2022). Nevertheless, tuning multilingual LLMs is still troubled by the scarcity of current SFT datasets. On the one hand, most instruction-based datasets are mainly in resource-rich languages (e.g., English or Chinese). To the best of our knowledge, there is currently no high-quality multilingual instruction-based SFT dataset for LLM training. On the other hand, most instructions are manufactured by experienced language speakers (e.g., Wei et al., 2022). Although the quality of instructions is well preserved, the amount of tasks is rather scarce for fine-tuning LLMs.

To overcome these two drawbacks, we determine to extend the generality of our proposed POLYLM via creating a multilingual SFT dataset – MULTIALPACA (Figure 6). Following the self-instruct paradigm proposed by recent studies (Wang et al., 2022; Taori et al., 2023), we query the available LLM for responses, iteratively collecting and filtering self-instruct examples to build our dataset. MULTIALPACA delivers comprehensive support on multilingualism, covering up to 11 languages including Arabic (Ar), German (De), Spanish (Es), French (Fr), Indonesian (Id), Japanese (Ja), Korean (Ko), Portuguese (Pt), Russian (Ru), Thai (Th), and Vietnamese (Vi). For each language, the number of tasks in MULTIALPACA varies from 9,515 to 14,671, yielding 132,701 tasks in total.

- 4.1 TASK FORMAT

We first form the format of our tasks by referring to Taori et al. (2023), where each task contains three parts: 1) “instruction” describes the requirements of the corresponding task; 2) “input” can complement the “instruction” to a complete question; and 3) “output” is a correct answer of the question. We notice that, Taori et al. (2023) constructed their dataset where each

MULTIALPACA Dataset

Ar, 11.06% De, 7.17% Es, 7.50% Fr, 8.54% Id, 9.13%

Vi, 10.48%

Th, 8.66%

Ru, 10.77%

Pt, 8.16%

Ko, 10.85% Ja, 7.68%

- Figure 6: Statistics on the number of self-instruct tasks for each language in MULTIALPACA. For English and Chinese subsets, we directly use the released tasks by ALPACA Taori et al. (2023) and CHINESE-ALPACA Cui et al. (2023) for POLYLM training.

Collect Multilingual Seed Tasks

Iterative Progress

MULTIALPACA Dataset

Sample Tasks as Demonstrations to Construct Prompts

Query LLMs to Obtain Tuples (Instruction, Input, Output)

Similarity Checking Format Checking

Add New Tasks to the Task Pool

- Figure 7: Illustration on the construction of MULTIALPACA . We first translate English seed tasks from ALPACA (Taori et al., 2023) into multilingual ones. For each language, we arrange iterative progress to collect the MULTIALPACA dataset, including constructing prompts, collecting tasks via querying LLMs, format checking, filtering on diversity, and adding the new tasks into the task pool.

“instruction” can be equipped with multiple “input-output” instances. For simplicity, we only assign each “instruction” with one “input-output” instance.

- 4.2 MULTIALPACA CONSTRUCTION As shown in Figure 7, we construct the MULTIALPACA dataset based on the following steps:4

Collecting Multilingual Seed Tasks We first obtain 175 seed tasks from Taori et al. (2023) to construct the multilingual ones for MULTIALPACA. After manually checking them, we remove the cases where answering the questions requires cultural backgrounds (e.g., idiom explanation, character-level riddle, and lyrics generation). Then, we marked the cases whose original “input” or “output” should be reserved (e.g., single-choice question, translation, bias identification, and code generation), where those tasks will directly use the original “input” or “output” across different languages for MULTIALPACA. Finally, we filter out 13 inappropriate seed tasks, and modified 23 ones marked due to the reuse of “input” or “output” parts. We translate the remaining 162 tasks into the other 11 languages, yielding multilingual seed tasks for each language.

Iterative Progress We manage the MULTIALPACA dataset construction progress as an iterative one with multiple rounds. For each round, we manage the following five substeps in order:

4See Appendix A for more details.

- • Prompt Construction We follow Taori et al. (2023) to construct the prompts for MULTIALPACA when querying LLM for completion. When handling each involved language, for each prompt, we sample two seed tasks and one MULTIALPACA task as the demonstrations, and guide the LLM to complete the other 17 tasks in the response. For each round, we construct 100 prompts for querying the completion by LLM.5
- • Response Collection We collect the responses from CHATGPT via the OpenAI API service. The model we use is “gpt-3.5-turbo-0301”, which supports the processing of tokens up to 4,096.
- • Format Checking When checking the format, we first remove the last task if the response is stopped due to the exceeding of max sequence length. Then, we use the pre-defined task format to help split the response string, so as to make sure each of the tasks contains “instruction”, “input”, and “output” parts.
- • Similarity Checking After that, to preserve the diversity of MULTIALPACA dataset, we further check the similarity between the tasks that are newly collected and those from the task pool. Following Taori et al. (2023), we compute the Rouge-L F-scores between the instruction of each newly collected task and those of all collected ones. For each newly collected task, it would be added to the task pool only if all the scores are lower than 0.7.
- • Task Pool Updating In the end, we update the task pool by adding the newly collected tasks, and arrange the next round for collecting MULTIALPACA self-instruct tasks.

MULTIALPACA Dataset Export Totally, we arrange 10 rounds in the iterative progress when constructing the MULTIALPACA dataset. We export all tasks from the task pool as the MULTIALPACA dataset for SFT learning.

- 5 MULTILINGUAL BENCHMARK

We aim to assess the capabilities of POLYLM from various perspectives: 1) the ability of large language models (LLMs) to understand and generate natural languages, as well as the ability to grasp world knowledge; 2) the performance of LLMs across different languages; and 3) their capacity to handle cross-lingual tasks. Following the experiment design of previous work (Scao et al., 2022; Ahuja et al., 2023), we gather a subset of datasets from previous NLP tasks to construct a multilingual benchmark. The brief statistics of all datasets in the benchmark can be found in Table 4. The details of how we frame all the tasks with prompting are listed in Appendix B.

Task category Task Test Lang. Metric Prompt

XNLI 5,010 15 Acc. [Premise], right? {Yes/Also/No}, [Hypothesis] XCOPA 500 11 Acc. [Prefix] {because/therefore} {choice1/choice2} [Suffix] PAWS-X 2,000 7 Acc. [Sentence1], right? {Yes/No}, [Sentence2] XWINOGRAD 83-2,325 6 Acc. [Prefix] {choice1/choice2} [Suffix]

NLU

Knowledge TydiQA 1,625-14,805 9 F1 [Context][Question][Answer] NLG MTG 200 5 Rouge [Prompt][Input][Output] MT WMT20 991-3,002 8 BLEU [INPUT] Translate this sentence from [SRC] to [TGT].

Table 4: Multilingual Benchmark

- 5.1 TASKS IN BENCHMARK

All the datasets in the above multilingual benchmark can be divided into four groups: Natural Language Understanding, Knowledge, Natural Language Generation and Machine Translation. The details of each dataset that we use for benchmarking are given below.

To assess the comprehension capability of large models across various languages, we collect the multilingual versions of datasets from seberal wide-used NLP benchmarks (Wang et al., 2018; 2019).

- 5Except for the first round where the task pool is empty, we arrange 10 prompts for completion due to the

small number of available tasks for demonstrations.

XNLI (Conneau et al., 2019) serves as a benchmark to evaluate a model’s proficiency in predicting textual entailment. The task entails the evaluation of whether two given sentences, A and B, convey the same meaning, are contradictory, or are unrelated. The dataset has been professionally translated into 14 languages from the original English XNLI dataset.

PAWS-X (Yang et al., 2019) is a benchmark to evaluate the model’s ability to judge whether one sentence is the paraphrase of another. It is professionally translated from the PAWS (Zhang et al., 2019) dataset into 6 diverse languages.

XWinograd (Tikhonov & Ryabinin, 2021) serves as a benchmark to measure a model’s common sense reasoning ability. Specifically, the task entails presenting the model with a brief contextual passage and requiring it to select the accurate term from a set of two options for a pronoun in the passage.

XCOPA (Ponti et al., 2020) is another benchmark intended to assess the proficiency of models in commonsense reasoning across languages. The dataset comprises translations and re-annotations of the English COPA Gordon et al. (2011), spanning 11 languages around the globe. Based on the given premise and prompt, the task is to choose the more plausible response between two answer choices that can be inferred from the premise.

TyDi QA (Clark et al., 2020) is a question-answering dataset covering 11 typologically diverse languages with 200K question-answer pairs. We use this dataset to evaluate the ability to grasp knowledge from natural text. Unlike previous datasets such as MLQA (Lewis et al., 2020) and MKQA (Longpre et al., 2020), this dataset is collected directly in each language without the use of translation. We select 5 languages out of 11 that are included in the pretraining corpora of POLYLM. Following the PaLM (Chowdhery et al., 2022), we evaluate models on the Gold passage task, which requires answering questions based on a passage that is guaranteed to contain the answer.

MTG (Chen et al., 2021) is used to assess the efficacy of large language models in generating longer responses across diverse usage scenarios and multiple languages. MTG covers four different generation tasks: Story Ending Generation (SG), Title Generation (TG), Question Generation (QG), and Summarization (Summ). The datasets are originally written in English, subsequently extended into four other languages (German, French, Spanish, and Chinese) through the use of machine translation and human annotation. The effectiveness of LLM-generated responses is evaluated using the average of Rouge1, Rouge2, and RougeL.

WMT20 (Barrault et al., 2020) is used to study the cross-lingual proficiency of large language models in accomplishing translation tasks, as the process of translation entails both comprehending the semantic of the input in one language and expressing it in another. We select translation tasks between English and each of the following languages as benchmark languages: German, Japanese, Russian, and Chinese. The results are evaluated using the SacreBLEU (Post, 2018) and the scores for BLEU (Papineni et al., 2002) on the test set are reported.

- 5.2 EVALUATION DESIGN

For metric evaluation, the tasks included in our benchmark can be divided into two categories: classification-style tasks and generation-style tasks.

Classification-style tasks require selecting the correct option from several options, such as the XNLI dataset. To evaluate these tasks, following the way in Gao et al. (2021), we design the problem in the form of a cloze test, where each option is filled in to construct a complete sentence. We then choose the correct answer by separately calculating the log-likelihood of each completed sentence and selecting the one with the highest value.

Generation-style tasks, such as machine translation, require generating answers with several natural sentences. For these tasks, we adopt greedy decoding for deterministic results. Considering the efficiency of decoding, we restrict the maximum number of generated tokens to 256. For foundation models, we choose the result before the first ‘\n’ as the answer, while for models that have undergone instruction tuning, we decode until the EOS token appears.

In evaluating foundation models, considering that models have not been able to understand instructions, we adopt in-context learning (Brown et al., 2020) to evaluate the model for generation-style tasks. We generally choose no more than five examples due to the model’s context window limita-

BLOOM-7.1B LLaMA-13B POLYLM-13B

BLOOM-7.1B LLaMA-13B POLYLM-13B

de

ar

70

60

zh

de

60

50

zh

en

50

40

vi

en

30

40

20

30

ko

es

tr

es

th

fr ru

ja fr

PAWS-X

XNLI

BLOOM-7.1B LLaMA-13B POLYLM-13B

BLOOM-7.1B LLaMA-13B POLYLM-13B

id

en

90

80

70

60

zh

fr

zh

it

50

40

20

30

vi

th

ru

ja

tr

pt

XWinograd

XCOPA

- Figure 8: Accuracy of NLU tasks under the zero-shot setting. Best reviewed in colors. Results indicate that POLYLM performs comparably or better than LLaMA-13B in the English scenario, and exhibits significant enhancements in multilingual evaluation.

tion. For tasks that have well-divided training/development sets, we randomly draw examples from them for each test sample. Otherwise, we draw examples randomly from the test sets except for the current sample.

- 6 EXPERIMENTS

In this section, we provide separate comparison results for the pre-training and SFT models. Then, we analyze the effectiveness of our model in three aspects: curriculum learning, multilingual instruction finetuning, and the scaling for model size.

- 6.1 COMPARISONS BETWEEN PRE-TRAINED FOUNDATIONAL MODELS For the pre-trained models, we selected two mainstream open-source models as our baselines.

• LLaMA (Touvron et al., 2023) is a pre-trained language model released by MetaAI, which includes 7B, 13B, 30B, and 65B versions. The pre-training dataset is sourced from publicly

BLOOM-7.1B LLaMA-13B POLYLM-13B

BLOOM-7.1B LLaMA-13B POLYLM-13B

ar

zh

60

20

15

40

10

ru

en

20

de

en

5

0

0

ko id

fr es

(a) F1 Scores on TyDiQA.

(b) Average Rouge Scores on MTG.

BLOOM-7.1B LLaMA-13B POLYLM-13B

de→en

40

en→zh

ja→en

30

20

10

en→ru

ru→en

0

en→ja

zh→en

en→de

(c) BLEU Scores on WMT20.

- Figure 9: Performance on knowledge, neural language generation and machine translation tasks under the one-shot setting. Best reviewed in colors.

available corpora. The 33B and 65B models are trained on 1.4 T tokens, while the 7B and 13B models are trained on 1 T tokens. To ensure an equal parameter count comparison with POLYLM, we mainly take the 13B version into consideration.

• BLOOM (Scao et al., 2022) is a multilingual model that covers 46 natural languages and 13 programming languages with a maximum of 176B parameters. Since BLOOM has not released a 13B version, we opt for the BLOOM-7.1B model as our baseline.

We evaluate POLYLM across various multilingual tasks, covering natural language understanding (NLU), knowledge, natural language generation (NLG) and machine translation (MT). To make a clearer comparison of the multilingual capabilities of different models, we present the results using radar charts, with detailed results available in the C.

Natural Language Understanding. Figure 8 shows the results on four NLU tasks under the zeroshot setting. POLYLM-13B shows comparable performance to the English-centric LLaMA-13B model in the English scenario. Moreover, it yields substantial improvements of 7.2% and 19.1% on PAWS-X and XNLI respectively. For languages other than English (the multilingual column), POLYLM-13B outperforms LLaMA-13B with average improvement up to 7.6%, 5.6%, 3%, and 11% on XCOPA, PAWS-X, XWinagrad, and XNLI, respectively. When compared to the multilingual language model BLOOM-7.1B, POLYLM-13B outperforms with an average improvement of

BLOOMZ-MT-7.1B LLaMA-Alpaca-13B

BLOOMZ-MT-7.1B LLaMA-Alpaca-13B

POLYLM-MultiAlpaca-13B

POLYLM-MultiAlpaca-13B

ar

de

60

70

zh

de

50

60

zh

en

40

50

vi

en

30

40

20

30

ko

es

tr

es

th

fr

ja fr

XNLI

PAWS-X

ru

BLOOMZ-MT-7.1B LLaMA-Alpaca-13B

BLOOMZ-MT-7.1B LLaMA-Alpaca-13B

POLYLM-MultiAlpaca-13B

POLYLM-MultiAlpaca-13B

en

id

90

80

60

70

zh

fr

zh

it

40

50

30

20

vi

th

ru

ja

XCOPA

XWinograd

tr

pt

- Figure 10: Performance of instruction-followed models on NLU tasks under the zero-shot setting. Best reviewed in colors.

4.2%, 4.1%, 3.4%, and 4% points on the respective tasks. This improvement can be attributed to the higher percent of multilingual text during pre-training and curriculum learning strategy.

Knowledge. We evaluate our model on grasping multilingual knowledge by using the TyDiQA benchmark in the one-shot setting. Upon careful analysis of Figure 9a, it is evident that BLOOM-

- 7.1B experiences significant performance drops in the Korean (ko) and Russian (ru) language directions, whereas LLaMA-13B and POLYLM-13B exhibit better balance across all five languages. Furthermore, POLYLM-13B has an additional advantage of an average 1.2-point lead over LLaMA-

- 13B.

Natural Language Generation. Figure 9b displays the Rouge scores of four diverse NLG tasks in multilingual settings. From a multilingual perspective, POLYLM-13B outperforms all other models across four languages, namely Chinese (zh), Spanish (es), French (fr), and German (de). Moreover, in terms of task types, POLYLM-13B performs the best in question generation (QG) and summarization (Sum) tasks, while also showing comparable performance to the best model LLaMA-13B in the text generation (TG) task. Across all MTG tasks and languages, POLYLM-13B has an average score advantage of 1.6 and 2.3 compared to LLaMA-13B and BLOOM-7.1B, respectively.

Machine Translation We focus on evaluating the translation performance on four typologically diverse languages from WMT20 datasets, including translation directions both from and to English. Results of Figure 9c show that POLYLM-13B achieves similar performance to LLaMA-13B in the multilingual to English directions and surpasses LLaMA-13B and BLOOM-7.1B with average BLEU scores of 5.4 and 15.8 in the English to multilingual directions.

BLOOMZ-MT-7.1B LLaMA-Alpaca-13B

BLOOMZ-MT-7.1B LLaMA-Alpaca-13B

[Figure 148]

POLYLM-MultiAlpaca-13B

[Figure 149]

POLYLM-MultiAlpaca-13B

ar

zh

60

25

50

20

40

15

30

10

ru

en

de

en

20

5

10

0

0

ko id

fr es

(a) F1 Scores on TyDiQA.

(b) Average Rouge Scores on MTG.

BLOOMZ-MT-7.1B LLaMA-Alpaca-13B

[Figure 150]

POLYLM-MultiAlpaca-13B

de→en

35

30

en→zh

ja→en

25

20

15

10

5

en→ru

ru→en

0

en→ja

zh→en

en→de

(c) BLEU Scores on WMT20.

- Figure 11: Performance of instruction-followed models on knowledge, neural language generation and machine translation tasks under the zero-shot setting. Best reviewed in colors.

- 6.2 COMPARISONS BETWEEN INSTRUCTION-FOLLOWED MODELS

This section focuses on evaluating the effectiveness of instruction-followed models founded on the pre-trained language models discussed in Section 6.1. We conduct a comparative analysis of POLYLM-MULTIALPACA-13B that is fine-tuned on POLYLM-13B using MULTIALPACA, against two other publicly available models:

- • BLOOMZ-MT-7B is initially pre-trained on BLOOM-7B, and later fine-tuned on the multilingual task mixture xP3-MT (Muennighoff et al., 2022).
- • LLaMA-Alpaca-13B is built based on the pre-trained model LLaMA-13B and fine-tuned on the English self-instruction dataset ALPACA (Taori et al., 2023).

Figure 10 and 11 present the performance comparisons of instruction-followed models with the zeroshot setting, considering various tasks and language directions. The results indicate that POLYLMMULTIALPACA-13B is comparable or superior to LLaMA-Alpaca-13B on all English tasks, although the latter is primarily trained on English-only instructions. On other non-English tasks, POLYLM-MULTIALPACA-13B significantly outperforms LLaMA-Alpaca-13B. This superiority can be attributed to the inclusion of more well-balanced multilingual datasets during the pre-training and instruction fine-tuning. In comparison to BLOOMZ-MT-7B, POLYLM-MULTIALPACA-13B has demonstrated consistent improvements across all tasks and languages. We have observed an outlier MTG, and we speculate that this may be due to the fact that MTG testsets are part of the xP3

XNLI

PAWS-X

w/o CL w/ CL

w/o CL w/ CL

65.0

55.0

60.0

50.0

Acc(%)

Acc(%)

55.0

45.0

50.0

40.0

45.0

35.0

30.0

40.0

ar de en es fr ru tr th vi zh

de en es fr ja ko zh

XWinograd

XCopa

w/o CL w/ CL

w/o CL w/ CL

85.0

70.0

80.0

60.0

Acc(%)

Acc(%)

75.0

50.0

70.0

40.0

65.0

60.0

30.0

en fr ja pt ru zh

id it th tr vi zh

(a) NLU tasks

MT

w/o CL w/ CL

31.00

26.00

BLEU

21.00

16.00

11.00

6.00

1.00

de2en ja2en ru2en zh2en en2de en2ja en2ru en2zh

(b) Machine translation task

- Figure 12: POLYLM-13B trained with curriculum learning reveals better performance in multiple languages in NLU and MT tasks.

dataset. We plan to refine our instruction tuning process for POLYLM by utilizing the xP3 dataset in order to delve deeper into this inconsistency.

Note that it is not feasible to fully assess the effectiveness of the model’s performance through downstream NLP tasks after instruction fine-tuning. Therefore, we have presented selected examples for qualitative analysis, which are fully outlined in Appendix D.

6.3 ANALYSIS Curriculum Learning. We validate the effectiveness of the curriculum learning strategy in NLU and MT tasks of multilingual benchmark (Section 5.1) by comparing the following variants:

- (1) w/o CL POLYLM-13B trained without curriculum learning, which is only optimized in pretrained dataset.
- (2) w/ CL POLYLM-13B trained with curriculum learning, using about 100B high-quality multilingual data selected from the pretrained dataset.

Please note that we only focus on the languages included during curriculum learning. Referring to Figure 12, the model with curriculum learning has achieved stable progress in mainly all languages in both NLU and MT tasks. First of all, the model performance is enhanced in most low-resource languages, indicating that the general knowledge can be effectively transferred to these languages through raising data proportion. Additionally, the model retains its superior performance in English,

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

(a) NLU tasks

[Figure 155]

(b) Machine translation task

Figure 13: The performance of models with different scales on different tasks.

WMT20 Machine Translation TyDiQA en2zh en2de en2ru en2ja zh2en de2en ru2en ja2en Avg. en ar id ko ru Avg.

Model

POLYLM-Alpaca-13B 10.0 17.1 5.0 6.3 12.2 23.8 24.7 10.0 13.6 53.7 45.5 55.3 29.9 40.4 45.0 POLYLM-MultiAlpaca-13B 21.9 25.9 17.9 16.2 25.0 34.2 31.8 14.3 23.4 58.7 50.7 52.1 30.1 40.3 46.4

Table 5: BLEU scores on WMT20 machine translation tasks and F1 scores on Tydi-QA task.

which illustrates that improving data quality for high-resource languages can achieve competitive results to training with larger amounts of data. Finally, it is worth noting that introducing more multilingual parallel data during the curriculum learning significantly boost the model performance on translation task.

Multilingual Self-instruction. Here we highlight the advantages of MULTIALPACA over Englishonly ALPACA (Taori et al., 2023), particularly in cross-lingual tasks (i.e., machine translation). As illustrated in Table 5, compared to the model fine-tuned only using ALPACA, POLYLMMULTIALPACA-13B exhibits substantial improvements in TyDiQA and multiple WMT20 translation tasks, with enhancements of +10 BLEU and +1.4% F1. These results suggest that MULTI-

ALPACA is capable of simulating the cross-lingual alignment ability of the foundational, as well as facilitating the comprehension of multilingual instructions.

Scaling for Model Size. In addition to the 13B model, we also release a smaller 1.7B model. Recent studies highlight the critical role of model size in the performance of large language models (LLMs), with much of this work focusing on English (Kaplan et al., 2020; Rae et al., 2021; Biderman et al., 2023; Touvron et al., 2023). In this section, we present results for POLYLM-13B and POLYLM-1.7B to investigate the impact of model size on multilingual abilities. Consistent with the aforementioned experimental setup for the validation of base model, we compare the two models using a one-shot setting. As illustrated in Figure 13, the 13B model significantly outperforms the 1.7B model across all compared multilingual tasks. We posit that multilingual problems are more complex than their monolingual counterparts and may depend more heavily on the model’s throughput. Moving forward, we plan to release additional models of varying sizes, with the ultimate goal of refining the scaling law for multilingualism.

- 7 CONCLUSION

Multilingualism poses an inevitable challenge for LLM due to the scarcity of resources. In this work, we release POLYLM – a new multilingual LLM, alone with MULTIALPACA – a multilingual instruction dataset, and a multilingual benchmark. Quantitative and qualitative analyses demonstrate the superiority of POLYLM over open-source models in non-English languages. We find that incorporating curriculum learning strategy can boost the performance of LLM on non-English languages, without impeding its English proficiency. In addition, fine-tuning LLM with multilingual instruction data can considerably improve zero-shot performance on these languages.

There is still ample opportunity for refinement in our work endeavors. For instance, while we briefly assess the model’s capacity to comprehend multilingual instructions, there is potential for further optimization through the amalgamation of data sources (Wang et al., 2023; Longpre et al., 2023), evolutionary methods (Xu et al., 2023) and diversification strategies (Zhou et al., 2023). Moreover, in our current version, we adopt absolute position encoding, which adheres to the early default configuration in Megatron toolkit (Shoeybi et al., 2020). Future iterations should incorporate techniques that facilitate the expansion of window size, such as rotary position encoding (Su et al., 2021; Chen et al., 2023) or ALiBi (Press et al., 2022).

Language serves as a conduit for culture, and the unique contributions of various languages enrich and diversify our global community. Nevertheless, the advancement of LLM may inadvertently amplify the influence of prominent languages and present a formidable obstacle for low-resource languages. In light of these concerns, we aspire that our research will motivate further inquiry and innovation in the field of multilingual LLM.

ETHICS STATEMENT

In this paper, we propose POLYLM, an LLM which offers a wider support on non-English languages. Our contributions are fully methodological: adding the support of multilingualism to LLM during training and SFT phases. However, when building our POLYLM model, it is unavoidable that our POLYLM might exhibit several common deficiencies of language models, e.g., hallucination and toxicity. Specifically, as the collected MULTIALPACA dataset are generated by CHATGPT, the pseudo tasks might give inappropriate pseudo tasks which are hardly filtered out, e.g., hallucinated reasoning and anti-fact statements (Brown et al., 2020; OpenAI, 2023). Besides, POLYLM may deliver toxic texts, which might be gender- or race-biased like other existing LLMs (Taori et al., 2023; Cui et al., 2023).

Despite the ethical concerns above, we think that those problems are of vital importance to the AI community to study the deficiencies of LLMs. We recommend that the users of POLYLM and MULTIALPACA deploy our released materials only for research proposals. Besides, we suggest the users better identify the deficiencies of those contents, and welcome the following researchers to facilitate further research on the alignment between the LLM outputs and human values with POLYLM and MULTIALPACA materials.

REFERENCES

Kabir Ahuja, Rishav Hada, Millicent A. Ochieng, Prachi Jain, Harshita Diddee, Samuel Maina, Tanuja Ganu, Sameer Segal, Maxamed Axmed, Kalika Bali, and Sunayana Sitaram. Mega: Multilingual evaluation of generative ai. ArXiv, abs/2303.12528, 2023.

Yuvanesh Anand, Zach Nussbaum, Brandon Duderstadt, Benjamin Schmidt, and Andriy Mulyar. Gpt4all: Training an assistant-style chatbot with large scale data distillation from gpt-3.5-turbo. https://github.com/nomic-ai/gpt4all, 2023.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

Lo¨ıc Barrault, Magdalena Biesialska, Ondˇrej Bojar, Marta R. Costa-juss`a, Christian Federmann, Yvette Graham, Roman Grundkiewicz, Barry Haddow, Matthias Huck, Eric Joanis, Tom Kocmi, Philipp Koehn, Chi-kiu Lo, Nikola Ljubeˇsi´c, Christof Monz, Makoto Morishita, Masaaki Nagata, Toshiaki Nakazawa, Santanu Pal, Matt Post, and Marcos Zampieri. Findings of the 2020 conference on machine translation (WMT20). In Proceedings of the Fifth Conference on Machine Translation, pp. 1–55, Online, November 2020. Association for Computational Linguistics. URL https://aclanthology.org/2020.wmt-1.1.

Yoshua Bengio, R´ejean Ducharme, and Pascal Vincent. A neural probabilistic language model. In Advances in neural information processing systems, 2000.

Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning, ICML 2009, volume 382, pp. 41–48. ACM, 2009. URL https://doi.org/10.1145/1553374. 1553380.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. arXiv preprint arXiv:2304.01373, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation, 2023.

Yiran Chen, Zhenqiao Song, Xianze Wu, Danqing Wang, Jingjing Xu, Jiaze Chen, Hao Zhou, and Lei Li. Mtg: A benchmarking suite for multilingual text generation. In NAACL-HLT, 2021.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garc´ıa, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark D´ıaz, Orhan Firat, Michele Catasta, Jason Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. ArXiv, abs/2204.02311, 2022.

Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Association for Computational Linguistics, 2020.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzm´an, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In ACL, 2019.

Yiming Cui, Ziqing Yang, and Xin Yao. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https: //aclanthology.org/N19-1423.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, Sep 2021. URL https://doi.org/10.5281/zenodo.5371628.

Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In International Conference on Artificial Intelligence and Statistics, 2010.

Andrew S. Gordon, Zornitsa Kozareva, and Melissa Roemmele. Semeval-2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In International Workshop on Semantic Evaluation, 2011.

Kenneth Heafield. Kenlm: Faster and smaller language model queries. In Proceedings of the sixth

workshop on statistical machine translation, pp. 187–197, 2011. Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv: Learning, 2016. Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza

Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Jo˜ao Carreira. Perceiver: General perception with iterative attention. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, volume 139 of Proceedings of Machine Learning Research, pp. 4651–4664. PMLR, 2021. URL http://proceedings.mlr.press/v139/ jaegle21a.html.

Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, H´erve J´egou, and Tomas Mikolov. Fasttext. zip: Compressing text classification models. arXiv preprint arXiv:1612.03651, 2016.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Taku Kudo and John Richardson. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.

M. Pawan Kumar, Benjamin Packer, and Daphne Koller. Self-paced learning for latent variable models. In Advances in Neural Information Processing Systems 23: 24th Annual Conference on Neural Information Processing Systems 2010, pp. 1189–1197. Curran Associates, Inc., 2010. URL https://proceedings.neurips.cc/paper/2010/hash/ e57c6b956a6521b28495f2886ca0977a-Abstract.html.

Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. MLQA: Evaluating cross-lingual extractive question answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, and Adam Lopez. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 9019–9052, 2022.

S. Longpre, Yi Lu, and Joachim Daiber. Mkqa: A linguistically diverse benchmark for multilingual open domain question answering. Transactions of the Association for Computational Linguistics, 9:1389–1406, 2020.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.

Tomas Mikolov, Martin Karafi´at, Luk´as Burget, Jan Cernock´y, and Sanjeev Khudanpur. Recurrent neural network based language model. In Takao Kobayashi, Keikichi Hirose, and Satoshi Nakamura (eds.), INTERSPEECH, pp. 1045–1048. ISCA, 2010. URL http://dblp.uni-trier.de/db/conf/interspeech/interspeech2010. html#MikolovKBCK10.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786, 2022.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Gray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id= TG8KACxEON.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Annual Meeting of the Association for Computational Linguistics, 2002.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023.

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023.

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 2227–2237, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1202. URL https://aclanthology.org/N18-1202.

Edoardo M. Ponti, Goran Glava s, Olga Majewska, Qianchu Liu, Ivan Vuli’c, and Anna Korhonen. XCOPA: A multilingual dataset for causal commonsense reasoning. arXiv preprint, 2020. URL https://ducdauge.github.io/files/xcopa.pdf.

Matt Post. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pp. 186–191, Belgium, Brussels, October 2018. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/ W18-6319.

Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation, 2022.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. URL https://s3-us-west-2. amazonaws. com/openaiassets/researchcovers/languageunsupervised/language understanding paper. pdf, 2018.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagn´e, Alexandra Sasha Luccioni, Fran¸cois Yvon, Matthias Gall´e, et al. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.

Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

Oleh Shliazhko, Alena Fenogenova, Maria Tikhonova, Vladislav Mikhailov, Anastasia Kozlova, and Tatiana Shavrina. mgpt: Few-shot learners go multilingual, 2022.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020.

Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990, 2022.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. CoRR, abs/2104.09864, 2021. URL https://arxiv.org/abs/2104.09864.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Alexey Tikhonov and Max Ryabinin. It’s all in the heads: Using attention heads as a baseline for cross-lingual transfer in commonsense reasoning, 2021.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30, NIPS 2017 4-9 December 2017, Long Beach, CA, USA, pp. 5998–6008, 2017. URL http://papers.nips.cc/paper/ 7181-attention-is-all-you-need.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. ArXiv, abs/1804.07461, 2018.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. In Neural Information Processing Systems, 2019.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022.

Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. How far can camels go? exploring the state of instruction tuning on open resources, 2023.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. URL https://openreview.net/ forum?id=gEZrGCozdqR.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In International Conference on Machine Learning, pp. 10524–10533. PMLR, 2020.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions, 2023.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mt5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934, 2020.

Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. Paws-x: A cross-lingual adversarial dataset for paraphrase identification. In EMNLP, 2019.

Seonghyeon Ye, Hyeonbin Hwang, Sohee Yang, Hyeongu Yun, Yireun Kim, and Minjoon Seo. In-context instruction learning. arXiv preprint arXiv:2302.14691, 2023.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414, 2022.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Yuan Zhang, Jason Baldridge, and Luheng He. PAWS: Paraphrase Adversaries from Word Scrambling. In NAACL, 2019.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. Lima: Less is more for alignment, 2023.

- A DETAILED SETTING FOR MULTIALPACADATASET CONSTRUCTION

- A.1 PROMPT FOR MULTIALPACA DATASET CONSTRUCTION

We show the used prompt when constructing MULTIALPACA dataset in Table 6. We mainly refer to Taori et al. (2023), and adopt our prompt to multilingual scenarios after minor revisions. Briefly, in the prompt, we list several requirements of the self-instruct tasks in the prompt, i.e., the used language, the format, the diversity, and the lengths of tasks within each single response. We also add three demonstrations to help the model generate the tasks which follow the pre-defined format.

#### Prompt template for MULTIALPACA dataset construction.

You are asked to come up with a set of 20 diverse task instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:

- 1. Try not to repeat the verb for each instruction to maximize diversity.
- 2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
- 3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
- 4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
- 5. The instructions should be in [language].
- 6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
- 7. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
- 8. Not all instructions require input. For example, when an instruction asks about some general information, “what is the highest peak in the world”, it is not necessary to provide a specific context. In this case, we simply put “¡noinput¿” in the input field.
- 9. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 200 words. There are 3 examples.

- 1. Instruction: [task1 instruction]

- 1. Input:

- [task1 input]

- 1. Output:

- [task1 output]

2. Instruction: [task2 instruction]

2. Input:

- [task2 input]

- 2. Output: [task2 output]

- 3. Instruction: [task3 instruction]

3. Input:

- [task3 input]

- 3. Output: [task3 output] Please generate the following 17 tasks that are similar to the above examples.

Table 6: Prompt for constructing MULTIALPACA tasks. We specify the language of generated tasks with the parameter “language”, and the used demonstrations with “task[123] {instruction,input,output}”.

100

Ar De Es Fr Id Ja Ko

Percentage(%)

80

Pt

Ru Th Vi

60

40

0 5000 10000 15000

Number of tasks before checking

Figure 14: The percentage (%) of the MULTIALPACA examples which pass the format and similarity checking. For each language, we compute the ratio of collected tasks to that before format and similarity checking during each round.

#### Language # tasks Ratio (%)

Ar 14,671 11.06 De 9,515 7.17 Es 9,958 7.50 Fr 11,332 8.54 Id 12,117 9.13 Ja 10,191 7.58 Ko 14,402 10.85 Pt 10,825 8.16 Ru 14,286 10.77 Th 11,496 8.66 Vi 13,908 10.48

Total 132,701 100.00 Table 7: The number of tasks for each language in MULTIALPACA dataset.

- A.2 FORMAT AND SIMILARITY CHECKING

After collecting the pseudo tasks for each language, we first remove the cases which contain website links. Then, we tokenize the “instruction”, “input”, and “output” with available tokenization toolkits.6

Besides, we found that some of the tasks may give redundant information within both “instruction” and “input” part. We revise those examples to make them available as much as possible. In detail, if the “input” part can be a sub-string of the “instruction”, we mark the “input” part as an empty one (using the placeholder “<noinput>”). Otherwise, we compute the Rouge-L F-score between the “instruction” and the “input” part, filtering out the tasks whose result is over 0.5. Specially, for the Ko, Vi, and Ar tasks, we determine the threshold of Rouge-L F-score as 0.3, 0.3, and 0.2 for higher diversity of tasks, respectively.

We show the percentage of remaining examples after similarity checking in Figure 14. For each language, we show the number of self-instruct tasks in MULTIALPACA in Table 7.

6nltk.tokenize.wordpunct tokenize for Ar, De, Es, Fr, Id, Pt, Ru, and Vi; kytea for Ja; Okt for Ko; thai tokenizer for Th.

### B DETAILS OF TASK FORMATTING We list the detailed format of all the tasks in our multilingual benchmark in the following tables.

Context → Using these eight simple techniques, you can fabricate a news story in the comfort of your own home., right?

Correct Answer → No, Only news reporters in a newsroom can write a news story, and it takes 20 steps to do it.

Incorrect Answer → Yes, Only news reporters in a newsroom can write a news story, and it takes 20 steps to do it. Also, Only news reporters in a newsroom can write a news story, and it takes 20 steps to do it.

Table 8: The task format of XNLI. We normalize the log-likelihood by token numbers of all the correct and incorrect answers and choose the one with the largest score as prediction.

Context → Il cursore sullo schermo del computer si e` mosso perch´e Correct Answer → l’utente ha spostato il mouse.

Incorrect Answer → l’utente ha cliccato il mouse. Table 9: The task format of XCOPA.

Context → He put snow on the smiley face because Correct Answer → snow was wet.

Incorrect Answer → the smiley face was wet. Table 10: The task format of XWinograd.

Context → The first category is monovalent verbs , where there is only one semantic argument and it consists of both unergative verbs and unaccusative verbs

., right?

Correct Answer → Yes, The first category is unergative verbs , where there is only one unaccountable argument and consists of both semantic verbs and monovalent verbs .

Incorrect Answer → No, The first category is unergative verbs , where there is only one unaccountable argument and consists of both semantic verbs and monovalent verbs .

Table 11: The task format of PAWS-X.

### C DETAILS OF EXPERIMENTAL RESULTS

- C.1 RESULTS OF PRETRAINED LANGUAGE MODELS
- C.2 RESULTS OF SFT MODELS

### D DEMONSTRATIONS FOR QUALITATIVE ANALYSIS

Context → Read the context and answer the question in one or a few words in English.

Context (English): Football games last for a total of 60 minutes in professional and college play and are divided into two-halves of 30 minutes and four-quarters of 15 minutes.[74][75] High school football games are 48 minutes in length with two-halves of 24 minutes and four-quarters of 12 minutes.[76] (...). Answer:

Target Completion → class B Table 12: The task format of Tydi-QA.

Context → Please generate a title for the following document in English

document: justin timberlake’s super bowl lii halftime show is approaching, and there are rumors circulating around the internet saying that nsync may have a reunion at the annual championship. well, it appears that fans of the boyband have to kiss that dream goodbye after joey fatone shuts down the rumors. tmz recently caught up with the 40-year-old hot dog purveyor and asked whether the reunion rumors were true. (...). title:

Target Completion → joey fatone shuts down nsync reunion rumors at the super bowl lii Table 13: The task format of Title Generation task in MTG.

Context → Write a story end of the following story in just a few sentences in English. story: john had a roommate he wanted to prank. john called and ordered ten pizzas to be delivered to the apartment. suddenly, john’s roommate got a call and had to leave. story ending:

Target Completion → it was too late to cancel the pizza order! john wound up paying for all the pizzas!

Table 14: The task format of Story Ending Generation task in MTG.

Context → Given a passage and a concept that can be found in this passage, please generate a question in English, the answer of which is this concept and is answerable after reading this passage. passage: a treaty is an official, express written agreement that states use to legally bind themselves. a treaty is the official document which expresses that agreement in words; and it is also the objective outcome of a ceremonial occasion which acknowledges the parties and their defined relationships. answer: themselves question:

Target Completion → who is responsible for the legally-bound obligations of the parties to a treaty?

Table 15: The task format of Question Generation task in MTG.

Context → Please generate a short summary of the given document in English document: a man whose girlfriend ran off with his stepfather and gave birth to a baby nearly four years ago has spoken of his delight after dna tests proved the baby is actually his. love rat stan crowther, 47, and rachel delaney, 18 at the time, set up home together and rachel then gave birth to a baby daughter, whom stan believed was his. but recent bombshell dna tests showed that the baby, living in chorley, lancashire, actually belongs to her former partner, and stan’s former stepson, ashley mercer, 27. (...). summary:

Target Completion → ashley mercer was 22 when he discovered his girlfriend rachel delaney, 18, was having an affair with his stepfather stan crowther, 43. stan had been married to ashley’s mother mandy rourke for ten years. stan and rachel moved in together and shortly after rachel, from chorley, lancashire, gave birth to a baby girl. stan assumed the baby to be his but dna tests have since revealed ashley is the four-year-old girl’s real father. ashley’s mother mandy rourke has now forgiven her ex-husband. stan and rachel are no longer together.

Table 16: The task format of Summarization task in MTG.

Context → Oil falls after Iran claims US offered to remove sanctions, Trump denies Translate this sentence from English to German.

Target Completion → Ol¨ f¨allt, nachdem Iran behauptet, die USA h¨atten Aufhebung der Sanktionen angeboten, Trump dementiert

Table 17: The task format of Translation.

en zh ar es fr ru th tr vi de Average bg el hi sw ur Average BLOOM-7.1B 53.9 35.5 33.8 48.7 49.8 42.6 34.9 34.9 47.4 39.6 42.1 39.3 35.5 46.7 37.9 41.9 40.3

- LLaMA-13B 35.5 34.6 34.1 33.4 33.6 33.6 34.6 34.0 34.1 35.2 34.3 33.9 34.5 35.7 33.2 34.1 34.3 {Poly}LM-13B 54.6 35.9 33.6 50.0 52.1 49.0 44.6 44.5 46.7 50.2 46.1 36.3 33.8 34.9 34.4 33.5 34.6

Table 18: Accuracy on XNLI.

en zh es fr ja ko de Average

BLOOM-7.1B 61.3 47.3 59.4 50.9 45.5 45.1 52.9 51.8 LLaMA-13B 53.7 45.2 52.1 54.5 45.0 47.1 53.0 50.1 POLYLM-13B 60.9 56.0 59.7 56.6 51.0 46.8 60.3 55.9

Table 19: Accuracy on PAWS-X.

en zh ja pt ru fr Average

BLOOM-7.1B 82.2 74.4 58.5 76.8 56.8 71.1 70.0 LLaMA-13B 86.8 70.0 59.9 71.5 70.8 68.7 71.3 POLYLM-13B 84.6 76.6 65.7 74.9 65.1 73.5 73.4

Table 20: Accuracy on XWinograd.

id it th tr vi zh Avg. et ht qu sw ta Avg.

BLOOM-7.1B 69.8 52.8 55.4 51.2 70.8 65.2 60.9 48.2 50.8 50.8 51.6 59.2 52.1 LLaMA-13B 57.8 67.2 54.6 53.0 53.8 58.4 57.5 48.2 52.8 50.2 51.2 54.4 51.4 POLYLM-13B 70.2 66.0 58.6 57.8 70.8 67.0 65.1 49.8 50.4 50.4 51.8 55.0 51.5

- Table 21: Accuracy on XCOPA. The left part presents the results of languages we mainly considered in the training phrase, while the right part shows the other languages in the testsets. ‘Avg.’ means the average accuracy.

ar en id ko ru Average fi bn sw te Average

BLOOM-7.1B 42.6 51.6 48.7 8.6 33.8 37.1 17.5 55.1 56.8 40.9 42.6 LLaMA-13B 49.7 54.4 43.4 49.7 41.8 47.8 40.2 32.0 33.5 8.5 28.6 POLYLM-13B 44.9 58.0 48.6 53.8 39.5 49.0 20.9 2.9 22.0 3.7 12.4

- Table 22: F1 scores on the TyDiQA-GoldP benchmark under one-shot conditions. The left part presents the results of languages we mainly considered in the training phrase, while the right part shows the other languages in the testsets.

Task zh en es fr de Average

BLOOM-7.1B 16.6 13.2 11.0 13.1 6.6 12.1 SG LLaMA-13B 16.5 12.5 2.0 8.1 0.7 8.0

POLYLM-13B 10.3 10.3 7.8 9.2 6.3 8.8

- BLOOM-7.1B 12.4 14.2 10.1 10.9 5.7 10.7

TG LLaMA-13B 11.5 19.6 15.9 16.8 10.6 14.9 POLYLM-13B 16.8 16.5 14.2 14.3 10.1 14.4

- BLOOM-7.1B 13.8 15.5 15.0 13.5 7.3 13.0

QG LLaMA-13B 20.2 16.3 14.6 13.1 4.5 13.7 POLYLM-13B 20.0 15.9 17.1 13.9 7.5 14.9 BLOOM-7.1B 11.0 10.3 10.0 9.7 7.5 9.7

Sum LLaMA-13B 14.6 18.1 8.1 10.3 7.6 11.7 POLYLM-13B 16.6 17.0 16.9 15.1 11.6 15.4 BLOOM-7.1B 13.5 13.5 11.5 11.8 6.8 11.4

Average LLaMA-13B 15.7 16.6 10.1 12.1 5.9 12.1 POLYLM-13B 17.5 14.9 14.0 13.1 8.9 13.7

- Table 23: Rouge scores on the MTG benchmark under one-shot conditions. Results are presented in two dimensions: language directions and task types. The bottom row shows the mean values of diverse language directions across all tasks. Similarly, the rightmost column depicts the average values of varied tasks across all language directions.

de→en ja→en ru→en zh→en Avg. en→de en→ja en→ru en→zh Avg.

BLOOM-7.1B 23.9 6.5 17.7 16.3 16.1 3.7 2.1 1.4 8.3 3.9 LLaMA-13B 36.9 13.6 32.7 22.6 26.5 23.6 9.6 16.8 7.4 14.3 POLYLM-13B 33.9 10.2 32.4 22.0 24.6 24.8 17.0 17.8 19.0 19.7

- Table 24: Translation BLEU scores on the WMT20 machine translation task under one-shot conditions. ‘Avg.’ means the average BLEU scores of translations to English or from English.

en zh ar es fr ru th tr vi de Average bg el hi sw ur Average

BLOOMZ-MT-7.1B 44.9 33.2 36.0 36.1 47.5 38.1 33.5 33.4 37.9 42.5 38.3 35.5 35.1 39.6 33.2 37.9 36.3 LLaMA-Alpaca-13B 35.7 34.6 33.2 33.3 33.4 33.4 36.2 34.7 34.6 33.7 34.3 34.3 34.1 35.8 33.0 32.8 34.0 POLYLM-MULTIALPACA-13B 54.3 36.0 33.3 47.0 49.6 46.7 38.4 44.1 44.6 48.0 44.2 36.1 32.7 34.1 33.1 33.8 34.0

##### Table 25: Accuracy of instruction-followed models on XNLI.

en zh es fr ja ko de Average

BLOOMZ-MT-7.1B 61.2 58.5 58.8 61.1 54.4 49.5 56.7 57.2 LLaMA-Alpaca-13B 53.3 45.3 53.5 54.1 46.0 48.3 54.3 50.7 POLYLM-MULTIALPACA-13B 65.3 55.9 63.0 59.8 52.3 53.7 64.9 59.3

Table 26: Accuracy of instruction-followed models on PAWS-X.

en zh ja pt ru fr Average

BLOOMZ-MT-7.1B 83.5 71.0 56.4 65.4 53.7 68.7 66.5 LLaMA-Alpaca-13B 88.6 67.3 61.4 73.0 72.1 78.3 73.5 POLYLM-MULTIALPACA-13B 83.9 73.6 65.2 72.2 67.9 71.1 72.3

Table 27: Accuracy of instruction-followed models on XWinograd.

id it th tr vi zh Avg. et ht qu sw ta Avg.

BLOOMZ-MT-7.1B 58.6 51.8 53.6 53.2 58.8 62.2 56.4 49.6 53.8 49.4 53.0 58.2 52.8 LLaMA-Alpaca-13B 55.4 70.8 54.6 53.0 53.0 60.0 57.8 47.2 53.0 51.8 51.0 56.0 51.8 POLYLM-MULTIALPACA-13B 71.6 66.8 60.2 58.8 71.8 75.6 67.5 48.4 52.0 50.4 50.8 55.4 51.4

Table 28: Accuracy of instruction-followed models on XCOPA.

en ar id ko ru Avg. bn fi sw te Avg.

BLOOMZ-MT-7.1B 22.4 36.6 26.9 5.8 9.1 20.2 26.7 2.4 14.4 26.5 17.5 LLaMA-Alpaca-13B 59.2 20.8 48.6 19.3 37.7 37.1 11.0 50.6 20.7 5.7 22.0 POLYLM-MULTIALPACA-13B 58.7 50.7 52.1 30.1 40.3 46.4 2.5 8.5 4.6 1.9 4.4

Table 29: F1 scores of instruction-followed models on the TyDiQA-GoldP benchmark.

Task zh en es fr de Average

BLOOMZ-MT-7.1B 12.1 12.6 9.8 12.1 5.6 10.4 SG LLaMA-Alpaca-13B 3.8 10.0 4.8 5.7 5.6 6.0

POLYLM-MULTIALPACA-13B 13.9 9.5 9.6 11.5 8.0 10.5 BLOOMZ-MT-7.1B 18.5 25.7 18.9 16.8 9.6 17.9

TG LLaMA-Alpaca-13B 9.3 25.1 15.5 15.5 10.1 15.1 POLYLM-MULTIALPACA-13B 15.3 23.7 16.6 15.9 11.7 16.6 BLOOMZ-MT-7.1B 23.8 31.9 29.6 26.4 9.2 24.2

QG LLaMA-Alpaca-13B 4.7 27.5 27.3 19.2 18.1 19.4 POLYLM-MULTIALPACA-13B 21.8 25.7 27.2 19.3 16.3 22.1

- BLOOMZ-MT-7.1B 16.5 18.4 19.7 18.4 11.0 16.8

Sum LLaMA-Alpaca-13B 2.6 22.0 16.9 15.2 13.4 14.0 POLYLM-MULTIALPACA-13B 15.4 21.5 18.1 15.2 13.0 16.6

- BLOOMZ-MT-7.1B 17.7 22.2 19.5 18.4 8.9 17.3

Average LLaMA-Alpaca-13B 5.1 21.2 16.1 13.9 11.8 13.6

POLYLM-MULTIALPACA-13B 16.6 20.1 17.9 15.5 12.3 16.5 Table 30: Rouge scores of instruction-followed models on the MTG benchmark.

de→en ja→en ru→en zh→en Avg. en→de en→ja en→ru en→zh Avg.

BLOOMZ-MT-7.1B 17.99 5.09 7.67 12.89 10.91 2.94 0.99 0.77 8.40 3.3 LLaMA-Alpaca-13B 30.0 9.9 20.3 16.4 19.14 16.8 3.6 10.6 3.0 8.5 POLYLM-MULTIALPACA-13B 34.2 14.3 31.8 25.0 26.32 25.9 16.2 17.9 21.9 20.5

Table 31: Translation BLEU scores of instruction-followed models on the WMT20 machine translation task.

Input Text

请给出⼀个计算标准差的公式。

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

标准差是衡量数据离散程度的⼀个统计量，其计算公式为std = sqrt(mean(abs(x x.mean())**2))。

标记

Examples

The Moon's orbit around Earth has

The smooth Borealis basin in the Northern Hemisphere covers 40%

31

Input Text

命名⼀种⻝物。 这种⻝物由以下配料制成：

- - 鸡蛋
- - ⾯粉
- - 奶油
- -⾁松

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

这种⻝物被称为⾁松蛋糕。

标记

Examples

The Moon's orbit around Earth has

32

The smooth Borealis basin in the Northern Hemisphere covers 40%

Input Text

从以下句⼦中选择与给出句⼦含义相同的句⼦。 过完这个周末后，这个项⽬应该完成。 选择：

- a) 这个项⽬很快就会结束了。
- b) 这个项⽬过了这个周末就完成了。
- c) 这个项⽬不会在周末前结束。

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

b) 这个项⽬过了这个周末就完成了。

标记

Examples

The Moon's orbit around Earth has

The smooth Borealis basin in the Northern Hemisphere covers 40%

33

Input Text

在指定的语境下，为⼀组单词选择相应的中⽂翻译。 下⾯是⼀ 组单词，为这些单词选择正确的中⽂翻译，语境为商业领域：

- - deadline
- - board meeting
- - marketing strategy

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

- - 截⽌⽇期
- - 董事会会议
- - 市场战略

标记

Examples

The Moon's orbit around Earth has

34

The smooth Borealis basin in the Northern Hemisphere covers 40%

Input Text

在给定的数据集中，找到所有包含某个关键词的条⽬。 数据集： 标题 ⽇期 作者 SARS疫情中的神经病理 2003年5⽉ 王⽼师 ⼉童肥胖问题的研究 2005年9⽉ 张医⽣ 123个分类问题的分析 2008年11⽉ 李教授 包含关键词“研究”的数据条⽬如下：

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

2005年9⽉，张医⽣的论⽂《⼉童肥胖问题的研究》中包括了关键词“研究”。

标记

Examples

The Moon's orbit around Earth has

35

The smooth Borealis basin in the Northern Hemisphere covers 40%

Input Text

在以下句⼦中找到连词“和”的使⽤，并说明其作⽤是连接哪些词语：我们今天早上起得很 晚，和昨天⼀样。

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

在这句话中，连词“和”连接了两个词语“今天早上”和“昨天”。

标记

Examples

The Moon's orbit around Earth has

36

The smooth Borealis basin in the Northern Hemisphere covers 40%

Use via API · Built with Gradio

Input Text

How does American cuisine vary by region?

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

American cuisine is characterized by its regional variations. Some of these variations include the types of cuisine that dominate di erent parts of the country, such as Southern cooking in the South or Midwest American cooking in the Midwest. Additionally, regional variations include local ingredients, cooking techniques, and even dishes. The cuisine of the American South is known for its use of corn, while the Midwest is known for its emphasis on meat dishes like barbecue and steak. Additionally, regional variations include dishes like pizza in the Northeast and barbecue in the South.

标记

Examples

37

The Moon's orbit around Earth has

The smooth Borealis basin in the Northern Hemisphere covers 40%

Use via API · Built with Gradio

Input Text

How do foreign nationals seek medical treatment in the United States?

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

Foreign nationals seeking medical treatment in the United States typically have to obtain a medical visa. The type of visa they need depends on the purpose of the visit and the type of medical treatment they need. They typically need a B-1/B-2 visa for visiting the United States for leisure, business, or medical treatment, and a J visa for medical treatments. The application process typically involves a medical exam and proof of health insurance. Once approved, they can begin their medical treatment.

标记

Examples

38

The Moon's orbit around Earth has

The smooth Borealis basin in the Northern Hemisphere covers 40%

Use via API · Built with Gradio

Input Text

Escribe una oración con el verbo "caminar" conjugado en pretérito imperfecto. Las mariposas ________ de flor en flor.

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

Las mariposas caminaban de flor en flor.

标记

Examples

The Moon's orbit around Earth has

39

The smooth Borealis basin in the Northern Hemisphere covers 40%

Input Text

Completa la serie numérica siguiente: 2, 4, 6, 8, __, 12

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

10

标记

Examples

The Moon's orbit around Earth has

40

The smooth Borealis basin in the Northern Hemisphere covers 40%

Input Text

Écrivez une courte biographie de Marie Curie.

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

Marie Curie était une physicienne et une chimiste française et polonaise, née à Varsovie en 1867 et décédée à Saint-Cloud en 1934. Elle est surtout connue pour ses découvertes dans le domaine de la radioactivité, qui ont révolutionné notre compréhension du fonctionnement des éléments chimiques.

标记

Examples

The Moon's orbit around Earth has

41

The smooth Borealis basin in the Northern Hemisphere covers 40%

Use via API · Built with Gradio

Input Text

Donnez-moi l'URL du site web de l'Université de Cambridge.

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

L'URL du site web de l'Université de Cambridge est http://www.cam.ac.uk.

标记

Examples

The Moon's orbit around Earth has

The smooth Borealis basin in the Northern Hemisphere covers 40%

42

Input Text

「過去に⾏われた⽇本のオリンピックの年は何ですか？」を答えてください

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

2020年の東京オリンピックは、2021年に延期されました。

标记

Examples

The Moon's orbit around Earth has

43

The smooth Borealis basin in the Northern Hemisphere covers 40%

Use via API · Built with Gradio

Input Text

明治時代における洋館建築の代表例を教えてください。

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

明治時代における洋館建築の代表例の⼀つは、東京の上野の国⽴博物館です。この建 物は、美しいデザイン、贅沢な内装、および優れた技術を備えています。

标记

Examples

The Moon's orbit around Earth has

44

The smooth Borealis basin in the Northern Hemisphere covers 40%

Input Text

  า "epiphany" หมาย งอะไร?

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

epiphiny หมาย ง "การ  ก าไ  นพบ ง ตนอยาก "  งเ น ง  าประ บใจมาก

标记

Examples

The Moon's orbit around Earth has

The smooth Borealis basin in the Northern Hemisphere covers 40%

45

Input Text

แปลง วเลขทศ ยมเ นเปอ เซน   วเลขทศ ยม: 0.75

tokens_to_generate

2048

temperature

1

temperature

top_k

6

topk

top_p

0

topp

清除 提交

Generated Text

 วเลขทศ ยมเห า นแปลงเ นเปอ เซน ไ  า 75 เปอ เซน 

标记

Examples

The Moon's orbit around Earth has

The smooth Borealis basin in the Northern Hemisphere covers 40%

46

