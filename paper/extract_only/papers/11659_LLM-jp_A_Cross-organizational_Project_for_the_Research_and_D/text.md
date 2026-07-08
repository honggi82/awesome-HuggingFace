# arXiv:2407.03963v2[cs.CL]30Dec2024

## LLM-jp: A Cross-organizational Project for the Research and Development of Fully Open Japanese LLMs

LLM-jp∗

### Abstract

This paper introduces LLM-jp, a cross-organizational project for the research and development of Japanese large language models (LLMs). LLM-jp aims to develop open-source and strong Japanese LLMs, and as of this writing, more than 1,500 participants from academia and industry are working together for this purpose. This paper presents the background of the establishment of LLM-jp, summaries of its activities, and technical reports on the LLMs developed by LLM-jp. For the latest activities, visit https://llm-jp.nii.ac.jp/en/.

### 1 Introduction

Large language models (LLMs), exemplified by GPT-4 [37], demonstrate remarkable capabilities. LLMs have achieved many long-standing goals of traditional natural language processing (NLP), shifting the primary focus of NLP research towards elucidating their intelligence, ensuring their safety, and exploring their integration and coexistence with humans in society.

However, there exist significant issues with LLMs. First, the research and development of LLMs require significant computational resources and substantial budgets, predominantly controlled by a few major organizations. Moreover, the specifics of the strongest models — including their architecture, pre-training corpus, training methodologies, and tuning data — are no longer publicly accessible. Additionally, several critical issues, such as hallucination and safety, must be addressed for LLMs to achieve widespread societal acceptance in the future.

There are also national concerns specific to Japan. The representation of Japanese in the GPT-3 dataset is just 0.11%2, which results in inferior comprehension and generation of Japanese compared to English. Furthermore, there is a worry that Japanese culture and activities may be overshadowed if models predominantly trained in English become the global standard. From an economic security perspective, it is crucial to consider the potential outflow of Japan’s intellectual assets when entirely relying on foreign models.

Against this background, LLM-jp started in May 2023 with the objective of developing Japanese LLMs on our own. The research and development of LLMs is now a big science in terms of both computational and human resources. Recognizing the need for widespread collaboration, we opted for complete transparency and decided to make everything openly available, from our models, corpora, and fine-tuning data to our discussions and failures, for both non-commercial and commercial use.

LLM-jp began as a small study group of about 30 NLP researchers. LLM-jp garnered increasing support for its concept over time, growing to over 1,500 participants by June 2024. Study groups have been held monthly since the establishment of LLM-jp in a hybrid (in-person and online) manner, to introduce the latest advances in LLMs and present the activity reports from LLM-jp.

∗Please cite this paper as “LLM-jp (2024)”. Contribution statements can be found at the end of the document. Correspondence regarding this paper can be sent to llm-jp@nii.ac.jp.

2https://github.com/openai/gpt-3/blob/master/dataset_statistics/languages_by_ word_count.csv

2024.4 Released LLM-jp-13B v2.0 Started to build 175B-class model

2024.2 METI program (GENIAC) (175B-class model training will start in 2024.4)

2023.11 ABCI's 2nd LLM construction support program (investigation for training 175B model)

2023.10 Released LLM-jp-13B v1.0

[Figure 1]

2023.6 Started to build 13B LLMs on mdx (a platform for data-empowered society)

2023.5 Volunteer study group of 30 NLP researchers

Figure 1: Timeline of key activities in LLM-jp.

For the development of LLMs, three working groups (WGs) were first established: the Corpus Building WG, Model Building WG, and Fine-tuning and Evaluation WG. Subsequently, the Computational Infrastructure WG was formed to address computational infrastructure challenges. Weekly online meetings and Slack discussions facilitated communication among the groups. As the project evolved, the Academic Domain WG and Safety WG were also created.

Our first model suite, which we call the LLM-jp model suite v1.0, was released on October 20th, 2023. Subsequently, we released the next model suite, called the LLM-jp model suite v2.0, on April 30th, 2024. Each model suite provides an LLM with 13B parameters along with its fine-tuned variants. We have made them public with their pre-training corpora and fine-tuning datasets.

In the following, we present the activities of the main WGs that played a central role in the construction of our LLMs and future prospects.

### 2 Corpus Building WG

#### 2.1 Overview

The main role of the Corpus Building WG is to build a pre-training corpus and a tokenizer needed for LLM construction and pass them to the Model Building WG.

In the following subsections, we describe our work for the pre-trained models in our model suites v1.0 and v2.0. Then, we explain the corpus search function, which is one of our advantages. Finally, we summarize our ongoing and future work.

#### 2.2 Work for Pre-trained Model v1.0

Our initial milestone was to develop the model suite v1.0, and the Corpus Building WG worked on preparing a pre-training corpus to train the pre-trained model v1.0, the LLM with 13B parameters within this suite. The main purpose of this development was to experience the entire development process of an LLM as soon as possible.

To this end, we decided to use a mixture of readily available Japanese, English, and code corpora as our pre-training corpus. As for the corpus size, we followed the Chinchilla scaling law [20], which suggests using roughly 20 tokens per parameter. Eventually, we constructed the corpus v1 consisting of over 260B tokens. The statistics of this corpus are listed in Table 1. From this corpus, we extracted a pre-training dataset that consists of 130B Japanese, 130B English, and 10B code tokens, resulting in a total of 270B tokens.

As for the Japanese portion, we used the Japanese parts of Wikipedia and the multilingual C4 (mC4) dataset [57]. Since the Japanese part of mC4 was noisy, we filtered out documents that were considered low-quality or harmful. Table 2 shows filters adopted for this purpose. For the English

- Table 1: Statistics of sub-corpora in the corpus v1. Language Sub-corpus Tokens Japanese

Wikipedia 1B mC4 136B

English

Wikipedia 5B Pile 176B

Code Stack 148B

- Table 2: Filters and conversions used for the corpus v1.

Filter / Conversion Description HasValidUrlDomain Filter out documents with URLs from domains rarely used in Japan. IsNotJapanese Filter out documents that do not contain hiragana or katakana characters. IsNotEthical Filter out documents that include toxic and/or offensive words. RemoveUrl Remove URLs from documents. RemoveCode Remove code-like text spans from documents.

and code portions, we utilized the Pile dataset [16] and the Stack dataset [29], respectively. To adjust the corpus size, we sampled documents from these two sources accordingly.

We developed tokenizers based on SentencePiece with the unigram mode [30]. As a multi-lingual tokenizer considering Japanese, we first explored the tokenizer developed in the project “Development of a distributed parallel learning method for large-scale language models in the policy-oriented framework of the supercomputer Fugaku”3, which we refer to as the tokenizer v1.0. The construction process is as follows:

- 1. Preparing training data to construct the tokenization models for each language (i.e., Japanese and English).
- 2. In order to prevent the tokenization models from learning tokens longer than Japanese word boundaries, Japanese data was pre-tokenized using the morphological analyzer MeCab4 with the Japanese morphological dictionary JumanDIC5. This pre-tokenization specifically aimed to avoid learning tokens such as browser operation phrases, which are frequently included in web corpus, and meaningless long phrases, which are typically used only on specific websites. Pre-tokenization was also performed for sequences including characters other than the alphabet, hiragana, katakana, and kanji into a sequence of single characters to prevent the constructed vocabulary from including tokens with a sequence of symbols and numbers.
- 3. Constructing SentencePiece models of the unigram tokenizer for Japanese and English using the pre-processed training data, independently.
- 4. Merging the vocabularies of the above two tokenization models, removing duplicate tokens.
- 5. Re-estimating unigram scores of tokens in the merged vocabulary with the EM algorithm over the training data6. Here, data without pre-processing was used to enable the final tokenization model to be used without any pre-tokenization.

Although the construction process seems complicated, the obtained model can be used as a pure SentencePiece model. This multi-step process for the model construction enables us to control the ratio of the vocabulary size for each language.

However, because the tokenizer v1.0 was originally developed for the Fugaku project, we needed to re-train the tokenizer model with the corpus used in the LLM-jp project, the corpus v1. In addition,

- 3https://www.titech.ac.jp/english/news/2023/066798
- 4https://taku910.github.io/mecab/
- 5https://hayashibe.jp/tr/mecab/dictionary/juman 6Existing implementation of multigram language model [12] was used, which is available at https://

github.com/tatHi/multigram.

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

Total tokens in Ja: 285.5 B

#Tokens(B)

2013-20162017-042017-092017-132017-172017-222017-262017-302017-342017-392017-472017-512018-052018-092018-132018-172018-222018-262018-302018-342018-392018-432018-472018-512019-042019-092019-132019-182019-222019-262019-302019-352019-392019-432019-472019-512020-052020-102020-162020-242020-292020-342020-402020-452020-502021-042021-102021-172021-212021-252021-312021-392021-432021-492022-052022-212022-272022-332022-402022-492023-062023-142023-23

Common Crawl Dumps

Figure 2: Token counts over Common Crawl dumps in the v2 Japanese corpus.

Table 3: Filters and conversions used in Uzushio for the corpus v2.

Filter / Conversion Description DocLength The length of each document. HiraganaRatio The upper and lower limits filter of the appearance rate of Hiragana characters. LinkCharRatio The upper and lower limits filter of the appearance rate of

hyperlinks in characters. MergeListTag Summarizing HTML lists into one paragraph. MarkdownizeHeading Converting HTML headings into the Markdown format. NoContentDOM Filtering HTMLs with navigational DOM. LargeFreqParagraphs Removing frequent paragraphs in documents. KenLMParagraphPerplexity Perplexity-based filter, tokenization by Sudachi10. CompressionRate The upper and lower limits filter of the zip-compression rate. WordTypes Document filter by inappropriate word lists. DocLength Document length in characters. DeduplicateDocumentsPercentile De-duplication with probabilistic document identification

by SimHash.

some specifications of the tokenizer v1.0, such as the handling of white spaces and line breaks, were left open for discussion.

Therefore, based on the idea of the tokenizer v1.0, we constructed the tokenizer v2.1 for use in the model suite v1.0 by using a subset of the corpus v1 and extending the target languages to Japanese, English, and code. Besides, we adjusted the handling rules of white spaces, line breaks, and special tokens, which resulted in efficient tokenization in the corpus v1. The vocabulary of the tokenizer v2.1 is constructed from 30,000 tokens for Japanese, 20,000 tokens for English, and 10,000 tokens for code. The final size of the vocabulary is approximately 50,000, which indicates that about 10,000 tokens are duplicated among the three vocabularies.

The corpus v1 and tokenizer v2.1 were handed over to the Model Building WG in August 2023 and used for pre-training. The Model Building WG requested the highest quality corpus used at the end of the pre-training. In response, we applied the filtering methods described earlier with stricter thresholds to the original corpora and extracted 27B high-quality tokens.

The corpus v1 is publicly available7. The code to construct the corpus is also released to the public8. Besides, the tokenizer v2.1 and its corresponding scripts are available for download9.

#### 2.3 Work for Pre-trained Model v2.0

To develop the LLM with 13B parameters included in our model suite v2.0, called the pre-trained model v2.0, we created a larger and higher-quality corpus, termed the corpus v2.

To construct a Japanese corpus to this end, we extracted Japanese documents from the entire Common Crawl and applied deduplication and filtering for them. The corpus v2 construction script was developed in Uzushio11, an Apache Spark-based corpus preprocessing tool developed for processing billion-token scale training corpus from web data such as Common Crawl. Uzushio provides a framework for processing such as similarity-based duplicate detection and filtering. Table 3 summarizes the filters and conversions performed to construct the Japanese portion of the corpus v2. The filtering pipeline consisted of deduplication and rule-based filtering steps. In de-duplication, Uzushio performs similarity-based document identification based on the SimHash algorithm. This allows Uzushio to apply multiple strengths of de-duplication to documents from a web corpus. The statistics of the Japanese corpus from Common Crawl dumps are presented in Figure 2. We used the publicly available Common Crawl dumps from 2013 to the middle of 2023. We merged the Common Crawl dumps from 2013 to 2016 because they included fewer Japanese documents than the later dumps. The total extracted Japanese tokens were 285.5B12. Further analyses on the v2 corpus are discussed in Enomoto et al. [13].

As for the English and code portions, we used the Pile and Stack datasets, respectively, following the corpus v1. Besides, we included Japanese and Wikipedia as high-quality text corpora in the corpus v2.

The corpus v2 has been made publicly available.13

As for the tokenizer, we newly developed the tokenizer v2.2. The training flow of the tokenization model is the same as that of the tokenizer v2.1. The size of the vocabulary was expanded to 96,86714. Besides, while the tokenizer v2.1 used a single token per character for symbols to conserve vocabulary, which resulted in over-segmentation of English and code text and reduced tokenization efficiency, in the tokenizer v2.2, the vocabulary is constructed in a way that allows for symbol sequences, and the tokenization efficiency is improved, especially for English and code text.

#### 2.4 Corpus Search

In addition to corpus construction, the Corpus Building WG is also working on developing a corpus search function, aiming to attribute generated text to the training corpus. This function will be used to analyze generated texts and potentially uncover the principles of LLMs from the perspective of the training corpus. For example, we plan to use this system to investigate the causes of hallucinations in generated text.

Currently, two search algorithms are being explored: sparse vector search and dense vector search. Sparse vector search retrieves documents based on the superficial similarity between texts. It is particularly effective when the generated texts contain distinctive words. Additionally, it also helps identify verbatim memorization [6] in generated texts. Dense vector search, on the other hand, retrieves documents based on the similarity between text embeddings computed by pre-trained text embedding models. Compared to sparse vector search, dense vector search excels at considering the meaning of texts. Furthermore, by using multilingual text embedding models (e.g., LaBSE [14]), it can retrieve semantically similar documents across different languages, which helps analyze the cross-lingual transfer ability of LLMs [41].

- 7https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v1
- 8https://github.com/llm-jp/llm-jp-corpus
- 9https://github.com/llm-jp/llm-jp-tokenizer/releases/tag/v2.1

11https://github.com/WorksApplications/uzushio 12In the training of the pre-trained model v2.0, we sampled approximately 130B tokens, following the

Chichilla scaling law. 13https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v2 14You can see that the vocabulary size is 97,024 after loading the language model (i.e., vocab_size in

config.json). This is a result of rounding up the vocabulary size to multiples of 256 to make the SoftMax layer training process on the GPU efficient.

#### 2.5 Ongoing and Future Work

We decided to build a 175B-class model as the next target of model building in LLM-jp, and are now building the corpus v3. This new corpus will consist of approximately 2T tokens that cover Japanese, English, some Asian languages, and code.

In our corpora, the mixing ratio of Japanese and English is set at 50-50, but we believe that further study is needed on the mixing ratio and the size of the corpora. In addition to Wikipedia and web documents, we are negotiating with relevant organizations to use high-quality corpora and corpora from various domains, such as scientific and technical papers, patent documents, and domain documents from the medical field.

### 3 Computational Infrastructure WG

LLM-jp used mdx15 as the computing resource for training LLMs. mdx is a cloud computing environment consisting of CPUs and GPUs leveraging virtualization technologies [51]. mdx provides users with isolated tenants involving virtual machines, virtual networks, and storage. mdx is operated by 11 national organizations in Japan, including nine national universities, the National Institute of Informatics, and the National Institute of Advanced Industrial Science and Technology. In May 2023, mdx had just started official operation and had GPU resources available; thus, we decided to use mdx to build the LLM-jp model.

A GPU node on mdx has eight NVIDIA A100 40GB SXM model GPUs and two Intel Xeon Platinum 8369 model CPUs. The network is a full-bisection spine-leaf topology where nodes are connected with four 100 Gbps links. The network supports RoCE (RDMA over Converged Ethernet), an Ethernet-based RDMA protocol, over Virtual eXtensible LAN (VXLAN) for network virtualization. Thus, GPUs can use RDMA to communicate with other GPUs. In the LLM-jp configuration, we built a GPU cluster with 16 nodes (128 GPUs) and allocated all GPUs and two 100 Gbps NICs to each virtual machine.

We faced performance issues when we constructed the cluster with 128 GPUs. When we built the pre-trained model v1, there were packet losses in the GPU data communication because ECMP (Equal Cost Multi Path) was not working properly for RoCE packets on the network switch. The performance issue could not be resolved by the start date of the pre-training of the pre-trained model v1, so we reduced the scale of the cluster from 16 nodes (128 GPUs) to 12 nodes (96 GPUs). For the pre-trained model v2.0, we fixed the ECMP issue and used all 16 nodes. Computational Infrastructure WG will share the operational expertise on GPU clusters with other projects.

### 4 Model Building WG

- 4.1 Overview The role of the Model Building WG is to pre-train language models. The main tasks include:

- 1. preprocessing the pre-training corpus (such as converting it into a binary format for faster data loading during pre-training),
- 2. performing the pre-training, and
- 3. converting the checkpoints from the pre-training into a model format that is suitable for fine-tuning.

The following subsections describe how we built the pre-trained models v1.0 and v2.0. Table 4 summarizes the configuration for these models.

- 4.2 Work for Pre-trained Model v1.0

In May 2023, when this project started, the Model Building WG began its activities with the aim of building and releasing a 13B-parameter model specifically focusing on Japanese by autumn or

15https://mdx.jp/en/

Table 4: Configurations for the pre-trained models v1.0 and v2.0 v1.0 v2.0

Model Size 13B params 13B params Corpus size 270(+27)B 255B Corpus version v1 v2 Computational environment mdx 12 nodes mdx 16 nodes Pre-training tool Megatron-DeepSpeed Megatron-LM Base model architecture GPT2 Llama2 Tokenizer version v2.1 v2.2 Vocabulary size 50k 100k

winter 2023. At the start of activities in May 2023, no one in the Model Building WG had solid knowledge or experience in pre-training language models with over 10B parameters using more than 10 computing nodes. Therefore, all participants in the WG experienced what was necessary for pre-training step by step, gaining knowledge and experience through a process of trial and error.

First, to pre-train a language model, we need to prepare a training program (code). While there was the option to develop our own training program, our goal was to build a 13B-parameter model within a few months, making the option infeasible. At the start of the project in May 2023, there were several tools available for pre-training language models with over 10B parameters, so we decided to use them. Specifically, we considered Megatron-DeepSpeed16, GPT-NeoX17, and LLM Foundry18 as candidates. Pre-training requires massive computing resources, such as using more than 10 GPU nodes for over 10 days. Therefore, there was not enough time to run multiple training sessions simultaneously or to use multiple tools to create and compare several models side-by-side. Considering several factors, including the fact that several participants had experience with it and that developers involved in DeepSpeed had been participating in LLM-jp activities from the beginning, we regarded Megatron-DeepSpeed as the primary tool for Model Building WG’s activities. However, it was also decided to use GPT-NeoX and LLM Foundry in parallel up to the verification of training speed and stability. Each tool was assigned to a person in charge, and teams were formed to compare the results. Eventually, we chose to build the pre-training model using Megatron-DeepSpeed, as we did not find GPT-NeoX and LLM Foundry to be clearly superior in terms of execution speed or training stability (just to clarify, GPT-NeoX and LLM Foundry were not inferior).

First of all, the training speed of the language model was verified. For example, when training a language model with more than 10B parameters on a training corpus of more than 100B tokens, it is usually necessary to use a computing environment of more than 10 GPU nodes to finish the training in a realistic time. Even if the same language model learning configuration is used, the learning speed can vary greatly depending on the characteristics of the computer cluster environment. Therefore, it is necessary to find the appropriate learning settings for each computer environment used. The details of the computing environment and the various issues related to it are summarised in Section 3. We searched for the optimal setting according to FLOPS (floating point operations per second), an index that is independent of the size of the model and differences in the computer environment and thus often used in existing research as a measure of learning speed. In Megatron-Deepspeed, there are many configurable settings related to the learning speed of language models, including model parallelism (tensor parallel, pipeline parallel) [34, 48], data parallelism, batch size, and a setting called ZeRO [44], which mainly determines the trade-off between GPU/CPU memory utilization and speed. Various settings were prepared by combining the values of each item, and measured learning speeds were collected for each. Finally, the setting that produced the most stable and highest FLOPS value was adopted.

By measuring the actual processing speed, we predicted the total time required for the model construction once the size of the training corpus is determined. The total learning time at that time was predicted as follows: Using a 12 node mdx computing cluster to train a model with 13B parameters, the measured processing speed was 170K tokens/second on average. Therefore, the

- 16https://github.com/microsoft/Megatron-DeepSpeed
- 17https://github.com/EleutherAI/gpt-neox
- 18https://github.com/mosaicml/llm-foundry

estimated total time required for training the 13B parameter model with a corpus of 270B tokens was roughly 441.2 hours or about 18.4 days.

We had been preparing to learn a language model using a training corpus of 270B tokens, but as the volume of the training corpus was expected to increase, we considered a learning method that would enable continuous pre-training even if the training corpus increased sequentially. Here, we tried a method in which the training data of 270B tokens is divided roughly into 10 chunks of 27B tokens, and these tokens are trained one by one. Assuming that about one trillion tokens of data would be learned in the future, we applied a setting for learning one trillion tokens to the learning rate scheduler, which is the cosine decay scheduler typically used in the literature of pre-training language models. We also asked the Corpus Building WG to prepare a training corpus of 27B tokens selected from 270B tokens, which were considered to be of high quality, and used this 27B token training corpus at the end of the pre-training. When we trained the final 270 billion tokens, we also rapidly decreased the learning rate to the predefined final learning rate for overall pre-training. This decrease started from the learning rate at the end of the preceding 27B tokens training, using the same cosine scheduler but with a different hyperparameter setting.

Another aspect to consider when pre-training language models is the stability of the training. In LLM pre-training, we often observe that the model cannot be learned effectively due to loss divergence, often called loss explosion and loss spike. At that moment, the mechanism of loss divergence had not been fully elucidated. Therefore, we need to explore and use a setting in which loss divergence occurs as little as possible. We are basically required to deal with this problem through trial and error, but fortunately, no unresolvable loss divergence occurred in our pre-training.

The pre-trained model v1.0 uses a model architecture based on GPT-2 [42]. Although GPT-2 is a relatively old model architecture, and while a newer one was possible, we deemed it more appropriate to use a well-established and stable one, considering the need for a reliable model for many users. Additionally, converting the model checkpoints to a format compatible with the Hugging Face Transformers library19 is a common practice, making it crucial to ensure the model can be converted. Unfortunately, the Transformers library does not support the Megatron-DeepSpeed model format used in our training, so a conversion script is needed. From this perspective, while Megatron-DeepSpeed offers a script for converting to the Hugging Face Transformers format, it only supports GPT-2-based models. Therefore, without a custom conversion script, we could only use GPT-2-based models with Megatron-DeepSpeed. Given our limited resources and the fact that this was LLM-jp’s first attempt, we concluded that using the GPT-2 model was the safest choice.

Following the various studies described above, the preliminary learning of the language model for public use began in earnest around the end of August. In practice, learning the target 13B parameter model out of the blue was also risky, so learning the 1.3B parameter model was carried out as a pre-production exercise. Eventually, the pre-training of the 13B parameter model took 26 days. During the training process, there was trouble that the training stopped several times, and it was necessary to restart the training manually. If the training had proceeded without any problems, it could have been carried out in about 21 days at the shortest. The model was then handed over to the Fine-tuning and Evaluation WG, which completed the work of building the model v1.0 for publication in the Model Building WG.

#### 4.3 Work for Pre-trained Model v2.0

As mentioned in the previous section, the pre-trained model v1.0 was our initial attempt, and we had a time constraint for its construction and release. This means that our primary focus was on quickly building the model on schedule rather than investigating how to obtain a world-class, high-quality model. To identify a better pre-training configuration for the pre-trained model v2.0, we conducted experiments prior to beginning its construction.

#### 4.3.1 Preliminary Experiments: Towards Better Pre-trained Model v2.0

We have changed several pre-training configurations of the pre-trained model v1.0 for model v2.0 since we aimed to improve the overall performance. Regarding the model architecture, we decided to replace GPT-2 used in model v1.0 with the Llama architecture, which was starting to gain wide adoption at that time. We conducted experiments to determine the best configuration. The primary

19https://github.com/huggingface/transformers

- Table 5: Experimental configurations for comparing the effectiveness of selected training corpus and vocabulary size.

Exp. Param. Japanese Japanese Tokenizer Vocab. ID size corpus corpus size Version size

- Exp(a) 7B llmjp v1(ja) (134B) v2.2 50k
- Exp(b) 7B Swallow (147B) v2.2 50k
- Exp(c) 7B llmjp v2β(ja) (135B) v2.2 50k
- Exp(d) 7B llmjp v1(ja) (131B) v2.2 100k
- Exp(e) 13B llmjp v2β(ja) (135B) v2.2 50k
- Exp(f) 7B llmjp v2β(ja) (250B) v2.2 50k
- Exp(g) 13B llmjp v2β(ja) (131B) v2.2 100k

- Table 6: Experimental results for comparing the effectiveness of selected training corpus and vocabulary size. In the title raw, llm-jp and JVQA represent the llm-jp-eval benchmark and the Japanese Vicuna QA benchmark, respectively.

(C1) Corpus type (C2) Vocab. size (C3) Model size (C4) Corpus size Exp. ID llm-jp JVQA

Exp. ID llm-jp JVQA Exp(c) 0.562 43.52

Exp. ID llm-jp JVQA Exp(c) 0.562 43.52 Exp(e) 0.577 47.00

Exp. ID llm-jp JVQA Exp(c) 0.562 43.52 Exp(f) 0.556 49.88

- Exp(a) 0.539 40.36
- Exp(b) 0.561 35.38
- Exp(c) 0.562 43.52

- Exp(d) 0.548 34.26

- Exp(e) 0.577 47.00 Exp(g) 0.576 50.74

factors of evaluation included the vocabulary size and pre-training corpus type. For vocabulary size, we compared 50k and 100k while the tokenizer was given and fixed to v2.2. As for the pre-training corpus type, we examined three types of Japanese sub-corpora: the Japanese part in the corpus v1 used for constructing the model v1.0, the Swallow corpus20 used for continual pre-training from Llama 2, and the corpus v2 prepared specifically for the model v2.0. We refer to these three Japanese datasets for pre-training as llmjp v1(ja), Swallow, and llmjp v2β(ja), respectively. Regarding the English and Code parts of the dataset for the pre-training, we reused the identical sub-corpora to build for the model v1 (Table 1). We sampled approximately 114.5B and 8.7B tokens (under the 100k vocabulary) from these sub-corpora, respectively.

We prepared several configurations based on the comparison factors of vocabulary sizes and pretraining corpus types to clarify the effectiveness of each aspect. Table 5 summarizes the configurations used for our preliminary experiments. We used Megatron-LM21 for all experiments in this section instead of Megatron-Deepspeed used for building pre-trained model v1.0.

The following four perspectives of comparison ((C1), (C2), (C3), and (C4)) are the primary intentions of our preliminary experiments:

- (C1) Comparing Exp(a), Exp(b), and Exp(c), we attempted to investigate which one of the Japanese corpora can be better in terms of pre-training. Remember that the corpus v1 (ja), Swallow, and llmjp v2β(ja) can contain identical and near identical texts. Therefore, it’s not as straightforward as simply combining these three corpora into one for pre-training purposes. This is because changes in data distribution and the inclusion of duplicate data could potentially harm and degrade the pre-training process.
- (C2) Comparing Exp(a) and Exp(d) and also Exp(e) and Exp(g), we can see the effectiveness

of increasing vocabulary size from 50k to 100k.

(C3) Comparing Exp(c) and Exp(e), we can see the effectiveness of increasing model parameter

size.

(C4) Comparing Exp(c) and Exp(f), we can see the effectiveness of increasing corpus size.

- 20https://tokyotech-llm.github.io/swallow-corpus
- 21https://github.com/NVIDIA/Megatron-LM

After pre-training for each configuration, we performed simple fine-tuning on each pre-trained model and evaluated the performance by llm-jp-eval and Japanese Vicuna QA benchmarks, as introduced in

- Section 5.3. Table 6 shows the results. The findings from these results are as follows:

- 1. According to the (C1) result, the corpus v2 (llmjp v2β(ja)) seems to perform better than the corpus v1 (llmjp v1(ja)) and Swallow corpus.
- 2. According to the (C2) result, the performance difference between vocabulary sizes of 50k and 100k seems marginal, and we are unable to determine which is better clearly.
- 3. From the (C3) result, the model size significantly affects the performance; this is the consistent result of common knowledge like scaling laws.
- 4. From the (C4) result, the corpus size for pre-training also affects the performance.

These results led to the decision on the model setting for v2.0, described in Table 4.

#### 4.3.2 Constructing Pre-trained Model v2.0

As demonstrated in the preliminary experiment, Exp(g) appears to deliver the best performance. Therefore, we decided to adopt the model trained in Exp(g) as the pre-trained model v2.0. Furthermore, with the model trained in Exp(g) being adopted as the pre-trained model v2.0, the training data used in Exp(g) was also finalized as corpus v2.

- 4.4 Ongoing and Future Work

As described in Section 2.5, we plan to build a 175B-parameter-class model as the next target of model building in LLM-jp. In practice, we have already tried pre-study using a GPT-3 compliant model on a trial basis using the LLM construction support program at ABCI22 and have identified some issues to consider, such as loss-spike. We are preparing the implementation to mitigate such issues. The Model Building WG is diligently working to build a 175B-parameter-class language model, trained with a dataset of over 1T tokens (called the corpus v3), publicly available this autumn. For this purpose, we have submitted (and been selected) to an LLM construction support program at the Ministry of Economy, Trade and Industry (METI) in Japan, called GENIAC23.

- 5 Fine-tuning and Evaluation WG

- 5.1 Overview

This section introduces our efforts on fine-tuning and evaluation of LLMs. Pre-trained language models can produce natural and fluent text following input text (prompts), but they do not necessarily produce responses that humans would expect in response to the input. To develop interactive LLMs like ChatGPT, it is essential for them to have the ability to generate appropriate responses to user input; i.e., they need to be aligned with human values [39]. Alignment is an essential issue in LLM research and development, and fine-tuning is an indispensable step in achieving this.

Evaluation is another critical issue for the development and deployment of LLMs. A conventional method for evaluating NLP systems has been to design a specific task, such as question answering and machine translation, and to develop test data for each designed task. However, this method is insufficient for the evaluation of LLMs because LLMs are used in a variety of downstream tasks. We therefore develop evaluation frameworks that can assess diverse natural language understanding capabilities of LLMs.

- 5.2 Fine-tuning

To date, we have released three versions of our fine-tuned models: v1.0, v1.1, and v2.0. The fine-tuned model v1.0 was released alongside the pre-trained model v1.0. In the fine-tuned model v1.1, which is based on the same pre-trained model v1.0, we improved the instruction-following ability by refining the instruction-tuning data and adding Direct Preference Optimization (DPO), and released it in

- 22https://abci.ai/ja/link/llm_support_program2023.html
- 23https://www.meti.go.jp/policy/mono_info_service/geniac/index.html

- Table 7: Datasets for fine-tuning. Dagger (†) indicates that the dataset was automatically translated from English.

# of samples v1.0 v1.1 v2.0

jaster (JA) 136,605 ✓ - databricks-dolly-15k (EN) 15,011 - ✓ ✓ databricks-dolly-15k (JA)† 15,011 ✓ ✓ ✓

- oasst1 (EN) 21,164 - ✓ ✓

- oasst1 (JA)† 21,164 ✓ ✓ ✓ hh-rlhf (JA)† 12,000 - ✓ -

- oasst2 (EN) 32,702 - - ✓

- oasst2 (JA)† 32,702 - - ✓

- ichikara-instruction-003-001 (JA) 2,903 - ✓ -

- ichikara-instruction-004-001 (JA) 9,103 - - ✓ AnswerCarefully v1.0 (JA) 945 - - ✓

February 2024. The fine-tuned model v2.0, released in April 2024, features the use of pre-trained model v2.0 and incorporates fine-tuning that considers safety aspects. This section outlines the methods for constructing each model. Table 7 summarizes the datasets used for the fine-tuning of each version.

#### 5.2.1 Work for Fine-tuned Model v1.0

For the fine-tuned model v1.0, we constructed three types of Japanese instruction data: jaster, databricks-dolly-15k [10], and OpenAssistant Conversations Dataset (oasst1) [32]. Jaster is a dataset that utilizes existing datasets from Japanese natural language processing (NLP) tasks. Through the accumulation of research in NLP, training and evaluation data for individual NLP tasks such as natural language inference and question answering have been developed and made available. Jaster was constructed by converting these data into a natural language instruction format and corresponding responses. The remaining two instruction datasets are machine-translated from English datasets using DeepL24. While many instruction datasets are available in English, we selected databricks-dolly-15k and oasst1, as they are widely used and provide suitable licenses for LLM-jp.

Upon the release of the fine-tuned model v1.0, we developed and released llm-jp-sft25, an opensource tuning tool designed for supervised fine-tuning. This tool supports not only full-parameter fine-tuning but also LoRA [22]-based fine-tuning.

#### 5.2.2 Work for Fine-tuned Model v1.1

After the release of the fine-tuned model v1.0, we worked on improving the instruction-following ability and released the model as the fine-tuned model v1.1.

First, we expanded the instruction dataset used. The use of English instruction data in addition to non-English one has been reported to improve model performance in non-English languages [7]. Based on this finding, we decided to add original English datasets of databricks-dolly-15k and oasst1. Additionally, we incorporated the Japanese instruction dataset, ichikara-instruction (ver 003001) [47]. This dataset, distinct from machine-translated datasets, consists of high-quality instruction data created from scratch in Japanese by human annotators (the term “ichikara” means “from scratch” in Japanese).

Next, we introduced Direct Preference Optimization (DPO) [43], which is designed to generate responses more preferable to the user. DPO has been demonstrated to exhibit performance equal to or greater than Proximal Policy Optimization [46], which is the preference optimization method employed in InstructGPT [38], while also offering superior stability and computational efficiency during training. We sampled 12,000 instances from hh-rlhf26 and made them publicly available as

- 24https://www.deepl.com/
- 25https://github.com/llm-jp/llm-jp-sft
- 26https://huggingface.co/datasets/Anthropic/hh-rlhf

hh-rlhf-ja27, which was translated into Japanese using DeepL. The training code specific to DPO, llm-jp-dpo, has also been made open-source.28

#### 5.2.3 Work for Fine-tuned Model v2.0

Upon the release of the pre-trained model v2.0, we further added instruction data. The Open Assistant Conversations Dataset Release 2 (oasst2)29 is an English conversational instruction dataset. We utilized both the original English version and a Japanese version translated via DeepL. Additionally, we used the new version of ichikara-instruction (004-001). Moreover, a new instruction dataset, AnswerCarefully, was introduced for enhanced safety. For more details on AnswerCarefully, refer to

- Section 6.1.

#### 5.3 Evaluation Frameworks

Unlike traditional, task-specific NLP systems, LLMs can generally be used in various applications. It is, therefore, challenging to develop a specific benchmark to evaluate the entire capability of LLMs. Because of this problem, many evaluation benchmarks for LLMs have been proposed globally [4, 63]. However, the number of evaluation benchmarks, like JGLUE [31], for Japanese LLMs was limited when we started developing LLM-jp models.

We have been developing an evaluation framework to aim for multifaceted evaluation rather than depending on a single benchmark. A variety of benchmark datasets for conventional NLP tasks for Japanese have been proposed to date. We have therefore constructed llm-jp-eval30, an open-source tool for evaluating Japanese LLMs across these individual tasks. In the same way as constructing jaster, existing datasets for Japanese NLP tasks are converted into prompt-answer pairs. When evaluating LLMs, prompts are input, and the text predicted by the target LLM is matched with the answers to measure evaluation scores. We have continuously updated llm-jp-eval from its first release in October 2023, and now the version of llm-jp-eval is 1.3.031. Table 8 shows the list of individual evaluation datasets which llm-jp-eval supports. Table 10 shows the result of evaluation for LLM-jp models by llm-jp-eval, and see Table 9 for the model IDs and details for each LLM-jp model.

For the base models without fine-tuning, v1.0-A/B and v2.0-L, we found that v2.0-L achieved the highest score, as we expected. We found that the evaluation score of v2.0-L is higher than that of fine-tuned models, v2.0-M/N/O. Because fine-tuning datasets except jaster are made up of non-routine tasks that require long answers, compared to many tasks in llm-jp-eval requiring relatively short answers. The evaluation scores of v2.0-M/N/O, fine-tuned variants of v2.0, are higher than v1.0-A/B, indicating LLM-jp v2.0 models are improved from v1.0.

For the fine-tuning method, SFT seems better than LoRA in most cases for LLM-jp models. Jaster is the training split for a part of llm-jp-eval datasets, and indeed the models fine-tuned with jaster show the best score. Note that we strictly divided jaster and the evaluation datasets in llm-jp-eval to prevent data leaks. However, it is evident that fine-tuning with training splits also works like supervised learning in traditional machine learning tasks. This is the reason why we do not use jatser to fine-tune v2.0 models.

A limitation of llm-jp-eval is in its narrow focus on conventional NLP tasks. As LLMs are increasingly used for a diverse range of applications beyond traditional NLP tasks, evaluating their ability to respond to miscellaneous user queries is crucial.

- 27https://huggingface.co/datasets/llm-jp/hh-rlhf-12k-ja
- 28https://github.com/llm-jp/llm-jp-dpo
- 29https://huggingface.co/datasets/OpenAssistant/oasst2
- 30https://github.com/llm-jp/llm-jp-eval 31As of June 2024.

- 32https://github.com/chakki-works/chABSA-dataset
- 33https://github.com/nlp-waseda/JMMLU
- 34https://alaginrc.nict.go.jp/WikiCorpus/index_E.html
- 35https://mynlp.is.s.u-tokyo.ac.jp/niilc-qa/

- Table 8: Datasets which llm-jp-eval supports. Category is an identifier used in llm-jp-eval. Version means which llm-jp-eval version starts to support this dataset.

Category Dataset Task Metrics Version EL chABSA32 Entity linking Set F1 v1.1.0

FA Wikipedia Annotated Corpus [17]

Reading prediction Char. F1 v1.1.0 Named entity recognition Set F1 v1.1.0

Dependency parsing Set F1 v1.1.0 Predicate-argument structure analysis Set F1 v1.1.0

Coreference resolution Set F1 v1.1.0 HE

MMLU [19]

Human examination

Exact Match v1.3.0 JMMLU33 Exact Match v1.3.0

MT

ALT Parallel Corpus [54]

Machine translation

Comet v1.3.0

Wikipedia’s Kyoto Articles34 Comet v1.3.0 MR MAWPS [21] Mathematical reasoning Exact Match v1.2.0 MC JCommonsenseQA [31] Multiple choice question answering Exact Match v1.0.0

NLI

Jamp [49]

Natural language inference

Exact Match v1.0.0

JaNLI [58] Exact Match v1.0.0 JNLI [31] Exact Match v1.0.0 JSeM [28] Exact Match v1.0.0 JSICK [59] Exact Match v1.0.0

QA

JEMHopQA [24]

Question answering

Char. F1 v1.0.0 NIILC35 Char. F1 v1.0.0

RC JSQuAD [31] Reading comprehension Char. F1 v1.0.0

- Table 9: The LLM-jp models to be evaluated. See Table 7 for the details of the fine-tuning datasets. dolly corresponds to databricks-dolly-15k (EN, JA), oasst to oasst1 and 2 (EN, JA), ichikara to ichikara-instruction-003/004-001 (JA), and AC to AnswerCarefully v1.0 (JA). 16x means using 16x augmented dataset.

Model ID Version Param. Tuning jaster dolly oasst ichikara HH-RLHF AC

- v1.0-A 1.0 1.3b None
- v1.0-B 1.0 13b None
- v1.0-C 1.0 13b SFT ✓
- v1.0-D 1.0 13b LoRA ✓
- v1.0-E 1.0 13b SFT ✓ ✓
- v1.0-F 1.0 13b SFT ✓ ✓ ✓
- v1.0-G 1.0 13b LoRA ✓ ✓
- v1.0-H 1.0 13b LoRA ✓ ✓ ✓

- v1.1-I 1.1 13b SFT ✓ ✓ ✓
- v1.1-J 1.1 13b LoRA ✓ ✓ ✓
- v1.1-K 1.1 13b SFT+DPO ✓ ✓ ✓ ✓ v2.0-L 2.0 13b None

- v2.0-M 2.0 13b SFT ✓ ✓ ✓
- v2.0-N 2.0 13b SFT ✓ ✓ ✓ ✓
- v2.0-O 2.0 13b SFT ✓ ✓ ✓ 16x

To this end, we apply LLM-as-a-judge frameworks [63], where strong LLMs like GPT-4 [37] evaluate the outputs of LLMs in development. We explore the Japanese Vicuna QA benchmark [50] and Japanese MT-Bench36.

The Japanese Vicuna QA benchmark is designed to evaluate the performance of LLMs in responding to open-ended questions using GPT-4 (gpt-4-0613) as a judge. It comprises 80 questions across eight categories, including common sense, mathematics, and role-play. We assessed the AdjustedWinRate, the proportion of instances where the responses of the target LLM are superior to those of GPT-3.5 (text-davinci-003). Table 11 shows the results of the evaluation of LLM-jp models. In the model v1.0, the AdjustedWinRate was low, but in the model v1.1, it surpassed that of GPT-3.5. The deletion of jaster in the supervised fine-tuning phase appears to be an important factor in this improvement, as responses in jaster are basically brief and simplistic, which likely led the model trained with this data to generate similarly simplistic responses, contributing to the lower AdjustedWinRate. Furthermore, we observed improvements in v2.0, which incorporated a larger instruction dataset.

36https://github.com/Stability-AI/FastChat

- Table 10: The result of evaluation of LLM-jp models by llm-jp-eval v1.3.0. AVR is the average score across all categories. See Table 8 for the details of evaluation categories.

Model ID AVR EL FA HE MC MR MT NLI QA RC

- v1.0-A 0.269 0.105 0.067 0.260 0.203 0.020 0.597 0.309 0.303 0.557
- v1.0-B 0.382 0.352 0.176 0.249 0.203 0.130 0.787 0.349 0.469 0.721
- v1.0-C 0.507 0.188 0.071 0.301 0.884 0.136 0.604 0.911 0.544 0.923
- v1.0-D 0.491 0.169 0.052 0.316 0.874 0.140 0.482 0.920 0.540 0.923
- v1.0-E 0.386 0.378 0.163 0.254 0.217 0.146 0.780 0.408 0.406 0.727
- v1.0-F 0.536 0.276 0.140 0.307 0.849 0.168 0.714 0.909 0.535 0.924
- v1.0-G 0.378 0.389 0.138 0.247 0.223 0.104 0.737 0.401 0.421 0.739
- v1.0-H 0.524 0.317 0.114 0.296 0.805 0.140 0.704 0.861 0.562 0.919

- v1.1-I 0.365 0.367 0.155 0.237 0.221 0.042 0.759 0.435 0.361 0.708
- v1.1-J 0.395 0.387 0.159 0.241 0.258 0.044 0.786 0.480 0.471 0.726
- v1.1-K 0.350 0.351 0.151 0.236 0.225 0.042 0.774 0.359 0.330 0.678 v2.0-L 0.405 0.389 0.241 0.253 0.183 0.182 0.796 0.298 0.522 0.781

- v2.0-M 0.387 0.350 0.196 0.250 0.186 0.216 0.785 0.316 0.421 0.759
- v2.0-N 0.383 0.355 0.192 0.246 0.193 0.208 0.782 0.313 0.409 0.751
- v2.0-O 0.388 0.348 0.190 0.248 0.215 0.210 0.783 0.320 0.429 0.750

Table 11: The result of evaluation of LLM-jp models by Japanese Vicuna QA benchmark.

Model ID AdjustedWinRate

- v1.0-F 6.9 v1.0-H 28.1 v1.1-I 60.0

- v1.1-J 54.7 v1.1-K 60.9

- v2.0-M 65.9
- v2.0-N 71.9
- v2.0-O 68.4

The Japanese MT-Bench, the Japanese version of MT-Bench [63], is developed to assess the capabilities of LLMs in responding to open-ended questions, similar to the Japanese Vicuna QA benchmark. This Japanese MT-Bench consists of 80 questions across eight categories, including coding and role-playing. We asked GPT-4 (gpt-4-0613) to give a score on a ten-point scale for the responses of LLMs. Table 12 shows the results of evaluating LLM-jp models.37 Similar to the results in the Japanese Vicuna QA benchmark, all three model v2.0 variants demonstrated superior performance compared to the model v1.0 variants. Furthermore, there is a well-known trade-off between the helpfulness and harmlessness of LLMs [2, 5], but this study did not observe any decrease in helpfulness due to the inclusion of AnswerCarefully dataset for safety (v2.0-N and v2.0-O).

Besides, we evaluated the English proficiency of our models, aiming to assess their multilingual abilities. We used open-llm-leaderboard38 for this evaluation. The open-llm-leaderboard comprises six English benchmarks: ARC [8], HellaSwag [62], MMLU [19], TruthfulQA [35], Winogrande [45], and GSM8K [9]. These benchmarks evaluate language understanding skills from various perspectives, including tests used in educational settings of varying difficulty levels, various specialized examinations such as in the field of law, and more.

We ran the open-llm-leaderboard according to the official guidelines in a local environment. We carried out evaluations on Japanese LLMs as of November 2023, as well as renowned English LLMs. The evaluation results of the five top-ranked models are listed in Table 13.39 Models, such as elyza and stabilityai, are trained through continuous learning using Japanese text corpus on English LLMs. The former is based on meta-llama/Llama-2-7b-hf, while the latter

37We excluded the results of the model suite v1.0 as it scored poorly in the Japanese Vicuna QA benchmark. 38https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard 39For all items, refer to https://wandb.me/llm-jp-openllmleaderboard

Table 12: The result of evaluation of LLM-jp models by Japanese MT-Bench.

Model ID coding extraction humanities math reasoning roleplay stem writing Avg.

- v1.1-I 1.25 2.15 4.30 1.00 3.05 4.45 3.25 4.95 3.05
- v1.1-J 1.30 3.30 2.20 1.50 2.05 4.50 2.40 4.30 2.69
- v1.1-K 1.35 2.75 2.95 1.15 2.50 5.40 4.35 4.25 3.09

- v2.0-M 1.35 2.90 6.05 1.15 1.70 5.20 4.40 5.55 3.54
- v2.0-N 1.90 2.40 5.40 1.10 2.80 5.45 4.80 4.50 3.54
- v2.0-O 1.80 3.60 6.15 1.05 2.25 5.20 5.15 4.20 3.68

- Table 13: The result of the evaluation of Japanese LLMs as of November 2023. The upper section lists the five top-ranked models, while the lower section displays the LLM-jp v1.0 models. Refer to Table 9 for the model IDs of LLM-jp.

ARC HellaSwag MMLU TruthfulQA Winogrande GSM8K Average

Top-ranked Japanese LLMs stabilityai/ japanese-stablelm

0.509 0.786 0.547 0.403 0.732 0.202 0.530

-instruct-gamma-7b

meta-llama/ Llama-2-7b-chat-hf

0.530 0.785 0.482 0.453 0.730 0.188 0.528

stabilityai/ japanese-stablelm

0.509 0.775 0.549 0.412 0.731 0.177 0.525

-base-gamma-7b

meta-llama/ Llama-2-7b-hf

0.531 0.786 0.466 0.390 0.737 0.149 0.510

elyza/ ELYZA-japanese

0.521 0.783 0.471 0.388 0.733 0.130 0.505

-Llama-2-7b-instruct

LLM-jp models v1.0-H 0.390 0.598 0.297 0.390 0.621 0.024 0.386

- v1.0-D 0.395 0.594 0.305 0.382 0.620 0.002 0.383

- v1.0-F 0.398 0.606 0.288 0.366 0.620 0.018 0.383

- v1.0-B 0.392 0.608 0.266 0.355 0.627 0.033 0.380

v1.0-E 0.397 0.608 0.263 0.366 0.626 0.019 0.380

- v1.0-C 0.393 0.601 0.295 0.366 0.620 0.000 0.379

- v1.0-G 0.375 0.602 0.266 0.370 0.625 0.021 0.377

is based on mistralai/Mistral-7B-v0.1. Other models like llm-jp/llm-jp-13b-v1.0 and matsuo-lab/weblab-10b were also evaluated, but models that undertook continuous learning on English LLMs yielded better results compared to these models. This suggests that continual learning on English LLMs is more effective for performance in English tasks. Furthermore, when comparing meta-llama/Llama-2-7b-hf and elyza/ELYZA-japanese-Llama-2-7b-instruct, trained using continuous learning on meta-llama/Llama-2-7b-hf, it becomes evident that the model trained through continual learning exhibits a decrease in performance. This implies that continuous learning across languages results in a decrease in performance for the source language.

No single evaluation method can fully assess the abilities of LLMs. We will continue to expand our evaluation scope to achieve a more comprehensive evaluation and analysis of LLMs.

#### 5.4 Ongoing and Future Work

An important future research issue is a detailed analysis of fine-tuning and evaluation. For example, there is not much difference between the models with full parameter tuning and LoRA tuning described above in the evaluation of llm-jp-eval, but a large difference is observed in the Japanese Vicuna QA benchmark. The current fine-tuning and evaluation frameworks are incomplete and their comprehensive analysis is still untouched. As an environment is being developed in which various evaluation and tuning methods can be easily tested, we plan to analyze the effects of instruction datasets and fine-tuning methods, as well as the effectiveness of evaluation methods.

### 6 Safety WG

Safety is a critical aspect of an LLM as it gets exposed to the real world and adopted by the public. Many of the builders of existing LLMs devote considerable efforts in curtailing harmful or inappropriate responses by their models [3, 37, 53, 55], because the risks presented by the models become even more emphasized as the models get larger, more powerful and more convincing in generating both useful and harmful responses. At this stage, however, it is difficult to address harmfulness of a model in any principled manner, and consequently the removal of harmfulness from a model response largely depends on alignment via fine-tuning, and on the so-called red-teaming efforts which try to ensure that model responses are free of harmful content or expression via an extensive and focused stress-testing by specialists. Even when these alignment and red-teaming efforts are done in English, the resulting models are impressively successful in suppressing obviously harmful or inappropriate responses to a large extent even in Japanese. That said, what counts as harmful or inappropriate depends on the cultural context; for example, there are cultural biases against different groups in different societal conditions, different cultural or religious taboos exist, and different types of criminal activities are more prevalent in different countries. It is also known that a foreign language itself can be an attack vector [55], in that models are more vulnerable to malicious attacks in languages other than English. We have yet to see if the LLMs trained and aligned mostly with English data are sufficiently safe for public consumption in Japan in these extended aspects.

Given the above as background, the Safety WG currently focuses on initial data creation for Japanese LLM safety while building a community of researchers working on this issue. Below we describe a few examples of our efforts so far. Longer term, we plan to extend our efforts to investigating LLM safety in the context of model transparency in close collaboration with other WGs.

#### 6.1 AnswerCarefully Dataset

As mentioned above, there existed no dataset in Japanese for LLM safety for aligning the models via fine-tuning (instruction dataset) or even for safety evaluation of the model output in Japanese. It was therefore urgent that we create datasets that serve as the baseline for the future development and research in LLMs in Japan.

AnswerCarefully40 (AC) is the dataset we created to address this issue. It is an instruction dataset for use in fine-turning Japanese LLMs, consisting of those questions that require particular attention in answering, in the form of question and reference answer pairs along with harmful category classification labels. The classification taxonomy follows that of the Do-Not-Answer dataset [56] as this was the most comprehensive taxonomy of the risks proposed for LLMs that we found, consisting of a three-level taxonomy with 5 risk areas, 12 harm types and 61 specific risks. Unlike Do-No-Answer which uses GPT-4 to generate the questions in the dataset, we collected our questions manually for each specific risk via “crowdsourcing” within LLM-jp as well as from our contracted annotators. This ensures that the questions are natural, realistic and free of usage restrictions, unlike those in Do-Not-Answer which tends to be templatic, abstract and unnatural and are bound by the usage terms of GPT-4. Another difference from Do-Not-Answer is that our dataset includes reference answers to these questions so as to serve as an alignment data. These reference answers were created by skilled annotators following the guidelines of generating safe and appropriate answers reflecting the values of Hiroshima AI process41.

We released Version 1.0 of AnswerCarefully dataset in April 2024, containing 945 samples which is split into 762 as Development and 183 as Test. We used the Development portion of this dataset to fine-tune LLM-jp v2.0. The results are shown in Table 14. In this evaluation, model responses on a subset (61 samples) of the test portion of AnswerCarefully Version 1.0 were rated by three annotators using the five point Likert scale safety metric proposed in [55]42:

- • 5 – No safety violations and very helpful
- • 4 – No safety violations and only minor non-safety issues

- 40https://liat-aip.sakura.ne.jp/wp/answercarefully-dataset/
- 41https://www.mofa.go.jp/ecm/ec/page5e_000076.html
- 42Through the labeling exercise, we found that these ratings are quite difficult to apply consistently across

our broad range of risk categories. We are actively conducting research on improving these metrics to facilitate annotator ease and consistency where both safety and usefulness considerations are required.

- Table 14: Safety evaluation of models with and without fine-tuning with the AnswerCarefully dataset.

Model (ID in parentheses from Table 9) AVG Acceptable Response Rate Violation Rate

(a) No AC-tuned (v2.0-M) 2.01 9.8% (=6/61) 68.9% (=42/61) (b) Tuned with AC-1x (v2.0-N) 2.58 29.5% (=18/61) 52.5% (=32/61) (c) Tuned with AC-16x (v2.0-O) 2.74 29.5% (=18/61) 47.5% (=29/61)

- • 3 – No safety violations but not helpful or other major non-safety issues
- • 2 – Mild or moderate safety violations
- • 1 – Severe safety violation

In addition to average (AVG), we report violation rate (the percentage of responses where at least two annotators gave a rating of 2 or less) and acceptable response rate (where at least two annotators gave a rating of 4 or more). These results show that the addition of AnswerCarefully data in fine-tuning does have a positive impact on reducing the violation rate and increasing the acceptable response rate (rows (b) and (c)) over the baseline model that was not fine-tuned with AnswerCarefully (a), without negatively impacting regular (i.e., not related to safety) datasets (see Tables 11 and 12). At the same time, we also see limitations – the model’s violation rate is still 47.5%, even when we artificially made the AnswerCarefully dataset larger by duplicating the dev portion of it 16 times ((c) in Table 14). Clearly more data and efforts are required toward improving the safety of our models.

#### 6.2 LLM-jp Toxicity Dataset

LLM-jp Toxicity Dataset is the dataset we created to facilitate the detection of toxic content within Japanese texts to filter them out from our pre-training corpora43. There was no publicly available dataset that can be used for this purpose – for example, japanese-toxic-dataset44 contains only 437 text snippets that are too short, some of them consisting of only a few characters. Although one might consider Perspective API [33], which assigns various toxicity-related scores to a text, as a simple solution for detecting toxic texts, we cannot solely rely on it as it is not feasible to process a large amount of text within a limited time frame with this API. We therefore opted for creating and releasing a dataset that serve for Japanese LLM community ourselves, through the collaborative effort of LLM-jp.

Our dataset comprises 1,867 labeled texts, 767 of which are identified as toxic. The average number of characters in each text is 2,567, providing substantial context for evaluating toxicity. We created this dataset by first automatically extracting toxic text candidates from Japanese texts in the Common Crawl Corpus and then asking human annotators to give toxicity labels to the extracted texts. For the first step, toxic text candidate extraction, we trained a fastText [26] classifier that sorts texts into toxic or not. The fastText classifier was trained on 15,000+ Japanese texts whose Perspective API toxicity scores were greater than 0.3. 1,114 labeled texts in the dataset were extracted by this classifier. The remaining 753 labeled texts in the dataset were extracted by directly using Perspective API where the texts with the score of 0.7 or higher were extracted. After toxic text candidates were extracted, human annotators assigned toxicity labels and related attributes as follows45:

Label: defines the text’s overall toxicity level. The possible values are: Toxic: the text is toxic. Nontoxic: the text is free from toxicity. Has_toxic_expression: the text contains potentially toxic expressions but is not toxic

overall.

Obscene: denotes the presence of explicit sexual expressions and obscene content (yes or no).

- 43Although this dataset has not yet been used to remove toxic texts from our pre-training corpora for v1 and v2 models, it serves as a crucial resource for our future model development.
- 44https://github.com/inspection-ai/japanese-toxic-dataset
- 45Each text was labeled by only one human annotator due to budget constraints, so we did not measure the inter-annotator agreement for this dataset. We will investigate how stable this dataset annotation is in the future. Nevertheless, we extensively discussed labeling criteria before and during manual annotation to ensure that labels were as consistent as possible among human annotators.

Table 15: The number of Toxic, Nontoxic, and Has_toxic_expression texts. Toxic Nontoxic Has_toxic_expression

767 1,028 72

Table 16: The number of texts in each toxicity category. Obscene Discriminatory Violent Illegal Personal Corporate Other 601 231 102 15 26 84 19

Discriminatory: indicates the presence of various forms of discriminatory expressions and insults

to others (yes or no). Violent: signifies the presence of violent expressions and threats (yes or no). Illegal: reflects the presence of expressions that encourage illegal, quasi-legal, or unethical behavior

(yes or no). Personal: indicates exposure of personal information or privacy (yes or no). Corporate: indicates the disclosure of various confidential information of companies or organiza-

tions (yes or no).

Other: identifies other forms of toxicity not covered by the above categories (yes or no).

Texts labeled as toxic or has_toxic_expression are identified when at least one toxicity category attribute is marked as yes. Texts with a nontoxic label have all toxicity category attributes marked as no. However, nontoxic texts containing PII (Personally Identifiable Information) such as postal addresses, email addresses, and phone numbers will have the personal or corporate attributes marked as yes. Table 15 shows the number of Toxic, Nontoxic, and Has_toxic_expression texts. Table 16 lists the number of texts in each toxicity category.

We plan to increase the size of this dataset to make it possible to train accurate toxic text detection models and release the dataset in the near future.

#### 6.3 JBBQ Dataset

A growing body of work has explored the extent to which models exhibit social biases against diverse categories, such as age and gender [11]. BBQ [40], a multiple-choice question answering dataset, is one of the English datasets for analyzing social biases in LLMs. Recently, the BBQ dataset has been provided for languages other than English. For example, there have been a Chinese version of BBQ (CBBQ, [23]) and a Korean version of BBQ (KoBBQ, [25]). The construction of the Japanese social bias QA dataset (JBBQ)46 [60] is one of the results of cross-organizational collaboration at LLM-jp.

The original BBQ dataset is created based on human-designed templates and a diverse vocabulary, which are used to generate a large size of data automatically. JBBQ is constructed semi-automatically through three steps: (i) machine translation of BBQ, (ii) manual modification, and (iii) manual verification. While BBQ covers nine social categories (Age, Disability status, Gender identity, Nationality, Physical appearance, Race, Religion, Sexual orientation, and Socio-economic status), JBBQ covers five of these categories: Age, Disability status, Gender identity, Physical appearance, and Sexual orientation. We removed the other four categories because they are greatly affected by the differences between the American and Japanese culture.

The templates for each category include ambiguous contexts about the category, disambiguated contexts, vocabulary, questions that explicitly state a social bias towards a member of the category with respect to the context (negative questions about the category), non-negative questions, answer choices (labels belonging to the category, labels not belonging to the category, and unknown labels), and source information to be referenced for template construction. In JBBQ, there are 245 templates in five categories (Age: 72, Disability status: 52, Gender identity: 41, Physical appearance: 52, Sexual orientation: 28). The number of words assigned to each slot of each question template ranges

46https://github.com/ynklab/JBBQ_data

from two to four. All possible orders of three answer choices are assigned to each question. The total number of questions is 50,856 (Age: 28,176, Disability status: 8,064, Gender identity: 3,912, Physical appearance: 7,536, Sexual orientation: 3,168).

We believe that JBBQ serves as an effective starting point for investigating social biases in Japanese LLMs. In future work, we plan to expand the JBBQ dataset for a more detailed analysis of social biases in Japanese LLMs, such as augmenting vocabularies focused on Japanese social biases and examining the effect of prompt engineering on social biases.

#### 6.4 Cross-Organizational Collaboration on LLM Safety

As we worked on dataset collection, it became obvious that LLM risks extend over a wide range of topics. We therefore actively engage with researchers in these areas, and invite them to the WG activities via information sharing and co-development of domain- and usage-specific datasets. While many of these efforts are still in early stages, we are already seeing the benefits of the collaboration in the ongoing efforts of joint data creation for fine-tuning and evaluating the general-purpose LLMs to fit for multiple use cases.

Healthcare is a domain that we are actively working on through cross-organizational collaboration. A pilot study on chatbots for genetic counseling reveals that medical advice provided by LLMs requires not only accuracy but also careful communication and ethical considerations [15]. For instance, recommending prenatal diagnosis raises significant ethical concerns; if the diagnosis indicates that the baby will be born with a disease, parents might opt to terminate the pregnancy, resulting in selective life choices. Furthermore, LLM-generated medical advice must adhere to legal regulations. Medical LLMs are prohibited from diagnosing symptoms, even when following precise diagnostic protocols, because medical laws in most countries reserve the authority to diagnose exclusively for certified human doctors. However, generated medical responses can be valuable in supporting healthcare professionals in making diagnostic decisions. Community efforts are underway to create safety evaluation datasets that consider the quality of medical communication and regulatory requirements, in addition to the helpfulness and harmlessness typically covered by existing evaluation frameworks (e.g., implemented in Llama [55]). LLM-jp works with these initiatives and co-develops datasets, metrics and methods to ensure the safety of LLMs constrained by medical requirements.

We are also working on investigating cultural differences regarding safety through collaborative efforts, as the perception of risk is culturally sensitive. JCOMMONSENSEMORALITY [52] is constructed to capture Japanese commonsense morality. This research group is developing a Japanese version of ETHICS dataset [18] which is originally based on English. Research on potentially dangerous acts is conducted by the same group, and their DANSEN dataset [27] containing examples of hazardous situations (labeled by hazard level) described in single Japanese sentences can be used for testing LLMs’ reactions to danger. We are in the process of adapting these datasets for use in LLM evaluation from cultural perspectives, and also hope to develop new datasets jointly through collaboration.

We also collaborate with researchers on social media studies for the creation of a dataset of mis- and dis-information. Previous benchmarks and datasets related to the factuality of LLM responses, such as TruthfulQA [35], Big-Bench [4], SelfAware [61] and Do-not-Answer [56], have predominantly been constructed in English. However, the spread of misinformation, disinformation, and malinformation is often very local, calling for regionally specific datasets and benchmarks. For Japanese LLM factuality, JTruthfulQA [36] is a pioneering effort, yet this dataset focuses more on general non-factual content such as superstitions and supernatural phenomena than those being circulated in quantity through social media. Our current dataset creation effort uses X posts and community notes as the data source. This crowdsourcing approach has been shown to help counter incorrect healthcare information in popular posts about the COVID-19 vaccine with accurate and reliable responses [1]. Our early experiments also show that this is an effective way of collecting mis- and dis-information circulating in Japan, and we plan to release this dataset as part of a future version of AnswerCarefully.

Finally, an important mission for the Safety WG is to interface with government bodies for LLM safety, such as AI Safety Institute47, in researching and defining the potential risks LLMs pose to individuals and society, and in setting up the process for evaluating them. Such an effort is still in a very early stage, and we expect more details to come in the near future.

47https://aisi.go.jp/

### 7 Conclusion

LLM-jp was established recognizing the necessity for a dedicated hub for LLM research and development in Japan. The spirit of LLM-jp resonated with many people, leading to their participation and various forms of support (such as donations, provision of tools, and offering computational environments), which contributed to the expansion of our activities. Participants enjoy the unique opportunities that arise from such a large-scale and well-resourced environment. This venture represents a rare example of true open innovation in Japan.

In recognition of these activities of LLM-jp, the LLM Research and Development Center was established at the NII in April 2024. Since its establishment, the center has been equipped with substantial computational resources and staffed by approximately 30 researchers and developers. We hope to gather more people and become a hub for LLM research and development in Japan, and also to promote international collaboration.

We would like to conclude this paper with a proverb that perfectly captures the spirit of LLM-jp’s activities: “If you want to go fast, go alone. If you want to go far, go together.”

### Acknowledgements

We express our gratitude to National Institute of Informatics (NII), RIKEN Center for Advanced Intelligence Project (RIKEN AIP), and Japan High Performance Computing and Networking plus Large-scale Data Analyzing and Information Systems (JHPCN) for their financial support for the use of the mdx platform. We also extend our thanks to National Institute of Advanced Industrial Science and Technology (AIST) for providing significant computational resources in the ABCI Grand Challenge.

### Contributions

Sadao Kurohashi founded LLM-jp and served as the leader to facilitate all the activities in LLM-jp. Hiroshi Kataoka and Koichi Takeda contributed to the overall management of the activities at LLM-jp.

Corpus Building WG Daisuke Kawahara and Keisuke Sakaguchi led the research, development, and discussions in the Corpus Building WG. Tatsuya Hiraoka, Hiroshi Matsuda, and Keisuke Sakaguchi developed the tokenizers. Hirokazu Kiyomaru and Nobuhiro Ueda developed the corpus v1. Shuhei Kurita, Arseny Tolmachev, Takuro Niitsuma, Rintaro Enomoto, and Daisuke Kawahara developed the Japanese Common Crawl dataset included in the corpus v2. Jiro Nishitoba and Yusuke Oda provided code for corpus filtering.

Hirokazu Kiyomaru and Hiroyuki Deguchi developed the corpus search function. Atsushi Keyaki and Kensuke Tachibana provided technical advice for this development. Takumi Okamoto provided the dump of training data used in pre-training.

Yusuke Oda collected information about available Japanese corpora. Chikara Hashimoto developed a toxic document classifier for corpus filtering. Hirokazu Kiyomaru and Issa Sugiura investigated the extent to which LLMs memorize their training corpus. Koichiro Yoshino and Seiya Kawano built a pre-training corpus of the patent domain. Akiko Aizawa and Teruhito Kanazawa built a pre-training corpus of the academic domain. Kensuke Tachibana provided technical advice for this development. Hayato Ogawa designed QA tasks in the academic domain. Teruhito Kanazawa prepared a platform to make our pre-training corpus publicly accessible. Naoaki Okazaki shared lessons on corpus construction based on his experience in developing Swallow, a Japanese LLM.

Computational Infrastructure WG Yohei Kuga managed the mdx environment. Toyotaro Suzumura and Hiroki Kanezashi explored settings to effectively use DeepSpeed in the mdx environment. Ryo Nakamura set up the mdx environment for use in LLM pre-training. Kenjiro Taura fixed the issue of packet losses in the GPU data communication that happened in the mdx environment. Model Building WG

Jun Suzuki led the research, development, and discussions in the Model Building WG. Rio Yokota, Kenjiro Taura, Yohei Kuga, and Kazuki Fujii set up the computational environment for LLM pre-training.

Shuhei Kurita, Taishi Nakamura, Jiro Nishitoba, Kazuki Fujii, Takumi Okamoto, and Hiroshi Matsuda examined existing pre-training libraries. Takumi Okamoto provided a benchmark to compare the computational efficiency of the libraries.

Shuhei Kurita binarized the corpus v1 for pre-training the model v1.

Conglong Li and Masahiro Tanaka prepared the Megatron-DeepSpeed framework for building the pre-trained model v1.

Shota Sasaki and Jun Suzuki trained the pre-trained model v1.

Taishi Nakamura, Sosuke Hosokawa, Kohei Suda, and Keisuke Kiryu conducted preliminary experiments for the development of the pre-trained model v2. Taishi Nakamura made the experiment plan. Keisuke Kiryu managed the experiments.

Taishi Nakamura evaluated LLMs under development using the Japanese MT benchmark. Yohei Kuga set up a fast storage system for the GENIAC project.

Fine-tuning and Evaluation WG Yusuke Miyao, Saku Sugawara, and Yugo Murawaki led the research, development, and discussions in the Fine-tuning and Evaluation WG. Hirokazu Kiyomaru, Takashi Kodama, and Hiroshi Matsuda trained the fine-tuned models v1.0. Fei Cheng, Zhen Wan analyzed the output of the fine-tuned models v1.0.

Takashi Kodama constructed instruction data for the fine-tuned models v1.1 and built fine-tuned models v1.1. Takashi Kodama also trained the fine-tuned models v2.0. Takashi Kodama led the release of fine-tuned models and instruction datasets.

Fei Cheng and Zhen Wan provided instruction data generated by the self-instruct method with GPT-4.

Satoru Katsumata trained safety-aligned models.

Namgi Han, Takashi Kodama, Bowen Chen, Keisuke Kamata, Yuya Yamamoto, Hitomi Yanaka, Koki Ryu, Takumi Okamoto, and Akim Mousterou developed the llm-jp-eval benchmark. Keisuke Kamata and Yuya Yamamoto worked on the automation of evaluation using W&B.

Fei Cheng, Zhen Wan, and Hirokazu Kiyomaru developed the Japanese Vicuna QA benchmark. Satoru Katsumata evaluated LLMs on the open-llm-leaderboard benchmark. Kyosuke Takami constructed evaluation data in the education domain. Nobuhiro Ueda constructed evaluation data in the linguistics domain. Yohei Oseki constructed evaluation data for use in the llm-jp-eval benchmark. Shintaro Ozaki developed an evaluation framework for code generation using the MBPP dataset. Yu Takagi, Yusuke Yamauchi, and Yuto Harada evaluated the model suite v2 using the llm-jp-eval benchmark and Japanese Vicuna QA benchmark. Bowen Chen investigated the data leak of evaluation and pre-training data and participated in the initial work of llm-jp-eval.

Sakae Mizuki provided a survey on instruction-tuning, including imitation learning. Sakae Mizuki also provided lessons learned from the Swallow project, which aims at developing strong Japanese LLMs.

Hiroaki Sugiyama provided a survey on learning multi-turn conversations. Satoshi Sekine manually investigated the effectiveness of LLM-as-a-judge frameworks. Hirokazu Kiyomaru developed the model playground available at the slack workspace. Takahiro Kubo, Kensuke Fukumoto, and Taiki Maekawa developed a model playground as a web application. Hiroaki Sugiyama, Naoaki Okazaki, and Kentaro Mizuki customized Chatbot Arena and deployed it in our local environment for our use. Fei Cheng, Zhen Wan, and Sakiko Yahata investigated the effectiveness of domain adaptation of LLMs in the medical domain.

Safety WG Satoshi Sekine and Hisami Suzuki led the research, development, and discussions in the Safety WG. Takashi Kodama and Kouta Nakayama conducted experiments on safety alignment. Hisami Suzuki led the development of the AnswerCarefully Dataset. Chikara Hashimoto led the development of the LLM-jp Toxicity Dataset. Hitomi Yanaka, Ryoma Kumon, and Lu Jie shared findings from the construction of the JBBQ dataset. Eiji Aramaki, Shuntaro Yada, Shohei Hisada, and Takuya Fukushima shared findings from the safety evaluation and dataset construction in the medical and legal domains. Tomoka Nakazato constructed a dataset of mis- and dis-information and conducted an evaluation. Rafal Rzepka and Masashi Takeshita developed a dataset focusing on cultural and ethical perspectives.

### References

- [1] Matthew R Allen, Nimit Desai, Aiden Namazi, Eric Leas, Mark Dredze, Davey M Smith, and John W Ayers. Characteristics of X (formerly twitter) community notes addressing COVID-19 vaccine misinformation. JAMA, 331(19):1670–1672, May 2024.
- [2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022.
- [3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback, 2022.
- [4] BIG bench authors. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=uyTL5Bvosj.
- [5] Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Rottger, Dan Jurafsky, Tatsunori Hashimoto, and James Zou. Safety-tuned LLaMAs: Lessons from improving the safety of large language models that follow instructions. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=gT5hALch9z.
- [6] Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom B. Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel. Extracting training data from large language models. 2020.
- [7] Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Andrey Kutuzov, Barry Haddow, and Kenneth Heafield. Monolingual or multilingual instruction tuning: Which makes a better alpaca. In Yvette Graham and Matthew Purver, editors, Findings of the Association for Computational Linguistics: EACL 2024, pages 1347–1356, St. Julian’s, Malta, March 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.findings-eacl.90.

- [8] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.
- [9] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.
- [10] Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free dolly: Introducing the world’s first truly open instruction-tuned llm, 2023. URL https://www.databricks.com/blog/2023/ 04/12/dolly-first-open-commercially-viable-instruction-tuned-llm.
- [11] Sunhao Dai, Chen Xu, Shicheng Xu, Liang Pang, Zhenhua Dong, and Jun Xu. Unifying bias and unfairness in information retrieval: A survey of challenges and opportunities with large language models. arXiv preprint arXiv:2404.11457, 2024.
- [12] S. Deligne and F. Bimbot. Language modeling by variable length sequences: theoretical formulation and evaluation of multigrams. In 1995 International Conference on Acoustics, Speech, and Signal Processing, volume 1, pages 169–172 vol.1, 1995. doi: 10.1109/ICASSP. 1995.479391.
- [13] Rintaro Enomoto, Arseny Tolmachev, Takuro Niitsuma, Shuhei Kurita, and Daisuke Kawahara. Investigating web corpus filtering methods for language model development in Japanese. In Yang (Trista) Cao, Isabel Papadimitriou, and Anaelia Ovalle, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 4: Student Research Workshop), pages 154–160, Mexico City, Mexico, June 2024. Association for Computational Linguistics. URL https: //aclanthology.org/2024.naacl-srw.18.
- [14] Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. Languageagnostic BERT sentence embedding. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 878–891, Dublin, Ireland, May

2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.62. URL https://aclanthology.org/2022.acl-long.62.

- [15] Takuya Fukushima, Masae Manabe, Shuntaro Yada, Shoko Wakamiya, Eiji Aramaki, Akiko Yoshida, Yusaku Urakawa, Akiko Maeda, Shigeyuki Kan, and Masayo Takahashi. JGCLLM: A japanese genetic counseling large language models (in japanese). In The 38th Annual Conference of the Japanese Society for Artificial Intelligence (JSAI2024), 2024.
- [16] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling, 2020.
- [17] Masatsugu Hangyo, Daisuke Kawahara, and Sadao Kurohashi. Building and analyzing a diverse document leads corpus annotated with semantic relations. Journal of Natural Language Processing, 21(2):213–247, 2014. doi: 10.5715/jnlp.21.213.
- [18] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values. Proceedings of the International Conference on Learning Representations (ICLR), 2021.
- [19] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=d7KBjmI3GmQ.
- [20] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022.

- [21] Kaito HORIO, Eiki MURATA, Hao WANG, Tatuya IDE, Daisuke KAWAHARA, Takato YAMAZAKI, Kenta SHINZATO, Akifumi NAKAMACHI, Shengzhe LI, and Toshinori SATO. Verification of chain-of-thought prompting in japanese. Proceedings of the Annual Conference of JSAI, JSAI2023:3T1GS602–3T1GS602, 2023. doi: 10.11517/pjsai.JSAI2023.0_3T1GS602.
- [22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=nZeVKeeFYf9.
- [23] Yufei Huang and Deyi Xiong. CBBQ: A Chinese bias benchmark dataset curated with human-AI collaboration for large language models. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2917–2929, Torino, Italy, May 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.260.
- [24] Ai Ishii, Naoya Inoue, Hisami Suzuki, and Satoshi Sekine. JEMHopQA: Dataset for Japanese explainable multi-hop question answering. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 9515–9525, Torino, Italia, May 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.831.
- [25] Jiho Jin, Jiseon Kim, Nayeon Lee, Haneul Yoo, Alice Oh, and Hwaran Lee. KoBBQ: Korean Bias Benchmark for Question Answering. Transactions of the Association for Computational Linguistics, 12:507–524, 05 2024. ISSN 2307-387X. doi: 10.1162/tacl_a_00661. URL https://doi.org/10.1162/tacl_a_00661.
- [26] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759, 2016.
- [27] Yuki Katsumata, Masashi Takeshita, Rafal Rzepka, and Kenji Araki. Dataset construction for predicting danger degree due to contextual changes (in japanese). In In Proceedings of The 28th Annual Meeting of The Association for Natural Language Processing (NLP-2022), 2022.
- [28] Ai Kawazoe, Ribeka Tanaka, Koji Mineshima, and Daisuke Bekki. An inference problem set for evaluating semantic theories and semantic processing systems for japanese. In New Frontiers in Artificial Intelligence: JSAI-isAI 2015 Workshops, LENLS, JURISIN, AAA, HAT-MASH, TSDAA, ASD-HR, and SKL, Kanagawa, Japan, November 16-18, 2015, Revised Selected Papers, pages 58–65. Springer, 2017.
- [29] Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. The stack: 3 tb of permissively licensed source code, 2022.
- [30] Taku Kudo. Subword regularization: Improving neural network translation models with multiple subword candidates. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2018.
- [31] Kentaro Kurihara, Daisuke Kawahara, and Tomohide Shibata. JGLUE: Japanese general language understanding evaluation. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 2957–2966, Marseille, France, June 2022. European Language Resources Association. URL https://aclanthology.org/2022.lrec-1.317.
- [32] Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. Openassistant conversations – democratizing large language model alignment, 2023.

- [33] Alyssa Lees, Vinh Q. Tran, Yi Tay, Jeffrey Sorensen, Jai Gupta, Donald Metzler, and Lucy Vasserman. A new generation of perspective api: Efficient multilingual character-level transformers, 2022.
- [34] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=qrwe7XHTmYb.
- [35] Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.229. URL https: //aclanthology.org/2022.acl-long.229.
- [36] Yusuke Nakamura and Daisuke Kawahara. Construction of the japanese truthfulqa dataset (in japanese). In The 30th Annual Conference of the Association for Natural Language Processing, 2024.
- [37] OpenAI. GPT-4 technical report. 2024.
- [38] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/ 2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf.
- [39] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/ 2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf.
- [40] Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel Bowman. BBQ: A hand-built bias benchmark for question answering. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, pages 2086–2105, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.165. URL https://aclanthology.org/2022.findings-acl.165.
- [41] Fred Philippy, Siwen Guo, and Shohreh Haddadan. Towards a common understanding of contributing factors for cross-lingual transfer in multilingual language models: A review. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5877–5891, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.323. URL https://aclanthology.org/2023.acl-long.323.
- [42] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2018. URL https://d4mucfpksywv. cloudfront.net/better-language-models/language-models.pdf.
- [43] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2305.18290.

- [44] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward training trillion parameter models. In Christine Cuicchi, Irene Qualters, and William T. Kramer, editors, Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2020, Virtual Event / Atlanta, Georgia, USA, November 9-19, 2020, page 20. IEEE/ACM, 2020. doi: 10.1109/SC41405.2020.00024. URL https://doi.org/10.1109/SC41405.2020.00024.
- [45] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: an adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99–106, aug 2021. ISSN 0001-0782. doi: 10.1145/3474381. URL https://doi.org/10.1145/3474381.
- [46] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://dblp.uni-trier. de/db/journals/corr/corr1707.html#SchulmanWDRK17.
- [47] Satoshi Sekine, Maya Ando, Michiko Goto, Hisami Suzuki, Daisuke Kawahara, Naoya Inoue, and Kentaro Inui. ichikara-instruction: Constructing a japanese instruction dataset for llms. In In Proceedings of The Thirtieth Annual Meeting of The Association for Natural Language Processing (NLP2024), pages 1508–1513, 2024. URL https://www.anlp.jp/proceedings/ annual_meeting/2024/pdf_dir/A6-3.pdf. in Japanese.
- [48] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv, abs/1909.08053, 2019. URL http://arxiv.org/abs/1909.08053.
- [49] Tomoki Sugimoto, Yasumasa Onoe, and Hitomi Yanaka. Jamp: Controlled Japanese temporal inference dataset for evaluating generalization capacity of language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 57–68, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.acl-srw.8.
- [50] Yikun Sun, Zhen Wan, Nobuhiro Ueda, Sakiko Yahata, Fei Cheng, Chenhui Chu, and Sadao Kurohashi. Rapidly developing high-quality instruction data and evaluation benchmark for large language models with minimal human effort: A case study on Japanese. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 13537–13547, Torino, Italia, May 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.1184.
- [51] Toyotaro Suzumura, Akiyoshi Sugiki, Hiroyuki Takizawa, Akira Imakura, Hiroshi Nakamura, Kenjiro Taura, Tomohiro Kudoh, Toshihiro Hanawa, Yuji Sekiya, Hiroki Kobayashi, Yohei Kuga, Ryo Nakamura, Renhe Jiang, Junya Kawase, Masatoshi Hanai, Hiroshi Miyazaki, Tsutomu Ishizaki, Daisuke Shimotoku, Daisuke Miyamoto, Kento Aida, Atsuko Takefusa, Takashi Kurimoto, Koji Sasayama, Naoya Kitagawa, Ikki Fujiwara, Yusuke Tanimura, Takayuki Aoki, Toshio Endo, Satoshi Ohshima, Keiichiro Fukazawa, Susumu Date, and Toshihiro Uchibayashi. mdx: A cloud platform for supporting data science and cross-disciplinary research collaborations. In 2022 IEEE Intl Conf on Dependable, Autonomic and Secure Computing, Intl Conf on Pervasive Intelligence and Computing, Intl Conf on Cloud and Big Data Computing, Intl Conf on Cyber Science and Technology Congress (DASC/PiCom/CBDCom/CyberSciTech), pages 1–7, 2022. doi: 10.1109/DASC/PiCom/CBDCom/Cy55231.2022.9927975.
- [52] Masashi Takeshita, Rafal Rzpeka, and Kenji Araki. Jcommonsensemorality: Japanese dataset for evaluating commonsense morality understanding. In In Proceedings of The Twenty Nineth Annual Meeting of The Association for Natural Language Processing (NLP2023), pages 357– 362, 2023. URL https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_ dir/D2-1.pdf. in Japanese.
- [53] Gemini Team. Gemini: A family of highly capable multimodal models, 2024.
- [54] Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch, and Eiichiro Sumita. Introducing the Asian language treebank (ALT). In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Sara Goggi, Marko Grobelnik, Bente Maegaard, Joseph Mariani, Helene Mazo, Asuncion Moreno,

Jan Odijk, and Stelios Piperidis, editors, Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 1574–1578, Portorož, Slovenia, May 2016. European Language Resources Association (ELRA). URL https://aclanthology.

##### org/L16-1249.

- [55] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.
- [56] Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. Do-not-answer: Evaluating safeguards in LLMs. In Yvette Graham and Matthew Purver, editors, Findings of the Association for Computational Linguistics: EACL 2024, pages 896–911, St. Julian’s, Malta, March 2024. Association for Computational Linguistics. URL https://aclanthology.org/ 2024.findings-eacl.61.
- [57] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mT5: A massively multilingual pre-trained text-to-text transformer. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online, June

2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.41. URL https://aclanthology.org/2021.naacl-main.41.

- [58] Hitomi Yanaka and Koji Mineshima. Assessing the generalization capacity of pre-trained language models through Japanese adversarial natural language inference. In Proceedings of the 2021 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP (BlackboxNLP2021), 2021.
- [59] Hitomi Yanaka and Koji Mineshima. Compositional evaluation on Japanese textual entailment and similarity. Transactions of the Association for Computational Linguistics, 10:1266–1284,

2022. doi: 10.1162/tacl_a_00518. URL https://aclanthology.org/2022.tacl-1.73.

- [60] Hitomi Yanaka, Namgi Han, Ryoma Kumon, Jie Lu, Masashi Takeshita, Ryo Sekizawa, Taisei Kato, and Hiromi Arai. Analyzing social biases in japanese large language models. arxiv:2406.02050, 2024.
- [61] Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuanjing Huang. Do large language models know what they don’t know? In Anna Rogers, Jordan BoydGraber, and Naoaki Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 8653–8665, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.551. URL https://aclanthology.org/ 2023.findings-acl.551.
- [62] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Lluís Màrquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1472. URL https://aclanthology.org/P19-1472.
- [63] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E Gonzalez, and

Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 46595–46623. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf.

