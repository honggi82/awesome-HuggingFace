# arXiv:2310.20707v2[cs.CL]5Mar2024

## WHAT’S IN MY BIG DATA?

Yanai Elazar1,2 Akshita Bhagia1 Ian Magnusson1 Abhilasha Ravichander1 Dustin Schwenk1 Alane Suhr3 Pete Walsh1 Dirk Groeneveld1 Luca Soldaini1 Sameer Singh4 Hannaneh Hajishirzi1,2 Noah A. Smith1,2 Jesse Dodge1

1Allen Institute for AI 2Paul G. Allen School of Computer Science & Engineering, University of Washington 3University of California, Berkeley 4University of California, Irvine

yanaiela@gmail.com https://github.com/allenai/wimbd wimbd.apps.allenai.org

ABSTRACT

Large text corpora are the backbone of language models. However, we have a limited understanding of the content of these corpora, including general statistics, quality, social factors, and inclusion of evaluation data (contamination). In this work, we propose WHAT’S IN MY BIG DATA? (WIMBD), a platform and a set of sixteen analyses that allow us to reveal and compare the contents of large text corpora. WIMBD builds on two basic capabilities—count and search—at scale, which allows us to analyze more than 35 terabytes on a standard compute node. We apply WIMBD to ten different corpora used to train popular language models, including C4, The Pile, and RedPajama. Our analysis uncovers several surprising and previously undocumented findings about these corpora, including the high prevalence of duplicate, synthetic, and low-quality content, personally identifiable information, toxic language, and benchmark contamination. For instance, we find that about 50% of the documents in RedPajama and LAION-2B-en are duplicates. In addition, several datasets used for benchmarking models trained on such corpora are contaminated with respect to important benchmarks, including the Winograd Schema Challenge and parts of GLUE and SuperGLUE. We open-source WIMBD’s code and artifacts to provide a standard set of evaluations for new text-based corpora and to encourage more analyses and transparency around them.

1 INTRODUCTION

Data is the foundation upon which machine learning (ML) is built. The introduction of new datasets drives progress, playing a crucial role in facilitating research and the creation of models with novel capabilities. Over time, the computational cost of AI experiments has dramatically increased, partly due to training increasingly large models on increasingly large datasets (Schwartz et al., 2020; Sevilla et al., 2022); today, some of the most impactful datasets are being created by scraping text from the entire publicly-available internet (Raffel et al., 2020; Together Computer, 2023; Penedo et al., 2023; Soldaini et al., 2024). These are some of the largest text datasets that have ever been built, and they are typically introduced with only a description of how they were made but no documentation of their contents. This is an important distinction, as we are now training models on massive text corpora without knowing what ideas, topics, toxicity, or personal information they contain.

Meanwhile, language models (LMs) have become ubiquitous and are used by people worldwide daily. These AI systems directly impact people’s lives, and thus, it has become vitally important to understand their capabilities and drawbacks. Models are only capable of learning from the data they were trained on, but analysis of pretraining corpora is hindered by lack of public release and by their massive size. Work analyzing the contents of web-scale corpora typically focuses on a subset of important dimensions, and there has been almost no work analyzing multiple datasets across the same dimensions. This means that ML practitioners have no practical tools to describe differences between datasets before choosing which one(s) to use.

###### Building Blocks

###### Analyses

###### WIMBD

###### Personally Identiﬁable Information (PII)

Most-Common Ngrams

###### Domain Distribution

###### Contamination

|Data|Contamination<br><br>|
|---|---|
|BoolQ MNLI XSum WSC|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>|

—————

Percentage

[Figure 5]

jurafsky@stanford.edu

Counts

==========

[Figure 6]

PII

********** ){ref-type=“ﬁg”} //////////

(206) 430-7757

[Figure 7]

[Figure 8]

208.80.152.2

Domain

Count

Search

- Figure 1: An overview of WIMBD. We implement two fundamental capabilities: Count and Search, allowing quick processing and access to large text corpora, which enables a wide range of analyses.

In this work, we propose to investigate the content of large text corpora using WHAT’S IN MY BIG DATA (WIMBD), a set of tools that enables practitioners to easily explore and quickly analyze large language datasets. We also use this tool to provide some of the first measurements across different web-scale datasets that are directly comparable. WIMBD has two components: (1) a search tool that enables programmatic access to search for documents containing a query using an Elasticsearch1 (ES) index. ES is a search engine that allows retrieving strings from a corpus, the documents where they appeared, and the number of times they appeared. (2) a count functionality, built using map-reduce (Dean & Ghemawat, 2008), allowing quick iteration over an entire dataset and extraction of relevant information, e.g., the character length distribution of documents, duplicates, domain counts, finding personally identifiable information (PII), and more. WIMBD is extendable and can be used to index, count, and analyze other corpora at scale (we benchmark the runtimes in Appendix D).

Using these tools, we perform a set of sixteen analyses on ten different English corpora used to train LMs, including C4 (used to train T5; Raffel et al., 2020), The Pile (used to train Pythia; Gao et al., 2020; Biderman et al., 2022; 2023), and RedPajama (used to reproduce Llama, Touvron et al., 2023, and to train RedPajama-INCITE; Together Computer, 2023). We divide our analyses into four categories: (1) data statistics (e.g., number of tokens and domain distribution; §4.2); (2) data quality (e.g., most frequent n-grams and measuring duplicate documents; §4.3); (3) community- and societyrelevant measurements (e.g., benchmark contamination and personally identifiable information detection; §4.4); and (4) cross-corpora analysis (e.g., comparing the most common n-gram and document overlap; §B.4). An illustration of WIMBD is presented in Figure 1.

Our work presents many insights on data distribution and anomalies. For example, inspecting the distribution over document lengths exposes anomalies where specific lengths are overrepresented relative to neighboring lengths; these anomalies often correspond to near-duplicate template-generated text or documents arbitrarily truncated to a specific character length. As another example, punctuation sequences are frequently the most common n-grams, such as a dash (‘-’) repeated ten times as the most common 10-gram in The Pile. WIMBD offers both retrospective documentation and grounding of model behavior to their training data and actionable insights for higher-quality corpora curation.

- 2 BACKGROUND: ON THE IMPORTANCE OF DATA UNDERSTANDING

There have been repeated calls for ML practitioners to provide better data documentation (e.g., McMillan-Major et al., 2023; Bender & Friedman, 2018; Mitchell et al., 2023; Pistilli et al., 2023; Paullada et al., 2021; Gebru et al., 2021). On the other hand, some of the most impactful ML models are increasingly opaque, specifically with respect to the most important component of recent advancements: data. With the increasingly competitive nature of the field, developers of systems like GPT-4 (OpenAI, 2023) and PaLM-2 (Google, 2023) have been offering little transparency into the most important development decisions, including the sources, size, and contents of their training data.

As web-scale datasets drive this rapid progress in modern ML systems, the gap between data transparency and documentation is more striking than ever (Kaddour et al., 2023). From a technical standpoint, the massive size of these datasets makes analysis of their contents challenging; even if OpenAI or Google shared their training data, it’s unclear where to start understanding it in its entirety. Tools like the Data Measurements Tool (Luccioni et al., 2021) and Know Your Data (Google, 2021) work towards improving data documentation, but focus on smaller datasets since the scale of web data leads to significant technical challenges. Our work aims to address this critical missing component.

1https://www.elastic.co/elasticsearch/

While other works support indexing and analyses of large corpora (Piktus et al., 2023a; Marone & Van Durme, 2023; Simig et al., 2022; Piktus et al., 2023b; Razeghi et al., 2022b), these efforts support a single corpus and often do not support programmatic access to the data or the analysis. Instead, we offer a holistic approach that combines search and counting with a package that allows programmatic access through wrappers on top of the ES API and extendable efficient counting capabilities.

Additional efforts are concerned with the effect of data on model behavior. Longpre et al. (2023) investigate how the composition of LMs’ pretraining data influences their downstream performance. Razeghi et al. (2022a) measure high correlation between term frequency and LMs’ few-shot reasoning capabilities with those terms. Shin et al. (2022) study the effect of pretraining corpora on in-context abilities. Seshadri et al. (2023) demonstrate that text-to-image models mimic biases from their training data. Akyurek et al. (2022) study fact tracing for identifying pretraining examples that enable a factual assertion, while Guu et al. (2023) offer a training run simulator, which allows making counterfactual queries on what a model would have learned under a different training procedure. These efforts separately built dedicated infrastructure to perform the studies. Our work provides a dedicated interface and tooling that allows performing a wide range of analyses on large-scale corpora, categorizing and offering novel analyses that highlight new insights into these large corpora.

### 3 WIMBD: THE PLATFORM

A core desideratum of WIMBD is to enable quick processing of terabytes of data. As such, we focus on uncomplicated, standard methods from the information retrieval and data management communities. WIMBD is comprised of two basic components: counting and search (retrieval). Fast counting and retrieving enable us to answer fundamental questions about data, as we demonstrate in Section 4. We summarize the framework abilities and types of analyses in Table 1. We run our experiments using a compute node machine with 224 CPUs and 882GB RAM, and an Elasticsearch cluster for the indexed corpora.

Table 1: Summary of the capabilities WIMBD provides and the analyses enabled by them.

Basic Ability Analyses

Document Counts, min/max doc length, #tokens, domain distribution, utterance date statistics, geolocation, language distribution, length distribution, toxic language, personally identifiable information, demographic sentiment co-occurrences

Exact Counts (§3.1)

Compressed Counts (§3.1) Duplicates, most & least common n-grams Search (§3.2) Benchmark contamination, n-gram counts

- 3.1 COUNTING

Due to the sparsity of language data and the scale of the data of interest, accurate counting can be challenging. We leverage the map-reduce framework (Dean & Ghemawat, 2008). We provide two approaches for counting, described below.

Exact Counts The exact counts approach is designed for cases where the number of possible values is tractable and can fit in memory. This fits cases where we are interested in calculating a bound number of variables of interest (e.g., number of documents,§4.2, or document length, §4.3.3).

Compressed Counts The compressed counts approach is designed for cases where the number of possible values is intractable. For instance, the total 10-grams in a large corpus can be very high, and the memory usage to compute all of them would be overwhelming. Similarly, finding duplicates requires keeping and comparing the strings of all documents in memory. In the case of C4, that would require over 800 GB of RAM. Instead, we apply a compression function (e.g., hashing, Bloom, 1970) to those values, reducing memory footprint while sacrificing some accuracy (due to hash collisions). For example, when finding the most common 10-grams, we store a table of counts where the keys in the table correspond to hashes of 10-grams. The hash table size is configurable according to the amount of memory available. The larger the hash table, the smaller the probability of hash collisions and, therefore, the higher the accuracy of the counts. E.g., unigram estimates are more accurate than 10-gram estimates since the number of possible values is much smaller.

- 3.2 SEARCHING

The second part of WIMBD allows fast text retrieval. For instance, we can get the number of documents mentioning a word or sequence (document frequency). It also allows more complex Boolean queries. While search and retrieval have numerous implementations, such as reverse indices, suffix arrays,

###### Table 2: Summary statistics of the corpora, along with the models trained on them. * signifies that the model was not trained on the exact version we consider, either due to some data mismatch, or the original data being private.

Corpus Origin Model Size (GB) # Documents # Tokens max(# Tokens) min(# Tokens) OpenWebText Gokaslan & Cohen (2019) GPT-2* (Radford et al., 2019) 41.2 8,005,939 7,767,705,349 95,139 128 C4 Raffel et al. (2020) T5 (Raffel et al., 2020) 838.7 364,868,892 153,607,833,664 101,898 5 mC4-en Chung et al. (2023) umT5 (Chung et al., 2023) 14,694.0 3,928,733,374 2,703,077,876,916 181,949 1 OSCAR Abadji et al. (2022) BLOOM* (Scao et al., 2022) 3,327.3 431,584,362 475,992,028,559 1,048,409 1 The Pile Gao et al. (2020) GPT-J/Neo & Pythia (Biderman et al., 2023) 1,369.0 210,607,728 285,794,281,816 28,121,329 0 RedPajama Together Computer (2023) LLaMA* (Touvron et al., 2023) 5,602.0 930,453,833 1,023,865,191,958 28,121,329 0 S2ORC Lo et al. (2020) SciBERT* (Beltagy et al., 2019) 692.7 11,241,499 59,863,121,791 376,681 1 peS2o Soldaini & Lo (2023) - 504.3 8,242,162 44,024,690,229 97,043 154 LAION-2B-en Schuhmann et al. (2022) Stable Diffusion* (Rombach et al., 2022) 570.2 2,319,907,827 29,643,340,153 131,077 0 The Stack Kocetkov et al. (2023) StarCoder* (Li et al., 2023) 7,830.8 544,750,672 1,525,618,728,620 26,298,134 0

suffix trees for exact match search, and dense retrieval for fuzzy search, in this work, we use ES, an inverted index. We build a wrapper on top of the ES API, allowing tailored and customized searches to fit our analysis requirements. We leave it to future work to explore other search alternatives.

- 4 WIMBD: THE ANALYSES

This section presents analyses conducted in WIMBD, grouped by category. First, we describe the ten corpora considered in this study (§4.1). We then consider four high-level categories, each split into several analyses: data statistics (§4.2), data quality (§4.3), and community- and society-relevant measurements (§4.4). Cross-corpus analyses, as well as elaborations and more analyses are presented in the appendix (§B). Our analyses are inspired by previous works (Dodge et al., 2021; Gao et al., 2020), but we expand them to multiple corpora, extend the types of analyses, and open-source our modular toolkit to encourage researchers to scrutinize their corpora. We offer the first extensive analyses on ten, combining extension of previous analyses and several novel ones.

- 4.1 CORPORA

We cover ten different large corpora, spanning across text-only (e.g., C4) to image captions (LAION2B-en) and code (The Stack). These corpora have been used in training language models (or similar large-scale models, such as Stable Diffusion; Rombach et al. 2022). A high-level description of these datasets using WIMBD is presented in Table 2, and further details about the construction and origin of these corpora are detailed in Appendix A.

- 4.2 DATA STATISTICS

Main Findings

- • Four out of the ten corpora we consider have ‘empty’ documents (meaning they contain only space-like characters), while The Pile and RedPajama contain the same longest document (with over 28 million tokens) of an encyclopedia.
- • While the most common source of webpages in C4 originates from www.nytimes.com, it consists of less than 0.05% of the total web pages, mC4-en most common domain is google.com (over 5% of the documents), and cdn.shopify.com contributes almost 6% to the total documents in LAION-2B-en.

- 4.2.1 SUMMARY STATISTICS

We begin by computing some summary statistics and present the results in Table 2. Using the

Exact Counts we compute the following high-level statistics of a corpus: (1) size, (2) number of documents, (3) number of tokens,2 (4) the size of the longest document, and (5) the size of the shortest document. Out of all corpora, mC4-en is the largest, which takes 14.7TB of disk, and 2.7 trillion tokens. After that comes The Stack with a size of 7.8TB, and more than 1.5 trillion tokens. Interestingly, four corpora contain documents with empty strings: LAION-2B-en (81 total), which typically contain a sequence of white spaces. In The Stack (1,350 total), RedPajama (3,877), and The

2We use Unicode text segmentation (Unicode, 2023) as a tokenizer, but we support any tokenizer supported by HuggingFace’s tokenizers library (Moi & Patry, 2023).

- Figure 2: Domain distribution of the ten most common domains per token for C4, LAION-2B-en, and RedPajama.

Pile (7,533), documents typically contain a mix of special characters that denote spacing (e.g., ‘\n’, or ‘\t’). In RedPajama, all of the empty strings are from the arXiv subset. The longest document in The Stack is a json file, with 26,298,134 tokens from http://jquery.com/. The longest document in The Pile and RedPajama is the same encyclopedia book called “INTERNATIONAL ENCYCLOPEDIA OF THE SOCIAL & BEHAVIORAL SCIENCES” from the Books3 subset with 28,121,329 tokens.

- 4.2.2 INTERNET DOMAIN DISTRIBUTION

Some corpora contain metadata information about the URL where the documents came from. As such, we employ the Exact Counts functionality, to parse the entire corpus, and extract information from the URLs about the (1) schemas (e.g., http, https), (2) domains (e.g., www.google.com, en.wikipedia.org, etc.), and (3) suffixes (e.g., com, org, de, etc.).

We apply these counts on the corpora that contain this information, namely C4, mC4-en, OSCAR, RedPajama, and LAION-2B-en. Starting with the domain analysis, we perform these counts twice: once when each domain is counted per document (yielding documents per domain) and another where each domain is counted per token (yielding tokens per domain). We present the results of three corpora per token in Figure 2 (and the full results in Appendix B.1). First, we note that C4 contains documents from a diverse set of domains, and even the percentage of the most common one, patents.google.com, is less than 0.05%. On the other hand, in the case of LAION2B-en, cdn.shopify.com is responsible for more than 6% of the documents. Similarly, arxiv.org is responsible for more than 12% of the documents in RedPajama. We showcase the results of the domains for the other corpora, as well as the schemas and suffixes in Appendix B.1.

- 4.3 DATA QUALITY Main Findings

- • The most common n-grams often correspond to repeated punctuation marks and duplicates.
- • While more than 60% of documents in The Pile are duplicates (unsurprisingly due to

oversampling), RedPajama and LAION-2B-en also contain about 50% duplicate documents.

• Document length distribution reveals interesting (and unexpected) outliers of documents, often resulting from duplicate documents and idiosyncratic data decisions.

- 4.3.1 MOST & LEAST COMMON n-GRAMS

Measuring outliers can reveal interesting insights about a corpus (Mitchell et al., 2023), We explore the most and least common token n-grams of each corpus using the Compressed Counts . We compute the 10K most common n-grams for all corpora, with n ∈ {1,2,3,10}. We report the results of the ten most common 10-grams in Table 3 and of the ten most common uni-, bi-, and tri-grams in Table 9 in the Appendix. Identical n-grams across corpora are highlighted in the same colors.

The different corpora contain a lot of uncleaned html or markdown format (e.g., ten times ‘?’ or ‘amp’), or boilerplate texts such as: “. You can follow any responses to this entry through” in C4, or “( Log Out / Change ) You are commenting using” in OSCAR, and formatting (“[1][2][3][”) in S2ORC and peS2o, which signifies references.

A striking finding from this analysis is the vast repetition of such 10-grams. For instance, ‘?’, ‘.’, and ‘-’ repeated ten times appear 9, 7.2, and 4.4 million times, respectively, in C4. We perform a manual analysis on the repeating question marks in C4 to better understand the scenarios where they

- Table 3: Most common 10-grams in five of the corpora we consider. n-grams from the top-10 that occur in more than one corpus are highlighted in the same color.

OpenWebText C4 mC4-en OSCAR The Pile n

Count - 3.64B . 602M

|-gram Co<br><br>- - - - - - - - - 3.4<br><br><br>. . . . . . . . . 1.05<br><br>= = = = = = = = = = 83 * * * * * * * * * * 59<br><br>|unt n-gram Co<br><br>M ? ? ? ? ? ? ? ? ? ? 9 M . . . . . . . . . . 7.27<br><br>0K - - - - - - - - - - 4.41 5K * * * * * * * * * * 3.87<br><br>|unt n-gram Co M . . . . . . . . . . 1.7 M - - - - - - - - - - 82 M 34 M * * * * * * * * * * 31<br><br>|unt n-gram Co 6B 77<br><br>3M \ \ \ \ \ \ \ \ \ \ 39 9M - - - - - - - - - - 17<br><br>4M . . . . . . . . . . 91.<br><br><br>|unt n-gram 3M - - - - - - - - - 5M = = = = = = = = = =<br><br>5M * * * * * * * * * *<br><br>6M ) { ref - type = " fig " }<br>|
|---|---|---|---|---|
|# # # # # # # # # # 30 amp ; amp ; amp ; amp ; amp ; 27<br><br>amp ; amp ; amp ; amp ; amp 26 — — — — — — — — — — 24 ... ... ... ... ... ... ... ... ... ... 88.<br><br>~ ~ ~ ~ ~ ~ ~ ~ ~ 83.|2K ! ! ! ! ! ! ! ! ! ! 1.91<br><br>8K . You can follow any responses to this entry through 78 5K 75<br><br>9K You can follow any responses to this entry through the 75 1K can follow any responses to this entry through the RSS 75<br><br><br>3K follow any responses to this entry through the RSS 2.0 74<br><br><br>|M \ / s \ / files \ / 1 \ 18 4K / s \ / files \ / 1 \ / 18 3K \ / \ / cdn.shopify.com \ / s \ / 18 2K / cdn.shopify.com \ / s \ / files \ / 18 2K \ / cdn.shopify.com \ / s \ / files \ 18 8K / \ / cdn.shopify.com \ / s \ / files 18<br><br>|3M * * * * * * * * * * 34. 3M = = = = = = = = = = 22. 2M ( Opens in new window ) Click to share on 15. 2M Log Out / Change ) You are commenting using your 13. 2M ( Log Out / Change ) You are commenting using 13. 2M . ( Log Out / Change ) You are commenting 13.<br><br>|9M / / / / / / / / / / 9M . . . . . . . . . . 7M # # # # # # # # # # 6M } - - - - - - - - 6M { ref - type = " fig " } ) 6M } = = = = = = = = =<br><br>|

188M 59.1M 56.2M 54.9M

; 38.3M 30.1M 28.9M

~ 21.8M

RedPajama S2ORC peS2o LAION-2B-en The Stack n

Count . 4.29B - 3.87B \ 2.75B

|-gram Co<br><br>. . . . . . . . . 67<br><br>- - - - - - - - - 50 \ \ \ \ \ \ \ \ \ 21<br><br><br>* * * * * * * * * * 19 = = = = = = = = = = 14<br><br>|unt n-gram Co 0M q q q q q q q q q q 30. 7M . . . . . . . . . . 5.4 3M + + + + + + + + + + 3.0 5M * * * * * * * * * * 1.9 5M º º º º º º º º º º 1.7<br><br>|unt n-gram Co<br><br>2M . . . . . . . . . . 1.42 9M [ 1 ] [ 2 ] [ 3 ] [ 45<br><br>3M ] [ 2 ] [ 3 ] [ 4 ] 45 3M 1 ] [ 2 ] [ 3 ] [ 4 45 3M [ 5 ] [ 6 ] [ 7 ] [ 45<br><br><br>|unt n-gram Co<br><br>M - - - - - - - - - - 1.6 7K 1.4 3K . . . . . . . . . . 1.1 3K \ \ \ \ \ \ \ \ \ \ 80 0K < br / > < br / > < br 79<br><br>|unt n-gram 5M - - - - - - - - - 3M * * * * * * * * * * 5M 0 0 0 0 0 0 0 0 0 0 9K = = = = = = = = = = 7K , " resolved " : " https : / /<br><br>|
|---|---|---|---|---|
|/ / / / / / / / / 79.3M · · · · · · · · · · 1.56M [ 6 ] [ 7 ] [ 8 ] [ 448K / > < br / > < br / > 796K " , " resolved " : " https : / / . . / . . / . 35.3M - - - - - - - - - - 1.11M ] [ 6 ] [ 7 ] [ 8 ] 448K br / > < br / > < br / 796K " resolved " : " https : / / registry.npmjs.org<br><br>. . / . . / . . 35.3M [ 5 ] [ 6 ] [ 7 ] [ 646K 5 ] [ 6 ] [ 7 ] [ 8 446K > < br / > < br / > < 576K resolved " : " https : / / registry.npmjs.org<br><br>. / . . / . . / 35.2M [ 1 ] [ 2 ] [ 3 ] [ 645K ] [ 7 ] [ 8 ] [ 9 ] 446K | Price : 1 Credit ( USD $ 1 ) 437K , , , , , , , , , , # # # # # # # # # # 33M [ 6 ] [ 7 ] [ 8 ] [ 644K 6 ] [ 7 ] [ 8 ] [ 9 444K vector | Price : 1 Credit ( USD $ 1 437K . tgz " , " integrity " : " sha512<br><br>| | | | |

2.62B 1.46B

/ 1.46B . . 1.42B . / / 1.42B / . 1B

938M

appear on the ten consecutive question marks symbols and categorize each appearance into writing, noise, and format occurrence. Analyzing 100 random documents, we found that 68% of documents use such n-grams as part of their writing style (e.g., ... $6???????????? How is that possible?, or ... So what do u think?????????????????????????). 18% are due to noise as we could not understand the context or content of the writing (e.g., ... e ??????????????? kap chit-koa ??), and finally, 14% of the documents were due to different format styles or issues (e.g., a sequence of question marks following by a ‘normal’ text, or a sequence of question marks between keywords).

- 4.3.2 DUPLICATES

Previous work has found that duplication can affect the quality of pretraining data, impacting sample efficiency (Lee et al., 2022; Tirumala et al., 2023) and memorization (Carlini et al., 2023). While more recent work finds contradictory evidence on data with less web-scraped text (Biderman et al., 2023), measuring duplication in pretraining data is necessary for future research on its effects. We calculate duplicates by matching documents with an MD5 hash of their texts (using Compressed Counts ). If more than a single document has the same hash, we consider them duplicates.3 We examine the duplication of document text and URLs within each dataset. While some datasets explicitly deduplicate their content, others do not, and some even oversample some sources.

139M

% of total uniq % of total

Table 4: Most frequent text duplicates from four datasets with text duplicates, along with their counts. Truncation for visualization is marked by [...].

60

| |
|---|

1.2B

460M

50

64.6M

Duplicate%

40

165M

3.7M

219M

30

Corpus Text

342M

20

1.8M

OSCAR In order to login you must be registered. Register ing Count: 1.8M takes only a few moments but gives you increas[...]

10

19.9M

The Pile {\n "info" : {\n "version" : 1,\n "author" : "xcode"\n Count: 3.8K }\n}

0

OSCAR The Pile RedPajama S2ORC LAION-2B-en

RedPajama ACCEPTED\n\n#### According to\nInternational Pla

- Figure 3: Percentages of document and document cluster duplicates in corpora with > 1% documents duplicated (corresponding to blue and orange bars). Duplicate counts are above bars.

nt NamesIndex\n\n#### Published in\nnull\n\n#### Count: 213.9K Original n[...] LAION-2B-en Front Cover Count: 1M

In Figure 3 we show counts and ratios of duplication across datasets with greater than 1% documents duplicated, and all datasets are shown in Table 13 in the appendix. These are based on two kinds of counts: (1) the count of documents in all clusters of duplicate text (in blue) and (2) the count of duplicate clusters (in orange). As expected, deduplicated corpora such as C4 have no exact duplicates (as those were filtered out of the corpus). In contrast, The Pile, which intentionally oversampled some data sources, has many duplicates (139M documents belonging to 64.6M duplicate text clusters). LAION-2B-en has the second highest ratio of duplicate documents (1.25B documents belonging to 342M duplicate text clusters), perhaps due to the smaller space of short sentences common in

3To test for hash collisions, we rerun the analysis with a different random seed. None of the > 7 billion hashes across the ten corpora had a different count. This could only occur if an identical number of collisions conflated an identical set of counts or, more likely, there were no collisions.

its image “alt text” source. Figure 15 in the appendix showcase the images of the most common duplicates in LAION-2B-en, with the most common images describe mainly receipts.

- Table 4 showcases duplicates with the most occurrences in four corpora. These duplicates vary dramatically in length and domain. LAION-2B-en, OSCAR, and RedPajama have clusters with the most occurrences, in the hundreds of thousands and above. Top duplicates in LAION-2B-en are shorter and describe products and website features. OSCAR’s top duplicates are all instances of website boilerplate.4 RedPajama’s top duplicates come from similar templated citation information.

- 4.3.3 DOCUMENT LENGTH DISTRIBUTION

101 103 105 107

Characters per Document

0.0

0.2

0.4

0.6

0.8

%ofDocuments

"In order to login you must be registered..."

FAQ for forum software phpBB

DeepMind Mathematics

OSCAR The Pile C4

Figure 4: Distribution over character document lengths (in log-scale) for C4, OSCAR and The Pile.

We compute document length distributions with Exact Counts . We expect a smooth distribution over document lengths, and deviation from such a distribution may indicate the presence of artificial documents or near duplicates.5 We compute the character length distribution and present results for three corpora in Figure 4 (additional results in Appendix B.2.3).

While C4 is free of duplicate documents, it include clusters of template-generated near-duplicate documents exposed by outliers of identical document lengths. Beyond template-generated user-facing copy (e.g., template-generated documents from a reverse phone lookup site, each associated with a unique phone number), we find clusters of template-generated JavaScript snippets, and large collections of unique documents, including numerous permutations of the same keywords, likely crafted for SEO purposes.

The Pile, featuring the longest documents, has a notable outlier with nearly 1% of its documents precisely 8,194 characters long. These outliers are derived from the DeepMind Mathematics dataset (Saxton et al., 2019), truncated to fit this length. The Pile also contains a significant number of short template-generated code snippets, e.g., a number of documents (of lengths 9, 18, and 36 tokens) each corresponding to a unique publication in various medical journals, and to auto-generated metadata files (of length 20 tokens) used in the Unity game engine. While OSCAR has no documents shorter than 100 characters, as those were filtered, it contains many near-duplicate documents that correspond to website boilerplate, e.g., template-generated FAQs about how to use the forum software phpBB.

- 4.4 COMMUNITY- AND SOCIETY-RELEVANT MEASUREMENTS

Main Findings

- • Instances of popular benchmarks like GLUE and SuperGLUE, were found in various corpora (e.g., C4 and RedPajama), render them unusable for fair model evaluation.
- • Automatic toxicity detection reveals that 1–16.5% of the documents in the corpora contain toxic language using an automatic classifier and between 0.01-16.6% using a taxonomy.
- • An estimated 200M, 4B, and 97M of email addresses, phone numbers, and IP addresses were found in the most PII-contaminated corpora per token (mC4-en).

- 4.4.1 BENCHMARK CONTAMINATION

As corpora grow and new evaluation datasets are created, the risk of contamination—where evaluation data are included in a (pre)training corpus—increases. As such, it is important to track contamination (Sainz et al., 2023; Jacovi et al., 2023).6 Using Search , we provide a contamination analysis of 82 datasets for four popular corpora: The Pile, C4, RedPajama, and OSCAR. We consider all datasets

- 4Many of these duplicate documents indicate that the user agent used to collect the dataset received automatic responses blocking it from crawling the website’s contents.
- 5Outlier lengths are those whose prevalence across the corpus is significantly higher than neighboring lengths. 6When evaluating a model trained on an existing corpus, one should exempt contaminated evaluation sets.

However, in the case of new corpus construction, practitioners may use WIMBD for decontaminating the corpus itself to maintain the evaluation data integrity.

%Contaminatedinstances

100.0

100

Corpus

The Pile C4 RedPajama OSCAR

| |
|---|

75

| |
|---|

67.5 67.5

| |
|---|

64.4

58.2 60.2

| |
|---|

52.6 52.8

50

49.4

45.5

45.0

32.2

30.4

29.3

29.2

25

18.7

18.6

13.9

11.1 11.1

10.9

9.9 9.9

7.5

6.2 6.2 5.9

5.3 3.1 3.1 3.4

5.2 3.5 3.5

5.1 5.1 5.1

4.8 4.9

2.0 2.0

1.9

1.6 0.3 0.3 0.2 0.2

1.4 1.4

1.2

1.0

0.6

0.3 0.2 0.2 0.2 0.2

0.1 0.1

0.1

0

super-glue_axbglue_ax head_qaglue_stsbstsb_multihealth_fact aeslc sick sem_evalsuper-glue_rteglue_rte liarsuper-glue_copawinograd_wscsuper-glue_wic

Dataset

Figure 5: Most contaminated evaluations test sets out of 82 PromptSource (Bach et al., 2022) datasets.

from PromptSource (Bach et al., 2022), a repository containing prompts for 279 different datasets (as of May 2023). We filter datasets we cannot automatically download, from Huggingface datasets (Lhoest et al., 2021), and datasets that do not have a test split. In addition, we only consider datasets that contain at least two inputs (e.g., natural language inference), leaving us with 82 datasets.

We measure contamination by testing whether all input fields are present in a single document and report the percentage of contaminated examples from the test set. Our contamination evaluation serves as an upper bound of exact-match dataset contamination. We provide more details of our analysis and design choices in Appendix B.3.1.

Contaminated datasets We present the results in Figure 5. We showcase all benchmarks whose contamination percentages are at least 5% in one of the four corpora. We find that RedPajama is the most contaminated dataset out of the four, where in eight out of the 15 corpora, its contamination rate is above 50%, and fully contaminated in the case of COPA (Roemmele et al., 2011). The Pile’s contamination rates are lower, but it is also contaminated with a few datasets, such as aesic (Zhang & Tetreault, 2019), WSC (Levesque et al., 2012) and WIC (Pilehvar & Camacho-Collados, 2019), which were included in the SuperGLUE evaluation benchmark (Wang et al., 2019).

Most examined datasets were not found in the corpora. It is important to note that while we find some contamination, most of the considered benchmarks do not appear in the corpora we investigated (67 out of the 82 datasets). For instance, Winogrande (Sakaguchi et al., 2021), a large corpus in the style of the Winograd schema, does not appear in any of the examined corpora.

- 4.4.2 PERSONALLY IDENTIFIABLE INFORMATION

PII is “information which can be used to distinguish or trace an individual’s identity, such as their name, social security number, biometric records, etc.” (Johnson III, 2007). Recent research has sought to extract PII from LMs (Carlini et al., 2021). These attacks highlight that LMs can ingest and reproduce PII contained in their training data, and show the risks of training on data that contains such information, even if the data remains private.

Table 5: Extrapolated PII frequencies. Count is the extrapolated frequency and Prec. is our identification precision accuracy, estimated by manual analysis of 100 random examples.

Corpus Email Addresses Phone Numbers IP Addresses Count Prec. Count Prec. Count Prec.

OpenWebText 364K 99 533K 87 70K 54 OSCAR 62.8M 100 107M 91 3.2M 43 C4 7.6M 99 19.7M 92 796K 56 mC4-en 201M 92 4B 66 97.8M 44 The Pile 19.8M 43 38M 65 4M 48 RedPajama 35.2M 100 70.2M 94 1.1M 30 S2ORC 630K 100 1.4M 100 0K 0 peS2o 418K 97 227K 31 0K 0 LAION-2B-en 636K 94 1M 7 0K 0 The Stack 4.3M 53 45.4M 9 4.4M 55

We document three kinds of personally identifiable information in pretraining corpora: phone numbers, email addresses, and IP addresses. We employ regular expres-

sions corresponding to each PII type using the Exact Counts . We provide more details about our methodology, the regexes, additional results, and error analyses in Appendix B.3.2. We conduct a manual analysis to estimate the precision of these methods on all corpora. The results of this analysis, as well as the extrapolated frequency of these matches, are presented in Table 5. Our identification method is highly precise (>80% precision) for email addresses on eight out of 10 corpora, and for phone numbers on five of the 10 corpora. Overall, most corpora contain a high volume of PII information, varying in type based on the corpus. For instance, RedPajama contain mainly phone numbers (70.2M) and a smaller amount of IP Addresses (1.1M), but S2ORC and peS2o contain mainly email addresses (630K and 418K, respectively) and no IP addresses were identified. The most common PII across corpora is phone numbers, followed by email addresses and IP addresses (except for The Stack, which has more IP addresses than email addresses: 4.4M vs. 4.3M, and peS2o, which has more email addresses than phone numbers). Finally, we observe that mC4-en contains the largest amount of PII, also when controlling for the number of tokens (Table 19 in the Appendix).

- 5 DISCUSSION

Data is one of the most poorly understood and studied components in ML research since “everyone wants to do the model work, not the data work” (Sambasivan et al., 2021). Yet, it is one of the most critical factors for successfully training a state-of-the-art language model. While the benefit of increasing model size is evident from the trend of recent years, it is not enough by itself, as the amount and quality of data are crucial (Kaplan et al., 2020).

Data Curation With the increasing data needed to train LMs (and other models for other modalities), it remains challenging to curate high-quality datasets. Besides the technical challenges of composing a large-scale dataset and the decisions that go into making it, these decisions and their influence on the final models are costly to assess due to the high computational resources required to train such models. With WIMBD, we hope to ease the decisions that go into crafting large-scale datasets by surfacing patterns and trends about what goes into them and what is left out from different aspects, such as data quality, community and society measurements, etc. Once decisions upon what data is important, and which should be left out of a dataset, practitioners can filter documents or passages that adhere to such decisions. The curation of the Dolma dataset (Soldaini et al., 2024) that happened while developing this work benefited from iterations over the insights from this work, such as the finding of ‘noisy’ most-common n-grams, and bugs in the initial ‘de-duplication’ implementation.

Data Documentation Adding to previous works that call for more data documentation, such as Datasheets (Gebru et al., 2021) and Data Statements (McMillan-Major et al., 2023), we argue for the importance of documenting such information. While previous works often focused and tailored the documentation for supervised-style datasets (e.g., “Is there a label or target associated with each instance?”, “How was the data associated with each instance acquired?” from Datasheets, and “What are the demographic characteristics of the annotators and annotation guideline developers?” from Data Statements) we call for more tailored documentation of large-scale pretraining corpora.7 This work offers a superset of the automatic full-corpus analyses proposed by Dodge et al. (2021); Gao

- et al. (2020), with several additions, categorization, and programmatic interface, allowing better understanding of the content of current and future large text corpora.

Grounding Models to their Training Data Unlike other factors of language model training, such as model architecture or optimizer choice, training data comes in the same natural language format as language model’s outputs and thus can be measured and described in all the same ways. As such, the data offers a unique opportunity for grounding models. For instance, a model’s ability to recall factual knowledge is derived from its training data (Jiang et al., 2020; Elazar et al., 2021a). On the other hand, models often perform better on frequent occurrences (Razeghi et al., 2022a; McCoy et al., 2023), and on documents similar to models’ training data (Longpre et al., 2023). The path to a holistic comprehension of model behavior is through the data, which requires an infrastructure investment to access big datasets and the right abstraction of data attributes.

- 6 CONCLUSION

In this work, we propose WIMBD, a framework for processing and analyzing large text corpora. Using WIMBD, we study ten different corpora that were used to train language models (or vision and language models, such as Stable Diffusion). We uncover interesting insights about these corpora using sixteen different analyses across four aspects: high-level statistics, data quality, communityand society- relevant measurements, and cross-data analysis. For instance, the most common source of texts for the LAION-2B-en dataset are the commercial websites Pinterest, Shopify, SlidePlayer, Amazon, and eBay. Regarding data quality, we find that about 50% of RedPajama and LAION-2Ben’s documents are duplicates. In addition, we find that many evaluation benchmarks, including several from GLUE and SuperGLUE, such as WSC, WIC, and RTE, are contaminated due to their appearance in corpora such as RedPajama. Besides the analyses, WIMBD offers an extendable platform for reproducing our analyses on other corpora, developing new ones, and answering research questions about data. We release all the code and artifacts for WIMBD to encourage researchers to adopt and extend our framework and analyze existing and new corpora.

7Many questions are still relevant for large pretraining corpora (e.g., “What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?”).

ACKNOWLEDGMENTS

We want to thank Ludwig Schmidt, Maarten Sap, and Emma Strubell, and the anonymous reviewers for discussions and feedback on this paper, Elizabeth Salesky for the help with Unicode rendering and getting excited about obscure Unicode characters with me, and Carissa Schoenick, Jon Borchardt, and Johann Dahm for assisting with visuals.

REFERENCES

Julien Abadji, Pedro Ortiz Suarez, Laurent Romary, and Benoît Sagot. Towards a cleaner documentoriented multilingual crawled corpus. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pp. 4344–4355, Marseille, France, June 2022. European Language Resources Association. URL https://aclanthology.org/2022.lrec-1.463.

Ekin Akyurek, Tolga Bolukbasi, Frederick Liu, Binbin Xiong, Ian Tenney, Jacob Andreas, and Kelvin Guu. Towards tracing knowledge in language models back to the training data. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 2429–2446, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/ 2022.findings-emnlp.180. URL https://aclanthology.org/2022.findings-emnlp.180.

Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don’t reach for the stars! arXiv preprint arXiv:2301.03988, 2023. URL https://arxiv.org/abs/ 2301.03988.

Stephen Bach, Victor Sanh, Zheng Xin Yong, Albert Webson, Colin Raffel, Nihal V. Nayak, Abheesht Sharma, Taewoon Kim, M Saiful Bari, Thibault Fevry, Zaid Alyafeai, Manan Dey, Andrea Santilli, Zhiqing Sun, Srulik Ben-david, Canwen Xu, Gunjan Chhablani, Han Wang, Jason Fries, Maged Al-shaibani, Shanya Sharma, Urmish Thakker, Khalid Almubarak, Xiangru Tang, Dragomir Radev, Mike Tian-jian Jiang, and Alexander Rush. PromptSource: An integrated development environment and repository for natural language prompts. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 93–104, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-demo.9. URL https://aclanthology.org/2022.acl-demo.9.

Iz Beltagy, Kyle Lo, and Arman Cohan. SciBERT: A pretrained language model for scientific text. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3615–3620, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1371. URL https://aclanthology.org/D19-1371.

Emily M. Bender and Batya Friedman. Data statements for natural language processing: Toward mitigating system bias and enabling better science. Transactions of the Association for Computational Linguistics, 6:587–604, 2018. doi: 10.1162/tacl_a_00041. URL https: //aclanthology.org/Q18-1041.

Stella Biderman, Kieran Bicheno, and Leo Gao. Datasheet for the pile, 2022. URL https:// arxiv.org/abs/2201.07311.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023. URL https: //openreview.net/forum?id=bpRTAnJ8LW.

Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPT-NeoX-20B: An open-source autoregressive language model. In Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pp. 95–136, virtual+Dublin,

May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.bigscience-1.9. URL https://aclanthology.org/2022.bigscience-1.9.

Burton H. Bloom. Space/time trade-offs in hash coding with allowable errors. Commun. ACM, 13(7): 422–426, jul 1970. ISSN 0001-0782. URL https://doi.org/10.1145/362686.362692.

Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pp. 2633–2650. USENIX Association, August 2021. ISBN 978-1-939133-243. URL https://www.usenix.org/conference/usenixsecurity21/presentation/carliniextracting.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= TatRHT_1cK.

Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, and Noah Constant. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=kXwdL1cWOAi.

Jeffrey Dean and Sanjay Ghemawat. Mapreduce: Simplified data processing on large clusters. Commun. ACM, 51(1):107–113, jan 2008. URL https://doi.org/10.1145/1327452.1327492.

Jesse Dodge, Maarten Sap, Ana Marasovi´c, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. Documenting large webtext corpora: A case study on the colossal clean crawled corpus. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 1286–1305, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlpmain.98. URL https://aclanthology.org/2021.emnlp-main.98.

Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, and Yoav Goldberg. Measuring and improving consistency in pretrained language models. Transactions of the Association for Computational Linguistics, 9:1012–1031, 2021a. URL https://aclanthology.org/2021.tacl-1.60.

Yanai Elazar, Hongming Zhang, Yoav Goldberg, and Dan Roth. Back to square one: Artifact detection, training and commonsense disentanglement in the Winograd schema. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10486–10500, Online and Punta Cana, Dominican Republic, November 2021b. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.819. URL https: //aclanthology.org/2021.emnlp-main.819.

Ali Emami, Kaheer Suleman, Adam Trischler, and Jackie Chi Kit Cheung. An analysis of dataset overlap on Winograd-style tasks. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 5855–5865, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.515. URL https://aclanthology.org/2020.coling-main.515.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020. URL https://arxiv.org/abs/2101.00027.

Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets. Commun. ACM, 64(12):86–92, nov

2021. ISSN 0001-0782. doi: 10.1145/3458723. URL https://doi.org/10.1145/3458723. Aaron Gokaslan and Vanya Cohen. Openwebtext corpus, 2019. URL https:// skylion007.github.io/OpenWebTextCorpus/.

Google. Know your data, 2021. URL https://github.com/pair-code/knowyourdata. Google. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. URL https://arxiv.org/

abs/2305.10403.

Kelvin Guu, Albert Webson, Ellie Pavlick, Lucas Dixon, Ian Tenney, and Tolga Bolukbasi. Simfluence: Modeling the influence of individual training examples by simulating training runs. arXiv preprint arXiv:2303.08114, 2023. URL https://arxiv.org/abs/2303.08114.

Alon Jacovi, Avi Caciularu, Omer Goldman, and Yoav Goldberg. Stop uploading test data in plain text: Practical strategies for mitigating data contamination by evaluation benchmarks. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 5075–5084, Singapore, December 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.emnlp-main.308.

Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423–438, 2020. doi: 10.1162/tacl_a_00324. URL https://aclanthology.org/2020.tacl-1.28.

Clay Johnson III. Us office of management and budget memorandum m-07-16, 2007. URL https: //georgewbush-whitehouse.archives.gov/omb/memoranda/fy2007/m07-16.pdf.

Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. Challenges and applications of large language models. arXiv preprint arXiv:2307.10169,

2023. URL https://arxiv.org/abs/2307.10169.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. URL https://arxiv.org/abs/2001.08361.

Denis Kocetkov, Raymond Li, Loubna Ben allal, Jia LI, Chenghao Mou, Yacine Jernite, Margaret Mitchell, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro Von Werra, and Harm de Vries. The stack: 3 TB of permissively licensed source code. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/ forum?id=pxpbTdUEpD.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris CallisonBurch, and Nicholas Carlini. Deduplicating training data makes language models better. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8424–8445, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.577. URL https://aclanthology.org/2022.acllong.577.

Hector J. Levesque, Ernest Davis, and Leora Morgenstern. The winograd schema challenge. In Proceedings of the Thirteenth International Conference on Principles of Knowledge Representation and Reasoning, KR’12, pp. 552–561. AAAI Press, 2012. ISBN 9781577355601. URL https: //dl.acm.org/doi/10.5555/3031843.3031909.

Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario Šaško, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Patry, Angelina McMillan-Major, Philipp Schmid, Sylvain Gugger, Clément Delangue, Théo Matussière, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar, François Lagunas, Alexander Rush, and Thomas Wolf. Datasets: A community library for natural language processing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 175–184, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. URL https: //aclanthology.org/2021.emnlp-demo.21.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023. URL https://arxiv.org/abs/2305.06161.

Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel Weld. S2ORC: The semantic scholar open research corpus. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 4969–4983, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.447. URL https://www.aclweb.org/anthology/ 2020.acl-main.447.

Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, et al. A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity. arXiv preprint arXiv:2305.13169, 2023. URL https://arxiv.org/abs/2305.13169.

Sasha Luccioni, Yacine Jernite, and Margaret Mitchell. Data measurements tool, 2021. URL https://huggingface.co/blog/data-measurements-tool.

Marc Marone and Benjamin Van Durme. Data portraits: Recording foundation model training data. arXiv preprint arXiv:2303.03919, 2023. URL https://arxiv.org/abs/2303.03919.

R. Thomas McCoy, Shunyu Yao, Dan Friedman, Matthew Hardy, and Thomas L. Griffiths. Embers of autoregression: Understanding large language models through the problem they are trained to solve. arXiv preprint arXiv:2309.13638, 2023. URL https://arxiv.org/abs/2309.13638.

Angelina McMillan-Major, Emily M. Bender, and Batya Friedman. Data statements: From technical concept to community practice. ACM J. Responsib. Comput., may 2023. doi: 10.1145/3594737. URL https://doi.org/10.1145/3594737.

Margaret Mitchell, Alexandra Sasha Luccioni, Nathan Lambert, Marissa Gerchick, Angelina McMillan-Major, Nazneen Ozoani, Ezinwanne Rajani, Tristan Thrush, Yacine Jernite, and Douwe Kiela. Measuring data. In arXiv, 2023. URL https://arxiv.org/abs/2212.05129.

Anthony Moi and Nicolas Patry. HuggingFace’s Tokenizers, April 2023. URL https://github.com/ huggingface/tokenizers.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. URL https://arxiv.org/ abs/2303.08774.

Amandalynne Paullada, Inioluwa Deborah Raji, Emily M. Bender, Emily Denton, and Alex Hanna. Data and its (dis)contents: A survey of dataset development and use in machine learning research. In Patterns, 2021. URL https://www.sciencedirect.com/science/article/pii/ S2666389921001847.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023. URL https://arxiv.org/abs/2306.01116.

Aleksandra Piktus, Christopher Akiki, Paulo Villegas, Hugo Laurençon, Gérard Dupont, Sasha Luccioni, Yacine Jernite, and Anna Rogers. The ROOTS search tool: Data transparency for LLMs. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 304–314, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-demo.29. URL https://aclanthology.org/2023.acl-demo.29.

Aleksandra Piktus, Odunayo Ogundepo, Christopher Akiki, Akintunde Oladipo, Xinyu Zhang, Hailey Schoelkopf, Stella Biderman, Martin Potthast, and Jimmy Lin. GAIA search: Hugging face and pyserini interoperability for NLP training data exploration. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 588–598, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-demo.57. URL https://aclanthology.org/2023.acl-demo.57.

Mohammad Taher Pilehvar and Jose Camacho-Collados. WiC: the word-in-context dataset for evaluating context-sensitive meaning representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 1267–1273, Minneapolis, Minnesota, June

2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1128. URL https: //aclanthology.org/N19-1128.

Giada Pistilli, Carlos Muñoz Ferrandis, Yacine Jernite, and Margaret Mitchell. Stronger together: On the articulation of ethical charters, legal tools, and technical documentation in ml. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’23, pp. 343–354, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701924. doi: 10.1145/3593013.3594002. URL https://doi.org/10.1145/3593013.3594002.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog post, 2019. URL https://openai.com/ research/better-language-models.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. URL http://jmlr.org/papers/v21/20-074.html.

Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot numerical reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 840–854, Abu Dhabi, United Arab Emirates, December 2022a. Association for Computational Linguistics. URL https://aclanthology.org/2022.findingsemnlp.59.

Yasaman Razeghi, Raja Sekhar Reddy Mekala, Robert L Logan Iv, Matt Gardner, and Sameer Singh. Snoopy: An online interface for exploring the effect of pretraining term frequencies on few-shot LM performance. In Wanxiang Che and Ekaterina Shutova (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 389–395, Abu Dhabi, UAE, December 2022b. Association for Computational Linguistics. URL https://aclanthology.org/2022.emnlp-demos.39.

Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In AAAI spring symposium: logical formalizations of commonsense reasoning, pp. 90–95, 2011. URL https://aaai.org/papers/02418-choice-of-plausible-alternatives-anevaluation-of-commonsense-causal-reasoning/.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, and Eneko Agirre. Did chatgpt cheat on your test?, Jun 2023. URL https://hitz-zentroa.github.io/lm-contamination/ blog/.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99–106, aug 2021. URL https://doi.org/10.1145/3474381.

Nithya Sambasivan, Shivani Kapania, Hannah Highfill, Diana Akrong, Praveen Paritosh, and Lora M Aroyo. “everyone wants to do the model work, not the data work”: Data cascades in high-stakes ai. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, CHI ’21, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450380966. doi: 10.1145/3411764.3445518. URL https://doi.org/10.1145/3411764.3445518.

David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. Analysing mathematical reasoning abilities of neural models. In International Conference on Learning Representations,

2019. URL https://openreview.net/forum?id=H1gR5iR5FX.

Teven Le Scao, Angela Fan, Christopher Akiki, Elizabeth-Jane Pavlick, Suzana Ili’c, Daniel Hesslow, Roman Castagn’e, Alexandra Sasha Luccioni, Franccois Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Rose Biderman, Albert Webson, Pawan Sasanka Ammanamanchi,

Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurenccon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa Etxabe, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris C. Emezue, Christopher Klamm, Colin Leong, Daniel Alexander van Strien, David Ifeoluwa Adelani, Dragomir R. Radev, Eduardo Gonz’alez Ponferrada, Efrat Levkovizh, Ethan Kim, Eyal Bar Natan, Francesco De Toni, Gérard Dupont, Germán Kruszewski, Giada Pistilli, Hady ElSahar, Hamza Benyamina, Hieu Trung Tran, Ian Yu, Idris Abdulmumin, Isaac Johnson, Itziar Gonzalez-Dios, Javier de la Rosa, Jenny Chim, Jesse Dodge, Jian Zhu, Jonathan Chang, Jorg Frohberg, Josephine L. Tobing, Joydeep Bhattacharjee, Khalid Almubarak, Kimbo Chen, Kyle Lo, Leandro von Werra, Leon Weber, Long Phan, Loubna Ben Allal, Ludovic Tanguy, Manan Dey, Manuel Romero Muñoz, Maraim Masoud, Mar’ia Grandury, Mario vSavsko, Max Huang, Maximin Coavoux, Mayank Singh, Mike Tian-Jian Jiang, Minh Chien Vu, Mohammad Ali Jauhar, Mustafa Ghaleb, Nishant Subramani, Nora Kassner, Nurulaqilla Khamis, Olivier Nguyen, Omar Espejel, Ona de Gibert, Paulo Villegas, Peter Henderson, Pierre Colombo, Priscilla A. Amuok, Quentin Lhoest, Rheza Harliman, Rishi Bommasani, Roberto L’opez, Rui Ribeiro, Salomey Osei, Sampo Pyysalo, Sebastian Nagel, Shamik Bose, Shamsuddeen Hassan Muhammad, Shanya Sharma, S. Longpre, Somaieh Nikpoor, Stanislav Silberberg, Suhas Pai, Sydney Zink, Tiago Timponi Torrent, Timo Schick, Tristan Thrush, Valentin Danchev, Vassilina Nikoulina, Veronika Laippala, Violette Lepercq, Vrinda Prabhu, Zaid Alyafeai, Zeerak Talat, Arun Raja, Benjamin Heinzerling, Chenglei Si, Elizabeth Salesky, Sabrina J. Mielke, Wilson Y. Lee, Abheesht Sharma, Andrea Santilli, Antoine Chaffin, Arnaud Stiegler, Debajyoti Datta, Eliza Szczechla, Gunjan Chhablani, Han Wang, Harshit Pandey, Hendrik Strobelt, Jason Alan Fries, Jos Rozen, Leo Gao, Lintang Sutawika, M Saiful Bari, Maged S. Al-shaibani, Matteo Manica, Nihal V. Nayak, Ryan Teehan, Samuel Albanie, Sheng Shen, Srulik Ben-David, Stephen H. Bach, Taewoon Kim, Tali Bers, Thibault Févry, Trishala Neeraj, Urmish Thakker, Vikas Raunak, Xiang Tang, Zheng Xin Yong, Zhiqing Sun, Shaked Brody, Y Uri, Hadar Tojarieh, Adam Roberts, Hyung Won Chung, Jaesung Tae, Jason Phang, Ofir Press, Conglong Li, Deepak Narayanan, Hatim Bourfoune, Jared Casper, Jeff Rasley, Max Ryabinin, Mayank Mishra, Minjia Zhang, Mohammad Shoeybi, Myriam Peyrounette, Nicolas Patry, Nouamane Tazi, Omar Sanseviero, Patrick von Platen, Pierre Cornette, Pierre Franccois Lavall’ee, Rémi Lacroix, Samyam Rajbhandari, Sanchit Gandhi, Shaden Smith, Stéphane Requena, Suraj Patil, Tim Dettmers, Ahmed Baruwa, Amanpreet Singh, Anastasia Cheveleva, Anne-Laure Ligozat, Arjun Subramonian, Aur’elie N’ev’eol, Charles Lovering, Daniel H Garrette, Deepak R. Tunuguntla, Ehud Reiter, Ekaterina Taktasheva, Ekaterina Voloshina, Eli Bogdanov, Genta Indra Winata, Hailey Schoelkopf, Jan-Christoph Kalo, Jekaterina Novikova, Jessica Zosa Forde, Xiangru Tang, Jungo Kasai, Ken Kawamura, Liam Hazan, Marine Carpuat, Miruna Clinciu, Najoung Kim, Newton Cheng, Oleg Serikov, Omer Antverg, Oskar van der Wal, Rui Zhang, Ruochen Zhang, Sebastian Gehrmann, Shachar Mirkin, S. Osher Pais, Tatiana Shavrina, Thomas Scialom, Tian Yun, Tomasz Limisiewicz, Verena Rieser, Vitaly Protasov, Vladislav Mikhailov, Yada Pruksachatkun, Yonatan Belinkov, Zachary Bamberger, Zdenvek Kasner, Alice Rueda, Amanda Pestana, Amir Feizpour, Ammar Khan, Amy Faranak, Ananda Santa Rosa Santos, Anthony Hevia, Antigona Unldreaj, Arash Aghagol, Arezoo Abdollahi, Aycha Tammour, Azadeh HajiHosseini, Bahareh Behroozi, Benjamin Olusola Ajibade, Bharat Kumar Saxena, Carlos Muñoz Ferrandis, Danish Contractor, David M. Lansky, Davis David, Douwe Kiela, Duong Anh Nguyen, Edward Tan, Emily Baylor, Ezinwanne Ozoani, Fatim T Mirza, Frankline Ononiwu, Habib Rezanejad, H.A. Jones, Indrani Bhattacharya, Irene Solaiman, Irina Sedenko, Isar Nejadgholi, Jan Passmore, Joshua Seltzer, Julio Bonis Sanz, Karen Fort, Lívia Macedo Dutra, Mairon Samagaio, Maraim Elbadri, Margot Mieskes, Marissa Gerchick, Martha Akinlolu, Michael McKenna, Mike Qiu, M. K. K. Ghauri, Mykola Burynok, Nafis Abrar, Nazneen Rajani, Nour Elkott, Nourhan Fahmy, Olanrewaju Samuel, Ran An, R. P. Kromann, Ryan Hao, Samira Alizadeh, Sarmad Shubber, Silas L. Wang, Sourav Roy, Sylvain Viguier, Thanh-Cong Le, Tobi Oyebade, Trieu Nguyen Hai Le, Yoyo Yang, Zachary Kyle Nguyen, Abhinav Ramesh Kashyap, Alfredo Palasciano, Alison Callahan, Anima Shukla, Antonio Miranda-Escalada, Ayush Kumar Singh, Benjamin Beilharz, Bo Wang, Caio Matheus Fonseca de Brito, Chenxi Zhou, Chirag Jain, Chuxin Xu, Clémentine Fourrier, Daniel Le’on Perin’an, Daniel Molano, Dian Yu, Enrique Manjavacas, Fabio Barth, Florian Fuhrimann, Gabriel Altay, Giyaseddin Bayrak, Gully A. Burns, Helena U. Vrabec, Iman I.B. Bello, Isha Dash, Ji Soo Kang, John Giorgi, Jonas Golde, Jose David Posada, Karthi Sivaraman, Lokesh Bulchandani, Lu Liu, Luisa Shinzato, Madeleine Hahn de Bykhovetz, Maiko Takeuchi, Marc Pàmies, María Andrea

Castillo, Marianna Nezhurina, Mario Sanger, Matthias Samwald, Michael Cullan, Michael Weinberg, M Wolf, Mina Mihaljcic, Minna Liu, Moritz Freidank, Myungsun Kang, Natasha Seelam, Nathan Dahlberg, Nicholas Michio Broad, Nikolaus Muellner, Pascale Fung, Patricia Haller, R. Chandrasekhar, R. Eisenberg, Robert Martin, Rodrigo L. Canalli, Rosaline Su, Ruisi Su, Samuel Cahyawijaya, Samuele Garda, Shlok S Deshmukh, Shubhanshu Mishra, Sid Kiblawi, Simon Ott, Sinee Sang-aroonsiri, Srishti Kumar, Stefan Schweter, Sushil Pratap Bharati, T. A. Laud, Th’eo Gigant, Tomoya Kainuma, Wojciech Kusa, Yanis Labrak, Yashasvi Bajaj, Y. Venkatraman, Yifan Xu, Ying Xu, Yun chao Xu, Zhee Xao Tan, Zhongli Xie, Zifan Ye, Mathilde Bras, Younes Belkada, and Thomas Wolf. BLOOM: A 176B-Parameter Open-Access Multilingual Language Model. ArXiv, abs/2211.05100, 2022. URL https://arxiv.org/abs/2211.05100.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https://openreview.net/forum?id=M3Y74vmsMcY.

Roy Schwartz, Jesse Dodge, Noah A. Smith, and Oren Etzioni. Green ai. Commun. ACM, 63(12): 54–63, nov 2020. ISSN 0001-0782. URL https://doi.org/10.1145/3381831.

Preethi Seshadri, Sameer Singh, and Yanai Elazar. The bias amplification paradox in text-to-image generation. arXiv preprint arXiv:2308.00755, 2023. URL https://arxiv.org/abs/2308.00755.

Jaime Sevilla, Lennart Heim, Anson Ho, Tamay Besiroglu, Marius Hobbhahn, and Pablo Villalobos. Compute trends across three eras of machine learning. In 2022 International Joint Conference on Neural Networks (IJCNN), pp. 1–8, 2022. URL https://ieeexplore.ieee.org/abstract/ document/9891914.

Seongjin Shin, Sang-Woo Lee, Hwijeen Ahn, Sungdong Kim, HyoungSeok Kim, Boseop Kim, Kyunghyun Cho, Gichang Lee, Woomyoung Park, Jung-Woo Ha, and Nako Sung. On the effect of pretraining corpora on in-context learning by a large-scale language model. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 5168–5186, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.380. URL https: //aclanthology.org/2022.naacl-main.380.

Daniel Simig, Tianlu Wang, Verna Dankers, Peter Henderson, Khuyagbaatar Batsuren, Dieuwke Hupkes, and Mona Diab. Text characterization toolkit (TCT). In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing: System Demonstrations, pp. 72–87, Taipei, Taiwan, November 2022. Association for Computational Linguistics. URL https://aclanthology.org/2022.aacl-demo.9.

Luca Soldaini and Kyle Lo. peS2o (Pretraining Efficiently on S2ORC) Dataset. Technical report, Allen Institute for AI, 2023. ODC-By, https://github.com/allenai/pes2o.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Raghavi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, A. Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Daniel Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hanna Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159, 2024. URL https://arxiv.org/abs/2402.00159.

Nishant Subramani, Sasha Luccioni, Jesse Dodge, and Margaret Mitchell. Detecting personal information in training corpora: an analysis. In Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023), pp. 208–220, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.trustnlp-1.18. URL https://aclanthology.org/2023.trustnlp-1.18.

MosaicML NLP Team. Introducing mpt-7b: A new standard for open-source, commercially usable llms, 2023. URL www.mosaicml.com/blog/mpt-7b. Accessed: 2023-05-05.

Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari S Morcos. D4: Improving llm pretraining via document de-duplication and diversification. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

Together Computer. RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset, April 2023. URL https://github.com/togethercomputer/RedPajama-Data.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971, 2023. URL https://arxiv.org/abs/2302.13971.

Unicode. Unicode Text Segmentation, Aug 2023. URL https://unicode.org/reports/tr29/.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/ file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf.

Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 483–498, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.41. URL https: //aclanthology.org/2021.naacl-main.41.

Rui Zhang and Joel Tetreault. This email could save your life: Introducing the task of email subject line generation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 446–456, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1043. URL https://aclanthology.org/P19-1043.

Xuhui Zhou, Maarten Sap, Swabha Swayamdipta, Yejin Choi, and Noah Smith. Challenges in automated debiasing for toxic language detection. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 3143–3155, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eaclmain.274. URL https://aclanthology.org/2021.eacl-main.274.

- A CORPORA: ELABORATION

We cover ten different corpora, including text-only corpora (e.g., C4), captions from image-captioning (LAION-2B-en), and code (The Stack). A high level description of these corpora using WIMBD is presented in Table 2, and details about the information contained in those corpora are detailed in Table 6.

We analyze all corpora fully, including the different subsets (e.g., The Pile is constructed of multiple sources, such as Wikipedia, arXiv, etc.). The only exceptions are mC4, and LAION, which the original released data consist of non-English texts as well, and we focus on the English subset. Note that while we focus on English text corpora, most of our analyses are not language dependent, and can be easily applied to other languages as well. The only exception is the toxic language analysis (§B.3.3) that relies on an English lexicon and classifier. However, we note that given non-English lexicon and classifier, the analysis can be easily repeated for other languages using our framework.

OPENWEBTEXT is an open-source reproduction8 (Gokaslan & Cohen, 2019) of the data used to train GPT-2 (Radford et al., 2019). Due to the limited information provided by Radford et al. (2019), and never releasing the data, it is unclear how similar OpenWebText is to the original data (WebText), but similar steps to the paper’s reports were conducted (such as deduplication, non-English filtering, min-length filtering, etc.).

C4 is the dataset used by Raffel et al. (2020) for training T5. The dataset: The Colossal Clean Crawled Corpus (C4 in short) is based on Common Crawl as a source of text that was scraped from the web. As such, a lot of the data is noisy, and a set of heuristics were employed to clean it up, such as filtering documents by length, obscene/bad words, duplicate texts, non-english, etc. C4 was not released by Raffel et al. (2020), and instead, it was scraped, cleaned, filtered, and released by Dodge

- et al. (2021).

MC4-EN is a multilingual version of C4 that was used to train mT5 (Xue et al., 2021), and later umT5 (Chung et al., 2023). We use the latest version (v.3.1.0) which was used to train umT5, containing documents collected from Common Crawl through August 2022, and in practice the portion of the data that is classified as English. The main difference of mC4-en over C4 is a higher confidence by a language classifier (from 0.7 to 0.96), while also allowing a 0.1% random set of documents that contain “bad words” to pass through, and adaptation of the “bad words” list that resulted in filtering more than 10% of the documents in a language.

OSCAR is a multilingual corpus based on Common Crawl (Abadji et al., 2022). It contains a length filter for improving data quality that filters out documents with short sentences. They also annotate the data with different labels, such as the language of the document, adult content, and language identification, which they use for different analyses. It is an ongoing effort, and the corpus is maintained and updated regularly.

THE PILE is a corpus consisting of 22 different domains (Gao et al., 2020). Unlike C4, the data was not scrapped from the web and then filtered, but pre-selected, with the motivation that this way the data will be of higher quality. The included domains in The Pile are diverse: they include data such as Wikipedia, Github, Arxiv, EuroParl, and more. By design, most datasets are upsampled in the hope to increase data quality, from 1.5x with domains such as OpenSubtitles, up to 3x with Wikipedia. Models such as GPT-J (Wang & Komatsuzaki, 2021), GPT-neo (Black et al., 2022) and Pythia (Biderman et al., 2023) were trained on this dataset.

REDPAJAMA is an open-source version reproduction of the data used to train LLaMA (Touvron et al., 2023), and was used to train RedPajama-INCITE (Together Computer, 2023).

S2ORC is a large corpus of English academic papers, which consists the abstracts, full text, including figures, tables, and references (Lo et al., 2020). The texts are automatically extracted from pdfs and LATEX sources.

8skylion007.github.io/OpenWebTextCorpus

PES2O is a derivative of S2ORC, cleaned and filtered to obtain a more usable version of the data intended to train language models. We use peS2o V2 (Soldaini & Lo, 2023).

LAION is a large dataset of images and captions scraped from Common Crawl (Schuhmann et al.,

- 2022). The main dataset (LAION-5B) contains 5.8 billion examples, of which 2.32 billion of the captions are in English (LAION-2B-en), which we use in this work. We focus on the text captions but demonstrate qualitative examples using the associated URLs and images when appropriate.

THE STACK (Kocetkov et al., 2023) is a source-code dataset that was collected for training language models, and parts of it were used to train SantaCoder (Allal et al., 2023) and MPT (Team, 2023). It was compiled from GHArchive9 with some filters: files that cannot contribute to training code such as binary files, files larger than 1MB, and some extensions. In addition, only repositories with permissive licenses were included (18 license types in the version v1.0, and 193 in version v1.1), and we use the v1.2. While the main purpose of code is to provide machine instructions to perform different functionalities, it also contain natural language in the form of comments: “Roughly 40 natural languages are present in docstrings and comments with English being the most prevalent. In python files, it makes up 96% of the dataset.”

- Table 6: Metadata information contained in the ten corpora we consider. Text refers to the main information contained in those datasets, while the type of text is different, e.g. The Stack contains source code, and LAION2B-en descibes images. URL indicates the URL that the document was collected from, or in the case of LAION2B-en, the link to the image that the text refers to. Scrape Date is the date that the document was scraped from the web, Date Added is the date the data was incorporated into the corpora. Domain/Lang indicates a subcategory of the text (e.g. field of study, the source from The Pile, code language in The Stack). ID is the document ID. Has Split signifies whether or not the released data contains a train-test split.

###### Corpus Text Url Scrape Date Date Added Domain/Lang ID Has Split

OpenWebText ✓ ✗ ✗ ✗ ✗ ✓ ✗ C4 ✓ ✓ ✓ ✗ ✗ ✗ ✓ mC4-en ✓ ✓ ✓ ✓ ✓ ✓ ✓ OSCAR ✓ ✓ ✓ ✗ ✓ ✓ ✗ The Pile ✓ ✗ ✗ ✗ ✓ ✗ ✓ RedPajama ✓ ✓

✓ ✓ ✓ ✓ ✗ S2ORC ✓ ✗ ✓ ✓ ✓ ✓ ✗ peS2o ✓ ✗ ✓ ✓ ✓ ✓ ✓ LAION-2B-en ✓ ✓

–

✗ ✗ ✗ ✓ ✗ The Stack ✓ ✗ ✓ ✓ ✓ ✓ ✗

–

9https://gharchive.org/

Corpus 1 25 50 75 99 N. C4 26 264 964 3,886 137,117 15,668,300 OSCAR 21 303 1,351 6,108 440,577 15,424,393 LAION-2B-en 1 6 11 25 892 1,470,243 mC4-en 48 580 1,448 5,984 477,951 62,209,454 RedPajama 26 264 963 3,882 136,937 15,658,463

- Table 7: Internet domain quantiles of each corpora with URL information. The values correspond to the number of tokens from each internet domain quantile. N. corresponds to the number of unique internet domains.

- B ADDITIONAL RESULTS

We provide additional details and extended results on all the corpora considered in this work. This appendix is structured in a similar way to the structure in the main paper, categorized by the four different high-level analyses: (1) Data Statistics (Appendix B.1), (2) Data Quality (Appendix B.2), (3) Community- and Society-Relevant Measurements (Appendix B.3), and (4) Cross-Data Analysis (Appendix B.4).

B.1 DATA STATISTICS

The summary statistics are composed of different analyses that mainly involve the additional metadata associated with the textual documents, such as the URL from which the document was extracted, the date it was collected, etc. We also consider some raw statistics about the corpora, described in the main paper (4.2). The analyses we propose for data statistics are the following:

- 1. Summary statistics (§4.2)
- 2. Internet domain distribution (§4.2.2, §B.1.1)
- 3. Internet domain schemes (§B.1.2)
- 4. Internet domain suffixes (§B.1.3)
- 5. Utterance date statistics (§B.1.4)
- 6. Geolocation (§B.1.5)
- 7. Language distribution (§B.1.6)

- B.1.1 INTERNET DOMAIN DISTRIBUTION

Here, we provide complete analyses on the five corpora that contain URL information in the corpus metadata. Using the Exact Counts , we conduct two analyses: (1) each domain is counted per document (yielding documents per domain), and another where each domain is counted per token in the document (yielding tokens per domain). The results are presented in Figure 6, where the (1) document per domain figures are presented on the left, and the (2) document per token figures are presented on the right.

In Table 7, we analyze the number of tokens in each domain, and calculate the 1, 25, 50, 75, and 99 quantiles of these distributions. Interestingly, the 1% quantile in LAION-2B-en include domains which have 1-or-less tokens.

- B.1.2 INTERNET DOMAIN SCHEMES

This analysis computes the domain schemes of the associated URLs using the Exact Counts . The results are presented in Figure 7. HTTP and HTTPS are two internet protocols, with the latter being an extension of the first that provides more secure communication. While the exact portion of websites across the web that uses each protocol is hard to assess, traffic that goes through Google primarily uses HTTPS - 95%.10.

10https://transparencyreport.google.com/https/overview, as of September 16th, 2023.

The trend of recent years shows an increase in the portion of HTTPS-supported websites, and as such, we can use this portion as a proxy for the internet age of a website: HTTP websites are more likely to be older. In addition, the portion of a corpus is an interesting comparison with the reported portion from Google’s traffic.

All corpora containing URL information show significant proportions from Google’s reports of 95% for the HTTPS protocol. OSCAR contains the highest proportion with 87.6% HTTPS URLs, while

- C4 is the lowest with only 62.5%.

- B.1.3 INTERNET DOMAIN SUFFIXES

Next, we compute the suffix distribution of the different corpora using the Exact Counts and present the results of the ten most common ones in Figure 8. Compared to the internet domain distribution, the suffixes provide us with a higher-level description of the sources of the documents.

Perhaps not surprisingly, the most common suffix is com, which is between 60.1% of the documents in OSCAR and 77.5% in LAION-2B-en. The distribution of suffixes for each dataset exhibits a long tail with a total of over 3,000 different suffixes in the different corpora. While the top 10 typically represent suffixes from English-speaking countries (e.g., co.uk, and ca), LAION-2B-en’s top-10 contains a lot of non-English speaking countries as well, such as Germany (de, 0.7%), Russia (ru, 0.5%), France (fr, 0.4%) and Italy (it, 0.4%).

- B.1.4 UTTERANCE DATE STATISTICS

In this section, we examine the temporal diversity of documents from corpora with either reliable creation timestamps in their metadata or URL source information from which creation time can be estimated. Language usage drifts, new concepts are introduced over time, and the truth of much commonsense knowledge depends on the date an utterance was made. While some datasets we consider (S2ORC and peS2o) have reliable, API-generated creation timestamps, most have creation dates that reflect the time of a document ingestion into the source dataset and not its origin date (C4, mC4-en, RedPajama, and LAION-2B-en). To characterize their temporal distribution, we directly count and bin documents by year for those with reliable creation time metadata. For datasets without this information, we fall back on using either the earliest date the URL associated with a document was indexed by the Internet Archive or the date of ingestion into the dataset (whichever is earlier).11 Note that such a procedure does not provide us with the timestamp of the document that was scraped, and as such, it serves as a lower bound on the document’s time creation. Given the limitations of the Internet Archive’s API, we do this for a 10,000 document random sample of each dataset, which allows a rough estimate of the collection time for documents in these corpora. Results are shown in Figure 9. We can see that RedPajama and OSCAR are dominated by documents created in the previous five years (as of September 2023), while other datasets have a more substantial proportion of documents from the first half of the 2010s and earlier. Notably, S2ORC and pes2o contain a non-negligible fraction of documents from the pre-internet era.

- B.1.5 GEOLOCATION

In this section, we gauge the geographic diversity of corpora with URL source information in their metadata. We use a commercially developed IP database 12 to estimate the country of origin for 100,000 randomly sampled URLs from each of the five corpora with this information included. While there are limitations to using the location of a hosting server as a stand-in for the content creator’s location (i.e., websites are not always hosted locally nor in one unique location), it does provide a rough geographic origin for source material. As seen in Figure 10, most web pages across corpora are hosted in the United States, with the bulk of the remainder distributed amongst the anglosphere. This is unsurprising given the focus on English-language sources in the construction of the corpora under consideration.

Table 8: Percentage of documents in English per dataset.

Corpus Percentage OpenWebText 99.68 C4 99.67 mC4-en 99.56 OSCAR 99.92 The Pile 96.12 RedPajama 96.93 S2ORC 96.44 peS2o 100.00 LAION-2B-en 95.90

- B.1.6 LANGUAGE DISTRIBUTION

Here, we aim to assess the proportion of languages in all corpora. We use the CLD213 classifier to make a prediction about what language is being used in each document, and use this prediction as a label that we analyze in aggregate. Note that we use the classifier label also in mixed-language documents (if CLD2’s is_reliable flag is False, we apply the label UN). Table 8 reports the percentages of English-language documents across corpora. As expected, the English fraction is quite high, given the targeted construction of most datasets we consider. The remaining percentages of non-English documents are broken down for the ten remaining most common languages in Figure 11. Note that the classifier we use, as with other classifiers, is imperfect, and as such the identified languages may be wrong.

- 11The Internet Archive is a massive library that has been preserving the web since 1996. https:

//archive.org

- 12This work includes IP2Location LITE data available from https://lite.ip2location.com
- 13https://github.com/CLD2Owners/cld2

C4 Domains per Document

www.nytimes.com

en.wikipedia.org

do5.b00kmedia.ru

www.latimes.com

www.theguardian.com

www.huffpost.com

patents.google.com

www.businessinsider.com

www.forbes.com

www.eventbrite.com

0.00 0.01 0.02 0.03 0.04

% of Documents

C4 Domains per Token

patents.google.com

en.wikipedia.org

en.m.wikipedia.org

www.nytimes.com

journals.plos.org

www.latimes.com

www.theguardian.com

www.forbes.com

www.huffpost.com

www.scribd.com

0.0 0.1 0.2 0.3 0.4

% of Documents

mC4-en Domains per Document

www.google.com

www.tripadvisor.com

www.ebay.com

www.walmart.com

www.tripadvisor.co.uk

en.wikipedia.org

finance.yahoo.com

www.thefreedictionary.com www.groupon.com www.ebay.co.uk

0.00 0.05 0.10 0.15 0.20

% of Documents

OSCAR Domains per Document

pubmed.ncbi.nlm.nih.gov

www.theguardian.com

unistore.www.microsoft.com

us.vestiairecollective.com

imgur.com

www.reuters.com

espas.secure.europarl.europa.eu

www.forbes.com

www.afternic.com:443

millenniumindicators.un.org

0.00 0.01 0.02 0.03

% of Documents

RedPajama Domains per Document

stackoverflow.com

en.wikipedia.org

de.wikipedia.org

sv.wikipedia.org

fr.wikipedia.org

nl.wikipedia.org

ru.wikipedia.org

it.wikipedia.org

es.wikipedia.org

arxiv.org

0 1 2 3 4 5

% of Documents

mC4-en Domains per Token

www.google.com

patents.google.com

www.patentsencyclopedia.com

www.tripadvisor.com

www.scribd.com

www.walmart.com

www.slideshare.net

issuu.com

patents.justia.com

www.tripadvisor.co.uk

0.00 0.25 0.50 0.75 1.00 1.25

% of Documents

OSCAR Domains per Token

www.drroyspencer.com

esr.ibiblio.org

smittenkitchen.com

worldwidescience.org

www.dailymail.co.uk

driftingthrough.com

downtown.utk.edu

usawatchdog.com

tim.blog

archives.augsburg.edu

0.00 0.05 0.10 0.15 0.20 0.25 0.30

% of Documents

RedPajama Domains per Token

arxiv.org

stackoverflow.com

en.wikipedia.org

www.gutenberg.org

de.wikipedia.org

fr.wikipedia.org

es.wikipedia.org

ru.wikipedia.org

math.stackexchange.com

it.wikipedia.org

0 2 4 6 8 10 12

% of Documents

LAION-2B-en Domains per Document

cdn.shopify.com

i.pinimg.com

i.ebayimg.com

images-na.ssl-images-amazon.com

www.specsserver.com

thumbs.dreamstime.com

i0.wp.com

render.fineartamerica.com

i.ytimg.com

images.slideplayer.com

0 1 2 3 4 5 6

% of Documents

LAION-2B-en Domains per Token

i.pinimg.com

cdn.shopify.com

images.slideplayer.com

images-na.ssl-images-amazon.com

i.ebayimg.com

ssl.c.photoshelter.com

ae01.alicdn.com

media.gettyimages.com

thumbs.dreamstime.com

us.123rf.com

0 1 2 3 4 5 6

% of Documents

###### Figure 6: Internet domain distributions of the ten most common domains for each corpus.

C4 Schemes

mC4-en Schemes

OSCAR Schemes

70

62.5

66.7

87.6

60

80

60

50

50

60

%ofDocuments

%ofDocuments

%ofDocuments

40

37.5

40

33.3

30

40

30

20

20

20

10

12.4

10

0

0

0

https http

https http

https http

Scheme

Scheme

Scheme

RedPajama Schemes

70

67.9

60

50

%ofDocuments

40

32.1

30

20

10

0

https http

Scheme

LAION-2B-en Schemes

80.1

80

70

60

%ofDocuments

50

40

30

19.9

20

10

0

https http

Scheme

###### Figure 7: Schema distributions of the ten most common domains for each corpus. We show the results for the five corpora that contain URL information.

C4 Suffixes

mC4-en Suffixes

OSCAR Suffixes

64.6

65.5

60.1

60

60

60

50

50

50

%ofDocuments

%ofDocuments

%ofDocuments

40

40

40

30

30

30

20

20

20

9.7

10

10

10

8.8

7.1

5.0

4.4 3.4

4.0 3.5

3.5

2.4 1.7 1.4 0.8 0.8 0.6

1.9 1.5 1.3 1.1 0.6 0.6

1.5 1.4 1.2 1.1 0.8 0.8

0

0

0

com org co.uk net edu com.au ca de gov org.uk

com org co.uk net com.au edu ca info org.uk in

com org co.uk net com.au edu info ca in us

Suffix

Suffix

Suffix

RedPajama Suffixes

62.3

60

50

%ofDocuments

40

30

20

14.9

10

4.3

3.0

1.6 1.3 1.1 0.9 0.5 0.5

0

com org co.uk net com.au edu ca info org.uk in

Suffix

LAION-2B-en Suffixes

80

77.5

70

60

%ofDocuments

50

40

30

20

10

8.1

2.4 1.7 0.9 0.7 0.6 0.5 0.4 0.4

0

com net co.uk org com.au de ca ru fr it

Suffix

###### Figure 8: Suffix distributions of the ten most common domains for each corpus. We show the results for the five corpora that contain URL information.

100

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

80

2023 2022 2021 2020 2019 2018

%ofDocuments

60

2010-2017 2000-2009 1990-1999 pre-1990

40

20

0

C4* mC4-en* OSCAR*RedPajama*S2ORC peS2oLAION-2B-en*

- Figure 9: Fraction of documents in each corpus produced per year. Corpora marked with * are estimates based on the Internet Archive index dates for a 10,000 document sample.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

C4 mC4-en OSCAR RedPajama LAION-2B-en

0

20

40

60

80

100

%ofDocuments

US UN Other

CA GB DE AU FR IE

NL SE

(a) Percentage of URLs by country

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

C4 mC4-en OSCAR RedPajama LAION-2B-en

0

20

40

60

80

100

%ofDocuments

US

Other

CA GB DE AU FR IE

NL SE

(b) Percentage of URLs (excluding unresolved URLS)

- Figure 10: Percentage of documents for each dataset originating in a given country. Only the nine most common countries across corpora are shown with the remainder combined in ’other.’ We label URLs we were unable to geolocate as UN (Unknown), and provide results with and without these documents included.

3.0

4.0

un

nl

nl

de es

3.5

2.5

de es

id ja pt pl fr zh-Hant

3.0

id ja pt pl fr zh-Hant

%ofDocuments

%ofDocuments

2.0

2.5

1.5

2.0

da

1.5

1.0

1.0

0.5

0.5

0.0

0.0

OpenWebTextC4 mC4-enOSCARThePileRedPajamaS2ORC peS2oLAION-2B-en

OpenWebTextC4 mC4-enOSCARThePileRedPajamaS2ORC peS2oLAION-2B-en

(a) Non-English language content

(b) Non-English language content excluding unknown languages

Figure 11: Percentage of non-English language documents detected in each corpus.

###### Table 9: Most common unigrams, bigrams and trigrams and their estimated counts.

OpenWebText C4 mC4-en OSCAR The Pile RedPajama S2ORC peS2o LAION-2B-en The Stack n-gram Count n-gram Count n-gram Count n-gram Count n-gram Count n-gram Count n-gram Count n-gram Count n-gram Count n-gram Count Unigrams

, 342M the 4.29B to 4.29B to 4.29B to 4.29B with 4.29B the 2.77B the 2.13B - 1.13B } 4.29B the 331M . 4.29B the 4.29B the 4.29B the 4.29B to 4.29B , 2.64B , 1.9B , 870M { 4.29B . 323M , 4.29B of 4.29B of 4.29B of 4.29B the 4.29B . 2.3B . 1.69B . 578M the 4.29B to 177M and 3.87B and 4.29B in 4.29B and 4.29B that 4.29B of 1.74B of 1.35B " 455M n 4.29B of 169M to 3.67B a 4.29B and 4.29B . 4.29B on 4.29B and 1.36B and 1.05B the 352M class 4.29B and 157M of 3.29B . 4.29B a 4.29B - 4.29B of 4.29B ) 1.11B ) 769M of 341M a 4.29B a 142M a 2.79B - 4.29B . 4.29B , 4.29B is 4.29B ( 1.11B in 766M and 320M ] 4.29B in 115M in 2.17B , 4.29B - 4.29B ) 4.29B in 4.29B - 1.02B ( 764M in 306M \ 4.29B - 91.3M is 1.6B " 4.29B , 4.29B " 4.29B for 4.29B in 985M - 749M / 249M [ 4.29B that 74.9M - 1.49B : 4.25B is 4.26B ( 4.28B as 4.29B to 904M to 705M : 247M > 4.29B

###### Bigrams

of the 39.6M of the 740M of the 4.29B of the 1.85B - - 4.29B of the 4.29B of the 433M of the 333M " " 257M } , 4.29B in the 29.2M . The 608M in the 4.29B , and 1.5B of the 1.3B , and 3.65B . The 302M . The 233M . . 96.5M { " 4.29B , and 29M , and 565M . The 4.29B . . 1.37B = = 1.02B in the 3.46B ) . 281M in the 208M of the 58.2M class = 4.29B . The 27.1M in the 523M . . 4.29B in the 1.28B . " 881M . The 3.38B in the 267M ) . 206M in the 39.5M ] , 4.29B , the 19.5M to the 321M , and 4.29B . The 1.17B , and 873M . . 2.54B , and 239M , and 181M T - 27.8M > < 4.29B to the 16.8M , the 296M , " 4.29B to the 825M * * 859M , the 2.15B , the 209M , the 162M at the 25.2M = = 4.29B . " 16.5M on the 257M " : 4.29B 774M in the 805M to the 2.06B ) , 164M to the 116M for sale 22.4M = " 4.29B

, but 13.2M . I 250M to the 4.09B , the 704M . The 793M on the 1.48B to the 151M ) , 111M , and 22.4M < / 4.29B on the 12.8M for the 208M , the 3.82B . I 674M " " 774M and the 1.32B ] . 134M ] . 104M on the 20.8M ; } 4.29B

. “ 10.9M . This 200M " , 3.6B on the 641M { \ 576M for the 1.27B . In 126M . In 97.1M - Shirt 19.6M : { 4.29B Trigrams

- - - 4.67M . . . 77.7M . . . 4.29B 774M - - - 4.26B . . . 1.62B et al . 98.6M et al . 76.3M " " " 123M class = " 4.29B . . . 4.6M . If you 63.5M " , " 2.93B . . . 735M = = = 926M - - - 686M al . , 50.7M al . , 38.6M . . . 49.2M > < / 4.29B , and the 2.46M . It is 52.8M " : " 2.71B \ \ \ 397M . " " 473M : / / 472M ) . The 44.5M ) . The 34M T - Shirt 19.4M : { " 4.29B one of the 2.42M as well as 50.8M : / / 1.84B - - - 248M * * * 303M * * * 326M . However , 35.6M . However , 28.3M < br / 11.5M - - - 4.29B a lot of 1.74M one of the 48.8M - - - 1.33B : / / 218M . . . 288M > < / 322M q q q 32M , and the 22.5M br / > 11.5M * * * 4.29B . This is 1.52M . This is 43.5M http : / 939M . If you 176M # # # 136M , and the 311M , and the 29.6M . In the 18.2M for sale in 10.5M " > < 4.29B . It is 1.51M , and the 41.7M https : / 832M ( 1 ) 152M ? " " 133M one of the 287M . In the 23.7M ) , and 16.8M : / / 9.58M " : { 4.29B

, according to 1.47M . You can 38.7M as well as 675M https : / 130M type = " 126M ( 1 ) 252M ) , and 23.6M ( Fig . 16M Royalty Free Stock 9.3M " : " 4.29B . " The 1.46M . However , 32.3M . If you 663M . It is 128M ] ( # 117M \ \ \ 244M ( Fig . 21.9M ] . The 15.5M http : / 6.09M " , " 4.29B

as well as 1.46M a lot of 29.3M one of the 619M as well as 115M - type = 116M https : / 243M . . . 20.8M ) . In 14.2M KEEP CALM AND 5.42M = = = 3.98B

B.2 DATA QUALITY

While we reported all the different analyses under data quality in the main paper, here we elaborate and provide the full results on all corpora and the different variations (e.g., most common unigrams, bigrams, and length distribution on token level). The analyses we propose for data quality are the following:

- 1. Most and least common n-grams (§4.3.1, §B.2.1)
- 2. Duplicate (§4.3.2, §B.2.2)
- 3. Document length distribution (§4.3.3, §B.2.3)

B.2.1 MOST & LEAST COMMON n-GRAMS

Most common n-grams In addition to the most common 10-grams reported in Section 4.3.1, we report the results for the most common unigrams, bigrams, and trigrams. Stop words and punctuation are the most common unigrams across the different datasets, with some differences in their ranking. Moving to bigrams, we observe more differences between the corpora. For instance, in LAION-2B-en, we observe some marketing mentions, such as “for sale” and “- Shirt”. “of the” and “in the” are repeating bigrams in all corpora. In the trigram results, we notice a larger diversion between the corpora. C4 contains common English expressions, such as “one of the”, “a lot of”, and “as well as”. However, LAION-2B-en contains much more marketing material, such as “T

- Shirt”, “for sale in”. OSCAR and The Pile have many n-grams that look like uncleaned html (“: / /”, ‘https : /”, “type = "”) or markdown (“–-”, “===”, “###”).

Least common n-grams Similarly to the most common n-grams, we look at the other side of n-grams distribution on the least common in a corpus. We showcase a random set of 25 unique unigrams from the different corpora in Figures 12 and 13. We observe two noticeable trends from such unigrams: (1) non-standard Unicode fonts like “negative squared latin” (for instance COTD in mC4-en), and (2) non-English strings. Non-English strings are quite diverse. The sample from OpenWebText contains unigrams from 12 languages other than English: Urdu, Arabic, Korean, Sanskrit, Hebrew, Armenian, Bengali, Persian, Japanese, Latvian, Sindhi, and Russian.

In addition to the unique unigrams inspection, we estimate the number of unique unigrams in each corpus and present the results in Table 10. The unique unigrams results reveal that a non-trivial amount of unique unigrams appear in these corpora. Even the smallest corpus, OpenWebText, contains more than 88 million unique unigrams, about 1.1% of the total unigrams in this corpus. The ratio of unique unigrams is about an order of magnitude smaller in the other corpora, except for LAION-2B-en, with over 554 million unique unigrams, which constitute 1.8% of the total unigrams.

Table 10: Estimated unique unigrams, and their percentage of the total unigrams.

Corpus Count Percentage OpenWebText 88,551,499 1.1 C4 759,392,762 0.5 mC4-en 4,290,392,741 0.2 OSCAR 1,280,686,454 0.3 The Pile 1,809,241,096 0.6 RedPajama 2,530,085,090 0.2 S2ORC 287,196,445 0.5 peS2o 201,729,350 0.5 LAION-2B-en 554,850,812 1.9 The Stack 4,294,966,820 0.3

[Figure 9]

(a) OpenWebText

[Figure 10]

- (b) C4

[Figure 11]

- (c) mC4-en

[Figure 12]

- (d) OSCAR

[Figure 13]

- (e) The Pile

Figure 12: Unique unigrams in OpenWebText, C4, mC4-en, OSCAR, and The Pile.

[Figure 14]

- (a) RedPajama

[Figure 15]

- (b) S2ORC

[Figure 16]

- (c) peS2o
- (d) LAION-2B-en

[Figure 17]

- (e) The Stack

[Figure 18]

Figure 13: Unique unigrams in RedPajama, S2ORC, peS2o, LAION-2B-en, and The Stack.

- Table 11: Top 5 most occurring text duplicates from datasets with duplicates (OpenWebText and C4 don’t have any duplicate documents). Truncation for visualization is marked by [...].

Corpus Property #1 Duplicate #2 Duplicate #3 Duplicate #4 Duplicate #5 Duplicate

mC4-en

Text ’, ’text-align:left; color:w hite;background-color:#0 564d1;’] //}); // ly.show(); var i_type = $("#fa[...]

Tada has the world’s lea ding smart parking techn ology and has many of the world’s top experts. A hug [...]

4K Ultra-clear picture with exquisite picture quality, p lug and play, H.265/H.26 5+, Max.512G SD card[...]

’, ’text-align:left; color:w hite;background-color:#0 564d1;’] //}); // ly.show(); var i_type = $("#fa[...]

‘, marker.on(’click’, ma rkerClick); if(type==0 & & index==0){ marker.emit (’click’, { target: marker } [...]

Count 154 114 80 76 73

OSCAR

Text In order to login you must be registered. Registering takes only a few moments but gives you increas[...]

JavaScript is disabled. For a better experience, please enable JavaScript in your browser before pro[...]

Privacy & Cookies: This site uses cookies. By co ntinuing to use this website , you agree to their use[...]

JavaScript seems to be d isabled in your browser. For the best experience on our site, be sure to tur[...]

You may not have to, it is u p to the administrator of th e board as to whether you need to register i[...]

Count 1,790,064 989,919 854,143 786,678 673,136

The Pile

Text {\n "info" : {\n "version" : 1,\n "author" : "xcode"\n } \n}

\r\n\r\n\r\n \r\n\r\n\r\n\r\n \tC-Track E-Filing\r\n\t\r\n \t\r\n\t\r\n\t\t\r\n\r\n\t\r\n\t \r\n\t\r\n\t\r\n\r\n\t\r\n\t\t\r \n\t\r\n\t\r\n\t\r\n \r\n\r\n\t\ r\n\t\r\n\t\r\n[...]

/* Localized versions of Inf o.plist keys */\n\n

<?xml version="1.0" enco ding="UTF-8"?>\n<!DO CTYPE plist PUBLIC " -//Apple//DTD PLIST 1.0/ /EN" "http://[...]

Count 3,775 2,941 2,913 2,744 2,714

RedPajama

Text ACCEPTED\n\n#### Acc ording to\nInternational Plant Names Index\n\n## ## Published in\nnull\n\ n#### Original n[...]

SYNONYM\n\n#### According to\nThe Catalo gue of Life, 3rd January 2011\n\n#### Published in\nnull\n\n#### Ori[...]

ACCEPTED\n\n#### Acc ording to\nThe Catalogue of Life, 3rd January 2011\n \n#### Published in\nnul l\n\n#### Or[...]

ACCEPTED\n\n#### Acc ording to\nNUB Generator [autonym]\n\n#### Publi shed in\nnull\n\n#### Or iginal name\nnull[...]

ACCEPTED\n\n#### Acc ording to\nInterim Register of Marine and Nonmarine Genera\n\n#### Published in\nnull\n[...]

Count 213,922 146,434 94,922 15,038 10,089

S2ORC

Text Abstract not submitted f or online publication\n\n\n\ n\n\u2022 Research which is freely available for red istrib[...]

Abstracts P1 - P16 are e ducational and not inclu ded for publication onli ne\n\n\n\n\nO R A L P R E S E N T[...]

Abstract withdrawn\n\n\n \n\u2022 Convenient onli ne submission \u2022 Tho rough peer review \n\u20 22 No space constraints [... ]

Educational abstract\n\nO1 Validation of a new autom ated volumetric breast d ensity measurement syste m [...]

Modeling and analysis of monkeypox disease using fractional derivatives\n\nT he frequency of monkeypo x [...]

Count 35 30 26 14 14

peS2o

Text Educational abstract\n\nO1 Validation of a new autom ated volumetric breast d ensity measurement syste m [...]

Reply on RC2\n\nThis man uscripts investigates the di screpancy of estimated v egetation influence on cat[. ..]

COP27 climate change con ference: urgent action n eeded for Africa and the world\n\nThe 2022 report of t[...]

Reply on RC2\n\nFollowin g your suggestion, we have revised the manuscript ve ry carefully. The lists be[. ..]

Reply on RC1\n\nThis pap er uses a 1D estuary model to explore the variability of overtide under varyin[...]

Count 14 7 6 4 4

LAION-2B-en

Text Front Cover Wall View 002 Market position of the s elected technologies

Pointwise: Reliable CFD meshing

Go to European Commi ssion website

Count 1,003,863 681,753 414,986 319,524 314,423

The Stack

Text #\n%\nRailCompiler: Inva lid movement.\n

//\n// WechatAuthSDK.h\ n// WechatAuthSDK\n//\n // Created by \u674e\u51ef on 13-11-29.\n// Copyright (c) 2013\u5e74 T[...]

OUTPUT_FORMAT (" elf32-littlearm", "elf32-big arm", "elf32-littlearm") \nENTRY(reset_handle r)\nSEARCH_DIR[...]

//\n// WBHttpRequest+We iboToken.h\n// WeiboSDK \n//\n// Created by Dannion Qiu on 14/11/6.\n// Cop yrigh[...]

//\n// WXApi.h\n// \u6240\ u6709Api\u63a5\u53e3 \n//\n// Created by Wechat on 12-2-28.\n// Copyright (c) 2012\u5e74 Tencent. A ll[...]

Count 45 43 29 24 20

- Table 12: Top 5 most occurring URL duplicates from datasets with URLs for each document and non-zero URL duplication.

LAION-2B-en OSCAR

Text Count Text Count UNLIKELY 33,142 https://international.thenewslens.com/tag/ 2,184 http://semantic.gs/driver_download_images/driver_download_certifications.png 27,162 https://arc.link/twitch/streaming/ 235 http://www.slickcar.com/products/hawkpadsa.jpg 10,700 https://zakiganj24news.blogspot.com/ 100 https://www.zeitauktion.info/assets/img/zeitauktion_placeholder.jpg 10,144 https://ywttvnews.com 100 https://static.uk.groupon-content.net/app/00/00/default0000.jpg 9,935 https://yellgh.com/our-services/ 100

- B.2.2 DUPLICATES

URL Duplicates We also examine duplication between document URLs for the datasets that have that metadata, which we show the top-5 URL duplicates from datasets with URL duplicates in Table 12. LAION’s most frequent URL (with 33,142 occurrences) is an invalid URL – “UNLIKELY”, likely resulting from a parsing error. The second most frequent URL (with 27,162 occurrences) from LAION-2B-en leads to an all-white image from a computer driver website, and in Figure 15, we see that among the top 25 duplicated URLs in LAION-2B-en, there are instances of image duplicates hosted at different URLs. Meanwhile, OSCAR has a notable artifact wherein, after the top two duplicate URLs, the next 234 URLs are duplicated exactly 100 times. Table 14 in the Appendix shows counts and ratios for these URL duplicates as previously specified for text hashes. These find that URL duplicate ratios are roughly an order of magnitude smaller than their text hash counterparts, and that the count of documents duplicated by URL is not dominated by only a few clusters.

139M

% of total uniq % of total

| |
|---|

60

1.2B

50

460M

64.6M

Duplicate%

40

165M

3.7M

219M

30

342M

20

1.8M

10

19.9M

33.9K

16.9K

22K 517K

232K

48.3K

0

mC4-en OSCAR ThePile RedPajama S2ORC peS2o LAION-2B-en TheStack

Figure 14: Percentages of text duplicates to totals for datasets with any. The percentages of documents and percentages of unique document clusters are each shown as bars. Duplicate counts are presented above the bars.

- Table 13: Statistics about text duplicates per dataset. Counts of duplicate documents and ratio of duplicate to total documents as well as equivalent counts for unique text clusters.

Corpus Duplicates Ratio of total Unique duplicates Uniq ratio of total OpenWebText 0 0.00 0 0.00 C4 0 0.00 0 0.00 mC4-en 48,255 0.00 21,991 0.00 OSCAR 164,740,386 0.38 19,934,531 0.07 The Pile 138,716,558 0.66 64,623,824 0.47 RedPajama 459,530,754 0.49 218,875,070 0.32 S2ORC 3,703,001 0.33 1,767,564 0.19 peS2o 33,903 0.00 16,924 0.00 LAION-2B-en 1,254,910,523 0.54 342,174,466 0.24 The Stack 517,396 0.00 232,151 0.00

- B.2.3 DOCUMENT LENGTH DISTRIBUTION

We elaborate on the results from the main paper and report the length distribution for all corpora, both for the character and token distribution. Figure 16 showcases these distributions, and Table 15 depicts the median token and character length distributions.

LAION-2B-en, containing image alt text, has the smallest average document lengths. Beyond the exact duplicates described above, which commonly describe products (especially home appliances), LAION-2B-en also contains a significant number of template-generated alt texts paired with maps describing the location of rental boats. The only outlier in OpenWebText in terms of document length

- Table 14: Statistics about URL duplicates for datasets with URLs for all documents. Counts of duplicate documents and ratio of duplicate to total documents as well as equivalent counts for unique URL clusters.

Corpus Duplicates Ratio of total Unique duplicates Unique ratio of total C4 0 0.00 0 0.00 mC4-en 0 0.00 0 0.00 OSCAR 5,958,969 0.01 2,542,577 0.01 LAION-2B-en 158,824,858 0.07 61,674,276 0.03

[Figure 19]

Figure 15: Images from the top 25 most duplicated URLs in LAION-2B-en.

is at exactly 100,000 characters; all documents over this length were chunked into multiple documents of length 100,000 by the dataset builders.

RedPajama also contains template-generated user-facing copy, including, e.g., placeholder pages for alumni of various secondary schools (each associated with a unique individual’s name). This analysis also reveals a collection of documents comprising nearly 0.01% of the dataset, containing what appear to be usernames or titles associated with pornographic content.

Finally, The Stack contains many template-generated new-duplicate documents; for example, a large number of auto-generated metadata files for Unity assets, each of length 20 tokens. It also contains a significant number of documents of length 20,000 characters that contain float and bit matrices.

The Pile also includes a significant number of auto-generated metadata files corresponding to Unity assets, e.g.:

|fileFormatVersion: 2 guid: e32f0a7fe2a7abc4289bc3c0e8a2b558 timeCreated: 1435687483 licenseType: Pro NativeFormatImporter: userData: assetBundleName: assetBundleVariant:|
|---|

as well as auto-generated files corresponding to publications in medical journals, e.g.:

|![](edinbmedj74198-0096){#sp1 .384}|
|---|

#### Characters Distribution

###### C4

mC4

###### OSCAR

The Pile

OpenWebText

0.07

0.5

0.08

0.04

0.8

0.06

0.07

0.4

0.06

0.03

0.05

0.6

0.05

0.3

0.04

0.04

0.02

0.4

0.03

0.2

0.03

0.02

0.02

0.01

0.2

0.1

0.01

0.01

0.0

0.0

0.00

0.00

0.00

101 102 103 104 105

102 103 104 105

102 103 104 105 106

101 103 105 107

103 104 105

Characters per Document

Characters per Document

Characters per Document

Characters per Document

Characters per Document

RedPajama

###### S2ORC

LAION-2B-en

The Stack

peS2o

0.200

0.0035

0.035

1.4

0.0035

0.175

0.0030

0.030

1.2

0.0030

0.150

0.0025

0.025

1.0

0.0025

0.125

0.0020

0.020

0.8

0.0020

0.100

0.0015

0.015

0.6

0.0015

0.075

0.0010

0.010

0.4

0.0010

0.050

0.0005

0.005

0.2

0.025

0.0005

0.0

0.000

0.000

0.0000

0.0000

101 103 105 107

101 102 103 104 105 106

100 101 102 103 104 105

101 103 105 107

103 104 105 106

Characters per Document

Characters per Document

Characters per Document

Characters per Document

Characters per Document

#### Tokens Distribution

###### C4

mC4

###### OSCAR

The Pile

OpenWebText

0.40

0.175

0.12

0.175

1.0

0.35

0.150

0.150

0.10

0.30

0.8

0.125

0.125

0.08

0.25

0.100

0.6

0.100

0.20

0.06

0.075

0.075

0.15

0.4

0.04

0.050

0.050

0.10

0.2

0.02

0.025

0.025

0.05

0.000

0.0

0.00

0.000

0.00

101 102 103 104 5

100 101 102 103 104 105

100 101 102 103 104 105 106

101 103 105 107

102 103 104

Tokens per Document

Tokens per Document

Tokens per Document

Tokens per Document

Tokens per Document

RedPajama

###### S2ORC

LAION-2B-en

The Stack

peS2o

0.8

0.175

0.016

0.0175

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7

0.7

0.014

0.150

0.0150

0.6

0.012

0.125

0.0125

0.5

0.010

0.100

0.0100

0.4

0.008

0.075

0.0075

0.3

0.006

0.050

0.0050

0.2

0.004

0.025

0.0025

0.1

0.002

0.0

0.000

0.000

0.0000

101 103 105 107

100 101 102 103 104 105

100 101 102 103 104 105

101 103 105 107

103 104 105

Tokens per Document

Tokens per Document

Tokens per Document

Tokens per Document

Tokens per Document

Figure 16: Distribution of document lengths for each of the datasets.

Table 15: Median document lengths for tokens and characters.

Corpus Median Token per Document Median Character per Document OpenWebText 634 3,185 C4 227 1,153 mC4-en 397 1,988 OSCAR 423 2,163 The Pile 361 1,835 RedPajama 514 2,604 S2orc 4,538 23,418 peS2o 4,582 23,852 LAION-2B-en 10 54 The Stack 430 1,953

B.3 COMMUNITY- AND SOCIETY-RELEVANT MEASUREMENTS

In this section, we provide additional results on the contamination and PII analyses from the main paper, as well as conduct two more analyses: toxic language and demographic sentiment co-occurrences. Overall the community- and society-relevant measurements contain the following analyses:

- 1. Benchmark contamination (§B.3.1)
- 2. Personally identifiable information (§B.3.2)
- 3. Toxic language (§B.3.3)
- 4. Demographic sentiment co-occurrences (§B.3.4)

- B.3.1 BENCHMARK CONTAMINATION

We measure contamination by testing whether all of the input fields are present in a single document, and report the percentage of examples from the test set that are contaminated and present the results in Table 16. We do not test for the presence of the labels as those are not always available, and they can come in different forms (e.g., in RTE they may appear either as ‘entailment’, ‘not-entailment’, or as ‘0’, ‘1’). Moreover, we do not test for consecutive appearance of these inputs, as they might appear in different orders and with different separators. As such, our contamination evaluation serves as an upper bound of exact-match dataset contamination. By employing exact match comparison with the pretraining data, we ignore minor changes in words or phrases that models trained on such similar texts may exploit. An example of such influence is introduced by Emami et al. (2020), who showed how high overlap between sentences in the Winograd Schema Challenge (Levesque et al., 2012) and pretraining corpora inflates the results on the test set, while Elazar et al. (2021b) argue that knowledge and reasoning capabilities from large pretraining corpora leak and inflate evaluation benchmarks.

Rationales of the Design Choices Here, we provide the rationals behind our design choices for the contamination experiment. Overall, our desiderata required a large benchmark that can be processed automatically, and that matched in an inspected corpora would be of high precision. We details these rationals in the following points:

- • Choice of task type. We chose to use tasks that include two or more inputs (e.g., natural language inference) as the co-occurrence of both inputs in the same document increase the likelihood of these inputs to originate from an existing evaluation dataset. In contrary, texts from tasks containing a single input (e.g., sentiment analysis) may naturally occur in some text corpus, which decreases the likelihood of contamination.
- • Ignoring the output. We decided to ignore the output of the inspected datasets since these can appear in different formats (e.g., numeric values, text labels, etc.).
- • Choice of PromptSource. Finally, we use PromptSource (Bach et al., 2022) as it is the only large scale benchmark which we could automatically process and discern the different input parts (e.g., this is important since many datasets contain additional fields like metadata which are not directly part of the task).

Note that different design choices can be made for inspecting additional contamination of benchmarks.

- Table 16: Contamination percentages of the 82 datasets filtered from PromptSource (Bach et al., 2022), in C4, OSCAR, The Pile, and RedPajama.

Dataset/Corpus C4 OSCAR The Pile RedPajama adversarial-qa-adversarialQA 0.03 0.03 0.03 0.03 adversarial-qa-dbert 0.00 0.00 0.00 0.00 adversarial-qa-dbidaf 0.00 0.00 0.00 0.00 adversarial-qa-droberta 0.10 0.10 0.10 0.10 aeslc 1.57 0.31 45.49 0.10 amazon-reviews-multi 2.28 2.10 1.48 2.06 billsum 0.06 0.06 0.03 0.06 cosmos-qa 0.00 0.00 0.00 0.00 crows-pairs 0.00 0.20 0.00 0.60 duorc-ParaphraseRC 0.00 0.00 0.00 0.00 duorc-SelfRC 0.01 0.00 0.02 0.02 esnli 0.04 0.08 1.13 1.24 gigaword 0.15 0.36 1.18 2.82 glue-ax 1.99 1.45 5.07 6.16 glue-mnli-matched 1.65 1.77 2.17 2.26 glue-mnli-mismatched 1.73 1.91 2.11 2.17 glue-mrpc 0.06 0.00 0.64 1.16 glue-qnli 0.13 0.04 1.48 1.21 glue-qnli 0.09 0.04 1.48 1.21 glue-rte 0.20 0.17 0.13 67.47 glue-stsb 3.48 3.12 11.09 9.86 glue-wnli 0.00 0.00 0.00 2.05 head-qa-en 5.22 5.29 5.11 5.94 health-fact 7.53 3.40 1.94 18.70 hlgd 0.00 0.00 0.00 0.00 liar 29.23 13.95 10.91 45.05

- math-dataset-algebra-linear-1d 0.00 0.00 0.00 0.00
- math-dataset-algebra-linear-2d 0.00 0.00 0.00 0.00 math-dataset-algebra-linear-2d-composed 0.00 0.00 0.00 0.00 math-qa 0.34 0.03 0.00 0.07 mc-taco 0.00 0.00 0.00 0.14 mocha 0.00 0.00 0.00 0.03 openai-humaneval 0.00 1.22 0.00 0.00 paws-x-en 0.05 0.00 0.15 0.20 paws-labeled-final 0.05 0.04 0.25 0.35 piqa 0.06 0.03 0.06 0.13 race-all 0.14 0.06 0.00 0.28 race-high 0.11 0.00 0.00 0.26 race-middle 0.21 0.21 0.00 0.35 ropes 0.00 0.00 0.00 0.00 samsum 0.00 0.00 0.00 0.12 scan-addprim-jump 0.00 0.00 0.05 0.16 scan-addprim-turn 0.00 0.00 0.08 0.00 scan-filler-num0 0.00 0.00 0.00 0.09 scan-length 0.00 0.00 0.03 0.00 scan-simple 0.02 0.00 0.10 0.26 scan-template-around 0.00 0.00 0.00 0.18 scan-template-jump 0.00 0.00 0.00 0.09 scan-template-opposite 0.00 0.00 0.04 0.16 scan-template-right 0.00 0.00 0.11 0.16 scicite 1.78 1.51 0.86 1.72 scitail-snli-format 0.09 0.38 0.28 0.71 scitail-tsv-format 0.09 0.38 0.28 0.71 sem-eval-2014 0.35 0.18 4.89 52.81 sick 0.31 0.18 4.79 52.61 snli 0.04 0.08 1.11 1.22 squadshifts-amazon 0.00 0.00 0.00 0.00 squadshifts-new-wiki 0.01 0.01 0.01 0.03 squadshifts-nyt 0.01 0.03 0.02 0.04 stsb-multi-mt 3.48 3.12 11.09 9.86 subjqa-books 0.00 0.00 0.00 0.00 subjqa-grocery 0.00 0.00 0.00 0.00 subjqa-movies 0.00 0.00 0.00 0.00 subjqa-restaurants 0.00 0.00 0.00 0.00 super-glue-axb 1.99 1.45 5.07 6.16 super-glue-axg 0.00 0.00 0.28 0.00 super-glue-boolq 0.00 3.05 0.00 0.03 super-glue-boolq 0.00 3.05 0.00 0.03 super-glue-cb 0.00 0.00 2.00 1.60 super-glue-copa 0.60 1.00 1.20 100.00 super-glue-multirc 0.00 0.00 0.00 0.00 super-glue-record 0.00 0.00 0.00 0.00 super-glue-rte 0.20 0.17 0.13 67.47 super-glue-wic 64.43 49.43 18.57 60.21 swag-regular 2.48 1.65 2.21 2.79 tab-fact-tab 0.00 0.00 0.00 0.00 wiki-qa 0.24 0.18 0.19 0.91 winograd-wsc-wsc273 29.30 30.40 32.23 58.24 winogrande-winogrande-xl 0.00 0.00 0.00 0.00 xnli-en 0.12 0.24 0.36 0.44 xsum 2.13 0.13 3.30 4.28 zest 0.00 0.00 0.00 0.00

- B.3.2 PII

We use three regular expressions inspired by Subramani et al. (2023) to identify email addresses, phone numbers, and IP addresses across pretraining corpora. In addition, we improved the phone numbers regex for better precision. These regexes provide us with a high precision performance (which we manually evaluate) and allows a fast PII identification. We apply postprocessing rules to the resulting matches, to improve the precision of detecting personal information by seeking to eliminate common classes of false positives (such as ISBN numbers that may be flagged as phone numbers). These rules are enumerated in Table 17.

Applying these regular expressions to the ten corpora we study in the paper, Table 20 contains the number of matches of each PII type in each corpus. For faster processing, we filter documents containing a large amount of special characters (such as documents with >50 consecutive “:)” emoticons). We further normalize this statistic, by the number of tokens in each pretraining dataset, in order to estimate the relative proportion of PII in each corpus. These results are in Table 19. We observe that even when controlling for the number of tokens in the different corpora, mC4-en has a large amount of personal information compared to the other pretraining corpora.

We manually evaluate the precision of the heuristics. In order to compute this statistic, we sample 100 examples of strings detected as PII (when available), for the three PII types, over the ten pretraining corpora in this study.These results are in Table 18. The nature of this retrieval task makes it challenging to estimate the recall of our method, and more work is needed on the topic. We show the types of examples that may be incorrectly identified as PII by our method in each corpus in Table 21.

- Table 17: Regular expressions and postprocessing rules used to identify three PII types (email/ phone numbers/IP addresses).

PII Type Regular Expression Postprocessing Filter Email Addresses [.\s@,?!;:)(]*([^\s@]+@[^\s@,?!;:)(]+?)[.\s@,?!;:)(]?[\s\n\r]

- (1) The username cannot be only "("
- (2) There must be a "." in the domain

- (1) ‘ISBN’, ‘DOI’, or "#" cannot appear in a context window of 50 characters from the match
- (2) Cannot contain URL

Phone Numbers \s+(?(\d{3}))?[-\. ]*(\d{3})[-. ]?(\d{4})

(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3} (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)

(1) ‘ISBN’, ‘DOI’, or "#" cannot appear in a context window of 50 characters from the match

IP Addresses

Assumptions and Limitations: We make a number of assumptions in doing this analysis, and we describe them below:

- • We choose three types of PII: phone numbers, email addresses and IP addresses. These three types of PII have relatively standardized formats (for example, IP addresses are always 32-bit numbers expressed in dotted decimal format), which allows us to construct regular expressions to search for these information types in text. However, the retrieved information types may not correspond to any one individual— for example, government organizations have email addresses and phone numbers.
- • Conversely, many types of personally identifiable information are not easily specifiable in the structured format we use for the information types in this study, and as a result we do not identify them in pretraining corpora.
- • While many types of information individually may not appear to identify a specific individual, they can be combined with information elsewhere on the internet to form PII. In this work, we only identify a small proportion of potential personal information that is present in pretraining datasets, but further work is needed to analyze the extent to which pretraining corpora include personal information as well as how this information can be sanitized.
- • Finally, we do not claim to estimate the risk level or sensitivity of the information types we extract from the pretraining corpus, acknowledging that this is highly context-dependent and personalized.

- Table 18: Extrapolated frequency of matches for regex searches of different kinds of PII (email/ phone numbers/IP addresses) in pretraining corpora. This is computed by multiplying the precision of our PII identification module for each pretraining corpus with the number of detections, in order to estimate the number of true matches. Prec. contain the precision of our identification method, as estimated by manual verification, on each corpora. Precision indicates the proportion of samples detected that we can reasonably infer as accurately matching the PII type. We sample 100,000 documents from each corpora, and analyze 100 samples of each detected PII type when available. * indicates that less than 100 samples for a PII type were found in a corpus, and we report the precision amongst the available PII detections. The number of samples for these corpora/PII type combinations are as follows: LAION-2B-en /Email Addresses (17), LAION-2B-en /IP Addresses (16), PeS2o/Phone Numbers

(13), PeS2o /IP Addresses (12), RedPajama/IP Addresses (95), S2ORC / Email Addresses (10), S2ORC / Phone Numbers (1), S2ORC / IP Addresses (0)

Corpus Email Addresses Phone Numbers IP Addresses Count Prec. Count Prec. Count Prec.

OpenWebText 363,789.4 99 532,929.8 87 70,430.0 54 OSCAR 62,802,224.0 100 107,163,132.4 91 3,237,420.6 43 C4 7,614,759.2 99 19,702,198.4 92 796,494.7 56 mC4-en 201,368,945.0 92 4,067,997,426.2 66 97,887,510.2 44 The Pile 19,882,348.2 43 38,019,831.8 65 4,078,794.7 48 RedPajama 35,217,396.0 100 70,264,985.9 94 1,126,129.5 *30 S2ORC 630,130.0 *100 1,465,947.0 *100 0.0 *0 PeS2o 418,136.9 97 226,937.5 *30.8 0.0 *0 LAION-2B-en 636,252.1 *94 1,029,066.6 7 0.0 *0 The Stack 4,329,620.3 53 45,473,381.9 9 4,481,490.7 55

- Table 19: Extrapolated ratios of PII frequency (the number of PII matches multiplied by the estimated precision),

PII ∗ Precision #Tokens

normalized by number of tokens in a corpus (

).

PII Type Email Addresses Phone Numbers IP Addresses OpenWebText 0.000047 0.000069 0.000009 OSCAR 0.000409 0.000698 0.000021 C4 0.000003 0.000007 0.000000 mC4-en 0.000423 0.008546 0.000206 The Pile 0.000070 0.000133 0.000014 RedPajama 0.000034 0.000069 0.000001 S2ORC 0.000011 0.000024 0.000000 PeS2o 0.000009 0.000005 0.000000 LAION-2B-en 0.000021 0.000035 0.000000 The Stack 0.000003 0.000030 0.000003

Corpus Email Addresses Phone Numbers IP Addresses OpenWebText 367,464 612,563 130,426 OSCAR 62,802,224 117,761,684 7,528,885 C4 7,691,676 21,415,433 1,422,312 mC4-en 218,879,288 6,163,632,464 222,471,614 The Pile 46,238,019 58,492,049 8,497,489 RedPajama 35,217,396 74,749,985 3,753,765 S2ORC 630,130 1,465,947 373,095 peS2o 431,069 736,810 239,912 LAION-2B-en 676,001 14,700,951 522,005 The Stack 8,169,095 505,259,799 8,148,165

Table 20: Frequency of matches for regex searches of different kinds of PII in pretraining corpora.

- Table 21: Abbreviated examples of incorrect detections by our method, for each PII type, in each pretraining dataset. The exact span that was matched is in red. Offensive content and personal information have been redacted from the presented examples.

Corpus Email Addresses Phone Numbers IP Addresses

skremoved) has joined * trayvonmartin sets ban on *!*@n***.*** * trayvonmartin has kicked whitepower from #n****

...2017 limitation 99 pcs. article id 472172730 ean 4012138149625 the model was produced in the usual minichamps...

... [stdout] awy was overriden from notenoughitems 1.6.1.9.jar 2014-03-24 20:25:06 [info] [minecraft-client]...

OpenWebText

C4 “you ever googled our email address? try googling “@fmr.com” and “charity” together, and you will get an idea”

on your mortgage. disclaimer - property reference 100103003249. the information displayed about this property

not load file or assembly ´smswrappers, version = 3.0.0.0

mC4-en smswrappe wrote in messagenews:a30c91p63 cj6vgr...4lfg7ve8@4ax.com... i bought gta iii at a garage sale and it did not

"stat-major-faults": 1213, "stattotal-memory": 3975217152, "stat-swap-in": 0

s not constitute the consent required by n.j.a.c. 11.5.6.1 (n) for the advertisement of listings exclusively

OSCAR - ...a getty images) michael jones9 october 2021 21:53 1633812509 andorra vs england player ratings: phil foden shi...

...latest update software comes with version number 10.0.0.163. currently the update available in the...

The Pile [@eiguren3].[]datalabel="table4"

t undefined behavior. for example, i get that b = 2083899728 and d = -552766888. the persistent thing you are

such damage. // according to ecma-262, sections 8.6.2.2 and 8.6.2.3 you’re not // allowed to override rea

RedPajama - watercolor baby bring a book card printable png v 1525458984 - watercolor baby bring a book card printable png

sh wikipedia) 18:54, 15 july 2013 (utc) if i can. 86.146.46.88 john of reading (talk) 06:38, 25 july 2013 (utc)

S2Orc - - -

PeS2o 65%@0.00262 izona institutional review board (approval number 2003521636a002). at baseline, the participants reported thei

-

LAION-2B-en NWA DemocratGazette/Michael Woods –03/15/2015– w@NWAMICHAELW...

queen creek 85142 e cherrywood dr - property id: 1311037210

gods and glory: war for the throne apk 3.8.10.1

The Stack remirror/ui@0.7.3 ermine the vision-agent service is running - hsd 15010872669 add missing heartbeatresponsetimersecs to the

atoaune — have you upgraded to oracle soa suite 12.2.1.1 and can’t find the partitions configuration any l

- Table 22: Toxic language percentages based on a taxonomy and a classifier over entire documents in the corpora we consider. Toxic language statistics in the corpora we consider. The document toxicity (the first two columns) reports the percentage of documents that contain at least one mention of toxic language detected by each of the approaches. The classifier is applied separately on each sentence. The fine-grained taxonomy mention (the last three columns) reports the number of toxic mentions overall, and their relative appearance normalized by the number of tokens in each corpus.

% Documents with Detected Toxicity Fine-grained Taxonomy Statistics

Corpus Classifier Taxonomy Offensive-minority Offensive-not-minority Harmless-minority OpenWebText 16.47 13.8 149K (1.92e-05) 3.55M (4.58e-04) 13.5M (1.74e-03) C4 5.75 0.01 158K (1.03e-06) 47 (3.06e-10) 146M (9.51e-04) mC4-en 6.09 0.15 31.4M (1.16e-05) 6.55M (2.42e-06) 2.85B (1.05e-03) OSCAR 9.58 8.97 8.91M (1.87e-05) 236M (4.95e-04) 549M (1.15e-03) The Pile 8.27 7.67 4.55M (1.59e-05) 84.7M (2.96e-04) 238M (8.32e-04) RedPajama 10.3 7.88 15.2M (1.49e-05) 283M (2.76e-04) 1.43B (1.40e-03) S2ORC 10.52 16.55 95.9K (1.60e-06) 8.02M (1.34e-04) 33M (5.52e-04) peS2o 9.56 17.0 47.8K (1.09e-06) 5.96M (1.35e-04) 26.7M (6.07e-04) LAION2B-en 1.09 0.89 2.69M (9.09e-05) 25.4M (8.55e-04) 182M (6.14e-03) The Stack 1.16 1.85 4.63M (3.04e-06) 84.8M (5.56e-05) 228M (1.50e-04)

- B.3.3 TOXIC LANGUAGE

How common is toxic language used in corpora? We employ two complementary methods for computing toxicity. The first is based on the work of (Zhou et al., 2021), who compiled a lexicon of terms (TOXTRIG) into three categories: possibly offensive minority identity mentions, possibly offensive non-identity mentions, and non-offensive minority identity mentions. It is then used by matching these “toxic triggers” over texts. The model-based method uses an SVM classifier trained on a dataset consisting of 200K examples based on Wikipedia and Twitter to identify toxic language.14 We apply such a classifier on each sentence separately and consider the document toxic in case any sentence is found to be toxic. We present the results in Table 22. C4 is the least toxic based on the taxonomy: only 0.01% were found to be toxic, which is expected due to the filters used in the curation process of the dataset. On the other hand, the classifier finds more documents to be toxic: 5.75%, which may indicate subtleties that the lexicon used for filtering documents from C4 did not catch. OpenWebText is the most toxic corpus based on the classifier, while PeS2o is the most toxic one based on the taxonomy, perhaps surprisingly, as it is not a web-based corpus.

Explicit Content Filtering The only dataset we analyze that explicitly filtered for toxic content (in the form of keyword matching) is C4. Indeed, the matching category from our analysis are the “Offensive-*” categories. Our analysis, that uses a fine-grained lexicon (Zhou et al., 2021), splits this category into “offensive-minority” and “offensive-not-minority”. In C4 we only found 47 mentions of the “offensive-not-minority” category, likely due to a difference in filter used to create C4 and our lexicon. In comparison, other datasets that did not employ such filters contain several million references of such phrases. Interestingly, C4 also contains 158K occurrences of the “offensive-minority” category, which were not filtered from the dataset.

- B.3.4 DEMOGRAPHIC SENTIMENT CO-OCCURRENCES

In this section, we turn to detecting biases in the corpora based on demographic factors. We constructed a set of unigrams and bigrams associated with gender (male and female pronouns), religion (the proper names of several major religions), and race (combinations of racial identifiers and words like man, woman, people, etc.). The sentiment of sentences containing these terms was computed using SpacyTextBlob and averaged over a given corpus. The results for all corpora are shown in Figure 17. The Stack is excluded from this analysis since the contexts in which these terms appeared were not typically natural language. Overall, we observe a neutral or weakly positive sentiment for sentences in which most of our demographic terms appear, with the exception of those including ‘black’ being uniformly more negative across all corpora. With minor exceptions we don’t observe substantial variation in the sentiment for individual terms among datasets. The weak positivity seen for all sources is in opposition to a related analysis performed in Gao et al. (2020), which measured weak negativity for most terms. It’s likely this is due to differences in the way

14https://github.com/dimitrismistriotis/alt-profanity-check

0.15

[Figure 20]

| | | |
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

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

OpenWebText C4

0.10

mC4-en* OSCAR The Pile

0.05

Sentiment

0.00

RedPajama S2ORC

0.05

peS2o LAION-2B-en

0.10

malefemale

asianblackhispanicwhite

athiestbuddhistchristianhindujewmuslim

0.15

- Figure 17: The average sentiment associated with several gender, racial, and religious demographic terms for each dataset. Note: averages for datasets marked with * were computed for 10% samples.

average sentiment is computed (we compute sentiment at the sentence level while Gao et al. (2020) computes sentiment only for the most frequent co-occurring terms).

developed

10000

Women's

10000

Dual

Automatic

Wide

illustration

patients

8000

8000

vector

Tips BMW

legal

Crystal

Close

companies

Hair

AP

Diamond

Pack

option

Print

Hand

Images

helps multiple

everyone

Ice

Toyota

LAION-2B-en

changes

6000

6000

Style

Electric

I've

Door

write giving

language

Paper

yet

skills

Glass

parents lost

Size

###### C4

choice

pay

thing several

Even goes

limited

financial

success types

sure

While

written

Stone

2020

DVD

Studio

Kids

includesstaff

Photo

Gallery

less

4000

needs course

4000

Product

minutes

OutCD

Modern

effect

did

Young

Fire

Shop

Me

late

Happy V

example known

History

IN

Light

words

learning

really

rentsilver

brown

energy

technology

third

hardsocial

built

George

am

™

Party

across

Pro

student

Family

holding

then shouldsaid

France

Go

type

equipment

early single season

national

War

2000

2000

town

bag

They

Will College 2007

He

says

Two

Washington October

taken

body

Black

blue

us

piece

gold 0

closehead

outside

music

center

last

gift

she

21

America

July

January

Great

House

machine

wall

machine

plans

North

States

paper

used

large

County

City

June

United

its

World

2016

black

year

an

my

>

or

Tote

BackgroundPants

Gifts

Vinyl

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

LAION-2B-en

C4

- Figure 18: 1,000 most common unigrams in LAION-2B-en (rank on x-axis), and their corresponding rank in C4 (y-axis), and visa-versa. The dashed red line corresponds to y = x. Points below and above that line indicates differences between the corpora. For instance, common unigrams in LAION-2B-en are of different adjectives and words often used to describe objects (e.g., Black, Light, Happy, Woman’s), but those are much less common in C4.

B.4 CROSS-DATA ANALYSIS Main Findings

- • Comparing unigrams of different corpora reveals distributional and topical differences.
- • OSCAR unigram distribution is the most similar to all other corpora on average.
- • 50% of RedPajama unique documents originate from C4 and 50% of OpenWebText unique

documents originate from The Pile.

• While mC4-en was supposedly a superset of C4, documents from C4 constitute only 0.04% of mC4-en, while the later being only 10x larger in size.

Using the analyses from the previous sections we can now perform targeted comparisons between different corpora. Such analysis is the first step of better understand the similarities and differences between corpora. We perform the following analyses:

- 1. Distributional similarities (§B.4.1)
- 2. Corpus overlap (§B.4.2)

B.4.1 DISTRIBUTIONAL SIMILARITY

Unigram Ranking Using the most common n-gram statistics (4.3.1), we can compare the ranking of these n-grams, to gain insights into their different usage between corpora. For the following analysis we consider the top 10,000 most common unigrams of two corpora, and display the 1,000 most common unigrams in one corpus as a function of the same unigram rank in the other corpus. In Figure 18 we display the rank of unigrams in C4 as a function of their ranks in LAION-2B-en. Some very common unigrams in LAION-2B-en describing objects such as “Two”, “Black”, “blue”, and “Light” are very common in LAION-2B-en - top 500 unigrams, but much more rare in C4’s top 1,000. Another category is car models such as BNW and Toyota whose ranking is about 900 in LAION-2B-en, but above 6,000 in C4. Figures 19-28 show the paired ranks for all corpora pairs.

7000

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |Clinton| | |
| | | | | | | |
| | | | | | | |
| | | |Obama| | | |
| | | | |Twitter| | |
| | | |men<br><br>that s<br><br>beg<br><br>co b|an urt ecamerecord<br><br>2011<br><br>earlier<br><br>40|goes<br><br>Europeanteams2008http<br><br>seven<br><br>they<br><br>mes<br><br>R<br><br>gr|re<br><br>owinsage|
| |fromthanhow You|home|available<br><br>issueanythingfacewanted<br><br>2017<br><br>Worldreasoactiorun|50nnning<br><br>generalweekselse mindcreatedbringfield users<br><br>diffi tak|cult es<br><br>Of builtperiodproduction account<br><br>levels|practice|
| | | | | | | |

Republicans

6000

5000

4000

###### C4

3000

2000

certainly

g

1000

ice

0

0 200 400 600 800 1000

OpenWebText

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | |Obama| |viole|nce|
| | | |wrote<br><br>him|self<br><br>vote bill|whose<br><br>press<br><br>Time<br><br>desp|s<br><br>ite|
| |fromtimehow he|re<br><br>already<br><br>playe on <|rs ceothers amongalmosttrying night<br><br>probably<br><br>national someoneclose<br><br>Americ wholeactio<br><br>tell le<br><br>th|a tn<br><br>emselvesgroups energyfront industry<br><br>Facebook forward Whitedidn'ttest<br><br>averag air<br><br>entidiffi 40<br><br>Mo<br><br>Se|e<br><br>recult<br><br>nday<br><br>ptember<br><br>step<br><br>British numbers<br><br>various<br><br>focusimpactaddres Coun<br><br>alt le|s ty<br><br>houg adin|
| | | | | | | |

8000

6000

mC4-en

4000

2000

h g

0

0 200 400 600 800 1000

OpenWebText

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | |According| | |
| | |Mr|war behind<br><br>reported<br><br>Americ<br><br>Wash film<br><br>him|a<br><br>ingtonself evidence<br><br>releasesensestrong air<br><br>isnMog|oaltnday<br><br>whose European Dcause morningSunday approa Street<br><br>Cou sci a|ch<br><br>rt ence nswe|
| |fromtimehow *|countrystory|Yorkhead<br><br>\<br><br>didn t bit<br><br>face<br><br>plan<br><br>seem<br><br>May<br><br>run|sning meanversionnearmain<br><br>June<br><br>soonlivingleave<br><br>40|currently size<br><br>uponindDr|ividu|
| | | | | | | |

6000

5000

4000

###### OSCAR

3000

2000

r

1000

l

0

0 200 400 600 800 1000

OpenWebText

5000

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | |doesn t<br><br>|can t<br><br>|interview<br><br>Street<br><br>Can|ada|
| | | |video<br><br>ago<br><br>deal<br><br>policy2017<br><br>wroterelea co b|sed<br><br>ntinue ecame<br><br>SouthMarch weeks<br><br>April questiopers<br><br>killed<br><br>tak<br><br>Da Se|nsonal<br><br>es<br><br>vid ptember<br><br>race career<br><br>looks<br><br>rules<br><br>admin Fe ro|istra brua exactlyund|
| |fromtimeget<br><br>te|am play<br><br>storycompan|ysocialseries move experiencetaken<br><br>decision deathfood<br><br>tell<br><br>him|self groupsfinal<br><br>instead changes received<br><br>averag 19<br><br>Firs|et<br><br>child<br><br>23<br><br>growth considadditi|onered|
| | | | | | | |

4000

3000

ThePile

2000

certainly

tion

ry y

1000

0

0 200 400 600 800 1000

OpenWebText

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |Clinton| | |
| | | | | |inves|tigati|
| | | | |American<br><br>anyoneannoun<br><br>Th cu|s<br><br>ced<br><br>at s tdecided there s<br><br>firetrade population<br><br>movement attention<br><br>leader partic<br><br>big<br><br>a|ularl<br><br>gest<br><br>nswe<br><br>effort|
| |fromtime!<br><br>You|@|move jobbitpolicy six close<br><br>decision Northcomp<br><br>himbl te|anies<br><br>ackself rm evidence needs turn<br><br>finalheld everyone weeks changesmindrateDepartme<br><br>News<br><br>interest con2|nt 2tentperiod<br><br>careerTV ind|practiceividu|
| | | | | | | |

4000

3000

RedPajama

2000

n

r ice

y l

1000

0

0 200 400 600 800 1000

OpenWebText

10000

8000

went

morning

trying

6000

don't

###### S2ORC

soon

U.S

February

San

violence

America

August

him

plans

She

Dr

July

4000

held

World

North

round

bit

member

want

my

city

South

actually

forces

security

Many

nearly

rulesparents

million

success

listNational

He

government

2000

head

financial

Now

run

longer site

others

accesseconomic

countriesneeds

close

together

five

\

what

children2015 given

makeright

involved

22

|

10

study

We

at

everyone

Republican

Why

Times

0

0 200 400 600 800 1000

OpenWebText

10000

told

email

8000

war

says

LAION-2B-en

6000

Dr

San

ago

soon

peS2o

City America

paid

ask

ever

really

pay

coming rights

July

sent

somethingthings

4000

me

held

deal

North

trade

member

think

became

offer

price

kindthought

security

United

2000

growing

despite

person

men

Most

back

Some

members

making working family

release include economic

old

longer site

give

practice

series

certain

global

full

especially

able

here

best

involved

means

us

Since R

effect

level

at thanmost

it s

edit

that s

everyone

Republican

0

0 200 400 600 800 1000

OpenWebText

10000

| | | | | | | |
|---|---|---|---|---|---|---|
| | |though|beg|an especially allow| | |
| | | |almost<br><br>policy|legal bill|rest<br><br>worth<br><br>seven<br><br>cli|mate|
| | |might<br><br>until|others<br><br>ago<br><br>sta|tes<br><br>access<br><br>rate<br><br>con<br><br>di|tent<br><br>rector<br><br>areas numbers<br><br>email languagelower<br><br>le|adin|
| |say|toopublic<br><br>point<br><br>start<br><br>doing|acrossfollowing<br><br>Obama<br><br>ho|urs<br><br>self<br><br>record age<br><br>type|land<br><br>hold|practice|
| |this<br><br>mostnowway<br><br>2<br><br>see throug good|h<br><br>system<br><br>free<br><br>real<br><br>full Sta All<br><br>c|tes<br><br>hildren video<br><br>night<br><br>edit<br><br>live class<br><br>actio|n building short<br><br>shows shot|Tuesday<br><br>decided<br><br>training<br><br>2009<br><br>Mostwatc Mi<br><br>Fe<br><br>bu J|h chae<br><br>brua y ame|
| | | | | | | |

interesting

8000

6000

g

4000

ice

2000

ry

l

0

0 200 400 600 800 1000

OpenWebText

| | | | | | | |
|---|---|---|---|---|---|---|
| | | |war|July|turned October<br><br>comp<br><br>Fe|letel<br><br>brua|
| |says|came|pay|ways risk|significant<br><br>Our<br><br>staff population<br><br>press<br><br>Mi<br><br>J|chae<br><br>ame|
| | |lot<br><br>four<br><br>s|chool water ago<br><br>anything<br><br>taken<br><br>probably<br><br>whole|turn|whose Europe<br><br>additi<br><br>jobs<br><br>Sin ind<br><br>le|on ce ividu<br><br>adin<br><br>exactly|
| |aro|und toomightOn laterdays<br><br>actually At<br><br>m|eanssocial started issue series<br><br>hand<br><br>May<br><br>sta|tes cost<br><br>groups<br><br>repog|oalrts<br><br>doesn't| |
| |not<br><br>he<br><br>my<br><br>most way<br><br>2<br><br>know<br><br>betw|eenhelp<br><br>per<br><br>times wit<br><br><<br><br>s|hinide<br><br>current<br><br>2012<br><br>role|developmentmean<br><br>election<br><br>stop<br><br>view<br><br>pr|ice events<br><br>word<br><br>focus<br><br>latest<br><br>cli|mate|
| | | | | | | |

10000

ry

y

8000

l

6000

TheStack

y

g

l

4000

2000

0

0 200 400 600 800 1000

OpenWebText

###### Figure 19: OpenWebText top 1,00 unigrams, and their corresponding indices in the other corpora.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | |cus|tomer| | |
| | | |visit<br><br>Please art<br><br>ide<br><br>ap|as<br><br>plication card<br><br>skills<br><br>kids<br><br>items<br><br>revie Drlat|w est conditions<br><br>travel<br><br>tools<br><br>Many<br><br>St<br><br>sendDay re|mem<br><br>creating<br><br>walk<br><br>limited|
| |by otherher<br><br>rea|lly<br><br>design<br><br>offer|planlow<br><br>results<br><br>training friends various<br><br>especialquite<br><br>staff|ly thirdBy<br><br>increasetestrate<br><br>period strong<br><br>pointsrunning<br><br>potential<br><br>inc|luded abilityworkedOfcouple<br><br>itselflate<br><br>attenti glo|on bal|
| | | | | | | |

4000

3000

OpenWebText

2000

ber ng d

1000

0

0 200 400 600 800 1000

C4

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | |e|ntirehighly<br><br>countries<br><br>losthold| |
| | | | |receivematterbuilt<br><br>held<br><br>stop<br><br>behin|abilityd<br><br>talk<br><br>Mrfit<br><br>hour glo|bal|
| | | |tookidea mov|e among<br><br>learning treatmentcommonhigher continuemarke lat stap|ting est<br><br>urchasendard<br><br>Park<br><br>G|oogl|
| | | |provides<br><br>understand monthpay provid result<br><br>includ<br><br>Som<br><br>fron ide pr|edes<br><br>e t<br><br>as<br><br>esentreporttest<br><br>oil<br><br>ma|Worldterial| |
| | |until comalr|eadyesabove<br><br>monthssingle| | | |
| | | | | | | |
| |by<br><br>other<br><br>her And| | | |0| |
| | | | | | | |

1600

1400

language

1200

1000

mC4-en

800

600

400

200

0

0 200 400 600 800 1000

C4

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | |chance<br><br>dev|creatingelop|
| | | |fron|t insuranceactivities<br><br>enviro bran<br><br>c|nment d<br><br>omputer<br><br>Then<br><br>programsexcellentrest multiple<br><br>requirplaye tra|emenrs ck|
| | | |idea<br><br>artstaff<br><br>eve pr pr|nts esent oblems<br><br>likelyphone issue<br><br>build common education happy modelpoints<br><br>posit co|ion deequipment directly<br><br>Day En|glish|
| | |dev<br><br>ra|elopment<br><br>planngefact<br><br>training<br><br>managemen whether<br><br>shortwholegovernm<br><br>seen|t ent<br><br>50<br><br>certain<br><br>de<br><br>revie n|w ews Thanks| |
| | | | | | | |
| |by<br><br>other<br><br>her 4| | | | | |
| | | | | | | |

2000

1750

1500

nged

1250

ts

OSCAR

1000

750

500

250

0

0 200 400 600 800 1000

C4

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | |plans<br><br>can t<br><br>| |
| | | |learn<br><br>ensure|throughout<br><br>card<br><br>store|extra<br><br>Many fit career feature send<br><br>Intern play G|ation ing oogl|
| | |products<br><br>offer easy|planspecial<br><br>designed 2017softw ga|are<br><br>mes<br><br>located stay<br><br>leavewee<br><br>rea<br><br>qui|ks<br><br>ding<br><br>ckly recently<br><br>site<br><br>w|s<br><br>all|
| | |experience<br><br>project<br><br>Un|doingiversity<br><br>story<br><br>provides million<br><br>saysfivewent<br><br>histor air eve ap|y<br><br>nts plication<br><br>party report<br><br>US<br><br>via<br><br>May2015<br><br>posit<br><br>behin sta|ion<br><br>d ndard<br><br>film<br><br>record| |
| |by otherher| | | | | |
| | | | | | | |

3500

3000

2500

ThePile

2000

al

1500

1000

language

500

0

0 200 400 600 800 1000

C4

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | |insurance items|tools|machine walk|
| | | |pleasefun<br><br>beautiful etc|usuallylocatedoil<br><br>skills kids<br><br>leavelat|est<br><br>looks travel couple<br><br>resources<br><br>plannin fully<br><br>hugre|g emem limited|
| | |food<br><br>price|questions<br><br>month let mind idea<br><br>needed<br><br>acco wh<br><br>prha<br><br>m|unt ite esentlf<br><br>eet<br><br>party<br><br>everyone<br><br>opportunity child<br><br>private<br><br>running<br><br>memberDr<br><br>inc|luded according terms<br><br>Of<br><br>decisionwritten<br><br>record<br><br>grou|ps|
| |by<br><br>otherher<br><br>arou|nd| | | | |
| | | | | | | |

2500

2000

ine

RedPajama

1500

ber

d

1000

500

0

0 200 400 600 800 1000

C4

10000

email

tell

someone

8000

Day

talk

phone

friends

ready

car

6000

piece

don't

hair

worked

###### S2ORC

bad

am

something

coming

4000

bit

creating

track

going

believe United

professional

said

learnvideo

return

built

team

box

That

English

share

@

National turn probably

write

2000

create

outside

benefits six

know

skin

issues

makes

room

taking whole

black

he

downplace

matter

red

First 40

done

needed

series

cannot

directly

contentmanagement

againstproduct

An

future

around

then

by

0

0 200 400 600 800 1000

C4

10000

love

8000

someone

morning

saw

talk

LAION-2B-en

6000

soon

peS2o

worked

bad

U.S

huge

came

customers

His

prices

4000

things

How

bit

creating

giving

think

@

believe United

live

build

And

return

business

probably

million property

English

2000

building

easy

currently

benefits six

address

run hoursmembers

financial

ones

completelyapply

her

enough

daily

lettaking needed

online

additional longer environmentfollow

always

done side

together

series

last

complete

taken

whatmake

family

step

means

find

'

between

by

0

0 200 400 600 800 1000

C4

| | | | | |dev|elop|
|---|---|---|---|---|---|---|
| | | | |pa|tients| |
| | | |companies|everyone changes<br><br>optio|n<br><br>helps<br><br>leg|al|
| | |sure<br><br>needsless<br><br>thing sev|eral<br><br>yet<br><br>pay While<br><br>includstaff|es<br><br>financial<br><br>skillschoice|success types<br><br>written<br><br>write giving<br><br>multiple<br><br>pare lostEv<br><br>go|nts en es<br><br>I've<br><br>limited|
| |then<br><br>rea|lly am<br><br>did<br><br>course<br><br>ach|socialrossard<br><br>example known<br><br>minutes<br><br>type<br><br>technology early energy|third<br><br>learning built<br><br>wor|ds<br><br>equipment<br><br>student<br><br>late<br><br>natio<br><br>eff|nal<br><br>ect<br><br>town|
| |an<br><br>its<br><br>us<br><br>shouldsaid<br><br>used<br><br>she<br><br>He<br><br>la<br><br>T|st<br><br>hey<br><br>large<br><br>bo|dymusic<br><br>single season<br><br>says<br><br>closhea|ed<br><br>2016<br><br>taken<br><br>outside<br><br>>black|World<br><br>StatesCounty<br><br>piece<br><br>June<br><br>ce|nter machine|
| | | | | | | |

ed

10000

8000

6000

language

d

4000

2000

ine

0

0 200 400 600 800 1000

C4

10000

road

completely

photos

8000

taking government

gives

came

looks

International

improve cut

Our

pay

6000

story

TheStack

wall

ability according

materials

National

It's

picture Since

love

several

walk

taken

board

doing

shows

quality

average

4000

why

customer

actually

paper follow

learning

building

plan

going

community

necessary

works

problem please

cost

against means

From

said

question

Theseweek

cases

less

people

2000

present

complete

And high

during

loss

Google

clear

apply

Some

latest

feature

being

different

already

release

light

web

children

safe

without

help where

non

information

case

not

companies

treatmentI ve

lives

0

0 200 400 600 800 1000

C4

###### Figure 20: C4 top 1,00 unigrams, and their corresponding indices in the other corpora.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | |04<br><br>Down|load<br><br>steel| | |
| | | |t<br><br>height|Vi|ew Compa<br><br>Book|ny|
| |or timeuse<br><br>i|201713<br><br>produc 50<br><br>|t studentscar property cost hours range Anshort 40 Ple<br><br>no<br><br>te|ase<br><br>n<br><br>st<br><br>application<br><br>please boxReadtable<br><br>August World<br><br>December whiteoffice<br><br>HealthSee Engl<br><br>ho<br><br>sa|ish wever les<br><br>problems reading mind<br><br>wide2007 interest meeteasily shows<br><br>extr rec 32 p|a<br><br>otenteive|
| | | | | | | |

8000

6000

OpenWebText

4000

2000

dial

0

0 200 400 600 800 1000

mC4-en

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |Template| | |
| | | |Posted<br><br>05<br><br>06| | | |
| | | |·<br><br>01|Top|N<br><br>Buy<br><br>d| |
| |or myinto bac|kthink<br><br>2019<br><br>e<br><br>50<br><br>|26<br><br>weight<br><br>main<br><br>D<br><br>file|s<br><br>application<br><br>code<br><br>September<br><br>White<br><br>account<br><br>essay doesn't<br><br>frontmodel<br><br>governmSome<br><br>Februa<br><br>historyho|ent<br><br>ry<br><br>wever International issue<br><br>item<br><br>believe<br><br>source deal<br><br>lawaddresslikelytakenparty<br><br>X<br><br>THE turnrec|tellent|
| | | | | | | |

8000

6000

###### C4

4000

2000

0

0 200 400 600 800 1000

mC4-en

| | | | |Template| | |
|---|---|---|---|---|---|---|
| | | |margin| | | |
| | | | | | | |
| | | | | | | |
| |or my than righ|t<br><br>00»|light<br><br>imageweight05 planmedia B<br><br>shortSo<br><br>me m|uth<br><br>thod citycompleteanagementpast<br><br>casino<br><br>School card storygiven<br><br>buildingresult<br><br>CollegeR Day<br><br>L<br><br>governm<br><br>Cen<br><br>str<br><br>qu<br><br>sa|ent<br><br>ter<br><br>ucture<br><br>estion<br><br>totalaccordingles<br><br>staffproduction<br><br>quitehalfenjoy<br><br>saveWest words<br><br>standaplayer learni<br><br>colle<br><br>disp ou|srd ng<br><br>ctio lay tside solution loss|
| | | | | | | |

8000

6000

###### OSCAR

4000

2000

n knowledge

0

0 200 400 600 800 1000

mC4-en

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | |Busin|ess<br><br>padding| | |
| | |Home|m|obile<br><br>Read|Park<br><br>Post Next in|terne|
| |or if there<br><br>onl<br><br>ev<br><br>bu|ine<br><br>en<br><br>siness<br><br>qualitybook<br><br>US wo|rking<br><br>options program cost position<br><br>buy<br><br>access<br><br>featuresplan rangeview<br><br>09 details<br><br>indus<br><br>em file<br><br>ev|try ail s<br><br>er<br><br>pay<br><br>2009<br><br>Well performance pm didn't<br><br>living<br><br>stay<br><br>visitCen<br><br>verstr<br><br>Ind co<br><br>w|ter<br><br>sionucture<br><br>ia mputer<br><br>entwomen<br><br>yourself hope eventspartyprivate<br><br>every<br><br>book<br><br>extr<br><br>rec|one<br><br>s<br><br>a<br><br>ent|
| | | | | | | |

8000

6000

ThePile

4000

t

2000

0

0 200 400 600 800 1000

mC4-en

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | |Buy| |
| | | |Price| |you'll| |
| | | | | | | |
| | | |Best<br><br>weight<br><br>08<br><br>Ple<br><br>03m|ase<br><br>obile Service can't professionalde be|signed<br><br>autifulpartsdrive<br><br>runni<br><br>un<br><br>cle|ng<br><br>it<br><br>an<br><br>loss|
| |or if there bec|ausestill<br><br>quality|foodcost2010<br><br>contacttitle mainte c|st ompaniesb relatedtechnologybuilding Not learn<br><br>modelL via eitherSome|First<br><br>problems thirdanythingincreasecommonprobablyevents currentlyevery|one|
| | | | | | | |

8000

7000

6000

5000

RedPajama

4000

3000

2000

1000

0

0 200 400 600 800 1000

mC4-en

10000

doesn't

books

Online

8000

trying

story White

car

6000

###### S2ORC

Black

border

soon

February

August

night

THE

April

stop

June

4000

State

visit

January

going look

art

job return

excellent

go

built

why

hard

2000

building

Now

More

ensure

enough

includes

course

locatedwide

air

just

matter

top

making

clearcertain

together27public

next

account

2011

added

called

lightposition

One

left

higher

only

or also

beautiful

0

0 200 400 600 800 1000

mC4-en

John

10000

didn't

TV

York

8000

went

got

trying

latestbrand

LAION-2B-en

ready

it's

6000

app

ago

peS2o

Black

border

October

August

customers

margin

4000

drive

extra

What

excellent

said

started

video

Next

2000

head

So always left

throughout

her

past

he

room 2015

article

collection

eventfar

air

unitreceived

plant

open

communitymonths

account

27

2012

side t

position

life

often

W

For

each

or In

£

06

Top

Steel

0

0 200 400 600 800 1000

mC4-en

10000

| | | | | |anything<br><br>allow<br><br>conti|nue|
|---|---|---|---|---|---|---|
| | |able|actually<br><br>often<br><br>already|given|average<br><br>almost<br><br>impact| |
| | | |include<br><br>options<br><br>ago ·<br><br>several|applicationfar can't<br><br>area|s<br><br>customer<br><br>total<br><br>growth<br><br>me lo|nu wer|
| | |don't<br><br>since sea h|rch<br><br>aving<br><br>research why<br><br>hours months<br><br>09|doing<br><br>minutes<br><br>loca<br><br>ver|tion<br><br>sion<br><br>third<br><br>law bordermedical<br><br>take<br><br>cle<br><br>fo|s<br><br>an<br><br>rwar|
| |or new<br><br>would If<br><br>workthroug<br><br>find<br><br>av<br><br>e|h<br><br>ailable<br><br>ach number products<br><br>2018<br><br>look<br><br>site<br><br>full 2016<br><br>market<br><br>page|26 de<br><br>children<br><br>single<br><br>pe|rfect box steel series<br><br>Read<br><br>close<br><br>news<br><br>living|Here<br><br>win Postshows|®|
| | | | | | | |

8000

6000

4000

2000

0

0 200 400 600 800 1000

mC4-en

| | | | | |especially| |
|---|---|---|---|---|---|---|
| | | |Best|season<br><br>frie|nds<br><br>coming| |
| | | | |didn't<br><br>air<br><br>taking<br><br>five<br><br>Wh|ile<br><br>mind<br><br>art hope| |
| | |I'm<br><br>lot water<br><br>|car| |problems<br><br>Postshows<br><br>pro|gram|
| |years mu|chWhat<br><br>come<br><br>sa<br><br>a|y<br><br>gain<br><br>needs<br><br>cost<br><br>person<br><br>big<br><br>Alsogetting<br><br>until<br><br>problem|city<br><br>can't seriesRead<br><br>paper|ensure period<br><br>choicelearni<br><br>p G|ng<br><br>retty roup|
| |from<br><br>beenthanmake<br><br>me<br><br>her<br><br>usNewfinde i|ach<br><br>process<br><br>anothe<br><br>2012|rprogramtimes<br><br>current 08<br><br>m|onth 2009<br><br>account<br><br>article<br><br>bla|ckmove<br><br>H<br><br>purch<br><br>crea|ase<br><br>ted|
| | | | | | | |

10000

8000

knowledge

6000

TheStack

s

4000

2000

throughout

0

0 200 400 600 800 1000

mC4-en

###### Figure 21: mC4-en top 1,00 unigrams, and their corresponding indices in the other corpora.

10000

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| |2020| | | | | |
| | | |Email<br><br>Tha|nks|marketing sale|Next|
| |haveotherpeople<br><br>Dec<br><br>And|ember online<br><br>February<br><br>him<br><br>More page s24 22u|sersoffer<br><br>commentsprovided low<br><br>etc receive<br><br>agree<br><br>cost<br><br>visit|31 learn<br><br>letrights<br><br>certainevent<br><br>ensure specialriskAnlight<br><br>cash<br><br>mattetell<br><br>send<br><br>site|r<br>s<br><br><br>usuallydesigned latestmember<br><br>books medicalfunction note<br><br>cause<br><br>2006digital<br><br>acti<br><br>hel<br><br>po|vities<br><br>d<br><br>you'retradepula|
| | | | | | | |

8000

OpenWebText

6000

4000

2000

r

0

0 200 400 600 800 1000

OSCAR

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |Comme|nts| |
| |2020|S|hare|Terms|09| |
| | | |View|topic|cr|ushe|
| |have2021were<br><br>Dec|ember<br><br>life<br><br>2012<br><br>2010 18|posts<br><br>USsearch<br><br>Servi law<br><br>00<br><br>m<br><br>e|ces edia<br><br>31<br><br>I veamountState<br><br>latersimplydewent<br><br>Online<br><br>choos<br><br>canMostt<br><br>h|classeead<br><br>actionlikely<br><br>similartoldtakes didn t fiveadditiontreatmentoutsidewhite<br><br>behitotalacti<br><br>lim|vitiesnd<br><br>ited|
| | | | | | | |

8000

6000

###### C4

4000

r

2000

0

0 200 400 600 800 1000

OSCAR

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | |Policy|browser Posts<br><br>©| | |
| |havelikeonly<br><br>DecJ|anuaryember go<br><br>S|hare<br><br>understand untilbecome 2008thoughstoryyet<br><br>Cli spe|ck itherecific<br><br>industry31plan<br><br>readingprofessionalevent<br><br>friendsStatesAn<br><br>furtherWhilewentrate<br><br>purceveJu<br><br>Da|sthasents ta likelylivingelse<br><br>additionaltaken<br><br>wantedtold Are<br><br>note staff<br><br>2006digitalpotenreainte 0|dytialrnet 2|
| | | | | | | |

8000

6000

mC4-en

4000

2000

effective

0

0 200 400 600 800 1000

OSCAR

10000

| | | | | |Learn| |
|---|---|---|---|---|---|---|
| | | | |Price| | |
| | | |Policy|Business| | |
| | | |Servi|ces<br><br>·|Namede0|livery2|
| |have2021only<br><br>Oct201|Augustober8<br><br>online 2013 account<br><br>accehttps sh lis|ss are t<br><br>addresseasy 26addbecomehardAmerican left<br><br>councann<br><br>que|tryot<br><br>everythingpaststions<br><br>companies interest<br><br>School month<br><br>legal<br><br>age<br><br>National close<br><br>Most<br><br>matteneedtellmain<br><br>site|red<br><br>s<br><br>Website<br><br>receivedwanted<br><br>leavenote<br><br>staff build China<br><br>positipoten<br><br>Hou|ontial<br><br>partsse|
| | | | | | | |

8000

6000

ThePile

4000

2000

opportunity

0

0 200 400 600 800 1000

OSCAR

| | | | | | | |
|---|---|---|---|---|---|---|
| |Reply| | | | | |
| | | | | | | |
| | | | |Thank|download<br><br>Learn| |
| |have& there And|around<br><br>2|009<br><br>Facebook<br><br>blog<br><br>options bad<br><br>I'mhours<br><br>commen<br><br>resul<br><br>visit yet<br><br>dap|t<br><br>ts<br><br>everythingayhandtemonthcertain<br><br>energywholetakingvariousprovides2007<br><br>changesfinancialFirst<br><br>perfect<br><br>site<br><br>Dr<br><br>wD|s ordsaylongerhigherunique<br><br>testareas<br><br>beautiful model<br><br>pretty child<br><br>comp stepfoc<br><br>re|uter us<br><br>quire wrong|
| | | | | | | |

8000

6000

RedPajama

4000

2000

0

0 200 400 600 800 1000

OSCAR

10000

Policy

Do

card

8000

wanted

6000

website November

###### S2ORC

August

She

September

Not

request

July

rights

4000

May

March

took

think

Statesmove

New

put

come

video

asked

vehicle

thought

US

2000

list

currently

machine

week

heart

makes

policy

details

China2006

online

event

called search children

program

givetop

near

focus

account

series

Datarather

post

amount

another

strong

$people

One

left

means

step

energy browserstudy

basedsystem

have

=

News

Posted

Just

Add

0

0 200 400 600 800 1000

OSCAR

10000

mail

Facebook

Do

Search

8000

says

nothing

LAION-2B-en

6000

Dr

soon

getting

peS2o

don't

November

Life

tax

August

Not

bring

board

4000

me

rights

bit

company

India

equipment

offer

go

learn

States

video

asked

write digital

teamwhy

2000

medical currently weeks

# back

issueitems

her

start 2009 following

details

event

words

always

allow

search

just

designed

program

openalready At

near

account

amount

23

left

find +

must

energy browserstudy

group

have2

happy

0

0 200 400 600 800 1000

OSCAR

| | | | | | | |
|---|---|---|---|---|---|---|
| | |lo|an<br><br>term|s<br><br>matte|r nothing<br><br>apply|wrong|
| | |able|address|cash<br><br>legal<br><br>con|ditionswantedsimilar<br><br>cause<br><br>mail| |
| | |PM|below bad<br><br>enough|tru|e<br><br>talk addition| |
| |beca|use<br><br>same<br><br>change<br><br>anothe<br><br>since|r<br><br>monthsresearch<br><br>try<br><br>it's<br><br>job<br><br>due|Thank<br><br>rights<br><br>energy games|COVID pretty<br><br>acti<br><br>po|vities<br><br>pula parts|
| |from<br><br>he<br><br>there<br><br>than<br><br>very years<br><br>should|she<br><br>site<br><br>More<br><br>So<br><br>By<br><br>place littlethree<br><br>O|ne<br><br>show<br><br>week<br><br>That<br><br>power Service<br><br>live<br><br>play Cli|ck<br><br>plan training short<br><br>shall<br><br>face<br><br>Repl|iesrather<br><br>Internationalyoung<br><br>model<br><br>child Name<br><br>doesn t<br><br>Learn<br><br>pap<br><br>file<br><br>cr<br><br>k|er ushe ids|
| | | | | | | |

10000

8000

effective

6000

4000

r

2000

r

0

0 200 400 600 800 1000

OSCAR

| | | | | | | |
|---|---|---|---|---|---|---|
| |NoveOct Ju|mberober ly<br><br>August| | |stay| |
| |says<br><br>Ju c<br><br>A|ne ookies pril|understand research He|alth<br><br>technology<br><br>gamesmatte|r men<br><br>soon<br><br>tax Hou|se|
| | |money wa|ter Contact<br><br>Facebook<br><br>away<br><br>American|car<br><br>interest|Here major<br><br>materials Out<br><br>India<br><br>limv|ehiclited you're|
| | |great say<br><br>might<br><br>keep<br><br>cou|rse<br><br>put<br><br>Email postsNow<br><br>gettingacross news<br><br>m|an<br><br>hand|Also<br><br>currently<br><br>fronpap|ter parts|
| |" 3<br><br>make<br><br>most way<br><br>after<br><br>here 8|post2012same No<br><br>don> t<br><br>full<br><br>pric u|e sers<br><br>called<br><br>job second term sof|s<br><br>tware<br><br>rightstopic<br><br>return<br><br>uses<br><br>can t<br><br>closeadd<br><br>con<br><br>h|ed<br><br>ditions<br><br>ead<br><br>common<br><br>unique stop<br><br>His<br><br>white ro|le|
| | | | | | | |

10000

effective

8000

6000

TheStack

4000

2000

0

0 200 400 600 800 1000

OSCAR

###### Figure 22: OSCAR top 1,00 unigrams, and their corresponding indices in the other corpora.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | |div| |partial| |
| | | |que| |width<br><br>hat<br><br>un| |
| |n<br><br>Let<br><br>d|m `<br><br>P<br><br>la<br><br>la|bel<br><br>V O|defined<br><br>lengthsigni|ficantly<br><br>functio|ns|
| |be herWe|function<br><br>somethin<br><br>code<br><br>beg|g<br><br>in<br><br>meaneither<br><br>See significantCourt space image positionverris co|ksion<br><br>nditions<br><br>additionparticular whole<br><br>showedstandard points user<br><br>page<br><br>pap s<br><br>a|er eems ir<br><br>style<br><br>struct<br><br>doesn't<br><br>heartfollow stopplan creafe<br><br>n|ltted<br><br>etcimagesote|
| | | | | | | |

8000

6000

OpenWebText

4000

2000

s

0

0 200 400 600 800 1000

The Pile

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | |Fig|beta| | |
| | |l| |import| | |
| |d| |V|Table<br><br>m|ass<br><br>29| |
| |> be willsaid thro|ugh<br><br>class frac<br><br>B<br><br>fig<br><br>13<br><br>J<br><br>U|.S<br><br>else<br><br>thought<br><br>higherratequestion const agewent<br><br>term face<br><br>low|ers positive changesNowtaken<br><br>according half certain<br><br>length<br><br>you're par<br><br>recsas|ty<br><br>growtheemsword<br><br>00<br><br>date<br><br>60complex percent 2013subject turne weekwri fe fo|d tes lt llowe|
| | | | | | | |

10000

8000

6000

###### C4

4000

2000

d

0

0 200 400 600 800 1000

The Pile

| | | |2d<br><br>aligned| | | |
|---|---|---|---|---|---|---|
| | |±| | | | |
| | | | |le| | |
| |k| | |pop|ulationknew<br><br>anti<br><br>determin|ed|
| |I hadthan<br><br>b kno|w<br><br>`<br><br>r<br><br>want<br><br>cells<br><br>X analys calleU|is d.Shand<br><br>effect<br><br>studies<br><br>Figure<br><br>rate America<br><br>levels seengovernearly<br><br>low maver<br><br>do|n<br><br>ment er terialsion ing ratherrole<br><br>night<br><br>half<br><br>tellperiodTHE<br><br>approac white<br><br>receive startcitypap<br><br>recde in|h d<br><br>ered<br><br>cisionord dividual<br><br>expected<br><br>comes<br><br>Whywritten<br><br>effecti<br><br>OF gav wri<br><br>for|ve<br><br>e te<br><br>ce<br><br>etc|
| | | | | | | |

8000

6000

mC4-en

4000

2000

0

0 200 400 600 800 1000

The Pile

| | | | |Template| | |
|---|---|---|---|---|---|---|
| | | |margin| | | |
| | | | | | | |
| | | | | | | |
| |or my than righ|t<br><br>00»|light<br><br>imageweight05 planmedia B<br><br>shortSo<br><br>me m|uth<br><br>thod citycompleteanagementpast<br><br>casino<br><br>School card storygiven<br><br>buildingresult<br><br>CollegeR Day<br><br>L<br><br>governm<br><br>Cen<br><br>str<br><br>qu<br><br>sa|ent<br><br>ter<br><br>ucture<br><br>estion<br><br>totalaccordingles<br><br>staffproduction<br><br>quitehalfenjoy<br><br>saveWest words<br><br>standaplayer learni<br><br>colle<br><br>disp ou|srd ng<br><br>ctio lay tside solution loss|
| | | | | | | |

8000

6000

###### OSCAR

4000

2000

n knowledge

0

0 200 400 600 800 1000

mC4-en

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | |mathbb|§<br><br>°<br><br>obtainedfactor| | |
| | |met|hodstring<br><br>com<br><br>O|pared<br><br>array<br><br>false color<br><br>diffe<br><br>e.g|rence Hereinstead<br><br>Yes<br><br>patient status effecti OFprim|ve<br><br>ary etc<br><br>scale|
| |I A than<br><br>b|much<br><br>T la code|bodyfurtheractionrelatedagegenerasourcface<br><br>co|el<br><br>nditions additionincrease accordingAnhouse un|derstand design<br><br>someone From sixproduction<br><br>stop29|building|
| | | | | | | |

7000

6000

5000

RedPajama

4000

3000

2000

1000

g

0

0 200 400 600 800 1000

The Pile

10000

someone

York

8000

_

got

title

6000

tried

###### S2ORC

Is

turned

him

eyes

4000

die

static

business

2000

know

text next

heart

service clear application 40

trial image

hours major

old called

ground productsindex

always handAtself let

Whileseries

elements32

Although2014

now

early

providedcontentBy

physical

find alcontrol

After associated

weight

B

be will We

pone

0

0 200 400 600 800 1000

The Pile

10000

himself

love

8000

someone

title

LAION-2B-en

ref

6000

peS2o

turned

him

house

felt

4000

big

gave

wantthink

American

false

come

return

thought

business

say

why

2000

home always handAt

But

idea

device

enough trial making

service

ground productsindex

true

event

world

six First

accessWhile

32

simple

2014

ablekey

what

strong

providedcontentBy

much

E

it |

infty

0

0 200 400 600 800 1000

The Pile

| | | | | | | |
|---|---|---|---|---|---|---|
| | | |values|matter<br><br>pro|bably simply longerspecies<br><br>s|amp|
| |_<br><br>ref<br><br>k|given|means|trying<br><br>diffe in|rence dividual| |
| | |somethin<br><br>presen<br><br>le|g<br><br>t<br><br>ast<br><br>treatment<br><br>thing sure<br><br>term<br><br>total<br><br>be|comele<br><br>major|move<br><br>distri Oh<br><br>form n|cectessa ote|
| | |r<br><br>much<br><br>think<br><br>court point<br><br>non<br><br>thinda|ysgs<br><br>shown<br><br>run<br><br>sec|tiontry °<br><br>past|target taking<br><br>production<br><br>Z<br><br>check<br><br>sample stop|war|
| |I her<br><br>theresaid shesee eac<br><br>wo<br><br>ve y|h<br><br>rk<br><br>ry earsdownlife<br><br>place<br><br>16|field<br><br>single headIt's action<br><br>O|play<br><br>particular<br><br>property event via|Herefilm<br><br>today<br><br>td<br><br>polic<br><br>de<br><br>stu 20|e<br><br>lta<br><br>den 10<br><br>building<br><br>treated|
| | | | | | | |

10000

es

8000

6000

ge

4000

2000

s

g

d

0

0 200 400 600 800 1000

The Pile

| | | | | | | |
|---|---|---|---|---|---|---|
| | |fig| | | |war|
| | | |didn't<br><br>govern|ment<br><br>approac|h<br><br>taking distri<br><br>doo|ct r|
| | | |higher<br><br>togethsu|errface<br><br>§<br><br>shows<br><br>pro<br><br>qu|bably<br><br>ality<br><br>near turn problems<br><br>today<br><br>market| |
| |him<br><br>y|ears<br><br>mightworld<br><br>come<br><br>The|se<br><br>give dueproblem<br><br>put<br><br>Then later<br><br>similar|yet<br><br>room solution<br><br>par|tyneeds<br><br>distribution doesn't<br><br>bad<br><br>produ|cts|
| |as 4<br><br>Ittwoouroverbeca eac<br><br>now 8|use h<br><br>here high year<br><br>during<br><br>l<br><br>large calle|d<br><br>expression<br><br>mathbb<br><br>action<br><br>children<br><br>term<br><br>read|le<br><br>view<br><br>below beta<br><br>points access|se events<br><br>subject<br><br>began<br><br>docum|ent|
| | | | | | | |

10000

8000

6000

TheStack

4000

2000

elements

0

0 200 400 600 800 1000

The Pile

###### Figure 23: The Pile top 1,00 unigrams, and their corresponding indices in the other corpora.

6000

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | |labe|l|
| | | | |v|Sc|ienc|
| | | |M x<br><br>C|ounty<br><br>Health<br><br>patients|complete<br><br>table<br><br>meth<br><br>fu|od<br><br>n|
| |; also<br><br>For|too|non25 due<br><br>21 August 2011<br><br>issues<br><br>mem|mberodel hoursheldviewaloriginalinternational<br><br>various sharearticle<br><br>someon<br><br>2022<br><br>commoDavidchildliv|e<br><br>esn groups worked<br><br>growth amountThen sawbuild<br><br>includedfocus playingwoman meeting| |
| | | | | | | |

5000

4000

OpenWebText

3000

e

2000

1000

0

0 200 400 600 800 1000

RedPajama

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| |^| |`|~| | |
| | | | |v<br><br>c|La|w|
| |0| |June<br><br>J Presidentm|en<br><br>et<br><br>cases<br><br>messageDepartmentWest record<br><br>t|groupsreleaseplayers<br><br>General<br><br>Sc|ienc<br><br>High|
| |youbeenWe righ|t<br><br>@<br><br>going<br><br>class his|tory<br><br>South24 wenthead<br><br>final Now me|mber elsecodecontinueparty matter someonbrin<br><br>dere|eg<br><br>viewal aireveryone<br><br>recently turnaddressIrunningve<br><br>anyo useractiv|ne ities|
| | | | | | | |

7000

6000

5000

4000

###### C4

3000

2000

e

1000

0

0 200 400 600 800 1000

RedPajama

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | |polic|e<br><br>evidence<br><br>Washington| |
| | | |sh|all national<br><br>policyLondon patients<br><br>en|worked<br><br>Why<br><br>conditions leading<br><br>succes learnin entir<br><br>pro<br><br>p|s g<br><br>e<br><br>gram<br><br>arent|
| | | |far<br><br>April<br><br>seen Schoolonce<br><br>result<br><br>source<br><br>issues comp<br><br>belie<br><br>field pr|anies<br><br>ve performanceesent plan<br><br>especiallyarthowever<br><br>original God<br><br>let<br><br>move<br><br>2009<br><br>reaso blac|n<br><br>k provides<br><br>front<br><br>impact<br><br>treatmentpopula table fu<br><br>u|r n sers|
| |you<br><br>In just 5|11|it's| | | |
| | | | | | | |

2500

2000

1500

s

mC4-en

s

1000

500

0

0 200 400 600 800 1000

RedPajama

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | |y| |
| |youup We For|returnstoryState|S<br><br>projectalongonce<br><br>Mr self due<br><br>roleTHou Drme<br><br>jo<br><br>C|sember b<br><br>ounty performance<br><br>national closemonthview<br><br>events countries howevereverythinglet<br><br>Londonpatients<br><br>v<br><br>It's proba sens tell<br><br>beg|bly<br><br>e<br><br>an late themselves<br><br>Court<br><br>that s economic takes<br><br>himself Park stopsimplyunique<br><br>conside THEanyo<br><br>methst<br><br>W<br><br>ne|red ne<br><br>udenod cess|
| | | | | | | |

8000

6000

###### OSCAR

4000

2000

t

ary

0

0 200 400 600 800 1000

RedPajama

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | |Black| |
| | | |October<br><br>buildin2010 Hou perspro|g se onalducts gamesoffershare<br><br>traini<br><br>ca<br><br>qu<br><br>liv|ng<br><br>reer<br><br>estions<br><br>es<br><br>themselves<br><br>ways<br><br>workedfocus lost<br><br>bornunique<br><br>Europ<br><br>mome<br><br>ens<br><br>pro<br><br>Mo|e<br><br>nt<br><br>ure<br><br>gram<br><br>st please|
| |youIn just 5|it s business<br><br>s<br><br>school we|ek<br><br>future everkeepfeel<br><br>mediamakes Mr<br><br>comes food<br><br>tak|en percent issue<br><br>period23<br><br>room<br><br>idea required received became<br><br>seems reaso|n<br><br>styleinvolved<br><br>fal|se|
| | | | | | | |

4000

3000

ThePile

2000

s

e

1000

0

0 200 400 600 800 1000

RedPajama

My

10000

anything

London

8000

it's

latest

6000

woman

getting

###### S2ORC

ago

am

September

East

July

June

4000

December

Science

lost

took

offer

became

plan

High

States

percent

https

student

science

hard yet

2000

simply

head

But

once online

past

you

too

turn

law

enough

needed2008decision 40evidence

place

experience One

third

takelastpublic class

naturalleading

global P

rather

acrossrealwomen

likelyareas

peopleyear

additional

expected

also

News

0

0 200 400 600 800 1000

RedPajama

10000

love

8000

LAION-2B-en

6000

peS2o

November

City

music

October

THE

meetingEast

4000

Department

doing

Center

name

lost

my

became

offer

comes

false

And

put

city

That

science

international

hard yet

2000

simply

country

#

you

turn

law 2015

ensure

makes

thoughmembers program

issues 2010

University

users

understand needed2008decision provides economic

policy

death

open

naturalleading I ve

necessary fun

five

especially

2017

here

strong

By

along

up

most

you re

saying

0

0 200 400 600 800 1000

RedPajama

10000

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |continue|himself<br><br>instea<br><br>meth|d<br><br>od|
| | | | |recent mean|impact| |
| | | |enough<br><br>source role<br><br>needs<br><br>me|mber events|included<br><br>contact<br><br>problems played| |
| |then<br><br>becau muc<br><br>d|se h<br><br>id<br><br>always point<br><br>researproces<br><br>typs|chs<br><br>tarte<br><br>future ever<br><br>exampleprogram<br><br>took<br><br>control<br><br>self<br><br>industry|main<br><br>try<br><br>article<br><br>crea|ted land<br><br>offers<br><br>popula det<br><br>pro ma|r ails<br><br>jects teria<br><br>stage|
| |you<br><br>sojust<br><br>any these<br><br>well<br><br>pa<br><br>da|rt<br><br>differentta class<br><br>placesystem school<br><br>working<br><br>ThatState|s World<br><br>moneyalong<br><br>November online<br><br>spacem<br><br>s|en<br><br>hort<br><br>party<br><br>shows text<br><br>polic<br><br>ou F|e tside<br><br>win<br><br>playing K file<br><br>East<br><br>La|w|
| | | | | | | |

8000

6000

resources

4000

l

professional

2000

0

0 200 400 600 800 1000

RedPajama

| | | | |especially|British| |
|---|---|---|---|---|---|---|
| | | |film<br><br>December<br><br>industry|education<br><br>cou C|rt<br><br>ollege|Best|
| | | |June<br><br>went<br><br>While<br><br>m|en<br><br>sens|eland<br><br>air<br><br>Sc|ienc|
| | |U.S school<br><br>Natio|nal early<br><br>months|idea<br><br>de|al California<br><br>simply<br><br>average<br><br>Black<br><br>Street leve pro|ls<br><br>gram|
| |years But<br><br>mads|tille<br><br>days<br><br>That able<br><br>his<br><br>Th|tory<br><br>ese<br><br>World makes<br><br>needs<br><br>hal|f Of<br><br>plan<br><br>room<br><br>First<br><br>Eng|lishtakes<br><br>analysis<br><br>Group<br><br>fo|llow High|
| |this<br><br>like<br><br>We<br><br>could now<br><br>goo|d<br><br>highplace<br><br>system<br><br>ht|tp<br><br>open<br><br>it's<br><br>2013<br><br>sure<br><br>e<br><br>jo|b<br><br>close<br><br>former<br><br>2009<br><br>mean|via<br><br>stop<br><br>allow<br><br>See| |
| | | | | | | |

10000

8000

e

6000

TheStack

s

4000

2000

0

0 200 400 600 800 1000

RedPajama

###### Figure 24: RedPajama top 1,00 unigrams, and their corresponding indices in the other corpora.

10000

10000

10000

j

sequences

vector

measurement

8000

8000

8000

tumor

analyzed

analyses

sensitivity

el

mice

analyses

kg

r

variation

binding

probability

OpenWebText

±

6000

6000

6000

ml bound

intensity spin

0.5

calculated

mC4-en

genetic

variables

h

Z

genes

Fig

functional

###### C4

dose

Table

experiment

cm

random

mechanism

4000

wave

4000

resulting

4000

estimate

Y

framework

findings

noise

proof

respectively

component

approaches

scorestechnique

Thus

regions

confirmed technique

signal

symptoms

O

k

influence

revealed

Thus

m

relative

en

Let

domain

materials

Q

negativeQ

performed

treated

pattern

factor O

represent

fields

initial

approximately According

smaller

difference 1825 long various presence determined

Using

W

33 80

t

cancer

produced

error

2000

2000

2000

elements

appropriate

culture

described

direction produced 29

X

testing

described

mm

method

compared associated

presented

cell

independent

followed

80

normal training states loss

involved strong

S

previous

knowledge primary source

models >

input

note

appropriate

physical

followed

unit

nature

culture

2010

stage

lossenconsider

Theneffective

cannot

physical

final

education

levels test showedrange

analysis

method

limited

sites especiallyadded

60 management

section

loss

2009

cases

original

consider

global

terms14

however requiredJ

total

environment

particularpoints

2015An

R

version

interestcertain

report

value

needs

test would

however

third

cost

100

term

short

due

2017

community

/ such

many

made

no

many

over

its

thisbeen

?

/ more

dataset

0

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

S2ORC

S2ORC

S2ORC

10000

10000

signaling

8000

electron

mice

8000

8000

correlation tumor

detected

6000

algorithm

statistical

mice

6000

6000

RedPajama

measurement

density

ThePile

###### OSCAR

concentration

observations

stability

scores

continuous

experimental

cm

indicates

4000

simulation

associationwave

concentration

mm

structures

variable

Figure

4000

regression

4000

interaction

outcomes

represents

indicated

reduction

bound

consistent influence

phase

evaluate

patterns

outcomes

symptoms

output

channel combined

zero volume fixed

TheoremEq

Therefore

°

finite solutions

channel

detectioncomparisonderivedcontrast areasdirectly improve decrease

membrane

authors

smaller

equal

processing

correlation findings

2000

subjects

selection

presented

resolution testing

flow independent

index

contains33

gexpression associated f

determined

2000

2000

e.g O

unittesting

scale

processes

variables appliedimpact students

reduce

presence

caused

plant

equalpatterneducation log

mass

showed

although

applications note First

elements limited

N R

solutions

e.g

ratio

active

80

95

physicalpreviouslydistancelarger

physicalevidence

matrix °

status

k

60

difference

direction needed

Since

feature

processing

involved

highly

respect especially

parameters showsdescribedresponse

rates

element

studies

analysis

presence

Ucomplex

section

types

considered

impact seen

medical

certain

increasespecific single

training

systems

simple

changes

position

single

various

levelsrelated because lower

provided22

interest take

period

fact

due

problem

several

report

/ been case

2017

day

best

If

way

when

/ been

up

/ more

pathway

signaling

0

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

S2ORC

S2ORC

S2ORC

10000

10000

8000

whether

According

values

species

practice

likely

areas

expression

7000

stress

regression

linear

patients

8000

response

8000

mode

difference

consistent

significant

patient

therapy

estimate

6000

density

therefore

scores

thus

potential

spin

able

LAION-2B-en

5000

6000

6000

comparison

given

however

risk

TheStack

interest peak

follows

peS2o

ability

basis procedure

4000

water

done

prior

capacity co

growth

dependent

average

effects

factor

Section

4000

4000

needs

relation

become

frequency

certain

value

effect

3000

times

education

unit

within

However

COVID

hand

range performance

complete

age

process

production

since

behavior

los

products

both

2000

needs

future

features

binding real

performance

better

version33

information

post

2000

2000

original

linesvolume

per

below

therewell most

without

together

real

pattern

region

2008

v

\ Notemeasuresindicatedirectionsmallerspatialleadingvelocitystatusspinequalcriteriafoodunderstandingscoresconfirmedappropriatenumbersstability

free normal

1000

focus

If

cell

full

methods

vectorregionslikely Theorem infectionIIbehaviorgeneratedcontainingerrorcrossderivedchildrendetectedbestinfluence2017limitedvariabledirectlyoverallmedium95completemolecularnmcontext

show

either

work

note

theysame

log

2009

Allscale

before All

we /

eachintoits

other

27

V

while

/ thesewill levellevels

define

right

:

literature

membrane

Proof

follows

initial

characteristicsconductedEqreductiondomain

literature

whereas

measurement

velocity

symptoms

improved

Hencevariation

0

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

S2ORC

S2ORC

S2ORC

###### Figure 25: S2ORC top 1,00 unigrams, and their corresponding indices in the other corpora.

10000

10000

10000

pH

Results

assessed

investigated

8000

k

8000

8000

tumor

inflammatory

simulation

measurements

experimental

liver

cluster

parameter

proteins

sensitivity

graph

optimal spectrum

variation

variables

linear

dose

OpenWebText

6000

genes

6000

variables

6000

liver

calculated

l

mC4-en

immune

calculated

bound

characteristics

gene

indicates

h

Z

min

###### C4

Table

dependent

density

Introduction

variable

g

infection mm

decrease

protein

intervention

bound

4000

obtain

4000

4000

consistent

differences

2019

represents

contrast

regions

Based

experiment

resulting

severe

expression

identified

obtained

influence

mass

mobserved

discussed \

fixed

obtained

stress

volume

procedure

fields

determined

32 35

G

37

leadsaffectedequalrepresent unit

vs

smallerpattern

generation

determined

combined

p

initial

respect

determined

score

n

h

2000

antidetermine

2000

2000

output practice appropriate

cancer

COVID

caused

critical highly

contains

culture

sets applications

applied

27

valuesg

greater

II

L

length states however

models potentialconditions performance changes

understanding

active

properties according

population

cell

produced

sectionscale

states knowledgetypes primary

80 language

primary 40

23

loss blood via required

error

input

established

brain

appropriate

difference systems

analysis

pressure

activity

finallead

products

standard

impactindividual

particular

lower showsrole

particular

2008

D

Here

according

standard learning

ability what

systems

contentsource body

recent

actionpositionAn

followcannot

R

short2010

2008

points

account

economic

amonghuman being

version

field

2016amount

40

food

series

p

class

including

since

after

year

or such

first

thistime

It

this

µ

pH

0

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

peS2o

peS2o

peS2o

10000

10000

COVID

cluster

inflammatory

8000

baseline

mice

analyses

8000

8000

observations

genes

induced

liver

assessed

6000

evaluated

variation

Fig

boundary

kg

6000

6000

RedPajama

mechanisms experiment

quantum

ThePile

###### OSCAR

studied

spin

tissue

decrease

vector

whereas

4000

association

dose criteria

framework

demonstrated

variable

Figure

infectioncontaining

4000

4000

responses

technique

Moreover

optimal

infection

resulting

Fig

patterns

proposed

boundary

resistance

accuracy

nm

sensitivity

severe

outcomes

indicates environmental confirmed

signal conducted

protein

evolution

equal

i.e

treated

mL

°

obtain

assessment

stress

concentrations estimated

particles

map lack

wave

Using

z

dose

2000

cmincreases

Q

selection 35

index

determinedfollows ·

identify

demonstrated components processes

2000

2000

distance

G

identify

population

contains USA vs

e.g

testing 31

participants

established

domain

H

proposed

overall

generation

vs

c

vsestablished language

co

proteinsimpactknowledgeindependent seen

strategy

here 23 individuals Y input

e.g

numbers

independent

r

constant Wsource

normal

cancer

lines

gas

mode

frequency °

types

k

difference average

stage

training

Therefore paper

highly cost

studies

provides

Here

developed via

effective ·

role

step

X

imagesource

specificKsize

x

function value<

medical

simple

clear

thus

five shortprovides

Table compared respectively

performancerole

amount

result

rate

conditionsriskstructure19

First

howeverlight

age

C

women

either

low

Epresent

along

terms

T

key

large

made

right

after

thisn

thistime

so

this

\

parameter

0

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

peS2o

peS2o

peS2o

comparison

growth

developed

10000

10000

reported

pain severe 2008

Fig

top

1000

near

becomesimulation

areas

nature

expression

serum

stress

food suggestedmainlyResultssoftware strategy mode

produced

applications

compared

patients

statushighest optimalparticle

8000

response

8000

mode

individual

determined

received

800

levels

procedure

brain

resistance mL degree authorsrathercannot

economic

×

effects

expected

literatureapplicationevents 2018

thus

complex

spin

followedindicatedposition

represent

active

identify

recent error

LAION-2B-en

networks

children analyzedincludearoundresulting

?

layer

6000

6000

TheStack

mg

presented

follows

protein

###### S2ORC

tissuemaximum components vector

become

600

find

function

according

higher

further

h

systemsaverage paper

access

treatment

Section

4000

4000

curve

results

become

software

shown

did

400

resolution

experience

various

shown

×

represents

within

hand

months

weight

those point

co

measure

gas37

First

called

numbers secondary

applied

products take feature

thenlevel

across along

change

flow

Table

above

Using

days

best

post

2000

2000

plant making

groups

random

signal

sources

nature

200

} support

per

below

single

zero

d

These

low

closehelpopen

lasttake

within total

whether

numberP

focus

called Data

32

features

children

J

commonrelative

As

way

them

single

software

|

specific

nowY

26

line proposed

they

i

In system

information

language receptor

beenx

15

11

this

To

this

=

}

revealed

evaluated

mainly

genetic

PCR

participants

therefore

analyzed

domain

measurements

represent

stablerepresents

evolution

0

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

peS2o

peS2o

peS2o

###### Figure 26: peS2o top 1,00 unigrams, and their corresponding indices in the other corpora.

| | | | |viewer| | |
|---|---|---|---|---|---|---|
| |Cover<br><br>Kids<br><br>Col Pa|lection ck<br><br>Metal|Cat| |Princess| |
| |Card<br><br>S|mall<br><br>Girl<br><br>Heart|CD<br><br>Phone<br><br>| |Tea<br><br>Close StudBoo|ksio|
| |Silver<br><br>Light<br><br>Table|Download|Galaxy|HeadMou<br><br>Pint<br><br>s|ntain<br><br>erest<br><br>ilver<br><br>PlanFilm<br><br>Max<br><br>Fi|eld|
| |stock A|ir<br><br>Power<br><br>Squar|e Me<br><br>bag Op|en<br><br>tree<br><br>Disn|ey<br><br>Valley<br><br>TO<br><br>01<br><br>Actio Wars coff|n<br><br>ee|
| |Photo<br><br>New<br><br>x<br><br>How Wedding black s|house25 Furniture<br><br>table<br><br>Great<br><br>Thu<br><br>19<br><br>Bir<br><br>Ap|mbnailthday<br><br>ple<br><br>Casual<br><br>front21San<br><br>Ju Ba|ne Flatckground<br><br>Angeles match<br><br>Recipes<br><br>Amer Co|ica lorsTipsMouse space diagram team<br><br>holdin|g Most|
| | | | | | | |

10000

8000

OpenWebText

6000

4000

2000

0

0 200 400 600 800 1000

LAION-2B-en

| |Wom|en's| | | | |
|---|---|---|---|---|---|---|
| |vector|illustration| | |Wide<br><br>Dual Autom|atic|
| |Print Ima<br><br>Pa<br><br>St|ges<br><br>ck<br><br>yle<br><br>Hand<br><br>HairDiamond|Crystal AP| |Ice<br><br>Tips BMW Close<br><br>Toyota| |
| |Photo<br><br>Size<br><br>Kids202<br><br>G|0<br><br>lass<br><br>Electric<br><br>Gallery|Paper<br><br>Door<br><br>DVD|Stone Pro|duct<br><br>Stud|io|
| |Light Part|y<br><br>Modern<br><br>Family Pro<br><br>Hap V<br><br>Y|py<br><br>oungOutCD<br><br>IN<br><br>ShopMeFire|War<br><br>Franc<br><br>Histor rens|e<br><br>y ilvert<br><br>Go<br><br>brown holdinG|eorgg|
| |Black<br><br>or<br><br>House City<br><br>blu|e<br><br>Two<br><br>North<br><br>Great<br><br>my<br><br>Unite<br><br>w|d<br><br>all<br><br>year<br><br>gold 0<br><br>21<br><br>Tote<br><br>July<br><br>bag<br><br>Ba|ckgroundPants<br><br>paper<br><br>gift Amer|ica<br><br>Washington October<br><br>Gifts<br><br>Will College 2007<br><br>Vinyl<br><br>Januar plansma|chiney|
| | | | | | | |

10000

8000

6000

###### C4

4000

2000

0

0 200 400 600 800 1000

LAION-2B-en

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |Pants|Unisex| |
| | |Logo|Casual<br><br>Painting<br><br>Ab|stract<br><br>Toy<br><br>Recipes|Holder<br><br>Traditio|nal|
| |vector<br><br>Dress<br><br>Pictu|re Girl<br><br>Colorin|g<br><br>R|ange Luxury|Lot<br><br>Close<br><br>Vinyl| |
| |Kids<br><br>Pink|Men Gift<br><br>GalleryHot<br><br>G|Superardenpattern<br><br>featuring<br><br>Country<br><br>Orange<br><br>Phone|Grand tree<br><br>ren|t<br><br>MountStory<br><br>USB<br><br>Credit<br><br>Cable Stud<br><br>Sa<br><br>Vo|io<br><br>fety<br><br>l|
| |3 7<br><br>Day<br><br>Long Table Part blu<br><br>on|y<br><br>e<br><br>e<br><br>Business<br><br>Old StreetUSA 3D|ReviewMachine bedroom UKGroup R<br><br>Abo<br><br>Op|ut<br><br>glassen<br><br>which<br><br>gift modernAmer<br><br>2009 years<br><br>36|ica NotOctober floorCanada Fe|Googlebrua<br><br>heart|
| | | | | | | |

10000

8000

christmas

6000

mC4-en

4000

2000

ryle

0

0 200 400 600 800 1000

LAION-2B-en

| | | | | |Dual| |
|---|---|---|---|---|---|---|
| |Leather<br><br>Men's| |Ba|Flatckground| | |
| |BagVintage<br><br>br|N|avy<br><br>Load<br><br>Ba|nd|Ice<br><br>oz<br><br>gr|ey|
| |Cover<br><br>Ideas<br><br>Wom<br><br>Colo<br><br>Fro<br><br>L<br><br>G|en's<br><br>r<br><br>nt<br><br>arge<br><br>lass<br><br>HandMini<br><br>ElectricHeart|Custom<br><br>TreeMovie featuring CD|Smart<br><br>Gas|icons<br><br>flowerAdult| |
| |PhotoGreen<br><br>Kids|Man<br><br>NaturalHapLiv<br><br>w|ingpy<br><br>all Oil<br><br>Live<br><br>Lake<br><br>Page|s<br><br>Bay<br><br>Rock Franc<br><br>Child|e<br><br>ren Area<br><br>Performanccheap<br><br>Chicagoholdin<br><br>Actio<br><br>coff<br><br>Sa<br><br>Ha<br><br>P|e<br><br>g<br><br>n<br><br>ee<br><br>fety<br><br>ll<br><br>lace|
| |Blackx<br><br>Sale<br><br>black City<br><br>Up|red Back room art<br><br>two<br><br>Tim|e<br><br>our<br><br>OF<br><br>40<br><br>Op<br><br>T|en<br><br>eam David<br><br>Tee<br><br>Jamessummer<br><br>Tex<br><br>jus<br><br>36|as<br><br>t<br><br>OnlyClose textmU|buildingense<br><br>p|
| | | | | | | |

10000

8000

6000

###### OSCAR

4000

2000

g

0

0 200 400 600 800 1000

LAION-2B-en

| | | | |Fall| | |
|---|---|---|---|---|---|---|
| |Ed L|ition arge<br><br>Gallery|S|ign|Wars| |
| |Size<br><br>S|mall<br><br>Video<br><br>Squar|e<br><br>Sports|Mou|ntain Story<br><br>Credit<br><br>Boo|ks|
| |Photo Case<br><br>DesignWall Women<br><br>Book<br><br>F|Watchull<br><br>Fo|od Real Line<br><br>Fire|AngelesAndroid<br><br>Space<br><br>Specia|l<br><br>Max<br><br>diagram End| |
| |Black<br><br>Green Daybr<br><br>202|0<br><br>Old kitchen<br><br>Great|Ind|ia|Canada<br><br>Florida<br><br>Who<br><br>Und<br><br>coff<br><br>L|er<br><br>ee<br><br>a|
| |8<br><br>that<br><br>Your Table<br><br>Slee<br><br>blu|ve<br><br>e housestyle cover<br><br>LaceCoffeC|e<br><br>Center<br><br>US<br><br>2021Cat<br><br>OF 2010<br><br>young De<br><br>|Fabricsigns what<br><br>food<br><br>2009<br><br>Hea<br><br>Ch|lth<br><br>ocolate<br><br>Not<br><br>Coat<br><br>floor<br><br>ToysSilk via BeaPor|traitr|
| | | | | | | |

10000

8000

6000

ThePile

4000

2000

0

0 200 400 600 800 1000

LAION-2B-en

| | | | | | | |
|---|---|---|---|---|---|---|
| |Ideas<br><br>Pink|Yellow<br><br>PhotographyDiamond| | |Beauty<br><br>Bea|r|
| |Print<br><br>Sale<br><br>Size<br><br>Card<br><br>Steel<br><br>Fro<br><br>Fas<br><br>St<br><br>L<br><br>G|nt<br><br>hion<br><br>yle<br><br>arge<br><br>lass|leather<br><br>jpg|Auto|Ice<br><br>flower gr|ey|
| |Kids<br><br>br|Short<br><br>N|avy<br><br>Oil| |Wars| |
| | |View<br><br>kitchen|pattern<br><br>Night Out Make<br><br>AP|DisnW F|eyay our<br><br>Story AwardsClass<br><br>Actio<br><br>EndLa|n<br><br>nd|
| |Black<br><br>Photo<br><br>or<br><br>stock<br><br>Vintage<br><br>black|style St<br><br>Street North<br><br>imag W<br><br>w<br><br>Of<br><br>Ea|ficees<br><br>rrings<br><br>Review E<br><br>Group<br><br>II<br><br>vs|party<br><br>Over<br><br>long<br><br>kids County<br><br>Be<br><br>near<br><br>News<br><br>box<br><br>Buttonso<br><br>Part Le|agueProject<br><br>Friday Will CollegeCanada t 29<br><br>Comp<br><br>Por<br><br>Fe<br><br>Copr<br><br>G<br><br>L|any<br><br>trait<br><br>brua<br><br>oducurt<br><br>eorg<br><br>a<br><br>British<br><br>Google|
| | | | | | | |

10000

8000

6000

RedPajama

4000

2000

le

t

ry

0

0 200 400 600 800 1000

LAION-2B-en

10000

10000

TO

Mouse

Chicago

Forest Chicago

Light

flowers

card

Full

Space

TV

Panel

Full

TV

card

8000

8000

Women

British

Day

Images Brown

Day Women

Park

Young

Park

Power

About

Small

3d

Class

Training

office

Set

Abstract

car

Board

car

6000

6000

School

TheStack

Set

###### S2ORC

Black

peS2o

Black

St

City

brown

brownpage

August

01

September

HD

Animal

HD

THE

Four

house

Green

July

looking

Design

July

4000

4000

How

January

January

North

art

https

http

What

http

video

Only

video

box

United

team

diagram

2000

2000

CA

white

her

heart

room

de

old people

2008

background

vs

USA

vs right

29

2018

women

againstAn

goodJ Shorts

yearpeople

design

To

threeBear

whenPrintable

Mebeautifuli

Braceletused

have

haveBedroom

;

=

Series

Royalty

Logodress

ThumbnailDecor

Greeting

LampQueen

Bags

Adult

Big

Thumbnail

Greeting

Premium

Rug

Tour

Lot

Vinyl

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

LAION-2B-en

LAION-2B-en

| | | | | | | |
|---|---|---|---|---|---|---|
| |Book<br><br>S|mall<br><br>Old|Super<br><br>Dark|Head|Italy holdin|g|
| |stock<br><br>House<br><br>Serie<br><br>Part|s<br><br>y<br><br>Park<br><br>house|Frame<br><br>Me<br><br>JuA|nepril|Januar|y|
| |Black<br><br>Case<br><br>UpF|ull|California<br><br>Abo|ut ft| | |
| |Top<br><br>design|room<br><br>North<br><br>USA<br><br>May<br><br>boo 3D|k<br><br>made More|Control<br><br>A|fter<br><br>three team<br><br>Comp<br><br>Actio<br><br>Store<br><br>gr|any<br><br>n<br><br>ey|
| |; Steel<br><br>he<br><br>R|r<br><br>oyalty<br><br>day<br><br>illustrationBednotpictu|res<br><br>year<br><br>card<br><br>System<br><br>Op|en<br><br>Interior<br><br>what<br><br>Disn<br><br>jus|ey<br><br>t<br><br>price<br><br>been<br><br>FOR<br><br>CoolResumeCare<br><br>J<br><br>FestivaSilk<br><br>End|l|
| | | | | | | |

10000

8000

6000

4000

2000

0

0 200 400 600 800 1000

LAION-2B-en

###### Figure 27: LAION-2B-en top 1,00 unigrams, and their corresponding indices in the other corpora.

| | | | | | | |
|---|---|---|---|---|---|---|
| |devconst src|z| | |000<br><br>Desc|riptio|
| |div<br><br>font<br><br>String l| |min template| | | |
| |width<br><br>_<br><br>mar<br><br>tran<br><br>06|gin<br><br>slation<br><br>Nam|e<br><br>container<br><br>|Y<br><br>log|documenta<br><br>1.5 settings|tion<br><br>absolute|
| |^ string<br><br>01|column en<br><br>contex<br><br>packa amp|t<br><br>ge<br><br>parent output<br><br>la48|37H tree<br><br>libra|ry<br><br>download<br><br>load<br><br>frame<br><br>W<br><br>fram brow|ewor ser|
| | |18 dependencies<br><br>50<br><br>D field<br><br>X|main<br><br>app<br><br>response<br><br>ar|makeia<br><br>unit<br><br>doc st|ylesheetu001b<br><br>90<br><br>normal<br><br>coordi<br><br>black<br><br>et<br><br>acc<br><br>|nate<br><br>ount|
| | | | | | | |

10000

n

8000

OpenWebText

6000

ute

4000

k

2000

s

0

0 200 400 600 800 1000

The Stack

| | | | | | | |
|---|---|---|---|---|---|---|
| | |align|float| |op| |
| |spanr font mar|gin| | |que| |
| |import<br><br>o<br><br>02<br><br>tran06<br><br>pl<br><br>0<br><br>l|slation<br><br>ugin<br><br>8<br><br>column<br><br>Namamp|e|37|52<br><br>com<br><br>exception tab<br><br>Us|er|
| |b|array<br><br>32|tag<br><br>la 35<br><br>co as|mmand sets<br><br>H<br><br>Y|functions<br><br>fram|ewor|
| |class<br><br>@<br><br>! or<br><br>16de|msgidlt<br><br>button typesoption break values26<br><br>hel|per<br><br>lengthP filesresponse conne<br><br>into|ction<br><br>theme unit<br><br>doc<br><br>tree<br><br>neev|endorDird<br><br>Get<br><br>THEUse<br><br>total<br><br>coordiwant<br><br>web<br><br>2011<br><br>000|alphanate000|
| | | | | | | |

10000

8000

6000

###### C4

4000

k

2000

s

0

0 200 400 600 800 1000

The Stack

| | | | | | | |
|---|---|---|---|---|---|---|
| |nbsp|dd| | | |alpha|
| |integrityString0.0pl|ugin<br><br>|h3<br><br>php<br><br>em| | |4.0|
| | | |0.5<br><br>disable<br><br>mo<br><br>a|d<br><br>dules<br><br>rguments<br><br>API<br><br>engines specified<br><br>keys<br><br>reso|lve<br><br>admin|absolute|
| |const|k<br><br>none<br><br>contex|t<br><br>instance<br><br>tag hidden<br><br>as|sets|expressiontransition<br><br>Source<br><br>settings| |
| |!<br><br>requires<br><br>02<br><br>border<br><br>ul|L<br><br>can<br><br>en<br><br>preformat<br><br>shou<br><br>m<br><br>s|ld<br><br>ax<br><br>ection<br><br>reference<br><br>more<br><br>app<br><br>output<br><br>assert<br><br>before|head<br><br>print<br><br>filter given<br><br>global details<br><br>do|cument<br><br>softwareeithercommon<br><br>remove<br><br>x2<br><br>pro<br><br>10|vide<br><br>00<br><br>sizeof|
| | | | | | | |

10000

8000

licenses

6000

mC4-en

ute

4000

2000

d

0

0 200 400 600 800 1000

The Stack

| | | | | | | |
|---|---|---|---|---|---|---|
| |div<br><br>void| |0|.00<br><br>shadow|a|rrow|
| |mar|gin| | | | |
| |static<br><br>l|define summary<br><br>column|6|4<br><br>Version|port<br><br>AS<br><br>duration|absolute|
| |` o obje de|ct<br><br>scriptionblockmenu| |catch<br><br>bar<br><br>pull<br><br>styl|es<br><br>fields NO<br><br>10|T<br><br>00|
| |if<br><br>return<br><br>06|kind<br><br>request<br><br>break<br><br>rel<br><br>31 2017<br><br>ev 1|ent<br><br>.0.1<br><br>40child<br><br>regexnavbar<br><br>infoOF application<br><br>matc<br><br>F|h<br><br>1.1.0ormy<br><br>protected<br><br>filenamethe|y<br><br>W<br><br>created<br><br>names<br><br>7.0.0<br><br>events<br><br>passwo<br><br>over000<br><br>clo<br><br>siz lig tt|rd<br><br>000<br><br>se e_t ht|
| | | | | | | |

10000

8000

transparent

6000

OSCAR

ute

4000

2000

0

0 200 400 600 800 1000

The Stack

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |na|vquot|jpg|Ex|ception<br><br>| |
| |pl|ugin api|Value<br><br>boolean assert<br><br>optional export<br><br>ob|j<br><br>ctx<br><br>Date<br><br>ove|rride Content siz|e_t|
| |tran|slation<br><br>th<br><br>heade<br><br>dt|r|API<br><br>catch 43filter<br><br>supportsTest<br><br>co|nfiguration port transition<br><br>exception click| |
| |span x A<br><br>link<br><br>de|scription<br><br>define time<br><br>kind<br><br>breakdat s|e ection<br><br>char<br><br>basetry<br><br>address runcall<br><br>server|root unitglobal|note<br><br>success stop<br><br>safeframe eventselemen operat web incl<br><br>|ts<br><br>ion continueudes<br><br>_blank|
| | | | | | | |

10000

8000

repository

6000

ThePile

4000

2000

ue

k

0

0 200 400 600 800 1000

The Stack

| | | | | |ANY| |
|---|---|---|---|---|---|---|
| |plna|uginv|str|byte| |outline|
| |github.|com|float<br><br>h2 radius|attribute| | |
| |font|dd<br><br>dt|offset<br><br>exportsym|bol<br><br>engines<br><br>ve|ctor<br><br>1.5|absolute|
| |param|html<br><br>z<br><br>in|terface<br><br>True<br><br>files<br><br>q<br><br>42|41<br><br>custom<br><br>specified<br><br>category<br><br>4749ma|ster<br><br>expression<br><br>Number<br><br>que|alpha|
| |span<br><br>%<br><br>00<br><br>o02String03<br><br>yo|u<br><br>target<br><br>all<br><br>O<br><br>50<br><br>coredisplay values ev<br><br>we|ent<br><br>ight<br><br>h3<br><br>response|fn<br><br>comm|aheadent<br><br>tbody<br><br>either 7.0.0 eventsblackacc siz lig|ount<br><br>e_t<br><br>ht|
| | | | | | | |

10000

8000

transparent

6000

RedPajama

4000

ute

2000

Component

0

0 200 400 600 800 1000

The Stack

10000

10000

metadata

Value

ahead

script

repository

API

card

dl

8000

8000

tr

layout

documentation

react

pull

auto

_

comment

Number

tr

port

LAION-2B-en

transparent

fill

6000

6000

###### S2ORC

assets

peS2o

`

05border

3.0

05

- 00

width

- 01

page

000

attribute

Use

Use

default

span

exception

Test

4000

4000

break

document

OF

Type

!

supports

options

my

row

select

1.0

false

static

57

stroke

59 52

valid

os

check

@!

51

child

2000

2000

date

coordinates black account

1000

link

he

classes duration

max

property

solid

service

2000

path

location position2017

48

language

requires

45

36

element

core

output application

\

fields

close provided

close

task

\

final

aria 60 In

2020

primary

parameter

40

light

r

useI

throwsfootersame

then

R

menuusing

License

tsd

Include

xs

md

ansi

svg

git0.00unsigned

sidebar

x1

xl000000

es

tt

0

0

0 200 400 600 800 1000

0 200 400 600 800 1000

The Stack

The Stack

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | |op<br><br>remove| |
| |string| |address<br><br>throw| |expect|4.0ed|
| |id<br><br>message<br><br>itemheigh|t| |device|AS<br><br>safe57<br><br>network<br><br>000<br><br>flags| |
| |span<br><br>`<br><br>lab|el<br><br>09<br><br>License<br><br>final|sym<br><br>item<br><br>in|bol<br><br>s<br><br>formation<br><br>theme37<br><br>details|stop<br><br>such<br><br>ste|p|
| |class<br><br>% integrity<br><br>01<br><br>ul<br><br>03<br><br>1|5<br><br>stroke<br><br>when struct used<br><br>shou<br><br>28 hel<br><br>ce|ld<br><br>per<br><br>nter under<br><br>childmedia<br><br>endif<br><br>45 model<br><br>double servic spa<br><br>coas|e ce<br><br>mmandsetscomponents<br><br>H<br><br>2.1.0<br><br>Classnow<br><br>wind<br><br>Inc<br><br>w|ow<br><br>lude<br><br>ithout<br><br>Array<br><br>want<br><br>black<br><br>web<br><br>framque|ewor|
| | | | | | | |

10000

8000

6000

4000

2000

k

0

0 200 400 600 800 1000

The Stack

###### Figure 28: The Stack top 1,00 unigrams, and their corresponding indices in the other corpora.

JS Distance (Unigrams Intersection)

JS Distance (Unigrams Union)

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
|0.14| | | | | | | | | |
|0.22|0.18| | | | | | | | |
|0.23|0.18|0.14| | | | | | | |
|0.28|0.27|0.22|0.28| | | | | | |
|0.25|0.23|0.18|0.17|0.25| | | | | |
|0.32|0.33|0.33|0.36|0.3|0.39| | | | |
|0.31|0.32|0.33|0.36|0.31|0.39|0.035| | | |
|0.4|0.41|0.35|0.4|0.4|0.42|0.39|0.4| | |
|0.54|0.53|0.45|0.49|0.42|0.47|0.49|0.49|0.49| |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
|0.19| | | | | | | | | |
|0.29|0.23| | | | | | | | |
|0.27|0.21|0.19| | | | | | | |
|0.32|0.33|0.28|0.33| | | | | | |
|0.27|0.26|0.24|0.21|0.29| | | | | |
|0.4|0.4|0.42|0.45|0.36|0.46| | | | |
|0.39|0.4|0.42|0.44|0.36|0.45|0.061| | | |
|0.51|0.5|0.43|0.49|0.52|0.51|0.56|0.56| | |
|0.61|0.61|0.53|0.58|0.49|0.56|0.58|0.59|0.62| |

OpenWebText

OpenWebText

C4

C4

mC4-en

mC4-en

[Figure 21]

[Figure 22]

0.6

0.5

0.5

Oscar

Oscar

0.4

0.4

The Pile

The Pile

0.3

0.3

RedPajama

RedPajama

0.2

0.2

S2ORC

S2ORC

0.1

0.1

peS2o

peS2o

0.0

0.0

LAION2B-en

LAION2B-en

The Stack

The Stack

OpenWebText C4 mC4-en Oscar ThePileRedPajama S2ORC peS2oLAION2B-en TheStack

OpenWebText C4 mC4-en Oscar ThePileRedPajama S2ORC peS2oLAION2B-en TheStack

(a) Intersection JS distance

(b) Union JS distance

Figure 29: The Jensen Shannon distance between the top 1,000 most common unigrams in each corpus. The lower the numbers the more similar the corpora are. OpenWebText, C4, mC4-en, OSCAR, The Pile and RedPajama are quite similar to one another (in terms of the common unigrams distribution), and S2ORC, peS2o, LAION-2B-en, and The Stack are quite different from all other corpora.

- Table 23: Top 10 exact text overlaps between more than 2 datasets. C4, OSCAR, and RedPajama share the most amount of documents, with over 1.6 million shared documents. Interestingly, even LAION-2B-en, an image-caption corpus overlaps with other corpora, such as C4 and RedPajama (which all share more than 30 thousand documents).

Corpus Intersection Count C4 ∩ OSCAR ∩ RedPajama 1,680,953 C4 ∩ mC4-en ∩ RedPajama 1,375,088 The Pile ∩ RedPajama ∩ The Stack 592,364 C4 ∩ The Pile ∩ RedPajama 118,432 C4 ∩ RedPajama ∩ LAION-2B-en 30,602 mC4-en ∩ OSCAR ∩ RedPajama 14,319 C4 ∩ mC4-en ∩ OSCAR 12,854 C4 ∩ mC4-en ∩ OSCAR ∩ RedPajama 12,854 OSCAR ∩ The Pile ∩ RedPajama 6,112 C4 ∩ OSCAR ∩ The Pile 6,096

Unigram Overlap Next, by comparing the 10,000 most common unigrams, we compare the similarity between each corpora pair using the Jensen Shannon distance using (1) the intersection and (2) the union of the two vocabularies. We present the results in Figure 29. On average, we find that OSCAR’s unigram distribution is the most similar to all other corpora (0.19 on average). The Stack, as expected, is the most distance corpus from all other corpora.

- B.4.2 CORPUS OVERLAP

In this analysis, we compute the overlap between the different corpora, by comparing (1) the texts, and (2) the URLs, when available. The pairwise results are presented in Figure 30 for the texts overlap, and Figure 31 for the URL overlap. We see that text overlap diminishes quickly to zero as more datasets are considered. Table 23 shows the largest text overlaps between more than two datasets. While the largest two are over 1 million document clusters, this is less than 1% of clusters in any of the involved datasets, and overlap size drops rapidly from there. This trend is similar for URL overlaps. The largest 3-corpora overlap is between C4, mC4-en, and OSCAR, with 6,767,877 shared URLS, while the rest of the overlaps share at most a single URL.

We find that documents from S2ORC and peS2o do not appear in other corpora. While it is likely that some of the academic papers are shared with other corpora, e.g., The Pile and RedPajama

Count of Overlaps (|D1 D2|)

Ratio of Overlaps to Unique Documents in D2 (|D1 D2| / |D2|)

[Figure 23]

[Figure 24]

8M 1.9K 94 333 3.9M 1.9K 0 0 2 1.8K

- 1E+00 5E-06 2E-08 1E-06 3E-02 3E-06 0E+00 0E+00 1E-09 3E-06
- 2E-04 1E+00 4E-04 6E-03 9E-04 5E-01 0E+00 0E+00 2E-05 8E-07

OpenWebText

OpenWebText

1.9K 365M 1.4M 1.7M 119K 365M 0 0 30.6K 423

C4

C4

94 1.4M 3.9B 257K 4.8K 1.7M 0 0 88.5K 113

- 1E-05 4E-03 1E+00 9E-04 4E-05 2E-03 0E+00 0E+00 6E-05 2E-07

- 4E-05 5E-03 7E-05 1E+00 3E-04 3E-03 0E+00 0E+00 5E-05 1E-06
- 5E-01 3E-04 1E-06 2E-04 1E+00 1E-03 0E+00 0E+00 1E-05 2E-02

- 2E-04 1E+00 4E-04 6E-03 7E-03 1E+00 0E+00 0E+00 2E-05 2E-02

mC4-en

mC4-en

UniqueDocumentsinDataset1(D1)

UniqueDocumentsinDataset1(D1)

333 1.7M 257K 287M 45.2K 1.7M 0 0 66K 530

OSCAR

OSCAR

3.9M 119K 4.8K 45.2K 137M 934K 0 0 14.2K 8.9M

The Pile

The Pile

1.9K 365M 1.7M 1.7M 934K 690M 0 0 31.2K 11.2M

RedPajama

RedPajama

0 0 0 0 0 0 9.3M 0 0 0

0E+00 0E+00 0E+00 0E+00 0E+00 0E+00 1E+00 0E+00 0E+00 0E+00

S2ORC

S2ORC

0 0 0 0 0 0 0 8.2M 0 0

0E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00 1E+00 0E+00 0E+00

peS2o

peS2o

2 30.6K 88.5K 66K 14.2K 31.2K 0 0 1.4B 12.2K

2E-07 8E-05 2E-05 2E-04 1E-04 5E-05 0E+00 0E+00 1E+00 2E-05

LAION-2B-en

LAION-2B-en

1.8K 423 113 530 8.9M 11.2M 0 0 12.2K 544M

2E-04 1E-06 3E-08 2E-06 7E-02 2E-02 0E+00 0E+00 9E-06 1E+00

The Stack

The Stack

OpenWebText C4 mC4-en OSCAR ThePileRedPajama S2ORC peS2oLAION-2B-enTheStack

OpenWebText C4 mC4-en OSCAR ThePileRedPajama S2ORC peS2oLAION-2B-enTheStack

Unique Documents in Dataset 2 (D2)

Unique Documents in Dataset 2 (D2)

- Figure 30: Overlaps of hashed full text between all pairs of datasets as counts and as ratio to dataset size.

[Figure 25]

C4 mC4-en OSCARLAION-2B-en

Unique URLs in Dataset 2 (D2)

C4

mC4-en

OSCAR

LAION-2B-en

UniqueURLsinDataset1(D1)

365M 149M 12.7M 6

149M 3.9B 120M 132

12.7M 120M 428M 4

6 132 4 2.2B

Count of Overlaps (|D1 D2|)

[Figure 26]

C4 mC4-en OSCARLAION-2B-en

Unique URLs in Dataset 2 (D2)

C4

mC4-en

OSCAR

LAION-2B-en

UniqueURLsinDataset1(D1)

- 1E+00 4E-02 3E-02 3E-09

4E-01 1E+00 3E-01 6E-08

3E-02 3E-02 1E+00 2E-09

- 2E-08 3E-08 9E-09 1E+00

Ratio of Overlaps to Unique URLs in D2 (|D1 D2| / |D2|)

- Figure 31: Overlaps of URL string between all pairs of datasets as counts and as ratio to dataset size.

that included arXiv as a data source, there are likely formatting differences that cause the exact string matching to be different. Interestingly, even S2ORC and peS2o do not contain any exact-text overlapping documents, despite peS2o being a cleaned version of S2ORC, due to a difference in formatting for parsed paper sections.

While RedPajama is 2.5 times larger than C4 in number of documents and 6.6 larger in number of tokens, we find that 50% of RedPajama unique documents originate from C4. This can be explained by larger documents (as evident from the largest average document length in The Stack of 2,800 tokens per document on average, compared to 420 tokens per document in C4, or by duplicate contents of

- C4 documents in RedPajama. Similarly, 50% of OpenWebText unique documents overlap with The Pile, which includes OpenWebText as a source. Another expected overlap is between datasets with Github as a source (RedPajama and The Pile), and The Stack (which purely consist of Github code).

Finally, we also notice that while mC4-en was created from a superset the Common Crawl data used to make C4, documents from C4 only constitute 0.04% of mC4-en, while the later is only 10 times larger in size. We speculate that this is due to formatting differences, between the C4 and mC4-en collection.

C LIMITATIONS

WIMBD has a few limitations, described below:

- • The search tool we use is Elasticsearch. While it is scalable, it was not designed for scaling with large text corpora. In addition, indexing these massive text corpora can take a few days,

- and keeping it running is costly. In the future, we hope to explore more cost effective and faster indexing tools.
- • Search is currently enabled using Elasticsearch, which only enables exact-match search. Fuzzy, and semantic search are important abilities that we currently do not support.

- Table 24: Time benchmark of the different analyses on C4. We ran all of these analyses on a 224-CPUs machine, with 881 Gb memory. * The contamination time was calculated on the test set of COPA, which contains 500 test examples. We also report the estimated cost in dollars based on Google’s pricing of the machine we used, that is $9.46 per hour.

#### Category Analysis Time Estimated Cost ($)

Summary Statistics 6:32 1 Internet Schemas 2:25 0.4 Internet Domains 5:38 0.9 Internet Domains per Token 3:32:07 33.4 Internet Suffixes 1:56 0.3 Utterance Date Statistics 2:12 0.3 Geolocation 1:17 0.2 Language ID 5:52 0.9

DataStatistics

- Top-1 9:08 1.4
- Top-2 2:14:26 21.2
- Top-3 5:45:10 54.4 Top-5 3:43:58 35.3 Top-10 8:43:40 82.6 Top-100 3:00:14 28.4 Bot-1 18:17 2.9 Duplicates 8:36 1.4 Length Distribution 8:56 1.4

DataQuality

Contamination *:48 0.1 Toxic Classifier 3:19:12 31.4 Toxic Taxonomy 3:15:27 30.8 PII 24:44 3.9 Demographic Sentiment 11:41:17 110.5

Measures

Comm.

Total 46:51:51 443.1

- D BENCHMARKING RUNTIMES

This section describes the benchmark times each analysis took to run on the C4 corpus. While C4 is not the largest corpora we analyze, it is a popular one, and representative in size. All out analyses were run on a Google cloud compute node with 882GB RAM and 224 CPUs. While the machine is rich in RAM, our analyses typically did not use more than 250GB, and the reason for choosing such machine was the availability of a machine with enough CPU cores, that came along with this amount of memory.

We report the benchmark runs in Table 24. All of the analyses we conducted took less than 12 hours to run, with 13 (out of 22) that took only several minutes, and all of the analyses on C4 took an estimated of 46 hours and 51 seconds (excluding repeated runs, and the contamination analyses on other evaluation datasets). Note that while the measured time for each run were calculated using the TIME command in linux, there is some variance, and those should be taken as a rough estimate.

We also calculate the estimated costs for each analysis and report it in the same table (Table 24). We use the estimated $9.46 per hour based on https://cloud.google.com/compute/all-pricing for our calculations, making the total cost on C4 $443.1.15

15This estimation does not include the Elasticsearch hosting costs.

- E TECHNICAL DETAILS

This section describes the algorithms for computing the most common, least common, and total number of unique n-grams in a large corpus. Each of these algorithms uses the same trick that was inspired by Bloom filters (Bloom, 1970) as described in section 3.1. As a result these algorithms do not provide exact results, and the accuracy is determined by the amount of memory available for the hash table.

- E.1 MOST COMMON n-GRAMS

To collect the (approximate) top-k n-grams we start by initializing a hash table of zeros (either u32 or u64) which represent occurrence counts for each n-gram, and an empty collection of the top-k n-grams. Then we iterate over the n-grams in the corpus and for each n-gram encountered we take its hash, increment the corresponding count in the hash table, and if that count is at least as large as the current minimum count in the top-k we add that n-gram to the top-k, potentially evicting another n-gram from the top-k.

After completing the iteration over the corpus the top-k will be complete and, in the absence of hash collisions, correct. However, the larger the corpus is relative to the hash table, the higher the probability of hash collisions. A large enough corpus will have more unique n-grams than there are entries in the hash table, which guarantees hash collisions in the table, leading to inflated counts for some n-grams and the potential for false positives in the top-k. That’s where the accuracy-memory tradeoff comes in. The final counts reported for the top-k n-grams will always be an upper bound of the true counts.

- E.2 LEAST COMMON n-GRAMS

To collect the (approximate) bottom-k n-grams we also start by initializing a hash table of u3216 zeros to represent occurrence counts for each n-gram, and an empty collection of the bottom-k n-grams. But this time we have to iterate over the corpus’ n-grams twice.

During the first iteration we tally up the counts just like we do in the top-k algorithm, except that we don’t add any n-grams to the bottom-k collection. During the second iteration we now already have the final counts of all n-grams, so we simply look up the count of each n-gram encountered and then add it to the bottom-k collection if its count is low enough, potentially evicting another n-gram.

Hash collisions might cause false negatives with the bottom-k, i.e. some rare n-grams may be missing from bottom-k if they had hash collisions with more frequent n-grams. The final counts reported will for the bottom-k n-grams always be a lower bound of the true counts.

- E.3 UNIQUE n-GRAMS

To estimate the number of unique n-grams we initialize a hash table of booleans set to ‘false’. Then we iterate over all n-grams in the corpus and for each n-gram encountered we take its hash and update the corresponding boolean in the table to ‘true’. After iterating over the whole corpus we simply have to tally up the number of ‘true’ entries. This number is the estimate for the number of unique n-grams, which will always be a lower bound of the actual number of unique n-grams.

16It’s not necessary to use u64 integers when collecting the bottom-k even if there’s a possibility of overflow counts, provided overflows are caught and kept at 232, since we only care about the exact count of rare n-grams which are unlikely to ever reach an overflow.

