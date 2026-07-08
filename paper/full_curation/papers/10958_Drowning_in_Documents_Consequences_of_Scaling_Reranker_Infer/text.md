# Drowning in Documents: Consequences of Scaling Reranker Inference

Mathew Jacob1,2,†, Erik Lindgren1, Matei Zaharia1, Michael Carbin1, Omar Khattab1 and Andrew Drozdov1,*

1Databricks, San Francisco, USA 2University of Illinois at Urbana-Champaign, Urbana, USA

#### Abstract

Rerankers, typically cross-encoders, are computationally intensive but are frequently used because they are widely assumed to outperform cheaper initial IR systems. We challenge this assumption by measuring reranker performance for full retrieval, not just re-scoring first-stage retrieval. To provide a more robust evaluation, we prioritize strong first-stage retrieval using modern dense embeddings and test rerankers on a variety of carefully chosen, challenging tasks, including internally curated datasets to avoid contamination, and out-of-domain ones. Our empirical results reveal a surprising trend: the best existing rerankers provide initial improvements when scoring progressively more documents, but their effectiveness gradually declines and can even degrade quality beyond a certain limit. We hope that our findings will spur future research to improve reranking.

#### Keywords

Reranking, Scaling Test-time Inference

## 1. Introduction

Contemporary information retrieval (IR) models are often either retrievers [1, 2, 3, inter alia] which pre-compute document representations for efficient retrieval or rerankers [4, 5, 6, inter alia] that jointly encode query–document pairs, often with cross-encoder architectures. These two paradigms pose a popular tradeoff between quality and cost: retrievers are orders of magnitude cheaper as they pre-index document representations, and while rerankers are more expensive, they are widely understood in the literature to boost quality and generalization due to their expressive modeling capabilities [7].

Modern IR systems often manage this tradeoff using multi-stage reranking pipelines [8, 9, 10], where a fast retriever identifies the initial top-𝐾 candidate documents and a reranker then re-scores only those 𝐾. If the models used are well-chosen—e.g. best-in-class retrievers and rerankers—it is generally assumed both (1) that introducing rerankers will consistently improve overall quality and (2) that increasing the number of reranked documents 𝐾 will progressively lead to even higher gains [11, 12, inter alia]. As a corollary, using rerankers to score the entire document set should also be an effective—albeit potentially unrealistic in terms of cost—approach for achieving high recall.

We test these seemingly innocuous assumptions using best-in-class rerankers across several underexplored public and enterprise IR benchmarks. Notably, our choice of datasets and models departs from conventional reranker research, which typically focuses on BM25 for first-stage retrieval and MSMarco for evaluation. We find that these assumptions are frequently false. Specifically, while rerankers initially help with small values for 𝐾, reranking with large 𝐾 decreases recall precipitously (Figure 1), often dropping beneath the quality of standalone retrievers. As a consequence, modern rerankers frequently perform worse than retrievers when both rank the full dataset. Upon studying this, we find that rerankers often score completely irrelevant documents, lacking semantic or lexical similarity to the query, very highly. These same documents, which we call phantom hits, are often given very low scores by the

ReNeuIR 2025 (at SIGIR 2025) – 4th Workshop on Reaching Efficiency in Neural Information Retrieval, July 17, 2025, Padua, Italy

*Corresponding author. †Work done during Mathew’s internship at Databricks.

andrew.drozdov@databricks.com (A. Drozdov)

© 2025 Copyright for this paper by its authors. Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0).

Reranking

Retriever

Full Retrieval

Retriever

Recall@10

Recall@K

Typical Error: Search sometimes has an accidental hit matching on an ambiguous term, fail to match a synonym, etc.

Reranker (Expected)

Reranker (Expected)

Reranker (Actual)

Reranker (Actual)

Phantom Hit: When search ﬁnds a document with no apparent connection to the query. Observed in neural rerankers in this paper.

10 Typical K Large K

10 Typical K Large K

Number of Documents Retrieved (K)

Number of Documents Retrieved (K)

Figure 1: Left: While rerankers are thought to outperform retrievers, we find instances where scaling the number of documents 𝐾 used in reranking often leads to a substantial decrease in recall. Right: Our definition of phantom hit, which is an unexpected error we observed in various neural rerankers.

retrievers, revealing a counterintuitive way in which retriever models can outperform reranker models in practice.

Our main findings are thus that (1) we empirically demonstrate that scaling K leads to a substantial decrease in recall in multiple settings, (2) we qualitatively reveal and investigate a pathology in reranking that we describe as Phantom Hits, where the reranker assigns a high score to documents that have no lexical or semantic overlap with the query, and (3) we demonstrate a proof-of-existence that more robust reranking options exist and can be implemented with large language models (LLMs). Overall, we establish that the current understanding of rerankers does not match their behavior in practice and call for more research to improve their robustness to noise.

## 2. Background and Related Work

Retrievers A first-stage retriever takes a query and searches the entire corpus to find relevant documents. Retrievers usually embed the documents offline and compute a cheap similarity score between the embedded query and documents to find the most relevant documents. For sparse vector methods, BM25 [13] has consistently proven to be a strong baseline. BM25 is a bag-of-words method based on lexical matching that uses an inverted index for fast search.

For dense retrieval, Transformers are used to encode the query and document separately [1, 2, 3]. This degree of independence enables dense embeddings to be computed for each document offline. At search time, only the query needs to be encoded, and then vector search algorithms are used to quickly find the most relevant documents [14, 15, 16].

Rerankers A cross-encoder [4, 5, 6] is a model that, given a query and document, outputs a relevance score for the pair. The cross-encoder attends to the query-document pairs jointly. The highly-expressive modeling of cross-encoders is widely understood in the literature [11, 7] to lead to much better accuracy and generalization.

Scaling Compute A growing body of work investigates the effects of scaling compute on different components of IR and RAG systems. Fang et al. [17] investigated the scaling laws of dense retrieval models, measuring how factors like model and data size affect performance of neural retrievers. Leng et al. [18] and Yue et al. [19] explore improving long-context RAG performance by scaling inference compute. There is also a broader body of work on scaling compute for LLM training [20] and inference [21, 22, 23].

## 3. Experimental Setup

We test (1) how first-stage retrievers and rerankers interact across different model pairings and reranking depth 𝐾 and (2) to compare retrievers against rerankers for full dataset retrieval.

### 3.1. Retrievers

We consider various retrievers with different cost and quality tradeoffs. For simplicity and reproducibility we use exact scoring for all retrievers. Refer to Table 2 in the Appendix for more information on retrievers.

Lexical Search BM25 is a lexical search approach that sparsely represents each document according to their token counts. We use the implementation from Pyserini [24] with the default parameters. BM25 is fast at scale when using inverted indices, although can not represent semantic similarity between text.

Dense Embeddings We use two proprietary embedding models, namely, voyage-2, a 1024dimensional dense embedding model from Voyage AI, and text-embedding-3-large, a 3072dimensional dense embedding model from OpenAI. The latter is the highest quality but most costly retrieval model that we consider.

### 3.2. Rerankers

We study several state-of-the-art open and closed rerankers. Refer to Table 2 in the Appendix for more information on rerankers.

Open Models We include two high quality open source cross-encoder models in our experiments: jina-reranker-v2-base-multilingual and bge-reranker-v2-m3. Both rerankers are published on huggingface. We access the Jina model through its API and run BGE locally.

Closed Models We include rerankers from Cohere and from Voyage AI: rerank-english-v3.0 and voyage-rerank-1. While these models have been described as cross-encoders in various blog posts, we cannot verify the precise architecture of closed models. We access the rerankers through their respective APIs.

### 3.3. Datasets

For our evaluation, we use a combination of eight academic and enterprise datasets to ensure adequate coverage of realistic retrieval workloads.

Academic Datasets Our experiments include evaluations across five diverse academic datasets. We include the biology and pony splits from BRIGHT [25] that require reasoning beyond lexical and semantic matching, the relic and doris-mae splits from BIRCO [26] that have been pathologically filtered down to the most challenging queries, and also scifact from BEIR [27], one of the most well established leaderboards for dense embeddings.

Enterprise Datasets To capture workloads that are representative of industry use cases, we benchmark rerankers across three internally curated datasets. FinanceBench is a RAG dataset derived from Islam et al. [28], and includes company-specific questions answerable from SEC filings. ManufacturingQA is derived from an internal knowledge base over technical documentation about manufacturing. This dataset is representative of real domain-specific customer queries and may contain alphanumeric product codes. DocumentationQA is an internal dataset that is comprised of users’ code-related questions with manually labeled answers grounded in documentation from an open-source software framework.

Additional Details We make two simplifications when preprocessing datasets. First, we truncate queries and documents to 512 tokens so that embeddings and rerankers are on a similar playing field, since many research papers have pointed out the challenges for long context retrieval with embeddings [29]. Second, we downsample negative documents to a maximum of 𝑁 = 10,000 documents—this reduces costs of experiments, and increasing N would only show a more profound version of existing trends. Additionally, for the enterprise datasets we were unable to evaluate against jina-reranker-v2-base-multilingual due to legal constraints. We provide a brief guide to ease reproducibility in Appendix A.

### 3.4. Why these datasets and models?

If we focused exclusively on weak retrieval, then we would have a false impression about reranker performance. For example, even though BM25 is well known to have impressive generalization capabilities, recent dense embeddings can greatly outperform BM25 across multiple benchmarks.

If we focused exclusively on the most popular benchmarks for reranker evaluation, such as MSMARCO and BEIR, then similarly we would have a false impression about reranker performance. Rerankers have been directly tuned for these evaluations either directly through including relevant samples in their training, or indirectly through validation. Our diverse selection of datasets not only covers challenging settings that rerankers have not been optimized for, it also is more representative of the real world setting where off-the-shelf models are used on enterprise data. In future work, we hope to investigate even larger corpora that contain tens or hundreds millions of passages such as TREC-RAG [30] (despite its reliance on MSMARCO) and TREC-Biogen [31].

## 4. Results & Analysis: Scaling Up Reranking

How well do modern rerankers perform when given different amounts of documents to rerank? We measure quality when reranking the top-𝑘 documents from different retrievers. We vary 𝑘 to sizes much larger than previous evaluations (larger than 5000)1 to better understand how rerankers behave in extreme settings. Figures 2 and 3 report Recall@10 for the rerankers, averaging across enterprise and academic datasets respectively. In the Appendix B, we include finer-grained results on the individual datasets.

1For cross-encoder evaluations, reranking the top-100 [5] or top-1000 [32, 33] documents is typical.

###### Average Recall@10 for Reranking on Academic Datasets

0.50

0.45

Recall@10

0.40

0.35

0.30

text-embedding-3-lg

voyage-2

bm25

0.25

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

jina-reranker-v2-base-multilingual

bge-reranker-v2-m3

voyage-rerank-1

listwise (gpt-4o-mini)

Figure 2: Recall@10 when reranking, averaged across academic datasets. The dashed line shows the first stage recall, and the solid line is the rerankers’ recall. The rerankers’ recall often degrades as the reranked 𝐾 increases.

###### Average Recall@10 for Reranking on Enterprise Datasets

0.9

0.8

Recall@10

0.7

0.6

text-embedding-3-lg

bm25

|voyage-2|
|---|

0.5

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

listwise (gpt-4o-mini)

bge-reranker-v2-m3

voyage-rerank-1

Figure 3: Recall@10 when reranking, averaged across enterprise datasets. The dashed line shows the first stage recall, and the solid line is the rerankers’ recall. The rerankers’ recall often degrades as the reranked 𝐾 increases.

### 4.1. Does scaling the number of documents for reranking help?

In the majority of cases, we see that reranking a small set of documents is an effective way to improve recall on a dataset (Figures 2 and 3). This claim is reinforced by the statistics from individual experiments,2 which show that rerankers improved Recall@10 over retrieval for at least one value of 𝐾 in 85.0% and 88.9% of the time across academic and enterprise datasets, respectively. The full results for individual experiments are included in Appendix B.

A closer look at the results reveals that even though rerankers are effective when reranking a few documents, that rerankers are much less consistently helpful when scaling the number of documents for reranking. To make this more clear, in Table 1 we include a fine-grained analysis of reranker average Recall@10 at different values of K by filtering for different cases of interest. Our initial filter (Helps) is to look for experiments where there exists at least one K such that reranker Recall@10 is greater than the retriever’s. We investigate four additional filters (Helps + X):

- • Never Hurts: There is no K such that reranker Recall@10 is less than the retriever’s.
- • Distracted: The reranker improves over the retriever at K, but there is K’ > K where the reranker’s Recall@10 for K’ is less than its Recall@10 for K.
- • Scaling is Best: The reranker’s maximum Recall@10 is at the maximum K.
- • Scaling Hurts: The reranker’s Recall@10 at the maximum K is worse than the retriever’s.

One of our key observations is that continuing to include more documents for the reranker is an ineffective way to scale computation. In very few cases did using the largest value of 𝐾 lead to to the best average Recall@10 (see “Helps & Scaling is Best” in Table 1). Furthermore, 53.3% and 44.4% of the time across academic and enterprise datasets, respectively, we saw that even when reranking a small 𝐾 was effective, that scaling 𝐾 lead to Recall@10 that was worse than retrieval alone (see “Helps & Scaling Hurts” in Table 1).

On the enterprise data, reranking the documents from BM25 always lead to an improvement, plus, reranking consistently seems to improve as the number of documents increases. BM25 plus reranking might seem promising, but the reality is that this can be outperformed by retrieval-only with strong dense embeddings. For example, we see that text-embedding-3-large is roughly 1.5x as effective as BM25 on the enterprise data, measured by average Recall@10. Retrieving a few documents

2An experiment is uniquely identified by a (dataset, retriever, reranker) tuple.

Academic Enterprise

SCI DM REL BIO PONY All FIN MQA DQA All # of Experiments 12 12 12 12 12 60 9 9 9 27 Helps 83.3 91.7 100.0 66.7 83.3 85.0 100.0 100.0 66.7 88.9 Helps & Never Hurts 25.0 8.3 75.0 0.0 8.3 23.3 11.1 22.2 33.3 22.2 Helps & Distracted 50.0 83.3 100.0 16.7 50.0 60.0 88.9 66.7 44.4 66.7 Helps & Scaling is Best 8.3 8.3 0.0 0.0 0.0 3.3 11.1 22.2 11.1 14.8 Helps & Scaling Hurts 41.7 66.7 16.7 66.7 75.0 53.3 44.4 55.6 33.3 44.4

- Table 1 Experiment-level statistics for Recall@10 when reranking. We filter for series using the base condition (Helps) and the four additional filters described in §4.1. Each cell shows the percentage (%) of experiments for which the conditions are met.

with a dense embedding would be better than reranking many documents with BM25.3 The stark contrast between BM25 and the dense embeddings is another example of the challenges for retrieval systems to generalize, and the necessity of many varied evaluation suites.

#### 4.1.1. How to fairly compare retrieval and reranking

Given that rerankers often show improved retrieval quality when reranking a few documents, one may assume that rerankers are more accurate than retrievers. Additionally, it’s rarely clear how effective a reranker really is, since a reranker is usually evaluated jointly with first stage retrieval. With this in mind, we develop a fair and simple protocol for comparing retrievers and rerankers: Rather than apply the reranker only on the top documents provided by the retriever, apply the reranker on all of the documents that are being retrieved from.

We use this protocol with our retrievers and rerankers of interest and present the results in Figure 4, where we plot Recall@K across all values of K. In line with our previous findings, we see that retrievers are better than rerankers in many settings.

###### Fair Comparison between Retrievers and Rerankers (Average Recall@K)

1.0

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | |Enterprise| |
| | | | | |
| | | | | |
| | | | | |

0.9

0.8

Recall@K

0.8

0.6

0.7

0.4

0.6

Academic

0.5

10 100 1000 5000

10 100 1000 5000

Number of Documents (K)

Number of Documents (K)

bm25-pyserini

bge-reranker-v2-m3

voyage-rerank-1

text-embedding-3-large

cohere-english

jina-reranker-v2-base-multilingual

voyage-2

- Figure 4: Recall@K for full retrieval, averaged across academic and enterprise datasets. The maximum recall can be less than 1 since we only display recall up to 5000 documents, and some datasets contain additional documents. Note the difference in y-axis.

3Our focus on strong retrievers and challenging datasets is in contrast to previous work that relies heavily on BM25 and MS MARCO [34].

### 4.2. Reranker Errors and Phantom Hits

In Table 1, we see that while reranking with a small number of documents can be beneficial, reranking with a large K often leads to worse performance: Recall@10 is lower than that of retrieval alone in 53.3% and 44.4% of experiments on academic and enterprise datasets, respectively. In a larger percentage of cases we find helpful rerankers will get distracted and Recall@10 will dip at higher K, though not always falling below the retriever. It is plausible that this decrease in recall could be due to missing labels, and rerankers finding previously unknown but relevant documents. However, when manually looking through the reranker’s predictions, we see that this is not the case. Instead, we attribute many reranker errors to what we call “phantom hits”. These are cases where the reranker prefers an irrelevant negative document to the positive ground truth when there is no apparent connection between the query and irrelevant document, lexical, semantic, or otherwise. In this error analysis, we seek a few representative examples of these “phantom hits”.

To mine potentially interesting examples, we filtered for queries where the Recall@10 decreased when reranking K=5000 instead of K=100 documents. We show examples of these reranker failures in Figures 5 and 6. In all the selected cases, the retriever (text-embedding-3-large) assigns a higher rank to the shown positive document, sometimes by a large margin. However, we see multiple cases where the reranker prefers irrelevant phantom hits over positive documents even when they have little or no text overlap with the query. We found this behavior is not isolated to a single reranker, and that seemingly random documents about “dishwashing” and “exercise” were preferred when there were clearly more topically similar options. More detailed rankings of each query are shown in Appendix C.

text-embedding-3-large + voyage-rerank-1

Query Many proteins in human cells can be post-translationally modified at lysine residues via acetylation.

Positive Document [Retriever: 1, Reranker: 39] Protein Lysine Acetylated/Deacetylated Enzymes and the Metabolism-Related Diseases Lysine acetylation is a reversible posttranslational ...

##### Negative Document [Retriever: 3337, Reranker: 5]

On the origins of ultra-fine anaphase bridges. Comment on: Chan KL, Palmai-Pallag T, Ying S, Hickson ID. Replication stress induces sister-chromatid bridging at fragile site loci in mitosis. Nat Cell Biol '09; 11:753-60.

text-embedding-3-large + bge-reranker-v2-m3

Query The binding orientation of the ML-SA1 activator at hTRPML2 is different from the binding orientation of the ML-SA1 activator at hTRPML1.

Positive Document [Retriever: 0, Reranker: 34] Human TRPML1 channel structures in open and closed conformations ... suggesting a distinct agonistbinding site from that found in TRPV1 ...

##### Negative Document [Retriever: 3645, Reranker: 2]

High-intensity functional exercise program and protein-enriched energy supplement for older persons dependent in activities of daily living: a randomised controlled trial. The aims of this randomised controlled trial were to determine if a high-intensity functional exercise program improves balance, gait ability, and lower-limb strength in older persons ...

- Figure 5: Two unexpected reranker error from Scifact. The ranks assigned by the retriever and reranker shown in parens (zero-indexed). Related words are highlighted.

text-embedding-3-large + cohere-rerank-english-v3.0

Query Less than 10% of the gabonese children with Schimmelpenning-Feuerstein-Mims syndrome (SFM) had a plasma lactate of more than 5mmol/L.

##### Positive Document [Retriever: 5, Reranker: 33]

Assessment of Volume Depletion in Children with Malaria Background ... To assist management of severely ill children, and to test the hypothesis that volume changes in fluid compartments reflect disease severity, we measured body compartment volumes in Gabonese children with malaria. ... Volumes in different compartments (TBW, ECW, and ICW) were not related to hyperlactataemia ...

##### Negative Document [Retriever: 4615, Reranker: 3]

The 5TMM series: a useful in vivo mouse model of human multiple myeloma. The present invention provides a combination sink and dishwashing apparatus having a sink sharing a common side wall with a cabinet which defines a closed space. The cabinet has a wire basket for holding and washing a plurality of dishes within the cabinet. ...

- Figure 6: An unexpected reranker errors from Scifact. The ranks assigned by the retriever and reranker shown in parens (zero-indexed). Related words are highlighted.

### 4.3. Listwise-Reranking with Large Language Models

Our results in the previous subsections show that scaling 𝐾 can eventually hurt reranker recall and eventually yield worse performance than retrieval. It remains unclear whether this is an issue with reranking in general, or whether certain rerankers are more robust to the failure mode of scaling K. We additionally explore listwise reranking using LLMs in order to see if there is at least one class of model where reranking is more robust. Compared to cross-encoders, this new approach is advantaged primarily for two reasons: the model uses more context when reranking and leverages a strong LLM backbone, gpt-4o-mini.

Implementation We follow the sliding window approach introduced in Sun et al. [35] using a window size of 20, stride size of 10, and gpt-4o-mini as the LLM.4,5 When our program fails to parse the output of the LLM due to syntax or logical errors (e.g. a repeated document ID), then we simply keep the retrieval ordering for that window.6 When inspecting the outputs from the model on failed windows, we observe that almost all of the errors occurred when the model thought that none of the documents being ranked were relevant to the query, and thus either didn’t follow the format specified in the prompt or refused to output a ranking altogether.

Experiment Running the pointwise cross-encoder on the full data is relatively inexpensive compared with the listwise LLM approach. The cross-encoder scores can be reused to compute Recall@10 for all values of 𝐾. In contrast, since the listwise approach begins with the lowest scoring documents from retrieval, this means that there is no opportunity to reuse computation between different values of 𝐾. This is exacerbated by the fact that gpt-4o-mini has a substantially higher cost per token than other rerankers.7 For these reasons, we only run the listwise approach with one of the retrievers and only across four settings of 𝐾 ∈ {50,100,500,1000}.

4Anecdotally, we found Llama-3.1-70B yielded similar results as gpt-4o-mini, although a full exploration of LLM performance is outside the scope of this study. 5We use sliding window with zero-shot prompting. Further performance improvements, potentially at lower inference cost,

can be can be achieved through finetuning Pradeep et al. [36] as well as alternative prompting algorithms [37, 38]. 6Errors like this occur about 5% of the time when 𝐾 = 100, aligning with previous reports [36]. 7Cost per 1M tokens (Feb-18-2024): voyage-2 is $0.05 and gpt-4o-mini is $0.15.

Results We report results for listwise reranking with LLMs in Figures 2 and 3 (additional results shown in the Appendix Figures 7, 8, and 9). Not only is Recall@10 substantially higher than competing rerankers, but consistently improves as we scale 𝐾. In contrast, we observed that pointwise crossencoders were robust in only 23.3% and 22.2% of academic and enterprise experiments, respectively (see “Helps & Never Hurts” in Table 1). This is an encouraging proof of existence that there are available rerankers with favorable properties in respect to scaling 𝐾.

## 5. Discussion

Our research has focused on empirically analyzing how effectively we can improve reranking by scaling up the number of included documents. We primarily focus on off-the-shelf pointwise cross-encoder rerankers and show that in many cases scaling 𝐾 hurt performance.8 In this section, in order to aid future research, we highlight three potential causes for unexpected failures in rerankers and how they might be addressed.

Negatives in Training We conjecture that today’s point-wise rerankers lack of robustness is partially due to exposure bias during reranking training. Namely, pointwise rerankers may see less negatives in training than their embedding model counterparts because large batch reranker training is computationally expensive and does not easily share negatives across the batch. Thus, we hypothesize that rerankers are trained on negatives selected from a subset of the corpus filtered by retrievers, which may explain the peak performance of rerankers occuring at a lower number of retrieved documents, where the documents the reranker scores more closely resemble its training data. Thus, while it is theoretically intuitive for rerankers to be considered strictly better models than embedding models, the computational expense of fully training a reranker is prohibitive enough to prevent this from being the case today.

For example, when training the mGTE embedding and reranking model in [39], the authors use 16,384 randomly selected in-batch negatives to train their embedding model, while only using 4 randomly selected negatives to train their cross-attention based reranker model. Additionally, some approaches to training embedding models are able to scale to millions of negative documents [40, 41]. Under-trained rerankers may explain why previous research has shown embeddings can match cross-encoder quality [42].

Reranking as Ensembling It’s confusing that rerankers can help when we’ve shown that they are overall worse than retrievers at tasks like full retrieval (§4.1.1). It’s similarly counter-intuitive that they can be worse than retrievers given that a cross-encoder and dense embedding are essentially the same model architecture, but trained with different input data. Perhaps these observations make more sense if we view reranking as a type of ensembling. Diversity is a useful property in ensembles [43], meanwhile, rerankers are overly specialized compared with embeddings due to the aforementioned exposure bias. From this view, it’s plausible that the reranker is providing alternative features (compared to embeddings) that are helpful to discriminate relevant from irrelevant documents only from the top of the list. The ensemble perspective is even more obvious when taking BM25 into account, which is case where the first stage and second stage are modeled completely differently, yet reranking remains helpful.

The ensembling perspective could be fruitful in future research. Scaling laws for dense retrieval explain how retrieval quality may improve with an increase in model parameters and training data [44], although they do not factor in reranking. Grefenstette et al. [45] shows that ensembling may be more productive than training a single large model under a fixed compute budget, and the same may apply to reranking.

8The one reranking alternative that we explore is listwise reranking with LLMs (§4.3). The listwise approach is encouraging and serves as a proof of existence that there are reranking architectures where scaling 𝐾 does help. Even so, this listwise approach is high latency and impractical, and is not our focus.

Robustness of Deep Learning There have been many studies that suggest deep learning models aren’t robust. Representative instances of this problem include vision models failing to classify images due to small perturbations [46], as well as text processing models changing their prediction when substituting words in their input with synonyms [47]. In the context of reranking, as one scales the number of documents, each additional document included creates a risk that the model may assign an inappropriately high score to an irrelevant document [48]. Similar observations have been made in other AI paradigms. For example, Best-of-N decoding can suffer with large N due to reward hacking, which leads to finding a more favorable proxy reward that doesn’t necessarily improve results on the downstream task [49, 50, 51, 52, 53, 54, 55].

Pointwise vs. Listwise Ranking The lack of robustness in rerankers is exacerbated when they assign scores independently to each document (pointwise). Using models that are trained with a listwise loss or listwise inference procedure has been shown to improve ranking quality [56, 57, 58, 59]. This is one explanation why listwise-reranking with large language models outperformed pointwise-reranking with cross-encoders in §4.3.

Effective Finetuning The research community has highlighted challenges associated with finetuning. One explanation for why our listwise-reranker may outperform the cross-encoder rerankers is that the listwise-reranker is based on gpt-4o-mini without any finetuning. The rerankers, on the otherhand, have almost definitely been finetuned for the reranking task, which may have lead to catastrophic forgetting or other modeling issues. Although the shortcomings of current finetuning recipes are well established [60], finetuning for ranking presents even further challenges since this procedure is used to convert models from text generation to embedding or classification [61, 62].

Favorable Conditions for Rerankers In our study, we have specifically chosen strong first stage retrieval methods (i.e. dense embeddings), as well as a diverse collection of reasonable datasets that are representative of real world workloads. Prior studies are often more limited to settings that are particularly well suited for reranking: using BM25 as the first stage retrieval and evaluated against MSMarco or similar datasets [34, 63, 64, 65, 66].

We include at least one result where reranking is consistently helpful—the BM25 portion of Figure 3 shows clear benefits from reranking on the Enterprise data split. Even so, we also find that dense embeddings alone are on par or better than sparse embeddings in the same data setting. The different trends observed when reranking atop sparse or dense embeddings in the first stage highlights the importance of exploring many scenarios when conducting reranker research. In practice, there are more free variables than simply the combination of dataset, retriever, and reranker. For instance, Bahri et al. [67] demonstrate the benefits of adaptive ranked list truncation—although this filtering would typically be applied prior to reranking, it may be beneficial to apply afterwards as well.

## 6. Conclusion

We empirically study how, in modern IR systems, scaling the inference compute of different rerankers impacts the quality of the retrieved output. We do this by testing modern OSS and closed source embedding models and rerankers on carefully curated academic and enterprise datasets. We find that, for modern cross-encoders, scaling the inference compute by reranking more documents ultimately leads to significant performance degredation on Recall. Furthermore, when ranking the entire corpus, we find that modern embedding models outperform cross-encoders. As a path forward, we present evidence that using LLMs for listwise reranking can outperform cross-encoders. This approach could serve as a potential teacher to improve cross-encoders or serve directly as a reranker when the costquality tradeoff is desirable. We hope our findings and analyses will be useful resources to practitioners as they deploy reranker pipelines, as well as spur future research to improve cross-encoder rerankers.

## Limitations

In our experiments, several of the models used are closed source, where we do not have access to the information like the training data, precise model architecture, and model size. As mentioned in §5, experimenting with different training strategies, training data distributions, and model sizes may confirm some of our hypotheses, as well as lead to new insights regarding rerankers.

## References

- [1] N. Reimers, I. Gurevych, Sentence-BERT: Sentence embeddings using Siamese BERT-networks, in: K. Inui, J. Jiang, V. Ng, X. Wan (Eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Association for Computational Linguistics, Hong Kong, China, 2019, pp. 3982–3992. URL: https://aclanthology.org/D19-1410. doi:10.18653/v1/D19-1410.
- [2] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, W.-t. Yih, Dense passage retrieval for open-domain question answering, in: B. Webber, T. Cohn, Y. He, Y. Liu (Eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Association for Computational Linguistics, Online, 2020, pp. 6769–6781. URL: https://aclanthology.org/2020.emnlp-main.550. doi:10.18653/v1/2020.emnlp-main.550.
- [3] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, E. Grave, Unsupervised dense information retrieval with contrastive learning, Transactions on Machine Learning Research

(2022). URL: https://openreview.net/forum?id=jKN1pXi7b0.

- [4] R. Nogueira, K. Cho, Passage re-ranking with bert, ArXiv abs/1901.04085 (2019). URL: https: //api.semanticscholar.org/CorpusID:58004692.
- [5] L. Gao, Z. Dai, J. Callan, Rethink training of bert rerankers in multi-stage retrieval pipeline, in: Advances in Information Retrieval: 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28 – April 1, 2021, Proceedings, Part II, Springer-Verlag, Berlin, Heidelberg, 2021, p. 280–286. URL: https://doi.org/10.1007/978-3-030-72240-1_26. doi:10.1007/978-3-030-72240-1_ 26.
- [6] M. Glass, G. Rossiello, M. F. M. Chowdhury, A. Naik, P. Cai, A. Gliozzo, Re2G: Retrieve, rerank, generate, in: M. Carpuat, M.-C. de Marneffe, I. V. Meza Ruiz (Eds.), Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Association for Computational Linguistics, Seattle, United States, 2022, pp. 2701–2715. URL: https://aclanthology.org/2022.naacl-main.194. doi:10.18653/v1/2022. naacl-main.194.
- [7] G. M. Rosa, L. H. Bonifacio, V. Jeronymo, H. Abonizio, M. Fadaee, R. de Alencar Lotufo, R. Nogueira, In defense of cross-encoders for zero-shot retrieval, ArXiv abs/2212.06121 (2022). URL: https: //api.semanticscholar.org/CorpusID:254564419.
- [8] I. Matveeva, C. Burges, T. Burkard, A. Laucius, L. Wong, High accuracy retrieval with multiple nested ranker, in: Proceedings of the29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’06, Association for Computing Machinery, New York, NY, USA, 2006, p. 437–444. URL: https://doi.org/10.1145/1148170.1148246. doi:10.1145/ 1148170.1148246.
- [9] L. Wang, J. Lin, D. Metzler, A cascade ranking model for efficient ranked retrieval, in: Proceedings of the 34th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’11, Association for Computing Machinery, New York, NY, USA, 2011, p. 105–114. URL: https://doi.org/10.1145/2009916.2009934. doi:10.1145/2009916.2009934.
- [10] R. Nogueira, W. Yang, K. Cho, J. Lin, Multi-stage document ranking with bert, arXiv preprint arXiv:1910.14424 (2019).
- [11] S. Humeau, K. Shuster, M.-A. Lachaux, J. Weston, Poly-encoders: Architectures and pre-training

- strategies for fast and accurate multi-sentence scoring, in: International Conference on Learning Representations, 2020. URL: https://openreview.net/forum?id=SkxgnnNFvH.
- [12] Y. Luan, J. Eisenstein, K. Toutanova, M. Collins, Sparse, dense, and attentional representations for text retrieval, Transactions of the Association for Computational Linguistics 9 (2021) 329–345. URL: https://aclanthology.org/2021.tacl-1.20. doi:10.1162/tacl_a_00369.
- [13] S. Robertson, S. Jones, Relevance weighting of search terms, Journal of the American Society for Information science 27 (1976) 129–146. doi:10.1002/asi.4630270302.
- [14] J. Johnson, M. Douze, H. Jégou, Billion-scale similarity search with gpus, IEEE Transactions on Big Data 7 (2019) 535–547.
- [15] Y. A. Malkov, D. A. Yashunin, Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs, IEEE transactions on pattern analysis and machine intelligence 42 (2018) 824–836.
- [16] R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, S. Kumar, Accelerating large-scale inference with anisotropic vector quantization, in: International Conference on Machine Learning,

2020. URL: https://arxiv.org/abs/1908.10396.

- [17] Y. Fang, J. Zhan, Q. Ai, J. Mao, W. Su, J. Chen, Y. Liu, Scaling laws for dense retrieval, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 1339–1349. URL: https://doi.org/10.1145/3626772.3657743. doi:10.1145/3626772.3657743.
- [18] Q. Leng, J. Portes, S. Havens, M. Zaharia, M. Carbin, Long context rag performance of large language models, 2024. doi:10.48550/arXiv.2411.03538.
- [19] Z. Yue, H. Zhuang, A. Bai, K. Hui, R. Jagerman, H. Zeng, Z. Qin, D. Wang, X. Wang, M. Bendersky, Inference scaling for long-context retrieval augmented generation, arXiv preprint arXiv:2007.00808

(2024). doi:10.48550/arXiv.2410.04343.

- [20] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, D. Amodei, Scaling laws for neural language models, ArXiv abs/2001.08361 (2020). URL: https://api.semanticscholar.org/CorpusID:210861095.
- [21] B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. Le, C. Ré, A. Mirhoseini, Large language monkeys: Scaling inference compute with repeated sampling, 2024. doi:10.48550/arXiv.2407.21787.
- [22] OpenAI, Learning to Reason with LLMs, LearningtoReasonwithLLMs, 2024. [Accessed 30-10-2024].
- [23] Z. Liang, Y. Liu, T. Niu, X. Zhang, Y. Zhou, S. Yavuz, Improving llm reasoning through scaling inference computation with collaborative verification, 2024. doi:10.48550/arXiv.2410.05318.
- [24] J. Lin, X. Ma, S.-C. Lin, J.-H. Yang, R. Pradeep, R. Nogueira, Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations, in: Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021), 2021, pp. 2356–2362.
- [25] H. Su, H. Yen, M. Xia, W. Shi, N. Muennighoff, H.-y. Wang, H. Liu, Q. Shi, Z. S. Siegel, M. Tang, R. Sun, J. Yoon, S. O. Arik, D. Chen, T. Yu, Bright: A realistic and challenging benchmark for reasoning-intensive retrieval, 2024. URL: https://arxiv.org/abs/2407.12883.
- [26] X. Wang, J. Wang, W. Cao, K. Wang, R. Paturi, L. Bergen, Birco: A benchmark of information retrieval tasks with complex objectives, ArXiv abs/2402.14151 (2024). URL: https: //api.semanticscholar.org/CorpusID:267782730.
- [27] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych, BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models, in: Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL: https: //openreview.net/forum?id=wCu6T5xFjeJ.
- [28] P. Islam, A. Kannappan, D. Kiela, R. Qian, N. Scherrer, B. Vidgen, Financebench: A new benchmark for financial question answering, 2023. arXiv:2311.11944.
- [29] D. Zhu, L. Wang, N. Yang, Y. Song, W. Wu, F. Wei, S. Li, LongEmbed: Extending embedding models for long context retrieval, in: Y. Al-Onaizan, M. Bansal, Y.-N. Chen (Eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Miami, Florida, USA, 2024, pp. 802–816. URL: https://aclanthology.org/

- 2024.emnlp-main.47.
- [30] R. Pradeep, N. Thakur, S. Sharifymoghaddam, E. Zhang, R. Nguyen, D. Campos, N. Craswell, J. Lin, Ragnarök: A reusable rag framework and baselines for trec 2024 retrieval-augmented generation track, ArXiv abs/2406.16828 (2024). URL: https://api.semanticscholar.org/CorpusID:270702738.
- [31] D. Gupta, D. Demner-Fushman, W. R. Hersh, S. Bedrick, K. Roberts, Overview of trec 2024 biomedical generative retrieval (biogen) track, ArXiv abs/2411.18069 (2024). URL: https://api. semanticscholar.org/CorpusID:274306244.
- [32] H. Zhuang, Z. Qin, R. Jagerman, K. Hui, J. Ma, J. Lu, J. Ni, X. Wang, M. Bendersky, Rankt5: Fine-tuning t5 for text ranking with ranking losses, Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (2022). URL: https: //api.semanticscholar.org/CorpusID:252993059.
- [33] M. Li, H. Zhuang, K. Hui, Z. Qin, J. Lin, R. Jagerman, X. Wang, M. Bendersky, Can query expansion improve generalization of strong cross-encoder rankers?, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 2321–2326. URL: https://doi.org/10.1145/3626772.3657979. doi:10.1145/3626772.3657979.
- [34] S. Hofstätter, N. Rekabsaz, C. Eickhoff, A. Hanbury, On the effect of low-frequency terms on neural-ir models, in: Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR’19, Association for Computing Machinery, New York, NY, USA, 2019, p. 1137–1140. URL: https://doi.org/10.1145/3331184.3331344. doi:10.1145/ 3331184.3331344.
- [35] W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, Z. Ren, Is ChatGPT good at search? investigating large language models as re-ranking agents, in: H. Bouamor, J. Pino, K. Bali (Eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Singapore, 2023, pp. 14918–14937. URL: https://aclanthology.org/2023.emnlp-main.923. doi:10.18653/v1/2023.emnlp-main.923.
- [36] R. Pradeep, S. Sharifymoghaddam, J. Lin, RankZephyr: Effective and robust zero-shot listwise reranking is a breeze!, arXiv:2312.02724 (2023).
- [37] C.-W. Huang, Y.-N. Chen, Instupr : Instruction-based unsupervised passage reranking with large language models, 2024. arXiv:2403.16435.
- [38] A. Parry, S. MacAvaney, D. Ganguly, Top-down partitioning for efficient list-wise ranking, ArXiv abs/2405.14589 (2024). URL: https://api.semanticscholar.org/CorpusID:269983636.
- [39] X. Zhang, Y. Zhang, D. Long, W. Xie, Z. Dai, J. Tang, H. Lin, B. Yang, P. Xie, F. Huang, et al., mgte: Generalized long-context text representation and reranking models for multilingual text retrieval, arXiv preprint arXiv:2407.19669 (2024).
- [40] L. Xiong, C. Xiong, Y. Li, K.-F. Tang, J. Liu, P. Bennett, J. Ahmed, A. Overwijk, Approximate nearest neighbor negative contrastive learning for dense text retrieval, arXiv preprint arXiv:2007.00808

(2020).

- [41] E. Lindgren, S. Reddi, R. Guo, S. Kumar, Efficient training of retrieval models using negative cache, Advances in Neural Information Processing Systems 34 (2021) 4134–4146.
- [42] A. Menon, S. Jayasumana, A. S. Rawat, S. Kim, S. Reddi, S. Kumar, In defense of dual-encoders for neural ranking, in: K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, S. Sabato (Eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, PMLR, 2022, pp. 15376–15400. URL: https://proceedings.mlr.press/ v162/menon22a.html.
- [43] R. Maclin, D. W. Opitz, Popular ensemble methods: An empirical study, J. Artif. Intell. Res. 11

(1999) 169–198. URL: https://api.semanticscholar.org/CorpusID:2594813.

- [44] Y. Fang, J. Zhan, Q. Ai, J. Mao, W. Su, J. Chen, Y. Liu, Scaling laws for dense retrieval, in: Annual International ACM SIGIR Conference on Research and Development in Information Retrieval,

2024. URL: https://api.semanticscholar.org/CorpusID:268723791.

- [45] E. Grefenstette, R. Stanforth, B. O’Donoghue, J. Uesato, G. Swirszcz, P. Kohli, Strength in numbers: Trading-off robustness and computation via adversarially-trained ensembles, ArXiv abs/1811.09300

- (2018). URL: https://api.semanticscholar.org/CorpusID:53717538.
- [46] H. Hosseini, R. Poovendran, Deep neural networks do not recognize negative images, 2017. doi:10.48550/arXiv.1703.06857.
- [47] R. Jia, A. Raghunathan, K. Göksel, P. Liang, Certified robustness to adversarial word substitutions, in: K. Inui, J. Jiang, V. Ng, X. Wan (Eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Association for Computational Linguistics, Hong Kong, China, 2019, pp. 4129–4142. URL: https://aclanthology.org/D19-1423. doi:10.18653/v1/D19-1423.
- [48] H. Zamani, M. Bendersky, D. Metzler, H. Zhuang, X. Wang, Stochastic retrieval-conditioned reranking, in: Proceedings of the 2022 ACM SIGIR International Conference on Theory of Information Retrieval, ICTIR ’22, Association for Computing Machinery, New York, NY, USA, 2022, p. 81–91. URL: https://doi.org/10.1145/3539813.3545141. doi:10.1145/3539813.3545141.
- [49] N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, P. Christiano, Learning to summarize from human feedback, in: Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Curran Associates Inc., Red Hook, NY, USA, 2020.
- [50] R. Nakano, J. Hilton, S. Balaji, J. Wu, O. Long, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, X. Jiang, K. Cobbe, T. Eloundou, G. Krueger, K. Button, M. Knight, B. Chess, J. Schulman, Webgpt: Browser-assisted question-answering with human feedback, ArXiv abs/2112.09332 (2021). URL: https://api.semanticscholar.org/CorpusID:245329531.
- [51] A. Pan, K. Bhatia, J. Steinhardt, The effects of reward misspecification: Mapping and mitigating misaligned models, ArXiv abs/2201.03544 (2022). URL: https://api.semanticscholar.org/CorpusID: 245837268.
- [52] L. Gao, J. Schulman, J. Hilton, Scaling laws for reward model overoptimization, in: A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, J. Scarlett (Eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, PMLR, 2023, pp. 10835–10866. URL: https://proceedings.mlr.press/v202/gao23h.html.
- [53] N. Lambert, R. Calandra, The alignment ceiling: Objective mismatch in reinforcement learning from human feedback, ArXiv abs/2311.00168 (2023). URL: https://api.semanticscholar.org/CorpusID: 264832734.
- [54] R. Rafailov, Y. Chittepu, R. Park, H. Sikchi, J. Hejna, W. B. Knox, C. Finn, S. Niekum, Scaling laws for reward model overoptimization in direct alignment algorithms, in: The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL: https://openreview.net/forum? id=pf4OuJyn4Q.
- [55] B. Stroebl, S. Kapoor, A. Narayanan, Inference scaling flaws: The limits of llm resampling with imperfect verifiers, 2024. URL: https://api.semanticscholar.org/CorpusID:274281169.
- [56] Q. Ai, K. Bi, J. Guo, W. B. Croft, Learning a deep listwise context model for ranking refinement, in: The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, SIGIR ’18, Association for Computing Machinery, New York, NY, USA, 2018, p. 135–144. URL: https://doi.org/10.1145/3209978.3209985. doi:10.1145/3209978.3209985.
- [57] Q. Ai, X. Wang, S. Bruch, N. Golbandi, M. Bendersky, M. Najork, Learning groupwise multivariate scoring functions using deep neural networks, in: Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval, ICTIR ’19, Association for Computing Machinery, New York, NY, USA, 2019, p. 85–92. URL: https://doi.org/10.1145/3341981.3344218. doi:10.1145/ 3341981.3344218.
- [58] R. Rahimi, A. Montazeralghaem, J. Allan, Listwise neural ranking models, in: Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval, ICTIR ’19, Association for Computing Machinery, New York, NY, USA, 2019, p. 101–104. URL: https: //doi.org/10.1145/3341981.3344245. doi:10.1145/3341981.3344245.
- [59] G. Gao, J. D. Chang, C. Cardie, K. Brantley, T. Joachims, Policy-gradient training of language models for ranking, ArXiv abs/2310.04407 (2023). URL: https://api.semanticscholar.org/CorpusID: 263829952.

- [60] D. Biderman, J. Ortiz, J. Portes, M. Paul, P. Greengard, C. Jennings, D. King, S. Havens, V. Chiley, J. Frankle, C. Blakeney, J. Cunningham, Lora learns less and forgets less, 2024. doi:10.48550/ arXiv.2405.09673.
- [61] X. Ma, L. Wang, N. Yang, F. Wei, J. Lin, Fine-tuning llama for multi-stage text retrieval, ArXiv abs/2310.08319 (2023). URL: https://api.semanticscholar.org/CorpusID:263908865.
- [62] P. BehnamGhader, V. Adlakha, M. Mosbach, D. Bahdanau, N. Chapados, S. Reddy, LLM2Vec: Large language models are secretly powerful text encoders, in: First Conference on Language Modeling,

2024. URL: https://openreview.net/forum?id=IW1PR7vEBf.

- [63] S. Hofstätter, M. Zlabinger, A. Hanbury, Interpretable & time-budget-constrained contextualization for re-ranking., in: G. D. Giacomo, A. Catalá, B. Dilkina, M. Milano, S. Barro, A. Bugarín, J. Lang (Eds.), ECAI, volume 325 of Frontiers in Artificial Intelligence and Applications, IOS Press, 2020, pp. 513–520. URL: http://dblp.uni-trier.de/db/conf/ecai/ecai2020.html#HofstatterZH20.
- [64] J. J. Lin, R. Nogueira, A. Yates, Pretrained transformers for text ranking: Bert and beyond, Proceedings of the 14th ACM International Conference on Web Search and Data Mining (2020). URL: https://api.semanticscholar.org/CorpusID:222310837.
- [65] I. Mokrii, L. Boytsov, P. Braslavski, A systematic evaluation of transfer learning and pseudo-labeling with bert-based ranking models, in: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’21, Association for Computing Machinery, New York, NY, USA, 2021, p. 2081–2085. URL: https://doi.org/10.1145/3404835.3463093. doi:10.1145/3404835.3463093.
- [66] A. Parry, T. Jaenich, S. MacAvaney, I. Ounis, Generative relevance feedback and convergence of adaptive re-ranking: University of glasgow terrier team at trec dl 2023, ArXiv abs/2405.01122

(2024). URL: https://api.semanticscholar.org/CorpusID:269502169.

- [67] D. Bahri, Y. Tay, C. Zheng, D. Metzler, A. Tomkins, Choppy: Cut transformer for ranked list truncation, in: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, Association for Computing Machinery, New York, NY, USA, 2020, p. 1513–1516. URL: https://doi.org/10.1145/3397271.3401188. doi:10.1145/ 3397271.3401188.
- [68] A. Salemi, H. Zamani, Towards a search engine for machines: Unified ranking for multiple retrievalaugmented large language models, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 741–751. URL: https://doi.org/10.1145/3626772.3657733. doi:10.1145/3626772.3657733.
- [69] D. Wadden, S. Lin, K. Lo, L. L. Wang, M. van Zuylen, A. Cohan, H. Hajishirzi, Fact or fiction: Verifying scientific claims, in: B. Webber, T. Cohn, Y. He, Y. Liu (Eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Association for Computational Linguistics, Online, 2020, pp. 7534–7550. URL: https://aclanthology.org/2020. emnlp-main.609. doi:10.18653/v1/2020.emnlp-main.609.
- [70] K. Thai, Y. Chang, K. Krishna, M. Iyyer, Relic: Retrieving evidence for literary claims, 2022. doi:10.48550/arXiv.2203.10053.
- [71] J. A. Wang, K. Wang, X. Wang, P. Naidu, L. Bergen, R. Paturi, Scientific document retrieval using multi-level aspect-based queries, Advances in Neural Information Processing Systems 36 (2024).

- A. Details for Reproducibility

- A.1. Model Details

Table 2 includes reference links for each embedding and reranker model used in this study.

- A.2. Computational Resources Experiments were run on commodity GPUs, and no individual run required more than 24 hours.
- A.3. Data Preprocessing

Summary of downsampling for datasets are in Table 3. For the datasets, we either downsampled the corpus size, number of queries, or made gold labels stricter, which we shall explain in more detail.

Corpus Downsampling When downsampling the size of the corpus, we made sure the maximum size of the corpus was 10,000 documents. To construct this new corpus, we first added all of the gold documents from the queries into the corpus to make sure those queries could be correctly answered. Then, we randomly sampled from the larger corpus without replacement to choose the remaining documents. This corpus downsampling was done for BRIGHT’s biology split.

Query Downsampling When downsampling the queries, we simply randomly selected the new queries from the main dataset. This downsampling was done for Scifact.

BRIGHT For the BRIGHT datasets, we use the ‘documents’ split and the gemini-generated reasoning queries, as this type of query reformulation was deemed to work more effectively in the BRIGHT paper.

BIRCO For the BIRCO datasets, the gold documents are densely labeled with ‘qrel’ scores, which is used to gauge how relevant a given gold document is to the query. The higher the qrel score, the more relevant that gold document is. In order to accomodate the binarization of relevance judgement scores that recall requires, we create a new dataset such that all the new gold documents are the gold documents of the original dataset that share the highest qrel score for a given query.

- A.4. Evaluation Metrics

We use Recall as our primary metric for evaluation because retrieval augmented generation (RAG) is increasingly becoming the main way that search engines are accessed [68]. In lieu of order-sensitive metrics such as nDCG, we include Recall@K with various values of K.

- B. Detailed Results We show Recall@10 for individual datasets across academic and enterprise splits in Figures 7, 8, and 9.
- C. Example Rankings

Reranker Failures We present and discuss unexpected reranker errors in §4.2. For the associated queries we report the top-8 retriever and reranker results in Figures 10, 11, 12, 13, 14, 15.

|Model Name|Type|Link|
|---|---|---|
|voyage-2<br><br>|E<br><br>|https://blog.voyageai.com/2024/05/05/voyage-large-2-instruct-instruction-tuned-and-rank-1-on-mteb/|
|text-embedding-3-large<br><br>|E|https://openai.com/index/new-embedding-models-and-api-updates/|
|voyage-rerank-1|R<br><br>|https://blog.voyageai.com/2024/05/29/voyage-rerank-1-cutting-edge-general-purpose-and-multilingual-reranker/|
|cohere-rerank-v3<br><br>|R|https://cohere.com/blog/rerank-3|
|bge-reranker-v2-m3<br><br>|R<br><br>|https://huggingface.co/BAAI/bge-reranker-v2-m3|
|jina-reranker-v2-base-multilingual<br><br>|R<br><br>|https://jina.ai/news/jina-reranker-v2-for-agentic-rag-ultra-fast-multilingual-function-calling-and-code-search/|

###### Table 2 Models, their type (E=Embedding or R=Reranker), and reference links.

|Split<br><br>|Benchmark|CDS|QDS|
|---|---|---|---|
|Scifact [69]|BEIR [27]|No<br><br>|Yes|
|RELIC [70]|BIRCO [26]<br><br>|No|No|
|DORIS MAE [71]|BIRCO [26]<br><br>|No<br><br>|No|
|Biology|BRIGHT [25]<br><br>|Yes|No|
|Pony<br><br>|BRIGHT [25]<br><br>|No|No|
|DocumentationQA<br><br>|Enterprise|No|No|
|FinanceBench [28]|Enterprise<br><br>|Yes<br><br>|N/A|
|ManufacturingQA|Enterprise|No<br><br>|No|

###### Table 3 Summary table for datasets used in our evaluation. CDS = Corpus Downsampled. QDS = Query Downsampled. Query details for Financebench are proprietary.

###### Average Recall@10 for Reranking, BEIR: Scifact (Academic)

0.90

0.85

Recall@10

0.80

0.75

0.70

text-embedding-3-lg

voyage-2

|bm25|
|---|

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

jina-reranker-v2-base-multilingual

bge-reranker-v2-m3

voyage-rerank-1

listwise (gpt-4o-mini)

###### Average Recall@10 for Reranking, BIRCO: RELIC (Academic)

0.50

0.45

0.40

Recall@10

0.35

0.30

0.25

0.20

text-embedding-3-lg

voyage-2

bm25

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

jina-reranker-v2-base-multilingual

bge-reranker-v2-m3

voyage-rerank-1

listwise (gpt-4o-mini)

###### Average Recall@10 for Reranking, BIRCO: DORIS-MAE (Academic)

0.5

Recall@10

0.4

0.3

0.2

text-embedding-3-lg

voyage-2

|bm25|
|---|

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

jina-reranker-v2-base-multilingual

bge-reranker-v2-m3

voyage-rerank-1

listwise (gpt-4o-mini)

- Figure 7: Recall@10 for reranking (Academic).

###### Average Recall@10 for Reranking, BRIGHT: Biology (Academic)

0.7

0.6

0.5

Recall@10

0.4

0.3

0.2

0.1

text-embedding-3-lg

voyage-2

bm25

0.0

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

jina-reranker-v2-base-multilingual

bge-reranker-v2-m3

voyage-rerank-1

listwise (gpt-4o-mini)

###### Average Recall@10 for Reranking, BRIGHT: Pony (Academic)

0.05

0.04

Recall@10

0.03

0.02

0.01

text-embedding-3-lg

voyage-2

bm25

0.00

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

jina-reranker-v2-base-multilingual

bge-reranker-v2-m3

voyage-rerank-1

listwise (gpt-4o-mini)

- Figure 8: Recall@10 for reranking (Academic cont.).

###### Average Recall@10 for Reranking, FinanceBench (Enterprise)

0.8

0.7

Recall@10

0.6

0.5

0.4

0.3

text-embedding-3-lg

voyage-2

bm25

0.2

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

listwise (gpt-4o-mini)

bge-reranker-v2-m3

voyage-rerank-1

###### Average Recall@10 for Reranking, ManufacturingQA (Enterprise)

0.925

0.900

0.875

Recall@10

0.850

0.825

0.800

0.775

text-embedding-3-lg

voyage-2

bm25

0.750

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

listwise (gpt-4o-mini)

bge-reranker-v2-m3

voyage-rerank-1

###### Average Recall@10 for Reranking, DocumentationQA (Enterprise)

0.90

0.85

Recall@10

0.80

0.75

0.70

0.65

text-embedding-3-lg

voyage-2

bm25

0.60

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

10 100 1000 5000 Number of Documents (K)

Retrieval-only

cohere-english

listwise (gpt-4o-mini)

bge-reranker-v2-m3

voyage-rerank-1

- Figure 9: Recall@10 for reranking (Enterprise).

text-embedding-3-large + voyage-rerank-1 (retriever top-8)

Query Many proteins in human cells can be post-translationally modified at lysine residues via acetylation.

Positive Document (Retriever: 1, Reranker: 39)

Protein Lysine Acetylated/Deacetylated Enzymes and the Metabolism-Related Diseases Lysine acetylation is a reversible posttranslational modifcation, an epigenetic phenomenon, referred to as transfer of an acetyl group from acetyl CoA to lysine e- amino group of targeted protein, which is modulated by acetyltransferases (histone/ lysine (K) acetyltransferases, HATs/KATs) and deacetylases (histone/lysine (K) deacetylases, HDACs/KDACs). Lysine acetylation regulates various metabolic processes, such as fatty acid oxidation, Krebs cycle, oxidative phosphorylation, angiogenesis and so on. Thus disorders of lysine acetylation may be correlated with obesity, diabetes and cardiovascular disease, which are termed as the metabolic complication. With accumulating studies on proteomic acetylation, lysine acetylation also involves in cell immune status and degenerative diseases, for example, Alzheimers disease and Huntingtons disease. This review primarily summarizes the current studies of lysine acetylation in metabolism modulation and in metabolism-related diseases, such as cardiovascular disease and fat metabolism disorder.

Negative Document (Retriever: 0, Reranker: 0)

The growing landscape of lysine acetylation links metabolism and cell signalling Lysine acetylation is a conserved protein post-translational modification that links acetyl-coenzyme A metabolism and cellular signalling. Recent advances in the identification and quantification of lysine acetylation by mass spectrometry have increased our understanding of lysine acetylation, implicating it in many biological processes through the regulation of protein interactions, activity and localization. In addition, proteins are frequently modified by other types of acylations, such as formylation, butyrylation, propionylation, succinylation, malonylation, myristoylation, glutarylation and crotonylation. The intricate link between lysine acylation and cellular metabolism has been clarified by the occurrence of several such metabolite-sensitive acylations and their selective removal by sirtuin deacylases. These emerging findings point to new functions for different lysine acylations and deacylating enzymes and also highlight the mechanisms by which acetylation regulates various cellular processes.

Positive Document (Retriever: 1, Reranker: 39)

Protein Lysine Acetylated/Deacetylated Enzymes and the Metabolism-Related Diseases Lysine acetylation is a reversible posttranslational modifcation, an epigenetic phenomenon, referred to as transfer of an acetyl group from acetyl CoA to lysine e- amino group of targeted protein, which is modulated by acetyltransferases (histone/ lysine (K) acetyltransferases, HATs/KATs) and deacetylases (histone/lysine (K) deacetylases, HDACs/KDACs). Lysine acetylation regulates various metabolic processes, such as fatty acid oxidation, Krebs cycle, oxidative phosphorylation, angiogenesis and so on. Thus disorders of lysine acetylation may be correlated with obesity, diabetes and cardiovascular disease, which are termed as the metabolic complication. With accumulating studies on proteomic acetylation, lysine acetylation also involves in cell immune status and degenerative diseases, for example, Alzheimers disease and Huntingtons disease. This review primarily summarizes the current studies of lysine acetylation in metabolism modulation and in metabolism-related diseases, such as cardiovascular disease and fat metabolism disorder.

- Negative Document (Retriever: 2, Reranker: 1)

BRIEFINGS IN FUNCTIONAL GENOMICS AND PROTEOMICS. VOL 5. NO 3. 209221ˆ doi:10.1093/bfgp/ell028 Histone acetylation in gene regulation Genetic information is packaged in the highly dynamic nucleoprotein structure called chromatin. Many biological processes are regulated via post-translational modifications of key proteins. Acetylation of lysine residues at the N-terminal histone tails is one of the most studied covalent modifications influencing gene regulation in eukaryotic cells. This review focuses on the role of enzymes involved in controlling both histone and non-histone proteins acetylation levels in the cell, with particular emphasis on their effects on cancer.

- Negative Document (Retriever: 3, Reranker: 2588)

Protein acetylation affects acetate metabolism, motility and acid stress response in Escherichia coli Although protein acetylation is widely observed, it has been associated with few specific regulatory functions making it poorly understood. To interrogate its functionality, we analyzed the acetylome in Escherichia coli knockout mutants of cobB, the only known sirtuin-like deacetylase, and patZ, the best-known protein acetyltransferase. For four growth conditions, more than 2,000 unique acetylated peptides, belonging to 809 proteins, were identified and differentially quantified. Nearly 65% of these proteins are related to metabolism. The global activity of CobB contributes to the deacetylation of a large number of substrates and has a major impact on physiology. Apart from the regulation of acetyl-CoA synthetase, we found that CobB-controlled acetylation of isocitrate lyase contributes to the fine-tuning of the glyoxylate shunt. Acetylation of the transcription factor RcsB prevents DNA binding, activating flagella biosynthesis and motility, and increases acid stress susceptibility. Surprisingly, deletion of patZ increased acetylation in acetate cultures, which suggests that it regulates the levels of acetylating agents. The results presented offer new insights into functional roles of protein acetylation in metabolic fitness and global cell regulation.

- Negative Document (Retriever: 4, Reranker: 1372)

Histone acetylation and deacetylation in yeast Histone acetylation and deacetylation in the yeast Saccharomyces cerevisiae occur by targeting acetyltransferase and deacetylase enzymes to gene promoters and, in an untargeted and global manner, by affecting most nucleosomes. Recently, new roles for histone acetylation have been uncovered, not only in transcription but also in DNA replication, repair and heterochromatin formation. Interestingly, specific acetylatable lysines can function as binding sites for regulatory factors. Moreover, histone deacetylation is not only repressive but can be required for gene activity.

- Negative Document (Retriever: 5, Reranker: 2589)

Targeting bromodomains: epigenetic readers of lysine acetylation Lysine acetylation is a key mechanism that regulates chromatin structure; aberrant acetylation levels have been linked to the development of several diseases. Acetyl-lysine modifications create docking sites for bromodomains, which are small interaction modules found on diverse proteins, some of which have a key role in the acetylation-dependent assembly of transcriptional regulator complexes. These complexes can then initiate transcriptional programmes that result in phenotypic changes. The recent discovery of potent and highly specific inhibitors for the BET (bromodomain and extra-terminal) family of bromodomains has stimulated intensive research activity in diverse therapeutic areas, particularly in oncology, where BET proteins regulate the expression of key oncogenes and anti-apoptotic proteins. In addition, targeting BET bromodomains could hold potential for the treatment of inflammation and viral infection. Here, we highlight recent progress in the development of bromodomain inhibitors, and their potential applications in drug discovery.

- Negative Document (Retriever: 6, Reranker: 1866)

Identification and Functional Characterization of N-Terminally Acetylated Proteins in Drosophila melanogaster Protein modifications play a major role for most biological processes in living organisms. Amino-terminal acetylation of proteins is a common modification found throughout the tree of life: the N-terminus of a nascent polypeptide chain becomes co-translationally acetylated, often after the removal of the initiating methionine residue. While the enzymes and protein complexes involved in these processes have been extensively studied, only little is known about the biological function of such N-terminal modification events. To identify common principles of N-terminal acetylation, we analyzed the amino-terminal peptides from proteins extracted from Drosophila Kc167 cells. We detected more than 1,200 mature protein N-termini and could show that N-terminal acetylation occurs in insects with a similar frequency as in humans. As the sole true determinant for N-terminal acetylation we could extract the (X)PX rule that indicates the prevention of acetylation under all circumstances. We could show that this rule can be used to genetically engineer a protein to study the biological relevance of the presence or absence of an acetyl group, thereby generating a generic assay to probe the functional importance of N-terminal acetylation. We applied the assay by expressing mutated proteins as transgenes in cell lines and in flies. Here, we present a straightforward strategy to systematically study the functional relevance of N-terminal acetylations in cells and whole organisms. Since the (X)PX rule seems to be of general validity in lower as well as higher eukaryotes, we propose that it can be used to study the function of N-terminal acetylation in all species.

- Negative Document (Retriever: 7, Reranker: 4712)

Acetylation of Histone H3 Lysine 56 Regulates Replication-Coupled Nucleosome Assembly Chromatin assembly factor 1 (CAF-1) and Rtt106 participate in the deposition of newly synthesized histones onto replicating DNA to form nucleosomes. This process is critical for the maintenance of genome stability and inheritance of functionally specialized chromatin structures in proliferating cells. However, the molecular functions of the acetylation of newly synthesized histones in this DNA replication-coupled nucleosome assembly pathway remain enigmatic. Here we show that histone H3 acetylated at lysine 56 (H3K56Ac) is incorporated onto replicating DNA and, by increasing the binding affinity of CAF-1 and Rtt106 for histone H3, H3K56Ac enhances the ability of these histone chaperones to assemble DNA into nucleosomes. Genetic analysis indicates that H3K56Ac acts in a nonredundant manner with the acetylation of the N-terminal residues of H3 and H4 in nucleosome assembly. These results reveal a mechanism by which H3K56Ac regulates replication-coupled nucleosome assembly mediated by CAF-1 and Rtt106.

###### Figure 10: Top-8 results from the retriever on Scifact. The ranks assigned by the retriever and reranker shown in parens (zero-indexed).

text-embedding-3-large + voyage-rerank-1 (reranker top-8)

Query Many proteins in human cells can be post-translationally modified at lysine residues via acetylation.

Positive Document (Retriever: 1, Reranker: 39)

Protein Lysine Acetylated/Deacetylated Enzymes and the Metabolism-Related Diseases Lysine acetylation is a reversible posttranslational modifcation, an epigenetic phenomenon, referred to as transfer of an acetyl group from acetyl CoA to lysine e- amino group of targeted protein, which is modulated by acetyltransferases (histone/ lysine (K) acetyltransferases, HATs/KATs) and deacetylases (histone/lysine (K) deacetylases, HDACs/KDACs). Lysine acetylation regulates various metabolic processes, such as fatty acid oxidation, Krebs cycle, oxidative phosphorylation, angiogenesis and so on. Thus disorders of lysine acetylation may be correlated with obesity, diabetes and cardiovascular disease, which are termed as the metabolic complication. With accumulating studies on proteomic acetylation, lysine acetylation also involves in cell immune status and degenerative diseases, for example, Alzheimers disease and Huntingtons disease. This review primarily summarizes the current studies of lysine acetylation in metabolism modulation and in metabolism-related diseases, such as cardiovascular disease and fat metabolism disorder.

- Negative Document (Retriever: 0, Reranker: 0)

The growing landscape of lysine acetylation links metabolism and cell signalling Lysine acetylation is a conserved protein post-translational modification that links acetyl-coenzyme A metabolism and cellular signalling. Recent advances in the identification and quantification of lysine acetylation by mass spectrometry have increased our understanding of lysine acetylation, implicating it in many biological processes through the regulation of protein interactions, activity and localization. In addition, proteins are frequently modified by other types of acylations, such as formylation, butyrylation, propionylation, succinylation, malonylation, myristoylation, glutarylation and crotonylation. The intricate link between lysine acylation and cellular metabolism has been clarified by the occurrence of several such metabolite-sensitive acylations and their selective removal by sirtuin deacylases. These emerging findings point to new functions for different lysine acylations and deacylating enzymes and also highlight the mechanisms by which acetylation regulates various cellular processes.

Negative Document (Retriever: 2, Reranker: 1)

BRIEFINGS IN FUNCTIONAL GENOMICS AND PROTEOMICS. VOL 5. NO 3. 209221ˆ doi:10.1093/bfgp/ell028 Histone acetylation in gene regulation Genetic information is packaged in the highly dynamic nucleoprotein structure called chromatin. Many biological processes are regulated via post-translational modifications of key proteins. Acetylation of lysine residues at the N-terminal histone tails is one of the most studied covalent modifications influencing gene regulation in eukaryotic cells. This review focuses on the role of enzymes involved in controlling both histone and non-histone proteins acetylation levels in the cell, with particular emphasis on their effects on cancer.

Negative Document (Retriever: 2608, Reranker: 2)

Guide to yeast genetics and molecular biology This volume and its companion, Volume 350, are specifically designed to meet the needs of graduate students and postdoctoral students as well as researchers, by providing all the up-to-date methods necessary to study genes in yeast. Procedures are included that enable newcomers to set up a yeast laboratory and to master basic manipulations. Relevant background and reference information given for procedures can be used as a guide to developing protocols in a number of disciplines. Specific topics addressed in this book include cytology, biochemistry, cell fractionation, and cell biology.

Negative Document (Retriever: 2713, Reranker: 3)

Antigen availability determines CD8+ T cell-dendritic cell interaction kinetics and memory fate decisions. T cells are activated by antigen (Ag)-bearing dendritic cells (DCs) in lymph nodes in three phases. The duration of the initial phase of transient, serial DC-T cell interactions is inversely correlated with Ag dose. The second phase, characterized by stable DC-T cell contacts, is believed to be necessary for full-fledged T cell activation. Here we have shown that this is not the case. CD8+ T cells interacting with DCs presenting low-dose, short-lived Ag did not transition to phase 2, whereas higher Ag dose yielded phase 2 transition. Both antigenic constellations promoted T cell proliferation and effector differentiation but yielded different transcriptome signatures at 12 hr and 24 hr. T cells that experienced phase 2 developed long-lived memory, whereas conditions without stable contacts yielded immunological amnesia. Thus, T cells make fate decisions within hours after Ag exposure, resulting in long-term memory or abortive effector responses, correlating with T cell-DCs interaction kinetics.

Negative Document (Retriever: 4858, Reranker: 4)

The 5TMM series: a useful in vivo mouse model of human multiple myeloma. The present invention provides a combination sink and dishwashing apparatus having a sink sharing a common side wall with a cabinet which defines a closed space. The cabinet has a wire basket for holding and washing a plurality of dishes within the cabinet. The common side wall is a part of the cabinet that defines the closed space and is positioned to form one side of the sink. The common side wall can be opened to allow the wire basket to slide from within the cabinet and into the sink, for loading and unloading the dishes. Within the cabinet, the invention contains a pump powered by a motor, the pump spraying water through a rotatably mounted spray arm onto the dishes to wash them, as with traditional dishwashing machines. The invention preferably includes a garbage disposal which is also powered by the motor. In an alternative embodiment, this invention can include two cabinets as described above, each located on opposing sides of the sink.

Negative Document (Retriever: 3337, Reranker: 5)

On the origins of ultra-fine anaphase bridges. Comment on: Chan KL, Palmai-Pallag T, Ying S, Hickson ID. Replication stress induces sister-chromatid bridging at fragile site loci in mitosis. Nat Cell Biol 2009; 11:753-60.

Negative Document (Retriever: 3364, Reranker: 6)

Killing by bactericidal antibiotics does not depend on reactive oxygen species. Bactericidal antibiotics kill by modulating their respective targets. This traditional view has been challenged by studies that propose an alternative, unified mechanism of killing, whereby toxic reactive oxygen species (ROS) are produced in the presence of antibiotics. We found no correlation between an individual cell’s probability of survival in the presence of antibiotic and its level of ROS. An ROS quencher, thiourea, protected cells from antibiotics present at low concentrations, but the effect was observed under anaerobic conditions as well. There was essentially no difference in survival of bacteria treated with various antibiotics under aerobic or anaerobic conditions. This suggests that ROS do not play a role in killing of bacterial pathogens by antibiotics.

Negative Document (Retriever: 3686, Reranker: 7)

Molecular mechanisms underlying tumor dormancy. Evidence suggests that dormant, microscopic tumors are not only common, but are highly prevalent in otherwise healthy individuals. Due to their small size and non-invasive nature, these dormant tumors remain asymptomatic and, in most cases, undetected. With advances in diagnostic imaging and molecular biology, it is now becoming clear that such neoplasms can remain in an asymptomatic, dormant stage for considerable periods of time without expanding in size. Although a number of processes may play a role in thwarting the expansion of microscopic tumors, one critical mechanism behind tumor dormancy is the ability of the tumor population to induce angiogenesis. Although cancer can arise through multiple pathways, it is assumed that essentially most tumors begin as microscopic, non-angiogenic neoplasms which cannot expand in size until vasculature is established. It is now becoming clear that cancer does not progress through a continuous exponential growth and mass expansion. Clinical cancer is usually manifested only in late, unavoidably symptomatic stages of the disease when tumors are sufficiently large to be readily detected. While dormancy in primary tumors is best defined as the time between the carcinogenic transformation event and the onset of inexorable progressive growth, it can also occur

- as minimal residual or occult disease from treated tumors or as micro-metastases. The existence of dormant tumors has important implications for the early detection and treatment of cancer. Elucidating the regulatory machinery of these processes will be instrumental in identifying novel early cancer biomarkers and could provide a rationale for the development of dormancy-promoting tumor therapies. Despite the high prevalence of microscopic, dormant tumors in humans and the significant clinical implications of their early detection, this area in cancer research has, to date, been under-investigated. In this mini review observations, models and experimental approaches to study tumor dormancy are summarized. Additionally, analogies and distinctions between the concepts of "tumor dormancy" and that of the "cellular dormancy" of tumor cells, as well as between the "exit from tumor dormancy" and the "onset of the angiogenic switch" are discussed.

###### Figure 11: Top-8 results from the reranker on Scifact. The ranks assigned by the retriever and reranker shown in parens (zero-indexed).

text-embedding-3-large + bge-reranker-v2-m3 (retriever top-8)

Query The binding orientation of the ML-SA1 activator at hTRPML2 is different from the binding orientation of the ML-SA1 activator at hTRPML1.

Positive Document (Retriever: 0, Reranker: 34)

Human TRPML1 channel structures in open and closed conformations Transient receptor potential mucolipin 1 (TRPML1) is a Ca2+-releasing cation channel that mediates the calcium signalling and homeostasis of lysosomes. Mutations in TRPML1 lead to mucolipidosis type IV, a severe lysosomal storage disorder. Here we report two electron cryo-microscopy structures of full-length human TRPML1: a 3.72-A apo structure at pH 7.0 in the closed state, and a 3.49-A agonist-bound structure at pH 6.0 in an open state. Several aromatic and hydrophobic residues in pore helix 1, helices S5 and S6, and helix S6 of a neighbouring subunit, form a hydrophobic cavity to house the agonist, suggesting a distinct agonist-binding site from that found in TRPV1, a TRP channel from a different subfamily. The opening of TRPML1 is associated with distinct dilations of its lower gate together with a slight structural movement of pore helix 1. Our work reveals the regulatory mechanism of TRPML channels, facilitates better understanding of TRP channel activation, and provides insights into the molecular basis of mucolipidosis type IV pathogenesis.

Positive Document (Retriever: 0, Reranker: 34)

Human TRPML1 channel structures in open and closed conformations Transient receptor potential mucolipin 1 (TRPML1) is a Ca2+-releasing cation channel that mediates the calcium signalling and homeostasis of lysosomes. Mutations in TRPML1 lead to mucolipidosis type IV, a severe lysosomal storage disorder. Here we report two electron cryo-microscopy structures of full-length human TRPML1: a 3.72-A apo structure at pH 7.0 in the closed state, and a 3.49-A agonist-bound structure at pH 6.0 in an open state. Several aromatic and hydrophobic residues in pore helix 1, helices S5 and S6, and helix S6 of a neighbouring subunit, form a hydrophobic cavity to house the agonist, suggesting a distinct agonist-binding site from that found in TRPV1, a TRP channel from a different subfamily. The opening of TRPML1 is associated with distinct dilations of its lower gate together with a slight structural movement of pore helix 1. Our work reveals the regulatory mechanism of TRPML channels, facilitates better understanding of TRP channel activation, and provides insights into the molecular basis of mucolipidosis type IV pathogenesis.

- Negative Document (Retriever: 1, Reranker: 96)

A small molecule restores function to TRPML1 mutant isoforms responsible for mucolipidosis type IV. Mucolipidosis type IV (MLIV) is an autosomal recessive lysosomal storage disorder often characterized by severe neurodevelopmental abnormalities and neuro-retinal degeneration. Mutations in the TRPML1 gene are causative for MLIV. We used lead optimization strategies to identify–and MLIV patient fibroblasts to test–small-molecule activators for their potential to restore TRPML1 mutant channel function. Using the whole-lysosome planar patch-clamp technique, we found that activation of MLIV mutant isoforms by the endogenous ligand PI(3,5)P2 is strongly reduced, while activity can be increased using synthetic ligands. We also found that the F465L mutation renders TRPML1 pH insensitive, while F408 impacts synthetic ligand binding. Trafficking defects and accumulation of zinc in lysosomes of MLIV mutant fibroblasts can be rescued by the small molecule treatment. Collectively, our data demonstrate that small molecules can be used to restore channel function and rescue disease associated abnormalities in patient cells expressing specific MLIV point mutations.

- Negative Document (Retriever: 2, Reranker: 82)

Lysosomal localization of TRPML3 depends on TRPML2 and the mucolipidosis-associated protein TRPML1. Mucolipidosis type IV is an autosomal recessive lysosomal storage disorder characterized by severe neurodegeneration, achlorhydria, and visual impairments such as corneal opacity and strabismus. The disease arises due to mutations in a group 2 transient receptor potential (TRP)-related cation channel, TRPML1. Mammals encode two additional TRPML proteins named TRPML2 and TRPML3. Information regarding the propensity of these proteins to multimerize, their subcellular distribution and mechanisms that regulate their trafficking are limited. Here we demonstrate that TRPMLs interact to form homo- and heteromultimers. Moreover, the presence of either TRPML1 or TRPML2 specifically influences the spatial distribution of TRPML3. TRPML1 and TRPML2 homomultimers are lysosomal proteins, whereas TRPML3 homomultimers are in the endoplasmic reticulum. However, TRPML3 localizes to lysosomes when coexpressed with either TRPML1 or TRPML2 and is comparably mislocalized when lysosomal targeting of TRPML1 and TRPML2 is disrupted. Conversely, TRPML3 does not cause retention of TRPML1 or TRPML2 in the endoplasmic reticulum. These data demonstrate that there is a hierarchy controlling the subcellular distributions of the TRPMLs such that TRPML1 and TRPML2 dictate the localization of TRPML3 and not vice versa.

- Negative Document (Retriever: 3, Reranker: 2013)

Nonclassical binding of formylated peptide in crystal structure of the MHC class lb molecule H2-M3 H2-M3 is a class Ib MHC molecule of the mouse with a 10(4)-fold preference for binding N-formylated peptides. To elucidate the basis of this unusual specificity, we expressed and crystallized a soluble form of M3 with a formylated nonamer peptide, fMYFINILTL, and determined the structure by X-ray crystallography. M3, refined at 2.1 A resolution, resembles class la MHC molecules in its overall structure, but differs in the peptide-binding groove. The A pocket, which usually accommodates the free N-terminus of a bound peptide, is closed, and the peptide is shifted one residue, such that the P1 side chain is lodged in the B pocket. The formyl group is coordinated by His-9 and a bound water on the floor of the groove.

- Negative Document (Retriever: 4, Reranker: 549)

AMPA receptor ligand binding domain mobility revealed by functional cross linking. Glutamate receptors mediate the majority of excitatory synaptic transmission in the CNS. The AMPA-subtype has rapid kinetics, with activation, deactivation and desensitization proceeding on the millisecond timescale or faster. Crystallographic, biochemical, and functional studies suggest that GluR2 Cys mutants which form intermolecular disulfide cross-links between the lower D2 lobes of the ligand binding cores can be trapped in a conformation that represents the desensitized state. We used multi-channel rapid perfusion techniques to examine the state dependence of cross-linking in these mutants. Under reducing conditions, both wild-type GluR2 and the G725C and S729C mutants have normal activation and desensitization kinetics, but the Cys mutants can be efficiently trapped in nonconducting states when oxidized. In contrast the I664C mutant is only partially inactivated under oxidizing conditions. For S729C, disulfide cross-links form rapidly when receptors are desensitized in the presence of glutamate, but receptors also become trapped at rest, in the absence of agonist. We assessed such spontaneous trapping in various conditions, including CNQX, a competitive antagonist; kainate, a weak partial agonist; or when desensitization was blocked by the L483Y mutation that stabilizes the D1 dimer interface. These experiments suggest that trapping in the absence of glutamate is due to two motions: Spontaneous breaking of the D1 dimer interface and hyperextension of the lower lobes of the ligand binding core. These data show that the glutamate binding domains are surprisingly mobile in the absence of ligand, which could influence receptor activity in the brain.

- Negative Document (Retriever: 5, Reranker: 799)

Structural Rearrangements of NR1/NR2A NMDA Receptors during Allosteric Inhibition Ionotropic glutamate receptor (iGluR) subunits contain a large N-terminal domain (NTD) that precedes the agonist-binding domain (ABD) and participates in subunit oligomerization. In NMDA receptors (NMDARs), the NTDs of NR2A and NR2B subunits also form binding sites for the endogenous inhibitor Zn(2+) ion. Although these allosteric sites have been characterized in detail, the molecular mechanisms by which the NTDs communicate with the rest of the receptor to promote its inhibition remain unknown. Here, we identify the ABD dimer interface as a major structural determinant that permits coupling between the NTDs and the channel gate. The strength of this interface also controls proton inhibition, another form of allosteric modulation of NMDARs. Conformational rearrangements

- at the ABD dimer interface thus appear to be a key mechanism conserved in all iGluR subfamilies, but have evolved to fulfill different functions: fast desensitization at AMPA and kainate receptors, allosteric inhibition at NMDARs.

- Negative Document (Retriever: 6, Reranker: 1850)

The Extracellular Surface of the GLP-1 Receptor Is a Molecular Trigger for Biased Agonism Ligand-directed signal bias offers opportunities for sculpting molecular events, with the promise of better, safer therapeutics. Critical to the exploitation of signal bias is an understanding of the molecular events coupling ligand binding to intracellular signaling. Activation of class B G protein-coupled receptors is driven by interaction of the peptide N terminus with the receptor core. To understand how this drives signaling, we have used advanced analytical methods that enable separation of effects on pathway-specific signaling from those that modify agonist affinity and mapped the functional consequence of receptor modification onto three-dimensional models of a receptor-ligand complex. This yields molecular insights into the initiation of receptor activation and the mechanistic basis for biased agonism. Our data reveal that peptide agonists can engage different elements of the receptor extracellular face to achieve effector coupling and biased signaling providing a foundation for rational design of biased agonists.

- Negative Document (Retriever: 7, Reranker: 259)

Role of TRPML and two-pore channels in endolysosomal cation homeostasis. The transient receptor potential (TRP) channels TRPML1, TRPML2, and TRPML3 (also called mucolipins 1-3 or MCOLN1-3) are nonselective cation channels. Mutations in the Trpml1 gene cause mucolipidosis type IV in humans with clinical features including psychomotor retardation, corneal clouding, and retinal degeneration, whereas mutations in the Trpml3 gene cause deafness, circling behavior, and coat color dilution in mice. No disease-causing mutations are reported for the Trpml2 gene. Like TRPML channels, which are expressed in the endolysosomal pathway, two-pore channels (TPCs), namely TPC1, TPC2, and TPC3, are found in intracellular organelles, in particular in endosomes and lysosomes. Both TRPML channels and TPCs may function as calcium/cation release channels in endosomes, lysosomes, and lysosome-related organelles with TRPMLs being activated by phosphatidylinositol 3,5-bisphosphate and regulated by pH and TPCs being activated by nicotinic acid adenine dinucleotide phosphate in a calcium- and pH-dependent manner. They may also be involved in endolysosomal transport and fusion processes, e.g., as intracellular calcium sources. Currently, however, the exact physiological roles of TRPML channels and TPCs remain quite elusive, and whether TRPML channels are purely endolysosomal ion channels or whether they may also be functionally active at the plasma membrane in vivo remains to be determined.

###### Figure 12: Top-8 results from the retriever on Scifact. The ranks assigned by the retriever and reranker shown in parens (zero-indexed).

text-embedding-3-large + bge-reranker-v2-m3 (reranker top-8)

Query The binding orientation of the ML-SA1 activator at hTRPML2 is different from the binding orientation of the ML-SA1 activator at hTRPML1.

Positive Document (Retriever: 0, Reranker: 34)

Human TRPML1 channel structures in open and closed conformations Transient receptor potential mucolipin 1 (TRPML1) is a Ca2+-releasing cation channel that mediates the calcium signalling and homeostasis of lysosomes. Mutations in TRPML1 lead to mucolipidosis type IV, a severe lysosomal storage disorder. Here we report two electron cryo-microscopy structures of full-length human TRPML1: a 3.72-A apo structure at pH 7.0 in the closed state, and a 3.49-A agonist-bound structure at pH 6.0 in an open state. Several aromatic and hydrophobic residues in pore helix 1, helices S5 and S6, and helix S6 of a neighbouring subunit, form a hydrophobic cavity to house the agonist, suggesting a distinct agonist-binding site from that found in TRPV1, a TRP channel from a different subfamily. The opening of TRPML1 is associated with distinct dilations of its lower gate together with a slight structural movement of pore helix 1. Our work reveals the regulatory mechanism of TRPML channels, facilitates better understanding of TRP channel activation, and provides insights into the molecular basis of mucolipidosis type IV pathogenesis.

Negative Document (Retriever: 2069, Reranker: 0)

-Adrenergic receptor antagonism prevents anxiety-like behavior and microglial reactivity induced by repeated social defeat. Psychosocial stress is associated with altered immune function and development of psychological disorders including anxiety and depression. Here we show that repeated social defeat in mice increased c-Fos staining in brain regions associated with fear and threat appraisal and promoted anxiety-like behavior in a -adrenergic receptor-dependent manner. Repeated social defeat also significantly increased the number of CD11b(+)/CD45(high)/Ly6C(high) macrophages that trafficked to the brain. In addition, several inflammatory markers were increased on the surface of microglia (CD14, CD86, and TLR4) and macrophages (CD14 and CD86) after social defeat. Repeated social defeat also increased the presence of deramified microglia in the medial amygdala, prefrontal cortex, and hippocampus. Moreover, mRNA analysis of microglia indicated that repeated social defeat increased levels of interleukin (IL)-1 and reduced levels of glucocorticoid responsive genes [glucocorticoid-induced leucine zipper (GILZ) and FK506 binding protein-51 (FKBP51)]. The stress-dependent changes in microglia and macrophages were prevented by propranolol, a -adrenergic receptor antagonist. ...

Negative Document (Retriever: 3054, Reranker: 1)

Hypoxia in relation to vasculature and proliferation in liver metastases in patients with colorectal cancer. PURPOSE To investigate hypoxia measured by pimonidazole binding, glucose transporter 1 (GLUT1) and carbonic anhydrase IX (CA-IX) expression, proliferation, and vascularity in liver metastases of colorectal cancer and to compare GLUT1 and CA-IX expression in corresponding primary tumors. METHODS AND MATERIALS Twenty-five patients with liver metastases of colorectal cancer, planned for metastasectomy, were included. The hypoxia marker pimonidazole and proliferation marker iododeoxyuridine were administered before surgery. After immunofluorescent staining of the frozen metastases, pimonidazole binding, vascularity, and proliferation were analyzed quantitatively. Thirteen paraffin-embedded primary tumors were stained immunohistochemically for GLUT1 and CA-IX expression, which was analyzed semiquantitatively in primary tumors and corresponding liver metastases. RESULTS In liver metastases, pimonidazole binding showed a pattern consistent with diffusion-limited hypoxia. The mean pimonidazole-positive fraction was 0.146; the mean distance from vessels to pimonidazole-positive areas was 80 microm. When expressed, often co-localization was observed between pimonidazole binding and GLUT1 or CA-IX expression, but microregional areas of mismatch were also observed. ...

Negative Document (Retriever: 3645, Reranker: 2)

High-intensity functional exercise program and protein-enriched energy supplement for older persons dependent in activities of daily living: a randomised controlled trial. The aims of this randomised controlled trial were to determine if a high-intensity functional exercise program improves balance, gait ability, and lower-limb strength in older persons dependent in activities of daily living and if an intake of protein-enriched energy supplement immediately after the exercises increases the effects of the training. One hundred and ninety-one older persons dependent in activities of daily living, living in residential care facilities, and with a Mini-Mental State Examination (MMSE) score of ? 10 participated. They were randomised to a high-intensity functional exercise program or a control activity, which included 29 sessions over 3 months, as well as to protein-enriched energy supplement or placebo. ...

Negative Document (Retriever: 3060, Reranker: 3)

Total atherosclerotic burden by whole body magnetic resonance angiography predicts major adverse cardiovascular events. OBJECTIVE The purpose of the present study was to investigate the relationship between the Total Atherosclerotic Score (TAS), a measurement of the overall atherosclerotic burden of the arterial tree by whole body magnetic resonance angiography (WBMRA), and the risk of major adverse cardiovascular events (MACE), defined as cardiac death, myocardial infarction, stroke and/or coronary revascularization, assuming that TAS predicts MACE. METHODS AND RESULTS 305 randomly selected 70 year-old subjects (47% women) underwent WBMRA. Their atherosclerotic burden was evaluated and TAS > 0, that is atherosclerotic changes, were found in 68% of subjects. During follow-up (mean 4.8 years), MACE occurred in 25 subjects (8.2%). Adjusting for multiple risk factors, TAS was associated with MACE (OR 8.86 for any degree of vessel lumen abnormality, 95%CI 1.14-69.11, p = 0.037). In addition, TAS improved discrimination and reclassification when added to the Framingham risk score (FRS), and ROC (Receiver Operator Curve) increased from 0.681 to 0.750 (p = 0.0421). CONCLUSION In a population-based sample of 70 year old men and women WBMRA, with TAS, predicted MACE independently of major cardiovascular risk factors.

Negative Document (Retriever: 3354, Reranker: 4)

Increased stress-induced inflammatory responses in male patients with major depression and increased early life stress. OBJECTIVE The authors sought to determine innate immune system activation following psychosocial stress in patients with major depression and increased early life stress. METHOD Plasma interleukin (IL)-6, lymphocyte subsets, and DNA binding of nuclear factor (NF)-kB in peripheral blood mononuclear cells were compared in medically healthy male subjects with current major depression and increased early life stress (N=14) versus nondepressed male comparison subjects (N=14) before and after completion of the Trier Social Stress Test. RESULTS Trier Social Stress Test-induced increases in IL-6 and NF-kappaB DNA-binding were greater in major depression patients with increased early life stress and independently correlated with depression severity, but not early life stress. Natural killer (NK) cell percentages also increased following stress. However, there were no differences between groups and no correlation between NK cell percentage and stress-induced NF-kappaB DNA-binding or IL-6. CONCLUSIONS Male major depression patients with increased early life stress exhibit enhanced inflammatory responsiveness to psychosocial stress, providing preliminary indication of a link between major depression, early life stress and adverse health outcomes in diseases associated with inflammation.

Negative Document (Retriever: 223, Reranker: 5)

Antigen presentation by major histocompatibility complex class I-B molecules. Class I-b genes constitute the majority of MHC class I loci. These monomorphic or oligomorphic molecules have been described in many organisms; they are best characterized in the mouse, which contains a substantial number of potentially intact genes. Two main characteristics differentiate class I-b from class I-a molecules: limited polymorphism and lower cell surface expression. These distinguishing features suggest possible generalizations regarding the evolution and function of this class. Additionally, class I-b proteins tend to have shorter cytoplasmic domains or in some cases may be secreted or may substitute a lipid anchor for the transmembrane domain. Some are also expressed in a limited distribution of cells or tissues. At least six mouse MHC class I-b molecules have been shown to present antigens to alpha beta or gamma delta T cells. Recent advances have provided insight into the physiological function of H-2M3a and have defined the natural peptide-binding motif of Qa-2. In addition, significant progress has been made toward better understanding of other class I-b molecules, including Qa-1, TL, HLA-E, HLA-G, and the MHC-unlinked class I molecule CD1. We begin this review, however, by arguing that the dichotomous categorization of MHC genes as class I-a and I-b is conceptually misleading, despite its historical basis and practical usefulness. With these reservations in mind, we then discuss antigen presentation by MHC class I-b molecules with particular attention to their structure, polymorphism, requirements for peptide antigen binding and tissue expression.

Negative Document (Retriever: 2882, Reranker: 6)

Increased expression of CYP1A1 and CYP1B1 in ovarian/peritoneal endometriotic lesions. Endometriosis is an estrogen-dependent disease affecting up to 10% of all premenopausal women. There is evidence that different endometriosis sites show distinct local estrogen concentration, which, in turn, might be due to a unique local estrogen metabolism. We aimed to investigate whether there was a site-specific regulation of selected enzymes responsible for the oxidative metabolism of estrogens in biopsy samples and endometrial and endometriotic stromal cells. Cytochrome P450 (CYP) 1A1 and CYP1B1 mRNA and protein expressions in deep-infiltrating (rectal, retossigmoidal, and uterossacral) lesions, superficial (ovarian and peritoneal) lesions, and eutopic and healthy (control) endometrium were evaluated by real-time PCR and western blot. Using a cross-sectional study design with 58 premenopausal women who were not under hormonal treatment, we were able to identify an overall increased CYP1A1 and CYP1B1 mRNA expression in superficial lesions compared with the healthy endometrium. CYP1A1 mRNA expression in superficial lesions was also greater than in the eutopic endometrium. Interestingly, we found a similar pattern of CYP1A1 and CYP1B1 expression in in vitro stromal cells isolated from ovarian lesions (n=3) when compared with stromal cells isolated from either rectum lesions or eutopic endometrium. In contradiction, there was an increased half-life of estradiol (measured by HPLC-MS-MS) in ovarian endometriotic stromal cells compared with paired eutopic stromal endometrial cells. Our results indicate that there is a site-dependent regulation of CYP1A1 and CYP1B1 in ovarian/peritoneal lesions and ovarian endometriotic stromal cells, whereas a slower metabolism is taking place in these cells.

Negative Document (Retriever: 3375, Reranker: 7)

Antioxidants attenuate the plasma cytokine response to exercise in humans. Exercise increases plasma TNF-alpha, IL-1beta, and IL-6, yet the stimuli and sources of TNF-alpha and IL-1beta remain largely unknown. We tested the role of oxidative stress and the potential contribution of monocytes in this cytokine (especially IL-1beta) response in previously untrained individuals. Six healthy nonathletes performed two 45-min bicycle exercise sessions at 70% of Vo(2 max) before and after a combination of antioxidants (vitamins E, A, and C for 60 days; allopurinol for 15 days; and N-acetylcysteine for 3 days). Blood was drawn at baseline, end-exercise, and 30 and 120 min postexercise. Plasma cytokines were determined by ELISA and monocyte intracellular cytokine level by flow cytometry. Before antioxidants, TNF-alpha increased by 60%, IL-1beta by threefold, and IL-6 by sixfold secondary to exercise (P < 0.05). After antioxidants, plasma IL-1beta became undetectable, the TNF-alpha response to exercise was abolished, and the IL-6 response was significantly blunted (P < 0.05). Exercise did not increase the percentage of monocytes producing the cytokines or their mean fluorescence intensity. We conclude that in untrained humans oxidative stress is a major stimulus for exercise-induced cytokine production and that monocytes play no role in this process.

###### Figure 13: Top-8 results from the reranker on Scifact. The ranks assigned by the retriever and reranker shown in parens (zero-indexed). A few documents are truncated to conserve space.

text-embedding-3-large + cohere-rerank-english-v3.0 (retriever top-8)

Query Less than 10% of the gabonese children with Schimmelpenning-Feuerstein-Mims syndrome (SFM) had a plasma lactate of more than 5mmol/L.

Positive Document (Retriever: 5, Reranker: 33)

Assessment of Volume Depletion in Children with Malaria Background The degree of volume depletion in severe malaria is currently unknown, although knowledge of fluid compartment volumes can guide therapy. To assist management of severely ill children, and to test the hypothesis that volume changes in fluid compartments reflect disease severity, we measured body compartment volumes in Gabonese children with malaria. Methods and Findings Total body water volume (TBW) and extracellular water volume (ECW) were estimated in children with severe or moderate malaria and in convalescence by tracer dilution with heavy water and bromide, respectively. Intracellular water volume (ICW) was derived from these parameters. Bioelectrical impedance analysis estimates of TBW and ECW were calibrated against dilution methods, and bioelectrical impedance analysis measurements were taken daily until discharge. Sixteen children had severe and 19 moderate malaria. Severe childhood malaria was associated with depletion of TBW (mean [SD] of 37 [33] ml/kg, or 6.7% [6.0%]) relative to measurement at discharge. This is defined as mild dehydration in other conditions. ECW measurements were normal on admission in children with severe malaria and did not rise in the first few days of admission. Volumes in different compartments (TBW, ECW, and ICW) were not related to hyperlactataemia or other clinical and laboratory markers of disease severity. Moderate malaria was not associated with a depletion of TBW.

###### Negative Document (Retriever: 0, Reranker: 2)

Mutation of the fumarase gene in two siblings with progressive encephalopathy and fumarase deficiency. We report an inborn error of the tricarboxylic acid cycle, fumarase deficiency, in two siblings born to first cousin parents. They presented with progressive encephalopathy, dystonia, leucopenia, and neutropenia. Elevation of lactate in the cerebrospinal fluid and high fumarate excretion in the urine led us to investigate the activities of the respiratory chain and of the Krebs cycle, and to finally identify fumarase deficiency in these two children. The deficiency was profound and present in all tissues investigated, affecting the cytosolic and the mitochondrial fumarase isoenzymes to the same degree. Analysis of fumarase cDNA demonstrated that both patients were homozygous for a missense mutation, a G-955–>C transversion, predicting a Glu-319–>Gln substitution. This substitution occurred in a highly conserved region of the fumarase cDNA. Both parents exhibited half the expected fumarase activity in their lymphocytes and were found to be heterozygous for this substitution. The present study is to our knowledge the first molecular characterization of tricarboxylic acid deficiency, a rare inherited inborn error of metabolism in childhood.

###### Negative Document (Retriever: 1, Reranker: 7)

Experimental and clinical studies on lactate and pyruvate as indicators of the severity of acute circulatory failure (shock). The increase in lactate (L) and pyruvate (P) content of arterial blood during experimental and clinical shock states and the extent to which such increases serve as measures of oxygen deficit and irreversible injury were investigated on an empirical basis. A standardized method for production of hemorrhagic shock in the Wistar rat was employed. During a 4-hour bleeding period, oxygen consumption of the rat was reduced to approximately 40% of control value, pH was reduced from 7.39 to 7.08, and a concurrent increase in L from 0.80 to 6.06 mm and in P from 0.07 to 0.18 mm were observed. Cumulative oxygen debt correlated with log L (r = 0.50; P < 0.0005) and both were significantly related to survival. Correlation of cumulative oxygen debt and survival, both with P and with computed values of the lactate pyruvate ratio (L/P) and excess lactate (XL), were of no higher magnitude. Partial correlation analysis demonstrated that neither the measurement of P nor the computation of L/P or XL improved predictability...

###### Negative Document (Retriever: 2, Reranker: 179)

A paediatric case of sideroblastic anaemia. Ultrastructural studies of erythroblasts cultured from marrow BFU-E in a methylcellulose micromethod. We examined the morphological and functional characteristics of erythroblasts derived from marrow erythroid progenitor cells grown in a methylcellulose microculture, which were taken from a female child with rare atypical sideroblastic anaemia (SA) partially responsive to pyridoxine. Colony formation was within the normal range in three successive cultures (median values: 82.25 CFU-E and 16.4 BFU-E derived colonies/6.6 X 10(4) cells) compared to growth by normal cells (65-315 CFU-E and 9-40 BFU-E). We evaluated in vitro differentiation by biochemical microassay of a cytosol enzyme involved in the haem pathway: uroporphyrinogen I synthase (UROS). The UROS values in the erythroid colonies from SA marrow were at the lowere end of the normal range (median values: 6.7 +/- 0.3 and 14.4 +/- 3.8 pmol uroporphyrinogen/h in CFU-E and BFU-E-derived colonies respectively versus 17.4 +/- 7.3 and 25 +/- 7.2 pmol/h in CFU-E and BFU-E colonies from normal subjects. Ultrastructural examination of the SA erythroblasts from non-cultured bone marrow or derived from cultured BFU-E revealed the characteristic deposition of iron in mitochondria around the nucleus of most cells (ringed sideroblasts). However, the majority of cultured cells had marked dyserythropoietic features, with a large number of bilobulated or trilobulated erythroblasts, multiple cytoplasmic vacuoles, numerous abnormalities of the nucleus, and excessive membrane material beneath the plasma membrane, all features difficult to observe in non-cultured marrows.

###### Negative Document (Retriever: 3, Reranker: 1185)

Neurological development of 5-year-old children receiving a low-saturated fat, low-cholesterol diet since infancy: A randomized controlled trial. CONTEXT Early childhood introduction of nutritional habits aimed at atherosclerosis prevention is compatible with normal growth, but its effect on neurological development is unknown. OBJECTIVE To analyze how parental counseling aimed at keeping children’s diets low in saturated fat and cholesterol influences neurodevelopment during the first 5 years of life. DESIGN Randomized controlled trial conducted between February 1990 and November 1996. SETTING Outpatient clinic of a university department in Turku, Finland. PARTICIPANTS A total of 1062 seven-month-old infants and their parents, recruited at well-baby clinics between 1990 and 1992. At age 5 years, 496 children still living in the city of Turku were available to participate in neurodevelopmental testing. INTERVENTION Participants were randomly assigned to receive individualized counseling aimed at limiting the child’s fat intake to 30% to 35% of daily energy, with a saturated:monounsaturated:polyunsaturated fatty acid ratio of 1:1:1 and a cholesterol intake of less than 200 mg/d (n = 540) or usual health education (control group, n = 522). ...

###### Negative Document (Retriever: 4, Reranker: 443)

PGAP2 mutations, affecting the GPI-anchor-synthesis pathway, cause hyperphosphatasia with mental retardation syndrome. Recently, mutations in genes involved in the biosynthesis of the glycosylphosphatidylinositol (GPI) anchor have been identified in a new subclass of congenital disorders of glycosylation (CDGs) with a distinct spectrum of clinical features. To date, mutations have been identified in six genes (PIGA, PIGL, PIGM, PIGN, PIGO, and PIGV) encoding proteins in the GPI-anchor-synthesis pathway in individuals with severe neurological features, including seizures, muscular hypotonia, and intellectual disability. We developed a diagnostic gene panel for targeting all known genes encoding proteins in the GPI-anchor-synthesis pathway to screen individuals matching these features, and we detected three missense mutations in PGAP2, c.46C>T, c.380T>C, and c.479C>T, in two unrelated individuals with hyperphosphatasia with mental retardation syndrome (HPMRS). The mutations cosegregated in the investigated families. PGAP2 is involved in fatty-acid GPI-anchor remodeling, which occurs in the Golgi apparatus and is required for stable association between GPI-anchored proteins and the cell-surface membrane rafts. ...

Positive Document (Retriever: 5, Reranker: 33)

Assessment of Volume Depletion in Children with Malaria Background The degree of volume depletion in severe malaria is currently unknown, although knowledge of fluid compartment volumes can guide therapy. To assist management of severely ill children, and to test the hypothesis that volume changes in fluid compartments reflect disease severity, we measured body compartment volumes in Gabonese children with malaria. Methods and Findings Total body water volume (TBW) and extracellular water volume (ECW) were estimated in children with severe or moderate malaria and in convalescence by tracer dilution with heavy water and bromide, respectively. Intracellular water volume (ICW) was derived from these parameters. Bioelectrical impedance analysis estimates of TBW and ECW were calibrated against dilution methods, and bioelectrical impedance analysis measurements were taken daily until discharge. Sixteen children had severe and 19 moderate malaria. Severe childhood malaria was associated with depletion of TBW (mean [SD] of 37 [33] ml/kg, or 6.7% [6.0%]) relative to measurement at discharge. This is defined as mild dehydration in other conditions. ECW measurements were normal on admission in children with severe malaria and did not rise in the first few days of admission. Volumes in different compartments (TBW, ECW, and ICW) were not related to hyperlactataemia or other clinical and laboratory markers of disease severity. Moderate malaria was not associated with a depletion of TBW.

- Negative Document (Retriever: 6, Reranker: 2624)

Neurocognitive Development in Children Experiencing Intrauterine Growth Retardation and Born Small for Gestational Age: Pathological, Constitutional and Therapeutic Pathways Interest in the neurocognitive and psychosocial outcomes in children who are born small for gestational age (SGA) has increased since the recent approval of growth hormone (GH) therapy in this indication. The objective of GH treatment in SGA children is to provide a symptomatic treatment for growth retardation. From a patient perspective, the ultimate goals of GH therapy are the reduction in the present or future risk of neurocognitive, psychological, social or occupational impairment, not the accompanying improvements in growth velocity and final height per se. Therefore, from a scientific perspective, neurocognitive and psychosocial endpoints become relevant domains of assessment to determine the final treatment benefit experienced by the patient born SGA. This article reviews recent available studies on developmental risks in SGA, and then transforms the empirical findings into an integrated conceptual framework on the sources and mediators of neurocognitive and psychosocial outcomes in intrauterine growth retardation and SGA. This framework depicts two distinct therapeutic pathways by which GH therapy may improve neurocognitive and behavioural outcomes. ...

- Negative Document (Retriever: 7, Reranker: 20)

Fumarase deficiency in dichorionic diamniotic twins. Fumarase deficiency is a rare autosomal recessive inborn error of metabolism of the Krebs Tricarboxylic Acid cycle. A heavy neurological disease burden is imparted by fumarase deficiency, commonly manifesting as microcephaly, dystonia, global developmental delay, seizures, and lethality in the infantile period. Heterozygous carriers also carry an increased risk of developing hereditary leiomyomatosis and renal cell carcinoma. We describe a non-consanguineous family in whom a dichorionic diamniotic twin pregnancy resulted in twin boys with fumarase deficiency proven at the biochemical, enzymatic, and molecular levels. Their clinical phenotype included hepatic involvement. A novel mutation in the fumarate hydratase gene was identified in this family.

###### Figure 14: Top-8 results from the retriever on Scifact. The ranks assigned by the retriever and reranker shown in parens (zero-indexed). A few documents are truncated to conserve space.

text-embedding-3-large + cohere-rerank-english-v3.0 (reranker top-8)

Query Less than 10% of the gabonese children with Schimmelpenning-Feuerstein-Mims syndrome (SFM) had a plasma lactate of more than 5mmol/L.

Positive Document (Retriever: 5, Reranker: 33)

Assessment of Volume Depletion in Children with Malaria Background The degree of volume depletion in severe malaria is currently unknown, although knowledge of fluid compartment volumes can guide therapy. To assist management of severely ill children, and to test the hypothesis that volume changes in fluid compartments reflect disease severity, we measured body compartment volumes in Gabonese children with malaria. Methods and Findings Total body water volume (TBW) and extracellular water volume (ECW) were estimated in children with severe or moderate malaria and in convalescence by tracer dilution with heavy water and bromide, respectively. Intracellular water volume (ICW) was derived from these parameters. Bioelectrical impedance analysis estimates of TBW and ECW were calibrated against dilution methods, and bioelectrical impedance analysis measurements were taken daily until discharge. Sixteen children had severe and 19 moderate malaria. Severe childhood malaria was associated with depletion of TBW (mean [SD] of 37 [33] ml/kg, or 6.7% [6.0%]) relative to measurement at discharge. This is defined as mild dehydration in other conditions. ECW measurements were normal on admission in children with severe malaria and did not rise in the first few days of admission. Volumes in different compartments (TBW, ECW, and ICW) were not related to hyperlactataemia or other clinical and laboratory markers of disease severity. Moderate malaria was not associated with a depletion of TBW.

Negative Document (Retriever: 15, Reranker: 0)

Lactic acidosis in patients with diabetes treated with metformin. To the Editor: From May 1995, when metformin was introduced in the United States, through June 30, 1996, the Food and Drug Administration (FDA) received reports of lactic acidosis in 66 patients treated with metformin. In 47 patients, the diagnosis was confirmed on the basis of circulating lactate values (>5 mmol per liter), in accordance with established criteria for the diagnosis of lactic acidosis (Table 1).1,2 Of the 47 patients with confirmed diagnoses, 43 had one or more risk factors for lactic acidosis. Thirty (64 percent) had preexisting cardiac disease, of whom 18 had histories of congestive heart failure. . . .

Negative Document (Retriever: 921, Reranker: 1)

A prospective study of plasma homocyst(e)ine and risk of myocardial infarction in US physicians. OBJECTIVE To assess prospectively the risk of coronary heart disease associated with elevated plasma levels of homocyst(e)ine. DESIGN Nested case-control study using prospectively collected blood samples. SETTING Participants in the Physicians’ Health Study. PARTICIPANTS A total of 14,916 male physicians, aged 40 to 84 years, with no prior myocardial infarction (MI) or stroke provided plasma samples at baseline and were followed up for 5 years. Samples from 271 men who subsequently developed MI were analyzed for homocyst(e)ine levels together with paired controls, matched by age and smoking. MAIN OUTCOME MEASURE Acute MI or death due to coronary disease. RESULTS Levels of homocyst(e)ine were higher in cases than in controls (11.1 +/- 4.0 [SD] vs 10.5 +/- 2.8 nmol/mL; P = .03). The difference was attributable to an excess of high values among men who later had MIs. The relative risk for the highest 5% vs the bottom 90% of homocyst(e)ine levels was 3.1 (95% confidence interval, 1.4 to 6.9; P = .005). After additional adjustment for diabetes, hypertension, aspirin assignment, Quetelet’s Index, and total/high-density lipoprotein cholesterol, this relative risk was 3.4 (95% confidence interval, 1.3 to 8.8) (P = .01). Thirteen controls and 31 cases (11%) had values above the 95th percentile of the controls. CONCLUSIONS Moderately high levels of plasma homocyst(e)ine are associated with subsequent risk of MI independent of other coronary risk factors. Because high levels can often be easily treated with vitamin supplements, homocyst(e)ine may be an independent, modifiable risk factor.

- Negative Document (Retriever: 0, Reranker: 2)

Mutation of the fumarase gene in two siblings with progressive encephalopathy and fumarase deficiency. We report an inborn error of the tricarboxylic acid cycle, fumarase deficiency, in two siblings born to first cousin parents. They presented with progressive encephalopathy, dystonia, leucopenia, and neutropenia. Elevation of lactate in the cerebrospinal fluid and high fumarate excretion in the urine led us to investigate the activities of the respiratory chain and of the Krebs cycle, and to finally identify fumarase deficiency in these two children. The deficiency was profound and present in all tissues investigated, affecting the cytosolic and the mitochondrial fumarase isoenzymes to the same degree. Analysis of fumarase cDNA demonstrated that both patients were homozygous for a missense mutation, a G-955–>C transversion, predicting a Glu-319–>Gln substitution. This substitution occurred in a highly conserved region of the fumarase cDNA. Both parents exhibited half the expected fumarase activity in their lymphocytes and were found to be heterozygous for this substitution. The present study is to our knowledge the first molecular characterization of tricarboxylic acid deficiency, a rare inherited inborn error of metabolism in childhood.

Negative Document (Retriever: 4615, Reranker: 3)

The 5TMM series: a useful in vivo mouse model of human multiple myeloma. The present invention provides a combination sink and dishwashing apparatus having a sink sharing a common side wall with a cabinet which defines a closed space. The cabinet has a wire basket for holding and washing a plurality of dishes within the cabinet. The common side wall is a part of the cabinet that defines the closed space and is positioned to form one side of the sink. The common side wall can be opened to allow the wire basket to slide from within the cabinet and into the sink, for loading and unloading the dishes. Within the cabinet, the invention contains a pump powered by a motor, the pump spraying water through a rotatably mounted spray arm onto the dishes to wash them, as with traditional dishwashing machines. The invention preferably includes a garbage disposal which is also powered by the motor. In an alternative embodiment, this invention can include two cabinets as described above, each located on opposing sides of the sink.

Negative Document (Retriever: 494, Reranker: 4)

Troponin concentrations for stratification of patients with acute coronary syndromes in relation to therapeutic efficacy of tirofiban. PRISM Study Investigators. Platelet Receptor Inhibition in Ischemic Syndrome Management. BACKGROUND A major challenge for physicians is to identify patients with acute coronary syndromes who may benefit from treatment with glycoprotein-IIb/IIIa-receptor antagonists. We investigated whether troponin concentrations can be used to stratify patients for benefit from treatment with tirofiban. METHODS We enrolled 2222 patients of the Platelet Receptor Inhibition in Ischemic Syndrome Management study with coronary artery disease and who had had chest pain in the previous 24 h. All patients received aspirin and were randomly assigned treatment with tirofiban or heparin. We took baseline measurements of troponin I and troponin T. We recorded death, myocardial infarction, or recurrent ischaemia after 48 h infusion treatment and at 7 days and 30 days. FINDINGS 629 (28.3%) patients had troponin I concentrations higher than the diagnostic threshold of 1.0 microg/L and 644 (29.0%) troponin T concentrations higher than 0.1 microg/L. 30-day event rates (death, myocardial infarction) were 13.0% for troponin-I-positive patients compared with 4.9% for troponin-I-negative patients (p<0.0001), and 13.7% compared wth 3.5% for troponin T (p<0.001). At 30 days, in troponin-I-positive patients, tirofiban had lowered the risk of death (adjusted hazard ratio 0.25 [95% CI 0.09-0.68], p=0.004) and myocardial infarction (0.37 [0.16-0.84], p=0.01). This benefit was seen in medically managed patients (0.30 [0.10-0.84], p=0.004) and those undergoing revascularisation (0.37 [0.15-0.93] p=0.02) after 48 h infusion treatment. By contrast, no treatment effect was seen for troponin-I-negative patients. Similar benefits were seen for troponin-T-positive patients. INTERPRETATION Troponin I and troponin T reliably identified high-risk patients with acute coronary syndromes, managed medically and by revascularisation, who would benefit from tirofiban.

Negative Document (Retriever: 3316, Reranker: 5)

Oxidant stress promotes disease by activating CaMKII. CaMKII is activated by oxidation of methionine residues residing in the regulatory domain. Oxidized CaMKII (ox-CaMKII) is now thought to participate in cardiovascular and pulmonary diseases and cancer. This invited review summarizes current evidence for the role of ox-CaMKII in disease, considers critical knowledge gaps and suggests new areas for inquiry.

Negative Document (Retriever: 414, Reranker: 6)

Serum high-density lipoprotein cholesterol, metabolic profile, and breast cancer risk. BACKGROUND The prevalence of metabolic syndrome (obesity, glucose intolerance, low serum high-density lipoprotein cholesterol [HDL-C], high serum triglycerides, hypertension) is high and increasing in parallel with an increasing breast cancer incidence worldwide. HDL-C represents an important aspect of the syndrome, yet its role in breast cancer is still undefined. METHODS In two population-based screening surveys during 1977-1983 and 1985-1987, serum HDL-C was assayed enzymatically among 38,823 Norwegian women aged 17-54 years at entry. Height, weight, blood pressure, serum lipids, fat and energy intake, physical activity, parity, oral contraceptive use, hormone therapy use, alcohol intake, and tobacco use were also assessed. We used Cox proportional hazards modeling to estimate the relative risk (RR) of breast cancer associated with serum HDL-C levels and to adjust for potential confounding variables. We performed stratified analyses to evaluate effect modification by body mass index (BMI) and menopausal status. All statistical tests were two-sided. RESULTS During a median follow-up of 17.2 years, we identified 708 cases of invasive breast cancer. In multivariable analysis, the risk of postmenopausal breast cancer was inversely related to quartile of HDL-C (P(trend) =.02). Among women with HDL-C above 1.64 mmol/L (highest quartile) versus below 1.20 mmol/L (lowest quartile), the relative risk was 0.75 (95% confidence interval [CI] = 0.58 to 0.97). The HDL-C association was confined to women in the heavier subgroup (BMI > or =25 kg/m2), for whom the relative risk of postmenopausal breast cancer in those with HDL-C above 1.64 mmol/L versus below 1.20 mmol/L was 0.43 (95% CI = 0.28 to 0.67; P(trend)<.001; P(interaction) =.001). CONCLUSION Low HDL-C, as part of the metabolic syndrome, is associated with increased postmenopausal breast cancer risk.

- Negative Document (Retriever: 1, Reranker: 7)

Experimental and clinical studies on lactate and pyruvate as indicators of the severity of acute circulatory failure (shock). The increase in lactate (L) and pyruvate (P) content of arterial blood during experimental and clinical shock states and the extent to which such increases serve as measures of oxygen deficit and irreversible injury were investigated on an empirical basis. A standardized method for production of hemorrhagic shock in the Wistar rat was employed. During a 4-hour bleeding period, oxygen consumption of the rat was reduced to approximately 40% of control value, pH was reduced from 7.39 to 7.08, and a concurrent increase in L from 0.80 to 6.06 mm and in P from 0.07 to 0.18 mm were observed. Cumulative oxygen debt correlated with log L (r = 0.50; P < 0.0005) and both were significantly related to survival. Correlation of cumulative oxygen debt and survival, both with P and with computed values of the lactate pyruvate ratio (L/P) and excess lactate (XL), were of no higher magnitude. Partial correlation analysis demonstrated that neither the measurement of P nor the computation of L/P or XL improved predictability...

###### Figure 15: Top-8 results from the reranker on Scifact. The ranks assigned by the retriever and reranker shown in parens (zero-indexed).

