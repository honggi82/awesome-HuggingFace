[Figure 1]

JACOLBERTV2.5: OPTIMISING MULTI-VECTOR RETRIEVERS TO CREATE STATE-OF-THE-ART JAPANESE RETRIEVERS WITH CONSTRAINED RESOURCES

[Figure 2]

arXiv:2407.20750v1[cs.IR]30Jul2024

Benjamin Clavié Answer.AI bc@answer.ai

July 31, 2024

ABSTRACT

Neural InformationRetrieval has advanced rapidly in high-resourcelanguages, but progress in lowerresource ones such as Japanese has been hindered by data scarcity, among other challenges. Consequently, multilingual models have dominated Japanese retrieval, despite their computational inefﬁciencies and inability to capture linguistic nuances. While recent multi-vector monolingual models like JaColBERT have narrowed this gap, they still lag behind multilingual methods in large-scale evaluations. This work addresses the suboptimal training methods of multi-vector retrievers in lowerresource setting, focusing on Japanese. We systematically evaluate and improve key aspects of the inference and training settings of JaColBERT, and more broadly, multi-vector models. We further enhance performance through a novel checkpoint merging step, showcasing it to be an effective way of combining the beneﬁts of ﬁne-tuning with the generalization capabilities of the original checkpoint. Building on our analysis, we introduce a novel training recipe, resulting in the JaColBERTv2.5 model. JaColBERTv2.5, with only 110 million parameters and trained in under 15 hours on 4 A100 GPUs, signiﬁcantly outperforms all existing methods across all common benchmarks, reaching an average score of 0.754, signiﬁcantly above the previous best of 0.720. To support future research, we make our ﬁnal models, intermediate checkpoints and all data used publicly available.

- 1 Introduction

Text-based InformationRetrieval (IR) [29, 3] is a Natural LanguageProcessing application that enables users to retrieve relevant documents for a given query [40]. Historically, the ﬁeld has been dominated by lexical-matching methods, where retrievalis performedby directly matching query terms with documentcontent[5], with both undergoingvarious normalization steps to facilitate this process [64]. Okapi BM25 [55], the most successful lexical-matching method, remains widely used today, either independently or in conjunction with more modern approaches [62].

Despite its inability to capture semantic information, BM25 is still considered a robust baseline that can be challenging to surpass, even with signiﬁcantly more computationally intensive approaches [25]. However, in recent years, there have been large advances in the development Neural IR approaches for the English language, with numerous models now consistently strongly outperforming BM25 [43].

These improvements have been attributed to two factors: the ﬁrst is the development of much more powerful neural architectures, such as the transformers architecture [66], and the pre-trained models based on it such as BERT [13]. The second factor is the release of large-scale retrieval training datasets, with Microsoft’s MS-MARCO [45] being widely credited as one of the most important factors to the rapid development of English Neural IR [79]. Moreover, there also exists a large range of high quality, domain-speciﬁc datasets on which Neural retrievers can be further trained, providing English models with even stronger generalisation capabilities [11, 62].

Outside of the English language, and some other very high-resources languages such as Mandarin Chinese [72], progress in Neural IR has been considerably slower and has yielded more modest results. In most mid and lower-

resource languages, as is the case in Japanese, retrieval benchmarks have largely been dominated by multilingual models [69, 8], taking advantage of the vast quantities of English training data to bolster their performances on all languages.

However, these models come with multiple trade-offs. Noticeably, their retrieval performance ceiling appears to be lower than monolingual models, as, in the case of English, monolingual retrievers trained on the same amount of English consistently show stronger performances than their multilingual counterpart [68, 73]. Their relative compute costs are also considerably higher, both at training time, as they require several billion query-documentpairs, and inference time, with a similarly performing multilingual retriever having 3-to-5 times as many parameters as monolingual variants [9, 8, 10]. Finally, multilingual models have been empirically shown to miss out on cultural understanding in some settings [67], which represents an ethical issue as well as a potential performance one in certain settings.

In the case of Japanese, the performance gap has been particularly wide, with monolingual neural models often representing a 40% average performance decrease from multilingual ones, with even starker degradation on larger-scale benchmarks [9].

All existing mono-lingual Japanese retrieval models have followed the most common approach to Neural IR, which consists in the use of single-vector dense representations, where a document is represented as a single averaged vector [76]. However, multi-vector retrievers, where documents are represented by multiple vectors, one for each token it contains, have shown remarkable training efﬁciency and out-of-domain generalisation since their introduction with the ColBERT model [26], and its further reﬁnement in ColBERTv2 [59]. The promising results of ColBERT models in English and their strong generalisation ability while training on a single domain, appeared to suggest the potential suitability of this approach in lower-resources settings, such as the Japanese language.

Following these ﬁndings, a recent study has introduced Japanese ColBERT variants, the JaColBERT and JaColBERTv2 models [9], respectively following the ColBERT and ColBERTv2 training recipes. These models, while trained exclusively on lower-quality Japanese training data, have demonstrated considerable performance improvements. They have vastly outperformed existing monolingual retrievers and reached competitive results with the best multilingual ones, albeit still falling short on large scale benchmarks containing millions of documents.

In spite of this, while there has been considerable improvements in understanding how to best train both general NLP [36, 22, 12, 1, 63] and retrieval models [21, 23, 8, 33], the training of ColBERT models has followed the standard ColBERTv2 recipe with some changes, but no systematic overhaul or in-depth modiﬁcations [44, 9, 38].

- 1.1 Contributions

In this study, we present the hypothesis that it is possible to outperform all existing retrieval methods, including the best multilingual models, in Japanese with an optimized multi-vector retriever training recipe.

To achieve this aim, we systematically evaluate potential improvements via many small-scale ablation studies, seeking to increase both training efﬁciency and downstream performance.

We begin by evaluating inference-time setting, and demonstrate that dynamic query-length is strictly superior to ﬁxedlength querying.

The impact of using various teacher models for knowledge distillation is then systematically evaluated, as well as the performance changes when ensembling various teachers, as is oftentimes the best option in English [20]. We show that this does not hold true in a Japanese setting, and generate teacher scores using the best-performing model in our ablation studies, BGE-M3 [8].

Our subsequent experiments then demonstrate that conventionally used training settings for ColBERT models are either computationally inefﬁcient, place unnecessary constraint on data gathering for no performance improvement, or are simply sub-optimal in terms of retrieval performance. Based on our results, we propose an improved training setting.

We propose an additional two ﬁnal steps to the training process. The ﬁrst one is a post-training step, where we leverage data with higher-quality Japanese to ﬁne-tune our ﬁnal model checkpoint, in the hopes of improving its results on common Japanese retrieval tasks. We then introduce a ﬁnal checkpoint averaging step, where the models resulting from this post-training steps are averaged with checkpoints from the pre-training phase, to create a model that retains the performance gains on tasks which are in-domain for the post-trained model, without losing any performance on out-of-domain tasks, further increasing the generalisation potential of our model.

Our resulting model, JaColBERTv2.5, is the best performing model on all evaluation datasets, reaching an average score of 0.752, representing a 4.5% improvement over JaColBERTv2 (0.720), a 60% performance improvement on

the best-performing monolingual single-vector retriever, GLuCoSE (0.470) and a 5.32% improvement over BGE-M3 (0.714), the strongest multilingual model. These results hold true for MIRACL [81], a large-scale retrieval benchmark on which previous versions of JaColBERT trailed signiﬁcantly behind BGE-M3, but JaColBERTv2.5 reaches a 6.87% performance improvement over it.

Our model prior to the post-training step, JaColBERTv2.4, also outperforms all other existing approaches, even while being fully out-of-domain on every evaluation dataset.

We achieve these results with a constrained compute budget, where the compute used for our ﬁnal model, teacher score generation included, cannot meaningfully exceed that of JaColBERTv2, to conﬁrm that downstream performance comes from our improved recipe rather than from increased computational budget.

Moreover, we obtain these results with a simpliﬁed training recipe, which fully discards the "positive" and "negative" labels assigned to each document, focusing entirely on the relative distribution of teacher scores instead. These results can help support future research in the area of data curation and hard-negative mining, and help simplifying both processes.

We make all the resources in this paper publicly available to support further research. This includes all of our training data, with teacher scores, for both the ablation and ﬁnal training runs1, the JaColBERTv2.52 and JaColBERTv2.4 models3, as well as all intermediate model checkpoints generated during training4.

Finally, while our study focused entirely on the Japanese language, we believe that our method can be directly applicable to any other language with at least moderate data volumes and lead to similar performance improvements, as even our 320,000 triplets vastly outperformed previous monolingual approaches.

- 2 Background

In this section, we will provide a brief overview of the mechanisms used by multi-vector retrieval models such as ColBERT and JaColBERT, on which our work builds.

- 2.1 ColBERT

Multi-vector retrievers, also sometimes called late-interaction models, were introduced and popularised by the Contextualized Late Interaction over BERT, or ColBERT, model architecture [26].

The ColBERT architecture relies on a simple intuition, which is that the dominant way of creating document representation, by creating a single vector for each document, causes too much information loss. While this information loss can be mitigated when evaluated on in-domain tasks, it can result in consistently poorer out-of-domain generalisation. To solve this problem, multi-vector retrievers do not create a single large vector to represent a document, but multiple smaller ones with each individual vector representing a single token.

Query Augmentation Identifying the fact that retrieval is by nature a very asymmetrical task, with queries often being short, a query-augmentation mechanism is also introduced. Rather than padding inputs to the maximum length with padding tokens, it leverages the masked-language-modeling objective of its base models [13]. To do so, it pads all queries with [MASK] tokens, which are then attended to by the model. These mask tokens have been heavily studied in subsequent work, and appear to provide useful term-weighting and amplify semantic information, resulting in improved retrieval performance. [15, 17]

MaxSim In order to perform scoring, the ColBERT authors introduce a new scoring mechanism, which they dub MaxSim, for Maximum Similarity. In this setting, the score of a [query, document] pair is obtained via the following formula, where E represents all the embeddings representing a given document or query:

max

Scorequery,document :=

Equery

j∈[|Edocument|]

i∈[|Equery|]

∗ EdocumentT

i

j

(1)

Effectively, for a given query token, its cosine similarity with every document token is computed, and the highest similarity is kept. This process is repeated over all query tokens, and the ﬁnal [query, document] score is represented as the sum of all those maximum similarities per query token, hence the name "MaxSim".

[Figure 3]

- 1https://huggingface.co/datasets/answerdotai/MMarco-japanese-32-scored-triplets
- 2https://huggingface.co/answerdotai/JaColBERTv2.5
- 3https://huggingface.co/answerdotai/JaColBERTv2.4
- 4https://huggingface.co/collections/bclavie/jacolbertv25-checkpoints-66a37d8da6b0d4d69c14f9c3

ColBERT Initial Performance This approach appears to consistently result in better out-of-domain generalisation, even when trained on considerably smaller data volumes than competitive single-vector approaches [26, 21]. However, its performance on in-domain tasks remained lower than that of single-vector retrievers, while requiring an order of magnitude more memory and storage space to index the same document volume due to the need of storing signiﬁcantly more vectors.

- 2.2 ColBERTv2

A subsequently improved version of ColBERT, named ColBERTv2, seeks to address these two issues [59]. This second version overhauls many parts of the original process, using a more modern training recipe, albeit without a clear evaluation of the impact of each component. Most notably on the training side, it introduces both in-batch negatives [24] and knowledge distillation [19]. To help alleviate the storage issue, it introduces a novel indexing approach, allowing for a 6-fold index-size reduction by clustering the individual token representations and ofﬂoading most of the stored information to the cluster centroids before compressing the vectors to just 2 bits. This method successfully brings the storage and memory requirements of ColBERTv2 down to the same order of magnitude as that of single-vector models, though still noticeably higher, while reaching even stronger results on out-of-domain datasets. Further work has shown that this approach also addresses the issue of weaker in-domain performance, with ﬁne-tuned versions of the model being able to outperform all other approaches on multiple benchmarks [56].

ColBERT and ColBERTv2 have garnered a lot of attention, with various studies attempting to better understand and improve its various mechanisms, such as the various effects of [MASK]-based query augmentation [17, 15], the impact of introducing full-word rather than token-level representations [21], potential improvements to its scoring approach [32], or to the mechanisms around its optimised indexing approach [58, 39].

- 2.3 JaColBERT

In Japanese, both of these approaches have been reproduced, with even greater success than their English equivalents [9]. JaColBERTv1, following the training recipe of the original ColBERT model, became the then-strongest monolingual Japanese retriever. However, it fell short of the strongest multilingual models on multiple benchmarks, with the most notable performancegap being found on large-scale retrievaltasks. Subsequently, JaColBERTv2, trained following the ColBERTv2 recipe, helped address these issues. JaColBERTv2, at the time of this work, is the strongest out-of-domain retrievers on all existing Japanese benchmarks. However, on MIRACL [81], a large-scale retrieval dataset which was used to train most multilingual retrievers and on which they are therefore in-domain, it still noticeably lags in performance.

- 3 Experiments

In this section, we will present the various steps of our experimental process. As this study focuses on systematically evaluating various approaches to using and training multi-vector models, we will conduct short training runs, also called ablations, on small data scales, to identify the best possible setting. We believe this sort of small-scale training to identify optimal model settings is well-suited to helping us reﬁne optimal training for constrained resources settings, as it has proved to be a particularly strong indicator of full-sized model performance. In recent months, it has notably become the preferred way of predicting model behaviour for large language models [1, 18, 63].

In each section, we will discuss the rationale for our proposed settings and, when relevant, the results and learnings of the relevant ablations.

Firstly, we will deﬁne our hardware constraints in Section 3.1, before discussing our choice of training data is Section 3.2. We then present an overview of our the baselines our models will be evaluated against in Section 3.3 and of our chosen valuation benchmarks in Section 3.4.

In Section 3.5, we will explore different approaches to deﬁning the query length on ColBERT’s query-augmentation mechanism.

We will then evaluate the impact of various alterations of the model’s training settings in Section 3.6, through a series of small-scale training runs.

Our next area of focus in Section 3.7 is systematically studying the use of various teacher models for knowledge distillation, in the context of Japanese-language retrieval. We will explore the impact of using a variety of models as teachers, as well as various ensembling methods.

Finally, we propose two ways of improving our ﬁnal model, after the original pre-training has concluded. In Section 3.8, we describe a post-training phase, where our model will be ﬁne-tuned on smaller-scale, considerably higherquality data. In Section 3.9, we discuss the introduction of checkpoint averaging [47] as a method to ensure that the ﬁnal model remains strong across the board, and mitigating potential performance losses on out-of-domain tasks during post-training.

For all training runs, both small and large scale, we follow JaColBERTv2 in initializing our models off the original JaColBERT checkpoint, which has been trained on a total of 8 million [query, positive_document,negative_document] training triplets [9].

- 3.1 Hardware Constraints

All experimentsconductedin this study are done under a compute constraint, in order to highlight that our ﬁnal model’s performance is a consequence of our improvements, rather than due to substantially increased compute.

JaColBERTv2 was trained for 28.5 hours on 8 NVidia A100 GPUs [9], representing a total training budget of 228 A100 hours. All teacher scores used by JaColBERTv2 were re-used from the original ColBERTv2 teacher scores [59], and therefore came at no additional compute cost.

Based on this, we constrain our total training compute budget to the same 228 hours, plus or minus 5%, resulting in a ﬁnal upper bound budget of 239 A100 hours. We include all time spent training each ablation model, generating teacher scores and training the ﬁnal JaColBERTv2.5 model under this budget.

- 3.2 Training Data

For both the ﬁnal model training and ablation runs, we follow existing practices [9, 69] and train our model using the Japanese split of MMarco [6]. MMarco is a machine-translated of MS Marco [45], a large English Information Retrieval (IR) dataset which has widely been credited as unlocking vast advances in English IR, thanks in large part to its scale and wide variety of queries. For a long time, no equivalent dataset existed in non-English languages, and efforts to create ones were largely unsuccessful due to the cost of such an endeavour. MMarco was introduced to bridge the gap between English and other languages, by providing a fully machine-translated version of MS Marco in 14 languages, and empirically showcasing that, while the resulting dataset produced poorer results than in English, it still contained useful signal on a scale usually not available in these languages.

Retrievals models are generally trained on triplets. These triplets can either be be standard triplets, as with ColBERTv1 [26] and JaColBERTv1 described above, where each triplet contains a single query, a single positive document, and a single negative document, or n-way triplets. n can be any number, and represents the total number of documents passed to the model: in the case of 16-way triplets, the model would be presented with a query, and 16 documents, rather than just 2 in the standard setting. Doing so allows us to more efﬁciently use knowledge distillation [19], where the model learns from teacher scores, generally generated by strong cross-encoder models, and attempts to emulate them or their distribution [50].

Our models are trained using 32-way triplets with knowledge distillation. This means that, for every single query, the model is given 32 documents per triplet, as well as teacher scores for every [query, document] pair. The goal of the model’s training is to attempt to learn to reproduce the provided scores, through a knowledge distillation loss function which we explore further in Section 3.6.4.

We use a downsample of the set of triplets used to train the original English ColBERTv2 model [59]. We downsample the triplets in two ways: ﬁrstly, in order to meet the compute constraints of this work, we randomly sample 3,200,000 triplets out of the 19,000,000 originally provided. Secondly, as ColBERTv2 was trained with 64-way triplets, we randomly sample 31 negative documents from the original 63 in every individual triplet. We choose to train on 3,200,000 triplets, which represents 40% of the 8,000,000 triplets JaColBERTv2 was trained on, in order to respect our compute constraints and allocate sufﬁcient compute to generating teacher scores.

As MMarco is a direct translation of MS Marco, it is possible to reuse the ColBERTv2 triplets with no further modiﬁcations. We use the teacher scores provided by the ColBERTv2 authors as our baseline teacher scores for all of our experiments, and will extensively cover the effect of different teachers in Section 3.7.

In recent years, higher quality datasets such as Mr.TyDi [79] and its improved version, MIRACL [81], have been introduced. However, both of them contain noticeably fewer queries and relevance labels per language than MMarco, and are more commonly used to train multi-lingual models [8, 69], while best-performing monolingual retrievers largely continue to pre-train on MMarco [37]. However, we believe that MIRACL could constitute a particularly suitable post-training dataset, which we will further explore in Section 3.8.

Ablation Training Data For all ablation training runs we conduct as part of our experiment, we use a further downsampled version of our ﬁnal training set. We create this set by sampling the ﬁrst 10% triplets of the full training set, resulting in 320,000 training triplets, which represents 4% of the original JaColBERTv2 training data. Following previous work [1, 18], we believe that this data volume is sufﬁcient to show trends which will scale to the ﬁnal training run.

- 3.3 Baselines

Our ﬁnal models are evaluated against a large range of representative baselines, including the current best-performing retrievers. To do so, we evaluate our model against BGE-M3 [8], the current best-performing multilingual embedding model. BGE-M3 is a multi-output model: it is capable of producing single-vector dense representations, but is also able to output sparse and computationally heavy multi-vector representations to act as a "self-reranker". As a result, we report results from BGE-M3 in two settings: dense, using only its single-vector retriever output, and all, leveraging all three forms of outputs in the way recommended by its original authors. BGE-M3’s model size is roughly 5.11x that of JaColBERT.

Results for the multilingual-E5 (mE5) family of models [69] are also presented in all three existing model sizes, small (JaColBERT-sized),˜ base(2.5x˜ JaColBERT) and large (5x˜ JaColBERT). The mE5 family is one of the most widely used model family for Japanese retrieval, and has consistently shown strong results on benchmarks [9].

We also report results for the best-performing single-vector retrievers in Japanese, GLuCoSE5, an embedding model based on LUKE [74], as well as Nagoya University’s SUP-SIMCSE family of models [65], in both base and large sizes.

Finally, we also report results for JaColBERTv1 and JaColBERTv2, the two previous best multi-vector retriever models for Japanese, respectively trained following the ColBERTv1 [26] and ColBERTv2 [59] training recipes.

- 3.4 Evaluation Data

We deﬁne two evaluation sets: one used for ﬁnal evaluation, described in Section 3.4.1 and a smaller, quicker-to-run one, that will be used for the various experiments we are planning to run to ﬁnd the optimal training setting presented in Section 3.4.2. All metric calculations are performed using the ranx evaluation toolkit [4].

- 3.4.1 Final Evaluation Data

For the ﬁnal evaluation, ﬁve commonly used evaluation datasets will be used to cover the model’s performance in a variety of settings. For each dataset, we choose a main evaluation metric in line with previous work in order to provide clear comparisons. However, detailed evaluation results for our models will also be reported.

Type # Queries # Documents Task Setting MIRACL General domain QA 860 6,953,614 Large-Scale Retrieval JSQuAD General domain QA 4420 1145 Small-Scale Retrieval JQaRA Trivia QA 1667 100 Reranking JaCWIR Synthetic Web QA 5000 100 Reranking ESCI Amazon Product Search 4206 149,999 Large-Scale Retrieval Table 1: Brief overview of the key information about the datasets used for evaluation. We consider as "Large-Scale Retrieval" any task where more than 100,000 documents are considered at once. For Reranking tasks, the number of documents is the number of documents to rerank.

[Figure 4]

[Figure 5]

JSQuAD is a QA dataset introduced in the JGLUE evaluation set [30], inspired by the English SQuAD [52]. We evaluate JSQuAD in the same setting as in previous studies [9], using Nouu.me’s evaluation benchmark, where the dataset is reduced to 1600 passages, and the model’s goal is to extract the relevant passage for each query in its top

[Figure 6]

5https://huggingface.co/pkshatech/GLuCoSE-base-ja

results. The metric we report for this dataset, following previous work, is Recall3. This task is the easiest amongst the evaluation set.

MIRACL [81] is a large-scale multilingual evaluation benchmarks. We use its Japanese subsplit, which is composed of over 6 million documents, extracted from Wikipedia, and contains human-created relevance judgements for 860 queries over this corpus. We choose to use exclusively MIRACL, rather than both MIRACL and Mr.TyDi [80], another large-scale multilingual information retrieval dataset, as MIRACL is a reﬁnement of Mr.TyDi, with additional judgements added and dubious labels removed. The main metric reported for this dataset is NDCG@10. It has been noted in the past that MIRACL contains "holes": that is, the positive judgements are not thorough, and the data contains many false negatives6. However, it remains the only large-scale evaluation benchmark for most languages it covers, including Japanese, and is one of the most commonly used non-English IR benchmark [8, 69, 9, 38].

JQaRA [61] is another dataset built from a QA dataset commonly used for Japanese QA evaluation, JAQKET [83]. The aim is, similarly to SQuAD, to ﬁnd a document containing the answer to a given query, over 1667 queries. The dataset was constructed via a mix of LLM usage, before going through human validation to ensure all negatives are negatives and all questions have at least one real positive passage. The task is presented as a hard reranking task: for each query, we are provided with 100 documents, with one or more of them containing the information necessary to answer the query and all other documents containing very adjacent information which does not directly address the query. These adjacent documents are called "hard negatives", as they’re purposefully designed to be hard to differentiate from positive examples. The main evaluation for this task is NDCG@10. All evaluations on JQaRa are conduced with the ofﬁcial evaluation code provided by the dataset author [61].

JaCWIR [60] is a medium-scale (500,000 documents) retrieval dataset, using a large variety of web-scrapped documents. It is an entirely auto-generated dataset, where GPT-3.5 [7] was asked to produce queries for which a given document would be relevant. We also use it as a reranking task, similarly to how it was introduced. For each of its 5,000 queries, the model must attempt to identify the relevant document among 99 hard negatives. The main evaluation metric for this task is NDCG@10. All evaluations on JaCWIR are conduced with the ofﬁcial evaluation code provided by the dataset author [60].

ESCI [53] is an addition to the JaColBERT evaluation set, which was not used in previous work. It is a "Shopping Queries Data Set" dataset provided by Amazon Science as part of the KDD Cup 2022. The goal of this dataset is to evaluate a model’s ability to match very short (1 to 5 tokens) queries with the textual description of relevant products. We use ESCI as a retrieval task, similarly to one of the settings it is available in in the Japanese Massive Text Embeddings Benchmark (JMTEB)7, an ongoing effort inspired by the English MTEB [43]. For any given product query, the model must attempt to retrieve the description of relevant products among 149,999 product descriptions. The main evaluation metric for this task is NDCG@10.

We provide an overviewof key information on each dataset in table 1. Despite the relative sparsity of Japanese retrieval benchmarks in comparison with English, we believe that these ﬁve datasets in these settings provide a good overview of retrieval models’ capabilities on a wide array of real-world relevant usages.

- 3.4.2 Development Evaluations

A large part of our study focuses on systematically evaluating a large variety of improvements to the ColBERT training and inference routine. As a result, we need a representative development set that is computationally inexpensive to run, while providing us with enough information to make decisions. We decide to use two evaluation sets, and report two key metrics for each of them. The ﬁrst one is JQaRA, as presented above. We choose JQaRA due to its small size, being presented as a reranking task, while it has consistently shown a good ability to discriminate between models and a good correlation to performance on other datasets [61]. We report both NDCG@10 and MRR@10 as our development metrics.

As our second task, we follow ongoing efforts in creating lighter embeddings benchmarks8 and introduce a smaller version of MIRACL’s Japanese split, which we dub MIRACL-small-ja. This dataset is built through hard-negative mining. Using BM25, we retrieve the top 250 results for each of the 860 MIRACL development queries. We then enrich this data with all positive examples, if they were not present in the BM25 results. The resulting dataset contains 197,610 documents and 860 queries.

[Figure 7]

- 6While there is no formal citation for this claim, it can deduced from the annotation process used for

MIRACL and has been frequently noted, notably as part of Cohere’s work on multilingual embeddings: https://huggingface.co/datasets/Cohere/miracl-en-queries-22-12

- 7https://huggingface.co/datasets/sbintuitions/JMTEB
- 8The team behind the English MTEB is currently developing a multilingual version of it, with a lite variant, including datasets

similar to MIRACL-small, currently under discussion. https://github.com/embeddings-benchmark/mteb/issues/784

While the ideal situation would be for the ﬁnal evaluation and development sets to be fully separate, this is not possible with the current amountof evaluation resources available. We pick these two datasets as we believe they provide a good balance between useful signal and minimising the risks of overﬁtting too strongly on these development evaluations, which would introduce a bias to our ﬁnal results.

- 3.5 Dynamic Query Length

ColBERT models use a query augmentation mechanism, leveraging the use of [MASK] tokens, which are appended to every query, replacing traditional padding [26], until the predeﬁned maximum query length is reached. These tokens have been shown to learn different types of information, occasionally acting as term-importance weights [15, 17].

However, the exact impact of mask tokens and the best way to use them has been understudied. Instead of variablelength augmentation, a past study has explored simply appending 8 [MASK] tokens to every query, regardless of the actual query length [20]. Another previous study has chosen to remove this augmentation mechanism entirely [21]. However, no study has compared the effects of these various choices against the original implementation.

We believe all three of these approaches to be suboptimal, and propose dynamic query length as a replacement. Dynamic query length effectively aims to improve the initial approach of padding each query with [MASK] tokens until the query maximum length by allowing it to more easily adapt to longer queries, while also borrowing from ﬁxed-length augmentation for edge cases.

Effectively, our approach is to set the maximum query length to the nearest higher multiple of 32 (ColBERT’s original maximum query length) before performing the [MASK]-padding. Additionally, if fewer than 8 augmentation tokens would be appended with the new query length, we ensure that at least 8 tokens are appended, overriding the maximum length.

To select the default query augmentation mechanism to use for JaColBERTv2, we evaluate all four of the discussed approaches.

- 3.5.1 Results

JQaRA MIRACL-small-ja Average NDCG@10 MRR@10 NDCG@10 Recall@5 NDCG@10 Overall

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Baseline 0.578 0.820 0.681 0.707 0.619 0.691 No Augmentation 0.557 0.817 0.632 0.661 0.595 0.667 Fixed 8 tokens aug. 0.577 0.813 0.670 0.692 0.624 0.688 Dynamic query length 0.581 0.820 0.681 0.707 0.631 0.700

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Table 2: Results of various query augmentation methods on our development set.

Results for this experiment are presented in Table 2. The results are pretty clear. On all datasets, disabling [MASK] augmentation is consistently largely outperformed by all query augmentation approaches. Fixed 8-token query augmentation, while performing considerably better than no augmentation, is similarly outperformed by both ﬁxed and dynamic query lengths on every dataset.

On JQaRA, where the query length ﬂuctuates more and some queries are considerably longer than others, dynamic query length outperforms a ﬂat, higher token limit, suggesting that appending too many [MASK] tokens can produce a slightly detrimental effect. An empirical analysis of the results obtained on JQaRA also reveals that dynamic query length has virtually no impact on queries that are noticeably shorter than the maximum query length in a ﬁxed query length setting. However, it noticeably improves NDCG@10 on queries which are nearer the maximum length, and would thus lose most of the query augmentation mechanism. This explains the small increase in overall NDCG@10 on the full dataset.

On MIRACL, where all queries have similar token counts and are all well under 32 tokens, the results for ﬁxed and dynamic query lengths are identical, as would be expected.

Based on the results of this experiment, every result we report on subsequent ablation, as well as the ﬁnal model results, will be using dynamic query length.

- 3.6 Training Setting

In this section, we will evaluate the impact of certain changes to common components of the retrieval model training pipeline. We will ﬁrst explore concerns around the impact of using in-batch negatives in Section 3.6.1. We next explore the optimal way of scheduling the model’s training, and the relevance of the recently recently schedule-free training method [12] in Section 3.6.2. We will then study the beneﬁts of score normalization, applied to both teacher and student models, in Section 3.6.3, as well as explore the use and impact of different commonly used knowledge distillation loss functions in Section 3.6.4.

Finally, we will present the results of all of our experimental small-scale runs and discuss their implications in Section 3.6.5.

- 3.6.1 In-Batch Negatives

In-batch negatives (IBNeg) are frequently used as a way to augment the training of retrieval models [24]. Effectively, within a given training batch, the IBNeg approach treats every other query’s positive documents as additional negative examples in a binary relevance classiﬁcation exercise. The query’s original positive example is treated as a positive label, and every other query’s positive example becomes a negative example. The model is asked to predict relevance over those newly created [query, document] pairs, and a cross-entropy loss is calculated over this prediction and then added to the model’s main training loss.

This method has shown modest but consistent across studies performance improvements when used with single-vector retriever models [24, 68]. This method is added to the ColBERT training recipe in the paper introducing ColBERTv2, following the promising results obtained by other models [59]. This choice was not thoroughly evaluated, and its impact is therefore unknown.

However, we empirically note that the resulting cross-entropy loss values are two orders of magnitude lower than the ones from the main KL-Divergence loss used in typical ColBERTv2 training. Moreover, we hypothesize that inbatch negatives are an unnecessary signal for training multi-vector models using 32-way triplets for multiple reasons, especially in non-English, data-constrained settings. Firstly, the information obtained from distilling the ranking distribution of a strong cross-encoder model over 32 documents should carry a much stronger signal than binary relevance labelling between a positive document and randomly sampled negatives. Secondly, this loss relies on the positive examples consistently being well-annotated and true positives, which is not guaranteed to be the case with lossy annotation processes, or even the partially-automated positive selection used in ColBERTv2 [59].

To conﬁrm the validity of our hypothesis, we will train two separate ablation models on the same data.

- 3.6.2 Scheduling

The learning rate scheduler used for the training of neural networks has been shown to have a potentially large impact on the performances of the resulting model [78]. There are are a few common schedulers yielding strong results, such as WSD (Warmup-Stable-Decay) [22], which increases the learning rate steadily before plateauing for the majority of training and entering a decay phase, which should ideally be performed on higher quality data, or Linear-Decay, where the model’s learning rate increases during a ﬁxed number of warmup steps before linearly lowering until the ﬁnal training step, among others. The original ColBERTv2, as well as JaColBERTv2 [9], used linear decay scheduling.

However, while tuned learning-rate scheduling consistently outperforms non-scheduled approached, it is not without constraints. Notably, for the best performance, it requires knowing the total number of steps in advance in all cases, or has constraints such having a higher quality data mix for schedulers relying on them for their annealing phase. Moreover, an optimal schedule for a large number of steps is not guaranteed to work as well for lower data quantities. These constraints are especially noticeable for retrieval models, which are expected to be put to use on a wide variety of downstream uses with varying data distributions, and therefore beneﬁt immensely from being able to easily resume training without huge performance impact.

Recently, schedule-free learning has been proposed [12]. This new approach, while not yet thoroughly tested across all domains, has empirically shown very encouraging results on a large number of benchmarks. In practice, it introduces additional calculations as part of the optimizer steps, allowing it to vary the learning rate without the need for a ﬁxed, pre-deﬁned schedule. This considerably simpliﬁes both the pre-training and ﬁne-tuning processes, as there is no need to optimise the scheduler used for training and similar parameters can be re-used for different data scales.

Schedule-free learning has been noted as potentially requiring a higher learning rate than scheduled learning, with optimal values empirically falling in the range of 1 to 10 times the original learning rate [12]. We thus conduct an experiment comparing the training setting used for JaColBERTv2 and its ablations and schedule-free learning, with

learning rates set to 1x, 3x and 5x the original 0.0001 learning rate. For all experiments, we retain the AdamW [36] optimizer used in previous work.

- 3.6.3 Score Normalisation

In the current ColBERTV2 training recipe, scores are unnormalised: the raw logits outputted by the teacher crossencoder are used as teacher scores, and the output of the maxsim scoring function is used as the student model’s score. These scores are on different scales, as the theoretical range for maxsim score is [0, Number of Query Tokens], while the range for cross-encoder logits include negative numbers and is on a scale dependant on the original model’s training, which varies teacher by teacher.

While some distillation losses can be seen as being partially robust to scale differences, which partially justiﬁed the use of KL-Divergence in the ColBERTv2 paper [59], we believe the lack of normalisation to be suboptimal for two main reasons. The ﬁrst one is that, given the large difference in scale, the loss calculation is likely to lead to a better approximation of information loss if operating on a similar scale. The second is that normalised scores allow the models to focus purely on the relative ranking of results, rather than absolute scores, the latter of which may provide less useful information due to the automated nature of triplet generations.

We experiment with two used normalization approaches: one where only the teacher scores are normalized [31], and one where both the teacher and student scores are normalized. In all cases, we use min-max normalization, deﬁned as:

scorenormalized =

score − scoreleast_relevant scoremost_relevant − scoreleast_relevant

[Figure 22]

(2)

Effectively, this gives a score of 1 to the most relevant document identiﬁed by the teacher and 0 to the least relevant one, regardless of their absolute score. Every other score is then placed on this scale depending on their distance to those two scores.

- 3.6.4 Loss functions

For knowledge distillation in retrieval models, two loss functions are commonly used [14, 54]: MarginMSE and KL-Divergence, the latter of which is the one used by ColBERTv2.

MarginMSE [20] consists in computing the Mean-Squared Error on the difference in the margin between the predictions of the model being trained and the teacher model. The margin deﬁned is the difference between the score the model gives to the positive document and the score it gives to negative documents. In the case of n-way training, this margin is calculated over every [positive_document_score, negative_document_n_score]. MarginMSE is thus computed as follows (where N is the batch size):

1 N

MarginMSE =

[Figure 23]

N

max(0,margin − (scoreteacher(xi) − scorestudent(xi)))2 (3)

i=1

Effectively, the training objective becomes for the student model’s margin to reproduce the teacher’s margin as closely as possible.

KL-Divergence [27], on the other hand, seeks to directly minimise the difference between the distributions of scores of the model being trained and its teacher. KL-Divergence loss is computed as follows (where N is the batch size):

1 N

KL-Div =

N

Pteacher(score|xi)log

[Figure 24]

i=1 score

Pteacher(score|xi) Pstudent(score|xi)

[Figure 25]

(4)

In effect, it computes an estimation of how much information appears to be lost between the teacher model’s distribution and the student model’s one, and minimising this loss becomes the primary training objective.

The use of either MarginMSE or KL-Divergence has been reported for knowledge distillation into various retrieval models. While both loss functions have been shown to be strictly superior to more traditional MSE-based losses [23, 20], there has been little head-to-head comparison of the two. Recently, the SPLADE-V3 authors anecdotally reported noticing overall similar performances between the losses, with MarginMSE favouring recall and KL-Divergence precision, and opted for a mixed-loss approach for their ﬁnal model, using a conjunction of both with a lower weight attributed to MarginMSE [31]. However, SPLADE-V3 was trained on only 8-way triplets, considerably fewer than our 32-way approach.

Our hypothesis is that, given our training setting with numerous negatives and its reduced reliance on having a strong positive, as only the distribution of scores matter, KL-Divergence remains the optimal choice to train ColBERT models with knowledge distillation. In order to test this hypothesis, we will compare its results with MarginMSE, as well as well as mixes of both with various MarginMSE weighting: λ = {0.2,0.1,0.05}.

- 3.6.5 Ablation Results

We present the results of all the ablation runs related to experiments detailed in the previous subsections in Table 3. As we run our experiments sequentially, the results are presented in the order of experiments, and the baseline for each new category is the best performing approach of the previous one.

JQaRA MIRACL-small-ja Average

[Figure 26]

[Figure 27]

[Figure 28]

NDCG@10 MRR@10 NDCG@10 Recall@5 NDCG@10 Overall In-Batch Negs. (IBNeg)

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

With IBNeg (baseline) 0.581 0820 0.681 0.707 0.631 0.697 Without IBNeg 0.580 0.820 0.682 0.713 0.631 0.699 Scheduling

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Linear Decay (baseline) 0.581 0.820 0.681 0.713 0.631 0.699 Schedule-free (1x LR, 1e-05) 0.576 0.820 0.669 0.707 0.623 0.693 Schedule-free (3x LR, 3e-05) 0.581 0.821 0.683 0.717 0.632 0.701 Schedule-Free (5x LR, 5-05) 0.575 0.815 0.681 0.709 0.628 0.695 Normalization

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

No norm (baseline) 0.581 0.820 0.681 0.713 0.631 0.699 Normalize teacher scores 0.565 0.802 0.680 0.717 0.623 0.691 Normalize student & teacher 0.585 0.827 0.691 0.716 0.638 0.705 Loss Function

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

KL-Divergence (baseline) 0.585 0.827 0.691 0.716 0.638 0.705 MarginMSE 0.583 0.827 0.672 0.699 0.628 0.695 Mixed (KL-Div λ = 1.0)

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

+ MarginMSE λ = 0.2 0.576 0.813 0.688 0.716 0.632 0.698 + MarginMSE λ = 0.1 0.582 0.816 0.687 0.714 0.635 0.700 + MarginMSE λ = 0.05 0.578 0.812 0.691 0.691 0.635 0.693 Table 3: Results of all training settings ablation results, compared to the relevant baseline

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

In-batch negatives Our results on in-batch negatives conﬁrm our hypothesis: they do not appear to provide useful training to the model, even resulting in slightly decreased performanceon MIRACL-small-ja. We believe this to be due to the factors we have highlighted with the main one being that the signal from distilling a teacher’s score distribution over 32 documents appears to constitute a strong enough learning signal on its own. Additionally, some of our positive examples may be false positives, and many of our negative examples are false negatives. In-batch negatives are costly to compute, as they require an additional scoring stage for every query against all the other queries’ positive documents, and materialising and computing a cross-entropy matrix. Moreover, they place additional constraints on training as we need to ensure higher data quality to fully leverage them. Removing the use of in-batch negatives thus represents an efﬁciency gain by lowering the compute and memory use of training at no performance cost. Following the results of this experiment, we remove the use of in-batch negatives from our training recipe.

Scheduling The results of our experimentson learning-rateschedulers highlight schedule-free learning [12] as a strong alternative to the Linear-Decay scheduler used in standard ColBERTv2 training. With a slight tweak to the learning rate, increasing it from the best-performing linear decay rate of 1e − 05 to 3e − 05, schedule-free learning results in a slight performance increase across the board. Additionally, schedule-free learning drastically reduces the level of optimisations required to operate at different data scales, and removes any costs associated with continuously stopping and restarting our model’s training to expose it to different data distributions when attempting to use it in a different domain. We choose to use schedule-free learning as part of our training. This leads to us disabling the use of gradient clipping [71], which has been observed to cause schedule-free learning runs to fail to converge [12].

Normalization Normalizing both teacher and student scores results in a sizeable performance increase on all datasets, while normalizing only teacher scores yield a consistent performance decrease. This is in line with our expectations. Moreover, normalized teacher scores, which appear to only function well when used in conjunction with normalized student scores, are a prerequisite to being able to best utilise an ensemble of teachers outputting scores on different

scales. Based on these clear results and this clear constraint, we introduce teacher and student score normalization to our training method.

Loss Functions Unlike our other experiments, our empirical results demonstrate that the currently most commonly used loss function for knowledge distillation in ColBERT models, KL-Divergence, is the best performing option. MarginMSE exhibits reduced performance on all datasets, with it being noticeably more pronounced on MIRACLsmall-ja. Interestingly, combining MarginMSE and KL-Divergence consistently results in worse results than using either loss on its own. We hypothesize that the varying quality of positive examples in our dataset could be a partial explanation for the substandard performance of MarginMSE, as it is calculated based on the margin between the positive example and negative examples, rather than strictly focusing on the score distribution in the same way as KLDivergence. As it performs strictly worse than KL-Divergence and additionally frees us for data quality constraints,we choose to discard MarginMSE and retain the KL-Divergence loss function for our model training.

- 3.7 Teacher Models

Another important part of knowledge distillation is the choice of teacher model. In Information Retrieval, teacher scores are most often obtained from the logits of cross-encoder reranker models, which assign a relevance score to query-document pairs [34]. Cross-encoders are powerful retrieval models, as they are aware of both the query and the document at scoring time, whereas most other retrieval methods encode documents and queries separately [42]. However, this means that they are particularly costly, as the model must run a full forward pass on every single pair in order to be able to output a score. As a result, they’re unsuitable as a single retriever for medium-to-large scale document collections, but are particularly powerful as teachers for distillation.

The original ColBERTv2 paper used scores from a lightweight 22 million parameter MiniLM [70]-based model, itself distilled from a larger, more powerful cross-encoder[62]. This choice was partially motivated due to the computational requirements of generating teacher scores for the entirety of the ColBERTv2 training set. As it comprises of over 19 million triplets, each made up of 64 individual query-document pairs, the score generation process for ColBERTv2 required computing cross-encoder scores for a total of 1.2 billion pairs.

However, larger cross-encoders generally result in better performance [62, 46], and successful distilled models in the general domains have shown that stronger teachers yield stronger distilled models [70, 57]. Speciﬁcally, anecdotal results have shown that using T5-3B based rerankers such as MonoT5-3B [46], based on a 3 billion parameters sequence-to-sequence model [51], consistently resulted in noticeably stronger distilled multi-vector models than distillation from smaller rerankers [75].

- 3.7.1 Single-Teacher

Models To investigate the performance of various rerankers as teachers on Japanese retrieval, we generate teacher scores using a wide variety of models. As for monolingual Japanese models, we use the JP-SMALL9 reranker, based on Multilingual-MiniLM [70], the JP-BASE10 and JP-LARGE11 rankers, based on Nagoya University’s SIMCSE models [16]. We also generate scores using the BGE-M3-reranker [8] (M3) model, the highest performing multi-lingual rerankers, as well as BGE-M3-jp12 (M3-jp), a version of it further ﬁne-tuned on small-scale Japanese datasets. We selected these models among the available Japanese rerankers for their strong results on existing benchmarks [82]. Finally, leveraging the fact that MMarco is a translation of MS Marco, we report the baseline performance of using the original ColBERTv2 triplets (original), re-used in JaColBERTv2. Finally, we evaluate using the scores of the MonoT5-3B model mentioned above, also generated on the English version of the dataset.

[Figure 70]

- 9https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-small-v1
- 10https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-base-v1
- 11https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-large-v1
- 12https://huggingface.co/hotchpotch/japanese-bge-reranker-v2-m3-v1

[Figure 71]

[Figure 72]

[Figure 73]

NDCG@10 MRR@10 NDCG@10 Recall@5 NDCG@10 Overall Baseline 0.585 0.827 0.691 0.716 0.638 0.705 Single Teacher

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

jp-small 0.567 0.807 0.703 0.737 0.635 0.704 jp-base 0.569 0.81 0.713 0.747 0.641 0.701 jp-large 0.577 0.816 0.713 0.741 0.645 0.712 M3 0.589 0.836 0.740 0.788 0.665 0.738 M3-jp 0.588 0.838 0.728 0.757 0.658 0.728 MonoT5-3B 0.587 0.835 0.594 0.642 0.591 0.665 Table 4: Results of ablation runs using various models as distillation teachers

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Results The results of models trained using various models to generate teacher scores are presented in Table 4. The impact of using different teachers is immediately clear. Interestingly, the jp-* models, in small, base and large sizes, all yield stronger results on MIRACL-small-ja than the original teacher’s scores. However, their use as teachers for our models appear to result in a signiﬁcant performance decrease on JQaRA compared the original scores. On the other hand, the BGE-Reranker-M3 models prove to be very strong teachers. Interestingly, the multi-lingual version of M3, which has not been ﬁne-tuned on Japanese data, vastly outperforms its ﬁne-tuned version on MIRACL-small-ja, and roughly matches its performance on JQaRA, resulting in a noticeably stronger average performance. Ultimately, BGE-Reranker-M3, in its non-ﬁnetuned multilingual version, appears to be strongest available teacher, reaching an overall score of 0.738, compared to the noticeably lower 0.705 score of the original training scores.

Finally, it is also worth noting that the MonoT5-3B reranker yields strong results on JQaRA, but leads to a sizeable performance drop on MIRACL-small-ja. This behaviour indicates a potentially interesting trend: models generating scores on English-language MSMarco rather than MMarco, as is the case for the original scores and MonoT5-3B, appears to result in strong performance on JQaRA and weaker results on MIRACL-small-ja, while the opposite holds true for Japanese-language models.13.

3.7.2 Ensembled Teachers

Ensembling Teachers Additionally, recent work on English Information Retrieval has increasingly highlighted that ensembling the scores of multiple teachers produce better distilled models [31], even when the ensembled teachers’ individual performance are largely similar [20]. The most common way of ensembling multiple teachers’ scores is via a two-step process, where the scores are ﬁrst normalized using min-max normalization as in Section 3.6.3, before being averaged. We believe that these results may not reproduce in a Japanese setting, as there exists far fewer Japanese base models and strong rerankers than in high-resource languages settings. However, we evaluate a large range of teacher ensembling as part of our study, using the teacher models described above.

[Figure 92]

13We do not explore this phenomenon further as generating scores with a 3 billion parameter model is particularly costly, and this analysis is outside the scope of this study.

[Figure 93]

[Figure 94]

[Figure 95]

NDCG@10 MRR@10 NDCG@10 Recall@5 NDCG@10 Overall

[Figure 96]

Baseline (orig.) 0.585 0.827 0.691 0.716 0.638 0.705 Best single teacher (M3) 0.589 0.836 0.740 0.788 0.665 0.738 Ensembled Teachers

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

M3 + M3-jp 0.587 0.835 0.733 0.764 0.660 0.730 M3 + M3-jp + orig. 0.585 0.832 0.742 0.778 0.664 0.734 M3 + M3-jp + large 0.587 0.828 0.739 0.766 0.663 0.730 M3 + M3-jp + large + base 0.589 0.834 0.738 0.763 0.664 0.731 M3 + MT5 0.598 0.837 0.702 0.745 0.650 0.721 M3-jp + MT5 0.596 0.832 0.694 0.734 0.645 0.714 M3 + M3-jp + MT5 0.598 0.832 0.743 0.769 0.671 0.736 M3 + orig. + MT5 0.597 0.837 0.713 0.747 0.655 0.724 M3-jp + orig. + MT5 0.596 0.835 0.707 0.740 0.652 0.720 M3 + M3-jp + orig. + MT5 0.596 0.835 0.727 0.760 0.662 0.730 All rerankers 0.589 0.841 0.711 0.759 0.65 0.725 Table 5: Results of ablation runs using various ensembles of models as distillation teachers. Best overall results are reported in bold, and best results within the ensembled category in italic. "orig." refers to the original training set, "MT5" to to MonoT5, "large" to jp-large and "base" to jp-base.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Results The results of various ensembling combinations are presented in Table 5. Overall, no ensembling combination outperforms simply using BGE-Reranker-M3 as a single teacher on all metrics, which conﬁrms our initial impression that its individual performance outweighs any gains from ensembling with weaker models.

It is worth noting that some ensembling combinations, such as BGE-Reranker-M3+M3-jp+MonoT5-3Breach promising results, with higher NDCG@10 results on both development sets. However, generating teacher scores is costly, particularly with the very large MonoT5-3B model, and the weaker performance of the ensembled teachers on nonNDCG metrics does not appear to justify this computational cost. We thus choose to retain BGE-Reranker-M3 as our single-model teacher for the full training run.

- 3.8 Post-Training

As discussed in Section 3.2, our models will be trained on MMarco, a dataset which was machine translated from English to Japanese in 2019, and therefore often contains lower quality Japanese or odd sentence constructions.

While we believe that this issue is unlikely to have a large impact on our ﬁnal model, we propose post-training (also called ﬁne-tuning) the model on smaller, higher quality datasets. To do so, we choose to use the previously discussed datasets MIRACL [81] and JQaRA [61], as well as JaGovFaqs, a subcomponent of JMTEB containing questions and answers from the Japanese Government’s FAQ sections.

Triplets Proportion (w/o MMarco) Proportion (with MMarco) MIRACL 115,365 64.01% 60.72% JQaRA 35,733 19.93% 14.17% JaGovFaqs 28,902 16.06% 15.21% Total without MMarco 180,000 100% 90.9% MMarco 18,000 10% (for reference) 9.9% Total with MMarco 198,000 110% (for reference) 100%

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Table 6: Overview of the datasets used in the post-training data mixes.

This post-training phase has two aims: the ﬁrst is to improve our model’s ability to understand Japanese in more diverse settings than the ones observed during pre-training. Our second aim is to highlight the potential gains that can be obtained with relatively small scale post-training on domain speciﬁc data, which should be reﬂected in potential performance improvements on JQaRA and MIRACL.

We present the full make-up of the post-training data in 6. We experiment with two post-training settings. The ﬁrst setting only contains the three datasets listed above. The second one additionally includes 10% data randomly sampled from the MMarco triplets used for pre-training, to address the issue of catastrophic forgetting, where the model forgets previous training when exposed to exclusively new data [28]

- 3.9 Checkpoint Averaging

Finally, we present the hypothesis that checkpoint averaging can improve our model’s generalisation ability. Checkpoint averaging, also called "model merging", consists in taking multiplied different checkpoints of similarly sized models and averaging their parameters at each layer to create a merged model.

This practice has a long history in statistical Machine Learning [48], with early research showing that averaging the parameters of a trained model during its learning rate decay phase can outperform the ﬁnal checkpoint [77]. More recently, this method has experienced renewed interest, with the merging of different Large Language Models (LLMs) showing noticeably stronger benchmark results than a single checkpoint [2]. During the writing of this work, Meta released the Llama-3.1 family of models, where the ﬁnal models consist of an averaged version of the ﬁnal few checkpoints [35], following the intuition of Polyak’s method [47].

In Information Retrieval, this practice is largely understudied, with no previous work reporting its use to the best of our knowledge. We believe that it is especially suitable to creating better overall retrievers by merging the weights of post-trained (ﬁne-tuned) models, which might hurt performance on datasets out of its ﬁne-tuning distribution, with the weights of the original model. We hypothesize that doing so will return most of the performance improvements on datasets similar to the post-training set while avoiding degradation on other tasks.

4 Final Experimental Setting

The ﬁnal experimental setting, used for our model training, is deduced from the results of the various ablation experiments detailed in the previous sections. We provide an overview of the ablation-informed decisions made in Table 7.

[Figure 130]

Setting Setting Used Changed

Query-Length Dynamic Yes In-Batch Negatives No Yes Scheduler Schedule-Free Yes Gradient Clipping Disabled Yes Learning Rate 3e-5 Yes Teacher Scores Normalization Min-Max Normalization Yes Student Scores Normalisation Min-Max Normalization Yes Mixed Loss No No Loss Function KL-Divergence No Teacher Model BGE-M3-Reranker Yes Batch size (per GPU) 16 No Maximum Document Length 300 No Warmup steps 5% of total No Gradient Accumulation Disabled No

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

- Table 7: Overview of the ﬁnal optimal training settings resulting from our experiments, and whether or not they represent a change from previously used settings. Settings in italic are retained from previous approaches with no further experiments within this paper.

We have shown that using dynamic query length at inference-time is strictly superior to ﬁxed-length queries in Section 3.5, and thus adopt dynamic query length for all our ﬁnal evaluations.

As for our training setting, following the ablation results, we have conﬁrmed the hypothesis presented in Section 3.6.1, and opt not use in-batch negatives. We adopt schedule-free learning, as presented in Section 3.6.2, due to its reduced constraints with no performance decrease. As suggested in Section 3.6.3, we normalize both teacher and student scores using min-max normalization, and use KL-Divergence loss, presented in Section 3.6.4. We train our models using knowledge distilled from the teacher scores of BGE-M3, as our results in Section 3.7 show it to be the best performing option on our development sets. We also report training settings that are unchanged from JaColBERTv2’s original settings14, to facilitate reproduction.

[Figure 145]

14After carefully analysing the data make-up of the ablation dataset and verifying it would not have a strong impact, maximum document length was set to 228 for ablation runs in order to minimise computational cost. For the full length training run, we reverted to JaColBERTv2’s 300 to account for longer outlier documents.

Potential Impact on Data Selection An important aspect of our training recipe changes are that the labels of "positives" and "negatives" for each document become unused. Indeed, our only learning metric is based on the KLDivergence loss on min-max normalized teacher scores, which trains our model to attempt to learn the score distribution of its teacher model. While we do not further modify the training data mix in this work, this lifts a considerable constraint for future endeavours, as the use of positive and negative labels often renders the curation of training data more difﬁcult due to the porous nature of relevance judgements. Indeed, curating the proper mix of "hard" and "easy" negatives is a complex process whose outcomes are not yet fully understood [49], and some "hard negatives" could very well be positives, and ﬁltering "too-hard-negatives" remains a mostly empirical, error-prone process [41].

Comparison to previous models Table 8 presents a comparison of an ablation run with the original JaColBERTv2’s training settings, an ablation run comprising of all our ﬁnal’s chosen parameters, JaColBERTv1, our base model, and JaColBERTv2, the previous best-performing Japanese ColBERT model. Our results appear to support our claim that the original JaColBERTv2 training recipe is highly suboptimal. With just 320,000 32-way triplets, our ablation model vastly outperforms the original setting, reaching an average score of 0.738 while the original settings only marginally outperforms JaColBERTv1, with a score of 0.675. Even more noteworthy, our ablation run, with just

- 4% of JaColBERTv2’s training data, outperforms it on both development evaluation sets. Finally, it is important to note that all models discussed here outperform JaColBERTv1’s score of 0.674, further showcasing the importance of nway-training and knowledge distillation, even in sub-optimal training settings.

[Figure 146]

JQaRA MIRACL-small-ja Average NDCG@10 MRR@10 NDCG@10 Recall@5 NDCG@10 Overall

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

JaColBERTv1 (Base Model)

[Figure 151]

0.550 0.811 0.652 0.681 0.601 0.674

[Figure 152]

[Figure 153]

JaColBERTv2 0.585 0.836 0.727 0.751 0.656 0.725 Original settings 0.578 0.820 0.681 0.707 0.630 0.697 Final Settings 0.589 0.836 0.740 0.788 0.665 0.738 Table 8: Results of JaColBERTv1 (the base model for both our experiments and JaColBERTv2), JaColBERTv2, the original training settings as well as our ﬁnal experimentalsetting ablation runs on our development evaluation set. Best results are indicated in bold.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Hardware Usage As part of these experiments, a total of 28 ablation models were trained. Each training run represents 1.3 hours of training time on 4 A100 GPUs, or 5.2 A100 hours per run. As a result, we estimate that a total of 146 A100 hours were spent on ablation runs. Additionally, 15 A100 hours were spent on generating teacher scores over the full 3,200,000 training set using the BGE-M3-Reranker model. Finally, an estimated 8 GPU hours were spent generating teacher scores with all models for the ablation runs, the majority of it dedicated to the MonoT5-3B model. This results in a total pre-ﬁnal training GPU usage of 169 A100 hours.

- 5 Final Results

The results for all newly introduced models and the baselines are presented in Table 9. In the interest of readability, we report the results for all models on the main evaluation of each dataset, as deﬁned in Section 3.4.1. Full evaluation results of our new models covering other, less commonly reported metrics, are available in Appendix A.

JSQuAD Recall@3

MIRACL NDCG@10

JQaRA NDCG@10

JaCWIR MAP@10

ESCI NDCG@10

[Figure 158]

Average Baselines (Multi)

[Figure 159]

[Figure 160]

bge-m3 0.939 0.728 0.539 0.864 0.399 0.694 bge-m3 (all) 0.958 0.752 0.576 0.906 0.380 0.714 multilingual-e5 (large) 0.953 0.706 0.554 0.876 0.320 0.682 multilingual-e5 (base) 0.934 0.647 0.471 0.852 0.347 0.650 multilingual-e5 (small) 0.934 0.636 0.492 0.869 0.331 0.652 Baselines (Mono)

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

GLuCoSE 0.798 0.348 0.309 0.686 0.207 0.470 sup-simcse-ja (base) 0.793 0.171 0.312 0.578 0.140 0.399 sup-simcse-ja (large) 0.777 0.199 0.392 0.474 0.140 0.396 JaColBERTv1 0.961 0.583 0.550 0.904 0.418 0.683 JaColBERTv2 0.968 0.667 0.585 0.919 0.462 0.720 JaColBERTv2.5

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Final checkpoint 0.973 0.756 0.601 0.928 0.462 0.744 + full post-train 0.970 0.780 0.608 0.924 0.451 0.747 + post-train (no mmarco) 0.972 0.772 0.613 0.923 0.452 0.746

[Figure 175]

[Figure 176]

[Figure 177]

- JaColBERTv2.4

(no post train, averaged)

0.973 0.757 0.601 0.929 0.463 0.745

[Figure 178]

- JaColBERTv2.5 (ﬁnal) (post-train, averaged) 0.974 0.778 0.618 0.928 0.462 0.752

[Figure 179]

- Table 9: Results for all baselines and newly introduced models on the main metric for all ﬁve evaluation datasets, as well as their averaged results. Best overall results are indicated in bold. Results in italics indicate that the model was exposed to the task’s training set.

Initial Results Immediately, we notice that all versions of JaColBERTv2.5, no matter the post-training and checkpoint averaging setup, largely outperform JaColBERTv2 and all previous approaches on all ﬁve benchmarks. The ﬁnal checkpoint resulting from the initial training phase reaches an average score of 0.744, with JaColBERTv2 previously reaching 0.720 and BGE-M3 in its all setting, combining dense, sparse and multi-vector representations, 0.714. More interestingly, this checkpoint also achieves an entirely out-of-domain performance of 0.756 NDCG@10 on MIRACL, largely surpassing JaColBERTv2’s out-of-domain 0.667 and outperforming BGE-M3 (all)’s score of 0.0752, despite the latter having been trained on MIRACL. These very strong results conﬁrm our intuition that the existing and most commonly used training recipe for multi-vector models, used in training JaColBERT, was largely suboptimal, and that our proposed improvements result in substantially stronger downstream results. Moreover, this conﬁrms our intuition that it is possible to reach state-of-the-art Japanese retrieval performance while using two orders of magnitude fewer data and compute than leading models.

Post-Training We then explore the results of our post-training step. While they both result in a slight average score improvement, bringing the average model scores to 0.747 when post-training with MMarco data re-injected and 0.746 without. However, these results are obtained via large gains on datasets which are now in-domain, namely MIRACL and JQaRA, while causing moderate degradation on all three other datasets. Interestingly, while adding MMarco data to the post-training set results in less pronounced performance increases on JQaRA, it counter-intuitively further increases the model’s performance on MIRACL, despite reducing its relative importance in the training set.

Checkpoint Averaging Finally, it is noticeable that averaging the two best-performing checkpoints of the initial training with the ﬁnal one result in a slight overall performance increase, although it does not represent a substantial improvement. As a result of the slight increase, we release this model as JaColBERTv2.4. However, averaging the two post-trained checkpoints, with and without MMarco, with the three checkpoints from the original run results in greatly improved performance across the board. Indeed, this ﬁnal version of the model, which is the version we name JaColBERTv2.5, largely outperforms all other variants, reaching an average score of 0.752. This model retains most of the performance gain of the most successful post-training runs on MIRACL, resulting in an increase in the NDCG@10 score on it from 0.757 to 0.778, just 0.002 points short of the best post-trained model (0.780). Even more interestingly, it is the strongest performing model of JQaRA, representing a 0.017NDCG@10 increase on the non post-trained model, but also a 0.005 increase on the post-trained versions.

These gains are achieved with little or no degradation on any of the other datasets, with averaging fully compensating for the degradation experienced in the pre-averaging post-trained models. This suggests that checkpoint averaging has a strong ability to revert catastrophic forgetting [28] in retrieval models, while retaining most of the domain-speciﬁc

performance gained from post-training.

Hardware Usage Our ﬁnal training run took 15.5 hours on 4 A100 GPUs, representing a total GPU usage of 62 hours. Post-training without MMarco took an hour on the same hardware setting, 1.2 hours with MMarco, representing a total usage of 8.8 hours over the two runs, which we round up to 9 for clearer reporting. Our ﬁnal training and post-training steps, in total, required 70 A100 hours. Combined with the 162 GPU hours budget spent on generating teacher scores, the total GPU usage of this study represents 233 A100 hours. While slightly above the JaColBERTv2 computational budget of 228 A100 hours, we fall within the upper bound of our allocated computational budget of 239 hours, while reaching signiﬁcantly stronger results than any previous approach.

- 6 Conclusion

In this work, we present JaColBERTv2.5, a model obtained by systematically evaluating the impact of potential improvements to the ColBERTv2 inference and training recipes, through small-scale ablation runs. Notably, we identify a better way to handle inference-time query length, devise a much improved training setting, and identify the optimal teacher model for knowledge distillation.

JaColBERTv2.5, while trained on only 40% as much data as JaColBERTv2, largely outperforms all other retrieval methods in Japanese, including multilingual models with ﬁve times the parameter count and trained using two orders of magnitude more compute and data.

Throughout our experiments, we have shown that multi-vector retrieval models can not only effectively bridge the gap between multilingual models and monolingual, but largely outperform the former, with only moderate compute resources and lower quality data than their English equivalents.

The results of our training recipe also show that it is possible to train models, using knowledge distillation, without any reliance on hard "positive" or "negative" labels, focusing entirely on the teacher score distribution instead. This represents valuable information for future work, as a step towards greatly simplifying the data curation process.

We have also demonstrated that checkpoint averaging, where the weights of multiple checkpoints of a similarly-shaped model are averaged to create a merged model, can greatly improve the generalisation potential of ﬁne-tuned JaColBERT models, while retaining the same out-of-domain performance as the original model.

We make both JaColBERTv2.515, our ﬁnal checkpoint resulting from averaging our ﬁnal model with post-training runs on two slightly different distributions, and JaColBERTv2.416, the outcome of merging the three best checkpoints of the original pre-training, publicly available.

We believe that our work can support the development of future mono-lingual retrievers, both in Japanese and other lower-resources languages. Notably, we believe that our improved training recipe can be directly applied to sparse retrieval models such as SPLADE [31], which has already shown strong potential in a Japanese setting17

In order to best support such future work, we make the entirety of our training data for both ablation and full training runs, teacher scores included, publicly available18. To further studies into better understanding of multi-vector retrieval models, we release all mid-training checkpoints, saved every 2,000 training steps19.

Finally, while our speciﬁc application case is focused on the Japanese language, all of our training recipe improvements are language-agnostic, and even our ablation-sized models, trained on just 320,000 triplets, vastly outperform previous monolingual approaches. As a result, we believe that our method can be directly applied to other languages and domains and yield large performance gains.

- 7 Ethical Considerations

We acknowledge the importance of ethical consideration in Natural Language Processing work. We have ensured to make our work as reproducible as possible, making both the the ﬁnal model, in-development versions of it, and the entirety of our training data publicly available to facilitate reproduction and future work.

[Figure 180]

- 15https://huggingface.co/answerdotai/JaColBERTv2.5
- 16https://huggingface.co/answerdotai/JaColBERTv2.4 17As demonstrated by an early release from the University of Tsukuba available at

https://huggingface.co/aken12/splade-japanese-v3.

- 18https://huggingface.co/datasets/answerdotai/MMarco-japanese-32-scored-triplets
- 19https://huggingface.co/collections/bclavie/jacolbertv25-checkpoints-66a37d8da6b0d4d69c14f9c3

There are no extreme ethical risks associated with our models. However, while they are not generative and will not, by themselves, generate harmful content, they fall in line with existing retrieval work. As such, our work is not exempt from potential biases, especially as the largest part of our training data is a lightly ﬁltered internet corpus which then underwent machine translation. It is possible that our models might unduly favor certain types of content, and may rank misinformation or harmful content highly for certain queries.

- 8 Acknowledgement

The author thank Yuichi Tateno for his extensive work in both creating benchmarks for, as well as training and benchmarking, Japanese rerankers, and Hayato Tsukagoshi for his willingness to share his work on Japanese SimCSE models and exploring the Japanese embedding training and data landscape. Further thanks extend to Benjamin Warner for sharing his expert advice on the various ways to optimise model training and detect inefﬁciencies, especially in the area of scheduling, as well as Alexis Gallagher, for very insightful feedback during the writing of this work. The author would also like acknowledge the helpful exchanges with Omar Khattab, Antoine Chafﬁn and Grifﬁn Adams for their eagerness to discuss and help clarify ideas, as well as Professor Makoto P. Kato for helpful exchanges around building better Japanese retrieval models.

References

- [1] ACHIAM, J., ADLER, S., AGARWAL, S., AHMAD, L., AKKAYA, I., ALEMAN, F. L., ALMEIDA, D., ALTENSCHMIDT, J., ALTMAN, S., ANADKAT, S., ET AL. Gpt-4 technical report. arXiv preprint arXiv:2303.08774

(2023).

- [2] AKIBA, T., SHING, M., TANG, Y., SUN, Q., AND HA, D. Evolutionary optimization of model merging recipes. arXiv preprint arXiv:2403.13187 (2024).
- [3] BAEZA-YATES, R., RIBEIRO-NETO, B., ET AL. Modern information retrieval, vol. 463. ACM press New York, 1999.
- [4] BASSANI, E. ranx: A blazing-fast python library for ranking evaluation and comparison. In European Conference on Information Retrieval (2022), Springer, pp. 259–264.
- [5] BELKIN, N., AND CROFT, W. Retrieval techniques. Annual review of information science and technology 22

(1987), 109–145.

- [6] BONIFACIO, L., JERONYMO, V., ABONIZIO, H. Q., CAMPIOTTI, I., FADAEE, M., LOTUFO, R., AND NOGUEIRA, R. MMARCO: A multilingual version of the MS MMARCO passage ranking dataset. arXiv preprint arXiv:2108.13897 (2021).
- [7] BROWN, T., MANN, B., RYDER, N., SUBBIAH, M., KAPLAN, J. D., DHARIWAL, P., NEELAKANTAN, A., SHYAM, P., SASTRY, G., ASKELL, A., ET AL. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
- [8] CHEN, J., XIAO, S., ZHANG, P., LUO, K., LIAN, D., AND LIU, Z. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216 (2024).
- [9] CLAVIÉ, B. Jacolbert and hard negatives, towards better japanese-ﬁrst embeddings for retrieval: Early technical report. arXiv preprint arXiv:2312.16144 (2023).
- [10] CONNEAU, A., KHANDELWAL, K., GOYAL, N., CHAUDHARY, V., WENZEK, G., GUZMÁN, F., GRAVE, É., OTT, M., ZETTLEMOYER, L., AND STOYANOV, V. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (2020), pp. 8440– 8451.
- [11] CRASWELL, N., MITRA, B., YILMAZ, E., CAMPOS, D., AND VOORHEES, E. M. Overview of the trec 2019 deep learning track. arXiv preprint arXiv:2003.07820 (2020).
- [12] DEFAZIO, A., XINGYU, YANG, MEHTA, H., MISHCHENKO, K., KHALED, A., AND CUTKOSKY, A. The road less scheduled. arXiv preprint arXiv:2405.15682 (2024).
- [13] DEVLIN, J., CHANG, M.-W., LEE, K., AND TOUTANOVA, K. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (2019), pp. 4171–4186.

- [14] FORMAL, T., PIWOWARSKI, B., AND CLINCHANT, S. Splade: Sparse lexical and expansion model for ﬁrst stage ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021), pp. 2288–2292.
- [15] FORMAL, T., PIWOWARSKI, B., AND CLINCHANT, S. A white box analysis of colbert. In Advances in Information Retrieval: 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part II 43 (2021), Springer, pp. 257–263.
- [16] GAO, T., YAO, X., AND CHEN, D. SimCSE: Simple contrastive learning of sentence embeddings, 2022.
- [17] GIACALONE, B., PAIEMENT, G., TUCKER, Q., AND ZANIBBI, R. Beneath the [mask]: An analysis of structural query tokens in colbert. In European Conference on Information Retrieval (2024), Springer, pp. 431–439.
- [18] HESTNESS, J., NARANG, S., ARDALANI, N., DIAMOS, G., JUN, H., KIANINEJAD, H., PATWARY, M. M. A., YANG, Y., AND ZHOU, Y. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409

(2017).

- [19] HINTON, G., VINYALS, O., AND DEAN, J. Distilling the knowledge in a neural network, 2015.
- [20] HOFSTÄTTER, S., ALTHAMMER, S., SCHRÖDER, M., SERTKAN, M., AND HANBURY, A. Improving efﬁcient neural ranking models with cross-architecture knowledge distillation. arXiv preprint arXiv:2010.02666 (2020).
- [21] HOFSTÄTTER, S., KHATTAB, O., ALTHAMMER, S., SERTKAN, M., AND HANBURY, A. Introducing neural bag of whole-words with colberter: Contextualized late interactions using enhanced reduction. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (2022), pp. 737–747.
- [22] HU, S., TU, Y., HAN, X., HE, C., CUI, G., LONG, X., ZHENG, Z., FANG, Y., HUANG, Y., ZHAO, W., ET AL. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395 (2024).
- [23] IZACARD, G., AND GRAVE, E. Distilling knowledge from reader to retriever for question answering. In ICLR 2021-9th International Conference on Learning Representations (2021).
- [24] KARPUKHIN, V., OGUZ, B., MIN, S., LEWIS, P., WU, L., EDUNOV, S., CHEN, D., AND YIH, W.-T. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2020), pp. 6769–6781.
- [25] KATO, S., TOGASHI, R., MAEDA, H., FUJITA, S., AND SAKAI, T. Lstm vs. bm25 for open-domain qa: A hands-on comparison of effectiveness and efﬁciency. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (2017), pp. 1309–1312.
- [26] KHATTAB, O., AND ZAHARIA, M. Colbert: Efﬁcient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (2020), pp. 39–48.
- [27] KIM, T., OH, J., KIM, N., CHO, S., AND YUN, S.-Y. Comparing kullback-leibler divergenceand mean squared error loss in knowledge distillation. In Proceedings of the 30th International Joint Conference on Artiﬁcial Intelligence (IJCAI) (08 2021), pp. 2628–2635.
- [28] KIRKPATRICK, J., PASCANU, R., RABINOWITZ, N., VENESS, J., DESJARDINS, G., RUSU, A. A., MILAN, K., QUAN, J., RAMALHO, T., GRABSKA-BARWINSKA, A., ET AL. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114, 13 (2017), 3521–3526.
- [29] KOBAYASHI, M., AND TAKEDA, K. Information retrieval on the web. ACM computing surveys (CSUR) 32, 2

(2000), 144–173.

- [30] KURIHARA, K., KAWAHARA, D., AND SHIBATA, T. Jglue: Japanese general language understanding evaluation. In Proceedings of the Thirteenth Language Resources and Evaluation Conference (2022), pp. 2957–2966.
- [31] LASSANCE, C., DÉJEAN, H., FORMAL, T., AND CLINCHANT, S. Splade-v3: New baselines for splade. arXiv preprint arXiv:2403.06789 (2024).
- [32] LEE, J., DAI, Z., DUDDU, S. M. K., LEI, T., NAIM, I., CHANG, M.-W., AND ZHAO, V. Rethinking the role of token retrieval in multi-vector retrieval. Advances in Neural Information Processing Systems 36 (2024).
- [33] LIN, J., ALFONSO-HERMELO, D., JERONYMO, V., KAMALLOO, E., LASSANCE, C., NOGUEIRA, R., OGUNDEPO, O., REZAGHOLIZADEH, M., THAKUR, N., YANG, J.-H., ET AL. Simple yet effective neural ranking and reranking baselines for cross-lingual information retrieval. arXiv preprint arXiv:2304.01019 (2023).
- [34] LIN, S.-C., YANG, J.-H., AND LIN, J. In-batch negatives for knowledge distillation with tightly-coupled teachers for dense retrieval. In Proceedingsof the 6th Workshop on RepresentationLearning for NLP (RepL4NLP-

2021) (2021), pp. 163–173.

- [35] LLAMATEAM. The llama 3 herd of models. HUMAN & MACHINE INTELLIGENCE, AI @ META REPORTS

(2024).

- [36] LOSHCHILOV, I., AND HUTTER, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101

(2017).

- [37] LOUIS, A. Decouvrir: A benchmark for evaluating the robustness of information retrieval models in french, 2024.
- [38] LOUIS, A., SAXENA, V., VAN DIJCK, G., AND SPANAKIS, G. Colbert-xm: A modular multi-vector representation model for zero-shot multilingual information retrieval. arXiv preprint arXiv:2402.15059 (2024).
- [39] MACAVANEY, S., AND TONELLOTTO, N. A reproducibility study of plaid. arXiv preprint arXiv:2404.14989

(2024).

- [40] MANNING, C. D. Introduction to information retrieval. Syngress Publishing„ 2008.
- [41] MERRICK, L., XU, D., NUTI, G., AND CAMPOS, D. Arctic-embed: Scalable, efﬁcient, and accurate text embedding models. arXiv preprint arXiv:2405.05374 (2024).
- [42] MITRA, B., CRASWELL, N., ET AL. An introduction to neural information retrieval. Foundations and Trends® in Information Retrieval 13, 1 (2018), 1–126.
- [43] MUENNIGHOFF, N., TAZI, N., MAGNE, L., AND REIMERS, N. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316 (2022).
- [44] NAIR, S., YANG, E., LAWRIE, D., DUH, K., MCNAMEE, P., MURRAY, K., MAYFIELD, J., AND OARD, D. W. Transfer learning approaches for building cross-language dense retrieval models. In European Conference on Information Retrieval (2022), Springer, pp. 382–396.
- [45] NGUYEN, T., ROSENBERG, M., SONG, X., GAO, J., TIWARY, S., MAJUMDER, R., AND DENG, L. MS MARCO: A human generated machine reading comprehension dataset. choice 2640 (2016), 660.
- [46] NOGUEIRA, R., JIANG, Z., AND LIN, J. Document ranking with a pretrained sequence-to-sequence model. arXiv preprint arXiv:2003.06713 (2020).
- [47] POLYAK, B. New stochastic approximation type procedures. Avtomatica i Telemekhanika 7 (01 1990), 98–107.
- [48] POLYAK, B. T., AND JUDITSKY, A. B. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization 30, 4 (1992), 838–855.
- [49] PRADEEP, R., LIU, Y., ZHANG, X., LI, Y., YATES, A., AND LIN, J. Squeezing water from a stone: a bag of tricks for further improving cross-encoder effectiveness for reranking. In European Conference on Information Retrieval (2022), Springer, pp. 655–670.
- [50] QU, Y., DING, Y., LIU, J., LIU, K., REN, R., ZHAO, W. X., DONG, D., WU, H., AND WANG, H. Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for ComputationalLinguistics: Human Language Technologies (2021), pp. 5835–5847.
- [51] RAFFEL, C., SHAZEER, N., ROBERTS, A., LEE, K., NARANG, S., MATENA, M., ZHOU, Y., LI, W., AND LIU, P. J. Exploring the limits of transfer learning with a uniﬁed text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.
- [52] RAJPURKAR, P., ZHANG, J., LOPYREV, K., AND LIANG, P. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing

(2016), pp. 2383–2392.

- [53] REDDY, C. K., MÀRQUEZ, L., VALERO, F., RAO, N., ZARAGOZA, H., BANDYOPADHYAY, S., BISWAS, A., XING, A., AND SUBBIAN, K. Shopping queries dataset: A large-scale esci benchmark for improving product search. arXiv preprint arXiv:2206.06588 (2022).
- [54] REN, R., QU, Y., LIU, J., ZHAO, W. X., SHE, Q., WU, H., WANG, H., AND WEN, J.-R. Rocketqav2: A joint training method for dense passage retrieval and passage re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (2021), pp. 2825–2835.
- [55] ROBERTSON, S. E., WALKER, S., JONES, S., HANCOCK-BEAULIEU, M. M., GATFORD, M., ET AL. Okapi at trec-3. Nist Special Publication Sp 109 (1995), 109.
- [56] SAAD-FALCON, J., KHATTAB, O., SANTHANAM, K., FLORIAN, R., FRANZ, M., ROUKOS, S., SIL, A., SULTAN, M., AND POTTS, C. Udapdr: Unsupervised domain adaptation via llm prompting and distillation of rerankers. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (2023), pp. 11265–11279.

- [57] SANH, V., DEBUT, L., CHAUMOND, J., AND WOLF, T. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 (2019).
- [58] SANTHANAM, K., KHATTAB, O., POTTS, C., AND ZAHARIA, M. Plaid: an efﬁcient engine for late interaction retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management

(2022), pp. 1747–1756.

- [59] SANTHANAM, K., KHATTAB, O., SAAD-FALCON, J., POTTS, C., AND ZAHARIA, M. ColBERTv2: Effective and efﬁcient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (2022), pp. 3715–3734.
- [60] TATENO, Y. Jacwir: Japanese casual web ir. HuggingFace Datasets, https://huggingface.co/datasets/hotchpotch/JaCWIR (2024).
- [61] TATENO, Y. Jqara: Japanese question answering with retrieval augmentation. HuggingFace Datasets, https://huggingface.co/datasets/hotchpotch/JQaRA (2024).
- [62] THAKUR, N., REIMERS, N., RÜCKLÉ, A., SRIVASTAVA, A., AND GUREVYCH, I. BEIR: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663 (2021).
- [63] TOUVRON, H., MARTIN, L., STONE, K., ALBERT, P., ALMAHAIRI, A., BABAEI, Y., BASHLYKOV, N., BATRA, S., BHARGAVA, P., BHOSALE, S., ET AL. Llama 2: Open foundation and ﬁne-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
- [64] TROTMAN, A., PUURULA, A., AND BURGESS, B. Improvements to bm25 and language models examined. In Proceedings of the 19th Australasian Document Computing Symposium (2014), pp. 58–65.
- [65] TSUKAGOSHI, H., SASANO, R., AND TAKEDA, K. Japanese SimCSE technical report. arXiv preprint arXiv:2310.19349 (2023).
- [66] VASWANI, A., SHAZEER, N., PARMAR, N., USZKOREIT, J., JONES, L., GOMEZ, A. N., KAISER, Ł., AND POLOSUKHIN, I. Attention is all you need. Advances in neural information processing systems 30 (2017).
- [67] WANG, B., LIU, Z., HUANG, X., JIAO, F., DING, Y., AW, A., AND CHEN, N. Seaeval for multilingual foundation models: From cross-lingual alignment to cultural reasoning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for ComputationalLinguistics: Human Language Technologies (Volume 1: Long Papers) (2024), pp. 370–390.
- [68] WANG, L., YANG, N., HUANG, X., JIAO, B., YANG, L., JIANG, D., MAJUMDER, R., AND WEI, F. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533 (2022).
- [69] WANG, L., YANG, N., HUANG, X., YANG, L., MAJUMDER, R., AND WEI, F. Multilinguale5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672 (2024).
- [70] WANG, W., WEI, F., DONG, L., BAO, H., YANG, N., AND ZHOU, M. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in Neural Information Processing Systems 33 (2020), 5776–5788.
- [71] WILSON, A. C., ROELOFS, R., STERN, M., SREBRO, N., AND RECHT, B. The marginal value of adaptive gradient methods in machine learning. Advances in neural information processing systems 30 (2017).
- [72] XIAO, S., LIU, Z., ZHANG, P., AND MUENNIGHOF, N. C-pack: Packaged resources to advance general chinese embedding. arXiv preprint arXiv:2309.07597 (2023).
- [73] XIAO, S., LIU, Z., ZHANG, P., AND MUENNIGHOF, N. C-pack: Packaged resources to advance general chinese embedding. arXiv preprint arXiv:2309.07597 (2023).
- [74] YAMADA, I., ASAI, A., SHINDO, H., TAKEDA, H., AND MATSUMOTO, Y. LUKE: Deep contextualized entity representations with entity-aware self-attention. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2020), pp. 6442–6454.
- [75] YANG, E., LAWRIE, D., MAYFIELD, J., OARD, D. W., AND MILLER, S. Translate-distill: Learning crosslanguage dense retrieval by translation and distillation. In European Conference on Information Retrieval (2024), Springer, pp. 50–65.
- [76] YATES, A., NOGUEIRA, R., AND LIN, J. Pretrained transformers for text ranking: Bert and beyond. In Proceedings of the 14th ACM International Conference on web search and data mining (2021), pp. 1154–1156.
- [77] YIN, G. Stochastic approximation via averaging: the polyak’s approach revisited. In Simulation and Optimization: Proceedings of the International Workshop on Computationally Intensive Methods in Simulation and Optimization held at the InternationalInstitute for Applied Systems Analysis (IIASA), Laxenburg, Austria, August 23–25, 1990 (1992), Springer, pp. 119–134.

- [78] ZHAI, X., KOLESNIKOV, A., HOULSBY, N., AND BEYER, L. Scaling vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022), pp. 12104–12113.
- [79] ZHANG, X., MA, X., SHI, P., AND LIN, J. Mr. TyDi: A multi-lingual benchmark for dense retrieval. In Proceedings of the 1st Workshop on Multilingual Representation Learning (2021), pp. 127–137.
- [80] ZHANG, X., MA, X., SHI, P., AND LIN, J. Mr. tydi: A multi-lingual benchmark for dense retrieval. In Proceedings of the 1st Workshop on Multilingual Representation Learning (2021), pp. 127–137.
- [81] ZHANG, X., THAKUR, N., OGUNDEPO, O., KAMALLOO, E., ALFONSO-HERMELO, D., LI, X., LIU, Q., REZAGHOLIZADEH, M., AND LIN, J. MIRACL: A multilingual retrieval dataset covering 18 diverse languages. Transactions of the Association for Computational Linguistics 11 (2023), 1114–1131.
- [82] 舘 野 祐 一. 日 本 語RERANKER 作 成 の テ ク ニ カ ル レ ポ ー ト. Technical Report, accessed at https://secon.dev/entry/2024/04/02/080000-japanese-reranker-tech-report/ [Tateno, Y.

(2024). Technical report on Japanese Reranker] (2024).

- [83] 鈴木潤AND 松田耕史AND 鈴木正敏AND 加藤拓真AND 宮脇峻平AND 西田京介. ライブコンペティショ ン：「AI 王クイズAI 日本一決定戦」. 自然言語処理, 28 (3), PP 888-894. [SUZUKI, J., MATSUDA, K., SUZUKI, M., KATO, T., MIYAWAKI, S., AND NISHIDA, K. (2021). LIVE COMPETITION: "AI KING: QUIZ AI JAPAN CHAMPIONSHIP". NATURAL LANGUAGE PROCESSING, 28 (3), PP. 888–894.] (2021).

A Full JaColBERTv2.5 results

Baseline Our Models JaColBERT v2

[Figure 181]

Post-Train (full) MIRACL

JaColBERT v2.5

JaColBERT v2.4

Final Checkpoint

Post-Train (no mmarco)

[Figure 182]

[Figure 183]

[Figure 184]

NDCG@10 0.667 0.778 0.757 0.756 0.772 0.780 MRR@10 0.688 0.795 0.780 0.781 0.793 0.802 Recall@10 0.802 0.887 0.869 0.871 0.880 0.887 Recall@100 0.961 0.989 0.987 0.987 0.987 0.990 JQaRA

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

NDCG@10 0.585 0.618 0.601 0.601 0.613 0.608 MRR@10 0.836 0.856 0.846 0.843 0.849 0.846 JaCWIR

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

MAP@10 0.919 0.928 0.929 0.928 0.923 0.924 Hit Rate@10 0.982 0.979 0.980 0.979 0.979 0.978 JSQuAD

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Recall@1 0.917 0.930 0.930 0.930 0.928 0.925 Recall@3 0.967 0.974 0.974 0.974 0.972 0.970 Recall@5 0.976 0.982 0.982 0.982 0.980 0.978 Recall@10 0.982 0.987 0.987 0.987 0.986 0.986 ESCI

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

NDCG@10 0.462 0.462 0.463 0.462 0.452 0.451 MRR@10 0.619 0.619 0.621 0.622 0.610 0.606 Recall@10 0.381 0.386 0.386 0.386 0.379 0.376

[Figure 205]

[Figure 206]

[Figure 207]

- Table 10: Presentation of the full results of different variants of our newly introduced models across a range of metrics. Best results are presented in bold. Results in italic indicate that the task is in-domain.

This figure "test.png" is available in "png"  format from:

http://arxiv.org/ps/2407.20750v1

