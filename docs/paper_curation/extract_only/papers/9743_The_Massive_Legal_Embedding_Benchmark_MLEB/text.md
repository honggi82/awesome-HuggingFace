# arXiv:2510.19365v1[cs.CL]22Oct2025

## The Massive Legal Embedding Benchmark (MLEB)

Umar Butler∗, Abdur-Rahman Butler, Adrian Lucas Malec Isaacus

21 October 2025

Abstract

We present the Massive Legal Embedding Benchmark (MLEB)1, the largest, most diverse, and most comprehensive open-source benchmark for legal information retrieval to date. MLEB consists of ten expert-annotated datasets spanning multiple jurisdictions (the US, UK, EU, Australia, Ireland, and Singapore), document types (cases, legislation, regulatory guidance, contracts, and literature), and task types (search, zero-shot classiﬁcation, and question answering). Seven of the datasets in MLEB were newly constructed in order to ﬁll domain and jurisdictional gaps in the open-source legal information retrieval landscape. We document our methodology in building MLEB and creating the new constituent datasets, and release our code, results, and data openly to assist with reproducible evaluations.

### 1 Introduction

In the context of information retrieval, embedding models convert documents and queries into sets of numbers known as ‘embeddings’ that can be compared with each other to identify relevant search results. Embeddings power the ‘retrieval’ component of retrieval-augmented generation (RAG) applications and are widely used across the legal tech industry. In legal RAG applications, lowquality embeddings lead to low-quality search results, which in turn lead to low-quality responses and increased hallucinations [1].

Despite their importance, limited attention has been paid to ensuring that embedding models are genuinely ﬁt for legal information retrieval. Previous attempts at building an industry-standard legal information retrieval benchmark have been limited in quality, size, and diversity. LegalBenchRAG [14] focuses on a small set of contract- and US-centric datasets, while the legal domain subset of the Massive Multilingual Text Embedding Benchmark (MTEB-Legal) [4] exhibits labeling issues and narrow topical coverage. Consequently, existing benchmark performance can fail to predict realworld eﬀectiveness at legal retrieval tasks. MLEB attempts to address those limitations by being larger, more diverse, and of higher quality than previous legal information retrieval benchmarks.

### 2 Related work 2.1 LegalBench-RAG

LegalBench-RAG evaluates legal information retrieval performance against four pre-existing evaluation sets: ContractNLI [10], Contract Understanding Atticus Dataset (CUAD) [7], M&A Under-

[Figure 1]_images/imageFile1.png>)

∗Corresponding author: research@isaacus.com 1https://isaacus.com/mleb

standing Dataset (MAUD) [16], and Privacy QA [15]. All datasets center around contracts, largely sourced from the US.

In practice, legal professionals and users seeking legal advice or knowledge tend to search for and be interested in a much broader range of document types than just contracts, including legislation, regulations, cases, and general legal literature.

LegalBench-RAG’s narrow focus on contracts thus limits its usefulness for the evaluation of legal embedding models that generalize well to other legal domains and multiple jurisdictions.

#### 2.2 MTEB-Legal

MTEB-Legal consists of eight datasets: AILA Casedocs [2], AILA Statutes [2], GerDaLIR Small [18], LeCaRDv2 [11], Consumer Contracts QA [9], Legal Summarization [12], Corporate Lobbying [13], and LegalQuAD [8].

Upon manual inspection of two of the English-language datasets in MTEB-Legal, AILA Casedocs2 and AILA Statutes3, we found they contained many query-passage pairs that were totally irrelevant to each other. According to the authors, the datasets had been created using an ‘automated methodology’ that paired ‘facts stated in certain [Indian] Supreme Court cases’ with cases and statutes that had been ‘cited by the lawyers arguing those cases’. This construction method was employed because ‘actually involving legal experts (e.g., to ﬁnd relevant prior cases / statutes) would have required a signiﬁcant amount of ﬁnancial resources and time’ [2].

In addition to mislabeling, we found that MTEB-Legal lacked diversity in the areas that matter most to legal practitioners and seekers of legal knowledge.

Speciﬁcally, of the remaining English-language datasets after exclusion of AILA Casedocs and AILA Statutes, two deal with consumer terms of service (Consumer Contracts QA and Legal Summarization), leaving only one (Corporate Lobbying) that deals with legislation, and none dealing with case law. All such datasets are largely representative of American law.

Regarding the non-English-language datasets in the legal split of MTEB, we argue that, in many cases, the legal systems of diﬀerent cultures may fundamentally diﬀer in ways that make crossjurisdictional comparisons (e.g., between the common law system used by Anglosphere countries and Sharia law) of the eﬀectiveness of legal embeddings inappropriate.

Furthermore, given that the legal split contains two German datasets, one Chinese dataset, and no other non-English datasets, and that those datasets are concentrated on three select legal tasks, we argue that the inclusion of non-English datasets largely introduces bias and noise in ways that are unlikely to be conducive to real-world performance on most English-language legal information retrieval tasks.

### 3 The Massive Legal Embedding Benchmark (MLEB)

Learning from the limitations of existing legal embedding evaluation sets, MLEB has been designed with four key objectives in mind, namely to:

- 1. be of high quality, both in terms of provenance and labeling;
- 2. consist of text processing tasks that have genuine real-world utility to legal tech professionals;
- 3. be meaningfully challenging in ways likely to require signiﬁcant legal knowledge and strong legal reasoning skills; and

[Figure 2]_images/imageFile2.png>)

- 2https://huggingface.co/datasets/mteb/AILA_casedocs
- 3https://huggingface.co/datasets/mteb/AILA_statutes

- 4. represent a broad variety of jurisdictions, legal areas, and types of legal texts.

To that end, MLEB contains ten diﬀerent evaluation sets spanning a range of diﬃculties (including tasks requiring legal reasoning as well as tasks requiring lexical analysis), problem types (speciﬁcally, retrieval, zero-shot classiﬁcation, and question answering), jurisdictions (the US, UK, EU, Australia, Ireland, and Singapore), and document types (decisions, legislation, regulatory guidance, contracts, and literature).

Of the ten datasets in MLEB, seven are entirely new, constructed either by having subject matter experts hand-label data or by adapting existing expert-labeled data.

Below, we present an overview of all the datasets included in MLEB alongside all the various features that make them unique.

[Figure 3]_images/imageFile3.png>)

Name Queries Domain Jurisdiction Description Bar Exam QA 117 Judicial US US bar exam questions paired with rele-

[Figure 4]_images/imageFile4.png>)

vant caselaw. [19]

SCALR 120 Judicial US Questions presented to the US Supreme Court paired with descriptions of the Court’s ﬁnal holdings. [6]

Singaporean Judicial Keywords

500 Judicial Singapore Judicial catchwords paired with Singaporean court judgments.

GDPR Holdings Retrieval

Australian Tax Guidance Retrieval

Irish Legislative Summaries

UK Legislative Long Titles

Contractual Clause Retrieval

License TL;DR Retrieval

500 Judicial EU GDPR case fact patterns paired with descriptions of court holdings.

112 Regulatory Australia Australian tax law questions paired with relevant Australian Government tax guidance and policies.

500 Regulatory Ireland Long titles paired with Irish acts.

78 Regulatory UK Long titles paired with UK acts.

90 Contractual Multinational NLI-style descriptions of types of contractual clauses paired with examples of those clauses.

65 Contractual Multinational Summaries of software licenses paired with their full texts.

Consumer Contracts QA

198 Contractual Multinational Questions about online terms of service paired with relevant clauses. [9]

[Figure 5]_images/imageFile5.png>)

- Table 1: An overview of the ten datasets making up MLEB, spanning six jurisdictions (the US, UK, EU, Australia, Ireland, and Singapore) and three domain types (Judicial, Contractual, Regulatory). Full URLs are provided in the Data Availability section (page 9).

#### 3.1 Bar Exam QA

Bar Exam QA [20] is a pre-existing evaluation set created by Stanford RegLab that tests the ability of information retrieval models to identify cases and legal literature (speciﬁcally, passages from

textbooks) relevant to US state bar exam questions. Bar Exam QA was selected to be a challenging retrieval task requiring lawyer-like analytical skills and extensive knowledge of American law for an information retrieval model to achieve high performance.

#### 3.2 SCALR

SCALR [6] is a pre-existing evaluation set created by Faiz Surani and Varun Iyer that pairs questions presented to the US Supreme Court with descriptions of the Court’s ﬁnal holdings. Uniquely, SCALR challenges models to predict, or at least be aware of, important US Supreme Court holdings. Half of SCALR was reserved for validation, leaving the other half for inclusion in the MLEB test set, with no queries overlapping between sets.

#### 3.3 Singaporean Judicial Keywords

Singaporean Judicial Keywords is a new evaluation set consisting of 500 catchword-judgment pairs sourced from the Singaporean Judiciary.

Singaporean Judicial Keywords was constructed by collecting all publicly available Singaporean court judgments, converting them into plain text with Inscriptis [17], cleaning them, and removing near duplicates with the simhash algorithm [3], and then using multiple complex regex patterns to extract catchwords from them before removing those catchwords and everything preceding them from judgments (in order to force models to focus on representing the core semantics of judgments’ texts rather than their metadata-rich cover sheets).

Uniquely, the keywords in this dataset are real-world annotations created by subject matter experts, namely, Singaporean law reporters, as opposed to being constructed ex post facto by third parties.

Additionally, unlike standard keyword queries, judicial catchwords are meant to capture the most essential and relevant concepts and principles to a case, even where those elements may never be explicitly referenced by it.

Such features make this dataset especially useful for the robust evaluation of the legal conceptual understanding and overall legal knowledge of information retrieval models.

#### 3.4 GDPR Holdings Retrieval

GDPR Holdings Retrieval is a new evaluation set consisting of 500 fact patterns paired with holdings in European regulatory and court decisions. This dataset was constructed by collecting all GDPRHub [5] articles, using regex to separate their facts and holdings sections and then converting those sections into plain text with Inscriptis. This dataset is intended to stress test the ability of an information retrieval model to retrieve relevant judicial and regulatory decisions given an arbitrary fact pattern.

#### 3.5 Australian Tax Guidance Retrieval

Australian Tax Guidance Retrieval is a new evaluation set consisting of 112 real-life tax questions posed by Australian taxpayers paired with 105 relevant Australian Government guidance and policy documents.

Questions in this dataset were sourced from the Australian Taxation Oﬃce (ATO)’s community forum where Australian taxpayers can ask accountants and ATO oﬃcials their tax questions. We found that, in many cases, users’ questions can be answered by reference to Australian Government guidance materials that, for whatever reason, taxpayers were unable to locate themselves.

We thus constructed this dataset by:

- 1. for each of the 14 sub-topics of the ATO Community forum that did not come under the parent topics ‘Online Services’ and ‘Tax Professionals’ (which were found to consist almost exclusively of practical questions around the use of ATO services rather than substantive tax law queries), selecting 8 questions that:

- (a) had at least one answer with at least one hyperlink (with, where there were multiple competing answers, the answer selected by the user as the best answer being used otherwise using the answers of ATO employees over those of tax professionals),
- (b) were about a substantive tax law problem and were not merely practical questions about, for example, the use of ATO services or how to ﬁle tax returns;

- 2. for each sampled question, visiting the hyperlink in the selected answer that appeared to be the most relevant to the question and then copying as much text from the hyperlink as appeared relevant to the question, ranging from a single paragraph to the entire document;
- 3. using a purpose-built Chrome extension to extract questions and relevant passages directly to Markdown to preserve the semantics of added markup; and
- 4. lightly cleaning queries and passages by replacing consecutive sequences of at least two newlines with two consecutive newlines and removing leading and trailing whitespace.

The queries in this dataset are valuable and challenging because users have gone to the eﬀort of asking them on a forum, indicating that traditional search engines failed to surface the answers they were looking for. The government materials are, in turn, also valuable because practicing subject matter experts, namely, accountants and ATO oﬃcials, have conﬁrmed them to be relevant.

#### 3.6 Irish Legislative Summaries

Irish Legislative Summaries is an evaluation set consisting of 500 Irish laws and their long titles, succinctly summarizing the subject matter, scope, and purpose of legislation. Similar to Singaporean Judicial Keywords, this dataset was constructed by collecting all publicly available Irish laws, converting them into plain text with Inscriptis, cleaning them, and removing near duplicates with the simhash algorithm, and then using regex to extract their long titles before removing those long titles and everything preceding them from legislation (in order to force models to focus on representing the core semantics of acts’ texts rather than their metadata-rich front matter). This dataset is meant to stress test the ability of an information retrieval model to retrieve relevant statutes to short queries describing them.

#### 3.7 UK Legislative Long Titles

UK Legislative Long Titles is an evaluation set consisting of 78 UK laws and their long titles. This dataset was constructed from all publicly available UK laws in the same way as Irish Legislative Summaries.

#### 3.8 Contractual Clause Retrieval

Contractual Clause Retrieval is an evaluation set consisting of 45 unique types of contractual clauses paired with 2 highly representative examples of each, resulting in 90 pairings.

This dataset was constructed by:

- 1. coming up with 45 contractual clause types and NLI-style statements (see Appendix A), which were intended to be as representative of the diverse types of contractual clauses common in commercial transactions while also being speciﬁc and distinct enough to avoid capturing substantively overlapping clause types. These included set-oﬀ clauses, authority to sign clauses, power of attorney, termination for cause clauses, good faith clauses, payment currency clauses, and many others;
- 2. for each clause type, sourcing two highly representative examples from online clause reference databases, stripping out irrelevant information where that might cause overlap with another clause type.

This dataset is intended to stress test the ability of information retrieval, zero-shot classiﬁcation, and NLI models to identify a broad range of common types of contractual clauses based solely on their deﬁnition, without prior examples.

- 3.9 License TL;DR Retrieval

License TL;DR Retrieval is an evaluation set consisting of 65 summary-license pairs sourced from tl;drLegal. This dataset was constructed by collecting all licenses publicly available on tl;drLegal and pairing their human-created summaries with their full texts. This dataset is intended to stress test the ability of an information retrieval model to match relevant open-source licenses with summaries of their terms.

#### 3.10 Consumer Contracts QA

Consumer Contracts QA [9] is a pre-existing evaluation set created by Noam Kolt that tests the ability of information retrieval models to retrieve relevant contractual clauses to questions about consumer contracts. The MLEB version of this dataset was derived by splitting the MTEB version in half so that some examples could be reserved for validation, with no queries overlapping between sets.

4 Results

#### 4.1 Performance

As of 21 October 2025, Isaacus’ Kanon 2 Embedder legal embedding model ranks ﬁrst on MLEB out of 20 other models, with an NDCG@10 score of 86.03, followed by Voyage 3 Large at 85.71 and Voyage 3.5 at 84.07.

The full results of the benchmark are presented below. All scores are NDCG@10 scores. We report both the task average (i.e., by evaluation set) and the domain average.

7

[Figure 6]_images/imageFile6.png>)

Rank Model Task avg Domain avg Judicial Contractual Regulatory

[Figure 7]_images/imageFile7.png>)

- 1 Kanon 2 Embedder 86.03 86.37 82.96 84.67 91.48
- 2 Voyage 3 Large 85.71 86.08 82.33 86.05 89.87
- 3 Voyage 3.5 84.07 84.62 79.13 85.92 88.83
- 4 Qwen3 Embedding 8B 82.96 83.45 78.49 82.88 88.99
- 5 Voyage 3.5 Lite 82.41 82.96 77.46 83.21 88.19
- 6 Qwen3 Embedding 4B 81.96 82.61 76.14 84.19 87.49
- 7 Gemini Embedding 80.90 81.50 75.49 77.88 91.13
- 8 Voyage Law 2 79.63 79.75 78.55 81.65 79.07
- 9 Text Embedding 3 Large 78.91 79.31 75.24 79.70 83.00
- 10 Jina Embeddings v4 78.62 79.16 73.78 76.44 87.26
- 11 Qwen3 Embedding 0.6B 77.13 77.66 72.37 76.61 84.01
- 12 EmbeddingGemma 75.16 76.08 66.85 76.07 85.32
- 13 Snowﬂake Arctic Embed L v2.0 74.08 74.70 68.50 76.41 79.17
- 14 Snowﬂake Arctic Embed M v2.0 73.94 74.60 67.99 73.36 82.46
- 15 Text Embedding 3 Small 73.88 74.65 67.00 74.24 82.71
- 16 Text Embedding Ada 002 72.59 73.62 63.29 73.59 83.99
- 17 Granite Embedding English R2 72.45 73.12 66.48 69.14 83.73
- 18 BGE M3 69.44 70.60 59.00 73.04 79.75
- 19 Mxbai Embed Large v1 69.33 70.50 58.81 72.10 80.60
- 20 E5 Large Instruct 68.11 69.00 60.04 65.62 81.36
- 21 Granite Embedding Small English R2 68.06 68.88 60.67 64.49 81.48

[Figure 8]_images/imageFile8.png>)

- Table 2: Performance of 21 embedding models on MLEB. All scores are NDCG@10 values. Task average represents the mean across all ten evaluation sets; domain average represents the mean across domain types (weighted by number of datasets per domain). Performance is also shown broken down by the three domain types: Judicial, Contractual, and Regulatory.

T() 10 best )erf(rming m( els (n MLEB

0.95

0.90

0.85

D(main

Ju icial

Score(NDCG@10)

C(ntractual

Regulat(r1

0.80

0.75

0.70

Voyage3.5

VoyageLaw2

Voyage3Large

Voyage3.5Lite

GeminiEmbeing

Kanon2Embeer

JinaEmbeings.4

Qwen3Embeing8B

Qwen3Embeing4B

TextEmbeing3Large

- Figure 1: Performance of the top 10 embedding models on MLEB, broken down by domain type. Scores represent NDCG@10 averaged across all datasets within each domain (Judicial, Contractual, Regulatory). Legal domain-adapted models (Kanon 2 Embedder, Voyage 3 Large, Voyage 3.5, Voyage Law 2) show particularly strong performance.

We observe that the qualities that make an embedding model perform well at general multilingual information retrieval tasks are not necessarily the same as those that make a model perform well at legal information retrieval. Currently, Gemini Embedding ranks 1st on MTEB and Voyage 3.5 ranks 23rd, whereas on MLEB, Gemini is only 7th and Voyage 3.5 is 3rd.

We also observe that strong performance on MLEB correlates with legal domain adaptation. The top scoring models on MLEB, including Kanon 2 Embedder and Voyage 3 Large and Voyage

- 3.5, have all been optimized in some way for the legal domain. Kanon 2 Embedder, in particular, was pretrained and ﬁnetuned largely on legal documents. Voyage Law 2, despite being an older and likely smaller model than most other high scoring embedding models, ranks 8th, ahead of OpenAI Text Embedding 3 Large. Similar to Kanon 2 Embedder, Voyage Law 2 was optimized speciﬁcally for legal information retrieval.
- 4.2 Speed

We assessed the total evaluation time (including network latency) of commercial models on MLEB with a batch size of 16 for documents and 1 for queries (reﬂective of real-world conditions under which batching of queries is less likely), except when evaluating Voyage AI models on datasets with especially long sequences, in which case a batch size of 1 was used to avoid hitting token rate limits.

#### 4.3 Limitations

Unfortunately, we note that we were unable to evaluate any of Cohere’s embedding models as their terms of service expressly forbid benchmarking of any kind. Additionally, we note that there is some potential for data leakage in respect of the models of Voyage AI, Jina, and Google, as their terms of service opt either a portion or all of their API users by default into sharing data for training purposes (which would likely include benchmarks given that customers tend to conduct their own evaluations of models).

Sco) vs inf ) nc tim

Kanon 2 Emb dd )

Vo0ag 3 La)g

0.86

Vo0ag 3.5

0.84

Vo0ag 3.5 Lit

0.82

G mini Emb dding

Vo0ag La. 2

0.80

T xt Emb dding 3 La)g

Sco)e(NDCG@10)

0.78

0.76

T xt Emb dding 3 Small

0.74

P)ovid )

T xt Emb dding Ada 002

Googl

Isaacus

0.72

O( nAI

Vo0ag

0.70

20 50 100

Total inf r nc tim (minut s )

- Figure 2: NDCG@10 score versus total evaluation time for commercial embedding models on MLEB. Evaluation time includes network latency and reﬂects real-world conditions (batch size of 16 for documents, 1 for queries). The plot demonstrates the speed-accuracy tradeoﬀ among commercial oﬀerings.

### 5 Conclusion

We presented the Massive Legal Embedding Benchmark (MLEB), a high-quality, expert-annotated open-source benchmark for legal information retrieval. Consisting of ten evaluation sets across six jurisdictions (the US, UK, EU, Australia, Ireland, and Singapore) and ﬁve document types (cases, legislation, regulatory guidance, contracts, and literature), MLEB is the most comprehensive benchmark for legal embeddings to date. Furthermore, MLEB represents a signiﬁcant new contribution to the legal information retrieval landscape, with seven datasets being released for the ﬁrst time in order to increase the representation of jurisdictions and legal domains outside of American law and contract law.

MLEB, its constituent datasets, and evaluation code have all been publicly released under opensource licenses. See the Data Availability section for complete access information. We encourage reproductions of our results and intend to continue improving MLEB with subsequent expansions.

### 6 Data Availability

All datasets comprising MLEB are publicly available on Hugging Face (https://huggingface.co/isaacus) under open-source licenses. The evaluation code and benchmark implementation are available on GitHub (https://github.com/isaacus-dev/mleb). Direct links to each dataset are provided below:

- • Bar Exam QA https://huggingface.co/datasets/isaacus/mteb-barexam-qa

- • SCALR https://huggingface.co/datasets/isaacus/mleb-scalr
- • Singaporean Judicial Keywords https://huggingface.co/datasets/isaacus/singaporean-judicial-keywords
- • GDPR Holdings Retrieval https://huggingface.co/datasets/isaacus/gdpr-holdings-retrieval
- • Australian Tax Guidance Retrieval https://huggingface.co/datasets/isaacus/australian-tax-guidance-retrieval
- • Irish Legislative Summaries https://huggingface.co/datasets/isaacus/irish-legislative-summaries
- • UK Legislative Long Titles https://huggingface.co/datasets/isaacus/uk-legislative-long-titles
- • Contractual Clause Retrieval https://huggingface.co/datasets/isaacus/contractual-clause-retrieval
- • License TL;DR Retrieval https://huggingface.co/datasets/isaacus/license-tldr-retrieval
- • Consumer Contracts QA https://huggingface.co/datasets/isaacus/mleb-consumer-contracts-qa

### 7 Disclosures

In the interests of transparency, Kanon 2 Embedder was created by Isaacus, a foundational legal AI research company, which also sponsored the creation of MLEB.

### References

- [1] Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. The distracting eﬀect: Understanding irrelevant passages in rag, 2025.
- [2] Paheli Bhattacharya, Kripabandhu Ghosh, Saptarshi Ghosh, Arindam Pal, Parth Mehta, Arnab Bhattacharya, and Prasenjit Majumder. Overview of the ﬁre 2019 aila track: Artiﬁcial intelligence for legal assistance. In FIRE 2019 Working Notes, volume 2517, pages 1–12. CEUR-WS.org, 2020.
- [3] Moses S Charikar. Similarity estimation techniques from rounding algorithms. In Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing (STOC ’02), pages 380–388. ACM, 2002.
- [4] Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzemiński, Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaeﬀer, Gabriel Sequeira, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, Ömer Çağatan, Akash Kundu, Martin Bernstorﬀ, Shitao Xiao, Akshita Sukhlecha, Bhavish Pahwa, Rafał Poświata, Kranthi Kiran GV,

- Shawon Ashraf, Daniel Auras, Björn Plüster, Jan Philipp Harries, Loïc Magne, Isabelle Mohr, Mariya Hendriksen, Dawei Zhu, Hippolyte Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Konrad Wojtasik, Taemin Lee, Marek Šuppa, Crystina Zhang, Roberta Rocca, Mohammed Hamdy, Andrianos Michail, John Yang, Manuel Faysse, Aleksei Vatolin, Nandan Thakur, Manan Dey, Dipam Vasani, Pranjal Chitale, Simone Tedeschi, Nguyen Tai, Artem Snegirev, Michael Günther, Mengzhou Xia, Weijia Shi, Xing Han Lù, Jordan Clive, Gayatri Krishnakumar, Anna Maksimova, Silvan Wehrli, Maria Tikhonova, Henil Panchal, Aleksandr Abramov, Malte Ostendorﬀ, Zheng Liu, Simon Clematide, Lester James Miranda, Alena Fenogenova, Guangyu Song, Ruqiya Bin Saﬁ, Wen-Ding Li, Alessia Borghini, Federico Cassano, Hongjin Su, Jimmy Lin, Howard Yen, Lasse Hansen, Sara Hooker, Chenghao Xiao, Vaibhav Adlakha, Orion Weller, Siva Reddy, and Niklas Muennighoﬀ. Mmteb: Massive multilingual text embedding benchmark, 2025.
- [5] GDPRHub Contributors. GDPRHub. https://gdprhub.eu. Accessed: October 2025.
- [6] Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya Narayana, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John Nay, Jonathan H. Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael Livermore, Nikon Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, and Zehua Li. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models, 2023. SCALR task created by Faiz Surani and Varun Iyer.
- [7] Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. Cuad: An expert-annotated nlp dataset for legal contract review. NeurIPS, 2021. https://github.com/TheAtticusProject/cuad.
- [8] Christoph Hoppe, David Pelkmann, Nico Migenda, Daniel Hötte, and Wolfram Schenck. Towards intelligent legal advisors for document retrieval and question-answering in german legal documents. In 2021 IEEE Fourth International Conference on Artiﬁcial Intelligence and Knowledge Engineering (AIKE), pages 29–32. IEEE, 2021.
- [9] Noam Kolt. Predicting consumer contracts. Berkeley Tech. LJ, 37:71, 2022.
- [10] Yuta Koreeda and Christopher D Manning. Contractnli: A dataset for documentlevel natural language inference for contracts. In Findings of EMNLP, 2021. https://stanfordnlp.github.io/contract-nli/.
- [11] Haitao Li, Yunqiu Shao, Yueyue Wu, Qingyao Ai, Yixiao Ma, and Yiqun Liu. Lecardv2: A large-scale chinese legal case retrieval dataset, 2023.
- [12] Laura Manor and Junyi Jessy Li. Plain english summarization of contracts. In Proceedings of the Natural Legal Language Processing Workshop 2019, pages 1–11. Association for Computational Linguistics, 2019.
- [13] John J Nay. Large language models as corporate lobbyists, 2023. Available at SSRN 4316615.
- [14] Nicholas Pipitone and Ghita Houir Alami. Legalbench-rag: A benchmark for retrievalaugmented generation in the legal domain, 2024.

- [15] Abhilasha Ravichander et al. Privacyqa: A dataset for privacy policy question answering. In EMNLP, 2019. https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP.
- [16] Steven H Wang, Antoine Scardigli, Leonard Tang, Wei Chen, Dimitry Levkin, Anya Chen, Spencer Ball, Thomas Woodside, Oliver Zhang, and Dan Hendrycks. Maud: An expertannotated legal nlp dataset for merger agreement understanding, 2023.
- [17] Albert Weichselbraun. Inscriptis - a python-based html to text conversion library optimized for knowledge extraction from the web. Journal of Open Source Software, 6(66):3557, 2021.
- [18] Marco Wrzalik and Dirk Krechel. Gerdalir: A german dataset for legal information retrieval. In Proceedings of the Natural Legal Language Processing Workshop 2021, pages 123–128. Association for Computational Linguistics, 2021.
- [19] Lucia Zheng, Neel Guha, Javokhir Arifov, Sarah Zhang, Michal Skreta, Christopher D. Manning, Peter Henderson, and Daniel E. Ho. A reasoning-focused legal retrieval benchmark. In Proceedings of the Symposium on Computer Science and Law on ZZZ, CSLAW ’25, page 169–193. ACM, March 2025.
- [20] Lucia Zheng, Neel Guha, Javokhir Arifov, Sarah Zhang, Michal Skreta, Christopher D Manning, Peter Henderson, and Daniel E Ho. A reasoning-focused legal retrieval benchmark, 2025.

### Appendix A Contractual Clause Types

The following table lists all 45 contractual clause types and their corresponding NLI-style statements used in the Contractual Clause Retrieval evaluation set.

[Figure 9]_images/imageFile9.png>)

Type Statement Set-oﬀ clause This is a contractual provision that permits a contracting

[Figure 10]_images/imageFile10.png>)

party to deduct liabilities owed to it by the counterparty from liabilities it owes to the counterparty.

Authority to sign clause This is a contractual provision that represents or warrants that a contracting party has authority to bind the entity it represents.

Power of attorney This is a contractual provision that grants a party a power of attorney to act on behalf of a contracting party.

Termination for cause clause This is a contractual provision that allows a contracting party to terminate the contract in the event of a breach or default by the counterparty.

Good faith clause This is a contractual provision that obligates a contracting party to act in good faith.

Payment currency clause This is a contractual provision that speciﬁes the currency in which payment is to be made.

Termination for change of control clause

This is a contractual provision that permits a contracting party to terminate the contract in the event of a change in the ownership or control of the counterparty.

Conditions precedent clause This is a contractual provision that speciﬁes conditions to be satisﬁed for obligations or eﬀectiveness to arise.

Counterparts clause This is a contractual provision that permits the contract to be executed in separate counterparts.

Workplace surveillance clause This is a contractual provision that states that an employer may surveil their employee.

No improvements clause This is a contractual provision that restricts a contracting party from making improvements to property.

Termination for insolvency clause

This is a contractual provision that allows a contracting party to terminate the contract in the event of the bankruptcy or insolvency of the counterparty.

Exclusivity clause This is a contractual provision that restricts a contracting party from dealing with other parties apart from the counterparty within a particular scope or territory.

Termination for force majeure clause

This is a contractual provision that allows a contracting party to terminate the contract in the event of conditions beyond their control.

[Figure 11]_images/imageFile11.png>)

Vesting clause This is a contractual provision that speciﬁes the schedule or conditions under which interests vest.

Termination for convenience clause

This is a contractual provision that allows a contracting party to terminate the contract for any reason.

Drag along clause This is a contractual provision that permits majority holders to require minority holders to sell their interests on the same terms in a sale.

Cumulative rights clause This is a contractual provision that speciﬁes rights and remedies are cumulative, not exclusive of those provided by law.

Lock-up clause This is a contractual provision that restricts a contracting party from selling or transferring securities for a period after issuance or listing.

Break fee clause This is a contractual provision that requires a contracting party to pay the counterparty in the event of failure to complete a transaction.

Waiver of jury trial clause This is a contractual provision that waives a contracting party’s right to a jury trial.

Return or destruction of materials clause

This is a contractual provision that requires a contracting party to return or destroy material.

Disclaimer This is a contractual provision that disclaims responsibility

for warranties, representations, liabilities or obligations. Novation clause This is a contractual provision that permits the

substitution of a new party in place of a contracting party. Conﬁdentiality clause This is a contractual provision that restricts the use of

information protected by a duty of conﬁdence. Injunctive relief clause This is a contractual provision that entitles a party to seek injunctive relief. Moral rights waiver clause This is a contractual provision that waives a contracting party’s moral rights in intellectual property.

Probation clause This is a contractual provision that permits an employer to terminate an employee without cause or notice during a probationary period.

Escalation to senior management clause

This is a contractual provision that permits the escalation of disputes to senior management.

This is a contractual provision that limits the liability of a contracting party for conditions beyond their control.

Limitation of liability for force majeure clause

[Figure 14]_images/imageFile14.png>)

Late payment interest clause This is a contractual provision that requires the payment of interest on overdue liabilities.

Choice of venue clause This is a contractual provision that speciﬁes the jurisdiction in which disputes over the contract should be resolved.

Clawback clause This is a contractual provision that entitles a contracting party to recover amounts previously paid to the counterparty.

No assignment clause This is a contractual provision that prohibits assignment of rights or obligations.

Cure period clause This is a contractual provision that speciﬁes a period during which a breaching contracting party may remedy or cure their breach.

Penalty clause This is a contractual provision that imposes a penalty on a contracting party in the event of a failure to perform a contractual obligation.

Suspension clause This is a contractual provision that entitles a contracting party to suspend their obligations.

Obligatory compliance with all applicable laws clause

This is a contractual provision that obligates a contracting party to comply with all relevant laws.

Further assurances clause This is a contractual provision that requires a contracting party to take necessary steps or actions to give eﬀect to the contract.

IP assignment clause This is a contractual provision that assigns intellectual property rights.

Tag along clause This is a contractual provision that entitles minority holders to participate in a sale by majority holders on the same terms.

Precedence of documents clause

This is a contractual provision that speciﬁes the order of precedence that parts of the contract or other documents take in interpretation of the agreement.

Bonus clause This is a contractual provision that provides for the payment of bonuses.

Right of ﬁrst refusal clause This is a contractual provision that entitles a contracting party to match an oﬀer before an interest is transferred.

Most favored nation clause This is a contractual provision that requires a contracting party to oﬀer the counterparty goods, services or interests on terms at least as favorable as those granted to anyone else.

[Figure 17]_images/imageFile17.png>)

