## CoverBench: A Challenging Benchmark for Complex Claim Verification

Alon Jacovi1 Moran Ambar1 Eyal Ben-David1 Uri Shaham1 Amir Feder1 Mor Geva1,2 Dror Marcus1 Avi Caciularu1 1Google Research 2Tel Aviv University alonjacovi@google.com

CoverBench: go/coverbench-paper

Abstract

Information-Rich Contexts

[Figure 1]

[Figure 2]

| | |
|---|---|

There is a growing line of research on verifying the correctness of language models’ outputs. At the same time, LMs are being used to tackle complex queries that require reasoning. We introduce CoverBench, a challenging benchmark focused on verifying LM outputs in complex reasoning settings. Datasets that can be used for this purpose are often designed for other complex reasoning tasks (e.g., QA) targeting specific use-cases (e.g., financial tables), requiring transformations, negative sampling and selection of hard examples to collect such a benchmark. CoverBench provides a diversified evaluation for complex claim verification in a variety of domains, types of reasoning, relatively long inputs, and a variety of standardizations, such as multiple representations for tables where available, and a consistent schema. We manually vet the data for quality to ensure low levels of label noise. Finally, we report a variety of competitive baseline results to show CoverBench is challenging and has very significant headroom. The data is available at https://huggingface.

# arXiv:2408.03325v2[cs.CL]26Nov2024

Wiki, Finance, Statistics Tables

Legal Contracts

Biomedical Abstracts

(etc.)

Complex Reasoning Claims Quantitative: Three companies (X, Y, Z) saw revenue increase 5%. Multi-hop: The lead actor in X’s first movie was Y.

+ long context reasoning, domain expertise, structured data…

Figure 1: CoverBench contains true and false claims that require implicit complex reasoning to verify in a variety of domains and settings.

sidered a reduction from NLI (Dagan et al., 2005; Bowman et al., 2015) or AIS (Rashkin et al., 2021).

In this work, we focus on complex claims. Naturally, as LMs are used frequently to solve complex queries (Suzgun et al., 2022), their verifying their outputs’ correctness may require multiple hops of reasoning (Geva et al., 2021), quantitative reasoning (Lewkowycz et al., 2022), domain expertise (Magesh et al., 2024), and so on, based on the reasoning level required in the original query. Are general open-ended complex reasoning tasks and complex claim verification equivalent? We argue that there is a difference: By focusing on a binary classification of given statements in a well-defined context, verifiers can—and should—be held to a higher standard compared to the source task. We propose such a standard in this work.

co/datasets/google/coverbench.

### 1 Introduction

Recent work has focused on measuring various properties of language models’ outputs (Leiter et al., 2022; Golovneva et al., 2022). One established property to measure is the correctness of generated text, against its context (Tang et al., 2024) or external sources (e.g., fact-checking, Pan et al., 2023a; Chen et al., 2023). This means, for example, to verify whether a summary correctly refers to its source document (Bishop et al., 2023; Krishna et al., 2023), or that some logical inference has been made correctly based on claims that can be verified (Jacovi et al., 2024). We refer to this as claim verification, where the claim is a falsifiable statement to be verified against a given grounding context (Honovich et al., 2022). It can be con-

To evaluate complex claim verification solutions, we collect CoverBench—a benchmark for this task—by leveraging a diverse set of nine datasets (§2) across different settings that require complex reasoning (Fig. 1). The benchmark targets a variety of language domains (Wikipedia, finance, biomedical, legal, statistics, and others), sources of complexity (structured data, quantitative reasoning,

###### Dataset Domain Task Sources of Complexity

FinQA (Chen et al., 2021) Finance QA Quantitative, multi-step, tables QRData (Liu et al., 2024) Statistics QA Long context, quantitative, multi-step, domain expertise, tables TabFact (Chen et al., 2020a) Wikipedia Verification Multi-step, tables MultiHiertt (Zhao et al., 2022) Finance QA Long context, quantitative, multi-step, tables HybridQA (Chen et al., 2020b) Wikipedia QA Very long context, tables ContractNLI (Chen et al., 2020b) Legal NLI Long context, domain expertise PubMedQA (Jin et al., 2019) Biomedical QA Domain expertise TACT (Caciularu et al., 2024) Various QA Quantitative, multi-step, tables Feverous (Aly et al., 2021) Wikipedia Verification Multi-step, quantitative, tables

Table 1: An overview of the datasets used in CoverBench (§2).

multi-step reasoning, domain expertise, reasoning over long context), and difficulty and quality (via various filtering steps, both manual and automatic).

To build CoverBench (§3), we convert all tasks to a unified format with declarative claims, metadata about the required type of reasoning, and parsing and standardization of all table representations (we use HTML, JSON, and Markdown). We additionally carefully sample false claims and challenging examples: In particular, challenging examples are selected through leveraging metadata and model-based selection. And similarly to TinyBenchmarks (Maia Polo et al., 2024), we select an efficient subset of examples that will be the most representative for model ranking.

CoverBench contains 733 examples of rich grounding contexts and complex claims based on them with correctness labels. The contexts are lengthy, with an average of 3,500 tokens. Many of the evaluated models, including recent competitive LMs, achieve performance near the random baseline, despite manual vetting we have done to ensure that the tasks are solvable. The best models achieve below a 70 Macro-F1 score, while smaller models—7–13 billion parameter LMs—achieve performance at the random baseline level. These results show a significant headroom for the task. The data is available at https://huggingface. co/datasets/google/coverbench.

### 2 Benchmark Scope

For our new benchmark about complex claim verification, we prioritize variety and difficulty. This section explains the facet of variety.

Domains. We aim to include a variety of language domains. In particular: The financial, Wikipedia, biomedical, and legal domains are well-represented in datasets for complex reasoning. A small quantity of examples in a large variety of other domains,

such as statistical inference and literature, were included as well, within their limited availability.

Sources of complexity. Complex reasoning can colloquially refer to many different tasks in practice. In the scope of this paper, we consider complex reasoning to include: (1) Reasoning over structured data (in particular, tables); (2) Reasoning over a long context; (3) Quantitative reasoning (calculation, aggregation, counting, and so on); (4) Reasoning requiring domain expertise; (5) Multi-hop reasoning (inter-dependent steps of reasoning). Importantly, we are interested not only in examples that exhibit any source of complexity, but specifically many unique combinations of them.

Datasets. In Tab. 1 we describe the datasets in CoverBench. Some contexts with tables (e.g., in MultiHiertt) contain a mix of table and text. Where possible, we leverage metadata to select the examples that require reasoning (e.g., as FinQA contains the answers’ calculation, we can select examples with multiple steps). Two originally-included datasets, SciTab (Lu et al., 2023) and REVEAL (Jacovi et al., 2024), were excluded during a manual inspection process described in §3. In Feverous, different examples can exhibit different sources of complexity, and we select examples that exhibit at least one. Extended details are available in §A.

### 3 Constructing CoverBench

We describe the process of collecting and building our benchmark. This involves conversion to a unified schema, negative sampling with seed models, and careful selection of informative examples.

#### 3.1 Conversion to the Schema

Many of the datasets described in §2 require nuanced transformation into the claim verification setting. Below are relevant details that require at-

tention in this transformation.

Schema. Each example in CoverBench contains the following: The grounding context; a falsifiable claim in the form of a declarative statement; a binary entailment label; the language domain of the instance; and the sources of complexity that make this instance require complex reasoning (see §2).

QA pairs to declarative statements. Claims which come in the format of a QA pair (e.g., “Q: What is the capital of France? A: Paris.”) were converted to declarative form (“The capital of France is Paris.”) through both manual annotation and a prompted LM (all generated statements were manually verified; details in §A).

Representing structure. The various table formats across datasets were parsed into a standard format, which we represented in one of three text formats, chosen at random: HTML, JSON, or Markdown.

#### 3.2 Sampling Negative Examples

TabFact, ContractNLI, PubMedQA, and Feverous are datasets that contain both true and false claims (in the case of PubMedQA, since answers are binary, the inverse answer can also be used.).

FinQA, QRData, MultiHiertt, HybridQA and TACT are QA datasets—and so they contain only positive examples as QA pairs. For these tasks we require a method to generate negative examplesclaims not entailed by the context. The negative examples should be difficult, which precludes simpler heuristics for negative sampling (Li et al., 2019).

Given the original QA format, a simple method to derive difficult negative cases is to use the question to generate models’ answers. The answers are compared to the gold answer, and if they are wrong, the new QA pair is selected as a negative example. These negative answers represent real model errors, so they are likely difficult for models to verify.

We used the following three seed models: GPT-4o (OpenAI et al., 2024), gemma-1.1-7b1.1-it (Gemma et al., 2024), and Mixtral-8x7bInstruct (Jiang et al., 2024). After extracting the classification from the models’ full answer, we compared the gold and model answers in two ways: Lowercased exact match (percentages, currencies other numbers were normalized to digits), and using a prompted LM judge (we used Mixtral-8x7b, and manually verified perfect accuracy for this simple task on a representative sample of 100 cases). The generated answer was considered incorrect if it was judged as different for both heuristics.

#### 3.3 Example Selection

Following the methodology above, we derived a total of roughly 7,000 examples. From this larger set, we selected a subset of examples with three goals: (I) Most difficult examples, without introducing label noise bias. (II) Examples that are least likely to suffer from memorization via data contamination. (III) Examples that are most indicative of the difference between models’ capabilities.

To clarify goal (I): It is expected that all datasets have annotation errors (Klie et al., 2022). E.g., if we simply select examples of model mistakes, while this will select difficult examples (Klie et al., 2023), it would also select a greater ratio of incorrectly-labeled examples. Thus, we cannot select examples of model mistakes deliberately.

- Goals (I) and (II) can be targeted by selecting examples that are incorrectly predicted by models in the claim-only baseline setting. In this setting, only the claim is provided to the model without its grounding context, so the example is intractable, which avoids the incorrect label bias. If the model is correct—it is via a random guess, some shallow heuristic (McCoy et al., 2019), or data contamination (Deng et al., 2023).1 By choosing examples where at least two models were incorrect in this baseline, we reduce the possibility of discarding correctness via a random guess.
- Goals (II) and (III) can be targeted by selecting model disagreements. Since on any disagreement one model was correct and one was incorrect, we can select examples—without using gold labelswhich demonstrably differentiate between models.

We made both selections by using two comparable-performance seed models: Mixtral8x7b-Instruct and Starling-LM-7B-beta-ExPO (Zhu et al., 2023). Performance on the final subset was 3 to 7 Macro-F1 points lower than on a random subset in our testing (§B).

#### 3.4 Vetting

Due to the various steps described above, we require a phase of manual inspection to check for solvability of the derived data and diagnose possible issues. We selected 10 examples at random from each dataset for a manual vetting phase. The

1We cannot guarantee against data contamination within the scope of this paper, since many models use hidden training data. Nevertheless, we rely on overall similarities between models and their similar internet-derived training data. We refer to the individual model developers to index CoverBench and check for contamination when reporting evaluations on it.

TACT QRData

HybridQA

Feverous

MultiHiertt

ContractNLI

PubMedQA

TabFact

FinQA

| | | | | | |
|---|---|---|---|---|---|

Wikipedia Finance

Biomed Legal

Statistics Other (17)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 2: Distribution of the source datasets and the text domains in CoverBench.

datasets had between 0 and 1 incorrect labels, indicating a rate of incorrect labels comparable with the academic datasets that the data is based on.

During this phase, two datasets originally selected were omitted: SciTab (Lu et al., 2023) was omitted due to a high level of label noise, and REVEAL (Jacovi et al., 2024) was omitted due to a loss of difficulty (notably, this loss of difficulty was an artifact of the conversion, and not present in the original data). PubMedQA also exhibited a loss of difficulty during the conversion process via a correlation between negation in the claim and a non-entailment label. However, this issue was neutralized with the claim-only sampling step (§3.3).

#### 3.5 The CoverBench Benchmark

The final challenge set contains 733 examples. Fig. 2 shows the distribution of the source datasets and domains. The examples have 3,500 tokens on average via the Mixtral-8x7b-Instruct tokenizer. The overall label distribution is balanced at 45:55 towards the positive class (58:42 to 44:56 per source dataset). More details are in §A.

### 4 Experimental Setup and Results

In this section, we describe the experiments and results we made to evaluate the difficulty of CoverBench. The Macro-F1 results are in Tab. 2. Overall, we see that a variety of competitive models struggle on this task. Fine-grained details of the experiment setting and our approach are available in §B.

Baseline details. The baselines use prompted LMs (Gemma et al., 2024; 01.AI et al., 2024; Qwen, 2024a,b; Reid et al., 2024; Touvron et al.,

###### Model 0-shot 0-shot CoT

NLI-Entailment-Verifier-xxl 43.7 T5-11b-TrueTeacher&ANLI 48.2 —

∗ Gemma-1.1-7b-it 43.4 46.1 ∗ Mixtral-8x7B-Instruct 45.3 49.0 Llama-2-13b-Chat 48.3 48.7 Qwen1.5-14B-Chat 49.0 50.8 ∗ Starling-LM-7B-beta-ExPO 47.7 52.1 Llama-2-70b-Chat 50.3 52.6 Yi-1.5-34B-Chat 51.0 54.2 Gemini 1.5 Flash 54.4 54.5 Qwen2-72B-Instruct 54.3 57.0 Gemini 1.5 Pro (05/2024) 59.9 62.1 ∗ GPT-4o (10/2024) 62.7 66.5 Claude 3.5 Sonnet (10/2024) 67.2 68.9

Table 2: Macro-F1 performance on CoverBench. 50 is the random baseline threshold, and below 50 implies a class bias, where 0 is the majority baseline. (∗) denotes seed models for sampling—we include their results for completeness, but note that they are unreliable.

2023; Anthropic, 2024) and off-the-shelf NLI classifiers (Gekhman et al., 2023; Sanyal et al., 2024). The LMs were prompted in 0-shot and 0-shot Chain-of-Thought (Wei et al., 2023) formats, after extensive “prompt engineering” using prompts from previous works and trial-and-error. While we made significant attempts with few-shot settings, none succeeded, likely due to the fact that the examples in CoverBench are often long (we have tested both simplified and full-length demonstrations), as few-shot is known to struggle in longcontext (Jiang et al., 2023; Cattan et al., 2024). For the NLI classifiers, contradiction and neutral were considered as not entailed. In cases where the context length exceeded the model’s context window, if the model did not include an extrapolation technique, the context was trimmed from its beginning.

### 5 Conclusions

We collect a new claim verification benchmark specifically targeting difficulty and complexity of reasoning, towards the goal of developing not only models with good complex reasoning, but better verification to classify when a generated claim is false. CoverBench involves a wide variety of domains and sources of complexity, has long inputs with thousands of tokens on average, and is relatively efficient in size. The benchmark provides a significant challenge to current models, and can serve as a groundwork for future work in the area.

### Limitations

Domain-specific LMs. In this work, we focused on validating the difficulty of the benchmark based on readily-available off-the-shelf LMs. Some of the tasks will likely be better addressed by LMs that use specialized tools or specialized prompting techniques (Kim et al., 2024), LMs that were specifically trained for a specific domain such as finance (Wu et al., 2023), and so on (Pan et al., 2023b). Our goal is to measure general ability in diverse settings, but specialized use-cases may benefit from investigating specialized models on a the relevant subset of CoverBench.

Data contamination disclaimer. As mentioned in the paper, despite some steps we took against this (such as generating negative examples, converting QA to declarative statements, converting tables to different formats than those in the original datasets, and sampling less-memorized examples), the problem of data contamination limits the ability of evaluation datasets currently. Evaluators using the data should take care to check for the individual examples’ presence in their models’ training data, if they have access to it, and otherwise they should not rely on the metrics for critical decisions about models whose training data is unknown. The role of the evaluations in this work is strictly to illustrate a robust general property of the data as a difficult evaluation set with large headroom.

### References

01.AI, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. 2024. Yi: Open foundation models by 01.ai. Preprint, arXiv:2403.04652.

Rami Aly, Zhijiang Guo, Michael Sejr Schlichtkrull, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Oana Cocarascu, and Arpit Mittal. 2021. FEVEROUS: Fact extraction and VERification over unstructured and structured information.

Anthropic. 2024. The Claude 3 Model Family: Opus, Sonnet, Haiku.

Jennifer A Bishop, Qianqian Xie, and Sophia Ananiadou. 2023. Longdocfactscore: Evaluating the fac-

tuality of long document abstractive summarisation. ArXiv, abs/2309.12455.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Conference on Empirical Methods in Natural Language Processing.

Avi Caciularu, Alon Jacovi, Eyal Ben David, Sasha Goldshtein, Tal Schuster, Jonathan Herzig, Gal Elidan, and Amir Globerson. 2024. Tact: Advancing complex aggregative reasoning with information extraction tools.

Arie Cattan, Alon Jacovi, Alex Fabrikant, Jonathan Herzig, Roee Aharoni, Hannah Rashkin, Dror Marcus, Avinatan Hassidim, Yossi Matias, Idan Szpektor, and Avi Caciularu. 2024. Can few-shot work in long-context? recycling the context to generate demonstrations. Preprint, arXiv:2406.13632.

Shiqi Chen, Yiran Zhao, Jinghan Zhang, Ethan Chern, Siyang Gao, Pengfei Liu, and Junxian He. 2023. Felm: Benchmarking factuality evaluation of large language models. ArXiv, abs/2310.00741.

Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. 2020a. Tabfact: A large-scale dataset for table-based fact verification. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Wang. 2020b. Hybridqa: A dataset of multi-hop question answering over tabular and textual data. Findings of EMNLP 2020.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al. 2021. Finqa: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711.

Ido Dagan, Oren Glickman, and Bernardo Magnini. 2005. The PASCAL recognising textual entailment challenge. In Machine Learning Challenges, Evaluating Predictive Uncertainty, Visual Object Classification and Recognizing Textual Entailment, First PASCAL Machine Learning Challenges Workshop, MLCW 2005, Southampton, UK, April 11-13, 2005, Revised Selected Papers, volume 3944 of Lecture Notes in Computer Science, pages 177–190. Springer.

Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark B. Gerstein, and Arman Cohan. 2023. Investigating data contamination in modern benchmarks for large language models. ArXiv, abs/2311.09783.

Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, and Idan Szpektor. 2023. Trueteacher: Learning factual consistency evaluation with large language models. Preprint, arXiv:2305.11171.

Team Gemma, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex CastroRos, Ambrose Slone, Amélie Héliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Clément Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikuła, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vlad Feinberg, Wojciech Stokowiec, Yu hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Warkentin, Ludovic Peran, Minh Giang, Clément Farabet, Oriol Vinyals, Jeff Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. 2024. Gemma: Open models based on gemini research and technology. Preprint, arXiv:2403.08295.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346– 361.

O. Yu. Golovneva, Moya Chen, Spencer Poff, Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. 2022. Roscoe: A suite of metrics for scoring step-by-step reasoning. ArXiv, abs/2212.07919.

Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. 2022. True: Re-evaluating factual consistency evaluation. ArXiv, abs/2204.04991.

Alon Jacovi, Yonatan Bitton, Bernd Bohnet, Jonathan Herzig, Or Honovich, Michael Tseng, Michael Collins, Roee Aharoni, and Mor Geva. 2024. A chain-of-thought is as strong as its weakest link: A benchmark for verifiers of reasoning chains. Preprint, arXiv:2402.00559.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, MarieAnne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of experts. Preprint, arXiv:2401.04088.

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. ArXiv, abs/2310.06839.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. 2019. Pubmedqa: A dataset for biomedical research question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577.

Joongwon Kim, Bhargavi Paranjape, Tushar Khot, and Hanna Hajishirzi. 2024. Husky: A unified, opensource language agent for multi-step reasoning.

Jan-Christoph Klie, Bonnie Webber, and Iryna Gurevych. 2023. Annotation Error Detection: Analyzing the Past and Present for a More Coherent Future. Computational Linguistics, 49(1):157–198.

Jan-Christoph Klie, Bonnie Lynn Webber, and Iryna Gurevych. 2022. Annotation error detection: Analyzing the past and present for a more coherent future. Computational Linguistics, 49:157–198.

Yuta Koreeda and Christopher Manning. 2021. ContractNLI: A dataset for document-level natural language inference for contracts. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 1907–1919, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Kalpesh Krishna, Erin Bransom, Bailey Kuehl, Mohit Iyyer, Pradeep Dasigi, Arman Cohan, and Kyle Lo. 2023. LongEval: Guidelines for human evaluation of faithfulness in long-form summarization. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1650–1669, Dubrovnik, Croatia. Association for Computational Linguistics.

Christoph Leiter, Piyawat Lertvittayakumjorn, M. Fomicheva, Wei Zhao, Yang Gao, and Steffen Eger. 2022. Towards explainable evaluation metrics for natural language generation. ArXiv, abs/2203.11131.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam

Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models. ArXiv, abs/2206.14858.

Jia Li, Chongyang Tao, Wei Wu, Yansong Feng, Dongyan Zhao, and Rui Yan. 2019. Sampling matters! an empirical study of negative sampling strategies for learning of matching models in retrievalbased dialogue systems. In Conference on Empirical Methods in Natural Language Processing.

Xiao Liu, Zirui Wu, Xueqing Wu, Pan Lu, Kai-Wei Chang, and Yansong Feng. 2024. Are llms capable of data-based statistical and causal reasoning? benchmarking advanced quantitative reasoning with data. arXiv preprint arXiv:2402.17644.

Xinyuan Lu, Liangming Pan, Qian Liu, Preslav Nakov, and Min-Yen Kan. 2023. Scitab: A challenging benchmark for compositional reasoning and claim verification on scientific tables. Preprint, arXiv:2305.13186.

Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, and Daniel E. Ho. 2024. Hallucination-free? assessing the reliability of leading ai legal research tools. Preprint, arXiv:2405.20362.

Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. 2024. tinybenchmarks: evaluating llms with fewer examples. arXiv preprint arXiv:2402.14992.

R. Thomas McCoy, Ellie Pavlick, and Tal Linzen. 2019. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In Annual Meeting of the Association for Computational Linguistics.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha GontijoLopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane

Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Liangming Pan, Xiaobao Wu, Xinyuan Lu, Anh Tuan Luu, William Yang Wang, Min-Yen Kan, and Preslav Nakov. 2023a. Fact-checking complex claims with program-guided reasoning. ArXiv, abs/2305.12744.

Liangming Pan, Xiaobao Wu, Xinyuan Lu, Anh Tuan Luu, William Yang Wang, Min-Yen Kan, and Preslav Nakov. 2023b. Fact-checking complex claims with program-guided reasoning. Preprint, arXiv:2305.12744.

The pandas development team. 2020. pandasdev/pandas: Pandas.

- Team Qwen. 2024a. Introducing qwen1.5.
- Team Qwen. 2024b. Qwen2 technical report.

Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and D. Reitter. 2021. Measuring attribution in natural language generation models. Computational Linguistics, 49:777–840.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Soumya Sanyal, Tianyi Xiao, Jiacheng Liu, Wenya Wang, and Xiang Ren. 2024. Are machines better at complex reasoning? unveiling human-machine inference gaps in entailment verification. Preprint, arXiv:2402.03686.

Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed Huai hsin Chi, Denny Zhou, and Jason Wei. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. In Annual Meeting of the Association for Computational Linguistics.

Liyan Tang, Philippe Laban, and Greg Durrett. 2024. Minicheck: Efficient fact-checking of llms on grounding documents. ArXiv, abs/2404.10774.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten,

Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. Preprint, arXiv:2307.09288.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. Preprint, arXiv:2201.11903.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Huggingface’s transformers: State-of-the-art natural language processing. Preprint, arXiv:1910.03771.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023. Bloomberggpt: A large language model for finance. ArXiv, abs/2303.17564.

Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. 2022. MultiHiertt: Numerical reasoning over multi hierarchical tabular and textual data. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6588–6600, Dublin, Ireland. Association for Computational Linguistics.

Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, Karthik Ganesan, Wei-Lin Chiang, Jian Zhang, and Jiantao Jiao. 2023. Starling-7b: Improving llm helpfulness & harmlessness with rlaif.

### A Benchmark Collection

Benchmark statistics are given in Tab. 3 and Fig. 3. Examples from each dataset in the benchmark are in Tab. 5.

The 17 domains which are included under “Other” in Fig. 2, and are represented by very few examples in the dataset (a total of 31 examples), are: business, media, culinary, emergency, environment, fashion, medicine, archaeology, ethics, real-estate, politics, technology, biology, transportation, retail, education, and architecture. These domains are all from the TACT dataset which has a wide variety of domains.

#### A.1 Datasets

Below we describe each dataset and what it was used for. The public test sets were used when available, and otherwise, the public dev sets. Each instance of CoverBench contains the precise ID of where the instance came from with respect to the source datasets.

FinQA (Chen et al., 2021) includes QA pairs that require quantitative reasoning (with a numerical answer) over a medium-length text and a table extracted from a financial report. Since the data includes the entire calculation behind the answer, we selected QA pairs that require at least two steps of calculations.

QRData (Liu et al., 2024) contains QA pairs that require statistical inference or causal inference over a large table. We selected QA pairs that require statistics, as difficult examples of quantitative reasoning.

TabFact (Chen et al., 2020a) contains Wikipedia tables and declarative statements based on the tables, some of which require reasoning over multiple cells in the table (e.g., aggregation or arg-max). We manually selected examples that require reasoning over multiple cells.

MultiHiertt (Zhao et al., 2022) similarly to FinQA includes QA pairs that require multi-step quantitative reasoning, with a numerical answer, in the finance domain. The contexts in MultiHiertt are long, with multiple tables (some of them hierarchical) interspersed throughout a long document.

HybridQA (Chen et al., 2020b) includes multihop QA pairs over a mix of text and a table in the Wikipedia domain.

ContractNLI (Koreeda and Manning, 2021) in-

Dataset Tokens Label Ratio (+) Size

ContractNLI 2,572 42.7% 12.1% Feverous 5,204 56.2% 16.5% FinQA 1,252 57.3% 16.0% HybridQA 19,297 55.8% 5.9% MultiHiertt 4,829 58.3% 8.2% PubMedQA 360 56.3% 16.2% QRData 5,512 57.5% 5.5% TACT 314 56.3% 4.4% TabFact 1,239 56.3% 15.3%

Table 3: Statistics per source dataset.

cludes natural language inference examples, where the premises are non-disclosure agreements, and the hypotheses are a set of 17 standardized conclusions. In the majority of cases, domain expertise is required to derive the correct label.

PubMedQA (Jin et al., 2019) includes binary (true-false) QA pairs over PubMed abstracts. A large quantity of examples require reasoning over multiple sentences to answer, and all require domain expertise.

TACT (Caciularu et al., 2024) includes text-table alignments alongside QA pairs that require multiple complex calculations, formalized as chains of table-manipulation queries, in a large variety of domains.

Feverous (Aly et al., 2021) includes claims over Wikipedia documents, many of which include tables and quantitative reasoning or multi-step reasoning. We selected the examples that require tables, and either quantitative reasoning or multi-step reasoning.

#### A.2 Conversion to the Schema

Conversion to declarative statements. For generating declarative statements, we used multiple LMs, including Llama-2-7b and Mixtral-8x7b; all generated statements were manually verified. Where the generated statements were incorrect, the authors of this work manually wrote the declarative statements.

Tables. Each dataset’s tables were parsed using the dataset’s associated source code when available, and otherwise, the parsing was written by us. All tables were parsed into pandas DataFrames (pandas development team, 2020). The tables’ text representations of HTML, JSON, and Markdown were de-

69.3% 53.5%

Tables

Multi Step Reasoning

Quantitative

52.5%

Reasoning Long Context

29.1% 28.4%

Domain Expertise

19.9% 18.1%

HTML

JSON Markdown

17.1%

0 20 40 60

Figure 3: Distribution of the sources of complexity in CoverBench. Long context in this figure refers to examples with over 3,000 tokens with the Mixtral-8x7bInstruct tokenizer.

rived using the default implementations in pandas (the JSON representation uses the “records” setting) and chosen at random for each example. Since some of the datasets use the same context for multiple different claims, the some contexts may appear multiple times in CoverBench, either with the same table representation or different ones. In the case of Feverous, due to the unique typesetting of the Wikipedia tables employed in that dataset, the tables were taken as-is.

Negative sampling. The final distribution of negative examples by their answering model, after all selection and sampling steps, are: 48.8% (Mixtral8x7B-Instruct), 28.8% (gpt-4o), 22.4% (gemma1.1-7b-it).

#### A.3 Prompts.

The prompts we used for the sub-tasks in the conversion process are given below.

QA to Declarative Statement:

“Edit the following question and answer into a declarative form.

For example, given the question "What is the population of Europe? Round to the nearest million." and answer "741,000,000", output "The population of Europe, rounded to the nearest million, is 741,000,000."

Question: [question] Answer: [answer] Declarative form:”

Final Answer Extraction:

Model Our sample Random sample

∗Gemma-1.1-7b-it 46.1 48.5 ∗Starling-LM-7B-beta-ExPO 52.1 58.2 ∗Mixtral-8x7B-Instruct 49.0 59.6 Gemini 1.5 Flash 54.5 57.8 Yi-1.5-34B-Chat 54.2 61.1 Qwen2-72B-Instruct 57.0 62.0 Gemini 1.5 Pro 62.1 68.8

Table 4: Macro-F1 performance on CoverBench (0-shot CoT) for comparison between our selection of examples (§3.3) and a random sample of the same size. (∗) denotes seed models for sampling—we include their results for completeness, but note that they are unreliable.

“Given a question and a model’s answer, extract the pure term answer from this text.

For example if the answer to the question “How much did the stocks rise in 2001?” is “The stocks in 2005 rose $50 from 2001.”, generate “$50”. If the model didn’t give a clear answer, output “None”.

Question: [question] Model answer: [model answer] Extracted final answer:” Gold Answer Comparison Check:

“Given two terms, Term 1 and Term 2, your task is to compare the two terms and say whether they are equivalent or not.

For example, “12%” and “0.12” are equivalent, and “2 thousand dollars” is equivalent to “$2,000”, but different values, different units or different entities are not equivalent. Generate “Yes” or “No”.

- Term 1: [model answer]
- Term 2: [gold answer] Equivalent:”

### B Experiment Details

#### B.1 Implementations

For the LM baselines, binary model decisions were parsed from model outputs under greedy decoding, according to the format below. If a model failed to comply with the prompt format, we additionally

concatenated “Final answer (correct or incorrect):” to the end of the model’s answer and re-prompted it to get the final answer for parsing.

The implementations and prompts were checked for soundness by seeing the gap between our measurements on a random subset compared to measurements in the literature, to observe that they are overall similar. The code to use the models was based on standard available HuggingFace (Wolf et al., 2020) code for each model, alongside their associated tokenizers and chat templates. For Gemini 1.5 Flash and Pro, the API version was used.

#### B.2 Additional Results

Tab. 4 shows a comparison of results between a random selection of examples and our model-based selection.

#### B.3 Prompts

Below are the prompts we used in the baseline evaluations. We have tested various edits of the prompts on a random subset of examples which were not included in the final benchmark.

Zero-Shot:

“Your task is to check if the Claim is correct according to the Evidence. Generate ’Correct’ if the Claim is correct according to the Evidence, or ’Incorrect’ if the claim is incorrect or cannot be verified.

Evidence: [context] Claim: [claim] Answer:”

Zero-Shot with Chain-of-Thought:

“Your task is to check if the Claim is correct according to the Evidence. Generate ’Correct’ if the Claim is correct according to the Evidence, or ’Incorrect’ if the claim is incorrect or cannot be verified.

Evidence: [context] Claim: [claim] Let’s think step-by-step:”

Dataset Context (snippet) Claim FinQA Paper Title: One-to-X analogical reasoning on word embeddings: a case for diachronic

For both Gigaword and NOW datasets (and the corresponding embeddings), using the cosine-based threshold increases recall and decreases precision (differences are statistically significant with t-test, p < 0.05).

armed conflict prediction from news texts Table 3: Average diachronic performance | | [EMPTY] | [BOLD] Algorithm | [BOLD] Precision | [BOLD] Recall | [BOLD] F1 | |—:|:———-|:—————–|—————:|————–:|:———-|

- | 0 | Giga | Baseline | 0.19 | 0.51 | 0.28 |
- | 1 | Giga | Threshold | 0.46 | 0.41 | [BOLD] 0.41 | [...]

QRData The Stanford University Heart Transplant Study was conducted to determine whether an experimental heart transplant program increased lifespan. Each patient entering the program was designated an official heart transplant [...] The data is in the CSV file heart_transplant.csv. heart_transplant.csv | | id | acceptyear | age | survived | survtime | prior | transplant | wait | |—-:|—–:|———-:|——:|:——–|——–:|:——|:———-|—–:|

The difference in survival rate between the control and treatment groups can be estimated using a confidence interval constructed with the normal approximation.

- | 0 | 15 | 68 | 53 | dead | 1 | no | control | nan |
- | 1 | 38 | 70 | 41 | dead | 5 | no | treatment | 5 | [...]

TabFact {"artist": {"0": "ophiolatry", "1": "ophiolatry", "2": "black flame", "3": "tangorodrim",

Black Flame’s title was released five months and two days after Ophiolatry’s 2008 title.

"4": "tangorodrim", "5": "triumfall"}, "title": {"0": "transmutation", [...]

MultiHiertt HEWLETT PACKARD ENTERPRISE COMPANY AND SUBSIDIARIES

Two amounts for Credit Card (a) exceeded the average of Amount for Credit Card (a) in 2012.

Management’s Discussion and Analysis of Financial Condition and Results [...]

- ## Table 0 ## <table><tr><td></td><td colspan="3">For the fiscal years ended October 31,</td>[...]<td>2019</td><td>2018</td><td>2017</td></tr><tr><td></td><td colspan="3">Dollars in millions</td></tr><tr><td>Net revenue [...] Fiscal 2019 compared with Fiscal 2018 Corporate Investments net revenue decreased by $36 million, or 6.6% (decreased 4.4% on a constant currency bases), in [...]
- ## Table 1 ## <table><tr><td> [...]

HybridQA 1998 IAAF World Half Marathon Championships The 7th IAAF World Half Marathon Championships was held on September 27, 1998, in the city of Uster, Switzerland. A total of 236 athletes, 139 men and 97 women, from 54 countries took part. Detailed reports on the event and an appraisal of the results was given. Complete results were published. Team Results – Men ’s <table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> <th></th> <th>Rank</th> <th>Country</th> [...]

Ethiopia finished in 3:05:18 at the IAAF World Half Marathon Championships of 1998 southeast of the country.

ContractNLI DEPARTMENT OF HOMELAND SECURITY NON-DISCLOSURE AGREEMENT I, __, an individual official, employee, consultant, or subcontractor of or to __ (the Authorized Entity), intending to be legally bound, hereby consent to the terms in this Agreement in consideration of my being granted conditional access to certain information, specified below, that is owned by, produced by, or in the possession of the United [...]

Confidential Information shall only include technical information.

PubMedQA To examine whether government-funded, low-income vision care programs improve use of eye care services by low-income individuals in Canada. Cross-sectional survey. 27,375 white respondents to the Canadian Community Health Survey (CCHS) Healthy Aging 2008/2009. Government-funded, low-income vision care programs were reviewed. The amount of assistance provided was compared with professional fee schedules for general/routine eye examinations [...]

Government assistance may improve utilization of eye care services by low-income individuals.

TACT [{"Job Title":"Registered Nurse (RN)","Job Description":"Full-Time position for caring for patients at Gentiva Home Hospice, Monday - Friday, daytime schedule, reports to Hospice, RN, Patient Care.","Location":"Bloomington, IL","Salary":"$20.04 [...]

The average salary described in the text is 27.52 per hour.

Feverous [H] Position: | [[Offensive_tackle|Offensive tackle]]/[[Guard_(gridiron_football)|Guard]] [H] Personal information | [H] Personal information [H] Born: | (1947-08-30) August 30, 1947 (age 73) [[Ponca_City,_Oklahoma|Ponca City, Oklahoma]] [H] Height: | 6 ft 2 in (1.88 m) [H] Weight: | 262 lb (119 kg) [...]

Jon Kolb ( played 200 games as a professional football player and he weighs 199lb).

Table 5: Examples from each of the source datasets used in CoverBench. The labels are true or false.

