# arXiv:2510.11391v3[cs.CV]18May2026

## DOCREWARD: A Document Reward Model for Structuring and Stylizing

Junpeng Liu1∗Yuzhong Zhao2∗Bowen Cao1 Jiayu Ding3 Yilin Jia4 Tengchao Lv5 Yupan Huang5 Wenshan Wu5 Shaohan Huang5 Nan Yang5 Li Dong5 Lei Cui5 Tao Ge5 Xun Wang5 Huitian Jiao5 Sun Mao5 FNU Kartik5 Si-Qing Chen5 Wai Lam1 Furu Wei5 1CUHK 2UCAS 3XJTU 4UMich 5Microsoft https://aka.ms/GeneralAI

### Abstract

Recent agentic workflows automate professional document generation but focus narrowly on textual quality, overlooking structural and stylistic professionalism, which is equally critical for readability. This gap stems mainly from a lack of effective reward models capable of guiding agents toward producing documents with high structural and stylistic professionalism. We introduce DOCREWARD, a Document Reward Model that evaluates documents based on their structure and style. To achieve this, we propose a textual-quality-agnostic framework that ensures assessments are not confounded by content quality, and construct DOCPAIR, a dataset of 117K paired documents covering 32 domains and 267 types. Each pair shares identical content but differs in structural and stylistic professionalism. DOCREWARD is trained using the Bradley-Terry loss. On a manually annotated benchmark, DOCREWARD outperforms GPT-5 by 14.6 percentage points in the same setting. Reinforcement learning experiments further show that DOCREWARD effectively guides agents toward generating documents with consistently higher structural and stylistic professionalism, highlighting its practical utility.

### 1 Introduction

Recent advances in agentic workflows have automated various complex tasks, such as code generation [1, 2, 3], image generation [4], visual understanding [5, 6], math reasoning [7], and travel planning [8]. A key focus of agentic workflows is the generation of professional documents, such as deep research [9, 10, 11] and technical documentation generation [12]. However, existing research on professional document generation primarily focuses on improving textual quality, often neglecting the visual structure and style—both of which are crucial for document professionalism. A well-organized structure facilitates seamless navigation for readers, while a consistent style makes the content more readable and engaging. Together, these aspects help convey information more clearly and effectively. The neglect of structure and style mainly stems from the lack of effective reward models, which are capable of guiding agentic workflows toward producing documents with professional structure and style.

However, building a reward model capable of providing a robust evaluation of visual structure and style is non-trivial, as it requires both comprehensiveness and textual-quality-agnosticism. Specifically, comprehensiveness refers to the ability to evaluate documents across diverse types, qualities, structures, and styles, while textual-quality-agnosticism, in this context, means that the model does not evaluate the inherent quality of the textual content itself, but instead assesses how well the structure and style of a document stand out, given the fixed content.

∗ Equal contribution. Work done during internship at Microsoft Research.

Preprint.

Human Preference Accuracy

Low professionalism High professionalism

[Figure 1]

[Figure 2]

82.3%

[Figure 3]

| | |
|---|---|
|[Figure 4]| |

67.7%

[Figure 5]

[Figure 6]

[Figure 7]

54.9% 54.2%

DREWARDOC

Claude4

GPT-4o

GPT-5

Stepwise feedback for more professional documents

-0.58 1.68 4.33 10.50

[Figure 8]

[Figure 9]

Agentic Workflows

Figure 1: DOCREWARD automatically assesses document professionalism according to the structure and style of documents, assisting existing agentic workflows for more professional document generation (left). It outperforms GPT-5 by 14.6 percentage points in human preference accuracy in the same setting (right).

To achieve this, we propose DOCREWARD, a Document Reward Model specialized in assessing document professionalism in structure and style, as illustrated in Figure 1. The model is trained under a textual-quality-agnostic framework. Specifically, we construct DOCPAIR, a multi-domain dataset of 117K document pairs across 32 domains and 267 types. Each pair comprises a high-professionalism sample and its low-professionalism counterpart; these documents share identical content but differ in structure and style. The construction of DOCPAIR involves three phases: 1) Curating HighQuality Professional Documents. We curate a collection of high-quality documents characterized by professional structure and style from diverse domains, such as government, education, and science. 2) Expanding Source Documents via Agents. Next, we extract the textual content and employ multiple generation agents to re-produce documents that preserve the original textual content while exploring diverse structural and stylistic variations. 3) Ranking Documents. The ranking labels are determined by a combination of human-verified heuristics and an oracle-based annotation method. This provides a scalable way to capture elements of structural and stylistic professionalism.

Based on the constructed dataset, we train DOCREWARD to take rendered document pages as input and output a score reflecting the document’s professionalism in structure and style. The predicted scores of paired documents are optimized using the Bradley-Terry loss [13, 14], which penalizes violations of the annotated ranking order.

To demonstrate the superiority and utility of DOCREWARD, we conduct both intrinsic and extrinsic evaluations. For the intrinsic evaluation, we establish a benchmark DOCPAIRBENCH of 1443 human-annotated pairs across multiple domains. Human annotators ranked each pair based on the professionalism of the documents’ structure and style. Notably, as shown in Figure 1 (right), DOCREWARD outperforms GPT-4o [15] and GPT-5 [16] by 27.4 and 14.6 percentage points, respectively, in accuracy on DOCPAIRBENCH, demonstrating its superiority over existing baselines. For the extrinsic evaluation, we evaluate DOCREWARD through two complementary experiments. 1) Best-of-N. DOCREWARD is used as a re-ranking model for improving agentic workflow without changing the agent itself. Human evaluation reveals that DOCREWARD as a reward model achieves a significantly higher win rate of 60.8%, compared to GPT-5’s 37.7%. 2) Reinforcement Learning. We further demonstrate the utility of DOCREWARD as the reward model for reinforcement learning of both open- and closed-source models. This integration improves the document generation performance of Qwen2.5-Coder [17] and GPT-4o in terms of structure and style.

Our main contributions are as follows:

- • We propose a textual-quality-agnostic framework for document reward modeling. By decoupling structure and style from textual content, it enables the reward model DOCREWARD to accurately capture complex structural and stylistic elements.
- • DOCPAIR: A large-scale multi-domain dataset comprising 117K document pairs across 32 domains and 267 document types, designed to equip DOCREWARD with comprehensiveness and textualquality-agnosticism.

- • Comprehensive experiments demonstrate that DOCREWARD not only outperforms GPT-5 in assessing structure and style in the same setting, but also serves as an effective reward model in reinforcement learning for agentic workflows.

### 2 Textual-Quality-Agnostic Framework

Document professionalism is characterized by its textual content, structure, and style. Although large language models excel at evaluating textual quality, they are limited in assessing structure and style. To bridge this gap, we propose a textual-quality-agnostic framework that trains reward models to evaluate professionalism independently of textual quality.

In this section, we formulate the framework and define its objectives. Let {Di}Ni=1 denote a set of N documents, where each document Di consists of textual content Dtext,i and rendered images Dimg,i. The document reward model Rθ assigns scores to documents that share the same textual content, such that these scores reflect their structural and stylistic quality. This process is formalized as follows:

Sim π∗, Argsort(Rθ(Dimg,1), . . . , Rθ(Dimg,N))

max

θ

s.t. Dtext,i = Dtext,j, ∀i, j, (1)

where “Sim” is a predefined similarity function that measures the agreement between the true and predicted orders. “Argsort” returns the indices of documents sorted by their predicted scores. π∗ denotes the true indices reflecting the relative ranking of the documents in terms of structure and style. Essentially, the term “Dtext,i = Dtext,j” ensures that the model evaluates professionalism in a textual-quality-agnostic manner, as it processes identical textual content from paired documents. Importantly, this framework is content-quality-independent rather than content-agnostic: the model still reads and understands the textual content to judge whether the structure and style are contextually appropriate, but its judgment is not influenced by how well the text itself is written. In this paper, document professionalism in structure and style is defined as follows:

- 1) Structure: Proper use of white space, appropriate margins, clear section breaks, well-structured text alignment, adequate paragraph spacing, proper indentation, inclusion of page headers and footers, and logical, coherent organization of content.
- 2) Style: Appropriate font choices (type, size, color, readability), clear heading styles, effective use of emphasis (bold, italics), bullet points, numbering, and consistent formatting.

By optimizing Rθ based on these factors, we obtain a reward model capable of assessing structural and stylistic professionalism in a comprehensive and textual-quality-agnostic way.

- 3 DocReward

Based on the proposed textual-quality-agnostic framework, we train DOCREWARD, a reward model specializing in assessing the structural and stylistic professionalism of documents. DOCREWARD is trained on DOCPAIR, a diverse dataset of 117K document pairs (Section 3.1), and is optimized with a preference-based objective for structural and stylistic assessment (Section 3.2). The following sections detail the data construction pipeline and model design.

#### 3.1 Data Construction

As shown in Figure 2, we first collect a set of high-quality real-world source documents. The source documents are then expanded by multiple generation agents, and the resulting documents are grouped by shared textual content. Finally, each group of documents is annotated with a ranking π∗ in terms of structure and style quality. The construction procedure is detailed step by step below:

Curating High-Quality Professional Documents. As illustrated in Figure 2 (top), we first curate a corpus of human-authored Microsoft Word documents that spans both highly formal institutional writing and everyday professional communication. We draw on two complementary sources:

1. Curating High-Quality Professional Documents

Professional source documents

[Figure 10]

[Figure 11]

###### Preprocessing & heuristic filtering

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Diverse domains:

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Government, Technical Academic, Business …

Very large or small files

Non English

Bad Structure

[Figure 21]

[Figure 22]

[Figure 23]

- 2. Expanding Source Documents via Agents

Professional source document

Textual context to document

###### Refinement for Better Structure and Style

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

|[Figure 31]<br><br>[Figure 32]<br><br>Real|
|---|

[Figure 33]

[Figure 34]

“… 3.714-Improved pension elections-public assistance beneficiaries \t\t 3.714-1… Definitions. The following…”

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Agent

Synth Synth

- 3. Ranking Documents

###### Real vs. Synth. Synth. vs. Synth.

“… compare two documents with a reference document and determine

[Figure 42]

[Figure 43]

[Figure 44]

|[Figure 45]<br><br>[Figure 46]<br><br>Real|
|---|

|[Figure 47]<br><br>[Figure 48]<br><br>Real|
|---|

[Figure 49]

[Figure 50]

[Figure 51]

|[Figure 52]<br><br>[Figure 53]<br><br>Synth|
|---|

|[Figure 54]<br><br>Synth|[Figure 55]<br><br>which one has better professionalism|
|---|---|
| |[Figure 56]<br><br>GPT-5|

|[Figure 57]<br><br>[Figure 58]<br><br>Synth|
|---|

|[Figure 59]<br><br>[Figure 60]<br><br>Synth|
|---|

[Figure 61]

ism …”

[Figure 62]

Synth

Same text content Real document as a reference

Figure 2: The data construction pipeline for DOCREWARD.

- 1) Government corpora: GovDocs1 [18] and NapierOne [19] are authoritative document collections sourced from government and public institutions, covering reports, forms, guidelines, and other professional materials with consistent structure and style.
- 2) Web document corpus: CommonCrawl, covering business, education, law, healthcare, etc.2

To ensure high quality, we apply a preprocessing and filtering pipeline before data construction. First, all files are converted to DOCX format to enable programmatic access and modification via python-docx.3 Next, we discard extreme or malformed cases (exceeding 20 pages, files larger than 1MB dominated by images, and files smaller than 10KB with trivial content). To efficiently reduce residual noise, we employ GPT-5 as a rigorous automated heuristic to flag poor structure/style on a [0,10] scale; documents scoring above 8 are retained. A manual inspection of 200 randomly sampled retained documents confirms that this automated filter preserves high-quality professional samples.

The domain distributions of the filtered corpus are shown in Figure 3. The corpus spans 32 domains and 267 document types, demonstrating substantial breadth and diversity, and it provides a highquality foundation for constructing subsequent document pairs. More data statistics are presented in Appendix A.5.

Expanding Source Documents via Agents. As shown in Figure 2(middle), to obtain documents with the same textual content but different structure and style, we construct two types of agents to synthesize documents given the textual content (and rendered pages) of the source documents. To further increase the diversity of the synthesized documents, each agent can be empowered by different LLMs. The two agents are detailed as follows:

other (7.9%)

technical (1.9%)

academic (2.3%)

business (3.4%)

government (32.2%)

legal (3.5%)

scientific (5.0%)

medical (5.7%)

- 1) Textual Content to Document. The textual content is first extracted from the source documents, discarding all formatting, styling, and layout information. Then, advanced models (e.g., GPT-4o, OpenAI o1 [20], Claude Sonnet 4 [21], and GPT-5) are used to synthesize DOCX documents via python-docx.
- 2) Refinement for Better Structure and Style. To further improve the structure and style of synthesized documents, we refine them by comparing them with the original human-authored documents in terms of structure and style. The refinement process consists of two stages: a) Agents are provided with the python-docx

non_profit (9.6%)

education (28.6%)

Figure 3: Top 10 domain distribution.

- 2https://commoncrawl.org/
- 3https://python-docx.readthedocs.io/

Setting 1: Scoring Setting 2: Best-of-N

- score-1
- score-2

- Doc 1

Setting 3: Reinforcement learning

Agent

- Doc 2

reward signal

argmax Reward

…

score

Action Model (generated doc)

score-N

Doc N

Setting 1: DocPairBench Setting 2: Best-of-N Setting 3: Reinforcement Learning

- score-1
- score-2

- Doc 1 reward signal

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

- Doc 2

argmax

DocReward Action

…

score

score-N

Doc Doc N

(generated doc)

Setting 1: DocPairBench Setting 2: Best-of-N Setting 3: Reinforcement Learning

- score-1
- score-2

- Doc 1 reward signal

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

- Doc 2

[Figure 74]

- A
- B

argmax

DocReward action

[Figure 75]

…

Doc

choice

score

score-N

pointwise pairwise

Doc N

(generated doc)

Figure 4: Three evaluation settings for DOCREWARD.

code, rendered pages, and structured textual representation of the synthesized document, along with the rendered pages of the original human-authored document, to generate a refinement plan. b) Based on the plan, the agents modify the python-docx code to produce refined documents.

The synthesized documents are grouped with their originals to facilitate subsequent processing. Please refer to Appendix A.3 and A.8 for details.

Ranking Documents. As shown in Figure 2 (bottom), the collected documents within each group share identical textual content and are organized into pairs. In the ranking stage, we assess the relative professionalism in terms of structure and style for each pair, under the following two cases:

- 1) Real vs. Synth. When comparing an original document with its agent-generated counterparts, the human-authored version is designated as the winner. This heuristic is grounded in a human inspection (on a randomly sampled set of 100 samples), which indicates that current state-of-the-art models (e.g., GPT-5, Claude Sonnet 4) produce documents with structural and stylistic quality inferior to high-quality human-authored documents curated by the rigorous filtering pipeline described earlier.
- 2) Synth. vs. Synth. For pairs where both documents are agent-generated, manual annotation is not scalable. We therefore employ a closed-source model as a proxy for human judgment. To mitigate its intrinsic biases, we adopt an oracle setting: the model is provided with a document triplet

{Dreal,Dsynth1,Dsynth2}, using the human-authored Dreal as a reference anchor. This transforms the annotation from a subjective preference task into an objective similarity matching against the ground-truth document (see the prompt in Appendix A.8). We validated this approach on 120 pairs, where it demonstrated 92.5% alignment with human experts, confirming its effectiveness.

Note that the above strategy is cost-effective and necessary for large-scale training data construction, while the test split, DOCPAIRBENCH, is constructed through strict human annotation. The data statistics of DOCPAIR are shown in Table 1.

Table 1: Data statistics of DOCPAIR.

Domains Types Doc. Avg. Page Pairs 32 267 69,137 3.2 117,108

#### 3.2 Model Structure and Optimization

We use Qwen2.5-VL [22] as the base model, which takes a multi-page document as input and outputs a scalar score via an added regression head (details provided in Appendix A.2).

Given a preference pair consisting of a winner Dimgw and a loser Dimgl , the reward model Rθ assigns scores to both documents and is optimized using the Bradley-Terry (BT) loss:

−log σ Rθ(Dimgw ) − Rθ(Dimgl ) , (2) where σ(x) = 1+1e−x . This objective encourages higher scores for preferred documents.

min

θ

4 Experiments

We first present the newly proposed benchmark DOCPAIRBENCH (Section 4.1) and then conduct intrinsic and extrinsic evaluations of DOCREWARD as shown in Figure 4 (Section 4.2 and 4.3).

#### 4.1 Human-Annotated DOCPAIRBENCH Construction

A subset of the curated documents in Section 3.1 is set aside as evaluation documents. To diversify the evaluation samples, we consider the following six types of documents using the method described in Section 3.1. Four of them are obtained via the Textual Content to Document agent, which generates

- Table 2: Accuracy of Models on the DOCPAIRBENCH benchmark across ten domains: Government (Gov.), Education (Edu.), Non-Profit Organization (NPO), Medical (Med.), Science (Sci.), Legal (Leg.), Business (Bus.), Academic (Acad.), Technology (Tech.), and Other (Oth.).

Models Gov. Edu. NPO Med. Sci. Leg. Bus. Acad. Tech. Oth. Overall

###### Pairwise Setting

Qwen2.5-VL-3B[22] 55.1 61.5 48.4 50.0 50.0 47.0 56.9 47.3 55.5 60.5 54.4 Qwen2.5-VL-7B[22] 57.7 61.5 53.9 53.8 54.0 59.0 56.0 52.7 49.1 58.1 56.1

- GPT-4o[15] 61.9 57.7 75.8 66.3 53.0 62.0 71.6 61.8 66.4 61.4 63.3

Claude Sonnet 4[21] 62.9 69.2 75.8 61.3 46.0 60.0 75.2 56.4 65.5 62.9 63.0

- GPT-5[16] 70.2 69.2 69.2 68.8 57.0 69.0 76.2 63.6 70.0 69.8 68.9 Pointwise Setting

Qwen2.5-VL-3B[22] 38.4 23.1 36.3 30.0 25.0 27.0 32.1 39.1 38.2 30.2 33.5 Qwen2.5-VL-7B[22] 49.4 34.6 46.2 41.3 35.0 52.0 49.5 31.8 48.2 48.5 46.0

- GPT-4o[15] 60.1 46.2 58.2 56.3 34.0 54.0 58.7 50.9 58.2 53.9 54.9

Claude Sonnet 4[21] 59.8 50.0 56.0 60.0 48.0 63.0 48.6 50.0 50.9 49.7 54.2

- GPT-5[16] 69.7 80.8 70.3 71.3 49.0 72.0 75.2 60.0 64.6 66.6 67.7

DocReward-3B (Ours) 87.2 76.9 80.2 75.0 69.0 85.0 87.2 79.1 74.6 77.3 80.6 DocReward-7B (Ours) 89.3 92.3 86.8 83.8 67.0 87.0 84.4 80.9 76.4 76.7 82.3

DOCX documents using different LLMs (i.e., GPT-4o, OpenAI o1, Claude Sonnet 4, and GPT-

- 5). One type comes from the Refinement for Better Structure and Style agent, where GPT-5 is employed to refine synthesized documents. The last type consists of the curated human-authored documents. Together, these six types constitute the origins of samples in our benchmark. For each set of documents sharing the same content but differing in structure and style, human experts meticulously rank their quality based on structure and style. To facilitate model evaluation, these ranked relationships are converted into a total of 1443 comparison pairs, each consisting of two documents and a binary label indicating the preferred one. To ensure the quality of human annotation, we evaluate consistency among human annotators using Cohen’s Kappa and observe a value of 0.834. The details of annotation protocol and reliability validation are presented in Appendix A.4.

#### 4.2 Results on DOCPAIRBENCH

As presented in Table 2, we evaluate methods under two settings, namely Pairwise and Pointwise. In the pairwise setting, models are given a document pair and asked to select the one with better structure and style, while in the pointwise setting, documents are scored independently. Performance is measured by accuracy with respect to human annotations. On DOCPAIRBENCH, DOCREWARD-3B and DOCREWARD-7B achieve substantial improvements over strong baselines including Qwen2.5VL, GPT-4o, Claude Sonnet 4, and GPT-5. In particular, DOCREWARD-7B attains an overall human preference accuracy of 82.3 percent, surpassing the strongest closed-source model (GPT-5, pointwise) by 14.6 points. These results indicate that DOCREWARD captures structural and stylistic quality signals that existing language models often overlook.

#### 4.3 Improving Document Generation with DOCREWARD

To demonstrate the utility of DOCREWARD, we conduct two complementary experiments.

Best-of-N. A document agent generates N candidates from the same textual content, and a reward model selects the best one based on its score. We compare three reward models including “Random”, GPT-5 and DOCREWARD where human annotators rank the selected outputs by structure and style. As shown in Table 3, DOCREWARD consistently outperforms the baselines, achieving a win rate of 60.8% against “Random” and GPT-5 reward models, indicating that its reward better aligns with human preferences. This evaluation shows that integrating DOCREWARD into a document agent improves output quality without modifying the agent itself. Details are provided in Appendix A.6.

Reinforcement Learning. We aim to enhance document generation agentic workflows that take plain text as input and generate professional documents. We consider two kinds of rewards: 1) Rrule that penalizes documents that either result from invalid python-docx code or

Table 4: Results of the reinforcement learning experiments. “Score” denotes the sigmoid-normalized DOCREWARD score. “Rank” denotes the average ranking among the listed methods.

- Table 3: Best-of-N evaluation results. DOCREWARD shows utility for professional document generation.

Reward Types Success↑ ROUGE-L↑ Score↑ Rank↓

Qwen2.5-Coder 30.0 20.61 0.0663 4.58 + rule 98.0 97.94 0.1785 4.06 + DocReward 100.0 97.95 0.3046 2.84

Rewards Win Lose Tie Random 24.6 66.2 9.2

GPT-5 37.7 40.0 22.3 DOCREWARD 60.8 16.9 22.3

GPT-4o 52.0 48.73 0.2682 3.18 + rule 66.0 62.15 0.3189 2.70 + DocReward 78.0 74.33 0.4486 2.02

|Reference Document|
|---|

|Base model + Rule + DocReward<br><br>Qwen2.5-Coder-1.5B + GRPO|
|---|

|Base model + Rule + DocReward<br><br>GPT-4o + Training-free GRPO|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

- Figure 5: Visualization of documents generated by various methods. “Execution error” denotes instances where the generated code encountered runtime failures, preventing the final document from being rendered.

differ from the input text after execution. Specifically, if the code executes successfully, then Rrule = ROUGE(doc_ori,doc_gen); else Rrule will be zero. 2) RDocReward that penalizes documents with poor structure and style. Overall, the total reward is defined as:

##### R = Rrule + α · Irule · σ(RDocReward), (3)

where α is a hyperparameter to balance the rewards, σ(·) is the Sigmoid operation to regularize the value range of DOCREWARD to (0,1), and Irule represents whether Rrule is larger than a threshold.

For open-source models, we adopt GRPO [23] as the RL algorithm, while employing trainingfree GRPO [24] for closed-source models.4 To evaluate the quality of RL-trained models, human annotators rank documents produced by six model variants based on the professionalism of structure and style. As shown in Table 4, rule-based rewards substantially improve document generation success rate for both Qwen2.5-Coder and GPT-4o. Further incorporating DocReward consistently enhances performance across both open- and closed-source models, yielding higher success rates, improved ROUGE-L scores, and better human rankings. These results demonstrate that DOCREWARD serves as an effective and generalizable reward model for professional structure and style. Figure 5 presents a visualization of documents generated by different methods.

4Training-free GRPO is a lightweight approach without model parameter updates.

Table 5: Out-of-domain (OOD) experiment. DOCREWARD generalizes effectively to unseen domains.

Table 6: Cross-lingual robustness evaluation. ∆ denotes the performance gap between English and non-English scenarios. DOCREWARD demonstrates strong cross-lingual robustness.

Model ID OOD

Qwen2.5-VL-3B 33.5 30.0 Qwen2.5-VL-7B 46.0 48.1

- GPT-4o 54.9 53.8

Claude Sonnet 4 54.2 49.1

- GPT-5 67.7 68.4

DocReward-3B 80.6 76.3 DocReward-7B 82.3 77.5

Model FR ES RU Non-EN EN ∆

Qwen2.5-VL-3B 35.0 33.8 22.5 30.4 33.5 -3.1 Qwen2.5-VL-7B 48.8 42.5 27.5 39.6 46.0 -6.4

- GPT-4o 47.5 52.5 42.5 47.5 54.9 -7.4

Claude Sonnet 4 57.5 51.3 42.5 50.4 54.2 -3.8

- GPT-5 57.5 76.3 47.5 60.4 67.7 -7.3

DocReward-3B 77.5 82.5 72.5 77.5 80.6 -3.1 DocReward-7B 78.8 88.8 66.3 77.9 82.3 -4.4

[Figure 97]

| |
|---|

page header

| |
|---|

numbering

| |
|---|

page footers

(a)

[Figure 98]

| |
|---|

page corners

| |
|---|

bullet points

page corners

| |
|---|

(b)

[Figure 99]

table corners

| |
|---|

| |
|---|

table borders

| |
|---|

| |
|---|

page corners

(c)

- Figure 6: Visualization of attention maps. DOCREWARD captures structural and stylistic elements, such as headings, alignment, and whitespace, in its evaluation of document professionalism.

#### 4.4 Robustness Test

Out-of-Domain Evaluation. To evaluate cross-domain robustness, certain domains are reserved exclusively as test sets rather than being used for training. Table 5 reports the out-of-domain results of different models. Firstly, DOCREWARD-7B (77.5) remains superior to all baseline models, including the closed-source model GPT-5 (68.4). This trend is consistent with the in-domain results. Secondly, the performance of DOCREWARD-7B decreases by merely 4.8 percentage points when transitioning from in-domain to out-of-domain evaluations. Such a small performance gap indicates that DOCREWARD generalizes effectively to unseen domains.

Cross-Lingual Robustness. To evaluate the cross-lingual robustness of DOCREWARD trained on English data, we conduct evaluations in French, Spanish, and Russian, with results shown in Table 6. Firstly, in non-English scenarios, DOCREWARD-7B model achieves a score of 77.9, substantially outperforming all baseline models. Secondly, all models, including the baselines, exhibit performance degradation in non-English settings. Notably, the performance drop of DOCREWARD (−4.4) is even smaller than that of closed-source models GPT-4o (−7.4) and GPT-5 (−7.3), indicating that DOCREWARD demonstrates strong cross-lingual robustness.

#### 4.5 Visualization of Attention Maps

To understand DOCREWARD’s decision-making process, we analyze attention maps. As shown in Figure 6, the model attends to structural and formatting cues when assessing document professionalism. Specifically, attention focuses on headings and numbering (Figure 6a), page headers and footers (e.g., “CS-66”, “DEC. 2006”), bullet points (Figure 6b), and table grids (Figure 6c), reflecting sensitivity to structural clarity, formatting consistency, alignment, and readability. Additionally, attention to page corners suggests that uniform margins and balanced whitespace are important indicators of professional layout design.

#### 4.6 Case Study

In Figure 7, we present a case study on documents with identical textual content but different structures and styles. In case (a), the allocation of whitespace is ineffective, with insufficient space

[Figure 100]

[Figure 101]

[Figure 102]

imbalanced layout

vertically aligned

vertically aligned

| |
|---|

| |
|---|

| |
|---|

misaligned

| |
|---|

proper font size

| |
|---|

small font size

| |
|---|

| |
|---|

missing table borders

| |
|---|

| |
|---|

| |
|---|

table borders for readability

(a) score: 1.21

(b) score: 2.11

(c) score: 5.34

Figure 7: Case study: DOCREWARD’s scores reflect structural and stylistic professionalism.

allocated for Last Name and excessive space for First Name, leading to an imbalanced layout. Key fields such as Faculty/Department, Country, and Country Code are not vertically aligned, causing a cluttered and disorganized layout. This poor alignment and inconsistent spacing result in a low score of 1.21 from DOCREWARD. Case (b) adopts a table-like arrangement, but the level-1 heading "The teaching staff member" is too small and does not stand out from the body text, diminishing its impact. Additionally, the lack of borders around input fields makes it hard to locate items, resulting in a moderate score of 2.11. Case (c) provides a clear and well-structured layout, with appropriately sized headings and improved readability, earning the highest rating of 5.34. These results show that DOCREWARD effectively captures document professionalism in structure and style.

### 5 Related Work

Aesthetic and Professionalism Assessment. In graphic design, AesthetiQ [25] trains an MLLM for layout prediction that uses MLLM’s aesthetic preferences for DPO over graphic layouts, while diffusion-based methods such as LACE [26] introduce differentiable constraints to directly optimize layout attributes. For web interfaces, Calista [27] uses explicit ratings and pairwise comparisons to model visual appeal, showing high correlation to human perception. Additionally, photo aesthetics are modeled using layout-aware CNNs such as A-Lamp [28], and similar techniques extend to video [29]. These studies focus on images or UI interfaces rather than multi-page documents, where professionalism depends on both structure and style.

Document AI. Document AI research mainly targets semantic parsing and content understanding. Models such as LayoutLM [30] and ReLayout [31], along with OCR-based pipelines [32], identify logical elements such as headings, tables, and semantic groups to support information extraction and classification. Recent work also explores automatic document or layout generation [33, 34, 35], but evaluation has primarily been limited to content correctness or basic formatting. As a result, the assessment of document professionalism—particularly visual structure and style—remains largely unexplored.

Preference Learning and Reward Models. A major challenge in professionalism assessment is acquiring feedback signals that reflect human judgment. Preference-based reward modeling addresses this issue by training on pairwise comparisons to approximate preferences, forming the basis of alignment methods like RLHF [36] and DPO [37].

### 6 Conclusion

In this paper, we introduced DOCREWARD, a document reward model designed to assess structural and stylistic professionalism. To enable professionalism assessment without being influenced by textual quality, we propose a textual-quality-agnostic framework for document reward modeling. Then, a multi-domain preference dataset, DOCPAIR, consisting of 117K paired documents, is constructed to enable the training of DOCREWARD using the Bradley-Terry loss. Comprehensive experiments demonstrate that DOCREWARD not only outperforms GPT-5 in assessing structure and

style in the same setting, but also serves as an effective reward model in reinforcement learning for agentic workflows.

### References

- [1] Sida Peng, Eirini Kalliamvakou, Peter Cihon, and Mert Demirer. The impact of ai on developer productivity: Evidence from github copilot. arXiv preprint arXiv:2302.06590, 2023.
- [2] Anthropic. Claude code: Best practices for agentic coding. https://www.anthropic.com/ engineering/claude-code-best-practices, April 2025. Accessed: 2025-09-24.
- [3] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2024.
- [4] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13294–13304, June 2025.
- [5] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.
- [6] Damiano Marsili, Rohun Agrawal, Yisong Yue, and Georgia Gkioxari. Visual agentic ai for spatial reasoning with a dynamic api. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19446–19455, 2025.
- [7] Yibo Yan, Shen Wang, Jiahao Huo, Philip S. Yu, Xuming Hu, and Qingsong Wen. MathAgent: Leveraging a mixture-of-math-agent framework for real-world multimodal mathematical error detection. In Georg Rehm and Yunyao Li, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), pages 69–82, Vienna, Austria, July 2025. Association for Computational Linguistics.
- [8] Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. Travelplanner: A benchmark for real-world planning with language agents. arXiv preprint arXiv:2402.01622, 2024.
- [9] OpenAI. Introducing deep research. https://openai.com/index/ introducing-deep-research/, February 2025. Accessed: 2025-09-24.
- [10] Xinbin Liang, Jinyu Xiang, Zhaoyang Yu, Jiayi Zhang, Sirui Hong, Sheng Fan, Xiao Tang, Bang Liu, Yuyu Luo, and Chenglin Wu. Openmanus: An open-source framework for building general ai agents, 2025.
- [11] Alibaba Cloud / Qwen. Deep research — qwen. https://chat.qwen.ai/?inputFeature= deep_research, 2025. Accessed: 2025-09-24.
- [12] Shubhang Shekhar Dvivedi, Vyshnav Vijay, Sai Leela Rahul Pujari, Shoumik Lodh, and Dhruv Kumar. A comparative analysis of large language models for code documentation generation. In Proceedings of the 1st ACM international conference on AI-powered software, pages 65–73, 2024.
- [13] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.
- [14] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [15] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [16] OpenAI. Gpt-5 system card. https://openai.com/index/gpt-5-system-card/, August

2025. Accessed: 2025-09-24.

- [17] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2.5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.
- [18] Simson Garfinkel, Paul Farrell, Vassil Roussev, and George Dinolt. Bringing science to digital forensics with standardized forensic corpora. digital investigation, 6:S2–S11, 2009.
- [19] Simon R Davies, Richard Macfarlane, and William J Buchanan. Napierone: A modern mixed file data set alternative to govdocs1. Forensic Science International: Digital Investigation, 40:301330, 2022.
- [20] OpenAI. Openai o1 system card. https://openai.com/index/ openai-o1-system-card/, December 2024. Updated: December 5, 2024. Accessed: 2025-09-24.
- [21] Anthropic. Introducing claude 4. https://www.anthropic.com/news/claude-4, May

2025. Accessed: 2025-09-24.

- [22] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. ArXiv, abs/2502.13923, 2025.
- [23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [24] Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, et al. Training-free group relative policy optimization. arXiv preprint arXiv:2510.08191, 2025.
- [25] Sohan Patnaik, Rishabh Jain, Balaji Krishnamurthy, and Mausoom Sarkar. Aesthetiq: Enhancing graphic layout design via aesthetic-aware preference alignment of multi-modal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23701–23711, 2025.
- [26] Jian Chen, Ruiyi Zhang, Yufan Zhou, and Changyou Chen. Towards aligned layout generation via diffusion model with aesthetic constraints. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.
- [27] Alexandros Delitzas, Kyriakos C. Chatzidimitriou, and Andreas L. Symeonidis. Calista: A deep learning-based system for understanding and evaluating website aesthetics. International Journal of Human-Computer Studies, 175:103019, 2023.
- [28] Shuang Ma, Jing Liu, and Chang Wen Chen. A-lamp: Adaptive layout-aware multi-patch deep convolutional neural network for photo aesthetic assessment. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 722–731. IEEE Computer Society, 2017.
- [29] Chang Liu and Han Yu. Ai-empowered persuasive video generation: A survey. ACM Computing Surveys, 55(13s):1–31, 2023.
- [30] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pretraining of text and layout for document image understanding. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1192–1200, 2020.

- [31] Zhouqiang Jiang, Bowen Wang, Junhao Chen, and Yuta Nakashima. Relayout: Towards real-world document understanding via layout-enhanced pre-training. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert, editors, Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 3778–3793. Association for Computational Linguistics, 2025.
- [32] Nishant Subramani, Alexandre Matton, Malcolm Greaves, and Adrian Lam. A survey of deep learning approaches for ocr and document understanding. arXiv preprint arXiv:2011.13534, 2020.
- [33] Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian-Guang Lou, and Dongmei Zhang. Layoutprompter: Awaken the design ability of large language models. Advances in Neural Information Processing Systems, 36:43852–43879, 2023.
- [34] Zecheng Tang, Chenfei Wu, Juntao Li, and Nan Duan. Layoutnuwa: Revealing the hidden layout expertise of large language models. arXiv preprint arXiv:2309.09506, 2023.
- [35] Jiaxu Tian, Xuehui Yu, Yaoxing Wang, Pan Wang, Guangqian Guo, and Shan Gao. Relayout: Integrating relation reasoning for content-aware layout generation with multi-modal large language models. arXiv preprint arXiv:2507.05568, 2025.
- [36] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.
- [37] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.
- [38] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372, 2024.

### A Appendix

#### A.1 Limitations and Broader Impacts

Limitations and Future Work: A limitation is that DOCREWARD is designed as a scalar reward model, providing only a numerical score to represent the overall structural and stylistic quality of a document. It currently cannot generate natural language feedback explaining why a document is perceived as unprofessional. Therefore, extending DOCREWARD from a scalar model to an interpretable reward model that can produce both quantitative scores and qualitative diagnostic rationales remains a key focus of our future work.

Broader Impacts: The document reward model proposed in this work advances agentic workflows for documents. It poses no obvious negative societal impact, as the focus is the textual-qualityagnostic training framework and methods.

#### A.2 Model Implementation Details

Our document reward model is built upon the Qwen2.5-VL multimodal architecture, and a regression head is added to predict scalar scores. The maximum input pixels are set to 300,000. It is configured with a maximum context length of 16,000 tokens. Training utilizes the AdamW optimizer with a learning rate of 1e-6 and a batch size of 256 over 3 epochs. All training was conducted on 8 NVIDIA A100 GPUs (80GB). The training code is based on LLaMA-Factory [38]. α and the threshold of Irule are set to 1 and 0.8, respectively.

#### A.3 Source Documents Expansion

To ensure that the reward model learns to assess differences in structure and style rather than content, we applied a rigorous filtering process. Using python-docx, we extracted text from pairs of Microsoft Word DOCX documents and computed their word counts. Only synthetic documents with a word count difference of no more than 20 words from the original document and a ROUGE-L score exceeding a threshold (i.e., 0.95) are retained, ensuring comparable content while isolating variation in structure and style. For the constructed training dataset DOCPAIR, both GPT-4o and GPT-5 serve as the base models of agents.

#### A.4 Annotation Protocol and Reliability

Annotation Guidelines. The detailed guidelines for human annotation are presented in Figure 8. The annotation guidelines consist of general principles that are formulated in an explicit, objective manner. For instance, extremely narrow margins that produce an almost fully saturated page layout are commonly regarded as unprofessional across different cultural and regional contexts.

Independence from annotators’ cultural and professional backgrounds. The annotation was performed by three Ph.D. students with complementary expertise in computer science/math, marketing, and design. We measured inter-annotator reliability using Cohen’s Kappa; the results are shown

- in Table 7. The high agreement indicates that the annotations follow clear, well-defined rules that do not depend on the annotators’ professional training or cultural background, demonstrating the guidelines’ generality and objectivity.

Table 7: Cohen’s Kappa among annotators. Annotator ID 1 2 3 Average

- 1 - 0.8340 0.8092 0.8215
- 2 0.8340 - 0.8590 0.8465
- 3 0.8092 0.8590 - 0.8341

Average 0.8215 0.8465 0.8341 0.8340

Guidelines for Human Annotation

Target: Each document group contains N documents. Their textual content is the same, but

their structure and style differ. The first document in each group is the original human-authored document, which serves as a reference during annotation. Based on the level of professionalism in structure and style across the N documents, the annotator should rank the documents. Note that there may exist cases where human-authored documents are not the best ones.

Annotation Format Example: For example, for the document group with ID 10655307, suppose the

human-annotated professionalism ranking is 1 > 5 > 3 > 2 > 4, where 1 is judged to be the most professionally structured and formatted document, and 4 is the least professional. Then the annotation format should be: 10655307 \t 15324

Evaluation Criteria:

- 1. **Layout and Design**:

- - Consistent formatting and spacing
- - Proper use of headings, subheadings, and other structures, and proper hierarchy (e.g., long paragraphs should use body text style rather than heading styles, and headings should not be formatted as body text)
- - Appropriate margins and white space usage (e.g., page margins or table column widths that are excessively wide or narrow are not appropriate)

- 2. **Readability and Typography**:

- - Consistent and appropriate font choices
- - Proper Text size (e.g., overly large or overly small text is not suitable)
- - Appropriate line spacing and clear paragraph structure
- - Proper Text alignment

- 3. **Professional Standards**:

- - Document structure and organization
- - Use of professional elements (headers, footers, page numbers)
- - Consistency across pages (if multiple pages provided)

- 4. **Visual Elements**:

- - Quality and placement of images, tables, or charts
- - Integration of visual elements with text
- - Professional presentation of data

Figure 8: Detailed guideline for human annotation.

- A.5 Document Types Distribution Document Types. The top 30 document types are presented in Figure 9.
- A.6 Details of Best-of-N Evaluation

For the Best-of-N experiment, the Textual Content to Document defined in Section 3.1 is adopted as the document agent, with the base model being GPT-5. Three reward models, including random, GPT-5, and DOCREWARD, are compared. Once the document agent generates candidates and the reward model selects the top-ranking document from N candidates, a highly educated annotator is asked to rank the three documents selected, according to the definitions of professional structure and style defined in Figure 8. As a result, we collected 130 comparison pairs between documents selected by each pair of reward models, and asked annotators to rank them. Finally, the win/lose/tie rate of each reward model is calculated on the comparison pairs against the other reward models. N is set to 8.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

job_description

government_form policy_document meeting_minutes

press_release course_syllabus supplementary_information worksheet employment_application_form supplementary_material job_posting

meeting_agenda parliamentary_written_reply

grant_application_form job_application_form application_form

official_report course_material

statute regulation specification_document lesson_plan

journal_article supporting_information parliamentary_question

guidance_document grant_application

public_notice research_publication position_description

0 500 1000 1500 2000 2500 3000

Number of Files

Figure 9: Top 30 Document Type Distribution.

#### A.7 Ablation Study of Inputs

In designing the input channels for DOCREWARD, we experimented with two different configurations: a purely visual channel method and a combination method of visual and additional parsing information. The experiment was conducted on a subset of the test set. The experimental results are summarized

- in Table 8. Results show that text and bounding boxes of text spans from an additional OCR module are not helpful for the assessment of professional structure and style.

Table 8: Additional text and bounding boxes of text spans are not helpful for the assessment of professional structure and style.

Inputs Accuracy image-only (3B) 85.00

image + OCR text & bbox (3B) 80.30 image-only (7B) 87.94 image + OCR text & bbox (7B) 84.41

#### A.8 Prompts

Domain and Type Classification Prompt

You are an expert document quality evaluator and domain classifier. Your task is to assess the professionalism, layout quality, and readability of documents based on their visual appearance, and classify the document’s domain.

You will be provided with screenshot images of document pages. First, classify the document domain and then evaluate the document on quality criteria.

**DOMAIN AND DOCUMENT TYPE CLASSIFICATION**: Classify the document on two levels:

- 1. **Domain Classification**: Choose the broad domain category (e.g., technical, personal, legal, scientific, government, financial, medical, business, education, marketing, academic, news, entertainment, sports, non_profit, religious, insurance, real_estate, automotive, travel, hospitality, retail, manufacturing, logistics, etc.)
- 2. **Document Type Classification**: Identify the specific document type within that domain. Examples include:

- - Technical: engineering_report, user_manual, software_documentation, specification_document, etc.
- - Personal: cv, personal_report, resume, personal_letter, etc.
- - Legal: legal_brief, legal_opinion, contract, regulatory_text, court_filing, etc.
- - Scientific: technical_paper, research_publication, scientific_study, laboratory_report, etc.
- - Government: regulation, white_paper, official_report, government_form, policy_document, etc.
- - Financial: audit_report, investment_report, financial_statement, banking_document, etc.
- - Medical: pharmaceutical_document, clinical_report, medical_manual, research_study, etc.
- - Business: corporate_memo, business_plan, presentation, financial_report, marketing_brochure, etc.
- - Education: thesis, textbook, academic_report, research_paper, course_material, etc.
- - Marketing: brand_guidelines, campaign_brief, advertising_proposal, market_analysis, social_media_strategy, etc.
- - Academic: dissertation, grant_proposal, conference_paper, journal_article, literature_review, etc.
- - News: press_release, news_article, interview_transcript, editorial, media_kit, etc.
- - Entertainment: production_notes, script, event_program, casting_call, performance_review, etc.
- - Sports: athlete_profile, game_report, coaching_guide, training_manual, tournament_bracket, etc.
- - Non_profit: annual_report, fundraising_proposal, impact_report, volunteer_handbook, grant_application, etc.
- - Religious: ceremony_program, sermon_notes, prayer_book, religious_text, pastoral_letter, etc.
- - Insurance: claims_form, policy_document, underwriting_report, risk_assessment, coverage_summary, etc.
- - Real_estate: lease_agreement, property_listing, market_analysis, appraisal_report, property_brochure, etc.
- - Automotive: parts_catalog, service_manual, recall_notice, safety_report, warranty_document, etc.
- - Travel: travel_guide, itinerary, visa_application, booking_confirmation, hotel_brochure, etc.
- - Hospitality: staff_handbook, menu, guest_services_guide, reservation_system, event_planning_document, etc.

- - Retail: inventory_report, product_catalog, customer_survey, sales_analysis, store_policy, etc.
- - Manufacturing: production_schedule, quality_control_report, equipment_manual, safety_protocol, process_documentation, etc.
- - Logistics: delivery_schedule, shipping_manifest, transportation_plan, warehouse_inventory, supply_chain_analysis, etc.

Choose the most specific and accurate document type that describes the document’s purpose and content. You may use other document types not listed above if they better describe the document.

Document Scoring Prompt for Proprietary Models (point-wise)

You are an expert document quality evaluator. Your task is to assess the professionalism, layout quality, and readability of documents based on their visual appearance.

You will be provided with screenshot images of document pages. Evaluate the document on the following criteria:

- 1. **Layout and Design**:

- - Professional appearance and visual appeal
- - Consistent formatting and spacing
- - Proper use of headings, subheadings, and hierarchy
- - Appropriate margins and white space usage
- - Overall visual balance and organization

- 2. **Readability and Typography**:

- - Font choices and consistency
- - Text size and legibility
- - Line spacing and paragraph structure
- - Text alignment and justification

- 3. **Professional Standards**:

- - Document structure and organization
- - Use of professional elements (headers, footers, page numbers)
- - Consistency across pages (if multiple pages provided)
- - Overall polish and attention to detail

- 4. **Visual Elements**:

- - Quality and placement of images, tables, or charts
- - Integration of visual elements with text
- - Professional presentation of data Rate the document on a scale from 0 to 10, where:
- - 9 to 10: Exceptional professional quality
- - 7 to 8: High professional standard
- - 5 to 6: Good professional appearance
- - 4: Average / acceptable quality
- - 2 to 3: Below average, needs improvement
- - 0 to 1: Poor quality, significant issues Your response should follow this format:

- 1. First, provide a detailed analysis of each evaluation criteria mentioned above
- 2. Then, conclude with a final numerical score on a new line starting with "SCORE: " followed by the number (e.g., "SCORE: 7.250")

Document Scoring Prompt for Proprietary Models(Pair-wise)

You are an expert document quality evaluator. Your task is to compare two documents and determine which one has better professionalism, layout quality, and readability based on their visual appearance.

You will be provided with screenshot images of all pages from two documents: Document A and Document B. Compare the documents on the following criteria:

- 1. **Layout and Design**:

- - Professional appearance and visual appeal
- - Consistent formatting and spacing
- - Proper use of headings, subheadings, and hierarchy
- - Appropriate margins and white space usage
- - Overall visual balance and organization

- 2. **Readability and Typography**:

- - Font choices and consistency
- - Text size and legibility
- - Line spacing and paragraph structure
- - Text alignment and justification

- 3. **Professional Standards**:

- - Document structure and organization
- - Use of professional elements (headers, footers, page numbers)
- - Consistency across pages
- - Overall polish and attention to detail

- 4. **Visual Elements**:

- - Quality and placement of images, tables, or charts
- - Integration of visual elements with text
- - Professional presentation of data

Your response should follow this format:

- 1. First, provide a detailed comparative analysis of each evaluation criteria for both documents
- 2. Then, conclude with your preference on a new line starting with "PREFERENCE: " followed by either "A" or "B" (e.g., "PREFERENCE: A", "PREFERENCE: B")

Choose the document that demonstrates superior overall quality, professionalism, and visual presentation.

Document Scoring Prompt for Proprietary Models (triple-wise)

You are an expert document quality evaluator. Your task is to compare two documents and determine which one has better professionalism, layout quality, and readability based on their visual appearance.

You will be provided with screenshot images of all pages from three documents: Document A, Document B, and the Original document (ground truth reference). The Original document serves as a reference standard. Compare Documents A and B on the following criteria:

- 1. **Layout and Design**:

- - Professional appearance and visual appeal
- - Consistent formatting and spacing
- - Proper use of headings, subheadings, and hierarchy
- - Appropriate margins and white space usage
- - Overall visual balance and organization

- 2. **Readability and Typography**:

- - Font choices and consistency
- - Text size and legibility
- - Line spacing and paragraph structure
- - Text alignment and justification

- 3. **Professional Standards**:

- - Document structure and organization
- - Use of professional elements (headers, footers, page numbers)
- - Consistency across pages
- - Overall polish and attention to detail

- 4. **Visual Elements**:

- - Quality and placement of images, tables, or charts
- - Integration of visual elements with text
- - Professional presentation of data

Your response should follow this format:

- 1. First, provide a detailed comparative analysis of each evaluation criteria for both documents, taking the Original document as reference for quality standards
- 2. Then, conclude with your preference on a new line starting with "PREFERENCE: " followed by either "A" or "B" (e.g., "PREFERENCE: A", "PREFERENCE: B")

Choose the document that demonstrates superior overall quality, professionalism, and visual presentation.

Prompt for Document Generation

Based on the following plain text content (extracted from a DOCX document), generate Python code using python-docx library to create a new, well-formatted DOCX document with appropriate styles and formatting:

Plain Text Content (no formatting): {editing_plan}

Output file: {output_file_path} TASK OVERVIEW: You are given ONLY the plain text content of a document (without any formatting,

styles, or structure information). Your job is to:

- 1. Analyze the text content to infer document structure (headings, paragraphs, lists, etc.)
- 2. Create a new DOCX document from scratch
- 3. Apply appropriate professional formatting and styles to make it look like a proper document
- 4. Add visual hierarchy, consistent formatting, and professional appearance IMPORTANT REQUIREMENTS:

- 1. Create a completely NEW DOCX document based on the plain text content
- 2. **PRESERVE ALL TEXT CONTENT**: Include every single word, sentence, paragraph, and character from the given plain text content. Do NOT omit, skip, or modify any text content.
- 3. **NO CONTENT CHANGES**: Only infer and apply formatting/structure. The actual text content must remain exactly the same as provided.
- 4. Analyze the text content to infer document structure and apply appropriate formatting
- 5. Generate Python code that creates a professional-looking document with proper hierarchy and styling
- 6. Ensure ALL provided text appears in the final document in the original order
- 7. **YOUR CODE WILL BE EXECUTED**: The generated Python code will be run directly, so it must be complete, executable, and include the document.save() function to save the DOCX file to the specified output path.
- 8. **DO NOT USE PLACEHOLDERS OR OMITTED CODE**: The generated code MUST be complete and explicit. Do NOT use comments or placeholders such as "# ... (Continue to add other sections and paragraphs similarly)" or "# Add more content here". The code must include ALL content from the original plain text, fully processed and added to the document.

**OUTPUT PATH REQUIREMENTS:**

- - You MUST use the exact output path provided: {output_file_path}
- - DO NOT create your own filename or path
- - DO NOT save to current directory with arbitrary names like ’output.docx’, ’document.docx’, etc.
- - DO NOT use variables like ’output_path’ without setting them to the exact provided path

CODE STRUCTURE REQUIREMENTS: Your generated Python code must follow this EXACT structure:

‘‘‘python import os from docx import Document from docx.shared import Inches, Pt from docx.enum.text import WD_ALIGN_PARAGRAPH from docx.enum.style import WD_STYLE_TYPE # Add other imports here...

# Create new document doc = Document()

# Add content here with appropriate formatting # Process the text content and add to document...

# Create output directory if needed os.makedirs(os.path.dirname(output_file_path), exist_ok=True) try:

print(’CODE: output_file_path = ’, output_file_path) except:

print(’CODE: output_file_path ERROR! ’) doc.save(output_file_path) ‘‘‘

- Prompt for Document Refinement (Phase 1 - Plan Generation)

You are a document formatting analysis expert. Your task is to analyze the differences between a previously generated document and the ground truth document, then create a specific refinement plan.

**Input Information:**

- **1. Previous Generated Code:** ‘‘‘python {previous_code} ‘‘‘
- **2. Previous Generated Document Screenshot:** {previous_doc_screenshot_info}
- **3. Ground Truth Document Screenshot:** {gt_screenshot_info}
- **4. Ground Truth Document Representation:** ‘‘‘ {gt_doc_repr} ‘‘‘

**Important Context Limitations:** Due to input context length constraints, the Ground Truth Document

Representation, Ground Truth Document Screenshot, and Previous Generated Document Screenshot may only contain the initial/front portions of the

documents. However, the Previous Generated Code is complete and contains the full implementation. When analyzing differences, focus primarily on the visible portions but consider that the documents may extend beyond what is shown.

**Task:** Compare the previous generated document with the ground truth document. Identify

the 5 most important differences and create a specific, actionable refinement plan with concrete implementation details needed to modify the previous generated code.

**Output Format:** Provide a detailed refinement plan with specific values and implementation

details: ## Top 5 Key Differences and Improvements Needed: For each improvement, specify:

- 1. **Location/Text**: Where the issue occurs (partial text content for identification, table position, paragraph number, etc.)
- 2. **What needs to be changed** (exact element/section)
- 3. **Current state** (what the code currently does)
- 4. **Target state** (what it should be)
- 5. **Specific implementation** (exact font sizes, spacing values, alignment settings, etc.)

### Example format:

**Issue**: [Specific formatting problem]

- - **Location**: Text containing "Document Header" or Table in section 2, row 1
- - **Current**: Font size 12pt, left alignment
- - **Target**: Font size 14pt, center alignment
- - **Implementation**: Set ‘run.font.size = Pt(14)‘ and ‘paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER‘

**Issue**: [Table formatting problem]

- - **Location**: Table with headers "Product Name, Price"
- - **Current**: No borders, default spacing
- - **Target**: 1pt black borders, 6pt cell padding
- - **Implementation**: Add table border properties with ‘width=1pt, color=black‘ and set cell margins to ‘6pt‘

Focus on providing exact values (font sizes in pt, spacing in pt/inches, specific color values, alignment constants) and concrete python-docx implementation steps. **Limit to exactly 5 most important differences** that will have the biggest visual impact.

- Prompt for Document Refinement (Phase 2 - Code Generation)

You are a document generation expert. Your task is to generate improved Python code that addresses the specific formatting issues identified in the refinement plan.

**Input Information:**

- **1. Previous Generated Code:** ‘‘‘python {previous_code} ‘‘‘
- **2. Refinement Plan:** ‘‘‘ {refinement_plan}

- ‘‘‘
- **3. Output File Path:**

- Output file: {output_file_path}

**Task:** Based on the previous code and the refinement plan, generate a **complete and

improved Python code** that creates a document matching the ground truth as closely as possible. This should be a standalone, executable script that generates the entire document from scratch.

**Requirements:**

- 1. **Generate complete Python code** - not just modifications, but a full working script
- 2. **Apply all improvements** specified in the refinement plan
- 3. **Create the entire document** structure and content to match ground truth
- 4. **Use appropriate libraries** (python-docx for high-level operations, direct XML manipulation for precise control)
- 5. **Include error handling** for robustness
- 6. **Save to specified output path** - the code must generate a complete document file
- 7. **DO NOT use main() function wrapper** - code should execute directly at top level
- 8. **Use exact output path provided**: {output_file_path}

**CODE STRUCTURE REQUIREMENTS:** Your generated Python code must follow this structure (NO main() function):

‘‘‘python import os from docx import Document from docx.shared import Inches, Pt from docx.enum.text import WD_ALIGN_PARAGRAPH # Add other imports as needed...

# Create new document doc = Document()

# Add all content here with appropriate formatting # Apply all improvements from refinement plan...

# Save the document output_file_path = "{output_file_path}" os.makedirs(os.path.dirname(output_file_path), exist_ok=True) doc.save(output_file_path) print("CODE: output_file_path = ", output_file_path) ‘‘‘

**Advanced Formatting Capabilities:**

- - **python-docx API**: Use for standard document operations
- - **Direct XML manipulation**: Use when python-docx doesn’t provide sufficient control
- - Access underlying XML: ‘element._element‘
- - XPath queries: ‘element.xpath()‘
- - Direct attribute setting: ‘element.set()‘ on XML nodes
- - Namespace operations: Use ‘qn()‘ for proper namespace handling
- - Document XML access: ‘document.element.body‘ for document-level changes

**Code Structure:** The code should be a complete script that:

- - Creates a new document
- - Builds the entire document structure and content
- - Applies all formatting to match the ground truth

- - Saves the complete document to output_file_path

**Output Format:** Provide a complete, executable Python script that implements the improvements

specified in the refinement plan.

**XML Manipulation Reference:** When python-docx API is insufficient, you can use direct XML manipulation. Here

are helper functions and examples for reference:

*Helper functions (include only if needed):* ‘‘‘python def set_xml_attribute(element, attr_name, attr_value):

"""Set XML attribute directly on element""" if hasattr(element, ’_element’):

element._element.set(qn(attr_name), attr_value) else:

element.set(qn(attr_name), attr_value)

def add_xml_element(parent, tag_name, **attributes): """Add XML element with attributes""" element = OxmlElement(qn(tag_name)) for attr, value in attributes.items():

element.set(qn(attr), value) parent.append(element) return element

‘‘‘

*Example XML operations:*

- - For precise spacing control: ‘p_element = paragraph._element; spacing_element

= add_xml_element(p_element, ’w:spacing’, before="120", after="120")‘

- - For table borders: ‘table_element = table._element; table_props = add_xml_element(table_element, ’w:tblPr’)‘
- - For direct attribute setting: ‘element._element.set(qn(’w:val’), ’value’)‘

**Focus on:**

- - Precise implementation of the refinement plan using both python-docx API and direct XML manipulation
- - Proper python-docx syntax and XML node manipulation for fine-grained control
- - Maintaining document integrity while applying improvements
- - Clear, maintainable code structure with comprehensive error handling
- - Complete document generation (not just partial modifications)

