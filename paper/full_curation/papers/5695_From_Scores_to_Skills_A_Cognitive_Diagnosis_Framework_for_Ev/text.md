# arXiv:2508.13491v2[cs.CE]23Aug2025

## From Scores to Skills: A Cognitive Diagnosis Framework for Evaluating Financial Large Language Models

### Ziyan Kuang1,2, Feiyu Zhu1,2, Maowei Jiang3, Yanzhao Lai4, Zelin Wang1,2, Zhitong Wang5, Meikang Qiu6, Jiajia Huang3, Min Peng1,2, Qianqian Xie1,2*, Sophia Ananiadou7

1School of Artifical Intelligence, Wuhan University 2Center for Language and Information Research , Wuhan University 3School of Computer Science, Nanjing Audit University 4Southwest Jiaotong University 5Beijing University of Financial Technology

- 6Computer and Cyber Sciences, Augusta University
- 7Computer Science, University of Manchester

###### Abstract

Large Language Models (LLMs) have shown promise for financial applications, yet their suitability for this high-stakes domain remains largely unproven due to inadequacies in existing benchmarks. Existing benchmarks solely rely on score-level evaluation, summarizing performance with a single score that obscures the nuanced understanding of what models truly know and their precise limitations. They also rely on datasets that cover only a narrow subset of financial concepts, while overlooking other essentials for real-world applications. To address these gaps, we introduce FinCDM, the first cognitive diagnosis evaluation framework tailored for financial LLMs, enabling the evaluation of LLMs at the knowledge-skill level, identifying what financial skills and knowledge they have or lack based on their response patterns across skill-tagged tasks, rather than a single aggregated number. We construct CPA-KQA, the first cognitively informed financial evaluation dataset derived from the Certified Public Accountant (CPA) examination, with comprehensive coverage of real-world accounting and financial skills. It is rigorously annotated by domain experts, who author, validate, and annotate questions with high inter-annotator agreement and fine-grained knowledge labels. Our extensive experiments on 30 proprietary, open-source, and domain-specific LLMs show that FinCDM reveals hidden knowledge gaps, identifies under-tested areas such as tax and regulatory reasoning overlooked by traditional benchmarks, and uncovers behavioral clusters among models. FinCDM introduces a new paradigm for financial LLM evaluation by enabling interpretable, skillaware diagnosis that supports more trustworthy and targeted model development, and all datasets and evaluation scripts will be publicly released to support further research.1

### 1 Introduction

Large language models (LLMs) are increasingly applied to financial tasks (Nie et al. 2024a), but their suitability for such high-stakes domains remains largely untested (Xie et al. 2024). While numerous open-domain benchmarks such as MMLU (Hendrycks et al. 2021), HELM (Liang et al. 2022),

*Corresponding author. Email: xieq@whu.edu.cn 1https://github.com/WHUNextGen/FinCDM

and HLB (Duan et al. 2024) have demonstrated the excellent general capabilities of LLMs like GPT-4 (Achiam et al. 2023), DeepSeek (Liu et al. 2024a), and LLaMA (Touvron et al. 2023), their effectiveness in the financial domainspecific tasks remains largely unknown.

To address this uncertainty, several benchmarks (Xie et al. 2023, 2024; Li et al. 2024a; Peng et al. 2025b), have been recently proposed to systematically evaluate the capabilities of LLMs specifically within the financial domain. However, existing financial LLM benchmarks (Xie et al. 2024; Li et al. 2024a; Peng et al. 2025b) often fail to reflect what matters for those tasks and applications, namely what the model knows, what it can reliably do, and where it is likely to fail. First, this issue stems from score flattening, where a dataset in existing benchmarks is reduced to a single number, making it unclear what knowledge the model has actually mastered. For example, in FinQA from MultiFinBen (Peng et al. 2025b), GPT-4o, a general-domain LLM, and FinMA (Xie et al. 2023), a financial LLM, achieve similar overall accuracy scores. However, GPT-4o tends to perform better on numerical computation tasks such as calculating the “net change in cash”, while FinMA is stronger at handling finance-specific conceptual questions, such as identifying “total equity”. Second, many prior datasets also exhibit coverage imbalance, where examples disproportionately rely on certain types of financial knowledge or skills, making it difficult to evaluate how models perform across the full range of real-world requirements. For example, in the accounting questions of FinEval (Guo et al. 2025), most items focus on a narrow set of concepts such as total revenue and net income, while overlooking other areas like equity changes or tax-related components. To address these limitations, we introduce FinCDM, the first cognitive diagnosis evaluation framework for financial LLMs. Inspired by cognitive diagnosis model (CDM) in educational assessment, FinCDM evaluates LLMs like human examinees, identifying what financial skills and knowledge they have or lack based on their response patterns across skill-tagged tasks, rather than a single aggregated number. First, we apply the

|question number: Q1|
|---|
|Question: Which of the following statements regarding tax and fee accounting is incorrect? ( )<br><br>A: When an enterprise incurs stamp duty, it should debit the "Taxes and Surcharges" account and credit the "Taxes and Fees Payable" account.<br>B: When an enterprise receives a refund of overpaid land value-added tax, it should debit the "Bank Deposits" account and credit the "Taxes and Surcharges" account.<br>C: Vehicle and vessel tax is calculated and paid based on the applicable tax amount. When an enterprise calculates the vehicle and vessel tax payable according to regulations, it should debit the "Taxes and Surcharges" account and credit the "Taxes and Fees Payable - Vehicle and Vessel Tax Payable" account.<br>D: When an enterprise calculates and pays the farmland occupation tax according to regulations, it should debit the "Construction in Progress" account and credit the "Bank Deposits" account.<br><br><br>Answer: B|

- K1

- K2

- K3

|Q1|
|---|

|Q1|
|---|

CPA Knowledge Taxonomy

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

- Topic-Skill 1
- Topic-Skill 2
- Topic-Skill 3

|Q2|
|---|

|Q2|
|---|

|1|0|0|0|0|
|---|---|---|---|---|
|0|1|1|0|0|
|1|0|1|1|0|
|0|1|0|1|1|
|0|0|1|1|1|

B7 Quality Management

- 1 1 0 1 0

- 0 1 1 1 1
- 1 1 1 1 1

- 1 1 1 0 1 0 1 1 1 1

...

- Matrix U LLMs’ Proficiency in Latent Skills

Matrix E Question–Skill Association

- Matrix V Skill–Concept Mapping

B2 Audit Test Process

A(19) Income Tax

|Q3|
|---|

|Q3|
|---|

F Tax Law

A Accountant

Kn

B Audit

A8 Liabilities

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

B1 Basic principles of audit

A(21) Debt Restructuring

- Topic-Skill 1
- Topic-Skill 2
- Topic-Skill 3

Answer Matrix X

Matrix F Model–Concept Mastery

FinCDM

human annotation

|Knowledge Points: A8 Liabilities|
|---|

|1|1|0|0|0|
|---|---|---|---|---|
|0|1|1|0|0|
|1|0|0|1|0|
|0|1|0|1|1|
|0|0|1|0|1|

Previous Concept 📘

|Question Authoring Order|
|---|

[Figure 1]

Annotation Opinions

Expert B: A8 Liabilities

Expert C: A8 Liabilities

###### Knowledge Points K1 K2 K3 ... Kn Performance

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

- Topic-Skill 1
- Topic-Skill 2
- Topic-Skill 3

|Q1|
|---|
|Knowledge Points: A8 Liabilities|

[Figure 2]

Concept Matrix Q

opinions consistent?

Cognitive Diagnosis

CPA-QKA Dataset

Skill

ACC

Figure 1: The whole framework of FinCDM.

similar overall scores on FinQA, highlighting how traditional aggregate metrics can obscure underlying capability variation. Second, our evaluation illustrates that prior benchmarks predominantly assess a narrow subset of financial concepts, resulting in inadequate coverage. In contrast, our new dataset systematically reveals previously overlooked weaknesses in areas like deferred tax liabilities, lease classification, and regulatory ratios, which are concepts rarely tested in existing benchmarks but critical in real-world applications. Third, by analyzing the knowledge-skill level mastery profiles produced by FinCDM, we identify latent associations between financial concepts and reveal distinct clusters of models exhibiting similar skill acquisition patterns. For example, GPT-3.5 and DeepSeek-VL share strengths in financial reporting and valuation, while FinGPT and FinQwen exhibit aligned capabilities in regulation and macroeconomic reasoning, reflecting different domain specialization strategies.

non-negative matrix co-factorization based CDM model to estimate LLM’s mastery on different financial skills with collected responses of LLMs, addressing the problem of score flattening. CDMs assume that the model’s answer to a task reflects whether the model possesses all the underlying skills required by the question. Therefore, by decomposing the responses to all questions based on knowledge labels annotated for each task, we can assess the model’s proficiency in different knowledge, revealing where it excels and where it struggles. Second, we introduce the first Chinese dataset named CPA-QKA rigorously annotated by domain experts, specifically constructed to support the knowledgeskill level evaluation of FinCDM, and provide comprehensive evaluation coverage. The dataset is grounded in the Certified Public Accountant (CPA) exam, the most recognized professional certification in accounting and financial reporting. We build the financial knowledge framework covering 70 key concepts from the content and skill specification outline of the CPA exam, ensuring comprehensive coverage of real-world financial concepts. Under the framework, our dataset construction involves a rigorous, two-phase annotation process. Domain experts first author multiple questions for per concept, and then perform verification to ensure accuracy, consistency, and high inter-annotator agreement (IAA). These verified questions are annotated with precise financial knowledge and skill tags, also employing IAA-based quality control. By annotating existing benchmarks using our framework, we further highlight their limitations, illustrating their narrow coverage of financial concepts and underscoring the need for the broader and deeper skill assessment that FinCDM provides. Figure 1 illustrates the complete workflow of FinCDM.

In summary, our major contributions are:

- 1. We propose FinCDM, the first cognitive diagnosis framework for financial LLM evaluation, which moves beyond aggregate metrics by assessing models’ proficiency at the knowledge-skill level.
- 2. We construct a new dataset and structured financial knowledge framework derived from the CPA exam, with high-quality human annotations. This supports reliable evaluation of FinCDM and exposes the narrow coverage of existing benchmarks.
- 3. We apply FinCDM to a broad set of 30 proprietary, opensource, and finance-specific models, uncovering their knowledge gaps, strengths, and behavioral patterns, offering actionable insights for model development and deployment.

Based on FinCDM, we evaluate 30 representative LLMs, including proprietary models, open-source general-purpose models of varying sizes, and finance-specific models. Our results support three key findings. First, FinCDM reveals significant differences in models’ mastery of financial knowledge that are not captured by traditional aggregate metrics. For example, Doubao shows strong performance on Chinese-specific regulations and specialized accounting areas, while Gemini demonstrates superior understanding of “Debt Restructuring”, “Lease”, “Post-Balance Sheet Events”, showcasing robust mastery in general accounting concepts. Despite these differences, both models achieve

### 2 Related Work

#### Financial Benchmark

Numerous benchmarks have been developed to evaluate LLMs in financial domains, covering tasks such as extraction, QA, reasoning, simulation, and retrieval. Early efforts like PIXIU (Xie et al. 2023) and FinBen (Xie et al. 2024) offer broad coverage but rely on aggregate metrics. Benchmarks like FinanceBench (Islam et al. 2023), BizBench

(Krumdick et al. 2024), and BizFinBench (Lu et al. 2025) focus on QA and business reasoning, while multilingual and low-resource evaluation is addressed by CFinBench (Nie et al. 2024b), CFLUE (Zhu et al. 2024), FinEval (Zhang

- et al. 2023), Golden Touchstone (Wu et al. 2024), Plutusben (Peng et al. 2025a), and the more recent MultiFinBen (Peng et al. 2025b). FinMTEB (Tang and Yang 2025) cover embedding-based retrieval and classification. Structured reasoning is evaluated in FinDABench (Liu et al. 2024b), Fino1 (Qian et al. 2025), FinChain (Xie et al. 2025), and agentic decision-making in InvestorBench (Li
- et al. 2024a), AlphaFin (Li et al. 2024b), and AveniBench (Klimaszewski et al. 2025). Multimodal understanding is addressed by M3FinMeeting (Zhu et al. 2025)and FinAudio (Cao et al. 2025).FinTagging (Wang et al. 2025) and FinDER (Choi et al. 2025) focus on fine-grained financial concept extraction and retrieval-augmented QA from 10-K filings and XBRL reports. Despite their breadth, these benchmarks rely on task-level aggregate metrics and lack concept-aware diagnostics.

#### Financial Dataset Design

A wide range of data sets have been developed to evaluate financial LLM in reasoning, QA, information extraction, summarization, and multimodal tasks. Numerical reasoning datasets include FinQA (Chen et al. 2021), TATQA (Zhu et al. 2021), ConvFinQA (Chen et al. 2022), DocFinQA (Reddy et al. 2024), and FinanceQA (Mateega, Georgescu, and Tang 2025), which require multistep or conversational numerical inference over financial reports. FinTextQA (Chen et al. 2024), FinLLMs (Yuan et al. 2024), and FinTruthQA (Xu et al. 2024) focus on long-form or formula-based QA, while SEC-QA (Lai et al. 2024) and MultiHiertt (Zhao et al. 2022) introduce multidocument and table-text hybrid challenges. Structured extraction is addressed by FinTagging (Wang et al. 2025), FiNERORD (Shah et al. 2023), FiNER-139 (Loukas et al. 2022), FinRED (Sharma et al. 2022), REFinD (Kaur et al. 2023), which allow evaluation of concept and relationship levels aligned with financial taxonomies. For numeral understanding, FinNum (Chen et al. 2019), FinNum-2 (Chen et al. 2020), and FiNCAT (Ghosh and Naskar 2022) annotate finegrained semantic types or numeral claims in finance text. The summarization is covered by ECTSum (Mukherjee et al. 2022), FNS-2020 (El-Haj et al. 2020), focusing on earnings calls and annual reports. Chinese and multilingual resources include FinEval (Zhang et al. 2023), CFLUE (Zhu et al. 2024), and UCFE (Yang et al. 2024), enabling QA and NLU in non-English contexts. Multimodal datasets such as FinMME (Luo et al. 2025) and FinLMM-R1 (Lan et al. 2025) support chart, image, and document–text alignment. Despite covering diverse formats and tasks, most existing datasets are task-driven with limited concept coverage.

#### Cognitive Diagnosis Model

Cognitive diagnosis models (CDMs), rooted in educational assessment, aim to infer mastery of latent knowledge attributes from observed response behaviors. Early interpretable models such as the Deterministic Inputs, Noisy

’And’ gate model (DINA)(De La Torre 2009), the Deterministic Inputs, Noisy ’Or’ gate model (DINO)(Templin and Henson 2006), and the Generalized DINA (GDINA) model(De La Torre 2011) employ binary latent attributes and probabilistic response functions; while valuable for interpretation, they depend heavily on accurate Q matrix specification and strict parametric assumptions (Gu and Xu 2019). To mitigate these limitations, matrix factorizationbased approaches, such as MF-DINA and logistic matrix factorization, embed examinees and attributes into low-dimensional spaces, enabling more flexible and robust modeling of item-attribute interactions. Neural networkbased CDMs such as NeuralCD (Wang et al. 2022) and recent graph-based models such as RCD (Gao et al. 2021) use deep architectures and disentangled graph learning to model nonlinear and noise-robust concept, exercise, and student representations. These neural approaches offer superior predictive power on large datasets but can suffer from overfitting and reduced interpretability when evaluated with few entities and many attributes.

### 3 Method

#### Preliminaries

In educational psychology, assessment plays a central role not only in certifying achievement but also in supporting learning and guiding instruction. Rather than reporting a single aggregate score, modern diagnostic assessments aim to provide interpretable and actionable insights into learners’ mastery of specific knowledge components (Frederiksen, Mislevy, and Bejar 1993; Leighton and Gierl 2007). This fine-grained feedback is foundational to formative assessment, enabling educators to design targeted interventions and fostering trust in the evaluation process (Kuh et al. 2011).

Cognitive diagnosis models (CDMs) (Leighton and Gierl 2007) were developed to meet these goals by explicitly modeling how a learner’s observed responses on assessment items reflect their their latent mastery of specific knowledge concepts. In the CDM framework, a particular knowledge domain is defined by K latent knowledge components (or attributes), and each learner is associated with a binary mastery profile α ∈ {0,1}K, where αk = 1 indicates mastery of the k-th skill. Each assessment item i ∈ {1,...,I} is linked to a subset of these skills through the expert-defined binary Q-matrix Q ∈ {0,1}I×K, where qik = 1 signifies that answering item i correctly requires mastery of skill k. Given a learner’s binary response vector X ∈ {0,1}I, where Xi = 1 denotes a correct answer to item i, CDMs estimate the probability of correctly answering each item by modeling the relationship among the learner’s mastery profile α, the item’s skill requirements qi, and additional model-specific parameters θ:

P(Xi = 1 | α,Q,θ) = fi(α,qi;θ)

where, fi denotes the item response function for item i. The primary learning objective in CDMs is to accurately infer each learner’s skill mastery profile α and estimate the

n

s

A18Govenmen

- n
- o

o

A(2 0)

a

g

- d
- e

- n
- o

e

n

o

pro

n

R e

a sset

p

d

a

A(2 1)

n

- m
- n

a

- o

o

a

Subsidy

o

d

H

- m

e

exp

e

- n

- o

u

- p

n

a

stru ct

cu

e

e

5

N o

e

d

co

a

1

A(1 9) n co m e

- o
- p

g

u

A

A(2 2) F

n

A(23) Fin an cial

cial

uity A(1

re n

ex ch a n g e

- n-

- m

o

- n etary

n

a

D e

A(2 4)

p

s

ents

olicies, acco u ntin g

urin g

ntal

an

estimates a n d

7)

cy co

bt

q

se

ch a n g

ent

Acco u

E

Rep ort

Fin

e

s

oreig n

m

co

g

es a n

R

Contin

ntin g

stru

rectio ns

nversi o n

er

A(13)

A(25)

4)

- d
- e ror

mater

th eir

n

A(1

In

w

balan ce

Tax

O

Eventsafter

A(12)

A(26) Bu sine ss

6)

wing fee

A(1

sheet

Merger

A(27) Con

yment

Financial

ro

Bo

Pa

salary

Statements A(28) Earningsper

1)

solidated

Share

A(

yee

Liabilities A9

A(10)

share

mplo

A(29) Fair value measurement

E

A8

sset Impairment

A(30) Government and

- A5 Investment Real

Estate

- A6 Long-term equity

investment and joint venture arangement

non-profit accounting

A7 A

F1 General Tax Law

nt

F2 VAT

nta

A4 Intangible Assets

u

cco

- F3 ConsuLamwption Tax

- F4 CorTpaoxra Ltaew Income

- F5 Personal Income TaxLaw

A

A3 Fixed Assets

A

A2 Inventory

A1 Total

F Tax Law

F6 Urban maintenance and constructiontax lawandtobaccoleaf taxlaw

E4 EconomicLegal System

E EconomicLaw

E3 ComSmysteercimal Legal

F7 Tariff Law and Ship Tonnage Tax Law

C

F8 Resource Tax Law and Environmental Protection Tax Law

cost

Financial

E2S Cystivielm Legal

management

E1 Baosifc lapwrinciples

F9 Urban land use tax law and

Ri sk

D

C Strate

dit

cultivated land occupation tax law

Ma n a g e

Au

- o

- m

p a

- n y

C6 Management Accounting

F(10) Real Estate Tax Law, Deed Tax Law and Land Value

g

B

C5 Cost Calculation

y a n d

Added Tax Law

m e nt

F(11) Vehicle Purchase Tax Law, Vehicle and Vehicle Tax Law and Stamp Tax Law

C4 Working Capital Management

F(12) International Taxation and Tax

Management Practice

C3

fundraising

Colection

Long-term

Administration

C2

invest

F(13) Tax

decision

Legal

Long-term

Law

Tax

ciples

ment decision

Manage ment Basi cs

and

C1 Finan cial

Administrative

F(14)

System

cprin

audit

D8 The

Basi

faced by

st

of

enterprises and their responses

on

D7 Risk

Man ag e m ent

Te

D 3

B1

D6 Ri sk an d Ri sk

process, syste m and

us

s

main risks

ss

Audit

eration

vario

ters

and

Strate gic

ce

balances

manage ment

Pro

B2

ns

a

method

of

m

udit

udit

transactio

sid

udit

D 5

cial

a

a

I m

G overn

Con

Overvie w

unt

D 4

A

mplete

ue

SaegcManagemen

C orp

C h

e

ple

3

a

acco

B

sp

e

nd iss

ntrol

port

n

Strate gic

nt

nterpris

4

oi ce

D1Ovevewo

ality

o

m e ntatio

Saegyand

B

D 2

e

ssi

orate

o

re

a n

cs

o

m

- B

5

- C

A n a

a

C

udit

u

e

e

S ra e g c

ce

ork

h

Q

Internal

g

Pro

E

a

A

E

w

7

ysis

6

n

n

B

B

a

8

M

B

Figure 2: 70 financial concepts covered by CPA-KQA.

model parameters θ from observed response data. By inferring fine-grained knowledge states, CDMs enable transparent and skill-specific evaluation, offering richer insights than traditional test scores.

#### Dataset Curation

Existing benchmarks typically rely on aggregate score evaluations and cover only a limited subset of financial concepts. To address these limitations, we propose a new dataset CPAKQA with domain experts-authored financial questions, to support the knowledge-skill level evaluation of financial LLMs using CDMs, and provide comprehensive knowledge coverage. To construct CPA-KQA, we referenced the CPA examination, a highly recognized professional certification in the financial industry, to ensure a comprehensive coverage of financial knowledge. It covers core financial areas including accounting, auditing, financial cost management, corporate strategy and risk management, economic law, and tax law. We build the financial knowledge framework covering 70 core concepts, derived from the content and skill specification outline of the CPA exam 2, such as “fixed assets”, “liabilities”, and “long-term investment decisions”, as shown in figure 2.

Expert Annotation and Quality Validation: Our data annotation involves two phases including question authoring and financial concepts tagging. The annotation team comprised three domain experts: an undergraduate student, a master’s student, and an associate professor specializing in finance. For each of the 70 financial concepts, three distinct questions were crafted by three annotators to ensure multiple cognitive perspectives, resulting in a total of 210 high-

2https://www.cicpa.org.cn/xxfb/tzgg/202502/ W020250228535255887805.pdf

Dataset Metric CPA-KQA Fineval-KQA

|Krippendorff’s Alpha<br><br>|0.937 -|
|---|---|
|σ KS-based measure|0.9904 0.9208 1 1<br><br>|

Table 1: Inter-annotator agreement metrics for human expert annotations on CPA-KQA and Fineval-KQA.

quality items. Annotators were explicitly instructed to craft questions ensuring clarity and precision. To ensure reliability and consistency, we conducted rigorous quality validation. In the first phase, each authored question was independently reviewed by the two other experts (excluding the original author) to assess relevance, clarity, and alignment with the intended financial concept. Discrepancies were resolved through collaborative discussions. Inter-annotator agreement for question-concept mappings in this phase was measured using Krippendorff’s alpha.

In the second stage, all questions are independently annotated by the same two experts, who assign one or more knowledge labels selected from a predefined set of 70 financial concepts. If the annotators’ assigned knowledge concepts are consistent with those originally provided by the question setter, the question proceeds without modification. In cases where the annotators disagree with each other or with the question setter, a three-way discussion is held to resolve the disagreement and determine the final label. Annotation consistency in this stage is measured using the σ score and the KS-based measure (Braylan, Alonso, and Lease 2022).

As shown in Table 1, our annotation achieved high consistency, with a Krippendorff’s alpha of 0.937, a σ score of 0.9904, and a KS-based measure of 1. These results confirm the high reliability and quality of our CPA-KQA dataset. Further annotation details are provided in the appendix.

FinEval-KQA To further reveal the coverage of financial knowledge in existing datasets, we introduce the FinEvalKQA dataset, augmenting the FinEval question set (Zhang et al. 2023), a widely used Chinese financial domain knowledge evaluation benchmark, with concept-level annotations based on the well-established framework of CPA-KQA. The FinEval dataset (Zhang et al. 2023) is a publicly available resource containing over 2,000 questions designed to evaluate financial models on a variety of financial knowledge areas. These questions span across multiple sub-domains, including financial analysis, accounting, and financial forecasting. We apply the CPA-KQA concept taxonomy to annotate a subset of FinEval with 101 questions that focuses on accounting-related knowledge.

Expert Annotation and Quality Validation: Our three expert annotators was tasked with mapping existing FinEval questions to the most relevant CPA-KQA concepts. A guideline document (see appendix for more details) was created to ensure that each question was annotated consistently with respect to the CPA-KQA taxonomy, and the annotators were instructed to focus on clarity, specificity, and relevance. To

ensure high-quality and consistent annotations, the interannotator agreement was evaluated using the same metrics as the original CPA-KQA annotations: σ and KS-based measure. We conducted a reconciliation process in which annotators discussed any discrepancies in their annotations to ensure alignment. The results showed strong consistency, with σ of 0.9208 and KS-based measure of 1, demonstrating the robustness of our annotation process.

#### Evaluation Framework

Building upon our annotated datasets, we now illustrate how we assess the mastery of financial knowledge of different LLMs. Specifically, using our newly constructed CPAKQA, we can obtain the observed response data from LLMs, which forms our response matrix X, and leverage the expert-annotated Q. We employ a non-negative matrix co-factorization inspired by SNMCF (Yu et al. 2023) and DINA (De La Torre 2009), modeling the process of evaluating LLM skill mastery explicitly through a probabilistic generative model. Our framework assumes the following generative process:

- 1. Latent skill representation for questions: For each ques-

tion i, generate a latent skill requirement vector ei from a Gamma prior:

ei ∼ Gamma(a,b), ei ∈ RT≥0.

- 2. Latent skill proficiency for models: For each LLM j, gen-

erate a latent skill proficiency vector uj from a Gamma prior:

uj ∼ Gamma(c,d), uj ∈ RT≥0.

- 3. Latent skill to financial concept mapping: For each financial concept k, generate a latent skill-to-concept association vector vk from a Gamma prior:

vk ∼ Gamma(e,f), vk ∈ RT≥0.

- 4. Generation of observed responses X: Each observed bi-

nary response xij indicating whether LLM j correctly answers question i is drawn from a Bernoulli distribution parameterized by the alignment between the latent skill vectors:

xij ∼ Bernoulli σ(e⊤i uj) , σ(z) =

1 1 + e−z .

- 5. Generation of question-concept association Q: Similarly,

the observed binary association qik, indicating whether question i is related to concept k, is drawn from a Bernoulli distribution parameterized by the alignment between their latent vectors:

1 1 + e−z .

qik ∼ Bernoulli σ(e⊤i vk) , σ(z) =

Through this generative formulation, we aim to factorize the matrices X and Q into low-dimensional latent representations with non-negative constraints:

X ≈ EU, Q ≈ EV, E,U,V ≥ 0

where E ∈ RM≥0×T represents the relationship between questions and T latent skills, capturing how strongly each ques-

tion is related to each latent skill. U ∈ RT≥×0 N captures the proficiency of each model in the latent skills. Each row in U corresponds to a latent skill, and each column represents the

proficiency of model j in that skill. V ∈ RT≥×0 K represents the relationship between latent skills and financial concepts, mapping the latent skills to the relevant concepts for each question. We estimate these latent matrices by optimizing the following joint objective:

∥W◦(X−EU)∥2F +β∥Q−EV ∥2F +λE∥E∥2F +λU∥U∥2F +λV ∥V ∥2F ,

min

E,U,V ≥0

The factorized matrices E, U, and V provide a rich, interpretable structure that reveals the relationships between questions, concepts, and models’ capabilities. See appendix for the optimization process to inference these latent matrices. Finally, we explicitly estimate each LLM’s mastery of financial concepts by combining the learned latent matrices U and V :

F = U⊤V, F ∈ RN≥0×K, F provides detailed, interpretable diagnostics, indicating the proficiency level of each LLM across the financial concepts.

#### Benchmarking

Based on FinCDM, we conduct a benchmark study covering both closed-source and open-source models, including those specifically tailored for the financial domain. In total, our evaluation involves over 30 Chinese-capable LLMs. Detailed model information can be found in Appendix. We evaluate models in the following categories:

- 1. Closed-source general models: These include GPT-4, GPT-4o, and GPT-4o-mini (Achiam et al. 2023); Claude 3.5 Sonnet, and Claude 3.7 Sonnet (Anthropic 2024); Gemini 1.5 Pro, Gemini 1.5 Flash, and Gemini 2.5 Pro Experimental (Team et al. 2024); Grok-3 (xAI 2024); Doubao-1.5-Pro-256k, and Doubao-1.5-Pro-32k (ByteDance 2024).
- 2. Open-source general models: These include Baichuan2-13B-Chat (Baichuan Inc. 2023), ChatGLM36B (Zhipu AI 2024), Falcon-7B (Almazrouei et al. 2023), GLM-4-32B-0414 and GLM-4-9B-0414 (GLM et al. 2024); Qwen2-72B-Instruct, Qwen2.5-7B-Instruct, Qwen3-0.6B, and Qwen3-235B-A22b (Yang et al. 2025; Team 2024) DeepSeek-Chat and DeepSeek-V3-0324 (Bi et al. 2024; Liu et al. 2024a) and Hunyuan (Tencent Hunyuan Team 2025).
- 3. Financial domain models: These include Finma-7bFull (Xie et al. 2023), CFGPT2-7B (Li et al. 2023).

For the evaluation, we prompt (see appendix for the prompt used in our experiments) each LLM to answer the questions in our datasets using a consistent and controlled setup. The decoding configuration is set with a temperature of 1.2 to promote diversity in generated responses, allowing models to explore varied responses under the same input. The maximum generation length is capped at 64 tokens to ensure concise answers, and each prompt instructs the model to return relevant and informative outputs. Each

model generates 10 responses per question. The final performance scores are computed by averaging over these 10 responses, providing a more reliable and robust estimate of model proficiency. All models are evaluated using a unified script and configuration pipeline to ensure fairness and comparability across systems.

#### Main Results

### 4 Results

Table 2: Results on CPA-KQA and Fineval-KQA. The “Con” column means concept, which represents the number of concepts for which the model achieved a mastery probability greater than 0.9. The data is sorted in descending order for ease of viewing.

CPA-KQA Fineval-KQA

Con Model Acc Con Model Acc 40/70 GLM4 0.63 13/38 Gemini1.5 0.66 39/70 Claude3.7 0.77 13/38 Gemini2.5pro 0.87 37/70 Qwen3-235b 0.73 11/38 Claude3.7 0.76 36/70 Doubao32k 0.82 11/38 Doubao1.5pro 0.88 36/70 Qwen-max 0.76 11/38 Qwen2.5-72b 0.77 34/70 Doubao256k 0.84 10/38 Claude3.5 0.69 34/70 Qwen2.5-72b 0.76 8/38 Gpt4 0.75 33/70 Claude3.5 0.74 8/38 Qwen-max 0.75 33/70 Gemini2.5pro 0.84 8/38 Qwen3-235b 0.79 31/70 Gemini1.5 0.64 7/38 DBRX 0.42 29/70 Hunyuan 0.64 7/38 Gpt4o-mini 0.48 27/70 Grok3 0.63 7/38 Qwen2.5-7b 0.68 25/70 Deepseek-v3 0.65 7/38 Glm4 0.69 25/70 Gpt4 0.63 7/38 Baichuan2 0.39 24/70 Deepseek-chat 0.65 6/38 Doubao256k 0.85 24/70 Qwen2.5-7b 0.62 6/38 Gpt4o 0.49 24/70 Glm4-9b 0.56 6/38 Llama2-70b 0.47 22/70 Gemini1.5pro 0.59 6/38 Llama3.1 0.48 21/70 Gpt4o-mini 0.50 6/38 Glm4-32b 0.67 20/70 Gpt4o 0.52 6/38 CFGPT2 0.48 19/70 Glm4-32b 0.57 5/38 Deepseek-chat 0.74 19/70 CFGPT2 0.38 5/38 Deepseek-v3 0.74 17/70 Chatglm3 0.34 5/38 Glm4-9b 0.56 17/70 Baichuan2 0.39 5/38 chatGlm3 0.37 13/70 DBRX 0.38 4/38 Gemini1.5pro 0.54 13/70 Llama2 0.45 4/38 Hunyuan 0.70 11/70 Llama3.1 0.57 3/38 Grok3 0.60 10/70 Qwen3-0.6b 0.24 3/38 Qwen3-0.6b 0.29 9/70 Finma7b 0.19 2/38 Falcon 0.20 0/70 Falcon7b 0.10 1/38 Finma7b 0.19

Table 2 and Figure 3 summarize the knowledge-skill level performance of various LLMs on our CPA-KQA dataset. Our key findings are as follows:

The Role of FinCDM. FinCDM provides a more informative evaluation than aggregate metrics by uncovering how a model’s overall performance is distributed across individual financial concepts, highlighting both strengths and blind spots that remain hidden under traditional evaluation. As shown in Figure 3, Claude 3.5 Sonnet achieves high average accuracy on CPA-KQA, but FinCDM reveals that this

performance is concentrated in a narrow subset of concepts. These differences are not reflected in its aggregate score, which suggests uniformly strong performance. FinCDM exposes this imbalance by diagnosing both the breadth and depth of the model’s conceptual coverage, offering a more interpretable and actionable analysis of model capabilities.

Divergent Knowledge Specialization across HighPerforming Models. While many top-performing models achieve similar overall passing rates, our analysis further reveals substantial variation in the specific financial concepts each model masters. This divergence reflects differing strengths in subdomains of financial knowledge, suggesting that high aggregate accuracy does not necessarily imply uniform competence. For instance, Gemini-2.5-Pro-Exp and Doubao-1.5-Pro-256k both attain high overall passing rates of 0.84, respectively. However, a closer inspection at the concept level reveals that Gemini excels in general accounting categories such as contingency and Lease, which are typically aligned with international financial reporting standards. In contrast, Doubao demonstrates stronger performance in financial cost management areas, particularly Long-term Investment Decisions, Long-term Financing Decisions, and Working Capital Management, reflecting its expertise in financial management domains. This pattern is consistent across other model pairs as well. These findings emphasize the importance of fine-grained, concept-level evaluation beyond aggregate metrics, and support the need for modular assessments that can reveal the unique strengths and limitations of each model.

Influence of Linguistic Resources on Model Performance. Our evaluation highlights the substantial impact of linguistic resource availability on a model’s performance in domain-specific tasks. Models with limited Chinese language capabilities consistently underperform, both in aggregate accuracy and in concept-level mastery. For example, Falcon-7B, which lacks robust pretraining on Chinese corpora, achieves a passing rate of only 0.15 and demonstrates minimal competence across financial concepts. These results underscore the necessity of adequate linguistic grounding for effective domain adaptation. Without sufficient coverage of the target language during pretraining or finetuning, models struggle not only with general comprehension but also with acquiring specialized financial knowledge.

Evaluating Datasets on Knowledge Mastery. To investigate how dataset structure shapes model understanding of financial knowledge, we evaluate multiple LLMs also on FINEVAL-KQA, an existing benchmark re-annotated at the knowledge-point level. As illustrated in Figure 4, FINEVALKPA exhibits significant structural imbalance, with a majority of questions concentrated on a few specific financial concepts, notably Financial Instruments, Fundamentals of Financial Management, Strategic Choices, Civil Law, and Commercial Law, the latter appearing as many as 13 times. This skewed distribution strongly influences model performance, causing mastery rates to be heavily dependent on the frequency and representation of concepts. In contrast, CPAKQA maintains a more balanced representation across a

[Figure 3]

Figure 3: Model knowledge mastery heatmap on the CPA-KQA dataset.

broader range of financial concepts, enabling more robust and generalizable assessments of conceptual understanding. For instance, Gemini1.5 demonstrates mastery across the relatively highest number of concepts but achieves a low overall accuracy due to limited mastery of frequently tested concepts such as extitCommercial Law , highlighting how uneven concept distribution can disproportionately impact model performance.

#### Ablation Studies

We further compare the performance of three categories of CDMs: Neural CDMs, Graph-based CDMs, and our employed Matrix Co-Factorization (MCF)-based method. Table 3 presents the results in terms of accuracy (acc), area under the curve (AUC), and root mean square error (RMSE). To compute these metrics, we first use the inferred latent matrices E and U to reconstruct the predicted response matrix Xˆ = EU. We then compute the metrics by comparing these predictions against the observed response matrix X. MCF outperforms both baselines by a significant margin, achieving the highest accuracy and AUC, and the lowest RMSE. These results correspond to absolute gains of +0.177 in accuracy and +0.146 in AUC, and a reduction of −0.167 in RMSE compared to the strongest baseline, RCD.

Table 3: Model performance comparison across architectures.

Model Accuracy (acc) AUC RMSE Matrix Co-Factorization (MCF) 0.9379 0.9873 0.2314 Neural CDM (CNN-based) (Wang et al. 2022) 0.7140 0.7789 0.4394 GNN-based CDM (Gao et al. 2021) 0.7469 0.8329 0.4110

#### Case Study

To evaluate the reliability of our diagnostic framework, we conduct a focused case study on Claude 3.5, examining its mastery of two specific financial concepts: F3 and F5. According to our cognitive diagnosis analysis (Figure 3), Claude 3.5 failed to demonstrate mastery in these two concepts. Upon closer inspection, we confirmed that the model answered all six related questions incorrectly (three questions per concept). This provides concrete evidence that our diagnostic framework effectively identifies specific knowledge gaps in financial LLMs. To further verify the reliability of our framework, we enlisted five certified auditing experts to independently review the six incorrectly answered questions along with Claude 3.5’s responses. All five experts hold at least an undergraduate degree in finance and auditing. They were asked to annotate the primary financial concept tested by each question without being informed of the original concept labels. As shown in Table 4, four of the five experts consistently assigned the questions to con-

F13 F14

A1 A2

F12

- A3 A4

A5

A6

A7

A8

A9

A10

A11

A12

A13

- A14
- A15
- A16
- A17
- A18
- A19
- A20
- A21
- A22

A23

A24

A25

A26

A27

A28 A29

A30 B1

B2

- B3

F11

F10

F9

F8

F7

F6

F5

F4

F3

F2

5

[Figure 4]

F1

- E1
- E2
- E3
- E4

- 0

- 1

- 2

- D7
- D8

D6

D5

D4

D3

D2

D1

C6

C5

C4

C3

C2

C1

B8

B7 B6 B5 B4

- Figure 4: Structure of FinEval-KQA. The lighter-colored blocks represent concepts covered by FinEval-KQA, while the darker-colored blocks indicate concepts missing from it. The further a block extends from the center, the more frequently that concept was assessed.

cepts F3 and F5. The fifth expert introduced minor variations, such as labeling one question as F2 or F4. The overall inter-annotator agreement was 0.80, indicating strong consensus among the experts. These results demonstrate that our diagnosis framework aligns closely with expert judgment in identifying these conceptual deficiencies.

- Table 4: Expert annotations of knowledge points associated with six incorrectly answered questions.

##### Question Index 1 2 3 4 5 6

- Expert 1 F3 F3 F3 F5 F5 F5
- Expert 2 F3 F3 F3 F5 F5 F5
- Expert 3 F3 F3 F3 F5 F5 F5
- Expert 4 F2 F3 F4 F5 F5 F5
- Expert 5 F3 F3 F3 F5 F5 F5

### 5 Conclusion

This work presents FinCDM, the first cognitive diagnosis framework for evaluating financial LLMs beyond conventional aggregate metrics. Inspired by educational assessment, FinCDM diagnoses model proficiency at the knowledge-skill level, enabling interpretable insights into what financial concepts a model has mastered or misunderstood. To support this, we introduce CPA-KQA, a highquality, expert-annotated benchmark grounded in the CPA exam, covering 70 core financial concepts with balanced

knowledge representation. Through evaluations on 30 diverse LLMs, FinCDM reveals that models with similar overall scores often differ markedly in concept-level mastery, exposes coverage gaps in existing benchmarks, and uncovers latent specialization patterns across models. Future directions include multilingual extensions, incorporation of multimodal financial content, and leveraging diagnostic feedback to inform instruction tuning and benchmark design.

### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Almazrouei, E.; Alobeidli, H.; Alshamsi, A.; Cappelli, A.; Cojocaru, R.; Debbah, M.; Goffinet, E.;´ Hesslow, D.; Launay, J.; Malartic, Q.; et al. 2023. The falcon series of open language models. arXiv preprint arXiv:2311.16867.

Anthropic. 2024. Introducing the Claude 3 model family. Accessed: 2025-07-30. Baichuan Inc. 2023. Baichuan2: Open-Source Large Language Models. Accessed: 2025-07-30.

Bi, X.; Chen, D.; Chen, G.; Chen, S.; Dai, D.; Deng, C.; Ding, H.; Dong, K.; Du, Q.; Fu, Z.; et al. 2024. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954.

Braylan, A.; Alonso, O.; and Lease, M. 2022. Measuring annotator agreement generally across complex structured, multi-object, and free-text annotation tasks. In Proceedings of the ACM Web Conference 2022, 1720–1730.

ByteDance. 2024. Doubao 1.5 Pro. Accessed: 2025-07-30.

Cao, Y.; Li, H.; Yu, Y.; Javaji, S. R.; He, Y.; Huang, J.; Zhu, Z.; Xie, Q.; Liu, X.-y.; Subbalakshmi, K.; et al. 2025. FinAudio: A Benchmark for Audio Large Language Models in Financial Applications. arXiv preprint arXiv:2503.20990.

Chen, C.-C.; Huang, H.-H.; Takamura, H.; and Chen, H.-H.

- 2019. Overview of the ntcir-14 finnum task: Fine-grained numeral understanding in financial social media data. In Proceedings of the 14th NTCIR Conference on Evaluation of Information Access Technologies, 19–27. Chen, C.-C.; Huang, H.-H.; Takamura, H.; and Chen, H.-H.
- 2020. Overview of the NTCIR-15 FinNum-2 Task: Numeral attachment in financial tweets. Development, 850(194): 1– 044.

Chen, J.; Zhou, P.; Hua, Y.; Loh, Y.; Chen, K.; Li, Z.; Zhu, B.; and Liang, J. 2024. Fintextqa: A dataset for long-form financial question answering. arXiv preprint arXiv:2405.09980.

Chen, Z.; Chen, W.; Smiley, C.; Shah, S.; Borova, I.; Langdon, D.; Moussa, R.; Beane, M.; Huang, T.-H.; Routledge, B.; et al. 2021. Finqa: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122.

Chen, Z.; Li, S.; Smiley, C.; Ma, Z.; Shah, S.; and Wang, W. Y. 2022. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering.

- arXiv preprint arXiv:2210.03849.

Choi, C.; Kwon, J.; Ha, J.; Choi, H.; Kim, C.; Lee, Y.; Sohn,

- J.-y.; and Lopez-Lira, A. 2025. Finder: Financial dataset for question answering and evaluating retrieval-augmented generation. arXiv preprint arXiv:2504.15800.

De La Torre, J. 2009. DINA model and parameter estimation: A didactic. Journal of educational and behavioral statistics, 34(1): 115–130.

De La Torre, J. 2011. The generalized DINA model framework. Psychometrika, 76(2): 179–199.

Duan, X.; Xiao, B.; Tang, X.; and Cai, Z. G. 2024. HLB: Benchmarking LLMs’ Humanlikeness in Language Use. arXiv preprint arXiv:2409.15890.

El-Haj, M.; Litvak, M.; Pittaras, N.; Giannakopoulos, G.; et al. 2020. The financial narrative summarisation shared task (FNS 2020). In Proceedings of the 1st Joint Workshop on Financial Narrative Processing and MultiLing Financial Summarisation, 1–12.

Frederiksen, N.; Mislevy, R. J.; and Bejar, I. I. 1993. Test theory for a new generation of tests. L. Erlbaum Associates.

Gao, W.; Liu, Q.; Huang, Z.; Yin, Y.; Bi, H.; Wang, M.-C.; Ma, J.; Wang, S.; and Su, Y. 2021. RCD: Relation map driven cognitive diagnosis for intelligent education systems. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, 501–510.

Ghosh, S.; and Naskar, S. K. 2022. Fincat: Financial numeral claim analysis tool. In Companion Proceedings of the Web Conference 2022, 583–585.

GLM, T.; Zeng, A.; Xu, B.; Wang, B.; Zhang, C.; Yin, D.; Zhang, D.; Rojas, D.; Feng, G.; Zhao, H.; et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793.

Gu, Y.; and Xu, G. 2019. The sufficient and necessary condition for the identifiability and estimability of the DINA model. Psychometrika, 84(2): 468–483.

Guo, X.; Xia, H.; Liu, Z.; Cao, H.; Yang, Z.; Liu, Z.; Wang, S.; Niu, J.; Wang, C.; Wang, Y.; et al. 2025. FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 6258–6292.

Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2021. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations.

Islam, P.; Kannappan, A.; Kiela, D.; Qian, R.; Scherrer, N.; and Vidgen, B. 2023. Financebench: A new benchmark for financial question answering. arXiv preprint arXiv:2311.11944.

Kaur, S.; Smiley, C.; Gupta, A.; Sain, J.; Wang, D.; Siddagangappa, S.; Aguda, T.; and Shah, S. 2023. REFinD: Relation extraction financial dataset. In Proceedings of the 46th international ACM SIGIR conference on research and development in information retrieval, 3054–3063.

Klimaszewski, M.; Chen, P.; Guillou, L.; Papaioannou, I.; Haddow, B.; and Birch, A. 2025. AVENIBENCH: Accessible and Versatile Evaluation of Finance Intelligence. In Proceedings of the Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), 111–117.

Krumdick, M.; Koncel-Kedziorski, R.; Lai, V. D.; Reddy, V.; Lovering, C.; and Tanner, C. 2024. Bizbench: A quantitative reasoning benchmark for business and finance. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 8309–8332.

Kuh, G. D.; Kinzie, J.; Buckley, J. A.; Bridges, B. K.; and Hayek, J. C. 2011. Piecing together the student success puzzle: Research, propositions, and recommendations: ASHE higher education report, volume 116. John Wiley & Sons.

Lai, V. D.; Krumdick, M.; Lovering, C.; Reddy, V.; Schmidt, C.; and Tanner, C. 2024. Sec-qa: A systematic evaluation corpus for financial qa. arXiv preprint arXiv:2406.14394.

Lan, K.; Zhu, J.; Li, J.; Cheng, D.; Chen, G.; and Jiang, C. 2025. FinLMM-R1: Enhancing Financial Reasoning in LMM through Scalable Data and Reward Design. arXiv preprint arXiv:2506.13066.

Leighton, J.; and Gierl, M. 2007. Cognitive diagnostic assessment for education: Theory and applications. Cambridge University Press.

Li, H.; Cao, Y.; Yu, Y.; Javaji, S. R.; Deng, Z.; He, Y.; Jiang, Y.; Zhu, Z.; Subbalakshmi, K.; Xiong, G.; et al. 2024a. Investorbench: A benchmark for financial decision-making tasks with llm-based agent. arXiv preprint arXiv:2412.18174.

Li, J.; Bian, Y.; Wang, G.; Lei, Y.; Cheng, D.; Ding, Z.; and Jiang, C. 2023. Cfgpt: Chinese financial assistant with large language model. arXiv preprint arXiv:2309.10654.

Li, X.; Li, Z.; Shi, C.; Xu, Y.; Du, Q.; Tan, M.; Huang, J.; and Lin, W. 2024b. Alphafin: Benchmarking financial analysis with retrieval-augmented stock-chain framework. arXiv preprint arXiv:2403.12582.

Liang, P.; Bommasani, R.; Lee, T.; Tsipras, D.; Soylu, D.; Yasunaga, M.; Zhang, Y.; Narayanan, D.; Wu, Y.; Kumar, A.; et al. 2022. Holistic evaluation of language models.

- arXiv preprint arXiv:2211.09110.

Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; et al. 2024a. Deepseekv3 technical report. arXiv preprint arXiv:2412.19437.

Liu, S.; Zhao, S.; Jia, C.; Zhuang, X.; Long, Z.; Zhou, J.; Zhou, A.; Lan, M.; Wu, Q.; and Yang, C. 2024b. Findabench: Benchmarking financial data analysis ability of large language models. arXiv preprint arXiv:2401.02982.

Loukas, L.; Fergadiotis, M.; Chalkidis, I.; Spyropoulou, E.; Malakasiotis, P.; Androutsopoulos, I.; and Paliouras, G. 2022. FiNER: Financial numeric entity recognition for XBRL tagging. arXiv preprint arXiv:2203.06482.

Lu, G.; Guo, X.; Zhang, R.; Zhu, W.; and Liu, J. 2025. BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs. arXiv preprint

- arXiv:2505.19457.

Luo, J.; Kou, Z.; Yang, L.; Luo, X.; Huang, J.; Xiao, Z.; Peng, J.; Liu, C.; Ji, J.; Liu, X.; et al. 2025. FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation. arXiv preprint arXiv:2505.24714.

Mateega, S.; Georgescu, C.; and Tang, D. 2025. Financeqa: A benchmark for evaluating financial analysis capabilities of large language models. arXiv preprint arXiv:2501.18062.

Mukherjee, R.; Bohra, A.; Banerjee, A.; Sharma, S.; Hegde, M.; Shaikh, A.; Shrivastava, S.; Dasgupta, K.; Ganguly, N.; Ghosh, S.; et al. 2022. Ectsum: A new benchmark dataset for bullet point summarization of long earnings call transcripts. arXiv preprint arXiv:2210.12467.

Nie, Y.; Kong, Y.; Dong, X.; Mulvey, J. M.; Poor, H. V.; Wen, Q.; and Zohren, S. 2024a. A survey of large language models for financial applications: Progress, prospects and challenges. arXiv preprint arXiv:2406.11903.

Nie, Y.; Yan, B.; Guo, T.; Liu, H.; Wang, H.; He, W.; Zheng, B.; Wang, W.; Li, Q.; Sun, W.; et al. 2024b. Cfinbench: A comprehensive chinese financial benchmark for large language models. arXiv preprint arXiv:2407.02301.

Peng, X.; Papadopoulos, T.; Soufleri, E.; Giannouris, P.; Xiang, R.; Wang, Y.; Qian, L.; Huang, J.; Xie, Q.; and Ananiadou, S. 2025a. Plutus: Benchmarking large language models in low-resource greek finance. arXiv preprint arXiv:2502.18772.

Peng, X.; Qian, L.; Wang, Y.; Xiang, R.; He, Y.; Ren, Y.; Jiang, M.; Zhao, J.; He, H.; Han, Y.; et al. 2025b. MultiFinBen: A Multilingual, Multimodal, and Difficulty-Aware Benchmark for Financial LLM Evaluation. arXiv preprint

- arXiv:2506.14028.

Qian, L.; Zhou, W.; Wang, Y.; Peng, X.; Huang, J.; and Xie, Q. 2025. Fino1: On the transferability of reasoning enhanced llms to finance. arXiv e-prints, arXiv–2502.

Reddy, V.; Koncel-Kedziorski, R.; Lai, V. D.; Krumdick, M.; Lovering, C.; and Tanner, C. 2024. Docfinqa: A long-context financial reasoning dataset. arXiv preprint arXiv:2401.06915.

Shah, A.; Gullapalli, A.; Vithani, R.; Galarnyk, M.; and Chava, S. 2023. FiNER-ORD: Financial Named Entity Recognition Open Research Dataset. arXiv preprint arXiv:2302.11157.

Sharma, S.; Nayak, T.; Bose, A.; Meena, A. K.; Dasgupta,

- K.; Ganguly, N.; and Goyal, P. 2022. FinRED: A dataset for relation extraction in financial domain. In Companion Proceedings of the Web Conference 2022, 595–597.

Tang, Y.; and Yang, Y. 2025. Finmteb: Finance massive text embedding benchmark. arXiv preprint arXiv:2502.10990.

Team, G.; Georgiev, P.; Lei, V. I.; Burnell, R.; Bai, L.; Gulati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.; et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Team, Q. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Templin, J. L.; and Henson, R. A. 2006. Measurement of psychological disorders using cognitive diagnosis models. Psychological methods, 11(3): 287.

Tencent Hunyuan Team. 2025. Hunyuan-A13B: An 80B-parameter MoE model with only 13B active parameters, 256K context support, and hybrid reasoning modes. Technical report, Tencent Hunyuan Model Team. Accessed: 2025-07-30.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozi`ere, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Wang, F.; Liu, Q.; Chen, E.; Huang, Z.; Yin, Y.; Wang, S.; and Su, Y. 2022. NeuralCD: a general framework for cognitive diagnosis. IEEE Transactions on Knowledge and Data Engineering, 35(8): 8312–8327.

Wang, Y.; Ren, Y.; Qian, L.; Peng, X.; Wang, K.; Han, Y.; Feng, D.; Liu, X.-Y.; Huang, J.; and Xie, Q. 2025. FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information. arXiv preprint

- arXiv:2505.20650.

Wu, X.; Liu, J.; Su, H.; Lin, Z.; Qi, Y.; Xu, C.; Su, J.; Zhong, J.; Wang, F.; Wang, S.; et al. 2024. Golden Touchstone: A Comprehensive Bilingual Benchmark for Evaluating Financial Large Language Models. arXiv preprint arXiv:2411.06272.

xAI. 2024. Introducing Grok-3. Accessed: 2025-07-30.

Xie, Q.; Han, W.; Chen, Z.; Xiang, R.; Zhang, X.; He, Y.; Xiao, M.; Li, D.; Dai, Y.; Feng, D.; et al. 2024. Finben: A holistic financial benchmark for large language models. Advances in Neural Information Processing Systems, 37: 95716–95743.

Xie, Q.; Han, W.; Zhang, X.; Lai, Y.; Peng, M.; Lopez-Lira,

- A.; and Huang, J. 2023. Pixiu: A comprehensive benchmark, instruction dataset and large language model for finance. Advances in Neural Information Processing Systems, 36: 33469–33484.

Xie, Z.; Sahnan, D.; Banerjee, D.; Georgiev, G.; Thareja, R.; Madmoun, H.; Su, J.; Singh, A.; Wang, Y.; Xing, R.; et al. 2025. FinChain: A Symbolic Benchmark for Verifiable Chain-of-Thought Financial Reasoning. arXiv preprint arXiv:2506.02515.

Xu, Z.; Zhou, P.; Shi, X.; Wu, J.; Jiang, Y.; Chong, D.; Ke,

- B.; and Yang, J. 2024. Fintruthqa: A benchmark dataset for evaluating the quality of financial information disclosure. arXiv preprint arXiv:2406.12009.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yang, Y.; Zhang, Y.; Hu, Y.; Guo, Y.; Gan, R.; He, Y.; Lei, M.; Zhang, X.; Wang, H.; Xie, Q.; et al. 2024. Ucfe: A user-centric financial expertise benchmark for large language models. arXiv preprint arXiv:2410.14059.

Yu, S.; Zeng, Y.; Pan, Y.; and Yang, F. 2023. Snmcf: a scalable non-negative matrix co-factorization for student cognitive modeling. IEEE Transactions on Knowledge and Data Engineering, 36(7): 3486–3500.

Yuan, Z.; Wang, K.; Zhu, S.; Yuan, Y.; Zhou, J.; Zhu, Y.; and Wei, W. 2024. Finllms: A framework for financial reasoning dataset generation with large language models. IEEE Transactions on Big Data.

Zhang, L.; Cai, W.; Liu, Z.; Yang, Z.; Dai, W.; Liao, Y.; Qin, Q.; Li, Y.; Liu, X.; Liu, Z.; et al. 2023. Fineval: A chinese financial domain knowledge evaluation benchmark for large language models. arXiv preprint arXiv:2308.09975.

Zhao, Y.; Li, Y.; Li, C.; and Zhang, R. 2022. MultiHiertt: Numerical reasoning over multi hierarchical tabular and textual data. arXiv preprint arXiv:2206.01347.

Zhipu AI. 2024. ChatGLM3: Open Bilingual Chat Models. Accessed: 2025-07-30.

Zhu, F.; Lei, W.; Huang, Y.; Wang, C.; Zhang, S.; Lv, J.; Feng, F.; and Chua, T.-S. 2021. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. arXiv preprint arXiv:2105.07624.

Zhu, J.; Li, J.; Wen, Y.; and Guo, L. 2024. Benchmarking Large Language Models on CFLUE–A Chinese Financial Language Understanding Evaluation Dataset. arXiv preprint arXiv:2405.10542.

Zhu, J.; Li, J.; Wen, Y.; Li, X.; Guo, L.; and Chen, F. 2025. M 3 FinMeeting: A Multilingual, Multi-Sector, and MultiTask Financial Meeting Understanding Evaluation Dataset. arXiv preprint arXiv:2506.02510.

### A Guidelines for Creating and Annotating CPA-KQA Items

#### Principles and Requirements for Item Creation

Consistency with CPA Exam Style Each assessment item must align closely with the CPA exam in terms of wording, question format, and option setting, reflecting the CPA exam’s rigorous standards for professional knowledge.

Originality and Accuracy All items must be original creations. Direct copying or quoting from existing CPA exam items or publicly available materials is strictly prohibited. Ensure accuracy, clarity, and absence of ambiguity in each item.

Specificity and Clarity Each item should clearly assess only one specified financial knowledge point. Avoid incorporating content irrelevant to the specified knowledge point to ensure accurate and effective evaluation.

Avoidance of Politically Sensitive Content Strictly avoid incorporating politically sensitive topics, events, or individuals in any assessment items. Content that may be viewed as politically controversial or sensitive is explicitly prohibited.

Item Formats Create three items per knowledge point, in Single-choice questions (at least one per knowledge point) Provide four clearly distinguishable options (A, B, C, D), with exactly one correct answer.

#### Required Format for Item Submission

Submit each item strictly following this format:

- • Knowledge Point: (Name of the Knowledge Point)
- • Question Stem: Clearly and completely stated Options (only for single-choice items): A. ... B. ... C. ... D. ...
- • Correct Answer: (Option letter or detailed solution)

#### Annotation and Quality Review Procedures

Experts will create and annotate items as follows:

- • Each expert independently creates items for their assigned knowledge points.
- • Upon completion, two other experts annotate each item on two dimensions: Quality Annotation: Mark as either ”Usable” or ”Unusable”. Criteria for ”Unusable” include unclear meaning, inaccurate wording, calculation errors, ambiguous answers, or controversial content. Knowledge Point Consistency Annotation: Verify if the item accurately corresponds to the specified knowledge point.
- • Items marked ”Unusable” or ”Inconsistent” by both reviewers are deleted directly.
- • Items marked by only one reviewer require one-time revision and must pass re-review.
- • Deleted items must be replaced until each knowledge point has three approved items.

### B Optimization Process for Non-negative Matrix Co-factorization

We estimate these latent matrices by optimizing the following joint objective:

∥W ◦ (X − EU)∥2F + α∥Q − EV ∥2F + λE∥E∥2F + λU∥U∥2F + λV ∥V ∥2F,

min

E,U,V ≥0

with multiplicative update rules for convergence to a local optimum:

(W ◦ X)U⊤ + αQV ⊤ (W ◦ EU)U⊤ + αEV V ⊤ + λEE

E ← E ◦

,

- U ← U ◦

E⊤(W ◦ X) E⊤(W ◦ EU) + λUU

,

- V ← V ◦

αE⊤Q αE⊤EV + λV V

,

where the symbol ◦ denotes element-wise multiplication, and division operations are also element-wise.

### C LLMs details

Category Model Creator Parameters Access Version Date

GLM-4-32B-0414 Zhipu AI undisclosed API 2025.4 GLM-4-9B-0414 Zhipu AI undisclosed API 2025.4 Hunyuan Tencent undisclosed API 2023.9 DeepSeek-Chat DeepSeek AI 236B (MoE) API 2024.5 DeepSeek-V3-0324 DeepSeek AI 671B (MoE) API 2025.3 DBRX-Instruct Databricks 132B (MoE) API 2024.3

- Qwen2-72B-Instruct Alibaba Cloud 72B API 2024.6 Qwen2.5-7B-Instruct Alibaba Cloud 7B API 2024.9
- Qwen3-0.6B Alibaba Cloud 0.6B API 2025.4 Qwen3-235B-A22b Alibaba Cloud 235B(MOE) API 2025.4 LLaMA2-70B Meta 70B API 2023.7 LLaMA3.1 405B Meta 405B API 2024.7 Baichuan2-13B-Chat Baichuan Inc. 13B Weights 2023.12 Falcon-7B TII 7B Weights 2023.5 ChatGLM3-6B Zhipu AI 6B Weights 2023.1

Open-Source

GPT-4o OpenAI undisclosed API 2024.5 GPT-4o-mini OpenAI undisclosed API 2024.7 GPT-4 OpenAI undisclosed API 2023.3 Gemini 1.5 Pro Google undisclosed API 2024.5 Gemini 1.5 Flash Google undisclosed API 2024.5 Gemini2.5 Pro Experimental 03-25

Google undisclosed API 2025.3

Closed-Source

Claude 3.5 Sonnet Anthropic undisclosed API 2024.1 Claude 3.7 Sonnet Anthropic undisclosed API 2025.2 GLM-4 Zhipu AI undisclosed API 2024.1 Grok 3 xAI undisclosed API 2025.2 Doubao-1.5-Pro-256k ByteDance undisclosed API 2025.1 Doubao-1.5-Pro-32k ByteDance undisclosed API 2025.1 Qwen-Max Alibaba Cloud undisclosed API 2025.1

Finma-7b-Full Finma 7B Weights 2023.9 CFGPT2-7B TongjiFinLab 7B Weights 2024.8

Financial

- Table 5: Models evaluated in this paper. The ”Access” column shows whether we have full access to the model weights or we can only access through API. The “Version Date” column shows the release date of the corresponding version of the model we evaluated.

[Figure 5]

### D prompt

- Figure 5: Prompt for multiple-choice questions in Intermediate Financial Accounting. For better readability, the English translation is displayed to the right of the corresponding Chinese text.

