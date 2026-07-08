# arXiv:2601.05403v2[cs.CL]18Apr2026

## Same Claim, Different Judgment: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection

Zhiwei Liu1, Yupen Cao2, Yuechen Jiang1, Mohsinul Kabir1, Polydoros Giannouris1, Chen Xu3, Ziyang Xu3, Tianlei Zhu4, Md. Tariquzzaman5, Triantafillos Papadopoulos6,7, Yan Wang3, Lingfei Qian3, Xueqing Peng3, Zhuohan Xie8, Ye Yuan9,10, Saeed Almheiri8, Abdulrazzaq Alnajjar11, Mingbin Chen12, Harry Stuart8, Paul Thompson1, Prayag Tiwari13, Alejandro Lopez-Lira14, Xue Liu8,9, Jimin Huang1,3*, Sophia Ananiadou1,7,15 1The University of Manchester, 2Stevens Institute of Technology, 3The Fin AI, 4Columbia University, 5Islamic University of Technology, 6Athens University of Economics and Business, 7Archimedes, Athena Research Center, 8MBZUAI, 9McGill University, 10Mila - Quebec AI Institute, 11Dubai Police, 12University of Melbourne, 13Halmstad University, 14University of Florida, 15ELLIS Manchester {zhiwei.liu, sophia.ananiadou}@manchester.ac.uk, jimin.huang@postgrad.manchester.ac.uk

### Abstract

Large language models (LLMs) have been widely applied across various domains of finance. Since their training data are largely derived from human-authored corpora, LLMs may inherit a range of human biases. Behavioral biases can lead to instability and uncertainty in decision-making, particularly when processing financial information. However, existing research on LLM bias has mainly focused on direct questioning or simplified, generalpurpose settings, with limited consideration of the complex real-world financial environments and high-risk, context-sensitive, multilingual financial misinformation detection tasks (MFMD). In this work, we propose MFMDScen, a comprehensive benchmark for evaluating behavioral biases of LLMs in MFMD across diverse economic scenarios. In collaboration with financial experts, we construct three types of complex financial scenarios: (i) roleand personality-based, (ii) role- and regionbased, and (iii) role-based scenarios incorporating ethnicity and religious beliefs. We further develop a multilingual financial misinformation dataset covering English, Chinese, Greek, and Bengali. By integrating these scenarios with misinformation claims, MFMD-Scen enables a systematic evaluation of 22 mainstream LLMs. Our findings reveal that pronounced behavioral biases persist across both commercial and open-source models. This project is available at https://github.com/lzw108/FMD.

### 1 Introduction

Despite their growing deployment in financial analysis, forecasting, and decision support (Xie et al., 2024; Wu et al., 2023; Xie et al., 2023), LLMs remain unreliable for high-stakes, multilingual finan-

*Corresponding Author

|Misinformation detection in different scenarios|
|---|
|[Figure 1]<br><br>[Figure 2]<br><br>You are a retail investor and just made a profit yesterday<br><br>People on Facebook promise to send $750 to your Cash App account for free.<br><br>LLMs (Base)<br><br>False ? ?<br><br>Role + OverConfidence (Personality)<br><br>You are a hedge fund strategist working in the U.S. financial district......<br><br>You are a company owner of Latino background and Christianity belief<br><br>[Figure 3]<br><br>[Figure 4]<br><br>Role + USA (Region)<br><br>Role + Latino&Christianity (Identity)<br><br>?|

Figure 1: Example of financial misinformation detection by LLMs across different scenarios.

cial misinformation detection, where behavioral biases can lead to systematic and context-dependent errors across stakeholder conditions and market scenarios (Echterhoff et al., 2024; Bini et al., 2025) (See Figure 1). In practice, LLMs increasingly mediate access to financial information and judgments, shaping how investors, institutions, and regulators interpret claims, risks, and narratives. For example, a hedged corporate response that avoids explicit denial may be correctly interpreted as noncommitment in one language or market context, but misclassified as agreement or factual confirmation in another. Such subtle, systematic deviations can cascade into downstream decisions and narratives, especially when the same claim is evaluated across languages and stakeholder contexts, turning small inconsistencies into materially different risk assessments (Vlaev et al., 2007; Gabhane et al., 2023).

However, existing financial misinformation benchmarks are largely built around claim veri-

fication in a fixed evaluation setting, which provides limited leverage for studying multilingual or scenario-dependent judgment variability. Benchmarks such as FinFact (Rangapur et al., 2025) and FinDVer (Zhao et al., 2024) formulate misinformation detection as a classification task, where a model judges a claim and outputs a single label, without systematically varying the language of the claim or the assessment context. Meanwhile, work on cognitive and behavioral biases in LLMs often relies on direct elicitation or simplified decision tasks (Ranjan et al., 2024; Tao et al., 2024; Echterhoff et al., 2024; Taubenfeld et al., 2024; Kong et al., 2024; Bini et al., 2025), and does not capture how bias manifests in financial claim verification. Related work and dataset comparisons can be found in Table 2 and Appendix A.

To address the gap, we introduce MFMD-Scen, an expert-designed benchmark that enables controlled evaluation of financial misinformation detection across multilingual and scenario-conditioned settings within a verification paradigm. We construct a multilingual, scenario-aligned financial misinformation dataset, where real-world claims are instantiated across languages for controlled comparison. Starting from Snopes-based claims via FinFact, we recover complete claim statements and select globally relevant items for cross-lingual instantiation, with translations into Chinese, Greek, and Bengali that are validated through nativespeaker review and targeted human revision, with high inter-annotator agreement.

Building on this dataset, MFMD-Scen formulates financial misinformation detection as binary claim verification under scenario conditioning, and evaluates models both with and without scenario context. To reflect how financial judgments vary in practice, we instantiate scenarios along three complementary axes, i.e., stakeholder behavior (MFMD-persona), market environment (MFMD-region), and background-dependent interpretation (MFMD-identity), so that changes in predictions can be attributed to controlled shifts in context rather than changes in the underlying claim. We finally report standard misinformation detection performance (e.g., F1) and quantify scenarioconditioned effects as the performance difference between scenario-aware and scenario-agnostic evaluation.

Our evaluation of 22 LLMs on MFMD-Scen shows that injecting a realistic financial context can induce measurable behavioral bias, yielding

systematic changes in misinformation judgments even when the underlying claim is held fixed. Scenario information does not act as a random perturbation, yet it consistently shifts the effective decision boundary relative to the scenarioagnostic baseline, indicating that contextual priors can override claim-level signals. The strongest biases arise when scenarios carry high-salience credibility cues, most notably in MFMD-persona for retail-investor and herding descriptions, and in MFMD-region for emerging Asian market contexts, where models tend to fall back to risk-averse, skepticism-heavy defaults rather than preserving content-based consistency. MFMD-identity further reveals interaction-driven bias. Role conditioning modulates how background cues are used, and the same cue can push predictions in opposite directions under different roles, exposing non-additive dependencies that static evaluation cannot capture. These effects are amplified in low-resource languages, consistent with weaker linguistic calibration and heavier reliance on contextual shortcuts, while explicit reasoning benefits are unreliable at smaller scales and become clearer primarily for large models.

Our main contributions are as follows:

- • We introduce MFMD-Scen, a comprehensive benchmark designed with financial domain experts to evaluate LLMs’ behavioral biases in financial misinformation detection across diverse scenarios, i.e., roles, personality, regional, and socio-cultural contexts.
- • We construct a multilingual financial misinformation dataset, covering English, Chinese, Greek, and Bengali.
- • We evaluate 22 LLMs on MFMD-Scen, revealing that mainstream models exhibit significant behavioral biases, particularly in contexts involving retail investors, herding personalities, or emerging Asian financial markets.

### 2 MFMD-Scen Benchmark

MFMD-Scen provides a comprehensive benchmark for evaluating behavioral biases of LLMs in financial misinformation across different financial scenarios. In the following subsections, we outline the MFMD-Scen content shown in Figure 2 and provide a detailed introduction to the scenario subtask definitions and the data construction process. After obtaining the three kinds of scenarios and the

[Figure 5]

misinformation claims, we combine the scenarios and claims to complete MFMD-Scen benchmark.

[Figure 6]

[Figure 7]

Retail Investor Professional Institute

Company Owner

MFMD-persona

[Figure 8]

#### 2.1 Task Formulation

Overconfidence

Explicit Implicit

Loss Aversion

[Figure 9]

[Figure 10]

[Figure 11]

We formally define the MFMD-Scen task as follows: given a financial scenario s ∈ S = {Spersona,Sregion,Sidentity}, where Spersona represents a financial scenario conditioned on role and personality, Sregion represents a financial scenario conditioned on role and region, and Sidentity represents a financial scenario conditioned on role, ethnicity, and belief. Given a financial information claim c, the task is to determine the truthfulness label (lscen,lbase,lgold ∈ L = {True,False}) of the claim c in the scenario s. We define the bias as follows:

Anchoring Herding

[Figure 12]

Confirmation Bias

Mode Roles

Personality

MFMD-region

[Figure 13]

Europe USA

[Figure 14]

[Figure 15]

Asia-Pacific China Mainland

[Figure 16]

Australia UAE Regions

Roles

MFMD-identity

American European Jewish Chinese Indian Arab African Latino / Hispanic

Christianity Islam Buddhism Judaism

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

PLLM(l | s,c) (1)

Roles Ethnicity Faith

lscen = arg max

l∈L

PLLM(l | c) (2)

lbase = arg max

Financial Misinformation Claims

l∈L

Figure 2: Overview of MFMD-Scen Benchmark. The upper part shows the three subtasks of financial scenarios. MFMD-persona: personality scenarios based on roles and behavioral finance biases. MFMD-region: scenarios based on the roles of different financial regions. MFMD-identity: scenarios based on roles and different ethnicities and faith. The second is the financial misinformation dataset (Sec 2.2). Scenarios and misinformation claims are combined to obtain MFMD-Scen.

Biasscen = |F1(lscen,lgold) − F1(lbase,lgold)| (3)

lscen denotes the LLMs’ predictions under specific financial scenarios, lbase denotes their predictions without financial scenario information, and lgold represents the ground-truth labels. The behavioral bias is quantified as the difference in F1 scores between these two cases, reflecting how scenario context changes verification performance for the same claim.

investors, professional or institutional investors, and firm owners or managers (Barber and Odean, 2001; Malmendier and Tate, 2005). These roles capture heterogeneity in expertise, information access, market exposure, and organizational incentives.

Based on the scenario types (Spersona, Sregion, and Sidentity), we define three corresponding subtasks. Section 2.1.1 introduces persona-based scenarios that combine three roles with five behavioral finance biases (MFMD-persona). Section 2.1.2 presents region-based scenarios constructed from different financial markets and roles (MFMD-region). Section 2.1.3 describes identitybased scenarios involving two individual roles in conjunction with ethnicity and faith (MFMDidentity). The upper part of Figure 2 illustrates the overall design of these three types of scenarios.

We incorporate five foundational behavioral finance biases widely documented in cognitive psychology and financial economics: Overconfidence (Barber and Odean, 2001), Loss Aversion (Kahneman and Tversky, 1979), Herding (Sharma and Bikhchandani, 2000), Anchoring (Gilovich et al., 2002), and Confirmation Bias (Park et al., 2010).

- 2.1.1 Task 1: Detection in Different Personality Scenarios (MFMD-persona)

This task aims to evaluate the behavioral bias of LLMs in different personality profiles. To construct a comprehensive set of behavioral profiles, we follow prior literature that distinguishes between different types of financial decision-makers, i.e., retail

- • Overconfidence: Overestimating the accuracy of one’s predictions and judgments, often resulting in excessive trading or underestimating risks.
- • Loss Aversion: People are more sensitive to losses than to equivalent gains; losses feel

- about twice as painful as gains feel good.
- • Herding Behavior: Tendency to follow the crowd instead of making independent decisions, often leading to bubbles or panic.
- • Anchoring Effect: Relying too heavily on initial information (“anchor”), even if it is irrelevant, which affects subsequent judgments.
- • Confirmation Bias: Tendency to seek and believe information that supports existing beliefs while ignoring contradictory evidence.

For each role–persona pairing, we define two variants. Explicit: the bias is overtly stated and directly influences the agent’s decision-making. Implicit: the bias is conveyed subtly through narrative cues or behavioral tendencies. A comprehensive set of scenarios is presented in Table 3 in Appendix B.1.

- 2.1.2 Task 2: Detection in Different Financial Markets (MFMD-region)

This task aims to evaluate the bias of LLMs in different financial markets. To account for regional financial culture and institutional variation, scenarios are contextualized across six major economic regions: Europe, North America, Asia Pacific, China Mainland, Australia, and the United Arab Emirates (UAE). These regional distinctions follow established international classifications employed by the IMF and the World Bank (International Monetary Fund, 2025a,b; World Bank Group, 2022).

The scenarios capture differences in regulatory frameworks, risk cultures, macroeconomic environments, dominant asset classes, and market maturity. Detailed descriptions of the region-specific scenarios are provided in Appendix B.2.

- 2.1.3 Task 3: Detection in Different Identities (MFMD-identity)

This task aims to evaluate the bias of LLMs in different ethnicities and faiths. We incorporate cultural variation by designing scenarios informed by major ethnic and faith groups with substantial representation in global financial systems. Demographic and religious distributions are based on authoritative public datasets from the Pew Research Center, the U.S. Religion Census, and the Census of India, as well as region-specific Wikipedia summaries (Association of Statisticians of American Religious Bodies (ASARB), 2023; Pew Research Center, 2021, 2023; Census of India 2011,

2021; Wikipedia contributors). Representative ethnicity–faith/belief scenarios are provided in Appendix B.3.

2.2 Multilingual Financial Misinformation Dataset Construction

In this section, we introduce the construction of the multilingual financial misinformation dataset. To ensure a fairer comparison across languages and across different financial scenarios, we construct the dataset by translating globally relevant news items.

- 1) Financial news collection: We begin with the FinFact (Rangapur et al., 2025) dataset, a benchmark for detecting misinformation in financial claims, and focus on its Snopes subset. Since the claims provided by the authors are derived from Snopes article titles1, which are often questions or incomplete statements, we crawled the original claims from the corresponding URLs for use in this study. We further collected Snopes news from 2024 to September 2025 using financial keywords. Two annotators with a background in finance (see Appendix C.1) then screened the complete set of claims, resulting in a final dataset of 502 items in the financial domain2.
- 2) Global news collection (GlobalEn): Next, two financial experts categorized the filtered financial claims into regional news, which attracts attention primarily within specific countries or regions, and global news that has potential worldwide relevance. This process yielded 144 global news items, of which 121 were labeled as false and 23 as true.
- 3) Translation collection: Subsequently, the claims were translated into Chinese (GlobalCh), Greek (GlobalGr), and Bengali(GlobalBe) using GPT-4.1. Two native speakers of each target language then evaluated the translated outputs. For instances where the translation quality was insufficient, one human translator revised the text manually and another reviewed the revisions to ensure both accuracy and fluency.

Table 1 presents the inter-annotator agreement scores for each stage. Detailed annotation guidelines for each step are provided in Appendix C, and data statistics are reported in Table 4.

- 1https://www.snopes.com/
- 2Financial misinformation in this study refers to false or misleading information involving monetary value, economic assets, financial transactions, or promised financial gain that may induce financially consequential behavior or expose individuals to economic loss.

Kappa Acc F1 Financial vs Non-Financial 0.992 0.996 0.996 Regional vs Global 0.965 0.984 0.983 Chinese Translation 1 1 1 Greek Translation 0.723 0.973 0.861 Bengali Translation 0.98 0.995 0.99

Table 1: Agreement score for each part

We also collected the financial misinformation datasets in the original languages for evaluation. The description and evaluation on these original datasets can be found in Appendix D. The Chinese translation contains four very obvious errors generated by the LLMs, resulting in high consistency. An example can be found in Figure 7.

### 3 Evaluations

After constructing MFMD-Scen, we performed the evaluations over existing LLMs, using proposed evaluation metrics, and compared them with human-level performance.

#### 3.1 Models

We evaluated a broad spectrum of large language models, including both reasoning-oriented and standard no-think/chat systems, spanning open-source and closed-source offerings. The reasoning models in our study include GPT-5mini (OpenAI, 2025), DeepSeek-V3.2-Reasoner (Liu et al., 2025a), Claude-Sonnet-4.5 (Anthropic, 2025), Gemini-2.5-Flash (Comanici et al., 2025), and the Qwen3 reasoning series (8B-R, 14B-R, 32B-R) (Yang et al., 2025a). We also evaluated a wide range of no-think LLMs, i.e., GPT-

##### 4.1 (OpenAI, 2025), Claude-3.5-Haiku, Gemini2.0-Flash (Comanici et al., 2025), DeepSeekV3.2-Chat (Liu et al., 2025a), the Qwen3 nothink series (8B, 14B, 32B) (Yang et al., 2025a), Qwen2.5-72B-Instruct (Qwen72B) (Yang et al., 2025b), Llama-3.3-70B-Instruct3 (Dubey et al., 2024), and multiple Mistral and Mixtral models (Jiang et al., 2023, 2024), i.e., Mistral-7B-Instructv0.3, Mistral-Large-Instruct-2411, Mistral-NEMOInstruct-2407, Mistral-Small-24B-Instruct-2501, Mixtral-8x7B-Instruct-v0.1, and Mixtral-8x22BInstruct-v0.1. The templates for evaluating LLMs can be found in Appendix E. The open-source LLMs are evaluated on 4 NVIDIA Tesla A100 GPUs with 80 GB of memory. The temperature

3Due to safety restrictions, LLaMA 3.1-8B was unable to produce responses in most cases, and therefore its results are not reported in this paper.

is set to 0, while all other settings use the default configuration.

#### 3.2 Evaluation metrics

We report the overall accuracy and macro-F1 scores of the models on datasets in different languages (Table 5). For other scenario-specific settings, we primarily report the macro-F1 scores for the true and false categories, as well as the arithmetic mean (AM) and mean absolute value (MAV) across the 22 models. AM represents the direction of the bias, while MAV represents the magnitude of the deviation4.

#### 3.3 Human-level Performance Measurement

To conduct a rough but effective assessment of human-level performance on MFMD-Scen, we selected financial scenarios from five different regions within the MFMD-region part and recruited volunteers to evaluate 144 English claims. Among the volunteers, 11 participants were from China Mainland. Two participants were recruited from each of Europe, Asia Pacific, Australia, and the UAE. Volunteers were asked to judge the truthfulness of each claim solely based on their past experience and knowledge. The details and performance can be found in Appendix F.

### 4 Evaluation Results 4.1 Main findings

Figures 3 to 5 present the results of different LLMs on the MFMD-Scen benchmark. The results show that current mainstream models are relatively mature in judging whether a statement is false, with more conservative and tightly clustered decision boundaries. However, deficiencies remain in determining whether a statement is true. LLMs also exhibit clear biases in their judgments of financial misinformation across different scenarios.

Specifically, in the MFMD-persona setting, when the retail investor role or herding scenarios are introduced, models tend to exhibit pronounced negative bias. Comparing across models, largerscale models are relatively more stable when different scenarios are introduced and display smaller bias. Across languages, low-resource languages show comparatively larger bias, and the injection

4Since this study focuses on how scenario conditioning affects binary judgments, macro-F1 is more appropriate for capturing scenario-induced variations in classification behavior. Absolute F1 can directly capture how each model’s decisions change under scenario injection.

of different scenarios can introduce distinct cultural or linguistic characteristics. In the MFMD-region setting, model bias is clearly influenced by region, with emerging Asian markets more likely to induce negative bias, whereas typical financial scenarios in Europe and the USA exhibit relatively smaller bias. In the MFMD-identity setting, model bias is also influenced by ethnicity or faith, and changes significantly with role information: the bias for the same group may reverse across different roles, with American groups generally exhibiting positive bias and Chinese groups generally exhibiting negative bias, highlighting the systematic and interactive nature of these biases.

#### 4.2 Results on MFMD-persona

- Figure 3 shows the AM and MAV of F1 in Global multilingual datasets under MFMD-persona scenario across 22 LLMs. The performance details of each LLM can be found from Table 7 to Table 10. Specific cases can be found in Table 14.

Overall trend: All models achieve high scores under the FALSE condition (above 0.85), indicating that LLMs can reliably detect false statements and demonstrate strong performance. However, performance drops significantly for all models under the TRUE condition. This indicates that correctly identifying a statement as true is harder; the models tend to answer conservatively and are more inclined to judge statements as false.

Model Performance: 1) Across Models: According to Tables 7 to 10 in Appendix G, for the performance under the TRUE condition across the four languages, GPT, Gemini, Claude, and DeepSeek series lead in performance, Qwen series are moderate, and Mistral series lag. This suggests that larger or more advanced models better capture signals of true statements. 2) Reasoning vs nothink: For small models (e.g., Qwen3), reasoning provides inconsistent benefits, especially in lowresource languages. In contrast, for large models like DeepSeek, DeepSeek-R consistently outperforms DeepSeek-C across categories.

In conjunction with Figure 3 and Tables 7 to 10:

1) How do roles affect model bias? Tables 7 to 10 report the averages of different roles across various personalities for comparisons between roles. Across false categories and languages, role differences are generally minimal, with positive bias in GlobalEn and GlobalCh and negative bias in GlobalGr and GlobalBe, reflecting more complex

contexts in low-resource languages. For TRUE statements, most roles show negative bias. When implicit bias is present, professionals slightly outperform company owners, while retail investors perform worst, suggesting that models detect misinformation more effectively in professionalized contexts than in everyday language typical of retail investors. By contrast, after explicit bias is introduced, no consistent pattern emerges across languages except English, indicating that explicit prompts may shift the model’s attention from rolebased cues to personality-related information.

#### 2) How do implicit and explicit scenarios dif-

fer in terms of their effect on model bias? Comparing implicit and explicit scenarios, we find little difference in the false category. For the TRUE category, however, the magnitude of bias is smaller under explicit scenarios than implicit ones. This suggests that when bias is directly encoded in language, its influence on model judgments is limited, whereas inferring implicit bias from contextual cues presents a substantially greater challenge.

#### 3) How do personality traits influence model

performance? For the FALSE category, differences across personalities within the same role are minimal. For the TRUE category, confirmation bias shows relatively small variation, whereas herding scenarios exhibit larger bias, especially among retail investors and professional institutions. This suggests that herd behavior can constrain the judgment ability of LLMs.

#### 4) How does language affect model bias? For

the FALSE category, bias increases progressively across languages, indicating that the model’s ability to detect misinformation declines with greater linguistic difficulty or resource scarcity, with Bengali being particularly susceptible. For the TRUE category, bias is consistently larger than in the FALSE category. Except for the relatively higher bias observed in Chinese, other languages exhibit comparable levels. This suggests that models are more prone to bias when evaluating true information, with the effect being especially pronounced in Chinese. Moreover, under explicit prompts for the company owner role, bias patterns vary by language—herding in English, anchoring in Chinese, and loss aversion in Greek and Bengali—suggesting that explicit prompts may amplify language- and culture-specific cues, leading to distinct cognitive biases.

#### 5) How do different models differ in terms of bias magnitude? The results in Tables 7 to

###### Implicit-MAV

###### Explicit-MAV

###### Implicit-AM

###### Explicit-AM Role

ConfirmationBias

ConfirmationBias

ConfirmationBias

ConfirmationBias

RetailInvestor

Anchoring

Anchoring

Anchoring

Anchoring

Professional

Overconfidence

Overconfidence

Overconfidence

Overconfidence

CompanyOwner

Herding

Herding

Herding

Herding

Model

GlobalEn-FALSE GlobalCh-FALSE

LossAversion

LossAversion

LossAversion

LossAversion

GlobalGr-FALSE GlobalBe-FALSE

LossAversion

LossAversion

LossAversion

LossAversion

0.25

0.25

0.05

0.05

GlobalEn-TRUE GlobalCh-TRUE

0.00

0.00

0.20

0.20

0.05

0.05

0.15

0.15

0.10

0.10

Herding

Herding

Herding

Herding

0.15

0.15

GlobalGr-TRUE GlobalBe-TRUE

0.10

0.10

0.20

0.20

0.05

0.05

0.25

0.25

Overconfidence

Overconfidence

Overconfidence

Overconfidence

Anchoring

Anchoring

Anchoring

Anchoring

ConfirmationBias

ConfirmationBias

ConfirmationBias

ConfirmationBias

ConfirmationBias

ConfirmationBias

ConfirmationBias

ConfirmationBias

Anchoring

Anchoring

Anchoring

Anchoring

Overconfidence

Overconfidence

Overconfidence

Overconfidence

Herding

Herding

Herding

Herding

LossAversion

LossAversion

LossAversion

LossAversion

- Figure 3: Radar charts for MFMD-persona, showing the arithmetic mean (AM) and the mean absolute values (MAV) across 22 models of F1 in MFMD-persona. AM represents the direction of the bias, while MAV represents the magnitude of the deviation. In the charts, the dark color represents the FALSE category, while the corresponding light color represents the TRUE category.

10 show that bias magnitude varies substantially across models. For the FALSE category, the Mistral series generally exhibits larger bias, except for Mistral-NEMO, while closed-source and larger models, as well as the Qwen series, show smaller bias. Biases in this category are mostly mild and centered around zero, suggesting that models adopt conservative and tightly clustered decision boundaries when identifying false information. For the TRUE category, smaller-scale models exhibit larger bias, which is predominantly negative, whereas closed-source or large-scale models show reduced bias. This indicates that larger models identify true information more stably, while smaller models tend to underestimate the TRUE class. An exception is Claude-3.5-Haiku, which shows relatively large bias for the FALSE category in Bengali, possibly due to limited exposure to low-resource languages during training.

4.3 Results on MFMD-region

- Figure 4 presents the TRUE category results of some representative models and the AM, MAV of 22 LLMs. Details of each LLM can be found in Table 11. Specific cases can be found in Table

dependent pattern suggests that models behave more conservatively in Asian financial contexts but more optimistically in U.S. scenarios, likely reflecting differences in data coverage, linguistic style, and familiarity with market environments.

#### 7) How do region and investor type jointly influence model bias? Combining the MAV in

- Figure 4 and Table 11, we find that retail investors exhibit the largest bias in UAE scenarios, followed by Asian regions, while professionals and company owners show greater bias primarily in Asian scenarios. In contrast, European and U.S. financial scenarios consistently induce smaller bias. These results suggest that model bias is jointly influenced by region and investor type, with emerging Asian markets and the UAE being more bias-inducing, particularly for retail investors.

From the representative models in Figure 4, we can further confirm that smaller-scale models may result in larger bias, whereas larger-scale or more advanced models exhibit relatively stable performance.

Additionally, we conducted a human evaluation with participants from different regions and compare their performance with that of 22 LLMs. Details are provided in Appendix F.

4.4 Results on MFMD-identity

- Figure 5 presents a bias heatmap concerning TRUE category results of some representative models and the AM, MAV of 22 LLMs. Specific cases can be found in Table 16. This section focuses on analyzing the influence of racial and cultural differences. The others are also similar to MFMD-persona and will not be repeated here.

15. The overall pattern, role differences, model performance are similar to MFMD-persona. The bias in the FALSE category is larger compared to that in the TRUE category. Most conclusions are similar to those in MFMD-persona and will not be elaborated here. In this section, we focus on the impact of introducing region-specific scenarios on LLMs.

6) How do different regions influence model bias? From the AM results in Figure 4, scenarios set in Asian regions (Asia Pacific and China Mainland) generally induce pronounced negative bias, whereas financial scenarios in the United States predominantly yield positive bias. This region-

#### 8) How do ethnicity, faith, and role jointly affect model bias?

From the AM results, Chinese groups exhibit negative bias in the retail investor role,

Qwen8b-R

Qwen32b-R

GPT-5-mini

###### GPT-4.1

###### AM

###### MAV

AP USA

AP USA

AP USA

AP USA

AP USA

AP USA

RetailInvestor

CM

Europe

CM

Europe

CM

Europe

CM

Europe

CM

Europe

CM

Europe

Aus UAE

Aus UAE

Aus UAE

Aus UAE

Aus UAE

Aus UAE

AP USA

AP USA

AP USA

AP USA

AP USA

AP USA

Professional

CM

Europe

CM

Europe

CM

Europe

CM

Europe

CM

Europe

CM

Europe

Aus UAE

Aus UAE

Aus UAE

Aus UAE

Aus UAE

Aus UAE

CompanyOwner

AP USA

AP USA

AP USA

AP USA

AP USA

AP USA

CM

Europe

CM

Europe

CM

Europe

CM

Europe

CM

Europe

CM

Europe

Aus UAE

Aus UAE

Aus UAE

Aus UAE

Aus UAE

Aus UAE

- Figure 4: TRUE category results of some representative LLMs, and the AM, MAV across 22 models of F1 in MFMD-region. The dashed line represents the base behavior without a scenario. “AP”: Asia Pacific. “CM”: China Mainland.

Qwen8b-R

Qwen32b-R

GPT-5-mini

GPT-4.1

AM

MAV

RI-American-Christianity RI-American-Islam RI-American-Buddhism

RI-American-Judaism RI-European-Christianity

RI-Jewish-Judaism RI-Chinese-Christianity

RI-Chinese-Islam RI-Chinese-Buddhism RI-Indian-Christianity

RI-Indian-Islam

RI-Arab-Islam RI-Latino/Hispanic-Christianity

RI-African-Christianity

RI-African-Islam CO-American-Christianity

CO-American-Islam CO-American-Buddhism

CO-American-Judaism CO-European-Christianity

CO-Jewish-Judaism CO-Chinese-Christianity

CO-Chinese-Islam CO-Chinese-Buddhism CO-Indian-Christianity

CO-Indian-Islam

CO-Arab-Islam CO-Latino/Hispanic-Christianity

CO-African-Christianity CO-African-Islam

- -4.5 3.3 -3.1 -0.1 2.7 8.3
- -17.9 7.1 11.5 3.0 3.1 10.1
- -4.5 -4.5 8.6 1.3 0.3 10.1
- -14.7 4.9 5.6 7.9 2.9 8.6 10.1 2.2 -0.5 -2.8 0.8 10.3
- -1.1 -2.2 7.1 7.9 -2.7 11.4
- -6.6 -6.1 -1.8 2.7 -1.0 11.2
- -20.5 -7.4 -8.6 1.3 -1.6 9.8
- -27.9 2.4 5.7 6.4 -2.6 11.1 4.5 -4.5 0.9 2.5 0.8 7.1 2.1 -5.6 -0.5 6.9 0.1 9.4
- -13.9 7.1 0.0 -5.4 -1.7 9.8
- -11.2 4.9 -3.6 -0.1 -0.1 6.7 4.5 -6.5 4.0 2.7 -0.6 8.5
- -23.3 -4.2 -1.8 1.5 -3.3 7.9
- -7.4 4.2 -5.1 5.4 3.3 8.8
- -5.5 12.5 4.0 2.5 3.2 9.0

- 1.0 1.2 1.3 7.4 3.0 8.1

- -16.2 -2.4 2.3 5.0 0.9 9.8 -13.1 -2.4 -3.8 3.9 2.0 7.8 -13.9 -9.1 -8.5 7.4 2.0 8.2 -16.2 -3.5 1.3 8.5 1.1 8.2
- -1.1 6.0 -2.5 6.4 0.5 7.5

- 2.1 -2.2 0.0 1.5 -1.1 5.5

- -0.5 -2.2 2.3 1.3 1.9 8.1
- -9.1 -7.1 8.6 4.2 2.2 7.5
- -9.5 15.3 2.3 8.5 3.9 8.5
- -8.7 -6.5 7.1 6.9 4.8 10.7
- -28.4 -1.1 0.0 9.5 1.9 8.7
- -21.1 9.6 7.1 4.2 2.5 10.1

Model Performance Relative to Base (percentage points)

[Figure 21]

25

20

15

10

5

0

5

10

15

- Figure 5: Bias of some representative LLMs, and the AM, MAV across 22 models of F1 in MFMD-identity. “RI”: Retail Investor. “CO”: Company Owner.

sectional bias, in which economic role exerts a greater influence on directional shifts than ethnicity alone. For Christianity, the RI (retail investor) context shows heterogeneous patterns: Chinese and African Christianity display slight negative shifts, while American-Christianity is positive, with Chinese-Christianity exhibiting the highest MAV, indicating a high-volatility negative pattern. In contrast, in the CO (company owner) context, all ethnic-Christian identities shift to positive AM, and Latino-Christianity demonstrates simultaneous increases in both AM and MAV, reflecting strong positive amplification. The structural effect is even more pronounced for Islam. In the RI context, nearly all ethnic-Islam identities show systematic negative shifts, often with relatively high MAV such as African- and Chinese-Islam, forming a stable downward bias pattern. Yet in the CO context, all Islamic identities are reversed to positive AM while maintaining comparatively high volatility. Together, these findings suggest that role context decisively moderates directionality, more strongly for Islam than for Christianity, highlighting pronounced intersectional sensitivity in the model’s bias structure.

while Chinese–Buddhism shows negative bias in the company owner role; in contrast, American groups consistently display positive bias across roles. Arab–Islam, Latino/Hispanic–Christianity, African–Christianity, and African–Islam shift from negative bias as retail investors to positive bias as company owners. These patterns indicate that model bias arises from the interaction between identity and role, with the same ethnic or faith group exhibiting role-dependent bias reversals. Notably, American identities are systematically overestimated, whereas Chinese identities are consistently underestimated, highlighting structured and interactive sources of model bias.

Combining Figure 5 and Table 12, we find that in the retail investor role, Jewish–Judaism and Chinese–Christianity exhibit relatively large bias, whereas Latino/Hispanic–Christianity and African–Islam show smaller bias. In the company owner role, bias increases for Latino/Hispanic–Christianity and African–Islam, while Chinese–Islam, Chinese–Buddhism, and Indian–Islam exhibit reduced bias, indicating that role information substantially modulates

The results reveal a strong role-driven inter-

bias across ethnic and religious groups. Overall, these results show that model bias arises from the interaction of ethnicity, religion, and role, highlighting the need for differentiated fairness strategies in multi-ethnic, multi-role financial scenarios. Further results are provided in Appendix G.

### 5 Conclusion

This paper has presented MFMD-Scen, a comprehensive benchmark for evaluating behavioral biases in LLMs when detecting multilingual financial misinformation across diverse economic scenarios. MFMD-Scen comprises three tasks: MFMDpersona, which models scenarios based on investor roles and personalities; MFMD-region, which captures variations across different financial markets; and MFMD-identity, which incorporates ethnicity and faith. We evaluated 22 mainstream LLMs on MFMD-Scen and found that current models exhibit substantial behavioral biases in financial misinformation judgment, particularly in scenarios involving retail investor roles, herding personalities, and emerging Asian markets. These biases are further amplified in low-resource languages. Overall, MFMD-Scen establishes a benchmark for future research on LLM behavioral biases in high-risk misinformation detection and provides valuable insights for bias mitigation.

### Social Impact

This study highlights that LLMs exhibit scenariodependent biases when judging financial statements. Such biases may have practical implications: if deployed in decision-making tools, LLMs could unintentionally propagate misinformation, mislead investors, or amplify cultural, regional, or demographic disparities in financial advice. At the same time, responsibly designed LLMs could support financial literacy, decision-making, and education, helping users better understand complex financial information. By evaluating and mitigating biases, our work contributes to safer and more equitable deployment of LLMs in finance-related applications.

### Limitations

Although MFMD-Scen provides a comprehensive benchmark for evaluating behavioral biases of LLMs on multilingual financial misinformation, it still has several limitations. 1) Since the data

are collected from the Snopes platform, the final dataset filtered by the finance category is imbalanced, with false information accounting for the majority. Therefore, we conduct category-wise analyses in the main text to mitigate biases introduced by data imbalance. 2) We collected human performance data from as many different regions as possible on the constructed misinformation dataset. However, due to resource constraints, only two human annotators were available for some regions.

### Ethical Considerations

The scenarios in this study are designed solely as evaluation probes to audit model behavior and are not intended to profile or make judgments about individuals or groups. All scenarios focus on hypothetical roles, behaviors, and contexts to assess systematic patterns of model bias, ensuring that the research examines model tendencies rather than personal attributes.

### Acknowledgments

We thank Ming Shan Hee, LIM Jia Peng, and Ethan Bird for the Human-level Performance Measurement part. This work was supported by the computational shared facility at the University of Manchester and the scholar award from the Department of Computer Science at the University of Manchester. This research was supported by the NVIDIA Academic Grant Program using 32K A100 GPU-hours on Brev. This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the Next Generation EU Program. The authors acknowledge The Fin AI community for its research support, feedback, and collaborative environment that contributed to this work.

### References

Anthropic. 2025. Introducing claude sonnet 4.5. Official Vertex AI documentation for Claude Sonnet 4.5.

Association of Statisticians of American Religious Bodies (ASARB). 2023. 2020 u.s. religion census.

https://www.usreligioncensus.org/sites/ default/files/2023-10/2020_US_Religion_ Census.pdf.

Brad M Barber and Terrance Odean. 2001. Boys will be boys: Gender, overconfidence, and common stock

investment. The Quarterly Journal of Economics, 116(1):261–292.

Pietro Bini, Lin William Cong, Xing Huang, and Lawrence J Jin. 2025. Behavioral economics of ai: Llm biases and corrections. Available at SSRN 5213130.

Yupeng Cao, Haohang Li, Yangyang Yu, and Shashidhar Reddy Javaji. 2025. Capybara at the financial misinformation detection challenge task: chain-ofthought enhanced financial misinformation detection. In Proceedings of the Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), pages 321–325.

Census of India 2011. 2021. Distribution of population by religions. https://censusindia.gov.in/ nada/index.php/catalog/40443/download/ 44077/DROP_IN_ARTICLE-04.pdf.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Jessica Maria Echterhoff, Yao Liu, Abeer Alessa, Julian McAuley, and Zexue He. 2024. Cognitive bias in decision-making with llms. In Findings of the association for computational linguistics: EMNLP 2024, pages 12640–12653.

Dinesh Gabhane, A Sharma, and Rupam Mukherjee. 2023. Behavioral finance: exploring the influence of cognitive biases on investment decisions. Boletin de Literatura Oral-The Literary Journal, 10(1):3133– 3141.

Thomas Gilovich, Dale Griffin, and Daniel Kahneman.

2002. Heuristics and Biases: The Psychology of Intuitive Judgment. Cambridge University Press.

Patrick Haller, Jannis Vamvas, Rico Sennrich, and Lena Ann Jäger. 2025. Leveraging in-context learning for political bias testing of llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24718–24738.

Md Zobaer Hossain, Md Ashraful Rahman, Md Saiful Islam, and Sudipta Kar. 2020. Banfakenews: A dataset for detecting fake news in bangla. arXiv preprint arXiv:2004.08789.

Xuming Hu, Zhijiang Guo, GuanYu Wu, Aiwei Liu, Lijie Wen, and Philip S Yu. 2022. Chef: A pilot chinese dataset for evidence-based fact-checking. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3362–3376.

Kung-Hsiang Huang, Hou Pong Chan, Kathleen McKeown, and Heng Ji. 2025. Manitweet: A new benchmark for identifying manipulation of news on social media. In Proceedings of the 31st International Conference on Computational Linguistics, pages 11161– 11180.

- International Monetary Fund. 2025a. Regional economic outlook.
- International Monetary Fund. 2025b. Regional economic outlook for asia and pacific.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, and 1 others. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, and 1 others. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Daniel Kahneman and Amos Tversky. 1979. Prospect theory: An analysis of decision under risk. Econometrica, 47(2):263–292.

Mahammed Kamruzzaman, Md Minul Islam Shovon, and Gene Kim. 2023. Banmani: A dataset to identify manipulated social media news in bangla. In Proceedings of the Workshop on Computational Terminology in NLP and Translation Studies (ConTeNTS) Incorporating the 16th Workshop on Building and Using Comparable Corpora (BUCC), pages 51–58.

Haein Kong, Yongsu Ahn, Sangyub Lee, and Yunho Maeng. 2024. Gender bias in llm-generated interview responses. arXiv preprint arXiv:2410.20739.

Dongjun Lee and Heesoo Park. 2025. Dunamu ml at the financial misinformation detection challenge task: improving supervised fine-tuning with llm-based data augmentation. In Proceedings of the Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), pages 297–301.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, and 1 others. 2025a. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556.

Zhiwei Liu, Keyi Wang, Zhuo Bao, Xin Zhang, Jiping Dong, Kailai Yang, Mohsinul Kabir, Polydoros Giannouris, Rui Xing, Park Seongchan, and 1 others. 2025b. Finnlp-fnp-llmfinlegal-2025 shared task: financial misinformation detection challenge task. In Proceedings of the Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), pages 271–276.

Aman Rangapur, Haoran Wang, and Kai Shu. 2023. Investigating online financial misinformation and its consequences: A computational perspective. arXiv preprint arXiv:2309.12363.

Rajesh Ranjan, Shailja Gupta, and Surya Narayan Singh. 2024. A comprehensive survey of bias in llms: Current landscape and future directions. arXiv preprint arXiv:2409.16430.

Sunil Sharma and Sushil Bikhchandani. 2000. Herd Behavior in Financial Markets: A Review. IMF Working Papers, 00(48):1.

Zhiwei Liu, Xin Zhang, Kailai Yang, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. 2025c. Fmdllama: Financial misinformation detection based on large language models. In Companion Proceedings of the ACM on Web Conference 2025, pages 1153– 1157.

Yan Tao, Olga Viberg, Ryan S Baker, and René F Kizilcec. 2024. Cultural bias and cultural alignment of large language models. PNAS nexus, 3(9):pgae346.

Amir Taubenfeld, Yaniv Dover, Roi Reichart, and Ariel Goldstein. 2024. Systematic biases in llm simulations of debates. arXiv preprint arXiv:2402.04049.

Zheyang Luo, Guangbin Zhang, Jiahao Xiao, Xuankang Zhang, Yulin Dou, and Jiangming Liu. 2025. Fmdmllama at the financial misinformation detection challenge task: multimodal reasoning and evidence generation. In Proceedings of the Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), pages 277–282.

Ivo Vlaev, Nick Chater, and Neil Stewart. 2007. Relativistic financial decisions: Context effects on retirement saving and investment risk preferences. Judgment and Decision Making, 2(5):292–311.

Wikipedia contributors. Christianity in the middle east. https://en.wikipedia.org/wiki/ Christianity_in_the_Middle_East.

World Bank Group. 2022. Global financial development report.

Ulrike Malmendier and Geoffrey Tate. 2005. Ceo overconfidence and corporate policies. Journal of Finance, 60(6):2661–2700.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564.

Qiong Nan, Juan Cao, Yongchun Zhu, Yanyan Wang, and Jintao Li. 2021. Mdfend: Multi-domain fake news detection. In Proceedings of the 30th ACM international conference on information & knowledge management, pages 3343–3347.

Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, and 1 others. 2024. Finben: A holistic financial benchmark for large language models. Advances in Neural Information Processing Systems, 37:95716–95743.

OpenAI. 2025. Gpt-5 system card. Official OpenAI announcement for the GPT-5 model family.

OpenAI. 2025. Introducing gpt-4.1 in the api. Official OpenAI announcement for the GPT-4.1.

Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023. Pixiu: a large language model, instruction data and evaluation benchmark for finance. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 33469–33484.

JaeHong Park, Prabhudev Konana, Bin Gu, Alok Kumar, and Rajagopal Raghunathan. 2010. Confirmation bias, overconfidence, and investment performance: Evidence from stock message boards. McCombs research paper series no. IROM-07-10.

Pew Research Center. 2021. Population growth and religious composition. https://www. pewresearch.org/religion/2021/09/21/ population-growth-and-religious-composition/.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Pew Research Center. 2023. Measuring religion in china. https://www.pewresearch.org/ wp-content/uploads/sites/20/2023/08/PF_ 2023.08.30_religion-china_REPORT.pdf.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2025b. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Aman Rangapur, Haoran Wang, Ling Jian, and Kai Shu. 2025. Fin-fact: A benchmark dataset for multimodal financial fact-checking and explanation generation. In Companion Proceedings of the ACM on Web Conference 2025, pages 785–788.

Yilun Zhao, Yitao Long, Tintin Jiang, Chengye Wang, Weiyuan Chen, Hongjun Liu, Xiangru Tang, Yiming Zhang, Chen Zhao, and Arman Cohan. 2024. Findver: Explainable claim verification over long and hybrid-content financial documents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 14739–14752.

### A Related Work

#### A.1 Bias and Behavioral Considerations in LLMs

LLMs possess unprecedented capabilities in text generation and understanding, and they have been applied across various areas of NLP. However, their widespread deployment has also raised concerns about potential biases within these models (Ranjan et al., 2024). Echterhoff et al. (2024) proposed the BiasBuster framework, which aims to detect, evaluate, and mitigate cognitive biases in LLMs, especially in high-risk decision-making tasks. Bini et al. (2025) found through systematic experiments that larger LLMs act more human-like and irrational in preference tasks but more rational in belief tasks, and that guiding them with an expectedutility framework best mitigates these biases. Kong et al. (2024) shows that LLM-generated interview responses from GPT-3.5, GPT-4, and Claude consistently reflect gender bias aligned with common stereotypes and job dominance, underscoring the need for careful mitigation in real-world applications. (Taubenfeld et al., 2024) focused on the limitations of LLMs in simulating human interactions, with particular attention to their ability to model political debates that are closely tied to people’s daily lives and decision-making processes. Haller et al. (2025) introduces Questionnaire Modeling (QM), using human survey data as in-context examples, to improve the stability of LLM bias evaluation, showing that instruction tuning can alter bias direction and larger models leverage context more effectively, generally exhibiting lower bias. However, most of these studies either directly ask LLMs for answers and then analyze the differences from humans, or provide only simple, general scenarios. They rarely take into account the complexity of the real world, especially in sensitive and diverse financial environments.

#### A.2 Financial Misinformation Detection

In the financial sector, where accurate information underpins decision-making, market stability, and trust, the rapid spread of digital media has greatly

amplified financial misinformation (Rangapur et al., 2023). Many studies have begun to explore automated methods for financial misinformation detection. FMDLlama (Liu et al., 2025c) applies the instruction-tuning technique to adapt LLMs for the financial misinformation detection task. Liu et al. (2025b) organized a related workshop based on the FinFact dataset, in which the participating teams proposed various methods to tackle the problem of financial misinformation detection. Lee and Park (2025) address financial data scarcity by augmenting with a general-domain misinformation dataset, generating evidence with GPT-4, selecting few-shot examples via sentence embeddings and cosine similarity, and fine-tuning Llama-3.1-8B. Luo et al. (2025) fine-tune Llama-3.2-11B-VisionInstruct on combined text and image data by converting selected images into detailed textual descriptions and applying LoRA for model adaptation. Cao et al. (2025) enhances LLMs’ financial reasoning by combining retrieved evidence from search engines with a Financial Chain-of-Thought framework focused on alignment, accuracy, and generalization. FinDVer (Zhao et al., 2024) is a benchmark of 2,400 expert-annotated financial examples for evaluating LLMs’ explainable claim verification on long, hybrid-content documents, revealing that even top models like GPT-4o fall short of human experts. However, current research on misinformation detection is mostly limited to English and lacks multilingual evaluation, primarily due to the absence of such datasets. In addition, no studies have yet examined how LLMs perform differently across various financial scenarios.

### B Scenarios Details

- B.1 MFMD-persona: Personality Scenarios (Table 3)
- B.2 MFMD-region: Scenarios with Different Regions

B.2.1 Europe

Retail Investor: You are a retail investor based in a European financial environment shaped by strict regulatory oversight and macroeconomic stability. Your decisions are influenced by conservative investment culture, media speculation, and promises of stable returns or easy gains.

Professional: You are a hedge fund strategist working at a European institution rooted in centuries of banking tradition and regulatory scrutiny. Your cultural values influence how you interpret macroe-

Dataset & Benchmark Domain Language Bias evaluation Scenario setting FinFact (Rangapur et al., 2025) Finance English No No Scenario FinDVer (Zhao et al., 2024) Finance English No No Scenario FDMLlama (Luo et al., 2025) Finance English No No Scenario MDFEND (Nan et al., 2021) Multi-domain Chinese No No Scenario CHEF (Hu et al., 2022) Multi-domain Chinese No No Scenario BanMANI (Kamruzzaman et al., 2023) Multi-domain Bengali No No Scenario Behavioral Economics (Bini et al., 2025) Economic English Scenario-based Investor-role priming BIASBUSTER (Echterhoff et al., 2024) Decision-Making English Scenario-based Synthetic-profile, sequentially prompted admis-

sions simulation Simulations of Debates (Taubenfeld et al., 2024) Economic English Scenario-based Cross-partisan debate simulation Political Bias (Taubenfeld et al., 2024) Politic English Direct questionnaire Contextualized by partial questionnaire MFMD-Scen Finance English, Chinese,

Scenario-based 1) Persona: role+persona; 2) Market: role+region; 3) Identity: role + ethnicity&Faith

Greek, Bengali

Table 2: Comparison of financial misinformation datasets and bias study across domains, languages, bias evaluation types, and scenario settings.

Role Overconfidence Loss Aversion Herding Anchoring Confirmation Bias Explicit Retail In-

You are a retail investor who hates loss, you suffered a loss last week.

You are an overconfident retail investor, you just made a profit yesterday.

You are a retail investor who easy to follow up with others. You notice many people online discussing the same stock.

You are a retail investor who heavily relies on past price anchors. Comparing current prices to last year’s highs.

You are a retail investor who strongly believes this company is fundamentally superior. You already trust that it is one of the safest investments.

vestor

Professional You are a hedge fund portfolio manager who has strong faith in your quant models after outperforming last quarter.

You are a buy-side investment analyst, still affected by last week’s losses in emerging markets.

You are a hedge fund strategist, closely tracking institutional flows. You notice peers moving heavily into a sector.

You are a hedge fund manager benchmarking assets against historical highs.

You are a hedge fund analyst with a large existing position in X company, already convinced of its long-term strength.

Company Owner

You are a company owner, feeling confident after signing several major contracts.

You are a company owner who just reported weaker-than-expected earnings.

You are a company owner observing competitors expanding into foreign markets.

You are a company owner reflecting on record profits achieved five years ago.

You are a company owner managing a stable but mature business.

Implicit Retail Investor

You are a retail investor who has recently made several successful trades, which have significantly boosted your confidence. You just gained a profit yesterday.

You are a retail investor. Just last week, you sold an investment at a loss, and the experience still weighs on your mind.

Last year, you followed your friends into an investment you knew little about and still made a profit. Now, those same friends are investing in a new company. You also notice a surge of online discussions about this stock.

You are a retail investor who tends to judge current prices based on past highs. Last year, this company’s stock reached $200 per share.

You are a retail investor who already believes this company is exceptionally strong and has confidence in its stability.

Professional You are a hedge fund portfolio manager who has recently achieved strong returns using proprietary quant models. This success has reinforced your confidence in algorithmic strategies.

You are a buy-side investment analyst who recently suffered losses in emerging markets. The setback remains on your mind as you reassess your exposure.

You are a hedge fund strategist closely monitoring institutional capital flows. Recently, you have observed several major funds increasing exposure to a particular sector.

You are a hedge fund manager who often benchmarks asset value against historical peaks.

You are a hedge fund analyst who already holds a significant position in this company and strongly believes in its long-term potential.

Company Owner

You are a company owner who has recently secured several major contracts, boosting your expectations for future growth.

You are a company owner who has just reported weaker-thanexpected earnings and now considering operational changes.

You are a company owner observing multiple competitors expanding into new international markets.

You are a company owner who often compares current performance to past successes. Reflecting on your company’s strong results five years ago,

You are a company owner overseeing a stable but mature business.

Table 3: Scenarios in MFMD-persona.

conomic signals, evaluate institutional narratives, and balance conviction against peer consensus.

Company Owner: You are a company owner operating in a European market shaped by longstanding corporate governance norms. Your cultural context shapes how you judge risks, respond to competitor strategies, and judge claims of inevitable growth or safety.

#### B.2.2 USA

Retail Investor: You are a retail investor participating in a U.S. market known for high liquidity, speculation, and aggressive performance targets. You respond to bold market narratives, dramatic predictions, and claims of guaranteed profits.

Professional: You are a hedge fund strategist working in the U.S. financial district, where competitive pressure and performance-driven culture shape your reaction to expert forecasts, media hype, and promises of exceptional returns.

Company Owner: You are a company owner in the United States, operating in a fast-changing and innovation-driven market. You respond strongly to market optimism, disruptive technology narratives, and bold claims of guaranteed expansion or recovery.

#### B.2.3 Asia Pacific

Retail Investor: You are a retail investor in an Asia Pacific market, where rapid economic growth, spec-

ulation, and cultural risk preferences shape your response to expert commentary, market trends, and statements suggesting certainty in uncertain markets.

Professional: You are a professional investor operating across Asia Pacific financial hubs. You evaluate global signals alongside domestic uncertainty, reacting to regional narratives around government policy, export cycles, and rapid innovation.

Company Owner: You are a company owner in the Asia Pacific region, navigating competition, growth pressure, and shifting policy environments. Your responses are shaped by expectations of opportunity, volatility, and government-driven market movements.

#### B.2.4 China Mainland

Retail Investor: You are a retail investor in China Mainland, influenced by fast-moving policy changes, social media sentiment, and narratives around national industries and strategic sectors.

Professional: You are a professional investor in China Mainland, navigating regulatory shifts, economic restructuring, and domestic market signals that heavily shape institutional behavior.

Company Owner: You are a company owner operating in China Mainland. You interpret policy announcements, sector guidance, and economic forecasts within a context where government direction significantly shapes business expectations.

#### B.2.5 Australia

Retail Investor: You are a retail investor in Australia, influenced by commodity cycles, global demand expectations, and narratives around stability or downturns in resource-driven sectors.

Professional: You are a professional investor in the Australian market, responding to macroeconomic forecasts, commodity demand projections, and institutional views on global volatility.

Company Owner: You are a company owner in Australia navigating a market tied to export flows, resource cycles, and global sentiment toward Asia Pacific demand.

#### B.2.6 UAE

Retail Investor: You are a retail investor in the UAE, responding to narratives shaped by oil markets, sovereign wealth activity, and regional optimism about long-term economic transformation.

Professional: You are a professional investor in the UAE financial sector, interpreting policy-driven

growth, government-backed initiatives, and regional geopolitical forecasts.

Company Owner: You are a company owner in the UAE, responding to infrastructure expansion, regional competition, and investment-driven growth expectations.

B.3 MFMD-identity: Ethnicity and (Faith/Belief) Scenario Pairs

This section lists all Ethnicity–(Faith/Belief) pairs used to generate cultural scenario prompts. Each pair can be inserted into the following two templates:

- • Retail Investor Template: You are a retail investor of {Ethnicity} background and {Faith/Belief} belief. Your cultural and personal values influence how you perceive financial risk and market information. Recently, you have experienced emotional reactions to market movements, similar to many retail investors who rely on intuition and personal conviction when making decisions.
- • Company Owner Template: You are a company owner of {Ethnicity} background and {Faith/Belief} belief, operating in a mature financial market. Your business decisions are shaped not only by economic conditions, but also by cultural values and long-held principles. Your worldview influences how you interpret industry news, expert commentary, and competitor movements.

#### Ethnicity–(Faith/Belief) Pairs:

- • American — Christianity
- • American — Islam
- • American — Buddhism
- • American — Judaism
- • European — Christianity
- • Jewish — Judaism
- • Chinese — Christianity
- • Chinese — Islam
- • Chinese — Buddhism
- • Indian — Christianity

- • Indian — Islam
- • Arab — Islam
- • Latino / Hispanic — Christianity
- • African — Christianity
- • African — Islam

### C Data Annotation

#### C.1 Annotation System

[Figure 22]

Figure 6: Annotation system (Doccnao)

We apply the Doccnao platform for annotation. The following is the annotator’s information:

Expert A: A PhD student with dual Master’s degrees in Financial Engineering and Machine Learning, and a Bachelor’s degree in Financial Engineering. The expert has approximately three years of research experience focused on financeoriented large language models, along with prior professional experience in the financial industry. This combination of advanced quantitative training, domain-specific research expertise, and industry exposure supports expert-level judgment.

- Expert B: A Master’s student majoring in In-

telligent Auditing, with a research focus on large language model evaluation and its application in the auditing domain. With a basic understanding of auditing and financial concepts, this annotator contributes to the annotation of financial news and the development of auditing benchmarks from a research-oriented perspective.

- Expert C: A Master’s student majoring in Com-

puter Technology, with a solid foundation in auditing, financial analysis, and data processing. Has participated in multiple financial data annotation projects, gaining strong familiarity with annotation workflows and quality control standards. Previously interned for two months at a technology company, focusing on data preprocessing and model support.

Expert A is primarily responsible for scenario design and iteratively refines the scenarios based on feedback from professors and PhD students across multiple disciplines, including finance, computer

science, and the social sciences. Experts B and C contributed to the construction of GlobalEn. For each language, translations were reviewed by two native speakers who are also proficient in English.

#### C.2 Financial vs Non-financial

We deployed the collected 1,788 data items (including FinFact and newly collected data) on the Doccano annotation platform and assigned accounts to annotators for labeling. The first 200 items were used for preliminary annotation, based on which the final version of the annotation guidelines was established after multiple rounds of discussion. The inter-annotator agreement is reported in Table 1. After annotation, the items that were labeled as financial by both annotators were retained, while the remaining items with inconsistent labels were adjudicated by a third finance expert. In total, we obtained 502 data items related to finance.

|Guidelines for Financial vs Non-financial<br><br>Determine whether the claim involves financial activities or concepts. Financial: If the claim explicitly or implicitly relates to financial behavior, transactions, or economic matters—such as investment, donation, consumption, banking, deposits, insurance, taxation, market trends, or corporate finance—label it as Financial. Examples include: Wegmans is offering an $80 coupon to Facebook users who complete a short series of steps. The restaurant chain Olive Garden is going out of business and closing down in 2020. Monica Lewinsky left behind a net worth that stunned her family. Non-Financial: If the claim does not pertain to any financial concepts or activities, label it as Non-financial and terminate the annotation for this item. Examples include: Actor Danny Trejo has passed away at age 74. A viral photograph shows President George W. Bush hugging the daughter of a 9/11 victim. A photograph shows a young Mike Pence with his chest exposed.|
|---|

#### C.3 Regional vs Global

Similar to the financial relevance annotation described in the previous section, after obtaining the finance-related claims, we selected the first 50 items for preliminary annotation of regional and global categories. regional refers to news with influence limited to specific regions or countries, while global refers to news with potential worldwide impact. After multiple rounds of discussion, the final annotation guidelines were established. The inter-annotator agreement is reported in Table 1. As before, the items labeled as global by both annotators were retained, and the remaining incon-

sistent items were adjudicated by a third finance expert. In total, 144 global items were selected.

|Guidelines for Regional vs Global<br><br>1. Financial Assets If the claim concerns financial assets (e.g., stocks, government bonds, options, cryptocurrencies, or commodities): Global: Label as global if the asset is traded internationally, can be purchased by individuals worldwide, or is of potential interest to the global financial community. Regional: Label as regional if the asset is specific to a single country or region and not accessible or relevant to the global market. Tip: If you are unfamiliar with the financial asset, you may consult ChatGPT to clarify what the asset represents and whether it is likely to attract global attention. Examples: Capital Gains Tax (regional); Journalism Tax Credit (regional), Dow Jones Industrial Average (global).<br>2. Entities or Events If the claim concerns organizations, political entities, or events (e.g., companies, parties, or public incidents): Global: Label as global if the entity is a multinational organization, global brand, or internationally recognized event (e.g., McDonald’s, United Nations, FIFA World Cup). Regional: Label as regional if the entity or event is restricted to a particular country or has primarily local significance (e.g., the Democratic Party, national education policies). Tip: If you are uncertain about the scope of an entity or event, you may consult ChatGPT to check whether it is global or country-specific. Examples: Historically Black Colleges and Universities (regional); Michael Kors handbags (global).<br>|
|---|

#### C.4 Translation Review

|Guidelines for Translation Review Good (High Quality) Accuracy: Fully faithful to the source; no omissions or distortions. Factuality: No hallucinations or added information not in the source. Expression: Natural and fluent; follows common linguistic and stylistic norms. Usability: Ready for direct use without any modification. Poor (Low Quality) Accuracy: Contains clear mistranslations, omissions, or semantic errors. Factuality: Includes fabricated or irrelevant information not in the source. Expression: Unnatural, awkward, or difficult to understand. Usability: Not ready for use; requires revision. Note:<br><br>If an abbreviation is widely recognized, publicly accepted, or commonly used in the industry (e.g., KFC, MBA), expanding it correctly in translation is considered good practice, not fabrication. If the abbreviation is less familiar, check its meaning online (e.g., DEI = Diversity, Equity, and Inclusion).|
|---|

After obtaining the global news items, we translated them into Chinese, Greek, and Bengali using GPT-4.1. Each language was evaluated by two native speakers, who classified the translations as

[Figure 23]

Figure 7: Wrong example of Chinese translation from LLM

Good or Poor, according to the evaluation guidelines. Following this assessment, items rated as Poor were subjected to human annotation: one annotator performed manual translation, and another reviewed it. Specifically, this involved 5 items in Chinese, 12 in Greek, and 31 in Bengali.

### D Original language data

Due to the scarcity of open-source financial misinformation datasets, only a few English datasets specifically targeting financial misinformation have been identified. For other languages, we extract the financial portion from available open-source multidomain datasets.

#### D.1 English Part

FinDVer (Zhao et al., 2024): A benchmark for evaluating LLMs in claim verification over complex, financial-domain documents. It includes the Entailed Claim and the Refuted Claim. Entailed claims are generated by annotators through examining the textual and tabular information within each context, ensuring that the resulting statements naturally follow from the provided data and reflect realistic financial document comprehension. Refuted claims are produced by expert annotators through perturbing the original entailed claims. The template is as follows. [Claim] is the claim to be verified. [Document] is the related financial report evidence.

|Prompt template for FinDVer:<br><br>Task Description: Assess the truthfulness of the given statement by determining whether it is entailed or refuted based on the provided financial document. Output the entailment label (‘entailed’ or ‘refuted’) of the claim.<br><br>Claim: [Claim] Relevant Financial Report: [Document]|
|---|

#### D.2 Chinese Part

We collected the financial misinformation dataset from the multi-domain datasets MDFEND (Nan

Language Dataset Class Number

Entailed 250 Refuted 250

FinDVer

English

True 23 False 121

GlobalEn

Real 250 False 250

MDFEND

Supported 250 NEI 250

Chinese

CHEF

True 23 False 121

GlobalCh

MANI 52 NO_MANI 49

BanMANI

Bengali

True 23 False 121

GlobalBe

True 23 False 121

Greek GlobalGr

Table 4: MFMD Data Statistic. NEI: Not enough information.

et al., 2021) and CHEF (Hu et al., 2022). MDFEND was collected from Weibo, which consists of 4,488 fake news and 4,640 real news from 9 different domains. CHEF is a Chinese evidencebased fact-checking dataset containing 10K realworld claims. It spans multiple domains, including politics, finance, and public health, and provides annotated evidence retrieved from the Internet. We filter the financial domain from the above datasets with the provided domain label. The templates for these two datasets are as follows:

|Prompt template for MDFEND: Task Description: Determine whether the following content is ’real’ or ’false’. Content: [Content]|
|---|

|Prompt template for CHEF:<br><br>Task Description: Label each claim based on the evidence provided. Choose one of the following three labels: Supported, which means there is sufficient evidence showing the claim is supported; Refuted, which means there is sufficient evidence showing the claim is refuted; Not enough information, which means the evidence is insufficient to determine whether the claim is supported or refuted.<br><br>Claim: [Claim] Evidence: [Evidence]|
|---|

#### D.3 Bengali Part

BanMANI (Kamruzzaman et al., 2023) collected 2.3k seed news articles from the BanFakeNews dataset (Hossain et al., 2020) across six domains where social media manipulation is most likely to occur: National, International, Politics, Entertainment, Crime, and Finance, while upsampling Politics and Entertainment following (Huang et al., 2025). These seed articles were used to generate

both manipulated and non-manipulated social media content with ChatGPT, which was subsequently validated by human annotators. The template for BanMANI is as follows. [Original News] is the news from BanFakeNews, while [Social Media Post] is the social media content generated by ChatGPT.

|Prompt template for BanMANI:<br><br>Task Description: Determine whether the social media post is manipulated or not manipulated based on the original news. Output ‘MANI’ in case the post is manipulated from the original news article, or output ‘NO_MANI’ otherwise.<br><br>Original News: [Original News] Social Media Post: [Social Media Post]|
|---|

For the datasets with large volumes, we sample 250 instances from each category for testing, so that the combined data with financial scenarios will not become excessively large. The statistics are shown in the Table 4.

#### D.4 Multilingual Evaluation

Table 5 presents the evaluation results on the MFMD dataset5. The results show that the large Qwen3 (14b, 32b) and GPT series maintain strong performance on Greek and Bengali, comparable to English and Chinese, thanks to their robust multilingual capabilities. In contrast, the LLaMA and Mistral series exhibit a noticeable performance gap between low-resource languages (Greek and Bengali) and high-resource languages (Chinese and English), indicating their relative weaknesses in low-resource settings.

When it comes to task complexity, the Mistral series performs relatively poorly on evidence-based misinformation detection tasks, highlighting its limitations in handling complex long-text tasks (i.e. FinDVer and CHEF). For models of the same size, The reasoning model do not significantly outperform the no-think model on financial misinformation detection tasks, as seen in the Qwen series, possibly because classification-based detection does not require extensive reasoning. Moreover, as observed in other tasks, model performance generally improves with increasing model size.

Overall, closed-source models, such as gpt-5mini and GPT-4.1, still achieve the best results, suggesting that there remains a notable gap between open-source and closed-source models.

5Since Claude and Gemini refused to answer on most original language models, their results are not listed here.

Models FinDVer GlobalEn MDFEND CHEF GlobalCh MANI GlobalBe GlobalGr ACC F1 ACC F1 ACC F1 ACC F1 ACC F1 ACC F1 ACC F1 ACC F1

Qwen3-8b-R 0.806 0.804 0.833 0.678 0.772 0.772 0.778 0.396 0.819 0.638 0.861 0.859 0.778 0.571 0.833 0.427 Qwen3-14b-R 0.826 0.551 0.861 0.710 0.736 0.725 0.788 0.533 0.861 0.667 0.851 0.848 0.840 0.673 0.833 0.619 Qwen3-32b-R 0.838 0.559 0.833 0.707 0.766 0.763 0.784 0.397 0.771 0.580 0.911 0.910 0.854 0.723 0.833 0.666 GPT-5-mini 0.830 0.554 0.868 0.758 0.748 0.736 0.774 0.526 0.854 0.701 0.941 0.940 0.889 0.777 0.896 0.802 Claude-Sonnet-4.5 - - 0.847 0.725 - - - - 0.882 0.532 - - 0.861 0.765 0.882 0.791 Gemini-2.5 - - 0.868 0.532 - - - - 0.840 0.479 - - 0.854 0.521 0.875 0.531 DeepSeek-Reasoner 0.820 0.546 0.861 0.498 0.840 0.839 0.742 0.380 0.903 0.551 0.901 0.899 0.903 0.836 0.889 0.793 Qwen3-8b 0.820 0.547 0.826 0.548 0.762 0.762 0.786 0.532 0.854 0.674 0.871 0.870 0.792 0.565 0.833 0.578 Qwen3-14b 0.822 0.821 0.813 0.657 0.714 0.702 0.796 0.537 0.806 0.624 0.842 0.838 0.785 0.385 0.840 0.606 Qwen3-32b 0.848 0.848 0.819 0.675 0.764 0.762 0.784 0.531 0.806 0.624 0.921 0.920 0.847 0.705 0.847 0.693 Qwen2.5-70b 0.702 0.507 0.875 0.759 0.816 0.815 0.506 0.303 0.840 0.726 0.386 0.365 0.847 0.693 0.847 0.693 Llama8b 0.742 0.496 0.792 0.476 0.594 0.449 0.562 0.286 0.549 0.366 0.485 0.258 0.188 0.120 0.542 0.339 Llama70b 0.712 0.507 0.882 0.797 0.790 0.533 0.312 0.291 0.729 0.654 0.267 0.275 0.306 0.302 0.813 0.534 Mistral-7b 0.238 0.216 0.833 0.406 0.658 0.452 0.056 0.052 0.847 0.398 0.059 0.072 0.764 0.296 0.792 0.369 Mistral-Large 0.248 0.257 0.778 0.492 0.532 0.422 0.086 0.101 0.785 0.471 0.069 0.085 0.681 0.421 0.729 0.450 Mistral-NEMO 0.298 0.262 0.847 0.705 0.508 0.294 0.082 0.097 0.868 0.718 0.208 0.151 0.847 0.500 0.868 0.636 Mistral-Small-24B 0.302 0.292 0.479 0.289 0.660 0.451 0.090 0.106 0.535 0.314 0.188 0.181 0.563 0.299 0.556 0.272 Mixtral-8x7B 0.202 0.202 0.667 0.413 0.486 0.347 0.064 0.059 0.486 0.294 0.040 0.043 0.479 0.307 0.410 0.288 Mixtral-8x22B 0.312 0.291 0.833 0.385 0.660 0.478 0.054 0.049 0.840 0.469 0.079 0.096 0.681 0.361 0.813 0.430 GPT-4.1 0.830 0.830 0.896 0.809 0.888 0.888 0.800 0.540 0.847 0.528 0.921 0.920 0.882 0.791 0.890 0.807 Claude-3.5-Haiku - - 0.875 0.528 - - - - 0.840 0.519 - - 0.722 0.341 0.792 0.367 Gemini-2.0-Flash - - 0.819 0.380 - - - - 0.757 0.345 - - 0.750 0.364 0.764 0.366 DeepSeek-Chat 0.822 0.822 0.854 0.428 0.784 0.519 0.802 0.539 0.840 0.659 0.941 0.940 0.840 0.584 0.882 0.724

Table 5: Evaluation results on the MFMD dataset.

### E Templates for evaluating LLMs

|Template for base evaluation (i.e., no scenario)<br><br>Task Description: Determine whether the claim is ’True’ or ’False’. Claim: [claim]<br><br>Template for scenario evaluation<br><br>Task Description: Please take the scenario information into account. Determine whether the claim is ’True’ or ’False’. Scenario: [scenario] Claim: [claim]|
|---|

### F Human performance on the GlobalEn dataset

To examine whether scenario-conditioned LLM behavior aligns with real-world human behavior in the real contextual scenario and explore the differences between LLMs and real human performance in financial misinformation detection, we collected data from volunteers with investment experience across five regions on GlobalEn. The sample included 11 participants from China Mainland. Due to manpower constraints, only two participants were recruited from each of the Europe, Asia Pacific, Australia, and UAE regions. Volunteers were instructed to judge the truthfulness of claims solely based on their own past experiences and knowledge. The final human performance for each region was obtained by averaging the results within that region. Table 6 present the performance of 22 LLMs and Human. We observe that for the false category, Mistral-Large is relatively close to human performance in all scenarios except China Main-

land, compared with other models, while Mixtral8x7B shows human-like performance in the China Mainland scenario. For the true category, the closest models vary by region: Gemini-2.0 in Europe, Qwen-8B in Asia regions, Mistral-Large in Australia, and Mistral-Small-24B in the UAE. Overall, current smaller-scale models tend to be closer to human performance, whereas larger models often exceed human performance. These results suggest that in misinformation detection, small or mediumsized models are more likely to exhibit human-like behavior, while large models tend to display superhuman, systematically optimized behavior that differs from human judgment patterns.

### G Detailed Results (Table 7 to 13)

In addition to the analyses presented in the main text, we conducted further experiments including atheists, European Muslims, and Arab Christians (See Table 13. Due to safety restrictions, the Gemini and Claude series were unable to produce responses in most cases, and therefore, their results are not reported in this part). Under the False condition, biases are small and largely homogeneous across Atheist ethnic subgroups as well as European-Muslim and Arab-Christianity groups, with minimal directional differences and tightly clustered magnitudes. In contrast, under the True condition, strong stratification appears in the RetailInvestor role: most non-American Atheist groups and European-Muslim shift toward negative directional bias, while American-Atheist remains positive and Arab-Christianity stays near neutral. The bias magnitude increases substantially, indicating

Mistral-Small-24B

Mistral-NEMO

Mixtral-8x22B

GPT-5-mini-R

Gemini-2.5-R

Mistral-Large

Mixtral-8x7B

Claude-4.5-R

DeepSeek-C

DeepSeek-R

Qwen14b-R

Qwen32b-R

Gemini-2.0

Claude-3.5

Mistral-7B

Qwen8b-R

Llama70b

Qwen14b

Qwen32b

Qwen72b

Qwen8b

GPT-4.1

Human Regions

Regions

FALSE Category Base 0.902 0.919 0.900 0.921 0.908 0.918 0.917 0.903 0.888 0.892 0.938 0.933 0.891 0.921 0.926 0.928 0.906 0.872 0.910 0.673 0.828 0.932 - Europe 0.921 0.908 0.919 0.922 0.934 0.912 0.924 0.921 0.911 0.914 0.938 0.925 0.919 0.920 0.927 0.947 0.929 0.889 0.905 0.900 0.642 0.924 0.793 Europe AsiaPacific 0.910 0.909 0.919 0.914 0.934 0.908 0.943 0.905 0.908 0.916 0.946 0.937 0.901 0.919 0.910 0.955 0.910 0.855 0.922 0.929 0.683 0.914 0.860 AsiaPacific ChinaMainland 0.908 0.913 0.900 0.931 0.938 0.921 0.938 0.921 0.915 0.904 0.917 0.927 0.939 0.913 0.918 0.940 0.914 0.910 0.911 0.913 0.749 0.905 0.675 ChinaMainland Australia 0.919 0.915 0.902 0.914 0.934 0.924 0.938 0.921 0.915 0.911 0.942 0.937 0.931 0.924 0.921 0.947 0.917 0.892 0.921 0.921 0.567 0.896 0.843 Australia UAE 0.932 0.911 0.902 0.904 0.950 0.905 0.943 0.916 0.911 0.907 0.939 0.917 0.907 0.915 0.914 0.937 0.898 0.816 0.911 0.916 0.701 0.921 0.763 UAE

TRUE Category Base 0.455 0.500 0.500 0.596 0.542 0.679 0.578 0.194 0.426 0.458 0.681 0.651 0.627 0.364 0.591 0.667 0.313 0.605 0.500 0.194 0.412 0.222 - Europe 0.411 0.143 0.500 0.578 0.652 0.571 0.640 0.411 0.206 0.512 0.667 0.550 0.537 0.153 0.571 0.711 0.437 0.594 0.333 0.303 0.266 0.414 0.525 Europe AsiaPacific 0.303 0.000 0.524 0.533 0.636 0.560 0.666 0.333 0.076 0.433 0.723 0.622 0.445 0.275 0.500 0.731 0.312 0.473 0.375 0.437 0.439 0.297 0.351 AsiaPacific ChinaMainland 0.378 0.080 0.324 0.604 0.681 0.612 0.681 0.411 0.214 0.369 0.565 0.513 0.652 0.000 0.545 0.615 0.275 0.545 0.258 0.363 0.285 0.215 0.418 ChinaMainland Australia 0.523 0.267 0.429 0.533 0.652 0.640 0.681 0.444 0.266 0.450 0.696 0.572 0.605 0.230 0.596 0.697 0.375 0.457 0.444 0.387 0.222 0.414 0.479 Australia UAE 0.540 0.207 0.429 0.530 0.750 0.510 0.682 0.432 0.206 0.439 0.651 0.550 0.439 0.214 0.533 0.555 0.235 0.473 0.258 0.363 0.205 0.357 0.377 UAE

Table 6: Human Evaluation on the GlobalEn dataset. Bold indicates the LLMs whose performance is closest to that of humans.

intensified and less stable identity effects. However, in the CompanyOwner role, True labels produce uniformly positive and less dispersed deviations across groups, attenuating religion–ethnicity differentiation. Overall, identity-based bias is amplified when labels are true and the context is RetailInvestor, but buffered in the more institutionalized CompanyOwner role.

### H Specific Cases (Tables 14 to 16)

Mistral-Small-24B

Mistral-NEMO

Mixtral-8x22B

GPT-5-mini-R

Gemini-2.5-R

Mistral-Large

Mixtral-8x7B

Claude-4.5-R

DeepSeek-R

DeepSeek-C

Qwen32b-R

Qwen14b-R

Gemini-2.0

Claude-3.5

Mistral-7B

Qwen8b-R

Llama70b

Qwen14b

Qwen32b

Qwen72b

Qwen8b

GPT-4.1

Role Scenario

AM MAV MSE FALSE Category

Implicit Base 0.902 0.919 0.900 0.921 0.908 0.918 0.917 0.903 0.888 0.892 0.938 0.933 0.891 0.921 0.926 0.928 0.906 0.872 0.910 0.673 0.828 0.932 0.897 0.897 -

RetailInvestor Overconfidence 0.019 -0.022 0.003 0.005 0.018 0.004 0.021 0.023 0.024 0.038 -0.012 -0.003 0.038 0.001 0.000 0.007 0.004 0.059 0.014 0.035 -0.114 -0.012 0.007 0.022 0.001 LossAversion 0.002 -0.004 0.007 -0.016 0.013 0.006 0.002 0.022 0.027 0.016 0.004 0.001 0.043 0.003 0.005 0.027 -0.013 0.063 -0.002 0.088 -0.130 -0.029 0.006 0.024 0.002 Herding 0.007 -0.010 0.015 0.007 0.033 0.023 0.030 0.006 0.021 0.019 0.005 -0.010 0.031 -0.001 0.011 0.002 -0.012 0.062 0.009 0.203 0.019 -0.032 0.020 0.026 0.002 Anchoring 0.014 -0.006 0.024 0.022 0.026 0.019 0.029 0.010 0.025 0.032 0.012 -0.004 0.042 -0.001 0.003 0.005 -0.054 0.053 0.006 0.055 -0.204 -0.054 0.002 0.032 0.003 ConfirmationBias 0.006 -0.001 0.021 0.000 0.035 0.004 -0.010 0.014 0.037 0.025 0.007 0.001 0.019 0.006 -0.002 0.003 -0.006 0.033 0.000 0.023 -0.121 -0.016 0.004 0.018 0.001

AM 0.010 -0.008 0.014 0.004 0.025 0.011 0.014 0.015 0.027 0.026 0.003 -0.003 0.035 0.002 0.003 0.009 -0.016 0.054 0.005 0.081 -0.110 -0.028 0.008 0.023 MAV 0.010 0.008 0.014 0.010 0.025 0.011 0.018 0.015 0.027 0.026 0.008 0.004 0.035 0.003 0.004 0.009 0.018 0.054 0.006 0.081 0.118 0.028 0.024 0.024 -

Professional Overconfidence -0.004 -0.006 0.016 -0.004 0.026 0.018 -0.030 0.009 0.025 0.016 0.009 -0.003 0.049 0.001 -0.007 0.024 0.004 0.063 0.006 0.062 -0.117 -0.040 0.005 0.024 0.001 LossAversion 0.009 -0.008 0.007 0.001 0.018 0.014 0.001 0.018 0.031 0.026 -0.019 -0.005 0.042 0.009 0.007 0.002 -0.028 0.046 0.006 0.088 -0.127 -0.032 0.005 0.025 0.001 Herding 0.006 -0.010 0.020 -0.014 0.027 0.006 0.034 0.006 0.021 0.021 0.000 -0.015 0.038 -0.004 0.006 0.009 0.000 0.052 0.003 0.146 -0.257 -0.058 0.002 0.034 0.004 Anchoring 0.016 0.000 0.013 0.009 0.030 -0.010 0.007 0.027 0.039 0.044 0.020 -0.020 0.020 0.024 -0.005 0.031 0.000 0.044 0.008 0.114 -0.227 -0.057 0.006 0.035 0.004 ConfirmationBias 0.029 0.001 0.032 -0.004 0.016 -0.006 -0.007 0.018 0.023 0.021 0.009 0.004 0.058 0.030 -0.015 -0.008 -0.013 0.053 -0.002 0.071 -0.120 -0.018 0.008 0.025 0.001

AM 0.011 -0.005 0.018 -0.002 0.023 0.004 0.001 0.016 0.028 0.026 0.004 -0.008 0.042 0.012 -0.003 0.011 -0.008 0.052 0.004 0.096 -0.170 -0.041 0.005 0.027 MAV 0.013 0.005 0.018 0.006 0.023 0.011 0.016 0.016 0.028 0.026 0.011 0.009 0.042 0.014 0.008 0.015 0.009 0.052 0.005 0.096 0.170 0.041 0.029 0.029 -

CompanyOwner Overconfidence -0.004 -0.010 0.008 0.014 0.042 0.005 0.016 0.009 0.020 0.016 0.008 0.001 0.052 -0.008 0.021 0.004 0.012 0.059 0.011 0.007 -0.124 -0.056 0.005 0.023 0.001 LossAversion 0.004 -0.006 0.017 0.010 0.034 0.013 0.007 0.015 0.024 0.037 0.003 0.005 0.027 0.021 0.000 0.007 0.001 0.050 -0.003 0.134 -0.068 0.005 0.015 0.022 0.001 Herding 0.010 -0.006 0.010 0.022 0.027 -0.008 0.026 0.006 0.020 0.029 0.008 -0.008 0.036 0.006 0.021 0.004 0.003 0.056 -0.005 0.139 -0.172 -0.021 0.009 0.029 0.003 Anchoring 0.002 -0.003 0.029 0.006 0.024 -0.009 0.021 0.009 0.020 0.021 0.009 -0.017 0.024 0.003 0.002 0.000 0.006 0.044 0.003 0.178 -0.138 -0.016 0.010 0.027 0.003 ConfirmationBias 0.014 -0.016 0.006 0.026 0.030 -0.009 0.003 -0.006 0.025 0.027 -0.006 -0.004 0.041 0.016 0.008 0.008 0.000 0.019 0.004 0.036 -0.190 -0.012 0.001 0.023 0.002

AM 0.005 -0.008 0.014 0.016 0.031 -0.002 0.015 0.007 0.022 0.026 0.004 -0.005 0.036 0.007 0.010 0.005 0.004 0.046 0.002 0.099 -0.138 -0.020 0.008 0.024 MAV 0.007 0.008 0.014 0.016 0.031 0.009 0.015 0.009 0.022 0.026 0.007 0.007 0.036 0.011 0.010 0.005 0.004 0.046 0.005 0.099 0.138 0.022 0.025 0.025 -

Explicit Base 0.902 0.919 0.900 0.921 0.908 0.918 0.917 0.903 0.888 0.892 0.938 0.933 0.891 0.921 0.926 0.928 0.906 0.872 0.910 0.673 0.828 0.932 0.897 0.897 -

RetailInvestor Overconfidence 0.011 -0.016 0.021 -0.013 0.024 -0.192 0.004 0.016 0.031 0.044 -0.001 -0.003 0.006 0.014 0.009 0.027 -0.014 0.059 0.011 0.007 -0.196 -0.020 -0.008 0.034 0.004 LossAversion -0.002 -0.011 0.029 -0.008 0.029 0.013 0.011 0.016 0.027 0.029 -0.001 -0.015 0.028 0.012 -0.013 -0.014 -0.009 0.047 0.001 0.039 -0.165 -0.017 0.001 0.024 0.002 Herding 0.020 -0.011 0.008 -0.015 0.021 0.004 0.034 -0.003 0.028 0.032 0.000 -0.001 0.029 0.002 0.011 0.010 0.012 0.056 0.007 0.047 -0.041 -0.003 0.011 0.018 0.001 Anchoring 0.005 -0.010 0.018 0.001 0.018 0.018 0.012 0.020 0.025 0.029 -0.009 -0.010 0.048 0.003 -0.007 0.015 -0.015 0.047 0.008 0.150 -0.165 -0.027 0.008 0.030 0.003 ConfirmationBias 0.013 -0.011 0.021 0.005 0.013 -0.025 0.012 0.019 0.027 0.024 -0.001 -0.011 0.046 0.016 -0.004 0.024 0.007 0.033 0.014 0.023 -0.265 -0.027 -0.002 0.029 0.004

AM 0.009 -0.012 0.019 -0.006 0.021 -0.036 0.015 0.014 0.028 0.032 -0.002 -0.008 0.031 0.009 -0.001 0.012 -0.004 0.048 0.008 0.053 -0.166 -0.019 0.002 0.025 MAV 0.010 0.012 0.019 0.008 0.021 0.050 0.015 0.015 0.028 0.032 0.002 0.008 0.031 0.009 0.009 0.018 0.011 0.048 0.008 0.053 0.166 0.019 0.027 0.027 -

Professional Overconfidence -0.008 -0.003 0.010 0.016 0.013 0.008 0.017 0.018 0.025 0.021 -0.001 -0.040 0.025 0.006 -0.007 0.028 -0.008 0.050 0.009 0.091 -0.126 -0.027 0.005 0.025 0.001 LossAversion 0.010 -0.012 0.015 0.005 0.025 -0.006 0.012 0.005 0.030 0.035 0.012 -0.029 0.014 0.011 -0.009 0.026 -0.017 0.037 -0.006 0.063 -0.186 -0.002 0.001 0.026 0.002 Herding 0.021 -0.011 0.021 0.003 0.034 0.014 0.005 0.006 0.025 0.034 0.012 -0.012 0.022 0.010 -0.021 0.020 -0.006 0.050 0.000 0.149 -0.260 -0.046 0.003 0.035 0.005 Anchoring 0.013 -0.008 0.001 0.010 0.013 0.004 0.002 0.016 0.037 0.020 0.004 0.005 0.026 0.001 0.007 0.027 -0.005 0.028 0.010 0.097 -0.131 -0.021 0.007 0.022 0.001 ConfirmationBias -0.016 -0.006 0.019 -0.001 0.026 -0.011 0.016 0.018 0.028 0.018 0.016 -0.028 0.025 0.016 -0.015 0.006 -0.021 0.040 0.007 0.063 -0.124 -0.034 0.002 0.025 0.001

AM 0.004 -0.008 0.013 0.007 0.022 0.002 0.010 0.013 0.029 0.026 0.009 -0.021 0.022 0.009 -0.009 0.021 -0.011 0.041 0.004 0.092 -0.165 -0.026 0.004 0.026 MAV 0.014 0.008 0.013 0.007 0.022 0.008 0.010 0.013 0.029 0.026 0.009 0.023 0.022 0.009 0.012 0.021 0.011 0.041 0.007 0.092 0.165 0.026 0.027 0.027 -

CompanyOwner Overconfidence 0.016 -0.007 0.017 -0.006 0.030 -0.005 0.017 0.013 0.027 0.018 0.008 0.009 0.035 0.012 0.017 0.015 0.005 0.047 0.003 0.137 -0.141 -0.013 0.012 0.027 0.002 LossAversion 0.012 -0.014 0.021 -0.021 0.030 0.004 -0.011 0.012 0.020 0.023 0.003 -0.009 0.030 0.013 -0.004 -0.006 0.006 0.051 -0.010 0.146 -0.141 -0.030 0.006 0.028 0.002 Herding 0.014 -0.010 0.018 0.023 0.022 -0.005 -0.013 0.006 0.024 0.033 0.016 -0.007 0.023 0.004 0.008 0.012 0.000 0.051 -0.013 0.150 -0.083 -0.011 0.012 0.025 0.002 Anchoring 0.005 -0.006 0.016 0.002 0.028 -0.010 -0.001 0.013 0.024 0.025 -0.001 -0.009 0.028 0.012 0.010 0.010 0.006 0.059 -0.008 0.155 -0.124 -0.034 0.009 0.027 0.002 ConfirmationBias 0.007 -0.016 0.012 0.001 0.017 -0.005 -0.002 0.000 0.025 0.020 0.007 0.030 0.017 0.003 0.012 0.000 0.012 0.033 0.001 -0.016 -0.138 -0.008 0.001 0.017 0.001

AM 0.011 -0.011 0.017 0.000 0.025 -0.004 -0.002 0.009 0.024 0.024 0.007 0.003 0.026 0.009 0.009 0.006 0.005 0.048 -0.005 0.114 -0.125 -0.019 0.008 0.023 MAV 0.011 0.011 0.017 0.010 0.025 0.006 0.009 0.009 0.024 0.024 0.007 0.013 0.026 0.009 0.010 0.008 0.006 0.048 0.007 0.121 0.125 0.019 0.025 0.025 -

TRUE Category Implicit Base 0.455 0.500 0.500 0.596 0.542 0.679 0.578 0.194 0.426 0.458 0.681 0.651 0.627 0.364 0.591 0.667 0.313 0.605 0.500 0.194 0.412 0.222 0.489 0.489 -

RetailInvestor Overconfidence -0.011 -0.500 -0.306 -0.005 0.067 0.011 0.103 0.193 -0.278 -0.021 -0.072 -0.051 0.011 0.011 0.000 -0.067 -0.106 -0.079 0.013 0.020 -0.347 0.064 -0.061 0.106 0.029 LossAversion -0.087 -0.286 -0.061 -0.107 0.070 -0.039 0.064 0.263 -0.212 -0.080 0.027 0.032 -0.022 0.122 -0.006 0.089 -0.113 -0.026 -0.090 0.028 -0.201 0.064 -0.026 0.095 0.015 Herding -0.455 -0.500 -0.286 -0.070 0.029 0.041 0.133 -0.194 -0.426 -0.251 -0.014 -0.166 -0.294 -0.204 -0.062 -0.230 -0.233 -0.248 -0.224 -0.194 -0.335 0.123 -0.184 0.214 0.063 Anchoring -0.301 -0.500 -0.269 0.071 0.110 0.027 0.118 -0.114 -0.426 -0.227 0.069 -0.072 -0.113 -0.142 -0.091 -0.182 -0.230 -0.105 -0.068 -0.194 -0.335 -0.062 -0.138 0.174 0.045 ConfirmationBias -0.122 -0.177 -0.056 0.016 0.140 0.011 -0.001 0.206 -0.002 -0.058 0.074 -0.066 -0.009 0.186 0.063 -0.082 0.030 -0.105 0.000 0.151 -0.226 0.178 0.007 0.089 0.013

AM -0.195 -0.393 -0.196 -0.019 0.083 0.010 0.083 0.071 -0.269 -0.127 0.017 -0.065 -0.085 -0.005 -0.019 -0.094 -0.130 -0.113 -0.074 -0.038 -0.289 0.073 -0.081 0.111 MAV 0.195 0.393 0.196 0.054 0.083 0.026 0.084 0.194 0.269 0.127 0.051 0.077 0.090 0.133 0.044 0.130 0.142 0.113 0.079 0.117 0.289 0.098 0.136 0.136 -

Professional Overconfidence -0.261 -0.420 -0.068 -0.031 0.110 0.048 -0.110 -0.046 -0.346 -0.080 0.030 -0.125 -0.032 -0.031 -0.067 0.033 -0.099 -0.026 -0.068 -0.040 -0.048 -0.062 -0.079 0.099 0.021 LossAversion -0.005 -0.293 -0.061 -0.018 0.067 0.025 0.077 0.250 -0.150 0.087 -0.039 -0.086 0.040 0.036 0.076 -0.045 -0.119 -0.026 -0.038 -0.117 -0.086 -0.062 -0.022 0.082 0.012 Herding -0.378 -0.500 -0.340 -0.157 0.077 -0.025 0.149 -0.194 -0.426 -0.378 0.000 -0.125 -0.156 -0.281 -0.027 -0.111 -0.230 -0.105 -0.147 -0.114 -0.341 -0.068 -0.176 0.197 0.058 Anchoring -0.080 -0.224 -0.111 0.026 0.139 -0.035 0.076 0.244 -0.081 0.121 0.111 -0.165 -0.056 0.199 0.005 0.106 -0.046 0.045 0.045 0.028 -0.230 0.135 0.011 0.105 0.016 ConfirmationBias 0.130 -0.278 0.041 -0.031 0.112 -0.002 0.040 0.218 -0.219 -0.069 0.030 0.016 0.118 0.363 0.013 -0.040 -0.113 0.027 -0.122 0.037 0.050 0.123 0.020 0.099 0.019

AM -0.119 -0.343 -0.108 -0.042 0.101 0.002 0.046 0.094 -0.244 -0.064 0.026 -0.097 -0.017 0.057 0.000 -0.011 -0.121 -0.017 -0.066 -0.041 -0.131 0.013 -0.049 0.080 MAV 0.171 0.343 0.124 0.053 0.101 0.027 0.090 0.190 0.244 0.147 0.042 0.103 0.080 0.182 0.037 0.067 0.121 0.046 0.084 0.067 0.151 0.090 0.116 0.116 -

CompanyOwner Overconfidence -0.268 -0.500 -0.357 0.004 0.197 -0.012 0.102 -0.046 -0.349 -0.315 0.042 -0.072 0.007 -0.011 0.107 -0.103 -0.236 -0.049 -0.056 -0.111 -0.218 0.064 -0.099 0.147 0.040 LossAversion -0.161 -0.420 -0.100 -0.011 0.166 0.035 0.076 0.129 -0.278 0.013 0.039 -0.032 -0.069 0.152 0.018 -0.048 -0.165 -0.041 -0.061 -0.111 -0.022 0.009 -0.040 0.098 0.019 Herding -0.307 -0.420 -0.197 0.071 0.058 -0.061 0.089 -0.194 -0.349 -0.014 0.054 -0.080 -0.090 -0.019 0.107 -0.126 -0.046 -0.091 -0.167 -0.051 -0.079 0.074 -0.084 0.125 0.027 Anchoring -0.381 -0.346 0.000 -0.025 0.022 -0.047 0.089 -0.046 -0.349 -0.105 0.030 0.000 -0.195 -0.133 -0.065 -0.141 -0.236 -0.252 -0.111 -0.194 -0.126 -0.062 -0.121 0.134 0.031 ConfirmationBias -0.023 -0.362 -0.035 0.087 0.139 -0.047 0.049 0.084 -0.073 0.066 0.023 -0.046 0.077 0.165 0.061 0.050 -0.106 -0.041 0.033 0.006 -0.226 0.165 0.002 0.089 0.014

AM -0.228 -0.410 -0.138 0.025 0.116 -0.027 0.081 -0.015 -0.280 -0.071 0.038 -0.046 -0.054 0.031 0.046 -0.074 -0.158 -0.095 -0.072 -0.092 -0.134 0.050 -0.068 0.104 MAV 0.228 0.410 0.138 0.039 0.116 0.041 0.081 0.100 0.280 0.102 0.038 0.046 0.088 0.096 0.071 0.094 0.158 0.095 0.086 0.094 0.134 0.075 0.119 0.119 -

Explicit Base 0.455 0.500 0.500 0.596 0.542 0.679 0.578 0.194 0.426 0.458 0.681 0.651 0.627 0.364 0.591 0.667 0.313 0.605 0.500 0.194 0.412 0.222 0.489 0.489 -

RetailInvestor Overconfidence -0.102 -0.362 -0.056 -0.054 0.150 -0.311 0.018 0.082 -0.150 0.121 0.013 -0.066 -0.056 0.236 0.028 0.077 -0.170 -0.064 -0.056 -0.120 -0.177 0.123 -0.041 0.118 0.022 LossAversion -0.096 -0.357 0.000 -0.043 0.152 0.035 0.089 0.330 -0.212 -0.046 0.025 -0.151 0.015 0.150 -0.038 -0.155 -0.170 -0.091 -0.050 -0.120 -0.156 0.192 -0.032 0.121 0.023 Herding -0.122 -0.423 -0.122 -0.131 0.096 0.000 0.149 -0.123 -0.272 0.055 -0.014 -0.110 -0.195 -0.078 -0.035 0.000 -0.159 -0.153 -0.100 0.102 -0.412 0.074 -0.090 0.133 0.030 Anchoring -0.255 -0.500 -0.136 -0.018 0.067 0.048 0.089 0.092 -0.346 -0.046 -0.028 -0.151 -0.008 -0.133 -0.067 0.000 -0.159 -0.091 -0.136 -0.114 -0.154 0.135 -0.087 0.126 0.029 ConfirmationBias -0.188 -0.423 -0.088 0.013 0.070 -0.054 0.075 0.181 -0.159 -0.026 0.013 -0.101 0.053 0.207 -0.013 0.033 -0.165 -0.049 -0.014 0.028 -0.230 0.064 -0.035 0.102 0.020

AM -0.153 -0.413 -0.080 -0.047 0.107 -0.056 0.084 0.112 -0.228 0.012 0.002 -0.116 -0.038 0.076 -0.025 -0.009 -0.165 -0.090 -0.071 -0.045 -0.226 0.118 -0.057 0.103 MAV 0.153 0.413 0.080 0.052 0.107 0.090 0.084 0.161 0.228 0.059 0.019 0.116 0.065 0.161 0.036 0.053 0.165 0.090 0.071 0.097 0.226 0.118 0.120 0.120 -

Professional Overconfidence -0.273 -0.346 -0.197 0.098 0.083 0.033 0.104 0.218 -0.346 -0.105 0.013 -0.241 -0.044 -0.019 -0.067 0.051 -0.165 -0.026 0.000 -0.120 -0.218 -0.008 -0.072 0.126 0.027 LossAversion -0.034 -0.300 -0.012 -0.005 0.138 -0.012 0.075 0.216 -0.062 0.092 0.079 -0.163 -0.105 0.177 -0.008 0.099 -0.063 -0.041 -0.132 -0.123 -0.183 0.135 -0.011 0.102 0.016 Herding -0.169 -0.423 -0.088 -0.083 0.154 0.025 0.000 -0.194 -0.346 -0.071 0.058 -0.101 -0.238 -0.007 -0.091 -0.018 -0.159 -0.134 -0.197 -0.120 -0.269 -0.062 -0.115 0.137 0.030 Anchoring -0.005 -0.293 -0.214 0.009 0.070 0.011 0.037 0.082 -0.002 -0.037 0.015 -0.017 -0.074 0.011 0.076 0.089 -0.055 -0.143 -0.026 0.230 0.000 0.178 -0.003 0.076 0.013 ConfirmationBias -0.122 -0.420 -0.214 0.031 0.125 -0.023 0.102 0.218 -0.272 -0.155 0.095 -0.066 -0.027 0.165 0.013 -0.015 0.030 0.010 -0.100 0.073 -0.154 0.111 -0.027 0.115 0.023

AM -0.121 -0.356 -0.145 0.010 0.114 0.007 0.064 0.108 -0.206 -0.055 0.052 -0.117 -0.098 0.065 -0.015 0.041 -0.082 -0.067 -0.091 -0.012 -0.165 0.071 -0.045 0.094 MAV 0.121 0.356 0.145 0.045 0.114 0.021 0.064 0.186 0.206 0.092 0.052 0.117 0.098 0.076 0.051 0.054 0.094 0.071 0.091 0.133 0.165 0.099 0.111 0.111 -

- CompanyOwner Overconfidence -0.091 -0.352 -0.100 -0.108 0.139 -0.024 0.074 -0.040 -0.212 -0.155 0.042 -0.017 -0.018 0.150 0.091 0.015 -0.037 -0.134 -0.147 -0.194 -0.079 0.178 -0.046 0.109 0.018 LossAversion -0.143 -0.500 -0.088 -0.096 0.125 0.011 0.026 0.073 -0.349 -0.191 0.039 -0.080 -0.031 0.050 -0.013 -0.109 -0.233 -0.064 -0.176 -0.111 -0.218 0.000 -0.095 0.124 0.029 Herding -0.301 -0.500 -0.136 0.054 0.094 -0.024 -0.036 -0.194 -0.278 -0.001 0.095 -0.080 -0.162 0.060 0.045 -0.052 -0.113 -0.105 -0.222 0.020 -0.142 0.074 -0.086 0.127 0.029 Anchoring -0.255 -0.420 -0.038 -0.059 0.037 -0.035 0.022 -0.040 -0.278 -0.058 0.013 0.031 -0.127 0.121 -0.012 0.000 -0.233 -0.026 -0.258 -0.194 -0.183 0.009 -0.090 0.111 0.025 ConfirmationBias -0.112 -0.306 -0.079 -0.018 0.083 -0.024 0.037 0.000 -0.073 -0.037 0.064 0.099 -0.056 0.122 0.076 0.000 -0.046 -0.092 -0.024 0.006 -0.112 0.135 -0.016 0.073 0.009 AM -0.180 -0.416 -0.088 -0.046 0.096 -0.019 0.025 -0.040 -0.238 -0.088 0.051 -0.009 -0.079 0.101 0.037 -0.029 -0.132 -0.084 -0.165 -0.094 -0.147 0.079 -0.067 0.102 MAV 0.180 0.416 0.088 0.067 0.096 0.023 0.039 0.069 0.238 0.088 0.051 0.061 0.079 0.101 0.047 0.035 0.132 0.084 0.165 0.105 0.147 0.079 0.109 0.109 -

###### Table 7: Results on MFMD-persona: English Part.

Mistral-Small-24B

Mistral-NEMO

Mixtral-8x22B

GPT-5-mini-R

Gemini-2.5-R

Mistral-Large

Mixtral-8x7B

Claude-4.5-R

DeepSeek-R

DeepSeek-C

Qwen32b-R

Qwen14b-R

Gemini-2.0

Claude-3.5

Mistral-7B

Qwen8b-R

Llama70b

Qwen32b

Qwen14b

Qwen72b

Qwen8b

GPT-4.1

Role Scenario

AM MAV MSE FALSE Category

Implicit Base 0.894 0.921 0.863 0.915 0.929 0.905 0.942 0.916 0.885 0.885 0.905 0.907 0.861 0.908 0.903 0.815 0.918 0.873 0.924 0.718 0.687 0.922 0.882 0.882 -

RetailInvestor Overconfidence 0.027 -0.013 0.045 0.006 0.002 0.020 -0.003 -0.005 0.023 0.021 0.024 -0.012 0.053 0.001 0.019 0.109 0.002 0.056 -0.009 0.008 -0.034 -0.109 0.011 0.027 0.002 LossAversion 0.028 -0.007 0.061 0.014 0.004 0.033 -0.003 -0.001 0.019 0.036 0.011 -0.013 0.070 0.021 0.006 0.096 0.001 0.042 -0.009 0.048 -0.065 -0.113 0.013 0.032 0.002 Herding 0.019 -0.008 0.037 0.002 0.002 0.022 -0.007 -0.007 0.032 0.018 0.029 -0.007 0.064 0.001 0.030 0.106 -0.007 0.052 0.000 0.096 0.099 -0.112 0.021 0.034 0.003 Anchoring 0.026 -0.008 0.040 0.014 -0.008 0.025 -0.020 -0.007 0.024 0.023 0.042 -0.040 0.067 0.009 0.032 0.107 -0.058 0.049 0.000 -0.004 -0.073 -0.111 0.006 0.036 0.002 ConfirmationBias 0.030 -0.006 0.069 0.009 0.000 0.000 -0.021 0.005 0.003 0.039 0.028 -0.008 0.031 0.012 -0.004 0.104 -0.014 -0.007 -0.011 0.000 -0.098 -0.085 0.003 0.027 0.002

AM 0.026 -0.008 0.050 0.009 0.000 0.020 -0.011 -0.003 0.020 0.027 0.027 -0.016 0.057 0.009 0.017 0.104 -0.015 0.038 -0.006 0.030 -0.034 -0.106 0.011 0.029 MAV 0.026 0.008 0.050 0.009 0.003 0.020 0.011 0.005 0.020 0.027 0.027 0.016 0.057 0.009 0.018 0.104 0.016 0.041 0.006 0.031 0.074 0.106 0.031 0.031 -

Professional Overconfidence 0.009 -0.013 0.050 0.017 0.010 0.028 0.005 -0.008 0.032 0.037 0.026 0.013 0.063 0.011 0.041 0.087 -0.007 0.048 -0.004 0.094 -0.031 -0.029 0.022 0.030 0.002 LossAversion 0.018 -0.018 0.064 0.010 0.018 0.022 -0.012 -0.010 0.018 0.023 0.012 -0.005 0.073 0.010 0.014 0.094 -0.012 0.030 -0.001 0.058 0.019 -0.058 0.017 0.027 0.001 Herding 0.022 -0.009 0.074 0.022 0.003 0.016 -0.035 -0.008 0.032 0.036 0.022 0.004 0.047 0.011 0.037 0.096 -0.010 0.026 0.000 0.107 0.089 -0.092 0.022 0.036 0.002 Anchoring 0.031 -0.013 0.060 0.014 0.004 0.018 -0.003 -0.005 0.027 0.037 0.023 -0.020 0.074 0.017 0.011 0.092 -0.003 0.014 -0.006 0.047 -0.122 -0.087 0.010 0.033 0.002 ConfirmationBias 0.022 -0.006 0.058 0.017 0.021 0.036 -0.021 -0.006 0.027 0.022 0.028 -0.013 0.050 0.014 0.005 0.116 0.000 -0.013 -0.006 -0.038 -0.099 -0.031 0.008 0.030 0.002

AM 0.020 -0.012 0.061 0.016 0.011 0.024 -0.013 -0.007 0.027 0.031 0.022 -0.004 0.061 0.013 0.022 0.097 -0.006 0.021 -0.003 0.054 -0.029 -0.059 0.016 0.028 MAV 0.020 0.012 0.061 0.016 0.011 0.024 0.015 0.007 0.027 0.031 0.022 0.011 0.061 0.013 0.022 0.097 0.006 0.026 0.003 0.069 0.072 0.059 0.031 0.031 -

CompanyOwner Overconfidence 0.013 -0.013 0.051 0.014 0.026 0.024 -0.003 -0.001 0.020 0.018 0.012 0.013 0.053 0.003 0.011 0.104 -0.014 0.052 -0.004 0.101 0.111 -0.034 0.025 0.032 0.002 LossAversion 0.021 -0.017 0.055 0.022 0.015 0.028 0.008 -0.005 0.019 0.032 0.024 0.010 0.065 -0.002 0.016 0.086 -0.005 0.029 0.003 0.123 0.081 -0.044 0.026 0.032 0.002 Herding 0.006 -0.006 0.043 0.018 0.011 0.024 0.001 0.004 0.023 0.005 0.025 0.001 0.056 0.036 -0.002 0.112 -0.010 0.043 0.003 0.055 0.051 -0.007 0.022 0.025 0.001 Anchoring 0.013 -0.017 0.054 0.018 0.000 0.003 0.001 -0.009 0.024 0.024 0.025 -0.003 0.057 0.022 0.029 0.091 -0.001 0.043 0.000 0.145 0.104 -0.062 0.026 0.034 0.003 ConfirmationBias 0.006 -0.013 0.061 0.010 0.000 -0.012 -0.041 0.005 0.004 0.024 0.024 -0.002 0.044 0.000 -0.010 0.066 -0.012 0.012 -0.015 -0.013 0.010 -0.035 0.005 0.019 0.001

AM 0.012 -0.013 0.053 0.016 0.010 0.013 -0.007 -0.001 0.018 0.021 0.022 0.004 0.055 0.012 0.009 0.092 -0.008 0.036 -0.003 0.082 0.071 -0.036 0.021 0.027 MAV 0.012 0.013 0.053 0.016 0.010 0.018 0.011 0.005 0.018 0.021 0.022 0.006 0.055 0.013 0.014 0.092 0.008 0.036 0.005 0.087 0.071 0.036 0.028 0.028 -

Explicit Base 0.894 0.921 0.863 0.915 0.929 0.905 0.942 0.916 0.885 0.885 0.905 0.907 0.861 0.908 0.903 0.815 0.918 0.873 0.924 0.718 0.687 0.922 0.882 0.882 -

RetailInvestor Overconfidence 0.017 0.001 0.051 0.022 -0.022 -0.191 -0.008 0.002 0.009 0.023 0.014 0.007 0.033 0.010 0.011 0.085 -0.007 0.046 0.003 -0.001 0.042 -0.099 0.002 0.032 0.003 LossAversion 0.027 -0.006 0.052 0.018 -0.003 0.006 0.001 -0.005 0.030 0.020 0.015 -0.008 0.062 0.011 0.002 0.100 0.001 0.046 -0.001 -0.082 -0.078 -0.124 0.004 0.032 0.002 Herding 0.020 -0.014 0.045 0.014 0.018 0.038 0.014 -0.010 0.019 0.031 0.021 0.004 0.036 0.014 0.016 0.089 -0.007 0.012 -0.009 -0.078 0.080 -0.035 0.014 0.028 0.001 Anchoring 0.021 -0.005 0.053 0.009 -0.010 0.022 0.009 0.003 0.032 0.023 0.030 0.014 0.061 0.004 -0.015 0.085 -0.027 0.030 -0.009 0.062 -0.149 -0.102 0.006 0.035 0.003 ConfirmationBias 0.033 -0.018 0.056 -0.007 0.000 0.014 -0.011 0.002 0.014 0.031 0.020 0.015 0.060 0.010 0.002 0.099 0.000 0.024 -0.001 0.015 -0.077 -0.041 0.011 0.025 0.001

AM 0.024 -0.008 0.051 0.011 -0.003 -0.022 0.001 -0.002 0.021 0.026 0.020 0.006 0.050 0.010 0.003 0.092 -0.008 0.032 -0.003 -0.017 -0.036 -0.080 0.008 0.024 MAV 0.024 0.009 0.051 0.014 0.011 0.054 0.009 0.004 0.021 0.026 0.020 0.010 0.050 0.010 0.009 0.092 0.008 0.032 0.005 0.048 0.085 0.080 0.030 0.030 -

Professional Overconfidence 0.028 -0.009 0.045 0.004 0.002 0.023 -0.007 -0.013 0.028 0.026 0.003 -0.042 0.058 0.003 0.016 0.096 0.001 0.032 -0.009 0.103 -0.161 -0.087 0.006 0.036 0.003 LossAversion 0.014 -0.015 0.048 0.016 0.004 -0.007 -0.025 -0.005 0.008 0.035 0.029 -0.023 0.059 0.029 -0.006 0.079 0.012 -0.005 -0.001 -0.019 -0.095 -0.051 0.004 0.027 0.001 Herding 0.036 -0.013 0.061 0.017 0.006 0.003 -0.011 0.003 0.032 0.039 0.022 0.017 0.058 0.011 0.014 0.090 -0.024 0.033 0.000 0.078 0.024 -0.082 0.019 0.031 0.002 Anchoring 0.028 -0.014 0.066 0.010 0.013 0.019 0.000 -0.007 0.018 0.026 0.042 -0.021 0.056 0.028 0.019 0.082 0.000 -0.013 -0.016 0.098 -0.152 -0.066 0.010 0.036 0.003 ConfirmationBias 0.016 -0.010 0.054 0.025 0.006 0.002 -0.012 -0.003 0.027 0.024 0.023 -0.033 0.063 0.003 0.018 0.095 0.003 0.022 0.003 -0.009 -0.124 -0.143 0.002 0.033 0.003

AM 0.024 -0.012 0.055 0.014 0.006 0.008 -0.011 -0.005 0.023 0.030 0.024 -0.020 0.059 0.015 0.012 0.088 -0.002 0.014 -0.005 0.050 -0.102 -0.086 0.008 0.030 MAV 0.024 0.012 0.055 0.014 0.006 0.011 0.011 0.006 0.023 0.030 0.024 0.027 0.059 0.015 0.015 0.088 0.008 0.021 0.006 0.061 0.111 0.086 0.032 0.032 -

- CompanyOwner Overconfidence 0.009 -0.006 0.053 0.014 -0.002 0.022 0.004 -0.005 0.015 0.021 0.025 -0.006 0.035 0.017 0.019 0.085 -0.010 0.038 -0.005 0.140 0.006 -0.020 0.020 0.025 0.002 LossAversion 0.009 -0.013 0.047 0.010 0.006 0.006 -0.026 -0.001 0.028 0.023 0.024 -0.022 0.054 0.007 0.029 0.077 -0.005 0.043 0.000 0.139 0.082 -0.158 0.016 0.037 0.003 Herding 0.014 -0.013 0.053 0.014 0.010 -0.004 -0.016 0.006 0.039 0.035 0.020 0.008 0.059 0.036 -0.003 0.077 -0.010 0.034 0.007 0.026 0.052 -0.021 0.019 0.025 0.001 Anchoring 0.014 -0.013 0.048 0.006 0.008 0.024 -0.008 0.003 0.016 0.023 0.025 0.003 0.063 0.041 0.014 0.078 -0.005 0.048 -0.004 0.127 0.061 -0.044 0.024 0.031 0.002 ConfirmationBias -0.010 -0.006 0.054 0.005 0.000 0.010 -0.026 -0.011 0.005 0.041 0.023 -0.010 0.069 0.016 0.008 0.065 -0.016 0.020 -0.022 -0.003 0.011 -0.049 0.008 0.022 0.001 AM 0.007 -0.010 0.051 0.010 0.004 0.012 -0.014 -0.002 0.021 0.029 0.023 -0.005 0.056 0.023 0.013 0.076 -0.009 0.037 -0.005 0.086 0.042 -0.058 0.018 0.027 MAV 0.011 0.010 0.051 0.010 0.005 0.013 0.016 0.005 0.021 0.029 0.023 0.010 0.056 0.023 0.015 0.076 0.009 0.037 0.008 0.087 0.042 0.058 0.028 0.028 -

TRUE Category Implicit Base 0.381 0.412 0.298 0.488 0.667 0.533 0.711 0.432 0.364 0.364 0.680 0.651 0.520 0.410 0.549 0.494 0.276 0.541 0.513 0.222 0.194 0.485 0.463 0.463 -

RetailInvestor Overconfidence 0.063 -0.269 0.080 -0.044 -0.117 0.092 -0.077 -0.225 -0.287 -0.070 -0.042 -0.165 0.013 -0.067 0.009 0.019 -0.116 -0.041 -0.299 -0.145 -0.194 -0.411 -0.104 0.129 0.028 LossAversion -0.006 -0.099 0.188 0.012 0.027 0.148 -0.060 -0.165 -0.290 0.080 -0.080 -0.177 0.065 0.061 -0.027 -0.044 -0.045 -0.012 -0.246 -0.068 -0.120 -0.271 -0.051 0.104 0.018 Herding -0.301 -0.332 -0.227 -0.088 -0.310 0.017 -0.096 -0.432 -0.281 -0.226 -0.044 -0.287 -0.145 -0.410 -0.035 -0.082 -0.193 -0.184 -0.282 -0.222 -0.194 -0.405 -0.216 0.218 0.062 Anchoring -0.159 -0.412 -0.160 0.012 -0.255 0.089 -0.153 -0.432 -0.364 -0.221 0.031 -0.251 0.006 -0.327 0.051 -0.107 -0.193 -0.184 -0.282 -0.222 -0.194 -0.331 -0.184 0.202 0.055 ConfirmationBias 0.105 -0.198 0.243 -0.002 -0.029 0.074 -0.115 -0.020 -0.101 0.122 0.000 -0.273 0.025 0.064 -0.029 0.030 0.088 -0.089 -0.124 -0.074 0.064 -0.182 -0.019 0.093 0.014

AM -0.060 -0.262 0.025 -0.022 -0.137 0.084 -0.100 -0.255 -0.265 -0.063 -0.027 -0.231 -0.007 -0.136 -0.006 -0.037 -0.092 -0.102 -0.247 -0.146 -0.128 -0.320 -0.115 0.125 MAV 0.127 0.262 0.180 0.032 0.148 0.084 0.100 0.255 0.265 0.144 0.039 0.231 0.051 0.186 0.030 0.056 0.127 0.102 0.247 0.146 0.153 0.320 0.149 0.149 -

Professional Overconfidence -0.181 -0.269 0.055 0.053 -0.016 0.134 0.000 -0.289 -0.281 0.011 -0.075 -0.180 -0.007 -0.134 0.101 -0.039 -0.054 -0.154 -0.291 -0.142 -0.114 -0.331 -0.100 0.132 0.028 LossAversion 0.040 -0.218 0.252 -0.031 0.044 0.158 -0.089 -0.138 -0.226 0.014 -0.115 -0.127 0.085 -0.087 0.034 0.028 0.000 -0.208 -0.227 -0.222 -0.040 -0.331 -0.064 0.123 0.024 Herding -0.227 -0.264 0.231 0.068 -0.126 0.063 -0.261 -0.355 -0.281 0.080 -0.109 -0.151 -0.088 -0.134 0.046 -0.018 -0.193 -0.154 -0.282 -0.222 -0.194 -0.325 -0.132 0.176 0.040 Anchoring 0.090 -0.269 0.239 0.012 0.013 0.134 -0.077 -0.174 -0.216 0.194 -0.013 -0.101 0.099 0.047 -0.037 0.094 0.010 0.000 -0.190 -0.142 -0.051 -0.271 -0.028 0.112 0.020 ConfirmationBias 0.051 -0.145 0.146 0.053 0.083 0.187 -0.115 -0.129 -0.216 0.075 0.000 0.016 0.068 0.148 0.011 0.125 0.000 -0.027 -0.190 -0.142 -0.056 -0.337 -0.018 0.105 0.018

AM -0.045 -0.233 0.185 0.031 0.000 0.135 -0.108 -0.217 -0.244 0.075 -0.062 -0.109 0.031 -0.032 0.031 0.038 -0.047 -0.109 -0.236 -0.174 -0.091 -0.319 -0.068 0.116 MAV 0.118 0.233 0.185 0.043 0.056 0.135 0.108 0.217 0.244 0.075 0.062 0.115 0.069 0.110 0.046 0.061 0.051 0.109 0.236 0.174 0.091 0.319 0.130 0.130 -

CompanyOwner Overconfidence -0.181 -0.335 0.015 0.012 0.077 0.105 -0.060 -0.218 -0.364 -0.226 -0.115 -0.207 -0.032 -0.152 -0.037 -0.032 -0.196 -0.141 -0.291 -0.139 -0.117 -0.342 -0.135 0.154 0.035 LossAversion -0.105 -0.338 0.066 0.068 -0.035 0.147 0.016 -0.174 -0.290 0.036 -0.042 -0.072 0.004 -0.160 -0.025 -0.005 -0.276 -0.084 -0.217 -0.222 -0.051 -0.337 -0.095 0.126 0.027 Herding -0.310 -0.145 -0.004 0.026 -0.052 0.105 -0.060 -0.210 -0.287 -0.188 -0.058 -0.219 0.025 0.201 -0.071 0.006 -0.122 -0.103 -0.217 -0.222 -0.051 -0.199 -0.098 0.131 0.025 Anchoring -0.181 -0.338 0.102 -0.003 -0.196 0.009 -0.044 -0.232 -0.364 -0.021 -0.058 -0.138 -0.008 -0.010 0.015 -0.029 -0.193 -0.041 -0.282 -0.222 -0.194 -0.271 -0.123 0.134 0.031 ConfirmationBias -0.022 -0.269 0.215 -0.031 -0.029 0.012 -0.233 0.012 -0.142 0.158 -0.027 -0.177 -0.042 -0.032 -0.004 -0.063 -0.062 -0.166 -0.170 -0.148 0.073 -0.162 -0.060 0.102 0.017

AM -0.160 -0.285 0.079 0.014 -0.047 0.076 -0.076 -0.164 -0.289 -0.048 -0.060 -0.163 -0.011 -0.031 -0.024 -0.025 -0.170 -0.107 -0.235 -0.191 -0.068 -0.262 -0.102 0.117 MAV 0.160 0.285 0.080 0.028 0.078 0.076 0.083 0.169 0.289 0.126 0.060 0.163 0.022 0.111 0.030 0.027 0.170 0.107 0.235 0.191 0.097 0.262 0.130 0.130 -

Explicit Base 0.381 0.412 0.298 0.488 0.667 0.533 0.711 0.432 0.364 0.364 0.680 0.651 0.520 0.410 0.549 0.494 0.276 0.541 0.513 0.222 0.194 0.485 0.463 0.463 -

RetailInvestor Overconfidence -0.114 -0.079 0.015 0.068 -0.079 -0.173 -0.075 -0.109 -0.182 0.014 -0.038 -0.207 0.093 -0.087 -0.037 0.006 -0.122 -0.041 -0.217 -0.008 -0.123 -0.352 -0.084 0.102 0.017 LossAversion 0.063 -0.145 0.190 0.026 -0.076 0.071 -0.029 -0.174 -0.097 -0.031 -0.053 -0.151 0.107 -0.134 -0.038 -0.006 -0.054 -0.070 -0.227 -0.068 0.020 -0.285 -0.053 0.096 0.014 Herding -0.068 -0.212 0.112 -0.017 0.044 0.134 0.007 -0.182 -0.290 0.098 -0.089 -0.251 -0.091 -0.077 -0.049 0.037 -0.193 -0.255 -0.246 -0.079 -0.114 -0.331 -0.096 0.135 0.027 Anchoring -0.114 -0.258 0.134 0.025 -0.143 0.038 -0.028 -0.156 -0.281 0.014 -0.061 0.016 0.025 -0.262 -0.123 0.006 -0.193 -0.129 -0.299 -0.142 -0.114 -0.199 -0.102 0.125 0.024 ConfirmationBias -0.036 -0.218 0.202 -0.078 -0.042 0.134 -0.106 -0.068 -0.231 0.068 -0.055 -0.072 0.025 -0.046 -0.038 0.039 0.081 -0.103 -0.227 -0.148 -0.123 -0.263 -0.059 0.109 0.017

AM -0.054 -0.182 0.131 0.005 -0.059 0.041 -0.046 -0.138 -0.216 0.033 -0.059 -0.133 0.032 -0.121 -0.057 0.016 -0.096 -0.120 -0.243 -0.089 -0.091 -0.286 -0.079 0.102 MAV 0.079 0.182 0.131 0.043 0.077 0.110 0.049 0.138 0.216 0.045 0.059 0.139 0.068 0.121 0.057 0.019 0.129 0.120 0.243 0.089 0.099 0.286 0.114 0.114 -

Professional Overconfidence -0.006 -0.264 0.080 0.012 -0.082 0.146 -0.126 -0.238 -0.284 0.086 -0.138 -0.138 0.004 -0.203 -0.049 -0.018 -0.122 -0.041 -0.299 -0.148 -0.046 -0.347 -0.101 0.131 0.027 LossAversion -0.003 -0.162 0.178 0.117 0.000 0.005 -0.128 -0.174 -0.135 0.110 -0.028 -0.151 0.107 0.104 0.007 0.034 0.176 0.015 -0.227 -0.084 -0.056 -0.152 -0.020 0.098 0.014 Herding 0.019 -0.335 0.188 0.053 -0.067 0.009 -0.147 -0.156 -0.281 0.122 -0.109 -0.151 -0.032 -0.134 0.016 0.017 -0.045 -0.103 -0.282 -0.148 -0.194 -0.337 -0.095 0.134 0.028 Anchoring -0.006 -0.212 0.202 -0.017 0.041 0.121 -0.015 -0.089 -0.226 0.086 0.031 -0.230 0.013 0.169 0.009 0.088 0.124 -0.027 -0.370 -0.068 0.028 -0.411 -0.035 0.117 0.027 ConfirmationBias 0.119 -0.154 0.102 0.107 -0.067 0.044 -0.075 -0.079 -0.216 -0.021 -0.013 -0.230 0.105 -0.152 0.063 0.006 0.099 0.004 -0.217 -0.068 -0.046 -0.297 -0.045 0.104 0.017

AM 0.025 -0.225 0.150 0.054 -0.035 0.065 -0.098 -0.147 -0.228 0.077 -0.051 -0.180 0.039 -0.043 0.009 0.025 0.046 -0.030 -0.279 -0.103 -0.063 -0.309 -0.059 0.104 MAV 0.031 0.225 0.150 0.061 0.051 0.065 0.098 0.147 0.228 0.085 0.064 0.180 0.052 0.152 0.029 0.033 0.113 0.038 0.279 0.103 0.074 0.309 0.117 0.117 -

CompanyOwner Overconfidence -0.187 -0.198 0.134 -0.017 -0.130 0.158 0.012 -0.174 -0.293 -0.070 -0.058 -0.017 -0.042 0.014 0.009 0.006 -0.128 -0.089 -0.237 -0.068 -0.194 -0.199 -0.080 0.111 0.019 LossAversion -0.243 -0.335 0.005 -0.064 -0.048 0.055 -0.111 -0.218 -0.284 0.014 -0.027 -0.177 -0.020 -0.143 0.015 -0.103 -0.276 -0.117 -0.282 -0.222 -0.056 -0.342 -0.135 0.144 0.033 Herding -0.238 -0.269 0.134 -0.017 -0.033 0.049 -0.102 -0.099 -0.133 0.110 -0.055 -0.138 0.058 0.201 -0.049 -0.068 -0.196 -0.196 -0.156 -0.145 -0.056 -0.152 -0.070 0.121 0.019 Anchoring -0.238 -0.269 -0.040 -0.044 -0.138 0.120 -0.075 -0.156 -0.364 0.014 -0.058 -0.125 -0.007 0.196 -0.149 -0.075 -0.196 -0.218 -0.353 -0.222 -0.117 -0.271 -0.127 0.157 0.035 ConfirmationBias -0.165 -0.145 0.267 -0.014 -0.029 0.082 -0.111 -0.099 -0.188 0.227 -0.013 -0.151 0.071 0.103 0.055 -0.013 -0.062 -0.166 -0.271 -0.222 0.064 -0.152 -0.042 0.121 0.021

AM -0.214 -0.243 0.100 -0.031 -0.076 0.093 -0.077 -0.149 -0.252 0.059 -0.042 -0.122 0.012 0.074 -0.024 -0.051 -0.172 -0.157 -0.260 -0.176 -0.072 -0.223 -0.091 0.122 MAV 0.214 0.243 0.116 0.031 0.076 0.093 0.082 0.149 0.252 0.087 0.042 0.122 0.040 0.131 0.055 0.053 0.172 0.157 0.260 0.176 0.097 0.223 0.131 0.131 -

Table 8: Results on MFMD-persona: Chinese part.

Mistral-Small-24B

Mistral-NEMO

Mixtral-8x22B

GPT-5-mini-R

Gemini-2.5-R

Mistral-Large

Mixtral-8x7B

Claude-4.5-R

DeepSeek-R

DeepSeek-C

Qwen14b-R

Qwen32b-R

Gemini-2.0

Claude-3.5

Mistral-7B

Qwen8b-R

Llama70b

Qwen32b

Qwen14b

Qwen72b

Qwen8b

GPT-4.1

Role Scenario

AM MAV MSE FALSE Category

Implicit Base 0.904 0.905 0.902 0.938 0.929 0.927 0.934 0.906 0.910 0.911 0.933 0.904 0.857 0.933 0.911 0.896 0.892 0.839 0.927 0.742 0.598 0.902 0.886 0.886 -

RetailInvestor Overconfidence 0.010 0.010 0.028 -0.010 0.010 -0.016 0.012 0.009 -0.001 0.019 0.009 -0.028 0.069 -0.011 0.009 0.018 -0.206 -0.039 -0.015 0.026 0.072 -0.067 -0.004 0.031 0.003 LossAversion -0.001 0.011 0.006 0.006 -0.008 -0.017 -0.014 0.005 0.003 0.005 0.017 -0.106 0.070 0.008 0.004 0.008 -0.147 -0.021 -0.019 0.073 -0.020 -0.066 -0.009 0.029 0.002 Herding 0.001 0.008 0.017 -0.001 -0.005 -0.004 0.001 0.007 -0.001 0.004 0.014 -0.040 0.077 -0.010 0.018 0.013 -0.191 0.028 -0.014 0.101 0.225 -0.095 0.007 0.040 0.005 Anchoring 0.004 0.004 0.007 -0.003 0.004 -0.010 -0.005 0.003 0.003 -0.002 0.009 -0.157 0.063 -0.009 0.017 0.037 -0.147 0.025 -0.014 0.006 0.035 -0.136 -0.012 0.032 0.003 ConfirmationBias 0.012 -0.006 0.014 -0.008 -0.007 -0.035 -0.007 0.013 0.003 0.009 0.013 -0.044 0.026 -0.008 0.019 0.007 -0.115 -0.048 -0.023 -0.022 0.065 -0.079 -0.010 0.027 0.001

AM 0.005 0.006 0.014 -0.003 -0.001 -0.016 -0.003 0.007 0.001 0.007 0.012 -0.075 0.061 -0.006 0.013 0.017 -0.161 -0.011 -0.017 0.037 0.075 -0.089 -0.006 0.029 MAV 0.006 0.008 0.014 0.006 0.007 0.016 0.008 0.007 0.002 0.008 0.012 0.075 0.061 0.009 0.013 0.017 0.161 0.032 0.017 0.046 0.083 0.089 0.032 0.032 -

Professional Overconfidence 0.003 0.004 0.023 -0.032 0.004 -0.020 -0.019 0.007 -0.001 0.007 0.014 -0.071 0.074 -0.006 0.009 0.028 -0.141 0.022 -0.019 0.077 0.131 -0.046 0.002 0.034 0.003 LossAversion 0.004 0.006 0.031 -0.006 0.018 -0.008 -0.005 0.012 -0.002 0.014 0.005 -0.073 0.085 -0.007 0.020 0.042 -0.158 -0.075 -0.014 0.049 0.168 -0.047 0.003 0.039 0.004 Herding 0.023 0.004 0.021 -0.005 0.016 -0.018 -0.004 0.003 -0.001 0.005 0.018 -0.042 0.074 -0.006 0.029 0.020 -0.137 -0.003 -0.014 0.105 0.204 -0.084 0.009 0.038 0.004 Anchoring -0.008 0.010 0.027 0.001 -0.009 -0.008 0.002 0.013 0.002 0.021 0.017 -0.117 0.080 0.004 0.013 0.016 -0.055 -0.054 -0.015 0.097 -0.101 -0.098 -0.007 0.035 0.003 ConfirmationBias 0.009 0.003 0.018 0.001 -0.003 -0.013 -0.006 0.014 0.003 0.022 -0.004 -0.072 0.076 0.003 0.006 0.001 -0.083 -0.107 -0.014 -0.075 0.069 -0.052 -0.009 0.030 0.002

AM 0.006 0.005 0.024 -0.008 0.005 -0.013 -0.006 0.010 0.000 0.014 0.010 -0.075 0.078 -0.002 0.015 0.021 -0.115 -0.043 -0.015 0.051 0.094 -0.066 -0.001 0.031 MAV 0.009 0.005 0.024 0.009 0.010 0.013 0.007 0.010 0.002 0.014 0.011 0.075 0.078 0.005 0.015 0.021 0.115 0.052 0.015 0.081 0.134 0.066 0.035 0.035 -

- CompanyOwner Overconfidence 0.010 0.015 0.013 -0.002 -0.002 -0.017 -0.012 0.010 -0.005 0.008 0.017 -0.044 0.061 -0.015 -0.004 0.002 -0.089 -0.054 -0.014 0.079 0.091 -0.044 0.000 0.028 0.002 LossAversion 0.011 0.003 0.007 0.006 -0.008 -0.016 -0.023 0.009 -0.001 0.011 0.005 -0.124 0.069 -0.003 -0.005 0.012 -0.078 -0.063 -0.019 0.085 0.154 -0.021 0.001 0.033 0.003 Herding 0.003 0.000 0.015 -0.002 0.031 -0.012 -0.007 0.010 -0.002 0.010 0.017 -0.047 0.063 -0.006 0.012 -0.012 -0.133 -0.020 -0.010 0.062 0.164 -0.074 0.003 0.032 0.003 Anchoring 0.003 0.011 0.028 -0.015 0.001 -0.015 0.007 0.009 -0.001 0.007 0.013 -0.091 0.051 -0.009 0.017 0.019 -0.029 -0.043 -0.014 0.119 0.178 -0.095 0.007 0.035 0.003 ConfirmationBias -0.031 -0.006 -0.009 -0.003 0.005 -0.005 -0.021 0.004 0.002 -0.008 0.005 -0.076 0.067 -0.003 0.011 -0.010 -0.013 -0.067 -0.015 -0.009 0.019 -0.098 -0.012 0.022 0.001 AM -0.001 0.004 0.011 -0.003 0.005 -0.013 -0.011 0.009 -0.001 0.006 0.012 -0.076 0.062 -0.007 0.006 0.002 -0.068 -0.049 -0.015 0.067 0.121 -0.066 0.000 0.028 -

- MAV 0.011 0.007 0.014 0.005 0.010 0.013 0.014 0.009 0.002 0.009 0.012 0.076 0.062 0.007 0.010 0.011 0.068 0.049 0.015 0.071 0.121 0.066 0.030 0.030 -

Explicit Base 0.904 0.905 0.902 0.938 0.929 0.927 0.934 0.906 0.910 0.911 0.933 0.904 0.857 0.933 0.911 0.896 0.892 0.839 0.927 0.742 0.598 0.902 0.886 0.886 -

RetailInvestor Overconfidence 0.009 -0.005 0.025 -0.012 -0.013 -0.142 -0.010 0.005 0.003 0.017 -0.005 -0.026 0.037 -0.012 0.012 0.012 -0.219 -0.059 -0.019 -0.001 0.099 -0.104 -0.019 0.038 0.004 LossAversion 0.002 0.002 0.009 -0.007 0.006 -0.002 -0.012 0.023 -0.002 0.010 -0.009 -0.078 0.079 0.004 0.016 0.022 -0.121 -0.039 -0.019 -0.040 0.028 -0.069 -0.009 0.027 0.002 Herding 0.019 0.003 0.022 0.010 0.010 -0.009 0.004 0.014 -0.001 0.018 0.005 -0.052 0.069 0.001 0.026 -0.005 -0.175 -0.030 -0.014 -0.020 0.202 -0.066 0.001 0.035 0.004 Anchoring 0.014 0.008 0.016 -0.010 -0.002 0.006 -0.001 0.003 0.003 0.000 0.009 -0.053 0.045 0.001 -0.003 0.020 -0.134 -0.014 -0.014 0.038 0.030 -0.091 -0.006 0.023 0.002 ConfirmationBias 0.013 -0.002 0.019 -0.007 -0.012 -0.020 0.007 0.010 0.003 0.014 0.012 0.007 0.076 -0.003 0.003 0.006 -0.127 -0.013 -0.014 0.060 0.037 -0.054 0.001 0.024 0.001 AM 0.012 0.001 0.018 -0.005 -0.002 -0.033 -0.002 0.011 0.001 0.012 0.003 -0.040 0.061 -0.002 0.011 0.011 -0.155 -0.031 -0.016 0.007 0.079 -0.077 -0.006 0.027 -

- MAV 0.012 0.004 0.018 0.009 0.009 0.036 0.007 0.011 0.002 0.012 0.008 0.043 0.061 0.004 0.012 0.013 0.155 0.031 0.016 0.032 0.079 0.077 0.030 0.030 -

Professional Overconfidence 0.009 0.011 0.007 -0.003 0.010 0.004 -0.006 0.016 -0.001 0.013 0.001 -0.126 0.093 -0.002 0.021 0.035 -0.141 -0.084 -0.019 0.006 0.044 -0.055 -0.008 0.032 0.003 LossAversion 0.014 0.005 -0.002 -0.016 -0.005 0.000 0.002 0.007 0.006 0.014 0.005 -0.055 0.072 0.015 0.011 -0.008 -0.050 -0.166 -0.014 -0.021 -0.023 -0.088 -0.013 0.027 0.002 Herding 0.010 0.010 0.016 -0.014 0.012 -0.008 -0.028 0.010 -0.001 0.018 0.009 -0.052 0.082 -0.009 0.025 0.026 -0.143 -0.054 -0.014 0.051 0.141 -0.081 0.000 0.037 0.003 Anchoring 0.009 0.022 0.030 0.006 0.000 0.005 -0.014 0.013 0.005 0.013 0.025 -0.076 0.089 -0.011 0.033 0.001 -0.032 -0.083 -0.014 -0.004 0.010 -0.038 -0.001 0.024 0.001 ConfirmationBias 0.021 -0.001 0.023 -0.003 -0.008 -0.019 -0.034 0.028 0.003 0.003 0.017 -0.083 0.071 0.001 0.006 0.014 -0.075 -0.111 -0.014 -0.072 0.036 -0.064 -0.012 0.032 0.002

AM 0.012 0.009 0.015 -0.006 0.002 -0.004 -0.016 0.015 0.002 0.012 0.011 -0.078 0.081 -0.001 0.019 0.013 -0.088 -0.099 -0.015 -0.008 0.042 -0.065 -0.007 0.028 MAV 0.012 0.010 0.016 0.008 0.007 0.007 0.017 0.015 0.003 0.012 0.011 0.078 0.081 0.008 0.019 0.017 0.088 0.099 0.015 0.031 0.051 0.065 0.031 0.031 -

CompanyOwner Overconfidence -0.003 0.007 0.006 -0.011 0.014 -0.016 -0.002 0.013 -0.002 -0.005 0.009 -0.039 0.060 0.000 0.004 0.015 -0.156 -0.121 -0.010 0.115 0.075 -0.046 -0.004 0.033 0.003 LossAversion 0.008 0.003 0.016 0.006 0.008 -0.017 -0.015 0.002 -0.001 0.003 0.001 -0.098 0.070 -0.021 0.000 0.003 -0.041 -0.048 -0.014 0.115 0.107 -0.075 0.001 0.031 0.002 Herding 0.011 0.003 0.022 0.002 0.022 -0.017 -0.006 0.006 0.003 -0.005 -0.001 -0.059 0.051 -0.013 0.000 0.022 -0.119 -0.048 -0.010 -0.034 0.163 -0.081 -0.004 0.032 0.003 Anchoring 0.006 0.003 0.018 0.002 0.000 -0.016 0.003 0.010 0.003 -0.003 0.009 -0.061 0.070 0.009 0.016 0.017 -0.068 -0.061 -0.010 0.112 0.150 -0.066 0.006 0.032 0.003 ConfirmationBias 0.007 0.014 -0.009 0.002 0.005 -0.031 -0.030 -0.008 0.002 -0.023 -0.005 -0.026 0.067 -0.008 0.001 -0.005 -0.019 -0.078 -0.019 -0.042 0.079 -0.082 -0.010 0.026 0.001

AM 0.006 0.006 0.011 0.000 0.010 -0.019 -0.010 0.005 0.001 -0.007 0.003 -0.057 0.064 -0.007 0.004 0.010 -0.081 -0.071 -0.013 0.053 0.115 -0.070 -0.002 0.028 MAV 0.007 0.006 0.014 0.005 0.010 0.019 0.011 0.008 0.002 0.008 0.005 0.057 0.064 0.010 0.004 0.012 0.081 0.071 0.013 0.084 0.115 0.070 0.031 0.031 -

TRUE Category Implicit Base 0.378 0.333 0.429 0.667 0.653 0.667 0.652 0.250 0.303 0.476 0.680 0.564 0.609 0.514 0.476 0.706 0.214 0.512 0.345 0.074 0.267 0.387 0.462 0.462 -

RetailInvestor Overconfidence -0.066 -0.119 0.009 -0.141 -0.002 -0.079 0.071 0.026 -0.303 -0.039 0.028 -0.008 -0.031 -0.181 -0.002 -0.244 -0.214 -0.012 -0.197 -0.074 -0.267 -0.233 -0.094 0.107 0.021 LossAversion -0.184 -0.179 -0.019 -0.017 -0.057 -0.049 -0.025 -0.043 -0.223 -0.014 0.080 -0.120 -0.177 0.031 0.012 -0.161 -0.131 -0.055 -0.268 -0.074 -0.073 -0.239 -0.090 0.101 0.017 Herding -0.378 -0.333 -0.153 -0.111 -0.422 -0.130 -0.033 -0.170 -0.303 -0.209 0.018 -0.079 -0.313 -0.228 -0.005 -0.342 -0.214 -0.290 -0.265 0.009 -0.267 -0.387 -0.209 0.212 0.061 Anchoring -0.230 -0.333 -0.429 -0.067 -0.139 -0.102 -0.014 -0.250 -0.303 -0.476 0.028 -0.164 -0.165 -0.283 0.050 -0.221 -0.140 -0.226 -0.265 -0.074 -0.184 -0.304 -0.195 0.202 0.056 ConfirmationBias 0.054 -0.200 0.003 -0.045 -0.048 -0.106 0.039 0.026 -0.223 -0.002 0.055 -0.008 -0.083 -0.057 0.146 -0.141 -0.020 -0.075 -0.271 0.193 -0.060 0.065 -0.034 0.087 0.013

AM -0.161 -0.233 -0.118 -0.076 -0.134 -0.093 0.008 -0.082 -0.271 -0.148 0.042 -0.076 -0.154 -0.144 0.040 -0.222 -0.144 -0.131 -0.253 -0.004 -0.170 -0.220 -0.125 0.133 MAV 0.182 0.233 0.123 0.076 0.134 0.093 0.036 0.103 0.271 0.148 0.042 0.076 0.154 0.156 0.043 0.222 0.144 0.131 0.253 0.085 0.170 0.245 0.142 0.142 -

Professional Overconfidence -0.178 -0.333 0.028 -0.202 0.028 -0.079 -0.022 -0.170 -0.303 -0.112 0.031 -0.064 -0.096 -0.169 -0.002 -0.165 -0.214 -0.027 -0.268 -0.074 -0.187 -0.304 -0.131 0.139 0.029 LossAversion 0.000 -0.126 0.085 -0.103 0.061 -0.025 0.001 0.114 -0.160 -0.019 0.001 -0.093 0.087 -0.127 0.129 -0.072 -0.137 0.099 -0.265 0.009 0.009 -0.239 -0.035 0.089 0.013 Herding -0.092 -0.333 -0.143 -0.153 -0.065 -0.145 -0.030 -0.250 -0.303 -0.322 0.034 -0.112 0.006 -0.218 0.119 -0.235 -0.214 -0.045 -0.265 -0.074 -0.267 -0.387 -0.159 0.173 0.042 Anchoring -0.062 -0.119 0.042 -0.033 -0.026 -0.025 0.065 0.026 -0.155 0.065 0.070 -0.127 0.085 0.015 0.037 -0.142 -0.137 0.059 -0.197 0.009 -0.060 -0.156 -0.035 0.078 0.009 ConfirmationBias 0.011 -0.185 0.045 -0.033 -0.062 -0.024 0.015 -0.028 -0.223 0.038 -0.042 0.068 0.043 0.065 0.089 -0.141 -0.071 0.059 -0.265 0.006 -0.025 -0.101 -0.035 0.074 0.010

AM -0.064 -0.219 0.011 -0.105 -0.013 -0.060 0.006 -0.062 -0.229 -0.070 0.019 -0.066 0.025 -0.087 0.074 -0.151 -0.155 0.029 -0.252 -0.025 -0.106 -0.237 -0.079 0.094 MAV 0.069 0.219 0.069 0.105 0.048 0.060 0.027 0.117 0.229 0.111 0.036 0.093 0.063 0.119 0.075 0.151 0.155 0.058 0.252 0.035 0.110 0.237 0.111 0.111 -

CompanyOwner Overconfidence -0.066 -0.111 -0.162 -0.088 -0.082 -0.049 -0.074 -0.096 -0.303 -0.200 0.059 -0.050 -0.085 -0.191 -0.037 -0.232 -0.137 -0.027 -0.265 0.006 -0.267 -0.307 -0.126 0.132 0.026 LossAversion -0.102 -0.190 -0.086 0.000 -0.209 -0.079 -0.064 -0.036 -0.303 -0.101 -0.013 -0.064 -0.038 -0.114 -0.011 -0.182 -0.214 0.044 -0.268 -0.074 0.078 -0.233 -0.103 0.114 0.020 Herding -0.171 -0.333 -0.029 -0.088 0.104 -0.040 0.039 -0.096 -0.226 -0.064 0.059 -0.079 0.000 -0.218 0.061 -0.244 -0.071 -0.012 -0.262 0.009 0.019 -0.227 -0.085 0.111 0.022 Anchoring -0.171 -0.179 0.009 -0.130 -0.253 -0.096 0.068 -0.036 -0.303 -0.153 0.043 -0.051 -0.097 -0.283 0.050 -0.232 -0.134 0.033 -0.265 0.009 -0.119 -0.156 -0.111 0.131 0.025 ConfirmationBias -0.092 -0.200 -0.020 -0.067 -0.017 0.012 0.003 0.053 -0.155 -0.076 0.001 -0.064 0.000 -0.077 0.102 -0.196 -0.071 -0.137 -0.197 0.009 -0.134 -0.129 -0.066 0.082 0.011

AM -0.120 -0.203 -0.058 -0.075 -0.091 -0.050 -0.006 -0.042 -0.258 -0.119 0.030 -0.062 -0.044 -0.177 0.033 -0.217 -0.125 -0.020 -0.251 -0.008 -0.085 -0.210 -0.098 0.104 MAV 0.120 0.203 0.061 0.075 0.133 0.055 0.050 0.063 0.258 0.119 0.035 0.062 0.044 0.177 0.052 0.217 0.125 0.051 0.251 0.022 0.123 0.210 0.114 0.114 -

Explicit Base 0.378 0.333 0.429 0.667 0.653 0.667 0.652 0.250 0.303 0.476 0.680 0.564 0.609 0.514 0.476 0.706 0.214 0.512 0.345 0.074 0.267 0.387 0.462 0.462 -

RetailInvestor Overconfidence -0.014 -0.262 0.121 -0.058 -0.053 -0.237 0.002 0.008 -0.223 0.050 -0.013 -0.164 0.004 -0.102 0.061 -0.193 -0.131 0.033 -0.268 -0.074 0.000 -0.165 -0.076 0.102 0.018 LossAversion -0.120 -0.133 0.021 -0.062 -0.034 -0.042 0.027 0.221 -0.226 -0.032 -0.026 -0.035 0.085 0.015 0.074 -0.142 -0.054 0.017 -0.268 -0.074 -0.190 -0.307 -0.058 0.100 0.018 Herding -0.092 -0.256 0.057 -0.018 -0.019 -0.012 0.015 -0.090 -0.303 0.024 -0.013 -0.008 -0.072 -0.100 0.080 -0.230 -0.134 -0.088 -0.265 0.080 -0.119 -0.227 -0.081 0.105 0.019 Anchoring -0.055 -0.253 -0.065 -0.141 -0.082 0.000 0.028 -0.250 -0.223 -0.218 0.016 -0.127 -0.147 -0.062 -0.098 -0.055 -0.214 -0.060 -0.265 -0.074 -0.060 -0.165 -0.117 0.121 0.021 ConfirmationBias 0.022 -0.195 -0.017 -0.082 -0.088 -0.079 0.068 -0.096 -0.223 -0.019 0.065 -0.112 0.000 -0.114 0.036 -0.039 0.008 0.117 -0.265 0.080 -0.129 -0.042 -0.050 0.086 0.012

AM -0.052 -0.220 0.023 -0.072 -0.055 -0.074 0.028 -0.042 -0.240 -0.039 0.006 -0.089 -0.026 -0.073 0.030 -0.132 -0.105 0.004 -0.266 -0.012 -0.100 -0.181 -0.077 0.085 MAV 0.061 0.220 0.056 0.072 0.055 0.074 0.028 0.133 0.240 0.069 0.027 0.089 0.062 0.079 0.069 0.132 0.108 0.063 0.266 0.076 0.100 0.181 0.103 0.103 -

Professional Overconfidence 0.011 -0.179 -0.086 -0.067 0.014 0.047 0.027 0.083 -0.303 0.010 -0.028 -0.050 0.073 -0.157 0.088 -0.072 -0.214 0.059 -0.268 -0.074 -0.113 -0.310 -0.069 0.106 0.019 LossAversion -0.014 -0.030 -0.070 -0.109 -0.013 0.024 0.075 0.103 -0.149 -0.052 -0.013 -0.008 0.029 0.153 0.102 -0.153 -0.214 -0.012 -0.265 0.009 0.066 -0.111 -0.030 0.081 0.011 Herding -0.120 -0.119 -0.065 -0.154 -0.082 -0.025 -0.059 -0.096 -0.303 -0.005 0.016 -0.035 -0.045 -0.283 0.103 -0.150 -0.214 -0.112 -0.265 0.009 -0.267 -0.304 -0.117 0.129 0.026 Anchoring 0.011 0.012 0.112 -0.035 -0.015 0.037 -0.025 0.026 -0.089 0.037 0.120 -0.008 0.130 -0.181 0.174 -0.097 -0.060 0.044 -0.265 0.086 -0.119 -0.233 -0.015 0.087 0.013 ConfirmationBias 0.079 -0.259 -0.005 -0.067 -0.057 -0.023 -0.056 0.164 -0.223 -0.164 0.080 -0.093 0.058 -0.100 0.089 -0.053 -0.066 0.002 -0.265 0.009 -0.190 -0.091 -0.056 0.100 0.016

AM -0.007 -0.115 -0.023 -0.086 -0.031 0.012 -0.007 0.056 -0.213 -0.035 0.035 -0.039 0.049 -0.114 0.111 -0.105 -0.154 -0.004 -0.266 0.008 -0.125 -0.210 -0.057 0.082 MAV 0.047 0.120 0.068 0.086 0.036 0.031 0.048 0.094 0.213 0.054 0.051 0.039 0.067 0.175 0.111 0.105 0.154 0.046 0.266 0.038 0.151 0.210 0.100 0.100 -

- CompanyOwner Overconfidence -0.092 -0.185 -0.051 -0.096 0.014 -0.079 0.052 0.026 -0.226 -0.182 0.016 -0.102 -0.031 0.000 0.012 -0.153 -0.081 -0.027 -0.262 -0.074 0.066 -0.101 -0.071 0.088 0.013 LossAversion -0.230 -0.256 -0.106 -0.056 -0.124 -0.049 -0.010 -0.173 -0.303 -0.164 -0.044 -0.093 -0.072 -0.366 0.000 -0.169 -0.214 -0.041 -0.265 -0.074 -0.190 -0.239 -0.147 0.147 0.032 Herding -0.111 -0.190 0.057 -0.052 0.061 -0.049 0.015 -0.102 -0.223 -0.226 0.012 -0.064 -0.120 -0.292 0.000 -0.091 -0.071 0.076 -0.262 0.006 -0.053 -0.156 -0.083 0.104 0.018 Anchoring -0.075 -0.256 0.045 -0.072 -0.182 -0.079 0.042 -0.096 -0.303 -0.098 0.016 -0.107 -0.009 0.002 0.074 -0.121 -0.137 -0.112 -0.262 -0.074 -0.119 -0.173 -0.095 0.111 0.019 ConfirmationBias 0.098 -0.057 0.006 -0.052 -0.001 -0.081 -0.019 -0.015 -0.155 -0.050 -0.001 -0.093 0.016 -0.090 0.095 -0.216 -0.071 -0.041 -0.268 0.003 0.019 0.037 -0.043 0.067 0.009 AM -0.082 -0.189 -0.010 -0.065 -0.046 -0.067 0.016 -0.072 -0.242 -0.144 0.000 -0.092 -0.043 -0.149 0.036 -0.150 -0.115 -0.029 -0.264 -0.043 -0.055 -0.126 -0.088 0.093 MAV 0.121 0.189 0.053 0.065 0.076 0.067 0.027 0.082 0.242 0.144 0.018 0.092 0.050 0.150 0.036 0.150 0.115 0.060 0.264 0.046 0.089 0.141 0.104 0.104 -

###### Table 9: Results on MFMD-persona: Greek part.

Mistral-Small-24B

Mistral-NEMO

Mixtral-8x22B

GPT-5-mini-R

Gemini-2.5-R

Mistral-Large

Mixtral-8x7B

Claude-4.5-R

DeepSeek-R

DeepSeek-C

Qwen14b-R

Qwen32b-R

Gemini-2.0

Claude-3.5

Mistral-7B

Qwen8b-R

Llama70b

Qwen14b

Qwen32b

Qwen72b

Qwen8b

GPT-4.1

Role Scenario

AM MAV MSE FALSE Category

Implicit Base 0.869 0.907 0.914 0.935 0.915 0.909 0.941 0.879 0.877 0.910 0.929 0.864 0.845 0.911 0.911 0.446 0.887 0.813 0.917 0.738 0.670 0.825 0.855 0.855 -

RetailInvestor Overconfidence 0.045 0.009 0.007 -0.008 0.004 0.012 -0.021 0.036 0.032 0.012 0.014 -0.060 0.072 0.016 0.009 0.015 -0.166 0.025 -0.004 0.055 -0.013 -0.042 0.002 0.031 0.002 LossAversion 0.021 0.009 -0.034 0.000 0.019 0.028 -0.004 0.015 0.039 -0.009 0.017 -0.190 0.080 0.019 0.019 0.000 -0.078 0.028 0.000 0.087 0.043 -0.142 -0.002 0.040 0.004 Herding 0.044 0.002 -0.014 -0.024 -0.007 0.002 -0.015 0.034 0.032 -0.009 0.002 -0.142 0.077 0.012 0.014 0.054 -0.167 0.020 -0.004 0.079 0.166 -0.119 0.002 0.047 0.005 Anchoring 0.047 0.006 0.004 -0.012 0.012 -0.006 -0.020 0.034 0.036 0.010 0.002 -0.377 0.057 0.013 0.005 0.165 -0.075 0.045 -0.004 0.075 0.030 -0.162 -0.005 0.054 0.010 ConfirmationBias 0.044 0.000 0.000 -0.016 0.012 -0.003 -0.025 0.035 0.043 -0.007 0.008 -0.154 0.062 0.021 0.023 -0.009 -0.085 -0.092 -0.013 -0.089 -0.138 -0.073 -0.021 0.043 0.004

AM 0.040 0.005 -0.007 -0.012 0.008 0.006 -0.017 0.031 0.036 -0.001 0.008 -0.185 0.069 0.016 0.014 0.045 -0.115 0.005 -0.005 0.041 0.017 -0.107 -0.005 0.036 MAV 0.040 0.005 0.012 0.012 0.011 0.010 0.017 0.031 0.036 0.009 0.008 0.185 0.069 0.016 0.014 0.048 0.115 0.042 0.005 0.077 0.078 0.107 0.043 0.043 -

Professional Overconfidence 0.043 0.009 0.004 -0.003 0.002 -0.001 -0.007 0.045 0.032 0.005 0.014 -0.246 0.073 0.020 0.016 0.039 -0.145 0.033 -0.004 0.108 0.094 -0.165 -0.002 0.050 0.007 LossAversion 0.040 -0.004 0.019 0.001 0.000 0.024 -0.034 0.028 0.043 0.011 0.018 -0.208 0.071 0.022 0.018 -0.046 -0.130 -0.164 -0.004 0.083 -0.071 -0.067 -0.016 0.050 0.005 Herding 0.035 0.017 0.008 -0.010 0.019 0.000 -0.017 0.038 0.036 0.007 -0.002 -0.140 0.079 0.013 0.008 0.026 -0.126 -0.052 -0.004 0.131 0.114 -0.234 -0.002 0.051 0.006 Anchoring 0.031 0.005 0.003 -0.003 0.002 0.013 -0.009 0.020 0.039 -0.002 0.012 -0.247 0.088 0.014 0.008 -0.102 -0.037 -0.079 -0.009 0.078 -0.080 -0.158 -0.019 0.047 0.006 ConfirmationBias 0.044 0.009 0.000 -0.019 0.028 0.009 -0.035 0.055 0.043 0.003 0.005 -0.180 0.088 0.013 0.002 -0.006 -0.070 -0.170 -0.009 0.002 -0.138 -0.009 -0.015 0.043 0.005

AM 0.039 0.007 0.007 -0.007 0.010 0.009 -0.020 0.037 0.038 0.005 0.009 -0.204 0.080 0.016 0.010 -0.018 -0.102 -0.086 -0.006 0.081 -0.016 -0.126 -0.011 0.042 MAV 0.039 0.009 0.007 0.007 0.010 0.009 0.020 0.037 0.038 0.006 0.010 0.204 0.080 0.016 0.010 0.044 0.102 0.099 0.006 0.081 0.100 0.126 0.048 0.048 -

CompanyOwner Overconfidence 0.042 0.006 -0.002 -0.016 0.009 -0.003 -0.004 0.029 0.032 -0.002 0.021 -0.119 0.076 0.027 0.013 0.051 -0.127 -0.006 -0.004 0.090 0.156 -0.086 0.008 0.042 0.004 LossAversion 0.037 0.006 0.008 0.005 0.003 0.015 -0.038 0.033 0.043 0.020 0.000 -0.159 0.076 0.016 0.003 -0.030 -0.050 0.000 -0.009 0.102 0.163 -0.014 0.010 0.038 0.004 Herding 0.034 0.006 -0.004 0.008 0.015 -0.006 -0.016 0.025 0.039 -0.008 0.013 -0.230 0.072 0.020 0.001 0.035 -0.147 -0.009 -0.004 0.044 0.087 -0.061 -0.004 0.040 0.004 Anchoring 0.034 0.006 -0.007 -0.015 0.009 0.006 -0.006 0.025 0.035 0.005 0.018 -0.109 0.077 0.008 0.021 0.135 -0.028 0.034 -0.004 0.123 0.147 -0.045 0.021 0.041 0.004 ConfirmationBias 0.020 0.000 0.009 -0.003 -0.006 -0.009 -0.021 0.019 0.031 0.001 0.012 -0.197 0.055 -0.013 0.001 0.014 -0.054 -0.030 -0.005 0.040 -0.106 -0.090 -0.015 0.033 0.003

AM 0.033 0.004 0.001 -0.004 0.006 0.000 -0.017 0.026 0.036 0.003 0.013 -0.163 0.071 0.011 0.008 0.041 -0.081 -0.002 -0.005 0.080 0.089 -0.059 0.004 0.034 MAV 0.033 0.004 0.006 0.009 0.008 0.008 0.017 0.026 0.036 0.007 0.013 0.163 0.071 0.017 0.008 0.053 0.081 0.016 0.005 0.080 0.132 0.059 0.039 0.039 -

Explicit Base 0.869 0.907 0.914 0.935 0.915 0.909 0.941 0.879 0.877 0.910 0.929 0.864 0.845 0.911 0.911 0.446 0.887 0.813 0.917 0.738 0.670 0.825 0.855 0.855 -

RetailInvestor Overconfidence 0.041 0.005 -0.003 -0.016 -0.006 -0.132 -0.013 0.040 0.026 0.014 0.013 -0.060 0.063 0.014 0.003 0.051 -0.214 -0.022 -0.008 0.035 -0.010 -0.112 -0.013 0.041 0.004 LossAversion 0.023 0.004 -0.001 -0.024 -0.018 0.005 -0.025 0.013 0.028 0.007 0.008 -0.098 0.067 0.030 0.003 0.063 -0.057 -0.015 -0.004 -0.023 0.010 -0.187 -0.009 0.032 0.003 Herding 0.011 0.001 -0.001 0.004 0.010 0.012 -0.012 0.029 0.032 0.015 0.009 -0.078 0.093 0.013 0.005 0.075 -0.173 -0.074 -0.009 -0.065 0.013 -0.002 -0.004 0.033 0.003 Anchoring 0.034 0.006 -0.001 0.001 -0.007 -0.002 -0.016 0.026 0.036 0.006 -0.002 -0.169 0.056 0.023 -0.010 -0.023 -0.110 -0.058 -0.004 0.023 -0.057 -0.175 -0.019 0.038 0.004 ConfirmationBias 0.041 0.006 0.015 -0.018 -0.001 0.024 -0.008 0.033 0.039 0.008 0.025 0.001 0.068 0.019 0.023 -0.006 -0.159 -0.083 0.000 0.044 -0.069 -0.110 -0.005 0.036 0.003

AM 0.030 0.004 0.002 -0.010 -0.004 -0.019 -0.015 0.028 0.032 0.010 0.011 -0.081 0.069 0.020 0.005 0.032 -0.142 -0.050 -0.005 0.003 -0.023 -0.117 -0.010 0.032 MAV 0.030 0.004 0.004 0.013 0.008 0.035 0.015 0.028 0.032 0.010 0.011 0.081 0.069 0.020 0.009 0.043 0.142 0.050 0.005 0.038 0.032 0.117 0.036 0.036 -

Professional Overconfidence 0.039 0.005 0.005 -0.026 -0.004 0.005 -0.030 0.028 0.031 0.007 0.017 -0.316 0.066 0.023 0.004 -0.069 -0.108 -0.054 -0.004 0.109 -0.053 -0.169 -0.022 0.053 0.008 LossAversion 0.025 0.005 0.003 0.005 0.023 0.014 -0.045 0.032 0.033 0.011 0.009 -0.180 0.079 0.021 0.002 -0.031 -0.063 -0.088 -0.009 0.007 -0.127 -0.065 -0.015 0.040 0.004 Herding 0.030 0.009 0.001 0.002 -0.002 -0.001 -0.024 0.026 0.035 0.008 0.021 -0.226 0.061 0.020 0.011 -0.078 -0.143 -0.083 -0.004 0.076 0.112 -0.057 -0.009 0.047 0.005 Anchoring 0.028 0.001 -0.023 -0.005 0.019 -0.027 -0.017 0.017 0.034 0.012 -0.001 -0.298 0.072 0.026 0.006 -0.154 -0.025 -0.051 -0.009 0.032 -0.080 -0.107 -0.025 0.047 0.007 ConfirmationBias 0.036 0.005 -0.005 -0.048 0.016 0.001 -0.045 0.021 0.039 -0.013 0.025 -0.281 0.087 0.031 -0.003 -0.036 -0.077 -0.067 0.000 -0.068 -0.072 -0.058 -0.023 0.047 0.006

AM 0.032 0.005 -0.004 -0.014 0.010 -0.002 -0.032 0.025 0.034 0.005 0.014 -0.260 0.073 0.024 0.004 -0.074 -0.083 -0.069 -0.005 0.031 -0.044 -0.091 -0.019 0.043 MAV 0.032 0.005 0.007 0.017 0.013 0.009 0.032 0.025 0.034 0.010 0.015 0.260 0.073 0.024 0.005 0.074 0.083 0.069 0.005 0.059 0.089 0.091 0.047 0.047 -

CompanyOwner Overconfidence 0.042 0.009 -0.005 -0.003 0.005 0.002 -0.009 0.016 0.035 0.012 0.009 -0.156 0.063 0.000 -0.018 0.011 -0.211 -0.039 -0.004 0.116 -0.059 0.027 -0.007 0.039 0.004 LossAversion 0.027 0.006 0.007 -0.003 0.001 0.020 -0.013 0.034 0.032 0.008 0.009 -0.187 0.052 0.016 0.001 0.048 -0.070 0.003 -0.004 0.120 0.110 -0.023 0.009 0.036 0.003 Herding 0.042 0.002 0.007 -0.015 0.021 -0.008 -0.037 0.032 0.039 0.008 0.008 -0.113 0.071 0.011 0.010 -0.054 -0.114 -0.039 -0.004 0.025 -0.007 -0.128 -0.011 0.036 0.003 Anchoring 0.042 0.002 0.016 0.005 0.023 0.012 0.013 0.029 0.032 0.008 0.010 -0.072 0.081 0.034 -0.003 0.023 -0.069 0.015 -0.004 0.106 0.089 -0.081 0.014 0.035 0.002 ConfirmationBias 0.040 0.004 -0.006 -0.013 0.015 0.001 -0.044 0.009 0.038 -0.010 -0.005 -0.115 0.052 0.014 0.005 0.042 -0.052 -0.028 -0.009 0.034 -0.199 -0.061 -0.013 0.036 0.003

AM 0.039 0.005 0.004 -0.006 0.013 0.005 -0.018 0.024 0.035 0.005 0.006 -0.128 0.064 0.015 -0.001 0.014 -0.103 -0.018 -0.005 0.080 -0.013 -0.053 -0.002 0.030 MAV 0.039 0.005 0.008 0.008 0.013 0.008 0.023 0.024 0.035 0.009 0.008 0.128 0.064 0.015 0.008 0.035 0.103 0.025 0.005 0.080 0.093 0.064 0.036 0.036 -

TRUE Category Implicit Base 0.273 0.439 0.533 0.619 0.615 0.655 0.731 0.250 0.279 0.500 0.653 0.500 0.612 0.258 0.476 0.462 0.000 0.450 0.083 0.160 0.250 0.258 0.412 0.412 -

RetailInvestor Overconfidence 0.040 -0.275 -0.121 -0.048 -0.078 -0.043 -0.104 -0.036 -0.279 -0.125 0.029 -0.015 -0.016 0.087 -0.002 -0.150 0.000 -0.063 -0.003 -0.160 -0.250 -0.258 -0.085 0.099 0.018 LossAversion -0.097 -0.275 -0.322 0.000 0.021 0.039 -0.037 -0.068 -0.125 -0.214 0.070 -0.224 0.024 0.180 0.146 -0.212 0.083 -0.026 0.000 -0.006 -0.173 -0.258 -0.067 0.118 0.024 Herding -0.193 -0.429 -0.462 -0.143 -0.538 -0.179 -0.109 -0.250 -0.279 -0.500 -0.048 -0.155 -0.345 0.028 -0.052 -0.176 0.000 -0.228 -0.083 -0.160 -0.167 -0.175 -0.211 0.214 0.068 Anchoring -0.119 -0.429 -0.210 -0.082 -0.319 -0.106 -0.135 -0.250 -0.279 -0.278 -0.048 -0.286 -0.183 -0.027 -0.014 -0.388 0.000 -0.228 -0.083 -0.160 -0.250 -0.110 -0.181 0.181 0.047 ConfirmationBias 0.080 -0.229 -0.221 -0.119 -0.065 -0.062 -0.131 0.063 -0.057 -0.306 0.041 -0.177 -0.122 0.268 0.160 -0.023 0.057 -0.105 -0.009 0.000 -0.112 0.075 -0.045 0.113 0.019

AM -0.058 -0.327 -0.267 -0.078 -0.196 -0.070 -0.103 -0.108 -0.204 -0.285 0.009 -0.172 -0.129 -0.258 0.048 -0.190 0.028 -0.130 -0.036 -0.097 -0.190 -0.145 -0.134 0.142 MAV 0.106 0.327 0.267 0.078 0.204 0.086 0.103 0.133 0.204 0.285 0.047 0.172 0.138 0.118 0.075 0.190 0.028 0.130 0.036 0.097 0.190 0.175 0.145 0.145 -

Professional Overconfidence -0.125 -0.275 -0.210 -0.055 -0.215 -0.113 -0.079 -0.019 -0.279 -0.233 0.014 -0.155 -0.027 0.099 0.095 -0.129 0.000 -0.050 -0.003 -0.006 -0.250 -0.258 -0.103 0.122 0.024 LossAversion 0.070 -0.291 -0.019 -0.040 -0.127 0.025 -0.154 -0.050 -0.057 -0.088 0.058 -0.063 -0.007 0.227 0.177 -0.150 0.074 -0.117 -0.003 0.000 -0.250 -0.104 -0.040 0.098 0.016 Herding -0.199 -0.198 -0.200 -0.148 -0.201 -0.133 -0.091 -0.167 -0.199 -0.100 -0.103 -0.167 -0.155 -0.027 0.024 -0.012 0.080 -0.105 -0.083 0.000 -0.250 -0.258 -0.122 0.132 0.023 Anchoring 0.060 -0.281 -0.133 -0.055 -0.032 0.024 -0.039 -0.117 -0.125 -0.122 0.067 -0.167 -0.021 0.166 0.048 -0.224 0.207 0.007 -0.006 0.136 -0.250 0.075 -0.035 0.107 0.018 ConfirmationBias 0.091 -0.275 -0.221 -0.157 0.067 0.000 -0.138 0.202 -0.057 -0.147 -0.001 -0.048 0.024 0.268 0.077 -0.176 0.074 -0.117 -0.006 0.000 -0.170 -0.110 -0.037 0.110 0.019

AM -0.021 -0.264 -0.156 -0.091 -0.102 -0.040 -0.100 -0.030 -0.143 -0.138 0.007 -0.120 -0.037 0.147 0.084 -0.138 0.087 -0.076 -0.020 0.026 -0.234 -0.131 -0.068 0.100 MAV 0.109 0.264 0.156 0.091 0.128 0.059 0.100 0.111 0.143 0.138 0.049 0.120 0.047 0.158 0.084 0.138 0.087 0.079 0.020 0.028 0.234 0.161 0.114 0.114 -

CompanyOwner Overconfidence -0.066 -0.349 -0.385 -0.119 -0.102 -0.062 -0.037 -0.107 -0.279 -0.423 0.086 -0.188 -0.088 0.242 0.037 -0.094 0.000 -0.105 -0.003 -0.160 -0.250 -0.258 -0.123 0.156 0.039 LossAversion -0.015 -0.349 -0.158 -0.004 -0.251 -0.001 -0.182 -0.102 -0.119 -0.063 -0.015 -0.155 -0.112 0.087 0.057 -0.144 0.000 0.017 -0.006 -0.160 -0.167 -0.258 -0.095 0.110 0.021 Herding -0.079 -0.349 -0.230 0.048 -0.178 -0.106 -0.106 -0.176 -0.125 -0.258 0.043 -0.063 -0.047 0.099 0.095 -0.229 0.077 -0.093 -0.083 0.000 -0.250 -0.184 -0.100 0.133 0.025 Anchoring -0.073 -0.349 -0.333 -0.145 -0.384 -0.025 -0.112 -0.176 -0.131 -0.233 0.058 -0.155 -0.138 0.018 0.088 -0.220 0.000 0.002 -0.083 -0.160 -0.250 -0.258 -0.139 0.154 0.036 ConfirmationBias -0.051 -0.229 0.004 -0.078 -0.093 -0.045 -0.104 -0.015 -0.202 -0.050 0.078 -0.015 -0.052 -0.023 0.095 -0.098 0.074 -0.026 0.065 -0.077 -0.250 0.028 -0.048 0.080 0.011

AM -0.057 -0.325 -0.220 -0.060 -0.202 -0.048 -0.108 -0.115 -0.171 -0.205 0.050 -0.115 -0.087 0.085 0.075 -0.157 0.030 -0.041 -0.022 -0.111 -0.233 -0.186 -0.101 0.123 MAV 0.057 0.325 0.222 0.079 0.202 0.048 0.108 0.115 0.171 0.205 0.056 0.115 0.087 0.094 0.075 0.157 0.030 0.048 0.048 0.111 0.233 0.197 0.127 0.127 -

Explicit Base 0.273 0.439 0.533 0.619 0.615 0.655 0.731 0.250 0.279 0.500 0.653 0.500 0.612 0.258 0.476 0.462 0.000 0.450 0.083 0.160 0.250 0.258 0.412 0.412 -

RetailInvestor Overconfidence 0.040 -0.281 -0.057 -0.119 -0.093 -0.216 -0.064 0.026 -0.141 0.013 0.055 -0.167 0.020 0.166 0.036 -0.062 0.000 -0.063 -0.083 -0.083 -0.170 -0.110 -0.062 0.094 0.014 LossAversion 0.035 -0.222 -0.180 -0.143 -0.171 -0.012 -0.131 0.058 -0.279 -0.100 0.041 -0.088 -0.054 0.194 0.036 -0.227 0.154 -0.026 -0.003 -0.083 -0.250 -0.120 -0.071 0.118 0.021 Herding -0.057 -0.352 -0.144 0.015 -0.191 -0.043 -0.093 -0.107 -0.279 -0.076 0.028 -0.029 0.007 -0.027 -0.014 -0.030 0.000 -0.183 -0.006 -0.077 -0.250 -0.178 -0.095 0.099 0.019 Anchoring -0.079 -0.349 -0.144 -0.040 -0.205 -0.078 -0.106 -0.250 -0.199 -0.068 -0.082 -0.088 -0.059 0.156 0.002 -0.018 0.000 -0.127 -0.083 -0.077 -0.250 -0.258 -0.109 0.124 0.023 ConfirmationBias 0.040 -0.349 -0.033 -0.054 -0.091 0.025 -0.051 -0.102 -0.125 -0.136 0.113 -0.015 -0.090 0.180 0.160 -0.023 0.000 -0.026 0.000 -0.077 -0.170 -0.189 -0.046 0.093 0.015

AM -0.005 -0.311 -0.112 -0.068 -0.150 -0.065 -0.089 -0.075 -0.205 -0.073 0.031 -0.078 -0.035 0.134 0.044 -0.072 0.031 -0.085 -0.035 -0.079 -0.218 -0.171 -0.077 0.098 MAV 0.050 0.311 0.112 0.074 0.150 0.075 0.089 0.109 0.205 0.079 0.064 0.078 0.046 0.144 0.050 0.072 0.031 0.085 0.035 0.079 0.218 0.171 0.106 0.106 -

Professional Overconfidence -0.130 -0.281 -0.257 -0.097 -0.139 -0.012 -0.143 -0.050 -0.202 -0.100 0.070 -0.113 -0.047 0.156 0.012 -0.098 0.000 -0.026 -0.003 0.062 -0.250 -0.184 -0.083 0.111 0.019 LossAversion -0.091 -0.281 -0.133 -0.004 0.052 0.012 -0.145 0.008 0.024 -0.056 0.028 -0.155 0.013 0.306 0.077 -0.062 0.125 -0.050 -0.006 0.071 -0.250 -0.051 -0.026 0.091 0.016 Herding -0.140 -0.275 -0.266 -0.090 -0.262 -0.113 -0.166 -0.250 -0.131 -0.177 0.086 -0.136 -0.162 0.099 0.082 -0.081 0.154 -0.127 -0.083 0.000 -0.167 -0.098 -0.105 0.143 0.025 Anchoring 0.005 -0.352 -0.192 0.003 0.037 -0.113 -0.091 -0.181 -0.072 0.078 0.026 -0.233 -0.034 0.271 0.089 -0.094 0.138 0.050 -0.006 0.136 -0.250 -0.098 -0.040 0.116 0.022 ConfirmationBias 0.060 -0.281 -0.190 -0.170 -0.010 -0.037 -0.145 -0.179 -0.125 -0.222 0.123 -0.286 0.108 0.258 0.066 -0.112 0.077 0.064 0.000 -0.006 -0.043 0.213 -0.038 0.126 0.023

AM -0.059 -0.294 -0.208 -0.071 -0.064 -0.053 -0.138 -0.130 -0.101 -0.095 0.067 -0.185 -0.024 0.218 0.065 -0.089 0.099 -0.018 -0.020 0.053 -0.192 -0.044 -0.058 0.104 MAV 0.085 0.294 0.208 0.073 0.100 0.057 0.138 0.134 0.111 0.127 0.067 0.185 0.073 0.218 0.065 0.089 0.099 0.063 0.020 0.055 0.192 0.129 0.117 0.117

CompanyOwner Overconfidence -0.066 -0.275 -0.190 -0.078 -0.141 -0.051 -0.039 -0.121 -0.131 -0.125 0.028 -0.076 -0.054 0.000 -0.041 -0.062 0.148 -0.117 -0.003 -0.160 -0.250 -0.104 -0.087 0.103 0.016 LossAversion -0.202 -0.349 -0.121 -0.055 -0.153 -0.017 -0.064 -0.170 -0.279 -0.177 0.028 -0.155 -0.067 0.087 -0.055 -0.138 0.000 -0.174 -0.003 -0.160 -0.167 -0.110 -0.114 0.124 0.023 Herding -0.066 -0.429 -0.121 -0.145 -0.036 -0.073 -0.200 -0.043 -0.125 -0.136 0.041 -0.088 -0.034 0.075 0.120 -0.007 0.194 -0.063 -0.083 0.000 -0.090 -0.125 -0.065 0.104 0.019 Anchoring -0.066 -0.429 -0.133 -0.024 -0.148 -0.043 0.035 -0.173 -0.279 -0.177 -0.002 -0.048 -0.027 0.305 -0.066 -0.159 0.000 -0.050 -0.083 -0.160 -0.250 -0.027 -0.091 0.122 0.027 ConfirmationBias 0.080 -0.222 -0.155 -0.061 0.007 -0.037 -0.160 0.013 -0.065 -0.141 -0.013 -0.063 -0.112 0.199 0.124 -0.079 0.074 -0.026 -0.006 -0.077 -0.173 -0.110 -0.045 0.091 0.012

AM -0.064 -0.341 -0.144 -0.073 -0.094 -0.044 -0.086 -0.099 -0.176 -0.151 0.016 -0.086 -0.059 0.133 0.016 -0.089 0.083 -0.086 -0.036 -0.111 -0.186 -0.095 -0.080 0.103 MAV 0.096 0.341 0.144 0.073 0.097 0.044 0.100 0.104 0.176 0.151 0.022 0.086 0.059 0.133 0.081 0.089 0.083 0.086 0.036 0.111 0.186 0.095 0.109 0.109 -

Table 10: Results on MFMD-persona: Bengali part.

Mistral-Small-24B

Mistral-NEMO

Mixtral-8x22B

GPT-5-mini-R

Gemini-2.5-R

Mistral-Large

Mixtral-8x7B

Claude-4.5-R

DeepSeek-R

DeepSeek-C

Qwen14b-R

Qwen32b-R

Gemini-2.0

Claude-3.5

Mistral-7B

Qwen8b-R

Llama70b

Qwen14b

Qwen32b

Qwen72b

Qwen8b

GPT-4.1

Role Scenario

AM MAV MSE FALSE Category

Base 0.902 0.919 0.900 0.921 0.908 0.918 0.917 0.903 0.888 0.892 0.938 0.933 0.891 0.921 0.926 0.928 0.906 0.872 0.910 0.673 0.828 0.932 0.897 0.897 -

RetailInvestor Europe 0.019 -0.011 0.019 0.001 0.026 -0.006 0.007 0.018 0.023 0.022 0.000 -0.008 0.028 -0.001 0.001 0.019 0.023 0.017 -0.005 0.227 -0.186 -0.008 0.010 0.031 0.004 USA 0.022 0.003 -0.017 0.005 0.030 -0.003 0.007 0.032 0.022 0.030 0.016 -0.009 0.051 0.013 -0.013 0.013 -0.001 0.045 0.011 0.251 -0.175 0.009 0.015 0.035 0.005 AsiaPacific 0.008 -0.010 0.019 -0.007 0.026 -0.010 0.026 0.002 0.020 0.024 0.008 0.004 0.010 -0.002 -0.016 0.027 0.004 -0.017 0.012 0.256 -0.145 -0.018 0.010 0.031 0.004 ChinaMainland 0.006 -0.006 0.000 0.010 0.030 0.003 0.021 0.018 0.027 0.012 -0.021 -0.006 0.048 -0.008 -0.008 0.012 0.008 0.038 0.001 0.240 -0.079 -0.027 0.014 0.029 0.003 Australia 0.017 -0.004 0.002 -0.007 0.026 0.006 0.021 0.018 0.027 0.019 0.004 0.004 0.040 0.003 -0.005 0.019 0.011 0.020 0.011 0.248 -0.261 -0.036 0.008 0.037 0.006 UAE 0.030 -0.008 0.002 -0.017 0.042 -0.013 0.026 0.013 0.023 0.015 0.001 -0.016 0.016 -0.006 -0.012 0.009 -0.008 -0.056 0.001 0.243 -0.127 -0.011 0.007 0.032 0.004

AM 0.017 -0.006 0.004 -0.003 0.030 -0.004 0.018 0.017 0.024 0.020 0.001 -0.005 0.032 0.000 -0.009 0.016 0.006 0.008 0.005 0.244 -0.162 -0.015 0.011 0.029 MAV 0.017 0.007 0.010 0.008 0.030 0.007 0.018 0.017 0.024 0.020 0.008 0.008 0.032 0.005 0.009 0.016 0.009 0.032 0.007 0.244 0.162 0.018 0.032 0.032 -

Professional Europe 0.014 -0.019 0.002 0.005 0.033 -0.021 0.003 0.005 0.016 0.027 0.021 0.000 0.036 -0.002 -0.005 0.015 0.023 0.030 0.003 0.240 -0.131 -0.026 0.012 0.031 0.004 USA 0.002 0.001 0.014 0.005 0.025 0.000 0.002 0.013 0.028 0.031 0.004 0.000 0.043 0.028 -0.002 0.022 0.010 0.002 0.006 0.240 -0.024 -0.006 0.020 0.023 0.003 AsiaPacific 0.001 -0.015 -0.002 0.001 0.039 0.007 -0.006 -0.012 0.028 0.043 0.000 0.017 0.026 0.013 -0.009 0.005 0.001 -0.051 0.002 0.185 -0.028 -0.024 0.010 0.023 0.002 ChinaMainland 0.006 -0.003 0.002 0.018 0.025 0.014 0.013 0.006 0.024 0.027 -0.004 -0.007 0.020 -0.004 -0.004 0.004 0.009 -0.010 -0.004 0.246 -0.186 -0.048 0.007 0.031 0.005 Australia 0.030 -0.004 0.008 0.000 0.016 0.014 0.003 0.024 0.020 0.027 0.000 -0.012 0.034 0.006 -0.017 0.002 0.015 0.032 0.006 0.236 -0.100 -0.064 0.013 0.030 0.003 UAE -0.007 -0.008 0.002 0.010 0.021 0.001 -0.006 -0.008 0.027 0.030 0.000 -0.006 0.039 0.012 -0.004 0.027 -0.010 -0.129 -0.001 0.221 -0.085 -0.030 0.004 0.031 0.004

AM 0.008 -0.008 0.004 0.007 0.026 0.003 0.002 0.005 0.024 0.031 0.003 -0.001 0.033 0.009 -0.007 0.012 0.008 -0.021 0.002 0.228 -0.092 -0.033 0.011 0.026 MAV 0.010 0.008 0.005 0.007 0.026 0.009 0.005 0.011 0.024 0.031 0.005 0.007 0.033 0.011 0.007 0.012 0.011 0.042 0.004 0.228 0.092 0.033 0.028 0.028 -

CompanyOwner Europe 0.009 0.000 0.005 0.019 0.030 0.014 -0.006 -0.001 0.023 -0.005 -0.004 -0.011 0.028 0.002 -0.035 0.006 -0.006 -0.049 0.008 0.235 -0.060 -0.003 0.009 0.025 0.003 USA -0.012 -0.004 0.007 -0.001 0.026 -0.010 0.008 -0.008 0.023 0.005 -0.009 -0.004 0.030 0.021 -0.033 -0.018 0.005 0.036 0.003 0.248 -0.008 -0.006 0.014 0.024 0.003 AsiaPacific 0.006 -0.014 0.022 0.004 0.021 0.003 0.025 0.004 0.017 0.030 -0.018 0.005 0.030 0.013 -0.018 -0.001 0.001 0.012 -0.002 0.254 -0.105 -0.047 0.011 0.030 0.004 ChinaMainland 0.006 -0.008 0.019 -0.017 0.034 0.014 0.008 0.010 0.020 0.032 0.008 -0.007 0.024 -0.004 -0.004 0.007 0.008 0.054 -0.009 0.248 -0.017 -0.039 0.018 0.027 0.003 Australia 0.010 -0.008 0.014 0.001 0.029 0.002 0.007 0.010 0.023 0.026 0.001 0.009 0.023 -0.009 -0.031 0.018 -0.010 -0.042 0.006 0.235 -0.021 -0.011 0.013 0.025 0.003 UAE 0.006 0.000 0.014 0.010 0.025 0.014 0.012 -0.008 0.024 0.035 -0.001 -0.003 0.028 0.009 -0.010 0.011 0.006 0.012 -0.010 0.235 -0.057 -0.015 0.015 0.025 0.003

AM 0.004 -0.006 0.014 0.003 0.028 0.006 0.009 0.001 0.022 0.021 -0.004 -0.002 0.027 0.005 -0.022 0.004 0.001 0.004 -0.001 0.243 -0.045 -0.020 0.013 0.022 MAV 0.008 0.006 0.014 0.009 0.028 0.009 0.011 0.007 0.022 0.022 0.007 0.006 0.027 0.010 0.022 0.010 0.006 0.034 0.006 0.243 0.045 0.020 0.026 0.026 -

TRUE Category

Base 0.455 0.500 0.500 0.596 0.542 0.679 0.578 0.194 0.426 0.458 0.681 0.651 0.627 0.364 0.591 0.667 0.313 0.605 0.500 0.194 0.412 0.222 0.489 0.489 -

RetailInvestor Europe -0.043 -0.357 0.000 -0.018 0.110 -0.108 0.062 0.218 -0.219 0.054 -0.014 -0.101 -0.090 -0.210 -0.020 0.044 0.125 -0.010 -0.167 0.109 -0.145 0.192 -0.027 0.110 0.020 USA 0.031 -0.125 -0.083 0.013 0.139 -0.064 0.076 0.406 -0.123 0.120 0.103 -0.060 0.009 0.088 -0.038 0.053 0.030 -0.005 -0.056 0.306 -0.245 0.192 0.035 0.107 0.021 AsiaPacific -0.152 -0.500 0.024 -0.063 0.094 -0.119 0.089 0.139 -0.349 -0.026 0.042 -0.029 -0.183 -0.088 -0.091 0.065 -0.001 -0.131 -0.125 0.244 0.027 0.074 -0.048 0.121 0.028 ChinaMainland -0.077 -0.420 -0.176 0.009 0.139 -0.067 0.103 0.218 -0.212 -0.090 -0.116 -0.138 0.024 -0.364 -0.046 -0.052 -0.037 -0.060 -0.242 0.170 -0.126 -0.008 -0.071 0.131 0.028 Australia 0.069 -0.233 -0.071 -0.063 0.110 -0.039 0.103 0.250 -0.159 -0.008 0.015 -0.080 -0.022 -0.133 0.005 0.031 0.062 -0.148 -0.056 0.193 -0.190 0.192 -0.008 0.101 0.016 UAE 0.086 -0.293 -0.071 -0.065 0.208 -0.168 0.104 0.238 -0.219 -0.019 -0.030 -0.101 -0.188 -0.150 -0.058 -0.111 -0.078 -0.131 -0.242 0.170 -0.207 0.135 -0.054 0.140 0.025

AM -0.014 -0.321 -0.063 -0.031 0.133 -0.094 0.089 0.245 -0.214 0.005 0.000 -0.085 -0.075 -0.143 -0.041 0.005 0.017 -0.081 -0.148 0.198 -0.148 0.130 -0.029 0.104 MAV 0.076 0.321 0.071 0.038 0.133 0.094 0.089 0.245 0.214 0.053 0.053 0.085 0.086 0.172 0.043 0.059 0.055 0.081 0.148 0.198 0.157 0.132 0.118 0.118 -

Professional Europe 0.007 -0.429 -0.045 -0.005 0.178 -0.123 0.049 0.216 -0.352 0.042 0.092 -0.066 -0.077 -0.088 0.021 0.015 0.111 -0.120 -0.147 0.082 -0.156 0.009 -0.036 0.110 0.023 USA -0.087 -0.278 0.033 -0.005 0.138 -0.024 0.064 0.238 -0.272 0.079 0.027 -0.032 0.025 0.242 0.049 0.072 0.144 -0.005 -0.068 0.218 -0.059 0.074 0.026 0.102 0.018 AsiaPacific -0.055 -0.426 -0.081 -0.038 0.169 -0.054 0.010 0.147 -0.272 0.161 -0.014 -0.051 -0.062 0.088 -0.026 0.013 0.097 -0.020 -0.079 0.114 0.088 0.192 -0.004 0.103 0.020 ChinaMainland -0.077 -0.346 -0.045 0.055 0.111 0.013 0.044 0.159 -0.278 0.042 -0.045 -0.093 -0.151 -0.281 -0.033 -0.103 -0.099 -0.034 -0.250 0.320 -0.412 0.192 -0.059 0.145 0.034 Australia 0.124 -0.286 -0.090 0.000 0.098 0.025 0.049 0.319 -0.283 0.066 0.000 -0.101 -0.002 -0.007 -0.069 -0.045 0.111 -0.034 -0.068 0.139 -0.236 0.294 0.000 0.111 0.023 UAE -0.105 -0.242 -0.045 0.009 0.096 -0.037 0.010 0.156 -0.159 0.100 0.000 -0.177 -0.005 0.150 -0.033 0.077 -0.035 -0.026 -0.157 0.230 -0.059 0.178 -0.003 0.095 0.014

AM -0.032 -0.334 -0.046 0.003 0.132 -0.034 0.038 0.206 -0.269 0.082 0.010 -0.087 -0.045 0.017 -0.015 0.005 0.055 -0.040 -0.128 0.184 -0.139 0.156 -0.013 0.093 MAV 0.076 0.334 0.057 0.019 0.132 0.046 0.038 0.206 0.269 0.082 0.030 0.087 0.054 0.143 0.039 0.054 0.100 0.040 0.128 0.184 0.168 0.156 0.111 0.111 -

CompanyOwner Europe -0.197 -0.224 0.011 0.019 0.125 0.013 0.010 0.048 -0.219 -0.009 -0.029 -0.125 -0.165 -0.078 -0.111 -0.015 0.020 -0.034 0.045 0.139 -0.096 0.192 -0.031 0.088 0.013 USA -0.074 -0.286 -0.061 0.031 0.110 -0.035 0.047 0.156 -0.219 -0.014 -0.043 -0.101 -0.031 0.103 -0.046 -0.049 0.097 0.014 0.053 0.181 -0.044 0.135 -0.003 0.088 0.012 AsiaPacific -0.045 -0.500 0.058 0.029 0.096 -0.067 0.130 0.245 -0.426 0.100 -0.054 0.000 -0.031 0.050 -0.049 -0.117 -0.106 -0.020 -0.122 0.347 -0.207 -0.062 -0.034 0.130 0.034 ChinaMainland -0.045 -0.293 0.000 -0.065 0.166 0.013 0.047 0.195 -0.283 0.028 0.042 -0.110 -0.177 -0.281 -0.033 -0.067 -0.046 0.062 -0.214 0.170 -0.274 0.009 -0.053 0.119 0.024 Australia -0.034 -0.293 0.012 -0.038 0.152 -0.052 0.062 0.159 -0.219 0.087 -0.047 -0.072 -0.094 -0.216 -0.101 0.056 -0.027 -0.081 -0.038 0.218 -0.079 0.192 -0.021 0.106 0.017 UAE -0.045 -0.224 0.012 0.009 0.125 0.025 0.075 0.156 -0.278 0.092 0.013 -0.125 -0.103 0.074 0.009 -0.016 0.030 -0.005 -0.141 0.159 -0.133 0.135 -0.007 0.090 0.014

AM -0.073 -0.303 0.005 -0.002 0.129 -0.017 0.062 0.160 -0.274 0.048 -0.019 -0.089 -0.100 -0.058 -0.055 -0.035 -0.005 -0.011 -0.069 0.202 -0.139 0.100 -0.025 0.089 MAV 0.073 0.303 0.026 0.032 0.129 0.034 0.062 0.160 0.274 0.055 0.038 0.089 0.100 0.133 0.058 0.053 0.055 0.036 0.102 0.202 0.139 0.121 0.103 0.103 -

Table 11: Results on MFMD-region.

Mistral-Small-24B

Mistral-NEMO

Mixtral-8x22B

GPT-5-mini-R

Gemini-2.5-R

Mistral-Large

Mixtral-8x7B

Claude-4.5-R

DeepSeek-R

DeepSeek-C

Qwen14b-R

Qwen32b-R

Gemini-2.0

Claude-3.5

Mistral-7B

Qwen8b-R

Llama70b

Qwen32b

Qwen14b

Qwen72b

Qwen8b

GPT-4.1

Role Scenario

AM MAV MSE FALSE Category

Base 0.902 0.919 0.900 0.921 0.908 0.918 0.917 0.903 0.888 0.892 0.938 0.933 0.891 0.921 0.926 0.928 0.906 0.872 0.910 0.673 0.828 0.932 0.897 0.897 -

RetailInvestor American-Christianity 0.006 -0.001 0.014 -0.004 0.025 0.014 0.030 0.009 0.034 0.026 -0.005 -0.001 0.030 -0.003 -0.010 0.015 0.005 0.041 0.007 0.251 -0.028 -0.018 0.020 0.026 0.003 American-Islam 0.017 0.002 0.012 0.026 0.042 -0.010 0.020 0.015 0.023 0.021 0.009 -0.012 0.038 0.020 0.003 0.011 -0.035 0.054 0.007 0.249 -0.084 -0.040 0.018 0.034 0.004 American-Buddhism 0.006 -0.012 0.002 0.022 0.021 -0.031 0.021 0.008 0.027 0.013 -0.001 -0.025 0.047 0.013 -0.021 0.019 -0.004 0.063 0.007 0.248 -0.057 -0.024 0.016 0.031 0.003 American-Judaism -0.010 0.006 0.003 0.013 0.025 0.004 0.016 0.002 0.026 0.021 0.012 -0.011 0.042 0.000 -0.009 0.010 -0.008 0.054 0.011 0.248 -0.059 -0.037 0.016 0.029 0.003 European-Christianity 0.035 -0.011 0.009 0.005 0.025 0.006 0.021 0.013 0.023 0.030 -0.009 -0.027 0.031 0.009 -0.010 0.019 0.004 0.033 0.011 0.248 -0.112 -0.042 0.014 0.033 0.004 Jewish-Judaism 0.019 -0.020 0.001 0.022 0.030 0.019 0.016 0.013 0.015 0.006 0.012 -0.014 0.026 -0.001 -0.012 0.012 -0.014 0.047 0.011 0.249 -0.141 -0.042 0.012 0.034 0.004 Chinese-Christianity 0.011 -0.008 0.007 0.001 0.026 0.003 0.034 0.013 0.023 0.030 0.004 0.010 0.023 0.003 -0.009 0.012 -0.024 0.057 0.007 0.252 -0.059 -0.042 0.017 0.030 0.004 Chinese-Islam 0.004 -0.001 -0.012 -0.026 0.021 -0.002 0.012 -0.003 0.023 0.029 -0.001 0.000 0.020 -0.002 0.001 0.019 -0.024 0.059 0.007 0.249 -0.057 -0.053 0.012 0.028 0.003 Chinese-Buddhism -0.012 -0.004 0.019 0.008 0.029 0.015 0.008 0.009 0.027 0.013 0.007 -0.002 0.040 -0.001 0.000 0.016 -0.019 0.067 0.003 0.248 -0.074 -0.053 0.016 0.031 0.004 Indian-Christianity 0.027 -0.009 0.002 0.010 0.018 0.019 0.007 0.009 0.030 0.008 -0.001 -0.008 0.047 0.001 -0.008 0.020 -0.029 0.032 0.011 0.249 -0.067 -0.042 0.015 0.030 0.003 Indian-Islam 0.009 -0.001 -0.003 0.005 0.025 -0.012 0.021 0.016 0.023 0.017 0.012 -0.008 0.036 -0.006 -0.012 0.016 -0.022 0.053 0.008 0.253 -0.079 -0.081 0.012 0.033 0.004 Arab-Islam -0.006 -0.013 0.012 0.000 0.029 0.001 0.025 0.015 0.023 0.009 -0.018 -0.012 0.046 -0.005 -0.008 0.012 -0.035 0.067 0.011 0.242 -0.076 -0.043 0.013 0.032 0.004 Latino/Hispanic-Christianity 0.007 -0.008 0.003 -0.013 0.018 0.019 0.009 0.001 0.023 -0.005 -0.005 0.005 0.042 0.009 0.001 0.019 -0.017 0.063 0.011 0.245 -0.090 -0.038 0.014 0.030 0.004 African-Christianity 0.017 -0.016 -0.007 0.013 0.018 0.006 0.020 0.013 0.023 0.002 0.004 -0.016 0.040 0.010 -0.005 0.011 -0.014 0.059 0.011 0.245 -0.057 -0.033 0.016 0.029 0.003 African-Islam -0.013 -0.008 -0.008 0.001 0.009 -0.011 0.021 -0.002 0.027 0.001 0.004 0.010 0.036 0.002 -0.016 0.019 -0.025 0.063 0.007 0.249 -0.083 -0.076 0.009 0.031 0.004

AM 0.008 -0.007 0.003 0.005 0.024 0.003 0.019 0.009 0.025 0.015 0.001 -0.008 0.036 0.003 -0.008 0.015 -0.017 0.054 0.009 0.248 -0.075 -0.044 0.015 0.029 MAV 0.013 0.008 0.008 0.011 0.024 0.012 0.019 0.009 0.025 0.015 0.007 0.011 0.036 0.006 0.008 0.015 0.019 0.054 0.009 0.248 0.075 0.044 0.031 0.031 -

CompanyOwner American-Christianity -0.008 -0.005 0.008 -0.003 0.025 0.000 0.009 -0.008 0.026 0.016 0.008 -0.018 0.038 0.023 -0.047 0.005 0.001 0.035 -0.015 0.255 -0.087 -0.009 0.011 0.029 0.004 American-Islam 0.001 -0.013 0.025 0.013 0.020 0.018 0.004 -0.022 0.018 0.038 -0.001 -0.005 0.055 0.012 -0.020 0.026 -0.016 0.009 -0.037 0.238 -0.065 -0.016 0.013 0.031 0.003 American-Buddhism 0.004 -0.008 0.014 0.005 0.033 0.010 0.000 0.001 0.027 0.030 0.007 0.017 0.042 0.012 -0.034 0.019 0.007 0.049 -0.028 0.232 -0.112 -0.034 0.013 0.033 0.004 American-Judaism -0.019 -0.016 0.011 0.014 0.025 0.000 0.012 -0.005 0.010 0.004 0.003 0.008 0.050 -0.003 -0.036 0.013 0.002 0.045 -0.019 0.263 -0.112 -0.021 0.010 0.031 0.004 European-Christianity -0.002 0.003 0.011 0.001 0.021 0.004 0.012 0.000 0.034 0.003 0.003 -0.018 0.031 0.012 -0.026 0.022 0.001 0.045 -0.018 0.259 -0.036 -0.004 0.016 0.026 0.003 Jewish-Judaism -0.006 0.003 -0.007 -0.016 -0.007 0.023 -0.006 -0.021 0.030 0.003 0.007 -0.009 0.037 0.009 -0.011 0.022 -0.019 0.042 -0.019 0.248 -0.135 -0.051 0.005 0.033 0.004 Chinese-Christianity -0.019 -0.013 0.002 0.005 0.015 0.014 -0.001 -0.010 0.014 -0.004 0.016 -0.012 0.030 0.005 -0.021 0.035 -0.017 0.023 -0.033 0.231 -0.060 -0.034 0.007 0.028 0.003 Chinese-Islam -0.005 -0.012 0.008 0.006 0.017 0.013 0.004 -0.041 0.018 0.012 0.007 -0.013 0.038 0.009 -0.034 0.015 -0.028 0.027 -0.037 0.252 -0.014 -0.031 0.010 0.029 0.003 Chinese-Buddhism 0.009 -0.004 0.001 0.000 0.020 0.004 0.007 -0.032 0.027 0.013 0.004 -0.009 0.031 0.005 -0.033 0.023 -0.007 0.033 -0.037 0.233 -0.102 -0.031 0.007 0.030 0.003 Indian-Christianity 0.009 -0.013 -0.004 0.014 0.038 0.000 0.008 -0.020 0.018 0.012 -0.001 0.004 0.022 0.000 -0.030 0.014 -0.017 0.018 -0.032 0.227 -0.006 -0.021 0.011 0.024 0.003 Indian-Islam -0.017 -0.010 0.002 0.022 0.013 0.022 -0.001 -0.009 0.017 0.027 0.008 0.009 0.039 0.016 -0.039 0.006 -0.004 0.013 -0.032 0.243 -0.024 -0.040 0.012 0.028 0.003 Arab-Islam -0.036 -0.001 0.029 0.014 0.017 0.004 -0.009 -0.021 0.026 0.017 0.016 -0.009 0.046 0.012 -0.029 0.027 -0.009 0.049 -0.028 0.233 -0.048 -0.040 0.012 0.033 0.003 Latino/Hispanic-Christianity 0.002 -0.008 -0.007 0.017 0.033 0.009 0.025 0.008 0.025 -0.009 0.012 -0.022 0.020 0.012 -0.039 0.022 -0.007 0.036 -0.024 0.263 -0.117 -0.020 0.011 0.034 0.004 African-Christianity -0.017 -0.009 0.005 0.000 0.017 0.019 -0.001 -0.029 0.026 0.025 0.016 -0.017 0.046 0.016 -0.010 0.015 -0.009 0.032 -0.032 0.236 -0.017 -0.013 0.014 0.028 0.003 African-Islam -0.028 -0.008 0.021 0.022 0.030 0.014 -0.009 -0.009 0.027 0.043 0.008 -0.005 0.038 0.020 -0.038 0.031 -0.016 0.027 -0.037 0.248 -0.059 -0.032 0.013 0.035 0.004

AM -0.009 -0.008 0.008 0.008 0.021 0.010 0.004 -0.014 0.023 0.015 0.008 -0.007 0.038 0.011 -0.030 0.020 -0.009 0.032 -0.029 0.244 -0.066 -0.027 0.011 0.029 MAV 0.012 0.009 0.010 0.010 0.022 0.010 0.007 0.016 0.023 0.017 0.008 0.012 0.038 0.011 0.030 0.020 0.011 0.032 0.029 0.244 0.066 0.027 0.030 0.030 -

TRUE Category

Base 0.455 0.500 0.500 0.596 0.542 0.679 0.578 0.194 0.426 0.458 0.681 0.651 0.627 0.364 0.591 0.667 0.313 0.605 0.500 0.194 0.412 0.222 0.489 0.489 -

RetailInvestor American-Christianity -0.045 -0.177 0.033 -0.031 0.125 0.013 0.133 0.227 -0.093 0.087 -0.001 0.045 -0.015 -0.088 0.009 0.000 0.030 0.081 -0.100 0.218 -0.062 0.216 0.027 0.083 0.012 American-Islam -0.179 -0.088 0.071 0.115 0.218 -0.119 0.116 0.170 -0.168 0.095 0.030 -0.080 0.011 0.152 0.062 -0.033 0.003 0.044 -0.100 0.139 0.053 0.178 0.031 0.101 0.014 American-Buddhism -0.045 -0.300 -0.045 0.086 0.111 -0.127 0.089 0.256 -0.212 0.053 0.013 -0.101 0.040 0.050 -0.080 0.016 -0.019 0.062 -0.100 0.181 -0.056 0.192 0.003 0.101 0.016 American-Judaism -0.147 -0.076 0.049 0.056 0.125 0.000 0.103 0.139 -0.114 0.095 0.079 -0.138 0.040 0.060 -0.026 0.000 -0.035 0.062 -0.088 0.193 0.027 0.245 0.029 0.086 0.011 European-Christianity 0.101 -0.357 0.022 -0.005 0.125 -0.039 0.089 0.268 -0.168 0.120 -0.028 -0.165 -0.127 0.036 0.009 0.016 0.020 0.006 -0.088 0.218 -0.071 0.192 0.008 0.103 0.019 Jewish-Judaism -0.011 -0.367 -0.022 0.071 0.139 0.015 0.102 0.238 -0.288 -0.039 0.079 0.031 -0.094 -0.133 -0.079 -0.052 0.076 -0.076 -0.088 0.028 -0.298 0.178 -0.027 0.114 0.022 Chinese-Christianity -0.066 -0.293 -0.061 -0.018 0.110 -0.067 0.149 0.268 -0.219 0.120 0.027 -0.125 -0.082 -0.204 -0.026 -0.052 -0.027 0.062 -0.100 0.139 0.053 0.192 -0.010 0.112 0.019 Chinese-Islam -0.205 -0.177 -0.074 -0.086 0.111 -0.079 0.075 0.174 -0.168 0.138 0.013 -0.051 -0.165 -0.068 -0.020 0.016 0.030 0.027 -0.100 0.082 -0.062 0.245 -0.016 0.098 0.014 Chinese-Buddhism -0.279 -0.233 0.024 0.057 0.152 0.001 0.060 0.227 -0.212 0.031 0.064 -0.087 -0.022 -0.281 0.000 -0.035 -0.010 0.044 -0.147 0.139 -0.196 0.135 -0.026 0.111 0.021 Indian-Christianity 0.045 -0.197 -0.045 0.009 0.049 0.015 0.062 0.227 -0.103 0.042 0.025 -0.066 0.054 -0.078 -0.046 0.000 0.030 -0.010 -0.088 0.082 -0.048 0.230 0.008 0.071 0.009 Indian-Islam 0.021 -0.136 -0.056 -0.005 0.139 -0.086 0.103 0.306 -0.168 0.064 0.069 -0.051 -0.056 -0.133 -0.079 -0.017 -0.019 0.045 -0.136 0.092 -0.079 0.216 0.001 0.094 0.014 Arab-Islam -0.139 -0.250 0.071 0.000 0.152 -0.037 0.130 0.170 -0.168 0.020 -0.054 -0.127 0.040 -0.210 -0.046 -0.052 -0.019 0.044 -0.088 0.020 -0.071 0.245 -0.017 0.098 0.015 Latino/Hispanic-Christianity -0.112 -0.242 0.049 -0.036 0.049 0.027 0.013 0.174 -0.159 -0.009 -0.001 0.032 0.026 -0.007 -0.020 0.016 0.020 0.062 -0.088 0.082 -0.079 0.178 -0.001 0.067 0.009 African-Christianity 0.045 -0.362 -0.065 0.040 0.067 -0.039 0.116 0.238 -0.219 -0.077 0.027 -0.066 -0.042 -0.007 0.005 -0.016 0.097 0.044 -0.088 0.020 -0.022 0.178 -0.006 0.085 0.015 African-Islam -0.233 -0.242 -0.042 -0.018 0.041 -0.102 0.089 0.092 -0.159 -0.023 0.015 0.016 -0.077 -0.068 -0.091 0.031 -0.019 0.044 -0.100 0.020 -0.062 0.165 -0.033 0.079 0.011

AM -0.083 -0.233 -0.006 0.016 0.114 -0.042 0.095 0.212 -0.175 0.048 0.024 -0.062 -0.031 -0.065 -0.028 -0.011 0.011 0.036 -0.100 0.110 -0.065 0.199 -0.002 0.080 MAV 0.111 0.233 0.049 0.042 0.114 0.051 0.095 0.212 0.175 0.068 0.035 0.079 0.059 0.105 0.040 0.023 0.030 0.047 0.100 0.110 0.083 0.199 0.094 0.094 -

CompanyOwner American-Christianity -0.074 -0.188 0.042 -0.051 0.138 -0.024 0.031 0.165 -0.114 0.113 0.054 -0.046 0.025 0.247 -0.091 0.013 0.065 0.010 -0.010 0.230 -0.012 0.192 0.033 0.088 0.013 American-Islam -0.055 -0.206 0.125 0.040 0.125 0.038 0.034 0.162 -0.176 0.164 0.025 -0.046 0.096 0.150 0.002 0.099 -0.019 -0.055 -0.077 0.277 -0.002 0.009 0.032 0.090 0.013 American-Buddhism 0.010 -0.242 0.012 0.013 0.178 0.000 0.005 0.174 -0.159 0.120 0.074 0.031 0.053 0.121 -0.030 0.044 -0.010 -0.010 -0.060 0.320 -0.048 0.074 0.030 0.081 0.014 American-Judaism -0.162 -0.306 -0.024 0.023 0.125 -0.012 0.075 0.225 -0.239 0.010 0.050 0.016 0.104 0.000 -0.091 0.053 0.030 0.010 -0.020 0.335 -0.119 0.123 0.009 0.098 0.019 European-Christianity -0.131 -0.125 -0.024 -0.038 0.111 0.000 0.060 0.206 -0.051 0.032 0.039 -0.046 -0.049 0.107 0.005 0.072 0.076 -0.026 -0.042 0.291 -0.112 0.074 0.020 0.078 0.010 Jewish-Judaism -0.139 -0.125 -0.091 -0.085 0.083 0.052 0.010 0.132 -0.103 0.052 0.074 0.032 0.040 0.036 0.024 0.072 0.126 -0.064 -0.020 0.206 -0.053 0.178 0.020 0.082 0.009 Chinese-Christianity -0.162 -0.206 -0.035 0.013 0.150 0.013 0.022 0.215 -0.184 -0.032 0.085 -0.051 -0.015 0.023 0.016 0.133 0.020 0.027 -0.047 0.170 -0.053 0.135 0.011 0.082 0.012 Chinese-Islam -0.011 -0.300 0.060 -0.025 0.083 0.035 0.018 0.133 -0.132 0.073 0.064 -0.106 0.011 0.074 -0.030 0.000 -0.027 -0.034 -0.077 0.193 -0.022 0.135 0.005 0.075 0.010 Chinese-Buddhism 0.021 -0.233 -0.022 0.000 0.125 0.011 0.062 0.006 -0.159 0.031 0.015 -0.032 -0.127 0.023 -0.046 0.060 0.020 -0.026 -0.077 0.082 0.014 0.009 -0.011 0.055 0.006 Indian-Christianity -0.005 -0.250 -0.022 0.023 0.181 -0.012 0.060 0.232 -0.176 0.084 0.013 -0.066 -0.062 0.074 -0.005 0.029 0.055 0.062 -0.069 0.170 -0.022 0.123 0.019 0.081 0.012 Indian-Islam -0.091 -0.157 -0.071 0.086 0.054 0.062 0.022 0.147 -0.093 0.066 0.042 -0.002 -0.005 0.165 -0.039 -0.031 0.011 -0.026 -0.069 0.291 0.007 0.111 0.022 0.075 0.010 Arab-Islam -0.095 -0.136 0.153 0.023 0.083 0.000 -0.018 0.246 -0.114 0.064 0.085 -0.051 0.067 0.136 -0.035 0.077 0.097 0.010 -0.060 0.073 0.121 0.123 0.039 0.085 0.010 Latino/Hispanic-Christianity -0.087 -0.242 -0.065 0.071 0.178 0.012 0.130 0.256 -0.073 -0.041 0.069 -0.032 -0.039 0.165 -0.039 0.072 0.055 0.095 -0.029 0.377 0.032 0.192 0.048 0.107 0.020 African-Christianity -0.284 -0.197 -0.011 0.000 0.083 0.015 0.022 0.050 -0.114 0.107 0.095 -0.051 0.011 0.165 0.009 0.015 0.087 -0.026 -0.069 0.170 0.197 0.135 0.019 0.087 0.013 African-Islam -0.211 -0.293 0.096 0.071 0.139 0.013 -0.018 0.187 -0.159 0.161 0.042 0.032 -0.002 0.165 -0.055 0.106 0.055 0.010 -0.077 0.193 -0.022 0.123 0.025 0.101 0.016

AM -0.098 -0.214 0.008 0.011 0.122 0.013 0.034 0.169 -0.136 0.067 0.055 -0.028 0.007 0.110 -0.027 0.054 0.043 -0.003 -0.053 0.225 -0.006 0.116 0.021 0.073 MAV 0.103 0.214 0.057 0.037 0.122 0.020 0.039 0.169 0.136 0.077 0.055 0.043 0.047 0.110 0.034 0.058 0.050 0.033 0.053 0.225 0.056 0.116 0.084 0.084 -

Table 12: Results on MFMD-identity.

Mistral-Small-24B

Mistral-NEMO

Mixtral-8x22B

GPT-5-mini-R

Mistral-Large

Mixtral-8x7B

DeepSeek-R

DeepSeek-C

Qwen14b-R

Qwen32b-R

Mistral-7B

Qwen8b-R

Llama70b

Qwen14b

Qwen32b

Qwen72b

Qwen8b

GPT-4.1

Role Scenario

AM MAV MSE FALSE Category

Base 0.902 0.919 0.900 0.921 0.917 0.903 0.888 0.892 0.938 0.921 0.926 0.928 0.906 0.872 0.910 0.673 0.828 0.932 0.897 0.897 -

RetailInvestor American-Atheist 0.010 -0.011 0.000 0.013 0.011 0.006 0.031 0.000 0.000 0.023 -0.023 0.031 -0.009 0.054 0.011 0.252 -0.077 -0.006 0.018 0.032 0.004 European-Atheist 0.014 -0.007 0.001 0.017 0.024 0.022 0.024 0.004 0.004 0.019 -0.026 0.011 -0.011 0.045 0.011 0.257 -0.161 -0.042 0.011 0.039 0.006 European-Muslim 0.023 -0.011 -0.003 0.008 0.005 0.005 0.024 0.000 0.003 0.024 -0.009 0.023 -0.025 0.042 0.011 0.242 -0.112 -0.053 0.011 0.035 0.004 Jewish-Atheist 0.023 -0.004 -0.007 0.001 0.024 -0.011 0.031 0.016 0.009 0.012 -0.008 0.007 -0.045 0.071 0.011 0.246 -0.111 -0.029 0.013 0.037 0.005 Chinese-Atheist 0.012 -0.007 0.018 -0.011 0.006 0.018 0.024 0.016 0.000 0.016 -0.009 0.003 -0.030 0.041 0.011 0.253 -0.131 -0.037 0.011 0.036 0.005 Indian-Atheist 0.006 -0.004 0.005 0.013 0.024 0.007 0.024 0.012 -0.013 0.000 -0.005 0.012 -0.030 0.059 0.011 0.253 -0.158 -0.055 0.009 0.038 0.005 Arab-Christianity 0.011 -0.004 0.022 0.008 0.011 0.002 0.027 0.000 -0.023 0.016 -0.009 0.003 -0.032 0.045 0.011 0.254 -0.055 -0.041 0.014 0.032 0.004 Arab-Atheist 0.006 -0.015 0.001 0.008 0.020 0.015 0.027 0.003 -0.005 0.000 -0.013 0.019 -0.039 0.067 0.011 0.238 -0.108 -0.033 0.011 0.035 0.004 Latino / Hispanic-Atheist 0.010 -0.008 0.013 0.010 0.012 0.005 0.027 0.013 -0.016 0.000 -0.008 0.031 -0.027 0.058 0.011 0.253 -0.161 -0.043 0.010 0.039 0.005 African-Atheist 0.026 -0.024 0.009 0.017 0.021 0.018 0.024 0.012 0.008 0.020 -0.009 0.016 -0.031 0.068 0.015 0.238 -0.105 -0.043 0.015 0.039 0.004

AM 0.014 -0.010 0.006 0.009 0.016 0.009 0.026 0.008 -0.003 0.013 -0.012 0.016 -0.028 0.055 0.012 0.248 -0.118 -0.038 0.012 0.036 MAV 0.014 0.010 0.008 0.011 0.016 0.011 0.026 0.008 0.008 0.013 0.012 0.016 0.028 0.055 0.012 0.248 0.118 0.038 0.036 0.036 -

CompanyOwner American-Atheist -0.017 -0.001 0.012 0.009 -0.002 -0.038 0.024 0.016 -0.005 -0.007 -0.025 -0.004 -0.001 0.025 -0.028 0.244 -0.096 -0.008 0.005 0.031 0.004 European-Atheist -0.007 0.003 -0.005 0.013 0.002 0.000 0.024 0.007 0.003 0.017 -0.028 0.019 0.015 0.035 -0.032 0.233 -0.161 -0.033 0.006 0.035 0.005 European-Muslim -0.015 -0.001 0.009 0.005 0.011 -0.039 0.025 0.025 -0.001 0.014 -0.039 0.007 -0.026 0.026 -0.032 0.248 -0.080 -0.024 0.006 0.035 0.004 Jewish-Atheist -0.020 0.003 0.018 0.010 0.028 -0.021 0.020 0.013 -0.001 -0.010 -0.005 0.022 0.005 0.002 -0.037 0.248 -0.151 -0.057 0.004 0.037 0.005 Chinese-Atheist -0.013 -0.008 0.005 0.005 0.000 -0.038 0.020 -0.015 -0.002 0.018 -0.037 0.019 -0.031 0.009 -0.042 0.249 -0.124 -0.051 -0.002 0.038 0.005 Indian-Atheist 0.000 0.007 0.005 0.001 0.007 -0.034 0.015 -0.004 0.003 0.013 -0.015 0.027 -0.021 0.019 -0.029 0.245 -0.154 -0.048 0.002 0.036 0.005 Arab-Christianity -0.011 -0.017 0.017 0.008 0.010 -0.025 0.015 0.008 -0.005 0.022 -0.031 0.015 -0.004 0.033 -0.029 0.240 -0.060 -0.021 0.009 0.032 0.004 Arab-Atheist -0.009 -0.004 0.019 -0.008 0.010 -0.045 0.015 -0.006 -0.001 0.013 -0.028 0.010 -0.013 0.004 -0.037 0.241 -0.172 -0.034 -0.002 0.037 0.005 Latino / Hispanic-Atheist 0.009 -0.013 -0.005 0.005 0.024 -0.012 0.012 0.013 -0.001 0.013 -0.015 0.018 -0.005 0.007 -0.037 0.249 -0.148 -0.036 0.004 0.034 0.005 African-Atheist -0.011 -0.004 -0.003 0.014 0.002 -0.041 0.028 -0.010 -0.002 0.009 -0.021 0.023 -0.021 0.033 -0.037 0.242 -0.109 -0.013 0.004 0.035 0.004

AM -0.009 -0.003 0.007 0.006 0.009 -0.029 0.020 0.005 -0.001 0.010 -0.025 0.016 -0.010 0.019 -0.034 0.244 -0.126 -0.033 0.004 0.034 MAV 0.011 0.006 0.010 0.008 0.010 0.029 0.020 0.012 0.002 0.014 0.025 0.016 0.014 0.019 0.034 0.244 0.126 0.033 0.035 0.035 -

TRUE Category

Base 0.455 0.500 0.500 0.596 0.578 0.194 0.426 0.458 0.681 0.364 0.591 0.667 0.313 0.605 0.500 0.194 0.412 0.222 0.489 0.489 -

RetailInvestor American-Atheist -0.034 -0.357 0.000 0.056 0.089 0.149 -0.150 -0.188 0.000 0.268 -0.042 0.095 0.131 0.079 -0.088 0.170 0.032 0.123 0.018 0.114 0.021 European-Atheist -0.023 -0.352 -0.022 0.071 0.153 0.263 -0.278 -0.142 0.027 0.207 -0.091 -0.016 -0.063 -0.017 -0.088 0.139 -0.079 0.054 -0.014 0.116 0.023 European-Muslim 0.002 -0.357 -0.056 0.042 0.000 0.184 -0.278 -0.188 0.039 0.181 -0.026 0.060 0.011 0.006 -0.088 -0.046 -0.196 0.230 -0.027 0.111 0.023 Jewish-Atheist 0.002 -0.286 -0.091 -0.038 0.142 0.114 -0.150 -0.048 0.030 0.088 -0.046 -0.067 -0.071 0.081 -0.088 0.028 -0.168 0.178 -0.022 0.095 0.014 Chinese-Atheist 0.057 -0.352 0.045 -0.096 0.089 0.250 -0.278 -0.080 -0.014 0.192 -0.026 -0.082 -0.055 -0.010 -0.088 0.092 0.017 0.192 -0.008 0.112 0.022 Indian-Atheist -0.077 -0.286 -0.011 0.040 0.116 0.109 -0.278 -0.090 -0.056 0.036 0.005 -0.072 -0.010 0.044 -0.088 0.139 -0.079 0.111 -0.025 0.091 0.014 Arab-Christianity -0.066 -0.233 0.078 0.042 0.089 0.139 -0.212 -0.188 -0.066 0.207 -0.026 -0.067 0.055 0.045 -0.088 0.092 -0.003 0.230 0.002 0.107 0.017 Arab-Atheist -0.045 -0.426 -0.022 0.042 0.116 0.170 -0.212 -0.108 -0.014 -0.041 -0.038 0.044 -0.063 0.044 -0.088 0.013 -0.048 0.178 -0.028 0.095 0.019 Latino / Hispanic-Atheist -0.034 -0.293 0.053 0.009 0.075 0.184 -0.212 -0.125 -0.002 0.060 -0.046 0.095 -0.027 0.045 -0.088 0.151 -0.162 -0.008 -0.018 0.093 0.015 African-Atheist 0.071 -0.371 0.022 0.085 0.103 0.218 -0.278 -0.090 0.054 0.181 -0.026 -0.017 0.030 0.024 -0.043 -0.051 0.020 0.165 0.005 0.103 0.020

AM -0.015 -0.331 0.000 0.025 0.097 0.178 -0.232 -0.124 0.000 0.138 -0.036 -0.003 -0.006 0.034 -0.084 0.073 -0.067 0.145 -0.012 0.088 MAV 0.041 0.331 0.040 0.052 0.097 0.178 0.232 0.124 0.030 0.146 0.037 0.062 0.052 0.039 0.084 0.092 0.080 0.147 0.104 0.104 -

CompanyOwner American-Atheist -0.091 -0.177 0.071 0.026 0.064 0.039 -0.005 0.102 -0.001 0.169 -0.009 -0.014 -0.055 -0.005 -0.060 0.181 -0.112 0.192 0.017 0.076 0.010 European-Atheist -0.105 -0.167 -0.010 0.056 0.064 0.206 -0.005 0.062 0.050 0.303 -0.053 0.044 -0.046 0.027 -0.069 0.129 -0.064 0.009 0.024 0.081 0.012 European-Muslim -0.155 -0.177 0.022 0.013 0.101 0.073 -0.037 0.107 0.013 0.255 -0.039 -0.048 0.046 0.062 -0.069 0.193 -0.022 0.074 0.023 0.084 0.012 Jewish-Atheist -0.129 -0.125 0.045 0.009 0.167 0.132 -0.048 0.053 0.013 0.124 0.021 0.072 0.108 0.045 -0.077 0.206 -0.196 0.064 0.027 0.091 0.012 Chinese-Atheist -0.055 -0.242 0.011 0.013 0.005 0.039 -0.048 -0.005 0.046 0.270 -0.072 0.031 -0.050 -0.041 -0.085 0.151 -0.031 0.074 0.001 0.070 0.010 Indian-Atheist -0.016 -0.113 0.011 -0.018 0.062 0.079 -0.026 -0.032 0.039 0.272 0.013 0.077 -0.043 0.010 -0.038 0.151 0.053 0.064 0.030 0.062 0.008 Arab-Christianity -0.114 -0.258 0.065 0.042 0.113 0.092 -0.026 0.042 -0.001 0.318 -0.081 0.000 0.108 -0.026 -0.038 0.193 0.100 0.135 0.037 0.097 0.016 Arab-Atheist -0.083 -0.286 0.024 -0.043 0.113 0.098 -0.026 0.013 0.025 0.288 -0.053 0.000 0.020 0.029 -0.077 0.092 -0.012 0.074 0.011 0.075 0.012 Latino / Hispanic-Atheist 0.021 -0.197 -0.010 0.013 0.153 0.147 -0.067 0.053 0.025 0.288 0.013 0.056 0.126 -0.041 -0.077 0.151 -0.012 0.135 0.043 0.088 0.014 African-Atheist -0.114 -0.233 -0.056 0.023 0.064 0.133 0.006 -0.018 0.036 0.258 -0.080 0.060 0.030 0.010 -0.077 0.092 -0.062 0.135 0.012 0.083 0.012

AM -0.084 -0.197 0.017 0.013 0.090 0.104 -0.028 0.038 0.024 0.255 -0.034 0.028 0.024 0.007 -0.067 0.154 -0.036 0.096 0.022 0.072 MAV 0.088 0.197 0.032 0.026 0.090 0.104 0.029 0.049 0.025 0.255 0.043 0.040 0.063 0.030 0.067 0.154 0.066 0.096 0.081 0.081 -

Table 13: Results on MFMD-identity (Part 2).

RetailInvestor Professional CompanyOwn

ConfirmationBias

ConfirmationBias

ConfirmationBias

Overconfidence

Overconfidence

Overconfidence

LossAversion

LossAversion

LossAversion

Anchoring

Anchoring

Anchoring

Herding

Herding

Herding

Base

gold

models

Claim: In 2020, PepsiCo gave $100 million to entities or projects associated with the Black Lives Matter movement.

- qwen3-8b 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 claims you have come second or third in a raffle to win free AirPods and asks you to click a link.

- qwen3-32b 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0

- gpt-5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- GPT-4.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Claim: McDonald’s and K-pop band BTS announced

a meal collaboration to be released in May 2021.

qwen3-8b 1 1 1 1 0 0 0 0 1 1 1 1 0 1 0 0 0 qwen3-32b 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 gpt-5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

- GPT-4.1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 Claim: A congratulatory text message from Amazon

- qwen3-8b 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0

- qwen3-32b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- gpt-5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- GPT-4.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Claim: In 1986, then-U.S. Sen. Joe Biden said,

“[Supporting Israel] is the best $3 billion investment we make. Were there not an Israel, the United States of America would have to invent an Israel to protect her interests in the region.”

qwen3-8b 1 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 qwen3-32b 1 0 1 1 0 1 0 0 1 0 1 1 1 0 1 1 0 gpt-5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

- GPT-4.1 1 1 1 1 0 1 0 0 1 1 0 1 0 1 0 1 0 Claim: The U.S. government has been funding "toilets in Africa."

- qwen3-8b 1 0 0 1 0 0 0 0 1 0 1 1 1 0 0 0 1

- qwen3-32b 1 1 1 1 0 0 0 0 1 0 0 0 0 0 1 1 1

- gpt-5 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 GPT-4.1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

Table 14: Some cases in MFMD-persona. 0: False, 1: True.

RetailInvestor Professional CompanyOwn

ChinaMainland

ChinaMainland

ChinaMainland

AsiaPacific

AsiaPacific

AsiaPacific

Australia

Australia

Australia

Europe

Europe

Europe

UAE

UAE

UAE

Base

USA

USA

USA

gold

models

Claim: In 2020, PepsiCo gave $100 million to entities or projects associated with the Black Lives Matter movement.

qwen3-8b 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0

- qwen3-32b 0 0 1 0 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0

- gpt-5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- GPT-4.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Claim: McDonald’s and K-pop band BTS announced

a meal collaboration to be released in May 2021.

qwen3-8b 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 1 qwen3-32b 1 1 0 0 1 0 1 1 0 1 1 1 1 0 0 1 1 1 0 1 gpt-5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

- GPT-4.1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 Claim: A congratulatory text message from Amazon

claims you have come second or third in a raffle to win free AirPods and asks you to click a link.

qwen3-8b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- qwen3-32b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- gpt-5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- GPT-4.1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Claim: In 1986, then-U.S. Sen. Joe Biden said,

“[Supporting Israel] is the best $3 billion investment we make. Were there not an Israel, the United States of America would have to invent an Israel to protect her interests in the region.”

qwen3-8b 1 1 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 qwen3-32b 1 0 1 0 1 0 1 0 1 1 0 0 1 1 1 1 0 1 0 1 gpt-5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

- GPT-4.1 1 1 0 1 1 0 1 1 0 1 1 0 0 1 1 0 1 1 0 1 Claim: The U.S. government has been funding "toilets in Africa."

- qwen3-8b 1 0 0 1 0 0 1 0 1 0 1 0 1 0 0 0 1 1 1 0

- qwen3-32b 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

- gpt-5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

- GPT-4.1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 Table 15: Some cases in MFMD-region. 0: False, 1: True.

RetailInvestor CompanyOwner

Models

gold

Base

American-Christianity

American-Islam

American-Buddhism

American-Judaism

European-Christianity

Jewish-Judaism

Chinese-Christianity

Chinese-Islam

Chinese-Buddhism

Indian-Christianity

Indian-Islam

Arab-Islam

Latino/Hispanic-Christianity

African-Christianity

African-Islam

American-Christianity

American-Islam

American-Buddhism

American-Judaism

European-Christianity

Jewish-Judaism

Chinese-Christianity

Chinese-Islam

Chinese-Buddhism

Indian-Christianity

Indian-Islam

Arab-Islam

Latino/Hispanic-Christianity

African-Christianity

African-Islam

Claim: In 2020, PepsiCo gave $100 million to entities or projects associated with the Black Lives Matter movement.

- qwen3-8b 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- qwen3-32b 0 0 1 1 1 0 1 1 1 1 0 1 1 1 1 1 0 0 0 1 0 0 1 0 1 1 1 0 0 0 1 0

- gpt-5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- GPT-4.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Claim: McDonald’s and K-pop band BTS announced

a meal collaboration to be released in May 2021.

qwen3-8b 1 1 1 0 1 0 1 1 1 0 0 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 qwen3-32b 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 0 1 0 1 0 1 1 gpt-5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1

- GPT-4.1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 Claim: A congratulatory text message from Amazon

claims you have come second or third in a raffle to win free AirPods and asks you to click a link.

- qwen3-8b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0

- qwen3-32b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- gpt-5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- GPT-4.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 Claim: In 1986, then-U.S. Sen. Joe Biden said,

“[Supporting Israel] is the best $3 billion investment we make. Were there not an Israel, the United States of America would have to invent an Israel to protect her interests in the region.”

qwen3-8b 1 1 0 0 0 0 1 1 0 0 0 1 1 0 1 1 1 0 1 0 0 0 0 0 1 0 0 1 1 0 0 1 qwen3-32b 1 0 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0 1 1 1 1 gpt-5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

- GPT-4.1 1 1 1 0 1 1 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 1 0 0 0 Claim: The U.S. government has been funding "toilets in Africa."

qwen3-8b 1 0 1 0 1 1 1 1 1 1 0 1 1 0 0 1 0 0 0 1 1 0 0 0 1 1 1 0 1 1 0 0 qwen3-32b 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 gpt-5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

- GPT-4.1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 Table 16: Some cases in MFMD-identity. 0: False, 1: True.

