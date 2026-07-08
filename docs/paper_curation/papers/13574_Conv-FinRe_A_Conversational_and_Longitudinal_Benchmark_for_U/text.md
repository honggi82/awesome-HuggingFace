# arXiv:2602.16990v2[cs.AI]17May2026

## Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation

Xueqing Peng∗

Lingfei Qian∗

Yueru He

Yi Han

Yan Wang

The Fin AI USA xueqing.peng@yale.edu

Columbia University USA

The Fin AI USA lingfei.qian@yale.edu

Georgia Institute of Technology USA

The Fin AI USA wy2266336@gmail.com

Dongji Feng∗

California State University USA dfeng@csumb.edu

Zhuohan Xie

MBZUAI UAE

Vincent Jim Zhang

The Fin AI USA

Rosie Guo

The Fin AI USA

Fengran Mo

University of Montreal Canada

Yankai Chen∗

Jimin Huang

The University of Manchester Manchester, United Kingdom

McGill University MBZUAI Canada

yankaichan3@gmail.com

The Fin AI USA

### Abstract

Most recommendation benchmarks evaluate how well a model imitates user behavior. In financial advisory, however, observed actions can be noisy or short-sighted under market volatility and may conflict with a user’s long-term goals. Treating what users chose as the sole ground truth, therefore, conflates behavioral imitation with decision quality. We introduce Conv-FinRe, a conversational and longitudinal benchmark for stock recommendation that evaluates LLMs beyond behavior matching. Given an onboarding interview, step-wise market context, and advisory dialogues, models must generate rankings over a fixed investment horizon. Crucially, ConvFinRe provides multi-view references that distinguish descriptive behavior from normative utility grounded in investor-specific risk preferences, enabling diagnosis of whether an LLM follows rational analysis, mimics user noise, or is driven by market momentum. We build the benchmark from real market data and human decision trajectories, instantiate controlled advisory conversations, and evaluate a suite of state-of-the-art LLMs. Results reveal a persistent tension between rational decision quality and behavioral alignment: models that perform well on utility-based ranking often fail to match user choices, whereas behaviorally aligned models

∗Corresponding authors.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

Xue (Steve) Liu

McGill University MBZUAI Canada

Jian-Yun Nie

University of Montreal Canada

can overfit short-term noise. The dataset is publicly released on Hugging Face1, and the codebase is available on GitHub2.

### CCS Concepts

• Information systems → Recommender systems; Similarity measures; Language models; • Human-centered computing → User centered design; • Computing methodologies → Natural language processing.

### Keywords

Personality Stock Recommendation, Conversational Benchmark, Utility Function, Rerank, Large Language Models

ACM Reference Format:

Yan Wang, Yi Han, Lingfei Qian, Yueru He, Xueqing Peng, Dongji Feng, Zhuohan Xie, Vincent Jim Zhang, Rosie Guo, Fengran Mo, Jimin Huang, Yankai Chen, Xue (Steve) Liu, and Jian-Yun Nie. 2018. Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 8 pages. https://doi.org/XXXXXXX.XXXXXXX

### 1 Introduction

Large language models (LLMs) have achieved remarkable progress across diverse application domains, demonstrating strong performance in language understanding [24, 43], reasoning [25, 44], and structured problem-solving [39, 40]. These capabilities have motivated their adoption as assistants for decision-making and recommendation tasks [7, 9, 15, 18, 19]. In most recommendation benchmarks, personalization is primarily measured by behavioral imitation: a recommendation is deemed correct if it matches what a user would click, rate, or choose [6, 14, 16, 28]. This behavior-centric

- 1https://huggingface.co/collections/TheFinAI/conv-finre
- 2https://github.com/The-FinAI/Conv-FinRe

- Table 1: Comparison of representative user-centric recommendation benchmarks. Dynamic indicates timeconditioned signals; Utility refers to relevance grounded in a user-dependent decision utility; Multi-view indicates multiple, potentially conflicting ranking views; Dialogue denotes a conversational interactive setting.

Work Domain Dynamic Utility Multi-view Dialogue

REASONER [6] Video FairEval [28] Music PerFairX [27] Movie&Music CEREAL [14] Movie LLM-REDIAL [16] Movie RecBench [18] General FAR-Trans [29] Finance

Ours Finance

paradigm is effective in many consumer domains, where feedback is a reliable proxy for utility.

Financial recommendation is different. Investor actions are often affected by short-term market noise, emotions, and shifting constraints, and may deviate from stable risk tolerance or long-term objectives [1, 22]. As a result, matching historical choices alone cannot tell whether an advisor is providing good financial guidance. A faithful mimic of noisy actions may be misaligned with the user’s underlying goals, while a purely rational advisor may ignore user intent and preferences.

Existing benchmarks, as summarized in Table 1, therefore struggle with three issues: behavior-as-truth, utility blindness, and singleview evaluation. Most user-centric benchmarks rely on relevance signals (clicks/ratings) [6, 18] without utility grounding, while finance datasets often emphasize prediction or trading objectives rather than user-specific decision quality [29]. Consequently, they cannot diagnose whether an LLM advisor is reasoning about risksensitive utility, blindly chasing market trends, or simply overfitting user noise.

To address this gap, we introduce Conv-FinRe, the first conversational and longitudinal benchmark that formulates financial recommendation as a multi-view alignment problem. Rather than evaluating whether an LLM simply matches user choices, the benchmark assesses model rankings against four complementary reference views, such as user choice (𝑦𝑢𝑠𝑒𝑟), rational utility (𝑦𝑢𝑡𝑖𝑙), market momentum (𝑦𝑚𝑜𝑚), and risk sensitivity (𝑦𝑠𝑎𝑓 𝑒), enabling diagnosis of whether a model relies on rational analysis, behavioral imitation, or short-term market signals. To support such evaluation, userspecific risk preferences are inferred from longitudinal decision trajectories via inverse optimization and used to construct utilityand risk-based reference rankings, without exposing the latent utility function to the model. Operationally, Conv-FinRe instantiates the task through onboarding interviews and step-wise advisory dialogues, where an LLM must reconcile competing advisory principles over time.

We evaluate a diverse set of state-of-the-art LLMs under ConvFinRe and reveal a fundamental tension between rational decision quality and behavioral alignment. While some models achieve strong utility-based rankings, they often conflate long-term risk with short-term market momentum; conversely, domain-specialized

models tend to overfit noisy user actions, mistaking transient behavior for stable preferences. Together, these results motivate ConvFinRe as a benchmark for multi-view, utility-grounded evaluation.

We make the following contributions: (1) We introduce ConvFinRe, a conversational and longitudinal benchmark for stock recommendation that evaluates LLMs beyond behavioral imitation by grounding assessment in investor-specific utility. (2) We formulate financial recommendation evaluation as a multi-view alignment problem and provide a diagnostic framework, supported by inverse optimization, that disentangles behavioral alignment from rational decision quality. (3) We conduct a systematic evaluation of stateof-the-art LLMs under Conv-FinRe, identifying distinct advisory behavior patterns under competing market signals and user noise.

### 2 Related Works

Personalized recommendation benchmarks are well studied in consumer domains such as e-commerce and media, where personalization is typically modeled from interaction histories or coarse user traits and supervision relies on a single relevance signal like ratings or clicks [6, 42]. Recent work enriches this setting by introducing structured explanations [6] and by evaluating LLMs either as representation enhancers or as end-to-end recommenders under point-wise, pair-wise, or list-wise protocols [7, 9, 18, 19], with further analysis of fairness, bias, and sequential alignment [17, 28]. In contrast, personalized stock recommendation poses additional challenges due to non-stationary assets and investor objectives constrained by risk tolerance and return–risk trade-offs [29, 34], with evidence that risk preferences evolve and must be inferred from behavior rather than static profiles [5]. While conversational and LLM-based financial advisors enable iterative preference elicitation [10, 23, 30, 33, 35, 36], existing benchmarks rarely evaluate recommendation quality using investor-specific utility as the core supervision signal, limiting their ability to assess true decision alignment.

### 3 Conv-FinRe

Figure 1 illustrates the overall pipeline of Conv-FinRe, from data collection and user profiling to multi-view conversation simulation and evaluation. The framework models longitudinal advisory interactions by integrating market signals, inferred user preferences, and competing expert recommendations, enabling fine-grained analysis of LLM alignment in personalized financial decision-making.

### 3.1 Task Formulation

We define the Multi-view Longitudinal Stock Recommendation task, which simulates iterative interactions between a personalized investment advisor and a user over a fixed investment horizon𝑇. Unlike conventional recommendation benchmarks that rely on a single "gold-standard" label, our task evaluates LLM alignment across four complementary reference views: User Choice (𝑦𝑢𝑠𝑒𝑟), representing the empirical selections made by the human participant; Rational Utility (𝑦𝑢𝑡𝑖𝑙), an idealized ranking derived from a calibrated utility function that represents the theoretically optimal balance between return and risk; Market Momentum (𝑦𝑚𝑜𝑚), a profit-oriented ranking based purely on recent cumulative returns; and Risk Sensitivity (𝑦𝑠𝑎𝑓 𝑒), a conservative ranking that

[Figure 1]

#### Figure 1: Overview of Conv-FinRe benchmark.

isolates the user’s specific risk-avoidance component by penalizing volatility and downside risk according to their inferred sensitivity.

In this setting, the LLM acts as a Personalized Investment Advisor whose objective is not explicitly specified. Instead, the model must infer the user’s latent financial preferences over time and reconcile conflicting advisory principles. At each decision step, the model is presented with the current market context, historical interaction trajectory, and recommendations from a panel of three specialized advisors grounded in 𝑦𝑢𝑡𝑖𝑙, 𝑦𝑚𝑜𝑚, and 𝑦𝑠𝑎𝑓 𝑒. The core challenge for the LLM is to synthesize these heterogeneous signals to produce a final ranking that reflects its inferred understanding of the user’s underlying financial objectives. This multi-view design enables us to diagnose whether an LLM’s misalignment with actual user behavior stems from an over-reliance on market momentum or a miscalculation of the user’s specific risk thresholds.

Formally, for the user 𝑖 at step 𝑡, given a candidate stock set S𝑡 and decision context I𝑡𝑖, the recommendation process is defined as:

𝜋𝑖,𝑡 = 𝑓𝜃 (I𝑡𝑖, S𝑡), (1)

where 𝑓𝜃 denotestheevaluatedLLM-basedadvisor,𝜋𝑖,𝑡 is thereranked list from the S𝑡. Specifically, the decision context I𝑡𝑖 is shown below:

I𝑡𝑖 = 𝑃𝑖, H1:𝑖𝑡−1, M𝑡 , (2)

where 𝑃𝑖 denotes the onboarding dialogue, a multi-turn introductory interaction used to elicit the user 𝑖’s background, financial goals, and initial risk tolerance. M𝑡 = {v𝑠,𝑡, x𝑠,𝑡}𝑠∈S𝑡 represents the current market state. Here, v𝑠,𝑡 = (𝜇𝑠,𝜎𝑠2, Drawdown𝑠) denotes the vector of raw performance metrics used for preference grounding, while x𝑠,𝑡 comprises the verbalized market signals (e.g., price trends, percentage returns, and volatility) presented to the LLM advisor. The term H1:𝑖𝑡−1 encapsulates the longitudinal interaction trajectory, consisting of multi-turn dialogues between the user and the three specialized advisors (Rational Utility, Market Momentum, and Risk Sensitivity). This history provides the model with both the user’s previous decision patterns and the conflicting advisory signals previously encountered.

### 3.2 Latent Preference Grounding via Inverse Optimization

The task itself does not assume access to the user’s true utility function. Instead, to enable principled analysis of advisory behaviors, we construct a latent preference signal that serves as a reference representation of each user’s underlying risk attitude. This preference signal is used to characterize advisory objectives within the benchmark, rather than being exposed to the model.

We assume that user 𝑖’s decision-making process is governed

by a latent utility function 𝑈𝑖,𝑡(𝑠), which balances expected return against volatility and downside risk [20, 26, 38]. For user𝑖 and stock

𝑠 ∈ S𝑡 at step 𝑡, the utility is defined as:

𝑈𝑖,𝑡(𝑠) = 𝜇˜𝑠,𝑡 − 𝜆𝑖𝜎˜𝑠,𝑡2 −𝛾𝑖Drawdown˜ 𝑠,𝑡, (3)

where 𝜇˜𝑠,𝑡, 𝜎˜𝑠,𝑡2 , and Drawdown˜ 𝑠,𝑡 denote the cross-sectionally standardized mean return, variance, and maximum drawdown of stock

𝑠 over a 7-day window preceding step 𝑡. (𝜆𝑖,𝛾𝑖) are the user-specific parameters, which are assumed to be time-invariant and capture the user’s sensitivity to volatility and downside risk.

We estimate (𝜆𝑖,𝛾𝑖) via Inverse Optimization [3, 4] using the

user’s longitudinal behavioral trajectory 𝐻1:𝑖 𝑇. Assuming a rational choice model with Gumbel-distributed noise, the probability

that user 𝑖 selects stock 𝑠∗ at step 𝑡 follows a Multinomial Logit model [32]:

exp 𝑈𝑖,𝑡(𝑠∗) 𝑠∈S𝑡 exp 𝑈𝑖,𝑡(𝑠)

, (4)

𝑃(𝑠∗ | 𝜆𝑖,𝛾𝑖, M𝑡) =

The global parameters are obtained by minimizing the Regularized Negative Log-Likelihood [2, 21] over the interaction horizon:

∑︁𝑇

log𝑃 𝑠𝑖,𝑡∗ | 𝜆𝑖,𝛾𝑖, M𝑡

L𝑖(𝜆𝑖,𝛾𝑖) = −

(5)

𝑡=1

+ 𝛼 ∥(𝜆𝑖,𝛾𝑖)∥22 ,

(𝜆𝑖,𝛾𝑖) = argmin 𝜆𝑖,𝛾𝑖

L𝑖. (6)

where 𝑠𝑖,𝑡∗ is the stock actually chosen by user 𝑖 at step 𝑡, and 𝛼 controls the strength of regularization.

Once (𝜆𝑖,𝛾𝑖) are estimated, we construct the reference views for evaluation. Specifically, the Rational Utility view (𝑦𝑢𝑡𝑖𝑙) is ranked by the full utility 𝑈𝑖,𝑡(𝑠), while the Risk Sensitivity view (𝑦𝑠𝑎𝑓 𝑒) is ranked by the personalized risk penalty term: 𝑅𝑖,𝑡(𝑠) = 𝜆𝑖𝜎˜𝑠,𝑡2 +𝛾𝑖Drawdown˜ 𝑠,𝑡.

### 3.3 Data Collection and Conversation Simulation

3.3.1 Data Collection. MarketData. We constructa compactstockuniverse from S&P 500

constituents using stratified sampling to ensure coverage of all eleven GICS sectors and balanced exposure to systematic risk. Candidate stocks are grouped by market beta, computed from five-year monthly returns, into low (𝛽 < 1), moderate (𝛽 ≈ 1), and high (𝛽 > 1) risk regimes, with approximately equal representation from

#### Table 2: Summary of the stock universe used in Conv-FinRe. Stocks are grouped by volatility tier based on market beta to ensure balanced exposure to systematic risk.

Ticker Company GICS Sector Beta Market Cap Low Volatility (𝛽 < 1)

PG Procter & Gamble Consumer Staples 0.36 $353B MRK Merck & Co., Inc. Health Care 0.38 $274B VZ Verizon Communications Inc. Communication Services 0.36 $188B Moderate Volatility (𝛽 ≈ 1)

LIN Linde plc Materials 0.95 $213B XOM Exxon Mobil Corporation Energy 0.95 $596B High Volatility (𝛽 > 1)

JPM JPMorgan Chase & Co. Financials 1.13 $841B AMZN Amazon.com, Inc. Information Technology 1.31 $2.6T MMM 3M Company Industrials 1.10 $81B SPG Simon Property Group, Inc. Real Estate 1.53 $62B TSLA Tesla, Inc. Consumer Discretionary 2.07 $1.4T

each group. The resulting universe comprises ten representative stocks (Table 2), enabling controlled longitudinal evaluation.

For each stock, we collect daily and intraday price data over a 30-day horizon (Aug. 6–Sep. 17, 2025) via the Yahoo Finance API3, which defines the market state in the simulation environment.

User Interaction Data. User interaction data are collected through a two-stage protocol. First, we obtain static user profiles from 10 participants via a structured questionnaire4 capturing investor demographics, financial capacity, investment experience, and risk attitudes. The questionnaire design follows regulatory suitability guidelines (MiFID II Article 255; FINRA Rule 21116) and is informed by Betterment’s approach7 on investor risk preferences and financial decision-making; and industry examples, such as JPMorgan Financial Health Check8, Charles Schwab Investor Risk Profile9, Vanguard Investor Financial Profile10, and Financial Group Plan11.

Second, we collect longitudinal decision trajectories using a custom asset simulation tool. Participants interact with a fixed universe of ten stocks over a 30-day horizon, observing daily and intraday price movements. At each step, users make incremental buy decisions and receive portfolio-level feedback, including realized returns and volatility, which is logged together with their actions to form a temporally ordered interaction trace. The simulation tool is publicly released for reproducibility12.

3.3.2 Conversation Simulation.

Conversation Generation. Building on the collected user interaction data, we construct a structured conversation simulation to instantiate the proposed longitudinal advisory task in a languagebased setting. Rather than collecting free-form dialogues from participants, we transform each user’s observed profile and behavioral

- 3https://pypi.org/project/yfinance/
- 4https://forms.gle/g7GnwqByq7mCoJgTA
- 5https://eur-lex.europa.eu/eli/dir/2014/65/oj/
- 6https://www.finra.org/rules-guidance/rulebooks/finra-rules/2111
- 7https://d-nb.info/116404222X
- 8https://am.jpmorgan.com/content/dam/jpm-am-aem/asiapacific/hk/en/literature/ account-forms/healthcheck_corporate.pdf
- 9https://www.studocu.vn/vn/document/royal-melbourne-institute-of-technologyvietnam/international-trade/charles-schwab-investor-risk-profile/100562992
- 10https://sustainableinvest.com/wp-content/uploads/Investor-Financial-ProfileQuestionnaire.pdf
- 11https://financialgroup.com/risk-profile-bq
- 12https://huggingface.co/spaces/TheFinAI/LetYourProfitsRun

#### Table 3: Structural statistics of Conv-FinRe benchmark. Token counts are computed using the cl100k_base tokenizer.

Scale Dialogue Users 10 Min Turns / Instance 4 Steps / User 23 Max Turns / Instance 26 Total Instances 230 Avg Turns / Instance 15

###### Turn Accounting Token Status

Unique Turns 270 Min Tokens / Instance 1,818 Prefix-Expanded Turns 3,450 Max Tokens / Instance 7,252

Avg Tokens / Instance 4,320.2

trajectory into a coherent multi-turn advisory conversation, enabling controlled and reproducible evaluation of LLMs. Each user trajectory is organized into two phases: an onboarding interview and a longitudinal advisory dialogue over a fixed horizon𝑇.

The onboarding phase verbalizes the static user profile obtained fromthequestionnaire.Usingtheuser’s survey responses as grounding signals, we generate a four-turn advisor–user dialogue that captures the user’s financial background, constraints, investment goals, and emotional reactions to risk. The generated language is constrained to match the user’s reported financial literacy, ensuring that preference signals are conveyed implicitly through natural expression rather than explicit financial terminology. This onboarding dialogue serves as the conversational realization of 𝑃𝑖 in the decision context I𝑡𝑖.

Following onboarding, the longitudinal advisory phase reflects the user’s sequential decision-making behavior observed in the asset simulation. At each step 𝑡, the conversation conditions on the historical interaction trajectory H1:𝑖𝑡−1 and the current market state M𝑡, represented by a 7-day market snapshot derived from the simulation environment. A panel of three specialized advisors provides recommendations based on the heterogeneous principles defined in our multi-view framework: Rational-Utility, MarketMomentum, and Risk-Sensitivity strategies. The user then finalizes a choice, potentially deviating from advisor suggestions and providing a subjective justification consistent with their observed behavior in the simulation. All advisor messages and user responses are appended to the conversation history, yielding a temporally ordered dialogue aligned with the underlying interaction trace. Overall, as shown in Table 3, the benchmark contains 10 users and 230 prefix-conditioned instances. Dialogue context grows from 4 to 26 turns (15 on average), yielding 270 unique turns that expand to 3,450 prefix-conditioned turns. Each instance contains 1,818–7,252 tokens (4,320.2 on average), highlighting the substantial longitudinal context complexity required for evaluation.

Conversation Quality Validation. We validate the quality of the simulated conversations from both a preference-grounding and a conversational realism perspective.

User Preference Consistency Validation: For each user 𝑖, the inferred latent parameters (𝜆𝑖,𝛾𝑖) represent the user’s sensitivity to volatility and downside risk. To assess whether these parameters meaningfully capture the user’s true investment psychology, financial experts translate each parameter pair into a concise naturallanguage summary of the user’s risk tolerance. Users are then asked to rate, on a 0–9 Likert scale (0 = not at all accurate, 9 = perfectly accurate), how well the summary reflects their own reasoning and emotions when making stock selection decisions. Across users, the

summaries receive a high average agreement score of 7.8, with low inter-user variance, indicating that the inferred preferences are largely consistent with users’ decision-making behavior.

Conversational Plausibility Validation: We further assess the realism of the simulated advisory dialogues. To avoid over-counting highly correlated interactions within the same user trajectory, we adopt a user-level sampling strategy. Specifically, for each of the 10 users, we randomly sample one decision step from the 23-step longitudinal conversation, resulting in 10 representative dialogue instances. Each sampled conversation is independently evaluated by three domain experts with backgrounds in finance and conversational systems. Experts rate each dialogue on a 0–9 Likert scale along four dimensions: role consistency, linguistic naturalness, behavioral plausibility of user responses, and cross-turn coherence. The final plausibility score for each conversation is computed as the average across the four dimensions and across experts. Overall, the simulated dialogues achieve a mean plausibility score of 8.1, indicating that the generated conversations closely resemble realistic financial advisory interactions rather than scripted or artificial exchanges.

### 3.4 Evaluation Metrics

Utility-based NDCG (uNDCG): We first evaluate whether modelgenerated rankings align with the user’s latent, utility-grounded preference structure. For user 𝑖 at step 𝑡, we compute uNDCG using

the calibrated utility 𝑈𝑖,𝑡(𝑠) as relevance. Given a ranking 𝜋𝑖,𝑡 over candidate set S𝑡, the discounted cumulative gain is:

∑︁|S𝑡 |

𝑈𝑖,𝑡(𝜋𝑖,𝑡[𝑘]) log2(𝑘 + 1)

DCG𝑖,𝑡 =

, (7)

𝑘=1

and the utility-based NDCG is defined as: uNDCG𝑖,𝑡 =

DCG𝑖,𝑡 IDCG𝑖,𝑡

, (8) where IDCG𝑖,𝑡 is computed from the utility-optimal ranking.

MRR and Hit Rate: To assess recovery of the user’s observed choice, let 𝑠𝑖,𝑡∗ denote the stock selected by user 𝑖 at step 𝑡, and let 𝜋𝑖,𝑡 (𝑠𝑖,𝑡∗ ) be its position in 𝜋𝑖,𝑡. The reciprocal rank is

1 𝜋𝑖,𝑡 (𝑠𝑖,𝑡∗ )

RR𝑖,𝑡 =

, (9)

with Mean Reciprocal Rank (MRR) obtained by averaging across users and steps. We additionally report Hit Rate at top-𝐾:

HR@𝐾𝑖,𝑡 = I 𝜋𝑖,𝑡 (𝑠𝑖,𝑡∗ ) ≤ 𝐾 , (10) and focus on 𝐾 ∈ {1, 3}, where I[·] denotes the indicator function. Expert Alignment Score (EAS): To analyze how models resolve competing advisory principles, we measure alignment with three expert rankings: Rational Utility, Market Momentum, and Risk Sensitivity. For model 𝑚, we compute step-wise alignment using Kendall’s 𝜏:

𝑖,𝑡 (𝑚) = 𝜏 𝜋𝑖,𝑡𝑚,𝜋𝑖,𝑡𝑒 , (11) where 𝑒 denotes the expert type. Final scores are obtained by averaging over all users and steps:

EAS(𝑒)

∑︁𝑁

##### ∑︁𝑇

1 𝑁𝑇

EAS(𝑒)(𝑚) =

EAS(𝑒)

𝑖,𝑡 (𝑚). (12)

𝑖=1

𝑡=1

#### Table 4: Overall performance on the longitudinal stock advisory task. The Random baseline is computed by averaging over 1,000 uniform random permutations per instance, serving as a sanity-check lower bound.

Model uNDCG↑ MRR↑ HR@1↑ HR@3↑ Random 0.73 ± 0.00 0.29 ± 0.01 0.10 ± 0.01 0.30 ± 0.01 GPT-5.2 0.94 ± 0.03 0.46 ± 0.02 0.29 ± 0.03 0.51 ± 0.03 GPT-4o 0.94 ± 0.00 0.56 ± 0.03 0.42 ± 0.03 0.60 ± 0.03 DeepSeek-V3.2 0.92 ± 0.00 0.51 ± 0.03 0.37 ± 0.03 0.55 ± 0.03 Qwen3-235B-A22B-Instruct 0.94 ± 0.00 0.47 ± 0.02 0.30 ± 0.03 0.52 ± 0.03 Qwen2.5-72B-Instruct 0.92 ± 0.01 0.63 ± 0.03 0.50 ± 0.03 0.69 ± 0.03 Llama-3.3-70B-Instruct 0.97 ± 0.00 0.52 ± 0.03 0.36 ± 0.03 0.59 ± 0.03 Llama3-XuanYuan3-70B-Chat 0.92 ± 0.00 0.65 ± 0.03 0.54 ± 0.03 0.69 ± 0.01

- 3.5 Evaluation Models

Our goal is to assess the capabilities and limitations of contemporary LLMs in conversational, personality-grounded longitudinal stock recommendation under the Conv-FinRe benchmark. To this end, we evaluate a diverse set of state-of-the-art LLMs, covering both proprietary and open-source families. Specifically, we include two closedsource general-purpose models, GPT-5.2 [31] and GPT-4o [13]. We further evaluate a range of open-source general models with strong reasoning and instruction-following capabilities, including DeepSeek-V3.2 [8], Qwen3-235B-A22B-Instruct [37], Qwen2.572B-Instruct [41], and Llama-3.3-70B-Instruct [12], and one financial domain conversational model: Llama3-XuanYuan3-70BChat13. All models are evaluated using the LM Evaluation Harness [11] under a unified interface. Proprietary models are accessed via official APIs, while open-source models are executed locally. Across all experiments, we standardize the maximum input context length to 8,192 tokens and constrain the maximum generation length to 126 tokens, ensuring fair and consistent comparison across models.

4 Experiments and Results

- 4.1 Overall Performance

Table4showsthatmostmodels achieve high uNDCG scores (0.92–0.97), indicating a strong baseline for ranking assets according to the Rational Utility. However, high uNDCG does not always translate into better recovery of the User Choice. While Llama-3.3-70B-Instruct leads in uNDCG (0.97), it shows lower Hit Rates, suggesting it prioritizes an "idealized" rational recommendation that balances long-term risk and return.

In contrast, Qwen2.5-72B-Instruct and Llama3-XuanYuan3-70BChat excel in MRR and HR@K, indicating they are more effective at mimicking the user’s realized, and often noisy, decision-making patterns. This gap reveals a fundamental trade-off: 𝑦𝑢𝑡𝑖𝑙 acts as a financially robust reference to align with the user’s latent psychology, whereas 𝑦𝑢𝑠𝑒𝑟 captures the empirical behavior which may deviate from pure rationality. The results suggest that models must navigate the tension between providing the most rational advice and maintaining empathetic behavioral alignment.

4.2 Expert Alignment Analysis

###### Table 5 reveals how models resolve competing advisory principles. A prominent trend is the strong coupling between Rational Utility

13https://huggingface.co/Duxiaoman-DI/Llama3-XuanYuan3-70B-Chat

#### Table 5: Alignment of model-generated rankings with heterogeneous advisory principles.

Model 𝜏(Utility)↑ 𝜏(Momentum)↑ 𝜏(Risk)↑ Random 0.00 ± 0.01 0.00 ± 0.01 0.00 ± 0.01 GPT-5.2 0.59 ± 0.02 0.56 ± 0.02 0.28 ± 0.02 GPT-4o 0.60 ± 0.02 0.60 ± 0.02 0.20 ± 0.02 DeepSeek-V3.2 0.51 ± 0.02 0.49 ± 0.02 0.26 ± 0.02 Qwen3-235B-A22B-Instruct 0.56 ± 0.02 0.55 ± 0.02 0.26 ± 0.02 Qwen2.5-72B-Instruct 0.52 ± 0.02 0.49 ± 0.02 0.22 ± 0.02 Llama-3.3-70B-Instruct 0.74 ± 0.02 0.73 ± 0.01 0.17 ± 0.02 Llama3-XuanYuan3-70B-Chat 0.47 ± 0.02 0.46 ± 0.02 0.15 ± 0.02

and Market Momentum alignment, which stems from the contextual collinearity of these signals during trending markets where high-momentum assets often dominate utility calculations. Llama-

- 3.3-70B-Instruct exemplifies this trend by achieving the highest alignment with both Utility and Momentum, yet its sharp decline in Risk alignment proves that it struggles to decouple downside protection from growth-oriented signals.

In contrast, DeepSeek-V3.2 demonstrates the most balanced profile across all evaluated models. By maintaining a stable and relatively high alignment with Safety while avoiding extreme bias toward return-driven metrics, DeepSeek-V3.2 shows a superior ability to integrate conflicting advisory signals into a compromise recommendation. The GPT series also exhibits similar balanced characteristics, though with slightly less consistency than DeepSeek in the safety dimension.

The behavior of Llama3-XuanYuan3-70B-Chat is particularly noteworthy given its background as a domain-specific LLM finetuned on financial corpora. Despite its lower expert alignment scores, it achieves high behavioral hit rates in Table 4, suggesting that its financial expertise manifests as empathetic alignment with User Choices rather than strict adherence to idealized mathematical formulas. XuanYuan3 acts as a seasoned human consultant who prioritizes the pragmatic, albeit noisy, preferences of real-world investors over rigid algorithmic consistency.

- 4.3 Preference Discovery Dynamics

Figure 2 illustrates the step-wise gain (Δ uNDCG) in utility-based alignment when conversational history is accessible. While gains are observed for several models, the dynamics are highly heterogeneous. Models like GPT-5.2 and DeepSeek-V3.2 show significant positive improvements in early to middle stages (steps 1–10), suggesting they successfully extract informative signals about the user’s latent risk preferences from initial interactions. The subsequent fluctuations and general plateauing across most models indicate that while LLMs can form a coarse preference representation early on, the inherent "noise" in longitudinal financial decisions makes consistent, long-term preference tracking challenging.

From a financial advisory perspective, these gains suggest that conversation history allows models to gauge baseline investment styles, identifying whether a user is inherently risk-averse or returnsensitive. However, the plateauing suggests that the marginal utility of additional historical context diminishes once a stable "investor persona" is established, at which point ranking performance becomes more contingent on the immediate market context M𝑡 than on further preference refinement.

[Figure 2]

- Figure 2: Step-wise improvement in utility-based alignment from conversational history.
- Figure 3 provides a finer-grained diagnostic by comparing av-

erage utility alignment with and without longitudinal context. Based on the relative shifts from the diagonal, we identify three archetypes.

Adaptive Advisors (GPT-5.2, DeepSeek-V3.2, Qwen3-235B) show clear improvements when history is available, indicating effective cross-turn preference integration and progressive alignment with the user’s latent risk profile.

Transaction-driven Analysts (GPT-4o, Llama-3.3-70B) remain close to the diagonal, achieving strong utility rankings but exhibiting limited gains from conversational context, suggesting reliance on contemporaneous market signals rather than personalization.

Behavioral Overfitters (Qwen2.5-72B, Llama3-XuanYuan3) experience degraded utility alignment when history is introduced, implying over-sensitivity to noisy user actions and a tendency to prioritize behavioral mimicry over stable preference inference.

The results reveal non-uniform preference discovery and highlight the need to separate surface imitation from genuine decisionutility alignment, as evidenced by XuanYuan’s performance drop.

[Figure 3]

Figure 3: Average utility alignment with and without conversational history.

### 5 Conclusion

We introduce Conv-FinRe, a conversational and longitudinal benchmark that shifts financial recommendation from surface-level behavioral matching to utility-grounded decision alignment. Through

inverse optimization of latent risk preferences, it supports multiview evaluation that separates rational decision quality from observed user behavior. Our results reveal a persistent tension between utility-based ranking and behavioral alignment: generalpurpose LLMs often optimize utility more effectively, while domainspecific models tend to overfit transient user actions. These findings expose the limits of behavior-only evaluation and motivate benchmarks that disentangle long-term investor preferences from short-term market noise.

### References

- [1] R Arran. 2023. Behavioral finance: The psychology behind financial decisionmaking. Business Studies Journal 15, 5 (2023), 1–2.
- [2] M Bertero. 2006. Regularization methods for linear inverse problems. In Inverse Problems: Lectures given at the 1st 1986 Session of the Centro Internazionale Matematico Estivo (CIME) held at Montecatini Terme, Italy, May 28–June 5, 1986. Springer, 52–112.
- [3] Dimitris Bertsimas, Vishal Gupta, and Ioannis Ch Paschalidis. 2012. Inverse optimization: A new perspective on the Black-Litterman model. Operations research 60, 6 (2012), 1389–1403.
- [4] Dimitris Bertsimas, Vishal Gupta, and Ioannis Ch Paschalidis. 2015. Data-driven estimation in equilibrium using inverse optimization. Mathematical Programming 153, 2 (2015), 595–633.
- [5] Agostino Capponi and Zhaoyu Zhang. 2020. Risk Preferences and Efficiency of Household Portfolios. arXiv preprint arXiv:2010.13928 (2020).
- [6] Xu Chen, Jingsen Zhang, Lei Wang, Quanyu Dai, Zhenhua Dong, Ruiming Tang, Rui Zhang, Li Chen, Xin Zhao, and Ji-Rong Wen. 2023. REASONER: an explainable recommendation dataset with comprehensive labeling ground truths. Advances in Neural Information Processing Systems 36 (2023), 14497–14515.
- [7] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering chatgpt’s capabilities in recommender systems. In Proceedings of the 17th ACM Conference on Recommender Systems. 1126–1132.
- [8] DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.
- [9] Dario Di Palma, Giovanni Maria Biancofiore, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia, and Eugenio Di Sciascio. 2023. Evaluating chatgpt as a recommender system: A rigorous approach. arXiv preprint arXiv:2309.03613

(2023).

- [10] Chongming Gao, Wenqiang Lei, Xiangnan He, Maarten De Rijke, and Tat-Seng Chua. 2021. Advances and challenges in conversational recommender systems: A survey. AI open 2 (2021), 100–126.
- [11] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2024. The Language Model Evaluation Harness. doi:10.5281/zenodo.12608602
- [12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783

(2024).

- [13] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
- [14] Jiyoon Lee, Joonghoon Kim, and Pilsung Kang. 2026. CEREAL: personality-driven LLM-based conversational recommendation dataset with contextually-enriched and realistic user interactions. Multimedia Tools and Applications 85, 2 (2026), 47.
- [15] Haohang Li, Yupeng Cao, Yangyang Yu, Shashidhar Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen Jiang, Zining Zhu, Kp Subbalakshmi, Jimin Huang, et al.

2025. Investorbench: A benchmark for financial decision-making tasks with llm-based agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2509–2525.

- [16] Tingting Liang, Chenxin Jin, Lingzhi Wang, Wenqi Fan, Congying Xia, Kai Chen, and Yuyu Yin. 2024. LLM-REDIAL: a large-scale dataset for conversational recommender systems created from user behaviors with llms. In Findings of the Association for Computational Linguistics ACL 2024. 8926–8939.
- [17] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, and Xiang Wang.

2023. Llara: Aligning large language models with sequential recommenders. CoRR

(2023).

- [18] Qijiong Liu, Jieming Zhu, Lu Fan, Kun Wang, Hengchang Hu, Wei Guo, Yong Liu, and Xiao-Ming Wu. 2025. Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders. arXiv preprint arXiv:2503.05493 (2025).

- [19] Hanjia Lyu, Song Jiang, Hanqing Zeng, Yinglong Xia, Qifan Wang, Si Zhang, Ren Chen, Chris Leung, Jiajie Tang, and Jiebo Luo. 2024. Llm-rec: Personalized recommendation via prompting large language models. In Findings of the Association for Computational Linguistics: NAACL 2024. 583–612.
- [20] Malik Magdon-Ismail and Amir F Atiya. 2004. Maximum drawdown. Risk Magazine 17, 10 (2004), 99–102.
- [21] Daniel McFadden. 1972. Conditional logit analysis of qualitative choice behavior.

(1972).

- [22] Khalid Mehraj and Vinay Kumar. 2025. Psychological Biases in Investment Decisions: A Behavioral Finance Approach. (2025).
- [23] Andreas Oehler and Matthias Horn. 2024. Does ChatGPT provide better advice than robo-advisors? Finance Research Letters 60 (2024), 104898.
- [24] Xueqing Peng, Lingfei Qian, Yan Wang, Ruoyu Xiang, Yueru He, Yang Ren, Mingyang Jiang, Jeff Zhao, Huan He, Yi Han, et al. 2025. MultiFinBen: A Multilingual, Multimodal, and Difficulty-Aware Benchmark for Financial LLM Evaluation. arXiv preprint arXiv:2506.14028 (2025).
- [25] Lingfei Qian, Weipeng Zhou, Yan Wang, Xueqing Peng, Han Yi, Yilun Zhao, Jimin Huang, Qianqian Xie, and Jian yun Nie. 2025. Fino1: On the Transferability of Reasoning-Enhanced LLMs and Reinforcement Learning to Finance. arXiv:2502.08127 [cs.CL] https://arxiv.org/abs/2502.08127
- [26] Mark Rubinstein. 2002. Markowitz’s" portfolio selection": A fifty-year retrospective. The Journal of finance 57, 3 (2002), 1041–1045.
- [27] Chandan Kumar Sah and Xiaoli Lian. 2025. PerFairX: Is There a Balance Between Fairness and Personality in Large Language Model Recommendations?. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2750–2759.
- [28] Chandan Kumar Sah, Xiaoli Lian, Tony Xu, and Li Zhang. 2025. FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness. arXiv preprint arXiv:2504.07801 (2025).
- [29] Javier Sanz-Cruzado, Nikolaos Droukas, and Richard McCreadie. 2024. FAR-Trans: An Investment Dataset for Financial Asset Recommendation. arXiv preprint arXiv:2407.08692 (2024).
- [30] Suraj Sharma, Joseph Brennan, and Jason Nurse. 2021. StockBabble: A conversational financial agent to support stock market investors. In Proceedings of the 3rd Conference on Conversational User Interfaces. 1–5.
- [31] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, et al. 2025. OpenAI GPT-5 System Card. arXiv:2601.03267 [cs.CL] https://arxiv.org/abs/2601.03267
- [32] Ying So and Warren F Kuhfeld. 1995. Multinomial logit models. In SUGI 20 conference proceedings, Vol. 1995. 1227–1234.
- [33] Yueming Sun and Yi Zhang. 2018. Conversational recommender system. In The 41st international acm sigir conference on research & development in information retrieval. 235–244.
- [34] Takehiro Takayanagi, Chung-Chi Chen, and Kiyoshi Izumi. 2023. Personalized dynamic recommender system for investors. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2246–2250.
- [35] Takehiro Takayanagi, Kiyoshi Izumi, Javier Sanz-Cruzado, Richard McCreadie, and Iadh Ounis. 2025. Are generative AI agents effective personalized financial advisors?. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval. 286–295.
- [36] Takehiro Takayanagi, Masahiro Suzuki, Kiyoshi Izumi, Javier Sanz-Cruzado, Richard McCreadie, and Iadh Ounis. 2025. FinPersona: An LLM-Driven Conversational Agent for Personalized Financial Advising. In European Conference on Information Retrieval. Springer, 13–18.
- [37] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https: //arxiv.org/abs/2505.09388
- [38] Amos Tversky and Daniel Kahneman. 1992. Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and uncertainty 5, 4 (1992), 297–323.
- [39] Yan Wang, Lingfei Qian, Xueqing Peng, Yang Ren, Keyi Wang, Yi Han, Dongji Feng, Fengran Mo, Shengyuan Lin, Qinchuan Zhang, Kaiwen He, Chenri Luo, Jianxing Chen, Junwei Wu, Chen Xu, Ziyang Xu, Jimin Huang, Guojun Xiong, Xiao-Yang Liu, Qianqian Xie, and Jian-Yun Nie. 2026. FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information. arXiv:2505.20650 [cs.CL] https://arxiv.org/abs/2505.20650
- [40] Yan Wang, Keyi Wang, Shanshan Yang, Jaisal Patel, Jeff Zhao, Fengran Mo, Xueqing Peng, Lingfei Qian, Jimin Huang, Guojun Xiong, Xiao-Yang Liu, and JianYun Nie. 2025. FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs. arXiv:2510.08886 [cs.CL] https://arxiv.org/abs/ 2510.08886
- [41] An Yang, Baosong Yang, Binyuan Hui, et al. 2024. Qwen2 Technical Report. arXiv preprint arXiv:2407.10671 (2024).
- [42] Qi Yang, Sergey Nikolenko, Alfred Huang, and Aleksandr Farseev. 2022. Personality-driven social multimedia content recommendation. In Proceedings of the 30th ACM International Conference on Multimedia. 7290–7299.
- [43] Tong Yu, Yongcheng Jing, Xikun Zhang, Wentao Jiang, Wenjie Wu, Yingjie Wang, Wenbin Hu, Bo Du, and Dacheng Tao. 2025. Benchmarking reasoning robustness in large language models. arXiv preprint arXiv:2503.04550 (2025).

[44] Yilun Zhao, Yitao Long, Hongjun Liu, Ryo Kamoi, Linyong Nan, Lyuhao Chen, Yixin Liu, Xiangru Tang, Rui Zhang, and Arman Cohan. 2024. DocMath-Eval: Evaluating Math Reasoning Capabilities of LLMs in Understanding Long and

Specialized Documents. arXiv:2311.09805 [cs.CL] https://arxiv.org/abs/2311. 09805

