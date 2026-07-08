# arXiv:2604.08362v2[cs.CL]21May2026

## Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces

Jiawei Chen1,2∗, Ruoxi Xu1,2∗, Boxi Cao1, Ruotong Pan3, Yunfei Zhang3, Yifei Hu3, Yong Du3, Tingting Gao3, Yaojie Lu1, Yingfei Sun2, Xianpei Han1, Le Sun1, Xiangyu Wu3, Hongyu Lin1 1 Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences 2 University of Chinese Academy of Sciences 3 Kuaishou Technology https://OmniBehavior.github.io

### Abstract

The emergence of Large Language Models (LLMs) has illuminated the potential for a general-purpose user simulator. However, existing benchmarks remain constrained to isolated scenarios, narrow action spaces, or synthetic data, failing to capture the holistic nature of authentic human behavior. To bridge this gap, we introduce OmniBehavior, the first user simulation benchmark constructed entirely from real-world data, integrating long-horizon, cross-scenario, and heterogeneous behavioral patterns into a unified framework. Based on this benchmark, we first provide empirical evidence that previous datasets with isolated scenarios suffer from tunnel vision, whereas real-world decision-making relies on long-term, crossscenario causal chains. Extensive evaluations of state-of-the-art LLMs reveal that current models struggle to accurately simulate these complex behaviors, with performance plateauing even as context windows expand. Crucially, a systematic comparison between simulated and authentic behaviors uncovers a fundamental structural bias: LLMs tend to converge toward a positive average person, exhibiting hyper-activity, persona homogenization, and a utopian bias. This results in the loss of individual differences and long-tail behaviors, highlighting critical directions for future high-fidelity simulation research1.

### 1 Introduction

Human behavior is not a sequence of isolated atoms, but a continuous narrative traversing long horizons and diverse scenarios [25]. The holistic modeling of such behavior is central to a wide range of disciplines, such as cognitive science [32, 30], behavioral economics [6, 12, 39], recommender systems [56, 9, 8], and interactive artificial intelligence [24, 35]. The recent breakthrough advancements of Large Language Models (LLMs) have raised an appealing prospect that a single LLM might serve as a general-purpose user simulator: predicting personalized behavior, and standing in for authentic users in the design and evaluation of interactive systems. To succeed, such a model must faithfully portray the cross-scenario, long-horizon, and heterogeneous structure of behavior, rather than merely fitting short, homogeneous interaction segments.

∗Equal contribution. This work was done when Jiawei and Ruoxi interned at Kuaishou. 1Both the code and dataset are publicly available: the code at GitHub and the dataset at Hugging Face.

Preprint.

[Figure 1]

(1) Real-World Data Collection (2) Raw Data Processing (3) OmniBehavior Benchmark

Span: 2025.09.01 – 2025.11.30

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Case A: Interest Evolution And CrossScenario Conversion in Tech (～30 Days)

###### Aggregate 5 Scenarios 22 Actions

Sep 5

Oct 12

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Video Browsing Live Streaming

[Figure 14]

Mall: Purchase Sony

Search: “Best Camera”

[Figure 15]

Video: Tech Review

- a. Behavior-level
- b. Text-level

[Figure 16]

[Figure 17]

Truncate spurious action

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Case B: Creator Loyalty And Full-Chain Engagement in Daily Life Sep 8

Cleaning

Regex + LLM Clean

[Figure 22]

Sep 26

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

E-commerce Advertisement

Representatives

[Figure 28]

Live:

Video: Food Review

Live: Engage in Q&A

Add to Cart

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

……

[Figure 33]

[Figure 34]

Sampling

Key Features

[Figure 35]

Authentic Real-World Data

Heterogeneous Multi-Scenario

Long-Horizon Sequence

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Customer Service Search Behavior

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Billion-scale from active Kuaishou users.

Interleave actions across 5 major scenes.

Continuous 3-month user logs (avg. 32k token).

[Figure 46]

LLM Human Dataset

KUAISHOU

Anonymize

Figure 1: Overview of OmniBehavior, a real-world comprehensive benchmark for evaluating LLM-based user simulators. The benchmark is constructed in three stages: (1) Data Collection: aggregation of real-world logs from the Kuaishou platform across several major scenarios, with Customer Service treated as part of the E-commerce scenario, yielding 5 scenarios in total after aggregation. (2) Data Processing: multi-modal fusion, two-level cleaning, representative sampling, and anonymization. (3) Benchmark Construction: the resulting dataset captures long-horizon, cross-scenario behavior traces, providing a high-fidelity testbed for evaluating LLM-based user simulators in real-world industrial settings.

While LLM-based simulators are rapidly advancing [48, 14, 43], the empirical basis for assessing their capabilities remains systematically insufficient. Specifically, existing benchmarks are often confined to isolated scenarios (e.g., exclusively focusing on video browsing [15], live streaming [38],

- E-commerce dialogue [57]) with narrow action spaces (e.g., click or watch). Such narrow focus overlooks the holistic nature of real human behavior, where authentic user preferences and actions are intrinsically interconnected across various scenarios and activity types. A user’s comprehensive digital footprint constitutes a cross-scenario continuum, composed of heterogeneous behaviors ranging from passive browsing and clicking to active searching, commenting, consulting, purchasing, etc. For instance, a purchase decision of a user may stem from interest sparked by a video viewed days earlier, while subsequent comments in a live stream reflect the actual post-purchase experience. This reliance on fragmented evaluation undermines the reliable evaluation of LLMs’ behavioral modeling capacity and risks systematic misinterpretation, thereby constraining both optimization and real-world deployment.

To bridge this gap, we propose OmniBehavior, to our knowledge, the first user simulation benchmark built entirely on real-world data that simultaneously captures long-horizon, crossscenario and heterogeneous behavioral patterns. OmniBehavior is collected from Kuaishou2, one of the world’s largest video platforms, with over 400 million daily active users. As illustrated in Figure 1, users on Kuaishou navigate across multiple scenarios such as video browsing, live streaming, advertising, and e-commerce, and interact through heterogeneous actions including searches, likes, shares, purchase and conversations. The combination of scenario diversity and richly structured interaction traces makes Kuaishou a uniquely suitable source for modeling real-world user behavior. Based on real user data from Kuaishou, we construct OmniBehavior through careful user sampling, long-horizon behavior aggregation, rigorous quality control, and strict anonymization procedures. Finally, we aggregate complete interaction traces with precise timestamps from 200 representative users over a three-month period, purposely selected to ensure high trajectory density and diversity, yielding a total of 2.12 million interactions. The trace lengths range from 50 to over 100,000, covering a total of 22 distinct actions across 5 different scenarios. To evaluate the ability of LLMs to accurately simulate individual users, we formulate a user-conditioned prediction task: given a user profile, the user’s historical behavior sequence, and the context of a specific scenario (e.g., video content, product information, or consultation background), the LLM is required to predict all corresponding user behaviors in that scenario (e.g., purchases and conversational responses).

2www.kuaishou.com

Drawing on OmniBehavior, we first conduct an in-depth investigation into real users’ behavioral patterns, and reveal that cross-scenario dependencies and long-horizon heterogeneous data are essential to transcending fragmented snapshots, enabling a panoramic understanding of user preferences and the modeling of truly authentic behavioral trajectories. Specifically, we find that (1) Single-scenario data suffers from tunnel vision. Statistically, integrating new scenarios consistently expands interest coverage by approximately 20-30%, merging disjointed facets into a holistic view that eliminates profiling bias. (2) Human decision-making inherently follows a long-horizon, crossscenario causal structure. Our backtracking analysis reveals that over 80% of conversion paths span multiple scenarios and extend over days. This confirms that long-sequence, heterogeneous data is essential for modeling authentic trajectories and preserving causal integrity. (3) Synthetic datasets exhibit significant statistical deviations when compared to authentic human data, underscoring the irreplaceable value of benchmarks grounded in real-world data. Given the profound complexity of these authentic patterns, it remains an open question whether state-of-the-art models can transcend fragmented data to capture holistic human decision-making. This necessitates a systematic evaluation of their true behavioral modeling capacities.

Building upon this benchmark, we perform a comprehensive evaluation of current state-of-the-art LLMs, including both closed-source models (e.g., Claude-4.5-Opus [4], GPT-5.2 [33]) and opensource models (e.g., GLM-4.7 [53], Qwen3-235B [50]), and find that current LLMs exhibit substantial limitations in faithfully modeling long-horizon, cross-scenario user behavior. For instance, even the best-performing LLM, Claude-4.5-Opus, achieves only an overall score of 44.55. Moreover, the F1 scores on binary behavior prediction tasks (e.g., like and share) of most models do not exceed 40%. Furthermore, although existing LLMs support long context windows, we observe that when the context is extended beyond 32K or even to 128K tokens, simply providing longer input sequences does not necessarily yield better performance. This finding highlights the importance of improving long-context reasoning capabilities and designing more effective memory management strategies.

Crucially, we further conduct a systematic comparison between simulated and authentic behaviors, uncovering a fundamental structural bias in current LLM-based simulators: Current LLMs tend to converge toward a positive average person representation, blurring individual-specific differences and discarding long-tail behaviors as well as negative interaction patterns, which severely limits their ability to accurately model real human behavior. Specifically, we first observe hyper-activity: LLMs consistently overestimate action probabilities compared to human baselines. This hampers their ability to model negative feedback signals, which are crucial for churn prediction. This quantitative error further leads to persona homogenization: quantitative analysis of behavioral vectors reveals that while real data exhibits distinct separation between individuals, simulated populations show severe overlap. This convergence towards an "average person" implies that models might discard specific, long-tail behavioral patterns in favor of generic responses, failing to reproduce the heterogeneity of real ecosystems. Finally, simulations are further distorted by a utopian bias: influenced by alignment mechanisms, LLMs are unrealistically positive and polite, and struggle to simulate the adversarial interactions of dissatisfied users. Collectively, these findings reveal that current LLMs do not merely fall short in accuracy, but exhibit a structural “positivity-andaverage” filter that prevents the modeling of authentic human diversity. Our work provides a critical cautionary framework for the use of AI in social science and behavioral modeling, suggesting that without addressing these intrinsic distortions, LLMs may remain mirrors of our ideals rather than maps of our reality.

The major contributions of this paper are summarized as follows:

- • We introduce OmniBehavior, to our knowledge, the first user simulation benchmark constructed entirely from authentic user interaction logs, integrating long-horizon, cross-scenario and heterogeneous behavior traces into a unified framework.
- • We provide the systematic analysis of real-world user behavior at scale, demonstrating that cross-scenario dependencies, long-horizon structures and heterogeneous signals are fundamental to accurate preference modeling.
- • We conduct a comprehensive evaluation of SOTA LLMs, revealing substantial capability gaps in modeling realistic user behavior, even with extended context lengths, and establishing strong baselines for future research.
- • We reveal a structural bias in LLM-based simulators, termed positivity-and-average bias, where models overestimate engagement, homogenize user behaviors, and suppress negative and longtail interactions, fundamentally limiting their applicability in real-world settings.

### 2 OmniBehavior Benchmark

In this section, we introduce the framework and construction of OmniBehavior. Unlike fragmented datasets, OmniBehavior is built upon end-to-end industrial logs from Kuaishou, introducing three interconnected challenges for LLMs. First, by covering all major scenarios within Kuaishou, it unifies diverse activities, requiring models to trace causal links across scenarios. Second, by aggregating complete three-month interaction histories, it necessitates modeling ultra-long behavioral sequences. Third, being built entirely on real user traces, it ensures simulations reflect authentic human logic. By faithfully preserving these structural properties, OmniBehavior establishes a rigorous standard for determining whether a user simulator can serve as a credible and robust proxy for real human behavior in industrial applications.

#### 2.1 Taxonomy

We derive the benchmark taxonomy encompassing 5 major scenarios and 22 distinct user actions on Kuaishou platform (one of the world’s largest video platforms). This framework ensures that the benchmark can model the holistic nature of user decision-making:

- • Video Browsing: A fundamental content consumption scenario. User behaviors include passive consumption (e.g., Watch, Completion) and active feedback (e.g., Like, Collect, Share, Comment, Download, Follow, Dislike, Unfollow).
- • Live Streaming: A live broadcast scenario focused on real-time interaction. User behaviors span social engagement (e.g., Like, Comment, Gift), content governance (e.g., Report), and commerce initiation (e.g., Add-to-Cart, Purchase).
- • Advertisement: A commercial scenario interleaved with feeds. Beyond standard feedbacks in video scenario, it involves conversion-specific actions including Click, Form Submission, Activation, and Purchase.
- • E-commerce: An E-commerce scenario focusing on transactional intent. It consists of decisionoriented actions (e.g. Add-to-Cart, Purchase) and customer service interactions, logging multiturn dialogues for product inquiries and after-sales service.
- • Search Behavior: An information retrieval scenario driven by explicit user intent, where users proactively express their needs through targeted search queries within the App.

#### 2.2 Construction Pipeline

This section presents the end-to-end construction pipeline of our benchmark, including data collection, cleaning, sampling, and anonymization. The pipeline is carefully designed to ensure data quality, representativeness, and compliance with privacy and ethical standards. By integrating multiple processing stages, we aim to construct high-fidelity user behavior sequences that accurately reflect real-world decision-making processes.

Data Collection. We collect interaction logs from the Kuaishou platform over a three-month period from September 1 to November 30, 2025. To preserve the holistic structure of user decision-making, we aggregate raw logs from the scenarios defined in Section 2.1 and interleave them based on timestamps to construct unified, chronological behavioral sequences. For each item, we retrieve rich metadata, including content features (e.g., captions, price, OCR, ASR), inherent item attributes (e.g., video duration, commodity description), and interaction statistics (e.g., view counts, paid counts).

Data Cleaning. We conduct a two-level cleaning pipeline to systematically eliminate noise: (1) Behavior-Level Cleaning: This stage eliminates spurious interactions like accidental touches or idle playback. We apply a truncation strategy based on the 99.9th percentile of viewing duration, setting thresholds at 879s for videos and 9,601s for live streams. Outliers exceeding these limits are discarded, removing a median of 91 noisy records per user; (2) Text-Level Cleaning: Raw textual data, such as OCR, ASR, and captions, frequently contains inherent noise. To mitigate this, we adopt a cascade cleaning strategy. First, we apply regular expressions to eliminate structural noise, such as stuttering and garbled characters. Second, we utilize the Qwen2.5-72B-Instruct [51] model to correct residual typos and semantic errors. This process achieves compression rates of 85.9% for OCR text and 5.2% for ASR text, significantly enhancing semantic density.

Data Sampling. To balance fidelity and evaluation efficiency, we employ a clustering-based sampling strategy to select a diverse set of representative users as evaluation targets. We first encode each user

to a comprehensive feature vector vu spanning four axes: (1) Demographics: User profiles including age and gender distributions; (2) Activity Level: Interaction intensity defined by total counts and active days; (3) Interest Distribution: Categorical preferences for content creators (e.g., Gaming, Playing, News); (4) Scenario Preference: Interaction frequency across the five major scenarios. This multi-dimensional construction ensures coverage of diverse behavioral patterns. Next, we apply K-Means clustering [31] to the normalized vectors, partitioning users into behavioral clusters. The user nearest to each cluster centroid is selected as the representative, yielding a final set of 200 users.

Data Anonymization. To safeguard user privacy and ensure ethical compliance, we implement a strict pipeline using a locally deployed Qwen3-235B [50] model. First, we use it to identify sensitive entities (e.g., names, phone numbers, addresses) and replace them with semantic placeholders (e.g., <NAME>, <PHONE>, <ADDRESS>). Second, we assess content toxicity and automatically prune interactions flagged as "Harmful", such as hate speech and violence. Finally, we conduct manual verification to ensure strict adherence to ethical standards.

Based on the above processing, OmniBehavior encompasses the complete interaction trajectories of 200 users across 5 scenarios collected over a three-month period. Most notably, the dataset features an average sequence length of 8,143 actions, providing rich context for user behavior modeling. Detailed distributions of sequence lengths and user demographics are visualized in Appendix A (Figures 11 and 12), further underscoring the benchmark’s long-horizon depth and population diversity.

### 3 Authentic User Behavior Analysis

To provide an in-depth understanding of authentic behavioral patterns, we conduct an empirical analysis of the OmniBehavior dataset. Specifically, we examine cross-scenario information gain and long-term causal dependencies. This analysis reveals that cross-scenario, long-horizon, and authentic data are essential for realistic user simulation.

#### 3.1 Multi-scenario Traces Enable Comprehensive User Modeling

To systematically assess whether single-scenario data is sufficient to model user preferences and behavioral patterns, we compare singlescenario and multi-scenario settings through qualitative profile reconstruction and quantitative measurements of interest coverage. Specifically, for each user, we extract one interest category and three keywords per interacted item using Claude-Sonnet-4.5, and aggregate them into user profiles.

[Figure 47]

[Figure 48]

[Figure 49]

Multi-Domain User Profile

- 1. Video

- 2. Ads

[Figure 50]

- 3. Live

[Figure 51]

- 4. Search

- 5. E-commerce

[Figure 52]

[Figure 53]

[Figure 54]

An Affective-rich content viewer fascinated by Rural-life.

[Figure 55]

[Figure 56]

A pragmatic household manager balancing Car-servicing duties and Health-care.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

###### A Cost-sensitive homemaker acting as a savvy Stream-buyer.

[Figure 63]

[Figure 64]

[Figure 65]

A beauty enthusiast exploring Hairfocused trends within her Fashionstyle.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

A rural lifestyle creator

[Figure 70]

and health-conscious parent driven by livestream commerce.

[Figure 71]

She bleeds fashion style with home

A dedicated nurturer prioritizing Home-center improvements and Child-care.

As illustrated by the user profile example in Figure 2(a), profiles derived from single-scenario data are often fragmented and biased, whereas multi-scenario data provides richer contextual signals that better capture the user’s stable and essential characteristics.

[Figure 72]

[Figure 73]

management, turning family care into engaging digital content.

[Figure 74]

(a) User profile based on single-scenario vs multi-scenario.

100.0% Interest Category 100.0%

|2.4% 0.4%<br><br>14.4% 8.3%<br><br>23.6%<br><br>14.1%<br><br>57.6%<br><br>26.2%<br><br>Interest Keyword|
|---|

1.0

0.8

Coverage(%)

Furthermore, to examine whether this observation generalizes beyond individual cases, we perform a quantitative analysis. Specifically, we calculate the cumulative growth of unique interest categories and keywords as more scenarios are incorporated. The average results across all users are demonstrated in Figure 2(b), where information coverage increases consistently with scenario diversity, confirming the statistical significance of the multi-scenario advantage.

0.6

0.4

0.2

0.0

Search +E-commerce +Ads +Live +Video

(b) Cumulative interest coverage with increasing scenarios.

Figure 2: Multi-scenario enables more comprehensive user modeling than single-scenario.

Root Cause Interest Accumulation Trigger Target Event

Live

E-commerce

Video

Video

[Figure 75]

[Figure 76]

[Figure 77]

###### Search

Live

Video

[Figure 78]

[Figure 79]

[Figure 80]

Advertisement

[Figure 81]

[Figure 82]

Xiaomi 17 Pro Max tempered glass screen protector review

Xiaomi 17 Pro vs. Xiaomi 17 Pro Max: Is the ¥1,000 Price Difference Worth It?

Xiaomi 17 Pro Max 5G Phone Pre-Sale

Xiaomi 17, vivo X300, and OPPO Find X9: Which Should You Choose?

Xiaomi 17 Pro Max 5G Phone Pre-Sale

Xiaomi Mobile Phone Flagship Store

Xiaomi Launch Event

Xiaomi 17 Pro Max New Product Pre-Sale

... (Long Interval) ...

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

9/25 12:46

9/27 11:45

9/27 17:59

9/30 7:43

10/03 7:24

10/08 01:52

10/08 15:28

10/08 23:36

- Figure 3: Case study of a cross-scenario causal chain, in which a search-initiated interest in “Xiaomi” accumulates across diverse interactions over 12 days and culminates in a purchase, highlighting the long-term and cross-scenario nature of user behavior.

3.2 Causal Chains of User Behavior Span Across Scenarios and Long Sequences

A profound understanding of user causal chains is a prerequisite for constructing a dataset that fully captures the authentic decision-making process. We randomly sample 180 high-value conversion events (e.g., “Purchase”) and utilize Claude-Sonnet-4.5 [5] to identify the specific interaction paths within the full historical logs that lead to the event (which we term as causal chains). All traced paths are further manually verified to ensure correctness.

0 50 Time Span (Days)

0

20

40

60

80

Count

(a)

2 Scenarios

40.1%

4 Scenarios

16.6%

1 Scenarios 18.2%

3 Scenarios 25.1%

(b)

Figure 4: Distributions of causal chain spans. (a) Time and (b) scenario counts demonstrate that user decisions naturally extend over long periods and across multiple scenarios.

- Figure 4 illustrates the distributions of temporal spans and scenario spans of the extracted causal chains. First, regarding temporal span, over 60% of decisions rely on cues from more than 3 days prior, significantly exceeding the temporal window of traditional sessionbased assumptions [36]. This necessitates a benchmark with sufficient temporal length. Second, crossscenario dependency is the norm, with 81.8% of chains spanning multiple scenarios. Together, these conclusions demonstrate that user decision-making is a naturally long-sequence and multi-scenario process, necessitating data that reflects this structure.

We further illustrate this with a representative case study in Figure 3, where a 12-day causal chain: spanning searches, cross-scenario interactions, and a live stream—culminates in a purchase. This confirms that decisions stem from long-term, cross-scenario accumulation, and that benchmarks limited to short sessions or single scenarios effectively perform "causal amputation" underscoring the necessity of ultra-long, multi-scenario environments.

#### 3.3 Distributional Divergence in Preference Evolution between Synthetic and Real User

To investigate whether synthetic user simulation data can accurately reproduce the complex dynamics of real user interest evolution, we conduct a comparative analysis between authentic and synthetic user trajectories. Specifically, we compare OmniBehavior with LoCoMo [28], a representative dataset that simulates long-horizon user memory through data synthesis. We adopt the interest extraction method described in Section 3.1 and track the temporal distribution of interest topics for both groups to characterize preference drift across the three-month observation period.

Real User Interest Trends

Synthetic User Interest Trends

InterestDistribution(%)

100

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

20

50

0

0

0 20 40 60 80 Timeline (Days)

0 20 40 60 80 Timeline (Days)

Anime

TraditionalArt

Dance

Clothing

VTuber

Entrepreneurship

Self-Development

Gaming

Figure 5: Comparison of interest evolution in OmniBehavior (authentic, left) and LoCoMo (synthetic, right). Real preferences evolve smoothly, whereas synthetic interests appear as rigid, task-driven spikes.

The resulting interest evolution patterns are shown in Figure 5. We extract interest categories from each interaction, map them into unified macro-level categories, and measure daily drift using 1 − Jaccard between category sets on consecutive days. Real users exhibit smoother and more stochastic interest changes, with an average drift rate of 0.6311, while synthetic users show more rigid, abrupt shifts with a much lower drift rate of 0.1698. This gap suggests that synthetic users fail to capture realistic temporal interest dynamics, underscoring the need for grounding user simulators in real-world interaction data.

Table 1: Comparison between 11 LLMs on the OmniBehavior Benchmark. We categorize user behaviors into three types: binary behaviors (e.g., clicks), continuous behaviors (e.g., duration), and textual behaviors (e.g., dialogue). The overall score represents the aggregated performance. The best/second best scores are bolded/underlined.

Video Live Ads E-commerce Binary Continuous Binary Binary Binary Textual

Overall Score Closed-source

Model

Claude-Opus-4.5 33.05 64.19 31.70 51.16 29.98 57.21 44.55 Claude-Sonnet-4.5 18.85 65.95 25.00 42.77 36.13 54.26 40.49 Claude-Haiku-4.5 22.84 63.26 26.11 30.00 26.37 50.29 36.48

Claude-Sonnet-4 25.29 64.62 28.86 36.81 16.50 49.13 36.87

Gemini-3-Flash 22.09 53.79 25.61 24.64 19.65 49.80 32.60 GPT-5.2 31.54 65.01 28.63 33.60 29.32 46.29 39.07 GPT-4o 27.88 62.75 28.15 25.24 28.66 44.92 36.27

Open-source

GLM-4.7 26.86 64.43 28.97 40.34 32.90 55.25 41.46 DeepSeek-V3 21.42 63.98 27.92 25.74 33.31 52.13 37.42

Kimi-K2-Instruct 23.30 64.80 28.60 31.19 29.94 47.83 37.61 Qwen3-235B 18.26 62.38 23.84 23.19 19.22 45.74 32.11

### 4 LLM Evaluation

#### 4.1 Experiment Settings

Task Definition. We formulate user behavior simulation as follows. Given a user profile pu, the user’s historical behavior sequence Hu and the context of a specific scenario ct, the simulator fθ aims to predict all corresponding user behavior(s) yt in that scenario. Formally, the historical behavior sequence is defined as Hu = {(c1,a1),...,(ct−1,at−1)}, where each interaction consists of a scenario context ci and the realized user behavior ai, covering both explicit signals (e.g., likes) and implicit signals (e.g., play time). The objective is to estimate yˆt = fθ(Hu,ct,pu), where yt may take the form of binary engagement decisions, continuous consumption durations, or textual feedback.

Metrics. We categorize user behaviors into 3 types, following prior settings [46, 47]. For binary behaviors (e.g., like, follow), we formulate simulation as a classification problem and report the F1-score. For continuous behaviors (watch duration), standard absolute error introduces bias toward longer videos. To mitigate this, we report Normalized Mean Absolute Error (NMAE), normalizes the error by the video’s duration. Finally, for textual behaviors (e.g., customer service dialog), we utilize Claude-Sonnet-4.5) as judge to quantify the performance of the simulator which measures four criteria (intent fidelity, persona mimicry, knowledge boundary, and semantic alignment).

Baselines. We evaluate a diverse set of representative LLMs, including both closed-source and opensource models. The closed-source models include the Claude series (Claude-Opus-4.5 [4], ClaudeSonnet-4.5 [5], Claude-Haiku-4.5 [3], and Claude-Sonnet-4 [2]), as well as Gemini-3-Flash [16] and the GPT series (GPT-5.2 [33] and GPT-4o [1, 19]). The open-source models include GLM-4.7 [53], DeepSeek-V3 [26], Kimi-K2-Instruct-0905 (hereafter Kimi-K2-Instruct) [44], and Qwen3-235BA22B-Instruct-2507 (hereafter Qwen3-235B) [50].

Implementation Details. We evaluate models on 6,000 behavior prediction tasks across different scenarios. To ensure the robustness of our evaluation, we carefully sample these tasks by enforcing temporal balance, scenario balance, and value distribution balance. We adopt a global 32k context window for main experimental evaluation and implement the memory management using the LangChain framework [29]. All experiments are conducted on NVIDIA A800-SXM4-80GB machines. Prompts and hyperparameter settings used in the experiments are provided in Appendix C.

#### 4.2 Overall Results

- Table 1 reports the overall performance of diverse LLM backbones on OmniBehavior. Results show that high-fidelity user simulation remains highly challenging. Even the best-performing model, Claude-Opus-4.5 achieves an overall score of 44.55, with most models clustered between 32 and 41. This low range indicates that general instruction tuning is insufficient for modeling the stochasticity, long-tail distributions, and cross-scenario causal dependencies inherent in human behavior, revealing a substantial gap between current LLMs and real human patterns. Closed-source models set the overall

performance ceiling, with Claude-Opus-4.5 achieving the highest score of 44.55 and leading 4 of 6 metrics. Yet this advantage is not absolute: the open-source GLM-4.7 ranks second overall with 41.46, outperforming Claude-Sonnet-4.5 (40.49) and GPT-5.2 (39.07). Open-weight models also excel in targeted settings, as DeepSeek-V3 surpasses Claude-Opus-4.5 in E-commerce binary behavior simulation (33.31 vs. 29.98). Thus, while closed-source models dominate overall, optimized open-weight models can rival or exceed them on specific tasks.

#### 4.3 Evaluation of Long Context Modeling

Effect of Context Window Size. We evaluate representative open-source and closed-source LLMs with context windows from 16K to 128K tokens on a subset of users (N = 66) whose interaction histories exceed 128K tokens. As shown in Figure 6, longer contexts do not consistently improve performance, suggesting that current LLMs, despite supporting ultra-long inputs, still struggle to effectively leverage extended interaction histories for long-context reasoning, consistent with prior findings [27].

Memory Management Mechanisms. Based on Qwen3235B, we compare two memory management approaches: history summarization and RAG-based retrieval of relevant scenarios, with implementation details in Appendix C.2. Figure 7 shows that both offer only modest gains, as RAG relies on semantic similarity and ignores causal dependencies, while summarization may discard critical behavioral details. This highlights the need to explicitly preserve the causal and temporal structure of user history.

### 5 The Structural Bias of LLM Simulator

38.00

36.00

| |
|---|

| |
|---|

| |
|---|

| |
|---|

34.00

32.00

30.00

16k 32k 48k 64k 96k 128k

Claude-Haiku-4.5 GPT-5.2 Qwen3-235B

Figure 6: Effect of context size.

40

30

Score

20

10

0

Video Live Ads E-commerce

Truncation

RAG

Summary

| |
|---|

| |
|---|

Figure 7: Effect of memory management.

To probe the underlying causes of the simulation gap, we systematically compare simulated and authentic behaviors, uncovering a fundamental structural bias in current LLM-based simulators: a "positivity-and-average" tendency. We demonstrate this through behavior probability distributions, emotional distributions, and cross-user behavioral differences.

Hyper-activity Bias. We compare real and simulated behavior distributions via positive interaction rates (positive outcomes among all behavior instances). Figure 8 shows that real behavior is sparse, with positive rates below 10%, while LLM simulators such as Qwen3-235B and Gemini-3-Flash overestimate user engagement by 40–60%. This hyperactivity bias shows poor modeling of implicit rejection, limiting their usage like user churn warning [41].

0.6 PositivePredictionRate

0.4

0.2

0.0

Video Live Ads E-commerce

|Human Claude-Opus-4.5<br><br>| |
|---|
<br><br>| |
|---|
<br><br>GPT-5.2 Kimi-K2-Instruct<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Gemini-3-Flash Qwen3-235B<br><br>| |
|---|
|
|---|

Figure 8: LLM simulators overestimate positive user interactions across scenes.

Utopian Tendency. We further examine affective realism in E-commerce customer service dialogues by analyzing emotional expression and language style. Using Claude-Sonnet-4.5, we assign each utterance a sentiment label in {-1,0,1}, from negative to positive. Figure 9 shows that real users often express strong negative emotions, while LLMgenerated utterances cluster around neutral or positive sentiment. Language style analysis further shows that LLM outputs are more polite, whereas real users are more direct and emotionally expressive (Appendix C.3). Overall, LLM simulations exhibit a positivity and politeness bias, producing an overly sanitized environment unsuitable for crisis management, malicious attacks, or adversarial social dynamics. Cases are provided in Appendix E.

| | |
|---|---|
| | |
| | |
| | |
| | |

1.5

1.0

0.5

0.0

-1.00 -0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00

Human

Gemini-3-Flash

GLM-4.7

Claude-Opus-4.5

GPT-5.2

DeepSeek-V3

Claude-Sonnet-4.5

GPT-4o

Qwen3-235B

Figure 9: LLM dialogues are more positive and miss real users’ negative emotions.

Personality Homogenization. These findings raise whether LLM simulators preserve personalized user characteristics. We analyze intra- and inter-user behavioral distances using 19-dimensional action rate vectors (e.g., like, share, activate). Intra-user distance measures consistency within a user’s history, while inter-user distance captures cross-user heterogeneity. Figure 10 shows that real users have much larger interuser than intra-user variation (Inter ≫ Intra, Ratio ≈ 0.29), whereas LLM-generated users show heavily overlapping distributions (Ratio ≈ 0.7−0.87). This indicates homogenized simulations that fail to preserve distinct user identities, likely due to generic pre-training patterns suppressing long-tail personalized signals. Results of other models are provided in Appendix C.4, demonstrating the same trends.

Ratio: 0.29

Ratio: 0.84

| | |
|---|---|

Ratio: 0.71 Ratio: 0.79

Intra Inter

Figure 10: Comparison of Intra-user and Interuser behavioral distances for real users and LLM-simulated users.

### 6 Related Works

- 6.1 LLMs as Human Simulators

Advancements in LLMs have enabled agents to simulate human cognition and interaction across diverse domains, including dialogue interactions [10, 55], recommender systems [45, 54], autonomous driving [21]. Seminal works like Generative Agents [34] and BASES [40] further extend this to social behaviors and web search. However, these approaches primarily operate within controlled, single-task environments or synthetic sandboxes. They often fail to capture the stochasticity, long-tail distributions, and cross-scenario dependencies inherent in real-world digital footprints. This gap underscores the need for benchmarks grounded in authentic industrial logs to rigorously assess alignment with real human behaviors. Our work addresses this need by providing a cross-scenario, real-world benchmark designed to evaluate the capabilities and limitations of LLMs in capturing diverse and long-term human behavior patterns.

- 6.2 Evaluation for Human Simulation

Existing evaluations for human simulation mainly fall into log-based behavioral prediction and roleplaying benchmarks. Log-based benchmarks have evolved from static interaction matrices [17, 7] to sequential logs and interactive environments [52, 58, 42, 20, 22]. However, they are usually restricted to isolated scenarios and thus miss cross-scenario dependencies, such as purchases triggered by short-video exposure. Role-playing benchmarks, in contrast, focus on persona consistency, memory, and multi-turn interaction [49, 28, 11, 18], but often rely on synthesized profiles or standardized tasks rather than noisy longitudinal logs. OmniBehavior bridges these gaps with large-scale real-world logs that capture cross-scenario, long-horizon, and heterogeneous behavioral patterns, enabling more ecologically valid evaluation of LLM simulators.

### 7 Discussion

Conclusions. We present OmniBehavior, a real-world benchmark for evaluating user simulation. Our statistical analysis shows that human behavior exhibits long-term and cross-scenario dependencies, suggesting that single-scenario settings and purely synthetic data are insufficient for realistic application. Experimental results reveal that current LLMs exhibit a substantial capability gap in modeling real-world user behaviors, regardless of context length. Furthermore, we identify structural bias in LLM-based simulators, characterized by a "positivity-and-average" tendency that homogenizes users and overestimates engagement. Such biases limit the reliability of LLM-based simulators for real-world scenarios. We hope OmniBehavior will encourage future work on modeling the long-horizon, cross-scenario, and heterogeneous nature of human behavior.

Limitations. Due to the rapid evolution of the field, the models and memory management algorithms covered in our current evaluation do not yet encompass all state-of-the-art approaches. To address this limitation, we plan to maintain a public leaderboard that supports community-contributed evaluation submissions, enabling continuous tracking and timely updates of the latest methods.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Anthropic. Introducing claude 4. https://www.anthropic.com/news/claude-4, May 2025.
- [3] Anthropic. Introducing claude haiku 4.5. https://www.anthropic.com/news/claude-haiku-4-5, October 2025.
- [4] Anthropic. Introducing claude opus 4.5. https://www.anthropic.com/news/claude-opus-4-5, November 2025.
- [5] Anthropic. Introducing claude sonnet 4.5. https://www.anthropic.com/news/claude-sonnet-4-5, September 2025.
- [6] W Brian Arthur. Designing economic agents that act like human agents: A behavioral approach to bounded rationality. The American economic review, 81(2):353–359, 1991.
- [7] James Bennett and Stan Lanning. The netflix prize. 2007.
- [8] Lucas Bernardi, Sakshi Batra, and Cintia Alicia Bruscantini. Simulations in recommender systems: An industry perspective. arXiv preprint arXiv:2109.06723, 2021.
- [9] Nicolas Bougie and Narimawa Watanabe. Simuser: Simulating user behavior with large language models for recommender system evaluation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), pages 43–60, 2025.
- [10] Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chateval: Towards better llm-based evaluators through multi-agent debate. arXiv preprint arXiv:2308.07201, 2023.
- [11] Jiawei Chen, Xinyan Guan, Qianhao Yuan, Guozhao Mo, Weixiang Zhou, Yaojie Lu, Hongyu Lin, Ben He, Le Sun, and Xianpei Han. Consistentchat: Building skeleton-guided consistent multi-turn dialogues for large language models from scratch. In The 2025 Conference on Empirical Methods in Natural Language Processing, 2025.
- [12] Geoffrey PE Clarkson and Herbert A Simon. Simulation of individual and group behavior. The American Economic Review, pages 920–932, 1960.
- [13] Cristian Danescu, Niculescu Mizil, Moritz Sudhof, Dan Jurafsky, Jure Leskovec, and Christopher Potts. A computational approach to politeness with application to social factors. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 250–259, 2013.
- [14] Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli Xu, and Yong Li. Large language models empowered agent-based modeling and simulation: A survey and perspectives. Humanities and Social Sciences Communications, 11(1):1–24, 2024.
- [15] Chongming Gao, Shijun Li, Wenqiang Lei, Jiawei Chen, Biao Li, Peng Jiang, Xiangnan He, Jiaxin Mao, and Tat-Seng Chua. Kuairec: A fully-observed dataset and insights for evaluating recommender systems. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pages 540–550, 2022.
- [16] Google. Gemini 3 flash: frontier intelligence built for speed. https://blog.google/products-andplatforms/products/gemini/gemini-3-flash/, December 2025.
- [17] F Maxwell Harper and Joseph A Konstan. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis), 5(4):1–19, 2015.
- [18] Tiancheng Hu, Joachim Baumann, Lorenzo Lupo, Nigel Collier, Dirk Hovy, and Paul Röttger. Simbench: Benchmarking the ability of large language models to simulate human behaviors. arXiv preprint arXiv:2510.17516, 2025.

- [19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [20] Eugene Ie, Chih-wei Hsu, Martin Mladenov, Vihan Jain, Sanmit Narvekar, Jing Wang, Rui Wu, and Craig Boutilier. Recsim: A configurable simulation platform for recommender systems. arXiv preprint arXiv:1909.04847, 2019.
- [21] Ye Jin, Xiaoxi Shen, Huiling Peng, Xiaoan Liu, Jingli Qin, Jiayang Li, Jintao Xie, Peizhong Gao, Guyue Zhou, and Jiangtao Gong. Surrealdriver: Designing generative driver agent simulation framework in urban contexts based on large language model. arXiv preprint arXiv:2309.13193, 5(7):8, 2023.
- [22] Andreas Konstantin Kruff, Christin Katharina Kreutz, Timo Breuer, Philipp Schaer, and Krisztian Balog. Sim4ia-bench: A user simulation benchmark suite for next query and utterance prediction. arXiv preprint arXiv:2511.09329, 2025.
- [23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
- [24] R Greer Lavery. Artificial intelligence and simulation: An introduction. In Proceedings of the 18th conference on Winter simulation, pages 448–452, 1986.
- [25] Kurt Lewin. Field theory in social science: selected theoretical papers (edited by dorwin cartwright.). 1951.
- [26] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [27] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics, 12:157–173, 2024.
- [28] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753, 2024.
- [29] Vasilios Mavroudis. Langchain. 2024.
- [30] James L McClelland. The place of modeling in cognitive science. Topics in Cognitive Science, 1(1):11–38, 2009.
- [31] James B McQueen. Some methods of classification and analysis of multivariate observations. In Proc. of 5th Berkeley Symposium on Math. Stat. and Prob., pages 281–297, 1967.
- [32] Mohd Naveed Uddin. Cognitive science and artificial intelligence: simulating the human mind and its complexity. Cognitive Computation and Systems, 1(4):113–116, 2019.
- [33] OpenAI. Gpt-5.2. https://openai.com/zh-Hans-CN/index/introducing-gpt-5-2/, 2025.
- [34] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.
- [35] Vladimir M Petrovi´c. Artificial intelligence and virtual worlds–toward human-level ai agents. IEEE Access, 6:39976–39988, 2018.
- [36] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pages 2685–2692, 2020.

- [37] Priyanshu Priya, Mauajama Firdaus, and Asif Ekbal. Computational politeness in natural language processing: A survey. ACM Computing Surveys, 56(9):1–42, 2024.
- [38] Changle Qu, Sunhao Dai, Ke Guo, Liqin Zhao, Yanan Niu, Xiao Zhang, and Jun Xu. Kuailive: A real-time interactive dataset for live streaming recommendation. arXiv preprint arXiv:2508.05633, 2025.
- [39] Julian Reiss. A plea for (good) simulations: nudging economics toward an experimental science. Simulation & gaming, 42(2):243–264, 2011.
- [40] Ruiyang Ren, Peng Qiu, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Hua Wu, Ji-Rong Wen, and Haifeng Wang. Bases: Large-scale web search user simulation with large language model based agents. arXiv preprint arXiv:2402.17505, 2024.
- [41] Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, et al. Towards understanding sycophancy in language models. arXiv preprint arXiv:2310.13548, 2023.
- [42] Jing-Cheng Shi, Yang Yu, Qing Da, Shi-Yong Chen, and An-Xiang Zeng. Virtual-taobao: Virtualizing real-world online retail environment for reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 4902–4909, 2019.
- [43] Patrick Taillandier, Jean Daniel Zucker, Arnaud Grignard, Benoit Gaudou, Nghi Quang Huynh, and Alexis Drogoul. Integrating llm in agent-based social simulation: Opportunities and challenges. arXiv preprint arXiv:2507.19364, 2025.
- [44] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.
- [45] Lei Wang, Jingsen Zhang, Hao Yang, Zhiyuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Ruihua Song, Wayne Xin Zhao, et al. When large language model based agent meets user behavior analysis: A novel user simulation paradigm. arXiv preprint arXiv:2306.02552, 2023.
- [46] Lei Wang, Jianxun Lian, Yi Huang, Yanqi Dai, Haoxuan Li, Xu Chen, Xing Xie, and Ji-Rong Wen. Characterbox: Evaluating the role-playing capabilities of llms in text-based virtual worlds. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6372–6391, 2025.
- [47] Noah Wang, Zy Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian Yang, et al. Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 14743–14777, 2024.
- [48] Zixu Wang, Bin Xie, Bingbing Xu, Shengmao Zhu, Yige Yuan, Liang Pang, Long Yang Du Su, Zixuan Li, Huawei Shen, and Xueqi Cheng. A survey on llm-based agents for social simulation: Taxonomy, evaluation and applications.
- [49] Qiuejie Xie, Qiming Feng, Tianqi Zhang, Qingqiu Li, Linyi Yang, Yuejie Zhang, Rui Feng, Liang He, Shang Gao, and Yue Zhang. Human simulacra: Benchmarking the personification of large language models. arXiv preprint arXiv:2402.18180, 2024.
- [50] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [51] An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoyan Huang, Jiandong Jiang, Jianhong Tu, Jianwei Zhang, Jingren Zhou, et al. Qwen2. 5-1m technical report. arXiv preprint arXiv:2501.15383, 2025.
- [52] Guanghu Yuan, Fajie Yuan, Yudong Li, Beibei Kong, Shujie Li, Lei Chen, Min Yang, Chenyun Yu, Bo Hu, Zang Li, et al. Tenrec: A large-scale multipurpose benchmark dataset for recommender systems. Advances in Neural Information Processing Systems, 35:11480–11493, 2022.

- [53] Z.ai. Glm-4.7: Advancing the coding capability. https://z.ai/blog/glm-4.7, December 2025.
- [54] Junjie Zhang, Yupeng Hou, Ruobing Xie, Wenqi Sun, Julian McAuley, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. Agentcf: Collaborative learning with autonomous language agents for recommender systems. In Proceedings of the ACM Web Conference 2024, pages 3679–3689, 2024.
- [55] Qingyu Zhang, Chunlei Xin, Xuanang Chen, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Qing Ye, Qianlong Xie, and Xingxing Wang. Ai-salesman: Towards reliable large language model driven telemarketing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 34790–34798, 2026.
- [56] Shuo Zhang and Krisztian Balog. Evaluating conversational recommender systems via user simulation. In Proceedings of the 26th acm sigkdd international conference on knowledge discovery & data mining, pages 1512–1520, 2020.
- [57] Nan Zhao, Haoran Li, Youzheng Wu, Xiaodong He, and Bowen Zhou. The jddc 2.0 corpus: A large-scale multimodal multi-turn chinese dialogue dataset for e-commerce customer service. arXiv preprint arXiv:2109.12913, 2021.
- [58] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1059–1068, 2018.

### LLM Usage Statement

Large Language Models were used only for limited auxiliary purposes during the preparation of this manuscript. In particular, the authors used LLMs to assist with language editing, including grammar correction, spelling checks, as well as to provide coding assistance during the experimental implementation process. All generated text and code suggestions were carefully reviewed, revised, tested, and approved by the authors. The LLMs did not contribute to the formulation of the research questions, conceptual ideas, experimental design, interpretation of results, or scientific conclusions. All research-level contributions, including the ideas, methodology, analysis, and final decisions, were made by the whole authors.

### Ethics Statement

This work is based on user behavior data that was sampled and processed under applicable legal, platform, and privacy-protection requirements. Only data that was permitted for research and benchmark construction was used. Before inclusion in the released benchmark, the raw data underwent multiple stages of quality control, representative sampling, and privacy-preserving cleaning. In particular, personally identifiable or sensitive information, such as names, phone numbers, addresses, and other user-specific identifiers, was removed or replaced with semantic placeholders (more details are provided in Section 2.2). We also filtered inappropriate or harmful content and manually verified the processed data to ensure compliance with ethical standards. All research analyses and experiments were conducted on the processed data, and the authors take responsibility for ensuring that the benchmark construction and usage comply with relevant data-governance and privacy-protection policies.

### A Data Statistics

#### A.1 Action Sequence Length Distribution

To provide a clearer picture of OmniBehavior’s composition, we report additional statistics on user behavior sequences and population attributes. Figure 11 illustrates that user action sequence lengths span multiple orders of magnitude on a logarithmic scale, range from relatively short traces to extremely long histories that can exceed 100k steps. Such characteristics reflect the inherently longhorizon and heterogeneous nature of real-world user behavior. They also underscore the necessity for models to effectively capture both short-term dynamics and long-range dependencies within a unified framework. Consequently, OmniBehavior presents substantial challenges for long-context modeling, particularly in terms of memory efficiency, temporal reasoning, and the ability to maintain coherent representations over ultra-long sequences.

Figure 11: Log-scaled distribution of user action sequence lengths, spanning hundreds to over 100k, requiring models to handle ultra-long contexts.

#### A.2 User Demographics and Interest Diversity

- Figure 12 complements this view by presenting the demographic and interest distributions of the sampled users. The results show that OmniBehavior covers users from diverse gender and age groups, while also spanning a broad spectrum of content interests. Such diversity shows that the dataset does not concentrate on a narrow user segment, but instead preserves substantial heterogeneity in both user background and preference structure. Taken together, these statistics indicate that our OmniBehavior benchmark is challenging not only because of its long behavioral horizons, but also because it captures a diverse and heterogeneous user population, which is essential for realistic user behavior simulation.

[Figure 98]

(a) Gender (b) Age (c) Interests

- Figure 12: Demographic and behavioral distributions of users in the benchmark. The charts show diversity across gender, age groups, interests, ensuring a representative population for user simulation.

B Benchmark Schema

- Figure 13 illustrates the detailed schema of the OmniBehavior benchmark across representative user scenarios on the Kuaishou platform, including Video Browsing, Live Streaming, E-commerce, and Advertisement. Search Behavior is not explicitly visualized because it does not involve a rich scenario context beyond the search query itself. Customer service next-utterance prediction is treated as part of the E-commerce scenario.

As shown in the figure, the input context consists of heterogeneous signals describing the interaction environment, such as timestamps, content titles, item attributes, and historical statistics, along with textual signals extracted from multimodal sources (e.g., OCR and ASR transcripts when available). Based on these contextual signals, the LLM-based simulator is required to generate simulated user actions. The action space varies across scenarios and includes behaviors such as liking, collecting, commenting, purchasing, and predicting watch duration or conversational responses.

This unified formulation enables OmniBehavior to capture diverse interaction patterns across multiple product surfaces while maintaining a consistent evaluation protocol. The resulting benchmark provides a structured and realistic environment for assessing the capability of LLM-based user simulators in modeling complex user behaviors.

(1) Video Browsing (2) Live Streaming

Context

Simulated Action

Simulated Action

Context

[Figure 99]

[Figure 100]

Verify Likes (Yes/No)

[Figure 101]

Category Likes (Yes/No)

[Figure 102]

[Figure 103]

[Figure 104]

Time

Time

[Figure 105]

Gifts (Yes/No) Click_cart (Yes/No)

[Figure 106]

Collect (Yes/No) Duration (Seconds)

[Figure 107]

[Figure 108]

[Figure 109]

Cover Title Cart Stats

[Figure 110]

ASR Title OCR Stats

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Other action (e.g., share follow)

[Figure 118]

Other action (e.g., share, follow)

(3) E-commerce (4) Advertisement

Context

Context

Simulated Action

Simulated Action

[Figure 119]

[Figure 120]

[Figure 121]

Likes (Yes/No) Comment (Yes/No) Activate (Yes/No)

[Figure 122]

[Figure 123]

[Figure 124]

Item

Time

Time

Sales

Add_to_cart (Yes/No) Purchase (Yes/No)

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Title OCR

Title Desc.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

NextUtterance Prediction Sales Submit_form (Yes/No)

[Figure 135]

[Figure 136]

Price

ASR

Source

[Figure 137]

- Figure 13: The OmniBehavior Benchmark Scope. We construct a unified simulation environment covering multiple major user activities on the Kuaishou platform, with customer service next-utterance prediction treated as part of the E-commerce scenario. The framework requires the agent to predict diverse behaviors (e.g., watch duration, purchase, comment, and customer-service responses) based on scenario-specific contexts, serving as a comprehensive testbed for high-fidelity user simulation.

### C Experimental Parameters and Details

#### C.1 Model Inference Details

In all main experimental evaluations, model inference was conducted via API calls with a fixed temperature of 0.1, while the top-p parameter was kept at each model’s default setting to ensure consistency and reduce sampling variance. Closed-source models, as well as DeepSeek-V3 and KimiK2-Instruct, were accessed through their official public APIs. In contrast, GLM-4.7 and Qwen3-235B were deployed locally using the vLLM framework [23] on NVIDIA A800-SXM4-80GB machines, enabling efficient large-scale inference under controlled hardware conditions.

#### C.2 Memory Management Results

We evaluate three representative memory management strategies: (1) Truncation, retains only the most recent context within the fixed window; (2) RAG, retrieves top-k relevant interactions (k=100) to ensure high recall over long user histories; and (3) Summary, incrementally compresses user history via periodic summarization with a 4K token buffer. For the RAG module, dense retrieval is implemented using the bge-m3 embedding model (1,024 dimension). The retrieval stage returns Top100 candidates with additional oversampling to improve recall, while vector indices are cached and periodically refreshed to balance efficiency. For the summarization module, we employ Qwen3-32B (no-thinking) with the same temperature setting to ensure stable outputs. A buffer-based incremental strategy is adopted, where summarization is triggered once the buffer reaches its 4K-token limit, and each summary is constrained to at most 1,000 tokens.

As shown in Table 2, different memory management strategies exhibit distinct trade-offs across scenarios. All methods are implemented using the same model (Qwen3-235B) in the main experiment and evaluated on the full user interaction history to ensure fairness. The Summary method achieves the best overall performance, particularly yielding substantial improvements in the Live Streaming scenario (31.06 vs. 23.84), suggesting its potential advantage in aggregating long-horizon interaction signals. In contrast, RAG provides marginal or even negative gains compared to Truncation in certain scenarios (e.g., E-commerce, 16.14 vs. 19.22), which may stem from retrieval noise or the disruption of sequential behavioral dependencies inherent in dense-retrieval-based context construction.

Overall, the results suggest that existing memory management strategies are insufficient for modeling real-world user behavior. Truncation preserves fidelity but lacks long-term reasoning, RAG retrieves relevant fragments but ignores causal structure, and summarization captures global trends but inevitably loses fine-grained behavioral details. These limitations call for the need of structure-aware memory mechanisms that explicitly model temporal order, cross-scenario dependencies, and causal relationships in user behavior sequences. We will explore more advanced memory architectures to improve long-horizon modeling in the future work.

Table 2: Comparison of memory management strategies on OmniBehavior using Qwen3-235B. Each scenario is formulated as a binary behavior prediction task (e.g., like, collect, share).

Method Video Live Ads E-commerce Avg. Truncation 18.26 23.84 23.19 19.22 21.13 RAG 17.9 24.44 23.03 16.14 20.38 (↓ 3.6%) Summary 20.7 31.06 24.86 20.45 24.27 (↑ 14.9%)

#### C.3 Complementary Analysis of Utopian Tendency

Following prior work on computational politeness and language strategies [13, 37], we decompose language style into five dimensions, including politeness markers, hedging language, blame avoidance, emotional control, and face-saving strategies. As shown in Figure 14, LLM-generated utterances consistently exhibit higher levels across these dimensions compared to real users. In contrast, real user language is more direct, emotionally expressive, and often confrontational in service failure scenarios. These results suggest that LLM-based simulators default to overly polite and controlled communication patterns, failing to capture the diversity and intensity of real-world user expressions.

Figure 15 illustrates a clear stylistic divergence between real user utterances and those generated by LLM-based simulators. The model-generated utterances (blue) are dominated by high-politeness markers and formal modal constructions. Expressions such as "Could you", "I would like", "Please" and "May I ask" account for a large portion of the generated vocabulary, reflecting a tendency toward highly polite and grammatically complete sentences. In contrast, the vocabulary of real users (red) is largely centered around operational issues, logistics, and post-purchase problem resolution. Frequent terms include "Refund", "Missing Item", "Fake", "Broken", "Hurry up", and "Intercept". Compared with model-generated language, real user expressions tend to be more transactional, direct, and emotionally charged, particularly when referring to service failures or delivery problems.

1.0

[Figure 138]

0.21 0.40 2.72 1.68 3.28 0.44 0.94 3.00 1.90 3.46 0.38 0.97 2.96 1.78 3.26 0.66 1.59 3.22 2.02 3.30 0.19 0.68 3.03 1.80 3.48 0.12 0.79 3.08 1.73 3.46

Human

0.8

Claude-Opus-4.5

0.6

Claude-Sonnet-4.5

GPT-5.2

0.4

GLM-4.7

0.2

DeepSeek-V3

0.0

Face Saving EmotionalControl

Hedging Language

Blame Avoidance

PolitenessMarkers

Figure 14: Language style comparison between real users and LLM-simulated users. LLM-generated utterances exhibit higher levels of politeness markers, hedging, and face-saving strategies, indicating a systematic tendency towards overly polite and nonconfrontational language.

These observations suggest that current LLM-based simulators exhibit a notable positivity bias or politeness alignment. While the models can simulate the intent of purchasing behavior, they struggle to reproduce the interaction style observed in real-world scenarios, especially in adversarial or frustration-driven contexts such as after-sales disputes. As a result, the user simulators display a more polite language style than real users, leading to a Utopian Tendency interaction pattern that overlooks the long-tail of abrupt, urgent, or dissatisfied behaviors commonly present in real E-commerce interactions [43].

[Figure 139]

[Figure 140]

(a) Real Users (b) LLM-based Simulators

- Figure 15: Vocabulary comparison of real users (left) and LLM-based simulators (right). The LLMgenerated language is dominated by highly polite expressions, reflecting a "Utopian Tendency" interaction style that contrasts with the friction vocabulary observed in real user interactions.

#### C.4 Personality Homogenization Results

To provide a comprehensive view, we report Intra-user and Inter-user behavioral distance distributions for all evaluated models in Figure 16. Consistent with the main findings, all LLM-based simulators exhibit substantially higher intra/inter ratios compared to real users, with heavily overlapping distributions across models. This result further confirms that personality homogenization is a general phenomenon rather than model-specific.

| |Ratio: 0.8|
|---|---|

| |Ratio: 0.79|
|---|---|

Ratio: 0.29

Ratio: 0.77

| |Ratio: 0.78|
|---|---|

| |Ratio: 0.87|
|---|---|

| |Ratio: 0.82|
|---|---|

Ratio: 0.66

| |Ratio: 0.72|
|---|---|

Ratio: 0.73 Ratio: 0.71

Ratio: 0.84

Intra Inter

- Figure 16: Intra-user vs. Inter-user behavioral distance distributions for all evaluated models. Compared to real users, all LLM-based simulators exhibit higher intra/inter ratios and substantial overlap between Intra/Inter-user distributions, indicating a consistent tendency towards persona homogenization.

### D Experimental Prompts

We have compiled all the prompts used in our experiments. Specifically, the prompts for binary, continuous, and text value predictions in the main experiment are presented in Figure 17, Figure 19, and Figure 21, respectively. Additionally, Figure 23 illustrates the prompt for raw data cleaning, Figure 22 shows the prompt for extracting key information from live streaming cover images, while Figure 18 details the prompt for identifying an item’s interest categories and keywords. Finally, the prompts for causal chain identification, sentiment classification and language style comparison are provided in Figure 24, Figure 20 and Figure 25.

You are a real user of the Kuaishou platform. Your core task is to infer the user’s interest preferences, spending level, and personality traits based on the given historical behavior sequence, and then simulate how the user would make a real decision in the current scenario. Core principles. Data driven. All inferences must be based on objective evidence from historical behavior data, avoiding unfounded speculation or assumptions. Behavioral consistency. New decisions should remain logically consistent with the user’s historical behavior patterns, reflecting stable preferences and habits. Individual differences. Fully respect each user’s uniqueness. Do not apply stereotypes or group labels. Discover genuine personal traits from the data. Context sensitivity. Decision prediction should consider the specific characteristics of the current scenario, balancing long term preferences with short term contextual factors. Authenticity first. Simulate choices a real user might make, including lack of interest, hesitation, or skipping, rather than always giving positive responses.

Input one is the user profile. This is the user’s basic platform information and can be used as background reference. Input two is the historical behavior trajectory. This is the user’s real operation record over a past period, covering multiple scenarios such as live streaming, marketplace, videos, and advertisements. Analyze the motivations and tendencies behind these behaviors to uncover implicit long term preferences and short term intentions. Input three is the current test scenario. The user is now facing the following situation. Prediction task. Please answer the following question from the user’s perspective. Output requirements. Please output only Yes or No. Do not output any other content, explanation, or analysis. Your answer:

Figure 17: Prompt for binary value prediction in the main experiment.

Task: Analyze the following user interaction content and extract:

- 1. One concise category (Category), such as “Beauty,” “Games,” “News,” etc.
- 2. Three specific keywords (Keywords). Ignore the interactive form of the text. Even if it is casual chat between friends, look beyond the social surface and identify the underlying topic being discussed. Content: "text" Output Format: Return only a JSON object containing two fields: "category" and "keywords" (a list of strings). Example: "category": "Technology", "keywords": ["iPhone", "Apple", "Smartphone"]

- Figure 18: Prompt for identifying an item’s interest categories and keywords.

You are a real user of the Kuaishou platform. Your core task is to infer the user’s interest preferences, spending level, and personality traits based on the given historical behavior sequence, and then simulate how the user would make a real decision in the current scenario. Core principles: Data driven. All inferences must be based on objective evidence from historical behavior data, avoiding unfounded speculation or assumptions. Behavioral consistency. New decisions should remain logically consistent with the user’s historical behavior patterns, reflecting stable preferences and habits. Individual differences. Fully respect each user’s uniqueness. Do not apply stereotypes or group labels. Discover genuine personal traits from the data. Context sensitivity. Decision prediction should consider the specific characteristics of the current scenario, balancing long term preferences with short term contextual factors. Authenticity first. Simulate choices a real user might make, including lack of interest, hesitation, or skipping, rather than always giving positive responses.

Input one is the user profile. This is the user’s basic platform information and can be used as background reference. Input two is the historical behavior trajectory. This is the user’s real operation record over a past period, covering multiple scenarios such as live streaming, marketplace, videos, and advertisements. Analyze the motivations and tendencies behind these behaviors to uncover implicit long term preferences and short term intentions.

Input three is the current test scenario. The user is now facing the following situation. Prediction task. Please answer the following question from the user’s perspective. Output requirements. Please output only a single integer. Do not output any other content, explanation, or unit. Your answer:

- Figure 19: Prompt for continuous value prediction in the main experiment.

You are a sentiment analysis expert. Analyze the sentiment of the text. Output only a number, with no explanation. Analyze the sentiment of the following text. You may output only one of the following three values:

- * -1: Negative sentiment (dissatisfaction, complaints, anger, disappointment, urging, questioning, etc.)
- * 0: Neutral (purely transactional inquiries, objective statements, no clear emotional tone)
- * 1: Positive sentiment (satisfaction, gratitude, praise, happiness, approval, etc.) Note: You must choose one value from -1, 0, or 1. No other values are allowed. Text: text Score:

- Figure 20: Prompt for sentiment classification in Utopian Tendency experiment.

You are a real user of an e commerce platform. Your core task is to infer the user’s communication style, personality traits, and current needs based on the given historical behavior sequence, and then simulate how the user would actually express themselves in a customer service conversation. Core principles. Data driven. All inferences must be based on objective evidence from historical behavior data, avoiding unfounded speculation or assumptions. Style consistency. The predicted expression should be consistent with the communication style, tone, and wording habits the user has shown in the past. Individual differences. Fully respect each user’s uniqueness and do not apply stereotypes or group labels. Context sensitivity. Make predictions by considering the context of the current conversation, the problems the user is encountering, and their emotional state. Authenticity first. Simulate what a real user would say, reflecting their unique communication style and current emotions. Input one is the user profile. This is the user’s basic platform information and can be used as background reference. Input two is the historical behavior trajectory. This is the user’s real operation record over a past period, covering multiple scenarios such as live streaming, marketplace, videos, and advertisements. Analyze the motivations and tendencies behind these behaviors to uncover the user’s communication style and personality traits. Input three is the current customer service conversation scenario. Output requirements. Please output only what the user would say. Do not output any other content, explanation, quotation marks, or analysis. Directly output the dialogue content. Your answer:

Figure 21: Prompt for text value prediction in the main experiment.

Please analyze this livestream cover image and extract the following basic information. Return the result in JSON format:

1. Live streaming type: What type of live streaming is this (E-commerce / gaming / chatting / talent performance, etc.) 2. Host characteristics: The host’s basic appearance features 3. Image text: Extract key text from the cover (**Note: Only extract core text such as live streaming title, product names, prices, promotional information, etc. Do NOT extract background decorative text, watermarks, platform labels, or meaningless symbols**) 4. Product information: If there are products, list the product type and brand 5. Scene description: Briefly describe the live streaming background and environment 6. Core selling point: What is the main focus or promotion of this live streaming

**Important rules:** - If any information is not present or unclear in the image, the corresponding field must be left empty - Only keep text that is directly relevant to the live streaming content - Do not output decorative text, platform UI text, or irrelevant symbols Please strictly follow the JSON format below and do not add any extra text: "Live streaming Type": "", "Host Characteristics": "", "Image Text": [], "Product Information": "", "Scene Description": "", "Core Selling Point": ""

Figure 22: Prompt for extracting and understanding key information from live streaming cover images.

# Role You are a professional data cleaning expert, skilled at extracting well-structured and semantically coherent main text from noisy OCR (Optical Character Recognition) and ASR (Automatic Speech Recognition) outputs. # Task Please perform deep cleaning on the provided original text. The goal is to transform it into clean text suitable for reading or further analysis. # Constraints & Rules 1. Deduplication and Merging: Remove adjacent paragraphs that are semantically identical or highly similar, and retain the one with the most complete information.

- 2. Noise Removal:

- * Remove meaningless garbled characters (e.g., AC:BU526, IC·BQ528, within 50 meters, and other interfering information).
- * Filter excessively redundant filler words, such as repeated occurrences of “uh,” “ah,” “that is to say,” retaining only those necessary for context.

- 3. Semantic Correction:

- * Correct obvious recognition errors (e.g., change “cumin cowhide” to “naturally revealed,” or infer based on context; if the correct meaning cannot be determined, keep the original).
- * Complete broken sentences and add commas, periods, or question marks appropriately based on tone and emphasis. 4. Formatting Standards:
- * Unify full-width and half-width punctuation.
- * Remove unnecessary leading/trailing spaces or special symbols. # Input Data {text} # Output Format Only output the final cleaned text. Do not explain the modification process.

Figure 23: Prompt for raw data cleaning.

You are a user behavior analysis expert focused on causal inference. Your task is to analyze a user’s historical interaction sequence, identify the key causal events that lead to a target behavior, and explain their roles. Input data includes the user history sequence and the target behavior. Task description: Review the entire history and identify key events that meaningfully contribute to the target behavior. Remove irrelevant noise such as random or unrelated actions. For each key event, evaluate its contribution strength on a scale from

- 4 to 10, where lower scores indicate weak background relevance and higher scores indicate decisive impact. Classify each event as one of the following types: Root Cause, meaning an early event that first sparked interest. Interest Accumulation, meaning a mid stage event that strengthened understanding or intent. Direct Trigger, meaning an event that occurred shortly before and directly caused the target behavior. If an event is part of a causal chain, indicate which earlier event it was derived from. Output requirements: Return only structured JSON data containing the selected key events. Each event should include a score, a logical classification, a brief Chinese explanation of how it contributed to the target behavior, and the ID of the preceding event if applicable. Do not include any additional text or formatting.

- Figure 24: Prompt for causal chain identification.

Please analyze the level of politeness in the following "user utterance". Please provide independent scores based on the following 5 dimensions. Each dimension should be scored on an integer scale from 0 to 4: Definition of Scoring Dimensions: # Politeness Markers 0 = No polite expressions at all; 1 = Occasional simple polite words; 2 = Some polite expressions are present; 3 = Clearly uses multiple polite expressions; 4 = Extensively uses politeness markers such as thanks, apologies, and courteous phrasing; # Hedging Language 0 = Direct, imperative, no softening; 1 = Slightly softened; 2 = Some vague or tactful phrasing is present; 3 = Multiple instances of softened tone; 4 = Extensive use of clear mitigating expressions such as "could you", "would it be possible", "perhaps", etc; # Blame Avoidance 0 = Strong blaming, accusatory, or aggressive language; 1 = Clearly assigns blame; 2 = Some blame is present but relatively restrained; 3

= Generally avoids direct blame; 4 = Completely avoids blame and may even take responsibility for possible misunderstanding; # Emotional Control 0 = Emotional outburst (many exclamation marks, insults, strongly negative wording); 1 = Clearly emotionally agitated; 2 = Some emotional expression is present; 3 = Emotion is fairly restrained; 4 = Completely calm and rational in expression; # Face Saving 0 = Commanding or oppressive wording; 1 = Strong pressure is applied; 2 = Neutral expression; 3 = Respects the other party’s position; 4

= Clearly preserves the other party’s face, showing understanding or consideration; Scoring Requirements: All dimensions must be integers from 0 to 4.

- Do not output explanations; - Do not output any extra text; - Output valid JSON only; Output Format: {"Politeness Markers": integer, "Hedging Language": integer, "Blame Avoidance": integer, "Emotional Control": integer, "Face Saving": integer} Now please evaluate the following text: {text}

- Figure 25: Prompt for language style comparison.

### E Case Study

To facilitate an intuitive understanding of our evaluation pipeline, we provide qualitative examples across multiple representative application scenarios. Specifically, we select four real-world settings: behavior prediction in live-streaming (Table 3), behavior prediction in E-commerce (Table 4), behavior prediction in video browsing scenarios (Table 5), and text prediction in customer service dialogues (Table 6) on OmniBehavior benchmark.

For each scenario, we present the raw model inputs, the outputs generated by all models evaluated in the main experiments, as well as the corresponding ground-truth user behaviors or responses. These examples are drawn directly from the evaluation datasets and are representative of typical user interactions in each scenario.

We emphasize that all examples shown in this appendix are anonymized and have been processed in accordance with data privacy and usage policies.

#### E.1 Case 1

- Table 3 shows a representative case study for the live streaming commenting prediction task. Models are required to infer user intentions based on demographic attributes, historical behavior sequences, and the current live streaming context. The ground truth indicates that the user posts a comment in the live streaming. While several models correctly predict the outcome, others fail to capture the behavior, demonstrating the challenge of modeling realistic user decision processes from long-horizon and heterogeneous interaction histories.

Prompt You are a real user of the Kuaishou platform. Your core task is: based on the given sequence of historical behaviors, infer the user’s interest preferences, spending power, and personality traits, and then simulate how they would realistically make decisions in a current scenario accordingly.

User Profile This is a female user on the platform aged 31–40, with a high school education or below, with a partner, and has children.

###### Historical Behavior Trace

This is the user’s real operation record over a past period. Please carefully analyze the motivations and tendencies behind these behaviors, and uncover the user’s implicit long-term preferences and short-term intentions:

- [Behavior 1] Time: 2025-10-14 14:16:24 Scene: Video browsing Details: This is a video titled "I really don’t want to miss you at all. You truly are a very good partner. Let’s Talk for fun.". The creator has 83,000 followers. The video is 23 seconds long. It currently has 17 impressions, 3 plays, and 1 full play. The video was published on 2024-03-06 11:57:19. Reaction: Watched for 6 seconds.
- [Behavior 2] Time: 2025-10-14 14:56:03 Scene: live streaming Details: This is a live streaming titled "New Streamer First Broadcast Traffic Sharing". The live streaming type is an entertainment-oriented, non-commerce live streaming. The live streaming has accumulated 8,974 viewers, with total views reaching 855,000. The live streaming currently has 29,000 likes and 9,022 comments. Reaction: Stayed in the live streaming for 46 minutes and 5 seconds, watched 4 times, liked the streamer 17 times, and sent 3 live chat messages......

###### Current Test Scenario

The user is currently encountering the following scenario: The current time is 2025-10-18 23:56:20. The user encounters a [live streaming] scenario. The detailed scenario information is as follows: This is a live streaming titled "Let’s Make Friends Together". The live streaming type is categorized as "Other," and it is an entertainment-oriented, non-commerce live streaming. The live streaming has accumulated 8,619 viewers, with total views reaching 13.481 million. The cumulative watch time has reached 83,000 hours. The live streaming cover includes the following: Streamer characteristics: Female, black hair tied up, wearing refined makeup, dressed in a black top, resting her face on her hand with a smile. Image text: "It’s getting cold, take good care of yourself." Scene description: Indoor background with green plant decorations; the overall image includes heart-shaped and star-like filter effects. The live streaming currently has 26,000 likes and 17,000 comments.

Prediction Task Please answer the following question from the perspective of this user: Will the user send a comment in the current live room? Your answer:

Ground Truth Human Yes

Model Predictions Claude-Opus-4.5 Yes ✓ Claude-Sonnet-4.5 No Claude-Haiku-4.5 No Claude-Sonnet-4 Yes ✓ Gemini-3-Flash Yes ✓ GPT-5.2 Yes ✓ GPT-4o Yes ✓ GLM-4.7 No DeepSeek-V3 Yes ✓ Kimi-K2-Instruct Yes ✓ Qwen3-235B Yes ✓

- Table 3: Comparison of prediction results for "Comment" behavior in live streamings scenario.

- E.2 Case 2
- Table 4 shows a representative case study for the E-commerce add-to-cart prediction task. Models must infer user intentions from demographic attributes, historical behavior sequences, and the current product context. The ground truth indicates that the user does not add the product to the cart, yet most models predict the opposite outcome. This failure suggests that current LLM-based simulators tend to overestimate user engagement or purchase intent, revealing limitations in capturing subtle preference signals from long behavioral histories.

Prompt You are a real user of the Kuaishou platform. Your core task is: based on the given sequence of historical behaviors, infer the user’s interest preferences, spending power, and personality traits, and then simulate how they would realistically make decisions in a current scenario accordingly.

User Profile This is a male user on the platform, aged 50+, with a high school education or below. He has a partner and children.

###### Historical Behavior Trace

This is the user’s real operation record over a past period. Please carefully analyze the motivations and tendencies behind these behaviors, and uncover the user’s implicit long-term preferences and short-term intentions:

- [Behavior 1]

- Time: 2025-10-09 10:16:46 Scenario: Video Browsing Title: "Does anyone support rural stoneground flour?" Author Stats: 255k followers. Video Duration: 29 seconds. Performance Metrics: 100,924 impressions, 93,326 views, 31,165 full views, 740 likes, 104 comments, 20 shares, 87 saves, 3 downloads. Publish Time: 2025-10-09 06:24:07. Reaction: Watched for 55 seconds. Used fast-forward and rewind operations. Completed the entire video.

[Behavior 2]

- Time: 2025-10-09 11:06:17 Scenario: Video Browsing Title: "American travel blogger shocked by food delivery robot in a Chongqing hotel". Author Stats: 86k followers. Video Duration: 1 minute 11 seconds. Performance Metrics: 16,470 impressions, 16,543 views, 5,208 full views, 335 likes, 16 comments, 7 shares, 10 saves, 1 download. Publish Time: 2025-10-04 18:11:31. Reaction: Watched for 1 minute 11 seconds. Completed the entire video.

[Behavior 3]

- Time: 2025-10-09 12:45:11 Scenario: Video Browsing Title: "Rogue software, auto-install, auto-billing—it’s going to cost me 3,160 RMB a year Uninstall Rogue Software" Author Stats: 1.169M followers. Video Duration: 38 seconds. Performance Metrics: 12,473 impressions, 6,620 views, 2,205 full views, 210 likes, 16 comments, 20 shares, 73 saves, 10 downloads. OCR: This rogue software is going to charge me 3,160 RMB a year. Uninstall it immediately. Just a minute ago, my phone suddenly alerted me to a 66 RMB charge. I checked my subscriptions, and it turned out to be this app. ASR: Just a minute ago, my phone alerted me to a 66 RMB charge. Publish Time: 2025-10-06 14:27:58. Reaction: Watched for 1 second.

Current Test Scenario

Timestamp: 2025-10-09 13:23:58 Scenario: E-commerce Scenario Details: Product: [Ma Chenggong] Yinghuai Stories Farmhouse Stone-Ground Flour; No Additives, No Extractions; Original Flavor Wheat Flour for Steamed Buns. Source: Order List Page. Product Category: Grains, Oils & Seasonings > Rice/Flour/Grains > Wheat Flour/Flour.

Prediction Task Please answer the following question from the perspective of this user: Will the user add the package to cart in the current shopping situation? Your answer:

Ground Truth Human No

Model Predictions Claude-Opus-4.5 Yes Claude-Sonnet-4.5 No ✓ Claude-Haiku-4.5 No ✓ Claude-Sonnet-4 Yes Gemini-3-Flash Yes GPT-5.2 Yes GPT-4o Yes GLM-4.7 Yes DeepSeek-V3 Yes Kimi-K2-Instruct Yes Qwen3-235B Yes

- Table 4: Comparison of prediction results for "Add-to-Cart" behavior in E-commerce scenario.

- E.3 Case 3
- Table 5 presents a representative case study for the video browsing "like" prediction task. Models must infer the user’s decision from demographic information, historical behavior sequences, and the current video context. The ground truth indicates that the user does not like the video, yet most models predict the opposite outcome. This pattern suggests that current LLM-based simulators tend to overestimate user engagement signals such as likes, highlighting the challenge of capturing nuanced user preferences from long and heterogeneous behavioral histories.

Prompt You are a real user of the Kuaishou platform. Your core task is: based on the given sequence of historical behaviors, infer the user’s interest preferences, spending power, and personality traits, and then simulate how they would realistically make decisions in a current scenario accordingly.

User Profile This is a male user on the platform, aged 18-23, with a high school education or below. He has no children.

###### Historical Behavior Trace

This is the user’s real operation record over a past period. Please carefully analyze the motivations and tendencies behind these behaviors, and uncover the user’s implicit long-term preferences and short-term intentions:

- [Behavior 1] Time: 2025-10-21 03:08:56 Scenario: Live Streaming Details: This is a live room. Live Cover Content: Image text: "Check in for 14 consecutive days to get 88 RMB. Current streak: 0/14 days. If the streak is broken, you will be ineligible for the grand prize. Check in today to claim." Reaction: Stayed in the live room for 42 seconds. Viewed 2 times.
- [Behavior 2] Time: 2025-10-21 06:54:09 Scenario: Live Streaming Title: "Money-making mini-games, instant withdrawals" Live Category: Other Scenarios / Entertainment (Non-e-commerce/Non-sales). Live Room Stats: 904 cumulative viewers, 355k total views, 1,933.7 total hours watched. Current interaction: 870 likes, 85 comments. Live Cover/Stream Description: Streamer Features: Pink long hair with straight bangs; wearing a pink zip-up jacket; exquisite makeup with orange-red lips; looking directly into the camera. Scene Description: The background features a large illustration of a pink anime character, with green plant leaves on the right. The overall environment follows an ACG or Cosplay style. Reaction: Dwell Time: Stayed in the live room for 13 minutes and 1 second. Viewed 1 time. Engagement: Liked the stream 56 times; shared the live room 1 time.
- [Behavior 3] Time: 2025-10-21 14:49:00 Scenario: Video browsing Details: This is a video titled "The most popular new beauty technology right now facial rejuvenation #anti aging folds #beauty transformation." The creator has seventy five thousand followers and has passed additional verification. The video duration is twenty six seconds...... Reaction: The user watched for thirty seconds, used fast forward playback, performed rewind viewing, and watched the entire video to completion.

###### Current Test Scenario

Timestamp: 2025-11-08 00:53:49 Scenario: Video Browsing Scenario Details: Title: "BeachParty PoolParty BeachVibes SexyBikini" Author Stats: 3,403 followers. Video Duration: 5 seconds. Performance Metrics: * Impressions: 8,755 Views: 2,567 Full Views (Completions): 1,511 Engagement: 69 likes, 7 comments, 4 shares, 17 saves, 12 downloads. Publish Time: 2025-11-04 19:04:11.

Prediction Task Please answer the following question from the perspective of this user: Do you predict the user will like this video? Your answer:

Ground Truth Human No

Model Predictions Claude-Opus-4.5 No ✓ Claude-Sonnet-4.5 No ✓ Claude-Haiku-4.5 No ✓ Claude-Sonnet-4 Yes Gemini-3-Flash Yes GPT-5.2 Yes GPT-4o Yes GLM-4.7 Yes DeepSeek-V3 No ✓ Kimi-K2-Instruct Yes Qwen3-235B Yes

Table 5: Comparison of prediction results for "Like" behavior in video browsing scenario.

#### E.4 Case 4

- Table 6 presents a case study of next-utterance prediction in an E-commerce customer service conversation. Given the user’s profile, historical behaviors, and the current dialogue context, models must generate the user’s most likely next response. The ground truth shows that the user questions the delayed shipment. While most models produce complaint-related responses, their tone and wording vary, highlighting the challenge of faithfully modeling realistic user language in adversarial service interactions.

Prompt You are a real user of the Kuaishou platform. Your core task is: based on the given sequence of historical behaviors, infer the user’s interest preferences, spending power, and personality traits, and then simulate how they would realistically make decisions in a current scenario accordingly.

User Profile This is a female user on the platform aged forty one to forty nine, with a high school education or below, who has a partner and has children.

###### Historical Behavior Trace

This is the user’s real operation record over a past period. Please carefully analyze the motivations and tendencies behind these behaviors, and uncover the user’s implicit long-term preferences and short-term intentions:

- [Behavior 1] Time: 2025-10-08 23:32:46 Scenario: Marketplace shopping Details: This is a product named Zhongjiu Ejiao Ferric Chloride Hemoglobin Capsules...... Reaction: The product was viewed but not added to the shopping cart and was successfully ordered and purchased.
- [Behavior 2] Time: 2025-10-21 06:54:09 Scenario: Live Streaming Title: "Money-making mini-games, instant withdrawals"...... Reaction: Dwell Time: Stayed in the live room for 13 minutes and 1 second. Viewed 1 time. Engagement: Liked the stream 56 times; shared the live room 1 time.

###### Current Test Scenario

This is a record of an e commerce customer service conversation. Order or inquiry background information This is a product inquiry type consultation. Conversation history: User: The electric comb still hasn’t been shipped. That’s fine, I will make Kuaishou fine you. Customer service: Human agent connected Customer service: I am here dear. How can I help you? The customer service agent will do their best to answer your questions.

Prediction Task Please answer the following question from the perspective of this user: What is the utterance they are most likely to say next? Your answer:

Ground Truth Human From September twenty second until now, your electric comb still has not been shipped?

Model Predictions Claude-Opus-4.5 I bought the electric comb days ago and it still hasn’t been shipped. Are you going to ship it or

not? If you don’t, I’ll file a complaint. Claude-Sonnet-4.5 It’s been so long and the electric comb still hasn’t been shipped. When exactly will it be sent? Don’t think ignoring me will solve it. I’ve already filed a complaint, just wait to be fined. Claude-Haiku-4.5 What is my order number? When will you ship it? I’ve already been waiting for several days. If you don’t ship it soon, I’ll report you. Claude-Sonnet-4 I’ve been urging you for days and it still hasn’t been shipped. With this kind of service attitude, I

will definitely complain to the end. Kuaishou will deal with scam shops like yours. Gemini-3-Flash Sorry, why hasn’t it been shipped yet? GPT-5.2 I’ll just ask you one thing. When exactly will this electric comb be shipped? If you don’t give me

a time, I’ll directly complain to the platform. GPT-4o Are you all pretending to be deaf? I’ve reported this several times already. Why haven’t you

handled it yet? GLM-4.7 Why hasn’t it been shipped yet? I already said I would file a complaint! DeepSeek-V3 Are you shipping this electric comb or not? If you don’t ship it, I’ll complain to the Kuaishou

platform and get you fined. Kimi-K2-Instruct I’ll give you my order number and file a complaint right now. I’ll get you heavily fined! Qwen3-235B If you don’t ship it soon, I’ll really complain to Kuaishou. Don’t think I won’t dare. The order

has been overdue for more than seven days. According to the rules, you owe me compensation. You must give me an explanation today.

Table 6: Comparison of utterance prediction results from models in customer service conversations within the E-commerce scenario, with a particular focus on cases where customers exhibit extreme dissatisfaction.

