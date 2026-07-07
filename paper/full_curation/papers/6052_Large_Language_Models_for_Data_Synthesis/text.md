# arXiv:2505.14752v4[cs.LG]8Jun2026

## LLMSynthor: Macro-Aligned Micro-Records Synthesis with Large Language Models

Yihong Tang

McGill University Montreal, Quebec, Canada yihong.tang@mail.mcgill.ca

Tong Nie

The Hong Kong Polytechnic University Hong Kong SAR, China tong.nie@connect.polyu.hk

Menglin Kong

McGill University Montreal, Quebec, Canada menglin.kong@mail.mcgill.ca

Wei Ma

The Hong Kong Polytechnic University Hong Kong SAR, China wei.w.ma@polyu.edu.hk

Junlin He

The Hong Kong Polytechnic University Hong Kong SAR, China junlin.he@polyu.edu.hk

Lijun Sun∗

McGill University Montreal, Quebec, Canada lijun.sun@mcgill.ca

### Abstract

Macro-aligned micro-records are essential for simulations in social science and urban studies. For instance, epidemic models of urban disease spread are only credible when micro-level records reproduce realistic individual mobility and contact patterns, while macro-level aggregates match macro-statistics such as case counts or travel flows. Still, collecting large-scale fine-grained data is often impractical, leaving researchers with only macro-statistics. While Large Language Models (LLMs) can generate realistic micro-records using rich real-world priors learned from vast corpora, naive recordby-record sampling is inefficient and fails to enforce alignment with target macro-statistics. Given this, we propose LlmSynthor, a framework that transforms a pre-trained LLM into a macroaware simulator capable of synthesizing realistic micro-records aligned with given macro-statistics. LlmSynthor incrementally constructs a synthetic dataset by iteratively generating batches of micro-records that reduce discrepancies between synthetic and target macro-statistics. By treating the LLM as a nonparametric copula over joint variable dependencies, the framework ensures alignment with target marginals and joint distributions. To improve efficiency, we introduce LLM Proposal Sampling, where the LLM generates discrepancy-guided proposals specifying variable constraints and record counts. Evaluations on synthetic and realworld datasets (mobility, e-commerce, population) encompassing diverse formats and settings show that LlmSynthor achieves high record realism, statistical fidelity, and practical utility, positioning it broadly applicable across economics, social science, urban studies, and beyond. Source codes of LlmSynthor are available at https://github.com/YihongT/LLMSynthor.

### CCS Concepts

#### • Computing methodologies → Simulation tools; Artificial intelligence; • Information systems → Data mining.

∗Corresponding author.

This work is licensed under a Creative Commons Attribution 4.0 International License. KDD 2026, Jeju Island, Republic of Korea.

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2259-2/2026/08 https://doi.org/10.1145/3770855.3818965

### Keywords

Dataset Synthesis; Urban Studies; Social Simulation; LLM

ACM Reference Format:

Yihong Tang, Menglin Kong, Junlin He, Tong Nie, Wei Ma, and Lijun Sun. 2026. LLMSynthor: Macro-Aligned Micro-Records Synthesis with Large Language Models. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD 2026), August 9–13, 2026, Jeju Island, Republic of Korea. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3770855.3818965

### 1 Introduction

High-stakes decisions in domains like public health and urban planning are increasingly supported by agent-based simulations of complex human behavior. At the micro-level, individual records capture behavioral detail such as mobility and contact patterns, which drive realistic dynamics. At the macro-level, aggregated statistics ensure consistency with population-level trends. Only when micro-records are collectively aligned with real-world macrostatistics can such simulations yield valid insights [14], yet these data are unattainable because large-scale collection is infeasible due to both prohibitive costs and stringent privacy constraints [5]. Consequently, researchers and policymakers must rely on macrostatistics, such as census reports, leaving a critical gap between macro-level observations and micro-level representations [46]. The challenge, therefore, is to synthesize realistic micro-records that are statistically faithful to the available macro-statistics.

This micro-macro synthesis task, however, is beyond the capabilities of existing generative paradigms. Current methods, from classical statistical models to modern deep generative networks [11, 12], are ill-equipped, as they all require access to a large volume of micro-records that are unavailable for model fitting. Furthermore, their reliance on rigid parametric assumptions or implicit model biases often leads to the generation of unrealistic records, such as a six-year-old with a doctorate, necessitating inefficient post-hoc fixes like rejection sampling. Most of these approaches also lack the generality to handle the heterogeneous or unstructured data common in social sciences and urban studies [24, 33], or require extensive manual engineering [31]. This highlights the need for a new framework that can synthesize realistic and complex micro-records that are statistically grounded, guided solely by macro-statistics.

The emergenceof LargeLanguageModels (LLMs) offers a promising yet insufficient solution. Harnessing rich real-world priors

statistical control. Figure 1 contrasts our framework with existing paradigms. In summary, our main contributions are:

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

###### LLMSynthor

###### Limitations of Existing Methods

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

###### Generative Models (GMs)

###### Macro-Aligned Micro-Records Synthesis

[Figure 11]

[Figure 12]

Aligned

Macro-Statistics Micro-Records

[Figure 13]

[Figure 14]

Micro-Records Micro-Records

GMs

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

- • We introduce LlmSynthor, which transforms a pre-trained LLM into a macro-aware simulator for synthesizing micro-records that preserve realism while aligning with target macro-statistics.
- • We propose two core methodological insights: (i) a nonparametric copula interpretation of LLMs for modeling joint variable dependencies, and (ii) LLM Proposal Sampling for efficient, large-scale, and controlled data generation.
- • We design a discrepancy-guided synthesis process that iteratively extends and aligns the synthetic and the target macro-statistics, providing rigorous, dataset-level statistical control.
- • Evaluations on synthetic and real-world datasets (mobility, ecommerce, population) across diverse formats and settings show that LlmSynthor consistently outperforms expert baselines in record realism, statistical fidelity, and practical utility, while maintaining broad versatility.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

9a.m.

[Figure 27]

5p.m. 3a.m.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

LLMSynthor Aggregated

[Figure 44]

[Figure 45]

Real

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Efficient Data Format-Agnostic

Macro-Statistics Control

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Rely on Micro-Records

Data-Specific Unrealistic

[Figure 67]

[Figure 68]

[Figure 69]

###### Supports Various Downstream Applications

[Figure 70]

###### Large Language Models (LLMs)

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Realistic Micro-Records

[Figure 75]

Event Simulation

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Synthetic Micro-Records

[Figure 84]

###### Micro-Records Macro-Statistics

[Figure 85]

Record-by-Record

Congestion Forecast under a Concert

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

9a.m.

[Figure 98]

…

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

LLM

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Aggregated

Real

Infection Spread

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Who gets exposed? Along which paths?

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Macro-Aligned

Inefficient Generation No Macro-statistics Control

[Figure 125]

- Figure 1: Comparison of generative approaches and LlmSynthor. Left: Existing methods either rely on micro-records (Generative Models) or cannot enforce macro-level consistency (LLMs), leading to unrealistic or inefficient generation. Right: LlmSynthor synthesizes realistic micro-records directly from macro-statistics, enabling efficient, formatagnostic, and macro-aligned generation.

### 2 Related Work

learned from vast corpora, they excel at generating realistic, complex, and even unstructured micro-records without requiring any fine-tuning, making them powerful universal generative priors [1, 18, 21]. However, existing approaches fall short in practice. Finetuning LLMs requires large volumes of micro-records and substantial computational resources, which are often unavailable in practice. In-context or few-shot prompting can condition local generations, but it still operates record by record, making the process both inefficient and incapable of enforcing macro-statistical alignment [37]. This limitation becomes critical at scale, where population-level simulations demand the rapid generation of large synthetic datasets that are not only realistic at the micro level but also statistically faithful in aggregate. Consequently, standard LLM-based generation may be time-consuming and yield datasets that appear plausible individually yet deviate significantly from target macro-statistics. The fundamental challenge is to enable LLMs to generate largescale synthetic data that remains both computationally efficient and globally consistent with target macro-statistics.

Dataset Synthesis. Early data synthesis methods focused on explicit statistical control, using techniques like iterative proportional fitting (IPF) [4], Bayesian networks [31], and copula models [26, 28] to match marginals and preserve dependencies. While interpretable, these methods often rely on strong assumptions and face challenges with scalability and heterogeneity. Deep generative models, including VAEs [2, 36], GANs [3, 41], and recent diffusion- or flow-based models [16, 20, 44], have improved realism and high-dimensional modeling. However, they tend to entangle marginals with dependencies and often require expensive retraining for new domains. LLM-based methods, such as GReaT [6] and HARMONIC [38], treat structured data as natural language, enabling zero-shot transfer and broad domain coverage through autoregressive decoding. However, these methods lack direct control over marginal and joint distributions, are sample inefficient, and struggle to scale with large or heterogeneous datasets.

LLMs as Data Generators. LLMs have shown exceptional versatility as data generators across various domains. They have been used to augment data [10], create instruction-finetuning datasets [23, 39], generate tabular data [6, 38], synthesize executable code [25, 27], and produce personal mobility data aligned with user preferences[35]androutines[15]. LLMsalsogeneratequestion-answering pairs to enhance model robustness [8, 34] and privacy-preserving text via topic modeling [32]. While existing methods excel at ensuring the semantic or functional quality of individual records, they often fail to control global statistical properties of the dataset. Some works introduce limited control, such as ensuring topic completeness or data coverage [45]. In parallel, recent work studies LLMs as attributed training-data generators and behavioral simulators, focusing on diversity, bias, and persona alignment rather than explicit aggregate control [13, 43]. However, these approaches lack explicit macroscopic statistical control. Most methods generate data record by record or in small, independent batches, without enforcing global statistical properties, highlighting the gap between generating realistic micro-records and ensuring statistical fidelity across the entire dataset, which is directly addressed by LlmSynthor.

To address these challenges, we present LlmSynthor, a framework that transforms a pre-trained LLM into a macro-aware simulator for micro-record synthesis. LlmSynthor incrementally constructs a synthetic dataset through an iterative feedback loop guided solely by target macro-statistics. At each iteration, LlmSynthor quantifies the discrepancy between the synthetic and target macrostatistics, and prompts the LLM to generate a corrective batch of micro-records that reduces this gap. This process is enabled by two core technical innovations. First, we interpret the LLM as a powerful nonparametric copula, enabling it to capture complex, nonlinear joint dependencies across variables without imposing rigid statistical assumptions. This allows LlmSynthor to align the synthetic data with the target by matching all available marginal and joint macro-statistics. Second, to overcome generation efficiency bottlenecks, we introduce LLM Proposal Sampling, where the LLM creates a generation plan of micro-record proposals, each defining a localized joint distribution over all variables with an associated generation count. This approach uniquely combines the LLM’s rich prior knowledge for micro-level realism with rigorous, macro-level

### 3 Methodology

- Definition 1 (Micro-record and Dataset). Let V be a predefined set of variables. A micro-record 𝑥 is defined as a set of variable-value pairs: 𝑥 = {(𝑣𝑖,𝑎𝑖)}𝑑𝑖=𝑥1, where 𝑣𝑖 ∈ V and 𝑎𝑖 is its value. A dataset is defined as a collection of such records, denoted by D = {𝑥𝑗}|D|𝑗=1 .
- Definition 2 (Macro-statistics). Let D be a dataset of micro-records. We define an operator Φ that maps the dataset into a set of macro-

statistics S, denoted as: Φ : D ↦→ S, where S = {𝜙𝑖}𝑖|S|=1. Here, each element 𝜙𝑖 represents a specific aggregated statistic derived from D.

Problem 1 (Macro-aligned Micro-records Synthesis). Given a set of target macro-statistics Starget, the objective is to construct a dataset of 𝑛 realistic micro-records, Dsynth = {𝑥ˆ𝑗}𝑛𝑗=1, such that the macrostatistics induced from Dsynth closely align with Starget. Formally, we seek minDsynth 𝑄 Φ(Dsynth), Starget , where Φ is the aggregation operator defined in Definition 2, and 𝑄 is a suitable discrepancy measure in the macro-statistics space.

In this work, a micro-record represents a data record corresponding to an individual entity. Each micro-record consists of a set of variables and their associated values. To accommodate the flexibility of unstructured data, the number of variables 𝑑𝑥 is allowed to vary across records. Macro-statistics summarize the distributional properties of a dataset and serve as the alignment targets for synthetic micro-records. In practice, macro-statistics are often the only information available from external sources. Formally, the statistics in S can be categorized into two types: (1) Marginal statistics, which describe the distribution of a single variable (e.g., a frequency vector of education levels), and (2) Joint statistics, which capture dependencies among multiple variables (e.g., a contingency table of education by employment status). While the aggregation operator Φ can, in principle, compute arbitrary statistics, its specific instantiation is application-dependent and determined by the macro-statistics available in a given context.

Consider a mobility dataset. A micro-record may represent a single trip,forexample𝑥 = {(origin,‘Times Square’), (destination, ‘Central Park’), (mode, ‘Bike’), (time, 17)}. Here, the variable set is V = {origin, destination, mode, time}, and each micro-record assigns concrete values to these variables. Alternatively, a microrecord may be unstructured, such as an entire daily travel diary, where the number of trips varies across individuals. In real-world settings, only macro-statistics are available from external sources,

denoted by Stargetmob . These may include marginal statistics, such as the frequency distribution of transportation modes, and joint statistics, such as a contingency table over origin-destination pairs. Ac-

cordingly, the aggregation operator Φmobtarget, defined in Definition 2, is instantiated for the mobility domain so that Φmobtarget(Dsynthmob ) produces statistics directly comparable to Stargetmob . The objective is therefore to construct a synthetic dataset of micro-records Dsynthmob such that Φmobtarget(Dsynthmob ) closely matches the target macro-statistics Stargetmob . For instance, if the target specifies that 30% of trips are made by bike (a marginal statistic) and 10% travel from Times Square to Central Park (a joint statistic), then the aggregated synthetic dataset must reproduce these proportions.

### 3.1 Overview

LlmSynthor comprises two core components: (i) Variable Dependency Inference, in which the LLM serves as a copula-like mechanism to capture dependencies among variables and identify informative joint macro-statistics beyond those explicitly provided; and (ii) Discrepancy-Guided Iterative Synthesis, where, at each iteration 𝑡, the LLM generates a batch of micro-records to progressively reduce the discrepancy 𝑄. An overview of the framework is shown in Figure 2, while a simplified illustrative example of the synthesis process is provided in Figure 10 in Appendix B.

### 3.2 Variable Dependency Inference

LlmSynthor operates exclusively on macro-statistics, which provide complementary but incomplete views of the underlying data distribution. While marginal statistics are indispensable for preserving the basic distributions of individual variables, joint macrostatistics may be either redundant or missing, depending on data availability and collection constraints. The goal of dependency inference is to identify an informative set of variable combinations whose dependencies are essential for faithful synthesis.

We draw inspiration from copula theory [30], which decomposes a multivariate distribution into marginal distributions and a dependency structure. Departing from parametric copula models, we interpret a pre-trained LLM as a nonparametric, semantics-aware copula that can infer salient dependency structures directly from macro-level information. Specifically, given the variable set V, the

LLM infers a collection of variable subsets C = {𝑐𝑘}𝑘|C|=1, where each 𝑐𝑘 ⊆ V represents a group of variables expected to exhibit strong statistical dependence. This inference leverages both (i) the semantic relationships encoded in variable names (e.g., “education” and “income”) and (ii) the available macro-statistics Starget:

C ∼ LLM pcopula(Starget, V) , (1)

where the prompt pcopula elicits informative variable combinations (see Appendix C.2). For example, given the variable set V = {origin,destination, mode, time},the LLMmayinferdependency

subsets {𝑐1 = [origin, destination, time], 𝑐2 = [time, activity]}, reflecting time-dependent travel demand and activity patterns.

Having inferred C, we retain all marginal statistics and selectively expand or filter joint macro-statistics in Starget to obtain an informative target set StargetC . In practice, StargetC is restricted to joint statistics that are explicitly available or can be reliably specified by practitioners. When certain inferred dependencies are not supported by available macro-statistics, they may be excluded from C. Alternatively, in settings where approximate control is acceptable, LLMs can be used as an optional mechanism to suggest or approximate missing joint statistics. The aggregation operator is then updated as Φtarget ↦→ ΦtargetC , ensuring that all induced statistics from any dataset are directly comparable to StargetC .

Figure 2(a) illustrates this process. In the mobility example, the LLM identifies a dependency𝑐1 over [origin, destination, time], leading to the replacement of the original OD statistic 𝜙OD with a time-conditioned statistic 𝜙TOD. By contrast, the dependency [time, activity] is already supported by available macro-statistics, and the corresponding statistic 𝜙TA is preserved.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

##### (b) Discrepancy-Guided Iterative Synthesis (Iteration t)

##### (a) Variable Dependency Inference

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

###### Input: Target Macro-Statistics & Variables

###### Updating Synthetic Macro-Statistics

###### Discrepancy Attribution LLM Proposal Sampling

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

###### Last Iteration’s Records

Target Macro-Statistics

Synthetic Macro-Statistics

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

𝓓𝒔𝒚𝒏𝒕𝒉(𝒕) = 𝓓𝒃𝒂𝒕𝒄𝒉(𝒕 𝟏) ∪ 𝓓𝒔𝒚𝒏𝒕𝒉(𝒕 𝟏)

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

###### Marginal Joint

[Figure 175]

Target Macro-Statistics

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Guide base on Discrepancies

[Figure 182]

[Figure 183]

[Figure 184]

Cumulate

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

𝓓𝒔𝒚𝒏𝒕𝒉(𝒕)

[Figure 190]

###### Cumulated Micro-Records

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

5p.m. … 10a.m.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

𝓢𝒔𝒚𝒏𝒕𝒉𝒕 𝓢𝒕𝒂𝒓𝒈𝒆𝒕𝓒

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

###### Variable Dependency Inference

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

𝓢𝒕𝒂𝒓𝒈𝒆𝒕𝓒 𝓢𝒔𝒚𝒏𝒕𝒉𝒕

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

𝚫(𝒕) = 𝓠( , )

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

𝚽𝒕𝒂𝒓𝒈𝒆𝒕𝓒 (𝓓𝒔𝒚𝒏𝒕𝒉(𝒕) )

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Aggregate

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

Discrepancy Attribution

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

###### Sampling

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

Synthetic Macro-Statistics 𝓢𝒔𝒚𝒏𝒕𝒉(𝒕)

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Top Discrepancies (𝓢𝒕𝒂𝒓𝒈𝒆𝒕𝓒 − 𝓢𝒔𝒚𝒏𝒕𝒉𝒕 )

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

A Batch of Micro-Records 𝓓𝒃𝒂𝒕𝒄𝒉(𝒕)

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

𝓢𝒕𝒂𝒓𝒈𝒆𝒕 𝒄 𝓢𝒕𝒂𝒓𝒈𝒆𝒕𝓒

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

Modify Joints

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

- Figure 2: Overview of LlmSynthor. The system (a) first uses an LLM to infer key variable dependencies, and then modifies the target macro-statistics to align with these inferred dependencies; (b) it then performs discrepancy-guided iterative synthesis. In each iteration, the newly generated micro-records are aggregated to form updated synthetic macro-statistics, which are compared with the target. The differences are attributed to specific variable groups, and guide the LLM to propose new microrecords that correct discrepancies. This feedback loop gradually aligns the synthetic data with the target macro-statistics.

### 3.3 Discrepancy-Guided Iterative Synthesis

2. Discrepancy Attribution. Given the updated synthetic macrostatistics Ssynth(𝑡) and the target macro-statistics StargetC , we next quantify their discrepancy 𝑄 StargetC , Ssynth(𝑡) , defined by:

With the refined target macro-statistics StargetC and the corresponding aggregation operator ΦtargetC in place, LlmSynthor enters a discrepancy-guided iterative synthesis loop that progressively constructs the synthetic dataset Dsynth. At each iteration 𝑡, (i) LlmSynthor first aggregates the current cumulative dataset to obtain synthetic macro-statistics Ssynth(𝑡) = ΦtargetC (Dsynth(𝑡) ), and (ii) measures the discrepancy Δ(𝑡) = 𝑄(StargetC , Ssynth(𝑡) ) between the synthetic and target statistics. Conditioned on these discrepancy signals, (iii) the LLM generates a batch of micro-records Dbatch(𝑡) that is designed to reduce Δ(𝑡), after which the cumulative dataset is updated as Dsynth(𝑡+1) = Dsynth(𝑡) ∪ Dbatch(𝑡) . This closed-loop process repeats until the synthetic dataset achieves satisfactory alignment with the target macro-statistics. We now elaborate on each step.

Δ(𝑡) = 𝛿1(𝑡),𝛿2(𝑡), . . .,𝛿(𝑡)

= 𝑄 StargetC , Ssynth(𝑡) , (2)

|StargetC |

where 𝑄(·, ·) is a discrepancy function applied element-wise to each macro-statistic. In this work, we implement 𝑄 as the directed

frequency difference StargetC − Ssynth(𝑡) . Positive values indicate underrepresented portions of the synthetic data relative to the target, while negative values indicate overrepresentation. These discrepancy signals explicitly identify which macro-level patterns require

correction and in which direction. Because ΦtargetC (·) operates on a discretized representation, each discrepancy is associated with specific bins or variable combinations (e.g., “time = 5-6 p.m. and mode = bike”). As a result, discrepancies are not only quantitative but also attributable to concrete subsets of variables, enabling them to be translated into actionable guidance for later generations.

1. Updating Synthetic Macro-statistics. At the beginning of iteration𝑡, the current synthetic dataset Dsynth(𝑡) is aggregated into macrostatisticsusingthetarget-aligned operatorSsynth(𝑡) = ΦtargetC Dsynth(𝑡) . This step produces a macro-level summary of the cumulative synthetic data that is directly comparable to the target statistics.

For example, consider the directed discrepancy for transportation

modes: 𝛿𝑀(𝑡) = {bike : +30%, car : −20%}. This indicates that bike trips are underrepresented in the synthetic data by 30%, motivating

the generation of additional bike-related records. Figure 9 provides an illustrative example. Further details are in Appendix A.

To ensure consistency across heterogeneous variables, each

3. Discrepancy-Guided LLM Proposal Sampling. The discrepancy signals Δ(𝑡) are then used to guide the generation of a new batch of micro-records. Rather than generating records one by one, which is inefficient and difficult to enforce distribution control at the macro-level, we shift the role of the LLM from a direct generator to a high-level planner. Specifically, we introduce LLM Proposal Sampling, in which the LLM generates a set of 𝑚 proposals:

micro-record 𝑥 ∈ Dsynth(𝑡) is mapped into a unified discretized space. Discrete variables are represented as frequency vectors, while continuous variables are discretized into bins (e.g., quantile-based) following the same scheme used by the target macro-statistics. For joint statistics involving multiple variables, the corresponding bins define contingency tables. This unified representation ensures that synthetic macro-statistics are directly comparable to the target macro-statistics, both in discretization and semantic meaning.

{𝜋1, . . .,𝜋𝑚} ∼ LLM pproposal(V, C, Δ(𝑡)) , (3)

using a prompt pproposal that instructs the LLM to design proposals aimed at reducing the identified discrepancies while respecting the inferred dependency structure C. The complete implemented prompt pproposal is provided in Appendix C.

Each generated proposal 𝜋𝑖 specifies a localized joint configuration over all variables together with the number of micro-records to generate. For discrete variables, this corresponds to assigning specific categorical values (e.g., mode = bike); for continuous variables, it specifies value ranges (e.g., time ∈ [17, 20]), from which values are sampled uniformly for simplicity. By explicitly planning record counts, proposal sampling enables quantitatively targeted generation rather than uncontrolled sampling.

Importantly, all proposals are aligned with the discretization schemeusedbyΦtargetC ,ensuringthattheireffectsonmacro-statistics are predictable. At the same time, the LLM’s learned priors ensure that the proposed joint configurations remain realistic. An example is shown in Figure 2(b), where one proposal 𝜋1 specifies 50 records with origin “Brooklyn”, destination “Times Square”, mode “transit”, activity “Shop”, and time range [6, 9].

Finally, records sampled from these proposals form a batch Dbatch(𝑡) , which is merged into the cumulative dataset as Dsynth(𝑡+1) = Dsynth(𝑡) ∪ Dbatch(𝑡) . By operating at the level of proposals rather than individual records, LlmSynthor acts as a high-level distributional controller, efficiently translating macro-level discrepancy signals into targeted micro-level generation while preserving realism.

Algorithm 1 Iterative Synthesis

Require: Target macro-statistics Starget, aggregation operator Φtarget, discrepancy

measure 𝑄, iterations𝑇

- 1: Dsynth(0) ← ∅ ⊲ initialize synthetic dataset
- 2: C ← LLM pcopula(Starget, V) ⊲ variable dependency inference
- 3: StargetC ←C Starget; ΦtargetC ←C Φtarget ⊲ refine targets and operator
- 4: for 𝑡 = 1 to𝑇 do
- 5: Ssynth(𝑡) ← ΦtargetC (Dsynth(𝑡) ) ⊲ updating synthetic macro-statistics
- 6: Δ(𝑡) ← 𝑄 StargetC , Ssynth(𝑡) ⊲ discrepancy attribution
- 7: {𝜋𝑖(𝑡) } ← LLM pproposal(V, C, Δ(𝑡) ) ⊲ plan proposals
- 8: Dbatch(𝑡) ← 𝑖 {𝑥ˆ𝑖(𝑡) }, 𝑥ˆ𝑖(𝑡) ∼ 𝜋𝑖(𝑡) ⊲ LLM proposal sampling
- 9: Dsynth(𝑡+1) ← Dsynth(𝑡) ∪ Dbatch(𝑡) ⊲ update dataset
- 10: end for
- 11: return Dsynth(𝑇)

Iterative Synthesis. Algorithm 1 summarizes the discrepancyguided iterative synthesis process. After initializing an empty synthetic dataset and inferring the variable dependency structure C, the algorithm iteratively expands Dsynth through a closed-loop procedure. First, at iteration 𝑡, the current cumulative synthetic

dataset Dsynth(𝑡) is aggregated into macro-statistics using the refined operator ΦtargetC , and compared with the target macro-statistics StargetC via the discrepancy measure 𝑄, yielding discrepancy signals Δ(𝑡) that identify which macro-level patterns are under- or overrepresented. Second, conditioned on the discrepancy signals Δ(𝑡) and the inferred dependency structure C, the LLM performs proposal planning by generating a set of proposal distributions {𝜋𝑖(𝑡)}. Each proposal specifies a localized joint configuration over variables and a corresponding record count, and is designed to directly

target the dominant mismatches revealed by Δ(𝑡). Third, synthetic records are sampled from these proposals to form a batch Dbatch(𝑡) , which is merged into the cumulative dataset to obtain Dsynth(𝑡+1). By iteratively aggregating, attributing discrepancies, and generating targeted batches, the algorithm progressively reduces macro-level mismatches while preserving micro-level realism, steering the synthetic dataset toward alignment with the target macro-statistics.

### 4 Experiments

To comprehensively evaluate LlmSynthor, our experiments are structured to answer three research questions (RQs). RQ1: How effectively can LlmSynthor synthesize realistic and usable microrecords from limited aggregate-level macro-statistics? RQ2: How does LlmSynthor’s statistical fidelity compare to state-of-the-art models trained on full micro-record datasets? RQ3: How effectively can LlmSynthor synthesize realistic and unstructured microrecords, without task-specific manual engineering? We address these RQs using three practical tasks detailed below. We perform experiments using the Chat Completion mode of GPT-4.1-nano [29].

### 4.1 Mobility Synthesis

Mobility synthesis aims to generate a complete dataset of realistic micro-records, each detailing an individual’s time-stamped origindestination trip, activity, and transport mode. This is an essential capability for urban applications like transport planning and event simulation, as comprehensive, individual-level mobility data for an entire population is impractical to collect.

Task Setup. To answer RQ1, we design a mobility synthesis task that mirrors a common real-world constraint: fusing aggregate information from multiple complementary data sources. From Open PFLOW [17], we extract trips (origin, destination, timestamp) and assign transport modes. Since OpenPFLOW lacks activity labels, we incorporate time-activity patterns from LLMob [15]. This task tests the ability to align spatiotemporal and behavioral data by generating 30,000 trips in a day in Tokyo to match both macrostatistics. As existing methods cannot handle such Mixed-Source synthesis without manual adaptations, we focus on a qualitative assessment of LlmSynthor’s unique capabilities.

Results. The results provide strong evidence for the first component of RQ1: LlmSynthor’s ability to generate realistic microrecords. Figure 3 compares the synthetic data against the target macro-statistics. The time-activity heatmaps on the left show close alignment, accurately capturing commuting peaks and midday activity rises. The OD flow heatmaps during the morning peak confirm that the synthetic trips reproduce key spatial patterns, matching high-density areas. These findings demonstrate that the synthesized population, in aggregate, successfully reproduces the guiding macro-level patterns. Furthermore, as shown in Figures 4 and 5, the generated micro-records also exhibit realistic internal structures, such as realistic correlations between travel mode and distance, reflecting their micro-level realism.

Controllable Mobility Synthesis for Events Simulation. To address the second component of RQ1 concerning the utility of the generated data, we demonstrate a key advantage of our framework: the ability to effortlessly incorporate arbitrary context into the

Time-Activity Distribution

Morning peak (6-9 a.m.) demand flow intensity heatmap

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

Real

[Figure 371]

[Figure 372]

[Figure 373]

Synthetic Real

Simulated OD flow during a concert at Tokyo Dome (20-24)

[Figure 374]

Synthetic

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

Case Study

[Figure 380]

[Figure 381]

[Figure 382]

Synthetic Real

#### Figure 3: Real vs. synthetic mobility patterns.

synthesis process. We test LlmSynthor’s controllability in a “whatif” scenario. We simulate a concert at Tokyo Dome (20-24h) by simply adding the prompt < There will be a concert from 20-24 at Tokyo Dome > during proposal generation. As shown in Figure 3, this simple intervention causes LlmSynthor to generate a surge of trips to the event location while preserving realistic background flows. This demonstrates LlmSynthor ’s potential as a powerful tool for scenario planning, allowing policymakers to simulate the effects of large events using detailed synthetic micro-records.

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Synthetic

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Real

Figure 4: Flow intensity maps for mobility data across seven time intervals: 0-6, 6-9, 9-12, 12-14, 14-17, 17-20, and 20-24.

Figure 4 shows a detailed comparison of spatial-temporal flow intensity between real and synthetic data across seven time intervals throughout the day. Each map captures the aggregate origin and destination activity within the Tokyo metropolitan area during a specific time window. The synthetic data successfully preserves major spatial patterns such as morning and evening commute flows, also capturing temporal variations in trip density. This highlights the model’s ability to maintain realistic spatiotemporal dynamics.

[Figure 397]

[Figure 398]

[Figure 399]

#### Figure 5: Distribution comparisons for mobility variables.

Figure 5 presents additional joint distribution visualizations across key mobility attributes. The left plot illustrates the correlation between transport mode and travel distance, showing that

synthetic records preserve realistic distance-dependent mode preferences (e.g., longer trips by car). The middle and right plots show marginal distributions for transport modes and time intervals, further confirming strong alignment between real and synthetic mobility behavior. Together, these results demonstrate that LlmSynthor can faithfully reproduce both spatial structure and behavioral signals critical for urban simulation and mobility planning.

### 4.2 E-Commerce Transaction Synthesis

Having demonstrated LlmSynthor’s unique capabilities, we now evaluate its statistical fidelity against strong, established baselines to answer RQ2. We consider a controlled e-commerce transaction synthesis task with a tractable likelihood, enabling a rigorous comparison with state-of-the-art tabular data synthesis models. This experiment is designed to test whether LlmSynthor, operating solely on macro-statistics, can match or exceed the performance of models trained with full access to micro-level records.

Task Setup. We construct a controlled environment in which each transaction is generated from a known Bayesian network over six variables, {𝑣𝐴,𝑣𝐺,𝑣𝐿,𝑣𝐶,𝑣𝑋,𝑣𝑀}, corresponding to user_age, user_gender, user_location_tier, product_category, price, and payment_method. The underlying joint distribution follows a structured probabilistic graphical model: 𝑝(𝑣𝐴,𝑣𝐺,𝑣𝐿,𝑣𝐶,𝑣𝑋,𝑣𝑀) = 𝑝(𝑣𝐴) 𝑝(𝑣𝐺) 𝑝(𝑣𝐿) 𝑝(𝑣𝐶 | 𝑣𝐴,𝑣𝐺) 𝑝(𝑣𝑋 | 𝑣𝐶) 𝑝(𝑣𝑀 | 𝑣𝐿). This known generative process allows for a precise assessment of how well different synthesis methods capture ground-truth dependencies. We generate a reference dataset of 2,000 transactions from this model and treat it as ground truth. From this dataset, we derive the target macro-statistics including all marginal statistics and joint contingency tables over correlated variable groups defined by the Bayesian network. These macro-statistics constitute the only input provided to LlmSynthor. In contrast, all baseline methods are trained directly on the full set of 2,000 micro-records. This design enables a direct and fair comparison of statistical fidelity, highlighting whether the joint dependencies inferred by LlmSynthor from macro-statistics alone can recover the true underlying structure.

Baselines. Given the structured tabular nature of the task, we compare LlmSynthor against representative methods spanning major paradigms of tabular data synthesis, all of which are trained with full access to the micro-level records. These include: (1) VAE- and GAN-based models (TVAE, CTGAN); (2) GANs with explicit dependency modeling (CopulaGAN); (3) autoregressive transformers (GReaT); (4) diffusion-based models (TabSyn); (5) classical statistical alignment methods (IPF); (6) hybrid LLM-probabilistic approaches (Spada, which combines LLM-induced dependency graphs with normalizing flows); and (7) LLM-based in-context generation (LLMICL). To ensure a fair comparison, we apply rejection sampling across all methods to enforce basic record-level validity.

Results. The results provide a clear affirmative answer to RQ2. Despite operating solely on macro-statistics, LlmSynthor consistently outperforms all baselines trained on the full micro-record dataset across both statistical fidelity and downstream utility.

We evaluate the synthesized data from two complementary perspectives. First, to assess statistical fidelity, Tables 1 and 2 report performance on marginal and joint distributions using multiple

Was ↓ Gap ↓ Tvd ↓ Gap ↓ Tvd ↓ Gap ↓ Tvd ↓ Gap ↓ Was ↓ Gap ↓ Tvd ↓ Gap ↓ age gender location category price payment

Methods Rej.% ↓

TVAE [41] 1.7 ± 0.2 2.06 0.032 0.008 0.01 0.056 0.043 0.054 0.02 113.194 0.085 0.017 0.013 CTGAN [41] 5.1 ± 1.1 4.429 0.057 0.117 0.065 0.162 0.076 0.080 0.028 138.998 0.059 0.088 0.022 CopulaGAN [9] 4.8 ± 1.0 4.82 0.027 0.052 0.016 0.045 0.031 0.057 0.024 151.239 0.047 0.045 0.014 GReaT [6] 0.8 ± 0.1 2.862 0.052 0.016 0.009 0.039 0.020 0.045 0.027 169.866 0.104 0.009 0.012 TabSyn [44] 1.9 ± 0.3 1.196 0.012 0.012 0.022 0.007 0.022 0.045 0.01 114.12 0.067 0.028 0.005 Spada [42] 4.1 ± 0.9 1.42 0.025 0.013 0.019 0.003 0.022 0.016 0.024 54.603 0.023 0.025 0.009 IPF [19] 3.9 ± 0.6 5.27 0.087 0.018 0.008 0.026 0.016 0.012 0.024 139.957 0.078 0.035 0.013 LLM-ICL [7] 0.1 ± 0.0 20.61 0.194 0.026 0.026 0.024 0.014 0.441 0.215 279.743 0.205 0.163 0.081

LlmSynthor (ours) 0.3 ± 0.1 1.13 0.023 0.002 0.008 0.002 0.012 0.010 0.022 12.762 0.011 0.003 0.004

#### Table 1: Marginal alignment evaluation (↓ is better). Methods differ by supervision: IPF, LLM-ICL, and Ours (LlmSynthor) are purely macro-supervised; Spada is macro+micro supervised; while others are standard micro-supervised models. Metrics include Wasserstein distance (Was), Total Variation Distance (Tvd), and the classifier two-sample test gap (Gap, defined as |Acc − 0.5|).

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Ours GReaT Ours GReaT

Jsd ↓ Gap ↓ Jsd ↓ Gap ↓ Jsd ↓ Gap ↓

Methods

[𝑣𝐴,𝑣𝐺,𝑣𝐶] [𝑣𝐶,𝑣𝑋] [𝑣𝐿,𝑣𝑀] TVAE 0.23 0.074 0.245 0.106 0.185 0.051 CTGAN 0.133 0.055 0.298 0.098 0.145 0.076 CopulaGAN 0.133 0.057 0.280 0.102 0.069 0.018 GReaT 0.087 0.058 0.382 0.177 0.038 0.020 TabSyn 0.083 0.022 0.237 0.082 0.027 0.015 Spada 0.085 0.024 0.225 0.033 0.103 0.03 IPF 0.117 0.049 0.504 0.154 0.113 0.025 LLM-ICL 0.437 0.216 0.591 0.242 0.476 0.081 Ours 0.071 0.022 0.134 0.020 0.007 0.007

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Ours GReaT

Table 2: Joint evaluations.

Figure 6: Qualitative Distributions and Comparisons.

|Methods<br><br>|Rej. %<br><br>|Age Gender Location Category Price Payment Was Tvd Tvd Tvd Was Tvd<br><br>|[v_A, v_G, v_C] [v_C, v_X] [v_L, v_M] Jsd Jsd Jsd|
|---|---|---|---|
|Ours (GPT-4.1-nano) Ours (Qwen2.5-7B) Ours (Qwen2.5-3B)<br><br>|0.3 ± 0.1 0.5 ± 0.1 0.5 ± 0.1<br><br>|1.13 0.002 0.002 0.022 12.762 0.003 1.03 0.001 0.013 0.007 8.691 0.001 0.93 0.001 0.002 0.003 14.367 0.002<br><br>|0.071 0.134 0.007 0.089 0.16 0.021 0.082 0.198 0.139<br><br>|

- Table 3: Quantitative ablation study on LLM backbones. We compare the synthesis quality of LlmSynthor using GPT-4.1-nano, Qwen2.5-7B-Instruct, and Qwen2.5-3B-Instruct against the TabSyn baseline. The results highlight the framework’s adaptability across models of varying parameter sizes, with bold values indicating the best performance in each category.

metrics, including Wasserstein distance (W), Total Variation Distance (TVD), Jensen-Shannon Divergence (JSD), and the Classifier Two-Sample Test (C2ST) gap (|acc − 0.5|). Across all metrics, LlmSynthor achieves the lowest divergence scores, indicating the closest match to the ground-truth distributions. The visualizations in Figure 6 further confirm that LlmSynthor most faithfully preserves the target dependency structures. Second, we evaluate the practical downstream utility of the synthetic data. We derive two rule-based downstream labels from the original transaction variables: discount_propensity, which measures price sensitivity using category-normalized price deviation together with age, payment method, and location tier; and lifetime_value_band,

which approximates customer value from transaction price, product category, payment method, and age. Both labels are computed using fixed, interpretable rules applied identically to real and synthetic data. We then train standard classifiers (logistic regression, decision trees, and random forests) on synthetic data generated by each method. As shown in Figure 6, models trained on LlmSynthor’s synthetic data generalize best to real data, demonstrating that its superior statistical fidelity translates directly into improved performance on downstream machine learning efficiency tasks.

We further evaluate semantic realism using three soft plausibility rules: R1 rejects unrealistically low prices for large-item categories such as Electronics and Furniture & Appliances; R2

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

Figure 7: Diversity, macro-level alignment, and conditional bias rapidly converge toward ground truth as more synthetic records are generated, while LlmSynthor also shows flexible efficiency, outperforming GReaT at low iteration counts and achieving the highest fidelity with more iterations.

Methods R1 (%) ↓ R2 (%) ↓ R3 (%) ↓ Rej. (%) ↓

TVAE 1.6 ± 0.1 0.1 ± 0.1 0.1 ± 0.0 1.8 ± 0.1 CTGAN 2.9 ± 0.1 1.9 ± 0.1 0.3 ± 0.0 5.1 ± 0.1 CopulaGAN 2.1 ± 0.1 2.1 ± 0.1 0.6 ± 0.0 4.8 ± 0.1 GReaT 3.5 ± 0.2 0.4 ± 0.0 0.0 ± 0.0 3.9 ± 0.1 TabSyn 0.7 ± 0.1 0.1 ± 0.0 0.1 ± 0.0 0.8 ± 0.1 Spada 3.6 ± 0.1 0.2 ± 0.0 0.4 ± 0.0 4.1 ± 0.1 IPF 0.6 ± 0.3 0.5 ± 0.2 0.7 ± 0.0 1.8 ± 0.0 LLM-ICL 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0

GT (Real Data) 0.5 ± 0.1 0.0 ± 0.0 0.1 ± 0.0 0.6 ± 0.1 LlmSynthor 0.5 ± 0.1 0.0 ± 0.0 0.0 ± 0.0 0.5 ± 0.1

#### Table 4: Rule-wise and total rejection rates (%) across 5 runs of 2,000 generated e-commerce transactions per method.

rejects extremely high Food & Beverages spending by teenagers; and R3 rejects ultra-expensive Electronics purchases by very senior users. These rules target rare but meaningful cross-variable inconsistencies in e-commerce transactions. As shown in Table 4, LlmSynthor achieves a 0.5% rejection rate, matching the real-data reference (0.6%) and outperforming most baselines. While LLMICL has a slightly lower rejection rate, its weaker distributional fit shows that semantic validity alone is insufficient, LlmSynthor better balances plausibility and statistical fidelity. Finally, as shown in Figure 7, LlmSynthor maintains high diversity while progressively mitigating bias. Efficiency analysis further demonstrates the framework’s flexibility: with only 10 iterations, LlmSynthor outperforms GReaT in both speed and quality. Increasing iterations further enhances fidelity, offering a controllable trade-off between computational cost and performance. In Table 3, we conduct robustness checks demonstrating stable SOTA performance when using smaller, open-source models (Qwen2.5-Instruct with 3B and 7B).

### 4.3 Population Synthesis

HavingestablishedLlmSynthor’s quantitativeadvantagesonstructured data, we now address RQ3 by evaluating its ability to generalize to complex, unstructured data formats. We consider a real-world population synthesis task, where records are inherently unstructured due to varying household sizes and hierarchical householdperson relationships. This setting tests whether LlmSynthor ’s general-purpose, macro-aware framework can outperform specialized population synthesis models that typically rely on extensive task-specific modeling and manual constraints.

Task Setup. We use population microdata from the American Community Survey (ACS) for households in South Carolina. The dataset contains both household- and person-level attributes, resulting in variable-length records that pose a significant challenge for conventional tabular synthesis methods. After preprocessing, we obtain approximately 15,000 households. The objective is to generate a synthetic population that preserves complex joint distributions across demographic, socioeconomic, and household-structure variables. To evaluate practical fidelity, we define 16 policy-relevant queries (e.g., the proportion of multigenerational households) grouped into six thematic categories. These queries serve as a proxy for real-world utility by evaluating whether the synthetic population preserves meaningful joint distributions over combinations of demographic and household variables.

Baselines. We compare LlmSynthor against strong, specialized population synthesis methods: (1) CP, a tensor factorization approach; (2) HMM, a specifically designed hierarchical probabilistic model for population synthesis; and (3) NVI, a deep variational inference framework. These baselines represent state-of-the-art techniques tailored specifically to population synthesis, providing a rigorous benchmark for evaluating the generality of LlmSynthor.

Methods Rej. Rate% Demog. Employment Equity Household Mobility Vuln.

CP 73.9 0.54 1.02 5.79 2.34 1.47 0.86 HMM 57.8 0.56 0.32 4.23 2.01 0.48 0.91 NVI 96.8 0.53 0.27 5.49 2.06 0.24 1.06

Ours 13.3 0.21 0.2 0.25 0.13 0.35 0.37

#### Table 5: Rejection rate and category-wise MRE across queries.

Check NVI CP HMM Real Ours Overall rejection rates ↓

Household rejection rate 96.8 ± 0.3 73.9 ± 0.6 57.8 ± 0.5 0.0 ± 0.0 13.3 ± 0.4 Person rejection rate 45.9 ± 0.4 34.1 ± 0.7 27.8 ± 0.6 0.0 ± 0.0 6.3 ± 0.2

Household-level checks ↓ householder_adult_consist. 4.5 ± 0.2 4.4 ± 0.3 15.2 ± 0.5 0.0 ± 0.0 3.8 ± 0.2 persons_all_valid 96.7 ± 0.4 73.5 ± 0.6 56.1 ± 0.7 0.0 ± 0.0 10.7 ± 0.3

Person-level checks ↓

age_range 0.1 ± 0.0 0.5 ± 0.1 0.3 ± 0.1 0.0 ± 0.0 0.0 ± 0.0 race_valid 0.0 ± 0.0 0.8 ± 0.1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 employment_age_consist. 41.6 ± 0.7 30.9 ± 0.6 24.5 ± 0.5 0.0 ± 0.0 4.4 ± 0.2 education_age_consist. 42.7 ± 0.8 8.1 ± 0.4 10.7 ± 0.5 0.0 ± 0.0 2.2 ± 0.1

#### Table 6: Rejection rates (%) by validation rule across methods. Rules with zero rejections for all methods are omitted for brevity. “Consist.” denotes consistency checks.

Results. The results provide a clear affirmative answer to RQ3. As shown in Table 5, LlmSynthor achieves the lowest mean relative error (MRE) across all six categories of policy-relevant queries. For example, on equity-related metrics, the MRE is reduced from 4.23 (HMM) to 0.25. Consistent improvements are also observed for demographic, employment, mobility, and vulnerability-related

queries. While LlmSynthor does not outperform all baselines on every individual query, its consistent category-level superiority indicates a stronger ability to capture the complex joint dependencies inherent in population data, leading to substantially higher practical utility. Beyond statistical accuracy, LlmSynthor exhibits a significant advantage in micro-record realism. As shown in Table 6, population data is governed by strict logical constraints (e.g., a child cannot be designated as a householder), which many probabilistic models struggle to enforce. As a result, traditional baselines incur extremely high rejection rates, reaching 96.8% for NVI and 73.9% for CP. In contrast, LlmSynthor reduces the overall rejection rate to 13.3%. By leveraging the semantic reasoning capabilities encoded in LLMs, LlmSynthor naturally respects complex intra-household rules, such as valid role assignments and age-consistent attributes, without requiring explicit constraint programming.

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

BachelorDoctorateElementaryHighSchoolMasterPreschool

education

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

#### Figure 8: Generated population data by LlmSynthor.

Figure 8 compares the marginal distributions of real data and synthetic data generated by LlmSynthor. LlmSynthor closely matches the real distributions across both household-level and individual-levelvariables,including skewed variables such asincome and age, as well as categorical variables such as employment and race. This suggests that the iterative structure-guided mechanism improves fidelity in challenging, high-variance settings.

### 4.4 Discussion

Overall, the experimental results provide strong evidence that LlmSynthor effectively bridges the micro-macro gap in data synthesis. Across diverse domains and data formats, LlmSynthor can generate realistic micro-records using only multi-source macrostatistics (RQ1), achieve superior quantitative fidelity compared to baselines with full access to micro-level data (RQ2), and generalize to complex, unstructured population data without requiring task-specific engineering (RQ3). Together, these results highlight the potential of macro-aware, LLM-driven synthesis as a generalpurpose alternative to specialized data generation pipelines. Regarding privacy issues, LlmSynthor operates exclusively on aggregate macro-statistics. By iteratively steering synthetic data toward population-level targets, the framework decouples generation from any individual record. As a result, the synthesized data reflects distributional properties rather than specific individuals, substantially reducing the risk of direct re-identification. This design aligns with “privacy by aggregation” principles and suggests a promising pathway for integrating formal privacy mechanisms, such as differential privacy applied at the macro-statistic level, in future extensions.

### 5 Conclusion

In this work, we introduced LlmSynthor, a macro-aware framework that bridges the micro-macro gap in data synthesis. We address the fundamental challenge of generating realistic, individuallevel micro-records when only aggregate macro-statistics are available. Our key contribution is a shift in how LLMs are used for data generation: rather than sampling records independently, we repurpose a pre-trained LLM as a macro-aware simulator that operates within a discrepancy-guided feedback loop, ensuring datasetlevel statistical alignment. Extensive experiments demonstrate that LlmSynthor synthesizes realistic and useful micro-records from multi-source macro-statistics (RQ1), consistently outperforms stateof-the-art baselines trained on full micro-level data (RQ2), and generalizes effectively to complex, unstructured population data without requiring task-specific engineering (RQ3). Together, these results highlight the strength of macro-guided synthesis as a general and scalable alternative to specialized generative pipelines. By enabling statistically grounded micro-record generation under severe data constraints, LlmSynthor opens new opportunities for data-driven social science, agent-based simulation, and evidencebased policymaking. More broadly, this work suggests a principled pathway for combining the expressive priors of large language models with explicit statistical control, offering a foundation for building reliable, high-fidelity synthetic worlds across a wide range of scientific and societal applications.

### 6 Limitations and Ethical Considerations

Despite these strengths, several limitations remain. First, LLMs encode inherent semantic priors that may occasionally conflict with target macro-statistics, leading to biased or over-regularized generations; in practice, this effect can be mitigated through more constrained prompting or by selectively suppressing semantic cues. Second, the scalability of the framework is currently bounded by the context window and reasoning capacity of the underlying LLM, particularly in high-dimensional settings with many variables, though this limitation is expected to ease as LLM capabilities continue to advance. Third, while LlmSynthor is well suited for mixed-type micro-records, it is not designed to directly model perceptual or tightly sequential data such as images, videos, graphs, or raw time series; nevertheless, it can naturally serve as a high-level macrocontroller to guide domain-specific generators in these modalities. From an ethical and privacy perspective, LlmSynthor operates exclusively on aggregate macro-statistics rather than individuallevel records, decoupling generation from any specific individual and ensuring that synthesized data reflects distributional properties instead of memorizing or reproducing personal data.

### GenAI Disclosure

GPT-4.1-nano was used for output generation, and LLMs for language editing only; all study design, implementation, evaluation, findings, and interpretations were author-verified.

### Acknowledgement

This study was supported by the Natural Sciences and Engineering Research Council of Canada through NSERC ALLRP 602710-24.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774

(2023).

- [2] Patricia A Apellániz, Juan Parras, and Santiago Zazo. 2024. An improved tabular data generator with VAE-GMM integration. In 2024 32nd European Signal Processing Conference (EUSIPCO). IEEE, 1886–1890.
- [3] Mrinal Kanti Baowaly, Chia-Ching Lin, Chao-Lin Liu, and Kuan-Ta Chen. 2019. Synthesizing electronic health records using improved generative adversarial networks. Journal of the American Medical Informatics Association 26, 3 (2019), 228–241.
- [4] Richard J Beckman, Keith A Baggerly, and Michael D McKay. 1996. Creating synthetic baseline populations. Transportation Research Part A: Policy and Practice 30, 6 (1996), 415–429.
- [5] Steven M Bellovin, Preetam K Dutta, and Nathan Reitinger. 2019. Privacy and synthetic datasets. Stan. Tech. L. Rev. 22 (2019), 1.
- [6] Vadim Borisov, Kathrin Seßler, Tobias Leemann, Martin Pawelczyk, and Gjergji Kasneci. 2022. Language models are realistic tabular data generators. arXiv preprint arXiv:2210.06280 (2022).
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
- [8] Arijit Ghosh Chowdhury and Aman Chadha. 2023. Generative data augmentation using LLMs improves distributional robustness in question answering. arXiv preprint arXiv:2309.06358 (2023).
- [9] Brian d’Alessandro, Cathy O’Neil, and Tom LaGatta. 2017. Conscientious classification: A data scientist’s guide to discrimination-aware classification. Big data 5, 2 (2017), 120–134.
- [10] Bosheng Ding, Chengwei Qin, Ruochen Zhao, Tianze Luo, Xinze Li, Guizhen Chen, Wenhan Xia, Junjie Hu, Anh Tuan Luu, and Shafiq Joty. 2024. Data augmentation using large language models: Data perspectives, learning paradigms and challenges. arXiv preprint arXiv:2403.02990 (2024).
- [11] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. Advances in neural information processing systems 27 (2014).
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.
- [13] Samuel Holt, Max Ruiz Luyten, Antonin Berthon, and Mihaela van der Schaar.

2025. G-sim: Generative simulations with large language models and gradientfree calibration. arXiv preprint arXiv:2506.09272 (2025).

- [14] Na Jiang, Fuzhen Yin, Boyu Wang, and Andrew T Crooks. 2024. A large-scale geographically explicit synthetic population with social networks for the united states. Scientific Data 11, 1 (2024), 1204.
- [15] Wang Jiawei, Renhe Jiang, Chuang Yang, Zengqing Wu, Ryosuke Shibasaki, Noboru Koshizuka, Chuan Xiao, et al. 2024. Large language models as urban residents: An llm agent framework for personal mobility generation. Advances in Neural Information Processing Systems 37 (2024), 124547–124574.
- [16] Sanket Kamthe, Samuel Assefa, and Marc Deisenroth. 2021. Copula flows for synthetic data generation. arXiv preprint arXiv:2101.00598 (2021).
- [17] Takehiro Kashiyama, Yanbo Pang, and Yoshihide Sekimoto. 2017. Open PFLOW: Creation and evaluation of an open dataset for typical people mass movement in urban areas. Transportation research part C: emerging technologies 85 (2017), 249–267.
- [18] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems 35 (2022), 22199–22213.
- [19] Stanislav Kolenikov. 2014. Calibrating survey data using iterative proportional fitting (raking). The Stata Journal 14, 1 (2014), 22–59.
- [20] Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, and Artem Babenko. 2023. Tabddpm: Modelling tabular data with diffusion models. In International Conference on Machine Learning. PMLR, 17564–17579.
- [21] Teyun Kwon, Norman Di Palo, and Edward Johns. 2024. Language models as zero-shot trajectory generators. IEEE Robotics and Automation Letters (2024).
- [22] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.
- [23] Ming Li, Lichang Chen, Jiuhai Chen, Shwai He, Jiuxiang Gu, and Tianyi Zhou. 2024. Selective reflection-tuning: Student-selected data recycling for llm instruction-tuning. In Findings of the Association for Computational Linguistics ACL 2024. 16189–16211.
- [24] Chao Ma, Sebastian Tschiatschek, Richard Turner, José Miguel Hernández-Lobato, and Cheng Zhang. 2020. Vaem: a deep generative model for heterogeneous mixed type data. Advances in Neural Information Processing Systems 33 (2020), 11237– 11247.

- [25] Daniel J Mankowitz, Andrea Michi, Anton Zhernov, Marco Gelmi, Marco Selvi, Cosmin Paduraru, Edouard Leurent, Shariq Iqbal, Jean-Baptiste Lespiau, Alex Ahern, et al. 2023. Faster sorting algorithms discovered using deep reinforcement learning. Nature 618, 7964 (2023), 257–263.
- [26] Roger B Nelsen. 2006. An introduction to copulas. Springer.
- [27] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language modelforcode with multi-turn program synthesis. arXivpreprint arXiv:2203.13474

(2022).

- [28] Ostap Okhrin, Alexander Ristig, and Ya-Fei Xu. 2017. Copulae in high dimensions: an introduction. Applied quantitative finance (2017), 247–277.
- [29] Openai. 2025. Introducing GPT-4.1 in the API. https://openai.com/index/gpt-4-1/.
- [30] M Sklar. 1959. Fonctions de répartition à n dimensions et leurs marges. In Annales de l’ISUP, Vol. 8. 229–231.
- [31] Lijun Sun, Alexander Erath, and Ming Cai. 2018. A hierarchical mixture modeling framework for population synthesis. Transportation Research Part B: Methodological 114 (2018), 199–212.
- [32] Bowen Tan, Zheng Xu, Eric Xing, Zhiting Hu, and Shanshan Wu. 2025. Synthesizing privacy-preserving text data via finetuning without finetuning billion-scale llms. arXiv preprint arXiv:2503.12347 (2025).
- [33] Yihong Tang, Ao Qu, Andy HF Chow, William HK Lam, Sze Chun Wong, and Wei Ma. 2022. Domain adversarial spatial-temporal network: A transferable framework for short-term traffic forecasting across cities. In Proceedings of the 31st ACM international conference on information & knowledge management. 1905– 1915.
- [34] Yihong Tang, Ao Qu, Xujing Yu, Weipeng Deng, Jun Ma, Jinhua Zhao, and Lijun Sun. 2026. From street views to urban science: Discovering road safety factors with multimodal large language models. Transportation Research Part C: Emerging Technologies 188 (2026), 105692.
- [35] Yihong Tang, Zhaokai Wang, Ao Qu, Yihao Yan, Zhaofeng Wu, Dingyi Zhuang, Jushi Kai, Kebing Hou, Xiaotong Guo, Jinhua Zhao, et al. 2024. ItiNera: Integrating Spatial Optimization with Large Language Models for Open-domain Urban Itinerary Planning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track. 1413–1432.
- [36] Syed Mahir Tazwar, Max Knobbout, Enrique Hortal Quesada, and Mirela Popa.

2024. Tab-VAE: A Novel VAE for Generating Synthetic Tabular Data.. In ICPRAM. 17–26.

- [37] Xindi Wang, Mahsa Salmani, Parsa Omidi, Xiangyu Ren, Mehdi Rezagholizadeh, and Armaghan Eshaghi. 2024. Beyond the limits: A survey of techniques to extend the context length in large language models. arXiv preprint arXiv:2402.02244

(2024).

- [38] Yuxin Wang, Duanyu Feng, Yongfu Dai, Zhengyu Chen, Jimin Huang, Sophia Ananiadou, Qianqian Xie, and Hao Wang. 2024. HARMONIC: Harnessing LLMs for tabular data synthesis and privacy protection. arXiv preprint arXiv:2408.02927

(2024).

- [39] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560 (2022).
- [40] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35

(2022), 24824–24837.

- [41] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni.

2019. Modeling tabular data using conditional gan. Advances in neural information processing systems 32 (2019).

- [42] Shuo Yang, Zheyu Zhang, Bardh Prenkaj, and Gjergji Kasneci. 2025. Doubling Your Data in Minutes: Ultra-fast Tabular Data Generation via LLM-Induced Dependency Graphs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 10348–10369.
- [43] Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander J Ratner, Ranjay Krishna, Jiaming Shen, and Chao Zhang. 2023. Large language model as attributed training data generator: A tale of diversity and bias. Advances in neural information processing systems 36 (2023), 55734–55784.
- [44] Hengrui Zhang, Jiani Zhang, Balasubramaniam Srinivasan, Zhengyuan Shen, Xiao Qin, Christos Faloutsos, Huzefa Rangwala, and George Karypis. 2023. Mixedtype tabular data synthesis with score-based diffusion in latent space. arXiv preprint arXiv:2310.09656 (2023).
- [45] Yu Zhang, Yunyi Zhang, Martin Michalski, Yucheng Jiang, Yu Meng, and Jiawei Han. 2023. Effective seed-guided topic discovery by integrating multiple types of contexts. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining. 429–437.
- [46] Meng Zhou, Jason Li, Rounaq Basu, and Joseph Ferreira. 2022. Creating spatiallydetailed heterogeneous synthetic populations for agent-based microsimulation. Computers, Environment and Urban Systems 91 (2022), 101717.

### A Implementation Details

Macro-statistics. We construct macro-statistics from the available target information. In real-world settings, these statistics can be directly specified by survey tables, census summaries, or other aggregate sources. In our experiments, because real micro-records are needed for baseline comparison, we derive the target macrostatistics from the ground-truth datasets. More generally, when micro-records are available, the same procedure can be used to obtain finer-grained macro-statistics for more sensitive discrepancy evaluation and correction. For continuous variables, we use a hierarchical binning scheme. First, each variable is partitioned into a fixed number of quantile-based main bins, with 6 bins by default, so that each bin contains approximately the same number of real records. This reduces bin sparsity and gives stable coverage across the full distribution, including heavy tails and outliers. Second, to capture local mismatches, we identify the main bin with the largest positive discrepancy between real and synthetic data, i.e., where the real proportion most exceeds the synthetic proportion, and subdivide it into a fixed number of uniformly spaced sub-bins. The sub-bin frequencies are normalized to sum to the frequency of the parent bin. This two-stage design captures both global and local distributional differences. Quantile-based main bins provide stable coarse statistics, while targeted sub-bin refinement focuses on regions where the synthetic data is most underrepresented. Discrete variables, as well as discretized continuous variables, are summarized as frequency tables. For joint macro-statistics, we compute empirical contingency tables over selected variable groups, using the same binning strategy for any continuous variables.

Discrepancy. We measure the discrepancy 𝑄(·, ·) between target and synthetic data as the entry-wise difference between their frequency tables, at both marginal and joint levels. For a target statistic 𝜙 and its synthetic estimate 𝜙ˆ, we define Q(𝜙,𝜙ˆ ) = 𝜙 − 𝜙.ˆ Since our synthesis process can only add records rather than remove them, we focus on positive discrepancies, corresponding to bins or contingency cells where the target proportion exceeds the synthetic proportion. These are the underrepresented regions that can be directly corrected through further generation. Because the macro-statistics are built from interpretable bins, each discrepancy can be attributed to a specific value range for continuous variables, a category for discrete variables, or a combination of variables for joint statistics. This attributability is crucial for the iterative synthesis loop: it tells the LLM where the synthetic data underestimates the target distribution and guides it to generate corrective records in those regions, rather than making undirected global adjustments.

Hyperparameters. We generate 5 proposals per iteration for 100 iterations. At each iteration, 3 joint variable combinations are inferred for grounding. LLM inference is performed using GPT4.1-nano with temperature 0.8, and Qwen2.5-7B-Instruct through the VLLM [22] framework. The generation hyperparameters are: max_new_tokens= 2048,temperature= 0.7,top_k= 20, andtop_p = 0.98. For macro-statistics, when only partial target statistics are available, such as externally provided aggregate statistics, we compute discrepancies using those statistics directly. When full microrecord data is available, we apply the two-level quantile-based binning scheme described above, using 6 main bins and 8 sub-bins for continuous variables. Discrete variables and discretized continuous

bins are handled uniformly when computing marginal and joint contingency tables. All experiments are repeated at least 3 times, and we report the mean results. The uncertainty in our results mainly comes from the stochasticity of LLM generation.

Hardware. All experiments are conducted on a server equipped with an Intel Xeon E5-2698 v4 CPU (40 threads), 252 GB of RAM, and four NVIDIA Tesla V100 GPUs with 32 GB of memory each.

B Running Examples

- B.1 Discrepancy-Guided Generation

This subsection explains how the model identifies the most significant discrepancies in each iteration and samples accordingly, thereby guiding the generation of micro-records to progressively reduce the gap between the synthetic and target distributions.

[Figure 420]

[Figure 421]

[Figure 422]

Target Synthetic Discrepancy

- Iteration 1
- Iteration 2
- Iteration 3

[Figure 423]

A B C

[Figure 424]

[Figure 425]

[Figure 426]

A B C

[Figure 427]

[Figure 428]

[Figure 429]

A B C

[Figure 430]

[Figure 431]

[Figure 432]

A B C

[Figure 433]

[Figure 434]

[Figure 435]

A B C

[Figure 436]

[Figure 437]

[Figure 438]

A B C

[Figure 439]

[Figure 440]

[Figure 441]

A B C

[Figure 442]

[Figure 443]

[Figure 444]

A B C

[Figure 445]

[Figure 446]

[Figure 447]

A B C

[Figure 448]

[Figure 449]

[Figure 450]

Largest

[Figure 451]

Largest

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

Generate more

[Figure 472]

- records with A

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

Generate more

[Figure 477]

- records with B and C

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

50%

[Figure 503]

30%

[Figure 504]

[Figure 505]

20%

[Figure 506]

50%

[Figure 507]

30%

[Figure 508]

[Figure 509]

20%

[Figure 510]

50%

[Figure 511]

30%

[Figure 512]

[Figure 513]

20%

[Figure 514]

100%

[Figure 515]

[Figure 516]

[Figure 517]

50%

[Figure 518]

30%

[Figure 519]

[Figure 520]

20%

[Figure 521]

50%

[Figure 522]

30%

[Figure 523]

[Figure 524]

20%

[Figure 525]

- 50%

[Figure 526]

[Figure 527]

30%

[Figure 528]

[Figure 529]

20%

（100, 60, 40）

（100, 60, 40）

（100, 60, 40） （0, 0, 0）

（25, 0, 0）

（25, 15, 10）

[Figure 530]

…

Figure 9: Discrepancy-guided Iterative Synthesis. The figure shows how discrepancies between target and synthetic distributions guide micro-record generation. At each iteration, the model samples from the largest positive discrepancies, highlighted in red, and generates records for underrepresented categories. Starting from an empty synthetic dataset (0, 0, 0), the model first adds records for category A, then iteratively corrects categories B and C, progressively improving alignment with the target distribution.

- B.2 Example of a Single Iteration

We provide an example to show how each iteration improves the alignment between the synthetic and target distributions.

[Figure 531]

[Figure 532]

[Figure 533]

(a) Variable Dependency Inference

𝐏𝐜𝐨𝐩𝐮𝐥𝐚 Inferred Dependencies 𝓒 External Sources

TransportMode

[Figure 534]

- 3. 'min', 'max', 'category' must use the actual values mentioned in

**Variables**: Categorical variables must be a single valid candidate string (case-sensitive), and numerical variables must be a list of two integer or float numbers (e.g., [3.0, 5.1]).

- 4. If a variable value has a high frequency, that value should be selected in multiple proposals.
- 5. Most generated samples should, as much as possible, prioritize satisfying components that are common across the Guidances, and the “num” for each proposal should be determined based on the frequencies specified in the Guidance.

[Figure 535]

[Figure 536]

###### LLM

[Figure 537]

[Figure 538]

[Figure 539]

###### ## Information

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

𝒄𝟏:[𝐭𝐢𝐦𝐞,𝐦𝐨𝐝𝐞] 𝒄𝟐:[𝐭𝐢𝐦𝐞,𝐚𝐜𝐭𝐢𝐯𝐢𝐭𝐲]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

Variables:

LLM Infer

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

###### Variables: 𝒱= 𝐭𝐢𝐦𝐞,𝐦𝐨𝐝𝐞,𝐚𝐜𝐭𝐢𝐯𝐢𝐭𝐲 time: integer, range 0~23 mode: transport mode, select from [‘transit’, ‘bike’, ‘car’] activity: activity type, select from [‘home’, ‘work’, ‘other’]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

Time

[Figure 575]

[Figure 576]

Updated Macro-Statistics 𝓢𝒕𝒂𝒓𝒈𝒆𝒕 𝓢𝒕𝒂𝒓𝒈𝒆𝒕𝓒 𝚽𝒕𝒂𝒓𝒈𝒆𝒕 𝚽𝒕𝒂𝒓𝒈𝒆𝒕𝓒 ModifyJoints

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

Macro-Statistics:

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

###### demand by time transport mode

activity

[Figure 603]

[Figure 604]

walk car

[Figure 605]

bike

10%

[Figure 606]

time

30%

20%

40%

transit

[Figure 607]

[Figure 608]

Marginal Joint

[Figure 609]

--Infer the variable combinations that demonstrate strong dependencies.

[Figure 610]

[Figure 611]

--Now return a pure, valid, and non-empty JSON in English that can be directly parsed by json.loads() in Python

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

(b) Discrepancy-Guided Iterative Synthesis (Iteration t)

[Figure 618]

In our implementation, each proposal defines a distribution for every variable. For discrete variables, it assigns a valid category, for continuous variables, it specifies a value range, such as [3.0, 5.1], from which values are sampled uniformly or by another simple scheme. The num field specifies how many records to generate from each proposal, allowing the LLM to allocate record counts according to the provided frequencies and guidance. In this way, the LLM can plan both diversity and statistical alignment by balancing the number and distribution of proposals. To make this process interpretable, we use chain-of-thought [40] prompting, asking the LLM to explain the rationale for each proposal with reference to the given statistics and constraints.

This proposal format is one practical instantiation of LLM Proposal Sampling, but the framework is extensible. A proposal distribution can also be defined as executable code, tool calls, or pointers to external generators, such as ControlNet-based diffusion models for images or specialized LLM agents for domain-specific content. Thus, LLM Proposal Sampling can act as a high-level distributional controller that guides external generators or hybrid pipelines toward statistically faithful and scenario-aligned synthetic data across data types and domains.

Figure 10: A simplified running example of discrepancyguided iterative synthesis.

### C.2 LLM as a Copula for Variable Dependency Inference

### C Prompts C.1 LLM Proposal Sampling

#### pcopula: LLM for Variable Dependency Inference

#### pproposal: LLM Proposal Sampling

## Information

**Variables:** ```{data_desc}``` ## Output Format: {{

## Output Format: ```json {{

"n_proposals": n, "proposal1": {{

- "1": ["var1", "var2", ...],
- "2": ["var1", "var2", ...],

"reason": "..." , "proposal": "...", "num": n1

...

}} ### Example Output JSON: {{

}}, "proposal2": {{ ... }}, ...

}} ```

- "1": ["Temperature", "Humidity"],
- "2": ["Humidity", "WeatherCondition"],

}}

## Information:

**Joint Guidance:** `{joint_guide}`

--Your task is to extract **at most {n_joints}** correlated variable groups based on the **Variables** summaries and present them in JSON format.

**Marginal Guidance:** `{marginal_guide}`

**Variables:** `{data_desc}`

--## Rules: Create less than {n_proposals} proposals totaling {n_samples} samples:

- 1. Ensure each group contains two or more variables.
- 2. Format the correlated variable groups according to the **Output Format**.

- 1. Each proposal must follow Joint and Marginal Guidance, do not improvise beyond the provided Guidance.
- 2. The reason for each proposal should explain the realistic meaning of this proposal and how it follows the provided Guidance by referencing frequencies one by one.

