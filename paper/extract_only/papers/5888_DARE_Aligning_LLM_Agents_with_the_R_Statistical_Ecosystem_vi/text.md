## DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval

#### Maojun Sun*1 Yue Wu*1 Yifei Xie*1 Ruijian Han1 Binyan Jiang1 Defeng Sun2 Yancheng Yuan2 Jian Huang12

Project Website DARE-R-Retriever RPKB GitHub

# arXiv:2603.04743v1[cs.IR]5Mar2026

### Abstract

Large Language Model (LLM) agents can automate data-science workflows, but many rigorous statistical methods implemented in R remain underused because LLMs struggle with statistical knowledge and tool retrieval. Existing retrieval-augmented approaches focus on function-level semantics and ignore data distribution, producing suboptimal matches. We propose DARE (Distribution-Aware Retrieval Embedding), a lightweight, plug-and-play retrieval model that incorporates data distribution information into function representations for R package retrieval. Our main contributions are: (i) RPKB, a curated R Package Knowledge Base derived from 8,191 high-quality CRAN packages; (ii) DARE, an embedding model that fuses distributional features with function metadata to improve retrieval relevance; and (iii) RCodingAgent, an R-oriented LLM agent for reliable R code generation and a suite of statistical analysis tasks for systematically evaluating LLM agents in realistic analytical scenarios. Empirically, DARE achieves an NDCG@10 of 93.47%, outperforming state-of-the-art open-source embedding models by up to 17% on package retrieval while using substantially fewer parameters. Integrating DARE into RCodingAgent yields significant gains on downstream analysis tasks. This work helps narrow the gap between LLM automation and the mature R statistical ecosystem.

1Department of Data Science and Artificial Intelligence, The Hong Kong Polytechnic University, Hong Kong SAR, China 2Department of Applied Mathematics, The Hong Kong Polytechnic University, Hong Kong SAR, China. Correspondence to: Ruijian Han <ruijian.han@polyu.edu.hk>, Yancheng Yuan <yancheng.yuan@polyu.edu.hk>, Jian Huang <j.huang@polyu.edu.hk>.

Preprint. March 6, 2026.

###### Function Retrieval Paradigm

Standard Semantic Retrieval

User query with data (q)

Ranked Results

Bi-Encoder

P( f ∣ q)

Function Database

(description focus)

[Figure 1]

genomic"Ihaveadataset ℱ = {d} with high dimensionality …”

[Figure 2]

[Figure 3]

DARE Approach

Data Proﬁle (c) Function Database ℱ = {dd;cdd}

Ranked Results

DARE Bi-Encoder

P( f ∣ q,c)

(Data-conditional)

Figure 1. Comparison of traditional semantic function search methods and Distribution-Aware Retrieval Embedding (DARE) method.

### 1. Introduction

Large Language Model (LLM)-based agents have rapidly emerged as a promising paradigm for automating data science workflows (Chen et al., 2025; Sun et al., 2025b). By integrating natural language reasoning, code generation, and tool execution, these agents are increasingly capable of translating user intent expressed in natural language into executable analytical pipelines. Recent systems have demonstrated strong performance in tasks such as data preprocessing, exploratory analysis, feature engineering, and model training (Sun et al., 2026), significantly reducing the manual effort traditionally required in end-to-end data science processes (Zhang et al., 2023; Sun et al., 2025a; Hong et al., 2025; Zhang et al., 2025).

Despite these advancements, existing LLM-based data science agents predominantly operate within Python-centric ecosystems (Orlanski et al., 2023; Zhao & Fard, 2025), and have limited capability in utilizing R, a language and environment specifically designed for statistical computing and analysis (R Core Team, 2025). This limitation results in the underutilization of decades of accumulated statistical knowledge. The statistical community has long developed a rich repertoire of theoretically rigorous methodologies for data analysis, many of which are implemented and actively maintained within the R ecosystem (Ihaka & Gentleman, 1996; R Core Team, 2025). Central to

this ecosystem is the Comprehensive R Archive Network (CRAN, https://cran.r-project.org/), which hosts thousands of peer-reviewed, highly specialized packages that encode domain expertise and methodological best practices.

However, current LLMs are trained on corpus that are heavily skewed toward general-purpose programming languages, especially Python. As a result, LLM-based agents often exhibit systematic performance gaps when interacting with R-based programming (Orlanski et al., 2023; Zhao & Fard, 2025). In practice, this imbalance leads to two major issues: (i) agents frequently default to Python-based implementations even when R provides more statistically appropriate or computationally efficient solutions, and (ii) when attempting to generate R code, agents often hallucinate function names, misuse parameter configurations, or fail to identify the correct statistical package.

A natural strategy to mitigate these limitations is RAG (Lewis et al., 2020), where agents retrieve external documentation or function descriptions from R package repositories. Existing embedding models primarily rely on semantic similarity between user queries and textual descriptions of functions. However, the applicability of statistical methods often depends not only on semantic intent but also on data distribution characteristics, such as sparsity structure, dimensionality regimes, distributional assumptions, and modality-specific constraints. General-purpose embedding models, trained on broad web corpora, frequently fail to capture these subtle yet crucial distributional conditions, resulting in retrieval errors that propagate to downstream code generation and execution failures.

To address this limitation, we first construct RPKB (R Package Knowledge Base), a curated dataset derived from 8,191 high-quality R packages on CRAN (https:// cran.r-project.org/), with functions spanning diverse statistical domains. RPKB provides structured function metadata, documentation, and usage information, serving as a valuable resource for statistical tool retrieval and LLM tool learning. Example entries and the data schema in RPKB are given in Section C of the Supplementary Materials.

Building upon RPKB, we propose DARE (DistributionAware Retrieval Embedding), a retrieval model that explicitly incorporates data distribution into function representations for retrieving R packages. By training a contrastive dual-encoder architecture conditioned on data profiles, DARE learns to distinguish between functions that are semantically similar but statistically incompatible under different data contexts. Importantly, DARE is designed as a lightweight and plug-and-play retrieval module that can be seamlessly integrated into LLM-based agent systems like LAMBDA (Sun et al., 2025a). Experimental results

demonstrate that DARE achieves state-of-the-art retrieval performance on RPKB, achieving an NDCG@10 of 93.47% (Normalized Discounted Cumulative Gain, a high-quality information retrieval ranking metric), outperforming prior state-of-the-art embedding models by up to 17% while using only 23M parameters. Figure 1 illustrates the difference between traditional semantic retrieval and DARE.

To investigate the practical impact of integrating DARE into LLM agents for real-world statistical analysis workflows, we further design RCodingAgent, an end-to-end Roriented LLM agent that supports automated statistical analysis through iterative reasoning, statistical tool retrieval, code generation, and execution-based validation. To systematically evaluate agent performance, we construct a diverse suite of 16 carefully designed R-based statistical analysis tasks spanning representative statistical domains, including hypothesis testing, goodness-of-fit analysis, survival analysis, and mixed-effects modeling. Experimental results show that integrating DARE into RCodingAgent significantly improves downstream statistical analysis performance by up to 56.25% across both frontier and lightweight LLMs, demonstrating its effectiveness in enabling reliable automated statistical workflows.

Our key contributions are summarized as follows:

- 1. We construct RPKB, a curated repository derived from 8,191 high-quality R packages and functions spanning diverse statistical domains. RPKB provides structured statistical tool knowledge and serves as a valuable resource for LLM-based tool learning and retrieval.
- 2. We propose DARE, a lightweight and plug-and-play distribution-aware retrieval embedding model that incorporates data distribution constraints into function representations. DARE achieves state-of-the-art retrieval performance on RPKB, outperforming prior embedding models with substantially larger parameter counts, while maintaining high computational efficiency.
- 3. We design RCodingAgent, an end-to-end R-oriented LLM agent that automates statistical analysis workflows, and construct a diverse suite of 16 R-based statistical analysis tasks for evaluation. When integrated into RCodingAgent, DARE significantly improves downstream statistical analysis performance.

### 2. Related Works

LLM Agents for Data Science. Recent works have empowered LLMs to act as data science agents by equipping them with planing, code execution environments, selfcorrection, and report generation (Sun et al., 2025b; Hong

###### DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval

User Query

(q ∼ Q)

###### Natural Language Query

[Figure 4]

(e.g., "I have a genomic dataset with high dimensionality...

f (̂ q,cq) f (̂ q,cq) = arg maxf ((q,cq), f)

s((q,cq), f)

[Figure 5]

Objective:

|Concat| |
|---|---|
| | |

f∈ℱ

###### Query Encoder

###### Query Embedding

Concat

###### InfoNCE Loss

(ε)

((eqq = ε((qq,cqq))))

Optional

exp(cos(eqi,efi)/τ) N ∑

- exp(cos(eqi,efi)/τ)

j=

- exp(cos(eqi,efj)/τ)

(all-MiniLM-L6-v2)

System Level Data Proﬁle

ℒi = − log

i =

(Normalized)

exp(cos(eqi,efj)/τ)

(e.g., "data_modality": "genomic sequence", "dimensionality": "high" ...

(cq ∼ C)

j=1

|Positive Pairs (Matched)| | | | | |
|---|---|---|---|---|---|
| | |Ne|gativ|e Pa|irs|
| | |(U|nma|tche|d)|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

|Similarity<br><br>s = cos(eeq,eef)| |
|---|---|
| | |

Shared Weights

Function Database ((f((dd;cdd)) ∼ ℱ))

###### Package Documentation

(e.g., R Package:Quality_Control_ Metrics_for_Gene_Signatures.

(d ∼ D)

[Figure 6]

R Package Description: Makes all the plots for the quality control ...

|Concat| |
|---|---|
| | |

###### Function Encoder

###### Function Embedding

Concat

(ε)

((ef = ε((qq,cqq))))

(all-MiniLM-L6-v2)

Data Proﬁle

(Normalized)

(e.g., "data_modality": "genomic sequence", "dimensionality": "high" ...

(cd ∼ C)

In-Batch Similarity Matrix (NN × NN)

Figure 2. The overall framework of DARE training process.

et al., 2025). While these agents show promise, they predominantly rely on Python. Efforts to incorporate R are limited and often suffer from lower code generation quality due to the scarcity of R in training data relative to Python (Li et al., 2023).

Dense Retrieval and RAG. Dense retrieval, which maps queries and documents into a shared latent space via biencoders, has evolved significantly from pioneering architectures like DPR (Karpukhin et al., 2020) and Contriever (Izacard et al., 2021). Recent advancements have shifted towards scaling up parameter counts and leveraging large-scale instruction tuning. Advanced models such as Snowflake Arctic (Merrick, 2024), Gte-large-en-v1.5 (Zhang et al., 2024), and Mxbai (Lee et al., 2024) have achieved dominance on general benchmarks by utilizing massive pre-training corpora and complex multi-stage fine-tuning strategies. However, these models are unable to fully utilize their abilities in statistical computing scenarios. First, their general-purpose design primarily emphasizes semantic similarity, making it difficult to capture statistical compatibility factors beyond textual semantics, such as data distribution characteristics and modeling assumptions. Second, the substantial increase in model size introduces significant computational overhead during retrieval. This limitation becomes particularly pronounced when searching over large-scale statistical tool repositories, such as extensive R package databases, where low-latency retrieval is critical for interactive agent workflows. To address these limitations, we introduce the DARE model, which achieves strong retrieval accuracy and effi-

ciency by explicitly incorporating data distribution information into the representation learning process, enabling distribution-aware statistical tool retrieval.

Tool Learning. Tool learning aims to teach LLMs to use external APIs (Wang et al., 2022; Shen et al., 2023). Most existing approaches rely on in-context learning (ICL) with API descriptions, which tightly couples tool selection with generation. Our work differs by introducing a dedicated, data-aware retrieval module that decouples tool selection from generation, enabling scalable access to thousands of statistical functions. When augmented with DARE, LLM agents are able to effectively utilize a broader range of statistical packages and achieve substantial performance improvements in downstream data analysis tasks.

### 3. Methodology

In this section, we present details of the database construction, proposed DARE framework (Figure 2), RCodingAgent and the evaluation methods.

#### 3.1. Database Construction

We constructed a specialized knowledge base by curating R packages from the Comprehensive R Archive Network (CRAN)1. Our data pipeline consists of three stages:

1https://cran.r-project.org/web/packages/ available_packages_by_name.html

✅

- 1. Extraction: We crawled raw documentation (HTML and PDF) from CRAN, extracting package-level metadata (e.g., Package Name, Description, Version, Author) and granular function-level details, specifically the Description, Usage, Arguments, and Value sections.
- 2. Function-Level Chunking and Filtering: We processed the corpus at the function level. To ensure the density of statistical knowledge, we implemented a rigorous filtering pipeline. We excluded: (i) generic utility functions (e.g., I/O operations, basic string manipulation); (ii) functions with vague or incomplete descriptions; (iii) auxiliary methods lacking specific algorithmic content. Our selection focused strictly on core statistical primitives and computational algorithms with clear analytical objectives.
- 3. Data Profile Generation: To achieve modeling on the data context, we employed Grok-4.1-fast to synthesize metadata from the unstructured documentation. The LLM was prompted to infer key statistical data attributes, such as data modality, distribution assumptions, and dimensionality.
- 4. Storage: The curated data were indexed using ChromaDB. The final repository comprises 8,191 highquality R functions spanning diverse statistical domains, serving as the retrieval corpora F.

More details about the database construction and selection, some examples and generation prompts are provided in the Supplementary Materials.

#### 3.2. Problem Formulation

Let F = {fk}|F|k=1 denote a repository of candidate R functions. Each function f ∈ F is formally defined as a tuple

f = (d,cd), where d denotes the natural-language docu-

mentation (e.g., package and function description), and cd denotes the structured data profile encoding the function’s inherent constraints (e.g., data modality, distribution, and dimensionality).

In the inference stage, the user provides a natural-language request q together with a dataset c. Based on this information, the system derives a query-side data profile cq, which can be automatically inferred from dataset characteristics and signals extracted from the user query (e.g., “data modality”: “tabular”, “feature type”: “numerical”, “distribution assumption”: “non-gaussian”, “dimensionality”: “high”).

Our goal is to retrieve the most appropriate R function fˆ∈ F that satisfies both the semantic intent of q and the datalevel context of cq:

fˆ(q,cq) = arg max f∈F

s (q,cq),f .

#### 3.3. DARE Modeling

We utilize a bi-encoder architecture with shared weights initialized from sentence-transformers/all-MiniLM-L6v2 (Reimers & Gurevych, 2019). We define the shared encoder network as ε(·), which maps input texts into a shared m-dimensional vector space. Let [·; ·] denote the textual concatenation operation.

The agent’s request is encoded into a query embedding eq = ε([q ; cq]) ∈ Rm. Correspondingly, each candidate function f is encoded into a function embedding ef = ε([d; cd]) ∈ Rm.

We compute the relevance score using cosine similarity between these representations:

e⊤q ef ∥eq∥2 ∥ef∥2

s eq,ef = cos(eq,ef) =

.

This factorization allows for efficient retrieval via Maximum Inner Product Search (MIPS) over the precomputed function embeddings {ef}f∈F.

We construct a supervised retrieval dataset D = {(qi,cq,i,fi)}Mi=1, where fi denotes the ground-truth function (comprising documentation di and profile cd,i) for the i-th query-context pair. We finetune the encoder using the InfoNCE (van den Oord et al., 2019) objective with in-batch negatives. Given a mini-batch B of size N, for each anchor query i, we treat the paired function fi as the positive sample and all other functions fj (j ̸= i) in the batch as negatives. The loss function for the i-th sample is defined as:

exp cos(eq

,ef

)/τ

i

i

Li = −log

,

N

exp cos(eq

,ef

)/τ

i

j

j=1

where τ is a learnable temperature parameter. The total batch loss is computed as L = N1 Ni=1 Li. This objective maximizes the similarity between the query embedding eq

i

and its corresponding function embedding ef

, while simultaneously minimizing the similarity to other candidates ef

i

j

that do not align with the query’s semantic and distributional constraints. Figure 2 illustrates the framework of DARE.

#### 3.4. RCodingAgent: DARE-Augmented Agentic Data Analysis

To support agent-based statistical analysis using the R programming language, we design and implement RCodingAgent, a prototype LLM-based agent that performs iterative reasoning, tool retrieval, R code generation, and executionbased validation to complete end-to-end

analytical tasks. As illustrated in Figure 3, RCodingAgent incorporates the DARE module to enhance downstream

!

###### DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval

###### Query

I have a high-dimensional genomic dataset named hidra_ex_1_2000.csv in my environment. I need to identify driver elements by estimating regulatory scores based on the counts provided in the data. Please set the random seed to 123 at the start. I need to ﬁlter for fragment lengths between 150 and 600 bp and use a DNA count ﬁlter of 5. For my evaluation, please print the ﬁrst value of the estimated scores (est_a) for the very ﬁrst region identiﬁed.

Retrieval LLM-based Data Science Agent

[Figure 7]

- Coding Step 1
- Coding Step 2
- Coding Step 3

###### DARE

# Load required library library(sharpr2)

Recall Top 3 Items

# Set random seed set.seed(123)

[Figure 8]

[Figure 9]

[Figure 10]

Package: Estimating_Regulatory_Scores_and_Identifying_ATACSTARR_Data :: sharpr2

# Load the data data_path <- "/home/xxx/data/id4406/hidra_ex_1_2000.csv" hidra_data <- read.csv(data_path) # Check the structure of the data cat("Data dimensions:", dim(hidra_data), "\n") cat("Column names:", colnames(hidra_data), "\n") head(hidra_data)

[Figure 11]

[Figure 12]

Package: Estimating_Regulatory_Scores_and_Identifying_ATACSTARR_Data :: sharpr2-package

[Figure 13]

[Figure 14]

Package: Estimating_Regulatory_Scores_and_Identifying_ATACSTARR_Data :: call_sig_reg

stdout

Data dimensions: 2000 4 Column names: start end PLASMID RNA

Observations

[Figure 15]

start end PLASMID RNA

[Figure 16]

- 1 13069 13432 10.2 7.8
- 2 57063 57343 3.6 3.2
- 3 57349 57757 6.4 6.4
- 4 127511 127762 5.4 4.0
- 5 127517 127753 9.2 7.4
- 6 236790 237138 4.8 3.4

[Figure 17]

{

"function_details_json": { "name": "sharpr2", "title": "sharpr2", "description": "For a HiDRA dataset on a given chromosome, this function calls tiled regions (the regions covered by at least one fragment), and calculates regulatory scores for each tiled region. The regulatory scores are based on standardized log(RNA/PLASMID).",

[Figure 18]

# Run sharpr2 function with specified parameters re <- sharpr2(hidra_data,

"usage": "sharpr2(data, l_min = 150, l_max = 600, f_rna = 10, f_dna = 0,\\n s_a = 300, verbose = FALSE, auto = TRUE, sig = TRUE, len = FALSE, \\n alpha = 0.05, win = 5, mse = FALSE, max_t = 1)",

l_min = 150, l_max = 600, f_dna = 5, f_rna = 0, sig = FALSE)

"arguments": {

"data": "A data.frame containing an ATAC-STARR dataset for one chromosome. The data.frame must contain four columns: \'start\', \'end\', \'PLASMID\', \'RNA\'. \'PLASMID\' and \'RNA\' are the values for DNA and RNA, which should be non-negative real numbers (average value over multiple replicates) or integers (counts).",

"l_min": "The fragments with a length smaller than l_min will not be processed. The default is 150.", "l_max": "The fragments with a length larger than l_max will not be processed. The default is 600.", "f_rna": "The fragments with an RNA count smaller than f_rna will not be processed. The default is

# Check the structure of the result cat("Result structure:\n") str(re)

10.",

"f_dna": "The fragments with an DNA count smaller than f_rna will not be processed. The default is 0.",

"s_a": "A variance hyperparameter in the prior for the latent regulatory scores. The default is 1000.", "verbose": "An indicator of whether to show processing information. The default is FALSE.", "auto": "An indicator of whether to automatically estimate the ridge coeﬃcient \\\\lambda from the data for each tiled region using a data-driven way described in the reference. The default is TRUE. If auto is TRUE, s_a is ignored and a ridge coeﬃcient is estimated for each tiled region separately. If auto is FALSE, a global user-deﬁned ridge coeﬃcient (1/s_a) is used.",

# Print the first value of est_a (estimated scores) for the first region cat("\nFirst value of est_a (estimated scores) for the very first region:\n") cat(re$score$est_a[1], "\n")

Result structure: List of 7

stdout

Observations

"sig": "An indicator of whether to identify signiﬁcant motif regions for the estimated scores. Only valid if auto=TRUE. The default is TRUE.",

"len": "An indicator of whether to model log(RNA/PLASMID) of each fragment as the average or the sum of the latent regulatory scores. The default is FALSE, which is the sum.",

$ score :List of 957

..$ :List of 4

"alpha": "A regional FWER to call high resolution driver elements (the signiﬁcant regulatory region). The default is 0.05.",

.. ..$ est_a : num [1:364] -0.219 -0.219 -0.219 -0.219 -0.219 ...

.. ..$ mse : logi NA

"win": "A window size for removing sporadic identiﬁed signiﬁcant regions. If a signiﬁcant consecutive region is small than win , it will be treated as false signals. The default is 5.",

.. ..$ var_nb: logi NA

"mse": "An indicator of whether mean square errors are included in the output results. The default is FALSE.",

.. ..$ lambda: logi NA

..$ :List of 4

"max_t": "A value between 0 and 1, indicating the proportion of non-zero eigenvectors used to

.. ..$ est_a : num [1:409] 0.465 0.465 0.465 0.465 0.465 ...

calculate \\\\lambda when auto=TRUE. The default is 1." }, "value": "score: the regulatory scores for each tiled region. This list contains four components: est_a (the regulatory scores at each locus), sd_e (the sqare root of the mean square error), var_nb (the variance of the esitmate at each locus), \\\\lambda (the ridge coeﬃcient).",

[Figure 19]

"examples": [ "data(hidra_ex)", "re <- sharpr2(hidra_ex[1:2000,], l_min = 150, l_max = 600, f_dna = 5, f_rna = 0, sig=FALSE)"

# Correctly access the first value of est_a for the very first region # re$score is a list where each element represents a tiled region with its own est_a vector first_est_a_value <- re$score[[1]]$est_a[1]

], "format": [], "details": "The default value of s_a is set to be 300, which is equivalent to a ridge coeﬃcient of 0.0033.

This default ridge coeﬃcient value is selected by the median of the estimated \\\\lambda from the ﬁrst library."

cat("First value of est_a (estimated scores) for the very first region:\n") cat(first_est_a_value, "\n")

} }

Final Observations

First value of est_a (estimated scores) for the very ﬁrst region: -0.2185472

stdout

Figure 3. An example of RCodingAgent for realistic statistical analysis.

statistical analysis workflows. Specifically, given a naturallanguage query q, the agent first invokes DARE to retrieve candidate R packages and functions that satisfy both analytical intent and data compatibility constraints. The retrieved functions are returned with structured metadata, including argument specifications and usage examples, which are injected into the LLM context to guide tool invocation and code generation via in-context learning, enabling accurate and executable statistical analysis.

3.5. Evaluating LLM Agents for Statistical Programming in R

To systematically evaluate LLM agents in realistic, execution-grounded statistical programming workflows, we

design an evaluation framework based on a suite of 16 representative R-based statistical analysis tasks. The evaluation focuses on assessing whether agents can correctly generate executable R code, and produce statistically valid analytical outputs under realistic data contexts. Upper panel of Figure 4 illustrates the task construction pipeline, bottom panel shows an overview of selected domains and packages.

Specifically, we collect high-quality example R scripts from the RPKB repository covering diverse statistical and scientific domains, including hypothesis testing, goodnessof-fit analysis, survival analysis, graphical model estimation, mixed-effects modeling, and statistical preprocessing. These scripts are executed to produce verified ground-truth outputs. Based on the underlying datasets and analytical

✅

###### DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval

[Figure 20]

[Figure 21]

[Figure 22]

ibrary(sharpr2) set.seed(123) data(hidra_ex) re <- sharpr2(hidra_ex[1:2000,], l_min = 150, l_max = 600, f_dna = 5, f_rna = 0, sig=FALSE) re reg_id <- 1 gt_value <- re$score[[reg_id]]$est_a[1] gt_value

ibrary(sharpr2) set.seed(123) data(hidra_ex) re <- sharpr2(hidra_ex[1:2000,], l_min = 150, l_max = 600, f_dna = 5, f_rna = 0, sig=FALSE) re reg_id <- 1 gt_value <- re$score[[reg_id]]$est_a[1] gt_value

ibrary(sharpr2) set.seed(123) data(hidra_ex) re <- sharpr2(hidra_ex[1:2000,], l_min = 150, l_max = 600, f_dna = 5, f_rna = 0, sig=FALSE) re reg_id <- 1 gt_value <- re$score[[reg_id]]$est_a[1] gt_value

[Figure 23]

###### Database R Packages

Collecting Executing

[Figure 24]

[Figure 25]

Dataset Provided

[Figure 26]

[Figure 27]

Answer Consistency Answer Conciseness Numerical Priority

[Figure 28]

[Figure 29]

stdout

[Figure 30]

16 Tasks

gt_value=-0.2185

[Figure 31]

Benchmark Synthesis Query

Filtering

OPTICS K-Xi Density-Based Clustering:fortify ica

Estimate a Log-Concave Probability Density from Iid Observa-tions:logconTwoSample

National Information Platforms for Nutrition Anthropometric Data Toolkit:ageHeaping

National Information Platforms for Nutrition Anthropometric Data Toolkit:ageChildren

Independent component analysis

Age heaping detection

Two-sample hypothesis test

Goodness-of-ﬁt

Auxiliary Functions to Estimate Centers of Biodiver-sity:getNarrowEndemics

Conditional Graphical LASSO for Gaussian Graphical Models with Censored and Missing Values:cglasso

Diversity Dynamics using Fossil Sampling Data:afﬁnity

Quality Scores for the Russell 3000:market proﬁtability

Narrow endemic identiﬁcation

Conditional graphical model estimation

Environmental afﬁnity assignment

Proﬁtability z-scoring

Computation of Ancestry Scores with Mixed Families and Unrelated Individuals:perfectwhiten

Various Linear Mixed Model Analyses:lmm.jack

Diversity Dynamics using Fossil Sampling Data:survivors

Set of Assumptions for Factor and Principal Component Analysis:communalities optimal solution

Covariance-preserving whitening

Factor analysis preprocessing

Survivorship analysis

Linear mixed model

Clinical Trial Simulation:cut data by event

Estimating Regulatory Scores and Identifying ATAC-STARR Data:sharpr2

Electronic Health Record (EHR) Data Processing and Analysis Tool:processLastDose

Simulating Data to Study Performance of Clustering Algorithms:overlap

Survival data truncation

Gaussian mixture overlap computation

Temporal data standardization

Regulatory score estimation

Figure 4. Upper panel: Pipeline for constructing R-based statistical evaluation tasks. Lower panel: Overview of selected domains and R packages covered in the benchmark.

objectives, we prompt LLMs to generate natural user-style analytical queries. Each generated query is manually verified and paired with its corresponding ground-truth output to ensure correctness and reproducibility. Detailed task specifications and query generation prompts are provided in the Supplementary Materials.

### 4. Experiments

#### 4.1. Experimental Setup

We first generated synthetic user queries using MiMo-V2flash to simulate realistic user interactions. For each function, we generated 30 queries and designed five prompt templates ranging from simple to complex to reflect users with different levels of data analysis expertise. In total, this process produced 245,730 queries for the dataset. We split the dataset into training and test sets, using 85% of the data for training and the remaining 15% for testing. In our experiment, each query considers only one ground-truth item. The model is initialized with all-MiniLM-L6-v2 and trained using the AdamW (Loshchilov & Hutter, 2019) optimizer for 100 epochs with a batch size of 256 and an initial learning rate of 1 × 10−4. Training is conducted on an NVIDIA A100 (80GB) GPU. Evaluation is performed after each epoch, and the checkpoint achieving the best

performance across multiple evaluation metrics is selected and released as DARE-R-Retriever (DARE). We evaluate the model using standard metrics commonly adopted in the information retrieval and search literature.

Baselines. We compared DARE against a wide range of open-sourced state-of-the-art embedding models, including BAAI/bge-m3 (Chen et al., 2024), intfloat/e5-large-v2 (Moreira et al., 2025), Snowflake/arctic-embed-l (Merrick, 2024), jinaai/jina-embeddings-v2-base-en (G¨unther et al., 2024), mixedbread-ai/mxbai-embed-large-v1 (Lee et al., 2024), WhereIsAI/UAE-Large-V1 (Li & Li, 2024), AlibabaNLP/gte-large-en-v1.5 (Zhang et al., 2024), and sentencetransformers/all-mpnet-base-v2 (Song et al., 2020), as well as our base model sentence-transformers/all-MiniLM-L6-v2 (Reimers & Gurevych, 2019).

To investigate the practical impact of integrating DARE into LLM agents for real-world statistical analysis workflows, we evaluate RCodingAgent across 16 statistical analysis tasks under two settings: without (w/o) and with DARE. Experiments are conducted across six representative LLMs, including deepseek-v3.2, gpt-5.2, mimo-v2-flash, grok-4.1fast, claude-haiku-4.5, and minimax-m2.1, to assess the robustness of DARE across diverse model capabilities. All agents are allowed at most 20 interaction steps for each task

attempt.

#### 4.2. Evaluation Metrics

Let Q denote the set of evaluation queries with size |Q|. For each query we obtain a ranked list of retrieved functions (f(1),...,f(K)). Let rel(f) ∈ {0,1} indicate whether a retrieved function is matched.

Recall@k. For each query q ∈ Q, we obtain a ranked list of retrieved functions f(1)(q),...,f((Kq)) . We report Recall@k, defined as the fraction of queries whose groundtruth function appears in the top-k retrieved results:

1 |Q| q∈Q

I{∃j ≤ k : rel(f((jq))) = 1}.

Recall@k =

In particular, Recall@1 corresponds to top-1 accuracy. NDCG@k. To capture rank sensitivity, we also report NDCG@k. Define the discounted cumulative gain

k

2rel(f(j)) − 1 log2(j + 1)

.

DCG@k =

j=1

Let IDCG@k be the maximum achievable DCG@k under an ideal ranking; then

DCG@k IDCG@k

.

NDCG@k =

With binary relevance, IDCG@k = 1 whenever at least one relevant item exists, so NDCG@k reduces to a logarithmically discounted score of how early the correct tool is retrieved.

Mean Reciprocal Rank (MRR@k). Evaluates how far down the ranked list the first relevant function appears. This metric is particularly critical for tool-augmented LLM agents, as placing the correct statistical tool higher in the prompt reduces token consumption and mitigates hallucination risks:

1 |Q| q∈Q

1 rank(q)

MRR@k =

,

where rank(q) is the position of the first ground-truth function for query q. If no relevant function is retrieved within

the top k results, the reciprocal rank rank1

is set to 0.

(q)

Efficiency Metrics. To evaluate the deployment feasibility of DARE in real-time agentic workflows, we report two key efficiency indicators:

• Average Latency (L): Measures the mean wall-clock time required to encode a single query-context pair.

Given a set of evaluation queries Q, and the total inference time Tseq under sequential execution (batch size = 1):

Tseq |Q|

× 1000 (ms/query).

L =

• Throughput (QPS): Measures the system’s processing capacity under parallel execution to saturate the hardware. Given |Q| queries processed in optimal batches (e.g., batch size = 128) with total inference time Tbatch:

QPS = |Q|

Tbatch

(queries/s).

Success Rate (SR). To evaluate the end-to-end performance of LLM agents on the statistical analysis tasks, we define a binary success indicator for each task. Let A(q) denote the execution output produced by the agent for a given query q, and G(q) represent the corresponding ground-truth answer. The correctness of the agent’s result is determined by an indicator function:

I(q) =

1, if A(q) = G(q) 0, otherwise

.

The overall Success Rate is then defined as the proportion of tasks where the agent’s output matches the ground truth:

1 |Q| q∈Q

I(q).

SR =

### 5. Experimental Results

In this section, we present a comprehensive evaluation of DARE. First, we assess its retrieval performance against open-sourced embedding models on the RPKB test set. Second, we investigate the efficiency of DARE. Lastly, we examine whether integrating DARE into RCodingAgent improves performance across 16 downstream statistical analysis tasks.

#### 5.1. Performance on Retrieval

We compared DARE against a diverse set of strong embedding baselines, ranging from compact sentence encoders (e.g., all-MiniLM-L6-v2) to large-scale state-of-the-art models (e.g., BGE-M3 and Snowflake/arctic-embed-l). The results are summarized in Table 1.

As shown in Table 1, DARE consistently establishes a new state of the art across all reported retrieval metrics. In particular, DARE achieves an NDCG@10 of 93.47%, substantially outperforming the strongest baseline, Snowflake/arcticembed-l, by a relative margin of 17.8%. This indicates that DARE not only retrieves the correct function but also ranks

- Table 1. Performance comparison with open-sourced SoTA embedding models on RPKB test set. Despite having only 23M parameters, DARE significantly outperforms all other models (up to 568M parameters) across all retrieval metrics.

###### Model Params NDCG@10 MRR@10 Recall@10 Recall@1

Snowflake/arctic-embed-l (Merrick, 2024) 335 M 79.32% 75.10% 92.35% 65.49% Intfloat/e5-large-v2 (Moreira et al., 2025) 335 M 75.13% 70.86% 88.38% 61.52% Jinaai/jina-embeddings-v2-base-en (G¨unther et al., 2024) 137 M 74.29% 69.65% 88.73% 59.69% BAAI/bge-m3 (Chen et al., 2024) 568 M 73.08% 68.43% 87.58% 58.47% Mixedbread-ai/mxbai-embed-large-v1 (Lee et al., 2024) 335 M 70.68% 65.65% 86.39% 55.08% WhereIsAI/UAE-Large-V1 (Li & Li, 2024) 335 M 70.66% 65.56% 86.58% 54.79% Alibaba-NLP/gte-large-en-v1.5 (Zhang et al., 2024) 434 M 66.39% 61.22% 82.57% 50.40% Sentence-transformers/all-mpnet-base-v2 (Song et al., 2020) 110 M 66.06% 60.57% 83.30% 49.37%

All-MiniLM-L6-v2 (Base Model) (Reimers & Gurevych, 2019) 23 M 61.27% 55.53% 79.36% 44.12% DARE (Ours) 23 M 93.47% 91.76% 98.63% 87.39%

it significantly higher than competing models. Moreover, DARE attains a remarkably high Recall@1 of 87.39%, corresponding to a 33.4% relative improvement over the best baseline, highlighting its strong ability to place the correct function at the top position.

Beyond top-ranked accuracy, DARE also demonstrates superior overall retrieval coverage. It achieves a Recall@10 of 98.63%, indicating that nearly all relevant functions are successfully retrieved within the top-10 candidates. In addition, the substantially higher MRR@10 of 91.76% reflects consistent ranking quality across queries, rather than gains driven by a small subset of easy cases. Together, these results suggest that DARE delivers both strong early precision and robust ranking stability.

A key challenge in function retrieval lies in distinguishing between functions that are statistically similar yet distributionally distinct (e.g., differentiating glm from glm.nb). General-purpose embedding models, which are primarily optimized for semantic similarity, often struggle in such finegrained scenarios. In contrast, DARE explicitly conditions representation learning on distributional characteristics, enabling more precise discrimination among closely related statistical functions. This capability is directly reflected in the large gains observed in rank-sensitive metrics such as NDCG@10 and MRR@10.

Notably, these performance improvements are achieved with remarkable efficiency. DARE adopts the architecture of all-MiniLM-L6-v2 and contains only 23M parameters, making it approximately 15 to 25 times smaller than leading competitors such as BAAI/bge-m3 (568M parameters) and Alibaba-NLP/gte-large (435M parameters). Despite its compact size, DARE substantially outperforms all large-scale baselines. Compared to the base model, DARE improves NDCG@10 from 61.27% to 93.47%, representing a dramatic absolute gain of over 32 points. This result confirms that the proposed distribution-conditional contrastive learning framework effectively injects critical domain-specific

###### System Performance: Latency vs QPS

DARE (Ours)

8000

7000

QPS(QueriesPerSecond)

6000

5000

all-mpnet-base-v2

jina-v2-base-en

4000

3000

e5-large-v2

uae-large-v1

arctic-embed-l

bge-m3

mxbai-large-v1

2000

gte-large-en-v1.5

4 6 8 10 12

Latency (ms)

Figure 5. Results of QPS and Latency.

knowledge that is largely absent from the pre-training of general-purpose language models.

#### 5.2. Inference Efficiency Analysis

Beyond retrieval accuracy, inference latency and throughput are critical factors for deploying LLM agents in real-time, iterative data science workflows. Figure 5 illustrates the system performance trade-off between Latency (ms) and Throughput QPS across evaluated models.

As visualized in Figure 5, DARE (represented by the green circle) dominates the efficiency landscape, occupying the ideal top-left corner. Our model achieves a remarkable throughput of 8,512 QPS with an ultra-low latency of 3.7ms. This performance is attributed to the lightweight architecture of our base model and the simple and efficient design.

In sharp contrast, high-performing general-purpose models such as BAAI/bge-m3, Snowflake/arctic-embed-l, and Alibaba-NLP/gte-large-en-v1.5 cluster in the bottom-right region. These models typically exhibit latencies exceeding

- Table 2. Comparison of End-to-End Success Rates for Various LLM Agents on statistical analysis tasks, with and without the DARE Module.

#### Model RCodingAgent (w/o DARE) RCodingAgent with DARE

Claude-haiku-4.5 6.25% 56.25% Deepseek-v3.2 18.75% 56.25% Gpt-5.2 25.00% 62.50% Grok-4.1-fast 18.75% 75.00% Mimo-v2-flash 12.50% 62.50% Minimax-m2.1 12.50% 68.75%

10ms and throughputs below 3,000 QPS due to their massive parameter counts (335M–568M). Consequently, DARE offers a speed advantage of approximately three to four times over these baselines. This efficiency is pivotal for agentic systems, where an agent may need to retrieve from thousands of candidate functions multiple times within a single reasoning chain. DARE ensures that adding rigorous statistical retrieval introduces negligible overhead to the overall inference pipeline.

#### 5.3. Impact on Agentic Data Analysis

To validate the practical utility of DARE, we conducted experiments for RCodingAgent on 16 statistical analysis tasks. Our findings are largely consistent with prior studies (Orlanski et al., 2023; Zhao & Fard, 2025), revealing that current LLMs exhibit limited capability in effectively utilizing R for statistical analysis tasks. As shown in Table 2, most models demonstrate very low end-to-end success rates on these tasks when relying solely on their inherent knowledge, highlighting a substantial limitation of reliable statistical tool usage.

Notably, Claude-haiku-4.5, Mimo-v2-flash, and Minimaxm2.1 initially showed poor performance on the benchmark. By contrast, augmenting these models with DARE leads to substantial performance improvements, increasing success rates to 50.00%, 50.00% and 56.25%, respectively. In addition, grok-4.1-fast exhibits the most pronounced improvement, with success rates rising from 18.75% to 75.00%, representing an absolute gain of 56.25%.

Even for advanced frontier models, DARE consistently provides meaningful gains. For example, gpt-5.2 improves from 25.00% to 62.50%. These results indicate that DARE effectively bridges the statistical tool utilization gap, enabling both lightweight and frontier models to reliably perform specialized data science tasks. Figure 3 presents a realistic example of statistical analysis. More cases can be found in the Supplementary Materials.

By supplying precise, distribution-aware retrieval signals, DARE reduces execution failures and enhances the robustness of LLM agents in statistical analysis workflows.

### 6. Future Directions

Despite the strong empirical performance of DARE, several promising research directions remain.

Enhancing native R proficiency of LLMs. Current LLMs exhibit limited native capability in statistical computing with R, partly due to the limited presence of high-quality R-oriented corpora in large-scale pretraining datasets. Future work may explore constructing large-scale R-centric knowledge corpora, such as curated tutorials, package documentation, and executable analytical workflows, to improve the internal statistical reasoning and programming proficiency of LLMs and complement retrieval-based augmentation.

Advancing structured tool learning and utilization. Improving tool learning strategies remains an important direction. In our current framework, retrieved statistical packages are provided to the LLM through demonstration-style contextual augmentation using structured JSON descriptions. While effective, this approach may not fully capture the hierarchical and compositional relationships among statistical tools. Future work could investigate more structured and adaptive strategies, such as dynamic tool abstraction, function-level reasoning graphs, or memory-based tool selection mechanisms, to better support complex statistical tool usage.

Expanding and sustaining the statistical knowledge base. Although our current repository contains over 8,191 highquality R functions, the broader R ecosystem includes a substantially larger collection of domain-specific packages and specialized tools. We plan to open-source our curated knowledge base to encourage community-driven expansion, with the goal of establishing a comprehensive and continuously evolving statistical knowledge infrastructure for LLM-based agents.

Integrate RCodingAgent into a mixture-of-experts agent system. A promising direction is to integrate RCodingAgent into a mixture-of-experts agent system. In this setting, RCodingAgent would serve as a specialized expert for R-based statistical analysis, coordinated with other agents

responsible for complementary tasks. Such a modular design could improve scalability and flexibility in complex end-to-end analytical workflows.

### References

- Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. M3-embedding: Multi-linguality, multifunctionality, multi-granularity text embeddings through self-knowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 2318– 2335, August 2024. doi: 10.18653/v1/2024.findings-acl.

137. URL https://aclanthology.org/2024. findings-acl.137/.

- Chen, K., Wang, P., Yu, Y., Zhan, X., and Wang, H. Large language model-based data science agent: A survey. arXiv:2508.02744, 2025.

G¨unther, M., Ong, J., Mohr, I., Abdessalem, A., Abel, T., Akram, M. K., Guzman, S., Mastrapas, G., Sturua, S., Wang, B., Werk, M., Wang, N., and Xiao, H. Jina embeddings 2: 8192-token general-purpose text embeddings for long documents. arXiv:2310.19923, 2024. URL https://arxiv.org/abs/2310.19923.

Hong, S., Lin, Y., Liu, B., Liu, B., Wu, B., Zhang, C., Li, D., Chen, J., Zhang, J., Wang, J., et al. Data interpreter: An llm agent for data science. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 19796– 19821, 2025.

Ihaka, R. and Gentleman, R. R: a language for data analysis and graphics. Journal of computational and graphical statistics, 5(3):299–314, 1996.

Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., and Grave, E. Unsupervised dense information retrieval with contrastive learning. arXiv:2112.09118, 2021.

Karpukhin, V., Oguz, B., Min, S., Lewis, P. S., Wu, L., Edunov, S., Chen, D., and Yih, W.-t. Dense passage retrieval for open-domain question answering. In EMNLP (1), pp. 6769–6781, 2020.

Lee, S., Shakir, A., Koenig, D., and Lipp, J. Open source strikes bread - new fluffy embeddings model. Availble at: https://www.mixedbread.ai/blog/mxbai-embed-largev1, 2024.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨uttler, H., Lewis, M., Yih, W.-t., Rockt¨aschel, T., et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 9459–9474, 2020.

Li, R. et al. Starcoder: may the source be with you! arXiv:2305.06161, 2023.

Li, X. and Li, J. AoE: Angle-optimized embeddings for semantic textual similarity. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1825– 1839, August 2024. doi: 10.18653/v1/2024.acl-long. 101. URL https://aclanthology.org/2024.

acl-long.101/.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https: //openreview.net/forum?id=Bkg6RiCqY7.

Merrick, L. Embedding and clustering your data can improve contrastive pretraining. arXiv:2407.18887, 2024. URL https://arxiv.org/abs/2407.18887.

Moreira, G. d. S. P., Osmulski, R., Xu, M., Ak, R., Schifferer, B., and Oldridge, E. Improving text embedding models with positive-aware hard-negative mining. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pp. 2169– 2178, 2025.

Orlanski, G., Xiao, K., Garcia, X., Hui, J., Howland, J., Malmaud, J., Austin, J., Singh, R., and Catasta, M. Measuring the impact of programming language distribution. In Proceedings of the 40th International Conference on Machine Learning, pp. 26619–26645. PMLR, 2023.

R Core Team. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria, 2025. URL https://www. R-project.org/.

Reimers, N. and Gurevych, I. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv:1908.10084, 2019.

Shen, Y., Song, K., Tan, X., Li, D., Lu, W., and Zhuang, Y. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Proceedings of the 37th Advances in Neural Information Processing Systems, pp. 38154– 38180, 2023.

Song, K., Tan, X., Qin, T., Lu, J., and Liu, T.-Y. MPNet: masked and permuted pre-training for language understanding. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 16857–16867, 2020. ISBN 9781713829546.

Sun, M., Han, R., Jiang, B., Qi, H., Sun, D., Yuan, Y., and Huang, J. LAMBDA: A large model based data agent. Journal of the American Statistical Association, pp. 1–13, 2025a.

Sun, M., Han, R., Jiang, B., Qi, H., Sun, D., Yuan, Y., and Huang, J. A survey on large language model-based agents for statistics and data science. The American Statistician, pp. 1–14, 2025b.

Sun, M., Xie, Y., Wu, Y., Han, R., Jiang, B., Sun, D., Yuan, Y., and Huang, J. Dsaeval: Evaluating data science agents on a wide range of real-world data science problems. arXiv:2601.13591, 2026.

van den Oord, A., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv:1807.03748, 2019. URL https://arxiv. org/abs/1807.03748.

Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., and Wei, F. Text embeddings by weaklysupervised contrastive pre-training. arXiv:2212.03533, 2022.

Zhang, S., Fan, J., Fan, M., Li, G., and Du, X. Deepanalyze: Agentic large language models for autonomous data science. arXiv:2510.16872, 2025.

- Zhang, W., Shen, Y., Lu, W., and Zhuang, Y. Data-copilot: Bridging billions of data and humans with autonomous workflow. arXiv:2306.07209, 2023.
- Zhang, X., Zhang, Y., Long, D., Xie, W., Dai, Z., Tang, J., Lin, H., Yang, B., Xie, P., Huang, F., Zhang, M., Li, W., and Zhang, M. mGTE: Generalized long-context text representation and reranking models for multilingual text retrieval. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 1393–1412. Association for Computational Linguistics, November 2024. doi: 10.18653/v1/2024.emnlp-industry.

103. URL https://aclanthology.org/2024. emnlp-industry.103/.

Zhao, Z. and Fard, F. Do current language models support code intelligence for r programming language? ACM Transactions on Software Engineering and Methodology, 34(8):1–39, 2025.

### A. Example Prompt of Data Profile Generation

This section presents an example prompt used to convert R documentation into structured Data Profiles. It instructs an LLM to exclude generic utilities and extract key statistical metadata such as data modality, distribution assumptions, and dimensionality.

Example Prompt of Data Profile Generation

You are an expert Data Scientist and R Developer. Your task is to filter and annotate R functions for a RetrievalAugmented Generation (DARE) system specialized in data science.

You will receive a JSON describing an R function (including package description, function description, usage, arguments).

Your goal is to:

- 1. FILTER: Determine if this function is a specific algorithm, statistical method, data transformation, or machine learning model.

- - REJECT basic utilities (e.g., file I/O, simple string manipulation, getters/setters, pure plotting without analytics).
- - KEEP only if it has clear data input requirements and analytical purpose.

- 2. EXTRACT: If the function is relevant, extract structured Data Information. Specifically, what kind of data does this algorithm expect? (e.g., sparse matrices, categorical factors, normally distributed vectors).

Output must be a valid JSON object adhering to this schema: { ”is relevant”: boolean, ”data profile”: { ”data modality”: ”string enum [tabular, time-series, text, image, graph, genomic/sequence, other]”, ”feature type”: ”string enum [numerical, categorical, mixed, binary, text-token, any]”, ”distribution assumption”: ”string (e.g., ’normal’, ’sparse’, ’poisson’, ’any’, ’unknown’)”, ”dimensionality”: ”string enum [low, high, any]”, ”missing data handling”: ”string (e.g., ’handles na’, ’must be complete’, ’unknown’)”, ”specific constraints”: [”list”, ”of”, ”strings”] }, ”task type”: ”string (e.g., ’classification’, ’clustering’, ’normalization’, ’gene signature scoring’)”, ”reasoning”: ”short explanation” }

### B. Example Prompt of DARE Training Query Generation

Here is an example prompt to generate queries from selected R functions. The generated queries are used to train the DARE model.

### Example Prompt of DARE Training Query Generation

You are an expert Data Scientist creating a training dataset for a Semantic Search Engine. Your goal is to generate **30 diverse user search queries** that would lead a user to the target R function described below.

**CRITICAL RULES:**

- 1. **NEVER mention the function name or the package name** in the queries. The user doesn’t know the tool exists yet; they are describing their problem.
- 2. **HEAVILY RELY on the Data Information**: The queries MUST describe the data structure, distribution, or constraints provided in the context (e.g., ”I have a sparse matrix...”, ”My data is high-dimensional...”, ”Handling zero-inflated count data...”).
- 3. **Variety**:

- - 12 queries should be long and detailed (e.g., ”I am working with a large tabular dataset that has high sparsity, and I need to...”).
- - 5 queries should be problem-focused (e.g., ”How to handle missing values in categorical time-series?”).
- - 5 queries should be ”Help me” style (e.g., ”I need a way to plot gene expressions where the data is...”).
- - 3 queries should be Instruction first then followed by data information (e.g. ”I need to visualize the structure of protein... Data Information: ’type’: ’tabular’, ’distribution’: ’sparse’..... ”)
- - 5 queries should be short/keyword-based but include data types (e.g., ”dim reduction for sparse arrays”).

**Input Format:** Function Description + Data Profile JSON.

**Output Format:** A valid JSON object with a single key ”queries” containing a list of 30 strings. Example: ”queries”: [”query 1”, ”query 2”, ...]

- C. Example of RPKB Here are some examples of the RPKB, which contains the function description and the data profile.

### Example of RPKB

{

"id": 22, "fc_id": "Reproducible_Data_Screening_Checks_and_Report_of _Possible_Errors::makeDataReport", "ground_truth_doc": "Data Constraints: {\"data_modality\": \"tabular\", \"

feature_type\": \"mixed\", \"distribution_assumption\": \"any\", \"dimensionality \": \"any\", \"missing_data_handling\": \"handles_na\", \"specific_constraints\": [\"data.frame/tibble /matrix\", \"handles labelled/haven_labelled as factors\", \"smartNum treats low unique nums as factors\"]}. Task: data_quality_assessment. \nR Package: Reproducible_Data_Screening_Checks_and_Report_of_Possible Errors. R Package Description: Data screening is an important first step of any statistical \n analysis. ’dataReporter’ auto generates a customizable data report with a thorough\n summary of the checks ......

"data_profile": { "data_modality": "tabular", "feature_type": "mixed", "distribution_assumption": "any", "dimensionality": "any", "missing_data_handling": "handles_na", "specific_constraints": [

"data.frame/tibble/matrix", "handles labelled/haven_labelled as factors", "smartNum treats low unique nums as factors"

]

}, "usage_guidance": "makeCodebook(data, vol = \"\", reportTitle = NULL, file = NULL,

...)", "example_code": "library(dataReporter) data(testData)

data(toyData) check(toyData) DF <- data.frame(x = 1:15) makeDataReport(DF) data(testData) makeDataReport(testData) # Overwrite any existing files generated by makeDataReport makeDataReport(testData, replace=TRUE) # Change output format to Word/docx: makeDataReport(testData, replace=TRUE, output = "word") # Only include problematic variables in the output document makeDataReport(testData, replace=TRUE, onlyProblematic=TRUE) # Add user defined check-function to the checks performed on

character variables:

# Here we add functionality to search for the string wally ( ignoring case)

wheresWally <- function(v, ...) { res <- grepl("wally", v, ignore.case=TRUE) problem <- any(res) message <- "Wally was found in these data" checkResult(list(problem = problem, message = message, problemValues = v[res])) } wheresWally <- checkFunction(wheresWally, description = "Search for the string ’wally’ ignoring case", classes = c("character") ) # Add the newly defined function to the list of checks used for

characters.

makeDataReport(testData, checks = setChecks(character = defaultCharacterChecks(add = "

wheresWally")),

replace=TRUE) #Handle non-supported variable classes using treatXasY: treat

raw as character and

#treat complex as numeric. We also add a list variable, but as lists are not

#handled through treatXasY, this variable will be caught in the

preChecks and skipped: toyData$rawVar <- as.raw(c(1:14, 1)) toyData$compVar <- c(1:14, 1) + 2i toyData$listVar <- as.list(c(1:14, 1)) makeDataReport(toyData, replace = TRUE, treatXasY = list(raw = "character", complex = "numeric"))",

"task_type": "data_quality_assessment", "domain": "Specialized Statistics & ML"

}

### Example of RPKB

{

"id": 960, "fc_id": "OPTICS_K-Xi_Density-Based_Clustering::fortify_ica", "ground_truth_doc": "Data Constraints: {\"data_modality\": \"tabular\", \"

feature_type\": \"numerical\", \"distribution_assumption\": \"non-gaussian\", \" dimensionality\": \"high\", \"missing_data_handling\": \"must_be_complete\", \"

specific_constraints\": [\"numeric matrix\", \"suitable for blind source separation\"]}. Task: independent_component_analysis. \nR Package: OPTICS_KXi_Density-Based_Clustering. R Package Description: Density-based clustering methods are well adapted to the clustering of high-dimensional data and enable the discovery of core groups of various shapes despite large amounts of noise. This package provides a novel density-based cluster extraction method, OPTICS kXi, and a framework to compare k-Xi models using ......

"feature_type": "numerical", "distribution_assumption": "non-gaussian", "dimensionality": "high", "missing_data_handling": "must_be_complete", "specific_constraints": [ "numeric matrix", "suitable for blind source separation"

]

}, "usage_guidance": "fortify_ica(m_data, ..., sup_vars = NULL)", "example_code": "library(opticskxi)

library(fastICA) library(plyr) df_ica <- fortify_ica(iris[-5], n.comp = 2) df_ica",

"task_type": "independent_component_analysis", "domain": "Specialized Statistics & ML"

}

### Example of RPKB

{

"id": 2460, "fc_id": "Estimate_a_Log-Concave_Probability_Density_from_I id_Observations::logconTwoSample", "ground_truth_doc": "Data Constraints: {\"data_modality\": \"tabular\", \" feature_type\": \"numerical\", \"distribution_assumption\": \"log-concave\", \" dimensionality\": \"low\", \"missing_data_handling\": \"must_be_complete\", \" specific_constraints\": [\"univariate iid observations\", \"two samples for comparison\"]}. Task: two-sample hypothesis test. ...... "data_profile": {

"data_modality": "tabular", "feature_type": "numerical", "distribution_assumption": "log-concave", "dimensionality": "low", "missing_data_handling": "must_be_complete", "specific_constraints": [

"univariate iid observations", "two samples for comparison"

]

}, "usage_guidance": "logconTwoSample(x, y, which = c("MLE", "smooth"), M = 999,

n.grid = 500, display = TRUE, seed0 = 1977) ",

"example_code": "library(logcondens)

- n1 <- 30

- n2 <- 25

- x <- rgamma(n1, 2, 1)

- y <- rgamma(n2, 2, 1) + 1 twosample <- logconTwoSample(x, y, which = c("MLE", "smooth")[1],

M = 999)

", "task_type": "two-sample hypothesis test", "domain": "Specialized Statistics & ML"

}

- D. Training and Evaluation Details To further support the experimental analysis, we provide some details in training process.

[Figure 32]

Figure 6. Training loss curves for the DARE model, demonstrating convergence behavior over training epochs.

- E. Example System Prompt For RCodingAgent Here is an example system prompt for the RCodingAgent.

### Example System Prompt For RCodingAgent

You are an advanced Data Science Agent specializing in R. Your goal is to solve complex data analysis tasks by writing robust, efficient, and ”tidy” R code. Core Guidelines:

- 1. File I/O Safety:

- - Input: specific data files are located at {data path}. ONLY load data from this directory.

- - Output: save all results (CSVs, models, images) to {working path}.

- 2. Package Management: Install any packages if you need.
- 3. Task Execution: Solving the problem by multiple step, but finally you only need to give a concise answer with numerical results.
- 4. System Integrity: IMPORTANTLY: You cannot perform any actions that may harm the system, such as deleting directories, important software, or software packages.

[Figure 33]

[Figure 34]

Figure 7. Evaluation loss curves for DARE in training process.

### Example System Prompt For RCodingAgent with DARE

You are an advanced Data Science Agent specializing in R equipped with a R package retrieval system. Your Ability: You will be provided with ”Retrieved R Documentation”. Use these documents to solve the task.

- - You MUST carefully analyze the ”Retrieved R Documentation”.
- - If the documents provide a specific function, prioritize using it over general knowledge.
- - Mimic the usage style, syntax, and pay attention to ”example” or ”usage” sections. Core Guidelines:

- 1. File I/O Safety: Input from {data path}, output to {working path}.

- 2. Task Execution: Solve the problem by multiple steps, and give a concise answer with numerical results.
- 3. System Integrity: Do not perform actions that harm the system.

- F. Construction of Statistical Analysis Evaluation Tasks We use the following prompt to generate evaluation queries from real R code and data.

Construction of Statistical Analysis Evaluation Tasks

Role: You are an expert in R programming and advanced statistics assisting in a research evaluation. Your task is to generate natural language queries based on R function metadata to test the retrieval and execution capabilities of an AI Agent.

Core Objective:

The queries must never mention the R package or function names. Instead, they must deeply embed the Distribution Assumptions, Data Modalities, and Specific Constraints found in the metadata. The goal is to force the model to retrieve the correct tool based on ”statistical logic” rather than ”name memorization.”

Task Instructions (4 Query Styles):

- 1. Beginner Style (Vague): Focus on the high-level scientific goal without technical jargon.
- 2. Practitioner Style (Rough): Use standard industry terminology and describe the operational context.
- 3. Rigorous Researcher Style (Detailed): Explicitly include statistical profiles (e.g., Poisson, Gaussian, Time-series). This is the core test for DARE’s performance.
- 4. Technical Logic Style (Parametric): Emphasize specific parameter configurations and the exact output required for evaluation. Hard Constraints for Output:

- - Mandatory Seed: Every query must require the user to start the script with set.seed(123).
- - Mandatory Verification: Every query must instruct the agent to print a specific value (the Ground Truth) for evaluation (e.g., ”print the p-value” or ”output the first value of est a”).

- - Pre-loaded Data: Assume the dataset is already available. The Agent must read the data directly (e.g., from a CSV or a specific object name).
- - No Leakage: Strictly forbid mentioning the fc id or the R Package name in the query.

Example Format:

”I have a dataset [Data Modality] that follows a [Distribution Assumption]. My goal is to [Task Type]. Please write an R script starting with set.seed(123) to process the data... Finally, print [Ground Truth Metric] so I can verify the results.”

- G. Example of Statistical Analysis Task

Here is an example task, illustrating the structure of the task type, data profile, ground truth verification, and the generated evaluation queries in four distinct styles.

### Example (ID: 960)

Task Type: Independent Component Analysis Data Profile: Tabular, Numerical, Non-Gaussian Distribution, High-dimensional.

#### Ground Truth Verification:

df ica$IC1[1] = 0.5040647 df ica$IC2[1] = -1.372758

#### Generated Evaluation Queries:

- 1. Beginner Style: I have a file named iris data.csv containing multiple numeric measurements. I suspect there are several independent underlying signals mixed together in this data, and I want to separate them into 2 distinct components. Can you write an R script that extracts these signals and adds them as new columns to the original data? Please set the seed to 123 for consistency and show me the first row of the two newly created columns.

- 2. Practitioner Style: I need to perform a feature extraction on iris data.csv to find independent components. Please use R to reduce the numeric features to 2 components. I’d like the results to be integrated back into the dataset (fortified). Make sure to set.seed(123) at the start and print the first row of the components so I can verify the values.

- 3. Rigorous Researcher Style: I am working with iris data.csv, which is a high-dimensional numeric matrix. Given that the data follows a non-Gaussian distribution, I need to perform Independent Component Analysis (ICA) for blind source separation. Please extract 2 components and ensure they are fortified into the data frame. Set the random seed to 123 and output the first row of ICA1 and ICA2 to confirm the computation.

- 4. Technical Logic Style: Using the numeric data in iris data.csv, I need to run a density-based ICA extraction to get 2 components. It’s crucial that the method supports a ’fortify’ mechanism to merge the components back into the observation table automatically. Use seed 123 for reproducibility and display the top row of the resulting ICA columns.

- H. Details of 16 Statistical Analysis Evaluation Tasks This section details the 16 representative statistical analysis tasks used to evaluate the RCodingAgent.

Table 3. Detailed profile of representative evaluation tasks.

Task Domain Detailed Information Independent component analysis Package/Function: OPTICS K-Xi Density-Based Clustering::fortify ica

Description: Density-based clustering utilities and visualization for ICA results Link: https://cran.r-project.org/web/packages/opticskxi/ vignettes/opticskxi.pdf

Two-sample hypothesis test Package/Function: Estimate a Log-Concave Probability Density from Iid Observations::logconTwoSample Description: Two-sample hypothesis testing based on log-concave density estimation. Link: https://cran.r-project.org/web/packages/logcondens/logcondens.pdf

Goodness-of-fit Package/Function: National Information Platforms for Nutrition Anthropometric Data Toolkit::ageChildren Description: Anthropometric age processing and child age standardization utilities. Link: https://cran.r-project.org/web/packages/nipnTK/nipnTK.pdf

Age heaping detection Package/Function: National Information Platforms for Nutrition Anthropometric Data Toolkit::ageHeaping Description: Detection and correction of age heaping phenomena in demographic data Link: https://cran.r-project.org/web/packages/nipnTK/nipnTK.pdf

Narrow endemic identification Package/Function: Auxiliary Functions to Estimate Centers of Biodiversity::getNarrowEndemics Description: Identification of narrow endemic species for biodiversity analysis. Link: https://cran.r-project.org/web/packages/sperich/sperich.pdf

Environmental affinity assignment Package/Function: Diversity Dynamics using Fossil Sampling Data::affinity Description: Environmental affinity assignment for fossil occurrence records Link: https://cran.r-project.org/web/packages/divDyn/divDyn.pdf

Survivorship analysis Package/Function: Diversity Dynamics using Fossil Sampling Data::survivors Description: Survivorship analysis across temporal fossil sampling intervals Link: https://cran.r-project.org/web/packages/divDyn/divDyn.pdf

Conditional graphical model estimation

Package/Function: Conditional Graphical LASSO (cglasso)

Description: Estimation of conditional graphical models under censoring and missingness Link: https://cran.r-project.org/web/packages/cglasso/cglasso.pdf

Clinical EHR preprocessing Package/Function: Electronic Health Record Data Processing and Analysis Tool::processLastDose Description: Temporal standardization and preprocessing of clinical dosage records. Link: https://cran.r-project.org/web/packages/EHR/EHR.pdf

Clinical trial simulation Package/Function: Clinical Trial Simulation::cut data by event Description: Survival data truncation based on event occurrence time. Link: https://cran.r-project.org/web/packages/simtrial/simtrial.pdf

Regulatory score estimation Package/Function: Estimating Regulatory Scores and Identifying ATAC-STARR Data::sharpr2 Description: Regulatory activity score estimation from high-throughput genomic assays Link: https://cran.r-project.org/web/packages/sharpr2/sharpr2.pdf

Task Domain Detailed Information Ancestry score computation Package/Function: Computation of Ancestry Scores with Mixed Families and Unrelated

Individuals::perfectwhiten Description: Covariance-preserving whitening for ancestry score computation. Link: https://cran.r-project.org/web/packages/PCFAM/PCFAM.pdf

Linear mixed model Package/Function: Various Linear Mixed Model Analyses::lmm.jack Description: Linear mixed-effects modeling with jackknife-based inference Link: https://cran.r-project.org/web/packages/minque/minque.pdf

Gaussian mixture overlap Package/Function: Simulating Data to Study Performance of Clustering Algorithms::overlap Description: Computation of Gaussian mixture overlap for clustering evaluation. Link: https://cran.r-project.org/web/packages/MixSim/MixSim.pdf

Factor Analysis Preprocessing Package/Function: Set of Assumptions for Factor and Principal Component Analysis::communalities Description: Preprocessing and communality estimation for factor analysis. Link: https://cran.r-project.org/web/packages/FactorAssumptions/FactorAssumptions.pdf

Financial quality scoring Package/Function: Quality Scores for the Russell 3000::market profitability Description: Profitability z-scoring and financial quality metric computation. Link: https://cran.r-project.org/web/packages/qmj/qmj.pdf

- I. Example Case: Performance Analysis of DARE on Data Analysis Task Here is an example for RCodingAgent with and without the DARE module for statistical analysis tasks.

### Query and Ground Truth

User Query: I have a high-dimensional genomic dataset named hidra ex 1 2000.csv in my environment. I need to identify driver elements by estimating regulatory scores based on the counts provided in the data. Please set the random seed to 123 at the start. I need to filter for fragment lengths between 150 and 600 bp and use a DNA count filter of 5. For my evaluation, please print the first value of the estimated scores (est a) for the very first region identified.

#### Ground Truth: -0.2185

- I.1. DARE By explicitly conditioning the query on distribution-aware metadata, DARE retrieves the correct statistical primitive.

### DARE Retrived Top 3

{

- 1. [ID: Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data::sharpr2] (Dist: 0.7704) Package: Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data :: sharpr2 {’function_name’: ’sharpr2’, ’package_version’: ’1.1.1.0’, ’package_name’: ’

Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data’, ’task_type’: ’regulatory_score_estimation’, ’ data_profile_json’: ’{"data_modality": "genomic/sequence", "feature_type": "numerical", "distribution_assumption": " poisson", "dimensionality": "high", "missing_data_handling": "must_be_complete", "specific_constraints": ["ATAC-STARR dataset", "columns: start, end, PLASMID, RNA", "non-negative reals or integers", "one chromosome", "fragment lengths 150-600 bp", "RNA/DNA count filters"]}’, ’function_details_json’: ’{"name": "sharpr2", "title": "sharpr2", "description": "For a HiDRA dataset on a given chromosome, this function calls tiled regions (the regions covered by at least one fragment), and calculates regulatory scores for each tiled region. The regulatory scores are based on standardized log(RNA /PLASMID).", "usage": "sharpr2(data, l_min = 150, l_max = 600, f_rna = 10, f_dna = 0,\\n s_a = 300, verbose = FALSE, auto

= TRUE, sig = TRUE, len = FALSE, \\n alpha = 0.05, win = 5, mse = FALSE, max_t = 1)", "arguments": {"data": "A data.

frame containing an ATAC-STARR dataset for one chromosome. The data.frame must contain four columns: \’start\’, \’end\’, \’PLASMID\’, \’RNA\’. \’PLASMID\’ and \’RNA\’ are the values for DNA and RNA, which should be non-negative real numbers ( average value over multiple replicates) or integers (counts).", "l_min": "The fragments with a length smaller than l_min will not be processed. The default is 150.", "l_max": "The fragments with a length larger than l_max will not be processed.

The default is 600.", "f_rna": "The fragments with an RNA count smaller than f_rna will not be processed. The default is 10.", "f_dna": "The fragments with an DNA count smaller than f_rna will not be processed. The default is 0.", "s_a": "A variance hyperparameter in the prior for the latent regulatory scores. The default is 1000.", "verbose": "An indicator of whether to show processing information. The default is FALSE.", "auto": "An indicator of whether to automatically estimate the ridge coefficient \\\\lambda from the data for each tiled region using a data-driven way described in the reference.

The default is TRUE. If auto is TRUE, s_a is ignored and a ridge coefficient is estimated for each tiled region separately.

If auto is FALSE, a global user-defined ridge coefficient (1/s_a) is used.", "sig": "An indicator of whether to identify significant motif regions for the estimated scores. Only valid if auto=TRUE. The default is TRUE.", "len": "An indicator of whether to model log(RNA/PLASMID) of each fragment as the average or the sum of the latent regulatory scores. The default is FALSE, which is the sum.", "alpha": "A regional FWER to call high resolution driver elements (the significant regulatory region). The default is 0.05.", "win": "A window size for removing sporadic identified significant regions. If a significant consecutive region is small than win , it will be treated as false signals. The default is 5.", "mse": "An indicator of whether mean square errors are included in the output results. The default is FALSE.", "max_t": "A value between 0 and 1, indicating the proportion of non-zero eigenvectors used to calculate \\\\lambda when auto=TRUE. The default is 1."}, "value": "score: the regulatory scores for each tiled region. This list contains four components: est_a ( the regulatory scores at each locus), sd_e (the sqare root of the mean square error), var_nb (the variance of the esitmate

at each locus), \\\\lambda (the ridge coefficient).", "examples": ["data(hidra_ex)", "re <- sharpr2(hidra_ex[1:2000,], l_min = 150, l_max = 600, f_dna = 5, f_rna = 0, sig=FALSE)"], "format": [], "details": "The default value of s_a is set to

be 300, which is equivalent to a ridge coefficient of 0.0033. This default ridge coefficient value is selected by the median of the estimated \\\\lambda from the first library."}’, ’function_title’: ’sharpr2’}

- 2. [ID: Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data::sharpr2-package] (Dist: 1.0781) Package: Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data :: sharpr2-package

{’function_name’: ’sharpr2-package’, ’task_type’: ’regulatory score estimation and driver element identification’, ’

data_profile_json’: ’{"data_modality": "genomic/sequence", "feature_type": "mixed", "distribution_assumption": "sparse", " dimensionality": "high", "missing_data_handling": "unknown", "specific_constraints": ["ATAC-STARR sequencing data", "highdefinition reporter assay library datasets"]}’, ’function_details_json’: ’{"name": "sharpr2-package", "title": "Estimating

regularoty scores and identifying high resolution driver elements for ATAC-STARR data", "description": "The package develops an algorithm for identifying high-resolution driver elements for datasets from an ATAC-STARR library.", "usage": "", "arguments": {}, "value": "", "examples": ["data(hidra_ex)", "re <- sharpr2(hidra_ex[1:2000,], l_min = 150, l_max = 600, f_dna = 5, f_rna = 0, sig=FALSE)"], "format": [], "details": ""}’, ’package_version’: ’1.1.1.0’, ’function_title’: ’ Estimating regularoty scores and identifying high resolution driver elements for ATAC-STARR data’, ’package_name’: ’ Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data’}

3. [ID: Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data::call_sig_reg] (Dist: 1.1114) Package: Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data :: call_sig_reg

{’function_title’: ’call_sig_reg’, ’data_profile_json’: ’{"data_modality": "genomic/sequence", "feature_type": "numerical", " distribution_assumption": "normal", "dimensionality": "high", "missing_data_handling": "unknown", "specific_constraints": ["requires sharpr2 output object", "tiled genomic regions", "z-score thresholds"]}’, ’package_name’: ’ Estimating_Regulatory_Scores_and_Identifying_ATAC-STARR_Data’, ’function_name’: ’call_sig_reg’, ’task_type’: ’ significant_regulatory_region_calling’, ’package_version’: ’1.1.1.0’, ’function_details_json’: ’{"name": "call_sig_reg", " title": "call_sig_reg", "description": "Given an object returned from the sharpr2 function, this function calls significant regions that contain driver elements for a specific tiled region based on a user-defined threshold.", "usage": "call_sig_reg(res, nr, threshold = 3.5, win = 10)", "arguments": {"res": "An object obtained from the sharpr2 function.", "nr": "An integer indicating the number of tiled region in res for which driver elements will be called.", "threshold": "

The cutoff to identify driver elements in the tiled region. The positions with a z-score larger than the threshold will be

called. The default is 3.5.", "win": "A window size for removing sporadic significant regions. If a significant consecutive region is small than win, it will be treated as false signals. The default is 10."}, "value": "sig_reg: identified regions containing driver elements.", "examples": ["data(hidra_ex)", "re <- sharpr2(hidra_ex[1:2000,], l_min =

150, l_max = 600, f_dna = 5, f_rna = 0, sig=TRUE)", "call_sig_reg(re,850, threshold=2.5)"], "format": [], "details": ""}’}

}

The top retrieved document is the correct one, which contains the correct function that directly addresses the user’s query. The agent can use the detailed usage and example code from this document to generate the correct R script for the task.

#### I.2. Correct Implementation by the RCodingAgent with DARE

With the correct R package documentation, the agent generates the following executable R script, successfully solve the problem.

### Code Log: Task ID 4406

###### [Agent] step 0: call LLM [Agent] tool call → jupyter code interpreter

# Set random seed and load specialized library set.seed(123) library(sharpr2)

# Load the genomic dataset data_path <- "/home/xxx/id4406/hidra_ex_1_2000.csv" hidra_data <- read.csv(data_path)

# Display basic info cat("Dataset dimensions:", dim(hidra_data), "\n") head(hidra_data)

[Tool Output] Dataset dimensions: 2000 4

start end PLASMID RNA 13069 13432 10.2 7.8 ... ... ... ...

- [Agent] step 1: call LLM [Agent] tool call → jupyter code interpreter

# Run sharpr2 with distribution-specific parameters result <- sharpr2( data = hidra_data, l_min = 150, l_max = 600, f_rna = 0, f_dna = 5, sig = FALSE, verbose = TRUE )

cat("\nResult names:", names(result), "\n") cat("Length of est_a:", length(result$score$est_a), "\n")

[Tool Output] Total reads after filtering: 1326. Call tiled regions: 957. Calculating regulatory scores ... Done! Result names: score region n reg n read sig reg motif cutoff Length of est a: 0

- [Agent] step 2: call LLM [Agent] tool call → jupyter code interpreter

# Examine nested structure for correct value extraction cat("$score structure:\n") str(result$score[[1]])

[Tool Output] $score structure: List of 4

.. $ est a : num [1:364] -0.219 -0.219 -0.219 -0.219 -0.219 ...

.. $ mse : logi NA

- [Agent] step 3: call LLM [Agent] tool call → jupyter code interpreter

# Extract the first value of est_a for the first region first_value <- result$score[[1]]$est_a[1] print(first_value)

[Tool Output] [1] -0.2185472

- [Agent] step 4: call LLM [Agent] final answer: Identified 957 tiled regions using sharpr2. The first estimated regulatory score (est a) for the first region is -0.2185472.

#### I.3. Failed Implementation by the RCodingAgent without DARE

This example illustrates a failure case where the agent operates without DARE. Lacking specialized tool documentation, the agent relies on its inherent knowledge and fails to identify the sharpr2 package. Instead, it attempts to solve the problem

using a generic log2-ratio heuristic.

### Code Log: Task ID 4406 (Failed Case - Without DARE)

- [Agent] step 0: call LLM [Agent] tool call → jupyter code interpreter

# Load required libraries and set seed set.seed(123)

# Load the data and explore its structure data <- read.csv("/home/xxx/id4406/hidra_ex_1_2000.csv") dim(data) head(data)

[Tool Output] [1] 2000 4

start end PLASMID RNA 13069 13432 10.2 7.8 ... ... ... ...

- [Agent] step 1: call LLM [Agent] tool call → jupyter code interpreter

# Calculate fragment length and apply filters manually data$fragment_length <- data$end - data$start

# Filter: lengths between 150-600 and PLASMID >= 5 data_filtered <- data[data$fragment_length >= 150 & data$fragment_length <= 600 & data$PLASMID >= 5, ] dim(data_filtered)

[Tool Output] [1] 1428 5

- [Agent] step 2: call LLM [Agent] tool call → jupyter code interpreter

# Estimate scores using a generic log2 ratio heuristic # Lacking specialized tool documentation for sharpr2 data_filtered$est_a <- log2((data_filtered$RNA + 1) / (data_filtered$PLASMID + 1))

# Print the first value for the first region cat("First estimated score (est_a):", data_filtered$est_a[1], "\n")

[Tool Output] First estimated score (est a): -0.3479233

- [Agent] step 3: call LLM [Agent] final answer: I have estimated the regulatory scores by applying filters and calculating a log2 ratio. The first value of the estimated score (est a) for the first region is

-0.3479233.

