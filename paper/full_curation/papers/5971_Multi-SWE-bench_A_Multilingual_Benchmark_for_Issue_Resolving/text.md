[Figure 1]

# arXiv:2504.02605v1[cs.SE]3Apr2025

## Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving

ByteDance Seed Leaderboard Benchmark RL Community GitHub Repo

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

### Abstract

The task of issue resolving is to modify a codebase to generate a patch that addresses a given issue. However, existing benchmarks, such as SWE-bench, focus almost exclusively on Python, making them insufficient for evaluating Large Language Models (LLMs) across diverse software ecosystems. To address this, we introduce a multilingual issue-resolving benchmark, called Multi-SWE-bench, covering Java, TypeScript, JavaScript, Go, Rust, C, and C++. It includes a total of 1,632 high-quality instances, which were carefully annotated from 2,456 candidates by 68 expert annotators, ensuring that the benchmark can provide an accurate and reliable evaluation. Based on Multi-SWE-bench, we evaluate a series of state-of-the-art models using three representative methods (Agentless, SWE-agent, and OpenHands) and present a comprehensive analysis with key empirical insights. In addition, we launch a Multi-SWE-RL open-source community, aimed at building large-scale reinforcement learning (RL) training datasets for issue-resolving tasks. As an initial contribution, we release a set of 4,723 well-structured instances spanning seven programming languages, laying a solid foundation for RL research in this domain. More importantly, we open-source our entire data production pipeline, along with detailed tutorials, encouraging the open-source community to continuously contribute and expand the dataset. We envision our Multi-SWE-bench and the ever-growing Multi-SWE-RL community as catalysts for advancing RL toward its full potential, bringing us one step closer to the dawn of AGI.

|[Figure 6]|
|---|
| |
| |

| | |
|---|---|
| | |

|[Figure 7]|
|---|
| |
| |

|[Figure 8]|
|---|
| |

Figure 1. Resolved rate (%) on Multi-SWE-bench (Claude-3.5-Sonnet).

Author contributions listed at end of paper. Correspondence to: {zandaoguang, shen.kai}@bytedance.com

#### Contents

- 1 Introduction 3
- 2 Related Work 4
- 3 Multi-SWE-bench 5

- 3.1 Benchmark Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3.1.1 Phase 1: Repository Selection . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.1.2 Phase 2: Pull Request Crawling . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.1.3 Phase 3: Environment Determination . . . . . . . . . . . . . . . . . . . . . 6
- 3.1.4 Phase 4: Pull Request Filtering . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.1.5 Phase 5: Manual Verification . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 3.2 Features of Multi-SWE-bench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 Multi-SWE-RL Open-Source Community 10
- 5 Experimental Setups 11

- 5.1 Evaluated LLMs and Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.2 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 6 Experimental Results 13

- 6.1 Performance on Multi-SWE-bench . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 6.1.1 Performance across Programming Languages . . . . . . . . . . . . . . . . 13
- 6.1.2 Performance across Various Methods and LLMs . . . . . . . . . . . . . . . 15
- 6.1.3 Performance across Different Repositories . . . . . . . . . . . . . . . . . . 17

- 6.2 Influencing Factors of Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 6.2.1 Issue Type . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 6.2.2 Characteristics of Issue Description . . . . . . . . . . . . . . . . . . . . . . 21
- 6.2.3 Characteristics of Fix Patches . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- 6.3 Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 6.4 Resource Consumption . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 6.5 Troubleshooting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- 7 Conclusions and Future Works 27

#### 1. Introduction

Automating software engineering tasks with large language models (LLMs) has gained considerable attention [Zan et al., 2023, Zheng et al., 2023b, Jiang et al., 2024, Jelodar et al., 2025] recently. Beyond code generation, the issue resolving task proposed by SWE-bench [Jimenez et al., 2023] changes the role of LLMs from code assistants to fully autonomous AI programmers. SWE-bench contains 2,294 issues from 12 widely-used open-sourced Python libraries. LLMs are tasked to generate a patch based on the issue description along with the buggy code repository. SWE-bench Verified is a subset of 500 human-validated issues selected from SWE-bench, chosen for appropriately scoped unit tests and well-specified issue descriptions. Within less than one year, the resolving rate on SWE-bench Verified increased from 0.40% [Jimenez et al., 2023] (for RAG+GPT3.5) to 65.40% [augment code, 2025] (for Augment Agent v0).

Although existing works based on SWE-bench demonstrate significant progress in Pythonbased issue resolving, the diversity of programming languages in real-world repositories presents additional challenges that remain unexplored. In particular, repositories in different languages follow distinct programming paradigms, idiomatic patterns, and runtime behaviors, which may impact the effectiveness of current approaches. This raises the question of whether the impressive performance of existing agents on Python issues can be generalized to other widely used languages, such as Java, TypeScript, JavaScript, Go, Rust, C, and C++.

To answer this question, we introduce Multi-SWE-bench, a multilingual benchmark for issue resolving, consisting of 1,632 issues across 7 widely used programming languages: Java, TypeScript, JavaScript, Go, Rust, C, and C++. To construct a reliable benchmark for evaluating the ability of agents to resolve real-world software issues, we employ a systematic five-phase pipeline. First, we select high-quality repositories from GitHub based on star ratings and runnability counts to ensure both popularity and practical usability. Second, we collect issue-related pull requests (PRs) along with their corresponding metadata. Third, we build Dockerized environments for each PR by extracting dependencies from CI/CD workflows and documentation to ensure reproducible execution. Fourth, we validate PRs by analyzing test outcomes across patch configurations, retaining only those with clear bug-fixing effects and no regressions. Fifth, we perform rigorous manual verification through dual annotation and cross-review, ensuring high-quality ground truth aligned with SWE-bench verified standards. By ensuring diversity, executability, and human-verified correctness, Multi-SWE-bench sets a high standard for evaluating LLMs on realistic and non-trivial issue-resolving tasks.

With its wide coverage of languages and issue types, Multi-SWE-bench introduces realistic challenges that push the boundaries of LLM-based software agents. We use Multi-SWE-bench to evaluate the generalizability of 3 representative methods (i.e., Agentless [Xia et al., 2024], SWE-agent [Yang et al., 2024], and OpenHands +CodeAct v2.1 [Wang et al., 2024b]) based on 9 top-performing frontier models (i.e., GPT-4o, OpenAI-o1, OpenAI-o3-mini-high, Claude-3.5Sonnet, Claude-3.7-Sonnet, DeepSeek-V3, DeepSeek-R1, Qwen2.5-72B-Instruct, and Doubao-

- 1.5-Pro). Our evaluation provides a comparative analysis of the overall effectiveness of these methods across seven programming languages, along with Python, offering insights into their cross-language capabilities. Furthermore, we conduct a fine-grained analysis of the key factors influencing model performance and investigate failure cases for each language to identify underlying challenges and limitations. Through comprehensive analysis and comparison, we provide a good understanding of existing models and shed light on future directions and further progress. For example, our findings show that models perform generally better when issue descriptions are longer, indicating a strong reliance on rich contextual grounding; in contrast, resolved rates drop sharply when fix patches exceed 600 tokens or touch more than one file,

exposing weaknesses in long-context retention and multi-file reasoning. Together, these findings delineate the current boundary of LLM capabilities in software engineering and define the key obstacles to real-world deployment.

Beyond Multi-SWE-bench, we launch the Multi-SWE-RL open-source community to address the pressing need for scalable, high-quality RL environments in software engineering. Recent models such as DeepSeek-R1 [Guo et al., 2025], OpenAI-o1 [Jaech et al., 2024], and OpenAI-o3 [OpenAI, 2025] have demonstrated the potential of RL even with simplistic reward signals. These advances reinforce our belief that "scaling RL in real-world software environments is a key pathway toward human-level intelligence". However, the creation of realistic, interactive environments remains a major bottleneck. As a first step toward scalable RL in software engineering, Multi-SWE-RL therefore launches a collaborative initiative to build training data and environment from real-world tasks. As the initial contribution to the Multi-SWE-RL community, we release a dataset of 4,723 containerized issue-resolving instances spanning 7 programming languages. Each instance is equipped with a reproducible execution environment, enabling plug-and-play training for RL agents in realistic software contexts. We envision this release as a spark—igniting broader community interest in the construction of RL training data and paving the way toward fully autonomous agent systems.

In summary, our main contributions are:

- • Multi-SWE-bench, a multilingual benchmark for issue resolving, consisting of 1,632 human-validated GitHub issues on 7 widely used programming language. It serves as a reliable support for comprehensively evaluating the performance of agents in real-world software development scenarios.
- • A large-scale evaluation of 9 state-of-the-art LLMs across 3 representative methods on Multi-SWE-bench, yielding diagnostic insights to guide future research.
- • Multi-SWE-RL, a community-driven open-source effort that initiates scalable RL data creation from real-world software tasks, laying the groundwork for long-term progress in multilingual software agent development.

#### 2. Related Work

The remarkable performance of LLMs in code-related tasks has motivated substantial research to study their role in automating software engineering. To evaluate the capabilities and limitations of existing approaches, a wide range of benchmarks for code-related tasks has been developed. Early efforts in this domain focused on primarily evaluating models in monolingual programlevel evaluations [Allamanis and Sutton, 2013, Raychev et al., 2016, Iyer et al., 2018, Chen et al., 2021b, Austin et al., 2021, Wang et al., 2023]. As LLMs advanced, benchmarks evolved in two key dimensions to better align with real-world software engineering scenarios. First, benchmarks shift from monolingual to multilingual tasks, with growing interest and practical needs in evaluating LLMs’ performance across multiple programming languages. Examples include Multilingual-HumanEval [Athiwaratkun et al., 2023] and HumanEval-X [Zheng et al.,

- 2023a], which extend the HumanEval [Chen et al., 2021a] benchmark to multiple languages, and MBXP [Athiwaratkun et al., 2022], which extends MBPP to multilingual scenarios. Second, benchmarks shift from program-level to repository-level tasks, focusing on more complex scenarios such as library-oriented code generation [Zan et al., 2022], repository-level code completion [Zhang et al., 2023, Liu et al., 2024a, Ding et al., 2024, Liu et al., 2024b, Yu et al., 2024], and bug fix [Mündler et al., 2024, Ouyang et al., 2024, Sun et al., Saavedra et al., 2024]. These evolving benchmarks aim to provide a more comprehensive evaluation of LLMs in real-world

###### Phase 3: Environment Determination

###### Phase 1: Repository Selection

Phase 4: PR Filtering

[Figure 9]

Repositories on GitHub

[Figure 10]

[Figure 11]

Environment configuration files

git checkout base commit

Search by keywords

Filter by stars

Run test suits

Execute

Manually analyze

[Figure 12]

High-quality repositories

[Figure 13]

git apply test.patch

[Figure 14]

Run.log

Environment dependencies

Verify runnability

[Figure 15]

Run test suits Execute

[Figure 16]

Runnable repositories

[Figure 17]

Repo-common PR-specific

git apply fix.patch

[Figure 18]

Test.log

###### Phase 2: PR Crawling

Run test suits

Generate

Rectify if fixable

[Figure 19]

Fix.log

[Figure 20]

[Figure 21]

Docker file

Error log

[Figure 22]

All pull requests (PRs)

Extract test cases

Build Build successfully failed

failed

Filter

[Figure 23]

[Figure 24]

[Figure 25]

Drop the PR if not fixable

Run.log Test.log Fix.log

[Figure 26]

Issue-related PRs

repo

- Test1 Passed Failed Passed
- Test2 Passed Passed Failed

[Figure 27]

Docker image

the

Crawl data for each PR

Launch

[Figure 28]

Raw data of the PR

Start

...... ...... ...... ......

Launch the repo successfully

base commit test.patch fix.patch instance id

[Figure 29]

Retain the PR if qualified

[Figure 30]

Docker container Success log

issue

[Figure 31]

Log analysis

Drop the PR if not qualified

Phase 5: Manual Verification

Questionnaire & Annotation system

[Figure 32]

Unverified Multi-SWE-Bench

[Figure 33]

Annotation Quality Assessment Multi-SWE-Bench

Data Annotation

[Figure 34]

[Figure 35]

[Figure 36]

Annotator training

Language stack determination

[Figure 37]

Annotator hiring

[Figure 38]

[Figure 39]

Figure 2. Construction of Multi-SWE-bench.

software development environments.

In addition to existing benchmarks, SWE-bench [Jimenez et al., 2023] has gained significant attention since its release. Instead of focusing on isolating code subtasks into separate datasets, SWE-bench addresses a broader range of tasks through repository-level issue resolving. These issue resolving tasks, including bug fixing, new feature requests, and optimization, which provide a more comprehensive evaluation of LLMs’ ability to automating software development. While SWE-bench is limited to textual context, SWE-bench Multimodal [Yang et al., 2025] and Visual SWE-bench [Zhang et al., 2024] extend evaluation to systems fixing bugs in visually-oriented and user-facing applications. SWE-Lancer [Miserendino et al., 2025] focuses on JavaScript and TypeScript, featuring over 1,400 freelance tasks from Upwork, including technical and managerial tasks. Despite these advancements, the performance of LLMs on other widely used programming languages remains underexplored. Our work aims to bridge this gap with Multi-SWE-bench, a large-scale multilingual benchmark for issue resolving, with 1,632 human-validated GitHub issues across 7 widely used languages.

#### 3. Multi-SWE-bench

Multi-SWE-bench consists of 1,632 issue-resolving tasks spanning 7 programming languages: Java, TypeScript, JavaScript, Go, Rust, C, and C++. This section will provide the construction process of Multi-SWE-bench, along with an analysis of its key features.

###### 3.1. Benchmark Construction

To evaluate the generalizability of LLMs as issue resolvers, seven programming languages are selected to construct Multi-SWE-bench through five phases. As shown in Fig. 2, the first four phases create a large pool of candidate data for each language, while the fifth phase finalizes the Multi-SWE-bench through manual verification.

###### 3.1.1. Phase 1: Repository Selection

We carefully curate a diverse set of high-quality GitHub repositories for each of the seven target programming languages. The selection process is guided by the following criteria:

- • Popularity and Maintenance: Repositories must have over 500 GitHub stars and demonstrate active maintenance for at least six months. In addition, we prioritize repositories frequently recommended in Google searches using keywords such as "high-quality", "well-maintained", and "popular".
- • CI/CD Support: Selected repositories are required to include CI/CD configurations (e.g., workflows under ".github/workflows/") to ensure automated testing and reproducibility.
- • Build Viability: After minimal manual setup, the latest commit must be buildable and testable in a clean environment, ensuring compatibility with modern tooling and infrastructure.

This filtering process results in a robust foundation of repositories that are representative of real-world, production-level codebases, laying the groundwork for subsequent phases.

###### 3.1.2. Phase 2: Pull Request Crawling

This phase aims to crawl issue-resolving pull requests (PRs) for each repository selected in phase

1. All PRs from the repository are collected and then filtered based on the following criteria:

- • Linked with at least one GitHub issue: The PR must be linked to at least one issue to ensure it addresses a clearly defined bug report or feature request.
- • Modified test files: The PR must include changes to test files, guaranteeing proper testing is in place to verify the correctness of the fix patches.
- • Merged into the main branch: The PR must be merged into the main branch, indicating it has been accepted by the repository’s maintainers and fully integrated.

After filtering, detailed information is gathered for each PR, including attributes such as issue description, base commit, fix.patch, and test.patch.

###### 3.1.3. Phase 3: Environment Determination

To enable faithful execution and evaluation of issue-resolving tasks, each PR must be reproducibly built and executed in an isolated environment. In this phase, we construct a Dockerbased runtime environment for every PR by automatically identifying and provisioning its necessary dependencies. The process begins with a manual inspection of environment-related artifacts, including CI/CD configuration files (e.g., GitHub Actions), repository documentation (e.g., README files), and exploratory trial runs. Through this analysis, we extract environment dependencies and classify them into two categories: repo-common dependencies, which are shared across the entire repository, and PR-specific dependencies, which are introduced or modified by the target PR.

Using the extracted dependency information, we generate a tailored Dockerfile and attempt to build a corresponding Docker image. If the build fails, we examine the error logs to identify missing dependencies, misconfigurations, or version conflicts. For fixable errors, we iteratively patch the Dockerfile or supporting scripts; otherwise, we discard the PR to ensure reliability. Once the image is successfully built, we verify whether the repository can be launched at the specific commit associated with the PR. This step ensures that all required services, packages,

and configurations are functional. If the launch fails, we again attempt corrective actions; if successful, we obtain a validated and executable containerized environment for downstream evaluation. This phase ensures that each PR is equipped with a clean and functional containerized environment, laying a necessary foundation for subsequent testing and analysis.

###### 3.1.4. Phase 4: Pull Request Filtering

With both the PR metadata and a working runtime environment established in the previous phases, we now perform semantic validation to ensure each PR meets the requirements of issue resolving. This is done by analyzing test behaviors under controlled patch configurations. For each PR, unlike SWE-bench which runs only relevant tests, we run the full test suit under the following three settings:

- • Run.log: Tests are executed on the base commit.
- • Test.log: The test.patch is applied to the base commit before execution.
- • Fix.log: Both the test.patch and the fix.patch are applied to the base commit before execution.

We then extract the execution status of each test case from these logs. Unlike SWE-bench, which considers only PASSED and FAILED, we also track NONE and SKIPPED status, as some test cases may be conditionally disabled or omitted after applying patches—resulting in inconsistent test counts across the three logs. Each test case is summarized by its status transition across the three settings. For instance, a test case with PASSED, FAILED, and PASSED statuses in run.log, test.log, and fix.log, respectively, is represented as PASSED→FAILED→PASSED. We apply the following filtering rules to determine eligible PRs:

- • PRs with any ANY→PASSED→FAILED transitions are discarded to ensure that no potential regressions are introduced by the fix.patch.
- • PRs without at least one ANY→FAILED→PASSED transition are discarded, as they do not demonstrate any effective bug fix.
- • PRs exhibiting abnormal transitions such as PASSED→NONE/SKIPPED→FAILED are discarded to eliminate ambiguous test behaviors.

After applying these criteria, we retain 2,456 issue-resolving instances spanning 39 repositories across 7 languages. For each instance, we extract test cases exhibiting transitions of the form Any→FAILED/PASSED/SKIPPED/NONE→PASSED, and include them in the dataset to enable fine-grained and reliable evaluation.

###### 3.1.5. Phase 5: Manual Verification

To ensure the reliability of Multi-SWE-bench in evaluating the issue-resolving capabilities of LLMs, we conduct comprehensive manual verification on the 2,456 issue-resolving instances retained from the previous phase. Our verification process follows the annotation guidelines of the recently released SWE-bench-verified1. In detail, we recruit 68 annotators through outsourcing, with the number per language proportional to the remaining annotation workload. All annotators were screened based on their qualifications, including at least two years of experience in the target language and a relevant bachelor’s degree or higher.

Before annotation, each annotator undergoes a one-hour training session covering the background, objectives, procedures, deliverables, and quality standards of the task. To further

1https://openai.com/index/introducing-swe-bench-verified

##### Table 1. Statistics of the Multi-SWE-bench. #A2P2P, #A2F2P, and #A2N2P represent the average counts of Any→PASSED&FAILED&NONE→PASSED unit tests.

|Repository|Instance<br><br>|Issue description<br><br>|Fix patches<br><br>|Unit tests|
|---|---|---|---|---|
|Org/Repo #Files #LoC|#Num|Avg. #Tokens|Avg. #Lines Avg. #Hunks Avg. #Files<br><br>|#A2P2P #A2F2P #A2N2P|

Java alibaba/fastjson2 4244 443.8k 6 459.2 10.5 1.3 1.2 1243.5 0.8 1020.5 elastic/logstash 562 59.9k 38 1600.4 212.3 10.0 4.6 554.7 1.9 256.2 mockito/mockito 986 84.0k 6 315.2 92.5 10.3 4.7 97.2 1.0 3.8 apache/dubbo 3939 402.1k 3 774.0 9.3 3.0 1.3 2.0 57.0 0.0 fasterxml/j-core 366 105.7k 18 304.7 33.8 4.8 2.1 2.0 85.6 0.0 fasterxml/j-dbind 1230 217.5k 42 621.5 35.1 3.9 2.1 2.0 73.8 0.0 fasterxml/j-dfmt-xml 206 23.0k 5 1071.8 98.4 10.4 3.2 2.0 94.2 0.0 google/gson 261 48.0k 5 365.8 35.8 4.6 1.8 2.0 62.6 0.0 google-ct/jib 604 75.5k 5 1094.6 15.2 3.2 2.6 2.0 96.2 0.0

TypeScript darkreader/darkreader 189 26.2k 2 749.5 13.0 2.0 1.5 41.0 3.5 0.0 mui/material-ui 27632 698.6k 174 508.6 331.2 20.2 12.0 5001.3 2.3 836.8 vuejs/core 509 128.2k 48 694.8 22.9 3.5 1.9 2920.4 3.0 0.0

JavaScript ag/gh-rdme-stats 69 11.8k 19 287.1 123.6 13.5 4.8 108.9 3.5 3.4 axios/axios 166 21.0k 4 490.8 179.5 7.8 4.0 68.5 1.2 0.0 expressjs/express 142 17.3k 4 177.5 7.2 2.2 1.5 808.2 1.5 65.2 iamkun/dayjs 324 17.1k 56 325.6 21.7 2.7 2.0 60.4 1.2 3.2 Kong/insomnia 526 182.0k 1 709.0 1.0 1.0 1.0 105.0 1.0 0.0 sveltejs/svelte 2800 105.9k 272 618.9 72.0 8.4 4.0 4904.2 5.5 0.0

Go cli/cli 737 165.1k 397 347.6 103.8 9.0 3.9 1997.0 2.9 31.0 grpc/grpc-go 981 260.8k 16 276.1 81.8 7.7 2.8 230.4 0.6 6.6 zeromicro/go-zero 960 117.6k 15 205.2 52.4 4.9 2.7 1318.9 0.3 43.9

Rust BurntSushi/ripgrep 98 45.4k 14 553.7 1604.9 21.9 7.5 233.2 1.1 8.1 clap-rs/clap 321 70.4k 132 987.0 147.1 15.7 4.7 489.5 3.1 378.8 nushell/nushell 1479 264.2k 14 795.6 155.0 10.6 4.3 798.6 2.6 336.6 rayon-rs/rayon 191 36.9k 2 153.5 637.5 5.5 2.0 113.5 0.5 171.0 serde-rs/serde 188 36.5k 2 171.5 72.5 3.0 3.0 0.0 0.0 294.5 sharkdp/bat 83 22.0k 10 638.2 239.5 14.1 5.9 152.7 1.7 33.6 sharkdp/fd 24 6.7k 14 167.8 55.8 7.8 4.5 186.5 1.1 0.0 tokio-rs/bytes 33 11.9k 5 188.0 45.0 5.6 1.8 23.2 0.4 91.6 tokio-rs/tokio 727 141.5k 25 590.0 139.8 10.6 3.5 26.6 0.0 287.4 tokio-rs/tracing 241 60.9k 21 472.0 597.2 39.3 7.1 30.8 0.2 182.0

C

facebook/zstd 276 119.8k 29 496.6 67.6 10.9 3.0 0.8 0.5 5.6 jqlang/jq 80 43.0k 17 429.8 26.1 2.7 1.8 27.2 1.0 0.1 ponylang/ponyc 285 80.2k 82 480.2 205.4 15.6 5.7 997.6 1.9 388.8

C++ catchorg/Catch2 399 58.0k 12 357.3 469.0 15.4 8.2 19.9 0.7 17.6 fmtlib/fmt 25 36.4k 41 397.7 36.8 3.0 1.1 9.3 0.0 9.3 nlohmann/json 477 124.7k 55 905.5 405.8 27.9 6.5 26.5 0.0 42.9 simdjson/simdjson 455 229.7k 20 320.2 768.5 35.5 11.0 18.6 0.0 41.5 yhirose/cpp-httplib 33 50.9k 1 240.0 1.0 1.0 1.0 272.0 1.0 0.0

support consistency and correctness during the annotation process, we establish dedicated discussion channels to provide real-time guidance and handle edge cases collaboratively. Each instance is independently labeled by two annotators. Upon completion, the two annotations are cross-reviewed to produce a single, agreed-upon final label. To ensure high annotation quality, we additionally form an internal quality assessment team of 14 experienced engineers. This team produces reference answers and verifies that the outsourced annotations for each language reach a minimum accuracy threshold of 80%. After rigorous manual verification, a total of 1,632 high-quality instances are retained as the final Multi-SWE-bench, filtered by annotation criteria from the verification questionnaire2: Q2.1=0 & Q3.1∈{2,3} & Q4.1∈{2,3}. All annotation results have been made publicly available to ensure the transparency of the dataset3.

- 2https://github.com/multi-swe-bench/multi-swe-bench/blob/main/docs/manual-verificat

ion/questionnaire-demo.pdf

- 3https://github.com/multi-swe-bench/multi-swe-bench/tree/main/docs/manual-verificat

ion/annotation-results

15mins 15mins - 1h 1h - 4h 4h

| |
|---|

400

300

#Issues

200

100

0

Java TS JS Go Rust C C++

Figure 3. Distribution of estimated time consumption of issues in Multi-SWE-bench.

###### 3.2. Features of Multi-SWE-bench

Tab. 1 presents an overview of the key statistics of Multi-SWE-bench, highlighting its coverage across a wide range of programming languages and repositories. It includes 1,632 issueresolving instances sourced from 39 diverse repositories, spanning 7 popular languages: Java, TypeScript, JavaScript, Go, Rust, C, and C++. These repositories vary significantly in size and complexity, with the number of files ranging from 24 to 27,632, and lines of code from 6.7k to 698.6k. This diversity ensures that Multi-SWE-bench reflects realistic and heterogeneous software development scenarios. In terms of issue descriptions, the complexity varies notably across repositories and languages. Java and Rust projects generally present longer and more detailed issue reports (e.g., "elastic/logstash" with 1600.4 tokens), suggesting more contextdependent reasoning is required. In contrast, JavaScript, Go, and C issues are typically brief and focused (e.g., "expressjs/express" with 177.5 tokens), implying simpler or more localized fixes. This variation in description length highlights the need for LLMs to adapt to both under- and over-specified problem statements. Similarly, patch complexity also differs significantly across languages. Rust and C++ projects frequently require large-scale edits, with some instances modifying over 200 lines and 7 files per patch (e.g., "BurntSushi/ripgrep" and "simdjson/simdjson"). Conversely, JavaScript, and TypeScript patches tend to be more localized and atomic, often involving under 3 hunks and fewer than 2 files. These contrasts emphasize the importance of handling both high-granularity refactoring and precision editing. Moreover, all repositories come with strong test coverage, providing reliable signals for verifying patch correctness, as confirmed by the manual verification in Sec. 3.1.5.

We further display the manual verification results in Tab. 3, which show that most instances have no serious issues and receive high scores, confirming the overall quality of the repositories selected in Sec. 3.1.1. As part of the manual annotation process in Multi-SWE-bench, we recorded the estimated time required to resolve each issue, categorized into four buckets: ≤15 minutes, 15 minutes–1 hour, 1–4 hours, and ≥4 hours (Fig. 3). Unlike SWE-Bench, we use this timebased annotation to define difficulty levels across all languages: easy (≤15 mins), medium (15 mins–1h), and hard (≥1h). Tab. 2 summarizes the distribution of difficulty levels by language. We observe clear trends across these categories: As difficulty increases, issues tend to have longer descriptions, and the corresponding patches involve more lines, hunks, and files. Interestingly, certain easy instances exhibit large-scale edits (e.g., Rust), which are typically due to highly repetitive and pattern-consistent changes. This highlights the advantage of time-based difficulty categorization over superficial metrics like token count or file span. Such categorization provides a more realistic measure of problem complexity and can better guide the development and evaluation of LLMs.

- Table 2. Feature distribution of Multi-SWE-bench instances by difficulty and language.

|Difficulty<br><br>|Instance<br><br>|Issue description<br><br>|Fix patches|Unit tests|
|---|---|---|---|---|
| |#Num<br><br>|Avg. #Tokens|Avg. #Lines Avg. #Hunks Avg. #Files<br><br>|#A2P2P #A2F2P #A2N2P|

Python

Easy 194 417.9 5.0 1.4 1.0 116.2 3.9 0 Medium 261 555.9 14.1 2.5 1.3 115.4 2.4 0 Hard 45 589.8 55.8 6.8 2.0 166.3 2.9 0

Java

- Easy 27 733.8 12.4 2.6 1.8 126.8 58.0 76.1 Medium 65 843.3 36.2 4.6 2.1 182.3 58.6 136.9 Hard 36 1039.0 246.1 11.9 5.4 389.1 21.8 136.9

TypeScript Easy 72 600.1 8.3 2.1 1.5 4806.8 2.0 0.0 Medium 88 566.9 74.3 8.8 4.3 4854.6 2.8 214.3 Hard 64 472.8 806.6 43.2 26.5 3706.1 2.7 1980.4

JavaScript Easy 10 282.4 4.7 1.8 1.6 616.8 1.2 35.1 Medium 105 505.8 15.5 2.6 2.1 3161.0 3.6 0.8 Hard 241 578.7 92.2 10.1 4.5 4169.9 5.2 0.3

Go

Easy 141 411.7 26.6 4.0 2.7 2181.0 2.6 20.4 Medium 153 331.4 49.6 6.9 2.6 1832.5 2.2 25.7 Hard 134 274.0 238.6 16.0 6.6 1704.2 3.4 46.7

Rust

Easy 66 808.2 318.7 7.0 3.3 465.2 3.2 212.0 Medium 126 814.7 113.6 10.6 3.7 343.0 1.8 300.5 Hard 47 599.4 629.0 45.2 10.3 232.3 1.1 334.0

C

Easy 30 551.4 16.4 3.7 2.2 424.8 0.8 208.2 Medium 54 449.9 36.7 5.5 2.5 715.5 1.0 228.2 Hard 44 460.2 381.1 28.0 8.7 702.5 2.4 306.3

C++

- Easy 28 494.5 25.2 4.4 2.2 45.0 0.1 15.7 Medium 59 427.5 204.2 7.6 3.3 18.2 0.1 23.0 Hard 42 904.2 763.7 47.2 11.1 9.3 0.0 47.2

- Table 3. Scoring statistics for Multi-SWE-bench from the verification questionnaire.

|Languages<br><br>|Q2.1 Serious Issue Flag|Q3.1 Clarity of Issue Description<br><br>|Q4.1 Coverage of Unit Tests|
|---|---|---|---|
| |#Score 0 #Score 1<br><br>|#Score 0 #Score 1 #Score 2 #Score 3<br><br>|#Score 0 #Score 1 #Score 2 #Score 3|
|Java TypeScript JavaScript Go Rust C C++|146 10<br><br>382 8 586 4<br><br>579 26 328 11<br><br>200 6 162 7<br><br>|2 2 44 98 5 56 121 200 0 6 13 567 5 10 276 288 4 20 165 139 2 4 115 79 0 6 96 60|10 5 17 114 31 76 133 142 55 172 305 54<br><br>44 100 151 284 23 50 74 181 13 55 83 49<br><br>7 21 45 89|

#### 4. Multi-SWE-RL Open-Source Community

Community Introduction. Multi-SWE-RL is an open-source community aimed at developing high-quality RL training datasets for complex software engineering tasks. Its purpose is to serve as the foundational infrastructure for training fully autonomous agents capable of addressing real-world software engineering challenges, paving the way toward achieving AGI. The need for such a community has become increasingly urgent as the potential of RL continues to expand. Notable models such as DeepSeek-R1 [Guo et al., 2025], OpenAI o1 [Jaech et al., 2024], and

- o3 [OpenAI, 2025] have demonstrated the power of RL, even with simple, rule-based reward signals. In light of these advancements, we are firmly convinced that “scaling RL in real-world environments is the path toward human-like intelligence”. However, the creation of such interactive environments and data trajectories is extremely challenging. For instance, the development
- of our Multi-SWE-bench took about one year to produce just high-quality 1,632 instances.

Therefore, we launched the Multi-SWE-RL community to harness the power of open-source collaborative contributions for building diverse RL environments.

Community Initialization. To bootstrap the Multi-SWE-RL community, we release an initial dataset comprising 4,723 issue-resolving instances spanning 76 widely-used open-source repositories and 7 programming languages: Java, TypeScript, JavaScript, Go, Rust, C, and C++. Each instance is equipped with a fully containerized execution environment to ensure reproducibility and ease of integration. This dataset was constructed using the same pipeline as Multi-SWEbench, excluding the manual verification process described in Sec. 3.1.5. Details about this release are available at Hugging Face dataset and Multi-SWE-RL contribution board. We envision this initial release as a spark—igniting broader community collaboration and fueling the construction of scalable, high-quality RL environments for real-world software engineering.

Contribution Guidelines and Recognition. We welcome contributions from the community to expand the Multi-SWE-bench and Multi-SWE-RL. To help new contributors get started, we provide a detailed demo that walks through the process of creating an issue-resolving instance, available at Contribution-demo.md. To recognize and incentivize community contributions, we maintain a rolling update schedule through periodic arXiv updates or follow-up technical reports, with new versions released every three months. Each update may include:

- • Newly added benchmarks for additional programming languages in Multi-SWE-bench, with new authors and contributors;
- • Newly contributed data to Multi-SWE-RL, with new authors and contributors;
- • Newly reported performance results from RL trials on Multi-SWE-bench using Multi-SWERL data, with new authors and contributors;
- • Newly open-sourced RL models with significantly enhanced performance, with new authors and contributors.

Our contribution incentive policy is detailed at Incentive-plan.md. We are committed to continuously refining our contribution strategy to encourage sustained open-source engagement, and we warmly invite the community to take part in shaping and scaling this collaborative effort.

#### 5. Experimental Setups

###### 5.1. Evaluated LLMs and Methods

Methods. We evaluate three representative methods for issue resolving: Agentless [Xia et al.,

- 2024], SWE-agent [Yang et al., 2024], and OpenHands + CodeAct v2.1 [Wang et al., 2024b]. These methods were specifically designed for Python as used in SWE-Bench [Jimenez et al., 2023]. We extended the methods to accommodate the multilingual nature of Multi-SWE-bench4.

- • Agentless5→MagentLess6: Agentless addresses the issue resolving task through a multistage fixed workflow, including hierarchical fault localization, code repair, and candidate patch selection via regression and reproduction tests. In MagentLess, we made the following key modifications to support multilingual adaptation and improve scalability:

- 1. We revised all prompts to accommodate the newly added languages.

4MagentLess and MopenHands are pronounced as /"mA:dZ@nt.l@s/ and /"moUp@n.hAndz/, respectively.

- 5https://github.com/OpenAutoCoder/Agentless
- 6https://github.com/multi-swe-bench/MagentLess

- 2. We replaced all file skeleton inputs with full file content, as extracting file skeletons is challenging in some programming languages.
- 3. We implemented function and class extraction for all languages using Tree-sitter7.
- 4. We pruned the extracted repository structures by retaining only files and directories with specific extensions, as repositories in certain languages (e.g., TypeScript) often contain an excessive number of files that may exceed LLM context limits.
- 5. We removed the candidate patch selection stage and retained only fault localization and code repair, as regression and reproduction testing is cumbersome to implement across languages and falls outside the scope of this work.

- • SWE-agent8→MSWE-agent9: SWE-agent is an agent-based approach that solves issues through multi-turn interactions via a predefined agent-computer interface (ACI). To support Multi-SWE-bench, we developed MSWE-agent with the following modifications:

- 1. We revised all prompts to accommodate the newly added languages.
- 2. We truncated overly long environment observations to ensure stable agent execution.
- 3. We added ".gitignore" to exclude compiled artifacts (e.g., ".o", ".bin") in languages like C/C++, which could otherwise interfere with "git apply".
- 4. We fixed language-specific commands that caused crashes or non-terminating behavior during execution to ensure stable agent execution.

- • OpenHands10→MopenHands11: OpenHands is a widely adopted platform for building software development agents. In MopenHands, we made the following key modifications to support multilingual adaptation:

- 1. We revised all prompts to support the newly added programming languages.
- 2. We added ".gitignore" to exclude compiled artifacts, as also done in MSWE-agent.
- 3. We fixed several implementation bugs, including an issue where "CmdRunAction" incorrectly rendered tab characters (\t) as spaces in "git diff" outputs, making patches unapplicable. To resolve this, we redirected the diff to a file and read it using "FileReadAction", which proved especially important in languages like Go.

We have systematically extended the above methods to support the multilingual setting of MultiSWE-bench. Still, there remains substantial room for improvement, particularly in languagespecific adaptation and overall robustness. We welcome community collaboration to further advance their capabilities.

LLMs. We evaluated 9 popular LLMs across the above three methods: GPT-4o (gpt-4o-2024-1120), OpenAI-o1 (o1-2024-12-17), OpenAI-o3-mini-high (o3-mini-2025-01-31 high), Claude-3.5Sonnet (claude-3-5-sonnet-20241022), Claude-3.7-Sonnet (claude-3-7-sonnet-20250219), DeepSeekV3, DeepSeek-R1, Qwen2.5-72B-Instruct, and Doubao-1.5-pro.

###### 5.2. Evaluation Metrics

Following SWE-Bench [Jimenez et al., 2023] and SWE-Lancer [Miserendino et al., 2025], we adopt Resolved Rate (%) as our primary evaluation metric, measuring the percentage of issues resolved. In addition, we report several other metrics to provide a more detailed analysis:

- 7https://tree-sitter.github.io
- 8https://github.com/SWE-agent/SWE-agent
- 9https://github.com/multi-swe-bench/MSWE-agent
- 10https://github.com/All-Hands-AI/OpenHands
- 11https://github.com/multi-swe-bench/MopenHands

Success Location (%) — the accuracy of fault localization at file level; and Average Cost ($) the average cost per issue.

#### 6. Experimental Results

###### 6.1. Performance on Multi-SWE-bench

In this subsection, we conduct a systematic evaluation of issue resolving performance on MultiSWE-bench along three key dimensions: (1) language-specific performance, examining variations in effectiveness across programming languages; (2) LLMs and agents comparison, evaluating the differential capabilities of various LLMs and methods; and (3) repository-level performance, analyzing the impact of repository characteristics on resolved rate.

###### 6.1.1. Performance across Programming Languages

- Tab. 4 presents the overall performance of the agents across eight programming languages, and
- Tab. 5 further details the results across three distinct difficulty levels. Based on these tables, several key observations can be drawn and outlined below.

Table 4. Resolved rate (%) of different models on Multi-SWE-bench.

Models Python Java TS JS Go Rust C C++ MagentLess GPT-4o 36.20 11.72 2.23 1.40 2.80 5.86 1.56 6.98 OpenAI-o1 48.20 21.09 5.80 5.06 4.44 7.11 1.56 5.43 OpenAI-o3-mini-high 46.40 5.47 0.45 2.81 3.97 7.95 3.91 1.55 Claude-3.5-Sonnet 42.40 14.84 4.91 1.97 5.14 5.02 1.56 3.88

- Claude-3.7-Sonnet 44.60 14.06 3.57 1.97 5.84 5.44 2.34 3.10 DeepSeek-V3 41.00 7.03 6.70 3.37 5.37 5.02 3.13 1.55 DeepSeek-R1 42.20 22.66 6.25 4.49 3.74 6.69 0.78 3.10 Qwen2.5-72B-Instruct 26.80 10.94 4.46 0.84 1.40 2.51 0.78 0.78 Doubao-1.5-pro 26.20 5.47 2.23 1.12 2.10 4.18 0.00 0.00

MSWE-agent GPT-4o 18.80 12.50 0.45 0.84 2.34 2.09 1.56 2.33 OpenAI-o1 28.80 21.88 4.02 4.21 4.67 4.18 3.91 3.88 OpenAI-o3-mini-high 28.60 16.41 4.91 4.21 3.97 5.02 2.34 5.43 Claude-3.5-Sonnet 24.80 20.31 8.04 4.21 5.84 6.69 4.69 6.98

- Claude-3.7-Sonnet 45.80 23.44 11.16 4.78 5.37 6.69 8.59 11.63 DeepSeek-V3 4.20 11.72 2.68 2.53 4.44 5.86 2.34 7.75 DeepSeek-R1 2.00 9.38 5.80 1.40 2.10 2.09 0.78 6.20 Qwen2.5-72B-Instruct 8.60 2.34 0.00 0.56 0.47 0.42 1.56 0.00 Doubao-1.5-pro 12.40 7.03 1.79 1.40 2.10 1.67 2.34 6.20

MopenHands GPT-4o 25.60 9.38 0.00 1.97 3.50 3.35 0.00 3.88 OpenAI-o1 16.00 3.91 0.45 3.65 3.74 2.51 3.13 3.88 OpenAI-o3-mini-high 20.40 10.16 0.45 3.37 2.34 5.02 1.56 6.98 Claude-3.5-Sonnet 39.00 14.84 11.61 1.97 6.78 12.13 3.13 12.40 Claude-3.7-Sonnet 52.20 21.88 2.23 5.06 7.48 15.90 8.59 14.73 DeepSeek-V3 27.80 9.38 1.34 1.12 0.70 4.60 3.13 7.75 DeepSeek-R1 26.00 8.59 0.45 2.53 0.00 4.60 2.34 4.65 Qwen2.5-72B-Instruct 4.40 3.13 0.00 0.84 1.40 1.67 0.78 2.33 Doubao-1.5-pro 8.80 0.78 0.00 1.12 1.64 0.84 0.00 3.10

Qwen2.5-72B-Instruct6.707.410.0020.002.131.520.0010.713.453.080.000.951.310.791.850.000.000.000.000.000.754.260.000.00 Doubao-1.5-pro15.460.000.0010.004.961.520.0014.294.981.540.000.950.000.000.000.002.220.000.000.830.002.130.000.00

DeepSeek-V341.2418.522.780.002.136.066.6717.8621.4610.770.001.900.003.973.708.476.670.001.560.830.004.260.000.00 DeepSeek-R141.2414.811.3910.000.0013.646.6714.2919.1610.770.002.860.001.591.853.390.000.000.002.070.000.000.000.00

Claude-3.5-Sonnet48.9725.9318.0610.0012.7724.246.6732.1436.0218.4613.643.815.237.143.7010.1713.330.001.560.832.248.510.002.38 Claude-3.7-Sonnet71.6548.152.7830.0011.3521.2113.3332.1444.8323.082.277.629.1513.4911.1115.2511.110.001.562.901.4914.892.272.38

OpenAI-o118.567.411.3930.008.513.036.677.1416.484.620.006.671.962.383.705.082.220.000.001.240.752.130.000.00 OpenAI-o3-mini-high31.4422.221.3940.004.9613.646.6714.2914.5610.770.003.811.960.790.006.786.670.000.001.660.004.260.002.38

MopenHands GPT-4o38.6629.630.000.008.516.060.0010.7119.546.150.002.861.962.380.003.394.440.000.001.660.002.130.000.00

Qwen2.5-72B-Instruct15.467.410.000.001.421.520.000.004.981.540.000.950.000.003.700.000.000.000.000.410.000.000.000.00 Doubao-1.5-pro17.5311.112.7810.005.671.523.3317.8610.737.692.271.900.650.793.703.390.002.780.000.830.004.260.002.38

DeepSeek-V37.2233.335.560.009.2210.610.0010.712.689.232.273.813.274.763.7010.170.000.000.002.070.752.130.002.38 DeepSeek-R12.5814.819.7210.006.384.553.3317.861.9212.316.821.900.000.790.005.080.000.000.000.830.002.130.000.00

Claude-3.5-Sonnet28.3548.1515.280.0013.4813.6413.3317.8625.6720.005.687.622.614.763.706.784.440.003.132.901.492.130.000.00 Claude-3.7-Sonnet61.8644.4420.830.0010.6413.6420.0028.5740.6127.699.097.624.582.387.4111.866.670.003.133.730.758.512.270.00

OpenAI-o140.7248.154.1710.008.516.0610.007.1424.1423.084.556.673.923.973.705.084.440.003.132.901.492.130.000.00 OpenAI-o3-mini-high42.7833.3311.1120.009.2212.123.3314.2921.4618.463.413.812.612.383.705.088.890.000.003.730.002.130.000.00

MSWE-agent GPT-4o25.7722.220.000.005.671.526.677.1416.0915.381.140.951.313.170.001.694.440.000.000.830.000.000.000.00

Qwen2.5-72B-Instruct44.3333.336.9420.003.556.060.003.5718.397.694.550.000.651.591.850.000.000.001.560.410.000.000.000.00 Doubao-1.5-pro39.1814.811.3910.003.5510.610.000.0020.314.623.410.002.611.590.000.004.440.001.561.240.002.130.000.00

DeepSeek-V357.7318.5211.1130.009.9315.156.670.0034.876.156.824.764.580.791.853.394.440.001.561.661.492.132.270.00 DeepSeek-R158.7651.8511.1130.007.8015.150.003.5736.0223.086.827.621.963.971.855.086.670.000.002.071.492.130.000.00

Claude-3.5-Sonnet61.8637.0411.1130.0011.359.093.3310.7134.1013.852.271.903.273.171.853.396.670.001.560.830.754.260.000.00 Claude-3.7-Sonnet64.4333.335.5620.0013.4810.613.337.1435.6313.853.413.813.923.171.853.3911.110.001.560.410.004.262.270.00

OpenAI-o168.0440.7411.1120.006.3816.673.3314.2940.2324.624.559.525.883.171.855.088.890.001.562.490.754.260.000.00 OpenAI-o3-mini-high67.0114.811.3930.009.9322.736.673.5738.314.620.004.761.312.381.851.694.440.000.000.830.752.134.550.00

MagentLess GPT-4o55.1522.224.1720.006.3813.643.3317.8627.9713.851.141.901.961.591.856.782.220.001.560.410.006.380.000.00

Models

EasyMediumHard PythonJavaTSJSGoRustCC++PythonJavaTSJSGoRustCC++PythonJavaTSJSGoRustCC++

Table5.Resolvedrate(%)ofdifferentmodelsonMulti-SWE-benchacrossvariousdifficultylevels.

Limited generalization beyond Python. From Tab. 4, it can be observed that existing LLMs and methods demonstrate strong performance in resolving Python issues but struggle to generalize effectively across other languages. For example, LLMs such as OpenAI-o1 and Claude-3.7-Sonnet achieve high resolved rates for Python but significantly lower effectiveness for most other languages. This performance disparity can be attributed to three main factors: (1) Benchmark difficulty: Multi-SWE-bench is inherently more challenging than SWE-Bench-Verified, with a higher proportion of medium and hard issues (77.1% for Multi-SWE-bench compared to 61.2% for SWE-Bench-Verified, as calculated from Tab. 2). (2) Method optimization bias: The three methods are initially optimized for Python, resulting in a performance bias that limits their effectiveness across other languages. (3) Language-specific complexity: Languages like TS and JS feature dynamic typing, asynchronous execution, and diverse runtime behaviors, while languages like C, C++, Rust, and Go involve manual memory management, complex build systems, and intricate type systems, which add to the difficulty for issue resolving.

Performance variations across language domains. Languages in Multi-SWE-bench can be largely categorized into four domains: high-level general-purpose programming (Python, Java), web development (TS, JS), systems programming (Go, Rust), and low-level/high-performance computing (C, C++). Based on Tab. 4, the performance generally follows a hierarchy, with highlevel general-purpose languages outperforming systems programming and low-level/highperformance computing languages, while web development languages perform the worst. Java ranks second after Python, though with a noticeable gap. Go and Rust exhibit inconsistent performance across models, generally outperforming TS and JS but falling behind Java. C and C++ show even greater variability, with some LLMs (e.g., Doubao-1.5-pro and Qwen2.572B-Instruct) struggling to handle them effectively due to the challenges posed by manual memory management and complex compilation pipelines. TS and JS consistently yield the lowest resolved rates, highlighting the difficulty LLMs face in handling their event-driven, asynchronous programming paradigms.

High sensitivity to issue difficulty. As shown in Tab. 5, LLM-based agents exhibit a performance that closely aligns with human-labeled difficulty, with resolved rates significantly decreasing as the issue difficulty increases from easy to hard. However, there are several exceptions. For example, the JS language on the SWE-agent and Claude-3.5-Sonnet exhibits a zero resolved rate for easy tasks, compared to 7.62% for medium tasks. This suggests that the MSWE-agent is less effective for JS. In addition to human-assigned difficulty, other factors, such as the number of files requiring modification to resolve the issue, also influence performance. For hard-level issues, existing LLMs and agents are mostly ineffective, with resolved rates approaching zero. This phenomenon indicates the limitations of these LLMs and agents: they are primarily capable of addressing issues that human developers can resolve in under 15 minutes and are insufficient for handling more complex tasks requiring over one hour of human effort. This finding further underscores the need for RL techniques aimed at advancing agents towards more human-like intelligence, particularly for tackling real-world complex scenarios.

###### 6.1.2. Performance across Various Methods and LLMs

Variation in LLMs’ performance. From Tab. 4, significant variability is observed in the performance of different LLMs across programming languages. Specifically, LLMs such as OpenAI-o1, OpenAI-o3-mini-high, Claude-3.5-Sonnet, and Claude-3.7-Sonnet show relatively strong performance, particularly in languages like Python, Java, and C. In contrast, LLMs like Qwen2.572B-Instruct and Doubao-1.5-pro exhibit rather lower resolved rates, particularly for hard-level issues shown in Tab. 5. Furthermore, the LLMs exhibit distinct language-specific biases in

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

###### Figure 4. Issue flow from locating to resolving.

their performance. For example, models like OpenAI-o1 and OpenAI-o3-mini-high perform consistently well across languages such as Python and Java, whereas they struggle significantly with languages like C and C++. On the one hand, this performance disparity is likely attributed to the models being better suited to handle higher-level languages like Python and Java, which are more prevalent in their training data. On the other hand, the challenges with C and C++ may arise from the models’ limited exposure to low-level language features, such as memory management and pointer manipulation, which are less represented in the training data.

Performance comparison of methods. Tab. 4 and Tab. 5 also provide a comprehensive comparison of the resolved rates across three issue-resolving methods, including MagentLess, MSWE-agent, and MopenHands. Overall, MopenHands outperforms the others in most cases, achieving the highest resolved rate in five out of seven languages, while MSWE-agent wins twice and MagentLess wins once. The better performance of MopenHands and MSWE-agent can be attributed to their more flexible workflow, which is better suited to another language beyond Python compared to MagentLess. In contrast, MagentLess follows a more rigid workflow optimized for Python, and the adaptation made to create the MagentLess limits its adaptability across other languages. However, a notable exception to this general trend is observed in the performance of the models DeepSeek-R1 and Qwen2.5-72B-Instruct. For these two models, MagentLess generally provides better results than MSWE-agent for languages except C and C++. This suggests that these models may be better suited to the fixed workflow of MagentLess.

Prioritizing accurate locating over editing and reproducing. MagentLess, MSWE-agent, and MopenHands generally resolve issues through two key steps: issue location and code editing to resolve the issue. To provide a more detailed analysis of how existing LLMs and methods perform across these steps, we present the issue flow in Fig. 4. An issue is considered successfully located if the fix patches generated by the LLMs hit the files of ground truth fix patches. As shown in Fig. 4, all three methods generally fail to locate issues more often than they succeed. Accurate issue localization is fundamental to the overall success of the resolution process, serving as a prerequisite for effective code editing. Compared to MopenHands, MagentLess achieves more accurate issue localization but struggles more with the code editing step, leading to a lower overall resolved rate. This disparity is particularly evident on Claude-3.7-Sonnet. This underscores the need for a balanced method that not only prioritizes precise issue identification but also enhances the model’s ability to generate effective fixes.

Number of turns required by MSWE-agent and MopenHands. Both MSWE-agent and MopenHands resolve the issue by multi-turn interactions. Fig. 5 shows the distribution of turns for successfully resolved an issue. The absence of a corresponding box plot indicates cases where no issues were successfully resolved, such as MSWE-agent with Qwen2.5-72B-Instruct on C++. The number of interaction turns required by two methods differs across models and languages. Specifically, MopenHands resolves issues in fewer turns than MSWE-agent when using GPT-4o for Java, whereas MSWE-agent requires fewer turns when resolving Python issues. However, MopenHands exhibits a rather higher degree of dispersion in the number of interaction turns compared to MSWE-agent, which is particularly evident on OpenAI-o3-mini-high. This suggests that MopenHands’ performance is less stable across different issues, requiring a varying number of turns depending on the complexity or nature of the issue.

###### 6.1.3. Performance across Different Repositories

To understand how repository characteristics affect performance, we examine two factors: (1) repository quality, which includes the number of stars, forks, PRs, and issues, and (2) repository complexity, which includes the number of code lines and files, and the language entropy.

[Figure 49]

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

Figure 5. Number of turns required across different programming languages.

Performance across repositories of varying quality. To assess repository quality, we examine key metrics including the number of stars, forks, PRs, and issues. Fig. 6 illustrates the average resolved rate across LLMs for the three methods in relation to the number of stars and forks. Similarly, Fig. 7 shows the average resolved rate in relation to the number of issues and PRs. Both Fig. 6 and Fig. 7 exhibit a general positive correlation between #Stars and #Forks, as well as #Issues and #PRs across the majority of repositories. Furthermore, repositories with higher resolved rates tend to cluster in the upper-right quadrant of both Fig. 6 and Fig. 7, suggesting that repositories with greater activity and community engagement (i.e., higher counts of stars, forks, issues, and PRs) are typically associated with a higher resolved rate. This trend is particularly evident for the MSWE-agent and MopenHands. In contrast, MagentLess exhibits relatively low variation in resolved rates across both Fig. 6 and Fig. 7, underscoring an important observation: while a greater number of stars, forks, issues, and PRs tend to correlate with higher resolved rates, these metrics do not provide a guarantee of a repository’s issue-resolving effectiveness.

Performance across repositories with different levels of complexity. To evaluate repository complexity, we consider several key metrics: the number of lines of code (#LoC), the number of files (#Files), and language entropy. Let 𝐿 = {𝑙1, 𝑙2, · · · , 𝑙𝑛} represent the set of programming languages used in the repository, with corresponding proportions {𝑝1, 𝑝2, · · · , 𝑝𝑛}. The language entropy of the repository is then calculated as:

∑︁𝑛

𝑝𝑖 log(𝑝𝑖)

𝐻(𝐿) = −

𝑖=1

where 𝑝𝑖 denotes the proportion of the repository written in language 𝑙𝑖. The average resolved rate across nine base LLMs with different repository complexity is presented in Fig. 8.

Fig. 8 shows a consistent trend in the resolved rate across varied repository complexity: All

| | | |
|---|---|---|
| | | |
| | | |
| | | |
|10|10| |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
|10|10| |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
|10|10<br><br>| |

0.4

0.4

10

10

10

Resolvedrate(%)

0.3

10

10

10

Resolvedrate(%)

#Forks

#Forks

#Forks

0.3

#Forks

#Forks

#Forks

0.2

10³

10³

10³

0.2

10³

10³

10³

0.1

0.1

0

10²

10²

10²

10³

10³

10³

0

10²

10²

10²

10³ 10 10

10³ 10 10

10³ 10 10

#Stars #Stars #Stars

(a) MagentLess (b) MSWE-Agent (c) MopenHands

- Figure 6. Relationship between resolved rate and the number of stars and forks of a repository.

10³ 10 10

10³

10

10³ 10 10

10³

10

10³ 10 10

10³

10

0

0.1

0.2

0.3

0.4

#Issues #Issues #Issues

#PRs

#PRs

#PRs

Resolvedrate(%)

10³ 10 10

10²

10³

10

10³ 10 10

10²

10³

10

10³ 10 10

10²

10³

10

0

0.1

0.2

0.3

0.4

(a) MagentLess (b) MSWE-Agent (c) MopenHands

#Forks

#Forks

#Forks

Resolvedrate(%)

- Figure 7. Relationship between resolved rate and the number of issues and PRs of a repository.

three methods exhibit fluctuations in performance with changes in #LoC, #Files, and language entropy, generally decreasing as the repository complexity increases. For the impact of #LoC, as #LoC increases, the resolved rate tends to decrease. However, Java-based repositories, such as gson, jib, j-core, j-dbind, and dubbo, show higher resolved rates despite their larger size. This suggests that factors beyond code size, such as lower language entropy, modularity, welldocumented code, and adherence to standardized practices, play a significant role in improving performance. For example, the gson repository demonstrates nearly-zero language entropy in

- Fig. 8. Similarly, the impact of #Files follows a trend similar to #LoC. The impact of language entropy shows a clearer trend than that of #LoC and #Files: repositories with lower entropy typically achieve higher resolved rates. This indicates that code simplicity and consistency play a crucial role in improving issue-resolving effectiveness on a repository.

###### 6.2. Influencing Factors of Performance

In this subsection, we investigate the factors influencing issue resolving performance, focusing on three key factors: (1) issue type, examining how different types of issues impact resolving effectiveness; (2) issue description characteristics, evaluating the role of description length in resolving issues; and (3) fix patch characteristics, analyzing how the length of fix patches and the number of involved files influence the resolving performance.

70

MagentLess

MSWE-agent MopenHands #LoC of repositories

60

50

Resolverate(%)

- 101

- 102

40

#LoC

30

20

10

0

gh-rdme-statsfd bytesexpressdayjsaxiosj-dfmt-xmlbatdarkreaderfmtrayonserderipgrepjqcpp-httplibgsonCatch2logstashtracingclapjibponycmockitoj-coresveltego-zerozstdjsoncoretokioinsomniaclij-dbindsimdjsongrpc-gonushelldubbofastjson2material-ui

70

MagentLess

MSWE-agent MopenHands #Files of repositories

- 102

- 103

- 104

60

50

Resolverate(%)

40

#Files

30

20

10

0

cpp-httplibfdfmtgh-rdme-statsbytes jqripgrepbatexpressaxiosdarkreaderserdej-dfmt-xmlrayontracinggsonzstdponycclapdayjsj-coreCatch2simdjsonjsoninsomniacorelogstashjibtokiogo-zerocligrpc-gomockitoj-dbindnushellsveltedubbofastjson2material-ui

70

MagentLess

100

MSWE-agent MopenHands Language entropy of repositories

60

50

Languageentropy

Resolverate(%)

40

10 1

30

20

10

0

10 2

gsondayjsserdetokioexpressgh-rdme-statsrayontracingfastjson2jibclibytesdubbogrpc-gomockitoj-corego-zeroj-dbindnushellsimdjsoncpp-httplibj-dfmt-xmlbatjsonfmtcoreripgrepfd axiosCatch2darkreaderclapmaterial-uiinsomniasveltezstdlogstashjqponyc

- Figure 8. Relation between resolved rate and the repository complexity on Multi-SWE-bench.

###### 6.2.1. Issue Type

- Tab. 6 lists the performance of the three methods on Multi-SWE-bench across different issue types and languages. Through a meticulous manual analysis of the annotation results in Sec. 3.1.5, we categorized all instances in Multi-SWE-bench into three issue types: bug fix (Bug Fix), new feature (New Feat.), and feature optimization (Feat. Opt.). We observe a consistent performance hierarchy across all methods and languages: bug fix issues are resolved with the highest success rates, followed by new features, with feature optimization being the most challenging. For instance, MSWE-agent achieves 17.97% on Java bug fixes but drops to 3.91% and 1.56% for new features and optimizations, respectively. MagentLess and MopenHands show a similar trend in all languages. These results highlight a fundamental limitation of current agent-based methods:

they are more effective at localized, symptom-driven repairs, but struggle with semantically demanding tasks such as implementing new functionality or refining existing behavior. The latter requires deeper intent understanding, multi-component reasoning, and cross-file context aggregation capabilities that remain underdeveloped in current LLM-based agents.

- Table 6. Resolved rate(%) on Multi-SWE-bench across different issue types (Claude-3.7-Sonnet).

|Languages<br><br>|MagentLess|MSWE-agent<br><br>|MopenHands|
|---|---|---|---|
| |Bug Fix New Feat. Feat. Opt.<br><br>|Bug Fix New Feat. Feat. Opt.<br><br>|Bug Fix New Feat. Feat. Opt.|
|Java TypeScript JavaScript Go Rust C C++|10.94 2.34 0.78<br><br>2.68 0.45 0.45<br><br>1.97 0.00 0.00<br><br>3.74 0.93 1.17 4.60 0.42 0.42 6.25 0.00 0.00<br><br>2.33 0.78 0.00<br><br><br><br><br>|17.97 3.91 1.56 9.38 1.34 0.45<br><br>4.21 0.56 0.00 3.27 0.70 1.40<br><br>5.44 1.26 0.00 7.81 0.78 0.00 7.75 3.1 0.78<br><br><br>|17.97 3.12 0.78 1.79 0.00 0.45<br><br>3.65 1.12 0.28<br>4.44 2.10 0.93 12.97 2.93 0.00<br><br><br>7.81 0.78 0.00 10.85 3.10 0.78|

###### 6.2.2. Characteristics of Issue Description

We aim to examine the impact of issue description length on issue-resolving performance.

- Fig. 9 illustrates the distribution of issue lengths (in tokens) in Multi-SWE-bench, which follows a power law, with the majority of issues being under 1,000 tokens. To explore the effect of description length, the issues are categorized into 5 intervals: <100, 100-400, 400-700, 700-1000, and >1000 tokens, as shown in Fig. 10. The absence of a corresponding bars indicates cases where no issues are successfully resolved.

60

#Issues

40

20

0

101 102 103 104

Description length (#tokens)

Figure 9. Histogram of issue description length (#tokens).

As shown in Fig. 10, there is no consistent relationship between issue description length and resolved rate. For example, in Python, issues with longer descriptions tend to have lower resolved rates, whereas in Go, longer descriptions are associated with higher rates. This discrepancy arises from two potential types of long issue descriptions: (1) detailed issues with precise issue position indications and resolving steps, and (2) complex issues that require extended descriptions to explain. These two possibilities have distinct impacts on the difficulty of resolving an issue, influencing the resolved rate in different ways. As for the performance among methods, compared with MSWE-agent and MopenHands, MagentLess generally performs better with longer, more detailed descriptions on average of the nine LLMs, especially for languages like Python, Java, and TS.

10.0

| | | | | |
|---|---|---|---|---|
| | | | | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |

| |
|---|

25.0

5.0

40.0

8.0

Resolverate(%)

20.0

4.0

30.0

6.0

15.0

3.0

20.0

4.0

10.0

2.0

10.0

2.0

5.0

1.0

0.0

0.0

0.0

0.0

<100100-400400-700700-1000>1000

<100100-400400-700700-1000>1000

<100100-400400-700700-1000>1000

<100100-400400-700700-1000>1000

#Tokens of issue description

#Tokens of issue description

#Tokens of issue description

#Tokens of issue description

(a) Python

(b) Java

(c) TS

(d) JS

5.0

| | | | | |
|---|---|---|---|---|
| | | | | |

| | |
|---|---|

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | |
|---|---|
| | |

8.0

8.0

12.5

4.0

Resolverate(%)

6.0

10.0

6.0

3.0

7.5

4.0

4.0

2.0

5.0

2.0

2.0

1.0

2.5

0.0

0.0

0.0

0.0

<100100-400400-700700-1000>1000

<100100-400400-700700-1000>1000

<100100-400400-700700-1000>1000

<100100-400400-700700-1000>1000

#Tokens of issue description

#Tokens of issue description

#Tokens of issue description

#Tokens of issue description

(e) Go

(f) Rust

(g) C

(h) C++

Figure 10. Influence of issue description length on resolved rate.

###### 6.2.3. Characteristics of Fix Patches

In this subsection, we investigate the impact of the ground-truth fix patches on the resolved rate, focusing on two key factors: (1) Fix patch length: We analyze how the length of fix patches affects performance, noticing that longer patches require more complex reasoning capabilities from LLMs. The fix patches are categorized into five intervals based on the length distribution shown in Fig. 11: <200, 200-600, 600-1000, 1000-1400, and >1400 tokens. (2) Number of files modified by fix patches: We examine how the cross-file nature of the fix patches influences performance, with more files requiring enhanced cross-file handling capabilities. The number of modified files is divided into four categories: 1, 1-5, 5-10, and >10, with the distribution shown in Fig.12. The absence of corresponding bars indicates cases where no issues are successfully resolved.

250

200

#Issues

150

100

50

0

102 103 104 105

Fix patches length (#tokens)

Figure 11. Histogram of fix patches length (#tokens).

1000

800

600

#Issues

400

200

0

100 101 102

#Files

Figure 12. Histogram of the number of files modified by fix patches.

Performance drops as fix patch length increases. As shown in Fig. 13, the length of fix patches significantly impacts the resolved rate, with shorter patches generally leading to higher success rates. Specifically, in the majority of cases, issues with descriptions >600 tokens exhibit a resolved rate approximately 50% lower than that of issues with descriptions <200 tokens. For most programming languages, the resolved rate for shorter fix patches is notably higher,

8.0

| |
|---|

| | |
|---|---|
| | |
| | |

| | | |
|---|---|---|
| | | |

| |
|---|
| |
| |

15.0

20.0

50.0

Resolverate(%)

6.0

40.0

15.0

10.0

30.0

4.0

10.0

20.0

5.0

2.0

5.0

10.0

0.0

0.0

0.0

0.0

<200200-600600-10001000-1400>1400

<200200-600600-10001000-1400>1400

<200200-600600-10001000-1400>1400

<200200-600600-10001000-1400>1400

#Tokens of fix patches

#Tokens of fix patches

#Tokens of fix patches

#Tokens of fix patches

(a) Python

(b) Java

(c) TS

(d) JS

| | |
|---|---|
| | |
| | |

| |
|---|
| |
| |

| |
|---|
| |

| | |
|---|---|

12.5

25.0

15.0

10.0

Resolverate(%)

10.0

20.0

8.0

10.0

7.5

15.0

6.0

5.0

10.0

4.0

5.0

2.5

5.0

2.0

0.0

0.0

0.0

0.0

<200200-600600-10001000-1400>1400

<200200-600600-10001000-1400>1400

<200200-600600-10001000-1400>1400

<200200-600600-10001000-1400>1400

#Tokens of fix patches

#Tokens of fix patches

#Tokens of fix patches

#Tokens of fix patches

(e) Go

(f) Rust

(g) C

(h) C++

Figure 13. Influence of fix patch length on resolved rate.

especially for MagentLess, which shows a peak in this range for languages like Python, Java, and C. This suggests that shorter patches are easier to handle, as they require less reasoning and simpler edits. For all three methods, the resolved rate for very long fix patches (>1000 tokens) drops significantly, even reaching zero for Java. This indicates that long patches, which likely require handling a larger scope of modifications, present greater challenges, especially for methods that may not be optimized for such complex tasks.

Cross-file fix patches lead to reduced effectiveness. Fig. 14 illustrates the relationship between the number of files modified by fix patches and the resolved rate. Consistent with the observation in Fig. 13, resolved rate drops significantly as the number of modified files increases across all three methods. This trend highlights the potential challenge of understanding and resolving issues that require changes across multiple files, which may demand more intricate handling or coordination between different parts of the repository. For issues resolved by modifications in a single file, MagentLess outperforms MSWE-agent and MopenHands in five out of seven programming languages. This suggests that MagentLess is more effective at resolving issues within the scope of a single file.

###### 6.3. Case Study

In this subsection, we analyze representative cases that highlight the strengths of agents, common failure patterns, and language-specific challenges, providing insights for future directions.

###### 6.3.1 Language-General Case

- • MSWE-agent and MopenHands often failed by exhausting the 50-round interaction limit, sometimes without even triggering the submit action, as seen in cases like axios__axios5919.traj, clap-rs__clap-5520.traj, and cli__cli-513.traj. Future work may explore strategies that enable agents to solve more complex tasks within a limited number of interaction rounds.
- • A significant number of failures across all three agent methods were due to incorrect fault localization, which led to an inability to identify and modify the relevant code, as seen in

12.5

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |
| |
| |

6.0

5.0

40.0

10.0

Resolverate(%)

4.0

30.0

4.0

7.5

3.0

20.0

5.0

2.0

2.0

10.0

2.5

1.0

0.0

0.0

0.0

0.0

1 1-5 5-10 >10

1 1-5 5-10 >10

1 1-5 5-10 >10

1 1-5 5-10 >10

#Files modified

#Files modified

#Files modified

#Files modified

(a) Python

(b) Java

(c) TS

(d) JS

| | |
|---|---|
| | |

| |
|---|
| |

| |
|---|
| |

| | |
|---|---|
| | |
| | |

8.0

12.5

- 6.0

10.0

Resolverate(%)

10.0

6.0

7.5

4.0

7.5

4.0

5.0

5.0

2.0

2.0

2.5

2.5

0.0

0.0

0.0

0.0

1 1-5 5-10 >10

1 1-5 5-10 >10

1 1-5 5-10 >10

1 1-5 5-10 >10

#Files modified

#Files modified

#Files modified

#Files modified

(e) Go

(f) Rust

(g) C

(h) C++

Figure 14. Influence of the number of files modified by fix patches on resolved rate.

cases such as elastic__logstash-14898.traj, alibaba__fastjson2-2285.traj, fasterxml__jacksondatabind-3560.traj, and apache__dubbo-7041.traj. This highlights the centrality of accurate fault localization and points to the potential of integrating software engineering techniques like SBFL [Abreu et al., 2007, Jones et al., 2002] into future agent designs.

- • In cases such as astropy__astropy-12907.traj and django__django-11299.traj, the model generated multiple valid actions in a single turn, but the hardcoded agent framework executed only the last, resulting in premature submission. This reveals a structural bottleneck in current agent design, where rigid control logic overrides model intent. It calls for a shift toward lightweight, model-centric agents with full decision autonomy delegated to the LLM.
- • Bug reproduction plays a critical role in successful repair. In cases such as nlohmann__json4537.traj, fmtlib__fmt-3248.traj, fasterxml__jackson-core-1142.traj, and google__gson-1093.traj, the model successfully reproduced the issue before producing an effective fix. In contrast, failure to reproduce often resulted in unresolved cases, as seen in catchorg__Catch2-1609.traj. However, reproduction is not always a prerequisite for success. Claude-3.5-Sonnet and Claude-3.7-Sonnet occasionally bypass reproduction and edit the code directly—yet still resolve the issue successfully, as in nlohmann__json-3601.traj, fmtlib__fmt-3729.traj, and googlecontainertools__jib-4035.traj. These cases suggest that agents should selectively invoke reproduction based on factors such as error traceability, edit confidence, and execution cost.

###### 6.3.2 Language-Specific Case

- • For certain TypeScript projects, the length of the extracted repository structure often exceeds the model’s maximum context length, preventing MagentLess from performing fault localization (e.g., mui__material-ui-25852.traj and mui__material-ui-37850.traj). This reveals the limited generalizability of fixed workflows like MagentLess when confronted with structurally irregular and language-specific scenarios, indicating significant room for improvement in both robustness and adaptability.
- • Tree-sitter fails to reliably extract code structures in JavaScript repositories that use loosely bound syntax such as arrow functions, preventing MagentLess from constructing contextual windows around candidate edits (e.g., iamkun__dayjs-2532.traj and iamkun__dayjs-2399.traj).

- Table 7. Average token consumption on Multi-SWE-bench. In. represents the average number of input tokens (in thousands), and Out. is the average number of output tokens (in thousands).

Python Java TS JS Go Rust C C++

Models

In. Out. In. Out. In. Out. In. Out. In. Out. In. Out. In. Out. In. Out.

MagentLess GPT-4o 36.15 4.20 52.10 2.74 241.18 2.01 29.48 2.53 25.14 2.72 48.23 2.71 50.26 2.64 76.38 2.44 OpenAI-o1 34.43 3.76 50.18 1.92 240.47 1.21 36.53 1.26 24.51 1.70 58.59 1.49 48.08 1.57 119.64 1.31 OpenAI-o3-mini-high 31.38 4.50 79.48 2.36 245.39 1.54 38.28 1.55 31.58 1.67 68.80 2.05 73.48 1.99 200.29 1.91 Claude-3.5-Sonnet 39.13 5.38 48.42 2.67 239.93 1.86 28.46 2.39 22.80 2.79 51.25 2.66 49.13 2.55 96.85 2.49 Claude-3.7-Sonnet 27.99 6.54 63.97 3.16 248.36 2.08 26.66 3.17 22.79 3.63 81.15 3.33 50.52 3.19 129.34 2.91 DeepSeek-V3 39.97 4.26 42.35 2.70 244.32 1.92 26.44 2.51 22.78 2.65 83.04 2.47 92.53 2.38 189.08 2.11 DeepSeek-R1 31.35 2.80 70.35 1.76 249.02 1.10 28.23 1.30 21.69 1.79 88.73 1.52 100.99 1.39 177.66 1.41 Qwen2.5-72B-Instruct 28.60 3.46 62.95 2.52 243.98 1.65 26.11 2.14 24.67 3.36 50.63 2.89 55.93 2.78 150.44 2.19 Doubao-1.5-pro 42.75 2.91 116.09 1.36 249.51 1.55 36.37 2.07 29.38 3.15 124.67 1.62 121.94 1.52 216.21 0.76

MSWE-agent GPT-4o 166.91 3.08 51.05 4.54 46.39 4.63 32.01 4.36 36.73 4.71 43.79 4.71 39.47 4.57 55.49 4.96 OpenAI-o1 243.44 1.64 33.36 2.99 30.05 3.56 25.70 3.19 37.71 3.49 39.51 3.71 34.05 3.17 29.24 3.28 OpenAI-o3-mini-high 240.23 1.82 26.37 3.64 18.27 3.39 21.33 3.99 26.46 3.41 32.84 4.03 23.24 3.78 32.39 4.90 Claude-3.5-Sonnet 33.30 5.55 32.09 3.89 21.51 3.10 23.94 3.66 21.06 3.06 35.47 4.03 31.16 3.44 38.22 4.32 Claude-3.7-Sonnet 31.86 4.46 38.96 4.79 32.08 4.92 32.16 4.60 33.79 4.56 40.59 4.56 38.41 4.34 36.96 4.67 DeepSeek-V3 12.63 22.83 35.08 4.14 15.73 2.15 19.78 3.23 15.34 2.43 33.98 5.47 16.26 2.07 31.28 4.18 DeepSeek-R1 11.76 2.65 17.51 2.69 9.91 1.66 9.36 2.43 10.47 1.85 13.98 2.86 11.34 2.44 14.64 3.06 Qwen2.5-72B-Instruct 164.42 6.69 53.43 9.26 39.58 7.82 35.21 6.45 22.53 8.38 36.49 7.93 28.90 5.76 67.29 11.69 Doubao-1.5-pro 72.58 1.30 37.75 3.73 19.18 2.46 32.90 3.68 25.39 3.91 38.09 4.04 29.03 3.65 32.67 3.59

MopenHands GPT-4o 25.35 1.24 22.01 1.32 35.76 1.60 35.51 1.50 23.96 1.52 40.40 1.45 34.80 2.08 34.61 1.66 OpenAI-o1 19.27 1.20 18.69 1.27 27.28 2.14 30.96 2.07 21.09 1.56 28.90 1.55 27.18 1.67 21.55 1.57 OpenAI-o3-mini-high 21.52 5.18 22.82 3.88 30.70 4.32 36.57 4.06 25.44 4.14 30.64 3.15 23.98 3.26 23.76 4.26 Claude-3.5-Sonnet 32.35 7.69 31.97 5.35 35.88 6.43 38.91 6.14 27.31 7.26 55.51 6.23 55.79 5.66 35.85 6.74 Claude-3.7-Sonnet 26.04 7.84 28.43 7.86 31.06 7.31 38.06 7.46 30.05 7.86 48.30 7.00 35.25 6.30 33.14 7.76 DeepSeek-V3 18.97 5.16 26.35 3.65 26.60 3.69 29.08 4.43 15.42 3.31 32.90 5.05 21.77 3.53 30.67 5.36 DeepSeek-R1 11.25 5.13 17.15 5.04 12.71 4.33 17.85 5.29 12.65 5.14 17.58 6.95 24.16 6.11 17.38 7.62 Qwen2.5-72B-Instruct 27.28 10.38 33.26 11.80 36.86 9.12 28.84 9.07 21.17 11.35 37.14 10.99 35.02 9.69 35.34 10.88 Doubao-1.5-pro 23.16 3.95 24.15 3.35 18.34 1.66 23.75 2.76 18.21 3.78 27.40 2.07 26.54 2.82 26.07 3.44

This exposes a structural brittleness in syntax-driven workflows when applied to syntactically permissive languages, motivating future extensions of MagentLess toward greater tolerance to parsing failure and language-specific irregularities.

• In some JavaScript projects, agents sometimes invoke pnpm to launch development servers as part of the repair routine. However, current agent frameworks lack support for managing long-lived, interactive processes, often resulting in premature termination or container crashes (e.g., sveltejs__svelte-12460.traj and sveltejs__svelte-10077.traj). Future agents should support persistent shell sessions and interactive service control, as enabled by frameworks like SWEReX [SWE-agent, 2025].

###### 6.4. Resource Consumption

In this subsection, we analyze the resource consumption across different languages, focusing on two key metrics: (1) the average token consumption and (2) the average cost per issue.

Average token consumption per issue. Tab. 7 compares the average token consumption for various languages using the GPT-4o tokenizer. Overall, token consumption varies between methods and languages. Among languages, TS exhibits the highest token consumption in MagentLess, whereas Python is the most token-intensive language in MSWE-agent. Notably, Go demonstrates relatively low token consumption in both input and output, likely due to its minimalistic syntax and clear conventions, which contribute to its compact representation and reduced token overhead. Additionally, in MSWE-agent for Python, we observe increased token usage on LLMs, including GPT-4o, OpenAI-o1, OpenAI-o3-mini-high, and Qwen2.5-72BInstruct. This is because we maintain the original SWE-agent implementation for Python, which does not incorporate the over-length truncation mechanism applied to other languages.

Table 8. Average cost ($) per issue of different models and methods on Multi-SWE-bench.

Models Python Java TS JS Go Rust C C++

MagentLess GPT-4o 0.1324 0.1576 0.6230 0.0990 0.0900 0.1476 0.1520 0.2153 OpenAI-o1 0.7417 0.8680 3.6795 0.6233 0.4698 0.9682 0.8153 1.8734 OpenAI-o3-mini-high 0.0543 0.0978 0.2767 0.0489 0.0421 0.0847 0.0896 0.2287 Claude-3.5-Sonnet 0.1981 0.1853 0.7478 0.1213 0.1102 0.1937 0.1856 0.3280 Claude-3.7-Sonnet 0.1821 0.2393 0.7763 0.1275 0.1229 0.2933 0.1994 0.4317 DeepSeek-V3 0.0075 0.0059 0.0192 0.0046 0.0045 0.0085 0.0091 0.0156 DeepSeek-R1 0.0105 0.0137 0.0373 0.0068 0.0070 0.0158 0.0172 0.0280 Qwen2.5-72B-Instruct 0.0051 0.0092 0.0324 0.0042 0.0046 0.0077 0.0084 0.0204 Doubao-1.5-pro 0.0055 0.0132 0.0279 0.0046 0.0041 0.0142 0.0138 0.0240

MSWE-agent GPT-4o 0.4480 0.1731 0.1623 0.1236 0.1390 0.1565 0.1444 0.1883 OpenAI-o1 3.7499 0.6797 0.6644 0.5772 0.7749 0.8151 0.7010 0.6353 OpenAI-o3-mini-high 0.2722 0.0450 0.0350 0.0410 0.0441 0.0538 0.0422 0.0572 Claude-3.5-Sonnet 0.1831 0.1546 0.1110 0.1266 0.1091 0.1669 0.1451 0.1794 Claude-3.7-Sonnet 0.1626 0.1887 0.1700 0.1654 0.1698 0.1901 0.1803 0.1810 DeepSeek-V3 0.0260 0.0070 0.0035 0.0049 0.0037 0.0084 0.0034 0.0068 DeepSeek-R1 0.0075 0.0083 0.0050 0.0066 0.0055 0.0082 0.0069 0.0088 Qwen2.5-72B-Instruct 0.0241 0.0106 0.0083 0.0072 0.0063 0.0079 0.0061 0.0134 Doubao-1.5-pro 0.0083 0.0052 0.0028 0.0046 0.0039 0.0053 0.0042 0.0046

MopenHands GPT-4o 0.0758 0.0682 0.1054 0.1038 0.0751 0.1155 0.1078 0.1031 OpenAI-o1 0.3608 0.3564 0.5374 0.5885 0.4099 0.5262 0.5081 0.4171 OpenAI-o3-mini-high 0.0465 0.0422 0.0528 0.0581 0.0462 0.0476 0.0407 0.0449 Claude-3.5-Sonnet 0.2124 0.1761 0.2041 0.2089 0.1908 0.2601 0.2523 0.2086 Claude-3.7-Sonnet 0.1957 0.2032 0.2028 0.2261 0.2080 0.2500 0.2002 0.2158 DeepSeek-V3 0.0070 0.0059 0.0059 0.0069 0.0047 0.0079 0.0054 0.0080 DeepSeek-R1 0.0128 0.0134 0.0113 0.0141 0.0130 0.0177 0.0168 0.0191 Qwen2.5-72B-Instruct 0.0077 0.0090 0.0084 0.0074 0.0073 0.0092 0.0084 0.0089 Doubao-1.5-pro 0.0037 0.0036 0.0025 0.0034 0.0031 0.0036 0.0037 0.0038

Average cost per issue. Tab. 8 presents the average cost ($) per issue on Multi-SWE-bench. Notably, DeepSeek-V3, DeepSeek-R1, and Qwen2.5-72B-Instruct achieve the lowest cost per resolved issue, staying below $0.03, benefiting from their cost-efficient pricing (below $0.14 per million input tokens). In contrast, OpenAI-o1 is the most expensive model, due to its high token price ($15 per million input tokens). Overall, MagentLess exhibits lower costs compared to MSWE-agent by virtue of its fixed workflow. Conversely, the workflows of both MSWEagent and MopenHands are determined by LLMs, requiring more interaction turns with the environment, which results in higher costs.

###### 6.5. Troubleshooting

During the construction of Multi-SWE-bench and Multi-SWE-RL, we encountered a range of practical and non-obvious challenges. We document the key issues below to facilitate reproducibility and guide future community contributions:

- • Test log inconsistency. The number of test cases differs between Test.log and Fix.log, as fix.patch may optimize control flow, eliminate redundant coverage, or merge test paths, which is commonly observed in repositories such as preactjs/preact.
- • Pre-fix build failures. Certain repositories fail to compile or execute tests before applying fix.patch, due to newly introduced symbols (e.g., functions or variables) in test.patch that are

- undefined without the fix.
- • Binary artifacts in C&C++. Agent runs may generate compiled binaries (e.g., ".o", ".bin") that block "git apply". We currently strip these via hard-coded filtering, though more robust handling is needed.
- • Evaluation nondeterminism. Java and C tests occasionally exhibit unstable behavior due to excessive thread concurrency, leading to inconsistent run.log outcomes. We mitigate this by reducing parallelism during evaluation.
- • Name casing mismatches. Some test names appear in lowercase in test.log but in uppercase in fix.log. We normalize all test names to lowercase to ensure alignment.
- • Unstable test identifiers. Some test names are dynamically generated with timestamps or random suffixes, making them non-deterministic. Such instances are excluded.
- • Log interleaving in Java. In some Java projects, test outputs from concurrent threads are interleaved without delimiters, making rule-based log parsing infeasible. This is likely due to unsynchronized multi-threaded logging.

#### 7. Conclusions and Future Works

We introduce Multi-SWE-bench, a multilingual benchmark for issue resolving, consisting of 1,632 human-validated GitHub instances on 7 widely used programming languages. Based on this benchmark, we evaluate nine popular models using three agent methods and conduct a thorough analysis of the results. Additionally, we establish the Multi-SWE-RL open-source community, aimed at building large-scale RL training datasets for issue-resolving tasks. To catalyze community involvement, we release 4,723 validated instances along with the complete data construction pipeline, encouraging the open-source community to continuously contribute and expand the dataset. Looking ahead, we plan to scale Multi-SWE-bench and Multi-SWE-RL to more instances, languages, and modalities. Beyond issue resolving, we would like to incorporate a broader range of software engineering tasks into our benchmark and RL community, such as end-to-end project generation [Zan et al., 2024, Starace et al., 2025], runtime environment setup [Zala et al., 2024, Hu et al., 2025, Eliseeva et al., 2025], bug reproduction [Wang et al., 2024a, 2025] and localization [Hossain et al., 2024], and software testing and maintenance [Lemner et al., 2024, Peng et al., 2025]. Overall, we envision our efforts as a step toward establishing a scalable and sustainable data-centric infrastructure that empowers future research.

#### Contributions

Daoguang Zan★

###### Project Lead

###### Core Contributors

Code Development Zhirong Huang★ Hanwu Chen Daoguang Zan

Paper Writing Wei Liu Daoguang Zan

Data Construction Zhirong Huang Hanwu Chen Linhao Zhang Daoguang Zan

Agent Integration and Evaluation Shulin Xin Lu Chen Qi Liu Xiaojian Zhong Aoyan Li Daoguang Zan

Result Analysis and Statistics Wei Liu Aoyan Li Shulin Xin Daoguang Zan

###### Other Contributors

Leaderboard Refinement Siyao Liu Linhao Zhang

Outsourced Hiring Yongsheng Xiao

Discussion and Support Liangqiang Chen Yuyu Zhang Jing Su Tianyu Liu Rui Long

###### Team Management

Kai Shen Liang Xiang

Names marked with ★ denote equal contribution.

#### Acknowledgments

We gratefully acknowledge all members of the Seed-Foundation-Code team. We thank Dong Chen, Ailun Yu, Shaoxin Lin, Yifan Shi, Bo Shen, Guangtai Liang, and Qianxiang Wang, former colleagues of Daoguang at Huawei Cloud, for their early discussion. We thank Changxin Pu and Xiang Gao at Bytedance for their support with data annotation. We thank Professors Jinxi Li from Shenzhen Technology University; Wei Zhang, Haiyan Zhao, and Zhi Jin from Peking University; and Xudong Lu and Lizhen Cui from Shandong University. We thank Shulin, Wei, Aoyan, Lu, and Qi for their dedication in the final sprint before the deadline. We are especially grateful to Zhirong, Hanwu, Linhao, and Daoguang for their countless late nights devoted to developing the dataset, without which this work would not have been possible.

#### References

- R. Abreu, P. Zoeteweij, and A. J. Van Gemund. On the accuracy of spectrum-based fault localization. In Testing: Academic and industrial conference practice and research techniquesMUTATION (TAICPART-MUTATION 2007), pages 89–98. IEEE, 2007.

M. Allamanis and C. Sutton. Mining source code repositories at massive scale using language modeling. In 2013 10th working conference on mining software repositories (MSR), pages 207–216. IEEE, 2013.

B. Athiwaratkun, S. K. Gouda, Z. Wang, X. Li, Y. Tian, M. Tan, W. U. Ahmad, S. Wang, Q. Sun, M. Shang, et al. Multi-lingual evaluation of code generation models. arXiv preprint arXiv:2210.14868, 2022.

- B. Athiwaratkun, S. K. Gouda, Z. Wang, X. Li, Y. Tian, M. Tan, W. U. Ahmad, S. Wang, Q. Sun,

- M. Shang, et al. Multi-lingual evaluation of code generation models. In ICLR, 2023.

augment code. Augment swe-bench verified agent, 2025. URL https://www.augmentcode. com/blog/1-open-source-agent-on-swe-bench-verified-by-combining-claud e-3-7-and-o1. 2025-03-31.

J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry,

- Q. Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

- M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda,
- N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin,

- B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet,

- F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage,

- M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating large language models trained on code, 2021a. URL https: //arxiv.org/abs/2107.03374.

- M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda,
- N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021b.

- Y. Ding, Z. Wang, W. Ahmad, H. Ding, M. Tan, N. Jain, M. K. Ramanathan, R. Nallapati,

- P. Bhatia, D. Roth, et al. Crosscodeeval: A diverse and multilingual benchmark for cross-file code completion. Advances in Neural Information Processing Systems, 36, 2024.

A. Eliseeva, A. Kovrigin, I. Kholkin, E. Bogomolov, and Y. Zharov. Envbench: A benchmark for

automated environment setup, 2025. URL https://arxiv.org/abs/2503.14443.

D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- S. B. Hossain, N. Jiang, Q. Zhou, X. Li, W.-H. Chiang, Y. Lyu, H. Nguyen, and O. Tripp. A deep dive into large language models for automated bug localization and repair. Proceedings of the ACM on Software Engineering, 1(FSE):1471–1493, 2024.

- R. Hu, C. Peng, X. Wang, and C. Gao. An llm-based agent for reliable docker environment configuration. arXiv preprint arXiv:2502.13681, 2025.
- S. Iyer, I. Konstas, A. Cheung, and L. Zettlemoyer. Mapping language to code in programmatic context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1643–1652, 2018.

A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

H. Jelodar, M. Meymani, and R. Razavi-Far. Large language models (llms) for source code analysis: applications, models and datasets, 2025. URL https://arxiv.org/abs/2503.1 7502.

J. Jiang, F. Wang, J. Shen, S. Kim, and S. Kim. A survey on large language models for code

generation, 2024. URL https://arxiv.org/abs/2406.00515.

- C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

J. A. Jones, M. J. Harrold, and J. Stasko. Visualization of test information to assist fault localization. In Proceedings of the 24th international conference on Software engineering, pages 467–477, 2002.

L. Lemner, L. Wahlgren, G. Gay, N. Mohammadiha, J. Liu, and J. Wennerberg. Exploring the integration of large language models in industrial test maintenance processes. arXiv preprint arXiv:2409.06416, 2024.

- T. Liu, C. Xu, and J. McAuley. Repobench: Benchmarking repository-level code auto-completion systems, 2024a. URL https://arxiv.org/abs/2306.03091.

- W. Liu, A. Yu, D. Zan, B. Shen, W. Zhang, H. Zhao, Z. Jin, and Q. Wang. GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model, 2024b. URL https://arxiv.org/abs/2406.07003.

- S. Miserendino, M. Wang, T. Patwardhan, and J. Heidecke. Swe-lancer: Can frontier llms earn $1 million from real-world freelance software engineering? arXiv preprint arXiv:2502.12115, 2025.

N. Mündler, M. Müller, J. He, and M. Vechev. Swt-bench: Testing and validating real-world bug-fixes with code agents. Advances in Neural Information Processing Systems, 37:81857–81887, 2024.

OpenAI. Openai o3-mini, 2025. URL https://openai.com/index/openai-o3-mini/.

Accessed: 2025-01-31.

- Y. Ouyang, J. Yang, and L. Zhang. Benchmarking automated program repair: An extensive study on both real-world and artificial bugs. In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, pages 440–452, 2024.

- X. Peng, C. Wang, M. Liu, Y. Lou, and Y. Wu. Code digital twin: Empowering llms with tacit knowledge for complex software maintenance. arXiv preprint arXiv:2503.07967, 2025.

V. Raychev, P. Bielik, and M. Vechev. Probabilistic model for code with decision trees. ACM SIGPLAN Notices, 51(10):731–747, 2016.

- N. Saavedra, A. Silva, and M. Monperrus. Gitbug-actions: Building reproducible bug-fix benchmarks with github actions. In Proceedings of the 2024 IEEE/ACM 46th International Conference on Software Engineering: Companion Proceedings, pages 1–5, 2024.

- G. Starace, O. Jaffe, D. Sherburn, J. Aung, C. J. Shern, L. Maksin, R. Dias, E. Mays, B. Kinsella,

- W. Thompson, J. Heidecke, M. Glaese, T. Patwardhan, and OpenAI. Paperbench: Evaluating ai’s ability to replicate ai research. 2025. URL https://openai.com/index/paperbench.

T. Sun, Y. Yang, X. Cheng, J. Yang, Y. Huo, Z. Ye, R. Yang, X. Guan, W. Zhang, H. Ji, et al. Repofixeval: A repository-level program repair benchmark from issue discovering to bug fixing.

SWE-agent. Swe-agent remote execution framework, 2025. URL https://github.com/SWE

-agent/SWE-ReX.

D. Wang, Z. Zhang, S. Feng, W. G. Halfond, and T. Yu. An empirical study on leveraging images in automated bug report reproduction. arXiv preprint arXiv:2502.15099, 2025.

- X. Wang, P. Gao, X. Meng, C. Peng, R. Hu, Y. Lin, and C. Gao. Aegis: An agent-based framework for general bug reproduction from issue descriptions. arXiv preprint arXiv:2411.18015, 2024a.

X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, H. H. Tran, F. Li, R. Ma, M. Zheng, B. Qian, Y. Shao, N. Muennighoff, Y. Zhang, B. Hui, J. Lin, R. Brennan, H. Peng, H. Ji, and G. Neubig. OpenHands: An Open Platform for AI Software Developers as Generalist Agents, 2024b. URL https://arxiv.org/abs/2407.16741.

- Z. Wang, S. Zhou, D. Fried, and G. Neubig. Execution-based evaluation for open-domain code generation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1271–1290, 2023.

- C. S. Xia, Y. Deng, S. Dunn, and L. Zhang. Agentless: Demystifying llm-based software engineering agents. arXiv preprint arXiv:2407.01489, 2024.

J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press. SWEagent: Agent-computer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793, 2024.

J. Yang, C. E. Jimenez, A. L. Zhang, K. Lieret, J. Yang, X. Wu, O. Press, N. Muennighoff,

- G. Synnaeve, K. R. Narasimhan, D. Yang, S. I. Wang, and O. Press. SWE-bench multimodal: Do ai systems generalize to visual software domains? In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=riTiq3i21b.
- H. Yu, B. Shen, D. Ran, J. Zhang, Q. Zhang, Y. Ma, G. Liang, Y. Li, Q. Wang, and T. Xie. Codereval: A benchmark of pragmatic code generation with generative pre-trained models. In Proceedings of the 46th IEEE/ACM International Conference on Software Engineering, pages 1–12, 2024.

A. Zala, J. Cho, H. Lin, J. Yoon, and M. Bansal. Envgen: Generating and adapting environments via llms for training embodied agents, 2024. URL https://arxiv.org/abs/2403.12014.

- D. Zan, B. Chen, D. Yang, Z. Lin, M. Kim, B. Guan, Y. Wang, W. Chen, and J. Lou. CERT: continual pre-training on sketches for library-oriented code generation. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pages 2369–2375, 2022.

D. Zan, B. Chen, F. Zhang, D. Lu, B. Wu, B. Guan, W. Yongji, and J.-G. Lou. Large language models meet nl2code: A survey. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7443–7464, 2023.

D. Zan, A. Yu, W. Liu, D. Chen, B. Shen, W. Li, Y. Yao, Y. Gong, X. Chen, B. Guan, et al. CodeS: Natural Language to Code Repository via Multi-Layer Sketch. arXiv preprint arXiv:2403.16443, 2024.

F. Zhang, B. Chen, Y. Zhang, J. Keung, J. Liu, D. Zan, Y. Mao, J.-G. Lou, and W. Chen. Repocoder: Repository-level code completion through iterative retrieval and generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2471–2484, 2023.

L. Zhang, D. Zan, Q. Yang, Z. Huang, D. Chen, B. Shen, T. Liu, Y. Gong, P. Huang, X. Lu, G. Liang, L. Cui, and Q. Wang. Codev: Issue resolving with visual data, 2024. URL https: //arxiv.org/abs/2412.17315.

- Q. Zheng, X. Xia, X. Zou, Y. Dong, S. Wang, Y. Xue, Z. Wang, L. Shen, A. Wang, Y. Li, T. Su, Z. Yang, and J. Tang. Codegeex: A pre-trained model for code generation with multilingual benchmarking on humaneval-x. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5673–5684, 2023a.

- Z. Zheng, K. Ning, Y. Wang, J. Zhang, D. Zheng, M. Ye, and J. Chen. A survey of large language models for code: Evolution, benchmarking, and future trends. arXiv preprint arXiv:2311.10372, 2023b.

