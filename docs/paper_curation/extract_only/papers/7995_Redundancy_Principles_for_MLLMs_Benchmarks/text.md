# arXiv:2501.13953v2[cs.CL]28May2025

### Redundancy Principles for MLLMs Benchmarks

##### Zicheng Zhang1,2*, Xiangyu Zhao1,2*, Xinyu Fang1,3, Chunyi Li1,2, Xiaohong Liu2, Xiongkuo Min2, Haodong Duan1†, Kai Chen 1†, Guangtao Zhai1,2†, 1Shanghai AI Laboratory, 2Shanghai Jiaotong University, 3Zhejiang University

3) Cross-Benchmark

2) Instances Number

###### 1) Dimensions

…

Bench N

Bench B

Effective Instances

High Redundancy

Low Redundancy

Anchor Bench

Bench A

Where Redundancy Exists？ Why Evaluate Redundancy?

MLLM Benchmark Design

MLLM Evaluation

[Figure 1]

Ensuring Independence Appropriate Instances Number

Bench

Bench A

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

- B

Bench

- C

[Figure 7]

[Figure 8]

#### c

Within One Domain Specfic

Reduce Redundant Instances

Bench D

Which To Choose?

[Figure 9]

Check Highly Redundant Dimensions

Figure 1: Where Redundancy Exists? and Why Evaluate Redundancy?

##### Abstract

##### 1 Introduction

Model Evaluation has always played a crucial role in the development of Multi-modal Large Language Models (MLLMs). Benchmarks serve not only as tools for assessing model accuracy but also as catalysts for driving innovation and improvements within the field. In the early stages, traditional model evaluation benchmarks such as GQA (Hudson and Manning, 2019), VQA-V2 (Antol et al., 2015), VizWiz (Bigham et al., 2010), and TextVQA (Singh et al., 2019) are characterized by relatively simple questions and answers, with responses often being a single word. This limits the depth of understanding and reasoning required from the models, making them less effective at evaluating the complex capabilities of modern MLLMs that are expected to handle more nuanced tasks. With the emergence of more powerful MLLMs (Achiam et al., 2023; Team et al., 2023; Chen et al., 2024b; Wang et al., 2024b; Li et al., 2024a; Liu et al., 2024), traditional evaluation frameworks have become inadequate to meet the flexible evaluation requirements. In response, a new generation of VQA benchmarks has arisen, such as MMBench (Liu et al., 2025), MMVet (Yu et al., 2023a), and MMMU (Yue et al., 2024).

With the rapid iteration of Multi-modality Large Language Models (MLLMs) and the evolving demands of the field, the number of benchmarks produced annually has surged into the hundreds. The rapid growth has inevitably led to significant redundancy among benchmarks. Therefore, it is crucial to take a step back and critically assess the current state of redundancy and propose targeted principles for constructing effective MLLM benchmarks. In this paper, we focus on redundancy from three key perspectives: 1) Redundancy of benchmark capability dimensions, 2) Redundancy in the number of test questions, and 3) Cross-benchmark redundancy within specific domains. Through the comprehensive analysis over hundreds of MLLMs’ performance across more than 20 benchmarks, we aim to quantitatively measure the level of redundancy lies in existing MLLM evaluations, provide valuable insights to guide the future development of MLLM benchmarks, and offer strategies to refine and address redundancy issues effectively.

1Equal Contribution. 2Corresponding Author.

As MLLMs have rapidly iterated and evolved, their diverse capabilities across various domains have garnered increasing attention, which has led to the development of specialized benchmarks to evaluate MLLMs’ performance in specific areas, like Mathematics Task (Lu et al., 2023; Zhang et al., 2025; Wang et al., 2024a; Zou et al., 2024), Optical Character Recognition (OCR) (Mishra et al., 2019; Liu et al., 2023a; Mathew et al., 2021), Medical Field (Hu et al., 2024), Remote Sensing (Li et al., 2024e), Agents (Yang et al., 2024), GUIs (Baechler et al., 2024), and so on.

The rapid proliferation of benchmarks has inevitably introduced significant redundancies, with overlapping capabilities being assessed and recurring questions appearing within and across benchmarks. Such redundancies create inefficiencies in model evaluation, repeatedly testing similar aspects of MLLM performance without contributing meaningful new insights. Additionally, this trend risks overemphasizing certain task types while neglecting others, potentially distorting research priorities. In this work, we address these challenges through a comprehensive and systematic exploration.

###### 1.1 Identifying Redundancy

Redundancy is an intrinsic and multifaceted issue in benchmarks, appearing in several key forms:

- • Redundancy across dimensions (intra-bench): Tasks within the same benchmark may evaluate overlapping capabilities of MLLMs, leading to repetitive assessments.
- • Redundancy among instances (intra-bench): Certain instances closely resemble others, providing minimal additional differentiation or insight for model evaluation.
- • Redundancy across benchmarks within specific domains: Benchmarks targeting specific domains often exhibit overlapping objectives or scopes, resulting in duplicated efforts across different evaluation sets.

###### 1.2 Ideal Redundancy Principles

Effective benchmarks should adhere to the following principles regarding redundancy:

• Independence of dimensions: Ideal benchmarks should ensure that its dimensions are largely independent, minimizing overlap between them. However, some degree of redundancy may be inevitable when certain capabilities naturally require the interaction of multiple foundational

skills, and redundancy should be carefully balanced to avoid excessive overlap.

- • Optimal instance count: A well-designed benchmark should strike a balance in the number of instances: neither too few nor too many, to ensure reliable and meaningful evaluations without introducing unnecessary redundancies.
- • Domain representativeness: A comprehensive benchmark targeted to a specific domain should represent the domain. This may involve purposeful overlap with other benchmarks within the same domain to reflect shared core capabilities.

###### 1.3 Benifits of Evaluating Redundancy

Evaluating and addressing redundancy offers several significant benefits, as shown in Fig. 1:

- • Optimizing benchmark design: 1). Determines whether certain dimensions within a benchmark warrant separate assessments or can be consolidated; 2). Identifies the minimal and sufficient number of instances required for accurate evaluation; 3). Assesses the necessity of introducing new benchmarks within specific domains.
- • Enhancing efficiency in MLLM evaluation: 1). Determines whether a benchmark deviates from the domain’s distribution ; 2). Identifies the anchor benchmarks required to evaluate model performance within the domain. By systematically addressing redundancy, we

can not only enhance the principles of benchmark design but also alleviate the resource demands of MLLM evaluation, creating a more streamlined and effective evaluation ecosystem.

##### 2 Redundancy Framework

We present a framework for evaluating redundancy among MLLM capabilities, defined as specific tasks within a benchmark. Our framework is grounded in the following prior assumption:

When evaluating similar capabilities, the performance rankings of MLLMs should exhibit strong correlation. Conversely, significant differences in these rankings suggest the evaluated capabilities are relatively independent.

Based on this principle, we propose the Performance Correlation Redundancy Framework, which quantifies redundancy by measuring the correlation of MLLM performance rankings. To ensure robustness and generalization capability, we leverage the comprehensive data from VLMEvalKit (Duan et al., 2024), which includes

diverse benchmarks and performance results from more than 100 MLLMs.

Dimensions Within Benchmark

Instances Within Benchmark

X1 X2

Xm

A% Instances

- 2.1 Dimensions Redundancy Assume a benchmark consists of a set of dimensions, denoted as X = {X1,X2,...,Xm}, where each Xi represents a specific dimension. Let N denote the number of MLLMs evaluated on these

dimensions. For a given dimension Xi, we denote the ranking of the N MLLMs on this dimension as Ri. To quantify the redundancy of Xi, we compute the average rank correlation between Ri and the rankings Rj of all other dimensions Xj (j ̸= i). Formally, the redundancy ρ(Xi) is defined as:

ρ(Xi) =

1 m − 1

m

j=1 j̸=i

CORR(Ri,Rj), (1)

where CORR(Ri,Rj) is the correlation coefficient between the rankings Ri and Rj.

- • High CORR(Ri,Rj) values can help identify potentially redundant dimension pairs.
- • ρ(Xi) represents the average redundancy level of dimension Xi, quantifying its overall overlap. By calculating the redundancy ρ(Xi) for all di-

mensions Xi in the benchmark and averaging these values, we can obtain the overall internal redundancy of the benchmark as well. Formally, the benchmark internal redundancy ρBI is defined as:

ρBI =

1 m

m

i=1

ρ(Xi), (2)

where ρ(Xi) is the redundancy of the i-th dimension as previously defined. This metric reflects the average similarity among all dimensions within the benchmark. A lower ρBI suggests that the dimensions are relatively independent and diverse.

- 2.2 Instances Redundancy

All Instances

MLLM Ranks R1~Rm

## …

MLLM Ranks

R1 Rm

R1

RA% RGT

Rank Corr.

Rank Corr.

Dimensions Redundancy

Instances Redundancy

(a) (b)

- Y1 Benchmarks Within Vertical Domain

- Y2

- K1
- K2

CrossBenchmark Redundancy

Rank Corr.

## …

Kl

Yl

MLLM Ranks K1~Kl

(c)

Figure 2: A quick look at the redundancy framework, where (a), (b), and (c) show the general process of computing dimensions redundancy, instances redundancy, and cross-benchmark redundancy respectively.

how representative the sampled subset is of the entire benchmark. To reduce the effect of randomness, the sampling process is repeated T = 100 times, and the average correlation result is recorded. We define the instance redundancy of the benchmark at sampling ratio A%, denoted as ρ(A%), as follows:

1 T 1≤t≤T

CORR(RA%t,RGT), (3)

ρ(A%) =

where RA%t represents the MLLM ranking based on the sampled A% instances at the tth time, and RGT is the MLLM ranking based on the full M instances within the MLLM benchmark. The interpretation of ρ(A%) is straightforward:

- • A higher ρ(A%) indicates that the sampled instances are highly representative of the entire benchmark, and the remaining 1 − A% instances contribute little additional information.
- • Conversely, a lower ρ(A%) suggests that the sampled instances are less representative, and more instances are needed to capture the variability of the full benchmark.

Let a benchmark contain M total instances (e.g., QA pairs). To evaluate redundancy, we begin by calculating the MLLM performance rankings obtained over the full set of all M instances, denoted as the ground-truth ranking RGT. We then randomly sample a subset of the instances, comprising A% of the total M, and compute the corresponding MLLM rankings, denoted as Rsample. To quantify the redundancy of the benchmark at a sampling ratio of A%, we calculate the correlation coefficient between Rsample and RGT. This correlation reflects

###### 2.3 Cross-Benchmark Redundancy

Consider Y = {Y1,Y2,...,Yl}, a collection of l benchmarks within a specific domain (e.g., object hallucination, visual reasoning, visual perception). Let N represent the number of MLLMs evaluated

1.0

Action Recognition

[Figure 10]

1.00 0.27 0.34 0.10 0.54 0.36 0.42 0.48 -0.05 -0.02 0.56 0.06 0.40 0.32 0.28 0.38 0.05 0.42 0.25 0.25

Attribute Comparison

- 0.27 1.00 0.07 0.08 -0.02 0.35 0.43 0.13 0.58 -0.04 0.31 -0.04 0.29 0.42 0.29 0.40 0.14 0.33 0.46 0.32

0.34 0.07 1.00 -0.02 0.39 0.14 0.36 0.07 0.22 0.31 0.07 0.19 0.11 0.22 -0.11 0.19 0.65 0.09 -0.24 -0.13

0.10 0.08 -0.02 1.00 0.37 0.17 0.17 0.23 -0.04 -0.25 0.24 -0.24 0.04 -0.11 0.20 0.07 -0.13 0.28 0.11 0.08

0.54 -0.02 0.39 0.37 1.00 0.38 0.40 0.22 -0.03 -0.15 0.45 0.08 0.36 0.08 0.03 0.17 0.22 0.17 -0.02 -0.03

0.36 0.35 0.14 0.17 0.38 1.00 0.39 0.34 0.38 -0.08 0.37 0.03 0.51 0.47 0.26 0.50 0.10 0.44 0.44 0.18

0.42 0.43 0.36 0.17 0.40 0.39 1.00 0.37 0.13 0.09 0.34 0.07 0.18 0.22 0.09 0.34 0.29 0.39 0.08 0.07

0.48 0.13 0.07 0.23 0.22 0.34 0.37 1.00 -0.01 -0.05 0.31 -0.12 0.30 0.19 0.17 0.14 -0.10 0.59 0.14 0.12

- -0.05 0.58 0.22 -0.04 -0.03 0.38 0.13 -0.01 1.00 0.08 0.01 0.02 0.27 0.40 0.19 0.45 0.36 0.07 0.29 0.33
- -0.02 -0.04 0.31 -0.25 -0.15 -0.08 0.09 -0.05 0.08 1.00 -0.05 0.30 -0.30 0.02 -0.28 0.00 0.40 0.05 -0.37 -0.13

0.56 0.31 0.07 0.24 0.45 0.37 0.34 0.31 0.01 -0.05 1.00 -0.06 0.56 0.00 0.38 0.31 -0.14 0.34 0.35 0.26

0.06 -0.04 0.19 -0.24 0.08 0.03 0.07 -0.12 0.02 0.30 -0.06 1.00 -0.08 0.25 -0.11 0.17 0.23 -0.09 -0.07 -0.03

0.40 0.29 0.11 0.04 0.36 0.51 0.18 0.30 0.27 -0.30 0.56 -0.08 1.00 0.34 0.53 0.43 -0.08 0.27 0.59 0.49

0.32 0.42 0.22 -0.11 0.08 0.47 0.22 0.19 0.40 0.02 0.00 0.25 0.34 1.00 0.26 0.40 0.28 0.27 0.52 0.41

- 0.28 0.29 -0.11 0.20 0.03 0.26 0.09 0.17 0.19 -0.28 0.38 -0.11 0.53 0.26 1.00 0.35 -0.34 0.33 0.63 0.56

Attribute Recognition

Celebrity Recognition

0.8

Function Reasoning

Future Prediction

Identity Reasoning

0.6

Image Emotion

Image Quality

Image Scene

Image

Style Image

0.4

Topic Nature

Relation

Object Localization

Ocr

0.2

Physical Property

0.38 0.40 0.19 0.07 0.17 0.50 0.34 0.14 0.45 0.00 0.31 0.17 0.43 0.40 0.35 1.00 0.15 0.25 0.50 0.46

Reasoning Physical Relation

0.05 0.14 0.65 -0.13 0.22 0.10 0.29 -0.10 0.36 0.40 -0.14 0.23 -0.08 0.28 -0.34 0.15 1.00 -0.01 -0.31 -0.18

Social Relation

0.42 0.33 0.09 0.28 0.17 0.44 0.39 0.59 0.07 0.05 0.34 -0.09 0.27 0.27 0.33 0.25 -0.01 1.00 0.28 0.20

0.0

Spatial

0.25 0.46 -0.24 0.11 -0.02 0.44 0.08 0.14 0.29 -0.37 0.35 -0.07 0.59 0.52 0.63 0.50 -0.31 0.28 1.00 0.67

Relationship Structuralized

Imagetext Understanding

0.25 0.32 -0.13 0.08 -0.03 0.18 0.07 0.12 0.33 -0.13 0.26 -0.03 0.49 0.41 0.56 0.46 -0.18 0.20 0.67 1.00

Relation

Relation

Reasoning

Reasoning

Reasoning

Recognition

Recognition

Recognition

Relationship

Relation

Ocr

Image

Image

Image

Quality

Physical

Understanding

Emotion

Comparison

Prediction

Localization

Scene

Structuralized

Image

Topic

Style

Nature

Function

Property

Image

Social

Physical

Identity

Celebrity

Attribute

Attribute

Future

Imagetext

Object

Spatial

Action

(a) 50+ SRCC dimensions redundancy.

1.0

Action Recognition

[Figure 11]

1.00 0.46 0.76 0.78 0.62 0.65 0.79 0.75 0.53 0.73 0.64 0.81 0.69 0.58 0.56 0.60 0.67 0.72 0.57 0.49

Attribute Comparison

0.46 1.00 0.63 0.63 0.71 0.72 0.69 0.57 0.75 0.58 0.52 0.53 0.74 0.82 0.62 0.47 0.61 0.67 0.78 0.66

Attribute Recognition

0.76 0.63 1.00 0.69 0.75 0.71 0.71 0.72 0.63 0.63 0.69 0.61 0.76 0.71 0.79 0.65 0.70 0.77 0.69 0.72

Celebrity Recognition

- 0.78 0.63 0.69 1.00 0.76 0.66 0.80 0.75 0.49 0.67 0.66 0.76 0.73 0.58 0.67 0.50 0.46 0.76 0.61 0.53

0.62 0.71 0.75 0.76 1.00 0.82 0.72 0.70 0.62 0.66 0.71 0.62 0.77 0.71 0.87 0.59 0.59 0.75 0.76 0.75

0.65 0.72 0.71 0.66 0.82 1.00 0.73 0.64 0.65 0.77 0.77 0.73 0.69 0.63 0.76 0.65 0.61 0.77 0.79 0.69

- 0.79 0.69 0.71 0.80 0.72 0.73 1.00 0.75 0.65 0.79 0.66 0.74 0.82 0.72 0.59 0.63 0.64 0.78 0.73 0.50

0.8

Function Reasoning

Future Prediction

Identity Reasoning

0.6

Image Emotion

0.75 0.57 0.72 0.75 0.70 0.64 0.75 1.00 0.52 0.70 0.70 0.76 0.70 0.64 0.62 0.52 0.61 0.81 0.61 0.48

Image Quality

0.53 0.75 0.63 0.49 0.62 0.65 0.65 0.52 1.00 0.59 0.58 0.52 0.57 0.78 0.63 0.54 0.70 0.60 0.67 0.68

Image Scene

0.73 0.58 0.63 0.67 0.66 0.77 0.79 0.70 0.59 1.00 0.76 0.84 0.67 0.58 0.60 0.53 0.55 0.79 0.67 0.50

Image

0.64 0.52 0.69 0.66 0.71 0.77 0.66 0.70 0.58 0.76 1.00 0.75 0.67 0.53 0.76 0.60 0.54 0.69 0.69 0.63

Style Image

0.4

0.81 0.53 0.61 0.76 0.62 0.73 0.74 0.76 0.52 0.84 0.75 1.00 0.60 0.49 0.55 0.47 0.50 0.76 0.53 0.40

Topic Nature

0.69 0.74 0.76 0.73 0.77 0.69 0.82 0.70 0.57 0.67 0.67 0.60 1.00 0.75 0.65 0.55 0.69 0.77 0.78 0.63

Relation

Object Localization

0.58 0.82 0.71 0.58 0.71 0.63 0.72 0.64 0.78 0.58 0.53 0.49 0.75 1.00 0.67 0.55 0.84 0.70 0.78 0.73

Ocr

- 0.56 0.62 0.79 0.67 0.87 0.76 0.59 0.62 0.63 0.60 0.76 0.55 0.65 0.67 1.00 0.62 0.60 0.68 0.73 0.88

0.60 0.47 0.65 0.50 0.59 0.65 0.63 0.52 0.54 0.53 0.60 0.47 0.55 0.55 0.62 1.00 0.63 0.52 0.65 0.62

0.67 0.61 0.70 0.46 0.59 0.61 0.64 0.61 0.70 0.55 0.54 0.50 0.69 0.84 0.60 0.63 1.00 0.62 0.70 0.67

0.72 0.67 0.77 0.76 0.75 0.77 0.78 0.81 0.60 0.79 0.69 0.76 0.77 0.70 0.68 0.52 0.62 1.00 0.70 0.58

- 0.57 0.78 0.69 0.61 0.76 0.79 0.73 0.61 0.67 0.67 0.69 0.53 0.78 0.78 0.73 0.65 0.70 0.70 1.00 0.75

0.2

Physical Property

Reasoning Physical Relation

Social Relation

0.0

Spatial

Relationship Structuralized

Imagetext Understanding

0.49 0.66 0.72 0.53 0.75 0.69 0.50 0.48 0.68 0.50 0.63 0.40 0.63 0.73 0.88 0.62 0.67 0.58 0.75 1.00

Relation

Relation

Reasoning

Reasoning

Reasoning

Recognition

Recognition

Recognition

Relationship

Relation

Ocr

Image

Image

Image

Quality

Physical

Understanding

Emotion

Comparison

Prediction

Localization

Scene

Structuralized

Image

Topic

Style

Nature

Function

Property

Image

Social

Physical

Identity

Celebrity

Attribute

Attribute

Future

Imagetext

Object

Spatial

Action

(d) 50− SRCC dimensions redundancy.

1.0

Action Recognition

[Figure 12]

1.00 0.20 0.35 -0.05 0.49 0.35 0.38 0.45 -0.08 0.10 0.55 0.27 0.40 0.20 0.29 0.39 0.11 0.38 0.27 0.24

Attribute Comparison

0.20 1.00 0.17 -0.07 -0.02 0.31 0.41 0.14 0.49 -0.02 0.22 -0.06 0.18 0.57 0.21 0.31 0.31 0.33 0.34 0.29

Attribute Recognition

0.35 0.17 1.00 -0.02 0.28 0.15 0.35 0.10 0.29 0.29 0.04 0.26 0.13 0.23 -0.01 0.19 0.65 0.08 -0.23 -0.13

Celebrity Recognition

- -0.05 -0.07 -0.02 1.00 0.25 0.07 0.12 0.11 -0.07 -0.26 0.07 -0.25 -0.09 -0.24 -0.03 -0.08 -0.05 0.18 -0.03 -0.09

0.49 -0.02 0.28 0.25 1.00 0.34 0.47 0.12 -0.04 -0.18 0.43 -0.02 0.38 0.03 0.11 0.16 0.12 0.24 0.05 -0.00

0.35 0.31 0.15 0.07 0.34 1.00 0.41 0.41 0.20 -0.10 0.35 -0.11 0.49 0.38 0.25 0.50 0.18 0.42 0.48 0.13

- 0.38 0.41 0.35 0.12 0.47 0.41 1.00 0.34 0.05 0.13 0.39 0.15 0.22 0.16 0.13 0.34 0.28 0.45 0.15 0.09

0.45 0.14 0.10 0.11 0.12 0.41 0.34 1.00 -0.14 0.10 0.45 0.00 0.23 0.09 0.04 0.07 -0.01 0.53 0.19 -0.03

-0.08 0.49 0.29 -0.07 -0.04 0.20 0.05 -0.14 1.00 0.03 -0.08 -0.09 0.27 0.36 0.17 0.33 0.26 -0.19 0.23 0.25

- 0.10 -0.02 0.29 -0.26 -0.18 -0.10 0.13 0.10 0.03 1.00 -0.08 0.64 -0.29 0.09 -0.22 0.01 0.30 0.03 -0.28 -0.02

0.55 0.22 0.04 0.07 0.43 0.35 0.39 0.45 -0.08 -0.08 1.00 -0.18 0.50 -0.03 0.33 0.24 -0.17 0.40 0.36 0.29

0.27 -0.06 0.26 -0.25 -0.02 -0.11 0.15 0.00 -0.09 0.64 -0.18 1.00 -0.15 0.23 -0.05 0.06 0.13 -0.08 -0.08 0.06

0.40 0.18 0.13 -0.09 0.38 0.49 0.22 0.23 0.27 -0.29 0.50 -0.15 1.00 0.26 0.55 0.47 -0.06 0.24 0.62 0.47

0.20 0.57 0.23 -0.24 0.03 0.38 0.16 0.09 0.36 0.09 -0.03 0.23 0.26 1.00 0.18 0.22 0.41 0.20 0.40 0.28

0.29 0.21 -0.01 -0.03 0.11 0.25 0.13 0.04 0.17 -0.22 0.33 -0.05 0.55 0.18 1.00 0.35 -0.30 0.30 0.63 0.52

0.39 0.31 0.19 -0.08 0.16 0.50 0.34 0.07 0.33 0.01 0.24 0.06 0.47 0.22 0.35 1.00 0.17 0.21 0.47 0.44

- 0.11 0.31 0.65 -0.05 0.12 0.18 0.28 -0.01 0.26 0.30 -0.17 0.13 -0.06 0.41 -0.30 0.17 1.00 0.11 -0.25 -0.24

0.8

Function Reasoning

Future Prediction

Identity Reasoning

0.6

Image Emotion

Image Quality

Image Scene

Image

Style Image

0.4

Topic Nature

Relation

Object Localization

Ocr

0.2

Physical Property

Reasoning Physical Relation

Social Relation

0.38 0.33 0.08 0.18 0.24 0.42 0.45 0.53 -0.19 0.03 0.40 -0.08 0.24 0.20 0.30 0.21 0.11 1.00 0.31 0.14

0.0

Spatial

0.27 0.34 -0.23 -0.03 0.05 0.48 0.15 0.19 0.23 -0.28 0.36 -0.08 0.62 0.40 0.63 0.47 -0.25 0.31 1.00 0.68

Relationship Structuralized

Imagetext Understanding

0.24 0.29 -0.13 -0.09 -0.00 0.13 0.09 -0.03 0.25 -0.02 0.29 0.06 0.47 0.28 0.52 0.44 -0.24 0.14 0.68 1.00

Relation

Relation

Reasoning

Reasoning

Reasoning

Recognition

Recognition

Recognition

Relationship

Relation

Ocr

Image

Image

Image

Quality

Physical

Understanding

Emotion

Comparison

Prediction

Localization

Scene

Structuralized

Image

Topic

Style

Nature

Function

Property

Image

Social

Physical

Identity

Celebrity

Attribute

Attribute

Future

Imagetext

Object

Spatial

Action

(b) 50+ PLCC dimensions redundancy.

1.0

Action Recognition

[Figure 13]

1.00 0.61 0.94 0.92 0.93 0.89 0.92 0.92 0.58 0.92 0.90 0.94 0.85 0.68 0.89 0.79 0.73 0.94 0.71 0.40

Attribute Comparison

0.61 1.00 0.66 0.67 0.70 0.72 0.60 0.62 0.76 0.58 0.58 0.57 0.75 0.83 0.67 0.59 0.62 0.68 0.77 0.51

Attribute Recognition

0.94 0.66 1.00 0.87 0.96 0.90 0.93 0.91 0.64 0.94 0.93 0.93 0.86 0.72 0.91 0.85 0.73 0.95 0.77 0.55

Celebrity Recognition

- 0.92 0.67 0.87 1.00 0.91 0.81 0.86 0.86 0.55 0.84 0.83 0.85 0.84 0.65 0.87 0.70 0.60 0.89 0.68 0.42
- 0.93 0.70 0.96 0.91 1.00 0.90 0.93 0.91 0.62 0.93 0.92 0.92 0.88 0.71 0.92 0.83 0.68 0.95 0.80 0.57

- 0.89 0.72 0.90 0.81 0.90 1.00 0.86 0.89 0.71 0.88 0.88 0.85 0.85 0.69 0.90 0.84 0.71 0.91 0.81 0.56

0.92 0.60 0.93 0.86 0.93 0.86 1.00 0.91 0.55 0.97 0.91 0.96 0.79 0.63 0.86 0.81 0.67 0.93 0.69 0.43

0.92 0.62 0.91 0.86 0.91 0.89 0.91 1.00 0.58 0.92 0.91 0.90 0.83 0.65 0.89 0.80 0.67 0.93 0.71 0.43

0.58 0.76 0.64 0.55 0.62 0.71 0.55 0.58 1.00 0.57 0.57 0.53 0.64 0.80 0.65 0.63 0.73 0.62 0.71 0.59

0.92 0.58 0.94 0.84 0.93 0.88 0.97 0.92 0.57 1.00 0.94 0.98 0.79 0.61 0.85 0.81 0.65 0.94 0.71 0.42

- 0.90 0.58 0.93 0.83 0.92 0.88 0.91 0.91 0.57 0.94 1.00 0.94 0.81 0.58 0.86 0.82 0.60 0.93 0.72 0.52

- 0.94 0.57 0.93 0.85 0.92 0.85 0.96 0.90 0.53 0.98 0.94 1.00 0.75 0.58 0.84 0.78 0.62 0.92 0.65 0.40

0.8

Function Reasoning

Future Prediction

Identity Reasoning

0.6

Image Emotion

Image Quality

Image Scene

Image

Style Image

0.4

Topic Nature

0.85 0.75 0.86 0.84 0.88 0.85 0.79 0.83 0.64 0.79 0.81 0.75 1.00 0.77 0.84 0.77 0.72 0.87 0.82 0.54

Relation

Object Localization

0.68 0.83 0.72 0.65 0.71 0.69 0.63 0.65 0.80 0.61 0.58 0.58 0.77 1.00 0.71 0.64 0.85 0.70 0.78 0.58

Ocr

0.89 0.67 0.91 0.87 0.92 0.90 0.86 0.89 0.65 0.85 0.86 0.84 0.84 0.71 1.00 0.76 0.70 0.89 0.78 0.66

0.2

Physical Property

0.79 0.59 0.85 0.70 0.83 0.84 0.81 0.80 0.63 0.81 0.82 0.78 0.77 0.64 0.76 1.00 0.69 0.81 0.73 0.53

Reasoning Physical Relation

0.73 0.62 0.73 0.60 0.68 0.71 0.67 0.67 0.73 0.65 0.60 0.62 0.72 0.85 0.70 0.69 1.00 0.69 0.72 0.48

Social Relation

0.94 0.68 0.95 0.89 0.95 0.91 0.93 0.93 0.62 0.94 0.93 0.92 0.87 0.70 0.89 0.81 0.69 1.00 0.76 0.48

0.0

Spatial

0.71 0.77 0.77 0.68 0.80 0.81 0.69 0.71 0.71 0.71 0.72 0.65 0.82 0.78 0.78 0.73 0.72 0.76 1.00 0.73

Relationship Structuralized

Imagetext Understanding

0.40 0.51 0.55 0.42 0.57 0.56 0.43 0.43 0.59 0.42 0.52 0.40 0.54 0.58 0.66 0.53 0.48 0.48 0.73 1.00

Relation

Relation

Reasoning

Reasoning

Reasoning

Recognition

Recognition

Recognition

Relationship

Relation

Ocr

Image

Image

Image

Quality

Physical

Understanding

Emotion

Comparison

Prediction

Localization

Scene

Structuralized

Image

Topic

Style

Nature

Function

Property

Image

Social

Physical

Identity

Celebrity

Attribute

Attribute

Future

Imagetext

Object

Spatial

Action

(e) 50− PLCC dimensions redundancy.

1.0

Action Recognition

[Figure 14]

1.00 0.04 0.12 0.00 0.24 0.13 0.15 0.21 0.01 0.01 0.30 0.07 0.16 0.04 0.08 0.15 0.01 0.15 0.07 0.06

Attribute Comparison

0.04 1.00 0.03 0.01 0.00 0.10 0.17 0.02 0.24 0.00 0.05 0.00 0.03 0.32 0.05 0.09 0.10 0.11 0.12 0.08

Attribute Recognition

- 0.12 0.03 1.00 0.00 0.08 0.02 0.12 0.01 0.08 0.08 0.00 0.07 0.02 0.05 0.00 0.04 0.42 0.01 0.05 0.02

- 0.00 0.01 0.00 1.00 0.06 0.01 0.02 0.01 0.01 0.07 0.01 0.06 0.01 0.06 0.00 0.01 0.00 0.03 0.00 0.01

0.24 0.00 0.08 0.06 1.00 0.12 0.22 0.01 0.00 0.03 0.19 0.00 0.15 0.00 0.01 0.03 0.02 0.06 0.00 0.00

0.13 0.10 0.02 0.01 0.12 1.00 0.17 0.17 0.04 0.01 0.13 0.01 0.24 0.14 0.06 0.25 0.03 0.18 0.23 0.02

- 0.15 0.17 0.12 0.02 0.22 0.17 1.00 0.12 0.00 0.02 0.15 0.02 0.05 0.03 0.02 0.11 0.08 0.20 0.02 0.01

0.21 0.02 0.01 0.01 0.01 0.17 0.12 1.00 0.02 0.01 0.20 0.00 0.05 0.01 0.00 0.00 0.00 0.28 0.03 0.00

0.01 0.24 0.08 0.01 0.00 0.04 0.00 0.02 1.00 0.00 0.01 0.01 0.07 0.13 0.03 0.11 0.07 0.04 0.05 0.06

0.01 0.00 0.08 0.07 0.03 0.01 0.02 0.01 0.00 1.00 0.01 0.41 0.08 0.01 0.05 0.00 0.09 0.00 0.08 0.00

0.30 0.05 0.00 0.01 0.19 0.13 0.15 0.20 0.01 0.01 1.00 0.03 0.25 0.00 0.11 0.06 0.03 0.16 0.13 0.09

- 0.07 0.00 0.07 0.06 0.00 0.01 0.02 0.00 0.01 0.41 0.03 1.00 0.02 0.05 0.00 0.00 0.02 0.01 0.01 0.00

0.16 0.03 0.02 0.01 0.15 0.24 0.05 0.05 0.07 0.08 0.25 0.02 1.00 0.07 0.31 0.22 0.00 0.06 0.38 0.22

0.04 0.32 0.05 0.06 0.00 0.14 0.03 0.01 0.13 0.01 0.00 0.05 0.07 1.00 0.03 0.05 0.17 0.04 0.16 0.08

- 0.08 0.05 0.00 0.00 0.01 0.06 0.02 0.00 0.03 0.05 0.11 0.00 0.31 0.03 1.00 0.13 0.09 0.09 0.40 0.27

Celebrity Recognition

0.8

Function Reasoning

Future Prediction

Identity Reasoning

0.6

Image Emotion

Image Quality

Image Scene

Image

Style Image

0.4

Topic Nature

Relation

Object Localization

Ocr

0.2

Physical Property

0.15 0.09 0.04 0.01 0.03 0.25 0.11 0.00 0.11 0.00 0.06 0.00 0.22 0.05 0.13 1.00 0.03 0.04 0.22 0.19

Reasoning Physical Relation

0.01 0.10 0.42 0.00 0.02 0.03 0.08 0.00 0.07 0.09 0.03 0.02 0.00 0.17 0.09 0.03 1.00 0.01 0.06 0.06

Social Relation

0.15 0.11 0.01 0.03 0.06 0.18 0.20 0.28 0.04 0.00 0.16 0.01 0.06 0.04 0.09 0.04 0.01 1.00 0.10 0.02

0.0

Spatial

0.07 0.12 0.05 0.00 0.00 0.23 0.02 0.03 0.05 0.08 0.13 0.01 0.38 0.16 0.40 0.22 0.06 0.10 1.00 0.47

Relationship Structuralized

Imagetext Understanding

0.06 0.08 0.02 0.01 0.00 0.02 0.01 0.00 0.06 0.00 0.09 0.00 0.22 0.08 0.27 0.19 0.06 0.02 0.47 1.00

Relation

Relation

Reasoning

Reasoning

Reasoning

Recognition

Recognition

Recognition

Relationship

Relation

Ocr

Image

Image

Image

Quality

Physical

Understanding

Emotion

Comparison

Prediction

Localization

Scene

Structuralized

Image

Topic

Style

Nature

Function

Property

Image

Social

Physical

Identity

Celebrity

Attribute

Attribute

Future

Imagetext

Object

Spatial

Action

(c) 50+ R2 dimensions redundancy.

1.0

Action Recognition

[Figure 15]

1.00 0.38 0.89 0.85 0.87 0.78 0.86 0.84 0.34 0.85 0.81 0.88 0.72 0.46 0.79 0.62 0.53 0.88 0.50 0.16

Attribute Comparison

0.38 1.00 0.43 0.45 0.48 0.52 0.36 0.38 0.58 0.34 0.34 0.33 0.57 0.68 0.45 0.34 0.39 0.46 0.59 0.26

Attribute Recognition

0.89 0.43 1.00 0.76 0.92 0.80 0.87 0.83 0.41 0.89 0.87 0.87 0.75 0.52 0.83 0.73 0.53 0.91 0.60 0.30

Celebrity Recognition

- 0.85 0.45 0.76 1.00 0.83 0.66 0.74 0.74 0.30 0.70 0.69 0.72 0.71 0.43 0.76 0.49 0.37 0.79 0.47 0.18

- 0.87 0.48 0.92 0.83 1.00 0.81 0.87 0.83 0.38 0.87 0.85 0.85 0.78 0.50 0.84 0.69 0.46 0.91 0.64 0.32

- 0.78 0.52 0.80 0.66 0.81 1.00 0.74 0.78 0.50 0.77 0.78 0.72 0.73 0.47 0.81 0.70 0.51 0.83 0.66 0.31

0.86 0.36 0.87 0.74 0.87 0.74 1.00 0.83 0.30 0.94 0.82 0.93 0.63 0.39 0.74 0.65 0.44 0.86 0.48 0.18

- 0.84 0.38 0.83 0.74 0.83 0.78 0.83 1.00 0.33 0.84 0.83 0.80 0.69 0.43 0.79 0.63 0.45 0.86 0.50 0.19

0.34 0.58 0.41 0.30 0.38 0.50 0.30 0.33 1.00 0.32 0.33 0.28 0.41 0.63 0.42 0.39 0.53 0.39 0.51 0.35

- 0.85 0.34 0.89 0.70 0.87 0.77 0.94 0.84 0.32 1.00 0.89 0.96 0.62 0.38 0.72 0.66 0.43 0.88 0.50 0.18

0.81 0.34 0.87 0.69 0.85 0.78 0.82 0.83 0.33 0.89 1.00 0.88 0.66 0.34 0.74 0.67 0.35 0.86 0.52 0.27

0.88 0.33 0.87 0.72 0.85 0.72 0.93 0.80 0.28 0.96 0.88 1.00 0.57 0.33 0.70 0.61 0.39 0.85 0.43 0.16

0.72 0.57 0.75 0.71 0.78 0.73 0.63 0.69 0.41 0.62 0.66 0.57 1.00 0.60 0.70 0.59 0.52 0.76 0.67 0.29

0.46 0.68 0.52 0.43 0.50 0.47 0.39 0.43 0.63 0.38 0.34 0.33 0.60 1.00 0.51 0.41 0.72 0.49 0.61 0.34

- 0.79 0.45 0.83 0.76 0.84 0.81 0.74 0.79 0.42 0.72 0.74 0.70 0.70 0.51 1.00 0.58 0.49 0.79 0.61 0.44

0.8

Function Reasoning

Future Prediction

Identity Reasoning

0.6

Image Emotion

Image Quality

Image Scene

Image

Style Image

0.4

Topic Nature

Relation

Object Localization

Ocr

0.2

Physical Property

0.62 0.34 0.73 0.49 0.69 0.70 0.65 0.63 0.39 0.66 0.67 0.61 0.59 0.41 0.58 1.00 0.48 0.65 0.53 0.28

Reasoning Physical Relation

0.53 0.39 0.53 0.37 0.46 0.51 0.44 0.45 0.53 0.43 0.35 0.39 0.52 0.72 0.49 0.48 1.00 0.47 0.52 0.23

Social Relation

0.88 0.46 0.91 0.79 0.91 0.83 0.86 0.86 0.39 0.88 0.86 0.85 0.76 0.49 0.79 0.65 0.47 1.00 0.57 0.23

0.0

Spatial

0.50 0.59 0.60 0.47 0.64 0.66 0.48 0.50 0.51 0.50 0.52 0.43 0.67 0.61 0.61 0.53 0.52 0.57 1.00 0.54

Relationship Structuralized

Imagetext Understanding

0.16 0.26 0.30 0.18 0.32 0.31 0.18 0.19 0.35 0.18 0.27 0.16 0.29 0.34 0.44 0.28 0.23 0.23 0.54 1.00

Relation

Relation

Reasoning

Reasoning

Reasoning

Recognition

Recognition

Recognition

Relationship

Relation

Ocr

Image

Image

Image

Quality

Physical

Understanding

Emotion

Comparison

Prediction

Localization

Scene

Structuralized

Image

Topic

Style

Nature

Function

Property

Image

Social

Physical

Identity

Celebrity

Attribute

Attribute

Future

Imagetext

Object

Spatial

Action

(f) 50− R2 dimensions redundancy.

- Figure 3: Visualizations of dimensions redundancy for MMBench (Liu et al., 2025) on Top-50 and Bottom-50 (marked as 50+ and 50−) MLLMs respectively. More benchmark results can be found in Appendix. B.

across these benchmarks. For a given benchmark Yi, let Ki denote the ranking of the N MLLMs based on their performance on Yi. To identify key anchor benchmarks within this domain (an anchor benchmark can serve as a representative over multiple other benchmarks), we focus on benchmarks that demonstrate high redundancy with others in the domain (Zohar et al., 2024). We define the redundancy of a benchmark ρ(Yi) as the average rank correlation coefficient between Ki and the rankings Kj of all other benchmarks Yj (j ̸= i) in the domain. Formally, ρ(Yi) is expressed as:

l

1 l − 1

CORR(Ki,Kj), (4)

ρ(Yi) =

j=1 j̸=i

where CORR(Ki,Kj) is the correlation coefficient between the rankings Ki and Kj. The interpretation of ρ(Yi) is as follows:

- • A higher ρ(Yi) indicates that benchmark Yi exhibits strong similarity with others in the domain, suggesting that it is highly representative of the domain’s capabilities or evaluation focus.
- • Conversely, a lower ρ(Yi) indicates that benchmark Yi shares less overlap with others, implying that it is less redundant and may capture unique/distinct aspects of the domain, or incorporate noises that are not related to the domain.

###### 2.4 Correlation Metrics

In this work, we adopt multiple metrics to describe the correlation between two set of performance numbers, including the Spearman Rank Correlation Coefficient (SRCC), the Pearson Linear Correlation Coefficient (PLCC), and the R² Score (R-squared Coefficient of Determination).

- • SRCC is an evaluation metric that measures rank similarity, capturing how well the relative order between two rankings aligns.
- • PLCC quantifies linear similarity, assessing how closely the rankings follow a linear relationship.
- • R² Score, on the other hand, evaluates the proportion of variance explained by the ranking relationship, serving as a measure of goodness-of-fit.

###### 2.5 Top-K Analysis

Considering that the performance of top-tier MLLMs often garners greater attention on benchmarks, we can streamline the redundancy analysis by focusing only on the top-K MLLMs with the highest overall performance on a given benchmark, rather than incorporating all MLLMs in the calculation. By selecting the top-K models, we can better target the analysis of benchmark redundancy across different performance tiers. This approach also simplifies the process of maintaining and updating our framework as new MLLMs are introduced.

0.35

0.30

0.25

AverageValues

0.20

0.15

0.10

0.05

0.00

Localization

Action

Prediction

Attribute

Attribute

Property

Physical

Physical

Emotion

Social

Imagetext

Spatial

Quality

Comparison

Ocr

Object

Structuralized

Style

Topic

Celebrity

Identity

Scene

Understanding

Image

Image

Image

Image

Image

Function

Nature

Relationship

Recognition

Recognition

Recognition

Relation

Relation

Relation

Reasoning

Reasoning

Reasoning

Future

(a) Top-50 SRCC redundancy.

0.7

0.6

0.5

AverageValues

0.4

0.3

0.2

0.1

0.0

Localization

Action

Prediction

Attribute

Attribute

Property

Physical

Physical

Emotion

Social

Imagetext

Spatial

Quality

Comparison

Ocr

Object

Structuralized

Style

Topic

Celebrity

Identity

Scene

Understanding

Image

Image

Image

Image

Image

Function

Nature

Relationship

Recognition

Recognition

Recognition

Relation

Relation

Relation

Reasoning

Reasoning

Reasoning

Future

(d) Bottom-50 SRCC redundancy.

0.35

0.30

0.25

AverageValues

0.20

0.15

0.10

0.05

0.00

Localization

Action

Prediction

Attribute

Attribute

Property

Physical

Physical

Emotion

Social

Imagetext

Spatial

Quality

Comparison

Ocr

Object

Structuralized

Style

Topic

Celebrity

Identity

Scene

Understanding

Image

Image

Image

Image

Image

Function

Nature

Relationship

Recognition

Recognition

Recognition

Relation

Relation

Relation

Reasoning

Reasoning

Reasoning

Future

(b) Top-50 PLCC redundancy.

0.8

0.7

0.6

AverageValues

0.5

0.4

0.3

0.2

0.1

0.0

Localization

Action

Prediction

Attribute

Attribute

Property

Physical

Physical

Emotion

Social

Imagetext

Spatial

Quality

Comparison

Ocr

Object

Structuralized

Style

Topic

Celebrity

Identity

Scene

Understanding

Image

Image

Image

Image

Image

Function

Nature

Relationship

Recognition

Recognition

Recognition

Relation

Relation

Relation

Reasoning

Reasoning

Reasoning

Future

(e) Bottom-50 PLCC redundancy.

0.175

0.150

0.125

AverageValues

0.100

0.075

0.050

0.025

0.000

Localization

Action

Prediction

Attribute

Attribute

Property

Physical

Physical

Emotion

Social

Imagetext

Spatial

Quality

Comparison

Ocr

Object

Structuralized

Style

Topic

Celebrity

Identity

Scene

Understanding

Image

Image

Image

Image

Image

Function

Nature

Relationship

Recognition

Recognition

Recognition

Relation

Relation

Relation

Reasoning

Reasoning

Reasoning

Future

(c) Top-50 R2 redundancy.

0.7

0.6

0.5

AverageValues

0.4

0.3

0.2

0.1

0.0

Localization

Action

Prediction

Attribute

Attribute

Property

Physical

Physical

Emotion

Social

Imagetext

Spatial

Quality

Comparison

Ocr

Object

Structuralized

Style

Topic

Celebrity

Identity

Scene

Understanding

Image

Image

Image

Image

Image

Function

Nature

Relationship

Recognition

Recognition

Recognition

Relation

Relation

Relation

Reasoning

Reasoning

Reasoning

Future

(f) Bottom-50 R2 redundancy.

- Figure 4: Bar plots of dimensions redundancy for MMBench (Liu et al., 2025) on Top-50 and Bottom-50 MLLMs. The redundancy values are computed by averaging the redundancy of each dimension with the redundancy of all other dimensions. 3 Experiment & Discussion

other dimensions, such as Spatial Relationship, Physical Property Reasoning, OCR, and Nature Relation, indicating that these tasks collectively represent the diverse abilities required to perform Structuralized Image-Text Understanding. In contrast, Image Topic and Image Scene exhibit relatively low redundancy with other dimensions, as shown in Figs. 4a to 4c. This could arise from the inherent complexity of assessing the overall topic and scene of an image, which is often less correlated with evaluating specific attributes or relationships. For instance, strong performance in recognizing individual attributes does not necessarily imply a comprehensive understanding of the overall topic or scene. However, Fig. 3b reveals that these two dimensions exhibit redundancy in terms of PLCC, suggesting potential overlaps within certain contexts. Another interesting insight arises from Celebrity Recognition, a knowledge-based task that remains relatively independent of other dimensions, which primarily measure perceptual abilities. As a result, it consistently exhibits significantly lower redundancy across SRCC, PLCC, and R². Conversely, high levels of redundancy are observed for Nature Relation and Spatial Relationship, as shown in Figs. 4a to 4c. This is attributed to the fact that these two dimensions serve as fundamental skills required by numerous other tasks, making their overlap a cornerstone of the broader evaluation framework.

We use the evaluation results of hundreds of MLLMs obtained through the VLMEvalKit (Duan et al., 2024) as our data source for conducting experiments and analysis. All the data sources are open-sourced and available on HuggingFace 1.

###### 3.1 Exploring Dimension Redundancy

To comprehensively demonstrate the application of our redundancy framework in MLLM benchmarks, we conduct a detailed case study using the widely adopted and dimensionally diverse MMBench benchmark (v1.1) (Liu et al., 2025). We categorize the MLLMs into two groups, Top-50 and Bottom-50, based on their overall performance in MMBench. This categorization enables us to highlight the differences in redundancy exhibited by MMBench when evaluating MLLMs with varying levels of capability. The dimension redundancy results are illustrated in Fig. 3 and Fig. 4, from which we derived several interesting insights.

Top-50 Redundancy. Figs. 3a and 3b visually illustrate the redundancy of SRCC and PLCC across various sub-dimensions, allowing for a quick analysis of which dimensions exhibit high correlations. For example, the tasks Image Emotion and Social Relation display strong redundancy, suggesting a significant overlap in the skills they assess. Similarly, Structuralized Image-Text Understanding demonstrates notable redundancy with several

Bottom-50 Redundancy. The results for the Bottom-50 redundancy, as shown in Figs. 4d to 4f, reveal a striking trend where nearly all dimensions

1https://huggingface.co/datasets/VLMEval/ OpenVLMRecords

1.00

0.95

0.90

MetricValue

0.85

0.80

Mean SRCC Mean PLCC Mean R2

0.75

Threshold 0.95

Above 0.95 Below 0.95

0.70

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Sample Percentage

(a) Instances redundancy with Top-50 MLLMs.

1.00

0.95

0.90

MetricValue

0.85

0.80

Mean SRCC Mean PLCC Mean R2

0.75

Threshold 0.95

Above 0.95 Below 0.95

0.70

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Sample Percentage

(b) Instances redundancy with Bottom-50 MLLMs.

- Figure 5: Visualizations of average instance redundancy for (a) Top-50 MLLMs and (b) Bottom-50 MLLMs across 18 LMM benchmarks (A-Bench (Zhang et al., 2024a), AI2D (Kembhavi et al., 2016), BLINK (Fu et al., 2025), HallusionBench (Guan et al., 2023), MMBench (Liu et al., 2025), MMMU (Yue et al., 2024), MME (Fu et al., 2024), MMStar (Chen et al., 2024a), MMT (Ying et al., 2024), MMVet (Yu et al., 2023b), OCRBench (Liu et al.,

- 2023b), Q-Bench (Wu et al., 2023; Zhang et al., 2024b), R-Bench-Dis (Li et al., 2024d), RealWorldQA (xAI, 2024), ScienceQA (Lu et al., 2022), SeedBench_IMG (Li et al., 2023), SeedBench2_Plus (Li et al., 2024b)). Notably, each data point represents the average of 100 sampling iterations to mitigate the impact of randomness.

exhibit significantly higher redundancy compared to the Top-50 redundancy. Specifically, most dimension pairs achieve SRCC and PLCC scores exceeding 0.6 (Figs. 4d and 4e), leading to an interesting conclusion: the dimensions appear to be more redundant for Bottom-50 MLLMs than for Top-50 MLLMs. This phenomenon can primarily be attributed to the fact that Bottom-50 MLLMs generally underperform across all capabilities. For these models, as their foundational abilities improve, incremental enhancements in one dimension often drive simultaneous improvements across others. This results in high consistency in performance rankings across dimensions, thereby causing relatively high dimensional redundancy. In contrast, the Top-50 MLLMs have already achieved relatively strong foundational capabilities. Consequently, more complex tasks across different dimensions introduce greater variability, allowing for more differentiation between performance in those dimensions. This leads to noticeably lower levels of redundancy for the Top-50 models. These findings emphasize the importance of carefully selecting the MLLMs included in redundancy analysis. Specifically, avoiding models with universally poor performance is crucial to ensure that the evaluation yields meaningful and accurate insights.

3.2 Exploration Instance Redundancy

We include the evaluation results from 18 publicly available benchmarks in VLMEvalKit (Duan et al.,

- 2024) in our experiments, with the average performance across benchmarks presented in Fig. 5.

We adopt a similarity threshold of 0.95 for partitioning2, This leads to an intriguing conclusion: a majority of existing MLLM benchmarks exhibit significant redundancy in their instances when ranking both Top-50 and Bottom-50 MLLMs, with at least 50% of the instances being redundant. This indicates that many benchmarks could reduce their instance counts by half without significantly affecting the ranking of MLLMs being tested. The R² score provides further insight, as it measures how effectively the final performance of MLLMs can be predicted using sampled instances. Compared to ensuring accurate ranking, achieving high accuracy in predicting the absolute performance of MLLMs requires a much larger number of instances. For example, both Top-50 and Bottom-50 MLLMs require over 90% of the instances to achieve an R² score greater than 0.95. This distinction highlights that fewer instances are sufficient for reliable ranking than for precise performance prediction.

We also compare redundancy tendencies between Top-50 and Bottom-50 MLLMs, as shown in Figs. 5a and 5b. Notably, at the same 0.95 threshold for SRCC and PLCC, Bottom-50 MLLMs require significantly fewer instances than Top50 MLLMs. This implies that accurately ranking higher-performing MLLMs (Top-50) demands more instances, while ranking lower-performing MLLMs (Bottom-50) can be achieved with fewer

2Ranks with SRCC and PLCC coefficients exceeding 0.95 are considered nearly identical, with only marginal differences in very few cases (Hauke and Kossowski, 2011).

1.00

0.95

0.90

0.85

MetricValue

0.80

BLINK

0.75

ScienceQA

MMMU

RealWorldQA

MMBench

0.70

MMStar

SEEDBench_IMG

AI2D

Threshold 0.95

0.65

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Sample Percentage

(a) Top-50 SRCC redundancy.

1.00

0.95

0.90

0.85

MetricValue

0.80

BLINK

0.75

ScienceQA

MMMU

RealWorldQA

MMBench

0.70

MMStar

SEEDBench_IMG

AI2D

Threshold 0.95

0.65

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Sample Percentage

(d) Bottom-50 SRCC redundancy.

1.00

0.95

0.90

MetricValue

0.85

0.80

BLINK

0.75

ScienceQA

MMMU

RealWorldQA

MMBench

0.70

MMStar

SEEDBench_IMG

AI2D

Threshold 0.95

0.65

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Sample Percentage

(b) Top-50 PLCC redundancy.

1.00

0.95

0.90

0.85

MetricValue

0.80

BLINK

0.75

ScienceQA

MMMU

RealWorldQA

MMBench

0.70

MMStar

SEEDBench_IMG

AI2D

Threshold 0.95

0.65

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Sample Percentage

(e) Bottom-50 PLCC redundancy.

1.00

0.95

0.90

MetricValue

0.85

0.80

BLINK

0.75

ScienceQA

MMMU

RealWorldQA

MMBench

0.70

MMStar

SEEDBench_IMG

AI2D

Threshold 0.95

0.65

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Sample Percentage

(c) Top-50 R2 redundancy.

1.00

0.95

0.90

0.85

MetricValue

0.80

BLINK

0.75

ScienceQA

MMMU

RealWorldQA

MMBench

0.70

MMStar

SEEDBench_IMG

AI2D

Threshold 0.95

0.65

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Sample Percentage

(f) Bottom-50 R2 redundancy.

- Figure 6: Benchmark-specific instance redundancy for (a) Top-50 MLLMs and (b) Bottom-50 MLLMs. The benchmarks include BLINK (Fu et al., 2025), ScienceQA (Lu et al., 2022), MMMU (Yue et al., 2024), RealWorldQA (xAI, 2024), MMBench (Liu et al., 2025), MMStar (Chen et al., 2024a), SeedBench_IMG (Li et al., 2023), and AI2D (Kembhavi et al., 2016). The selection of the Top-50 and Bottom-50 MLLMs is based on the corresponding benchmark.

instances. Consequently, the redundancy of benchmark instances correlates strongly with the capability of the MLLMs being evaluated: the stronger the MLLMs, the lower the redundancy of the benchmark instances.

From the benchmark-specific results (Fig. 6), the redundancy gap between Top-50 and Bottom50 MLLMs remains consistent across different benchmarks. Further examination reveals considerable variation in redundancy levels between benchmarks. For example, in the Top-50 redundancy analysis, RealWorldQA (xAI, 2024) demonstrates relatively low redundancy, requiring nearly 80% of the instances to reach saturation, while other benchmarks require far fewer. However, for Bottom-50 MLLMs, redundancy levels across benchmarks increase significantly, and the differences between them narrow. This illustrates that benchmark redundancy is more prominent when evaluating less capable MLLMs.

It is important to note that the conclusions above are based on the statistical analysis of mainstream benchmarks. Specialized benchmarks, with unique design goals or tasks, require case-by-case analyses to assess their instance redundancy accurately. Therefore, while these results provide general insights into redundancy trends for standard benchmarks, further evaluation is necessary for niche or task-specific benchmarks.

3.3 Exploring Cross-Benchmark Redundancy To analyze cross-benchmark redundancy, we focus on the Math domain, specifically examining several popular mathematical benchmarks: MathVista (Lu et al., 2023), MathVision (Zhang et al., 2025), MathVerse (Wang et al., 2024a), and DynaMath (Zou et al., 2024). We utilize the available evaluation results of 37 MLLMs listed on the OpenCompass Reasoning Leaderboard3 and assess their ranking performance across these math benchmarks. The corresponding heatmap is presented in Fig. 8. The results reveal that, although all four benchmarks are designed to evaluate the mathematical abilities of MLLMs, the correlations between them are not particularly strong. Among them, MathVista (Lu et al., 2023) exhibits the least redundancy, showing the lowest correlation with the other benchmarks. In contrast, MathVerse and MathVision demonstrate high redundancy, indicating a strong correlation with other benchmarks. These differences suggest varying levels of overlap in their evaluation focus areas.

To better understand the variability across benchmarks, we analyze their task distributions. While MathVerse and MathVision are focused on standard mathematical tasks, resulting in the highest correla-

3https://huggingface.co/spaces/opencompass/ Open_LMM_Reasoning_Leaderboard

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Q: Subtract all tiny purple shiny cubes. Subtract all large purple balls. How many objects are left? A: 9

Q: In the diagram of the food web shown what will most directly be affected by the loss of the trees? A: horses

Q: What is the lowest value in blue bar? A: 7

Q: Was this a square pizza? A: no

(a) (b) (c) (d)

- Figure 7: Examples of tasks excluded from the MathVista benchmark. (a), (b), and (c) showcase tasks derived from the general-vqa category, including Scientific Figure Understanding, General VQA, and Chart/Table/Diagram QA. Panel (d) presents questions extracted from the CLEVR dataset but categorized as math-targeted-vqa.

[Figure 20]

tion and substantial overlap with other benchmarks, MathVista includes 30%-40% of questions outside traditional mathematics, such as tasks related to Scientific Figure Understanding, General VQA, and Chart/Table/Diagram QA (see Fig. 7(a)(b)(c) for examples). As discussed in Sec. 2.3, low redundancy can arise from unique elements specific to a domain or from irrelevant tasks, which we consider “noise" within the dataset. For instance, general VQA tasks, while broadly useful, have limited relevance to assessing mathematical ability and contribute to this noise. To quantify the impact, we remove general VQA tasks from MathVista and recalculate its redundancy with other benchmarks. After this refinement, the redundancy between MathVista and other mathematical benchmarks significantly increases, aligning more closely with their task profiles. Additionally, we identify and exclude CLEVR-derived questions categorized as math-targeted vqa within MathVista, which also have limited relevance to mathematical capabilities (examples in Fig. 7(d)). This further increases overlap with specialized mathematical benchmarks, demonstrating that removing irrelevant tasks improves alignment and reduces noise.

Figure 8: Cross-benchmark redundancy map. MathVision and MathVersion are more focused on the core domain of mathematics (with relatively higher redundancy across other math benchmarks), making them more suitable for benchmarking the mathematical capabilities of MLLMs in a narrow sense.

##### 4 Conclusion

In conclusion, this paper addresses the pervasive issue of redundancy in MLLM benchmarks, impacting both the effectiveness and efficiency of model evaluation. We identify redundancy at three levels: dimension, instance, and cross-benchmark redundancy, and propose a framework with actionable guidelines to improve benchmark design. By promoting the independence of dimensions, optimizing instance counts, and ensuring purposeful redundancy within specific domains, our framework streamlines evaluations and enhances reliability. Case studies further demonstrate its utility in refining current practices, paving the way for more efficient and accurate MLLM assessments.

Therefore, we propose the following principles for benchmark design within a domain:

- • A benchmark intended to broadly assess model performance in one domain should demonstrate relatively high redundancy with other in-domain benchmarks, reflecting comprehensive coverage of diverse sub-capabilities.
- • A specialized benchmark should display lower redundancy with other benchmarks, focusing on distinct capabilities to fill the vacancy, complement broader assessments, and provide a unique perspective on specific topics in a domain.

- 5 Limitations The limitations of this work is as follows:

- • The assumption that MLLM performance rankings should show strong correlation when evaluating similar capabilities may not always hold. In some cases, performance on seemingly similar tasks could diverge due to subtle task differences, domain-specific nuances, or differences in model strengths.
- • The use of correlation metrics (SRCC, PLCC, and R²) to quantify redundancy may be limited in capturing the full complexity of model performance across different tasks and domains. These metrics may not adequately account for differences in task difficulty, model behavior under various conditions, or the impact of outliers.
- • The redundancy value is not fixed when using different selections of MLLMs for calculation. This bias could result in misleading conclusions about the redundancy or uniqueness of certain benchmarks.

- 6 Acknowledgements

This research was partly supported by grants of National Natural Science Foundation of China (NSFC, Grant No. 62171281), Science and Technology Commission of Shanghai Municipality (STCSM, Grant No. 20DZ1200203, 2021SHZDZX0102), National Key R&D Program of China (No.2022ZD0161600), the Shanghai Postdoctoral Excellence Program (No.2023023), China Postdoctoral Science Fund (No.2024M751559), and Shanghai Artificial intelligence Laboratory.

##### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433.

Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor C˘arbune, Jason Lin, Jindong Chen, and Abhanshu Sharma. 2024. Screenai: A vision-language model for ui and infographics understanding. arXiv preprint arXiv:2402.04615.

Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. 2010. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. 2024a. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024b. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. 2024. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. 2024. Mme: A comprehensive evaluation benchmark for multimodal large language models.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, WeiChiu Ma, and Ranjay Krishna. 2025. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. 2023. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. arXiv preprint arXiv:2310.14566.

Jan Hauke and Tomasz Kossowski. 2011. Comparison of values of pearson’s and spearman’s correlation coefficients on the same sets of data. Quaestiones geographicae, 30(2):87–93.

Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. 2024. Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22170–22183.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning

and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. 2016. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. 2024a. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. 2024b. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790.

Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. 2024c. Seedbench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125.

Chunyi Li, Jianbo Zhang, Zicheng Zhang, Haoning Wu, Yuan Tian, Wei Sun, Guo Lu, Xiaohong Liu, Xiongkuo Min, Weisi Lin, et al. 2024d. R-bench: Are your large multimodal model robust to real-world corruptions? arXiv preprint arXiv:2410.05474.

Xiang Li, Jian Ding, and Mohamed Elhoseiny. 2024e. Vrsbench: A versatile vision-language benchmark dataset for remote sensing image understanding. arXiv preprint arXiv:2406.12384.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruction tuning. Advances in neural information processing systems, 36.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. 2025. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer.

Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. 2023a. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895.

Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. 2023b. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS).

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209.

Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. 2019. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–952. IEEE.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.

Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. 2024a. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024b. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. 2023. Qbench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181.

xAI. 2024. Realworldqa dataset. Available at https://huggingface.co/datasets/xai-org/ RealworldQA.

Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. 2024. Gpt4tools: Teaching large language model to use tools via self-instruction. Advances in Neural Information Processing Systems, 36.

Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. 2024. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023a. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023b. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. 2024. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. 2025. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision.

Zicheng Zhang, Haoning Wu, Chunyi Li, Yingjie Zhou, Wei Sun, Xiongkuo Min, Zijian Chen, Xiaohong Liu, Weisi Lin, and Guangtao Zhai. 2024a. A-bench: Are lmms masters at evaluating ai-generated images? arXiv preprint arXiv:2406.03070.

Zicheng Zhang, Haoning Wu, Erli Zhang, Guangtao Zhai, and Weisi Lin. 2024b. Q-bench: A benchmark for multi-modal foundation models on low-level vision from single images to pairs. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, et al. 2024. Apollo: An exploration of video understanding in large multimodal models. arXiv preprint arXiv:2412.10360.

Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. 2024. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836.

##### A Metrics Equation

To evaluate the consistency and accuracy of predictions, we employ three widely used metrics: the Spearman Rank Correlation Coefficient (SRCC), the Pearson Linear Correlation Coefficient (PLCC), and the Coefficient of Determination (R2). These metrics provide complementary perspectives on model performance, capturing rank-based, linear, and variance-explained relationships, respectively. The mathematical definitions are detailed below.

1) The SRCC measures the rank-based relationship between predicted and true values. It is defined as:

6 ni=1 d2i n(n2 − 1)

SRCC = 1 −

, where:

di = rank(xi) − rank(yi),

and n is the number of data points. A higher SRCC indicates a stronger monotonic relationship between the rankings of predicted and ground truth values.

2) The PLCC quantifies the linear relationship between predicted and true values. It is computed as:

n i=1(xi − x¯)(yi − y¯)

PLCC =

,

n i=1(xi − x¯)2 ni=1(yi − y¯)2

where:

- • xi and yi are the data points,
- • x¯ and y¯ are the means of x and y, respectively.

- A higher PLCC indicates a stronger linear relationship between predicted and ground truth values.

3) The R2 score represents the proportion of variance in the ground truth values that is explained by the predictions. It is defined as:

R2 = 1 −

n i=1(yi − yˆi)2 n i=1(yi − y¯)2

where:

- • yi are the ground truth values,
- • yˆi are the predicted values,
- • y¯ is the mean of the ground truth values. An R2 score closer to 1 indicates a better fit between the predictions and the ground truth.

- B Extra Dimensions Redundancy Maps

We present the dimension redundancy maps for AI2D (Kembhavi et al., 2016) and SEED-Bench (Li et al., 2024c), as shown in Fig. 9 and Fig. 10.

###### 1. Key Observations from the Redundancy Maps:

- • In Fig. 9, it is evident that the dimension ‘lifeCycles’ exhibits the highest redundancy, particularly with ‘typesOf’.
- • Similarly, in Fig. 10, the ‘Instance Identity’ dimension shows the highest redundancy and is most closely related to ‘Scene Understanding’.

###### 2. Trends in Top-50 vs. Bottom-50 Redundancy:

- • A clear pattern emerges when comparing the Top-50 and Bottom-50 redundancy maps. Nearly all Bottom-50 dimensions display significantly higher redundancy than their Top-50 counterparts. This observation supports our conclusion that dimensions tend to exhibit greater redundancy for Bottom50 MLLMs compared to Top-50 MLLMs.
- • This phenomenon can be attributed to the overall underperformance of Bottom-50 MLLMs across various capabilities. As these models begin to improve, enhancements in their foundational abilities often lead to simultaneous progress across multiple dimensions. This results in a high degree of similarity in performance rankings, contributing to elevated dimensional redundancy.
- • In contrast, Top-50 MLLMs already possess relatively strong foundational capabilities. As a result, more challenging tasks across different dimensions introduce greater differentiation, reducing redundancy and creating more distinct performance profiles.

###### 3. Implications for Redundancy Analysis:

• To ensure a reasonable and accurate evaluation during redundancy analysis, it is crucial to exclude MLLMs with consistently poor performance. Including such models could skew the analysis by disproportionately inflating redundancy, as their universal underperformance does not provide meaningful insights into inter-dimensional relationships.

##### C Redundancy Practice Recommendations

To ensure benchmarks are reliable and efficient, we recommend incorporating redundancy detec-

tion into the benchmark design process after its initial testing on a set of MLLMs. This critical step identifies potential redundancies across dimensions/instances/cross-benchmark overlaps, leading to more precise and meaningful evaluations.

###### 1. Dimension Redundancy Check.

Calculate the dimensional redundancy within the benchmark, with particular attention to dimensions exhibiting overall high redundancy. Analyze the redundancy heatmap to identify pairs of dimensions with exceptionally strong correlations, as these may indicate overlapping capabilities being assessed. For such cases, evaluate whether these dimensions are truly necessary or whether they assess similar or redundant skills.

###### 2. Instance Redundancy Check.

Compute the instance redundancy curve to determine whether a smaller subset of benchmark instances can produce results comparable to the full instance set. If significant instance redundancy is identified, the benchmark should be reviewed, and redundant instances should be reduced. This not only streamlines the evaluation process but also optimizes resource usage without compromising the accuracy of results.

###### 3. Cross-benchmark Redundancy Check.

If the benchmark is intended to serve as a representative for a specific domain, measure its cross-benchmark redundancy relative to other benchmarks within the domain. Higher redundancy indicates stronger representativeness, making it a reliable choice for tasks requiring domain coverage. Conversely, if the goal is to fill a vacancy in the specific domain (e.g., focusing on a specific topic in mathematics that is not covered by previous benchmarks) maintaining low redundancy is a more favorable choice. For use cases focusing on core capabilities within a specific domain under limited resources, it is recommended to select the benchmark with the highest cross-benchmark redundancy. This ensures that the benchmark comprehensively covers the essential skills while minimizing unnecessary overlaps.

1.0

[Figure 21]

1.00 0.05 0.19 0.17 0.15 0.15 -0.02 0.06 0.21 -0.10 -0.04 0.04 0.11 -0.03 0.10

atomStructure

- 0.05 1.00 0.30 0.41 0.53 0.65 0.44 0.36 0.35 0.35 0.24 0.08 0.44 0.10 0.43

0.19 0.30 1.00 0.52 0.66 0.48 0.47 0.24 0.57 0.47 0.30 0.31 0.59 0.07 0.32

0.17 0.41 0.52 1.00 0.53 0.34 0.53 0.25 0.53 0.46 0.32 0.12 0.60 -0.14 0.38

0.15 0.53 0.66 0.53 1.00 0.78 0.78 0.51 0.73 0.63 0.53 0.40 0.84 0.37 0.49

0.15 0.65 0.48 0.34 0.78 1.00 0.59 0.55 0.64 0.59 0.22 0.34 0.55 0.18 0.41

-0.02 0.44 0.47 0.53 0.78 0.59 1.00 0.41 0.69 0.66 0.64 0.38 0.76 0.35 0.52

- 0.06 0.36 0.24 0.25 0.51 0.55 0.41 1.00 0.29 0.31 0.26 0.17 0.35 0.10 0.27

eclipses

faultsEarthquakes

0.8

foodChainsWebs

lifeCycles

0.6

moonPhaseEquinox

partsOfA

partsOfTheEarth

0.4

0.21 0.35 0.57 0.53 0.73 0.64 0.69 0.29 1.00 0.62 0.42 0.29 0.64 0.08 0.34

photosynthesisRespiration

- -0.10 0.35 0.47 0.46 0.63 0.59 0.66 0.31 0.62 1.00 0.43 0.22 0.60 0.36 0.23
- -0.04 0.24 0.30 0.32 0.53 0.22 0.64 0.26 0.42 0.43 1.00 0.26 0.61 0.46 0.18

0.04 0.08 0.31 0.12 0.40 0.34 0.38 0.17 0.29 0.22 0.26 1.00 0.37 0.24 0.20

0.11 0.44 0.59 0.60 0.84 0.55 0.76 0.35 0.64 0.60 0.61 0.37 1.00 0.44 0.43

- -0.03 0.10 0.07 -0.14 0.37 0.18 0.35 0.10 0.08 0.36 0.46 0.24 0.44 1.00 0.07

rockCycle

rockStrata

0.2

solarSystem

typesOf

volcano

0.0

0.10 0.43 0.32 0.38 0.49 0.41 0.52 0.27 0.34 0.23 0.18 0.20 0.43 0.07 1.00

waterCNPCycle

atomStructurefaultsEarthquakeseclipsesfoodChainsWebsmoonPhaseEquinoxlifeCycles photosynthesisRespirationpartsOfApartsOfTheEarth rockCyclerockStratasolarSystemtypesOfvolcanowaterCNPCycle

(a) 50+ SRCC dimensions redundancy.

1.0

[Figure 22]

1.00 0.44 0.41 0.47 0.59 0.54 0.52 0.43 0.55 0.48 0.47 0.51 0.46 0.58 0.26

atomStructure

0.44 1.00 0.48 0.77 0.75 0.64 0.71 0.60 0.62 0.61 0.66 0.73 0.67 0.53 0.45

eclipses

0.41 0.48 1.00 0.46 0.56 0.51 0.52 0.42 0.50 0.61 0.51 0.45 0.58 0.46 0.32

faultsEarthquakes

0.8

- 0.47 0.77 0.46 1.00 0.75 0.69 0.85 0.75 0.65 0.56 0.66 0.86 0.78 0.54 0.43

0.59 0.75 0.56 0.75 1.00 0.80 0.74 0.70 0.74 0.71 0.80 0.70 0.81 0.64 0.52

- 0.54 0.64 0.51 0.69 0.80 1.00 0.80 0.68 0.70 0.63 0.76 0.75 0.80 0.56 0.41

0.52 0.71 0.52 0.85 0.74 0.80 1.00 0.74 0.59 0.59 0.78 0.86 0.89 0.58 0.42

0.43 0.60 0.42 0.75 0.70 0.68 0.74 1.00 0.44 0.50 0.70 0.74 0.66 0.38 0.33

- 0.55 0.62 0.50 0.65 0.74 0.70 0.59 0.44 1.00 0.58 0.61 0.69 0.64 0.63 0.52

- 0.48 0.61 0.61 0.56 0.71 0.63 0.59 0.50 0.58 1.00 0.53 0.53 0.61 0.50 0.50

foodChainsWebs

lifeCycles

0.6

moonPhaseEquinox

partsOfA

partsOfTheEarth

0.4

photosynthesisRespiration

rockCycle

0.47 0.66 0.51 0.66 0.80 0.76 0.78 0.70 0.61 0.53 1.00 0.68 0.83 0.55 0.42

rockStrata

0.2

0.51 0.73 0.45 0.86 0.70 0.75 0.86 0.74 0.69 0.53 0.68 1.00 0.79 0.47 0.48

solarSystem

0.46 0.67 0.58 0.78 0.81 0.80 0.89 0.66 0.64 0.61 0.83 0.79 1.00 0.59 0.44

typesOf

0.58 0.53 0.46 0.54 0.64 0.56 0.58 0.38 0.63 0.50 0.55 0.47 0.59 1.00 0.46

volcano

0.0

0.26 0.45 0.32 0.43 0.52 0.41 0.42 0.33 0.52 0.50 0.42 0.48 0.44 0.46 1.00

waterCNPCycle

atomStructurefaultsEarthquakeseclipsesfoodChainsWebsmoonPhaseEquinoxlifeCycles photosynthesisRespirationpartsOfApartsOfTheEarth rockCyclerockStratasolarSystemtypesOfvolcanowaterCNPCycle

(d) 50− SRCC dimensions redundancy.

1.0

[Figure 23]

1.00 0.10 0.14 0.19 0.21 0.25 0.01 0.06 0.24 -0.10 -0.02 0.02 0.14 0.02 0.19

atomStructure

0.10 1.00 0.38 0.41 0.54 0.67 0.42 0.29 0.41 0.41 0.28 0.04 0.47 0.12 0.37

eclipses

0.14 0.38 1.00 0.46 0.65 0.54 0.49 0.23 0.52 0.50 0.39 0.32 0.63 0.11 0.37

faultsEarthquakes

0.8

0.19 0.41 0.46 1.00 0.42 0.32 0.51 0.19 0.39 0.46 0.37 0.09 0.55 -0.13 0.35

foodChainsWebs

0.21 0.54 0.65 0.42 1.00 0.79 0.75 0.48 0.74 0.58 0.52 0.37 0.83 0.36 0.50

lifeCycles

0.6

0.25 0.67 0.54 0.32 0.79 1.00 0.60 0.49 0.68 0.58 0.26 0.36 0.62 0.20 0.46

moonPhaseEquinox

- 0.01 0.42 0.49 0.51 0.75 0.60 1.00 0.36 0.67 0.66 0.63 0.39 0.75 0.32 0.49

0.06 0.29 0.23 0.19 0.48 0.49 0.36 1.00 0.30 0.26 0.22 0.16 0.28 0.12 0.20

0.24 0.41 0.52 0.39 0.74 0.68 0.67 0.30 1.00 0.51 0.44 0.25 0.63 0.13 0.31

- -0.10 0.41 0.50 0.46 0.58 0.58 0.66 0.26 0.51 1.00 0.41 0.20 0.59 0.36 0.23
- -0.02 0.28 0.39 0.37 0.52 0.26 0.63 0.22 0.44 0.41 1.00 0.31 0.66 0.43 0.23

- 0.02 0.04 0.32 0.09 0.37 0.36 0.39 0.16 0.25 0.20 0.31 1.00 0.40 0.21 0.21

partsOfA

partsOfTheEarth

0.4

photosynthesisRespiration

rockCycle

rockStrata

0.2

solarSystem

0.14 0.47 0.63 0.55 0.83 0.62 0.75 0.28 0.63 0.59 0.66 0.40 1.00 0.41 0.49

typesOf

0.02 0.12 0.11 -0.13 0.36 0.20 0.32 0.12 0.13 0.36 0.43 0.21 0.41 1.00 0.12

volcano

0.0

0.19 0.37 0.37 0.35 0.50 0.46 0.49 0.20 0.31 0.23 0.23 0.21 0.49 0.12 1.00

waterCNPCycle

atomStructurefaultsEarthquakeseclipsesfoodChainsWebsmoonPhaseEquinoxlifeCycles photosynthesisRespirationpartsOfApartsOfTheEarth rockCyclerockStratasolarSystemtypesOfvolcanowaterCNPCycle

(b) 50+ PLCC dimensions redundancy.

1.0

[Figure 24]

1.00 0.47 0.41 0.47 0.56 0.50 0.48 0.43 0.52 0.48 0.48 0.54 0.46 0.57 0.20

atomStructure

- 0.47 1.00 0.62 0.82 0.82 0.76 0.79 0.66 0.70 0.68 0.73 0.80 0.77 0.61 0.43

0.41 0.62 1.00 0.70 0.76 0.74 0.75 0.56 0.72 0.73 0.67 0.60 0.77 0.65 0.44

- 0.47 0.82 0.70 1.00 0.88 0.85 0.92 0.78 0.75 0.71 0.75 0.88 0.88 0.64 0.45

- 0.56 0.82 0.76 0.88 1.00 0.90 0.89 0.75 0.80 0.80 0.84 0.79 0.92 0.74 0.54

0.50 0.76 0.74 0.85 0.90 1.00 0.89 0.75 0.79 0.74 0.79 0.78 0.89 0.70 0.47

0.48 0.79 0.75 0.92 0.89 0.89 1.00 0.77 0.75 0.73 0.81 0.85 0.93 0.70 0.49

0.43 0.66 0.56 0.78 0.75 0.75 0.77 1.00 0.56 0.58 0.75 0.73 0.72 0.46 0.26

0.52 0.70 0.72 0.75 0.80 0.79 0.75 0.56 1.00 0.70 0.66 0.72 0.76 0.72 0.54

0.48 0.68 0.73 0.71 0.80 0.74 0.73 0.58 0.70 1.00 0.60 0.60 0.72 0.61 0.44

0.48 0.73 0.67 0.75 0.84 0.79 0.81 0.75 0.66 0.60 1.00 0.73 0.84 0.61 0.41

0.54 0.80 0.60 0.88 0.79 0.78 0.85 0.73 0.72 0.60 0.73 1.00 0.83 0.55 0.43

0.46 0.77 0.77 0.88 0.92 0.89 0.93 0.72 0.76 0.72 0.84 0.83 1.00 0.72 0.51

- 0.57 0.61 0.65 0.64 0.74 0.70 0.70 0.46 0.72 0.61 0.61 0.55 0.72 1.00 0.50

eclipses

faultsEarthquakes

0.8

foodChainsWebs

lifeCycles

0.6

moonPhaseEquinox

partsOfA

partsOfTheEarth

0.4

photosynthesisRespiration

rockCycle

rockStrata

0.2

solarSystem

typesOf

volcano

0.0

0.20 0.43 0.44 0.45 0.54 0.47 0.49 0.26 0.54 0.44 0.41 0.43 0.51 0.50 1.00

waterCNPCycle

atomStructurefaultsEarthquakeseclipsesfoodChainsWebsmoonPhaseEquinoxlifeCycles photosynthesisRespirationpartsOfApartsOfTheEarth rockCyclerockStratasolarSystemtypesOfvolcanowaterCNPCycle

(e) 50− PLCC dimensions redundancy.

1.0

[Figure 25]

1.00 0.01 0.02 0.04 0.04 0.06 0.00 0.00 0.06 0.01 0.00 0.00 0.02 0.00 0.04

atomStructure

- 0.01 1.00 0.14 0.17 0.29 0.45 0.18 0.08 0.17 0.17 0.08 0.00 0.22 0.01 0.14
- 0.02 0.14 1.00 0.21 0.42 0.29 0.24 0.05 0.27 0.25 0.15 0.10 0.40 0.01 0.14

eclipses

faultsEarthquakes

0.8

0.04 0.17 0.21 1.00 0.18 0.10 0.26 0.04 0.16 0.21 0.14 0.01 0.31 0.02 0.13

foodChainsWebs

0.04 0.29 0.42 0.18 1.00 0.62 0.56 0.23 0.54 0.34 0.27 0.14 0.69 0.13 0.25

lifeCycles

0.6

0.06 0.45 0.29 0.10 0.62 1.00 0.36 0.24 0.46 0.33 0.07 0.13 0.38 0.04 0.21

moonPhaseEquinox

0.00 0.18 0.24 0.26 0.56 0.36 1.00 0.13 0.45 0.44 0.40 0.15 0.56 0.10 0.24

partsOfA

- 0.00 0.08 0.05 0.04 0.23 0.24 0.13 1.00 0.09 0.07 0.05 0.03 0.08 0.01 0.04

0.06 0.17 0.27 0.16 0.54 0.46 0.45 0.09 1.00 0.26 0.20 0.06 0.40 0.02 0.10

- 0.01 0.17 0.25 0.21 0.34 0.33 0.44 0.07 0.26 1.00 0.17 0.04 0.35 0.13 0.05

0.00 0.08 0.15 0.14 0.27 0.07 0.40 0.05 0.20 0.17 1.00 0.10 0.44 0.19 0.05

- 0.00 0.00 0.10 0.01 0.14 0.13 0.15 0.03 0.06 0.04 0.10 1.00 0.16 0.04 0.04

0.02 0.22 0.40 0.31 0.69 0.38 0.56 0.08 0.40 0.35 0.44 0.16 1.00 0.16 0.24

- 0.00 0.01 0.01 0.02 0.13 0.04 0.10 0.01 0.02 0.13 0.19 0.04 0.16 1.00 0.02

partsOfTheEarth

0.4

photosynthesisRespiration

rockCycle

rockStrata

0.2

solarSystem

typesOf

volcano

0.0

0.04 0.14 0.14 0.13 0.25 0.21 0.24 0.04 0.10 0.05 0.05 0.04 0.24 0.02 1.00

waterCNPCycle

atomStructurefaultsEarthquakeseclipsesfoodChainsWebsmoonPhaseEquinoxlifeCycles photosynthesisRespirationpartsOfApartsOfTheEarth rockCyclerockStratasolarSystemtypesOfvolcanowaterCNPCycle

(c) 50+ R2 dimensions redundancy.

1.0

[Figure 26]

1.00 0.22 0.17 0.22 0.32 0.25 0.23 0.19 0.27 0.23 0.23 0.29 0.21 0.32 0.04

atomStructure

- 0.22 1.00 0.39 0.67 0.67 0.57 0.62 0.43 0.49 0.46 0.54 0.63 0.60 0.37 0.18

0.17 0.39 1.00 0.48 0.58 0.55 0.57 0.31 0.51 0.53 0.45 0.36 0.59 0.43 0.19

- 0.22 0.67 0.48 1.00 0.77 0.72 0.86 0.60 0.56 0.50 0.56 0.77 0.77 0.40 0.20

0.32 0.67 0.58 0.77 1.00 0.82 0.79 0.57 0.65 0.64 0.71 0.62 0.85 0.54 0.29

0.25 0.57 0.55 0.72 0.82 1.00 0.79 0.56 0.62 0.55 0.63 0.61 0.79 0.49 0.22

- 0.23 0.62 0.57 0.86 0.79 0.79 1.00 0.60 0.56 0.53 0.66 0.73 0.86 0.49 0.24

0.19 0.43 0.31 0.60 0.57 0.56 0.60 1.00 0.31 0.34 0.57 0.54 0.52 0.21 0.07

0.27 0.49 0.51 0.56 0.65 0.62 0.56 0.31 1.00 0.48 0.44 0.52 0.58 0.51 0.29

- 0.23 0.46 0.53 0.50 0.64 0.55 0.53 0.34 0.48 1.00 0.36 0.37 0.52 0.38 0.20

eclipses

faultsEarthquakes

0.8

foodChainsWebs

lifeCycles

0.6

moonPhaseEquinox

partsOfA

partsOfTheEarth

0.4

photosynthesisRespiration

rockCycle

0.23 0.54 0.45 0.56 0.71 0.63 0.66 0.57 0.44 0.36 1.00 0.54 0.70 0.38 0.16

rockStrata

0.2

0.29 0.63 0.36 0.77 0.62 0.61 0.73 0.54 0.52 0.37 0.54 1.00 0.68 0.30 0.18

solarSystem

0.21 0.60 0.59 0.77 0.85 0.79 0.86 0.52 0.58 0.52 0.70 0.68 1.00 0.52 0.26

typesOf

0.32 0.37 0.43 0.40 0.54 0.49 0.49 0.21 0.51 0.38 0.38 0.30 0.52 1.00 0.25

volcano

0.0

0.04 0.18 0.19 0.20 0.29 0.22 0.24 0.07 0.29 0.20 0.16 0.18 0.26 0.25 1.00

waterCNPCycle

atomStructurefaultsEarthquakeseclipsesfoodChainsWebsmoonPhaseEquinoxlifeCycles photosynthesisRespirationpartsOfApartsOfTheEarth rockCyclerockStratasolarSystemtypesOfvolcanowaterCNPCycle

(f) 50− R2 dimensions redundancy.

- Figure 9: Visualizations of dimensions redundancy for AI2D (Kembhavi et al., 2016) on Top-50 and Bottom-50 MLLMs (marked as 50+ and 50−).

Instance Attributes Instance

Identity Instance Interaction

Instance Location Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

Instance Attributes

Instance Identity

Instance Interaction

Instance Location

Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

1.00 0.64 0.12 0.44 0.35 0.58 0.41 0.41 0.38

0.64 1.00 0.49 0.74 0.47 0.79 0.56 0.14 0.54

0.12 0.49 1.00 0.53 0.40 0.30 0.51 -0.01 0.37

0.44 0.74 0.53 1.00 0.45 0.63 0.49 0.14 0.34

0.35 0.47 0.40 0.45 1.00 0.33 0.48 0.14 0.18

0.58 0.79 0.30 0.63 0.33 1.00 0.48 0.15 0.40

0.41 0.56 0.51 0.49 0.48 0.48 1.00 0.28 0.30

0.41 0.14 -0.01 0.14 0.14 0.15 0.28 1.00 0.12

0.38 0.54 0.37 0.34 0.18 0.40 0.30 0.12 1.00

[Figure 27]

0.0

0.2

0.4

0.6

0.8

1.0

(a) 50+ SRCC dimensions redundancy.

Instance Attributes Instance

Identity Instance Interaction

Instance Location Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

Instance Attributes

Instance Identity

Instance Interaction

Instance Location

Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

1.00 0.64 0.11 0.38 0.36 0.59 0.43 0.41 0.34

0.64 1.00 0.49 0.71 0.44 0.78 0.60 0.12 0.53

0.11 0.49 1.00 0.53 0.42 0.33 0.54 -0.04 0.42

0.38 0.71 0.53 1.00 0.51 0.57 0.62 0.09 0.39

0.36 0.44 0.42 0.51 1.00 0.27 0.56 0.27 0.19

0.59 0.78 0.33 0.57 0.27 1.00 0.53 0.08 0.41

0.43 0.60 0.54 0.62 0.56 0.53 1.00 0.24 0.40

0.41 0.12 -0.04 0.09 0.27 0.08 0.24 1.00 0.14

0.34 0.53 0.42 0.39 0.19 0.41 0.40 0.14 1.00

[Figure 28]

0.0

0.2

0.4

0.6

0.8

1.0

(b) 50+ PLCC dimensions redundancy.

Instance Attributes Instance

Identity Instance Interaction

Instance Location Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

Instance Attributes

Instance Identity

Instance Interaction

Instance Location

Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

1.00 0.41 0.01 0.14 0.13 0.34 0.18 0.17 0.12

0.41 1.00 0.24 0.51 0.19 0.61 0.36 0.02 0.28

0.01 0.24 1.00 0.28 0.17 0.11 0.30 0.00 0.18

0.14 0.51 0.28 1.00 0.26 0.32 0.38 0.01 0.15

0.13 0.19 0.17 0.26 1.00 0.07 0.31 0.07 0.04

0.34 0.61 0.11 0.32 0.07 1.00 0.28 0.01 0.17

0.18 0.36 0.30 0.38 0.31 0.28 1.00 0.06 0.16

0.17 0.02 0.00 0.01 0.07 0.01 0.06 1.00 0.02

0.12 0.28 0.18 0.15 0.04 0.17 0.16 0.02 1.00

[Figure 29]

0.0

0.2

0.4

0.6

0.8

1.0

(c) 50+ R2 dimensions redundancy.

Instance Attributes Instance

Identity Instance Interaction

Instance Location Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

Instance Attributes

Instance Identity

Instance Interaction

Instance Location

Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

1.00 0.80 0.71 0.75 0.66 0.79 0.59 0.43 0.73

0.80 1.00 0.79 0.78 0.88 0.84 0.65 0.24 0.76

0.71 0.79 1.00 0.82 0.71 0.79 0.73 0.29 0.76

0.75 0.78 0.82 1.00 0.72 0.78 0.85 0.38 0.78

0.66 0.88 0.71 0.72 1.00 0.81 0.68 0.16 0.70

0.79 0.84 0.79 0.78 0.81 1.00 0.66 0.28 0.83

0.59 0.65 0.73 0.85 0.68 0.66 1.00 0.35 0.65

0.43 0.24 0.29 0.38 0.16 0.28 0.35 1.00 0.16

0.73 0.76 0.76 0.78 0.70 0.83 0.65 0.16 1.00

[Figure 30]

0.0

0.2

0.4

0.6

0.8

1.0

(d) 50− SRCC dimensions redundancy.

Instance Attributes Instance

Identity Instance Interaction

Instance Location Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

Instance Attributes

Instance Identity

Instance Interaction

Instance Location

Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

1.00 0.97 0.91 0.94 0.91 0.94 0.85 0.47 0.91

0.97 1.00 0.94 0.95 0.93 0.97 0.86 0.39 0.95

- 0.91 0.94 1.00 0.93 0.85 0.93 0.85 0.39 0.92

0.94 0.95 0.93 1.00 0.90 0.91 0.93 0.45 0.90

0.91 0.93 0.85 0.90 1.00 0.88 0.84 0.31 0.85

0.94 0.97 0.93 0.91 0.88 1.00 0.82 0.40 0.98

0.85 0.86 0.85 0.93 0.84 0.82 1.00 0.43 0.81

0.47 0.39 0.39 0.45 0.31 0.40 0.43 1.00 0.37

- 0.91 0.95 0.92 0.90 0.85 0.98 0.81 0.37 1.00

[Figure 31]

0.0

0.2

0.4

0.6

0.8

1.0

(e) 50− PLCC dimensions redundancy.

Instance Attributes Instance

Identity Instance Interaction

Instance Location Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

Instance Attributes

Instance Identity

Instance Interaction

Instance Location

Instances Counting

Scene Understanding

Spatial Relation

Text Understanding

Visual Reasoning

1.00 0.95 0.83 0.89 0.83 0.88 0.71 0.22 0.83

0.95 1.00 0.88 0.91 0.87 0.93 0.74 0.16 0.90

0.83 0.88 1.00 0.86 0.72 0.86 0.72 0.15 0.85

0.89 0.91 0.86 1.00 0.82 0.83 0.86 0.20 0.81

0.83 0.87 0.72 0.82 1.00 0.77 0.71 0.10 0.71

0.88 0.93 0.86 0.83 0.77 1.00 0.67 0.16 0.96

0.71 0.74 0.72 0.86 0.71 0.67 1.00 0.18 0.66

0.22 0.16 0.15 0.20 0.10 0.16 0.18 1.00 0.14

0.83 0.90 0.85 0.81 0.71 0.96 0.66 0.14 1.00

[Figure 32]

0.0

0.2

0.4

0.6

0.8

1.0

(f) 50− R2 dimensions redundancy.

- Figure 10: Visualizations of dimensions redundancy for SEED-Bench (Li et al., 2024c) on Top-50 and Bottom-50 MLLMs (marked as 50+ and 50−).

