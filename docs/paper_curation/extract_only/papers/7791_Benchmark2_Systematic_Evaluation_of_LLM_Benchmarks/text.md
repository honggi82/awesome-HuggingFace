## Benchmark2: Systematic Evaluation of LLM Benchmarks

Qi Qian1∗ Chengsong Huang2∗ Jingwen Xu1 Changze Lv1 Muling Wu1 Wenhao Liu3

Xiaohua Wang1 Zhenghua Wang1 Zisu Huang1 Muzhao Tian1 Jianhan Xu3 Kun Hu3 He-Da Wang3 Yao Hu3 Xuanjing Huang1† Xiaoqing Zheng1†

1College of Computer Science and Artificial Intelligence, Fudan University 2Washington University in St. Louis 3Xiaohongshu Inc. qqian23@m.fudan.edu.cn chengsong@wustl.edu {zhengxq, xjhuang}@fudan.edu.cn

# arXiv:2601.03986v1[cs.CL]7Jan2026

### Abstract

The rapid proliferation of benchmarks for evaluating large language models (LLMs) has created an urgent need for systematic methods to assess benchmark quality itself. We propose BENCHMARK2, a comprehensive framework comprising three complementary metrics: (1) Cross-Benchmark Ranking Consistency, measuring whether a benchmark produces model rankings aligned with peer benchmarks; (2) Discriminability Score, quantifying a benchmark’s ability to differentiate between models; and (3) Capability Alignment Deviation, identifying problematic instances where stronger models fail but weaker models succeed within the same model family. We conduct extensive experiments across 15 benchmarks spanning mathematics, reasoning, and knowledge domains, evaluating 11 LLMs across four model families. Our analysis reveals significant quality variations among existing benchmarks and demonstrates that selective benchmark construction based on our metrics can achieve comparable evaluation performance with substantially reduced test sets.

### 1 Introduction

The evaluation of large language models (LLMs) has become increasingly important as these models are deployed across diverse real-world applications. Benchmarks serve as the primary instruments for measuring model capabilities, guiding both research directions and practical deployment decisions. However, the explosive growth in the number of benchmarks—with hundreds now available across domains such as mathematics, reasoning, instruction following, and knowledge understanding—raises a fundamental question: How do we know if a benchmark itself is good?

Although benchmarks serve an important role in the field, surprisingly little attention has been paid to the benchmark quality. Current practice often treats benchmarks as ground truth without

questioning their reliability or validity. This oversight can lead to several problems, as illustrated in Figure 1: (1) Ranking Inconsistency—different benchmarks may produce conflicting model rankings, making it unclear which benchmark to trust; (2) Low Discriminative Power—some benchmarks fail to differentiate between models of varying capabilities, clustering all models within a narrow performance range; and (3) RankInconsistent Items—individual test instances may exhibit counter-intuitive behavior where stronger models fail but weaker models succeed.

Consider a concrete example: if Benchmark A ranks Model X above Model Y in mathematical reasoning, while Benchmarks B and C consistently show the opposite ranking, should we trust Benchmark A? Similarly, if a benchmark shows minimal performance differences between a state-of-the-art model and a much smaller model, does this indicate that these two models have similar capabilities, or does it suggest that the benchmark cannot effectively distinguish between them?

To address these challenges, we propose BENCHMARK2, a novel framework for evaluating benchmark quality through three complementary metrics (Figure 1, bottom row): Cross-Benchmark Ranking Consistency (CBRC), which measures ranking correlation with external domain benchmarks, Discriminability Score (DS), which measures the benchmark’s ability to distinguish between models of varying capabilities; and Capability Alignment Deviation (CAD), which penalizes counterintuitive instances where weaker models outperform stronger ones within the same family, ensuring hierarchical consistency.

We conduct comprehensive experiments across 15 widely-used benchmarks spanning three major domains (mathematics, general reasoning, and knowledge & understanding), evaluating 11 LLMs across four model families with clear capability hierarchies. Our analysis reveals substantial quality

|Are these benchmarks truly reliable, discriminative, and consistent?<br><br>Benchmark A<br><br>Benchmark B<br><br>Benchmark C<br>Benchmark D 1 < 4 < 3 < 2<br><br><br>Ranking models by performance<br><br>1 < 2 < 3 < 4<br><br>1 < 2 < 3 < 4<br>1 < 2 < 4 < 3<br><br><br>Consistency Score A (High)<br>Consistency Score B (High)<br>Consistency Score C (Medium)<br><br><br>Benchmark2: Systematic Evaluation of LLM Benchmarks<br><br>Problem1: Ranking Inconsistency and Bias Problem2: Low Discriminative Power of Gaps Problem3: Prevalence of Rank-Inconsistant Items<br><br>[Figure 1]<br><br>Model 1 is the best!<br><br>Model 2 is superior!<br><br>Both are inferior to Model 3.<br><br>[Figure 2]<br><br>Hardly any difference between them. Maybe 1%?<br><br>Model 1 is 40% ahead of Model 2.<br><br>ItemItemii ItemItemii ItemItemii ItemItemii<br><br>Rank-Consistent items:<br><br>1 > 2<br><br>ItemItemii ItemItemii<br><br>Test result:<br><br>Rank-Inconsistant items:<br><br>2 > 1<br><br>Test result:<br><br>ItemItemij<br><br>✓<br><br>✗<br><br>[Figure 3]<br><br>Metric1: Cross-Benchmark Ranking Consistency Metric2: Discriminability Score Metric3: Capability Alignment Deviation<br><br>Calculating performance gaps based on benchmark results. Calculating the proportion of Rank-Inconsistent Items<br><br>Capability Alignment Deviation B = NumberInversionof comparisons<br><br>Models from the same family Larger Size = Better Performance for Item i in Benchmark B<br><br>for in<br><br>Test result (i, x)<br><br>Model x<br><br>All combinations from the same , Model y Model Family<br><br>,<br><br>:<br><br>:<br><br>Test result (i, y)<br><br>Model x<br><br>Model y<br><br>is weaker than if Testresult(i,x) > Testresult(i,y) :<br><br>Inversion += 1<br><br>Benchmark A<br><br>0.1 < 0.4 < 0.7 < 1.0 Discriminability<br><br>Score A (High)<br><br>+ 0.3 + 0.3 + 0.3<br><br>Benchmark C<br><br>0.5 < 0.6 < 0.7 < 0.8 Discriminability<br><br>Score C (Low)<br><br>+ 0.1 + 0.1 + 0.1<br><br>Benchmark B<br><br><br>0.2 < 0.4 < 0.7 < 0.9 Discriminability<br><br>Score B (Medium)<br><br>+ 0.2 + 0.3 + 0.2<br><br>Consistency Score D (Low)<br><br>computing consistency with other Benchmarks<br><br>Model 1, 2, 3, 4 are from different Model Family<br><br>Model-Agnostic Performance in Ascending Order|
|---|

- Figure 1: Overview of BENCHMARK2 framework. Top row: Three key problems with existing LLM benchmarks—ranking inconsistency across benchmarks, low discriminative power of performance gaps, and prevalence of rank-inconsistent test items. Bottom row: Our three complementary metrics addressing each problem—CrossBenchmark Ranking Consistency (CBRC) measures alignment with peer benchmarks, Discriminability Score (DS) quantifies performance gap magnitudes, and Capability Alignment Deviation (CAD) identifies items violating expected capability hierarchies within model families.

variations among existing benchmarks and identifies specific characteristics that distinguish highquality benchmarks from problematic ones.

Beyond benchmark quality assessment, we demonstrate a practical application of our framework: selective benchmark construction. By identifying high-quality test instances based on our metrics, we construct reduced benchmark versions that achieve comparable evaluation performance to full benchmarks while providing greater efficiency.

Our contributions are summarized as follows:

- • We formalize the problem of benchmark quality assessment and propose three complementary metrics that capture different aspects of benchmark reliability.
- • We conduct the first large-scale systematic evaluation of benchmark quality across 15 benchmarks and 11 models spanning four families, providing empirical insights into the state of LLM evaluation.
- • We show that filtering instances via quality metrics achieves comparable evaluation performance using only 35% of the original data.

### 2 Related Work

#### 2.1 LLM Benchmarks

The landscape of LLM evaluation has expanded dramatically. General-purpose benchmarks such

as MMLU (Hendrycks et al., 2021a), BBH (Suzgun et al., 2023), and ARC (Clark et al., 2018) measure broad capabilities across multiple domains. Domain-specific benchmarks have emerged for mathematics (MATH-500, AIME, OlympiadBench), reasoning and comprehension (DROP (Dua et al., 2019), CommonsenseQA (Talmor et al., 2019)), and knowledge understanding (IFEval(Zhou et al., 2023), SuperGPQA (M-A-P Team et al., 2025)). More challenging benchmarks have been introduced to address ceiling effects, including OmniMath (Gao et al., 2024) for advanced mathematics. However, this proliferation has occurred largely without systematic quality assessment.

#### 2.2 Benchmark Quality Analysis

Several studies have examined issues with existing benchmarks. Bowman and Dahl (2021) discussed dangers of benchmark-driven research and the need for more robust evaluation practices. Data contamination, where LLMs inadvertently encounter test data during training, has been identified as a significant concern (Xu et al., 2024; Sainz et al., 2023). Benchmark saturation has motivated dynamic benchmarks (Kiela et al., 2021). Research on evaluation methodology has addressed statistical significance in model comparisons (Dror et al., 2018) and limitations of single-number metrics

(Ethayarajh and Jurafsky, 2020). Liang et al. (2023) introduced HELM for holistic evaluation across multiple dimensions.

Our work complements these efforts by providing metrics specifically designed for benchmark quality assessment. Unlike prior work that focuses on identifying specific issues or proposing new evaluation paradigms, our framework provides systematic, quantitative metrics for assessing benchmark reliability, discriminability, and capability alignment.

### 3 Methodology

In this section, we present BENCHMARK2, our framework for benchmark evaluation. Formally, consider a set of benchmarks B = {B1,B2,...,Bn} and a set of candidate models M = {M1,M2,...,Mm}. Let sij denote the performance score of model Mj evaluated on benchmark Bi. Based on this formulation, we propose three complementary approaches to assess benchmark quality:

#### 3.1 Cross-Benchmark Ranking Consistency

Cross-Benchmark Ranking Consistency (CBRC) evaluates whether a benchmark’s ranking corroborates with others in the same domain. The underlying rationale is that effective benchmarks measuring similar capabilities should produce highly correlated model rankings.

Definition. For a benchmark Bi, we compute its ranking consistency as the average Kendall’s τ correlation with other benchmarks in the same domain:

1 n − 1 j̸=i

τ(ri,rj) (1)

CBRC(Bi) =

where ri denotes the ranking of models induced by benchmark Bi, and τ(·,·) is Kendall’s tau correlation coefficient.

Interpretation. CBRC values range from -1 to 1, where 1 indicates perfect agreement with other benchmarks, 0 indicates no correlation, and negative values indicate inverse rankings. We consider CBRC > 0.7 as indicating high consistency, 0.4–

- 0.7 as moderate, and < 0.4 as low.

#### 3.2 Discriminability Score

A high-quality benchmark should effectively differentiate between models of varying capabilities. If

all models achieve similar scores regardless of their actual ability differences, the benchmark provides limited useful information.

Definition. We define the Discriminability Score (DS) based on the normalized score spread and the statistical significance of pairwise differences:

1[|sij − sik| > ϵ] m(m − 1)/2

σi s¯i · j<k

DS(Bi) =

(2)

where σi is the standard deviation of scores on benchmark Bi, s¯i is the mean score, and the second term represents the proportion of model pairs with practically significant differences (we set ϵ = 0.02 as the minimum meaningful difference).

Interpretation. Higher DS values indicate better discriminability. We empirically find that benchmarks with DS > 0.4 provide good differentiation, while those with DS < 0.2 offer minimal discrimination between models.

#### 3.3 Capability Alignment Deviation

This metric operates at the instance level, identifying individual test questions that may be problematic. The key insight is that if a question is well-designed, stronger models should generally outperform weaker models on it, maintaining alignment with expected capability hierarchies.

Model Family Hierarchy. Rather than establishing a global ordering across all models—which can be unreliable due to different training data and optimization objectives across model families—we leverage the natural capability hierarchy within model families. For a model family F (e.g., Qwen2.5-Instruct), we define an ordering based on parameter count:

F = {M1 ≻ M2 ≻ ··· ≻ Mk} (3)

where M1 has the most parameters and Mk has the fewest. This within-family ordering is more reliable than cross-family comparisons.

Definition. For a benchmark Bi, we first compute the raw inversion rate by aggregating inversions across all model families {F1,F2,...,FF}:

F f=1 invFf(Bi)

inv_rate(Bi) =

F f=1 compF

(Bi)

f

(4)

For each family F = {M1 ≻ M2 ≻ ··· ≻ Mk}, an inversion on question q occurs when a stronger model fails but a weaker model succeeds:

invF(Bi) =

1[¬cjq ∧ clq] (5)

q∈Qi j<l

where Qi is the set of questions in benchmark Bi, cjq indicates whether model Mj correctly answers question q.

We then apply an exponential transformation to convert the inversion rate to a score where higher values indicate better alignment:

CAD(Bi) = e−λ·inv_rate(Bi) (6)

where λ > 0 is a scaling parameter that controls the sensitivity of the transformation. In our experiments, we set λ = 12 based on empirical analysis to ensure meaningful differentiation across the observed range of inversion rates.

Interpretation. CAD ranges from 0 to 1, where 1 indicates perfect alignment (no inversions) and values approaching 0 indicate severe capability hierarchy violations. We consider CAD > 0.6 as indicating good quality, 0.4–0.6 as acceptable, and

- < 0.4 as indicating significant quality issues.

- 3.4 Stability Score

To assess the reliability of selective benchmark evaluation, we introduce the Stability Score, which measures the consistency of model rankings across multiple sampling iterations.

Definition. For a selective benchmark Bs with selection ratio r, we perform K bootstrap sampling iterations (we use K = 100). In each iteration k, we sample r · |B| instances and compute the resulting model ranking rk. The Stability Score is defined as the average pairwise ranking correlation:

Stability(Bs) =

2 K(K − 1) i<j

τ(ri,rj) (7)

where τ(·,·) is Kendall’s tau correlation coefficient between rankings from different bootstrap samples.

Interpretation. Stability Score ranges from -1 to 1, where 1 indicates that the selective benchmark produces identical rankings regardless of which specific instances are sampled, and lower values indicate higher variance in rankings. We consider Stability > 0.7 as high, 0.5–0.7 as moderate, and

- < 0.5 as low.

#### 3.5 Benchmark Quality Score

We also provide a combined score for overall assessment:

BQS(Bi) = α · CBRC(Bi) + β · DS(Bi)

(8)

+ γ · CAD(Bi)

where CBRC denotes the normalized CBRC score, and α, β, γ are weighting parameters. Details on normalization and weight selection are provided in Appendix C.

4 Experimental Setup

#### 4.1 Benchmarks

To demonstrate the broad applicability of our framework, we select a diverse set of 15 benchmarks across three major domains:

Mathematics (5 benchmarks). AIME 2024 (Mathematical Association of America, 2024a), OmniMath (Gao et al., 2024) for advanced mathematical problem solving, OlympiadBench (He et al., 2024), AMC (Mathematical Association of America, 2024b) and MATH-500 (Hendrycks et al., 2021b).

General Reasoning (5 benchmarks). BigBench Hard (BBH) (Suzgun et al., 2023) for challenging tasks, DROP (Dua et al., 2019) for reading comprehension, ARC (Clark et al., 2018) for scientific reasoning, CommonsenseQA (Talmor et al., 2019) for commonsense reasoning and SIQA (Sap et al., 2019) for social intelligence.

Knowledge & Understanding (5 benchmarks). SuperGPQA (M-A-P Team et al., 2025) for graduate-level reasoning, MMLU-Pro (Wang et al., 2024) , IFBench (Pyatkin et al., 2025) and IFEval (Zhou et al., 2023) for instruction following capabilities and EQ-Bench (Paech, 2023) for emotional intelligence.

4.2 Models We evaluate 11 models across four families with clear capability hierarchies based on model size:

- • DeepSeek-R1-Distill-Qwen (DeepSeek-AI, 2025): 1.5B, 7B, 32B
- • Llama-3.1-Instruct (Grattafiori et al., 2024): 8B, 70B
- • Qwen2.5-Instruct (Yang et al., 2024): 1.5B, 7B, 72B
- • Qwen3 (Yang et al., 2025): 1.7B, 8B, 32B

CBRC DS CAD

Benchmark

BQS Score σ Score Range σ Score σ

Mathematics

AIME 2024 0.52 0.10 0.74 0–53 0.22 0.85 0.13 0.79 OmniMath 0.76 0.13 0.79 0–62 0.27 0.61 0.22 0.75 OlympiadBench 0.75 0.12 0.76 0–64 0.32 0.61 0.23 0.73 AMC 22-24 0.70 0.18 0.36 16–67 0.11 0.46 0.09 0.55 MATH-500 0.70 0.18 0.16 49–87 0.07 0.62 0.10 0.55

General Reasoning

ARC 0.79 0.03 0.11 56–95 0.07 0.87 0.05 0.65 BBH 0.75 0.02 0.25 30–89 0.12 0.66 0.05 0.60 DROP 0.71 0.08 0.20 41–87 0.05 0.61 0.07 0.56 CommonsenseQA 0.75 0.07 0.17 37–85 0.11 0.57 0.02 0.54 SIQA 0.73 0.05 0.17 24–53 0.09 0.23 0.03 0.40

Knowledge & Understanding

IFEval 0.75 0.11 0.23 37–87 0.10 0.63 0.10 0.58 EQ-Bench 0.75 0.08 0.27 17–82 0.17 0.53 0.07 0.56 IFBench 0.71 0.11 0.31 11–32 0.08 0.51 0.07 0.55 SuperGPQA 0.79 0.05 0.34 9–40 0.08 0.43 0.07 0.55 MMLU-Pro 0.65 0.10 0.40 27–71 0.17 0.36 0.18 0.51

Table 1: Comprehensive benchmark quality metrics across three domains. CBRC: Cross-Benchmark Ranking Consistency (Kendall’s τ with peer benchmarks). DS: Discriminability Score; Range shows model performance spread (min–max %). CAD: Capability Alignment Deviation (higher is better). BQS: Combined Benchmark Quality Score. σ denotes standard deviation.

Selection Rationale. To ensure architectural diversity and enable reliable CAD computation across multiple scales, we selected models from four distinct development lineages. This strategy mitigates the risk of family-specific bias while providing 1–3 comparison pairs within each family, yielding a total of 10 pairs per benchmark instance.

Held-Out Validation. To verify that our metrics generalize beyond the models used in their computation, we additionally evaluate on Qwen2.5-Base (1.5B, 7B, 32B)—base models that share the same architecture as Qwen2.5-Instruct but were not used in metric computation and have fundamentally different training (no instruction tuning).

#### 4.3 Evaluation Protocol

For each model-benchmark pair, we use standardized prompting templates following the original benchmark specifications where available. We use greedy decoding for reproducibility and evaluate using exact match or execution-based metrics as appropriate for each benchmark.

5 Results

#### 5.1 Comprehensive Quality Analysis

- Table 1 reveals distinct quality profiles across domains.

Mathematics exhibits the widest quality variation (BQS: 0.55–0.79). AIME 2024 achieves exceptional discriminability (DS = 0.74) and capability alignment (CAD = 0.85), while MATH-500 shows potential ceiling effects with low discriminability (DS = 0.16).

General Reasoning presents a qualitydiscriminability trade-off. ARC achieves the highest capability alignment (CAD = 0.87) but limited discriminability (DS = 0.11), whereas BBH maximizes discriminability (DS = 0.25) at the cost of alignment (CAD = 0.66). SIQA exhibits problematic quality across all model families (CAD = 0.23).

Knowledge & Understanding shows the most consistent quality profile (BQS: 0.51–0.58), with IFEval and SuperGPQA achieving strong crossbenchmark consistency (CBRC ≥ 0.75).

Two patterns emerge: (1) high discriminability and high capability alignment rarely co-occur, and (2) benchmarks with objective evaluation criteria consistently achieve higher CAD scores.

#### 5.2 Model Performance Analysis

Table 2 reveals clear capability hierarchies within each model family, validating our within-family CAD computation approach. Within the DeepSeek family, performance scales consistently from

Mathematics General K&U Avg

Model

F S ∆ Rk F S ∆ Rk F S ∆ Rk F S ∆ Rk DeepSeek-R1-Distill-Qwen 32B 57.5 80.1 +22.6 3→3 78.8 85.5 +6.7 3→3 52.8 67.5 +14.7 5→4 63.1 77.7 +14.6 4→3

- 7B 42.8 57.7 +14.9 5→5 63.9 47.7 -16.2 10→10 36.6 22.1 -14.5 10→11 47.8 42.5 -5.3 8→9 1.5B 26.9 26.0 -0.9 10→10 38.3 13.5 -24.8 13→13 20.4 6.1 -14.3 13→13 28.5 15.2 -13.3 13→13

Llama-3.1-Instruct 70B 30.1 32.1 +2.0 8→9 79.8 86.5 +6.7 2→2 60.9 79.8 +18.9 3→3 56.9 66.1 +9.2 5→5

- 8B 13.3 4.2 -9.1 13→13 67.9 56.6 -11.3 8→9 47.5 54.2 +6.7 7→8 42.9 38.3 -4.6 10→10 Qwen2.5-Instruct

72B 49.4 79.3 +29.9 4→4 78.7 85.0 +6.3 4→4 62.0 81.3 +19.3 1→2 63.4 81.9 +18.5 3→2

- 7B 29.5 37.7 +8.2 9→8 73.3 69.4 -3.9 6→6 51.4 58.3 +6.9 6→6 51.4 55.1 +3.7 6→6 1.5B 13.7 6.2 -7.5 12→12 54.3 30.0 -24.3 11→12 29.8 20.1 -9.7 11→12 32.6 18.8 -13.8 11→12

Qwen3 32B 63.5 97.1 +33.6 1→1 81.7 94.2 +12.5 1→1 61.9 85.7 +23.8 2→1 69.1 92.3 +23.2 1→1

- 8B 58.4 82.9 +24.5 2→2 78.3 82.9 +4.6 5→5 56.3 60.6 +4.3 4→5 64.3 75.5 +11.2 2→4 1.7B 34.9 49.7 +14.8 6→6 69.6 57.5 -12.1 7→8 44.8 37.2 -7.6 8→9 49.8 48.1 -1.7 7→8

τ 0.96 0.96 0.85 0.93 Stability 0.76 0.73 0.58 0.69

- Table 2: Model performance comparison between full benchmarks (F) and selective evaluation (S). ∆ = score difference, Rk = rank change among all 14 models (Full→Selective). Bottom rows show ranking consistency (Kendall’s τ) and stability score per domain.

Model

Mathematics General K&U Avg

F S ∆ Rk F S ∆ Rk F S ∆ Rk F S ∆ Rk

32B 34.5 45.0 +10.5 7→7 67.5 61.6 -5.9 9→7 39.7 56.7 +17.0 9→7 47.2 54.4 +7.2 9→7 7B 15.2 6.4 -8.8 11→11 52.8 42.8 -10.0 12→11 23.8 30.4 +6.6 12→10 30.6 26.5 -4.1 12→11 1.5B 1.5 1.1 -0.4 14→14 5.8 7.8 +2.0 14→14 4.2 5.3 +1.1 14→14 3.9 4.7 +0.8 14→14

Avg |∆Rk| 0.0 1.0 1.3 1.0

- Table 3: Held-out model validation using Qwen2.5-Base family (1.5B, 7B, 32B). These base models were not used in computing CAD, DS, or CBRC metrics, which were derived exclusively from instruction-tuned models. F = Full benchmark score (%), S = Selective evaluation score (%), ∆ = score difference, Rk = rank among all 14 models (Full→Selective). Avg |∆Rk| shows the average absolute rank change for held-out models (lower is better).

28.5% (1.5B) to 47.8% (7B) to 63.1% (32B) average. Similar patterns hold for Llama (42.9% to 56.9%), Qwen2.5 (32.6% to 51.4% to 63.4%), and Qwen3 (49.8% to 64.3% to 69.1%).

The selective evaluation results demonstrate that our quality-based instance selection maintains strong ranking consistency with full benchmarks. The average Kendall’s τ of 0.93 indicates that selective evaluation preserves the relative ordering of models while using only 35% of the original instances. Notably, larger models within each family consistently show positive ∆ values on selective benchmarks, confirming that high-quality instances better differentiate capable models. The rank change column (Rk) shows that most models maintain similar rankings between full and selective evaluation, with only minor position shifts

occurring primarily among mid-tier models. 5.3 Held-Out Model Validation

To validate that our metrics generalize beyond the models used in their computation, we evaluate on held-out models that were not included in computing CAD, DS, or CBRC metrics. Specifically, we use Qwen2.5-Base (1.5B, 7B, 32B), which are base models without instruction tuning. Table 3 presents these results.

The held-out validation demonstrates strong generalization of our selective benchmark approach. Mathematics shows perfect rank preservation (Avg |∆Rk| = 0.0), indicating that selective evaluation ranks held-out models identically to full benchmarks in this domain. General Reasoning and Average scores show moderate variation (1.0), while

###### Effect of Selection Ratio on Benchmark Quality Metrics

1.0

1.0

0.9

0.9

Ranking Stability DS

0.8

Ranking/DS

0.8

Stability

0.7

0.6

0.7

0.5

| |
|---|

0.6

| |
|---|

0.4

| |
|---|

0.3

0.5

20 40 60 80 100

Selection Ratio (%)

- Figure 2: Effect of selection ratio on benchmark quality metrics. The optimal point at 35% (marked) achieves a good balance between ranking consistency (τ = 0.93), stability (0.69), and discriminability (DS = 0.47).

Knowledge & Understanding shows slightly higher variation (1.3). Notably, the extreme-performing

- 1.5B model maintains its 14th rank consistently across all domains, demonstrating that our selection method is particularly reliable at the capability distribution tails.

#### 5.4 Selective Benchmark Construction

We select test instances with high CAD scores (indicating low inversion rates) and high discriminability contributions, creating filtered benchmarks containing approximately 35% of the original instances.

#### 5.4.1 Selection Ratio Analysis

Figure 2 illustrates the effect of selection ratio on benchmark quality metrics. As the selection ratio increases from 10% to 100%, we observe distinct patterns across the three metrics.

Ranking consistency (τ) increases rapidly from 0.88 at 10% to 0.93 at 35%, then plateaus near 0.99 for higher ratios. Stability shows an inverse pattern, starting high at low selection ratios (where only the most reliable instances are included) and gradually decreasing as more instances are added. DS decreases steadily as selection ratio increases, as the most discriminative instances are selected first. The optimal point at 35% achieves a balanced trade-off: ranking τ of 0.93, stability of 0.69, and DS of 0.47—substantially better than the full benchmark’s stability of 0.59 while maintaining comparable ranking consistency.

#### 5.4.2 Metric Combination Ablation

- Table 4 compares metric combinations for instance selection. Key findings from the ablation study:

Method Level Rank τ Stability DS Single Metric

CAD only (score > 0.15) Inst 0.93 0.61 0.32 DS only (top 35%) Inst 0.95 0.50 0.48 CBRC only (top-2) Bench 0.62 0.62 0.28

Two Metrics

CAD + DS (ours) Inst 0.93 0.69 0.47 CAD + CBRC Mix 0.67 0.78 0.25 DS + CBRC Mix 0.59 0.58 0.33

Three Metrics CAD + DS + CBRC Mix 0.65 0.76 0.29 Full benchmark – 1.00 0.59 0.34

- Table 4: Metric combination ablation. Level: Inst=Instance, Bench=Benchmark. Our CAD+DS combination (bold) achieves the best balance.

Threshold Retained τ Stab.

0.40 58% 0.87 1.00 0.15 84% 0.93 0.69 0.05 95% 0.96 0.55 0.01 98% 0.98 0.52 None 100% 0.95 0.50

- Table 5: CAD threshold sensitivity analysis. Higher threshold is more restrictive (fewer instances retained).

CAD alone provides good stability. Filtering by CAD alone (score > 0.15) yields good ranking consistency (0.93) and reasonable stability (0.61) by removing noisy instances, but reduces discriminability (0.32).

DS alone maximizes discriminability. Selecting high-discriminability instances preserves the best discriminability (0.48) but provides lower stability (0.50).

Combined approach balances objectives. Our CAD+DS combination achieves strong ranking consistency (0.93) and improved stability (0.69) compared to the full benchmark (0.59), while maintaining good discriminability (0.47).

#### 5.4.3 Threshold Sensitivity Analysis

Table 5 presents results for different CAD threshold values. The threshold analysis reveals a trade-off between ranking consistency and stability. A very restrictive threshold (0.40) achieves perfect stability (1.00) but lower ranking consistency (0.87) due to insufficient instances. The threshold of 0.15 provides an optimal balance, achieving strong ranking consistency (0.93) and good stability (0.69) while retaining 84% of instances.

##### Strategy Rank τ Stability DS

Random selection 0.91 ± 0.04 0.59 ± 0.02 0.34 ± 0.02 High accuracy 0.87 0.68 0.33 Low accuracy 0.79 0.61 0.26 Medium difficulty 0.93 0.51 0.47 Longest instances 0.89 0.59 0.34 Shortest instances 0.76 0.62 0.31

CAD + DS (ours) 0.93 0.69 0.47 Full benchmark 1.00 0.59 0.34

- Table 6: Comparison of selection strategies. High/Low accuracy: instances where most models succeed/fail. Medium difficulty: moderate average accuracy. Longest/Shortest: most/fewest tokens.

Benchmark DeepSeek Llama Qwen2.5 Qwen3

MATH-500 0.48 0.58 0.74 0.67 AIME 2024 0.67 1.00 1.00 0.88 AMC 22-24 0.37 0.58 0.57 0.43 OlympiadBench 0.54 0.93 0.91 0.39 OmniMath 0.61 0.98 0.84 0.39 DROP 0.58 0.76 0.58 0.62 ARC 0.80 0.92 0.87 0.93 BBH 0.75 0.63 0.62 0.62 SIQA 0.24 0.21 0.20 0.27 CommonsenseQA 0.57 0.54 0.59 0.55 IFEval 0.49 0.72 0.75 0.66 IFBench 0.62 0.52 0.49 0.44 EQ-Bench 0.60 0.61 0.54 0.43 SuperGPQA 0.54 0.39 0.35 0.44 MMLU-Pro 0.57 0.51 0.55 0.54

- Table 7: CAD breakdown by model family (transformed scores, higher is better). Values represent e−λ·inv_rate with λ = 12, where 1.0 indicates perfect alignment.

#### 5.4.4 Baseline Comparison

- Table 6 compares our selection against baselines. High-accuracy selection provides good stability (0.68) but lower ranking consistency (0.87). Medium-difficulty achieves comparable discriminability (0.47) but lower stability (0.51). Our approach achieves the best balance of stability (0.69) and discriminability (0.47) while maintaining strong ranking consistency (0.93).

5.5 CAD Breakdown by Model Family

- Table 7 reveals family-specific patterns. Llama achieves near-perfect CAD on several benchmarks (AIME: 1.00, OmniMath: 0.98), while Qwen3 shows higher variation across benchmarks (OlympiadBench: 0.39, EQ-Bench: 0.43). SIQA exhibits consistently low CAD across all families (0.20–0.27), indicating inherent design issues.

### 6 Discussion and Future Directions

- 6.1 What Makes a High-Quality Benchmark?

Our analysis identifies three key characteristics: (1) High discriminability—top benchmarks (AIME, OmniMath, OlympiadBench) achieve DS > 0.7 with wide score ranges; (2) Strong capability alignment—benchmarks with CAD > 0.6 respect withinfamily hierarchies and feature objective evaluation; (3) Balanced quality profile—the highest BQS scores emerge from benchmarks balancing multiple dimensions, as exemplified by AIME 2024 (BQS = 0.79).

- 6.2 Implications for Benchmark Development

We recommend that benchmark developers: (1) target DS > 0.2 and CAD > 0.6 as minimum thresholds; (2) prefer objective evaluation criteria; (3) consider selective construction using CAD+DS metrics; and (4) monitor family-specific CAD variation as an indicator of potential biases.

- 6.3 Methodological Considerations

On CBRC and circularity. Using benchmarks to evaluate benchmarks raises circularity concerns. We mitigate this by selecting widely-adopted reference benchmarks, aggregating across multiple benchmarks, and complementing CBRC with two reference-independent metrics (CAD and DS).

On model and CAD scope. Our evaluation spans four model families, with held-out validation confirming generalization. CAD requires multiple model sizes within a family, limiting applicability to single-variant proprietary models, but ensures reliable capability ordering.

### 7 Conclusion

We presented BENCHMARK2, a framework for evaluating LLM benchmark quality through three complementary metrics: Cross-Benchmark Ranking Consistency, Discriminability Score, and Capability Alignment Deviation. Our evaluation across 15 benchmarks and 11 models reveals significant quality variations among widely-used benchmarks, and demonstrates that selective construction can maintain evaluation fidelity using only 35% of original instances. We hope this framework helps practitioners assess benchmark reliability. Future directions include extending to generation-based evaluations with LLM-as-judge and developing dynamic quality monitoring for benchmark degradation.

### Limitations

Our study has several limitations that suggest directions for future research. First, our evaluation focuses on three domains (mathematics, reasoning, and knowledge understanding); although our metrics are domain-agnostic by design, extending validation to additional domains such as code generation, machine translation, and dialogue systems remains important future work. Second, our analysis is restricted to text-based LLM benchmarks; as multimodal large language models become increasingly prevalent, extending our framework to vision-language, audio-language, and video understanding benchmarks represents a natural next step. Third, while our evaluation spans 11 models across four diverse families, incorporating a broader range of models including proprietary systems would enhance generalizability.

### Ethics Statement

This work involves evaluation of existing public benchmarks and models, and does not introduce new data collection or human subjects research. We use only publicly available benchmarks and evaluate models through their official APIs or publicly released weights. Our framework is intended to improve benchmark quality and thereby contribute to more reliable AI evaluation.

### References

Samuel R Bowman and George E Dahl. 2021. What will it take to fix benchmarking in natural language understanding? arXiv preprint arXiv:2104.02145.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. 2018. The hitchhiker’s guide to testing statistical significance in natural language processing. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, pages 1383–1392.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019.

DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2368–2378.

Kawin Ethayarajh and Dan Jurafsky. 2020. Utility is in the eye of the user: A critique of nlp leaderboards. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 4846–4853.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. 2024. Omni-MATH: A universal olympiad level mathematic benchmark for large language models. arXiv preprint arXiv:2410.07985.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In Proceedings of the International Conference on Learning Representations (ICLR).

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, and 1 others. 2021. Dynabench: Rethinking benchmarking in nlp. arXiv preprint arXiv:2104.14337.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E Gonzalez, Hao Zhang, and Ion Stoica. 2023. vLLM: Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, and 1 others. 2023. Holistic evaluation of language models. Transactions on Machine Learning Research.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, and 1 others. 2024. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574.

M-A-P Team, Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, and 1 others. 2025. SuperGPQA: Scaling LLM evaluation across 285 graduate disciplines. arXiv preprint arXiv:2502.14739.

Ruijie Xu, Zengzhi Luo, Siyuan Chen, Jing He, Junhe Duan, Fengzhe Wu, Qi Zhang, and Xuanjing Xu. 2024. Benchmarking benchmark leakage in large language models. arXiv preprint arXiv:2404.18824.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Mathematical Association of America. 2024a. AIME 2024: American invitational mathematics examination. https: //www.maa.org/math-competitions/ aime-american-invitational-mathematics-examination.

Qwen An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Junyang Lin, and 25 others. 2024. Qwen2.5 technical report. ArXiv, abs/2412.15115.

Mathematical Association of America. 2024b. AMC 10/12: American mathematics competitions 2022-2024. https://www.maa.org/ math-competitions/amc-1012.

ModelScope Team. 2024. EvalScope: Evaluation framework for large models. https://github.com/ modelscope/evalscope.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Samuel J. Paech. 2023. EQ-Bench: An emotional intelligence benchmark for large language models. arXiv preprint arXiv:2312.06281.

Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hannaneh Hajishirzi. 2025. Generalizing verifiable instruction following. In Advances in Neural Information Processing Systems, volume 38.

### A Experimental Setup Details

We conduct all experiments using the EvalScope framework (ModelScope Team, 2024), an opensource evaluation toolkit that provides standardized benchmark implementations and consistent evaluation protocols. For model deployment and inference, we utilize the vLLM framework (Kwon et al., 2023), a high-throughput serving engine optimized for large language models.

Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre. 2023. Nlp evaluation in trouble: On the need to measure llm data contamination for each benchmark. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10776– 10787.

Table 8 summarizes the key inference parameters used across all experiments. We use greedy decoding for reproducibility and set the maximum new tokens to 16384 to accommodate long-form reasoning outputs. All experiments were conducted on NVIDIA A100 80GB GPUs, with smaller models (1.5B–8B parameters) evaluated using single-GPU deployment and larger models (32B–72B parameters) utilizing multi-GPU tensor parallelism. The complete evaluation across all 15 benchmarks and 14 models required approximately 500 GPU-hours.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. 2019. Social IQA: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, pages 4463–4473.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, and Jason Wei. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4149–4158.

### B CAD Transform Parameter Selection

The Capability Alignment Deviation (CAD) metric applies an exponential transformation to convert raw inversion rates into interpretable scores:

##### Parameter Value

Temperature 0.7 Top-p 0.8 Max new tokens 16384 Tensor parallelism Model-dependent GPU memory utilization 0.90

Table 8: Inference configuration for all model evaluations using vLLM framework.

CAD(Bi) = e−λ·inv_rate(Bi). The choice of λ affects the sensitivity of the transformed scores to variations in raw inversion rates. We conduct a systematic analysis to select an appropriate value based on five criteria: (1) Median Mapping—the median raw inversion rate should map to a score in the range [0.15, 0.35]; (2) Quality Separationdifferent quality levels should exhibit meaningful score differences; (3) Excellent Quality Rewardbenchmarks with low inversion rates (raw_cad < 0.03) should receive high scores (> 0.65); (4) Poor Quality Penalty—benchmarks with high inversion rates (raw_cad > 0.25) should receive low scores (< 0.10); and (5) Dynamic Range—the transformation should preserve meaningful variation across the main data distribution.

- Table 9 presents the evaluation of candidate λ values against these criteria. Based on this analysis, we select λ = 12 as it achieves the highest total score (0.68) by providing strong quality separation

- (0.93), perfect reward for excellent benchmarks
- (1.00), full penalty for poor benchmarks (1.00), and complete dynamic range preservation (1.00).

- Table 10 shows how raw inversion rates translate to CAD scores with λ = 12, providing practitioners with concrete reference points for interpreting CAD values in practice.

### C BQS Weight and Normalization Details

The Benchmark Quality Score (BQS) combines three metrics with different native scales. To ensure meaningful aggregation, we apply normalization and empirically-tuned weights.

CBRC Normalization. CBRC (Kendall’s τ) ranges from −1 to 1, while DS and CAD both range from 0 to 1. To align scales, we normalize CBRC using a linear transformation:

CBRC(Bi) + 1 2

CBRC(Bi) =

(9)

This maps the CBRC range [−1,1] to [0,1], where 0 indicates perfect negative correlation, 0.5 indi-

cates no correlation, and 1 indicates perfect positive correlation.

Weight Selection. We assign weights α = 0.3, β = 0.3, and γ = 0.4 based on the following considerations:

- • CAD receives the highest weight (0.4) because it operates at the instance level and directly measures whether individual test items respect capability hierarchies—a fundamental property of well-designed benchmarks.
- • CBRC and DS receive equal weights (0.3 each) as they capture complementary benchmark-level properties: external consistency (CBRC) and internal discriminative power (DS).

Final Formula. The complete BQS formula is:

CBRC(Bi) + 1 2

BQS(Bi) = 0.3 ·

+ 0.3 · DS(Bi)

+ 0.4 · CAD(Bi)

(10)

### D Detailed Model Performance

This section presents the complete performance matrix for all 14 models across the 15 benchmarks, organized by domain.

In the Mathematics domain (Table 11), DeepSeek-R1-Distill-Qwen-32B shows strong performance on competition-style benchmarks, achieving the highest score on AIME 2024 (53.3%). The Qwen3 family demonstrates consistently strong results across all mathematics benchmarks, with Qwen3-32B achieving the highest scores on MATH-500 (87.0%) and AMC 22-24 (67.2%).

In General Reasoning (Table 12), Qwen3-32B achieves the highest scores on DROP (85.7%) and ARC (95.0%). The results show clear capability hierarchies across model families, with larger models consistently outperforming their smaller counterparts.

λ Median Separation Excellent Poor Range Total

3 0.00 1.00 1.00 0.00 0.90 0.54 6 0.00 1.00 1.00 0.38 1.00 0.61 9 0.00 1.00 1.00 0.70 1.00 0.66

12 0.00 0.93 1.00 1.00 1.00 0.68 15 0.00 0.72 0.70 1.00 1.00 0.57 18 0.00 0.54 0.70 1.00 1.00 0.53 24 0.00 0.30 0.97 1.00 1.00 0.52 30 0.00 0.17 0.81 1.00 1.00 0.45

- Table 9: Lambda parameter selection analysis. Median: median raw_cad maps to 0.15–0.35 score range; Separation: quality level separation; Excellent: excellent quality (raw_cad < 0.03) receives high score; Poor: poor quality (raw_cad > 0.25) receives low score; Range: dynamic range in main data distribution. The weighted total uses coefficients 0.30, 0.25, 0.20, 0.15, 0.10 respectively.

Raw CAD Transformed Score

0.00 1.000

- 0.02 0.787
- 0.03 0.698 0.05 0.549 0.08 0.383 0.10 0.301 0.12 0.237 0.15 0.165 0.20 0.091 0.25 0.050 0.30 0.027 0.40 0.008

- Table 10: Raw CAD to transformed score mapping with λ = 12. Quality levels: Excellent (raw_cad < 0.03), Good (0.03–0.08), Acceptable (0.08–0.15), Concerning (0.15–0.25), Poor (> 0.25).

For Knowledge & Understanding (Table 13), the larger instruction-tuned models generally achieve higher performance, with IFEval and EQ-Bench showing clearer capability hierarchies across model families.

### E Statistical Reliability Analysis

We compute 95% confidence intervals for all metrics using bootstrap sampling with 1000 iterations. Table 14 presents these intervals across all benchmarks. The results reveal that CBRC estimates show moderate uncertainty (typical CI width of 0.3–0.5), while CAD estimates are notably more stable (typical CI width < 0.1). This stability arises because CAD aggregates over many instance-level comparisons, reducing variance. The DS metric shows higher variability, particularly for smaller benchmarks like AIME 2024 (CI: [0.54, 1.19]), reflecting sensitivity to the specific model set evaluated.

### F Cross-Benchmark Correlation Analysis

We compute pairwise Kendall’s τ correlations between benchmarks within each domain to under-

stand the consistency of model rankings across different evaluation instruments.

In Mathematics (Table 15), we observe high correlations among most benchmarks. MATH-500 and AMC 22-24 show strong correlation (τ = 0.88), while OlympiadBench and OmniMath form a nearly perfectly correlated pair (τ = 0.99). AIME 2024 shows moderate correlations with other benchmarks (τ ≈ 0.62–0.71), reflecting its unique difficulty level.

In General Reasoning (Table 16), DROP and BBH show the highest correlation (τ = 0.85), both requiring complex reasoning. SIQA and CommonsenseQA show strong alignment (τ = 0.80), as both focus on social and commonsense understanding.

The Knowledge & Understanding domain (Table 17) exhibits a relatively uniform correlation structure, with IFEval and EQ-Bench showing strong alignment (τ = 0.80).

##### Model MATH-500 AIME 2024 AMC 22-24 OlympiadBench OmniMath

DeepSeek-R1-Distill-Qwen-1.5B 68.4 16.7 35.8 6.8 6.7 DeepSeek-R1-Distill-Qwen-7B 73.6 16.7 39.6 49.4 34.8 DeepSeek-R1-Distill-Qwen-32B 74.6 53.3 41.8 61.8 56.2 Llama-3.1-Instruct-8B 49.4 0.0 16.4 0.7 0.2 Llama-3.1-Instruct-70B 68.2 23.3 26.9 18.3 13.9 Qwen2.5-Instruct-1.5B 49.8 0.0 18.7 0.0 0.0 Qwen2.5-Instruct-7B 75.2 13.3 40.3 8.1 10.4 Qwen2.5-Instruct-72B 83.4 16.7 50.7 52.2 44.1 Qwen3-1.7B 72.2 10.0 40.3 28.9 22.9 Qwen3-8B 84.6 26.7 61.9 61.8 57.1 Qwen3-32B 87.0 36.7 67.2 64.8 62.0

Table 11: Mathematics domain: Model performance (%) on each benchmark.

Model DROP ARC BBH SIQA CommonsenseQA

DeepSeek-R1-Distill-Qwen-1.5B 42.0 56.7 30.8 24.9 37.3 DeepSeek-R1-Distill-Qwen-7B 66.2 83.9 69.4 37.6 62.4 DeepSeek-R1-Distill-Qwen-32B 76.6 93.2 89.6 51.4 83.1 Llama-3.1-Instruct-8B 66.8 86.3 67.5 42.3 76.7 Llama-3.1-Instruct-70B 87.9 93.9 85.3 49.6 82.5 Qwen2.5-Instruct-1.5B 45.9 78.1 39.7 41.5 66.5 Qwen2.5-Instruct-7B 70.2 90.6 71.1 52.2 82.4 Qwen2.5-Instruct-72B 73.7 94.2 87.3 52.8 85.4 Qwen3-1.7B 67.6 89.0 72.8 45.9 72.7 Qwen3-8B 82.0 94.1 83.5 48.9 83.0 Qwen3-32B 85.7 95.0 89.9 53.7 84.4

Table 12: General Reasoning domain: Model performance (%) on each benchmark.

Model IFEval IFBench EQ-Bench SuperGPQA MMLU-Pro

DeepSeek-R1-Distill-Qwen-1.5B 37.1 11.0 17.1 9.8 27.1 DeepSeek-R1-Distill-Qwen-7B 58.1 13.1 49.8 18.0 43.8 DeepSeek-R1-Distill-Qwen-32B 73.6 19.0 76.8 31.7 63.0 Llama-3.1-Instruct-8B 76.7 25.6 66.7 20.8 47.6 Llama-3.1-Instruct-70B 87.2 32.2 82.1 35.5 67.4 Qwen2.5-Instruct-1.5B 41.1 14.5 48.2 17.3 27.8 Qwen2.5-Instruct-7B 74.2 23.6 72.9 28.9 57.3 Qwen2.5-Instruct-72B 86.6 32.7 78.4 40.5 71.9 Qwen3-1.7B 70.6 22.5 63.0 20.5 47.4 Qwen3-8B 84.5 30.6 76.6 28.0 61.7 Qwen3-32B 86.6 32.4 80.1 39.4 71.0

Table 13: Knowledge & Understanding domain: Model performance (%) on each benchmark.

CBRC DS CAD Mean 95% CI σ Mean 95% CI σ Mean 95% CI σ

Benchmark

MATH-500 0.76 [0.55, 0.91] 0.09 0.27 [0.09, 0.52] 0.12 0.80 [0.76, 0.83] 0.02 AIME 2024 0.66 [0.39, 0.88] 0.13 0.84 [0.54, 1.19] 0.16 0.92 [0.83, 1.00] 0.04 AMC 22-24 0.72 [0.51, 0.88] 0.09 0.43 [0.24, 0.65] 0.11 0.69 [0.62, 0.75] 0.03 OlympiadBench 0.79 [0.61, 0.91] 0.08 0.82 [0.51, 1.15] 0.16 0.77 [0.75, 0.79] 0.01 OmniMath 0.78 [0.61, 0.90] 0.07 0.85 [0.58, 1.16] 0.16 0.78 [0.75, 0.80] 0.01

DROP 0.68 [0.39, 0.90] 0.13 0.30 [0.11, 0.54] 0.12 0.80 [0.80, 0.81] 0.00 ARC 0.73 [0.51, 0.90] 0.10 0.27 [0.06, 0.51] 0.12 0.94 [0.93, 0.95] 0.00 BBH 0.70 [0.43, 0.90] 0.12 0.31 [0.17, 0.46] 0.08 0.82 [0.82, 0.83] 0.00 SIQA 0.70 [0.43, 0.90] 0.12 0.29 [0.10, 0.52] 0.11 0.49 [0.47, 0.51] 0.01 CommonsenseQA 0.68 [0.42, 0.88] 0.12 0.28 [0.07, 0.51] 0.12 0.79 [0.77, 0.81] 0.01

IFEval 0.69 [0.47, 0.85] 0.09 0.31 [0.19, 0.45] 0.07 0.79 [0.76, 0.82] 0.02 IFBench 0.55 [0.22, 0.80] 0.15 0.32 [0.23, 0.39] 0.04 0.74 [0.69, 0.78] 0.02 EQ-Bench 0.70 [0.54, 0.85] 0.08 0.35 [0.12, 0.61] 0.12 0.77 [0.71, 0.83] 0.03 SuperGPQA 0.65 [0.39, 0.83] 0.11 0.39 [0.22, 0.57] 0.09 0.67 [0.67, 0.68] 0.00 MMLU-Pro 0.59 [0.21, 0.85] 0.17 0.44 [0.25, 0.64] 0.10 0.54 [0.53, 0.54] 0.00

Table 14: Bootstrap 95% confidence intervals for all metrics across benchmarks (1000 iterations).

MATH-500 AIME 2024 AMC 22-24 OlympiadBench OmniMath

MATH-500 1.00 0.65 0.88 0.75 0.75 AIME 2024 – 1.00 0.62 0.71 0.69 AMC 22-24 – – 1.00 0.70 0.70 OlympiadBench – – – 1.00 0.99 OmniMath – – – – 1.00

Table 15: Mathematics domain: Pairwise Kendall’s τ correlation between benchmarks.

DROP ARC BBH SIQA CommonsenseQA

DROP 1.00 0.74 0.85 0.58 0.56 ARC – 1.00 0.71 0.76 0.74 BBH – – 1.00 0.65 0.63 SIQA – – – 1.00 0.80 CommonsenseQA – – – – 1.00

Table 16: General Reasoning domain: Pairwise Kendall’s τ correlation between benchmarks.

IFEval IFBench EQ-Bench SuperGPQA MMLU-Pro

IFEval 1.00 0.75 0.80 0.62 0.57 IFBench – 1.00 0.54 0.47 0.43 EQ-Bench – – 1.00 0.80 0.67 SuperGPQA – – – 1.00 0.69 MMLU-Pro – – – – 1.00

Table 17: Knowledge & Understanding domain: Pairwise Kendall’s τ correlation between benchmarks.

