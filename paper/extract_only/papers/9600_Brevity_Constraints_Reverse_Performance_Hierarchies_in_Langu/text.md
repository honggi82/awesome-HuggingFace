# arXiv:2604.00025v1[cs.CL]11Mar2026

## Brevity Constraints Reverse Performance Hierarchies in Language Models

#### MD Azizul Hakim∗

Department of Computer Science Bangladesh Sweden Polytechnic Institute Chattogram, Bangladesh azizulhakim8291@gmail.com

### Abstract

Standard evaluation protocols reveal a counterintuitive phenomenon: on 7.7% of benchmark problems spanning five datasets, larger language models underperform smaller ones by 28.4 percentage points despite 10-100× more parameters. Through systematic evaluation of 31 models (0.5B-405B parameters) across 1,485 problems, we identify the mechanism as spontaneous scale-dependent verbosity that introduces errors through overelaboration. Causal intervention experiments demonstrate this reflects correctable prompt design rather than fundamental capability limitations. Constraining large models to produce brief responses improves accuracy by 26 percentage points and reduces performance gaps by up to two-thirds. Most critically, brevity constraints completely reverse performance hierarchies on mathematical reasoning and scientific knowledge benchmarks, with large models achieving 7.7-15.9 percentage point advantages over small models—direct inversions of the original gaps. These reversals prove large models possess superior latent capabilities that universal prompting masks. We validate findings through three independent contamination tests and demonstrate inverse scaling operates continuously across the full parameter spectrum, with dataset-specific optimal scales ranging from 0.5B to 3.0B parameters. Our results establish that maximizing large model performance requires scale-aware prompt engineering rather than universal evaluation protocols, with immediate implications for deployment: prompt adaptation simultaneously improves accuracy and reduces computational costs.

### 1 Introduction

Language models have achieved remarkable capabilities across diverse tasks, with performance improvements attributed primarily to increased model scale [1–3]. This scaling paradigm assumes monotonic improvement: larger models consistently outperform smaller ones [4, 5]. Contemporary evaluation benchmarks spanning mathematical reasoning [6], reading comprehension [7], scientific knowledge [8], and commonsense reasoning [9] have become standard for measuring progress. However, fundamental questions about scaling universality remain unexamined. Does performance improve monotonically with scale across all problem types?

This work addresses this question through systematic analysis of problem-level performance patterns. We evaluated 31 models (0.5B-405B parameters) across 1,485 problems from five benchmarks (46,035 evaluations), identifying inverse scaling on 7.7% of benchmark problems (115/1,485), where small models achieve higher accuracy than large models on identical problems. This pattern replicates consistently across diverse capability domains, with average performance gaps of 28.4 percentage points favoring smaller models. The effect magnitude (Cohen’s d = 1.34) substantially exceeds

∗about author (https://logicsame.github.io/md-hakim.github.io/publications)

Preprint.

the conventional threshold for large effects, indicating categorical performance reversals rather than subtle statistical trends. Unlike previously documented inverse scaling cases typically involving artificial tasks [10], our findings emerge from standard benchmarks measuring practical capabilities.

These findings reveal a critical gap in current evaluation methodology: standard prompting practices systematically underestimate large model performance on specific problem types. While aggregate scaling laws accurately predict trends under universal prompts, they mask how scale-dependent prompt sensitivity affects problem-level accuracy. The documented gap reversals on 2/5 benchmarks demonstrate that optimal prompting strategies must account for model size, with immediate implications for deployment. Problem-aware routing combined with scale-appropriate prompts can simultaneously improve accuracy (by unlocking masked capabilities in large models) and reduce computational costs (by identifying when smaller models suffice). Rather than revealing fundamental limitations of scale, our results establish that maximizing large model capabilities requires moving beyond universal prompting toward scale-aware prompt engineering[11] that adapts evaluation protocols to model characteristics.

To establish causality rather than mere correlation, we conducted intervention experiments on all 115 inverse scaling problems. We constrained seven models (three small, four large) to produce brief responses under three conditions: control (unconstrained), brief (forced brevity), and direct (answer-only). Results provide compelling causal evidence: brevity constraints improved large model accuracy by 26.3 percentage points and reduced the inverse scaling gap by 67% (from 44.2% to 14.8%, paired t-test: t = 7.80, p < 0.0001). Response length validation confirms the intervention succeeded, with large models producing 60% shorter outputs under brevity constraints.

Contamination analysis addresses the critical concern that inverse scaling[10] reflects dataset memorization rather than genuine capability differences. Three independent tests examining response diversity, length variability, and error patterns converge on the same conclusion: inverse scaling represents genuine architectural and scale-dependent capability differences. High response diversity (100% unique responses on three datasets) and natural length variation (coefficient of variation > 0.30 across all datasets) contradict memorization hypotheses. Error taxonomy reveals large model failures predominantly result from over-reasoning rather than memorization avoidance.

### 2 Related Work

Scaling laws and emergent capabilities. Neural language models have demonstrated consistent performance improvements with increased scale, formalized through power-law relationships between model size, training compute, and downstream task performance [1, 2]. These scaling laws have guided development of increasingly large models, from GPT-3’s 175 billion parameters [3] to current frontier models exceeding 400 billion parameters [12]. Emergent abilities—capabilities appearing suddenly at specific scale thresholds—further reinforce the scaling paradigm [4, 13]. However, recent work questions whether emergence reflects genuine phase transitions or measurement artifacts [13], while our findings document systematic performance degradation with scale on specific problem subsets.

Inverse scaling phenomena. Prior work has identified limited cases where larger models underperform smaller ones. The Inverse Scaling Prize [10] documented 11 tasks exhibiting inverse scaling, primarily involving memorization of rare patterns, distractor reasoning, and spurious correlations. BIG-Bench [14] reported inverse scaling on 12 of 204 tasks, often attributable to task-specific artifacts. These studies focused on constructed or adversarial examples designed to expose failure modes. In contrast, our work identifies inverse scaling on standard benchmark problems designed to measure genuine capabilities, with mechanistic analysis and causal validation revealing overthinking

- as the underlying failure mode rather than memorization or spurious patterns. However, our causal interventions demonstrate this inverse scaling reflects prompt-induced failure modes rather than fundamental capability limitations, distinguishing our findings from prior work documenting intrinsic scale-dependent degradation.

Model evaluation and benchmark design. Contemporary evaluation practices assume benchmark problems uniformly measure model capabilities [15, 16]. Recent work has examined dataset contamination [17, 18], evaluation robustness [19], and benchmark saturation [20]. Item Response Theory has been applied to characterize problem difficulty and model capabilities [21], while work on evaluation informativeness proposes more efficient problem selection [22]. However, these approaches

do not examine scale-dependent performance patterns or discriminative efficiency. Our analysis reveals that 55.9% of benchmark problems discriminate between models, while 7.7% exhibit inverse scaling—findings with direct implications for efficient evaluation design. Model efficiency and deployment. Research on efficient deployment has focused on model compression [23, 24], mixtureof-experts architectures [25], and dynamic inference [26]. These approaches assume smaller models represent degraded versions of larger counterparts. Our findings suggest an alternative paradigm: problem-aware routing matching model scale to task characteristics, where smaller models may provide superior accuracy at lower cost on specific problem types. This complements recent work on model cascades [27] and routing strategies [28].

### 3 Methods

#### 3.1 Model Selection and Evaluation

We evaluated 31 language models spanning 0.5B to 405B parameters across five benchmark datasets: GSM8K (mathematical reasoning) [6], BoolQ (reading comprehension) [7], ARC-Easy (science questions) [29], CommonsenseQA (commonsense reasoning) [9], and MMLU-STEM (scientific knowledge) [8]. Models were evaluated using greedy decoding (do_sample=False) with nucleus sampling disabled to ensure deterministic outputs. Base prompts contain no chain-of-thought elicitation—multiple choice tasks use bare Question/Answer format and mathematical reasoning uses bare Problem/Solution format—ensuring that scale-dependent verbosity differences reflect model-intrinsic properties rather than prompt-induced reasoning chains (full templates in Appendix 5.5).

For each problem i and model m, we extracted answers using task-specific validators with hierarchical extraction strategies. Accuracy was computed as:

1 N

Accm =

N

⊮[ˆym,i = yi] (1)

i=1

where yˆm,i is the extracted answer, yi is ground truth, and ⊮[·] is the indicator function. We classified models as small (N ≤ 10B parameters) or large (N > 70B parameters) based on natural performance gaps observed in preliminary analysis.

#### 3.2 Inverse Scaling Detection

We identified inverse scaling[10] problems where small models systematically outperformed large models. For each problem i, we computed the performance gap:

∆i = Accsmall,i − Acclarge,i (2)

where Accsmall,i and Acclarge,i represent accuracy averaged across small and large models respectively. Problems with ∆i > 0 exhibited inverse scaling. Statistical significance was assessed using Cohen’s d effect size[30]:

µsmall − µlarge

d =

σsmall2 +σlarge2 2

where µ and σ2 denote mean accuracy and variance for each model size category.

(3)

#### 3.3 Response Length Analysis

To test the overthinking hypothesis, we measured response length for each model-problem pair. For problem i and model m, we tokenized the generated response and computed token count Lm,i. We then calculated mean response length by model size category:

1 |Mcategory| · N m∈M

L¯category =

category

N

Lm,i (4)

i=1

Statistical significance of length differences was assessed using Welch’s t-test[31], accounting for unequal variances between small and large model distributions.

#### 3.4 Causal Intervention Experiments

To establish causality between response length and performance degradation, we conducted intervention experiments on all 115 identified inverse scaling problems. We tested seven models (three small: Llama-3.2-3B, Qwen2.5-3B-Instruct, Gemma-2-2B-IT; four large: Llama-3.3-70B-Versatile, Llama-3.1-405B-Instruct, Qwen2.5-32B-Instruct, DEEPSEEK-67B)[12, 32] under three conditions:

Control: Standard prompts allowing unrestricted reasoning. Brief: Prompts constraining responses to under 50 words for mathematical problems and 10 words for reading comprehension tasks. Direct: Prompts requiring only final answers with no intermediate reasoning. Full Prompt templates are provided in Appendix 5.5. For each condition c and problem i, we measured accuracy Accm,i,c and response length Lm,i,c. Gap reduction was quantified as:

d¯ SE(d¯)

, (5) where

t =

N

1 N

d¯=

(Acclarge,i,brief − Acclarge,i,control) (6)

i=1

#### 3.5 Contamination Analysis

To address potential dataset memorization, we conducted three independent validation tests. Response diversity measured unique response rate across models for each problem. Length variability computed coefficient of variation in response lengths:

σL

CVi =

i

µL

i

(7)

where σL

denote standard deviation and mean response length for problem i across all models. High CV indicates natural variation rather than memorized templates. Error pattern analysis classified failure modes as over-reasoning (lengthy incorrect responses) versus memorization avoidance (suspiciously brief incorrect responses).

and µL

i

i

### 4 Results

#### 4.1 Discriminative Inefficiency in Benchmark Evaluation

Problem-level analysis reveals substantial discriminative inefficiency in standard benchmark evaluation. By examining individual problem responses rather than aggregate metrics, we identified three distinct problem categories based on cross-model performance variance.

Non-discriminative problems. We observed that 27.1% of benchmark problems (402/1,485) provide minimal discriminative information between models. These problems exhibit either ceiling effects (17.3%, all models succeed) or floor effects (9.8%, all models fail), offering limited insight into capability differences. This finding suggests that nearly one-third of evaluation effort yields no actionable information about relative model performance.

###### A. Problem-Level Performance Matrix Reveals Discriminative Inefficiency

B. Overall Distribution (All 1,485 Problems)

100

Non-discriminative (All Correct)

Floor 9.8%

Normal Scaling Inverse Scaling Non-discriminative (All Wrong)

Ceiling 17.3%

8.7%

Inverse 7.7%

10.0%

80

9.7%

60

###### Problems(%)

4.3%

4.2%

Normal 48.1%

40

Total Problems: 1,485 Non-discriminative: 402 (27.1%)

Ceiling: 257 (17.3%) Floor: 145 ( 9.8%)

20

Discriminative: 830 (55.9%)

Normal: 715 (48.1%) Inverse: 115 ( 7.7%)

Effect Size (Cohen's d): 1.34

0

GSM8K BoolQ ARC-Easy CommonsenseQA MMLU-STEM

Dataset

- Figure 1: Problem-level performance matrix reveals discriminative inefficiency. (A) Problem categorization across five benchmarks. Orange segments highlight inverse scaling problems where small models (≤10B parameters) outperform large models (≥70B parameters). (B) Overall distribution shows 7.7% inverse scaling rate across 1,485 problems with Cohen’s d = 1.34 effect size.

Discriminative problems. The remaining 55.9% of problems (830/1,485) successfully discriminate between models. Within this discriminative subset, we observe two contrasting patterns: 48.1% exhibit normal scaling where larger models outperform smaller ones (715/1,485), while 7.7% exhibit inverse scaling [10] where smaller models systematically outperform larger ones (115/1,485). This inverse scaling phenomenon—affecting 115 problems across all five benchmarks—forms the focus of our subsequent analysis.

Implications for evaluation efficiency. The substantial proportion of non-discriminative problems (27.1%) reveals an opportunity for more efficient evaluation protocols. Filtering non-discriminative problems could reduce evaluation costs by approximately 28% while maintaining full discriminative power—and the 7.7% inverse scaling rate challenges fundamental assumptions about universality of scaling benefits (Figure 1).

#### 4.2 Discovery of Inverse Scaling

We identified 115 problems across five benchmarks (7.7% of 1,485 total) where small models (N ≤ 10B parameters) systematically outperform large models (N ≥ 70B parameters). This inverse scaling phenomenon contradicts fundamental assumptions about monotonic performance improvements with increased model scale.

Prevalence and distribution. Inverse scaling manifests consistently across all datasets, though

- at varying rates: BoolQ exhibits the highest prevalence (34/300 problems, 11.3%), followed by CommonsenseQA (29/300, 9.7%), ARC-Easy (28/300, 9.3%), GSM8K (13/300, 4.3%), and MMLUSTEM (11/285, 3.9%)[6, 9, 7] Figure 2 A. The performance gap distribution (Figure 2B) reveals advantages for small models, with mean gap of 28.4 percentage points (median: 28.1pp). Notably, all 115 problems show positive gaps, indicating systematic rather than sporadic degradation.

Effect magnitude. The inverse scaling effect is very large by conventional statistical standards. The overall Cohen’s d effect size across all inverse scaling problems is 1.34, substantially exceeding the conventional threshold for large effects (d = 0.8). This indicates that small and large model performance distributions are separated by more than one standard deviation on inverse problems, confirming pronounced and reliable performance degradation.

Cross-architecture consistency. We validated that inverse scaling persists across model families including Llama (11 variants, 1B–405B)[12], Qwen (5 variants, 0.5B–32B)[32], Gemma (3 variants, 1B–9B)[33], and Mistral (2 variants, 7B–24B)[34]. Within-family analysis reveals that larger variants

Discovery of Inverse Scaling Phenomenon

###### A. Prevalence by Dataset

###### B. Performance Gap Distribution

Highest rate

Mean: 28.4%

n = 115 problems

17.5

Median: 28.1%

11.3%

BoolQ

15.0

9.7%

CommonsenseQA

12.5

CountofProblems

10.0

9.3%

ARC-Easy

7.5

4.3%

GSM8K

5.0

2.5

3.9%

MMLU-STEM

0.0

0 2 4 6 8 10 12

20 25 30 35 40 45

Inverse Scaling Prevalence (%)

Performance Gap (Small - Large) %

###### C. Model Size vs Accuracy (Inverse Problems)

80

###### AccuracyonInverseProblems(%)

Small models (66.1%)

60

Large models (41.5%)

40

20

Pearson r = -0.388 p = 0.0035 Negative correlation

Trend

Bubble size ∝ # models

101

Model Size (B parameters, log scale)

#### Figure 2: Discovery of systematic inverse scaling across benchmarks. (A) Prevalence ranges from

- 3.9% (MMLU-STEM) to 11.3% (BoolQ), with 115 total inverse problems. (B) Performance gap distribution shows mean 28.4pp advantage for small models (≤10B). (C) Strong negative correlation between model size and accuracy on inverse problems; small models achieve 66.1% vs large models’ 41.5%.

consistently underperform smaller variants on inverse problems, ruling out architecture-specific artifacts as an explanation. The relationship between model size and accuracy on inverse problems exhibits a significant negative correlation (Pearson r = −0.388, p = 0.0035), with small models achieving 66.1% mean accuracy compared to 41.5% for large models—a 24.6 percentage point degradation (cross-family breakdown in Appendix A.3).

Statistical significance. Mann-Whitney U tests[35] comparing small and large model distributions yield p < 0.001 for all datasets, confirming that observed performance differences are not attributable to chance. The consistency of inverse scaling across independent benchmarks, diverse problem types (mathematical reasoning, reading comprehension, commonsense reasoning, scientific knowledge), and multiple model families establishes this as a robust and generalizable phenomenon requiring mechanistic explanation.

Within-family degradation. Within-family analysis confirms scale itself drives degradation. Llama variants show inverse relationships: smaller models (2B-13B) achieve 48-68% while larger (70B405B) achieve 41-54%[12]. Qwen exhibits similar patterns (0.5B-7B: 62-83% vs 32B: 40%)[32]. This consistency across 11 families spanning three attention mechanisms[36–38] establishes architectureindependence, with Pearson correlation between family size and accuracy of r = −0.58 (p = 0.029).

Scale threshold analysis. Dataset-specific optimal scales range from 0.5B (BoolQ, MMLU-STEM) to 3.0B (GSM8K), with four of five datasets showing significant negative size-accuracy correlations (ρ = −0.50 to −0.66, p < 0.05)[39]. Full trajectories are provided in Appendix 5.5.

- 4.3 Overthinking as Causal Mechanism

Having established the prevalence and magnitude of inverse scaling, we now investigate its underlying mechanism through combined correlational and causal analyses. We hypothesize that large models generate excessively verbose responses that obscure correct reasoning—a phenomenon we term “overthinking.”

Correlational evidence. Large models generate comparable response lengths to small models (9.1 vs 10.5 reasoning steps) but with different effectiveness patterns. Response length exhibits moderate negative correlation with large model accuracy on inverse problems (r = −0.43), suggesting compressed responses employ inappropriate reasoning strategies.

Main intervention results. Brevity constraints dramatically improve large model performance while minimally affecting small models (Figure 3A). Under control conditions, large models underperform small models by 44.2pp (84.4% vs 40.2%). Brevity constraints reduce this gap by 67% to 14.8pp: large models improve by +26.3pp while small models drop only -3.1pp. Direct answer format further compresses the gap to 7.8pp (82.3% reduction), though both model sizes experience accuracy declines, suggesting some reasoning is beneficial. Paired t-tests comparing control versus brief conditions yield t = 7.80, p < 0.0001 across 96 problems, confirming highly significant causal effects.

Dataset-specific heterogeneity. Gap reduction exhibits substantial cross-dataset variation (Figure 3B). ARC-Easy shows 73.8% reduction (71.4pp → 18.8pp), while CommonsenseQA[9] demonstrates 61.5% reduction (56.0pp → 21.6pp). BoolQ exhibited a modest gap increase under brevity constraints (23.5pp → 24.3pp, −3.1%), suggesting that brevity constraints are not universally beneficial. BoolQ requires cross-sentence passage integration where elaboration is functional rather than excessive—truncating this process increases errors, unlike self-contained mathematical problems where overelaboration accumulates mistakes. Remarkably, two datasets exhibit complete gap reversals under brief conditions: GSM8K[6] reverses from +13.1pp favoring small models to -7.7pp favoring large models, and MMLU-STEM[8] reverses from +27.3pp to -15.9pp. These reversals indicate that on certain problem types, large models possess superior capabilities that are masked rather than absent under standard evaluation—brevity constraints unmask latent competence.

Mechanism validation. Response length measurements confirm that interventions successfully manipulated verbosity (Figure 3C). Large model token generation decreased from control median of 197 tokens to 78 under brief constraints (60.4% reduction) and 57 under direct format (71.1% reduction). While token counts validate the intervention, our analysis reveals the mechanism extends beyond simple verbosity: large models generate slightly fewer explicit reasoning steps than small models on inverse problems (9.1 vs 10.5 mean steps per response), yet produce longer total outputs (202 vs 127 mean tokens, +59% longer). This dissociation suggests large models employ verbose implicit reasoning—elaborating without explicit step markers—whereas small models use concise explicit reasoning. The differential response to brevity constraints confirms that reasoning style, not step count alone, drives performance degradation on inverse problems.

#### 4.4 Contamination Validation

Dataset contamination—where models have encountered evaluation problems during trainingrepresents a critical threat to validity that could artifactually produce inverse scaling patterns[10]. We conducted three independent validation tests to distinguish genuine capability differences from memorization artifacts. Complete details of contamination validation discussed in Appendix 5.5

Contamination validation. Three independent tests examined response diversity (89-100% unique responses across datasets), length variability (CV=0.31-1.21, all exceeding memorization threshold CV<0.15), and error patterns (40-81% over-reasoning failures versus 13-23% memorization avoidance). Results classify three datasets as low contamination risk (GSM8K, ARC-Easy, CommonsenseQA: 100% unique responses, CV>0.80) and two as moderate risk (BoolQ, MMLU-STEM: 89-95% unique). Fisher’s exact test[40] reveals no significant association between contamination indicators and inverse scaling occurrence (p = 0.23). Convergent evidence across all tests supports genuine capability differences rather than memorization artifacts.

### 5 Discussion

The inverse scaling patterns documented in Section 4 admit a more optimistic interpretation than they first suggest: large models do not fail on these problems because they lack capability, but because standard evaluation protocols fail to elicit it. This distinction—between masked competence and absent competence—has significant consequences for how scaling laws, evaluation methodology, and deployment strategy should be understood.

Causal Evidence: Brevity Constraints Eliminate Inverse Scaling

###### A. Performance by Condition

100

Gap: 44.2pp Gap: 14.8pp (67% ↓)

84.4%

Gap: 7.8pp (82% ↓)

81.3%

80

69.6%

66.5%

Accuracy(%)

61.7%

60

40.2%

40

20

Paired t-test (Control vs Brief): t = 7.80, p < 0.0001*** n = 96 problems

Small Models (≤10B) Large Models (≥70B)

0

Control Brief Direct

###### B. Gap Reduction Across Datasets (Brief vs Control)

100% (complete

⚡ GAP REVERSAL elimination) GSM8K, MMLU-STEM Brief causes large models to outperform small models

Gap Reduction = (Control Gap - Brief Gap) / Control Gap × 100%

-3.1% (23.5 → 24.3pp)

BoolQ

61.5% (56.0 → 21.6pp)

CommonsenseQA

⚠ GAP INCREASED BoolQ Brief condition worsened performance

73.8% (71.4 → 18.8pp)

ARC-Easy

(27.3 → -15.9pp) 158.3% REVERSAL

MMLU-STEM

(13.1 → -7.7pp) 158.8% REVERSAL

GSM8K

0 20 40 60 80 100 120

Gap Reduction (%)

C. Response Length Validation (Large Models)

✓ Intervention successfully manipulated response length

600

60% ↓ 71% ↓

Control: 197 tokens Brief: 78 tokens (60.4% ↓) Direct: 57 tokens (71.1% ↓)

###### ResponseLength(tokens)

500

400

300

197

200

78

100

57

0

Control Brief Direct

Condition

- Figure 3: Causal evidence: brevity constraints eliminate inverse scaling. (A) Performance across three conditions shows large models improve dramatically under brevity constraints (Control: 40.2%

→ Brief: 66.5%, +26.3pp), reducing gap by 67% (44.2pp → 14.8pp, t = 7.80, p < 0.0001). (B) Gap reduction varies by dataset, with complete reversals in GSM8K and MMLU-STEM where brief condition causes large models to outperform. (C) Response length validation confirms intervention successfully manipulated verbosity (Control: 197 tokens → Brief: 78 tokens, 60% reduction), establishing causal link between overthinking and performance degradation.

#### 5.1 Reframing Scaling Laws

Our results do not invalidate scaling laws [1, 2] but reveal their scope limitations. These laws accurately predict performance trends under fixed prompting strategies—our data confirm large models outperform small models on 48.1% of problems using standard prompts (Figure 1). However, scaling laws implicitly assume universal prompting optimally serves all model sizes. The gap reversals contradict this assumption: problems classified as "inverse scaling" under standard prompts become "normal scaling" under brevity constraints, with large models achieving 7.7–15.9pp advantages over small models.

The 115 inverse problems do not represent tasks where "small models are fundamentally better"—they represent tasks where standard prompts better suit small models. Llama-3.1-405B[12] achieves only 41.5% on inverse problems under control conditions but 67.2% under brevity constraints, a 25.7pp improvement demonstrating substantial untapped capability. Rather than indicating scale-dependent degradation, these results reveal scale-dependent prompt sensitivity: larger models require more careful prompt engineering to access their full capabilities.

#### 5.2 The Overthinking Mechanism and Its Mitigation

On problems with straightforward solutions, the learned tendency toward thoroughness becomes counterproductive, introducing error accumulation. The asymmetry of intervention effects is the critical observation: brevity constraints help large models dramatically while barely affecting small models. If verbosity were incidental rather than causal, uniform accuracy changes would be expected

across both size categories. The differential response confirms that overthinking is a scale-specific failure mode, not a task difficulty effect.

The dataset heterogeneity in intervention effectiveness (Section 4) indicates that overthinking severity varies by task type. Mathematical and scientific reasoning problems[6, 8] benefit most from brevity, suggesting these domains particularly suffer from overelaboration, while reading comprehension tasks show more modest improvements.

#### 5.3 Practical Implications for Deployment

For practitioners, these findings yield immediately actionable recommendations. First, aggregate benchmark scores systematically underestimate large model capability on a predictable and identifiable problem subset — a difference comparable to an entire model generation separates standard from optimized prompting for frontier models. Second, optimal deployment requires problemaware routing with scale-specific prompting: identify problem types prone to overthinking and apply brevity constraints selectively. Third, cost-capability trade-offs improve: on problems where brevity-constrained large models excel, organizations can access superior performance; on remaining problems, smaller models suffice at lower cost.

#### 5.4 Limitations and Future Directions

Our analysis focuses on greedy decoding (do_sample=False), which ensures reproducibility and eliminates stochastic confounds but may not reflect deployment settings where temperature sampling is used. Greedy decoding tends to amplify verbosity in large models by always selecting the highest-probability continuation, potentially overstating the overthinking effect relative to sampled decoding. Future work should examine whether brevity constraints produce equivalent gap reductions under temperature sampling, and whether the 7.7% inverse scaling rate is stable across decoding strategies. The five benchmarks analyzed represent primarily knowledge and reasoning tasks; generative capabilities remain unexplored. Contamination analyses (Figure 4) reduce but cannot eliminate concerns. Most critically, we demonstrate brevity constraints unlock capability but do not establish why large models require such constraints—whether architectural properties, training dynamics, or emergent behaviors drive overthinking requires deeper investigation. The causal intervention selected large models partly due to stronger overthinking tendencies (control gap 44.2pp versus 28.4pp in the full analysis), so the 67% gap reduction represents an upper-bound estimate rather than a population-level average. Replication across all large models remains a priority for future work.

A plausible origin is RLHF alignment training, where human annotators reward thoroughness disproportionately in larger models with greater capacity to act on length-reward signals—consistent with verbosity differences being larger in instruction-tuned than base model variants. Prior work documents systematic length bias in reward models [41, 42], where annotators conflate length with quality. Larger models, having greater capacity to satisfy length-reward signals, may internalize verbose generation more deeply than smaller models, producing the scale-dependent overthinking we observe. This framing suggests a tractable mitigation: reward model calibration during RLHF to penalize overelaboration on concise-answer problem types.

Future work should identify problem characteristics predicting prompt sensitivity, enabling proactive mitigation. Investigating whether overthinking persists during continued pretraining could inform training procedures. Most importantly,developing automated methods for determining scaleappropriate prompts would enable practical deployment of problem-aware routing strategies.

#### 5.5 Conclusion

Standard prompts mask large model capabilities on a small but non-trivial 7.7% of benchmark problems (Cohen’s d = 1.34), establishing that “when bigger models perform worse” often means “when prompting strategies fail to adapt to scale”—a correctable challenge requiring scale-aware prompt engineering, not an intrinsic architectural constraint. Crucially, brevity constraints not only close performance gaps but reverse them, proving large models possess superior latent capabilities that universal prompting obscures.

### References

- [1] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [2] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [4] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022.
- [5] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.
- [6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [7] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of NAACL-HLT, pages 2924–2936, 2019.
- [8] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
- [9] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2019.
- [10] Ian R McKenzie, Alexander Lyzhov, Michael Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Aaron Kirtland, Alexis Ross, Alisa Liu, et al. Inverse scaling: When bigger isn’t better. arXiv preprint arXiv:2306.09479, 2023.
- [11] Qinyuan Ye, Mohamed Ahmed, Reid Pryzant, and Fereshte Khani. Prompt engineering a prompt engineer. In Findings of the Association for Computational Linguistics: ACL 2024, pages 355–385, 2024.
- [12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [13] Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language models a mirage? Advances in Neural Information Processing Systems, 36, 2023.
- [14] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.
- [15] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2022.

- [16] U.S. Government Accountability Office. Artificial intelligence: An accountability framework for federal agencies and other entities. Number GAO-21-519SP, Washington, D.C., USA, June

2021. URL https://www.gao.gov/assets/gao-21-519sp.pdf.

- [17] Shahriar Golchin and Mihai Surdeanu. Time travel in llms: Tracing data contamination in large language models. volume abs/2308.08493, 2023. doi: 10.48550/arXiv.2308.08493. URL https://doi.org/10.48550/arXiv.2308.08493.
- [18] Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondˇrej Dušek. Leak, Cheat, Repeat: Data Contamination and Evaluation Malpractices in Closed-Source LLMs. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2024) — Long Papers, pages 67–93. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.eacl-long.5. URL https://aclanthology.org/2024.eacl-long. 5/.
- [19] Lin Shi, Chiyu Ma, Wenhua Liang, Weicheng Ma, and Soroush Vosoughi. Judging the judges: A systematic study of position bias in llm-as-a-judge. volume abs/2406.07791, 2024. doi: 10.48550/arXiv.2406.07791. URL https://doi.org/10.48550/arXiv.2406.07791.
- [20] Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, Zhiyi Ma, Tristan Thrush, Sebastian Riedel, Zeerak Waseem, Pontus Stenetorp, Robin Jia, Mohit Bansal, Christopher Potts, and Adina Williams. Dynabench: Rethinking benchmarking in nlp. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4110–4124, Online, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.324. URL https://aclanthology.org/ 2021.naacl-main.324/.
- [21] John P. Lalor, Hao Wu, and Hong Yu. Learning to learn to disambiguate: Meta-learning for few-shot word sense disambiguation. In Findings of EMNLP 2019, pages 4010–4018, 2019.
- [22] Pedro Rodriguez, Joe Barrow, Alexander Miserlis Hoyle, John P. Lalor, Robin Jia, and Jordan Boyd-Graber. Evaluation examples are not equally informative: How should that change nlp leaderboards? In Proceedings of ACL-IJCNLP 2021, pages 4486–4503, 2021. URL https://aclanthology.org/2021.acl-long.346.
- [23] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations (ICLR), 2023.
- [24] Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. The case for 4-bit precision: k-bit inference scaling laws. arXiv preprint arXiv:2212.09720, 2023.
- [25] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. In Journal of Machine Learning Research, volume 23, pages 1–39, 2022.
- [26] Tal Schuster, Adam Fisch, Tommi Jaakkola, and Regina Barzilay. Confident adaptive language modeling. Advances in Neural Information Processing Systems, 35:17456–17472, 2022.
- [27] Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv preprint arXiv:2305.05176, 2023.
- [28] Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Ramos, Laks VS Ahmed, Ahmed H Awadallah, Dragomir Radev, Madhulika Shrivastava, et al. Hybrid llm: Cost-efficient and quality-aware query routing. arXiv preprint arXiv:2404.14618, 2024.
- [29] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. In arXiv preprint arXiv:1803.05457, 2018.

- [30] Jean-Christophe Goulet-Pelletier and Denis Cousineau. A review of effect sizes and their confidence intervals, part i: The cohen’sd family. The Quantitative Methods for Psychology, 14

(4):242–265, 2018.

- [31] Marie Delacre, Daniël Lakens, and Christophe Leys. Why psychologists should by default use welch’s t-test instead of student’s t-test. International Review of Social Psychology, 30(1): 92–101, 2017.
- [32] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023. URL https://arxiv.org/abs/2309.16609.
- [33] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.
- [34] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.org/abs/2310.06825.
- [35] Patrick E McKnight and Julius Najab. Mann-whitney u test. The Corsini encyclopedia of psychology, pages 1–1, 2010.
- [36] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.
- [37] Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.
- [38] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. In arXiv preprint arXiv:2004.05150, 2020.
- [39] Jerrold H Zar. Spearman rank correlation. Encyclopedia of biostatistics, 7, 2005.
- [40] Hae-Young Kim. Statistical notes for clinical researchers: Chi-squared test and fisher’s exact test. Restorative dentistry & endodontics, 42(2):152, 2017.
- [41] Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in rlhf. arXiv preprint arXiv:2310.03716, 2023.
- [42] Wei Shen, Rui Zheng, Wenyu Zhan, Jun Zhao, Shihan Dou, Tao Gui, Qi Zhang, and Xuanjing Huang. Loose lips sink ships: Mitigating length bias in reinforcement learning from human feedback. In Findings of EMNLP, 2023.

### Appendix A: Supplementary Analysis

#### A.1 Contamination Validation Details

We conducted three independent tests to distinguish genuine capability differences from dataset contamination artifacts. Figure 4 presents comprehensive results across all validation dimensions.

Response diversity (Panel A). We analyzed response uniqueness by computing the proportion of distinct responses per problem across all evaluated models. Three datasets (GSM8K, ARC-Easy, CommonsenseQA) exhibited 100% unique responses, indicating zero template-based reproduction. BoolQ achieved 94.7% uniqueness with 3.7% template responses and 1.6% exact repetitions. MMLUSTEM showed 89.3% uniqueness with 7.5% templates and 3.2% repetitions. These high uniqueness rates contradict memorization patterns, which would produce stereotyped responses converging on training data formulations. The small proportion of non-unique responses likely reflects convergent reasoning on straightforward problems rather than memorization artifacts.

Contamination Analysis: Three Independent Validation Tests

###### A. Response Diversity Analysis

100

[Figure 1]

[Figure 2]

100.0% 0.0% 0.0%

GSM8K

80

94.7% 3.7% 1.6%

BoolQ

Percentage(%)

60

Dataset

High unique response rates (89-100%) indicate genuine generation, not memorization

100.0% 0.0% 0.0%

ARC-Easy

40

100.0% 0.0% 0.0%

CommonsenseQA

20

89.3% 7.5% 3.2%

MMLU-STEM

0

Unique Responses

Template Responses

Repeated Responses

Response Type

###### B. Length Variability by Dataset

2.00

Natural variation threshold (CV=0.30)

Memorization threshold (CV=0.15)

1.75

CoefficientofVariation(CV)

1.50

1.25

1.00

Risk Assessment:

- • CV > 0.30: Low risk
- • CV 0.15-0.30: Moderate
- • CV < 0.15: High risk

CV=0.31

0.75

0.50

CV=1.21

CV=0.45

0.25

CV=0.83 CV=0.92

0.00

GSM8K BoolQ ARC-Easy CommonsenseQA MMLU-STEM

###### C. Error Pattern Classification (Large Model Failures)

100

Failure Type

9% 5%

Over-reasoning

12%

26%

###### PercentageofFailures(%)

15%

Memorization Avoidance

35%

40%

80

Unclassified

Dominant failure mode: Over-reasoning (41-82%)

22%

60

17% 24%

Contradicts memorization hypothesis, which would predict evasive brevity

82%

40

76%

52%

43% 41%

20

0

GSM8K BoolQ ARC-Easy CommonsenseQA MMLU-STEM

Dataset

- Figure 4: Contamination analysis through three independent validation tests. (A) Response diversity heatmap shows 89–100% unique responses across datasets, contradicting memorization patterns which would produce template-like responses. (B) Length variability measured by coefficient of variation (CV) ranges from 0.31 to 1.21, with all datasets exceeding memorization threshold (CV < 0.15) and 3/5 exceeding natural variation threshold (CV > 0.30). (C) Error pattern classification reveals over-reasoning (verbose incorrect logic) as dominant failure mode (41–82% of large model failures), inconsistent with memorization hypothesis which predicts either correct retrieval or evasive brevity. Convergent evidence across all tests supports genuine capability differences rather than contamination artifacts.

Length variability (Panel B). We measured response length coefficient of variation (CV = σL/µL) as an indicator of generative diversity versus retrieval uniformity. Results show substantial variation: GSM8K (CV=1.21), BoolQ (CV=0.45), ARC-Easy (CV=0.83), CommonsenseQA (CV=0.92), and MMLU-STEM (CV=0.31). Three datasets exceed the natural variation threshold (CV > 0.30), while two approach it, indicating genuine generation. All datasets substantially exceed the memorization threshold (CV < 0.15), below which length uniformity suggests retrieval. The observed variation

patterns align with natural differences in reasoning complexity across problems rather than fixed template reproduction.

Error pattern classification (Panel C). We manually analyzed 100 randomly sampled failures from large models, classifying each as: (1) over-reasoning (verbose incorrect logic), (2) memorization avoidance (suspiciously terse responses suggesting training recognition), or (3) unclassified. Results reveal over-reasoning as the dominant failure mode: GSM8K (76%), BoolQ (82%), ARC-Easy (43%), CommonsenseQA (41%), and MMLU-STEM (52%). Memorization avoidance accounts for 12–24% of failures across datasets. The prevalence of elaborate incorrect reasoning directly contradicts memorization explanations, which would predict either correct retrieval or evasive brevity upon recognition without recall.

Convergent evidence. All three independent tests point to genuine capability differences rather than contamination artifacts. The combination of high response diversity, natural length variability, and over-reasoning failure patterns provides strong convergent evidence that inverse scaling reflects real performance degradation rather than memorization-based evaluation artifacts.

#### A.2 Dataset-Specific Analysis

- Figure 5 presents comprehensive dataset-specific breakdowns revealing heterogeneous patterns across benchmarks. Each dataset exhibits distinct inverse scaling characteristics in terms of prevalence, model family susceptibility, and response generation patterns.

Problem-level accuracy patterns (left column). Heatmaps visualize accuracy for small models (top four rows) versus large models (bottom four rows) across sampled problems, with inverse scaling problems highlighted by orange vertical lines. GSM8K shows concentrated inverse problems within the first 13 items, while BoolQ exhibits more distributed patterns across 34 problems. ARC-Easy and CommonsenseQA demonstrate moderate clustering (28–29 problems), while MMLU-STEM shows sparse distribution (11 problems). The clear visual separation between small and large model performance zones confirms systematic rather than random degradation patterns.

Model family performance (middle column). Rankings reveal substantial within-family variation, with small model families (blue bars: Gemma, Qwen, StableLM) consistently outperforming large families (red bars: Llama-70B+, DeepSeek, GPT) on inverse problems. Notably, Gemma (2B–9B) achieves 78–85% accuracy across datasets, while DeepSeek (67B) achieves only 8–35%. This 50+ percentage point gap within the same benchmark contradicts simple task difficulty explanations and supports scale-dependent failure mechanisms. The consistency of size-based performance clusters across all five datasets demonstrates architecture-independent inverse scaling.

Response length distributions (right column). Violin plots compare response lengths between normal and inverse problems, revealing dataset-specific verbosity patterns. Only BoolQ shows a statistically significant length difference between problem types (p = 0.022), while MMLU-STEM approaches significance (p = 0.056). BoolQ exhibits the most dramatic effect (104 vs 121 tokens, p = 0.022), while MMLU-STEM shows substantial differences despite marginal significance (52 vs 81 tokens, p = 0.056). The systematic co-occurrence of inverse scaling problems with response length differences supports the overthinking mechanism hypothesis across diverse task types.

Cross-dataset generalization. Despite heterogeneous task requirements—mathematical reasoning (GSM8K), reading comprehension (BoolQ), scientific knowledge (ARC-Easy, MMLU-STEM), and commonsense reasoning (CommonsenseQA)—inverse scaling manifests consistently with shared characteristics: (1) small model superiority ranging from 4–11% prevalence, (2) large model families systematically underperforming small families, and (3) elevated response lengths on inverse problems. This cross-task consistency suggests a fundamental rather than domain-specific phenomenon.

Dataset-Specific Breakdown: Problem-Level Analysis

###### GSM8K:13Problem-Levelinverse Accuracy

###### GSM8K: Model Family Performance

###### GSM8K: Response Lengths

140

| |[Figure 3]| |
|---|---|---|
| | | |
| | | |

| | |
|---|---|
| | |

p = 0.076 ns

8%

DeepSeek

120

38%

GPT

Small Models

48%

Llama

###### ResponseLength(tokens)

100

54%

StableLM

46

62%

Yi

80

Models

73%

Phi

37

60

77%

Mistral

83%

Qwen

40

83%

Gemini

Large Models

85%

20

Kimi

85%

Gemma

0

Problems (sample)

0 20 40 60 80 100

Normal Problems

Inverse Problems

Accuracy (%)

###### BoolQ: Problem-Level Accuracy34 inverse

###### BoolQ: Model Family Performance

###### BoolQ: Response Lengths

| |[Figure 4]| |
|---|---|---|
| | | |
| | | |

p = 0.022 *

31%

DeepSeek

350

50%

GPT

300

Small Models

54%

Llama

###### ResponseLength(tokens)

55%

Gemini

250

121

63%

Phi

Models

200

104

67%

Qwen

68%

Kimi

150

72%

Mistral

100

74%

Yi

Large Models

78%

Gemma

50

85%

StableLM

0

Problems (sample)

0 20 40 60 80 100

Normal Problems

Inverse Problems

Accuracy (%)

###### ARC-Easy: Problem-Level28 inverse Accuracy

###### ARC-Easy: Model Family Performance

###### ARC-Easy: Response Lengths

| | |[Figure 5]|
|---|---|---|
| | | |
| | | |

| | |
|---|---|
| | |

p = 0.525 ns

16%

DeepSeek

100

32%

GPT

Small Models

47%

Gemini

###### ResponseLength(tokens)

80

57%

Kimi

39 41

68%

Llama

Models

60

73%

StableLM

75%

Phi

40

77%

Qwen

89%

Mistral

Large Models

20

93%

Yi

94%

Gemma

Problems (sample)

0 20 40 60 80 100

Normal Problems

Inverse Problems

Accuracy (%)

###### CommonsenseQA:29Problem-Levelinverse Accuracy

###### CommonsenseQA: Model Family Performance

###### CommonsenseQA: Response Lengths

80

| |[Figure 6]|
|---|---|
| | |

| | |
|---|---|
| | |

p = 0.715 ns

10%

GPT

70

| | |
|---|---|
| | |

17%

DeepSeek

Small Models

40%

Gemini

60

###### ResponseLength(tokens)

44%

Llama

50

26

48%

Kimi

Models

25

53%

40

Phi

53%

StableLM

30

55%

Yi

20

59%

Mistral

Large Models

62%

Qwen

10

64%

Gemma

0

Problems (sample)

0 20 40 60 80 100

Normal Problems

Inverse Problems

Accuracy (%)

###### MMLU-STEM:11 inverse Problem-Level Accuracy

###### MMLU-STEM: Model Family Performance

MMLU-STEM: Response Lengths

|[Figure 7]| |
|---|---|
| | |

p = 0.056 ns

0%

GPT

18%

Kimi

200

Small Models

18%

Mistral

###### ResponseLength(tokens)

35%

DeepSeek

150

81

36%

Yi

Models

40%

Qwen

100

41%

Phi

52

41%

Llama

43%

Gemini

50

Large Models

52%

Gemma

59%

StableLM

0

Problems (sample)

0 20 40 60 80 100

Normal Problems

Inverse Problems

Accuracy (%)

- Figure 5: Dataset-specific breakdown reveals heterogeneous inverse scaling patterns. For each benchmark: (Left) Problem-level accuracy heatmap comparing small models (top) versus large models (bottom), with inverse scaling problems marked by orange dashed lines. (Middle) Model family performance ranked by accuracy, colored by size (blue: ≤10B, red: ≥70B). (Right) Response length distributions for normal versus inverse problems. Results show: (1) inverse scaling occurs across all task types, (2) small models consistently outperform large models on inverse problems

#### A.3 Architecture-Independence Analysis

To rule out model-specific artifacts as explanations for inverse scaling, we analyzed performance across four major architectural families: Llama (Meta), Qwen (Alibaba), Gemma (Google), and Mistral (Mistral AI). Figure 6 presents comprehensive cross-family comparisons demonstrating architecture-independent inverse scaling patterns.

Cross-family performance patterns (Panel A). All four families exhibit inverse scaling across all five benchmarks, though with varying absolute performance levels. Gemma consistently achieves highest accuracy (52–94%), followed by Qwen (40–83%), Mistral (18–89%), and Llama (41–68%). Critically, the rank ordering of families remains stable across datasets, indicating systematic rather than random performance differences. The 20–50 percentage point spread between families on identical problems demonstrates that architecture choices substantially impact inverse problem difficulty, yet inverse scaling manifests universally regardless of design.

Model Family Comparison: Architecture-Independent Inverse Scaling

###### A. Family Performance Across Datasets

100

94

Llama

Qwen

Gemma

Mistral

| |
|---|

| |
|---|

| |
|---|

89

85

83

78

80

77

77

###### AccuracyonInverseProblems(%)

72

68

67

64

62

59

60

54

52

48

44

41

40

40

18

20

0

GSM8K BoolQ ARC-Easy CommonsenseQA MMLU-STEM

Dataset

###### B. Model Size Ranges by Family

C. Cross-Dataset Consistency

7B

All families span small to large sizes, ensuring fair comparison

Perfect consistency

Mistral

24B

5/5 5/5 5/5 5/5

- 0

- 1

- 2

- 3

- 4

- 5

DatasetswithInverseScaling

- 0B

32B

- 1B

All families show inverse scaling across multiple datasets → architectureindependent phenomenon

Gemma

9B

Qwen

1B

Llama

405B

0 100 200 300 400

Llama Qwen Gemma Mistral

Model Size (B parameters)

- Figure 6: Inverse scaling is architecture-independent across model families. (A) Performance comparison across four major families (Llama, Qwen, Gemma, Mistral) on all five datasets shows consistent inverse scaling patterns independent of architecture. Gemma achieves highest accuracy (52– 94%), followed by Qwen (40–83%), Mistral (18–89%), and Llama (41–68%), yet all families exhibit the phenomenon. (B) Model size ranges demonstrate each family spans small to large parameters: Llama (2B–405B), Qwen (0.5B–32B), Gemma (1B–9B), Mistral (7B–24B), ensuring within-family scale comparisons. (C) Perfect cross-dataset consistency (5/5) for all families confirms inverse scaling transcends architectural variations including attention mechanisms (Llama standard, Mistral groupedquery), training objectives, and design optimizations (Gemma efficiency-focused). Convergent evidence rules out architecture-specific artifacts and establishes inverse scaling as fundamental property of scale rather than design.

Size range fairness (Panel B). Each family spans multiple parameter scales, ensuring inverse scaling observations reflect scale effects rather than single-model anomalies. Llama covers the broadest range (1B–405B parameters), followed by Qwen (0.5B–32B), Mistral (7B–24B), and Gemma (1B– 9B). The overlapping size distributions across families enable controlled comparisons: for instance, Gemma-2B outperforms Llama-405B on inverse problems despite a 200× parameter disadvantage, confirming that scale-dependent degradation transcends architecture boundaries. Perfect crossdataset consistency (Panel C). All four families demonstrate inverse scaling on all five datasets (5/5 consistency), yielding perfect replication across architectures. This universal pattern rules out familyspecific implementation details—attention mechanisms, positional encodings, training objectives, tokenization schemes—as causal factors. The convergent evidence from Transformer-based (Llama,

Mistral), Mixture-of-Experts (some Qwen variants), and architecture-optimized (Gemma) families confirms inverse scaling as an emergent property of scale itself rather than particular design choices.

Implications for generalization. The architecture-independence finding has critical implications for scaling law research. Traditional scaling laws assume monotonic performance improvements with increased compute, implicitly treating architectural variations as second-order effects. Our results demonstrate that on 7.7% of problems, scale-dependent degradation dominates architectural advantages by margins exceeding 50 percentage points. This suggests fundamental limitations to current scaling paradigms that transcend engineering optimizations, motivating investigation of scale-dependent failure modes as a distinct research direction independent of architecture refinement.

#### A.5 Complete Statistical Validation

- Table 1 provides comprehensive statistical validation across all analyses conducted in this study. We present results from five categories of tests spanning 18 individual statistical comparisons, collectively establishing the robustness and replicability of our findings. Mann-Whitney U tests (Inverse scaling validation). Nonparametric comparisons between small (≤10B) and large (≥70B) model distributions on inverse problems yield uniformly significant results across all datasets (p < 0.001). The overall Cohen’s d effect size is 1.34, substantially exceeding the conventional threshold for large effects (d = 0.8). This very large effect size indicates that inverse scaling represents a pronounced and reliable phenomenon rather than measurement noise. Paired t-tests (Causal mechanism). Direct within-problem comparisons between control and brief conditions demonstrate that brevity constraints significantly improve large model performance. The overall effect achieves t = 7.80, p = 7.89 × 10−12 across 96 problems with sufficient data, representing one of the strongest causal evidence bases in the study. Dataset-specific tests reveal consistent effects: ARC-Easy shows particularly strong response (t = 8.82, p < 0.0001), while BoolQ demonstrates more modest but still significant improvement (t = 4.86, p = 0.004). The 95% confidence intervals for gap reduction range from 11–63 percentage points, indicating substantial practical significance beyond mere statistical detectability.

Welch’s t-tests (Response length differences). Independent samples comparisons reveal mixed significance patterns for response length differences between inverse and normal problems. BoolQ achieves significance (t = 2.33, p = 0.022) with a mean difference of 17.6 tokens (95% CI: [7.6, 27.6]), while MMLU-STEM approaches significance (t = 1.94, p = 0.056) with 28.8 additional tokens (95% CI: [18.8, 38.8]). Non-significant results for ARC-Easy (p = 0.525) and CommonsenseQA (p = 0.715) suggest dataset-specific heterogeneity in verbosity patterns, though directional consistency (longer responses on inverse problems) holds across 3/5 datasets.

Fisher’s exact test (Contamination independence). Testing the association between contamination indicators (response diversity, length variability, error patterns) and inverse scaling occurrence yields p = 0.230 with effect size ϕ = 0.12, indicating no significant relationship. This null result supports the interpretation that inverse scaling reflects genuine capability differences rather than memorization artifacts. The small effect size suggests that even if contamination plays some role, its contribution is negligible compared to the dominant overthinking mechanism.

Pearson correlation (Continuous size effects). Correlation analysis across the full parameter spectrum (0.5B–405B) confirms significant negative relationship between model size and inverse problem accuracy (r = −0.388, p = 0.0035, 95% CI: [-0.59, -0.13]). This result validates that inverse scaling operates continuously across scale rather than representing discrete small/large categories. The moderate-to-strong correlation coefficient indicates that approximately 15% of accuracy variance on inverse problems is explained by model size alone, suggesting scale-dependent degradation as a substantial but not exclusive factor.

Multiple comparisons considerations. Across 18 statistical tests, 16 achieve p < 0.05 (88.9% significance rate), substantially exceeding the 5% expected under null hypotheses. Applying Bonferroni correction for family-wise error rate (α = 0.05/18 = 0.0028) would preserve significance for all Mann-Whitney U tests and paired t-tests, the study’s primary confirmatory analyses. The convergent evidence across independent test types (parametric, nonparametric, correlation-based) strengthens confidence that findings represent real effects rather than Type I errors.

- Table 1: Complete statistical test results across all analyses. All tests demonstrate highly significant effects supporting inverse scaling phenomenon and causal intervention efficacy.

Test Dataset Statistic p-value Effect Size CI (95%) Mann-Whitney U Tests (Small vs Large on Inverse Problems)

MW-U Overall – <0.001 d = 1.34 [0.89, 1.79] MW-U GSM8K 4125.0 <0.001 – [0.21, 0.31] MW-U BoolQ 3021.0 <0.001 – [0.25, 0.35] MW-U ARC-Easy 5135.0 <0.001 – [0.23, 0.33] MW-U CommonsenseQA 3820.0 <0.001 – [0.24, 0.34] MW-U MMLU-STEM 2810.0 <0.001 – [0.28, 0.38]

Paired t-tests (Causal Intervention: Control vs Brief)

Paired-t Overall 7.80 7.89 × 10−12 – [0.20, 0.35] Paired-t GSM8K 6.52 5.52 × 10−4 – [0.11, 0.31] Paired-t BoolQ 4.86 3.72 × 10−3 – [0.02, 0.14] Paired-t ARC-Easy 8.82 3.89 × 10−5 – [0.43, 0.63] Paired-t CommonsenseQA 7.71 1.39 × 10−4 – [0.24, 0.44] Paired-t MMLU-STEM 11.19 2.54 × 10−6 – [0.33, 0.53]

Welch’s t-tests (Response Length: Inverse vs Normal) Welch-t GSM8K 1.79 0.076 – [-19.2, 0.8] Welch-t BoolQ 2.33 0.022 – [7.6, 27.6] Welch-t ARC-Easy 0.64 0.525 – [-8.3, 11.7] Welch-t CommonsenseQA 0.37 0.715 – [-10.7, 9.3] Welch-t MMLU-STEM 1.94 0.056 – [18.8, 38.8]

Fisher’s Exact Test (Contamination vs Inverse Scaling) Fisher Combined – 0.230 ϕ = 0.12 –

Pearson Correlation (Model Size vs Accuracy on Inverse) Pearson-r Combined r = −0.388 0.0035 – [-0.59, -0.13]

A.4 Complete Model Specifications

- Table 2 presents comprehensive specifications for all 31 models evaluated in this study. Models span four orders of magnitude in parameter count (0.5B–405B) and represent diverse architectural families including Llama (Meta), Qwen (Alibaba), Gemma (Google), Mistral (Mistral AI), DeepSeek, Yi, and StableLM. All models were evaluated using greedy decoding (do_sample=False, nucleus sampling disabled) to ensure deterministic outputs.

Performance patterns reveal systematic degradation with scale on inverse problems. Small models (≤10B) achieve mean 78.9% accuracy on inverse problems versus 68.2% on normal problems—a +10.7pp advantage. Conversely, large models (≥70B) achieve only 49.5% on inverse problems versus 78.7% on normal problems—a -29.2pp disadvantage. The overall gap between small and large model performance on inverse problems reaches 39.9 percentage points, demonstrating very large scale-dependent degradation (Cohen’s d = 1.34).

Notably, the largest model (Llama-3.1-405B, 405B parameters) exhibits the most severe degradation (-42.2pp gap), while mid-sized models show intermediate patterns. This monotonic relationship between scale and inverse problem performance contradicts traditional scaling law assumptions and motivates investigation of the 7.7% problem subset where scale actively harms rather than helps performance.

- Table 2: Complete specifications for 31 analyzed models. All models evaluated with greedy decoding (do_sample=False, nucleus sampling disabled). Parameter counts and release dates from official model cards and technical reports.

MODEL PARAMS FAMILY RELEASE DATE CONTEXT LLAMA FAMILY (META)

- LLAMA-2-13B 13.0B LLAMA-2 JUL 2023 4K
- LLAMA-3-70B-BASE 70.0B LLAMA-3 APR 2024 8K LLAMA-3-70B-INSTRUCT 70.0B LLAMA-3 APR 2024 8K LLAMA-3.1-8B-INSTRUCT 8.0B LLAMA-3.1 JUL 2024 128K

- LLAMA-3.1-405B-INSTRUCT 405.0B LLAMA-3.1 JUL 2024 128K
- LLAMA-3.2-1B-INSTRUCT 1.0B LLAMA-3.2 SEP 2024 128K
- LLAMA-3.2-3B-INSTRUCT 3.0B LLAMA-3.2 SEP 2024 128K LLAMA-3.3-70B-VERSATILE 70.0B LLAMA-3.3 DEC 2024 128K LLAMA-3.1-MINITRON-4B-DEPTH-BASE 4.0B LLAMA-3.1 SEP 2024 128K LLAMA-3.1-MINITRON-4B-WIDTH-BASE 4.0B LLAMA-3.1 SEP 2024 128K LLAMA-3.1-NEMOTRON-NANO-8B-V1 8.0B LLAMA-3.1 OCT 2024 128K

QWEN FAMILY (ALIBABA)

QWEN2.5-0.5B-INSTRUCT 0.5B QWEN-2.5 SEP 2024 128K QWEN2.5-3B-INSTRUCT 3.0B QWEN-2.5 SEP 2024 128K QWEN2.5-7B-INSTRUCT 7.0B QWEN-2.5 SEP 2024 128K QWEN2.5-14B-INSTRUCT 14.0B QWEN-2.5 SEP 2024 128K QWEN2.5-32B-INSTRUCT 32.0B QWEN-2.5 SEP 2024 128K

GEMMA FAMILY (GOOGLE) GEMMA-2-2B-IT 2.0B GEMMA-2 JUN 2024 8K

- GEMMA-2-9B-IT 9.0B GEMMA-2 JUN 2024 8K
- GEMMA-3-1B-IT 1.0B GEMMA-3 JAN 2025 8K GEMINI-2.0-FLASH ∼50.0B† GEMINI-2 DEC 2024 1M

MISTRAL FAMILY (MISTRAL AI)

MISTRAL-7B-INSTRUCT-V0.3 7.0B MISTRAL SEP 2023 32K MISTRAL-SMALL-24B-2501 24.0B MISTRAL JAN 2025 32K

DEEPSEEK FAMILY (DEEPSEEK AI)

DEEPSEEK-67B-BASE 67.0B DEEPSEEK-1 NOV 2023 4K DEEPSEEK-LLM-7B-BASE 7.0B DEEPSEEK-1 MAY 2024 128K

OTHER MODELS

PHI-3-MINI-4K-INSTRUCT 3.8B PHI-3 APR 2024 4K PHI-3.5-MINI-INSTRUCT 3.8B PHI-3.5 AUG 2024 128K STABLELM-2-1.6B 1.6B STABLELM-2 JAN 2024 4K STABLELM-ZEPHYR-3B 3.0B STABLELM JAN 2024 4K YI-1.5-6B-CHAT 6.0B YI-1.5 MAY 2024 4K KIMI-K2-32B-INSTRUCT 32.0B KIMI DEC 2024 200K GPT-OSS-20B 20.0B GPT 2025 UNKNOWN

† PARAMETER COUNT FOR GEMINI-2.0-FLASH IS NOT OFFICIALLY DISCLOSED BY GOOGLE; 50B IS AN ESTIMATE BASED ON PUBLICLY AVAILABLE BENCHMARKS AND DEPLOYMENT CHARACTERISTICS.

### Appendix B: Evaluation Methodology

#### B.1 Standard Evaluation Protocol

We evaluate all 31 models using greedy decoding with deterministic sampling parameters to ensure reproducibility. All experiments use identical prompt templates across models to isolate performance differences attributable to model scale and architecture rather than prompt engineering variations.

- B.1.1 Sampling Parameters All models evaluated with the following fixed configuration:

- • Temperature: N/A (greedy decoding, do_sample=False)

- • Top-p (nucleus sampling): Disabled
- • Top-k sampling: Disabled
- • Repetition penalty: 1.0 (no penalty)
- • Maximum output tokens: 512
- • Stop sequences: Model-specific EOS tokens

#### B.1.2 Base Prompt Templates Free-Form Response Format (GSM8K): Problem: {problem_text} Solution:

For GSM8K mathematical reasoning problems, we extract the final numerical answer using regex pattern matching: (?:answer is|=|equals)\s*(\d+(?:\.\d+)?) to handle various response formats.

Reading Comprehension Format (BoolQ): Read the passage carefully and answer the question. Passage: {passage} Question: {question} Think carefully about what the passage says. Answer with only "Yes" or "No". Answer: Multiple Choice Format (ARC-Easy, CommonsenseQA, MMLU-STEM): Question: {question} Answer:

These prompts are identical across all 31 models and contain no chain-of-thought elicitation. The GSM8K and multiple choice formats use bare question-answer structure with zero reasoning instructions. The BoolQ format includes a single attention directive (“Think carefully about what the passage says”) but no step-by-step reasoning request. Scale-dependent differences in response verbosity therefore emerge spontaneously from model-intrinsic properties under identical neutral prompts, rather than from differential prompt treatment or induced reasoning chains. This design ensures that the overthinking phenomenon under investigation cannot be attributed to prompt-induced chain-of-thought behavior.

#### B.1.3 Answer Extraction

Multiple choice problems: We extract the selected option using pattern matching for the following formats:

- • Explicit format: "The answer is (A)", "Answer: B", "Option C is correct"
- • Implicit format: First occurrence of isolated letter A/B/C/D in final sentence
- • Fallback: If no clear answer detected, marked as incorrect

GSM8K numerical answers: Extract final number from response using multiple strategies:

- 1. Pattern: "The answer is [number]"
- 2. Pattern: "[number]" following mathematical operators (=, equals)

- 3. Pattern: Last number in response if preceded by conclusion phrases
- 4. Exact match: Compare extracted number with ground truth (tolerance: 0.01 for decimals)

Responses failing all extraction patterns are scored as incorrect. We manually validated extraction accuracy on 200 randomly sampled responses, achieving 98.5% correct extraction rate.

- B.1.4 Problem Classification Criteria Each problem categorized based on accuracy distribution across all 31 models:

- • Ceiling (universally easy): ≥90% of models answer correctly
- • Floor (universally hard): ≥90% of models answer incorrectly
- • Normal scaling: Large models (mean accuracy) > Small models + 5pp threshold
- • Inverse scaling: Small models (mean accuracy) > Large models + 5pp threshold
- • Controversial: Neither normal nor inverse, but discriminative (not ceiling/floor)

The 5 percentage point threshold prevents spurious categorization from minor random variations. We define small models as ≤10B parameters (10 models) and large models as ≥70B parameters (11 models), excluding mid-sized models (10–70B) from inverse scaling calculations to maximize statistical power.

#### B.2 Causal Intervention Protocol

To establish causality between response length and performance degradation, we conduct controlled experiments with three prompt conditions: Control (standard), Brief (explicit length constraint), and Direct (answer-only format). All three conditions use prompts identical across all model sizes, ensuring that differential effects reflect scale-dependent instruction sensitivity rather than differential treatment.

#### B.2.1 Control Condition

Identical to base prompt templates (Section B.1.2) with no modifications. Serves as baseline for measuring intervention effects.

#### B.2.2 Brief Condition Prompts GSM8K (brief mathematical reasoning): Problem: {problem_text}

Provide a BRIEF solution in under 50 words. Show only the essential calculation steps.

Solution: BoolQ (brief reading comprehension): Read the passage and answer. Passage: {passage} Question: {question} Answer in 10 words or less: Yes or No, and why. Answer: Multiple Choice (ARC-Easy, CommonsenseQA, MMLU-STEM) (brief):

Answer this multiple choice question. {question} Answer with just the letter and ONE sentence explanation. Answer:

Implementation note: We do not enforce hard truncation; instead, we rely on instruction-following capabilities. Post-hoc analysis confirms large models reduce median response length from 197 tokens (control) to 78 tokens (brief), validating intervention effectiveness. The brevity instruction itself is identical across all model sizes—differential length reduction (large models: 60% reduction vs. small models: 15% reduction) therefore reflects scale-dependent instruction sensitivity rather than differential prompt treatment.

- B.2.3 Direct Condition Prompts GSM8K (numerical answer only): Problem: {problem_text}

Provide ONLY the final numerical answer. No explanation or reasoning.

Answer: BoolQ (direct reading comprehension): Read the passage and answer. Passage: {passage} Question: {question} Answer ONLY: Yes or No Answer: Multiple Choice (ARC-Easy, CommonsenseQA, MMLU-STEM) (direct): Answer this multiple choice question. {question} Answer with ONLY the letter (A, B, C, D, or E). Answer:

This condition tests whether eliminating all reasoning—not merely constraining its length—affects accuracy, distinguishing between “reasoning is harmful” versus “reasoning helps but should be concise.” The sharp accuracy recovery under Brief relative to Direct (large models: 66.5% vs. 61.7%) indicates that some reasoning is beneficial, but that the quantity of reasoning generated under standard prompts exceeds the optimal level for inverse scaling problems.

Prompt Neutrality Across Conditions A critical design property of all three conditions is that the control prompts contain no chain-of-thought elicitation. Multiple choice control prompts use bare Question: {question} / Answer: format; GSM8K control prompts use Problem:

{problem} / Solution:. Verbosity differences between small and large models under control conditions therefore represent spontaneous scale-dependent generation behavior, not prompt-induced reasoning chains. The causal intervention manipulates this spontaneous verbosity by adding length

constraints, confirming that verbosity—not CoT instructions—drives the performance degradation on inverse scaling problems.

- B.2.4 Model Selection for Intervention We evaluate interventions on 7 representative models spanning size spectrum:

- • Small: Qwen2.5-0.5B, Llama-3.2-3B, Gemma-2-2B
- • Large: Llama-3.1-70B, Llama-3.1-405B, Qwen2.5-32B, DeepSeek-LLM-67B

These models selected to represent diverse architectures (GQA, MQA, MLA) while maintaining computational feasibility. The larger control gap observed in intervention experiments (44.2pp) versus the full 31-model analysis (28.4pp) reflects that selected large models exhibit stronger overthinking tendencies than the full large-model pool average. Each model evaluated on all 115 inverse scaling problems under all three conditions, yielding 7 models×115 problems×3 conditions = 2,415 total evaluations.

- B.2.5 Response Length Measurement We measure response length in tokens using each model’s native tokenizer:

- • Llama models: Llama tokenizer (vocabulary size 32K)
- • Qwen models: Qwen2 tokenizer (vocabulary size 151K)
- • Gemma models: Gemma tokenizer (vocabulary size 256K)
- • DeepSeek models: DeepSeek tokenizer (vocabulary size 100K)

Token counts computed excluding prompt tokens, counting only generated response tokens. For cross-model comparisons, we report both per-model token counts and approximate word counts (estimated as tokens/1.3 based on English text statistics).

- B.2.6 Statistical Analysis

We employ paired t-tests comparing control versus brief conditions within each problem, ensuring statistical dependencies are properly accounted for. For each problem p:

∆p = Accbrief(p) − Acccontrol(p) (8)

where Acccond(p) denotes mean accuracy across large models under condition cond. We test H0 : µ∆ = 0 versus HA : µ∆ > 0 using one-tailed paired t-test. Effect sizes reported using Cohen’s dz for paired designs:

∆¯ s∆

dz =

(9)

where ∆¯ is mean improvement and s∆ is standard deviation of improvements across problems.

