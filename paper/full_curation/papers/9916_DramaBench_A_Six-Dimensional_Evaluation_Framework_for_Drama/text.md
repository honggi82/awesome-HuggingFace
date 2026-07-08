# arXiv:2512.19012v2[cs.CL]23Dec2025

## DramaBench: A Six-Dimensional Evaluation Framework for Drama Script Continuation

Shijian MA1, Yunqi HUANG2, Yan LIN1,∗ 1University of Macau 2University College London mas8069@foxmail.com, yunqi.huang.23@ucl.ac.uk, yanlin@um.edu.mo ∗Corresponding author

### Abstract

Drama script continuation requires models to maintain character consistency, advance plot coherently, and preserve dramatic structurecapabilities that existing benchmarks fail to evaluate comprehensively. We present DramaBench, the first large-scale benchmark for evaluating drama script continuation across six independent dimensions: Format Standards, Narrative Efficiency, Character Consistency, Emotional Depth, Logic Consistency, and Conflict Handling. Our framework combines rulebased analysis with LLM-based labeling and statistical metrics, ensuring objective and reproducible evaluation. We conduct comprehensive evaluation of 8 state-of-the-art language models on 1,103 scripts (8,824 evaluations total), with rigorous statistical significance testing (252 pairwise comparisons, 65.9% significant) and human validation (188 scripts, substantial agreement on 3/5 dimensions). Our ablation studies confirm all six dimensions capture independent quality aspects (mean |r| = 0.020). DramaBench provides actionable, dimensionspecific feedback for model improvement and establishes a rigorous standard for creative writing evaluation.

### 1 Introduction

Drama scripts are a unique form of creative writing that combine narrative storytelling with specific structural constraints. Unlike general prose or dialogue generation, drama script continuation demands that language models simultaneously handle multiple quality dimensions: maintaining established character personalities, advancing plot through meaningful events, preserving logical consistency with prior context, managing emotional arcs, and escalating dramatic conflicts—all while adhering to screenplay formatting standards.

Despite the growing capabilities of large language models in creative writing tasks, existing benchmarks for evaluating script generation focus

primarily on full-script generation from scratch (Mirowski et al., 2023; Zhou et al., 2024) or lack quantitative, multi-dimensional evaluation frameworks. Story generation benchmarks like ROCStories (Mostafazadeh et al., 2016) evaluate story coherence through cloze tests but do not capture the screenplay-specific requirements of format compliance, character voice distinctiveness, or dramatic structure. General NLG evaluation frameworks like UniEval (Zhong et al., 2022) provide multidimensional assessment but are not specialized for the unique challenges of drama script continuation.

This gap creates three critical challenges for evaluating drama script continuation systems: (1) Multi-dimensional quality: Script quality cannot be captured by a single aggregate score—a script may have perfect format but poor characterization, or vice versa; (2) Subjective evaluation: Traditional human evaluation of creative writing is expensive, slow, and suffers from low inter-annotator agreement; (3) Lack of actionable feedback: Aggregate quality scores fail to identify specific weaknesses for model improvement.

We address these challenges with DramaBench, a comprehensive benchmark that evaluates drama script continuation through six independent dimensions using a novel LLM Labeling + Statistical Analysis methodology (Figure 1). Unlike LLMas-a-Judge approaches that ask models to directly score quality (Zheng et al., 2023), our framework uses LLMs as structured data annotators that extract categorical labels (e.g., “driver beat” vs “static beat”), which are then converted to objective metrics through statistical formulas. This approach ensures reproducibility, interpretability, and enables the extracted labels to serve as training data for model improvement.

#### Our contributions are:

1. DramaBench benchmark: The first largescale benchmark for drama script con-

[Figure 1]

- Figure 1: Overview of the DramaBench evaluation framework. The pipeline consists of three components: (1) Task input with script context and model continuation, (2) Six independent evaluation dimensions (Format, Narrative, Character, Emotion, Logic, Conflict), and (3) Structured LLM labeling framework that extracts categorical labels (not direct scores) which are then aggregated into objective metrics. This approach ensures reproducibility and provides actionable feedback for model improvement.

tinuation, containing 1,103 professionallystructured scripts with scene-boundary-aware context-continuation splits (8,824 modelscript evaluations across 8 SOTA models).

- 2. Six-dimensional evaluation framework: A hybrid evaluation system combining rulebased analysis (Format Standards) with LLMlabeled dimensions (Narrative Efficiency, Character Consistency, Emotional Depth, Logic Consistency, Conflict Handling), each with precisely defined annotation units and statistical metrics.
- 3. Rigorous validation: Comprehensive statistical significance testing (252 Mann-Whitney U tests with FDR correction, 65.9% significant comparisons), ablation studies confirming dimension independence (mean correlation |r| = 0.014), and human validation on 188 scripts showing substantial agreement on 3/5 dimensions (κ = 0.42–0.53).
- 4. Systematic analysis: In-depth error taxonomy classifying 10,850 errors, 24 case studies demonstrating model strengths/weaknesses, and model-specific error profiles revealing actionable improvement opportunities.
- 5. Public release: All evaluation scripts, metrics,

and analysis code will be released to enable reproducible drama generation research.

Our evaluation reveals that no single model excels across all dimensions: GPT-5.2 leads in overall robustness (narrative efficiency, character consistency, logic), Qwen3-Max specializes in emotional depth, and Gemini-3-Pro excels at conflict management. All models achieve near-perfect format compliance, but show significant variance in logic consistency (2–5% error rates). These findings demonstrate that multi-dimensional evaluation is essential for understanding model capabilities and guiding targeted improvements.

### 2 Related Work

#### 2.1 Story and Drama Generation

Early story generation benchmarks focused on commonsense reasoning and narrative coherence. ROCStories (Mostafazadeh et al., 2016) introduced a 100K-story corpus with the Story Cloze Test, evaluating models’ ability to select the correct ending from alternatives. While foundational, this approach evaluates understanding rather than generation, and does not address screenplay-specific requirements like format compliance or character voice.

Recent work has explored drama and screenplay generation systems. Dramatron (Mirowski et al.,

- 2023) uses hierarchical LLM prompting to generate complete scripts with scene breakdowns, character descriptions, and dialogue, validated through a 15expert user study. IBSEN (Zhou et al., 2024) proposes a director-actor agent framework where a director agent plans plot points and actor agents generate character-specific dialogue. However, these are generation systems rather than benchmarksthey lack standardized test sets, quantitative metrics, or large-scale model comparisons.

- 2.2 Multi-Dimensional NLG Evaluation

UniEval (Zhong et al., 2022) frames NLG evaluation as boolean question answering, training a unified model to assess coherence, consistency, fluency, and relevance across multiple tasks. While general-purpose, it does not capture screenplayspecific dimensions like dramatic structure or character voice distinctiveness. LitBench (Fein et al., 2025) evaluates creative writing quality but focuses on prose rather than screenplay format.

Recent surveys on LLM-as-a-Judge (Gu et al., 2024; Li et al., 2024) highlight reliability concerns when LLMs directly score outputs, noting issues with position bias, verbosity bias, and inconsistent criteria application. We address these concerns by using LLMs for structured labeling rather than direct scoring.

- 2.3 Evaluation Benchmarks

MT-Bench (Zheng et al., 2023) evaluates conversational AI through multi-turn dialogues judged by GPT-4, demonstrating strong correlation with human preferences. However, its focus on instructionfollowing and factual correctness differs from creative writing evaluation. HelloBench (Que et al.,

- 2024) assesses long-context generation capabilities but lacks domain-specific metrics for dramatic structure.

Positioning: DramaBench is the first benchmark to combine (1) large-scale quantitative evaluation (8,824 samples), (2) drama-specific multidimensional metrics, (3) statistical significance validation, and (4) human-LLM agreement analysis for screenplay continuation.

### 3 Dataset

#### 3.1 Data Collection and Processing

DramaBench is constructed from a curated subset of 1,103 English drama scripts sourced from the english_short_drama_scripts_dataset,

Statistic Value Total scripts 1,103 Total evaluations 8,824 Models evaluated 8 Avg script length 113 lines Avg context length 51 lines (381 tokens) Avg continuation length 62 lines (401 tokens) Split ratio (context:cont.) 45.7:54.3 Scene boundary splits 767 (69.5%) Midpoint splits 336 (30.5%) Construction cost $0 (rule-based) Reproducibility 100% (deterministic)

Table 1: DramaBench dataset statistics. The benchmark uses deterministic scene-boundary splitting for reproducible evaluation.

which contains professionally-written short-form drama scripts following Fountain screenplay format. We selected high-quality scripts through a filtering pipeline that removed duplicates, truncated scripts, and format-corrupted samples.

#### 3.2 Scene-Boundary-Aware Splitting

To create continuation tasks, we split each script into context and continuation segments using a scene-boundary-aware algorithm:

- 1. Default split: Identify the 50% midpoint (by line count)
- 2. Boundary optimization: Search within a ±20% window for scene heading markers (INT. or EXT.)
- 3. Final split: Use scene boundary if found (69.5% of cases), otherwise use midpoint

This approach ensures that continuations begin at natural narrative breakpoints, improving the realism of the generation task.

#### 3.3 Dataset Statistics

Table 1 summarizes DramaBench’s key statistics. With an average context of 381 tokens and continuation of 401 tokens, the task requires models to maintain consistency over moderate-length contexts while generating substantial creative content.

#### 3.4 Quality Control

All scripts in DramaBench meet the following criteria: (1) valid Fountain format structure, (2) minimum 27 lines and maximum 269 lines (ensuring sufficient context and continuation), (3) presence of

dialogue and character interactions (excluding pure narrative descriptions), and (4) English language content. This filtering ensures that the benchmark evaluates genuine screenplay continuation rather than format correction or translation tasks.

### 4 Evaluation Framework

- 4.1 Overview: LLM Labeling + Statistical Analysis

Our evaluation methodology addresses two key limitations of existing approaches: (1) LLM-asJudge systems that ask models to directly score quality suffer from bias and low reproducibility (Gu et al., 2024); (2) Human evaluation is expensive and slow.

We introduce a LLM Labeling + Statistical Analysis framework where LLMs act as structured data annotators rather than judges. For each dimension, we define:

- • Annotation unit: The granularity of analysis (line-level, event-level, dialogue-level, scenelevel, fact-level, or global-level)
- • Label set: Categorical labels assigned by the LLM (e.g., “Driver/Static/Redundant” for narrative beats)
- • Structured prompt: Clear instructions specifying label definitions and classification criteria
- • Statistical metric: Objective formulas converting label counts to quality scores

This approach ensures reproducibility (same inputs yield same labels), interpretability (scores trace back to specific labeled instances), and data reusability (labels serve as training data for DPO or reward modeling).

4.2 Six Evaluation Dimensions 4.2.1 Format Standards (Rule-Based) Objective: Evaluate screenplay format compliance and writing style.

Method: Pure rule-based analysis using Fountain format specification (Traugott and Nolan,

2012) (no LLM required).

#### Metrics:

• Format Error Rate: Percentage of lines violating Fountain rules (target: < 1%)

- • Novelization Index: Ratio of prose-like narrative to screenplay-appropriate action (target: < 0.35)
- • Dialogue-Action Ratio: Balance between dialogue and action lines (target: 1.0–2.0 for short drama)

#### 4.2.2 Narrative Efficiency

Objective: Evaluate plot progression density and identify “padding” behavior.

Annotation unit: Event-level (story beats (McKee, 1997; Snyder, 2005) extracted from action descriptions)

LLM task: Extract independent story beats and classify each as:

- • Driver: Advances plot (e.g., discovering evidence, confronting antagonist)
- • Static: Descriptive actions (e.g., looking out window, lighting cigarette)
- • Redundant: Repeats known information Metrics:
- • Effective Narrative Rate (ENR): Ndriver

Ndriver+Nstatic+Nredundant

- • Beats Per Page: TokenNCountdriver /250

#### 4.2.3 Character Consistency

Objective: Detect Out-Of-Character (OOC) behavior.

Annotation unit: Dialogue-level Preprocessing: LLM generates persona profiles

from context (Li et al., 2016), capturing speech patterns and personality traits.

LLM task: Classify each dialogue line as:

- • In_Character: Fits established persona
- • Neutral: Generic response, cannot judge
- • OOC: Violates persona (e.g., polite character suddenly rude without buildup)

#### Metrics:

- • OOC Rate: Total DialogueNOOC Lines (lower is better)

- • Voice Distinctiveness: TotalNin-characterDialogue

- 4.2.4 Emotional Depth Objective: Evaluate emotional arc dynamics within scenes.

Annotation unit: Scene-level LLM task: For each protagonist:

- 1. Identify opening emotion using the ValenceArousal model (Russell, 1980) (Valence: Positive/Negative, Arousal: High/Low)
- 2. Identify closing emotion
- 3. Classify change: Shift or Static
- 4. Detect Complex_Emotion (simultaneous opposing emotions, e.g., bitter smile)

Metrics:

- • Arc Score: 1 if Shift, 0 if Static
- • Complexity Ratio: Proportion of scenes with complex emotions

- 4.2.5 Logic Consistency Objective: Detect factual contradictions using atomic fact verification (Thorne et al., 2018; Bowman et al., 2015; Williams et al., 2018).

Annotation unit: Fact-level LLM task:

- 1. Extract: Identify hard constraints from context (e.g., “A’s leg is broken”, “only one gun available”)
- 2. Verify: For each fact, classify continuation’s handling as:

- • Violated: Contradicts the fact
- • Maintained: Respects the fact
- • Irrelevant: Doesn’t involve the fact

Metrics:

- • Logic Break Rate: N Nviolated

violated+Nmaintained

- • Context Coherence: Nmaintained (memory of context)

- 4.2.6 Conflict Handling Objective: Evaluate whether plot effectively advances conflict.

Annotation unit: Global-level (entire continuation)

LLM task: Identify core conflict from context, then classify handling:

- • Escalation (+2 pts): Intensifies conflict following principles of dramatic structure (Freytag, 1863; Field, 2005) (preferred for serial drama)
- • Twist (+2 pts): Introduces new obstacle/complication
- • Pause (+1 pt): Delays resolution
- • Resolution (0 pts): Fully resolves (discouraged mid-series)
- • Dropped (−5 pts): Ignores established conflict

#### Metrics:

- • Conflict Score: Weighted average based on classification
- • Drop Rate: Frequency of “Dropped” classifications

### 5 Experimental Setup

#### 5.1 Models Evaluated

We evaluate 8 state-of-the-art language models representing diverse architectures and training paradigms:

- 1. Claude Opus 4.5 (Anthropic, 2025)
- 2. DeepSeek v3.2 (DeepSeek-AI, 2025)
- 3. GLM-4.6 (Team GLM, 2024)
- 4. Gemini 3 Pro Preview (Google DeepMind, 2025)
- 5. Kimi K2 Thinking (Moonshot AI, 2025)
- 6. MiniMax M2 (MiniMax AI, 2025)
- 7. GPT-5.2 (OpenAI, 2025)
- 8. Qwen3-Max (Qwen Team, 2025)

All models were prompted with the same context and instruction to continue the script while maintaining consistency with established characters, plot, and style.

#### 5.2 Evaluation Process

LLM Evaluator: We use Qwen3-Max as the primary evaluator for all LLM-labeled dimensions. Format Standards uses deterministic rule-based analysis.

Cost: Format analysis costs $0 (rule-based). LLM-labeled dimensions incur API costs based on evaluator token usage.

Reproducibility: All evaluation prompts, extracted labels, and scripts are deterministic and will be released.

6 Results

#### 6.1 Overall Performance

Table 2 presents model rankings across all six dimensions. No single model dominates—each shows distinct specialization patterns.

#### Key findings:

- • Format compliance is universal: All 8 models achieve 0% format error rate, demonstrating that Fountain format can be reliably learned from prompts.
- • GPT-5.2 excels in robustness: Ranks 1st in 3/6 dimensions (Narrative, Character, Logic), making it the most well-rounded model.
- • Qwen3-Max specializes in emotion: Achieves the highest emotional arc rate (92.8%) but ranks 6th in narrative efficiency.
- • Gemini-3-Pro masters conflict: Best conflict handling score (1.867) but weakest emotional depth (8th).
- • Logic consistency shows largest variance: Error rates range from 2.0% (GPT-5.2) to 5.3% (GLM-4.6), indicating this dimension most differentiates models.

#### 6.2 Statistical Significance Testing

We conducted 252 Mann-Whitney U tests (Mann and Whitney, 1947) (28 pairwise model comparisons × 9 metrics) with Benjamini-Hochberg FDR correction (Benjamini and Hochberg, 1995) (q = 0.05). Results in Table 3 show that 65.9% of comparisons are statistically significant, confirming that observed differences are not due to random variation.

Most differentiating metrics: Beats per Page (26/28 significant, 20 large effect sizes) reveals the

largest performance gaps, with GPT-5.2 generating

- 3.5× more driver beats than GLM-4.6. Least differentiating metric: Conflict Score

shows only 8/28 significant comparisons, indicating models have uniformly mastered conflict escalation/management.

6.3 Dimension-Specific Analysis

- 6.3.1 Format Standards

All models achieve near-perfect format compliance (mean error rate: 0.0%), with only 228 total format warnings across 8,824 evaluations (2.6% sample error rate). This demonstrates that Fountain screenplay format can be reliably learned through prompt engineering alone, without specialized training.

Writing style varies: GPT-5.2 produces the most concise, screenplay-appropriate writing (Novelization Index: 0.11), while DeepSeek v3.2 generates more descriptive prose (0.24).

- 6.3.2 Narrative Efficiency

Mean ENR across all models is 93.3%, indicating high plot progression density. However, the distribution is left-skewed (skewness: −1.81), with most scripts near-perfect but a long tail of low-efficiency outliers.

Beat density: Average of 3.09 beats per page, with GPT-5.2 achieving 3.52 (highest) and GLM4.6 at 2.43 (lowest). Statistical testing reveals this is the most differentiating dimension.

- 6.3.3 Character Consistency

Mean OOC rate is remarkably low (0.97%), with 87.1% of scripts achieving perfect character consistency. The distribution is extremely right-skewed (skewness: +11.06), suggesting most models maintain character voice well, but occasional catastrophic failures occur (12.9% outlier rate).

Voice distinctiveness: Models average 84.1% in-character dialogue, indicating strong persona adherence.

- 6.3.4 Emotional Depth

90.3% of continuations include emotional shifts (vs static), demonstrating that models understand the importance of emotional arcs in drama. Complex emotions appear in 97.5% of scenes, with an average of 2.57 complex emotional moments per continuation.

Valence vs Arousal: Arousal shifts are more common (62.8%) than valence shifts (29.0%), sug-

Model Format Narrative Character Emotional Logic Conflict Avg Rank Error↓ ENR↑ OOC↓ Arc↑ Break↓ Score↑

GPT-5.2 0.000 (4th) 0.981 (1st) 0.006 (1st) 0.899 (4th) 0.020 (1st) 1.843 (2nd) 2.2 Qwen3-Max 0.000 (5th) 0.924 (6th) 0.008 (2nd) 0.928 (1st) 0.039 (5th) 1.781 (7th) 4.3 Gemini-3-Pro 0.000 (1st) 0.938 (2nd) 0.013 (7th) 0.875 (8th) 0.036 (4th) 1.867 (1st) 3.8 Claude Opus 4.5 0.000 (2nd) 0.929 (4th) 0.012 (5th) 0.883 (7th) 0.025 (3rd) 1.818 (5th) 4.3 DeepSeek v3.2 0.000 (3rd) 0.921 (7th) 0.009 (3rd) 0.894 (5th) 0.023 (2nd) 1.802 (6th) 4.3 MiniMax M2 0.000 (8th) 0.926 (5th) 0.009 (6th) 0.886 (6th) 0.041 (6th) 1.825 (4th) 5.8 Kimi K2 Thinking 0.000 (6th) 0.932 (3rd) 0.010 (8th) 0.912 (3rd) 0.043 (7th) 1.755 (8th) 5.8 GLM-4.6 0.000 (7th) 0.915 (8th) 0.011 (4th) 0.913 (2nd) 0.053 (8th) 1.836 (3rd) 5.3

- Table 2: Model performance across six dimensions. Bold indicates best performance. All models achieve 0% format error rate. Rankings vary significantly by dimension, confirming the need for multi-dimensional evaluation.

Metric Sig. Large Medium Small Tests Effect Effect Effect

Beats/Page 26/28 20 4 2 Context Coherence 26/28 4 12 10 Complex Emotions 23/28 1 8 14 ENR 22/28 2 6 14 Logic Break Rate 21/28 1 4 16 OOC Rate 18/28 1 3 14 Arc Score 15/28 0 2 13 Voice Distinct. 14/28 2 1 11 Conflict Score 8/28 0 0 8

Total 166/252 31 27 108

- Table 3: Statistical significance summary. Beats per Page is the most differentiating metric (26/28 significant, 20 large effects), while Conflict Score shows the least variance (models uniformly master conflict handling).

Aspect Success Failure

(Claude Opus 4.5) (MiniMax M2)

Logic Break 0.0% 100.0% Facts 7/8 maintained 3/8 violated

Context Both scripts establish physical/situational constraints that must be respected in continuation.

Success Excerpt

“She tries to rise but collapses, too weak from the birth.” [Maintains: postbirth weakness]

Failure Excerpt

“Ranran’s eyes SNAP open in her bedroom—normal, undamaged.” [Violation: Was in surgery, not bedroom]

Error Type N/A Spatial contradic-

tion, state reset

gesting models find it easier to modulate emotional intensity than to flip positive/negative sentiment.

Table 4: Qualitative case study: Logic Consistency. Success case maintains all context constraints; failure case contradicts established facts through spatial/temporal violations. See Appendix A for all dimensions.

#### 6.3.5 Logic Consistency

Mean logic break rate is 3.6%, with the highest variance across models (17.6% outlier rate). GPT-5.2 and DeepSeek v3.2 maintain the strongest logical coherence (2.0–2.3%), while GLM-4.6 shows the highest error rate (5.3%).

Context memory: Models maintain an average of 6.88 facts per script, indicating moderate context retention.

#### 6.3.6 Conflict Handling

Models overwhelmingly prefer Escalation (79.1% of scripts), with very low drop rates (1.5%). This aligns with serial drama conventions where conflict should intensify rather than resolve mid-episode.

Secondary conflicts: 75.6% of continuations introduce or develop secondary conflicts, showing sophisticated multi-threaded plot management.

### 7 Analysis

#### 7.1 Case Studies

We present illustrative success and failure cases across dimensions. Table 4 shows a representative comparison for Logic Consistency, demonstrating how models differ in maintaining context constraints. Extended case studies for all six dimensions are provided in Appendix A.

Narrative Success (GPT-5.2, script_1404): Achieves ENR = 1.0 with 18 driver beats and zero static/redundant beats. Every action advances plot: protagonist discovers hidden camera, realizes betrayal, devises counter-plan, and confronts antagonist—all within a 62-line continuation.

Narrative Failure (DeepSeek v3.2, script_6207): ENR = 0.33 with 67% static beats. Continuation consists primarily of character facial expressions, contemplative pauses, and

[Figure 2]

- Figure 2: Model performance comparison across six dimensions. All 8 SOTA models displayed on a single radar chart, revealing distinct capability profiles: GPT-5.2 shows balanced excellence across all dimensions, Qwen3-Max specializes in Emotional Depth, while Gemini 3 Pro excels in Conflict Handling. No single model dominates all dimensions.

repetitive descriptions of emotional states without plot advancement.

Logic Failure (MiniMax M2, script_4755): 100% violation rate (all facts contradicted). Context establishes protagonist in surgery; continuation has protagonist waking in bedroom, directly contradicting established location.

Character Success (Claude Opus 4.5, script_0005): 0% OOC rate with all 31 dialogue lines perfectly matching established personas. A grief-stricken protagonist maintains vengeful tone throughout, creating consistent character voice.

- 7.2 Error Taxonomy

We classified 10,850 errors across all evaluations.

- Figure 3 shows the top 10 error types with abbreviated labels for readability (OOC = Out-ofCharacter, D-A Imbal. = Dialogue-Action Imbalance, Low ENR = Low Effective Narrative Rate).

3. Redundant Beats (1,211 occurrences, 0.99% of all beats)

#### Model-specific patterns:

- • GPT-5.2: Minimal errors across all categories (best overall profile)
- • Qwen3-Max: Dialogue imbalance (243 occurrences) but excellent emotional depth
- • GLM-4.6: Excessive prose (102 occurrences) and highest logic violations

#### 7.3 Cross-Dimensional Insights

Format independence: Format standards show near-zero correlation with content dimensions (mean |r| = 0.040), validating that structural compliance is orthogonal to semantic quality.

#### Most common errors:

Robustness cluster: Narrative Efficiency, Character Consistency, and Logic Consistency show weak positive correlations (r ≈ 0.05), suggesting models with strong foundational capabilities tend to excel across these dimensions.

- 1. Dialogue-Action Imbalance (1,354 occurrences, 15.3%)
- 2. Low Information Gain (1,305 occurrences, 14.8%)

[Figure 3]

Figure 3: Top 10 error types from the error taxonomy (10,850 total errors). Dialogue-Action Imbalance and Low Information Gain are the most common.

Creative specialization: Emotional Depth and Conflict Handling are independent of other dimensions (r < 0.02), indicating these are distinct capabilities that can be optimized separately.

### 8 Ablation Studies

#### 8.1 Dimension Independence Validation

We computed Spearman correlations between dimension pairs across 8,824 evaluations. Format Standards is excluded from correlation analysis as all models achieved 100% compliance (zero variance). Results for the remaining five dimensions confirm extreme independence:

- • Mean absolute correlation: |r| = 0.014 (near-zero)
- • Maximum correlation: |r| = 0.035 (Narrative ↔ Emotional)
- • Consistency across models: Standard deviation = 0.0053

This validates that all five content dimensions capture distinct, non-redundant aspects of script quality. No dimension can be eliminated without loss of information.

#### 8.2 Human-LLM Agreement Analysis

We validated our LLM evaluator (Qwen3-Max) against human expert annotations on 188 scripts. Table 5 shows agreement metrics.

Strong agreement (3/5 dimensions): Logic, Emotional Depth, and Conflict show moderate-tosubstantial agreement, validating their reliability.

[Figure 4]

Figure 4: Spearman correlation matrix (5×5) between content dimensions. Near-zero correlations (mean |r| = 0.014) confirm that each dimension captures independent quality aspects. Format Standards excluded due to 100% compliance across all models.

Dimension Metric Agreement

Logic Consistency Pearson r 0.48*** Emotional Depth Cohen’s κ (Cohen, 1960) 0.53 Conflict Handling Cohen’s κ 0.42

Narrative Efficiency Pearson r 0.07 (n.s.) Character Consistency Pearson r −0.04 (n.s.)

Table 5: Human-LLM agreement on 188 scripts. Three dimensions show moderate-to-substantial agreement, while two reveal evaluator bias. *** p < 0.001.

Weak agreement (2/5 dimensions): Narrative Efficiency and Character Consistency show no significant correlation, indicating evaluator-specific biases in beat classification and OOC judgment. Model rankings on these dimensions should be interpreted with caution.

### 9 Discussion

#### 9.1 Implications

Multi-dimensional evaluation is essential: Our results demonstrate that aggregate quality scores mask important nuances. A model may excel at conflict management while failing at character consistency—insights lost in single-score systems.

Dimension-specific optimization opportunities: Error taxonomy and case studies reveal actionable improvement targets. For example, GLM4.6’s high logic violation rate suggests targeted training on context memory, while Qwen3-Max’s dialogue imbalance could be addressed through

prompt engineering.

Data closed-loop: All extracted labels (OOC dialogues, redundant beats, logic violations) can serve as negative samples for Direct Preference Optimization (DPO), enabling continuous model improvement.

#### 9.2 Limitations

Evaluator bias: Human-LLM agreement analysis reveals that Qwen3-Max exhibits systematic biases on narrative efficiency and character consistency. Future work should employ multi-evaluator ensemble voting to mitigate single-model bias.

Single language: DramaBench currently covers only English scripts. Extension to multilingual screenplay evaluation requires language-specific format parsers and evaluators.

Genre diversity: Our dataset focuses on shortform drama. Evaluation of full-length feature scripts, comedy, or experimental formats may require dimension modifications.

#### 9.3 Future Work

Multi-evaluator ensembles: Combining judgments from multiple LLM evaluators (GPT-4, Claude, Gemini) through consensus voting could improve reliability on dimensions with weak human-LLM agreement.

Prompt engineering refinement: Incorporating explicit examples from human annotations in evaluation prompts may reduce evaluator bias.

Extended human validation: A larger-scale human evaluation study (500+ scripts, 5+ annotators) would enable more robust inter-annotator agreement analysis and evaluator calibration.

Domain extension: Adapting the framework to other creative writing domains (novels, poetry, interactive fiction) could demonstrate generalizability.

### 10 Conclusion

We present DramaBench, the first large-scale benchmark for evaluating drama script continuation through six independent dimensions. Our LLM Labeling + Statistical Analysis framework ensures objective, reproducible evaluation while providing actionable feedback for model improvement. Comprehensive evaluation of 8 state-of-theart models on 1,103 scripts reveals that no single model excels universally—GPT-5.2 leads in robustness, Qwen3-Max in emotional depth, and Gemini3-Pro in conflict handling. Statistical validation

(65.9% of 252 comparisons significant), ablation studies (mean correlation |r| = 0.020), and human validation (substantial agreement on 3/5 dimensions) confirm the framework’s rigor. By establishing a rigorous standard for creative writing evaluation, DramaBench enables targeted model improvements and advances research in controllable, multi-faceted text generation.

### Ethics Statement

All scripts in DramaBench are from publicly available datasets. No personal information, copyrighted material without permission, or sensitive content was included. Model evaluations were conducted using official APIs with proper usage agreements. We acknowledge potential biases in LLM evaluators and recommend ensemble approaches for production use.

### Limitations

In addition to limitations discussed in Section 8.2, we note: (1) DramaBench evaluates continuation quality but not full-script generation coherence over long spans; (2) Our format analysis assumes Fountain format—other screenplay formats (Final Draft, Celtx) would require parser modifications; (3) Evaluation cost scales linearly with script count and model count (8,824 evaluations required significant API usage); (4) Human validation covered only 17% of the dataset—larger validation sets would strengthen claims.

### Acknowledgements

The authors acknowledge support from the Science and Technology Development Fund of Macau (Grant No. 0016/2025/ITP1).

### References

Anthropic. 2025. Claude opus 4.5 system card. Technical report, Anthropic.

Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1):289–300.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages

632–642, Lisbon, Portugal. Association for Computational Linguistics.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556.

Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, and Nick Haber. 2025. LitBench: A benchmark and dataset for reliable evaluation of creative writing. arXiv preprint arXiv:2507.00769.

Syd Field. 2005. Screenplay: The Foundations of Screenwriting, revised and updated edition. Delta, New York.

Gustav Freytag. 1863. Die Technik des Dramas. S. Hirzel, Leipzig. English translation: Freytag’s Technique of the Drama: An Exposition of Dramatic Composition and Art (1894).

Google DeepMind. 2025. Gemini 3 pro model card. Technical report, Google DeepMind.

Jiawei Gu, Xuhui Wang, Yiming Chen, Lei Zhang, and Yang Liu. 2024. A survey on LLM-as-a-Judge. arXiv preprint arXiv:2411.15594.

Haitao Li, Qingyao Zhang, and Jia Liu. 2024. LLMs-asJudges: A comprehensive survey on LLM-based evaluation methods. arXiv preprint arXiv:2412.05579.

Jiwei Li, Michel Galley, Chris Brockett, Georgios P. Spithourakis, Jianfeng Gao, and William B. Dolan. 2016. A persona-based neural conversation model. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 994–1003, Berlin, Germany. Association for Computational Linguistics.

Henry B Mann and Donald R Whitney. 1947. On a test of whether one of two random variables is stochastically larger than the other. The Annals of Mathematical Statistics, 18(1):50–60.

Robert McKee. 1997. Story: Substance, Structure, Style, and the Principles of Screenwriting. HarperCollins, New York.

MiniMax AI. 2025. MiniMax-M2: Model for max coding and agentic workflows. https://github. com/MiniMax-AI/MiniMax-M2.

Piotr Mirowski, Kory W. Mathewson, Jaylen Pittman, and Richard Evans. 2023. Co-writing screenplays and theatre scripts with language models: An evaluation by industry professionals. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, pages 1–34. ACM.

Moonshot AI. 2025. Kimi K2: Open agentic intelligence. https://moonshotai.github. io/Kimi-K2/. GitHub: https://github.com/ MoonshotAI/Kimi-K2.

Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. 2016. A corpus and cloze evaluation framework for deeper understanding of commonsense stories. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 839–849. Association for Computational Linguistics.

OpenAI. 2025. GPT-5.2 system card. Technical report, OpenAI.

Haoran Que, Feiyu Duan, Liqun He, Yutao Mou, Wangchunshu Zhou, Jiaheng Liu, Wenge Rong, Zekun Moore Wang, Jian Yang, Ge Zhang, Junran Peng, Zhaoxiang Zhang, Songyang Zhang, and Kai Chen. 2024. HelloBench: Evaluating long text generation capabilities of large language models. arXiv preprint arXiv:2409.16191.

Qwen Team. 2025. Qwen3-max: Just scale it. https: //www.alibabacloud.com/blog/602621.

James A. Russell. 1980. A circumplex model of affect. Journal of Personality and Social Psychology, 39(6):1161–1178.

Blake Snyder. 2005. Save the Cat! The Last Book on Screenwriting You’ll Ever Need. Michael Wiese Productions.

Team GLM. 2024. ChatGLM: A family of large language models from GLM-130B to GLM-4 all tools. arXiv preprint arXiv:2406.12793.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics.

John August Traugott and Stuart Nolan. 2012. Fountain: A markup language for screenwriting. https:// fountain.io. Accessed: 2025-12-19.

Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, and 1 others. 2023. Judging LLM-as-a-Judge with MT-Bench and chatbot arena. arXiv preprint arXiv:2306.05685. NeurIPS 2023 Datasets and Benchmarks Track.

Ming Zhong, Yang Liu, Da Yin, Yuning Meng, and Jiawei Han. 2022. Towards a unified multidimensional evaluator for text generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2023– 2038. Association for Computational Linguistics.

Senyu Zhou, Yao Han, Zhengyu Chen, Zhe Li, Dongming Yu, and Hao Chen. 2024. IBSEN: Directoractor agent collaboration for controllable and interactive drama script generation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1585–1602. Association for Computational Linguistics.

### A Extended Case Studies

This appendix provides detailed qualitative case studies for each of our six evaluation dimensions, demonstrating representative success and failure patterns.

#### A.1 Narrative Efficiency

Metric Success Failure

Script ID script_1404 script_6207 Model Qwen3-Max DeepSeek V3.2 ENR 100.0% 33.3% Driver Beats 18/18 4/12 Static Beats 0 8

Table 6: Narrative Efficiency case comparison.

Success Pattern (High narrative momentum): Every beat advances the plot directly.

OLD GU: “Cancel the betrothal banquet.” CHAIRMAN TANG: (relieved) “Of course, Old Gu—” OLD GU: (interrupting) “Cancel it so we can plan a wedding instead.” TANG NUANNUAN: “What?!” OLD GU: “I don’t marry ornaments. I marry forces of nature.”

Failure Pattern (Excessive static beats): 8 of 12 beats are static description with no plot advancement.

MU CHANGXUAN wipes her tears, a soft smile lingering. [STATIC] MU CHANGXUAN: “I have the best family in all the land.” [STATIC] She steps forward, embracing each brother in turn. [STATIC] A SERVANT appears: “The carriage is ready.” [STATIC]

#### A.2 Character Consistency

Context: Zhou Sheng—a man betrayed by his exgirlfriend, killed with his mother by a hit-and-run

Metric Success Failure

Script ID script_0005 script_3350 Model Claude Opus 4.5 GPT-5.2 OOC Rate 0.0% 100.0% Voice Distinctiveness 48.4% 0.0%

Table 7: Character Consistency case comparison.

driver. Established traits: grief-stricken, vengeful, determined.

Success: Voice matches established persona throughout.

ZHOU SHENG (V.O.): “Mom... I couldn’t protect you.” ZHOU SHENG: “I need the name of the driver who hit us.” ZHOU SHENG: “Tell them... I decline.” (refusing settlement) ZHOU SHENG (V.O.): “I have more money than God. And I have nothing left to lose.”

Maintains angry, vengeful, grief-driven character consistently.

#### A.3 Emotional Depth

Metric Success Failure

Script ID script_5570 script_5719 Model GPT-5.2 Gemini 3 Pro Complex Emotions 7 instances 0 instances Arc Type SHIFT STATIC

Table 8: Emotional Depth case comparison.

Success: Multiple complex (dual/opposing) emotions in single moments.

SHEN ZHIYI: “Anger is the only thing that kept me alive.” CAI YUE: “Then learn to hide it.” (She traces her new face with disbelief and revulsion mixed) “To become a lie that no one dares doubt... takes blood.”

7 instances of simultaneous opposing emotions detected.

Failure: Flat emotional trajectory (panic → panic → terrified). No emotional shifts, no complex moments throughout continuation.

#### A.4 Logic Consistency

See Table 4 in the main text for detailed analysis of Logic Consistency success/failure patterns.

#### A.5 Conflict Handling

Context: A modern bioengineer awakens in ancient China, targeted for castration. Core conflict: survival against unknown enemies.

###### Metric Success Failure

Script ID script_0014 script_2362 Model Claude Opus 4.5 Qwen3-Max Classification ESCALATION DROPPED Score +2.00 −5.00

Table 9: Conflict Handling case comparison.

Success (ESCALATION): Conflict intensifies progressively.

- • Initial threat: Assassination attempt (conflict established)
- • “Twelve days until marriage” (new complication added)
- • “Someone doesn’t want this marriage” (stakes raised)
- • “The man who woke up isn’t the same” (internal conflict)
- • Result: 3 secondary conflicts introduced

Failure (DROPPED): Conflict abandoned via time skip.

- • Initial conflict: Father sells daughter, mother commits suicide
- • THEN: “DISSOLVE TO: 20 YEARS LATER”
- • Father’s guilt? DROPPED
- • Daughter’s immediate fate? SKIPPED
- • Result: Core conflict left unresolved

#### A.6 Format Standards

Unlike the five content dimensions above, Format Standards presents no success/failure contrast: all 8 models achieved 0.0% format error rate on Fountain screenplay format. This uniformity is itself a significant finding—it demonstrates that screenplay format compliance is a solved problem for current SOTA models, learnable purely through prompt engineering without specialized training. This validates our framework’s design choice to separate structural compliance (Format) from semantic quality (other five dimensions).

