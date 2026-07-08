## Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding

Jaehyun Jeon1*, Min Soo Kim1, Jang Han Yoon1, Sumin Shim1, Yejin Choi1, Hanbin Kim1, Dae Hyun Kim1†, Youngjae Yu2† 1Yonsei University 2Seoul National University {jaehyun.jeon, dhkim16}@yonsei.ac.kr youngjaeyu@snu.ac.kr

# arXiv:2505.05026v5[cs.CL]4Jun2026

### Abstract

User interface (UI) design goes beyond visuals to shape user experience (UX), underscoring the shift toward UI/UX as a unified concept. While recent studies have explored UI evaluation using Multimodal Large Language Models (MLLMs), they largely focus on surface-level features, overlooking how design choices influence user behavior at scale. To fill this gap, we introduce WiserUI-Bench, a novel benchmark for multimodal understanding of how UI/UX design affects user behavior, built on 300 realworld UI image pairs from industry A/B tests, with empirically validated winners that induced more user actions. For future design progress in practice, post-hoc understanding of why such winners succeed with mass users is also required; we support this via expert-curated key interpretations for each instance. Experiments across multiple MLLMs on WiserUI-Bench for two main tasks, (1) predicting the more effective UI image between an A/B-tested pair, and (2) explaining it post-hoc in alignment with expert interpretations, show that models exhibit limited understanding of the behavioral impact of UI/UX design. We believe our work will foster research on leveraging MLLMs for visual design in user behavior contexts.

### 1 Introduction

User interface (UI) design is central to application development. While visual aesthetics lay its foundation, the fundamental goal is to steer user behavior during interaction, such as encouraging signups or purchases, and thereby boost service revenue (Fogg, 2002). This behavioral focus expands to UI/UX design, uniting visual form with user experience (UX) considerations (Norman, 1988).

Practitioners validate such decisions through large-scale A/B tests that randomly assign users to alternative UI variants and measure which design

*Currently at NC AI †Co-corresponding authors

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Figure 1: A real-world, behavior-aware design decision example from WiserUI-Bench, grounded in A/B test results, illustrating how UI changes steer user actions.

more effectively drives the desired action (Kohavi et al., 2009). For example, as shown in Figure 1, version B achieves higher sign-ups overall, interpreted as reduced user friction from fewer options. Such interpretation reflects an abductive reasoning process (Hobbs et al., 1993; Bhagavatula et al., 2020), inferring the most plausible explanations for aggregate outcomes that are used to guide subsequent design decisions (Quin et al., 2024). Given that predicting and explaining these results is nontrivial even for humans, hinging on complex cognitive processes underlying user experience, assessing multimodal understanding of UI/UX design becomes a crucial challenge, requiring not only identifying visual elements but also reasoning about how they influence user behavior.

Recent efforts have explored how to evaluate UI design quality with Multimodal Large Language Models (MLLMs); however, they all fall short in capturing the behavioral impact of UI/UX design, as summarized in Table 1. While some focus solely on basic visual, heuristic-based attributes (Yang and Li, 2024; Wu et al., 2024), others rely on handwritten critiques of individual UIs, lacking validation against real user data and offering limited coverage of cognitive aspects (Duan et al., 2024).

To address these limitations, we introduce

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

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

[Figure 44]

[Figure 45]

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

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

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

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Figure 2: Overview of WiserUI-Bench and two main tasks. Each instance contains a UI image pair with verified A/B test winner and expert-curated key interpretations explaining it. These span the three cognitive UX dimensions; the example includes all three, though most cover fewer. MLLMs are evaluated on (1) selecting the more effective UI/UX design by predicting the verified winner, and (2) explaining the effectiveness of a given winner, measured by whether the model captures each expert interpretation.

WiserUI-Bench (Pairwise UI/UX Design Understanding Benchmark), a novel benchmark for evaluating a model’s multimodal understanding of how UI/UX design affects user behavior. WiserUIBench comprises 300 real-world UI image pairs with winning variants verified by A/B tests from actual companies, spanning varied contexts and sourced from reliable platforms. Each pair includes expert-curated key interpretations, 684 in total, written post-hoc with knowledge of the winner to explain its effectiveness, covering core behavioral drivers. An example instance is illustrated in the upper part of Figure 2. Grounded in large-scale real user behavior data, our benchmark captures authentic usage and preference patterns, providing a reliable basis for evaluation. Each interpretation is tagged with one of the three cognitive UX dimensions articulated by Norman (2007): perception (what users notice), memory (what users remember), and action (how users act).

Our benchmark supports two core tasks, illustrated in the lower part of Figure 2: (1) selecting the more effective UI/UX design from a given pair by predicting the A/B test-verified winner, and (2) interpreting the UI/UX effectiveness of the given winner by assessing alignment between modelgenerated and expert-curated interpretations. This task setup captures the essential capabilities for UI/UX assessment, not only to identify which design better guides user behavior, but also to reliably explain why in a post-hoc manner.

Experiments on a broad range of proprietary and open-source MLLMs on WiserUI-Bench show that existing models struggle with visual reasoning in user behavior contexts. Models perform near-random on selection, and their post-hoc interpretations remain insufficient to reach expert level. We expect that WiserUI-Bench and our novel UI/UX-focused tasks will spur further research on this critical and underexplored, user-centered area, advancing behavior-aware reasoning about visual design in MLLMs.

In summary, our main contributions are:

- • We introduce a novel and challenging task of multimodal UI/UX understanding that centers on the behavioral impact of design.
- • We release WiserUI-Bench, a benchmark of 300 real-world UI pairs uniquely constructed from verified results of large-scale industry A/B tests, along with 684 carefully curated expert key interpretations.
- • Through comprehensive experiments and analyses across diverse MLLMs, we uncover their limited ability to understand the behavioral impact of UI/UX design, highlighting a critical gap in current visual reasoning.

### 2 Related Work

#### 2.1 Visual Reasoning

Visual reasoning benchmarks for Multimodal Large Language Models (MLLMs) evaluate models’ abil-

Large-Scale User Validation

Expert Interpretation

Benchmark # Samples Unit Source Quality Source Type

Evaluation Objective

Guideline Violation Detection BetterWeb (Wu et al., 2024)

Yang and Li (2024) 382 Single Legacy UI Mobile ✗ ✗

892 Pair Synthetic Mobile ✗ ✗ Basic Visual Quality UICrit (Duan et al., 2024)

983 Single Legacy UI Mobile ✗ ✓ UI Design Critique WiserUI-Bench (Ours)

✓ (Real A/B Test)

Actual Production

UI/UX Design Understanding (User Behavior-Focused)

✓

300 Pair

Mobile + Web

Table 1: Comparison between WiserUI-Bench and existing UI evaluation benchmarks.

ity to process visual information and reason at a cognitive level. While some target general tasks (Yue et al., 2024; Liu et al., 2024b; Fu et al., 2024; Li et al., 2024b), others focus on specific domains like mathematics or science (Hao et al., 2025; Lu et al., 2024; Xu et al., 2025). However, few benchmarks address reasoning about human behavior or preference patterns in visual contexts. Moreover, although some studies explore visual reasoning across multiple images (Cheng et al., 2025; Zhao et al., 2024), such cases remain rare. Our task incorporates both aspects, reasoning over paired UI images and aligning with behavior patterns, as captured by actual A/B test outcomes.

#### 2.2 UI Evaluation

Recent advances have driven progress in evaluating UI quality with MLLMs. Yang and Li (2024) propose a benchmark for detecting guideline violations in individual screens, while Wu et al. (2024) generates synthetic UI pairs via perturbations such

- as color noise or spacing changes. However, both are limited to heuristic analysis of surface-level appearance. Duan et al. (2024) evaluates single screens using expert-critique annotations, but it lacks validation with real-world user behavior and relies on RICO (Deka et al., 2017), an outdated, mobile-only dataset that limits its relevance to today’s production-level designs. Prior work also suggests that pairwise evaluation yields more reliable signals, particularly for subjective tasks like ours that target behavior patterns captured by realworld A/B tests (Abel et al., 2020; Bawabe et al., 2021). In this context, WiserUI-Bench offers a stronger foundation by grounding design effectiveness in large-scale user behavior data and requiring models to engage in deeper, nuanced reasoning over both visuals and UX in a pairwise format.

#### 2.3 Simulation of Human Behavior

Efforts to simulate human behavior with AI are ongoing. Park et al. (2023) places LLM-driven characters in a sandbox town, producing socially plausible interactions. Lu et al. (2025a) moves closer to real behavior by using actual platform logs to predict next user actions from prior context, yet remains text-only. In the UI domain, Lu et al. (2025b) and Wang et al. (2025a) deploy personabased agents to browse interfaces and examine how designs may steer user behavior; the former checks realism only via qualitative interviews, while the latter compares against an actual A/B test but only under a single scenario. By contrast, our work grounds evaluation in numerous real-user A/B test results and introduces diverse multimodal tasks that reason over UI variant image pairs to predict and explain user behavior.

3 WiserUI-Bench

We propose WiserUI-Bench, a new benchmark for evaluating multimodal understanding of UI/UX design, a key factor in crafting interfaces that influence user behavior.

#### 3.1 Benchmark Construction

Data Sources To construct a benchmark that faithfully reflects real-world UI/UX design decisions, we collect UI images from industry-validated A/B testing data. Specifically, we aggregate data from widely recognized A/B testing and analysis platforms1, which showcase large-scale A/B tests conducted by companies across various industries and regions. These sources provide both UI variants and verified outcomes based on actual user behavior data. By grounding our benchmark in such real, rigorously tested cases, we ensure a credible

1https://vwo.com/success-stories/, https: //goodui.org/leaks/, https://abtest.design/

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Figure 3: Distribution of (a) UI change element types and (b) expert interpretations for WiserUI-Bench.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

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

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Figure 4: Benchmark construction pipeline. (a) Raw A/B test data were collected from reliable platforms, (b) then UI images were refined for clarity. (c) Finally, key interpretations explaining the test outcomes were curated by UI/UX experts.

basis for evaluating UI/UX design understanding in practical scenarios.

Data Curation Pipeline As illustrated in Figure 4, our curation process involves multiple stages to ensure high-quality annotations. First, we extract raw UI images and accompanying textual descriptions from our selected platform sources. Next, we refine these images by eliminating any visual markers that were externally added, such as arrows or circles, using inpainting techniques, to preserve a clean, unbiased evaluation of visual reasoning.

Finally, we construct key expert interpretations explaining the winning variant’s superior performance. Such expert involvement is essential, as interpreting user behavior often requires deep domain expertise. Three experienced UI/UX experts aware of the A/B test outcomes independently annotated each instance, instructed to identify (i) the key UI modification and (ii) its corresponding behavioral implication. As a pre-defined quality control step, only interpretations where at least two

experts independently converged on substantially overlapping findings were retained, yielding 684 final interpretations (562 with full, 122 with two-way agreement) from 845 initial candidates. Each interpretation was further tagged with the most relevant law from a curated set of 12 UX laws. Since user behavior is often shaped by multiple factors, a single instance may include multiple interpretations. See Appendix A.1 for details.

#### 3.2 Benchmark Analysis

Our pipeline produces a benchmark of 300 realworld UI image pairs, each with a definitive winner identified through A/B test results and annotated with expert-curated key interpretations of the core design factors driving user behavior, totaling 684 entries. We include detailed statistics in Appx. A.2.

Ensuring contextual diversity is essential, as effective UI/UX strategies vary widely across use cases (Oinas-Kukkonen and Harjumaa, 2018). For example, page types correspond to different stages of the user journey, requiring distinct strategies. To capture this, WiserUI-Bench spans 11 page types, 13 industry domains, and two device types.

Moreover, we annotate UI differences using element–attribute associations, specifying the UI element being modified and the corresponding attribute change (e.g., a button with a position change), to capture real-world UI variants with concurrent modifications. On average, each instance contains 1.53 element-level and 1.99 attribute-level annotations, spanning 19 element types and 14 attribute types. We further group the element types by their primary functional roles (Figure 3(a)): Interactive Control, which directly elicits intentional user actions; Content & State Display, which primarily conveys information, semantics, or system state; and Container & Layout Structure, which

organizes spatial layout and guides attention at a structural level.

WiserUI-Bench also exhibits diversity in the interpretations used to explain user behavior, as shown in Figure 3(b), with an average of 2.28 interpretations per instance. The tagged UX laws are grouped into three cognitive dimensions of UX, perception, memory, and action, following Norman (2007), with four laws assigned to each.

### 4 Task Description

Proper evaluation of a model’s multimodal understanding of UI/UX design requires assessing not only its ability to predict which design more effectively guides user behavior, but also whether it can explain why in a post-hoc manner, grounding its reasoning in visual and behavioral cues. To this end, we define two main tasks, selection and interpretation, as illustrated in Figure 2, both essential for demonstrating genuine understanding. Success requires identifying key visual differences between two UI variants, as well as inferring the underlying behavioral goal from context (e.g., promoting sign-ups) and how the interface supports that goal.

- Task 1: UI/UX Design Selection The first task assesses whether the model can identify which of two UI images is more effective at influencing user behavior, with the correct answer determined by large-scale A/B testing run by real companies. Although the task is framed as a binary choice, solving it effectively requires nuanced visual reasoning beyond surface-level comparison.
- Task 2: UI/UX Design Interpretation The second task evaluates a model’s post-hoc interpretation ability on the observed results. Unlike the first task, it presents the winning UI/UX design upfront and asks the model to explain the factors contributing to its effectiveness in guiding user behavior. The model’s explanation is then compared to expert-curated key interpretations, with the degree of alignment serving as a measure of the quality of its expert-level interpretive understanding.

### 5 Experiments

Models We evaluate a diverse set of MLLMs, spanning both proprietary and open-source models. On the proprietary side, we include o1 (OpenAI et al., 2024b), GPT-5.1 (OpenAI, 2025), GPT4o (OpenAI et al., 2024a), Claude 4.5 Sonnet (Anthropic, 2025), Claude 3.5 Sonnet (Anthropic,

- 2024), all noted for their strong visual reasoning. For open-source models, we evaluate widely adopted models such as Qwen3-VL (30B-A3B, 8B) (Bai et al., 2025a), Qwen2.5-VL (7B, 32B) (Bai et al., 2025b), InternVL-2.5 (8B, 38B) (Chen et al.,
- 2025), LLaVA-NeXT-7B (Liu et al., 2024a) and LLaVA-OneVision-7B (Li et al., 2024a). Refer to Appendix B.1 for details. 5.1 UI/UX Design Selection

#### 5.1.1 Evaluation Metrics

We report model accuracies across input-order variations of each UI image pair, reflecting the multi-image input setting. Specifically, we report: (1) First Accuracy (FA), when the more effective UI/UX design is presented first in the input; (2) Second Accuracy (SA), when it is presented second; and (3) Average Accuracy (AA), the mean of FA and SA, reflecting overall correctness. Finally, we compute (4) Consistent Accuracy (CA), which checks whether the model selects the same, correct UI across both input orders of each pair. This metric isolates the model’s core visual reasoning ability by ensuring that its choices are based on content rather than position bias, a well-known issue in pairwise comparison settings (Wang et al., 2025b; Zhang et al., 2023). (Random guessing yields a CA of 25%: since there’s a 50% chance of being correct in each order.) All evaluations are conducted over three independent runs, and we report the averaged results. Experiment details, including the prompts, are provided in Appendix B.2.

#### 5.1.2 Results

Overall Performance Table 2 summarizes model performance on the UI/UX design selection task. Most AA scores hover only slightly above the 50% random baseline for a binary choice. Moreover, they are inflated by strong position bias, especially in proprietary models and the Qwen-VL family, which tend to pick the second image. When we examine CA, the order-invariant metric, performance drops back to near-random levels. Interestingly, InternVL-2.5-38B scores slightly higher than proprietary models on CA, but still fall short of being adequate. Contrary to expectations, o1 shows the strongest bias and a lower CA than GPT-4o, suggesting that self-reasoning of o1 does not transfer well to this task. Most smaller (7B/8B) models perform significantly worse than larger ones, confirming the scaling trend. Overall, existing MLLMs struggle to consistently predict the more effective

Overall IC CSD CLS Mixed # Cases (300) (100) (79) (49) (72)

FA SA AA CA AA CA AA CA AA CA AA CA

Random 50.00 50.00 50.00 25.00 50.00 25.00 50.00 25.00 50.00 25.00 50.00 25.00 o1 16.56 97.78 57.17 15.56 55.67 16.00 55.70 11.81 56.46 11.56 61.34 21.76 GPT-5.1 35.67 81.33 58.50 33.33 58.67 33.33 59.28 35.02 54.42 25.17 60.19 37.04 GPT-4o 31.89 88.33 60.11 30.11 59.67 31.00 61.60 29.96 51.70 17.69 64.81 37.50 Claude 4.5 Sonnet 34.33 79.33 56.83 32.33 53.50 29.00 60.76 37.97 55.10 34.69 58.33 29.17 Claude 3.5 Sonnet 26.11 86.67 56.39 24.22 56.67 22.67 57.59 29.96 51.70 12.24 57.87 28.24 Qwen3-VL-30B-A3B 16.67 88.00 52.33 15.00 53.33 16.67 51.69 12.66 54.08 17.69 50.46 13.43 Qwen3-VL-8B 19.67 87.67 53.67 19.33 51.67 16.00 57.38 23.63 49.66 16.33 55.09 21.30 Qwen2.5-VL-32B 33.67 81.44 57.56 31.00 58.67 31.67 56.54 28.69 58.84 28.57 56.25 34.26 Qwen2.5-VL-7B 23.00 77.89 50.44 12.89 48.17 8.67 56.12 19.41 43.88 9.52 51.85 13.89 InternVL-2.5-38B 55.67 60.00 57.83 34.56 56.17 34.33 59.70 36.29 55.10 28.57 59.95 37.04 InternVL-2.5-8B 58.33 45.67 52.00 21.33 52.17 23.00 48.73 15.61 57.14 25.85 51.85 22.22 LLaVA-NeXT-7B 54.44 40.56 47.50 10.78 48.50 12.67 43.67 8.44 46.60 8.16 50.93 12.50 LLaVA-OneVision-7B 19.78 81.56 50.67 10.44 50.00 11.00 54.43 13.50 47.28 3.40 49.77 11.11

- Table 2: Results of the UI/UX design selection task on WiserUI-Bench, shown overall and by UI change element types. Best and second-best model scores for each metric are marked in bold and underlined, respectively. For CA, the best and second-best UI element type performances within each model are highlighted in blue and yellow , respectively. All metrics are reported as percentages (%).

Model Method FA SA AA CA

GPT-4o

Zero-Shot 31.89 88.33 60.11 30.11 CoCoT 33.00 83.33 58.17 29.67 Self-Refine 32.44 82.67 57.56 29.89 DDCoT 36.56 86.89 61.72 34.78 MAD (R1) 51.22 67.44 59.33 39.00 MAD (R3) 34.44 80.89 57.67 29.89

Claude 3.5 Sonnet

Zero-Shot 26.11 86.67 56.39 24.22 CoCoT 29.78 81.78 55.78 27.22 Self-Refine 25.67 76.44 51.06 20.44 DDCoT 26.22 86.67 56.44 23.56 MAD (R1) 39.00 72.89 55.94 30.00 MAD (R3) 23.33 85.44 54.39 21.44

- Table 3: Results of the UI/UX design selection task across reasoning strategies on WiserUI-Bench, figures in percentages (%).

based models do so on CSD, which often involves content-level changes (e.g., images or text). By contrast, Container & Layout Structure (CLS) is the most challenging category, with sharply reduced CA across most models, reflecting limited sensitivity to structural layout changes, such as card or tile-level differences. Notably, Claude 4.5 Sonnet perform relatively better on CLS than other models.

Effect of Reasoning Strategies We additionally evaluate various reasoning strategies on GPT-4o and Claude 3.5 Sonnet to assess their effectiveness, including Contrastive Chain-of-Thought (CoCoT) (Zhang et al., 2024), Self-Refine (Madaan et al., 2023), Duty-Distinct Chain-of-Thought (DDCoT) (Zheng et al., 2023), and Multi-Agent Debate (MAD) (Liang et al., 2024; Du et al., 2023). For MAD, we stop and moderate the debate after the first (R1) and third (R3) rounds. As shown in Table 3, AA scores remain similar to or below the zero-shot setting, while CA scores often surpass it, especially for MAD (R1) across models, DDCoT on GPT-4o, and CoCoT on Claude 3.5 Sonnet. This gain is mainly due to reduced position bias, enabling more consistent decisions regardless of input order. MAD (R1) performs well by generating and moderating conflicting perspectives, which suits this human behavior pattern-driven task, benefiting from multi-perspective reasoning. However, excessive internal debate may hurt performance, as seen in the drop with MAD (R3).

UI/UX design in real-world UI pairs.

Performance Across UI Element Types The UI element-wise results in Table 2 show performance variations across change type categories, while preserving overall trends: GPT-4o and InternVL2.5-38B generally achieve the highest AA and CA, respectively. Focusing on CA, Mixed cases, where changes span multiple functional UI element groups, often yield relatively high performance, suggesting that aggregated multi-functional changes may provide richer behavioral cues than single-group modifications. Interactive Control (IC) and Content & State Display (CSD) show broadly comparable performance overall, although their relative ranking varies across models. Across model families, GPT-based models show relative strength on Mixed cases, whereas Claude-

[Figure 211]

Figure 5: Results of the UI/UX design interpretation task by each UX dimension on WiserUI-Bench.

Interpretation Instance-level Recall Recall

Model

o1 64.18 78.33 GPT-5.1 68.71 79.00 GPT-4o 50.15 66.67 Claude 4.5 Sonnet 67.40 80.33 Claude 3.5 Sonnet 63.16 78.67

Qwen3-VL-30B-A3B 64.62 74.00 Qwen3-VL-8B 47.22 56.00 Qwen2.5-VL-32B 60.23 73.00 Qwen2.5-VL-7B 43.71 61.67 InternVL-2.5-38B 45.32 59.33 InternVL-2.5-8B 35.09 51.67 LLaVA-NeXT-7B 11.99 21.67 LLaVA-OneVision-7B 21.78 33.33

- Table 4: Results of the UI/UX design interpretation task on WiserUI-Bench, figures in percentages (%).

5.2 UI/UX Design Interpretation 5.2.1 Evaluation Metrics

To judge how well a model’s generated interpretation explains the observed outcome, we compare it against expert-curated key interpretations for each instance. Since we let models produce free-form explanations, their outputs often differ substantially in wording and verbosity even when conveying similar meaning; accordingly, we prioritize semantic alignment over surface-level lexical overlap. Specifically, we instruct the model to enumerate all relevant factors in its explanation, and each expert interpretation is independently evaluated in a binary manner, whether it is fully captured by the explanation or not. We report two metrics: Interpretation Recall, measuring the proportion of expert interpretations captured, and Instance-level Recall, which counts an instance as correct if at least one expert interpretation is captured. Binary inclusion judgments are made using a GPT-4o-based evaluator (Liu et al., 2023), validated against human judgments on 1,000 random samples (83.0% accuracy, Cohen’s kappa 0.66). The prompts used, along with the results of BLEU, ROUGE, METEOR,

and BERTScore (Papineni et al., 2002; Lin, 2004; Banerjee and Lavie, 2005; Zhang et al., 2020) are reported in the Appendix B.3.

#### 5.2.2 Results

Overall Performance Table 4 reports performance on the UI/UX design interpretation task. Recent models such as GPT-5.1 and Claude 4.5 Sonnet are among the strongest performers, although the overall results remain far from saturated. Some models, such as o1 and Claude 3.5 Sonnet, also show relatively strong post-hoc interpretation given the more effective variant, despite being weaker on selection in Task 1. By contrast, models such as GPT-4o and InternVL-2.5-38B, which were relatively strong on Task 1 selection, do not maintain the same level of advantage under this setting. Taken together, these observations suggest that selecting the better design and explaining its advantage are related but dissociable capabilities. Within the smaller-model regime, Qwen-VL variants align more closely with expert interpretations than other 7B/8B models, and larger Qwen-VL models consistently outperform their smaller counterparts. The persistent gap between Interpretation Recall and Instance-level Recall suggests that models often recover only part of the expert interpretations, rather than covering it comprehensively.

Performance Across UX Aspects We further analyze Interpretation Recall scores by UX dimension to examine which types of UX interpretations are more readily recovered, as a true understanding of UI/UX requires capturing the full range of cognitive aspects—perception, memory, and action. As shown in Figure 5, o1 performs best on Perception, indicating relatively stronger ability to explain which visual cues are likely noticed by users, whereas GPT-5.1 achieves the highest score on Action, suggesting better performance on interaction-oriented explanations. More broadly,

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

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

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

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

[Figure 274]

- Figure 6: Example responses from InternVL-2.5-38B on the UI/UX design selection task, showing model outputs under different input image orders for each case. Blue marks correct UI change perception, while red denotes incorrect or hallucinated perception. Purple indicates user behavior pattern-aligned reasoning, whereas misaligned reasoning in orange.

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

- Figure 7: Example responses from o1, GPT-4o, and Qwen2.5-VL-32B on the UI/UX design interpretation task. Green and blue highlight points capturing the expert interpretations fully.

no single model dominates uniformly across all three dimensions. Performance remains below or around 70% across most dimensions, underscoring the need for more holistic modeling of UI/UX understanding across multiple cognitive aspects.

### 6 Discussion

Case Study For the UI/UX design selection task, we analyze the success and failure modes of the strong-performing model, InternVL-2.5-38B, as shown in Figure 6, focusing on cases where the model makes consistent decisions under both input image orders. In Case 1, the model correctly perceives the UI change as the presence of a CTA button and selects the more effective UI/UX, providing a reason that behaviorally aligns with the user behavior pattern implied by the A/B test outcome, hypothesizing that fewer CTAs better focus user attention. In Case 2, although the model correctly identifies differences such as pricing badges and

CTA text, it fails to select the winning variant identified by the A/B test, instead offering misaligned reasoning that higher visual saliency negatively impacts user behavior by inducing commitment pressure. In Case 3, the model fails at perception level, hallucinating non-existent changes instead of recognizing the actual button color difference, which ultimately leads to an incorrect final decision.

Figure 7 shows a case from the UI/UX design interpretation task. The UI on the right was verified by A/B test results as the more effective design, with experts attributing its effectiveness to a simplified background and an enlarged demo button, along with their associated effects, factors that models must explicitly capture for an accurate interpretation. o1 succeeds by directly highlighting these key points. GPT-4o correctly identifies the relevant UI elements, but provides an incorrect or vague explanation for their impact. Qwen2.5-VL32B correctly notes the minimalist design, but stops

- at identifying the button, missing the key detail of its enlargement and resulting in an incomplete explanation. As shown here, models often identify and emphasize salient surface-level differences, but fall short in capturing deeper behavioral implications and UX reasoning, limiting interpretive depth relative to experts.

### 7 Conclusion

We introduce a new task that evaluates multimodal understanding of how design drives user actions in the critical field of UI/UX. To support this, we present WiserUI-Bench, a behavior-focused benchmark uniquely built from diverse, real-world A/B-tested UI variants and expert-curated key interpretations, providing a reliable basis for assessment. Our two tasks, selection and interpretation, go beyond surface-level recognition, requiring models to reason about the behavioral implications of visual design, a cognitively challenging problem. Extensive experiments across both proprietary and opensource MLLMs reveal they exhibit only shallow levels of behavior-aware visual reasoning. Achieving stronger performance on this significant task, particularly across the analyzed UI elements or UX dimensions, may require further training on large-scale, behaviorally grounded UI/UX data that reflects real-world user dynamics. We envision WiserUI-Bench and our work as a step toward multimodal systems capable of not only understanding but also generating visual interfaces grounded in user behavior contexts, aligning with human cognition to optimize user experience. More broadly, we hope to catalyze future research toward deeper UI/UX and human behavior literacy in AI.

### 8 Limitations

Given that user experience patterns vary across cultural contexts and societal norms, cultural biases inevitably exist in our benchmark (Karreman et al., 2016). Also, interactive UIs were not extensively explored in our work. Although the number of 300 UI image pairs may seem limited, publicly available real-world A/B test results are extremely rare, as companies rarely disclose them. Moreover, independently conducting production-level UI experiments at a comparable level and scale was practically infeasible. However, as the first benchmark in this area, we focused on reliability, diversity, and annotation quality, covering a wide range of realworld UI changes and providing expert-curated key

interpretations.

### 9 Ethical Considerations

To ensure expert-level annotation quality, we recruited three annotators with professional UI/UX backgrounds who are currently employed at global IT companies specializing in UX and AI. We clearly informed them of the academic nature of the task, and they voluntarily participated in the study. We provided each annotator with monetary compensation equivalent to approximately $225 USD, reflecting fair market rates for domain-specific labor, and issued payments within 48 hours of annotation completion. We instructed annotators to focus solely on task-relevant elements and ensured that they were not exposed to or asked to label any sensitive, offensive, or triggering content.

Our benchmark does not include any private, user-identifiable, or confidential information. All materials were curated exclusively from publicly accessible promotional case studies (e.g., Success Stories from VWO2) released by the originating platforms, while strictly excluding any restricted or paywalled content. We acknowledge that while some source platforms include copyright notices intended to protect their proprietary assets, they do not provide explicit open redistribution licenses, such as Creative Commons, for their publicly accessible case materials. To responsibly address the absence of explicit licenses and to ensure compliant academic use, WiserUI-Bench is released strictly for non-commercial academic research under a non-commercial license (e.g., CC BY-NCSA 4.0). Rather than redistributing raw image assets directly, we primarily release source URLs alongside our expert annotations and automated image-processing scripts. We will further maintain transparent communication channels within the repository to accommodate potential inquiries from original content owners and, if necessary, adjust specific entries in accordance with responsible academic data governance.

### Acknowledgments

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea Government (MSIT) (No. RS-2024-00338140, Development of learning and utilization technology to reflect sustainability of generative language models

2https://vwo.com/success-stories

and up-to-dateness over time), and by the Culture, Sports and Tourism R&D Program through the Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism in 2026 (RS-2024-00361757, Development of Multimodal UX Evaluation Platform Technology for XR Spatial Responsive Content Optimization)

### References

Edward Abel, Ixent Galpin, Norman Paton, and John Keane. 2020. Pairwise comparisons or constrained optimization? a usability evaluation of techniques for eliciting decision priorities. International Transactions in Operational Research, 29.

- Anthropic. 2024. The claude 3 model family: Opus, sonnet, haiku. https://www-cdn.anthropic.com/ de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/ Model_Card_Claude_3.pdf.
- Anthropic. 2025. System Card: Claude Sonnet 4.5.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025a. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2.5-vl technical report. Preprint, arXiv:2502.13923.

Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, Ann Arbor, Michigan. Association for Computational Linguistics.

Sarah Bawabe, Laura Wilson, Tongyu Zhou, Ezra Marks, and Jeff Huang. 2021. The ux factor: Using comparative peer review to evaluate designs through user preferences. Proc. ACM Hum.-Comput. Interact., 5(CSCW2).

Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Scott Wen tau Yih, and Yejin Choi. 2020. Abductive commonsense reasoning. Preprint, arXiv:1908.05739.

G. Bradski. 2000. The OpenCV Library. Dr. Dobb’s Journal of Software Tools.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye,

Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, and 23 others. 2025. Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling. Preprint, arXiv:2412.05271.

Ziming Cheng, Binrui Xu, Lisheng Gong, Zuhe Song, Tianshuo Zhou, Shiqi Zhong, Siyu Ren, Mingxiang Chen, Xiangchao Meng, Yuxin Zhang, Yanlin Li, Lei Ren, Wei Chen, Zhiyuan Huang, Mingjie Zhan, Xiaojie Wang, and Fangxiang Feng. 2025. Evaluating mllms with multimodal multi-image reasoning benchmark. Preprint, arXiv:2506.04280.

Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols, and Ranjitha Kumar. 2017. Rico: A mobile app dataset for building data-driven design applications. In Proceedings of the 30th annual ACM symposium on user interface software and technology, pages 845–854.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning.

Peitong Duan, Chin-Yi Cheng, Gang Li, Bjoern Hartmann, and Yang Li. 2024. Uicrit: Enhancing automated design evaluation with a ui critique dataset. In Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology, pages 1–17.

Hermann Ebbinghaus, Henry A Ruger, and Clara E Bussenius. 1913. Memory: A contribution to experimental psychology.

Paul M Fitts and James R Peterson. 1964. Information capacity of discrete motor responses. Journal of experimental psychology, 67(2):103.

Brian J Fogg. 2002. Persuasive technology: using computers to change what we think and do. Ubiquity, 2002(December):2.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. 2024. Mme: A comprehensive evaluation benchmark for multimodal large language models. Preprint, arXiv:2306.13394.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. 2025. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. Preprint, arXiv:2501.05444.

William E Hick. 1952. On the rate of gain of information. Quarterly Journal of experimental psychology, 4(1):11–26.

Jerry R Hobbs, Mark E Stickel, Douglas E Appelt, and Paul Martin. 1993. Interpretation as abduction. Artificial intelligence, 63(1-2):69–142.

Clark L Hull. 1932. The goal-gradient hypothesis and maze learning. Psychological review, 39(1):25.

Joyce Karreman, Pietro Romeo, and Qian Li. 2016. Cross-cultural hci and ux design: a comparison of chinese and western user interfaces. Unpublished work.

Kurt Koffka. 1922. Perception: an introduction to the gestalt-theorie. Psychological bulletin, 19(10):531.

Ron Kohavi, Roger Longbotham, Dan Sommerfield, and Randal M Henne. 2009. Controlled experiments on the web: survey and practical guide. Data mining and knowledge discovery, 18:140–181.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a. Llava-onevision: Easy visual task transfer. Preprint, arXiv:2408.03326.

Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. 2024b. Seedbench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17889–17904, Miami, Florida, USA. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore. Association for Computational Linguistics.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024b. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. 2024. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR).

Yuxuan Lu, Jing Huang, Yan Han, Bingsheng Yao, Sisong Bei, Jiri Gesi, Yaochen Xie, Zheshen, Wang, Qi He, and Dakuo Wang. 2025a. Prompting is not all you need! evaluating llm agent simulation methodologies with real-world online customer behavior data. Preprint, arXiv:2503.20749.

Yuxuan Lu, Bingsheng Yao, Hansu Gu, Jing Huang, Jessie Wang, Yang Li, Jiri Gesi, Qi He, Toby Jia-Jun Li, and Dakuo Wang. 2025b. Uxagent: A system for simulating usability testing of web design with llm agents. Preprint, arXiv:2504.09407.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, and 1 others. 2023. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594.

George A Miller. 1956. The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological review, 63(2):81.

Jakob Nielsen. 1994. Enhancing the explanatory power of usability heuristics. In Proceedings of the SIGCHI conference on Human Factors in Computing Systems, pages 152–158.

Jakob Nielsen. 2000. End of web design. Alertbox [en línea] Disponible en: www. useit. com/alertbox/20000723. html, 23(06):2000.

Don Norman. 2007. Emotional design: Why we love (or hate) everyday things. Basic books.

Donald A Norman. 1988. The psychology of everyday things. Basic books.

Harri Oinas-Kukkonen and Marja Harjumaa. 2018. Persuasive systems design: key issues, process model and system features 1. In Routledge handbook of policy design, pages 87–105. Routledge.

OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Ma˛dry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, and 401 others. 2024a. Gpt4o system card. Preprint, arXiv:2410.21276.

OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, and 244 others. 2024b. Openai o1 system card. Preprint, arXiv:2412.16720.

OpenAI. 2025. GPT-5.1 Instant and GPT-5.1 Thinking System Card Addendum.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Vilfredo Pareto. 1919. Manuale di economia politica con una introduzione alla scienza sociale, volume 13. Società editrice libraria.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.

Federico Quin, Danny Weyns, Matthias Galster, and Camila Costa Silva. 2024. A/b testing: A systematic literature review. Journal of Systems and Software, 211:112011.

Hedwig Von Restorff. 1933. Über die wirkung von bereichsbildungen im spurenfeld. Psychologische Forschung, 18:299–342.

Dakuo Wang, Ting-Yao Hsu, Yuxuan Lu, Hansu Gu, Limeng Cui, Yaochen Xie, William Headean, Bingsheng Yao, Akash Veeragouni, Jiapeng Liu, Sreyashi Nag, and Jessie Wang. 2025a. Agenta/b: Automated and scalable web a/btesting with interactive llm agents. Preprint, arXiv:2504.09723.

Ziqi Wang, Hanlin Zhang, Xiner Li, Kuan-Hao Huang, Chi Han, Shuiwang Ji, Sham M. Kakade, Hao Peng, and Heng Ji. 2025b. Eliminating position bias of language models: A mechanistic approach. Preprint, arXiv:2407.01100.

Jason Wu, Yi-Hao Peng, Xin Yue Amanda Li, Amanda Swearngin, Jeffrey P Bigham, and Jeffrey Nichols. 2024. Uiclip: a data-driven model for assessing user interface design. In Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology, pages 1–16.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, Wenhai Wang, Jifeng Dai, and Jinguo Zhu. 2025. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. Preprint, arXiv:2504.15279.

Bo Yang and Shanping Li. 2024. Uisgpt: Automated mobile ui design smell detection with large language models. Electronics, 13(16):3127.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan

Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and 3 others. 2024. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR.

Daoan Zhang, Junming Yang, Hanjia Lyu, Zijian Jin, Yuan Yao, Mingkai Chen, and Jiebo Luo. 2024. Cocot: Contrastive chain-of-thought prompting for large multimodal models with multiple image inputs. Preprint, arXiv:2401.02582.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. Preprint, arXiv:1904.09675.

Xinlu Zhang, Yujie Lu, Weizhi Wang, An Yan, Jun Yan, Lianke Qin, Heng Wang, Xifeng Yan, William Yang Wang, and Linda Ruth Petzold. 2023. Gpt-4v(ision) as a generalist evaluator for vision-language tasks. Preprint, arXiv:2311.01361.

Bingchen Zhao, Yongshuo Zong, Letian Zhang, and Timothy Hospedales. 2024. Benchmarking multiimage understanding in vision and language models: Perception, knowledge, reasoning, and multi-hop reasoning. Preprint, arXiv:2406.12742.

Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, and Sibei Yang. 2023. Ddcot: Duty-distinct chain-ofthought prompting for multimodal reasoning in language models. Advances in Neural Information Processing Systems, 36:5168–5191.

### A WiserUI-Bench Details

#### A.1 Curation Details

Image Preprocess During benchmark construction process, we observed that some UI images from our data sources included manual indicators, such as circled regions, numbered markers, and arrows, highlighting specific UI elements related to reasoning points from A/B test results (see Figure 17). While helpful for human interpretation, these indicators introduced artificial attention cues that could potentially bias model evaluation.

To mitigate this, we cleaned the dataset using inpainting techniques. We first identified common indicator patterns via edge detection and color-based segmentation, including circular highlights drawn in semi-transparent colors, numbered markers, and arrows. Our automated pipeline primarily utilized OpenCV (Bradski, 2000) for marker detection and inpainting. For a small number of complex cases where automated methods were insufficient, we applied a hybrid refinement approach using AI-based editing tools (e.g., Gemini3). These tools were used solely to remove attached markers or visual indicators, without modifying the underlying UI layout or design elements. This process ensured a cleaner, more naturalistic dataset for fair visual reasoning evaluation.

Interpretation Annotation As shown in Figure 20, three UI/UX experts were presented with a pair of UI images, one designated as the winner based on A/B test results, along with the corresponding source website link. Experts were instructed to provide key interpretations explaining why the winning variant supported user actions more effectively than the alternative. We engaged domain experts for this task, as user behavior is often subconscious and shaped by implicit context, making evaluation by non-experts inherently difficult.

Annotators were guided by two resources: textual descriptions from the original sources, which often highlight key design intentions, and the Nielsen Norman 10 Usability Heuristics (Nielsen, 1994), a robust theoretical framework widely used in usability research, which served as broad conceptual criteria. The resulting interpretations were then mapped to 12 curated UX laws that remain widely applied in practice. These laws provide concrete, actionable guidance across diverse aspects of user experience, which can be categorized into

3https://gemini.google.com

three cognitive dimensions: Perception, Memory, and Action. This classification is inspired by the framework proposed by Norman (2007), which introduced the concept of Emotional Design. The full list of UX laws and their classifications is presented in Table 5.

#### A.2 Statistics Details

Figure 18 presents the distribution of WiserUIBench across page types, industry domains, device types. For clarification on the page type classification, the homepage refers to the main entry point of a website, typically offering broad navigation and general content. In contrast, a landing page is usually a standalone page optimized for a specific user action, such as signing up or making a purchase.

Full distribution of UI change attribute types is shown in Figure 19. For clarification, element-level counts reflect the number of distinct UI components affected, while attribute-level counts capture the total number of modified attributes across all components. For example, an instance annotated as <Input Field: Padding, Color / Button: Color> is counted as 2 element changes and 3 attribute changes.

B Experiment Details

##### B.1 Setup B.1.1 Model Configuration For proprietary MLLMs, we accessed the models via their official API endpoints as follows:

- • o1 (OpenAI et al., 2024b): o1-2024-12-17
- • GPT-5.1 (OpenAI, 2025): gpt-5.1-2025-11-13 (reasoning effort: none)
- • GPT-4o (OpenAI et al., 2024a): gpt-4o-2024-08-06
- • Claude 4.5 Sonnet (Anthropic, 2025): claude-sonnet-4-5-20250929 (thinking type: disabled)
- • Claude 3.5 Sonnet (Anthropic, 2024): claude-3-5-sonnet-20240620

For open-source models, we used publicly available weights from the Huggingface Hub4. Specific model versions are as follows:

4https://huggingface.co/models

- • Qwen3-VL-30B-A3B (Bai et al., 2025a): Qwen/Qwen3-VL-30B-A3B-Instruct
- • Qwen3-VL-8B (Bai et al., 2025a): Qwen/Qwen3-VL-8B-Instruct
- • Qwen2.5-VL-32B (Bai et al., 2025b): Qwen/Qwen2.5-VL-32B-Instruct
- • Qwen2.5-VL-7B (Bai et al., 2025b): Qwen/Qwen2.5-VL-7B-Instruct
- • InternVL-2.5-38B (Chen et al., 2025): OpenGVLab/InternVL2_5-38B
- • InternVL-2.5-8B (Chen et al., 2025): OpenGVLab/InternVL2_5-8B
- • LLaVA-NeXT-7B (Liu et al., 2024a): llava-hf/llava-v1.6-mistral-7b-hf
- • LLaVA-OneVision-7B (Li et al., 2024a): llava-hf/llava-onevision-qwen2-7b-ovhf

The temperature for all models was set to 0.2.

#### B.1.2 Resources

All experiments using open-source models were conducted on a Linux machine (Ubuntu 20.04.6 LTS) equipped with 4 NVIDIA RTX A6000 GPUs, each with 48GB VRAM. We used the following software libraries and frameworks: transformers version 4.53.2 and vllm version 0.9.2.

B.2 UI/UX Design Selection

- B.2.1 Prompts

Figures 8 through 14 show the prompts used, including those designed for diverse reasoning strategies.

- B.2.2 Token Counts

Token counts across reasoning strategies on proprietary models are presented in Table 6.

- B.2.3 Standard Deviation

Table 7 presents the standard deviation across three experiment runs for each model and reasoning method. For metric abbreviation, FS and SS denote the standard deviation when the first or second UI image input was the more effective one, while AS and CS refer to the standard deviation of total correct and both correct cases, respectively.

- B.2.4 Additional Case Study Figure 21 and 22 present the additional case studies across different models. Figure 23 shows an example comparing various reasoning strategies executed with GPT-4o. B.3 UI/UX Design Interpretation
- B.3.1 Prompts Figure 15 presents the prompt used for models’ interpretation generation, and Figure 16 presents the prompt used for its GPT-4o-based evaluation.

- B.3.2 Results on Text Similarity Metrics Alongside Interpretation Recall and Instance-level Recall from GPT-4o-based evaluation, we report BLEU, ROUGE, METEOR, and BERTScore (Papineni et al., 2002; Lin, 2004; Banerjee and Lavie, 2005; Zhang et al., 2020) in Table 8 as complementary lexical-overlap metrics.

The consistently low scores observed in BLEU, ROUGE, and METEOR primarily stem from verbosity and vocabulary differences between concise expert interpretations and more elaborative modelgenerated explanations, as also illustrated in Figure 7. As such, these metrics largely capture stylistic divergence rather than true semantic misalignment, In contrast, our recall-based metrics are explicitly designed to assess semantic inclusion, thereby providing a more faithful characterization of the alignment between expert and model interpretations, which motivates their use as the main evaluation criterion.

- B.3.3 Additional Case Study Figure 24 presents an additional case study across different models. C Icon Attribution

Icons used in this paper are from Flaticon (https://www.flaticon.com), and they are attributed to the respective authors as required by Flaticon’s license.

UX Dimension UX Law Description

Elements sharing a boundary or background are perceived as a unified group. This can be a visible border, a colored box, or a distinct region in the layout.

Law of Common Region (Koffka, 1922)

Items positioned close together are seen as belonging to the same group. This mental shortcut helps users quickly identify relationships between adjacent elements.

Law of Proximity (Koffka, 1922)

Perception

Humans naturally perceive and interpret complex images in their simplest forms. It reduces cognitive effort by helping us group and understand elements more quickly.

Law of Prägnanz (Koffka, 1922)

People group together elements that look alike, assuming they serve a similar purpose. Consistency in color, shape, or size leads the user to see these items as related.

Law of Similarity (Koffka, 1922)

People remember a distinct element more easily than others that blend in. This is also known as the Isolation Effect, emphasizing our attention on anything unusual.

Von Restorff Effect (Von Restorff, 1933)

Users tend to remember the first and last items in a series more than those in the middle. This influences how they recall sequential information or actions.

Serial Position Effect (Ebbinghaus et al., 1913)

Memory

On average, people can hold only about seven items in their working memory at once. Going beyond this limit causes cognitive overload.

Miller’s Law (Miller, 1956)

Users expect new interfaces to operate similarly to those they’ve used before. Leveraging established patterns lowers cognitive load and accelerates adoption.

Jakob’s Law (Nielsen, 2000)

Decision time rises with the number and complexity of choices. Too many options can overwhelm users and slow their actions.

Hick’s Law (Hick, 1952)

Roughly 80% of outcomes stem from 20% of causes. Focusing on the most impactful features or tasks delivers the greatest improvement.

Pareto Principle (Pareto, 1919)

Action

Fitts’s Law (Fitts and Peterson, 1964)

The time required to reach a target depends on its size and distance. Larger, closer elements are easier to access quickly and accurately.

Motivation increases as people get closer to finishing a task. Providing clear progress markers encourages users to continue and complete it faster.

Goal-Gradient Effect (Hull, 1932)

Table 5: Overview of UX laws used for interpretation annotation, and their categorization by cognitive dimension (Perception, Memory, Action).

Models Methods Input Tokens Output Tokens o1 Zero-Shot 1292.5 (± 559.77) 1347.7 (± 489.76)

Zero-Shot 1453.4 (± 634.41) 248.8 (± 57.98) CoCoT 1458.4 (± 634.41) 329.7 (± 63.29)

Self-Refine 5240.4 (± 1935.80) 820.9 (± 142.43)

GPT-4o

DDCoT 1521.4 (± 634.41) 415.3 (± 73.42)

MAD (R1) 4942.4 (± 1903.15) 483.1 (± 58.5)

MAD (R3) 11501.5 (± 4435.48) 1158.1 (± 82.60)

Zero-Shot 1686.6 (± 1062.43) 401.4 (± 72.94)

CoCoT 1692.5 (± 1062.58) 487.0 (± 82.28) Self-Refine 6230.7 (± 3245.69) 1201.4 (± 207.32) DDCoT 1754.5 (± 1064.29) 582.2 (± 97.13) MAD (R1) 5895.5 (± 3224.31) 834.1 (± 125.13)

Claude 3.5 Sonnet

MAD (R3) 13853.4 (± 7512.63) 1936.5 (± 276.14)

Table 6: Overview of Token Counts (µ ± σ) in the UI/UX design selection task on WiserUI-Bench.

Model Method FS SS AS CS o1 Zero-Shot 1.528 2.082 3.606 0.577

Zero-Shot 1.528 1.000 2.082 1.528 CoCoT 4.000 2.646 1.732 1.732 Self-Refine 2.517 1.000 3.215 0.577 DDCoT 2.082 2.309 0.577 2.517 MAD (R1) 3.215 3.055 6.245 2.646

GPT-4o

- MAD (R3) 0.577 2.082 1.732 3.215

Claude 3.5 Sonnet

Zero-Shot 1.528 2.000 3.512 2.887 CoCoT 3.055 3.512 0.577 0.577 Self-Refine 2.646 3.512 4.619 1.528 DDCoT 2.082 1.732 0.577 2.517 MAD (R1) 1.000 1.528 2.309 4.359

- MAD (R3) 1.000 2.887 3.786 3.786

Qwen2.5-VL-32B Zero-Shot 1.732 2.309 2.082 2.000 Qwen2.5-VL-7B Zero-Shot 2.646 0.577 2.517 3.215 InternVL-2.5-38B Zero-Shot 2.646 1.000 3.000 3.055 InternVL-2.5-8B Zero-Shot 1.000 2.646 2.646 2.000 LLaVA-NeXT-7B Zero-Shot 3.512 1.155 4.583 6.028 LLaVA-OneVision-7B Zero-Shot 4.041 2.309 4.359 3.512

- Table 7: Results on standard deviation across three experiment runs in the UI/UX design selection task for each model and reasoning strategy on WiserUI-Bench.

Model BLEU-2 BLEU-4 ROUGE-L METEOR BERTScore

o1 0.0181 0.0041 0.0972 0.1630 0.8414 GPT-4o 0.0162 0.0040 0.0875 0.1557 0.8233 Claude 3.5 Sonnet 0.0141 0.0037 0.0654 0.1418 0.8076

Qwen2.5-VL-32B 0.0098 0.0025 0.0442 0.1016 0.8085 Qwen2.5-VL-7B 0.0180 0.0048 0.0789 0.1481 0.8331 InternVL-2.5-38B 0.0128 0.0034 0.0574 0.1225 0.8080 InternVL-2.5-8B 0.0111 0.0032 0.0510 0.1124 0.8082 LLaVA-NeXT-7B 0.0077 0.0018 0.0386 0.0908 0.8127 LLaVA-OneVision-7B 0.0157 0.0043 0.0718 0.1416 0.8255

- Table 8: Results of text similarity metrics per expert interpretation for the UI/UX design interpretation task on WiserUI-Bench.

The two screenshots show two different versions of the same page. Identify the key UI differences between the two versions, and then evaluate which variant is more effective UI/UX design that leads to better user experience and conversion.

You should end your answer with following the format (No bold, etc): More effective: <First/Second>

- Figure 8: Prompt used for the UI/UX design selection task with zero-shot strategy.

You are an expert in designing UI/UX for web/apps. The two screenshots show two different versions of the same page. Identify the key UI differences between the two versions, and then evaluate which variant is more effective UI/UX design that leads to better user experience and conversion. Think step by step. You should end your answer with the following format (No bold, etc): More effective: <First/Second>

- Figure 9: Prompt used for the UI/UX design selection task with CoCoT strategy.

You are an expert in designing UI/UX for web/apps. The two screenshots show two different versions of the same page. We want to identify the key UI differences between the two versions, and then evaluate which variant is more effective UI/UX design that leads to better user experience and conversion. Review your previous answer and find problems with your answer. Previous answer: {previous_answer}

- Figure 10: Prompt used for the UI/UX design selection task with Self-Refine strategy on the reviewing step. Prompt inputs are in bold.

You are an expert in designing UI/UX for web/apps. The two screenshots show two different versions of the same page. We want to identify the key UI differences between the two versions, and then evaluate which variant is more effective UI/UX design that leads to better user experience and conversion. Your previous answer: {previous_answer} Feedback on your previous answer: {feedback}

Based on the feedback for your previous answer, improve your answer.

You should end your answer with following the format (No bold, etc): More effective: <First/Second>

- Figure 11: Prompt used for the UI/UX design selection task with Self-Refine strategy on the improving step. Prompt inputs are in bold.

You are an expert in designing UI/UX for web/apps. The two screenshots show two different versions of the same page. We want to identify the key UI differences between the two versions and then evaluate which variant is more effective UI/UX design that leads to better user experience and conversion. You should

- (1) Think step by step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions.
- (2) With the aim of helping answer the original question, try to answer the sub-questions.
- (3) Give your final answer according to the sub-questions and sub-answers.

You should end your answer with the following format (No bold, etc): More effective: <First/Second>

- Figure 12: Prompt used for the UI/UX design selection task with DDCoT strategy.

And you are a debater. Hello and welcome to the debate.

The two screenshots show two different versions of the same page. Identifying the key UI differences between the two versions, you think the {first/second} variant is more effective UI/UX design that leads to better user experience and conversion. It’s not necessary to fully agree with each other’s perspectives, as our objective is to find the correct answer.

The opponent proposed that: {opponent_opinion}

- Figure 13: Prompt used for the UI/UX design selection task with MAD strategy on the debate step arguing the first/second version better drives user actions. Prompt inputs are in bold.

You are an expert in designing UI/UX for web/apps. And you are a moderator. There will be two debaters involved in a debate. The two screenshots show two different versions of the same page. They are discussing which variant is more effective UI/UX design that leads to better user experience and conversion. At the end of each debate round, you will evaluate answers and decide which is correct. Now the {number} round of debate for both sides has ended. Argue from the side arguing for the first version: {first_reason} Argue from the side arguing for the second version: {second_reason}

Then what is the correct answer? Please give the final answer that you think is correct and summarize your reasons.

You should end your answer with following the format (No bold, etc): More effective: <First/Second>

- Figure 14: Prompt used for the UI/UX design selection task with MAD strategy on the moderation step. Prompt inputs are in bold.

You are an expert in designing UI/UX for web/apps. The two screenshots show two different versions of the same page. If the SECOND version has proven to be more effective UI/UX design that leads to better user experience and conversion, provide all possible and appropriate interpretations for its superiority.

- Figure 15: Prompt used for the UI/UX design interpretation task.

Given two statements are interpretations why one UI is better than another in terms of guiding user behavior effectively.

- Statement 1: {model_reason}
- Statement 2: {reference_reason}

Your task is to assess whether the first statement contains the same reasoning and essential content as the second statement. If it does, answer with ’Yes’. Otherwise, answer with ’No’.

- Figure 16: Prompt used for GPT-4o-based evaluation on the UI/UX design interpretation task. Prompt inputs are in bold.

[Figure 285]

Figure 17: Example of visual indicator removal in WiserUI-Bench. Left: Original image with markers added by data source website. Right: After inpainting, with markers removed while preserving UI elements.

[Figure 286]

[Figure 287]

[Figure 288]

Figure 18: Detailed statistics of WiserUI-Bench on (a) page type, (b) industry domain, (c) device type

[Figure 289]

[Figure 290]

###### Figure 19: Detailed statistics of WiserUI-Bench on UI change attribute type

[Figure 291]

[Figure 292]

###### Figure 20: A screenshot of the interpretation annotation interface for UI/UX experts based on original data sources and guidelines. After writing the key interpretations, they also selected the most relevant UX law.

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

###### Figure 21: A qualitative example from the UI/UX design selection task on WiserUI-Bench: Both Claude 3.5 Sonnet and InternVL-2.5-38B selected the A/B test winner.

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

###### Figure 22: A qualitative example from the UI/UX design selection task on WiserUI-Bench: GPT-4o selected the A/B test winner, whereas LLaVA-NeXT-7B did not.

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

###### Figure 23: A qualitative example from the UI/UX design selection task on WiserUI-Bench, showcasing the effect of diverse reasoning strategies executed on GPT-4o.

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

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

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

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

###### Figure 24: A qualitative example from the UI/UX design interpretation task on WiserUI-Bench: InternVL-2.5-8B captured only the memory part, while Qwen2.5-VL-32B identified two key interpretations. The highlights in the models’ responses indicate the parts that align with the expert-curated key interpretations.

