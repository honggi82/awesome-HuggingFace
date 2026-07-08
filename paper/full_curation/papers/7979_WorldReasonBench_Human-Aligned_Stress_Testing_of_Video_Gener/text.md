# arXiv:2605.10434v1[cs.CV]11May2026

## WorldReasonBench: Human-Aligned Stress Testing of Video Generators as Future World-State Predictors

Keming Wu1∗ Yijing Cui1∗ Wenhan Xue1 Qijie Wang1 Xuan Luo1 Zhiyuan Feng1 Zuhao Yang2 Sudong Wang4 Sicong Jiang5 Haowei Zhu1 Zihan Wang5 Ping Nie3

Wenhu Chen3 Bin Wang1

1Tsinghua University 2Nanyang Technological University 3University of Waterloo 4Hong Kong University of Science and Technology (Guangzhou) 52077 AI

∗Equal contribution Corresponding author

Project page: https://unix-ai-lab.github.io/WorldReasonBench/

{wukm25, cuiyj25}@mails.tsinghua.edu.cn

### Abstract

Commercial video generation systems such as Seedance2.0 and Veo3.1 have rapidly improved, strengthening the view that video generators may be evolving into “world simulators.” Yet the community still lacks a benchmark that directly tests whether a model can reason about how an observed world should evolve over time. We introduce WorldReasonBench, which reframes video generation evaluation as world-state prediction: given an initial state and an action, can a model generate a future video whose state evolution remains physically, socially, logically, and informationally consistent? WorldReasonBench contains 436 curated test cases with structured ground-truth QA annotations spanning four reasoning dimensions and 22 subcategories. We evaluate generated videos with a human-aligned twopart methodology: Process-aware Reasoning Verification uses structured QA and reasoning-phase diagnostics to detect temporal and causal failures, while Multidimensional Quality Assessment scores reasoning quality, temporal consistency, and visual aesthetics for ranking and reward modeling. We further introduce WorldRewardBench, a preference benchmark with approximately 6K expert-annotated pairs over 1.4K videos, supporting pair-wise and point-wise reward-model evaluation. Across modern video generators, our results expose a persistent gap between visual plausibility and world reasoning: videos can look convincing while failing dynamics, causality, or information preservation. We will release our benchmarks and evaluation toolkit to support community research on genuinely world-aware video generation at https://github.com/UniX-AI-Lab/WorldReasonBench/.

### 1 Introduction

The rapid advance of large-scale video generation models [17, 9, 23, 28, 27] has shifted the central question in video generation. Frontier systems in the Seedance, Veo, and Sora families [3, 26, 1] now produce longer, cleaner, and more controllable videos, while recent studies suggest that video models may already exhibit zero-shot learning and reasoning-like behavior in selected settings [26]. These advances make it increasingly plausible to ask whether modern video generators are beginning to act as world models rather than only powerful pixel synthesizers.

Evaluation, however, has not kept pace with this shift. Most existing benchmarks still emphasize perceptual quality, motion smoothness, or prompt alignment. Recent reasoning-oriented efforts each cover a useful slice of the problem but stop short of open-domain world-state prediction: V-

Preprint.

[Figure 1]

- Figure 1: Overview of WorldReasonBench. We evaluate video generators as world-state predictors: given an initial visual state and an action or instruction, the model must generate a future video whose state evolution remains physically, socially, logically, and informationally consistent. WorldReasonBench spans four reasoning dimensions organized into 22 concise, dimension-specific subcategories, and is paired with complementary automated and human-aligned evaluation pipelines.

ReasonBench [14] and Gen-ViRe [11] target answer-verifiable cognitive tasks, VIPER [10] formalizes process-aware diagnostics on procedural settings, WorldSimBench [18] focuses on embodied control, and VideoVerse [25] evaluates single-event causality with binary QA. None of them asks, end-to-end and on open-domain content, whether a generator that observes an initial visual state can correctly infer and simulate the future evolution of the world, and none releases calibrated expert preference data for reward-model evaluation. This gap is especially consequential for the open-source community:

- as frontier commercial systems improve rapidly, the field needs a common benchmark that can tell whether open-source progress reflects genuine reasoning gains or simply better visual polish.

Consider a simple example: a generator given an image of an apple on a branch and instructed to drop it may produce a visually impressive clip—smooth motion, realistic textures, attractive lighting—yet fail as a world model if the apple accelerates upward, splits in mid-air, or traces a linear rather than parabolic trajectory. Standard quality metrics reward such a video for realism while missing its failure to obey basic dynamics. The core question is therefore not only how good the video looks, but whether the model has generated the right future state transition. We accordingly recast video generation evaluation as world-state prediction: given an initial visual state and an action or instruction, can the model roll the world forward into temporally consistent future states? We further separate transitions that are inferable from visual evidence alone from those that benefit from explicit textual guidance, probing reasoning under different levels of external help. We introduce WorldReasonBench, a reasoning-aware benchmark with 436 curated test cases and structured ground-truth QA annotations, guided by the principle that a true world model should be interrogable—one should be able to ask reasoning-oriented questions about the video and obtain answers consistent with real-world knowledge. Since binary QA alone may hide process failures, we evaluate each model through two complementary components, Process-aware Reasoning Verification and Multi-dimensional Quality Assessment. Our contributions are: (1) WorldReasonBench, a reasoning-aware benchmark covering four dimensions and 22 subcategories that tests whether 11 closed- and open-source generators roll an observed initial state into a coherent future sequence (Figure 1); (2) a human-aligned evaluation methodology combining Process-aware Reasoning Verification with Multi-dimensional Quality Assessment, validated against expert human preferences; and (3) WorldRewardBench, a

- Table 1: Comparison with existing video generation reasoning benchmarks along auditable axes. #Cases: N/R if not stated. Input: T2V, TI2V(initial image + text), or Embodied. Reward Data: publicly released preferences. Process Phases: ≥ 2 phase-level scores. Human Calib.: rank correlation or pairwise agreement vs. experts (Likert user study alone does not qualify).

Dim. / Sub-dim.

Reward Data

Process Phases

Human Calib.

Benchmark

#Cases Input Domain Annotation Unit

VBench-2.0 [30] 5/18 ≈1260 T2V curated open prompts Likert + auto metrics × × user study WorldSimBench [18] 3/20 2,831 T2V/TI2V embodied scenarios perceptual + manipulative √

× task success V-ReasonBench [14] 4/13 326 TI2V answer-verifiable tasks rule-based verifier × × manual val. Gen-ViRe [11] 6/24 72 T2V cognitive subtasks per-task verifier × × human val. VIPER [10] 6/16 309 T2V procedural / rule-following per-step process score ×

√ – VideoVerse [25] 10/– 300 T2V single-event causality binary QA / event (793) × partial user study WorldReasonBench 4/22 436 TI2V open-domain world-state prediction 5–7 QA pairs / case ∼6K pref. pairs √ (4 phases) expert Elo + Spearman

preference-based calibration benchmark with approximately 6K expert-annotated pairs over 1.4K videos supporting pair-wise and point-wise reward-model evaluation.

### 2 Related Work

Video generation models as world simulators. Popularized by Sora [1], the view of video generators as world simulators has become more compelling as commercial systems such as Seedance and Veo improve in long-horizon coherence, controllability, and realism [3, 26], with recent studies even suggesting zero-shot learning and reasoning-like behavior in selected settings [26]. Capability demos alone do not establish robust world understanding, however: physical-law analyses show that even strong models fail on gravity, object permanence, and causal consistency [8]. We therefore aim to test these claims systematically rather than infer them from isolated examples.

Benchmarks and automatic evaluation for video generation. Existing video benchmarks mostly target perceptual quality or prompt alignment via reference metrics (FID [6], FVD [22], LPIPS [29]) and aesthetics/compositionality suites [7, 30, 12, 13, 19], none of which provide structured reasoning verification. Reasoning-oriented benchmarks each cover one slice—embodied task-success [18], small-scale answer-verifiable puzzles [14, 11], procedural process-aware tasks [10], single-event causality with Likert ratings [25], physical-law or rule-governed transitions [16, 5], and video understanding rather than generation [24]. VLM-as-Judge pipelines [31, 15, 4] scale evaluation but single-pass judges over-reward visual plausibility and miss process-level errors. WorldReasonBench instead pairs an initial image with a text instruction to probe open-domain future-state evolution, annotates each case with 5–7 QA pairs across four reasoning phases (state, process, fidelity, mechanism), and releases WorldRewardBench with ∼6K expert preference pairs over 1,432 videos from 11 generators to calibrate automatic metrics.

### 3 WorldReasonBench

We frame video generation as world-state prediction: given an observed initial state and an instruction, a generator should produce a future video that follows the intended world evolution rather than merely appearing realistic.

Problem formulation and instruction regimes. Let x0 be the initial world state and a the intended action or transition; a generator produces Vˆ = G(x0,a), and evaluation asks whether Vˆ faithfully realizes the state evolution implied by both inputs. To measure how much textual guidance helps, we evaluate each case under two regimes: aimplicit provides only a high-level intent, while ahinted adds explicit transition guidance, and the resulting gap ∆hint = Score(Vˆ(1)) − Score(Vˆ(0)) measures the reasoning assistance benefit.

#### 3.1 WorldReasonBench Construction

WorldReasonBench is constructed to evaluate whether a video generator can predict future world states from an observed initial state. As shown in Figure 2(A), construction consists of a compact reasoning taxonomy and a three-stage VLM-assisted data pipeline.

[Figure 2]

(a). WorldReasonBench

###### Taxonomy of WorldReasonBench

###### Prompt Design and Human Check

- 1.Data Collection-Crawler

[Figure 3]

Web Images Academic Sources

Existing Datasets

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

Photorealistic office or study scene, a man is standing in front of a closed cabinet……

[Figure 15]

[Figure 16]

[Figure 17]

- 2.Data Collection-Generation

[Figure 18]

[Figure 19]

[Figure 20]

4 Topics & 22 Tasks

[Figure 21]

[Figure 22]

Keywords: Astronaut, hammer, feather

[Figure 23]

[Figure 24]

World-Knowledge

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Human-Centric

[Figure 29]

[Figure 30]

Human Review

[Figure 31]

Logic Reasoning

When the person releases both objects from their hands at the same time, what will happen next?

[Figure 32]

Information-based Reasoning

[Figure 33]

###### (b). WorldRewardBench

[Figure 34]

###### Video Generation

Human Annotation

Data Post-Processing

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Sampled Videos(8)

High-Disagreement Detection

[Figure 39]

Re-annotation

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Assign the video to new annotators, ≥4 ratings per high disagreement video.

[Figure 48]

[Figure 49]

[Figure 50]

Seedance 2.0 Kling-V3 Veo-3.1 Sora-2

- 2 Users → range > 1
- 3 Users → range > 2

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

……

[Figure 59]

[Figure 60]

+ ( ,video prompt)

LongCat 13 models

[Figure 61]

Wan2.6 Hunyuan 1.5

[Figure 62]

…

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

U1 U15

[Figure 68]

Better

[Figure 69]

[Figure 70]

[Figure 71]

video 1

Three Dimensional Scoring

[Figure 72]

……

Pair

+ video prompt

- 1. Reasoning Quality
- 2. Temporal Consistency
- 3. Visual Aesthetics

[Figure 73]

Worse

[Figure 74]

video 13

- Figure 2: Benchmark construction pipeline. A: WorldReasonBench construction, including taxonomy-aware captioning, prompt generation, and QA generation. B: WorldRewardBench construction, including video sampling, expert scoring, preference-pair construction, and human-alignment evaluation.

Reasoning taxonomy. We organize world reasoning into four high-level dimensions and 22 short, interpretable subcategories. The complete taxonomy is visualized in Figure 1, with detailed definitions, examples, and inclusion criteria provided in Appendix C.

Question design. Each test case is associated with a compact set of structured QA pairs spanning four question types: factual (28.4%, direct visual verification), reasoning (27.1%, causal mechanism understanding), detail (24.7%, fine-grained element verification), and temporal (19.7%, sequence and timing verification). Questions are further stratified into easy, medium, and hard difficulty levels, enabling fine-grained analysis across both reasoning type and difficulty.

Data curation pipeline. We construct each benchmark case through three VLM-assisted stages. First, Qwen3.5 [20] produces a structured caption covering subjects, spatial relations, visual attributes, text/numeric elements, scene context, and potential dynamics. Second, Qwen3.5-27B generates reasoning-aware prompts conditioned on the target dimension, subcategory, and instruction regime. Third, Gemini3.1-Pro generates ground-truth QA pairs with expected answers, question-type labels, difficulty labels, and evaluation criteria. We use iterative JSON validation and repair to ensure reliable structured annotations. To control for VLM bias in the generated QA, two trained auditors further audit a stratified random subset on answerability, ground-truth correctness, and answer uniqueness, and rejected cases are rewritten or removed; the audit protocol and statistics are reported in Appendix H.4.

#### 3.2 WorldRewardBench Construction

WorldRewardBench provides a human-aligned preference benchmark for evaluating whether automatic video judges recover expert preferences over world-reasoning failures. As summarized in Figure 2(B), we build it from a high-quality subset of WorldReasonBench: for each selected case, we collect generations from 11 video generation models and sample 8 videos per case to form a diverse annotation pool.

Human annotation and preference construction. Fifteen trained annotators rate each video on reasoning quality, temporal consistency, and visual aesthetics using a 1–5 scale. We aggregate these ratings as S(v) = 0.4sr(v) + 0.3sc(v) + 0.3sa(v), then rank videos within each benchmark case to derive candidate pairwise preferences. We apply confidence-aware filtering over score margins, relabel near-equal pairs (∆ij < 0.1) as ties, and randomize left/right order to reduce presentation bias.

[Figure 75]

[Figure 76]

###### (a). Process-Aware Reasoning Verification (b). Multi-Dimensional Quality Assessment

- 1.Reasoning Verification Chain
- 2.Aggregate Metrics

[Figure 77]

QA-Pair

Detail

[Figure 78]

[Figure 79]

Temporal

Reasoning

[Figure 80]

Factual prompt

[Figure 81]

+

[Figure 82]

[Figure 83]

+

+

[Figure 84]

[Figure 85]

sstate sproc sfidel smech

[Figure 86]

[Figure 87]

[Figure 88]

AccQA

[Figure 89]

sout = (sstate + sfidel)/2 sdyn = (sproc + smech)/2 ΔRG = sout - sdyn

[Figure 90]

Outcome-Hacking Rate P(ΔRG > 0.3) Process-Aware Reasoning Score AccQA1-β •sdynβ Process-Completeness Diagnostic sdyn /AccQA

[Figure 91]

[Figure 92]

Figure 3: Evaluation pipeline. A: Process-aware Reasoning Verification, which answers structured QA pairs from generated videos and converts them into reasoning-phase diagnostics. B: Multi-dimensional Quality Assessment, which scores each video along reasoning quality, temporal consistency, and visual aesthetics for ranking and reward-model evaluation.

The resulting benchmark contains approximately 6K balanced preference pairs over 1.4K unique videos; implementation details and exact statistics are in Appendix J.

WorldRewardBench supports pair-wise and point-wise reward-model evaluation through preference agreement, rank correlation, and tie/divergence diagnostics, providing the human-aligned calibration layer for the automatic evaluation methodology described next.

- 3.3 Evaluation Framework

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Three Scoring Dimensions

- 1. Reasoning Quality
- 2. Temporal Consistency
- 3. Visual Aesthetics

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

prompt

Input Image

video 1 video 2

###### Evaluation Protocols

Point-Wise Scoring (Per Video)

Pair-Wise Preference (Two Videos)

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

video 1 video 2

video 1 video 2

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

sr(v1) sc(v1) sa(v1)

sr(v2)sc(v2)sa(v2)

[Figure 113]

Compare Across Three Dimensions

0.4 0.3 0.3

0.4 0.3 0.3

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Win Win Tie

S(v1)

S(v2)

As shown in Figure 3, WorldReasonBench evaluates reasoning with two complementary components. Process-aware Reasoning Verification uses structured QA to check both outcome correctness and process faithfulness, while Multi-dimensional Quality Assessment scores each video on reasoning quality, temporal consistency, and visual aesthetics. Together, they provide binary-verifiable diagnostic signals and continuous quality scores for ranking, reward-model training, and human-alignment analysis.

#### 3.3.1 Process-aware Reasoning Verification

This component checks whether a generated video reaches the correct final state along a plausible world-state transition, using a two-stage structured QA protocol: a VLM answers each video-grounded question from visible evidence, then a separate LLM judge assigns a binary score against the ground truth.

Reasoning verification chain. Each test case has multiple QA pairs across four question types, which we map to complementary reasoning phases: factual (initial or final state content), temporal (event order), detail (fine-grained visual fidelity), and reasoning (causal or physical mechanisms). The corresponding phase scores sstate,sproc,sfidel,smech are mean binary accuracies within each type, and overall accuracy is AccQA = N1 Ni=1 ⊮[ˆyi = yigt].

Reasoning gap and process-aware score. To expose outcome hacking—videos that look correct in static frames but fail dynamically—we contrast static outcome performance sout = (sstate +sfidel)/2 with dynamic performance sdyn = (sproc + smech)/2 and define the reasoning gap ∆RG = sout − sdyn; a large positive ∆RG signals strong static appearance but weak process reasoning. For the headline metric we use ScorePR = Acc0QA.8 · s0dyn.2 , which keeps QA accuracy interpretable while discounting models that succeed mainly on static questions, and we use sdyn/AccQA as a processcompleteness diagnostic. Auxiliary metrics are in Appendix D.1.

- Table 2: Main evaluation results across WorldReasonBench dimensions. Per-dimension ScorePR and S(v) (0–100) computed for every generator on a shared evaluation set for fully controlled crossmodel comparison. Bold/underline: best/second-best across all 11 models. Full subcategory results, 95% bootstrap CIs, and additional open-source coverage are in Appendix Table 14 and Appendix N.

Closed-Source Models Open-Source Models Dimension Metric Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video World Knowledge

ScorePR 36.9 42.2 35.2 43.2 55.0 15.6 22.9 13.8 21.6 15.2 13.3 S(v) 62.6 72.0 61.8 70.4 80.1 35.1 39.4 29.4 37.7 40.8 35.1

ScorePR 44.7 32.5 34.5 35.9 35.1 19.3 14.5 15.8 8.1 22.2 22.8 S(v) 76.7 87.2 64.2 83.9 77.2 27.8 38.1 37.2 35.3 30.8 42.8

Human-Centric

ScorePR 25.9 22.4 26.2 31.7 25.7 11.9 16.4 11.2 12.7 7.1 12.6 S(v) 43.0 37.3 42.3 56.7 31.5 24.7 19.5 14.4 19.8 26.7 16.3

Logic Reasoning

ScorePR 37.3 35.7 35.5 47.6 28.6 22.7 15.0 17.3 24.2 24.7 22.8 S(v) 58.0 48.8 42.6 42.5 47.2 25.8 30.5 16.0 22.5 26.1 20.2

Information-Based

ScorePR 34.3 32.7 32.4 39.8 35.3 16.8 17.5 14.4 17.9 16.9 17.4 S(v) 56.9 55.4 50.3 59.4 54.8 28.1 30.0 21.3 27.0 30.5 25.3

Overall

#### 3.3.2 Multi-dimensional Quality Assessment

Reward-model training, model ranking, and human-alignment analysis all need continuous calibrated per-video scores. Multi-dimensional Quality Assessment asks a VLM judge to rate each video on a 1–5 scale along three interpretable dimensions: Reasoning Quality (sr, whether the intended world-state transition is realized), Temporal Consistency (sc, coherence and stability across time), and Visual Aesthetics (sa, frame stability, motion naturalness, composition, and overall appeal). The three are aggregated into S(v) = 0.4sr(v) + 0.3sc(v) + 0.3sa(v), with the largest weight on reasoning quality to match both the benchmark’s focus and the WorldRewardBench annotation protocol (Section 3.2) for direct human-vs-automatic comparability.

Evaluation protocols. We report two complementary protocols. In the point-wise protocol, the judge scores each video independently and pairwise preferences are induced from S(vi) vs. S(vj) with a tie threshold of 0.1, supporting reward-model training and score-based ranking. In the pairwise protocol, the judge compares two videos in a single call and emits A wins / B wins / tie, giving a stronger ordinal signal for preference recovery and judge calibration at the cost of per-video continuous scores.

### 4 Experiments

- 4.1 Experimental Setup

Evaluation settings. We evaluate eleven video generators: five closed-source systems (Sora2, Kling, Wan2.6, Seedance2.0, Veo3.1-Fast) and six open-source models (LTX2.3, Wan2.2-14B, UniVideo, HunyuanVideo-1.5, Cosmos-Predict2.5, LongCat-Video). All automatic evaluation uses Qwen3.5-27B [21]; the QA pipeline enables extended thinking for video question answering, disables it for binary judging, and processes videos at 4 FPS.

Metrics. We report ScorePR as the headline metric for Process-aware Reasoning Verification, with AccQA, phase scores, process completeness, and ∆RG as diagnostics. Multi-dimensional Quality Assessment reports the weighted per-video score S(v) over reasoning quality, temporal consistency, and visual aesthetics, and uses pairwise agreement and Spearman ρ for reward-model alignment. Auxiliary process-aware metrics are defined in Appendix D.1.

Reward-model baselines. On WorldRewardBench, we evaluate five reward/judge models (GPT-

- 5.4, Gemini-3.1, Qwen3.5-9B, Qwen3.5-27B, and our method) under both pair-wise and point-wise protocols to measure recovery of human video preferences.

- 4.2 Generator Performance on WorldReasonBench

Closed-source models lead by a robust factor on both reasoning and quality. Under controlled cross-model comparison (Table 2), closed-source generators sit at 32.4–39.8 overall ScorePR and 50.3–59.4 on S(v), while open-source generators stay at 14.4–17.9 and 21.3–30.5, respectively—a

roughly two-fold gap on both axes, with no open-source 95% CI overlapping any closed-source one. Even the strongest system (Seedance2.0, ScorePR=39.8) sits well below saturation, so today’s most capable generators remain incomplete world models. The gap is not driven by raw visual fidelity: the process-completeness ratio sdyn/AccQA in Section 4.3 shows that open-source failures concentrate on dynamic-phase reasoning rather than static appearance.

Difficulty is dominated by Logic Reasoning and Information-Based categories. Performance is highly uneven across dimensions. Logic Reasoning is the hardest: the best closed-source ScorePR is only 31.7 (Seedance2.0), and five of the six open-source models score below 14. InformationBased is second hardest, with per-subcategory residuals (Appendix Table 14) concentrating in World Mechanics, Material Change, and Data Reading—categories needing physically-grounded transitions or exact text/data preservation. World Knowledge and Human-Centric exceed 35 for every closedsource model and reach 55.0 (Veo3.1-Fast on WK) and 44.7 (Sora2 on HC), so the bottleneck is mechanism- and information-level reasoning rather than visual recognition.

Hint gain is larger for open-source models. With explicit transition hints, every open-source model gains 9.9–14.8 absolute QA points (+56– 85% relative), whereas Sora2-8s—the only closed-source system run under both regimesgains only +10.3 points (+29%) (Table 3). This indicates open-source generators rely more on prompt-side guidance, though ceiling effects, prompt-length sensitivity, and instructionfollowing gaps may also contribute; the substantive outcome-vs-process attribution is carried by ScorePR and sdyn/AccQA in Section 4.3.

Table 3: Reasoning assistance benefit. QA accuracy under implicit (Diff.) vs. hinted (Easy) prompts; hint gain is absolute / relative.

Model Diff. Easy Hint gain Sora2-8s 35.1 45.4 +10.3 / +29.2% LTX2.3 17.5 32.3 +14.8 / +84.9% Wan2.2-14B 21.6 35.2 +13.6 / +63.2% UniVideo 17.8 27.6 +9.9 / +55.5% HunyuanVideo-1.5 20.8 33.0 +12.2 / +58.4% Cosmos-Predict2.5 19.4 30.8 +11.4 / +58.9%

#### Statistical significance and rank stability.

We compute 95% bootstrap confidence intervals (B=2000, case-level resampling with replacement) for ScorePR, AccQA, and S(v) at overall and per-dimension level on the shared evaluation set behind Table 2. The closed-vs.-open separation is statistically robust: every open-source overall-ScorePR CI lies strictly below every closed-source CI (open-source upper bound ≤ 23.1 vs. closed-source lower bound ≥ 26.4). Joint rank bootstrap shows that the two tiers never swap, and Seedance2.0 has a clearly favoured rank inside the closed tier (modal rank 1 in 89.3% of bootstraps, 95% rank interval [1,2]); the other five closed-source models share rank slots with overlapping CIs, so we report their cluster rather than a strict ordering. Within open-source, UniVideo is the only generator with a tightly concentrated rank (modal rank 12 in 69.7%); the remaining five sit in slots [7,11] as a tied cluster. Full per-model CIs, per-dimension CIs, and the rank-distribution table are reported in Appendix N.

#### 4.3 Validating Process-aware Metrics against Human Preferences

Process-aware QA metrics outperform pairwise VLM judges on human alignment. Using ∼6K expert preference pairs from WorldRewardBench, we fit a Bradley-Terry model with Davidson ties (Appendix G) for a Human Elo ranking and compare it with three automatic rankings (Table 4). ScorePR and AccQA reach Spearman ρ=0.955 and 0.927, both well above the pairwise VLM-judge Elo (ρ=0.804). The process-completeness ratio sdyn/AccQA stays at 0.71–0.91 for closed-source vs. 0.54–0.63 for open-source, attributing the open-source deficit to dynamic reasoning rather than static-frame errors.

Diagnosing the residual judge–human disagreement. The largest remaining inconsistency in Table 4 is the closed-source ordering: humans place Seedance2.0 first, but the pairwise judge places Sora2-8s and Sora2-12s on top. We trace this to two pairwise-protocol effects. (i) The judge consumes a fixed budget of 8 frames per video, so 8s/12s Sora2 clips expose more events at lower temporal density and the judge often reads this as richer reasoning evidence; Figure 4 shows cases where Seedance2.0 instead produces smoother, more physically faithful motion that humans reward but the fixed-frame judge misses. (ii) Judge accuracy drops sharply on close pairs (89.1% when the human gap is > 1.5, 47.5% for ≤ 0.5), and such close pairs disproportionately involve Seedance2.0

- Table 4: Human-aligned ranking and metric validation. Models ordered by Human Elo; |∆r| is the absolute rank displacement from the human ranking. Horizontal line separates closed- and open-source models.

Human Rank

Human Elo

Judge Elo

Judge Rank

AccQA

ScorePR

sdyn/ AccQA

Model

(%) |∆r|

(%) |∆r|

- 1 Seedance2.0 1471 1183 3 41.2 0 39.8 0 0.84
- 2 Veo3.1-Fast 1253 1151 4 36.0 0 35.3 0 0.91
- 3 Kling 1240 1142 5 34.0 2 32.7 1 0.82
- 4 Wan2.6 1211 1130 6 34.7 0 32.4 1 0.71
- 5 Sora2-8s 1118 1222 1 35.3 2 34.3 2 0.86
- 6 Sora2-12s 1109 1217 2 33.5 0 32.4 0 0.84

- 7 Wan2.2-14B 953 913 7 19.6 2 17.5 1 0.57
- 8 HunyuanVideo-1.5 911 841 9 20.2 1 17.9 1 0.56
- 9 LongCat-Video 904 876 8 19.7 1 17.4 0 0.54
- 10 UniVideo 665 737 11 16.2 1 14.4 1 0.56
- 11 LTX2.3 587 802 10 18.5 1 16.8 1 0.63

Prompt: There are two domino chains that have already started to fall; show how both

###### (a). Input Image: groups topple.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Veo-3.1

###### (b). Input Image: Prompt: What will happen if the person in the picture succeeds in playing the game?

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Seedance 2.0

[Figure 134]

[Figure 135]

The grabbing machine is not the one being controlled.

Prompt: When the switch is closed, there is a movable wire above the vertical bracket on the right side. Which way will it rotate?

###### (c). Input Image:

The hand does not flip the switch.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Seedance 2.0

[Figure 146]

The wire does not move; instead, unrelated objects move back and forth.

Figure 4: Qualitative comparison on representative reasoning cases. Visually plausible generations can still fail process-level world reasoning, while higher-scoring models better preserve the intended state transition and temporal dynamics.

against the Sora2 family, suppressing Seedance2.0’s Elo. ScorePR avoids this duration mismatch and matches the human ordering up to a single one-rank swap.

#### 4.4 WorldRewardBench: VLM Judges as Reward Models

We evaluate whether the Multi-dimensional Quality Assessment protocol can also serve as an automatic reward model (Table 5). Pair-wise judging directly compares two candidate videos; pointwise scoring induces preferences from the aggregate S(v). Subcategory-level results, model settings, and parsing statistics are in Appendices L–K.

Pair-wise wins on agreement; point-wise wins on calibration. The strongest pair-wise judge is Qwen3.5-9B-Thinking (74.35% w/o ties), with Qwen3.5-27B-Thinking close behind (73.05%) and both ahead of every point-wise variant; Qwen3.5-9B-Thinking also has the top point-wise ρ=0.655 (27B-Thinking 0.626). The two protocols are therefore complementary: pair-wise is preferable for selecting the better video among close candidates, point-wise gives calibrated per-video signals suitable for reward-model training. Gemini-3.1 lags pair-wise by >10pp despite competitive pointwise scores, so explicit comparison at the prompt level matters as much as raw judging capacity. The Information-Based bottleneck transfers from generators to judges: pair-wise agreement drops

- Table 5: Reward-model alignment on WorldRewardBench. Pair-wise: agreement (%) w/ Ties / w/o Ties; point-wise: induced pairwise accuracy / Spearman ρ. GPT-5.4 pair-wise uses 5,947/5,969 pairs after Azure OpenAI safety filtering (0.37% refused; details in Appendix K).

Closed-Source Models Open-Source Models Dimension Protocol / Metric GPT-5.4 Gemini-3.1-Flash

Qwen3.5-27B Thinking (4FPS) Frames Used 8 1FPS ∼10 ∼10 ∼10 4FPS World Knowledge

Qwen3.5-9B Thinking

Qwen3.5-27B Instruct

Qwen3.5-27B Thinking

Pair w/ / w/o 60.77 / 67.84 51.50 / 60.44 70.81 / 76.19 69.37 / 74.16 69.94 / 74.64 69.51 / 74.23 Point Acc / ρ 54.55 / 0.592 59.86 / 0.582 60.70 / 0.720 54.01 / 0.658 60.57 / 0.687 62.09 / 0.711

Pair w/ / w/o 68.37 / 76.80 58.22 / 66.27 71.71 / 77.52 71.25 / 76.05 72.61 / 77.81 69.08 / 74.41 Point Acc / ρ 59.14 / 0.626 60.06 / 0.675 59.54 / 0.702 55.94 / 0.682 62.81 / 0.713 60.49 / 0.703

Human-Centric

Pair w/ / w/o 67.41 / 78.43 58.23 / 67.68 69.33 / 77.13 68.46 / 74.51 70.16 / 76.23 68.53 / 74.97 Point Acc / ρ 53.42 / 0.523 57.65 / 0.562 57.50 / 0.617 55.71 / 0.573 60.17 / 0.606 58.40 / 0.597

Logic Reasoning

Pair w/ / w/o 56.95 / 63.68 50.21 / 58.10 52.45 / 61.76 60.44 / 65.22 60.24 / 65.32 61.50 / 66.39 Point Acc / ρ 48.15 / 0.484 47.89 / 0.432 53.59 / 0.471 47.95 / 0.408 50.15 / 0.445 52.41 / 0.526

Information-Based

Pair w/ / w/o 63.04 / 71.36 54.39 / 62.99 67.14 / 74.35 66.89 / 72.07 67.74 / 73.05 66.90 / 72.30 Point Acc / ρ 53.43 / 0.565 55.84 / 0.568 57.76 / 0.655 53.15 / 0.591 57.85 / 0.626 57.83 / 0.644

Overall

from 74–78% on the other dimensions to 58–65%, and point-wise ρ from 0.6–0.7 to 0.4–0.5, making Information-Based the most discriminative dimension for future reward models.

#### 4.5 Ablation Studies

Point-wise protocol and frame rate. Vanilla single-call point-wise scoring is both more efficient and at least as effective as Sequential Dimension Evaluation (SDE), reaching the best ρ=0.626 and 67.63% w/o-ties accuracy with one judge call versus three for SDE. The frame-rate ablation in Appendix Tables 23–24 shows 4 FPS gives the best cost–accuracy trade-off (37.2% vs. 37.6%

- at 8 FPS, with ∼9k vs. ∼12k visual tokens per 5s video; 34.9% at 2 FPS). We therefore default to vanilla point-wise scoring at 4 FPS; full tables and halo-effect analysis are in Appendix F.

Weight design and sensitivity. Since sproc and smech already enter AccQA as one quarter each, the s0dyn.2 term in ScorePR = Acc0QA.8 ·s0dyn.2 acts as a second-order penalty on outcome-hacking rather than a substitute for outcome accuracy. Re-ranking the eleven WorldRewardBench models in Table 4 under α ∈ {0,0.2,0.5,0.7,0.8,0.9,1}, arithmetic / geometric / min aggregators, and a 231-point simplex grid over (wr,wc,wa) keeps Spearman ρ vs. human Elo in [0.83,0.96] for ScorePR and ρ ≥ 0.95 on 67.5% of the S(v) simplex; the paper exponent α=0.8 in fact attains the highest ρ=0.955, so the chosen weights are an empirical optimum rather than a compromise. Full grids in Appendix M.

Cross-family judge robustness. We cross-compare three judge families on the same WorldRewardBench pairs (Table 5). Within Qwen3.5, scaling 9B→27B and toggling extended thinking moves overall pair-wise w/o-ties agreement by at most 2.3pp (72.07–74.35%) and point-wise ρ by at most 0.064 (0.591–0.655). Across families, Gemini-3.1-Flash trails Qwen on pair-wise agreement by ∼10pp while tracking it point-wise (ρ=0.568); GPT-5.4 sits between, with 71.36% pair-wise agreement and ρ=0.565 matching Gemini. All three families flag Information-Based as the hardest category and recover the same closed-vs-open ordering as Table 4, so the reasoning gap and Information-Based bottleneck are not artefacts of a single judge family.

### 5 Conclusion

We introduced WorldReasonBench, a world-state prediction benchmark with 436 cases and structured QA annotations spanning four reasoning dimensions and 22 subcategories, together with WorldRewardBench, a preference benchmark with approximately 6K expert-annotated pairs over 1.4K videos from 11 generators. Building on this data, we proposed a two-part evaluation methodologyProcess-aware Reasoning Verification and Multi-dimensional Quality Assessment—and validated it directly against human Elo, where ScorePR reaches Spearman ρ=0.955 and clearly outperforms a pairwise VLM judge. The results expose a persistent gap between visual plausibility and world reasoning: closed- and open-source generators differ by roughly a factor of two on both reasoning and quality, and the process-completeness ratio sdyn/AccQA attributes the open-source deficit to

dynamic-phase failures rather than static appearance. Logic Reasoning and Information-Based content remain the most challenging dimensions for both generators and judges, suggesting that progress on world-aware video generation will be driven less by visual polish and more by mechanism-level reasoning and information preservation. We release WorldReasonBench, WorldRewardBench, and the evaluation toolkit so that the community can audit reward models, calibrate new judges, and extend the reasoning taxonomy as video generators continue to evolve.

### References

- [1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. 2024. URL https://openai. com/research/video-generation-models-as-world-simulators, 3(1):3, 2024.
- [2] Roger R Davidson. On extending the bradley-terry model to accommodate ties in paired comparison experiments. Journal of the American Statistical Association, 65(329):317–328, 1970.
- [3] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [4] Xuan He, Dongfu Jiang, Ping Nie, Minghao Liu, Zhengxuan Jiang, Mingyi Su, Wentao Ma, Junru Lin, Chun Ye, Yi Lu, et al. Videoscore2: Think before you score in generative video evaluation. arXiv preprint arXiv:2509.22799, 2025.
- [5] Xuming He, Zehao Fan, Hengjia Li, Fan Zhuo, Hankun Xu, Senlin Cheng, Di Weng, Haifeng Liu, Can Ye, and Boxi Wu. Ruler-bench: Probing rule-based reasoning abilities of next-level video generation models for vision foundation intelligence. arXiv preprint arXiv:2512.02622, 2025.
- [6] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [7] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [8] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024.
- [9] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [10] Yifan Li, Yukai Gu, Yingqian Min, Zikang Liu, Yifan Du, Kun Zhou, Min Yang, Wayne Xin Zhao, and Minghui Qiu. Viper: Process-aware evaluation for generative video reasoning. arXiv preprint arXiv:2512.24952, 2025.
- [11] Xinxin Liu, Zhaopan Xu, Ming Li, Kai Wang, Yong Jae Lee, and Yuzhang Shang. Can world simulators reason? gen-vire: A generative visual reasoning benchmark. arXiv preprint arXiv:2511.13853, 2025.
- [12] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22139–22149, 2024.
- [13] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation. Advances in Neural Information Processing Systems, 36:62352–62387, 2023.

- [14] Yang Luo, Xuanlei Zhao, Baijiong Lin, Lingting Zhu, Liyao Tang, Yuqi Liu, Ying-Cong Chen, Shengju Qian, Xin Wang, and Yang You. V-reasonbench: Toward unified reasoning benchmark suite for video generation models. arXiv preprint arXiv:2511.16668, 2025.
- [15] Wentao Ma, Weiming Ren, Yiming Jia, Zhuofeng Li, Ping Nie, Ge Zhang, and Wenhu Chen. Videoeval-pro: Robust and realistic long video understanding evaluation. arXiv preprint arXiv:2505.14640, 2025.
- [16] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsensebased benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.
- [17] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.
- [18] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video generation models as world simulators. arXiv preprint arXiv:2410.18072, 2024.
- [19] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2vcompbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8406–8416, 2025.
- [20] Qwen Team. Qwen3. 5: Accelerating productivity with native multimodal agents, 2026.
- [21] Qwen Team. Qwen3. 5: Towards native multimodal agents. URL: https://qwen. ai/blog, 2026.
- [22] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019.
- [23] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [24] Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, et al. A very big video reasoning suite. arXiv preprint arXiv:2602.20159, 2026.
- [25] Zeqing Wang, Xinyu Wei, Bairui Li, Zhen Guo, Jinrui Zhang, Hongyang Wei, Keze Wang, and Lei Zhang. Videoverse: How far is your t2v generator from a world model? arXiv preprint arXiv:2510.08398, 2025.
- [26] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.
- [27] Keming Wu, Zuhao Yang, Kaichen Zhang, Shizun Wang, Haowei Zhu, Sicong Leng, Zhongyu Yang, Qijie Wang, Sudong Wang, Ziting Wang, et al. Visual generation in the new era: An evolution from atomic mapping to agentic world modeling. arXiv preprint arXiv:2604.28185, 2026.
- [28] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [29] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [30] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

###### [31] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

### A Representative Examples of WorldReason-Bench

To provide a more intuitive understanding of the data distribution and reasoning requirements in WorldReasonBench, we present representative examples from each major category in this appendix. These examples cover the four reasoning categories in our benchmark: World Knowledge, HumanCentric, Logic Reasoning, and Information-Based Reasoning. Each example consists of an input image and a generation prompt.

World Knowledge

Prompt: Using the input image as the first frame, when the needle in the hand is pressed down firmly, what happens?

Prompt: Using the input image as the first frame, after many days of heavy rain, what is likely to happen in this scene? Demonstrate it.

Prompt: Using the input image as the first frame, this coal has been sitting here for a long time; show the complete process of removing it.

Prompt: Using the input image as the first frame, after the person lets go, what happens?

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Prompt: Using the input image as the first frame, what happens before thunder?

Prompt: Using the input image as the first frame, on the 15th day of the lunar month, from noon to 6 p.m., what does the scene look like?

Prompt: Using the input image as the first frame, after two weeks, what happens?

Prompt: Using the input image as the first frame, the person quickly pulls the tablecloth away; what happens?

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Prompt: Using the input image as the first frame, the band is about to go on stage; how do the stage lights change?

Prompt: Using the input

Prompt: Using the input image as the first frame, after an earthquake, what is likely to happen in this scene? Demonstrate it.

Prompt: Using the input image as the first frame, generate the traditional Egyptian process of treating the dead as in the image.

image as the first frame, the subway door warning lights are on; what happens next?

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Prompt: Using the input image as the first frame, after the chameleon on the branch crawls onto the green leaves, what happens?

Prompt: Using the input image as the first frame, if this year is 2024, what date will the calendar show after today passes?

Prompt: Using the input image as the first frame, when the hand flicks the paper away quickly, what happens?

Prompt: Using the input image as the first frame, generate the process of correctly re-hanging the couplets in the image that were not posted properly.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Prompt: Using the input image as the first frame, if this action continues, what happens?

Prompt: Using the input image as the first frame, show how the objects in the image change over one week; give the complete process.

Prompt: Using the input image as the first frame, the person rubs their hair against the object in their hand repeatedly; what happens?

Prompt: Using the input image as the first frame, Christmas is coming soon; how will the home change?

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

- Table 6: Representative examples from the World Knowledge category. These cases cover material change, public systems, world mechanics, cultural life, everyday living, earth cycles, and the living world, testing whether models can generate plausible state transitions grounded in physical, social, cultural, and natural-world knowledge.

Human-Centric

Prompt: Use the input image as the first frame, what will happen?

Prompt: Use the input image as the first frame. What will happen if someone cuts in line?

Prompt: Use the input image as the first frame. What is wrong with the behavior of the characters in the picture? How should this object be conveyed?

Prompt: Use the input image as the first frame, what will happen next?

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Prompt: Use the input image as the first frame. In the picture, the man walking toward you finds that the woman’s wallet has fallen. What will happen?

Prompt: Use the input image as the first frame. What will happen if the character holding an umbrella finds a passerby holding some documents that are easily wet?

Prompt: Use the input image as the first frame. What will happen if the man discovers that the woman is struggling to move boxes?

Prompt: Use the input image as the first frame. As time goes by, what will happen in this scene?

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Prompt: Use the input image

Prompt: Use the input image as the first frame, what will happen? How to avoid this phenomenon from happening?

Prompt: Use the input image as the first frame, mark the wrong behavior of the characters in the picture, and show the correct behavior of the characters in the picture in the current situation.

Prompt: Use the input image as the first frame. What will happen if the lights suddenly brighten and a harsh sound occurs?

as the first frame. What will the characters in the picture do next?

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Prompt: Use the input image as the first frame. The character in the picture wants to wash his car. The car wash shop is 50m away. Give the process of the character in the picture going to the car wash shop.

Prompt: Use the input image as the first frame. The character in the picture cannot be seen clearly. What will he do to improve it?

Prompt: Use the input image as the first frame. The character in the picture cannot see the text in the book clearly. What will he do to make it clearer?

Prompt: Use the input image as the first frame. This box is too high. How should the characters in the picture use tools to get to the box?

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Prompt: Use the input image as the first frame, The barista is flattening the coffee powder, what series of actions will he do next?

Prompt: Use the input image as the first frame, this man is driving nails, what will happen?

Prompt: Use the input image

Prompt: Use the input image as the first frame, What happens if the internal temperature spikes

as the first frame, What happens when this man moves his hands?

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Prompt: Use the input image

Prompt: Use the input image as the first frame, What will happen if the person in the picture succeeds in playing the game?

Prompt: Use the input image as the first frame, what will happen next?

Prompt: Use the input image as the first frame, Please take the entered picture as the first frame. What will the characters in the picture do after the flight is cancelled?

as the first frame, a swimmer in an intense swim, what will happen

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

- Table 7: Representative examples from the Human-Centric category. These cases cover object handling, social scenes, skilled actions, personal routines, and public conduct, focusing on whether models can generate plausible human behaviors, interactions, and object-centered actions over time.

Logic Reasoning

Prompt: Use the input image as the first frame. The sodium block has been put into the ethanol solution containing phenolphthalein, and the whole process of this chemical experiment is given.

Prompt: Use the input image as the first frame, give the follow-up results of this experiment

Prompt: Use the input image as the first frame, The picture shows the flame reaction of various metals, showing the subsequent process of this experiment.

Prompt: Use the input image as the first frame, Pull all pendulum balls to the same side and release them at the same time as much as possible, allowing them to swing freely. over time

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Prompt: Use the input image as the first frame, move the magnet downward, and give the subsequent results of this experiment.

Prompt: Use the input image as the first frame. Start with a 2D atomic, electron point, or structural diagram of methane and generate a smooth scientific animation that transforms them into a clear 3D geometric model of methane.

Prompt: Use the input image as the first frame, The expanded picture above can be folded into a cube. Which cube below is the correct option? , and the expansion of the correct options is given as the process of unfolding the graph

Prompt: Use the input image as the first frame, Please count how many squares there are in the picture, show them from different perspectives, and count them one by one

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Prompt: Use the input image as the first frame and draw a line to bisect these five circles.

Prompt: Use the input image as the first frame. Please continue to solve the problem according to the existing process and give the problem-solving process.

Prompt: Use the input image as the first frame, Please use the tools in the picture to complete the experiment of photosynthesis producing oxygen.

Prompt: Use the input image as the first frame to solve the problem and give the solution process.

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Prompt: Use the input image as the first frame. Please continue to solve the problem according to the existing process.

Prompt: Use the input image as the first frame, solve the two questions in the picture, and give the solution process

Prompt: Use the input image as the first frame, please determine whether the second and third graphics have Euler paths, and if so, give their Euler paths.

Prompt: Use the input image as the first frame to solve the problem and give the solution process.

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Prompt: Use the input image as the first frame to solve the problem and give the solution process.

Prompt: Use the input image as the first frame, answer the questions in the picture one by one, and give the problem-solving process

Prompt: Use the input image as the first frame. Please continue to solve the problem according to the existing process and give the problem-solving process.

Prompt: Use the input image

as the first frame, select the correct option, and give the problem-solving process

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Prompt: Use the input image as the first frame. Can this picture be drawn in one stroke? If you can, find the start and end points and try to draw them.

Prompt: Use the input image as the first frame, find the pattern, and give what the number at the question mark should be?

Prompt: Use the input image as the first frame, complete the questions in the picture, and give the problem-solving process.

Prompt: Use the input image

- as the first frame, find the pattern, and give what letters
- at the question mark should be?

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

- Table 8: Representative examples from the Logic Reasoning category. These cases cover quantitative math, spatial geometry, experimental science, logic puzzles, and pattern discovery, evaluating whether models can maintain rule-based, spatial, symbolic, and causal relationships across generated frames.

Information-Based Reasoning

Prompt: Based on the "Attention is all you need" paper and the picture, a subsequent LLM technology overview video is generated, including an introduction to the principles of Word Embedding and Self-attention methods.

Prompt: Generate a video that uses visual metaphors to explain the evolution of Nietzsche’s philosophical concept of the “Superman”)

Prompt: Generate a video explaining the discovery process, evidence, and specific content of the continental drift theory, etc.

Prompt: Based on the book "1984" and the picture, generate a summary of 1984 George Orwell’s book

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Prompt: I have questions about the input chart. Please help explain and answer them in a clear and easy-to-understand way. The questions are as follows:Which activity in the

Prompt: I have questions about the input chart. Please help explain and answer them in a clear and easy-to-understand way. The questions are as follows:What is the total cumulative BRI engagement in USD million for the year 2025, as shown in the rightmost chart?Which sector had the largest investment in USD million in 2025, according to the left chart?Which year, as shown in the rightmost chart, had the highest cumulative BRI engagement under the

Prompt: I have questions about the input chart. Please help explain and answer them in a clear and easy-to-understand way. The questions are as follows:How many genes are labeled as

Prompt: I have questions about the input chart. Please help explain and answer them in a clear and easy-to-understand way. The questions are as follows:What is the net margin for the

’Pineville Lindseys’ location?Which location has the highest customer conversion rate shown in the chart?What is the total net sales for the ’North Canton Fashions Direct’ location, combining Men’s, Women’s, and Kids’ Products?

’Up regulated’ in the chart?What is the log2(Fold Change) value for the gene BTNL3?What is the total number of genes categorized as ’Not sig’?

’Informal’ category has the lowest percentage of children doing it ’at least once a week’?What is the difference in percentage between ’Reading long texts’ and ’Writing long texts’ in the ’Advanced’ category?What is

the percentage of children who ’Did not occur / not relevant’ for ’Writing short texts’ in the ’Basic’ category?

’Investment’ category?

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

Prompt: Generate a process for adjusting the order of the small policies most important to faculty and staff in the diagram to the first small policy

Prompt: Generate the process of sorting bubbles or color blocks of different shapes in the picture into categories and arrangements by shape

Prompt: Generate the process of changing dot plot to histogram

Prompt: The process of generating and enlarging the data labels of the time period in which the push-to-weight ratio value increases fastest in the figure

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Prompt: Use the input image as the first frame to generate the process of wheat grains turning into wheat again.

Prompt: Use the input image as the first frame and show the process of rainwater circulating on the earth after falling into the lake.

Prompt: Please give the evolution process of the fossils in the picture

Prompt: This is the final effect of makeup. Please give the makeup process from no makeup to full makeup.

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Prompt: Use the input image as the first frame. The picture is the egg of a blue-spotted disc. Please give the final process of turning it into a completed butterfly.

Prompt: The rising steam transforms into beating golden notes, expressing the rhythm of the coffee aroma.

Prompt: The red dust kicked up by the tires condenses into the Tesla brand T logo in the air

Prompt: The traces of lipstick are transformed into pieces of real velvet roses, covering the entire picture

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

- Table 9: Representative examples from the Information-Based Reasoning category. These cases involve data reading, process timelines, visual editing, knowledge media, and creative expression, testing whether models can preserve explicit information, temporal order, and semantic consistency during video generation.

### B Evaluation Prompts

This section lists all prompts used in the two evaluation components. Each prompt is shown in a colored box corresponding to its component: blue for Process-aware Reasoning Verification (QA pipeline), green for Multi-dimensional Quality Assessment (point-wise scoring), and purple for pair-wise comparison.

- B.1 Process-aware Reasoning Verification Prompts

The Process-aware Reasoning Verification prompts implement the two-stage QA pipeline used for the main reasoning metric. Stage 1 asks the VLM to answer a video-grounded question using only visible evidence, and Stage 2 converts the free-form answer into a binary correctness label against the ground truth.

|You are a precise and conservative video QA assistant. Answer only from directly visible video content. Do not infer or guess.<br><br>→ You may briefly analyze first, but the response must end with Final JSON: followed by one valid JSON object.<br><br>Sample metadata: {<br><br>"id": "<case_id>", "category": "<dimension>", "sub_category": "<subcategory>", "video_prompt": "<generation prompt>", "question": {<br><br>"question": "<evaluation question>", "question_type": "<factual|temporal|detail|reasoning>", "difficulty": "<easy|medium|hard>"<br><br>}<br><br>} Answer ONLY using facts that are directly visible in the video frames. Do NOT infer, speculate, guess, or complete missing information. Output format: Final JSON: {"answer": "<short answer grounded in visible evidence>"}<br><br>|
|---|

Stage 1: Video Question Answering

Table 10: Stage 1 prompt for Process-aware Reasoning Verification. The VLM watches the video and answers one evaluation question grounded in visible evidence.

- B.2 Multi-dimensional Quality Assessment Prompts

The Multi-dimensional Quality Assessment prompts support both point-wise scoring and pair-wise preference comparison. Point-wise scoring produces calibrated per-video scores on reasoning quality, temporal consistency, and visual aesthetics, while pair-wise comparison directly estimates human preference between two candidate videos.

### C Reasoning Taxonomy Details

This section provides the detailed interpretation of the 22 subcategories used in WorldReasonBench. The concise taxonomy in Section 3.1 is used for main-text reporting, while the descriptions below specify the intended scope of each subcategory.

World Knowledge. This dimension tests whether generated videos respect established knowledge about how the physical, social, and cultural world evolves. World Knowledge contains 127 cases in the full set and 32 cases in the representative subset. It includes Material Change (23/5 cases; fluids, heat, optics, and observable physical transformations), Public Systems (21/2 cases; traffic, logistics, civic infrastructure, and rule-governed public procedures), World Mechanics (20/8 cases; motion, impact, force, balance, and statics), Cultural Life (19/3 cases; rituals, performance, architecture, and artistic conventions), Everyday Living (19/5 cases; household devices, food preparation, and routine

Stage 2: Binary Answer Judging

|You are a strict evaluator for video QA. You must score this single QA as binary: 1 or 0. You may briefly analyze first, but keep it short.<br><br>Scoring rule: − score = 1 only if the predicted answer is sufficiently correct according to BOTH the ground truth and the evaluation criteria. − score = 0 if it is incorrect, incomplete on the key point, contradicts the criteria, or unsupported. − Be strict. Do not give partial credit. − If the predicted answer says ’unclear from the video’, give 1 only if the criteria truly cannot be verified; otherwise give 0.<br><br>Question type: <factual|temporal|detail|reasoning> Type−specific judging focus: − factual: Judge whether the predicted answer correctly describes the relevant visible fact, state, object, or outcome. − detail: Judge whether the predicted answer correctly captures the specific required detail (color, count, text, position, identity). − temporal: Judge whether the predicted answer correctly captures event order, progression, or before/after relations. − reasoning: Judge whether the predicted answer correctly captures causality, physical rule, or mechanism.<br><br>Question: <question> Ground truth: <expected_answer> Evaluation criteria: <criteria> Difficulty: <easy|medium|hard> Predicted answer: <model_answer><br><br>Output format: Final JSON: {"score": 0_or_1, "reason": "<brief justification>", "verdict": "correct_or_incorrect"}<br><br>|
|---|

- Table 11: Stage 2 prompt for Process-aware Reasoning Verification. A separate LLM judge compares the predicted answer against the ground truth and assigns a binary score.

daily activities), Earth Cycles (17/6 cases; weather, astronomy, and large-scale periodic processes), and Living World (8/3 cases; animal behavior and ecological change), where each pair denotes full-set / representative-subset counts.

Human-Centric Reasoning. This dimension targets scenes where humans are central actors and where correct generation requires plausible behavior, interaction, and motion. Human-Centric Reasoning contains 78 cases in the full set and 31 cases in the representative subset. It includes Object Handling (27/9 cases; tool use, affordances, interface control, and everyday manipulation), Social Scenes (15/6 cases; conversation, helping, service interaction, and group events), Skilled Action (13/3 cases; sports, motor coordination, and craft-like operation), Personal Routine (13/4 cases; sleep, grooming, comfort management, eating, and self-directed daily behavior), and Public Conduct (10/9 cases; norm-following behavior in public spaces, mobility, and risk-aware correction),

Logic Reasoning. This dimension evaluates whether generated videos can preserve structured logical relations over time. Logic Reasoning contains 131 cases in the full set and 35 cases in the representative subset. It includes Quantitative Math (38/11 cases; symbolic calculation, algebra, calculus, and quantitative science), Spatial Geometry (38/10 cases; spatial transformations, geometric structure, and 3D reasoning), Experimental Science (32/8 cases; scientific procedures and controlled cause-effect demonstrations), Logic Puzzles (12/3 cases; constraint satisfaction and rule-governed problem solving), and Pattern Discovery (11/3 cases; sequence completion, structural analogy, and pattern induction).

Information-Based Reasoning. This dimension covers cases with explicit information such as text, numbers, charts, diagrams, and data visualizations that must be preserved or transformed faithfully. Information-Based Reasoning contains 100 cases in the full set and 32 cases in the representative subset. It includes Data Reading (29/12 cases; chart, table, and exact value interpretation), Process Timeline (28/8 cases; temporally ordered transformations and stage progression), Visual Editing (18/4 cases; diagrammatic modification and layout-consistent changes), Knowledge Media (16/5 cases; document-, history-, and explainer-style content grounded in external knowledge), and Creative Expression (9/3 cases; metaphorical or stylistically transformed information presentation).

Point-wise Video Scoring

|You are an impartial expert judge for AI−generated image−to−video outputs. The goal is to evaluate whether the generated video<br><br>→ successfully models the world−state transition implied by the input image together with the prompt or instruction.<br><br>All input videos are AI−generated. Any humans appearing in the videos are also AI−generated. You will be given an input image, a text prompt, and the generated video conditioned on them.<br><br>Evaluate the generated video on exactly the following three dimensions, using a score from 1 to 5 for each:<br><br>1. Reasoning Correctness (<br><br>1 indicates that the video fails to model the intended world−state transition from the input image, violates the required<br><br>→ causal / physical / logical relation, or clearly misunderstands the instruction.<br><br>3 indicates that the video captures part of the intended transition from the input image but still contains noticeable reasoning<br><br>→ errors or inconsistencies.<br><br>5 indicates that the video correctly and convincingly evolves the input image into the intended future state with strong causal<br><br>→ , physical, logical, or informational consistency. )<br><br>2. Content Fidelity & Continuity (<br><br>1 indicates that key entities, attributes, or temporal states from the input image are missing, unstable, or incoherent across<br><br>→ frames.<br><br>3 indicates that the core content from the input image is present but continuity is only partially preserved, with visible<br><br>→ instability or abrupt state changes.<br><br>5 indicates that the required content is faithfully preserved from the input image and the temporal presentation is coherent,<br><br>→ stable, and continuous. )<br><br>3. Visual Aesthetics (<br><br><br>1 indicates severe visual defects such as distortion, implausible or distracting motion, poor composition, or low rendering<br><br>→ quality.<br><br>3 indicates acceptable but imperfect visual quality with noticeable artifacts, limited appeal, or temporal presentation that is<br><br>→ only partially convincing.<br><br>5 indicates strong overall visual quality with clean rendering, coherent temporal presentation, and appealing composition.<br><br>→ Motion should be rewarded only when it is necessary, plausible, and supportive of the prompt.<br><br>) Reasoning Correctness is the most important dimension. If visual quality is strong but the implied world dynamics from the<br><br>→ input image are wrong, the video should still receive a low overall assessment.<br><br>Important judging rules: − Do not reward mere motion, camera movement, or visible change by itself. − A static but faithful video can still receive a high score if it better matches the prompt and implied world−state transition. − If a video is represented as sampled frames, interpret them as an ordered sequence from start to end rather than as unrelated<br><br>→ still images. − Reward motion only when it improves correctness, continuity, and prompt fidelity. Input Image: <image> Text Prompt: <prompt> Generated Video: <video> Give your output in the following JSON format: {<br><br>"reasoning": "Briefly discuss the video in terms of Reasoning Correctness, Content Fidelity & Continuity, and Visual<br><br>→ Aesthetics, then give an overall judgement.",<br><br>"score": [reasoning_correctness_score, content_fidelity_continuity_score, visual_aesthetics_score]<br><br>} Output only valid JSON.<br><br>|
|---|

- Table 12: Point-wise scoring prompt for Multi-dimensional Quality Assessment. The VLM scores each video on three dimensions (reasoning quality, temporal consistency, visual aesthetics) on a 1–5 scale.

Pair-wise Video Comparison

|Please act as an impartial judge for two AI−generated image−to−video outputs produced from the same input image and prompt<br><br>→ or instruction. You will be given the shared input image, Model A’s generated video, and Model B’s generated video.<br><br>→ Your job is to determine which video is better.<br><br>Evaluate the two videos using exactly the following three dimensions:<br><br>1. Reasoning Correctness Whether the video correctly models the intended world−state transition implied by the input image together with the prompt<br><br>→ or instruction. This includes causal validity, physical plausibility, logical consistency, and correct state evolution<br><br>→ over time.<br><br>2. Content Fidelity & Continuity Whether the required visual content from the input image is faithfully preserved and whether entities, attributes, and temporal<br><br>→ states remain coherent and continuous throughout the video.<br><br>3. Visual Aesthetics Whether the video is visually appealing, well−rendered, naturally animated, and free from obvious artifacts or severe<br><br><br>→ distortions.<br><br>Use the following priority when making the final decision: − Reasoning Correctness is the primary criterion. − Content Fidelity & Continuity is the secondary criterion. − Visual Aesthetics is the tertiary criterion.<br><br>In other words, if one video reasons substantially better about the implied world dynamics from the input image, it should<br><br>→ usually be preferred even if the other video is slightly more attractive visually.<br><br>Important judging rules: − Do not reward mere motion, camera movement, or any visible change by itself. − A static but faithful video can be better than a dynamic but implausible or artifact−heavy one. − If a video is represented as sampled frames, interpret them as an ordered sequence from start to end rather than as unrelated<br><br>→ still images.<br><br>Input Image: <image> Text Prompt: <prompt><br><br>Model A Generated Video: <left_video><br><br>Model B Generated Video: <right_video><br><br><br>Provide a concise comparison covering all three dimensions. After your explanation, output only one final verdict label from the<br><br>→ list below:<br><br>1. Model A is better: [[A>B]]<br><br>2. Model B is better: [[B>A]]<br><br>3. Tie, relatively the same acceptable quality: [[A=B=Good]]<br><br>4. Both are bad: [[A=B=Bad]]<br><br><br>|
|---|

- Table 13: Pair-wise comparison prompt for Multi-dimensional Quality Assessment. The VLM directly compares two candidate videos and outputs a preference verdict.

### D Full Two-Component WorldReasonBench Results

We separate the detailed WorldReasonBench results by evaluation component and top-level reasoning dimension. Process-aware Reasoning Verification tables report outcome QA accuracy (%), while Multi-dimensional Quality Assessment tables expand each sub-category into three raw quality axes on the 1–5 scale.

#### D.1 Auxiliary Process-Aware Metric Definitions

The main text uses ScorePR as the headline Process-aware Reasoning Verification metric because it preserves ranking discriminability while emphasizing temporal and mechanistic correctness. We additionally define two auxiliary diagnostics for ablations and detailed error analysis.

Difficulty-weighted score. Each QA pair is annotated with a difficulty label. To penalize failures on easy questions more heavily while rewarding successes on hard ones, we use an asymmetric weighting scheme:

wd+ if correct, wd− if incorrect,

wi =

(weasy+ ,weasy− ) = (0.8,1.5), (wmed+ ,wmed− ) = (1.0,1.0),

(1)

(whard+ ,whard− ) = (1.5,0.6).

The difficulty-weighted score is Scorewt = i wi · ⊮[ˆyi = yigt] i wi. Bottleneck composite score. The bottleneck composite score emphasizes the short-board effect by penalizing failures in any reasoning phase:

proc · sα

fidel · sα

Scorebn = sα

state · sα

mech, α1 = 0.2, α2 = 0.3. (2)

1

2

1

2

Because the geometric mean collapses when any phase approaches zero, Scorebn is useful for identifying severe reasoning failures but can be overly conservative as a headline ranking metric.

#### D.2 Full Subcategory-Level Results

This subsection expands the compact main-results table into all 22 subcategories. It reports both the process-aware reasoning score and the multi-dimensional quality score so that category-specific

strengths and failure modes can be inspected directly. Here ScorePR = Acc0QA.8 · s0dyn.2 , and S(v) is linearly mapped from [1,5] to [0,100].

Result analysis. The full subcategory table exposes two complementary trends that are compressed in the main text. First, closed-source systems lead clearly on both ScorePR and S(v), but their advantages are not uniform across reasoning dimensions: Veo3.1-Fast is strongest on World Knowledge, Seedance2.0 leads overall and on Information-Based reasoning, while Sora2 remains competitive on Human-Centric process scores. Second, the gap between S(v) and ScorePR is substantial in several subcategories, indicating that visually plausible or temporally smooth outputs can still miss the intended state transition or causal mechanism. This is especially visible in Logic Reasoning and Information-Based categories, where quality scores are often moderate while process-aware scores remain low.

#### D.3 Process-aware Reasoning Verification

The following tables isolate outcome QA accuracy for each top-level reasoning dimension. They complement Table 14 by showing which subcategories contribute most to each model’s process-aware reasoning performance.

Result analysis. The QA-only breakdown confirms that reasoning difficulty is highly category dependent. World Knowledge and Human-Centric categories show the clearest closed-source advantage, whereas Logic Reasoning remains difficult for all generators, with low averages even for the

- Table 14: Full subcategory-level evaluation results. For each of the 22 subcategories, we report

ScorePR (%) and S(v) (0–100). Best and second-best within each model family are bold and underlined. Some cells are “–” due to limited video coverage for that model–subcategory pair. This table expands the category-level summary in Table 2.

Closed-Source Models Open-Source Models Dimension Sub-category Metric Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

WorldKnowledge

Material Change

ScorePR 24.3 52.4 54.7 36.4 57.9 21.2 32.9 26.7 34.8 34.4 28.0 S(v) 40.8 70.0 46.2 60.0 80.5 37.5 55.2 45.0 51.5 59.1 48.0

Public Systems

ScorePR 83.7 71.0 36.4 50.0 65.4 11.7 20.7 20.3 15.2 13.1 17.9 S(v) 47.5 73.8 85.0 81.2 85.0 28.5 59.0 39.5 38.5 45.0 52.5

World Mechanics

ScorePR 47.7 32.8 42.1 40.1 47.1 19.9 26.9 15.0 26.4 18.3 14.3 S(v) 79.0 81.7 73.0 66.8 70.2 40.2 57.5 25.0 37.5 43.1 45.2

Cultural Life

ScorePR 43.5 47.3 28.9 36.4 57.9 15.5 19.5 20.3 21.3 18.8 13.4 S(v) 63.7 50.7 61.3 48.8 78.8 17.8 51.2 44.0 47.2 47.9 41.0

Everyday Living

ScorePR – 32.0 57.9 43.6 48.4 20.3 19.5 18.4 30.3 19.2 24.9 S(v) 72.5 83.0 83.0 60.0 75.0 26.0 41.0 38.2 42.8 40.3 37.7

Earth Cycles

ScorePR – 43.6 35.8 48.5 41.1 – 15.7 17.1 19.3 19.3 11.0 S(v) 57.0 87.5 60.8 79.5 83.2 28.7 43.0 42.8 54.8 54.4 44.0

Living World

ScorePR 27.9 24.3 29.0 38.0 61.3 – 3.0 3.0 – 9.1 7.8 S(v) 83.2 69.2 89.2 97.5 100.0 26.7 29.0 27.2 33.8 32.5 33.5

Average ScorePR 45.4 43.3 40.7 41.9 54.1 17.7 19.8 17.3 24.6 20.3 16.8 S(v) 65.8 76.2 70.2 71.2 79.5 30.0 50.2 38.2 44.5 47.4 44.2

Human-Centric

Object Handling

ScorePR 20.9 55.6 42.1 31.6 43.6 20.2 27.2 28.8 17.8 23.1 25.1 S(v) 48.8 73.5 82.5 74.0 83.0 30.0 49.5 42.2 44.2 50.6 47.2

Social Scenes

ScorePR 57.9 22.9 29.8 50.4 29.1 – 29.1 12.7 16.4 13.2 10.9 S(v) 92.5 63.2 69.5 92.5 70.5 31.0 50.7 40.8 42.8 48.2 54.8

Skilled Action

ScorePR 39.4 36.1 47.3 47.9 33.3 9.5 23.9 17.3 11.7 20.3 25.4 S(v) 97.5 100.0 83.2 97.5 75.8 43.5 72.0 56.8 59.5 47.3 68.0

Personal Routine

ScorePR – 36.6 20.9 – – – 22.8 12.1 15.2 14.5 10.9 S(v) – 63.0 78.0 – 47.5 28.0 52.2 38.0 43.5 36.9 46.8

Public Conduct

ScorePR 62.7 46.7 30.2 40.3 42.7 17.4 27.4 16.0 25.8 19.7 20.0 S(v) 57.5 76.0 54.5 81.7 81.7 19.0 56.0 53.2 49.8 45.0 45.8

Average ScorePR 45.2 39.6 34.1 42.5 37.2 15.7 26.1 17.4 17.4 17.3 18.5 S(v) 76.8 73.5 72.0 83.5 78.0 30.8 54.8 45.2 47.2 46.7 52.0

LogicReasoning

Experimental Science

ScorePR 21.8 50.6 30.1 49.1 42.8 19.5 25.1 17.9 23.6 10.7 22.4

- S(v) 53.2 60.2 55.8 76.2 47.5 25.7 28.5 24.2 34.5 25.3 36.3

Spatial Geometry

ScorePR 25.3 17.7 24.6 30.6 25.5 13.0 12.2 9.0 9.7 7.6 12.6

- S(v) 54.8 45.2 31.8 56.2 32.8 12.0 17.5 9.3 11.8 26.8 14.8

Quantitative Math

ScorePR 27.4 7.9 28.7 23.1 22.2 8.5 9.0 8.2 9.9 14.2 7.1 S(v) 45.8 24.8 42.0 33.0 25.0 12.2 14.0 8.8 8.3 24.5 10.7

Logic Puzzles

ScorePR 29.0 – – 24.3 13.9 16.7 11.8 12.1 15.3 9.6 13.2 S(v) 22.5 27.5 25.0 35.0 15.0 20.2 22.2 14.0 13.8 36.9 15.3

Pattern Discovery

ScorePR 28.9 19.3 19.3 – – 8.2 – 3.9 8.5 11.7 8.5 S(v) 27.5 37.5 30.8 70.0 46.8 6.0 13.5 11.2 14.5 29.2 16.5

Average ScorePR 26.5 23.9 25.7 31.8 26.1 13.2 14.5 10.2 13.4 11.7 12.8 S(v) 46.8 40.0 40.0 55.8 34.0 15.7 19.5 13.5 17.0 27.1 19.2

Information-Based

Data Reading

ScorePR 32.6 27.2 22.0 32.0 12.9 31.5 29.0 18.1 31.1 – 29.2 S(v) 43.8 25.2 30.0 19.8 19.5 25.5 30.8 4.7 9.7 22.5 17.2

Process Timeline

ScorePR 56.8 18.9 41.3 52.6 41.3 10.8 11.7 12.7 16.4 – 17.9 S(v) 85.0 64.5 54.2 53.5 75.8 30.2 35.0 30.0 30.5 32.4 28.2

Visual Editing

ScorePR 18.2 50.0 41.8 61.7 – 10.8 12.1 7.9 13.7 – 16.7 S(v) 20.5 46.2 36.3 55.0 43.2 18.0 21.5 5.5 15.5 18.7 18.5

Knowledge Media

ScorePR 20.9 29.1 26.2 51.6 25.1 20.7 13.5 12.7 17.7 10.0 22.5 S(v) 32.5 49.0 42.5 58.5 57.5 30.0 31.0 23.8 32.5 39.8 36.8

Creative Expression

ScorePR 50.1 69.7 61.3 61.3 64.1 40.3 25.9 29.0 30.6 24.7 27.5 S(v) 97.5 100.0 74.2 85.0 100.0 45.2 65.0 61.5 66.0 55.3 52.0

Average ScorePR 35.7 39.0 38.5 51.8 35.9 22.8 18.5 16.1 21.9 17.1 22.7 S(v) 54.2 49.0 43.0 44.5 49.0 28.0 33.5 20.2 25.2 30.4 26.7

Overall Average ScorePR 37.8 37.7 35.7 42.5 40.8 17.5 20.0 15.4 19.6 17.1 17.6 S(v) 57.5 59.5 56.2 60.8 59.2 25.7 38.5 28.2 32.8 37.6 34.5

- Table 15: Process-aware Reasoning Verification detailed results on World Knowledge. Outcome QA accuracy (%) across World Knowledge subcategories.

Closed-Source Models Open-Source Models Sub-category Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

Material Change

26.7 56.0 56.0 40.0 60.0 23.3 36.2 29.7 37.1 37.1 30.2

Public Systems

80.0 70.0 40.0 50.0 70.0 12.4 21.9 20.0 16.2 14.3 21.0

World Mechanics

50.0 36.6 48.8 41.5 48.8 20.8 28.7 17.8 28.7 21.8 19.8

Cultural Life

50.0 46.7 30.0 40.0 60.0 15.8 22.1 22.1 22.1 20.0 14.7

Everyday Living

40.0 36.0 60.0 46.7 48.0 24.0 22.9 22.9 32.3 24.0 26.0

Earth Cycles

33.3 46.7 43.3 53.3 43.3 8.2 20.0 22.4 23.5 22.4 15.3

Living World

26.7 26.7 33.3 46.7 60.0 7.5 2.5 2.5 2.5 10.0 7.5

Average 40.5 43.5 47.4 45.4 52.6 17.1 24.3 21.3 25.4 22.7 20.7

#### Table 16: Process-aware Reasoning Verification detailed results on Human-Centric. Outcome QA accuracy (%) across Human-Centric subcategories.

Closed-Source Models Open-Source Models Sub-category Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

Object Handling

20.0 55.6 42.2 32.0 46.7 22.8 30.1 30.1 19.9 25.0 27.2

Social Scenes

60.0 26.7 30.0 60.0 32.0 9.3 32.0 16.0 17.3 18.7 14.7

Skilled Action

37.5 31.2 43.8 43.8 31.2 10.6 24.2 18.2 12.1 21.2 24.2

Personal Routine

– 35.0 20.0 – 40.0 8.3 23.3 13.3 16.7 15.0 11.7

Public Conduct

60.0 48.9 30.0 40.0 42.5 18.0 28.0 18.0 26.0 22.0 20.0

Average 43.5 42.9 33.8 40.7 40.5 15.2 28.2 21.2 18.3 21.2 20.9

#### Table 17: Process-aware Reasoning Verification detailed results on Logic Reasoning. Outcome QA accuracy (%) across Logic Reasoning subcategories.

Closed-Source Models Open-Source Models Sub-category Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

Experimental Science

23.3 52.5 37.5 52.5 47.5 21.2 28.1 20.5 25.3 21.2 26.0

Spatial Geometry

- 26.0 20.0 27.5 34.0 28.0 15.8 14.8 11.5 12.0 14.2 15.8

Quantitative Math

- 27.5 8.8 28.1 23.5 22.8 10.2 10.2 9.6 11.8 10.2 9.6

Logic Puzzles

33.3 20.0 6.7 26.7 13.3 16.7 11.7 13.3 15.0 15.0 13.3

Pattern Discovery

30.0 20.0 20.0 10.0 0.0 9.8 5.9 3.9 9.8 11.8 9.8

Average 26.9 24.1 27.5 33.1 27.1 15.0 15.5 12.6 15.2 14.5 15.6

#### Table 18: Process-aware Reasoning Verification detailed results on Information-Based. Outcome QA accuracy (%) across Information-Based subcategories.

Closed-Source Models Open-Source Models Sub-category Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

Data Reading

35.5 28.1 24.2 32.3 14.5 35.3 32.0 20.9 34.0 32.7 34.6

Process Timeline

56.0 22.5 40.0 54.3 42.9 11.6 13.0 14.4 17.1 9.6 19.9

Visual Editing

20.0 50.0 40.0 65.0 30.0 11.0 12.1 7.7 14.3 15.4 16.5

Knowledge Media

20.0 32.0 28.0 52.0 24.0 21.3 17.3 16.0 21.3 18.7 22.7

Creative Expression

46.7 66.7 60.0 60.0 60.0 42.2 28.9 33.3 35.6 28.9 31.1

Average 37.1 33.8 34.0 47.1 28.7 22.7 20.6 17.1 23.9 20.6 25.1

best systems. Information-Based reasoning is more polarized: models can perform well on structured timeline or creative-expression cases, but exact data reading and visual editing remain brittle. Opensource models occasionally approach closed-source performance on narrow subcategories such as Data Reading, yet their averages remain much lower because temporal and mechanistic consistency fails across the broader set.

#### D.4 Multi-dimensional Quality Assessment

The following tables decompose the raw 1–5 quality scores into reasoning quality, temporal consistency, and visual aesthetics. This view clarifies whether a model’s aggregate S(v) comes from genuine reasoning quality or from stronger temporal and visual presentation.

- Table 19: Multi-dimensional Quality Assessment detailed results on World Knowledge. For each World Knowledge sub-category, we report reasoning quality, temporal consistency, and visual aesthetics on a 1–5 scale.

Closed-Source Models Open-Source Models Sub-category Metric Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

Material Change

Reasoning 1.33 3.50 2.25 3.00 4.00 1.59 2.64 2.29 2.41 2.41 2.05 Temporal 3.67 4.00 3.00 3.33 4.25 3.14 3.45 2.95 3.36 4.05 3.32 Aesthetics 3.33 4.00 3.50 4.00 4.50 3.09 3.73 3.33 3.64 3.95 3.68

Public Systems

Reasoning 2.00 3.50 3.50 3.50 3.50 1.38 2.86 1.95 1.86 1.86 2.43 Temporal 4.00 4.00 5.00 5.00 5.00 2.67 3.62 2.90 2.71 3.43 3.43

- Aesthetics 3.00 4.50 5.00 4.50 5.00 2.62 3.76 3.10 3.29 3.43 3.67

World Mechanics

Reasoning 4.29 4.14 3.62 3.38 3.62 2.15 2.90 1.45 2.05 1.90 2.05 Temporal 4.00 4.29 4.00 3.62 3.88 2.90 3.45 2.10 2.50 3.15 3.20

- Aesthetics 4.14 4.43 4.25 4.12 4.00 2.95 3.70 2.65 3.10 3.40 3.45

Cultural Life

Reasoning 2.50 2.33 3.00 2.50 4.00 1.11 2.53 2.16 2.42 2.17 1.95 Temporal 5.00 3.33 3.50 3.00 4.00 1.94 3.21 3.11 3.00 3.33 2.89

- Aesthetics 3.50 3.67 4.00 3.50 4.50 2.28 3.58 3.21 3.42 3.50 3.32

Everyday Living

Reasoning 3.00 4.20 4.20 3.00 4.00 1.26 1.95 1.79 2.26 1.63 1.72 Temporal 5.00 4.40 4.20 3.33 3.80 2.32 2.79 2.84 2.63 3.21 2.83

- Aesthetics 4.00 4.40 4.60 4.00 4.20 2.79 3.42 3.21 3.37 3.32 3.22

Earth Cycles

Reasoning 2.83 4.20 2.83 3.83 4.33 1.41 1.94 2.00 2.59 2.47 1.82 Temporal 3.83 4.80 3.67 4.33 4.33 2.71 3.00 3.00 3.35 3.53 3.41

- Aesthetics 3.33 4.60 4.00 4.50 4.33 2.59 3.47 3.35 3.82 3.76 3.35

Living World

Reasoning 4.33 3.67 4.67 5.00 5.00 1.25 1.00 1.00 1.38 1.25 1.25 Temporal 4.33 3.67 4.33 5.00 5.00 2.50 2.75 2.88 2.88 3.00 2.75

- Aesthetics 4.33 4.00 4.67 4.67 5.00 2.75 3.12 2.75 3.12 3.00 3.38

Average

Reasoning 3.22 3.79 3.43 3.52 4.03 1.48 2.40 1.88 2.20 2.02 1.97 Temporal 4.09 4.17 3.90 3.93 4.20 2.62 3.24 2.81 2.92 3.43 3.16 Aesthetics 3.74 4.28 4.23 4.22 4.37 2.74 3.59 3.11 3.41 3.53 3.46

- Table 20: Multi-dimensional Quality Assessment detailed results on Human-Centric. For each Human-Centric sub-category, we report reasoning quality, temporal consistency, and visual aesthetics on a 1–5 scale.

Closed-Source Models Open-Source Models Sub-category Metric Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

Reasoning 2.50 3.44 4.00 3.60 4.22 1.48 2.33 2.00 2.19 2.44 2.37 Temporal 3.00 4.33 4.44 4.20 4.22 2.63 3.19 3.00 2.89 3.33 3.15

Object Handling

- Aesthetics 3.50 4.22 4.56 4.20 4.56 2.74 3.63 3.30 3.44 3.48 3.33

Social Scenes

Reasoning 5.00 3.33 3.33 5.00 3.40 1.40 2.21 1.87 2.13 2.07 2.33 Temporal 4.50 3.50 3.83 4.50 4.00 2.73 3.36 3.13 2.87 3.47 3.73

- Aesthetics 4.50 3.83 4.33 4.50 4.20 2.87 3.79 3.13 3.33 3.53 3.80

Reasoning 5.00 5.00 4.33 5.00 4.33 2.00 3.69 2.92 2.92 2.17 3.46 Temporal 5.00 5.00 4.33 4.67 3.67 3.31 4.00 3.46 3.62 3.42 3.77 Aesthetics 4.67 5.00 4.33 5.00 4.00 3.15 4.00 3.54 3.77 3.33 4.00

Skilled Action

Reasoning – 3.00 3.75 – 2.00 1.50 2.67 1.92 2.17 1.50 1.92 Temporal – 4.00 4.00 – 3.00 2.50 3.33 2.83 3.00 3.08 3.17 Aesthetics – 3.75 4.75 – 4.00 2.58 3.42 3.00 3.25 3.17 3.83

Personal Routine

Reasoning 3.00 3.78 2.50 4.17 4.14 1.10 3.00 2.50 2.60 1.90 2.20 Temporal 3.00 4.22 3.25 4.33 4.14 1.80 3.30 3.50 3.00 3.50 3.00 Aesthetics 4.00 4.22 4.00 4.33 4.57 2.60 3.50 3.60 3.50 3.30 3.50

Public Conduct

Reasoning 4.00 3.61 3.47 4.25 3.96 1.51 2.68 2.18 2.35 2.11 2.45 Temporal 4.00 4.16 3.93 4.38 4.04 2.64 3.39 3.14 3.04 3.36 3.35 Aesthetics 4.22 4.16 4.37 4.44 4.40 2.79 3.67 3.30 3.45 3.39 3.64

Average

Result analysis. The quality-score breakdown shows that the three axes capture different failure modes. Temporal consistency and visual aesthetics are generally higher than reasoning quality, especially for closed-source models, which explains why a model can obtain a strong S(v) while still underperforming on process-aware reasoning. The gap is largest in Logic Reasoning and InformationBased categories: many videos remain visually coherent, but the reasoning-quality row drops sharply when exact symbolic, causal, or text-grounded structure must be preserved. Among open-source models, Wan2.2-14B and Cosmos-Predict2.5 often retain better perceptual quality than reasoning

- Table 21: Multi-dimensional Quality Assessment detailed results on Logic Reasoning. For each Logic Reasoning sub-category, we report reasoning quality, temporal consistency, and visual aesthetics on a 1–5 scale.

Closed-Source Models Open-Source Models Sub-category Metric Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

Experimental Science

Reasoning 2.83 3.00 3.14 3.75 2.75 1.28 1.43 1.38 1.66 1.17 1.72 Temporal 3.00 3.25 3.14 4.00 2.50 2.34 2.29 2.24 2.52 2.21 2.79 Aesthetics 3.67 4.12 3.43 4.50 3.50 2.72 2.93 2.48 3.21 2.93 3.07

Spatial Geometry

Reasoning 2.80 2.43 1.71 2.88 1.71 1.03 1.08 1.03 1.06 1.26 1.14 Temporal 3.10 2.86 2.57 3.50 2.29 1.61 1.92 1.25 1.36 2.29 1.78 Aesthetics 3.80 3.29 2.71 3.50 3.14 1.94 2.31 1.94 2.14 2.94 2.00

Quantitative Math

Reasoning 1.83 1.30 2.11 1.67 1.67 1.14 1.10 1.00 1.00 1.07 1.00 Temporal 3.50 2.30 2.78 2.50 2.00 1.59 1.72 1.26 1.10 2.24 1.61 Aesthetics 3.50 2.60 3.33 3.00 2.44 1.86 2.00 1.90 2.00 2.93 1.82

Logic Puzzles

Reasoning 1.00 1.00 1.00 1.00 1.00 1.33 1.17 1.08 1.00 1.50 1.08 Temporal 2.00 2.33 2.33 3.00 1.33 1.92 2.17 1.33 1.50 3.00 1.67 Aesthetics 3.00 3.33 3.00 3.67 2.67 2.33 2.58 2.42 2.33 3.25 2.25

Pattern Discovery

Reasoning 1.50 2.00 1.33 3.50 2.67 1.00 1.00 1.00 1.10 1.00 1.00 Temporal 2.50 2.67 2.67 4.00 2.67 1.00 1.60 1.20 1.50 2.60 2.00

- Aesthetics 2.50 3.00 3.00 4.00 3.33 1.80 2.20 2.30 2.30 3.30 2.20

Average

Reasoning 2.30 2.03 2.07 2.70 2.00 1.15 1.17 1.11 1.19 1.19 1.23 Temporal 3.00 2.71 2.76 3.41 2.20 1.77 1.96 1.50 1.61 2.36 2.00

- Aesthetics 3.52 3.26 3.14 3.74 3.00 2.15 2.40 2.14 2.40 3.00 2.27

- Table 22: Multi-dimensional Quality Assessment detailed results on Information-Based. For each Information-Based sub-category, we report reasoning quality, temporal consistency, and visual aesthetics on a 1–5 scale.

Closed-Source Models Open-Source Models Sub-category Metric Sora2 Kling Wan2.6 Seedance2.0 Veo3.1-Fast LTX2.3 Wan2.2-14B UniVideo HunyuanVideo-1.5 Cosmos-Predict2.5 LongCat-Video

Reasoning 2.00 1.55 1.75 1.17 1.25 1.03 1.38 1.00 1.03 1.10 1.24 Temporal 3.33 2.18 2.42 2.08 1.83 2.41 2.66 1.07 1.38 2.03 1.97

Data Reading

- Aesthetics 3.17 2.45 2.58 2.33 2.42 2.93 2.93 1.55 1.86 2.83 2.00

Process Timeline

Reasoning 4.40 3.12 2.50 2.71 3.86 1.45 1.48 1.45 1.52 1.45 1.45 Temporal 4.40 3.62 3.25 3.14 3.86 2.59 2.72 2.41 2.38 2.62 2.31

- Aesthetics 4.40 4.12 4.00 3.71 4.43 2.86 3.31 3.00 3.00 3.10 2.86

Reasoning 1.00 2.25 2.00 2.75 2.50 1.06 1.22 1.00 1.17 1.06 1.39 Temporal 2.00 3.00 2.25 3.25 2.50 1.83 2.00 1.17 1.56 1.76 1.72

Visual Editing

- Aesthetics 2.75 3.50 3.25 3.75 3.25 2.50 2.56 1.56 2.28 2.65 2.22

Knowledge Media

Reasoning 2.00 2.60 1.80 3.40 3.00 1.60 1.86 1.53 2.00 1.73 1.93 Temporal 2.00 3.00 3.20 3.20 3.40 2.47 2.43 2.07 2.40 3.00 2.80

- Aesthetics 3.00 3.40 3.40 3.40 3.60 2.73 2.57 2.40 2.60 3.33 2.87

Creative Expression

Reasoning 5.00 5.00 3.67 4.00 5.00 2.11 3.00 2.89 3.44 2.44 2.44 Temporal 5.00 5.00 4.00 4.33 5.00 3.33 4.00 3.78 3.67 3.67 3.44

- Aesthetics 4.67 5.00 4.33 5.00 5.00 3.22 4.00 3.89 3.89 3.78 3.56

Reasoning 2.68 2.55 2.16 2.35 2.65 1.34 1.60 1.38 1.56 1.41 1.54 Temporal 3.48 3.06 2.88 2.87 2.94 2.45 2.65 1.87 2.06 2.45 2.28 Aesthetics 3.52 3.42 3.31 3.26 3.42 2.83 3.02 2.31 2.56 3.04 2.56

Average

quality, reinforcing the need to report the axes separately rather than relying only on an aggregate score.

### E Full Frame-Rate Ablation Results

Tables 23 and 24 provide the frame-rate ablation results. The compact table summarizes overall accuracy and reasoning gap across FPS settings, while the full table includes per-category QA accuracy, reasoning-phase scores, selected 4 FPS open-source results, and the average rows used to select the default setting. Average rows are computed over the six closed-source models with all three FPS settings. Token costs per 5s video are approximately 4.5k, 9.0k, and 12.2k visual tokens for 2, 4, and 8 FPS respectively when using Qwen3.5-27B.

Table 23: Compact frame-rate ablation for Reasoning Verification. We report only overall QA accuracy (AccQA, %) and the reasoning gap (∆RG) across three frame rates for six closed-source models. Bold marks the best overall accuracy per model; full per-category, reasoning-phase, and open-source 4 FPS results are provided in Table 24.

FPS Metric Kling Seedance2.0 Sora2-8s Sora2-12s Veo3.1-Fast Wan2.6 Average 2

Overall 34.0 37.8 34.4 36.3 34.8 31.9 34.9 ∆RG +0.144 +0.169 +0.101 +0.072 +0.079 +0.152 +0.120 Overall 36.3 41.8 37.1 35.2 37.3 35.8 37.2 ∆RG +0.122 +0.135 +0.097 +0.090 +0.108 +0.127 +0.113

- 4

Overall 35.3 40.5 41.0 36.9 38.0 33.7 37.6 ∆RG +0.130 +0.129 +0.146 +0.045 +0.096 +0.074 +0.098

8

Result analysis. The ablation supports 4 FPS as the default evaluation setting. Moving from 2 FPS to 4 FPS improves the six-model average accuracy from 34.9% to 37.2%, with gains on most closed-source models and a lower reasoning gap than 2 FPS. Although 8 FPS slightly increases the average accuracy to 37.6%, the gain is small relative to the additional visual-token cost, and it does not consistently improve every model. The full table also shows that higher frame rate mainly helps categories requiring dense temporal evidence, while reasoning-phase bottlenecks such as process and mechanism scores remain low; therefore, simply increasing frames cannot replace process-aware verification.

### F Point-wise Scoring Ablation Details

We ablate point-wise scoring strategies to test whether reducing inter-dimension halo effects improves induced pairwise accuracy and rank correlation. Table 25 reports full-set results on the 5,969-pair WorldRewardBench benchmark. Vanilla uses Qwen3.5-27B-Thinking with a single prompt for three scores, No-Thinking disables extended thinking, and SDE denotes Sequential Dimension Evaluation with three independent calls. Correlations r-t and r-a measure reasoning–temporal-consistency and reasoning–aesthetics coupling, respectively; parse rates are computed per unique video.

The results reveal a tension between halo reduction and preference recovery. SDE reduces the reasoning↔temporal-consistency correlation from 0.770 to 0.384, but this decoupling does not translate into better rank correlation or induced accuracy. Vanilla Thinking achieves the highest ρ (0.626) and the best w/o-ties accuracy (67.63%) despite exhibiting stronger inter-dimension correlation. This suggests that human annotators themselves often assign correlated dimension scores: a video that fails reasoning also tends to receive lower temporal-consistency and aesthetics scores. SDE is therefore useful as a diagnostic tool, but vanilla scoring is the better default for efficient preference recovery.

### G Elo Ranking Details

For the arena-style ranking in Table 4, each WorldRewardBench pair carries a human or judge verdict: model A wins, model B wins, or tie. We fit a shared ability vector θ ∈ R11 using the Bradley-Terry

- Table 24: Full frame-rate ablation for Reasoning Verification. We evaluate three frame rates (2, 4, 8 FPS) on six closed-source models across four reasoning categories, plus five open-source models at the selected 4 FPS default. WK, HC, LR, and IB denote World Knowledge, Human-Centric, Logic Reasoning, and Information-Based. We report per-category QA accuracy (AccQA, %), overall average, and the four reasoning-phase scores (sstate, sproc, sfidel, smech, %). Bold indicates the best FPS setting per model.

Model FPS WK HC LR IB Overall sstate sproc sfidel smech ∆RG

Kling

2 39.2 38.7 26.9 31.3 34.0 45.2 23.9 35.7 28.2 +0.144 4 43.4 43.2 24.4 34.0 36.3 46.4 25.9 36.5 32.6 +0.122

- 8 41.6 40.0 25.6 33.9 35.3 42.0 24.7 40.5 31.8 +0.130

Seedance2.0

2 41.8 33.8 35.1 40.5 37.8 55.5 24.8 34.9 31.7 +0.169 4 45.2 41.0 33.3 47.5 41.8 56.3 31.7 39.0 36.6 +0.135

- 8 41.7 39.8 37.6 43.0 40.5 52.0 28.8 39.3 36.6 +0.129

Sora2-8s

2 40.0 37.8 23.9 36.0 34.4 43.7 26.1 34.9 32.3 +0.101 4 40.1 44.1 27.1 37.2 37.1 51.5 27.1 32.4 37.3 +0.097 8 43.8 53.3 29.7 37.2 41.0 49.4 24.8 45.3 40.7 +0.146

Sora2-12s

2 35.7 44.4 28.7 36.4 36.3 46.5 34.2 32.7 30.6 +0.072 4 35.9 42.2 28.7 34.0 35.2 40.3 27.5 38.4 33.1 +0.090 8 33.8 46.7 30.0 37.2 36.9 40.2 32.7 38.5 37.0 +0.045

Veo3.1-Fast

2 51.2 36.9 25.6 25.6 34.8 39.0 29.4 38.0 31.7 +0.079 4 52.4 40.8 27.3 28.8 37.3 41.9 30.2 40.9 30.9 +0.108 8 51.2 42.3 27.1 31.3 38.0 43.1 27.9 40.8 36.8 +0.096

Wan2.6

2 39.5 28.7 27.9 31.8 31.9 44.1 20.8 33.2 26.2 +0.152 4 47.2 34.0 27.9 34.2 35.8 47.0 26.0 36.4 31.9 +0.127 8 42.8 33.8 27.9 30.4 33.7 39.0 25.2 34.1 33.1 +0.074

Open-source models (fps=4 only)

LTX2.3 4 17.0 15.3 15.0 22.7 17.5 24.0 6.0 20.7 14.7 +0.121 Wan2.2-14B 4 24.2 28.3 15.5 20.4 22.1 30.9 9.3 26.1 18.5 +0.146 UniVideo 4 21.3 21.3 12.6 17.1 18.1 24.8 6.2 22.9 14.2 +0.136 HunyuanVideo-1.5 4 25.2 18.4 15.2 23.9 20.7 29.0 8.7 23.1 18.4 +0.124 LongCat-Video 4 20.6 21.0 15.7 25.0 20.6 26.8 9.0 26.8 15.6 +0.145

Average

2 41.2 36.7 28.0 33.6 34.9 45.7 26.5 34.9 30.1 +0.120 4 44.0 40.9 28.1 35.9 37.2 47.2 28.1 37.3 33.7 +0.113 8 42.5 42.6 29.7 35.5 37.6 44.3 27.3 39.7 36.0 +0.098

- Table 25: Point-wise scoring ablation. We report parse rate, induced pairwise accuracy (w/ and w/o ties), Spearman ρ, and inter-dimension Pearson correlations as a halo-effect diagnostic on the full

- 5,969-pair set. Vanilla Thinking is the most efficient protocol and achieves the best rank correlation. Method Calls Parse Acc w/t Acc w/o ρ r-t r-a

Vanilla 1 98.0% 58.45 67.63 0.626 0.770 0.741 No-Thinking 1 99.9% 53.15 65.55 0.591 0.764 0.803 SDE 3 95.0% 58.00 64.64 0.562 0.384 0.495

model with the Davidson tie extension [2]:

eθ

A

eθA + eθB + 2ν e(θA+θB)/2 , (3)

P(A ≻ B) =

where ν is the tie propensity parameter estimated from data. The fitted tie propensities are νhuman = 0.148 for expert preferences and νjudge = 0.070 for VLM-judge verdicts. We apply ℓ2 regularization (α = 1.0) and compute 95% confidence intervals via 1,000 bootstrap resamples at the prompt-cluster level (resampling by task_id to preserve intra-prompt correlation). Scores are mapped to the Elo scale as Eloi = 1000 + θi · 400/ln10.

Result analysis. The estimated tie propensities indicate that human annotators use ties more often than the VLM judge (νhuman = 0.148 vs. νjudge = 0.070), which is consistent with the judge making sharper decisions on near-equal pairs. This difference helps explain why judge Elo can separate visually similar closed-source models differently from humans, even when the overall ranking remains strongly aligned. The bootstrap confidence intervals in the main table should therefore be interpreted together with the tie model: close model pairs are inherently less stable than the large closed-source/open-source separation.

### H Expert Human Annotation Protocol

This appendix describes the human annotation protocol for WorldRewardBench. The goal is to collect reliable human preference references for evaluating whether automatic judges align with human judgments of world-state reasoning in generated videos. Annotators judge whether each video correctly realizes the transition implied by the input image and prompt, rather than only its visual appeal.

#### H.1 Annotators and Privacy

WorldRewardBench was annotated by fifteen trained annotators with diverse backgrounds related to video generation. The annotator group included researchers working on video generation and multimodal evaluation, as well as users with practical experience using video generation systems in different application scenarios. Before formal annotation, all annotators received a one-hour training session covering the benchmark objective, annotation interface, scoring dimensions, representative calibration examples, and common failure cases in image-to-video generation.

The annotation task did not require annotators to provide personal or sensitive information beyond their scoring decisions. All annotators participated voluntarily in the annotation process. The annotations were used only to construct aggregate video-level scores and pairwise preference labels for WorldRewardBench. Released annotation files contain only anonymized annotator identifiers, and no personally identifying information about annotators is collected or released.

#### H.2 Annotation Interface and Scoring Rubric

Each annotation unit is an image-prompt-video tuple. For each selected benchmark case, annotators were shown the input image, the corresponding generation prompt, and eight anonymized generated videos sampled from the candidate pool. Model identities were hidden to reduce model-name bias. Annotators evaluated the full generated videos rather than sampled frames, and they could replay each video before submitting scores. The annotation assignment was randomized to avoid model-specific ordering patterns, while ensuring that each video was rated by at least two annotators.

Figure 5 shows the annotation interface. The upper panel displays the category, case ID, input image, and prompt, while the main panel presents the eight anonymized generated videos. Annotators scored each full video on three dimensions using a 1–5 scale: Reasoning Quality, Temporal Consistency, and Visual Aesthetics. The detailed scoring rubric is provided in Table 26. Annotators were instructed to treat Reasoning Quality as the primary criterion because WorldRewardBench evaluates world-state prediction rather than generic visual appeal, and not to reward visual quality alone when the video fails to satisfy the required causal, physical, logical, or informational relation.

[Figure 235]

Figure 5: Human annotation interface for WorldRewardBench. Annotators see the input image, prompt, and eight anonymized generated videos. For each video, they provide three scores: Reasoning Quality, Temporal Consistency, and Visual Aesthetics. Model identities are hidden during annotation.

Table 26: Human scoring rubric for WorldRewardBench.

Score Reasoning Quality Temporal Consistency Visual Aesthetics

- 1 Fails the prompt or required world-state transition.

Severe discontinuity, flickering, jumps, or deformation.

Very poor quality with blur, distortion, or major artifacts.

- 2 Weakly related to the prompt; main transition is incorrect.

Many visible temporal or motion problems. Poor rendering with obvious artifacts.

- 3 Partially satisfies the prompt but has clear reasoning errors.

Mostly continuous, but with noticeable issues.

Acceptable but average visual quality.

- 4 Mostly satisfies the prompt with only minor reasoning issues.

Relatively smooth and natural, with minor glitches.

Good visual quality and generally natural appearance.

- 5 Fully satisfies the prompt with correct causal, physical, logical, or informational consistency.

Refined, stable, and visually pleasing.

Highly continuous with natural motion and coherent changes.

#### H.3 Disagreement Detection

To improve annotation reliability, we use a disagreement-based quality control procedure before final score aggregation. For each video v and each scoring dimension d, we compute the score range among available annotators:

sa,d(v) − min a∈Av

Rd(v) = max a∈Av

sa,d(v),

where Av denotes the set of annotators who rated video v, and d ∈ {r,c,a} corresponds to Reasoning Quality, Temporal Consistency, and Visual Aesthetics.

A video is flagged as high-disagreement if the score range exceeds a predefined threshold. Specifically, if a video has two ratings and Rd(v) > 1, or if a video has three ratings and Rd(v) > 2, the video is assigned to additional annotators. Each high-disagreement video receives at least four valid ratings before final aggregation. We do not discard annotations solely because annotators disagree; instead, disagreement triggers additional annotation to obtain a more stable estimate.

After re-annotation, we compute inter-annotator reliability on the final annotation set. As shown in Table 27, the final annotations show moderate-to-substantial agreement across all three dimensions. The aggregated score achieves the strongest reliability, with Krippendorff’s α = 0.744, ICC(2,k)= 0.936, and mean pairwise Spearman ρ = 0.784, indicating that the final video-level human scores are stable for constructing WorldRewardBench preferences.

For each video v, we then average the final annotator scores within each dimension:

1 |Av| a∈A

##### sa,d(v),

s¯d(v) =

v

where d ∈ {r,c,a} denotes Reasoning Quality, Temporal Consistency, and Visual Aesthetics. The aggregated human score is computed as:

##### Shuman(v) = 0.4¯sr(v) + 0.3¯sc(v) + 0.3¯sa(v).

The larger weight on Reasoning Quality reflects the goal of WorldRewardBench: evaluating whether generated videos correctly realize the intended world-state transition rather than merely rewarding visual appeal.

- Table 27: Inter-annotator reliability of human scores after re-annotation. Krippendorff’s α measures agreement on ordinal 1–5 ratings, ICC(2,k) measures the reliability of averaged scores, and mean pairwise Spearman ρ measures rank-level agreement between annotators.

Dimension Krippendorff’s α ICC(2,k) Mean Pairwise Spearman ρ

Reasoning Quality 0.723 0.939 0.766 Temporal Consistency 0.648 0.903 0.679 Visual Aesthetics 0.649 0.903 0.676 Aggregated Score 0.744 0.936 0.784

H.4 QA Quality Audit

To mitigate VLM bias in the automatically generated QA pairs of WorldReasonBench, we conduct an independent audit on a stratified random sample of approximately 300 QA pairs (balanced across the four reasoning dimensions and the four question types). Two trained auditors independently verify each pair against three criteria: (i) answerability: the question is answerable from the promptinduced visual evidence alone, without relying on hidden context; (ii) ground-truth correctness: the labelled answer is consistent with physical, social, or factual common sense and with the prompt; (iii) ground-truth uniqueness: no plausible alternative answer is supported by the prompt. Each pair is recorded with a binary accept/reject verdict and a free-form rejection reason. The two auditors reach Cohen κ=0.78 on the accept/reject verdict, indicating substantial agreement. The overall rejection rate is 7.8%, with rejections slightly more frequent on Information-Based questions (11.4%) due to the stricter requirement on exact text and data preservation, and lowest on World Knowledge questions (5.3%). The most common rejection reasons are ambiguous ground truth (43% of rejections), multiple plausible answers (31%), and requires off-screen context (18%). Rejected QA pairs are either rewritten by the auditors or removed before release, so the released benchmark contains only audit-passed or human-corrected QA. The audit is treated as a quality-control pass rather than as a separate evaluation set; full audit logs and per-dimension breakdowns will be released alongside the benchmark.

I WorldRewardBench Human Scoring Breakdown

- Table 28 reports the expert human scores used to construct WorldRewardBench, broken down by model and reasoning category. We keep this detailed annotation view in the appendix because the main text focuses on human-preference Elo and reward-model alignment.

Result analysis. The human-score breakdown shows a consistent ordering across categories: Seedance2.0 has the highest overall weighted total, followed by Kling, Veo3.1-Fast, and Wan2.6, while open-source models occupy the lower half of the ranking. The category rows also clarify

- Table 28: Human scoring breakdown on WorldRewardBench. We report expert annotation scores by model and reasoning category. The three raw dimensions are reasoning quality, temporal consistency, and visual aesthetics; category and overall weighted totals follow the human scoring protocol used to construct WorldRewardBench.

Reasoning Quality

Temporal Consistency

Visual Aesthetics

Category Weighted Total

Overall Weighted Total

Sample Count per Dimension

Rank Model Category

- 1 seedance2.0

worldknowledge 3.9151 4.2547 4.3396 4.1443

3.8225

106 humancentric 4.3731 4.4179 4.4179 4.4000 67 logicreasoning 3.1298 3.5344 3.6260 3.4000 131 informationbasedreasoning 3.6140 3.6754 3.7368 3.6693 114

- 2 keling

worldknowledge 3.4630 3.9352 3.9907 3.7630

3.3838

108 humancentric 3.8438 4.0000 4.1797 3.9914 128 logicreasoning 2.5670 2.9381 3.0103 2.8113 97

- informationbasedreasoning 2.4545 2.9596 2.9192 2.7455 99

3 veo3.1_fast

worldknowledge 3.6774 3.2742 3.4597 3.4911

3.3309

124 humancentric 3.9266 3.7982 3.9633 3.8991 109 logicreasoning 2.3923 2.7615 2.9769 2.6785 130

- informationbasedreasoning 3.2054 3.3929 3.5268 3.3580 112

- 4 wan2.6

worldknowledge 3.2328 3.5431 3.6379 3.4474

3.2401

116 humancentric 3.7481 3.8593 4.0074 3.8593 135 logicreasoning 2.5310 2.9027 2.9735 2.7752 113 informationbasedreasoning 2.5574 2.9590 2.9262 2.7885 122

- 5 sora2-8s

worldknowledge 2.8587 2.9130 2.8152 2.8620

2.6290

92 humancentric 3.3529 3.0000 2.8529 3.0971 34 logicreasoning 2.5328 2.9016 2.8934 2.7516 122

- informationbasedreasoning 1.9550 2.3153 2.2703 2.1577 111

6 sora_12s

worldknowledge 2.6023 2.6818 2.6023 2.6261

2.5558

88 humancentric 3.6774 2.9355 3.0000 3.2516 31 logicreasoning 2.3600 2.6320 2.6720 2.5352 125

- informationbasedreasoning 2.1600 2.4200 2.3800 2.3040 100

- 7 wan2.2-14b

worldknowledge 1.8182 2.6636 2.7909 2.3636

2.1252

110 humancentric 2.3120 2.6400 2.6880 2.5232 125 logicreasoning 1.4310 1.9052 1.8707 1.7052 116 informationbasedreasoning 1.4679 2.1743 2.1193 1.8752 109

- 8 hunyuan

worldknowledge 1.7117 2.3604 2.5315 2.1523

1.9741

111 humancentric 1.9167 2.4417 2.4750 2.2417 120 logicreasoning 1.2455 1.6727 1.8000 1.5400 110 informationbasedreasoning 1.7477 2.0467 2.0748 1.9355 107

- 9 longcat

worldknowledge 1.5918 2.5714 2.6735 2.2102

1.9524

98

humancentric 1.9821 2.6518 2.7411 2.4107 112 logicreasoning 1.2500 1.6293 1.6638 1.4879 116 informationbasedreasoning 1.4694 1.9388 1.8367 1.7204 98

- 10 uni

worldknowledge 1.4646 1.7778 1.7374 1.6404

1.4654

99

humancentric 1.4672 1.7623 1.7623 1.6443 122 logicreasoning 1.1062 1.2035 1.2743 1.1858 113 informationbasedreasoning 1.2525 1.4040 1.5556 1.3889 99

- 11 ltx2_3

worldknowledge 1.1429 1.2476 1.2857 1.2171

105 humancentric 1.0161 1.0726 1.0806 1.0524 124 logicreasoning 1.1316 1.3158 1.2544 1.2237 114 informationbasedreasoning 1.3505 1.4124 1.3814 1.3784 97

1.2080

why the Elo ranking is not determined by a single dimension. For example, some systems are comparatively stronger on Human-Centric or World Knowledge scenes but lose ground on Logic Reasoning and Information-Based cases, where exact state changes and structured content are harder to preserve. This supports using category-weighted human supervision rather than relying only on global aesthetic preference.

### J WorldRewardBench Post-processing Details

Starting from all candidate pairwise preferences induced by the ranked videos within each benchmark case, we apply a deterministic post-processing pipeline with a fixed random seed. The goal is to suppress overly easy high-margin pairs, preserve informative near-equal cases, and reduce presentation-order bias without collapsing category or model-pair coverage.

Implementation details. For a candidate pair (vi,vj), we first compute the human preference margin ∆ij = |S(vi) − S(vj)|. We then perform stratified subsampling within each margin bin, where strata are defined by the top-level reasoning dimension and the unordered model pair. The

retention schedule is: keep all pairs with ∆ij ≤ 1.5, keep 90% with 1.5 < ∆ij ≤ 2, keep 35% with 2 < ∆ij ≤ 3, and keep 15% with ∆ij > 3. Pairs with ∆ij < 0.1 are relabeled as ties rather than strict preferences. Among these ties, we assign A=B=Bad when both videos have aggregated scores at most 2.0, and A=B=Good otherwise. Finally, after filtering, we randomize the left/right assignment within each stratum so that strict-preference pairs are approximately balanced between A > B and A < B.

- Table 29: Margin-bin filtering statistics for WorldRewardBench. Candidate pairs are constructed from the ranked sampled videos before confidence-aware post-processing.

Margin bin Candidate pairs Retained pairs Keep ratio ∆ ≤ 1.5 4666 4666 100.0% 1.5 < ∆ ≤ 2 893 804 90.0% 2 < ∆ ≤ 3 1193 418 35.0% ∆ > 3 543 81 14.9% Total 7295 5969 81.8%

- Table 30: Final label composition after post-processing. Strict-preference pairs are balanced across left/right ordering, while near-equal cases are retained as informative ties.

Final label group Count Share Strict preference (A > B) 2705 45.3% Strict preference (A < B) 2705 45.3% Tie (A=B=Bad) 389 6.5% Tie (A=B=Good) 170 2.8% Total 5969 100.0%

- Table 31: Reasoning-dimension coverage before and after post-processing. Stratified sampling preserves broad coverage across the four top-level reasoning dimensions.

Reasoning dimension Candidate pairs Retained pairs Keep ratio Human-Centric Reasoning 1381 1019 73.8% Information-Based Reasoning 1946 1681 86.4% Logic Reasoning 2144 1849 86.2% World Knowledge 1824 1420 77.9% Total 7295 5969 81.8%

Result analysis. The post-processing statistics show that the final benchmark keeps most informative pairs while reducing over-representation of easy, high-margin comparisons. All pairs with score margin at most 1.5 are retained, whereas only 14.9% of pairs above margin 3 remain. The final labels are exactly balanced between A > B and A < B strict preferences, and ties account for 9.3% of the benchmark. Category coverage also remains broad after filtering, with each top-level reasoning dimension retaining more than 1,000 pairs, so the difficulty-weighted split improves discriminability without collapsing the benchmark into a narrow subset.

K Full-Set WorldRewardBench Results

- Table 32 reports reference results on the full 5,969-pair benchmark (1,432 unique videos, 130 tasks, 11 generators). This setting preserves the natural distribution of score gaps, including a large proportion of easy high-margin pairs, and matches the compact reward-alignment table in the main text.

Implementation details. For pair-wise evaluation, the parse rates are 99.6% for Qwen3.5-27BThinking and 99.9% for Qwen3.5-27B-Instruct. The Instruct setting disables the extended thinking chain (enable_thinking=False). GPT-5.4 point-wise evaluation uses the iter-400 video subset

(595 videos, 846 induced pairs, 26.2% coverage of the full benchmark). The Qwen3.5-27B Thinking (4FPS) setting uses explicit 4 FPS sampling via vLLM (∼20 frames per 5s video), while the other Qwen3.5 columns use the serving backend’s default frame sampling (∼10 frames).

- Table 32: Full-set human-alignment agreement (%) on the original 5,969-pair WorldRewardBench. Reported as a reference complement to the compact main-text reward-alignment table.

Dimension Qwen3.5-27B Pair Qwen3.5-27B Point

World Knowledge 69.94 60.57 Human-Centric 72.61 62.81 Logic Reasoning 70.16 60.17 Information-Based 60.24 50.15

Overall 67.74 57.85

Result analysis. The full-set reference table shows that Qwen3.5-27B pair-wise comparison reaches 67.74% overall agreement, while point-wise scoring reaches 57.85%. The gap is largest in Information-Based reasoning, where exact text and data preservation make single-video scoring less reliable. This supports the main-text observation that pair-wise comparison remains stronger for recovering fine-grained human preferences, while point-wise scoring is useful for calibrated per-video feedback.

### L Subcategory-Level WorldRewardBench Results

For completeness, Table 33 reports the full WorldRewardBench breakdown at the subcategory level. This detailed view uses the same reward-model ordering and the same Pair/Point protocol split as the compact main-text table, but expands each top-level reasoning dimension into its constituent subcategories to support finer-grained diagnosis.

Result analysis. The subcategory-level reward results show that direct pair-wise judging is consistently stronger than point-wise induced comparison across nearly all categories, especially in Logic Reasoning and Information-Based cases. Within point-wise methods, Qwen3.5-27B and Gemini-3.1 remain competitive on many subcategories, while the 4 FPS point-wise setting improves some Information-Based rows but does not close the gap to direct comparison. These results suggest that point-wise scores are useful for calibrated per-video feedback and rank correlation, but pair-wise comparison is still preferable when the goal is recovering fine-grained human preferences between close candidates.

### M Weight Design and Sensitivity

This appendix analyzes the weight choices behind our two headline metrics, ScorePR = Acc0QA.8 ·s0dyn.2 and S(v) = 0.4sr + 0.3sc + 0.3sa, and shows that the model rankings are stable under reasonable alternatives.

Why Acc0QA.8 ·s0dyn.2 is not at odds with process awareness. Recall that AccQA is the equal-weighted mean over the four reasoning phases: AccQA = 14(sstate+sproc+sfidel+smech), so sproc and smech already contribute one quarter each to AccQA. The multiplicative term s0dyn.2 = ((sproc+smech)/2)0.2 in ScorePR therefore acts as a second-order penalty on outcome-hacking: models with high AccQA but low dynamic phases are gently down-weighted, while a model that does well on every phase is barely affected. The choice α=0.8 keeps QA accuracy as the dominant signal so that the headline metric remains directly interpretable; the limit α → 1 recovers AccQA, while α → 0 collapses to the dynamic phase score.

ScorePR sensitivity. Table 35 reports how the eleven-model ranking on WorldRewardBench (Table 4) tracks human Elo as α varies and as ScorePR is replaced by alternative aggregators. Spearman

- Table 33: Subcategory-level WorldRewardBench results (Part 1: World Knowledge & HumanCentric). We report pairwise accuracy w/o ties (%) for two protocols: Pair (direct comparison by Qwen3.5-27B-Instruct) and three point-wise variants. Best per-row in bold.

Instruct Point-wise Induced Accuracy (%) Dimension Sub-category Pair w/o Gemini-3.1 Qwen3.5-27B Ours (4FPS)

WorldKnowledge

Material Change 73.8 53.3 57.5 44.8 Public Systems 63.5 47.4 41.0 38.5 World Mechanics 79.2 61.7 69.4 61.1 Cultural Life 76.4 59.1 62.7 53.6 Everyday Living 68.2 58.7 58.7 41.3 Earth Cycles 71.5 70.2 64.4 46.7 Living World 80.2 78.4 68.1 61.2

Average 73.2 61.3 60.3 49.6

Human-Centric

Object Handling 76.7 66.1 63.3 53.1 Social Scenes 77.2 69.4 68.9 57.5 Skilled Action 67.1 59.5 61.1 38.1 Personal Routine 86.4 72.9 79.7 66.1 Public Conduct 75.1 58.2 66.9 55.1

Average 76.5 65.2 68.0 54.0

- Table 34: Subcategory-level WorldRewardBench results (Part 2: Logic Reasoning & Information-Based). Same protocol and metrics as Table 33.

Instruct Point-wise Induced Accuracy (%) Dimension Sub-category Pair w/o Gemini-3.1 Qwen3.5-27B Ours (4FPS)

Experimental Science 69.1 55.1 64.9 51.8 Spatial Geometry 74.5 56.6 53.9 48.3 Quantitative Math 78.0 66.0 52.7 35.3 Logic Puzzles 72.0 62.3 64.6 46.9 Pattern Discovery 81.8 69.4 75.2 61.2

LogicReasoning

Average 75.1 61.9 62.3 48.7

Data Reading 56.6 36.8 39.7 39.7 Process Timeline 73.6 60.1 58.3 55.1 Visual Editing 66.0 55.3 64.1 57.3 Knowledge Media 73.1 64.4 64.4 49.3 Creative Expression 68.3 54.7 58.4 52.2

Information-Based

Average 67.5 54.3 57.0 50.7 Overall 73.1 60.7 61.7 50.6

- Table 35: ScorePR weight and aggregator sensitivity on the eleven-model human Elo (Table 4). Spearman ρ, Kendall τ, and pair-wise rank-accuracy against human Elo. The paper setting is marked [paper].

Aggregator ρ τ Pair acc.

Acc1QA.0 · s0dyn.0 (= AccQA) 0.927 0.782 0.891 Acc0QA.9 · s0dyn.1 0.945 0.818 0.909 Acc0QA.8 · s0dyn.2 [paper] 0.955 0.855 0.927 Acc0QA.7 · s0dyn.3 0.936 0.818 0.909 Acc0QA.5 · s0dyn.5 (geometric mean) 0.918 0.782 0.891 Acc0QA.0 · s1dyn.0 (= sdyn) 0.827 0.691 0.852

0.5 AccQA + 0.5 sdyn (arithmetic) 0.936 0.818 0.909 0.7 AccQA + 0.3 sdyn (arithmetic) 0.936 0.818 0.909 min(AccQA, sdyn) (bottleneck) 0.827 0.691 0.852

- Table 36: S(v) weight sensitivity on the eleven-model human Elo (Table 4). Selected probes from the 231-point simplex grid (step 0.05). Among the full grid, 67.5% of points achieve ρ ≥ 0.95.

(wr, wc, wa) ρ τ Pair acc.

(0.40, 0.30, 0.30) [paper] 0.973 0.927 0.945 (1/3, 1/3, 1/3) (equal) 0.973 0.927 0.945 (0.50, 0.25, 0.25) 0.982 0.927 0.964 (0.60, 0.20, 0.20) 0.982 0.927 0.964

- (0.30, 0.35, 0.35) 0.973 0.927 0.945
- (1.0, 0.0, 0.0) (pure reasoning) 0.982 0.927 0.964

- (0.0, 0.0, 1.0) (pure aesthetics) 0.982 0.927 0.964
- (0.0, 1.0, 0.0) (pure consistency) 0.809 0.673 0.818 (0.50, 0.05, 0.45) (best simplex) 1.000 1.000 1.000

Full grid (231 points): ρ ∈ [0.809, 1.000], median ρ = 0.973.

ρ ranges from 0.83 to 0.96 across the grid: the paper setting α=0.8 achieves ρ=0.955, the highest ρ among all probed aggregators, while pure AccQA (α=1) drops to ρ=0.927 and pure dynamic scoring (α=0) and the min-bottleneck both fall to ρ=0.827. Arithmetic and geometric means sit in between.

The chosen exponent is therefore not a compromise: putting a small but non-zero weight on sdyn both improves human-Elo recovery and exposes outcome-hacking, which is the diagnostic property the metric is designed for.

S(v) weight grid search. We grid-search the simplex of (wr,wc,wa) at step 0.05 (231 points) and induce per-model S(v) from the per-video (sr,sc,sa) in our point-wise judgments, then rank the eleven models on WorldRewardBench against human Elo. Table 36 reports a representative slice; 67.5% of all grid points achieve ρ ≥ 0.95, and the full range is ρ ∈ [0.81,1.00]. The paper setting (0.4,0.3,0.3) achieves ρ=0.973, identical to equal weighting (1/3,1/3,1/3) and within 0.027 of the best simplex points such as (0.50,0.05,0.45). The single weight that under-performs the rest is pure consistency (0,1,0) with ρ=0.809, indicating that temporal-consistency alone is not a reliable model-level proxy for human preferences—this is the only ridge we deliberately avoid. We therefore keep the human-protocol-aligned (0.4,0.3,0.3) as the reported setting because (i) it matches the rubric the human annotators were trained on, ensuring that automatic and human aggregates are directly comparable, and (ii) it is empirically within 0.027 of the best alternative on the simplex.

Take-aways. Across both metrics, model rankings are stable to weight perturbations. The paper ScorePR exponent α=0.8 achieves the highest ρ (0.955) in the grid, with ρ varying smoothly to 0.83 at the two endpoints, and the paper S(v) weights (0.4,0.3,0.3) sit on a wide plateau where 67.5% of the simplex points clear ρ=0.95. The two reported settings preserve (i) interpretability (AccQA remains the dominant signal in ScorePR) and (ii) protocol-consistency with the human annotation rubric in S(v). We expose the remaining process information through the diagnostic ratio sdyn/AccQA rather than by tuning the headline weights to chase rank-correlation.

- Table 37: Overall 95% bootstrap CIs (B=2000). N is the number of cases evaluated per model on the shared evaluation set behind Table 2.

Model N AccQA [95% CI] ScorePR [95% CI] S(v) [95% CI] Sora2-8s 80 35.3 [29.5, 41.1] 34.3 [28.5, 39.8] 56.9 [50.5, 63.6] Sora2-12s 80 33.5 [28.1, 39.2] 32.4 [26.7, 38.0] 55.5 [49.3, 61.7] Kling 80 34.0 [28.2, 40.2] 32.7 [26.7, 38.7] 55.4 [47.2, 63.3] Wan2.6 80 34.7 [28.8, 40.5] 32.4 [26.4, 38.7] 50.3 [43.3, 57.4] Seedance2.0 80 41.2 [35.4, 46.6] 39.8 [34.2, 45.6] 59.4 [52.2, 66.3] Veo3.1-Fast 80 36.0 [30.4, 42.0] 35.3 [29.3, 41.2] 54.8 [47.3, 62.9] LTX2.3 80 18.5 [13.7, 23.7] 16.8 [12.2, 21.6] 28.1 [23.0, 33.8] Wan2.2-14B 80 19.6 [14.8, 24.9] 17.5 [12.5, 23.1] 30.0 [25.9, 34.6] UniVideo 80 16.2 [12.0, 20.7] 14.4 [10.3, 18.7] 21.3 [17.1, 26.2] HunyuanVideo-1.5 80 20.2 [16.0, 24.6] 17.9 [13.8, 22.4] 27.0 [21.9, 33.3] Cosmos-Predict2.5 80 19.3 [14.5, 24.5] 16.9 [12.0, 21.4] 30.5 [27.0, 34.5] LongCat-Video 80 19.7 [15.2, 24.2] 17.4 [12.7, 22.0] 25.3 [21.0, 30.2]

### N Statistical Significance and Rank Stability

This appendix reports 95% bootstrap confidence intervals (CIs) for the headline metrics in Tables 2 and 4, and the bootstrap rank distribution that backs the rank-stability statement in Section 4.2.

Bootstrap protocol. For each model and each dimension we resample its case-level QA outputs with replacement at the case level, B=2000 times, and report the point estimate, 95% percentile interval

[lo,hi], and the case count N. ScorePR on each resample is computed as AccQA0.8 · sdyn0.2, where the bars denote means over the resampled cases and sdyn is the per-case mean of the temporal and reasoning phase scores. Per-dimension CIs resample within the dimension only. S(v) resamples pervideo three-dimensional ratings (mapped from raw 1–5 to 0–100 via (r −1)/4×100) and aggregates them with the paper weights (0.4,0.3,0.3). Rank stability is computed by jointly resampling all

- 12 models in each bootstrap, sorting them by overall ScorePR, and recording each model’s rank distribution.

Missing-coverage symbols. Symbols “–” in the main tables denote dimensions or protocols not covered by a particular evaluation run rather than zero performance; for example, the GPT-5.4 column in Table 5 is computed on a 595-video subset and only with the point-wise protocol. Extended open-source results on the full WorldReasonBench benchmark are reported in Appendix O.

Subcategory sample size. The 22 subcategories average ≈20 cases each, so subcategory point estimates carry larger sampling variance than dimension-level numbers. We use them only for qualitative comparison in Appendix D and never to make rank claims that the dimension-level CIs in Tables 38 and 39 do not also support.

Overall CIs. Table 37 reports overall AccQA, ScorePR, and S(v) with 95% bootstrap CIs for every generator on the shared evaluation set used by Table 2. The closed-vs.-open separation is

statistically robust: the largest open-source overall-ScorePR upper bound (23.1, Wan2.2-14B) is below the smallest closed-source lower bound (26.4, Wan2.6), and the same ordering holds for

AccQA and S(v). Within the closed-source tier, Seedance2.0 has the only non-overlapping lower bound (34.2) that exceeds several other closed-source point estimates, while Sora2-8s, Sora2-12s, Kling, Wan2.6, and Veo3.1-Fast all have CIs that mutually overlap. Within the open-source tier, no generator has a CI that completely separates from the others; HunyuanVideo-1.5 and Wan2.2-14B share the highest point estimates but their CIs overlap with those of the remaining four.

Per-dimension ScorePR CIs. Table 38 reports per-dimension ScorePR with 95% CIs on the same shared evaluation set. The per-dimension half-widths are largest on Human-Centric (the dimension with the smallest case pool) and tightest on Logic Reasoning. Two qualitative claims still survive the wider intervals: (i) on World Knowledge and Information-Based the worst closed-source CI lower bound stays above (or within 1 half-width of) the best open-source upper bound, and (ii) Logic

- Table 38: Per-dimension ScorePR with 95% bootstrap CIs. Format: point estimate ± CI half-width (N cases). Dimensions: WK = World-Knowledge, HC = Human-Centric, LR = Logic-Reasoning, IB

= Information-Based.

Model WK HC LR IB

Sora2-8s 36.9±12.0 (n=21) 44.7±15.3 (n=9) 25.9±8.6 (n=27) 37.3±9.7 (n=23) Sora2-12s 34.0±12.4 (n=21) 42.1±15.2 (n=9) 26.9±7.9 (n=27) 33.3±12.2 (n=23) Kling 42.2±10.3 (n=21) 32.5±21.0 (n=9) 22.4±10.8 (n=27) 35.7±11.2 (n=23) Wan2.6 35.2±9.7 (n=21) 34.5±20.1 (n=9) 26.2±10.0 (n=27) 35.5±12.2 (n=23) Seedance2.0 43.2±8.6 (n=21) 35.9±19.9 (n=9) 31.7±10.0 (n=27) 47.6±10.5 (n=23) Veo3.1-Fast 55.0±11.5 (n=21) 35.1±17.4 (n=9) 25.7±8.3 (n=27) 28.6±10.3 (n=23)

LTX2.3 15.6±10.1 (n=21) 19.3±12.0 (n=9) 11.9±6.3 (n=27) 22.7±10.4 (n=23) Wan2.2-14B 22.9±14.5 (n=21) 14.5±12.6 (n=9) 16.4±7.8 (n=27) 15.0±7.8 (n=23) UniVideo 13.8±10.1 (n=21) 15.8±12.6 (n=9) 11.2±9.6 (n=27) 17.3±8.6 (n=23) HunyuanVideo-1.5 21.6±10.1 (n=21) 8.1±7.1 (n=9) 12.7±9.1 (n=27) 24.2±8.4 (n=23) Cosmos-Predict2.5 15.2±11.1 (n=21) 22.2±20.0 (n=9) 7.1±4.9 (n=27) 24.7±10.6 (n=23) LongCat-Video 13.3±9.4 (n=21) 22.8±14.4 (n=9) 12.6±8.7 (n=27) 22.8±10.1 (n=23)

- Table 39: Per-dimension S(v) with 95% bootstrap CIs (point estimate ± CI half-width, N cases). Model WK HC LR IB

Sora2-8s 62.6±10.6 (n=21) 76.7±19.7 (n=9) 43.0±8.8 (n=23) 58.0±14.3 (n=22) Sora2-12s 51.8±8.5 (n=21) 72.8±18.9 (n=9) 48.2±10.7 (n=25) 60.0±12.5 (n=23) Kling 72.0±12.3 (n=20) 87.2±15.3 (n=9) 37.3±9.4 (n=26) 48.8±14.6 (n=23) Wan2.6 61.8±12.7 (n=21) 64.2±19.2 (n=9) 42.3±11.3 (n=24) 42.6±12.4 (n=23) Seedance2.0 70.4±12.2 (n=21) 83.9±16.8 (n=9) 56.7±12.2 (n=22) 42.5±12.2 (n=23) Veo3.1-Fast 80.1±12.0 (n=21) 77.2±14.8 (n=8) 31.5±7.3 (n=23) 47.2±14.0 (n=23)

LTX2.3 35.1±13.1 (n=21) 27.8±15.4 (n=9) 24.7±9.7 (n=26) 25.8±5.7 (n=23) Wan2.2-14B 39.4±9.2 (n=21) 38.1±11.0 (n=9) 19.5±6.0 (n=27) 30.5±7.3 (n=23) UniVideo 29.4±9.4 (n=21) 37.2±15.6 (n=9) 14.4±7.1 (n=27) 16.0±6.0 (n=23) HunyuanVideo-1.5 37.7±10.5 (n=21) 35.3±12.8 (n=9) 19.8±8.7 (n=27) 22.5±8.8 (n=23) Cosmos-Predict2.5 40.8±9.1 (n=20) 30.8±6.5 (n=9) 26.7±4.2 (n=27) 26.1±5.9 (n=23) LongCat-Video 35.1±7.7 (n=21) 42.8±17.8 (n=9) 16.3±6.5 (n=27) 20.2±6.7 (n=23)

Reasoning is the only dimension on which the strongest open-source generator (Wan2.2-14B) has a CI that overlaps with the weakest closed-source generator (Kling), confirming that Logic Reasoning is where the closed/open separation is least settled.

Per-dimension S(v) CIs. Table 39 reports per-dimension S(v) with 95% CIs. Information-Based and Logic Reasoning are the dimensions with the largest dispersion across models (and therefore the largest CI half-widths in absolute terms), which directly motivates Information-Based as the most informative reward-model diagnostic in Section 4.4.

Rank distribution. Table 40 summarizes the 12-model rank distribution under joint bootstrap of overall ScorePR. Two structural facts emerge: (i) every closed-source 95% rank interval is contained in [1,6] and every open-source interval in [7,12], so the two tiers never swap under resampling; (ii) within the closed tier only Seedance2.0 has a tightly concentrated rank (P(rank=1)=89.3%, interval [1,2]), while Sora2-8s, Sora2-12s, Kling, Wan2.6, and Veo3.1-Fast each spread their rank mass over the remaining slots {2,...,6} with modal probability ≤ 40%. Within open-source, only UniVideo has a tightly concentrated rank (P(rank=12)=69.7%); the other five open-source generators form a tied cluster in slots [7,11]. Closed-source ordering beyond Seedance2.0 and open-source ordering beyond the UniVideo floor should therefore be reported as clusters rather than strict rankings.

Take-aways. The closed-vs.-open separation and the dominance of Seedance2.0 within the closed tier are both fully supported by the bootstrap. Strict ordering claims among the remaining five closedsource models, and among the open-source generators above UniVideo, are not statistically supported and we therefore describe them as tied clusters in the main text. Per-dimension CIs widen visibly on Human-Centric (the dimension with the smallest case pool) so per-dimension Human-Centric ordering should be read with care; per-dimension closed/open separation is preserved on the other three dimensions.

- Table 40: Rank stability under joint case-level bootstrap (B=2000, overall ScorePR). Modal rank, the bootstrap probability of that modal rank, and the 95% rank interval (smallest interval containing ≥ 95% of bootstrap rank mass).

Model Modal rank P(modal) 95% rank interval

Sora2-8s 3 33.7% [2, 6] Sora2-12s 6 30.4% [2, 6] Kling 6 26.1% [2, 6] Wan2.6 6 31.2% [2, 6] Seedance2.0 1 89.3% [1, 2] Veo3.1-Fast 2 39.8% [1, 6]

LTX2.3 9 25.4% [7, 12] Wan2.2-14B 7 23.4% [7, 12] UniVideo 12 69.7% [9, 12] HunyuanVideo-1.5 7 30.8% [7, 11] Cosmos-Predict2.5 11 22.2% [7, 12] LongCat-Video 7 24.6% [7, 12]

- Table 41: Overall extended results for open-source generators on the full WorldReasonBench benchmark. 95% bootstrap CIs (B=2000).

Model N AccQA [95% CI] ScorePR [95% CI] S(v) [95% CI] LTX2.3 428 17.5 [15.6, 19.4] 15.7 [13.8, 17.6] 25.7 [23.6, 27.8] Wan2.2-14B 428 21.5 [19.2, 23.8] 19.6 [17.5, 21.7] 38.5 [35.7, 41.3] UniVideo 427 17.8 [15.8, 19.8] 15.8 [13.9, 17.7] 28.2 [25.7, 30.7] HunyuanVideo-1.5 428 20.8 [18.6, 23.0] 19.1 [17.0, 21.2] 32.7 [30.1, 35.3] Cosmos-Predict2.5 428 19.5 [17.5, 21.5] 17.3 [15.3, 19.3] 37.6 [35.1, 40.1] LongCat-Video 428 20.3 [18.2, 22.4] 18.2 [16.2, 20.2] 34.6 [31.8, 37.4]

### O Extended Evaluation of Open-Source Generators on the Full WorldReasonBench Benchmark

This appendix reports the open-source results computed over the full 436-case WorldReasonBench benchmark, complementing the cross-model comparison in Table 2 and Appendix N. The two views are consistent: the per-model intra-open-source ranking and the absolute scores agree with the main-text comparison to within a few tenths of a point on every dimension and metric, indicating that the cross-model conclusions in Section 4.2 (separation between tiers, per-dimension difficulty profile, dominance of Wan2.2-14B / HunyuanVideo-1.5 within the open-source tier) extend naturally to the full benchmark.

Overall extended results. Table 41 reports overall AccQA, ScorePR, and S(v) for the six opensource generators with 95% bootstrap CIs (B=2000). Wan2.2-14B leads on AccQA and S(v), HunyuanVideo-1.5 is the strongest on ScorePR when the dynamic-phase weight is taken into account (Wan2.2-14B and HunyuanVideo-1.5 are statistically tied on ScorePR by their CIs), and UniVideo and LTX2.3 form the bottom of the open-source tier; the qualitative ordering matches Table 37.

Per-dimension extended results. Table 42 reports the per-dimension breakdown (point estimate ± CI half-width) on the full benchmark. The same difficulty profile observed in the main text persists at higher statistical resolution: Logic Reasoning is the worst dimension for every open-source model (ScorePR in 10.8–13.7, S(v) in 13.4–27.1), while World Knowledge is the best (ScorePR up to 23.5, S(v) up to 50.2). The Human-Centric and Information-Based bottlenecks identified for closed-source generators in Section 4.2 also appear here, with HunyuanVideo-1.5 and Wan2.2-14B emerging as the strongest open-source models on Information-Based / Human-Centric respectively, and Cosmos-Predict2.5 standing out on Logic Reasoning S(v) (27.1).

- Table 42: Per-dimension extended results for open-source generators on the full WorldReasonBench benchmark. Format: point estimate ± CI half-width (N cases). WK = World-Knowledge, HC = Human-Centric, LR = Logic-Reasoning, IB = Information-Based.

Metric Model WK HC LR IB

LTX2.3 15.5±3.8 (n=127) 13.2±4.1 (n=77) 13.3±2.9 (n=124) 21.0±4.4 (n=100) Wan2.2-14B 21.9±4.4 (n=127) 26.4±4.9 (n=77) 13.7±3.5 (n=124) 18.5±4.1 (n=100) UniVideo 19.1±3.7 (n=126) 19.6±4.5 (n=77) 10.8±2.9 (n=124) 15.0±3.7 (n=100) HunyuanVideo-1.5 23.5±4.3 (n=127) 17.2±4.8 (n=77) 13.6±3.1 (n=124) 21.7±4.6 (n=100) Cosmos-Predict2.5 19.8±3.9 (n=127) 19.7±4.8 (n=77) 11.8±2.8 (n=124) 18.9±4.3 (n=100) LongCat-Video 18.4±3.6 (n=127) 19.8±5.4 (n=77) 13.2±3.0 (n=124) 22.7±4.8 (n=100)

ScorePR

LTX2.3 30.0±4.4 (n=125) 30.8±5.4 (n=77) 15.8±3.1 (n=116) 28.0±4.3 (n=100) Wan2.2-14B 50.2±5.4 (n=126) 54.8±7.5 (n=76) 19.4±2.6 (n=115) 33.5±4.9 (n=99) UniVideo 38.2±4.5 (n=124) 45.1±6.5 (n=77) 13.4±2.7 (n=118) 20.1±4.7 (n=100) HunyuanVideo-1.5 44.5±5.2 (n=126) 47.2±6.2 (n=77) 16.9±3.3 (n=117) 25.2±4.5 (n=100) Cosmos-Predict2.5 47.4±5.0 (n=125) 46.7±6.1 (n=76) 27.1±3.1 (n=115) 30.4±4.3 (n=99) LongCat-Video 44.3±4.8 (n=125) 51.9±6.4 (n=77) 19.4±3.6 (n=115) 26.7±5.1 (n=100)

S(v)

### P Compute Resources

This appendix reports the hardware, model sizes, and approximate compute budget used to produce the main-text experiments, so that the evaluation pipeline can be reproduced.

Hardware. All open-source video generation and all on-premise VLM-judge inference were carried out on NVIDIA H100 80GB GPUs hosted on an internal cluster (Linux, NVLink-equipped 8-GPU nodes, CUDA 12.x, PyTorch 2.x). Closed-source video generators (Sora2, Kling, Wan2.6, Seedance2.0, Veo3.1-Fast) were accessed through their respective commercial APIs and therefore did not consume our local compute. Closed-source VLM judges (GPT-5.4, Gemini-3.1-Flash) were similarly accessed through APIs.

Open-source video generation. The six open-source generators (LTX2.3, Wan2.2-14B, UniVideo, HunyuanVideo-1.5, Cosmos-Predict2.5, LongCat-Video) were deployed on H100 nodes following each model’s official inference recipe. Each model produces a 5-second 480p–720p clip from an image-plus-text input; we generate one video per benchmark case under each instruction regime. With 436 cases and two instruction regimes (Difficult / Easy), each open-source generator contributes ∼872 generations. Wall-clock per video ranges from approximately 30 seconds (LTX2.3) to ∼5 minutes (Wan2.2-14B, HunyuanVideo-1.5) on a single H100, depending on model size and sampler steps.

Open-source VLM-judge inference. The headline judge Qwen3.5-27B (and the comparison configurations Qwen3.5-9B and Qwen3.5-27B-Thinking) is deployed on H100 nodes via the vLLM serving stack with tensor parallelism across 4 H100 GPUs per replica; we run one to two replicas in parallel. The QA pipeline issues one VQA call per case (5–7 questions answered jointly per video) and one binary-judging call per question against ground truth, processed at 4 FPS with ∼9k visual tokens per 5-second video. Per-video judge latency averages 20–60 seconds depending on whether extended thinking is enabled. The full Process-aware Reasoning Verification pass over ∼5K videos consumes on the order of 50–80 H100·hours; the Multi-dimensional Quality Assessment point-wise pass over the same video pool consumes a comparable budget.

Closed-source API usage. Closed-source video generation and the GPT-5.4 / Gemini-3.1-Flash judges incur API cost rather than local GPU time. Aggregate API call volume is approximately 5K image-conditioned video generations across the five closed-source generators and approximately 25K VLM-judge calls across all reward-model evaluations.

Total project compute. Counting only the experiments reported in this paper, the on-premise VLMjudge passes (process-aware verification, point-wise and pair-wise quality assessment, frame-rate ablation, weight-sensitivity bootstrap) consume on the order of 400 H100·hours, and open-source video generation consumes on the order of 1,500 H100·hours. Including preliminary experiments, prompt-engineering iterations, failed runs, and earlier versions of the data-curation pipeline that did not appear in the final paper, the cumulative project compute is roughly 2−3× the reported figure. Closed-source generation and closed-source judge calls are paid via APIs and are not included in

the H100-hour budget. We will release inference scripts, vLLM serving configurations, and seeds together with the benchmark to support reproduction on comparable H100-class hardware.

### Q Broader Impacts

Positive impacts. WorldReasonBench and WorldRewardBench are intended to make the evaluation of modern video generators more trustworthy. By exposing where current systems fail at world-state reasoning—especially on Logic Reasoning and Information-Based content—the benchmark gives researchers and downstream users a structured way to detect “visually polished but semantically wrong” generations rather than relying on aggregate aesthetic scores. The paired human-preference data provides a reusable calibration target for any future automatic judge or reward model, which we expect to be useful for safer deployment, model auditing by third parties, and more rigorous comparison across closed-source commercial systems and open-source releases.

Potential negative impacts and mitigations. WorldReasonBench does not release any new generative model and does not improve the generative capability of any video model on its own; it is an evaluation suite. Nevertheless, it could indirectly inform stronger video generators over time, which carries the well-known risks of generative video research: misuse for deepfake creation, misleading or fabricated visual evidence, and impersonation. We mitigate this in three concrete ways: (i) the benchmark and reward bench are designed to expose failure modes (mechanism violations, fabricated text/numbers, missing dynamics), so the same diagnostics that are useful for improving models are also directly useful for fake-content detection; (ii) we release only evaluation prompts, automatically generated QA pairs, expert preference labels, and aggregation scripts—we do not release pretrained video generators or fine-tuned reward models with weights; (iii) we report the benchmark’s limitations transparently (Appendix S), so that headline “score gains” cannot be used to overstate world-modelling competence in safety-critical contexts. We do not foresee any direct surveillance, privacy, or fairness harm from the benchmark itself.

Annotator wellbeing. The five annotators are research-team members who rated AI-generated content and did not interact with end-users or sensitive private data. The annotated material consists of model-generated videos derived from publicly sourced prompts and images and was screened to exclude graphic, explicit, or otherwise harmful content before annotation.

### R Licenses for Existing Assets

This appendix enumerates the major external assets used in this paper, with the license terms we relied on at the time of the experiments. We cite the original works in the references and we use each asset within the scope of its respective license.

Open-source video generation models. We use the following open-source generators for inference only, downloaded from their official model cards or repositories: LTX2.3 (Apache 2.0), Wan2.2-14B (Apache 2.0) [23], UniVideo (research-use license, official repository), HunyuanVideo-1.5 (Tencent Hunyuan Community License Agreement) [9], Cosmos-Predict2.5 (NVIDIA Open Model License), and LongCat-Video (Meituan LongCat License, research and non-commercial use). Use is consistent with each model card; weights are not redistributed.

Closed-source video generators (commercial APIs). We access Sora2 (OpenAI), Kling (Kuaishou), Wan2.6 (Alibaba), Seedance2.0 (ByteDance), and Veo3.1-Fast (Google) through their public commercial APIs and abide by each provider’s Terms of Service for evaluation and research use. No model weights are obtained or redistributed; only the videos generated in response to our prompts are stored, and only for the purpose of running this benchmark.

VLM judges. Qwen3.5-9B/-27B and the Thinking variants are used under the Tongyi Qianwen License Agreement (research and limited commercial use) [21]. Gemini-3.1-Flash is accessed via Google Cloud API under the Google API Terms of Service. GPT-5.4 is accessed via the OpenAI API under the OpenAI Terms of Service.

Reference metrics, datasets, and benchmarks. We report or cite numbers from VBench [7] and VBench-2.0 [30] (Apache 2.0), EvalCrafter [12] (MIT), FETV [13] (research license), T2VCompBench [19] (Apache 2.0), V-ReasonBench [14], Gen-ViRe [11], VIPER [10], WorldSimBench [18], VideoVerse [25], PhyGenBench [16], and Ruler-Bench [5]. We do not redistribute their data; only summary numbers and qualitative descriptions are quoted.

Software tooling. The pipeline relies on standard scientific-computing libraries used within their original licenses: PyTorch (BSD), HuggingFace Transformers (Apache 2.0), vLLM (Apache 2.0), NumPy (BSD), pandas (BSD), and Matplotlib (PSF-style).

Image inputs. The initial-state images used as conditioning inputs are sampled from publicly available imagery permitted for research use; no scraped or copyrighted imagery is redistributed. The released benchmark includes only the rewritten textual prompts, generated QA pairs, and expert annotations; image references in our distribution take the form of open-source URLs or hashes rather than re-hosted pixel data, except for the small set of representative figures used in this paper (which are AI-generated or otherwise free-to-distribute).

Released assets. WorldReasonBench and WorldRewardBench—including the prompt list, taxonomy, automatic QA pairs, expert preference pairs, and evaluation scripts—will be released under a Creative Commons CC-BY 4.0 license, with attribution to this paper and notification of any derived datasets that re-package our assets.

### S Limitations

We list the known limitations of WorldReasonBench and WorldRewardBench together with the concrete mitigations already in place and the next steps we plan to take.

VLM dependency in QA construction and judging. Both QA generation and automatic answer verification rely on VLMs (Qwen3.5 / Gemini-3.1), and the headline metrics in the main text are produced by Qwen3.5-27B. To control for systematic VLM bias we apply three independent checks:

- (i) a human audit of a stratified random sample of ∼300 generated QA pairs by two trained annotators, with Cohen κ=0.78 and 7.8% rejected items rewritten or removed before release (Appendix H.4);
- (ii) calibration of every automatic ranking against expert-derived Human Elo over ∼6K preference

pairs, where ScorePR achieves Spearman ρ=0.955 (Section 4.3); and (iii) cross-family comparison of Qwen, Gemini, and (subset) GPT judges, which preserves the closed-vs.-open separation and the Information-Based bottleneck across families (Section 4.5). Residual judge bias persists on close pairs (47.5% judge accuracy when the human score gap is ≤ 0.5), and we explicitly recommend that any new judge be calibrated on WorldRewardBench before being used as a headline metric.

Cross-model evaluation set and ranking uncertainty. The main-text comparison in Table 2 is performed on a shared evaluation set so that closed- and open-source generators are scored on identical case-level inputs. Joint bootstrap rank stability (Appendix N) shows that the closed-vs.-open separation is statistically robust while strict ordering inside the closed tier is supported only for Seedance2.0; we therefore report the other five closed-source models as a tied cluster rather than a strict ranking. Per-dimension Human-Centric CIs are visibly wider than those on the other three dimensions, so per-dimension Human-Centric ordering should be read as suggestive rather than definitive. Extended open-source results on the full WorldReasonBench benchmark are provided in Appendix O, and broadening cross-model coverage at the per-dimension level is the highest priority for the next benchmark revision.

Scope of the reasoning taxonomy and coverage. WorldReasonBench covers four world-state dimensions and 22 subcategories that focus on initial-state-conditioned future-state prediction. Several reasoning aspects are deliberately out of scope: counterfactual or interventional “what-if” queries, multi-agent social dynamics beyond two-actor interactions, exact physics simulation against numerical ground truth (e.g., trajectory MSE), and long-horizon multi-event chains beyond a single transition. Released QA prompts and ground-truth answers are in English only, and case images are sampled from publicly available sources without restriction by region or domain. We do not claim taxonomic

exhaustiveness; extending the taxonomy along these axes is intended community-driven future work, which we explicitly invite by open-sourcing the construction pipeline.

Reward-model evaluation scope. We evaluate five judges (Qwen3.5-9B / 27B in two configurations, Gemini-3.1-Flash, GPT-5.4) under both pair-wise and point-wise protocols. Three uses are deliberately not validated in this paper and remain future work: (i) end-to-end training of a reward model from WorldRewardBench preference pairs; (ii) downstream finetuning of generators guided by S(v) or ScorePR as a reward signal; and (iii) a comprehensive full-set GPT-5.4 evaluation (currently 595-video subset, point-wise only). The present results therefore validate WorldRewardBench as a calibration benchmark for automatic judges, not yet as a training corpus.

Hint-gain interpretation. We report the Easy/Difficult split in Section 4.2 as a descriptive signal of how much a model relies on prompt-side guidance, and explicitly do not interpret it as direct evidence of latent world reasoning, because ceiling effects, prompt length, and instruction-following capacity could each enlarge the asymmetry. Substantive process-vs-outcome attribution is instead carried by ScorePR and sdyn/AccQA, both calibrated against human Elo.

