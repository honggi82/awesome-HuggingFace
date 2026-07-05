# arXiv:2605.22109v1[cs.AI]21May2026

## Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?

##### Caixin Kang1,2 Tianyu Yan2,3 Sitong Gong2,3 Mingfang Zhang1 Liangyang Ouyang1,2 Ruicong Liu1,2 Bo Zheng2 Huchuan Lu3 Kaipeng Zhang2 Yoichi Sato1 Yifei Huang1,2

1The University of Tokyo 2Shanda AI Research Tokyo 3Dalian University of Technology {cxkang, ysato}@iis.u-tokyo.ac.jp hyf015@gmail.com

Code Dataset

[Figure 1]

[Figure 2]

### Abstract

Multimodal Large Language Models (MLLMs) are increasingly deployed in humanfacing roles where personality perception is critical, yet existing benchmarks evaluate this capability solely on numerical Big Five score prediction, leaving open whether models truly perceive personality through behavioral understanding or merely prejudge through superficial pattern matching. We address this gap with three contributions. (i) A new task: we formalize Grounded Personality Reasoning (GPR), which requires MLLMs to anchor each Big Five rating in observable evidence through a chain of rating, reasoning, and grounding. (ii) A new dataset: we release MM-OCEAN (1,104 videos, 5,320 MCQs), produced by a multiagent pipeline with human verification, with timestamped behavioral observations, evidence-grounded trait analyses, and seven categories of cue-grounding MCQs. (iii) Benchmark and analysis: we design a three-tier evaluation (rating, reasoning, grounding) plus four sample-level failure-mode metrics: Prejudice Rate (PR), Confabulation Rate (CR), Integration-failure Rate (IR), and Holistic-grounding Rate (HR), and benchmark 27 MLLMs (13 closed, 14 open). The analysis uncovers a striking Prejudice Gap: across the field, 51% of correct ratings are not grounded in retrieved cues, and the Holistic-Grounding Rate spans only 0–33.5%. These findings expose a disconnect between getting the right score and reasoning for the right reason, charting a roadmap for grounded social cognition in MLLMs.

### 1 Introduction

Multimodal Large Language Models (MLLMs) are rapidly entering high-stakes, human-centric applications: AI-powered interview screening [46], mental-health triage from facial and vocal cues [25], social robots and companion digital humans that adapt to user traits [63, 9], and intelligent game NPCs that modulate behavior based on player affect [18]. At the heart of all these systems lies a shared capability: personality perception, the inference of stable psychological characteristics from observable behavior, with the Big Five (OCEAN) model [30] as the de facto target of inference.

But how well do current MLLMs actually understand the people they observe? Traditional benchmarks for apparent personality recognition (APR), such as ChaLearn First Impressions [56, 14], frame the task as numerical regression on Big Five trait scores. This formulation cannot distinguish a model that “gets it right” from one that merely “guesses right”: a model may achieve low prediction error by exploiting superficial correlations (e.g., smiling faces → high agreeableness) without genuinely understanding the supporting evidence, i.e., the right answer for the wrong reason.

This distinction between genuine perception and superficial prejudice carries practical stakes. Half a century of person-perception research shows that accurate trait inference rests on integrating specific

Preprint.

[Figure 3]

- Figure 1: Overview of MM-OCEAN. Multimodal inputs are processed by a multi-agent humancollaborative pipeline, filtered by text-only LLMs, and reviewed by experts to produce a benchmark supporting three tasks: ordinal Big Five rating (T1), open-ended evidence-grounded reasoning (T2), and structured cue-grounding Multiple-Choice Questions (MCQs) (T3).

behavioral micro-cues such as gaze and posture shifts, not on gestalt impressions [17, 2, 42]. By definition, a rating that cites no such cues is a prejudice, not a perception. Regulation has begun to enforce the same standard. The EU AI Act now classifies personality-based hiring and education systems as high-risk and mandates an explainable evidence trail for each deployed prediction [12]. A personality judgment is trustworthy only if grounded in behavioral evidence. To formalize this requirement we introduce Grounded Personality Reasoning (GPR), which requires a model to (1) perceive fine-grained multimodal behavioral cues, (2) reason about how these cues map to personality traits via evidence-based analysis, and (3) demonstrate these abilities on structured multiple-choice probes that target specific sub-skills (e.g., microexpression localization, temporalcausal reasoning).

To evaluate GPR we construct MM-OCEAN, comprising 1,104 videos and 5,320 cue-grounding MCQs built by a five-stage multi-agent human-collaborative annotation pipeline (Figure 1). A threetier evaluation framework probes the perception chain at increasing depth: ordinal personality rating (Task 1), open-ended rating reasoning (Task 2), and structured cue grounding (Task 3; tested via targeted multiple-choice questions). Because aggregate task scores hide which step failed on a given sample, we add four sample-level failure-mode rates: Prejudice rate (PR; right rating, wrong cues), Confabulation rate (CR; plausible rationale, wrong cues), Integration-failure rate (IR; right cues, wrong rating), and Holistic-Grounding rate (HR; all three correct).

Benchmarking 27 representative MLLMs (13 proprietary, 14 open-source) reveals a striking Prejudice Gap: 51% of all correct ratings come without grounded cue retrieval, and the Holistic-Grounding Rate spans only 0–33.5%. Moreover, recent reasoning-capable MLLMs dominate the upper leaderboard, but the prejudice phenomenon is universal, even at the closed-source frontier, ∼ 15% of correct ratings remain ungrounded. Consequently, today’s MLLMs often “get the right score for the wrong reason,” a gap our benchmark is designed to detect. In summary, our contributions are as follows:

- • Task. We formalize Grounded Personality Reasoning (GPR), distinguishing genuine perception from prejudice via a rating–reasoning–grounding chain.
- • Dataset. We release MM-OCEAN (1,104 videos, 5,320 MCQs) with timestamped atomic observations, evidence-grounded trait analyses, and seven categories of cue-grounding MCQs, produced by an Observer–Psychologist–Examiner–Aligner pipeline with human verification.

- Table 1: Positioning of MM-OCEAN against related benchmarks. Mod.: V/A/T = video/audio/text; Format: Reg/Cls/Open = regression/classification/open-ended.

Benchmark Domain Mod. Scale Format Evid. Reason. Cue-Gnd. Human

grounded eval probes val. Apparent personality recognition (video)

ChaLearn FI v1/v2 [56, 14] Big Five V+A ∼10k Reg ✗ ✗ ✗ partial UDIVA [54] Big Five (dyadic) V+A 188 sessions Reg ✗ ✗ ✗ partial

General-purpose video benchmarks for MLLMs MVBench [36] 20 video tasks V 4k MCQ MCQ ✗ ✗ partial (by task) partial Video-MME [16] 30 subtasks V+A 2.7k MCQ MCQ ✗ ✗ partial (by domain) ✓ TempCompass [41] Temporal video V 7k items Mixed ✗ ✗ partial (5 asp.) ✓ Perception Test [55] Perceptual video V+A 11.6k MCQ+Loc. ✗ ✗ partial ✓

###### Social cognition / Theory of Mind (text-only)

SocialIQA [62] Social commonsense T 38k MCQ ✗ ✗ ✗ ✓ FANToM [32] ToM in dialogue T ∼10k Open+MCQ ✗ partial ✗ ✓ Hi-ToM [68] Higher-order ToM T 600 MCQ ✗ ✗ ✗ ✗

###### Video emotion / affective understanding

MELD [57] Conversational emotion V+A+T 13k utter. Cls ✗ ✗ ✗ ✓ EmoBench [61] Emotional intelligence T 400 MCQ ✗ partial partial (2 asp.) ✓

MM-OCEAN (ours) GPR (Big Five + cue-grounding) V+A+T 1,104 / 5,320 Multi ✓ ✓ ✓(7 cat.) ✓

• Benchmark and analysis. We design a three-tier evaluation framework (rating, reasoning, grounding) and four sample-level failure-mode metrics (PR/CR/IR/HR). Across 27 MLLMs we uncover the Prejudice Gap, the discriminative power of HR, the prevalence of reasoning-capable variants among top performers, and two failure archetypes (confident raters vs. cautious reasoners).

### 2 Related Work

Psychological background: the Big Five model. The Big Five (OCEAN) model [44, 30] is the most empirically supported personality taxonomy in psychology, validated across languages and cultures and routinely used in clinical and social-science research [8]. Following ChaLearn First Impressions and most prior APR work, we adopt Big Five as the target of inference throughout MM-OCEAN.

Apparent personality recognition. The ChaLearn Looking at People challenges [56, 14] established apparent personality recognition (APR), where models predict Big Five scores from short video clips via deep multimodal fusion [26], from CNN aggregation [26] to Transformer architectures [60]. All existing APR benchmarks remain pure regression with numerical labels, providing no mechanism to evaluate why a particular score was assigned. GPR reframes the task to require behaviorally grounded reasoning, not numerical outputs alone.

Video understanding benchmarks for MLLMs. Recent benchmarks evaluate MLLMs’ video understanding across temporal reasoning (TempCompass [41], MVBench [36]), long-form comprehension (Video-MME [16], EgoSchema [43]), and multi-task assessment [15]. While some touch on human-centric understanding through emotion recognition [57] or action detection, none simultaneously target personality from video, require evidence-grounded reasoning, evaluate the reasoning chain itself, and supply fine-grained cue-grounding probes; MM-OCEAN fills these gaps along all four dimensions (Table 1).

Social cognition and theory of mind. ToM benchmarks (FANToM [32], OpenToM [70], HiToM [68]) test reasoning about momentary mental states from text, and recent multimodal extensions probe higher-order social cognition such as deception in multi-party interactions [31] and multispeaker attention [52, 53]. Our work extends this line to personality perception, a higher-order social-cognitive task requiring multimodal integration over longer time spans and reasoning about stable trait dispositions; GPR additionally requires reasoning to be grounded in observable evidence.

### 3 MM-OCEAN: Benchmark Construction

##### 3.1 Task Definition: Grounded Personality Reasoning

Input. A Grounded Personality Reasoning (GPR) instance is a short video V = (Vvis,Vaud,Vtxt) comprising a sequence of RGB frames Vvis ∈ RT×H×W×3, an audio waveform Vaud, and a speech transcription Vtxt. We denote the set of Big Five traits by T = {E,A,C,N,O} and the ordinal personality scale by L = {1,2,3,4,5} (Very Low to Very High).

- Table 2: Seven cue-grounding MCQ categories generated by the Examiner. Two clusters: Reasoning (semantic / causal inference) and Visual Grounding (pixel- / time-level localization).

Category Cognitive target Example Reasoning cluster

Personality Attrib.(Pers) Behavior → trait mapping “Which Big Five trait does the behavior at 11.6–

14.8s most strongly support?”

Counterfactual(Counter) Alternative-scenario inference “If the behavior at 11.6–14.8s were absent, which

trait rating would change most?”

Temporal-Causal(TempC) Cause-effect across time “Which causal chain best links the person’s actions

across the video?”

Mixed Emotion(Mixed) Complex affective state “During 1.9–8.4s, the person’s emotional state is

###### best characterized as . . .” Visual Grounding cluster

Micro-expression(Micro) Subtle facial signal detection “When does a notable micro-expression change

relevant to the rated trait occur?”

Spatial Loc.(Spat) Body-part-level localization “At ∼12.6s within bbox (0.30, 0.66, 0.09, 0.09),

what is the most prominent action?”

Temp-Spatial Jnt.(TSJnt) Joint time×space grounding “When and where does the subject make a head-

coordinated emphatic gesture?”

[Figure 4]

- Figure 2: MM-OCEAN overview. (a) Three-layer sunburst over benchmark scope, three evaluation tasks, and the seven cue-grounding categories. (b) Atomic-observation density across the four perceptual channels; bounding-box geometry is attached to every Expression / Action observation. Outputs across the three tasks. A model fθ must produce:

T1 (Rating) yˆi ∈ L, ∀i ∈ T , (1)

- T2 (Reasoning) (Oˆ,Rˆ) = fθ(V ), Oˆ = {ok}Kk=1, Rˆ = {ri | i∈T }, (2)
- T3 (Grounding) aˆq ∈ {A,B,C,D,E,F}, ∀q ∈ Q, (3)

where each observation ok = (dk,tsk,tek,desck,bk) records a perceptual dimension dk ∈ {Expression, Action, Audio, Background}, start/end timestamps (in seconds), a free-text description

desck, and a body-part tag bk (e.g., face, hand); each reasoning chain ri=(ℓi,Ei,rati) comprises the predicted trait level ℓi∈L, an evidence set Ei⊆{1,...,K} of observation indices (OBS-IDs), and a free-text rationale rati; Q is the set of seven cue-grounding MCQs for V . The grounding constraint Ei⊆{1,...,K} — every trait judgment must cite at least one observed cue — is what distinguishes GPR from Apparent Personality Recognition (APR), which evaluates only yˆi.

##### 3.2 Multi-Agent Human-Collaborative Annotation Pipeline

MM-OCEAN is constructed through a five-stage pipeline that interleaves four LLM agents (Observer, Psychologist, Examiner, and Aligner) with two complementary human roles: 24 trained annotatorverifiers (Stage 1) and a pool of expert reviewers (Stage 5), as visualized in Figure 1. The full annotation protocol, web-tool design, and inter-annotator agreement are detailed in Appendix B.

- Stage 1. Atomic-Cue Annotation (Observer + Human). The Observer agent receives the video and transcription and emits atomic behavioral observations, i.e., the smallest indivisible behavioral events (e.g., a single eyebrow raise, a brief pause), each tagged with a unique OBS-ID, a perceptual

- dimension (Expression, Action, Audio, Background), preliminary timestamps, a factual description, and body-part labels. 24 trained human annotators then review every drafted cue, labelling it correct, incorrect, or nonexistent and pruning the latter two; for every retained Expression or Action observation, the annotator further refines its timestamps and tight bounding-box via a frame-accurate web tool we built. 78.2% of Observer drafts are accepted, 14.6% corrected, and 5.9% deleted; pairwise verdict agreement on overlap pool is 77% (App. B).
- Stage 2. Trait Reasoning (Psychologist). The Psychologist receives the verified observations and produces, for each Big Five trait, a structured analysis containing a trait-level assessment (mapped from the GT scores in the First Impressions [14] to five ordinal levels), a reasoning chain citing cues as evidence, and a confidence-weighted rationale.
- Stage 3. MCQ Generation (Examiner). The Examiner consumes the verified observations and Psychologist analyses and generates seven cue-grounding MCQs spanning a cognitive taxonomy (Table 2, Figure 2) organized from reasoning to visual grounding. The reasoning cluster probes higherorder social-cognitive abilities established in psychology and video QA: Personality Attribution [17] (behavior→trait inference), Counterfactual reasoning [59], Temporal-Causal chains [69], and Mixed Emotion discrimination [34]. The visual-grounding cluster probes fine-grained perceptual localization: Micro-expression [13, 72] detection, Spatial Localization of body regions [74, 39], and joint TemporalSpatial grounding [75, 40]. Each MCQ has six options: one correct answer and five distractors covering three failure modes (text-derivable, plausible-but-wrong-segment, near-miss).
- Stage 4. Quality Assurance (Aligner). The Aligner performs automated quality assurance on the MCQs through two layers: deterministic code checks (timestamp range, bounding-box validity) and LLM-level semantic review (consistency between MCQ correct answers and the personality analyses; factual alignment with the observations). Full Aligner protocol in Appendix A. Cross-judge robustness validation via Claude 4.5/Gemini 2.5 confirms stable T2 ranking (ρ ≥ 0.92, App. J).
- Stage 5. Filtering and Expert Review (Human + Text-only LLMs). Each MCQ passes through a two-step quality gate. (a) Text-leakage filter. Every MCQ is answered by two text-only LLMs (GPT4o-mini and Gemini Flash) using only the question stem and options (no video, no observations); items that both LLMs answer correctly are flagged as transcript-derivable and dropped, ensuring every retained question requires multimodal grounding. (b) Expert review. Trained expert annotators review the surviving MCQs from the video, providing the final human correction and quality control.

##### 3.3 Dataset and Statistics

Source. MM-OCEAN draws its videos from the ChaLearn First Impressions V2 dataset [14], which contains ∼10K fifteen-second clips of single-person speech with crowd-sourced Big Five trait scores and ASR-extracted transcriptions.

Statistics. The released benchmark comprises 1,104 test videos accompanied by three layers of finegrained annotations: ∼13.5K human-verified atomic behavioral observations across four perceptual channels (Expression, Action, Audio, Background); 5,520 trait-level personality analyses; and 5,320 cue-grounding MCQs (averaging 4.8 retained per video after filter). Continuous Big Five scores are discretized into the five ordinal levels of L; the per-trait class distribution is reported in Appendix Table A1. Figure 2 jointly visualizes the resulting dataset structure.

### 4 Evaluation Framework

MM-OCEAN evaluates each model through three tasks of increasing cognitive depth (Figure 3): ordinal personality rating (T1), open-ended rating reasoning (T2), and structured cue grounding (T3); cross-task diagnostic rates (§4.4) then localize where the personality-reasoning chain breaks.

##### 4.1 Task 1: Ordinal Personality Rating

Given V , the model predicts yˆi ∈ L for each trait i∈T (Eq. 1). Over a test set Dtest of N videos, we report exact-match accuracy and mean absolute error:

AccT1 =

N

1 5N

1 5N n, i

⊮ y ˆi(n) = yi(n) , MAET1 =

n=1 i∈T

y ˆi(n) − yi(n) , (4)

[Figure 5]

- Figure 3: The five-stage multi-agent human-collaborative annotation pipeline. Observer drafts atomic observations → Annotator verifies and localizes them (Stage 1) → Psychologist produces evidence-grounded Big Five analyses (Stage 2) → Examiner generates seven categories of cuegrounding MCQs (Stage 3) → Aligner enforces four consistency checks C1–C4 (Stage 4) → Stage 5 applies text-leakage filtering (a) and expert review (b).

complemented by Spearman’s ρ in the appendix. Ordinal levels align with both human judgment and generative MLLM output formats better than continuous scores.

##### 4.2 Task 2: Open-Ended Rating Reasoning

Given V , the model produces (Oˆ,Rˆ) (Eq. 2): an open-ended explanation of why the rating was given. An AI-as-Judge J evaluates models output against GT along four dimensions: Evidence Coverage, Logical Coherence, Grounding Accuracy, and Directional Accuracy, collected in DJ with |DJ| = 4. Each dimension returns a score sd ∈ [1,10]; we report the per-sample composite and its mean:

N

1 |DJ| d∈D

1 N

sd(fθ(V ); GT), ST2 =

ST2(Vn,fθ). (5)

ST2(V,fθ) =

n=1

J

##### 4.3 Task 3: Structured Cue Grounding

Task 3 isolates the ability to ground personality judgments in specific observable cues through structured multiple-choice probes. For each q∈QV in one of the seven cognitive categories C defined in Table 2, the model outputs aˆq (Eq. 3). We report overall and per-category accuracy:

1 |Qc| q∈Q

1 |Q| q∈Q

⊮[ˆaq = a⋆q], Acc(Tc3) =

AccT3 =

##### 4.4 Cross-Task Diagnosis: Gaps and Failure Modes

⊮[ˆaq = a⋆q], c ∈ C. (6)

c

Beyond per-task accuracy, MM-OCEAN’s three tasks combine to reveal where a model’s personalityreasoning chain breaks. We define five quantities that jointly localize the failure: two population-level signals and four sample-level rates (three failures + one success).

Population-level signals. We rank all evaluated models on each task. The Rating–Grounding Misalignment (RGM) of a model is its average T2/T3 rank minus its T1 rank; a large positive RGM flags a model that rates correctly without comparably grounded downstream support. To probe whether grounding has democratized at the same pace as overall capability, we also report the

closed-vs-open frontier-mean (top-3 within each ecosystem) gap ∆Tk =MopenTk −MclosedTk as a robust ecosystem-level snapshot. We refer to the field-wide phenomenon that most “correct” ratings come

without grounded evidence — captured jointly by high PR, low HR, and within-model rating-vsgrounding rank disconnect RGM>0 — as the Prejudice Gap (§5.2).

RGM(m) = 12[rkT2(m) + rkT3(m)] − rkT1(m), (7) ∆Tk = MopenTk − MclosedTk . (8)

Sample-level failure modes. Each prediction either succeeds or fails on three independent axes (rating, reasoning, cue retrieval), placing the outcome into one of 23=8 cells. Four cells correspond to interpretable archetypes: Prejudice Rate (PR; right rating, wrong cues), Confabulation Rate (CR; right rating, incoherent reasoning), Integration-failure Rate (IR; right cues, wrong rating), and Holistic-Grounding Rate (HR; all three correct). Formally, we binarize each task outcome by a threshold θk:

⊮[ˆyi = yi⋆], R2 = S

⊮[ˆaq = a⋆q], (9)

rk = ⊮[Rk ≥ θk], R1 = |T1|

10 , R3 = |Q1

T 2

V |

q

i

with defaults θ1=θ3=0.5 (majority-correct) and θ2=0.7 (the ≥ 7 judge bucket; sensitivity in Appendix I); the four rates are then

PR(m)=Pr[r3=0|r1=1], CR(m)=Pr[r2=0|r1=1], (10) IR(m)=Pr[r1=0|r3=1], HR(m)=Pr[r1=1∧r2=1∧r3=1]. (11)

PR/CR/IR are minimized; HR is capturing full three-tier success. A 3×3×3 threshold sweep confirms that the HR ranking is stable (ρ≥0.92 across all 27 combos; Appendix I).

### 5 Benchmarking Results

##### 5.1 Models and Evaluation Protocol

We evaluate 27 representative MLLMs spanning 12 families: GPT [1, 29, 47, 50, 48, 49, 51], Gemini [64, 65, 11, 22, 24, 23], Claude [6, 4, 5, 3], Qwen-VL [66, 67, 7, 71, 58], Gemma [21], Llama [45], GLM [20, 27], InternVL [10], MiniCPM [73], MiMo [35], Step [28], and LLaVA [37, 38]; 13 are proprietary and 14 are open-source, with the full list and parameter sizes in Table 3. We uniformly sample frames per video and use the same structured prompt per task for all models; open-source models are served via vLLM [33]. For Task 2 we use GPT4o-mini as the AI-as-Judge, with a confidently-wrong consistency check in Appendix J. A cross-judge robustness check with Claude Haiku 4.5 and Gemini 2.5 Flash-Lite confirms the T2 ranking is stable across judge families (Spearman ρ ≥ 0.92, Appendix J). Compute resources are detailed in Appendix Z.

##### 5.2 Leaderboard and the Prejudice Gap

- Table 3 reports the full leaderboard, sorted by Holistic-Grounding Rate (HR). Our evaluation uncovers a pervasive Prejudice Gap across the 27 evaluated MLLMs, the mean Prejudice Rate is PR=51.3%, where over half of correct ratings are ungrounded. Meanwhile, the mean Holistic-Grounding Rate is only HR=10.4%, with the field’s best model (Gemini 3 Flash) reaching just 33.5%. A traditional T1-only leaderboard would credit a model with 50–56% rating accuracy as “competent at personality assessment,” yet on the same model, Prejudice Rate (PR) is typically 40–87% (Table 3), most of those correct ratings rely on cues the model could not actually recover. Per-model PR-vs-T1 and PR/CR/IR/HR fingerprint visualizations are in Appendix R and S.

The phenomenon is universal across the model landscape. Even at the proprietary frontier (Gemini 3 Flash, GPT-5.5, Gemini 3.1 Pro), Top-3 mean PR ≈ 14.5%, leaving 1 in 7 correct ratings ungrounded; at the open-source frontier (Qwen3.5-397B, Qwen3-VL-235B, Qwen3-VL-30B), Top-3 mean PR ≈47.0%. While the performance gap between open and closed frontiers remains narrow for rating (∆T1=−5.6%) and explanation (∆T2=−3.6%) , it widens for cue retrieval (∆T3=−26.6%; full table in Appendix V). Personality scoring and verbal reasoning have largely democratized; behavioral cue retrieval has not, and the open-source frontier is where prejudice is most prevalent. §5.3 drills into where this gap concentrates and how it interacts with per-sample failure modes.

Table 3: Main MM-OCEAN leaderboard (27 models, sorted by HR within each group). T2: T2Avg4 AI-as-Judge composite (1–10). HR/PR/CR/IR: holistic-grounding / prejudice / confabulation / integration-failure rates (%) from Eqs. (10–11) at default thresholds. RGM: rating–grounding misalignment, the rank-based per-model gap (Eq. 7). Bold: best per column; underline: second best.

Per-task accuracy Failure-mode rates (%)

Model Size T1↑ MAE↓ T2↑ T3↑ HR↑ PR↓ CR↓ IR↓ RGM Reference

Random baseline – 20.0 – – 16.7 – – – – –

Proprietary Models Gemini 3 Flash [24] API 64.1 0.42 6.65 66.5 33.5 17.2 44.7 28.7 +0.5 GPT-5.5 [50] API 56.0 0.51 6.65 66.4 28.0 15.5 46.4 36.5 −0.5 Gemini 3.1 Pro [23] API 57.3 0.50 6.59 70.6 27.4 10.8 53.2 33.4 +0.0 Gemini 2.5 Pro [11] API 50.1 0.61 6.40 65.2 20.0 16.9 52.3 48.6 −7.0 GPT-5.4 [48] API 48.7 0.60 6.48 52.6 17.9 33.0 46.1 48.7 −9.5 Gemini 2.5 Flash [11] API 43.1 0.69 6.37 56.5 16.9 28.0 43.9 59.0 −16.5 Claude Opus 4.6 [4] API 50.2 0.59 6.50 49.7 16.8 40.7 48.4 46.1 −5.5 Claude Sonnet 4.6 [5] API 46.6 0.63 6.37 45.6 12.5 46.1 47.4 54.3 −9.0 Claude Haiku 4.5 [3] API 50.6 0.57 6.51 41.0 12.4 55.9 44.6 46.4 −1.5 GPT-5.4-mini [49] API 49.6 0.59 6.38 40.5 11.3 55.4 49.4 48.2 −2.0 o4-mini [51] API 48.0 0.62 6.05 43.4 7.8 48.2 71.7 51.7 −5.0 GPT-4o [29] API 53.3 0.55 6.03 31.9 4.5 69.7 75.7 38.3 +11.0 GPT-4o-mini [29] API 47.6 0.62 5.44 17.5 0.3 87.9 95.2 45.5 +5.0

Open-source Models Qwen3.5-397B-A17B [58] 397B 53.1 0.55 6.45 48.1 15.9 41.5 54.3 40.9 +0.0 Qwen3-VL-235B-A22B [7] 235B 51.5 0.58 6.39 44.2 12.8 47.0 56.4 43.8 +1.0 Qwen3-VL-30B-A3B [7] 30B 55.8 0.52 6.34 43.0 12.4 52.4 58.8 38.8 +8.0 Gemma-4-31B-it [21] 31B 55.7 0.52 6.02 57.0 11.3 29.8 75.4 39.2 +4.5 Llama-4-Maverick-FP8 [45] 402B 55.9 0.53 6.01 36.6 5.8 64.5 75.8 37.4 +14.0 GLM-4.6V [27] 108B 48.2 0.62 5.86 42.0 4.6 52.7 80.7 51.2 −1.0 MiMo-VL-7B-RL [35] 7B 51.1 0.56 5.82 38.9 3.6 56.5 84.1 42.2 +8.0 Qwen3-VL-8B [7] 8B 50.0 0.60 5.80 37.0 2.8 62.4 85.5 47.4 +5.0 Step3-VL-10B [28] 10B 42.4 0.71 5.51 36.3 0.9 62.3 92.9 62.1 −5.5 MiniCPM-o 2.6 [73] 8B 44.7 0.65 4.79 28.6 0.6 67.8 95.1 55.5 +1.5 Qwen2.5-Omni-7B [71] 7B 43.8 0.66 5.10 27.8 0.5 79.9 95.8 55.6 +0.0 Qwen2.5-VL-7B [7] 7B 45.1 0.65 4.67 23.9 0.1 86.3 98.9 58.9 +4.5 InternVL3-8B [10] 8B 43.8 0.65 4.84 26.4 0.0 75.6 99.8 54.9 +0.0 LLaVA-NeXT-Video-7B [38] 7B 36.0 0.87 1.94 16.7 0.0 82.3 100.0 62.9 +0.0

##### 5.3 Where Prejudice Concentrates: Cognitive and Per-Sample Diagnostics

We drill into the Prejudice Gap along two complementary lenses: per-category cognitive sub-abilities (which T3 categories are systemic bottlenecks of cue retrieval) and per-sample diagnostic rates (which competence combinations break or succeed jointly).

Per-category breakdown. Mean accuracy across the 27 models reveals a stable difficulty hierarchy: Temporal-Causal Reasoning is the easiest (64.8%), while Spatial Localization (30.7%) and Microexpression Localization (34.6%) are the hardest. The Top-3 closed advantage concentrates almost entirely on the visual-grounding cluster (Figure 4), with +19.5pp on Spatial Localization and +21.8pp on Temporal-Spatial Joint, versus only 6–11pp gaps on every reasoning-cluster category. Even the strongest closed model (Gemini 3.1 Pro) attains only 57% on Spatial Localization and 71% on Temporal-Spatial Joint, so fine-grained spatiotemporal grounding is a benchmark-wide bottleneck and the most actionable target for the next generation of open-source MLLMs. Full per-category accuracies are in Appendix M (Table A6).

HR as a highly discriminatory measure. The PR/CR/IR/HR columns of Table 3 (defined in §4.4) decompose per-sample errors into interpretable archetypes. HR spans 0.0% (LLaVA-NeXT, InternVL3) to 33.5% (Gemini 3 Flash); its coefficient of variation CV≈0.93 is far larger than any single-task metric (T1 ≈ 0.13, T2 ≈ 0.16, T3 ≈ 0.36), so conditioning on joint rating–reasoning–grounding success amplifies the spread between models well beyond any individual accuracy. Across the 27 models, HR rankings remain strongly correlated with the equally-weighted task mean (Spearman ρ≈0.97 vs. rank by T¯ = (T1 + T2/10 + T3)/3). Informative exceptions exist, Gemma-4-31B-it ranks 5th by task mean but only 13.5th by HR, indicating its T1 and T3 successes are distributed

###### T3 per-category accuracy radar (Top-3 Closed vs Top-3 Open vs 27-model mean)

Personality Attribution

0

Gemini 3.1 Pro

↑ better downstream worse on rating (Cautious Reasoners)

GPT-5.5

80%

Δ = +10

Temporal-Spatial Joint

Counterfactual

Gemini 2.5 Pro

Gemini 3 Flash

5

AveragerankonT2+T3(Reasoning+Grounding)

GPT-5.4 Claude Opus 4.6

60%

Δ = +6

Δ = +22

Qwen3.5-397B

| |
|---|

Gemini 2.5 Flash

40%

Gemma-4-31B

Claude Sonnet 4.6

| |
|---|

10

20%

Claude Haiku 4.5

o4-mini

Qwen3-VL-235B

Qwen3-VL-30B

| |
|---|

GPT-5.4-mini

Δ = +20

15

| |
|---|

GLM-4.6V

Δ = +11

Spatial Loc.

Temporal-Causal

MiMo-VL-7B Qwen3-VL-8B

GPT-4o

Llama-4-Maverick

| |
|---|

Step3-VL-10B

20

↓ better on rating worse downstream

Δ = +22

| |
|---|

(Confident Raters)

Qwen2.5-Omni-7B MiniCPM-o 2.6 InternVL3-8B

Δ = +9

GPT-4o-mini

Qwen2.5-VL-7B

Confident Raters (PRG ≥ +5, n=6)

Micro-expression Mixed Emotion

25

Cautious Reasoners (PRG ≤ −5, n=7)

LLaVA-NeXT-7B

Balanced (|PRG| ≤ 4, n=12)

PRG = 0 (diagonal)

All 27 models (mean) Top-3 Proprietary Top-3 Open-source

25 20 15 10 5 0

- Figure 4: Per-category cognitive radar (T3). Top-3 closed vs. Top-3 open accuracy across the seven cue-grounding MCQ categories. The closed-source advantage concentrates on the visual-grounding cluster (Spatial Localization, Micro-expression, Temporal-Spatial Joint).

Reasoning cluster | Visual Grounding cluster

Rank on T1 (Rating)

Figure 5: RGM archetypes scatter (T1 rank vs. avg. T2+T3 rank). Confident Raters (RGM≥ +5) lie above the diagonal: good on T1 but worse downstream. Cautious Reasoners (RGM≤−5) lie below: good downstream but rate poorly.

across different videos rather than co-occurring per video. This is the very pattern that the HR conditional, and the broader PR/CR/IR cross-task taxonomy (§4.4), is designed to expose.

The failure profile separates two model archetypes. Rating–Grounding Misalignment (RGM, Eq. 7) cleanly partitions models into two archetypes (Figure 5). Confident Raters (RGM≥ +5, n = 5) score well on T1 but fail downstream: Llama-4-Maverick-FP8 (RGM +14) ranks 4 on T1 but only 17/19 on T2/T3. Cautious Reasoners (RGM≤ −5, n = 5) exhibit the opposite pattern: Gemini 2.5 Flash (RGM −16.5) rates poorly (rank 25) but excels on T2 and T3. The remaining 17 models lie in the balanced middle band (|RGM|≤4). The decomposition offers clear diagnostic utility, as confident raters need better grounding while cautious reasoners need better rating calibration.

##### 5.4 Additional Analyses

Beyond the headline findings of §5.2–5.3, we run a set of auxiliary analyses (full results in the appendix) organized around three questions. To localize where the difficulty in MM-OCEAN lives, we examine per-trait T1 accuracy across the five OCEAN dimensions (Appendix N) and the perdimension T2 score breakdown across the four AI-as-Judge axes (Appendix O); jointly these reveal which traits and which reasoning aspects are intrinsically hardest. To probe which model attributes correlate with strong GPR performance, we compare open-source models grouped by parameter scale (Appendix P), trace generation-over-time effects within each closed-source family (Appendix W), and report an observational comparison of reasoning-capable vs. non-reasoning subsets (Appendix Q). To stress-test our methodology and benchmark integrity, we measure T1 prediction-distribution calibration relative to ground truth (Appendix C), positional-bias σ on the A–F option-letter distribution as a cheap cue-retrieval health signal (Appendix X), inter-model rank correlation on per-video task scores (Appendix Y), and a confidently-wrong consistency check on the AI-as-Judge (Appendix J).

### 6 Discussion and Conclusion

We introduce Grounded Personality Reasoning (GPR) and MM-OCEAN, a multi-granularity benchmark requiring MLLMs to ground personality judgments in observable evidence. Evaluating 27 MLLMs reveals a pervasive Prejudice Gap: 51% of correct ratings lack grounded evidence, and the mean Holistic-Grounding Rate (HR) is only 10.4%. These results show that traditional rating-only evaluations systematically overestimate competence by crediting ungrounded predictions. While proprietary and open-source models perform similarly on rating and explanation (∆ < 6%), a substantial −26.6% gap exists in cue retrieval. As a highly discriminative metric, HR reveals that reasoning-intensive models increasingly lead the field. Prioritizing fine-grained spatiotemporal

grounding in post-training is therefore essential for developing the next generation of trustworthy, personality-aware MLLMs.

Limitations and future work. MM-OCEAN focuses on apparent personality from short, singlespeaker English video clips; throughout this work, this denotes the specific construct from First Impressions V2. We evaluate Task 2 reasoning quality via an AI-as-Judge protocol. Natural extensions include cross-cultural and multilingual videos, multi-judge ensembles for Task 2 reliability, and richer grounding operationalizations beyond MCQ-based cue retrieval. We hope MM-OCEAN catalyzes MLLMs that genuinely understand, rather than merely judge, the people they observe; ethical considerations and responsible-use guidelines are discussed in Appendix L.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Nalini Ambady and Robert Rosenthal. Thin slices of expressive behavior as predictors of interpersonal consequences: A meta-analysis. Psychological bulletin, 111(2):256, 1992.
- [3] Anthropic. Claude Haiku 4.5. https://www.anthropic.com/claude/haiku, 2025. Claude Haiku 4.5; accessed 2026-05-04.
- [4] Anthropic. Claude Opus 4.6. https://www.anthropic.com/claude/opus, 2025. Claude Opus 4.6; accessed 2026-05-04.
- [5] Anthropic. Claude Sonnet 4.6. https://www.anthropic.com/claude/sonnet, 2025. Claude Sonnet 4.6; accessed 2026-05-04.
- [6] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 1(1):4, 2024.
- [7] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [8] Murray R Barrick and Michael K Mount. The big five personality dimensions and job performance: a meta-analysis. Personnel psychology, 44(1):1–26, 1991.
- [9] Yiyi Cai, Xuangeng Chu, Xiwei Gao, Sitong Gong, Yifei Huang, Caixin Kang, Kunhang Li, Haiyang Liu, Ruicong Liu, Yun Liu, et al. Towards interactive intelligence for digital humans. arXiv preprint arXiv:2512.13674, 2025.
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.
- [11] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [12] European Parliament Council and the. Regulation (eu) 2024/1689 of the european parliament and of the council of 13 june 2024 laying down harmonised rules on artificial intelligence and amending regulations (ec) no 300/2008,(eu) no 167/2013,(eu) no 168/2013,(eu) 2018/858,(eu) 2018/1139 and (eu) 2019/2144 and directives 2014/90/eu,(eu) 2016/797 and (eu) 2020/1828 (artificial intelligence act). off. J. Eur. Union, 50:202, 2024.
- [13] Paul Ekman and Wallace V Friesen. Nonverbal leakage and clues to deception. Psychiatry, 32

(1):88–106, 1969.

- [14] Hugo Jair Escalante, Heysem Kaya, Albert Ali Salah, Sergio Escalera, Ya˘gmur Güçlütürk, Umut Güçlü, Xavier Baró, Isabelle Guyon, Julio CS Jacques Junior, Meysam Madadi, et al. Modeling, recognizing, and explaining apparent personality from videos. IEEE Transactions on Affective Computing, 13(2):894–911, 2020.
- [15] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. Advances in Neural Information Processing Systems, 37:89098–89124, 2024.
- [16] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118, 2025.
- [17] David C Funder. On the accuracy of personality judgment: a realistic approach. Psychological review, 102(4):652, 1995.
- [18] Francesco Garavaglia, Renato Avellar Nobre, Laura Anna Ripamonti, Dario Maggiorini, and Davide Gadia. Moody5: Personality-biased agents to enhance interactive storytelling in video games. In 2022 IEEE Conference on Games (CoG), pages 175–182. IEEE, 2022.
- [19] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. Datasheets for datasets. Communications of the ACM, 64(12):86–92, 2021.
- [20] Team Glm, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024.
- [21] Google DeepMind. Gemma 4. https://deepmind.google/models/gemma/, 2025. Gemma4-31B-it; accessed 2026-05-04.
- [22] Google DeepMind. Gemini 3. https://blog.google/products/gemini/gemini-3/,

2025. Multimodal large language model.

- [23] Google DeepMind. Gemini 3.1 pro. https://deepmind.google/models/gemini/, 2025. Gemini 3.1 Pro; accessed 2026-05-04.
- [24] Google DeepMind. Gemini 3 flash. https://deepmind.google/models/gemini/, 2025. Gemini 3 Flash; accessed 2026-05-04.
- [25] Jonathan Gratch, Ron Artstein, Gale M Lucas, Giota Stratou, Stefan Scherer, Angela Nazarian, Rachel Wood, Jill Boberg, David DeVault, Stacy Marsella, et al. The distress analysis interview corpus of human and computer interviews. In Lrec, volume 14, pages 3123–3128. Reykjavik, 2014.
- [26] Ya˘gmur Güçlütürk, Umut Güçlü, Marcel AJ van Gerven, and Rob van Lier. Deep impression: Audiovisual deep residual networks for multimodal apparent personality trait recognition. In European conference on computer vision, pages 349–358. Springer, 2016.
- [27] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.
- [28] Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun, et al. Step3-vl-10b technical report. arXiv preprint arXiv:2601.09668, 2026.
- [29] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [30] Oliver P John, Laura P Naumann, and Christopher J Soto. Paradigm shift to the integrative big five trait taxonomy. Handbook of personality: Theory and research, 3(2):114–158, 2008.
- [31] Caixin Kang, Yifei Huang, Liangyang Ouyang, Mingfang Zhang, Ruicong Liu, and Yoichi Sato. Can mllms read the room? a multimodal benchmark for assessing deception in multi-party social interactions. arXiv preprint arXiv:2511.16221, 2025.
- [32] Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. Fantom: A benchmark for stress-testing machine theory of mind in interactions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14397–14413, 2023.
- [33] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
- [34] Jeff T Larsen, A Peter McGraw, and John T Cacioppo. Can people feel happy and sad at the same time? Journal of personality and social psychology, 81(4):684, 2001.
- [35] Jiaze Li, Jingyang Chen, Yuxun Qu, Shijie Xu, Zhenru Lin, Junyou Zhu, Boshen Xu, Wenhui Tan, Pei Fu, Jianzhong Ju, et al. Xiaomi mimo-vl-miloco technical report. arXiv preprint arXiv:2512.17436, 2025.
- [36] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.
- [37] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [38] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.
- [39] Ruicong Liu, Takehiko Ohkawa, Mingfang Zhang, and Yoichi Sato. Single-to-dual-view adaptation for egocentric 3d hand pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 677–686, 2024.
- [40] Ruicong Liu, Yifei Huang, Liangyang Ouyang, Caixin Kang, and Yoichi Sato. Sfhand: A streaming framework for language-guided 3d hand forecasting and embodied manipulation. arXiv preprint arXiv:2511.18127, 2025.
- [41] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731–8772, 2024.
- [42] Yunfei Liu, Ruicong Liu, Haofei Wang, and Feng Lu. Generalizing gaze estimation with outlierguided collaborative adaptation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3835–3844, 2021.
- [43] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023.
- [44] Robert R McCrae and Paul T Costa. Validation of the five-factor model of personality across instruments and observers. Journal of personality and social psychology, 52(1):81, 1987.
- [45] Meta. Llama 4 Maverick. https://ai.meta.com/blog/ llama-4-multimodal-intelligence/, 2025. Llama-4-Maverick (FP8 variant); accessed 2026-05-04.

- [46] Iftekhar Naim, Md Iftekhar Tanveer, Daniel Gildea, and Mohammed Ehsan Hoque. Automated analysis and prediction of job interview performance. IEEE Transactions on Affective Computing, 9(2):191–204, 2016.
- [47] OpenAI. GPT-5. https://openai.com/index/introducing-gpt-5/, 2025. Large language model.
- [48] OpenAI. GPT-5.4. https://openai.com/, 2025. GPT-5.4; accessed 2026-05-04.
- [49] OpenAI. GPT-5.4-mini. https://openai.com/, 2025. GPT-5.4-mini; accessed 2026-05-04.
- [50] OpenAI. GPT-5.5. https://openai.com/, 2025. GPT-5.5; accessed 2026-05-04.
- [51] OpenAI. o4-mini. https://openai.com/index/introducing-o3-and-o4-mini/, 2025. Reasoning-capable language model.
- [52] Liangyang Ouyang, Yifei Huang, Mingfang Zhang, Caixin Kang, Ryosuke Furuta, and Yoichi Sato. Multi-speaker attention alignment for multimodal social interaction. arXiv preprint arXiv:2511.17952, 2025.
- [53] Liangyang Ouyang, Ruicong Liu, Caixin Kang, Yifei Huang, and Yoichi Sato. Socialdirector: Training-free social interaction control for multi-person video generation. arXiv preprint arXiv:2605.10079, 2026.
- [54] Cristina Palmero, Javier Selva, Sorina Smeureanu, Julio Junior, Jacques CS, Albert Clapés, Alexa Moseguí, Zejian Zhang, David Gallardo, Georgina Guilera, et al. Context-aware personality inference in dyadic scenarios: Introducing the udiva dataset. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1–12, 2021.
- [55] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023.
- [56] Víctor Ponce-López, Baiyu Chen, Marc Oliu, Ciprian Corneanu, Albert Clapés, Isabelle Guyon, Xavier Baró, Hugo Jair Escalante, and Sergio Escalera. Chalearn lap 2016: First round challenge on first impressions-dataset and results. In European conference on computer vision, pages 400–418. Springer, 2016.
- [57] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. Meld: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 527–536, 2019.
- [58] Qwen Team. Qwen3.5. https://qwen.ai/blog?id=qwen3.5, 2025. Large language model.
- [59] Neal J Roese. Counterfactual thinking. Psychological bulletin, 121(1):133, 1997.
- [60] Hossein Saberi and Reza Ravanmehr. Transformer-based personality trait recognition enhanced by contextual augmentation. International Journal of Web Research, 9(1):1–24, 2026.
- [61] Sahand Sabour, Siyang Liu, Zheyuan Zhang, June Liu, Jinfeng Zhou, Alvionna Sunaryo, Tatia Lee, Rada Mihalcea, and Minlie Huang. Emobench: Evaluating the emotional intelligence of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5986–6004, 2024.
- [62] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social iqa: Commonsense reasoning about social interactions. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 4463–4473, 2019.
- [63] Cheng Tang, Chao Tang, Steven Gong, Thomas M Kwok, and Yue Hu. Robot character generation and adaptive human-robot interaction with personality shaping. arXiv preprint arXiv:2503.15518, 2025.

- [64] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [65] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [66] Qwen Team. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.
- [67] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [68] Yufan Wu, Yinghui He, Yilin Jia, Rada Mihalcea, Yulong Chen, and Naihao Deng. Hi-tom: A benchmark for evaluating higher-order theory of mind reasoning in large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10691–10706, 2023.
- [69] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021.
- [70] Hainiu Xu, Runcong Zhao, Lixing Zhu, Jinhua Du, and Yulan He. Opentom: A comprehensive benchmark for evaluating theory-of-mind reasoning capabilities of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8593–8623, 2024.
- [71] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.
- [72] Wen-Jing Yan, Xiaobai Li, Su-Jing Wang, Guoying Zhao, Yong-Jin Liu, Yu-Hsin Chen, and Xiaolan Fu. Casme ii: An improved spontaneous micro-expression database and the baseline evaluation. PloS one, 9(1):e86041, 2014.
- [73] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [74] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In European conference on computer vision, pages 69–85. Springer, 2016.
- [75] Zhu Zhang, Zhou Zhao, Yang Zhao, Qi Wang, Huasheng Liu, and Lianli Gao. Where does it exist: Spatio-temporal video grounding for multi-form sentences. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10668–10677, 2020.

### Appendix Contents

##### Appendix Overview

The appendix is grouped thematically. Cross-references in the main text use the labels below. PIPELINE & DATASET

- §A Annotation Pipeline Prompts Observer / Psychologist / Examiner / Aligner contracts.
- §B Human Annotation Protocol 24-annotator team, web tool, quality metrics.

- §H MCQ Examples per Category One worked example per of the seven categories.

- §K Dataset Documentation Datasheets-for-Datasets disclosure.
- §L Ethics and Responsible Use Dataset bias and misuse caveats.

- TASK 1. ORDINAL PERSONALITY RATING

- §N Per-Trait T1 Difficulty Per-Big-Five accuracy and MAE breakdown.

§C Full T1 Prediction Distribution Per-model calibration vs. ground truth. §D Off-by-N Error Distribution Magnitude profile of T1 misranking.

TASK 2. OPEN-ENDED RATING REASONING

- §O T2 Per-Dimension Breakdown Per-model means on the four AI-as-Judge axes.

§E Logical-Coherence Distribution Per-model coherence-score histograms. TASK 3. STRUCTURED CUE GROUNDING

§M Per-Category Accuracy Full numerical breakdown over the seven categories. §F Question Difficulty Distribution Item-level difficulty histogram. §G Additional Model-Level Analyses Expertise vectors, parameter efficiency, substitutability.

FAILURE MODES & DIAGNOSTICS

- §R PR vs. T1 Visualization Trustworthy-zone scatter (5/27 reach it).
- §S Failure-Mode Fingerprint Per-model PR / CR / IR / HR heatmap.
- §T T1 → HR Rank Reordering Per-model rank-shift slope chart.
- §U Right Rating With vs. Without Grounding

Worked side-by-side example contrasting two models.

- §V Closed-vs-Open Task Gap Frontier-mean task-level differences.

§I Threshold Sensitivity Robustness of PR/CR/IR/HR to threshold choice. ECOSYSTEM-LEVEL EFFECTS

§P Open-Source Size Scaling Task accuracy by parameter count. §Q Effect of Reasoning Capability Reasoning vs. non-reasoning subsets (observational).

- §W Generation-over-Time Effects Per-family generation curves.

- §Y Inter-Model Rank Correlation Spearman ρ across the Top-10 models.

METHODOLOGY §X Positional Bias Option-letter skew σ and its T3 correlation. §J AI-as-Judge Protocol Judge prompt, rubric, reliability, and cross-judge robust-

ness.

- §Z Compute Resources GPU hours, API costs, and overhead breakdown.

### A Annotation Pipeline Prompts

This appendix documents the four LLM-agent prompts of the construction pipeline (§3.2). Full prompt text and JSON output schemas are released with the dataset; below we summarize each agent’s contract.

- Stage 1 (a). Observer. Input: a 15-second video and its transcription. System role: “You are a noninterpretive behavior recorder. Record only what is observable; never explain why.” Output schema: a JSON list of atomic observations, one per indivisible behavioral event, each with the fields obs_id, dimension ∈{Expression, Action, Audio, Background}, t_start, t_end, description (factual, ≤20 words), and body_part when applicable. The Observer is explicitly forbidden from making personality or affect claims; this is enforced by prompt instruction and validated via tag-vocabulary checking at parse time.
- Stage 2. Psychologist. Input: a video, its transcription, and the human-verified atomic observation list from Stage 1. System role: “You are an expert personality psychologist applying Funder’s Realistic Accuracy Model.” Output schema: a JSON object with five trait_analysis entries (one per OCEAN trait), each containing a level ∈L mapped from the GT continuous score, an evidence list of cited obs_ids (≥1 required), and a rationale (≤100 words) that links the cited cues to the trait. The grounding constraint is enforced at parse time: any analysis that fails to cite at least one valid obs_id is rejected and re-queried.
- Stage 3. Examiner. Input: the verified observations and Psychologist analyses. System role: “You are an exam writer probing fine-grained social cognition.” Output schema: seven MCQs covering the seven categories of Table 2, each with a question stem, six labeled options (A–F), a correct_answer letter, an explanation citing supporting obs_ids, and a distractor_strategy tag indicating which of three failure modes (text-derivable, plausible-butwrong-segment, near-miss) each distractor exploits. The Examiner is required to use timestamp anchors (t_start–t_end) verbatim from the Observer pool when constructing temporal MCQs, and to use bbox coordinates verbatim from the Stage-1 verifier’s bbox refinements when constructing spatial MCQs.
- Stage 4. Aligner. Input: the seven Examiner-generated MCQs plus the upstream observations and analyses. Operation: a two-layer quality-assurance pipeline. Layer 1 (deterministic code checks): timestamp ranges within the video duration, bbox coordinates in [0,1]4, no out-of-vocabulary categories. Layer 2 (LLM semantic review): consistency between MCQ correct answer and the Stage2 trait conclusion (e.g., a question about Extraversion High should not have a correct answer about Extraversion Low), and factual alignment between MCQ option text and the cited observation. The Aligner can only modify MCQ fields; it cannot edit observations or analyses, which are treated as immutable upstream artifacts. We log every Aligner correction together with the failed-check identifier; aggregate correction rates are reported in Appendix B.

### B Human Annotation Protocol

Annotator team and training. Stage 1 verification was performed by 24 trained annotators. Each annotator completed a training session covering Big Five trait definitions, the four Observer dimensions, common mistakes (e.g., conflating “smiling” with “Agreeableness” before recording the cue), and the bounding-box drawing tool. 1,633 unique videos were submitted, of which 199 were assigned to two independent annotators for quality monitoring. Stage 5 expert review was performed by a smaller panel of psychology-trained reviewers who adjudicated edge cases.

Web tool. The annotation tool is a custom React application with three integrated views (Figure A1): (i) a frame-accurate video scrubber with ≤1-frame stepping, hotkeys for play/pause and ±1-frame seek, and timestamp display in seconds and frames; (ii) an OBS list view with editable per-cue timestamps, dimension dropdown, free-text description, body-part tag, and quality label (correct / incorrect / nonexistent); (iii) an overlay canvas for drawing tight bounding boxes on the current frame. Annotators draw timestamps and bounding boxes directly for every retained Expression or Action observation; the tool stores the original Observer draft alongside the annotator-corrected version for downstream auditing. JSON-schema validity is enforced on save.

[Figure 6]

- Figure A1: Annotation web tool. Three-pane layout: frame-accurate video scrubber, atomic-cue list with edit controls, and bbox overlay on the current frame. Annotators verify each Observer-drafted cue, refine timestamps to frame precision, tighten bbox geometry, and label dimension / body-part.

Per-stage decision rules. For each Observer-drafted observation, the verifier selects one of three labels: correct (kept verbatim), incorrect (reworded), or nonexistent (deleted). Incorrect / nonexistent rates by dimension are reported below. For every retained Expression or Action observation, the verifier additionally tightens the bbox and snaps the start/end timestamps to the closest perceptuallymeaningful frame.

Agreement and correction rates. Across the full annotation campaign, 24 annotators judged a total of 45,609 Observer-drafted clues and drew 36,677 bounding boxes. The aggregate quality breakdown is: 78.2% of clues accepted as correct, 14.6% corrected (wrong description, timestamp, or dimension), and 5.9% deleted (nonexistent or hallucinated cue). Annotators also contributed 605 bonus clues (cues the Observer missed that annotators added).

Annotator quality control. Per-annotator acceptance rates varied from 66.9% to 91.9%. Annotations with abnormally high acceptance rates were flagged during quality monitoring and filtered from the released dataset, removing approximately 8% of submitted videos. The retained core annotators have a mean acceptance rate of 75.7% and a combined correction-plus-deletion rate of 23%, demonstrating that human review adds substantial value beyond the LLM Observer.

Inter-annotator agreement. 199 videos were assigned to two independent annotators via a structured overlap pool. On the 147 pairs where both annotators submitted, the pairwise verdict agreement rate is 77.0% (range 50–100%), confirming that the three-way verdict (correct / wrong / missing) is reproducible across annotators despite the inherent subjectivity of fine-grained behavioral cue assessment. The Stage 5 panel reviews each MCQ that survives the text-leakage filter (Stage 5a). Across the released benchmark, ∼7% of post-filter MCQs receive an expert-corrected option, ∼2% receive a question-stem rewording, and ∼1.5% are dropped entirely.

### C Full T1 Prediction Distribution

- Table A1 reports each model’s class distribution over the five ordinal levels and its total-variation distance to ground truth.

- Table A1: Task 1 prediction distribution (%) across the five ordinal levels (VL, L, M, H, VH), and total-variation distance (TVD) to ground truth (lower is better, 27 models). Ground truth is listed first as a reference. Models follow the leaderboard order of Table 3. Open-source small models systematically collapse onto Medium, while extreme classes (Very Low/Very High) are under-predicted by every model.

Model VL L M H VH TVD ↓ Reference

Ground truth 1.8 17.8 47.5 30.4 2.5 – Proprietary Models

Gemini 3 Flash 0.4 18.1 43.3 37.5 0.7 7.4 GPT-5.5 0.1 13.9 48.3 36.9 0.8 7.4 Gemini 3.1 Pro 0.3 19.0 43.8 36.3 0.6 7.1 Gemini 2.5 Pro 2.2 20.8 35.1 40.3 1.6 13.3 GPT-5.4 0.0 15.6 47.0 36.5 0.8 6.1 Gemini 2.5 Flash 1.2 19.9 28.5 48.9 1.5 20.6 Claude Opus 4.6 0.2 20.2 44.9 34.1 0.5 6.1 Claude Sonnet 4.6 0.1 18.7 46.1 34.2 0.9 4.7 Claude Haiku 4.5 0.1 15.0 46.2 37.5 1.2 7.1 GPT-5.4-mini 0.8 20.8 56.0 21.9 0.5 11.5 o4-mini 0.3 21.3 46.7 31.2 0.5 4.3 GPT-4o 0.1 16.9 47.5 33.6 1.9 3.3 GPT-4o-mini 0.0 16.8 49.4 32.9 0.9 4.5

Open-source Models

Qwen3.5-397B-A17B 0.1 18.7 50.3 30.7 0.2 4.0 Qwen3-VL-235B-A22B 0.1 17.5 42.2 39.4 0.7 9.1 Qwen3-VL-30B-A3B 0.2 19.3 41.0 39.1 0.3 10.3 Gemma-4-31B-it 0.6 16.8 48.2 33.0 1.4 3.4 Llama-4-Maverick-FP8 0.2 23.9 38.5 37.1 0.4 12.7 GLM-4.6V 0.2 22.3 43.1 34.1 0.3 8.3 MiMo-VL-7B-RL 0.2 14.8 70.9 14.1 0.1 23.4 Qwen3-VL-8B 0.3 24.4 40.9 33.0 1.4 9.2 Step3-VL-10B 0.8 20.7 34.4 43.0 1.2 15.5 MiniCPM-o 2.6 0.1 26.4 64.7 8.8 0.0 25.8 Qwen2.5-Omni-7B 0.0 26.7 61.2 12.1 0.0 22.6 Qwen2.5-VL-7B 0.1 22.6 60.3 17.0 0.0 17.6 InternVL3-8B 0.2 20.1 72.6 7.0 0.1 27.4 LLaVA-NeXT-Video-7B 0.0 63.4 19.5 17.0 0.0 45.6

D Off-by-N Error Distribution

- Table A2 shows the distribution of T1 errors by absolute magnitude. The vast majority of errors (∼ 80%) are off-by-one across the 27 models, explaining why MAE is weakly discriminative on MM-OCEAN and why we primarily report accuracy.

E Task 2 Logical-Coherence Distribution

- Table A3 shows how each model’s Task 2 Logical Coherence scores distribute across five score buckets. Top models concentrate around the 8–9 bucket, while weaker models concentrate in 4–5. Virtually no model reaches the 10/10 bucket (only LLaVA-NeXT at 0.4%), confirming that our Judge does not inflate scores.

- Table A2: Task 1 error-magnitude distribution (%) across 27 models. Exact: correct predictions; Off-k: |y−yˆ| = k on the five-level ordinal scale; Mean: average absolute deviation. The vast majority (∼ 80%) of T1 errors are Off-1, so MAE is only weakly discriminative on MM-OCEAN and we emphasize exact-match accuracy in the main tables.

Model Exact ↑ Off-1 Off-2 Off-3 Off-4 Mean ↓ Proprietary Models

Gemini 3 Flash 63.4 30.6 5.6 0.4 0.0 0.43 GPT-5.5 55.3 37.2 7.1 0.5 0.0 0.53 Gemini 3.1 Pro 56.6 36.2 6.9 0.4 0.0 0.51 Gemini 2.5 Pro 49.5 40.3 8.8 1.3 0.1 0.62 GPT-5.4 48.0 42.8 8.7 0.5 0.0 0.62 Gemini 2.5 Flash 42.7 45.3 10.8 1.1 0.0 0.70 Claude Opus 4.6 49.6 41.2 8.5 0.7 0.0 0.60 Claude Sonnet 4.6 45.9 44.2 9.2 0.8 0.0 0.65 Claude Haiku 4.5 49.8 41.9 7.7 0.5 0.0 0.59 GPT-5.4-mini 48.7 42.4 8.0 0.9 0.0 0.61 o4-mini 47.2 42.9 9.1 0.8 0.0 0.64 GPT-4o 52.6 39.3 7.5 0.7 0.0 0.56 GPT-4o-mini 46.9 43.4 9.0 0.6 0.0 0.63

###### Open-source Models

Qwen3.5-397B-A17B 52.3 39.9 7.2 0.6 0.0 0.56 Qwen3-VL-235B-A22B 50.8 39.9 8.5 0.8 0.0 0.59 Qwen3-VL-30B-A3B 55.2 36.9 7.1 0.8 0.0 0.54 Gemma-4-31B-it 54.8 37.2 7.3 0.7 0.0 0.54 Llama-4-Maverick-FP8 55.3 35.6 8.3 0.8 0.0 0.55 GLM-4.6V 47.6 42.2 9.3 0.9 0.0 0.63 MiMo-VL-7B-RL 50.4 42.4 6.7 0.5 0.0 0.57 Qwen3-VL-8B 49.6 41.0 8.7 0.8 0.0 0.61 Step3-VL-10B 42.1 45.1 11.8 1.1 0.0 0.72 MiniCPM-o 2.6 43.9 46.0 9.3 0.8 0.0 0.67 Qwen2.5-Omni-7B 42.9 47.2 9.1 0.8 0.0 0.68 Qwen2.5-VL-7B 44.3 46.0 8.9 0.8 0.0 0.66 InternVL3-8B 43.0 48.0 8.3 0.7 0.0 0.67 LLaVA-NeXT-Video-7B 35.8 42.8 19.3 2.1 0.0 0.88

### F Question Difficulty Distribution

Across the 27 evaluated models, the number of models correctly answering a given question follows a bell-shaped distribution (Table A4). The distribution’s long left tail contains 153 questions (1.8%) that no model answered correctly, a “human-only” subset of MM-OCEAN.

### G Additional Model-Level Analyses

Relative expertise per model. Based on the per-category means in Table A6 and the radar in Figure 4, for each model we compute its accuracy in each of the seven cue-grounding categories and subtract the model’s own seven-category mean, yielding a centered expertise vector that isolates relative strengths from absolute capability. Two robust patterns emerge across all 27 models. First, every evaluated MLLM has positive deviation on the reasoning cluster and negative deviation on the visual-grounding cluster, confirming a field-wide preference for semantic reasoning over fine-grained perceptual localization. Second, individual signature strengths exist. Gemini 3.1 Pro is unusually strong on Temporal-Causal (+27pp above its own mean), Gemini 3 Flash on Personality Attribution (+12), and Gemma-4-31B-it leads the open-source field on Temporal-Causal (+27).

Parameter efficiency. Complementing Table A8 and Figure A4, we measure parameter efficiency as billion-parameters per %-T3-above-chance, where chance is 100%/6≈16.7%. The most efficient

- Table A3: Task 2 Logical Coherence score distribution (%) across five buckets, and per-model mean (27 models). Top models concentrate ∼50% of samples in the 8–9 bucket, while weak models concentrate in 4–5. Virtually no model reaches the 10/10 bucket, confirming the Judge does not inflate scores. Models sorted within each group by mean.

Model 1–3 4–5 6–7 8–9 10 Mean ↑ Proprietary Models

Gemini 3 Flash 6.3 9.1 30.1 54.4 0.0 6.97 GPT-5.5 5.2 10.6 32.2 51.9 0.0 6.94 Gemini 3.1 Pro 6.5 11.2 29.4 53.0 0.0 6.91 Gemini 2.5 Pro 10.3 11.5 29.2 49.1 0.0 6.70 GPT-5.4 7.2 11.7 33.3 47.7 0.0 6.78 Gemini 2.5 Flash 12.5 6.3 32.9 48.3 0.0 6.73 Claude Opus 4.6 8.6 10.0 30.7 50.7 0.0 6.84 Claude Sonnet 4.6 9.2 12.4 31.7 46.6 0.0 6.69 Claude Haiku 4.5 7.3 11.1 30.8 50.9 0.0 6.89 GPT-5.4-mini 8.7 10.7 34.7 45.9 0.0 6.65 o4-mini 11.3 15.0 33.1 40.6 0.0 6.39 GPT-4o 8.6 20.4 29.7 41.3 0.0 6.40 GPT-4o-mini 11.4 26.8 41.5 20.3 0.0 5.86

###### Open-source Models

Qwen3.5-397B-A17B 7.6 10.6 32.8 48.9 0.0 6.81 Qwen3-VL-235B-A22B 9.4 11.2 29.7 49.7 0.0 6.70 Qwen3-VL-30B-A3B 10.4 12.1 29.4 48.1 0.0 6.62 Gemma-4-31B-it 11.2 13.3 39.6 35.9 0.0 6.36 Llama-4-Maverick-FP8 10.4 18.7 33.4 37.5 0.0 6.32 GLM-4.6V 11.7 18.5 36.4 33.4 0.0 6.20 MiMo-VL-7B-RL 10.3 21.4 40.4 28.0 0.0 6.14 Qwen3-VL-8B 11.5 17.4 38.6 32.4 0.0 6.21 Step3-VL-10B 14.1 19.8 39.5 26.6 0.0 5.96 MiniCPM-o 2.6 25.2 27.5 32.6 14.6 0.0 5.17 Qwen2.5-Omni-7B 19.4 27.3 37.7 15.6 0.0 5.43 Qwen2.5-VL-7B 24.2 32.2 34.6 8.9 0.0 5.05 InternVL3-8B 19.4 30.7 40.3 9.7 0.0 5.30 LLaVA-NeXT-Video-7B 97.3 1.7 0.3 0.2 0.4 1.28

open-source model is MiMo-VL-7B-RL at 0.315B per %-T3-above-chance, followed by Gemma4-31B-it (0.77) and Qwen3-VL-30B-A3B (1.13). At the heavy end, the 235–402B models cost ∼8–10× more parameters per percentage point. Data quality and post-training appear to matter more than parameter count.

Substitutability matrix. Drawing on Table 3, we define open-source substitutability as the withintask gap between the strongest open-source model and the closed Top-3 median. T1 (rating) is fully substitutable (gap <2pp). T2 (reasoning) is partially substitutable (gap ∼0.2 on 10-pt scale). T3 (cue grounding) is the bottleneck: only Gemma-4-31B-it (57.0%) approaches the proprietary Flash class (56.5%), and no open model reaches Flash-Pro tier (65+%).

### H MCQ Examples per Category

We provide one worked example per of the seven cue-grounding categories, drawn from the released benchmark. All seven examples are from a single representative test video to allow cross-category comparison of the same behavioral context. Each item shows the question, all six options, the correct answer letter, and a one-line explanation. Full distractor strategies and OBS-ID evidence chains are in the released JSON files.

- Table A4: Number of models correctly answering each MCQ (out of 27) and the corresponding number of questions. The distribution is roughly bell-shaped with a long left tail: 153 questions (1.8%) are answered correctly by no model.

# correct models # questions # correct models # questions

27 (all solved) 134 13 285 26 494 12 284 25 456 11 322 24 385 10 324 23 373 9 284 22 331 8 283 21 380 7 287 20 360 6 292 19 334 5 293 18 290 4 269 17 304 3 263 16 288 2 269 15 275 1 178 14 285 0 (human-only) 153

###### (1) Personality Attribution

Q: During 11.6s–14.8s, the person exhibits a specific behavioral pattern. Which Big Five personality dimension is most directly supported by this behavior?

- A. High Conscientiousness, as the gesture reflects a planned and goal-oriented sequence.
- B. High Extraversion, because the individual is demonstrating significant social energy.
- C. High Agreeableness, due to the clear non-verbal effort to build rapport with the audience.
- D. Low Openness, given that the subject is sharing a very common experience.
- E. Medium Neuroticism, because his expression shows some on-camera tension.
- F. Low Agreeableness, because his direct presentation could be interpreted as forceful. Correct: C. The leaning-in + widened eyes + raised brows during 11.6–14.8s is a textbook rapport-building gesture, the strongest single cue for High Agreeableness in the clip.

###### (2) Counterfactual Reasoning

Q: If the person had not exhibited the behavior at 11.6–14.8s, which personality assessment would be most affected?

- A. Conscientiousness would seem higher, because the casual chewing-while-speaking would no longer be visible.
- B. Agreeableness might rise from Medium to High, because this potentially assertive gesture is removed.
- C. Extraversion would seem much lower, as his main moment of social engagement would be discounted.
- D. Agreeableness might drop from High to Medium, as this key rapport-building gesture would be absent.
- E. Openness would be most affected, since this gesture is the central part of his creative expression.
- F. Neuroticism would increase, as the confident gesture’s absence would emphasize earlier self-consciousness.

- Correct: D. Removing the rapport-building moment leaves no other strong evidence for High Agreeableness; the rating would conservatively drop one level.

###### (3) Temporal-Causal Reasoning

Q: Which of the following best describes the causal chain linking the person’s actions at different moments across the video?

- A. A focused moment of assertion (∼7s), a subsequent pause for reflection (∼9s), then a cognitive withdrawal (∼11s).
- B. A cognitive withdrawal for formulation (∼2s), a preparatory pause for structure (∼5s), then a focused re-engagement for assertion (∼7s).
- C. An expressive social gesture (∼5s), a period of anxious self-monitoring (∼9s), and a compensatory return to neutrality (∼13s).
- D. A neutral opening (∼1s), an emotional rapport display (∼4s), then a logical conclusion to reinforce the argument (∼12s).
- E. A cognitive withdrawal (∼2s), a focused re-engagement (∼7s), and a concluding pause for planning (∼11s).
- F. An attempt at direct engagement (∼1s), a disengaging gaze due to distraction (∼5s), then a forced re-focus (∼12s).

- Correct: B. The video shows look-away → pause → assertive re-engagement; B is the only option that matches the actual ordering of all three observed behaviors.

(4) Mixed Emotion Discrimination

Q: During 1.9–8.4s, the person’s emotional state is best characterized as which of the following?

- A. Engaged interest mixed with a feeling of clear surprise.
- B. Subtle amusement tempered by professional composure.
- C. Joyful excitement mixed with playful hesitation.
- D. Quiet confidence tempered by a hint of genuine curiosity.
- E. Mild anticipation mixed with focused readiness to act.
- F. Calm satisfaction tempered with thoughtful recall.

Correct: E. Setting up the action verbally (anticipation) plus tightened lips and gaze fixation (readiness to act) match E; the other options either over-claim affect (excitement, surprise) or claim emotions for which no cue is observable.

(5) Micro-expression Localization

Q: At which time interval does a notable change in the person’s facial micro-expression occur, most relevant to their High Agreeableness rating?

- A. [8.4s–9.6s]
- B. [2.5s–3.7s]
- C. [11.8s–13.0s]
- D. [0.1s–1.3s]
- E. [4.5s–5.7s]
- F. [6.2s–7.4s]

- Correct: C. The interval [11.8,13.0]s captures the eye-widening + brow-raise transition that anchors the High Agreeableness rating; the other intervals contain only neutral baseline expression.

###### (6) Spatial Localization Verification

Q: At around 12.6s, a region was detected at (x=0.296,y=0.657,w=0.093,h=0.091) (normalized, top-left origin). What is the most prominent expression or action in this region?

- A. Subtly raised eyebrows signaling engagement.
- B. An averted gaze directed well above the camera.
- C. Jaw movement associated with active chewing.
- D. A neutral facial expression showing no tension.
- E. Slightly upturned mouth corners forming a smile.
- F. A slightly opened mouth used for verbal emphasis.

- Correct: F. The provided bbox tightly frames the subject’s mouth at 12.6s, where he is leaning forward and speaking emphatically with the mouth slightly open.

- (7) Temporal-Spatial Joint Localization

Q: At which moment and location does the subject make a strong non-verbal gesture to emphasize a point to the viewer, involving a coordinated change in head position and facial expression?

- A. [2.6s] at (x=0.52,y=0.49,w=0.18,h=0.40)
- B. [2.8s] at (x=0.39,y=0.58,w=0.13,h=0.09)
- C. [0.2s] at (x=0.67,y=0.88,w=0.17,h=0.12)
- D. [13.8s] at (x=0.22,y=0.56,w=0.19,h=0.29)
- E. [10.4s] at (x=0.37,y=0.50,w=0.17,h=0.27)
- F. [4.9s] at (x=0.38,y=0.41,w=0.15,h=0.08)

- Correct: D. At 13.8s the subject has leaned forward (peak emphatic gesture), and the bbox covers his face and upper torso at the moment the gesture peaks; the other options either point to neutral baseline timestamps or to too-tight bboxes that miss the head-coordinated motion.

### I Failure-Mode Taxonomy: Threshold Sensitivity

The per-sample binary outcomes rk = ⊮[Rk ≥ θk] defined in Eq. (9) depend on three thresholds (θ1,θ2,θ3). Our default setting is θ1 = θ3 = 0.5 (majority-correct: 3 of 5 traits on T1, 4 of 7 MCQs on T3) and θ2=0.7 (“acceptable” judge quality, the lower bound of the ≥7 score bucket in Appendix Table A3). We assess robustness by sweeping each threshold independently in the ranges θ1 ∈ {0.4,0.5,0.6}, θ3 ∈ {0.4,0.5,0.6}, and θ2 ∈ {0.6,0.7,0.8}, and recomputing PR, CR, IR, HR from Eqs. (10–11) for every combination.

What stability looks like. The qualitative claims of §5.3 are robust if (i) the Spearman rank correlation of HR with the default-threshold HR remains ρ≥0.9 across the swept grid, (ii) the Top-3 closed and Top-3 open identities are invariant, and (iii) the sign of every ∆Tk (§5.2) is preserved.

Sweep results. We sweep all 3×3×3=27 threshold combinations and recompute HR for each of the 27 models. Across the full grid, the Spearman rank correlation of HR with the default-threshold HR is ρ∈[0.925,1.000], confirming that the leaderboard ordering is highly stable. The Top-3 closed and open identities are preserved for 21 of the 27 combinations. The only regime where Top-3 changes is θ2=0.8 (requiring a Judge score ≥8/10), which collapses the field-best HR to ∼4.7% and makes rankings noisy. At θ2∈{0.6,0.7}, the Top-3 is invariant regardless of θ1 or θ3.

The ∆Tk closed-vs-open task-level gaps in Table A10 are computed from raw accuracy means, not from binarized rates, so they are unaffected by any threshold choice.

Practitioner default. We recommend (θ1,θ2,θ3) = (0.5,0.7,0.5). Per-sample Rk values will be released alongside the dataset so that future work can recompute the rates under any preferred threshold.

### J AI-as-Judge Protocol

Judge model. For Task 2 we use GPT-4o-mini (temperature 0, single sample) as a single-judge AI evaluator across all 27 evaluated MLLMs. We deliberately use a model that is not on the leaderboard’s high end to avoid self-preference bias.

Four-dimension rubric. The judge scores each per-trait Task 2 reasoning output on four independent axes, each in [1,10]:

- • Evidence Coverage: does the rationale cite multimodal cues spanning the ∼15s clip, or does it rely on a single static impression?
- • Logical Coherence: do the cited cues genuinely entail the trait level, or is the explanation a non-sequitur?
- • Grounding Accuracy: are the cited cues observable in the video, or fabricated / generic?
- • Directional Accuracy: is the directional claim (high vs. low) consistent with the cited evidence?

The composite per-sample score ST2 (Eq. 5) is the simple mean of the four. The judge sees the model’s Task 2 output, the ground-truth Big Five trait level, and the human-verified atomic observations as reference; it does not see the video.

###### AI-as-Judge Confidently-Wrong Consistency Δ tightly clustered at +2.1–+3.4 (σ_Δ = 0.27 across 27 models)

| |7.52 4.13<br><br>7.58 4.28<br>8.14 4.84<br><br><br>7.82 4.51<br><br>7.77 4.47<br>8.10 4.88<br><br><br>8.00 4.78 7.95 4.75<br><br><br>6.63 3.46<br>7.50 4.36 7.31 4.19 7.08 3.96<br><br><br>6.39 3.29<br>7.36 4.26<br>8.07 4.99 7.74 4.74 7.99 5.00<br><br><br>7.87 4.88<br><br>6.78 3.80<br>7.27 4.33<br><br><br>7.84 4.90<br>8.03 5.13 7.30 4.44 7.83 4.97<br><br><br>7.87 5.09<br>6.19 3.71 3.28 1.22<br>|[Figure 7]<br><br>3.39 3.31 3.31 3.30 3.29 3.22 3.22 3.20 3.17 3.14 3.12 3.12 3.10 3.10 3.08 3.00 2.99 2.99 2.97 2.95 2.94 2.90 2.86 2.86 2.78 2.48 2.06|
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

Llama-4-Maverick

GPT-4o Claude Sonnet 4.6

|[Figure 8]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

- 2
- 3
- 4
- 5
- 6
- 7
- 8

Qwen3-VL-30B

o4-mini Claude Opus 4.6

Gemini 2.5 Pro Qwen3-VL-235B

InternVL3-8B

GLM-4.6V Step3-VL-10B

Judgescore(1–10)

GPT-4o-mini Qwen2.5-VL-7B

Qwen3-VL-8B

GPT-5.4 Gemini 3 Flash

Claude Haiku 4.5

Gemini 3.1 Pro Qwen2.5-Omni-7B

MiMo-VL-7B Qwen3.5-397B Gemini 2.5 Flash Gemma-4-31B GPT-5.4-mini GPT-5.5 MiniCPM-o 2.6 LLaVA-NeXT-7B

T2 | T1 correct T2 | T1 wrong Δ (drop)

- Figure A2: Confidently-wrong consistency of the AI-as-Judge. Each row is a model: T2 mean given T1-correct (left), T2 mean given T1-wrong (middle), and the drop ∆ (right). The ∆ column is tightly clustered across all evaluated models, showing that the Judge tracks correctness rather than style.

Confidently-wrong consistency check. A standard robustness threat for AI-as-Judge is that it may simply mirror the surface “style” of the response rather than its correctness. We test this by partitioning each model’s Task 2 outputs by whether Task 1 was correct on that sample, and computing the conditional Judge mean for each partition (Figure A2). Across all 27 evaluated models, the Judge gives systematically lower scores (∆≈2.1–3.4 points) when T1 was wrong, even though Task 2 prose itself can look plausible to a casual reader. The cross-model standard deviation of this ∆ is only σ∆=0.27, indicating that the Judge applies the correctness-sensitive penalty uniformly rather than rewarding stylistic fluency.

Robustness of AI-as-Judge evaluation. The tight across-model consistency of the confidently-wrong penalty (Figure A2) indicates that Judge-based scoring in Task 2 is not simply rewarding style, but is tracking a latent correctness signal. Combined with the strong negative correlation between positional bias and MCQ accuracy (Figure A10), this suggests that MM-OCEAN is self-diagnostic: the same data that scores a model also reveals why that model fails.

Judge prompt skeleton. The full judge prompt is in our supplementary material. The system message instructs the judge to be a strict psychology grader, the user message provides the model output, the GT trait level, and the human-verified observation list, and the response schema is a JSON object with the four dimension scores plus a one-sentence justification per dimension. We discard outputs that fail JSON-validation (rare; <0.3% of all samples) and re-query.

Cross-judge robustness. A reasonable concern with any single-judge protocol is that the leaderboard reflects judge-specific bias rather than genuine model differences, especially given that the GPT4o-mini judge shares architectural lineage with two of the evaluated models (GPT-4o, GPT-4omini). To rule this out, we re-judge all 27 models’ Task-2 outputs on a stable 200-video random subset (seed=42) with two alternative judges drawn from different families: Claude Haiku 4.5 (Anthropic) and Gemini 2.5 Flash-Lite (Google), both comparable in capability tier to GPT-4omini. We use the identical prompt, rubric, and four-dimension scoring protocol.

- Table A5 summarizes the results. The Spearman rank correlations of ST2 between GPT-4o-mini and each alternative judge are ρ=0.94 (p < 10−13) and ρ=0.92 (p < 10−11) respectively, confirming

- Table A5: Cross-judge robustness. Left: Spearman ρ between primary (GPT-4o-mini) and alternative judges. Right: within-family self-preference check for GPT-family models.

Rank correlation Within-family check Judge pair ρ p-value Model GPT-4o-mini Avg alt ∆

GPT-4o-mini vs Haiku 4.5 0.94 < 10−13 GPT-4o 6.04 5.08 +0.97 GPT-4o-mini vs Flash-Lite 0.92 < 10−11 GPT-4o-mini 5.41 4.31 +1.10

that the T2 ranking is not an artifact of the judge’s model family. The Top-3 identities by ST2 are preserved under both alternative judges.

A within-family check reveals that GPT-4o-mini scores its own family (GPT-4o, GPT-4o-mini) approximately +1.0 point higher on the 10-point scale than the cross-family judge average. This is a modest absolute inflation consistent with the known self-preference tendency of LLM judges, but it does not distort the relative ranking (ρ ≥ 0.91). Haiku 4.5 scores ∼0.7 lower and Flash-Lite ∼1.0 lower than GPT-4o-mini on average across all models, a global calibration shift rather than a model-specific bias.

### K Dataset Documentation

We document MM-OCEAN following the Datasheets for Datasets framework [19] (full Q&A in the released artifact; key items below).

Motivation. The dataset was created to evaluate whether MLLMs can ground personality judgments in observable behavioral evidence (§1). It was created by the authors with no commercial sponsorship.

Composition. Each instance is a 15-second single-speaker English video clip drawn from ChaLearn First Impressions V2 [14], paired with: (i) a transcription, (ii) human-verified atomic behavioral observations across four perceptual channels (Expression, Action, Audio, Background), (iii) five pertrait Big Five personality analyses with cited evidence, and (iv) seven cue-grounding multiple-choice questions covering the seven categories of Table 2. The released split contains 1,104 videos, ∼13.5K verified observations, 5,520 trait analyses, and 5,320 MCQs (4.8 MCQs/video on average after the text-leakage filter). The Big Five labels are inherited from First Impressions V2 crowd-sourced annotations and discretized into five ordinal levels.

Collection process. Videos: drawn from First Impressions V2’s existing test split. Annotations: produced by the multi-agent pipeline of §3.2. Stage 1 verification was performed by 24 trained annotators (1,633 unique videos submitted, 45,609 clues judged, 36,677 bboxes drawn) using the web tool described in Appendix B. Annotators were compensated at the local research-assistant hourly rate.

Preprocessing / cleaning. Beyond the multi-agent pipeline (Stages 1–5), we apply a text-leakage filter (Stage 5a) that drops any MCQ that two text-only LLMs (GPT-4o-mini and Gemini Flash) can both answer correctly from question + options alone. Videos retaining <3 MCQs after filtering are dropped from the released split.

Uses. Intended use is academic research on grounded personality reasoning and trustworthy multimodal evaluation. We discourage downstream deployment in personality screening or hiring without explicit consent and fairness audits (see Appendix L).

Distribution. Released under the same license as ChaLearn First Impressions V2 for the underlying videos plus a research-only license for our annotation layers, on [release URL after acceptance].

Maintenance. The first author maintains the repository and a public issue tracker for label corrections. Errata releases are versioned and announced on the project page.

- Table A6: Task 3 per-category accuracy (%). Left: mean across all 27 models with min–max range. Right: Top-3 closed vs. Top-3 open averages and their absolute gap. The closed advantage concentrates on the visual-grounding cluster.

Category Mean (n=27) Range Top-3 Closed Top-3 Open ∆(pp) Reasoning cluster

Personality Attribution (Pers) 41.7 15–70 67.3 57.5 +9.8 Counterfactual (Counter) 53.6 15–80 76.0 70.0 +6.0 Temporal-Causal (TempC) 64.8 16–92 91.0 80.0 +11.0 Mixed Emotion (Mixed) 54.8 15–80 78.3 69.0 +9.3

###### Visual Grounding cluster

Micro-expression (Micro) 34.6 17–61 60.3 38.5 +21.8 Spatial Loc. (Spat) 30.7 10–57 54.0 34.5 +19.5 Temporal-Spatial Jnt. (TSJnt) 37.4 13–71 65.3 43.5 +21.8

### L Ethics and Responsible Use

Dataset bias. MM-OCEAN inherits the cultural and linguistic biases of ChaLearn First Impressions V2 [14], which is predominantly composed of Western-context English speakers. Personality perception is itself subjective and culturally situated; absolute trait scores should not be interpreted as objective ground truth across cultures.

Misuse risks. Automated personality assessment from video carries risks of misuse, including discriminatory screening (e.g., in hiring), surveillance, and over-claimed psychometric validity. MMOCEAN is designed as a diagnostic research tool for measuring grounded-reasoning capabilities of MLLMs, not a deployment-ready personality-scoring system. We discourage downstream use without explicit consent, transparency about model limitations, and fairness audits of any system built on top of it.

Operationalization caveats. Task 3 operationalizes grounding as MCQ-based retrieval over a predefined cue set; a high Prejudice Rate may in part reflect MCQ-design choices (which cues are queried, which distractors are used) rather than a model’s general inability to ground its judgment in observable behavior. The Task 2 AI-as-Judge, while scalable, may not capture every reasoning dimension; multi-judge or human-judge validation is recommended for high-stakes downstream use.

### M T3 Per-Category Accuracy: Full Numerical Breakdown

- Table A6 reports the full per-category Task 3 accuracy referenced in §5.3 of the main paper. The left half lists the all-model mean and min–max range across the 27 evaluated MLLMs; the right half compares the Top-3 closed and Top-3 open averages and their absolute gap. The Mean column reproduces the all-model difficulty hierarchy referenced in the main text; the ∆ column gives the precise per-category closed-vs-open gap visualized in Figure 4.

N Per-Trait T1 Difficulty

- Table A7 reports mean T1 accuracy and MAE per Big Five trait, averaged across all 27 evaluated models. The trait-difficulty hierarchy is stable across the field: Extraversion / Agreeableness / Conscientiousness cluster at 53–55%; Openness sits slightly lower (49%); Neuroticism is the universally hardest trait (37.7%, MAE 0.87). Even the strongest models (Gemini 3.1 Pro, Gemini 3 Flash, GPT-5.5) attain only 50–58% on Neuroticism, suggesting that current MLLMs are bottlenecked by inferring internal emotional states from short (∼15s) clips. Per-trait per-model details are in the trait-by-model heatmap in our supplementary material.

##### Table A7: Mean T1 accuracy and MAE per Big Five trait, averaged across all 27 models.

###### Trait Mean Acc (%) ↑ Mean MAE ↓

Extraversion 54.5 0.51 Agreeableness 53.8 0.52 Conscientiousness 53.3 0.52 Openness 49.2 0.58

Neuroticism 37.7 0.87

Neuroticism is the universally hardest Big Five trait

###### (a) Per-trait T1 accuracy (mean over 27 models)

(b) Per-trait T1 MAE

1.0

60

0.87

MeanT1MAE(lowerisbetter)

54.5 53.8 53.3

0.8

49.2

50

MeanT1accuracy(%)

40

0.58

37.7

0.6

0.51 0.52 0.52

30

0.4

20

0.2

10

0

0.0

ExtraversionAgreeablenessConscientiousness Openness Neuroticism

ExtraversionAgreeablenessConscientiousness Openness Neuroticism

- Figure A3: Per-trait T1 difficulty across the 27 evaluated MLLMs. Neuroticism is the universally hardest trait; the gap from Openness (49.2%) to Neuroticism (37.7%) is larger than the gap from Extraversion to Openness, indicating that internal-state inference (Neuroticism) requires capabilities qualitatively beyond surface trait recognition.

### O T2 Per-Dimension Breakdown

The four AI-as-Judge dimensions exhibit different difficulty levels across the 27 models: Logical Coherence (mean 6.17) and Grounding Accuracy (mean 6.43) are easier; Directional Accuracy (mean 5.66) sits in between; Evidence Coverage (mean 5.14) is the hardest. Models can write coherent on-topic explanations but tend to under-cite specific behavioral evidence, a finding consistent with the Confabulation Rate (CR) being the dominant failure pattern in mid-tier models (Table 3). A complete model-by-dimension heatmap is provided in the supplementary material.

### P Open-Source Size Scaling

Open-source MLLMs at three parameter scales (Table A8). Scaling from ≤ 8B to 9–32B yields +17pp on T3, but scaling further to 100B+ adds essentially nothing on T3 (−2.8pp due to Gemma4-31B-it being the strongest open model on T3, beating the Qwen3-VL-235B-A22B and Llama-4Maverick despite being ∼10× smaller). Sheer parameter count is not what limits open-source MCQ performance; data quality and post-training appear to matter more.

### Q Effect of Reasoning Capability (Observational)

Caveat. The two subsets compared in this section differ on multiple confounding dimensions: parameter count, model family, generation, and training data. Reasoning-capable variants are typically newer, larger, and drawn from the strongest families. The aggregate gaps below should therefore be read as a descriptive field-level pattern (“reasoning-capable variants tend to lead the leaderboard”), not as a controlled causal effect of explicit thinking. We report this analysis for completeness but do not headline it among our central findings.

Setup. We split the 27 evaluated MLLMs into a reasoning-capable subset (n=13, models exposing an explicit thinking/reasoning mode) and a non-reasoning subset (n=14), and compare the group means on each task and on HR.

##### Table A8: Open-source models grouped by parameter scale.

Scale Count T1 mean T2-Avg4 T3 mean

≤ 8B 7 44.9 4.71 28.5 9–32B 3 51.3 5.96 45.5 ∼ 100B+ 4 52.2 6.18 42.7

Open-Source Size Scaling: 9–32B band leads on T3

80

| |T1 (%)<br><br>T2-Avg4 (×10)<br>| | | |
|---|---|---|---|---|
| |5.96<br><br>6.18<br><br>T3 (%)| | | |
| |51.3 52.2| | | |
| |44.9<br><br>4.71<br><br>28.5<br><br>45.5<br><br>42.7| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

70

60

Score/Accuracy

50

40

30

20

10

0

≤8B (n=7)

9–32B (n=3)

100B+ (n=4)

- Figure A4: Open-source size scaling. Each panel plots the per-band mean across ≤8B, 9–32B, and ∼100B+. T3 plateaus past ∼30B while T1 and T2 keep rising slowly, suggesting cue-grounding requires post-training quality more than parameter count.

Observed pattern. The reasoning-capable subset leads by +18.3pp on T3, +11.5pp on HR, and only +2.6pp on T1; mean RGM is also more negative (−3.8 vs. +3.5, Figure A5). Read together with Appendix W, the most parsimonious interpretation is that newer-generation, larger, top-family models (which happen to be reasoning-capable) dominate cue retrieval, while T1 saturates earlier and is therefore less sensitive to these confounded dimensions. We leave a controlled experiment (matched-size, matched-family, with vs. without thinking mode) to future work.

### R Per-Model PR vs. T1 Visualization

- Figure A6 plots each evaluated MLLM as a point in (T1 accuracy, Prejudice Rate) space. The shaded Trustworthy zone marks the desirable corner — high T1 and low PR. Only 5 of 27 models reach it (Gemini 3 Flash, GPT-5.5, Gemini 3.1 Pro, Gemini 2.5 Pro, and Gemma-4-31B-it, the only open-source model in the zone). The rest of the field clusters above the PR=51.3% reference line, confirming that the Prejudice Gap is a field-wide rather than archetype-specific phenomenon.

S Per-Model Failure-Mode Fingerprint

- Figure A7 renders the four sample-level failure rates from Eqs. (10–11) as a per-model heatmap, sorted by HR. Compared to scanning the same numbers in Table 3, the heatmap makes the field-wide pattern visually immediate: HR (green, “trustworthy”) is concentrated almost entirely in the top three rows; PR (red, “prejudice”) and CR (orange, “confabulation”) saturate for the field’s mid-to-bottom; IR (blue, “integration-failure”) is broadly elevated across the table.

T T1 → HR Rank Reordering

The Failure-Mode Fingerprint (Figure A7) and the PR-vs-T1 scatter (Figure A6) both visualize rates.

- Figure A8 provides a complementary rank-based view of the same phenomenon: for each model, we draw a line from its T1 rank (left, rating-only leaderboard) to its HR rank (right, trustworthy-reasoning leaderboard). Models that drop ≥5 ranks under HR are colored red (Confident Raters); models that

- Table A9: Reasoning-capable vs. non-reasoning models (observational). Group means across the 27 evaluated MLLMs. The gap concentrates on T2/T3/HR, but the two subsets differ on size, family, and generation, so the gap is not a controlled effect of reasoning capability.

Subset T1(%) T2-Avg4 T3(%) HR(%) RGM(mean)

Reasoning (n = 13) 51.0 6.35 52.2 16.4 −3.8 Non-reasoning (n = 14) 48.5 5.38 33.9 4.9 +3.5

∆ (Reasoning − Non-reasoning) +2.6 +0.97 +18.3 +11.5 −7.3

Reasoning-Capable Models Win on T2/T3 and HR — but not on T1

| |Reasoning-capable (n=13)| | | | |
|---|---|---|---|---|---|
| |Non-reasoning (n=14)| | | | |
| |51.0<br><br>52.2 48.5<br><br>Δ = +2.5<br><br>Δ = +18.3| | | | |
| | | | | | |
| |33.9| | | | |
| |Δ = +11.5| | | | |
| |16.4<br><br>Δ = +1.0| | | | |
| |6.3 5.4 4.9| | | | |
| | | | | | |

70

60

50

Score/Accuracy

40

30

20

10

0

T1 (%) T2-Avg4 (1–10) T3 (%) HR (%)

- Figure A5: Reasoning-capable vs. non-reasoning subset means. Differences are large on T3 / HR and small on T1, but the two subsets differ on size, family, and generation, so this gap is observational.

climb ≥ 5 ranks are colored green (Cautious Reasoners); models with |∆rank| < 5 are gray. The picture confirms what RGM (Eq. 7) measures: a small handful of models reorder substantially when HR replaces T1 as the ranking criterion. Llama-4-Maverick-FP8 (T1 rank 4 → HR rank 17) and GPT-4o (rank 5 → rank 18) drop the most, while Gemini 2.5 Flash (rank 24 → rank 5) and GPT-5.4 (rank 17 → rank 5) climb the most. Most models stay close to the diagonal, indicating that for the bulk of the field rating success and trustworthy reasoning are coupled.

### U Worked Example: Right Rating With vs. Without Grounding

To make the prejudice phenomenon concrete, we contrast two models on the same video, the same trait, and the same correct rating. The video is W4tz3plvvKI.001 (a young man discussing his current and aspirational video-editing software); the trait is Extraversion with ground-truth level Low. Both GPT-4o and Gemini 3 Flash predict Low, both pass T2 with comparable Judge scores, yet only Gemini 3 Flash answers the structured cue-grounding probes correctly.

###### Only 5 of 27 MLLMs reach the Trustworthy zone

100

Proprietary Open-source

| |
|---|

Reasoning-capable

Qwen2.5-VL-7B LLaVA-NeXT-7B

| |
|---|

80

| |
|---|

correctratingswithoutretrievedcues

GPT-4o

PrejudiceRate,PR(\%)↓better

MiniCPM-o 2.6

| |
|---|

Llama-4-Maverick

| |
|---|

60

| |
|---|

field PR = 51.3%

Qwen3-VL-30B

| |
|---|

| |
|---|

40

Gemma-4-31B

Trustworthy zone (T1 ≥ 50%, PR ≤ 30%)

20

Gemini 2.5 Pro GPT-5.5 Gemini 3 Flash

Gemini 3.1 Pro

0

35 40 45 50 55 60 65

Task 1 accuracy (\%) → better

Figure A6: Right Rating, Wrong Cues. T1 accuracy vs. Prejudice Rate across 27 MLLMs.

Per-model failure-mode fingerprint (sorted by HR)

###### HR (trustworthy)

###### PR (prejudice)

###### CR (confabulation)

IR (integration-fail)

|33.5|
|---|

|17.2|
|---|

|44.7|
|---|

|28.7|
|---|

Gemini 3 Flash GPT-5.5 Gemini 3.1 Pro Gemini 2.5 Pro

|28.0|
|---|

|15.5|
|---|

|46.4|
|---|

|36.5|
|---|

|27.4|
|---|

|10.8|
|---|

|53.2|
|---|

|33.4|
|---|

|20.0|
|---|

|16.9|
|---|

|52.3|
|---|

|48.6|
|---|

|17.9|
|---|

|33.0|
|---|

|46.1|
|---|

|48.7|
|---|

GPT-5.4 Gemini 2.5 Flash Claude Opus 4.6

|16.9|
|---|

|28.0|
|---|

|43.9|
|---|

|59.0|
|---|

|16.8|
|---|

|40.7|
|---|

|48.4|
|---|

|46.1|
|---|

|15.9|
|---|

|41.5|
|---|

|54.3|
|---|

|40.9|
|---|

Qwen3.5-397B

|12.8|
|---|

|47.0|
|---|

|56.4|
|---|

|43.8|
|---|

Qwen3-VL-235B Claude Sonnet 4.6

|12.5|
|---|

|46.1|
|---|

|47.4|
|---|

|54.3|
|---|

|12.4|
|---|

|55.9|
|---|

|44.6|
|---|

|46.4|
|---|

Claude Haiku 4.5 Qwen3-VL-30B

|12.4|
|---|

|52.4|
|---|

|58.8|
|---|

|38.8|
|---|

|11.3|
|---|

|55.4|
|---|

|49.4|
|---|

|48.2|
|---|

GPT-5.4-mini Gemma-4-31B

|11.3|
|---|

|29.8|
|---|

|75.4|
|---|

|39.2|
|---|

|7.8|
|---|

|48.2|
|---|

|71.7|
|---|

|51.7|
|---|

o4-mini Llama-4-Maverick

|5.8|
|---|

|64.5|
|---|

|75.8|
|---|

|37.4|
|---|

|4.6|
|---|

|52.7|
|---|

|80.7|
|---|

|51.2|
|---|

GLM-4.6V GPT-4o

|4.5|
|---|

|69.7|
|---|

|75.7|
|---|

|38.3|
|---|

|3.6|
|---|

|56.5|
|---|

|84.1|
|---|

|42.2|
|---|

MiMo-VL-7B Qwen3-VL-8B Step3-VL-10B

|2.8|
|---|

|62.4|
|---|

|85.5|
|---|

|47.4|
|---|

|0.9|
|---|

|62.3|
|---|

|92.9|
|---|

|62.1|
|---|

|0.6|
|---|

|67.8|
|---|

|95.1|
|---|

|55.5|
|---|

MiniCPM-o 2.6 Qwen2.5-Omni-7B

|0.5|
|---|

|79.9|
|---|

|95.8|
|---|

|55.6|
|---|

|0.3|
|---|

|87.9|
|---|

|95.2|
|---|

|45.5|
|---|

GPT-4o-mini Qwen2.5-VL-7B

|0.1|
|---|

|86.3|
|---|

|98.9|
|---|

|58.9|
|---|

|0.0|
|---|

|75.6|
|---|

|99.8|
|---|

|54.9|
|---|

InternVL3-8B LLaVA-NeXT-7B

|0.0|
|---|

|82.3|
|---|

|100.0|
|---|

|62.9|
|---|

##### Figure A7: Per-model failure-mode fingerprint (sorted by HR).

Task 2: open-ended reasoning (both models predict Low Extraversion)

- • GPT-4o (T2-Avg4 = 7.8): “The person speaks in a calm and measured tone, with limited outward energy or enthusiasm, suggesting a preference for introspection over sociability. This behavior aligns with a lower level of extraversion, as they do not exhibit assertiveness or high talkativeness.”
- • Gemini 3 Flash (T2-Avg4 =8.2): “The speaker exhibits a low-energy vocal tone and frequent gaze aversion, which are characteristic of introversion and internal cognitive processing. His limited use of expansive gestures and soft speech further support a lower level of extraversion.”

The two rationales are surface-level very similar. Both correctly arrive at Low Extraversion, both cite generic cues (calm tone, low energy). The Judge gives them within 0.5 points of each other.

###### Rating vs. Trustworthy Reasoning: who rides T1 priors?

Drops ≥5 in HR (Confident Raters) Climbs ≥5 in HR (Cautious Reasoners) Stable (|Δ|<5)

0

Gemini 3 Flash Gemini 3 Flash

Gemini 3.1 Pro

GPT-5.5

GPT-5.5

Gemini 3.1 Pro

Llama-4-Maverick

Gemini 2.5 Pro

Qwen3-VL-30B

GPT-5.4

5

Gemma-4-31B

Gemini 2.5 Flash

GPT-4o

Claude Opus 4.6

Qwen3.5-397B Qwen3.5-397B Qwen3-VL-235B Qwen3-VL-235B

Claude Sonnet 4.6 Claude Haiku 4.5 Claude Haiku 4.5

MiMo-VL-7B

10

Rank(1=best)

Claude Opus 4.6

Qwen3-VL-30B

Gemini 2.5 Pro

GPT-5.4-mini

Qwen3-VL-8B

Gemma-4-31B

GPT-5.4-mini

15

GPT-5.4

Llama-4-Maverick

GPT-4o GPT-4o-mini

MiMo-VL-7B

Claude Sonnet 4.6

Qwen3-VL-8B

20

Step3-VL-10B

GPT-4o-mini

Gemini 2.5 Flash

25

Step3-VL-10B

T1 rank (rating only)

HR rank (trustworthy reasoning)

- Figure A8: Rank reordering from T1 to HR. Red = drops ≥5 in HR (Confident Raters), green = climbs ≥5 in HR (Cautious Reasoners), gray = stable.

Task 3: structured cue-grounding probe (same trait, divergent outcome)

Q. During 4.9s–8.7s, the person exhibits a specific behavioral pattern. Which Big Five personality dimension is most directly supported by this behavior?

- A. Low Extraversion due to an internally-focused cognitive processing style (correct)
- B. High Agreeableness because the action shows a patient and calm demeanor.
- C. High Conscientiousness due to the deliberate and careful thought process.
- D. High Extraversion as the person is actively engaging with their thoughts.
- E. Low Neuroticism because the behavior indicates a high level of comfort.
- F. Low Openness since the behavior suggests a resistance to new information. GPT-4o picked C (incorrect). Gemini 3 Flash picked A (correct).

The verified observation at [4.9,8.7]s reads “the subject’s gaze drifts down and to the left while the face remains neutral and speech continues,” a textbook indicator of internally-directed cognitive processing and therefore of Low Extraversion. Both models predicted Low Extraversion at the rating level, so the GT trait label was within reach for both. But only Gemini 3 Flash can localize which behavioral window actually anchors the rating. GPT-4o’s matching T1 + plausible T2 reasoning are not, on this sample, supported by retrievable cues, the very pattern the Prejudice Rate is designed to detect. The same divergence is visible across the remaining six MCQs of this video, which span Counterfactual / Mixed-Emotion / Spatial / Temporal-Spatial / Micro-expression / Temporal-Causal categories.

### V Closed-vs-Open Frontier-Mean Task Gap

- Table A10 reports the frontier-mean (top-3 within each ecosystem) by task, referenced from §5.2. Closed leads on every task, but the gap is task-dependent: small on T1 and T2, several times larger on T3. Read together with the Prejudice Rate distribution (Figure A6), this confirms that the Prejudice Gap is a field-wide phenomenon whose ecosystem-level component concentrates on cue retrieval rather than rating or verbal reasoning.

Table A10: Closed-vs-open frontier-mean task gap. Top-3 within each ecosystem averaged by task. ∆(abs.) is the absolute difference in percentage points (pp); ∆(rel.) is the corresponding relative percent change.

Metric Top-3 Closed Top-3 Open ∆(abs.) ∆(rel.) Winner

T1 Accuracy (%) 59.13 55.80 −3.33pp −5.6% Closed (narrow) T2-Avg4 (1–10) 6.63 6.39 −0.24 −3.6% Closed (narrow) T3 Accuracy (%) 67.84 49.77 −18.06pp −26.6% Closed (large)

Generation-over-Time Effects: Newer generations gain on T3 more than on T1

###### GPT

###### Claude

###### Gemini

Qwen-VL

80

###### T1 T3

70

71

64 66

66

65

60

57

Accuracy(%)

56

56

53

52

56 50

51

50 50

50

50

49 53

50

47 46

45

43

44

40

43

41

40

37

30

32

24

20

Haiku4.5 Sonnet4.6 Opus4.6

2.5Flash 2.5Pro 3Flash 3.1Pro

Qwen2.5-VL-7B Qwen3-VL-8BQwen3-VL-30B-A3BQwen3-VL-235B-A22B

GPT-4o GPT-5.4-mini GPT-5.4 GPT-5.5

- Figure A9: Generation-over-time per family. T3 (cue grounding) improves much more steeply across each family’s generations than T1 (rating), which saturates quickly.

### W Generation-over-Time Effects

Within each closed-model family, every new generation improves T3 substantially while T1 saturates earlier (Figure A9). GPT-4o → GPT-5.5: T3 jumps from 31.9% to 66.4% (+34.5pp). Claude Haiku → Sonnet → Opus: T3 climbs from 41.0% to 49.7%. Gemini 2.5 Pro → 3.1 Pro: T3 from 65.2% to 70.6%. Within Gemini, the Flash variants outperform Pro on T1 (Gemini 3 Flash 64.1% vs. 3.1 Pro 57.3%), while Pro variants outperform on T3 (3.1 Pro 70.6% vs. 3 Flash 66.5%). Higher inference budget appears most useful for multi-step MCQ reasoning and less so for fast intuitive ratings. Open-source families (Qwen-VL 2.5→3) show the same direction. Each generation roughly halves the gap to closed Top-3.

#### X Positional Bias We measure positional bias as

(pℓ(m) − p¯)2 × 100, p¯ = 16.6%, (12)

σ(m) = 16

ℓ∈{A,...,F}

where pℓ(m) is the fraction of MCQs for which model m selects option letter ℓ. Across the 27 models, the top three with the lowest σ are GPT-5.5 (0.7), Gemini 2.5 Flash (0.9), Gemini 2.5 Pro (0.9); the worst are MiniCPM-o 2.6 (19.5), LLaVA-NeXT (11.5), GPT-4o-mini (10.2), InternVL3-8B (10.0). Every model with σ>10 ranks in the bottom third on T3. Figure A10 plots σ against T3 accuracy across the 27 models; the strong negative correlation (r≈−0.68) makes positional-bias σ a cheap early-warning signal of cue-retrieval collapse.

### Y Inter-Model Rank Correlation

For each pair of models in the top-10 (by HR), we compute the Spearman rank correlation of per-video task scores. Mean off-diagonal ρ for T1 is 0.56 (range 0.41–0.74); for T3 it is 0.39 (range 0.29–0.56). Top models agree more on which videos are intrinsically easy or hard to rate (T1) than on which questions they answer correctly (T3). This confirms that T3 separates per-model competence more sharply than T1 (which carries a stronger shared video-difficulty signal), making T3 the cleaner discriminator and HR (which conditions on T3) the sharpest combined metric. Figure A11 shows the full 10×10 correlation matrices for T1 and T3.

###### Positional Bias Strongly Predicts Cue-Grounding Performance (r = -0.70, n=26)

80

Gemini 3.1 Pro

70

GPT-5.5

σ > 10 collapse zone

Gemini 3 Flash

Gemini 2.5 Pro

Task3cue-groundingaccuracy(%)

60

Gemma-4-31B

50

40

Qwen3-VL-8B Step3-VL-10B

30

MiniCPM-o 2.6 InternVL3-8B

Qwen2.5-VL-7B

20

LLaVA-NeXT-7B

Proprietary Open-source

| |
|---|

GPT-4o-mini

| |
|---|

linear fit (r=-0.70)

10

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

Positional bias σ (option-letter distribution width)

- Figure A10: Positional bias σ vs. Task 3 accuracy across the 27 evaluated MLLMs. Models with σ>10 collapse to the bottom third on T3, regardless of family or scale.

[Figure 9]

Gemini3FlashGPT-5.5Gemini3.1ProGemini2.5ProGPT-5.4Gemini2.5FlashClaudeOpus4.6Qwen3.5-397BQwen3-VL-235BClaudeSonnet4.6

Gemini 3 Flash

GPT-5.5

Gemini 3.1 Pro

Gemini 2.5 Pro

GPT-5.4

Gemini 2.5 Flash

Claude Opus 4.6

Qwen3.5-397B

Qwen3-VL-235B

Claude Sonnet 4.6

1.00 0.59 0.66 0.53 0.51 0.42 0.55 0.57 0.58 0.51

0.59 1.00 0.64 0.53 0.67 0.41 0.65 0.64 0.61 0.63

0.66 0.64 1.00 0.58 0.56 0.44 0.60 0.60 0.52 0.55

0.53 0.53 0.58 1.00 0.53 0.56 0.53 0.54 0.52 0.48

0.51 0.67 0.56 0.53 1.00 0.48 0.65 0.61 0.59 0.68

0.42 0.41 0.44 0.56 0.48 1.00 0.42 0.41 0.51 0.42

0.55 0.65 0.60 0.53 0.65 0.42 1.00 0.64 0.59 0.74

- 0.57 0.64 0.60 0.54 0.61 0.41 0.64 1.00 0.62 0.61
- 0.58 0.61 0.52 0.52 0.59 0.51 0.59 0.62 1.00 0.62

0.51 0.63 0.55 0.48 0.68 0.42 0.74 0.61 0.62 1.00

T1 per-video Spearman ρ (top-10 by HR)

[Figure 10]

Gemini3FlashGPT-5.5Gemini3.1ProGemini2.5ProGPT-5.4Gemini2.5FlashClaudeOpus4.6Qwen3.5-397BQwen3-VL-235BClaudeSonnet4.6

Gemini 3 Flash

GPT-5.5

Gemini 3.1 Pro

Gemini 2.5 Pro

GPT-5.4

Gemini 2.5 Flash

Claude Opus 4.6

Qwen3.5-397B

Qwen3-VL-235B

Claude Sonnet 4.6

1.00 0.37 0.50 0.40 0.38 0.40 0.39 0.33 0.31 0.34

- 0.37 1.00 0.46 0.44 0.46 0.40 0.39 0.39 0.34 0.31

0.50 0.46 1.00 0.45 0.35 0.41 0.36 0.38 0.29 0.33

0.40 0.44 0.45 1.00 0.38 0.51 0.37 0.33 0.31 0.31

- 0.38 0.46 0.35 0.38 1.00 0.37 0.52 0.40 0.44 0.44

0.40 0.40 0.41 0.51 0.37 1.00 0.36 0.33 0.31 0.30

- 0.39 0.39 0.36 0.37 0.52 0.36 1.00 0.44 0.49 0.56

- 0.33 0.39 0.38 0.33 0.40 0.33 0.44 1.00 0.47 0.44

0.31 0.34 0.29 0.31 0.44 0.31 0.49 0.47 1.00 0.48

- 0.34 0.31 0.33 0.31 0.44 0.30 0.56 0.44 0.48 1.00

T3 per-video Spearman ρ (top-10 by HR)

[Figure 11]

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

Spearmanρ

Inter-model rank correlation: T1 (left) is more uniform; T3 (right) better separates models

- Figure A11: Pairwise per-video Spearman rank correlation among the Top-10 (by HR) MLLMs. Left: T1 (rating); right: T3 (cue grounding). The lower-saturation T3 panel confirms that T3 carries less shared video-difficulty signal and more model-specific competence signal than T1.

### Z Compute Resources

All open-source models are served on NVIDIA H200 GPUs via vLLM [33]; all proprietary models are accessed through official APIs. Table A11 summarizes the estimated compute for each phase of the project.

##### Table A11: Estimated compute resources.

Phase GPU (H200-hrs) API cost (USD) Dataset construction

Annotation pipeline (Observer/Psychologist/Examiner/Aligner) – ∼$200 Human annotation (24 annotators, 45K clues) – –

Main benchmark evaluation

14 open-source models × 1,104 videos (unified prompt) ∼60 – 13 proprietary models × 1,104 videos (API) – ∼$400 AI-as-Judge (GPT-4o-mini, ∼149K calls) – ∼$120

Robustness and auxiliary experiments

Cross-judge robustness (Haiku + Flash-Lite, 200-video subset) – ∼$60 Threshold sensitivity sweep (27 combos, offline recomputation) <1 – Preliminary / debugging / failed runs (∼50% overhead) ∼30 ∼$100

###### Total ∼90 ∼$880

